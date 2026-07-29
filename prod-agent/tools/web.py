from __future__ import annotations

from dataclasses import dataclass
from http.cookiejar import CookieJar
import json as json_module
import re
import threading
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import (
    HTTPRedirectHandler,
    HTTPCookieProcessor,
    ProxyHandler,
    Request,
    build_opener,
)


class WebToolError(ValueError):
    pass


@dataclass(frozen=True)
class WebResponse:
    status: int
    url: str
    headers: tuple[tuple[str, str], ...]
    body: bytes
    truncated: bool

    @property
    def text(self) -> str:
        content_type = self.header("Content-Type") or ""
        charset = "utf-8"
        for field in content_type.split(";")[1:]:
            key, separator, value = field.strip().partition("=")
            if separator and key.lower() == "charset":
                charset = value.strip("\"' ")
                break
        try:
            return self.body.decode(charset, errors="replace")
        except LookupError:
            return self.body.decode("utf-8", errors="replace")

    def header(self, name: str) -> str | None:
        lowered = name.lower()
        for key, value in self.headers:
            if key.lower() == lowered:
                return value
        return None

    def render(self, max_chars: int | None = None) -> str:
        content_type = self.header("Content-Type") or ""
        if "text" in content_type.lower() or "json" in content_type.lower():
            body = self.text
        else:
            body = self.body.hex()
        text = (
            f"status={self.status}\nurl={self.url}\n"
            f"content-type={content_type}\n\n{body}"
        )
        if max_chars is not None and len(text) > max_chars:
            marker = "\n...[output truncated]"
            if max_chars <= len(marker):
                return marker[:max_chars]
            return text[: max(0, max_chars - len(marker))] + marker
        return text


def _origin(url: str) -> tuple[str, str, int]:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise WebToolError("event base URL must be HTTP or HTTPS")
    if parsed.username or parsed.password:
        raise WebToolError("URLs may not contain credentials")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as error:
        raise WebToolError(f"invalid URL port: {error}") from error
    return parsed.scheme.lower(), parsed.hostname.lower().rstrip("."), port


class _SameOriginRedirects(HTTPRedirectHandler):
    def __init__(self, origin: tuple[str, str, int], max_redirects: int) -> None:
        super().__init__()
        self.origin = origin
        self.max_redirects = max_redirects

    def redirect_request(
        self,
        request: Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> Request | None:
        if _origin(new_url) != self.origin:
            raise WebToolError("redirects may not leave the event host")
        count = int(getattr(request, "_event_redirect_count", 0)) + 1
        if count > self.max_redirects:
            raise WebToolError("redirect limit exceeded")
        redirected = super().redirect_request(
            request, file_pointer, code, message, headers, new_url
        )
        if redirected is not None:
            setattr(redirected, "_event_redirect_count", count)
        return redirected


class EventWebSession:
    """A task-local cookie session restricted to the configured event origin."""

    _BLOCKED_HEADERS = {
        "host",
        "content-length",
        "transfer-encoding",
        "connection",
        "proxy-authorization",
        "proxy-connection",
    }

    def __init__(
        self,
        base_url: str,
        *,
        default_headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 30.0,
        max_body_bytes: int = 1_000_000,
        max_request_bytes: int = 1_000_000,
        max_redirects: int = 5,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.origin = _origin(self.base_url)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_body_bytes < 1 or max_request_bytes < 1 or max_redirects < 0:
            raise ValueError("web bounds must be non-negative and bodies positive")
        self.timeout_seconds = float(timeout_seconds)
        self.max_body_bytes = int(max_body_bytes)
        self.max_request_bytes = int(max_request_bytes)
        self.default_headers = self._headers(default_headers or {})
        self.cookies = CookieJar()
        self._opener = build_opener(
            ProxyHandler({}),
            HTTPCookieProcessor(self.cookies),
            _SameOriginRedirects(self.origin, max_redirects),
        )
        self._lock = threading.RLock()

    def _url(self, value: str) -> str:
        if not isinstance(value, str) or not value or len(value) > 8_192:
            raise WebToolError("URL must contain 1 to 8192 characters")
        absolute = urljoin(self.base_url, value)
        if _origin(absolute) != self.origin:
            raise WebToolError("web requests may only target the event host")
        parsed = urlsplit(absolute)
        return urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path or "/", parsed.query, "")
        )

    def _headers(self, headers: Mapping[str, str]) -> dict[str, str]:
        clean: dict[str, str] = {}
        for name, value in headers.items():
            if not isinstance(name, str) or not isinstance(value, str):
                raise WebToolError("header names and values must be strings")
            if (
                not name
                or re.fullmatch(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+", name) is None
                or name.lower() in self._BLOCKED_HEADERS
                or name.lower().startswith("proxy-")
                or any(
                    ord(character) < 32 or ord(character) == 127
                    for character in value
                )
            ):
                raise WebToolError(f"unsafe HTTP header: {name!r}")
            clean[name] = value
        return clean

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        form: Mapping[str, str] | None = None,
        json: Any | None = None,
    ) -> WebResponse:
        method = method.upper()
        if method not in {"GET", "POST"}:
            raise WebToolError("only GET and POST requests are allowed")
        if form is not None and json is not None:
            raise WebToolError("form and json bodies are mutually exclusive")
        if method == "GET" and (form is not None or json is not None):
            raise WebToolError("GET requests may not contain a body")
        request_headers = {
            **self.default_headers,
            **self._headers(headers or {}),
            "Accept-Encoding": "identity",
        }
        body: bytes | None = None
        if form is not None:
            if not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in form.items()
            ):
                raise WebToolError("form keys and values must be strings")
            body = urlencode(form).encode("utf-8")
            request_headers.setdefault(
                "Content-Type", "application/x-www-form-urlencoded"
            )
        elif json is not None:
            try:
                body = json_module.dumps(
                    json, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
            except (TypeError, ValueError) as error:
                raise WebToolError(f"JSON body is not serializable: {error}") from error
            request_headers.setdefault("Content-Type", "application/json")
        if body is not None and len(body) > self.max_request_bytes:
            raise WebToolError("request body exceeds configured byte limit")
        request = Request(
            self._url(url),
            data=body,
            headers=request_headers,
            method=method,
        )
        try:
            with self._lock:
                try:
                    response = self._opener.open(
                        request, timeout=self.timeout_seconds
                    )
                except HTTPError as error:
                    response = error
                with response:
                    response_body = response.read(self.max_body_bytes + 1)
                    final_url = response.geturl()
                    if _origin(final_url) != self.origin:
                        raise WebToolError("response escaped the event host")
                    return WebResponse(
                        status=int(response.status),
                        url=final_url,
                        headers=tuple(response.headers.items()),
                        body=response_body[: self.max_body_bytes],
                        truncated=len(response_body) > self.max_body_bytes,
                    )
        except WebToolError:
            raise
        except (HTTPError, URLError, OSError, TimeoutError) as error:
            raise WebToolError(f"web request failed: {error}") from error

    def get(self, url: str, **kwargs: Any) -> WebResponse:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> WebResponse:
        return self.request("POST", url, **kwargs)


class WebSessionPool:
    """Create exactly one persistent event web session for each task id."""

    def __init__(self, base_url: str, **session_options: Any) -> None:
        self.base_url = base_url
        self.session_options = session_options
        self._sessions: dict[str, EventWebSession] = {}
        self._lock = threading.Lock()

    def for_task(self, task_id: str) -> EventWebSession:
        if not isinstance(task_id, str) or not task_id or len(task_id) > 200:
            raise WebToolError("task id must contain 1 to 200 characters")
        with self._lock:
            session = self._sessions.get(task_id)
            if session is None:
                session = EventWebSession(self.base_url, **self.session_options)
                self._sessions[task_id] = session
            return session
