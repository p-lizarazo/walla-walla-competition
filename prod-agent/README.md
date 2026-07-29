# Production Agent

This directory contains the unattended Agent Jeopardy runner. It is designed
to poll the live board, rank all open variants, solve independent tasks in
parallel, verify candidates, and serialize submissions.

## Architecture

- `main.py` coordinates phase polling, prioritization, worker pools,
  verification, cooldowns, and the single submission lane.
- `config.py` validates environment controls and exposes an immutable
  `Config`.
- `models.py` defines board, task, candidate, evidence, confidence, priority,
  and attempt-state records.
- `game_client.py` owns authenticated board/task/file/submission requests and
  provides a separate HTTP session per worker thread.
- `scheduler.py` ranks every open variant by expected net points per second,
  combining calibrated accuracy, the 25% wrong-answer penalty, solve latency,
  race survival, cooldowns, and a bounded initial 500-point boost.
- `solver.py` is the Anthropic SDK tool loop. It always requests
  `claude-haiku-4-5`; 400/500-point tiles can receive bounded extended-thinking
  budgets while lower tiers stay latency-first.
- `tools/` provides bounded file/search/archive access, resource-limited Python,
  persistent event-host-only HTTP sessions, safe status reads, and candidate
  evidence recording.
- `playbooks.py` loads generic methods from `playbooks.json`. Selection
  combines category, value tier, and an explicit or inferred task signature.
- `submission.py` is the only path to the answer API. Solver workers only
  produce candidates.

Computed answers should move directly from deterministic code into the
submission lane. Do not place raw answers or secrets in logs, playbooks, or
model summaries.

## Modes

`AGENT_MODE=practice_eval` runs calibration on the practice board. Use it to
measure solve rate, elapsed time, confidence calibration, tool failures, and
cooldown behavior without point penalties. Practice still uses the global
submission interval and should exercise the same solver and verifier used in
competition. It also appends one compact result for every attempted tile to
`practice_results.jsonl`, including failed, skipped, and unsubmitted attempts.

`AGENT_MODE=scored` is the default competition mode. It observes whichever
scored board is currently playable, prioritizes expected net value and race survival, and
submits only candidates that pass the configured confidence threshold.
Scored errors cost points and trigger increasing per-tile cooldowns, so a
model-produced guess is not a candidate.

## Configuration and controls

Required:

```bash
export JEOPARDY_BASE_URL=https://event-host
export TEAM_API_KEY=...
```

The model proxy defaults to `$JEOPARDY_BASE_URL/anthropic`, and its API key
defaults to the team key. Useful controls from `Config.from_env()`:

| Variable | Default | Purpose |
|---|---:|---|
| `AGENT_MODE` | `auto` | `auto`, `scored`, or `practice_eval` |
| `WORKERS` | `6` | Concurrent solver workers |
| `VERIFIER_WORKERS` | `1` | Independent verifier capacity for uncertain 400/500s |
| `CPU_WORKERS` | `2` | Bounded concurrent local-compute capacity |
| `MAX_TURNS` | `6` | Model/tool turns for 100–300-point attempts |
| `MAX_TURNS_400` | `10` | Model/tool turns for 400-point attempts |
| `MAX_TURNS_500` | `12` | Model/tool turns for 500-point attempts |
| `MAX_TOKENS` | `1536` | Output ceiling for 100–300-point attempts |
| `MAX_TOKENS_400` | `3072` | Output ceiling for 400-point attempts |
| `MAX_TOKENS_500` | `4096` | Output ceiling for 500-point attempts |
| `THINKING_ENABLED` | `0` | Enable Haiku 4.5 extended thinking on high tiers |
| `THINKING_MIN_POINTS` | `400` | Lowest tier that receives thinking |
| `THINKING_BUDGET_400` | `1024` | 400-point thinking budget |
| `THINKING_BUDGET_500` | `1536` | 500-point thinking budget |
| `MAX_TOOL_OUTPUT` | `6000` | Tool output admitted to model context |
| `PYTHON_TIMEOUT_SECONDS` | `60` | Local computation timeout |
| `PYTHON_MEMORY_MB` | `1024` | Local computation memory budget |
| `BOARD_POLL_SECONDS` | `1.5` | Board refresh and cancellation cadence |
| `SUBMISSION_INTERVAL_SECONDS` | `3.1` | Global submission spacing |
| `STRONG_CONFIDENCE_THRESHOLD` | `0.90` | Normal submit threshold |
| `URGENT_CONFIDENCE_FLOOR` | `0.80` | Lowest allowed urgent threshold |
| `TEMPERATURES` | `0.0,0.15,0.3` | Worker inference temperatures |
| `PLAYBOOKS_PATH` | `playbooks.json` | Generic method catalog |

Keep secrets in injected environment variables, never in `.env` files shipped
with the agent. Tune concurrency against the shared model rate and the
container's two-CPU/two-GB limits.

Parallelism is width-first: six solver workers attack different tiles, while
one verifier worker is reserved for a second Haiku pass at another temperature
when an uncertain 400/500 misses the confidence gate. Two local Python slots
allow useful compute overlap without letting subprocesses consume the whole
container. A 1.5-second board refresh cooperatively cancels queued work and
running solvers between model/tool calls when another team claims their tile.
Lower tiers use small model budgets for throughput; 400s and 500s receive
progressively larger turn and token ceilings.

## Lightweight practice results

Practice calibration needs outcomes, not a production telemetry subsystem. In
`practice_eval` mode, serialize one append per completed attempt behind a
single lock. Each line is an independent JSON object, so an interrupted run
keeps all completed records:

```json
{"timestamp":"...","task_id":"...","category":"...","points":400,"signature":"binary-records","playbook_version":1,"temperature":0.0,"elapsed_seconds":12.3,"tool_turns":5,"confidence":0.94,"submitted":true,"result":"correct","answer_sha256":"...","failure_stage":null,"error_type":null}
```

Use the field list in `playbooks.json` as the compact schema. Record an
outcome for every attempt, not only correct submissions. For failures,
`result` can be `incorrect`, `unavailable`, `unverified`, `tool_error`,
`model_error`, or `exception`, with a short class-like `error_type`.

Never record the raw answer, prompt, tool input/output, response body, cookies,
headers, or environment values. An answer SHA-256 digest is safe and useful
for detecting repeated candidates. Disable this practice ledger in scored
mode, and do not include `practice_results.jsonl` in the deployment zip.

## Evaluation workflow

1. Start with `AGENT_MODE=practice_eval` and a small worker count.
2. Exercise every category and value tier, including both variants in stacked
   cells.
3. Analyze `practice_results.jsonl` for every attempted tile, and use dashboard
   logs only to diagnose individual failures.
4. Compare correct/incorrect rates by category, tier, method, confidence band,
   temperature, elapsed time, and tool failure.
5. Replay failures with the same files, improve generic parsing or
   verification, and add only reusable methods to `playbooks.json`.
6. Before scored mode, verify that logs contain no raw answer or secret and
   that every accepted candidate has direct provenance plus an independent
   check.

Do not copy practice outputs into playbooks. The useful artifact is the
general method: state preservation, timestamp normalization, structured binary
parsing, temporal document resolution, bounded optimization, or targeted code
repair.

## Packaging and deployment

Validate the production code with:

```bash
python -m pytest prod-agent/tests -q
python ops/build_agent.py                # writes ./agent.zip
python ops/validate_agent.py agent.zip   # proves the runner can boot it
```

Artifact creation and deployment are owned by the repository ops pipeline.
Keep runtime modules, tests, practice results, and credentials separated so
that pipeline can package only the production allowlist.

`ops/build_agent.py` packages the runtime tree, flattening `main.py` to the
archive root, and excludes tests, evals, caches, dot-files, `.env` files, and
practice ledgers.

`ops/validate_agent.py` then checks the archive's *behavior*: it extracts the
zip to a temporary directory and imports every shipped module, plus `main.py`
itself, with nothing but the extracted copy on `sys.path`. That is what
catches a module that was refactored away while something still imports it —
a failure that otherwise appears only as a `ModuleNotFoundError` in
`/api/agent/logs`, after the deploy has already replaced a working agent.

The `agent` GitHub Actions workflow (`.github/workflows/agent.yml`) runs those
three steps on every push, on Python 3.12 with the hosted image's packages.

It then uploads the validated archive's *contents* as the artifact named
`agent`. That indirection is deliberate: GitHub re-zips every artifact on
download, so uploading `agent.zip` itself would hand you a wrapper zip
containing `agent.zip`, and submitting that wrapper puts `agent.zip` — not
`main.py` — at the archive root, which fails to deploy. Uploading the contents
makes GitHub's wrapper the real archive, so the run summary's **`agent`**
artifact downloads as `agent.zip` with `main.py` at its root.

Deploy by dragging that downloaded `agent.zip` onto the dashboard, or:

```bash
curl -X POST "$JEOPARDY_BASE_URL/api/agent/submit" \
  -H "X-Api-Key: $TEAM_API_KEY" -F file=@agent.zip
```

Uploading deploys and restarts immediately. Dependencies in
`requirements.txt` install at submission time; the running container has no
general outbound internet.

## Dashboard monitoring

Keep the event dashboard open during practice and scored rounds. Monitor agent
status, restarts, live logs, solved tiles, score, model usage, and rate-limit
pressure. Human summaries should make phase transitions, queue pressure,
confidence decisions, submissions, and cooldowns visible without exposing
task answers.

The equivalent endpoints are:

- `GET /api/me` for agent state and model usage.
- `GET /api/agent/status` for deployment health.
- `GET /api/agent/logs?tail=500` for recent container output.
- `POST /api/agent/start` and `POST /api/agent/stop` for lifecycle control.

On a crash or sustained error loop, inspect the latest dashboard lines, make a
narrow fix, rerun targeted practice evaluation, rebuild the zip, and redeploy.
A submitted answer must always originate from the hosted agent during scored
rounds.
