# Agent Jeopardy Learnings

`learnings.jsonl` is the append-only machine-readable ledger. `main.py`
records a method summary and submission outcome for every future attempt.

## Operating requirements

- Treat latency as a scoring mechanism. Launch independent high-value tiles
  concurrently; only the submission lane is serialized for the global limit.
- Use Claude Haiku 4.5 through the event Anthropic proxy. The runner rotates
  temperatures `0.0`, `0.25`, and `0.5` across workers by default.
- Prioritize 500-point tiles. The selection queue spreads the initial wave
  across categories, then takes remaining tiles in descending point order.
- Compute an answer into `answer.txt`, validate it locally, and submit that
  exact file content. This avoids a model retyping an otherwise correct token.
- Preserve the event's HTTP session for Dark Web tasks and obey every
  stateful-step requirement in the responses.

## Proven practice approaches

| Category | Method | Outcome |
|---|---|---|
| Heavy Compute 500 | Multi-start nearest-neighbor tour construction, followed by 2-opt improvement and duplicate/distance validation. | Correct |
| The Dark Web 500 | Retain cookies, extract and order required fragments, and include per-step headers during the final stateful walk. | Correct |
| Ancient Scrolls 400 | Parse amendment effective dates, exclude revoked amendments, then select the latest valid amendment at the requested date. | Correct |
| Cryptic 400 | Parse the binary format directly from its specification with `struct`, then confirm the maximum is unique. | Correct |
| Needle in the Haystack 400 | Normalize all documented timestamp formats, split per-user sessions at gaps over 30 minutes, then rank UTC session-start days. | Correct |
| Heavy Compute 400 | Use meet-in-the-middle subset-sum search, enforce the requested subset size, and verify the exact total and index uniqueness. | Correct |

## Constraints observed in practice

- The event accepts only one submission every three seconds. The runner queues
  requests and retries a rate-limited computed answer after the server delay.
- The baseline's single inference call is insufficient. Reliable solves use
  Python for parsing, calculation, and verification, and preserve exact
  computed values through file-based submission.
- Practice submissions are safe for calibration; scored submissions must be
  performed by the hosted agent.
