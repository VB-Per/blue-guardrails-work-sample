# agent-checks

Three cheap checks that find agent hallucinations in a trace, with no gold dataset and no model calls.
They run on four hand-written agent traces shaped like OpenTelemetry GenAI spans: one clean, three with a planted failure at the claim, tool-call and trajectory level.
Companion code for the article [How to detect hallucinations in AI agents in production](../guide.html).

<!-- {{ARTICLE_URL}} is a placeholder: the page build fills in the published article URL. -->

## Quick start

1. Open a terminal in this folder. You need Python 3.11, 3.12 or 3.13 (all tested). The checks use the standard library only: no installs, no API keys.
2. Run every check on every trace: `python run.py` (or `python3 run.py`). The last line should read `7 flags in 3 of 4 traces. Passed every check: 01_clean.` Add `-v` to print why each flag fired.
3. Run the tests: `pip install "pytest>=7"`, then `pytest -q`. Expect `17 passed`.

## Expected output

```text
TRACE                    CHECK       VERDICT  EVIDENCE SPAN     ISSUE                     VALUE
-----------------------  ----------  -------  ----------------  ------------------------  ----------------
01_clean                 tool_args   PASS     -                 -                         -
01_clean                 claims      PASS     -                 -                         -
01_clean                 trajectory  PASS     -                 -                         -
02_unsupported_claim     tool_args   PASS     -                 -                         -
02_unsupported_claim     claims      FLAG     1e9ea827a9b6137d  unsupported_date          November 28
02_unsupported_claim     claims      FLAG     1e9ea827a9b6137d  unsupported_number        12
02_unsupported_claim     claims      FLAG     1e9ea827a9b6137d  unsupported_entity        Hamburg
02_unsupported_claim     trajectory  PASS     -                 -                         -
03_wrong_tool_argument   tool_args   FLAG     dd146fd1b3ea8c90  schema_violation          include=tracking
03_wrong_tool_argument   tool_args   FLAG     dd146fd1b3ea8c90  ungrounded_argument       order_id=A-78
03_wrong_tool_argument   claims      PASS     -                 -                         -
03_wrong_tool_argument   trajectory  FLAG     233022537a66b809  answer_after_failed_step  get_order
04_built_on_failed_step  tool_args   PASS     -                 -                         -
04_built_on_failed_step  claims      PASS     -                 -                         -
04_built_on_failed_step  trajectory  FLAG     182880851334387f  answer_after_failed_step  issue_refund

7 flags in 3 of 4 traces. Passed every check: 01_clean.
```

The tests compare each trace's flags with the hand labels stored in the trace file, so a missed failure and an extra flag both fail. They also cover the edge cases below and the judge hook.

## The traces

Every trace is the same order-support agent with three tools (`get_order`, `check_stock`, `issue_refund`).

| Trace | Level | What was planted | Caught by |
|---|---|---|---|
| `01_clean` | none | Nothing. The answer only repeats the `get_order` result. | passes all checks |
| `02_unsupported_claim` | claim | `check_stock` returns stock 0 in Leipzig, restock 2026-12-15. The answer says 12 units in Hamburg, shipping November 28. | `claims` (3 flags) |
| `03_wrong_tool_argument` | tool call | The test trace from the article. The user asks about A-77; the model calls `get_order(order_id="A-78", include="tracking")`. `tracking` is not in the schema enum, and A-78 appears nowhere before the call. The call fails and the answer says the order is on its way. | `tool_args` (2), `trajectory` (1) |
| `04_built_on_failed_step` | trajectory | `issue_refund` times out; the answer confirms the refund. Every number in it is supported by earlier context, so only the trajectory check fires. | `trajectory` |

## The checks (`checks.py`)

1. **`tool_args`**: validates each `tool_call` against the JSON Schema in `gen_ai.tool.definitions` (type, enum, required, properties, additionalProperties, items). Then provenance: an argument named `*id`, `*number`, `*email`, `*path` or `*sku` must appear in the system instructions, a user message or an earlier tool result.
2. **`claims`**: pulls dates, IDs, numbers and capitalised names out of each answer and looks for each one in what the agent saw at that step. `39.90` matches `39.9`, `November 20` matches `2026-11-20`.
3. **`trajectory`**: links tool calls to results by `gen_ai.tool.call.id`. A call with `error.type` or an empty result is a failed step. An answer after a failed step is flagged unless it says the step failed, or a later call of the same tool succeeded.

The model's own earlier output never counts as evidence, so an ID it invented in step 1 cannot support step 2.

**Optional LLM judge, off by default.** Some errors have nothing to match: in trace 02, "in stock" contradicts `"stock": 0` with no number in the sentence. `check_trace(spans, judge=fn)` sends each answer and its evidence to `fn` as one prompt (`judge_prompt`: one narrow issue type, domain context, flagged spans with a label and explanation). `llm_judge_stub` only raises `NotImplementedError`. Wire in your own model client:

```python
from checks import check_trace

def my_judge(prompt):
    ...  # call a model you have evaluated; return [{"text": ..., "label": ..., "explanation": ...}]

findings = check_trace(spans, judge=my_judge)
```

## Notebook

`notebook.ipynb` walks through the same flow one check at a time, with outputs saved. To re-run it: `pip install nbconvert ipykernel`, then `jupyter nbconvert --to notebook --execute --inplace notebook.ipynb`.

## Trace format

Each file holds one trace: `trace_id`, a `_note` describing the scenario, `labels` (the planted failures, used by the tests) and `spans`. Spans use flattened attributes (`span_id`, `start_time`, `attributes` as a dict), the same shape as the snippet in the article. Raw OTLP JSON stores attributes as key/value lists, so flatten them first. Structured values can be objects or JSON strings; `attr()` reads both.

Attribute names follow the OpenTelemetry GenAI semantic conventions, checked on 27 Nov 2026 against the [semantic-conventions-genai](https://github.com/open-telemetry/semantic-conventions-genai) repository, where [opentelemetry.io](https://opentelemetry.io/docs/specs/semconv/gen-ai/) now points: `gen_ai.operation.name` (`invoke_agent`, `chat`, `execute_tool`), `gen_ai.system_instructions`, `gen_ai.input.messages` and `gen_ai.output.messages` (message parts of type `text`, `tool_call` and `tool_call_response`), `gen_ai.tool.definitions`, `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`, `error.type`. The conventions are at Development status, and every content attribute is opt-in: if your instrumentation does not capture content, these checks have nothing to read.

## Differences from the snippet in the article

- A small built-in schema validator instead of `jsonschema`, so nothing needs installing. Use `jsonschema` for real schemas.
- Evidence leaves out the model's own earlier messages.
- The claim check for numbers, dates, IDs and names is new; the article describes it but the snippet does not include it.
- The trajectory rule links failures by call ID, skips answers that admit the failure, and clears a failure after a successful retry.

## What this does NOT do

- **It is a teaching example, not a detector to deploy.** The patterns are tuned to four synthetic traces. On real traffic expect false positives (a greeting with the user's name, "within 24 hours" from a policy the prompt lacks, a total the model computed correctly) and misses (claims with no number, date, ID or name, a wrong year, paraphrase, conflation, drift from the system prompt).
- **It checks support, not truth.** A claim is "supported" if the agent saw it, even if the tool result was wrong.
- **Real detection needs domain context and an evaluated judge**: what the application does, which sources count, what counts as an issue there, and a verifier measured against human labels on your own traces. The article's validation section covers how.
- **It is not Blue Guardrails' product** and does not show how their detector works. It is an independent example by Vighnesh Suresh Babu, not affiliated with or endorsed by Blue Guardrails.
- **No model produced these traces.** They are hand-written; `demo-model` and `demo` are placeholders.

## Files

```text
checks.py            the three checks and the judge hook
run.py               runs every check on every trace and prints the table
traces/*.json        four traces with hand labels
tests/test_checks.py pytest suite
notebook.ipynb       the same flow, step by step, executed
pytest.ini           lets pytest import checks.py from the repo root
```
