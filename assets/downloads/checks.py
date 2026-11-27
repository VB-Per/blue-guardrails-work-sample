"""Deterministic hallucination checks for agent traces shaped like OpenTelemetry GenAI spans.

Companion code for the article "How to detect hallucinations in AI agents in production".
A teaching example: standard library only, no API keys, no model calls by default.

Three checks, one per level of the article's table:
  tool_args   Tool-call arguments: JSON Schema subset plus provenance of ID-like values.
  claims      Numbers, dates, IDs and names in answer text that nothing the agent saw supports.
  trajectory  Answers written after a failed or empty tool call, without saying it failed.
Plus an optional LLM-as-judge hook (check_trace(..., judge=fn)) that is off by default.
"""
import json
import re
from collections import namedtuple

Finding = namedtuple("Finding", "check issue span value detail")
CHECKS = ("tool_args", "claims", "trajectory")

# Argument names whose values must come from the conversation, not the model (same rule as the article)
GROUNDED_ARG = re.compile(r"(^|_)(id|number|email|path|sku)$")

MONTHS = "january february march april may june july august september october november december".split()
MONTH = "|".join(MONTHS + [m[:3] for m in MONTHS])
ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})(?!\d)")                                   # 2026-11-28
MONTH_DAY = re.compile(rf"\b({MONTH})\.? (\d{{1,2}})(?:st|nd|rd|th)?(?:,? \d{{4}})?\b", re.I)  # November 28
DAY_MONTH = re.compile(rf"\b(\d{{1,2}}) ({MONTH})\.?(?: \d{{4}})?\b", re.I)                  # 28 November
ID_LIKE = re.compile(r"\b[A-Z]{1,5}-\d+\b")                                                # A-77, K-200
NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*(?!\w)")                                   # 12, 39.90, 1,200
CAPITALIZED = re.compile(r"\b[A-Z][A-Za-z]+\b")                                            # Hamburg, DHL
ADMITS_FAILURE = re.compile(
    r"\b(could not|couldn't|cannot|can't|unable|failed|not able|didn't go through|"
    r"did not go through|not find|no results|try again|went wrong|error)\b", re.I)


# ---------- reading spans ----------

def attr(span, key, default=None):
    """Read an attribute. Exporters often store structured content as a JSON string; accept both."""
    v = span["attributes"].get(key, default)
    return json.loads(v) if isinstance(v, str) and v[:1] in "[{" else v


def parts(messages, kind):
    return [p for m in messages or [] for p in m.get("parts", []) if p.get("type") == kind]


def ordered(spans, op):
    return [s for s in sorted(spans, key=lambda s: s["start_time"])
            if s["attributes"].get("gen_ai.operation.name") == op]


def answers(span):
    """Text the model wrote in this chat span."""
    return [p.get("content", "") for p in parts(attr(span, "gen_ai.output.messages", []), "text")]


def evidence(span):
    """Everything the model saw at this step except its own earlier output:
    system instructions, user (and system) messages, and tool results."""
    system = attr(span, "gen_ai.system_instructions", []) or []
    chunks = [system] if isinstance(system, str) else [p.get("content", "") for p in system]
    for m in attr(span, "gen_ai.input.messages", []) or []:
        if m.get("role") == "assistant":
            continue  # the model's own earlier words are not evidence for its later words
        for p in m.get("parts", []):
            if p.get("type") == "text":
                chunks.append(p.get("content", ""))
            elif p.get("type") == "tool_call_response":
                chunks.append(json.dumps(p.get("response"), ensure_ascii=False))
    return "\n".join(str(c) for c in chunks)


def mentioned(value, text):
    return re.search(rf"(?<![\w-]){re.escape(value)}(?![\w-])", text) is not None


# ---------- 1. tool arguments ----------

TYPES = {"object": dict, "array": list, "string": str, "integer": int,
         "number": (int, float), "boolean": bool, "null": type(None)}


def is_type(value, t):
    return isinstance(value, TYPES[t]) and not (isinstance(value, bool) and t in ("integer", "number"))


def validate(value, schema, path=""):
    """Small JSON Schema subset: type, enum, required, properties, additionalProperties, items.
    Yields (path, bad_value, message). Use the jsonschema package for full schemas."""
    types = schema.get("type")
    types = [types] if isinstance(types, str) else types or []
    if types and not any(is_type(value, t) for t in types):
        yield path, value, f"expected {'/'.join(types)}, got {type(value).__name__}"
        return
    if "enum" in schema and value not in schema["enum"]:
        yield path, value, f"{value!r} is not one of {schema['enum']}"
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                yield f"{path}.{key}".lstrip("."), "(missing)", "required argument missing"
        for key, v in value.items():
            if key in props:
                yield from validate(v, props[key], f"{path}.{key}".lstrip("."))
            elif schema.get("additionalProperties") is False:
                yield f"{path}.{key}".lstrip("."), v, "argument not in the schema"
    if isinstance(value, list) and "items" in schema:
        for i, v in enumerate(value):
            yield from validate(v, schema["items"], f"{path}[{i}]")


def check_tool_args(spans):
    out = []
    for s in ordered(spans, "chat"):
        tools = {t.get("name"): t for t in attr(s, "gen_ai.tool.definitions", []) or []}
        seen = evidence(s)
        for call in parts(attr(s, "gen_ai.output.messages", []), "tool_call"):
            name, args = call.get("name"), call.get("arguments") or {}
            args = json.loads(args) if isinstance(args, str) else args
            if name not in tools:
                out.append(Finding("tool_args", "unknown_tool", s["span_id"], name,
                                   "tool is not in gen_ai.tool.definitions for this call"))
                continue
            for path, bad, msg in validate(args, tools[name].get("parameters") or {}):
                out.append(Finding("tool_args", "schema_violation", s["span_id"],
                                   f"{path}={bad}", f"{name}: {msg}"))
            for k, v in (args.items() if isinstance(args, dict) else []):
                if GROUNDED_ARG.search(k) and not mentioned(str(v), seen):
                    out.append(Finding("tool_args", "ungrounded_argument", s["span_id"], f"{k}={v}",
                                       f"{name}: {v!r} appears in no instruction, user message or tool result"))
    return out


# ---------- 2. claims against tool output ----------

def month(name):
    return [m[:3] for m in MONTHS].index(name[:3].lower()) + 1


def facts(text, claim=False):
    """Pull dates, IDs, numbers and names out of text. Returns {kind: {normalized: surface}}.
    Dates compare on (month, day); a fuller version would also compare years."""
    found = {"date": {}, "id": {}, "number": {}, "entity": {}}

    def take(kind, key):
        def sub(m):
            found[kind].setdefault(key(m), m[0])
            return " "  # mask it so digits in dates and IDs are not read as numbers
        return sub

    text = ISO_DATE.sub(take("date", lambda m: (int(m[2]), int(m[3]))), text)
    text = MONTH_DAY.sub(take("date", lambda m: (month(m[1]), int(m[2]))), text)
    text = DAY_MONTH.sub(take("date", lambda m: (month(m[2]), int(m[1]))), text)
    text = ID_LIKE.sub(take("id", lambda m: m[0]), text)
    text = NUMBER.sub(take("number", lambda m: float(m[0].replace(",", ""))), text)
    if claim:  # names: capitalized words that do not start a sentence
        for sentence in re.split(r"(?<=[.!?:;])\s+", text.strip()):
            lead = re.match(r"\W*", sentence).end()
            for m in CAPITALIZED.finditer(sentence):
                if m.start() > lead:
                    found["entity"].setdefault(m[0].lower(), m[0])
    else:  # evidence side: every word counts as support
        found["entity"] = {w: w for w in re.findall(r"[a-z0-9]+", text.lower())}
    return found


def check_claims(spans):
    out = []
    for s in ordered(spans, "chat"):
        texts = answers(s)
        if not texts:
            continue
        support = facts(evidence(s))
        for text in texts:
            for kind, items in facts(text, claim=True).items():
                saw = "" if kind == "entity" else "; agent saw: " + (", ".join(support[kind].values()) or "none")
                for key, surface in items.items():
                    if key not in support[kind]:
                        detail = f"{kind} {surface!r} is in the answer but in nothing the agent saw{saw}"
                        out.append(Finding("claims", f"unsupported_{kind}", s["span_id"], surface, detail))
    return out


# ---------- 3. trajectory ----------

def check_trajectory(spans):
    """Flag answers written after a failed or empty tool call that do not admit the failure.
    A candidate for the verifier, not a verdict: the answer may not rely on the failed call."""
    out, failed = [], {}  # gen_ai.tool.call.id -> (tool, span_id, reason)
    for s in sorted(spans, key=lambda s: s["start_time"]):
        a, op = s["attributes"], s["attributes"].get("gen_ai.operation.name")
        if op == "execute_tool":
            tool, result = a.get("gen_ai.tool.name"), attr(s, "gen_ai.tool.call.result")
            if a.get("error.type") or result in (None, "", [], {}):
                reason = a.get("error.type") or "empty result"
                failed[a.get("gen_ai.tool.call.id")] = (tool, s["span_id"], reason)
            else:  # a later successful call of the same tool is treated as a retry that fixed it
                failed = {k: v for k, v in failed.items() if v[0] != tool}
        if op != "chat":
            continue
        for text in answers(s):
            if failed and not ADMITS_FAILURE.search(text):
                for call_id, (tool, span_id, reason) in failed.items():
                    out.append(Finding("trajectory", "answer_after_failed_step", s["span_id"], tool,
                                       f"answer follows {tool} call {call_id} ({reason}, span {span_id}) "
                                       "and does not say it failed"))
    return out


# ---------- optional LLM judge (off by default) ----------

JUDGE_RULE = ("unsupported_claim: the answer states a fact (number, date, name, status or outcome) that "
              "conflicts with, or is missing from, the evidence the agent had at this step.")
JUDGE_DOMAIN = "Customer support agent for a home goods shop. Sources: order, stock and refund tools."


def judge_prompt(answer, evidence_text, rule=JUDGE_RULE, domain=JUDGE_DOMAIN):
    """Narrow rule + domain context + span-level output, as the article recommends."""
    label = rule.split(":")[0]
    return (f"Domain: {domain}\nIssue type: {rule}\n\n"
            f"Evidence (everything the agent saw at this step):\n{evidence_text}\n\n"
            f"Answer to check:\n{answer}\n\n"
            f'Return a JSON list: [{{"text": "<exact flagged span>", "label": "{label}", '
            '"explanation": "<what the evidence says instead>"}], or [] if nothing breaks the rule.')


def llm_judge_stub(prompt):
    """Stub. Replace with a call to a model you have evaluated against human labels.
    Must return a list of {"text", "label", "explanation"} dicts."""
    raise NotImplementedError("No LLM judge is wired up. Pass judge=your_function to check_trace().")


def check_trace(spans, judge=None):
    """Run all deterministic checks on one trace. judge=None (default) makes no model calls."""
    findings = check_tool_args(spans) + check_claims(spans) + check_trajectory(spans)
    if judge:
        for s in ordered(spans, "chat"):
            for text in answers(s):
                for item in judge(judge_prompt(text, evidence(s))):
                    findings.append(Finding("llm_judge", item["label"], s["span_id"],
                                            item["text"], item["explanation"]))
    return findings
