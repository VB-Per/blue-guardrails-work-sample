"""Each planted failure is caught, the clean trace passes, and the edge cases behave as documented."""
import copy
import json

import pytest

from checks import check_trace, check_trajectory, facts, llm_judge_stub, validate
from run import load_traces, results

TRACES = load_traces()


def found(spans, **kw):
    return {(f.check, f.issue, f.span, f.value) for f in check_trace(spans, **kw)}


def labelled(trace):
    return {(x["check"], x["issue"], x["span"], x["value"]) for x in trace["labels"]}


def spans_of(name):
    return copy.deepcopy(TRACES[name]["spans"])


def by_op(spans, op):
    return [s for s in spans if s["attributes"].get("gen_ai.operation.name") == op]


# ---------- the four traces ----------

@pytest.mark.parametrize("name", sorted(TRACES))
def test_flags_match_the_hand_labels_exactly(name):
    """Every planted failure is caught and nothing else is flagged."""
    assert found(TRACES[name]["spans"]) == labelled(TRACES[name])


def test_clean_trace_passes_every_check():
    assert check_trace(TRACES["01_clean"]["spans"]) == []


def test_claim_check_catches_invented_number_date_and_name():
    issues = {(f.issue, f.value) for f in check_trace(TRACES["02_unsupported_claim"]["spans"])}
    assert issues == {("unsupported_number", "12"), ("unsupported_entity", "Hamburg"),
                      ("unsupported_date", "November 28")}


def test_tool_call_trace_gives_the_three_flags_described_in_the_article():
    issues = {(f.check, f.issue, f.value) for f in check_trace(TRACES["03_wrong_tool_argument"]["spans"])}
    assert issues == {("tool_args", "schema_violation", "include=tracking"),
                      ("tool_args", "ungrounded_argument", "order_id=A-78"),
                      ("trajectory", "answer_after_failed_step", "get_order")}


def test_confirmed_refund_after_timeout_is_caught_only_by_the_trajectory_check():
    assert [f.check for f in check_trace(TRACES["04_built_on_failed_step"]["spans"])] == ["trajectory"]


def test_run_reports_seven_flags_and_one_clean_trace():
    rows = results(TRACES)
    assert sum(r[2] == "FLAG" for r in rows) == 7
    assert {r[0] for r in rows if r[2] == "FLAG"} == {
        "02_unsupported_claim", "03_wrong_tool_argument", "04_built_on_failed_step"}


# ---------- edge cases ----------

def test_answer_that_admits_the_failure_is_not_flagged():
    spans = spans_of("04_built_on_failed_step")
    by_op(spans, "chat")[-1]["attributes"]["gen_ai.output.messages"][0]["parts"] = [
        {"type": "text", "content": "Sorry, the refund for order A-77 did not go through. Please try again later."}]
    assert check_trajectory(spans) == []


def test_empty_tool_result_counts_as_a_failed_step():
    spans = spans_of("01_clean")
    by_op(spans, "execute_tool")[0]["attributes"]["gen_ai.tool.call.result"] = {}
    assert [f.value for f in check_trajectory(spans)] == ["get_order"]


def test_successful_retry_clears_the_failed_step():
    spans = spans_of("04_built_on_failed_step")
    failed = next(s for s in spans if s["attributes"].get("error.type"))
    retry = copy.deepcopy(failed)
    retry["span_id"], retry["start_time"] = "0000000000000001", failed["end_time"]
    del retry["attributes"]["error.type"]
    retry["attributes"]["gen_ai.tool.call.result"] = {"refund_id": "R-1", "status": "issued"}
    assert check_trajectory(spans + [retry]) == []


def test_attributes_exported_as_json_strings_give_the_same_flags():
    spans = spans_of("03_wrong_tool_argument")
    for s in spans:
        s["attributes"] = {k: json.dumps(v) if isinstance(v, (dict, list)) else v
                           for k, v in s["attributes"].items()}
    assert found(spans) == labelled(TRACES["03_wrong_tool_argument"])


def test_schema_subset_catches_missing_wrong_type_and_extra_arguments():
    schema = {"type": "object", "required": ["order_id", "amount"], "additionalProperties": False,
              "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}}}
    errors = {(path, msg) for path, _, msg in validate({"amount": "39.90", "note": "x"}, schema)}
    assert errors == {("order_id", "required argument missing"), ("amount", "expected number, got str"),
                      ("note", "argument not in the schema")}
    assert list(validate({"order_id": "A-77", "amount": 39.9}, schema)) == []
    assert list(validate({"order_id": "A-77", "amount": True}, schema))  # a bool is not a number


def test_fact_extraction_normalizes_dates_and_numbers():
    f = facts("Order A-77: 1,200 units at 39.90 EUR, due Dec 15 or 15 December 2026.", claim=True)
    assert set(f["date"]) == {(12, 15)}
    assert set(f["id"]) == {"A-77"}
    assert set(f["number"]) == {1200.0, 39.9}
    assert set(f["entity"]) == {"eur"}


# ---------- optional LLM judge ----------

def test_llm_judge_is_off_by_default_and_the_stub_refuses_to_run():
    assert all(f.check != "llm_judge" for f in check_trace(TRACES["02_unsupported_claim"]["spans"]))
    with pytest.raises(NotImplementedError):
        check_trace(TRACES["02_unsupported_claim"]["spans"], judge=llm_judge_stub)


def test_judge_hook_gets_answer_and_evidence_and_its_flags_are_reported():
    prompts = []

    def fake_judge(prompt):
        prompts.append(prompt)
        return [{"text": "is in stock", "label": "unsupported_claim",
                 "explanation": "check_stock returned stock 0"}]

    flags = [f for f in check_trace(TRACES["02_unsupported_claim"]["spans"], judge=fake_judge)
             if f.check == "llm_judge"]
    assert [(f.issue, f.value) for f in flags] == [("unsupported_claim", "is in stock")]
    assert '"stock": 0' in prompts[0] and "12 units" in prompts[0]
