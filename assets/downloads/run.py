"""Run every check on every trace in traces/ and print one table.

    python run.py        table
    python run.py -v     table plus the explanation for each flag
"""
import json
import pathlib
import sys

from checks import CHECKS, check_trace

TRACES = pathlib.Path(__file__).parent / "traces"


def load_traces(folder=TRACES):
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))}


def results(traces):
    """One row per finding, and one PASS row for each check that found nothing."""
    rows = []
    for name, trace in traces.items():
        findings = check_trace(trace["spans"])
        for check in CHECKS:
            hits = [f for f in findings if f.check == check]
            rows += [(name, check, "FLAG", f.span, f.issue, f.value, f.detail) for f in hits]
            rows += [] if hits else [(name, check, "PASS", "-", "-", "-", "")]
    return rows


def table(rows, verbose=False):
    head = ("TRACE", "CHECK", "VERDICT", "EVIDENCE SPAN", "ISSUE", "VALUE")
    widths = [max(len(str(r[i])) for r in [head, *rows]) for i in range(len(head))]
    line = lambda r: "  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip()
    out = [line(head), line(["-" * w for w in widths])]
    for r in rows:
        out.append(line(r[:6]))
        if verbose and r[6]:
            out.append(" " * (widths[0] + 2) + "> " + r[6])
    flags = [r for r in rows if r[2] == "FLAG"]
    clean = sorted({r[0] for r in rows} - {r[0] for r in flags})
    out.append(f"\n{len(flags)} flags in {len({r[0] for r in flags})} of {len({r[0] for r in rows})} traces. "
               f"Passed every check: {', '.join(clean) or 'none'}.")
    return "\n".join(out)


if __name__ == "__main__":
    print(table(results(load_traces()), verbose="-v" in sys.argv))
