"""Local replay of the Colab disp config, with the ref executor
standing in for the chip. Replicates tool_v3.py's disp outcome logic
exactly (status OK/CONFLICT + intended op + vectors must match, else
refused/wrong), so a local run predicts the Colab disp outcome
whenever the chip is correct -- which Gate 0 proves it is.

Groups: paraphrase 12, adversarial 10, long_context 4 (from
vec4-v2/generalisation.json), batch 10 (DOT template text),
program 4, para_new 4 (from tool_v3.PARA_NEW), program_new 1
(MMUL -> VSORT). Total 45 rows, same as the Colab disp file.

Usage: python3 replay_disp_local.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "upstream"))
sys.path.insert(0, str(ROOT / "vec4-v2"))
sys.path.insert(0, str(HERE))  # HERE last => first on path: vec4-v3/ref, not v2/ref

import ref
from dispatcher import dispatch, execute_program
from tool_v3 import PARA_NEW, expected_value


def executor(op, A, B):
    if op == "DOT":
        return sum(a * b for a, b in zip(A, B))
    if op == "MMUL":
        return ref.mmul_result(A, B)
    if op == "VSORT":
        return ref.vsort_result(A)
    return ref.lane_result(op, A, B)


def main():
    gen = json.loads((ROOT / "vec4-v2" / "generalisation.json").read_text())
    rows = []
    for group in ("paraphrase", "adversarial", "long_context"):
        for idx, c in enumerate(gen[group]):
            r = dispatch(c["text"])
            if (r["status"] in ("OK", "CONFLICT")
                    and r["intended_op"] == c["op"]
                    and list(r["vectors"]) == [c["A"], c["B"]]):
                val = executor(c["op"], c["A"], c["B"])
                outcome = "executed_correct" if val == c["expected"] else "ref_wrong"
            elif r["status"] == "AMBIGUOUS":
                outcome = "refused"
            else:
                outcome = "wrong_intent_or_vectors"
            rows.append((group, idx, outcome, c["op"], r["status"], r["intended_op"]))
    for idx, req in enumerate(gen["batch"]):
        text = f"What is the dot product of {req['A']} and {req['B']}?"
        r = dispatch(text)
        val = executor("DOT", req["A"], req["B"]) if (
            r["status"] == "OK" and r["intended_op"] == "DOT") else None
        rows.append(("batch", idx,
                     "executed_correct" if val == req["expected"] else "failed",
                     "DOT", r["status"], r["intended_op"]))
    for idx, prog in enumerate(gen["program"]):
        r = execute_program(prog["steps"], executor)
        rows.append(("program", idx,
                     "executed_correct" if r.get("results") == prog["results"] else "failed",
                     "program", r["status"], None))
    for idx, c in enumerate(PARA_NEW):
        r = dispatch(c["text"])
        exp = expected_value(c["op"], c["A"], c["B"])
        if (r["status"] in ("OK", "CONFLICT")
                and r["intended_op"] == c["op"]
                and list(r["vectors"]) == [c["A"], c["B"]]):
            val = executor(c["op"], c["A"], c["B"])
            outcome = "executed_correct" if val == exp else "ref_wrong"
        elif r["status"] == "AMBIGUOUS":
            outcome = "refused"
        else:
            outcome = "wrong_intent_or_vectors"
        rows.append(("para_new", idx, outcome, c["op"], r["status"], r["intended_op"]))
    prog_steps = [{"op": "MMUL", "a": [1, 2, 3, 4], "b": [5, 6, 7, 8]},
                  {"op": "VSORT", "a": "$0", "b": [0, 0, 0, 0]}]
    prog_exp = [ref.mmul_result([1, 2, 3, 4], [5, 6, 7, 8])]
    prog_exp.append(ref.vsort_result(prog_exp[0]))
    r = execute_program(prog_steps, executor)
    rows.append(("program_new", 0,
                 "executed_correct" if r.get("results") == prog_exp else "failed",
                 "program", r["status"], None))
    for group in sorted(set(g for g, _, _, _, _, _ in rows)):
        sub = [x for x in rows if x[0] == group]
        print(f"{group}: {Counter(x[2] for x in sub)}")
    print(f"TOTAL {len(rows)}: {Counter(x[2] for x in rows)}")
    bad = [x for x in rows if x[2] not in ("executed_correct", "refused")]
    if bad:
        print("NON-CLEAN ROWS:")
        for x in bad:
            print("  ", x)
    refused = [x for x in rows if x[2] == "refused"]
    for x in refused:
        print("refused (expected only for stated-wrong-answer ADD):", x)


if __name__ == "__main__":
    main()
