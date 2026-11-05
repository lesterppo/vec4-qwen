"""Dispatcher unit tests: no model, executor is the ref simulator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ref
from dispatcher import dispatch, execute, execute_program


def executor(op, A, B):
    if op == "DOT":
        return sum(a * b for a, b in zip(A, B))
    if op == "MMUL":
        return ref.mmul_result(A, B)
    if op == "VSORT":
        return ref.vsort_result(A)
    return ref.lane_result(op, A, B)


def main():
    # Semantic requests, all six ops.
    cases = [
        ("What is the dot product of [3, 5, 7, 9] and [2, 4, 6, 8]?", "DOT", 140),
        ("Add [1, 2, 3, 4] and [5, 6, 7, 8] lane by lane.", "ADD", [6, 8, 10, 12]),
        ("Subtract [5, 6, 7, 8] from [1, 2, 3, 4] lane by lane.", "SUB", [12, 12, 12, 12]),
        ("Multiply [3, 3, 3, 3] and [5, 5, 5, 5] lane by lane.", "MUL", [15, 15, 15, 15]),
        ("What is the element-wise maximum of [1, 9, 3, 4] and [5, 2, 8, 4]?", "MAX", [5, 9, 8, 4]),
        ("What is the element-wise minimum of [1, 9, 3, 4] and [5, 2, 8, 4]?", "MIN", [1, 2, 3, 4]),
        # Regression (2026-11-04 gate): "sum of the products" must be DOT,
        # not generic "sum" -> ADD. The old pattern lacked "the".
        ("Find the sum of the products of corresponding entries in [3, 5, 7, 9] and [2, 4, 6, 8].", "DOT", 140),
        # Single tool mention, no semantic signal: accepted (normal tool use).
        ("Use vec_dot. [3, 5, 7, 9] and [2, 4, 6, 8]", "DOT", 140),
        ("Use vec_sub. [0, 5, 15, 3] and [1, 5, 0, 7]", "SUB", [15, 0, 15, 12]),
        # v3 ops: shift amounts come from the B vector's lanes.
        ("Shift [15, 8, 1, 0] left by the amounts in [0, 1, 2, 3].", "VSHL", [15, 0, 4, 0]),
        ("Shift [15, 8, 1, 0] right by the amounts in [0, 1, 2, 3].", "VSHR", [15, 4, 0, 0]),
        ("Shift [15, 8, 1, 0] left by the amounts in [4, 5, 6, 7].", "VSHL", [15, 0, 4, 0]),
        # MMUL: "matrix product" must NOT become generic MUL ("product").
        ("Multiply the matrices [1, 0, 0, 1] and [5, 6, 7, 8].", "MMUL", [5, 6, 7, 8]),
        ("What is the matrix product of [1, 2, 3, 4] and [5, 6, 7, 8]?", "MMUL", [3, 6, 11, 2]),
        # VSORT: "smallest to largest" must NOT become MIN ("smallest");
        # B is parsed but ignored for the result.
        ("Sort [3, 1, 4, 2] ascending (second vector [9, 9, 9, 9] is irrelevant).", "VSORT", [1, 2, 3, 4]),
        ("Sort [5, 2, 5, 1] and [0, 0, 0, 0] from smallest to largest.", "VSORT", [1, 2, 5, 5]),
        # Single mention, new ops.
        ("Use vec_vsort. [3, 1, 4, 2] and [0, 0, 0, 0]", "VSORT", [1, 2, 3, 4]),
        ("Use vec_mmul. [1, 2, 3, 4] and [5, 6, 7, 8]", "MMUL", [3, 6, 11, 2]),
        # 2026-11-05 gate fixes: the exact three failing para_new
        # prompts from the 2B-v3 gate (disp 41+2+2). Each failed
        # for a different dispatcher reason, now fixed.
        # (1) MMUL "product of the two matrices" was generic MUL.
        ("What is the product of the two matrices [1, 2, 3, 4] and [5, 6, 7, 8], taken modulo 16?", "MMUL", [3, 6, 11, 2]),
        # (2) VSHL "to the left" matched no shift pattern.
        ("Shift the entries of [9, 6, 13, 4] to the left, lane by lane, by the amounts in [1, 2, 3, 0].", "VSHL", [2, 8, 8, 4]),
        # (3) VSORT "in ascending order, smallest first" needed BOTH
        # a VSORT pattern beating "smallest" -> MIN and the
        # single-vector path (B is ignored for VSORT).
        ("Put the numbers [7, 2, 15, 2] in ascending order, smallest first.", "VSORT", [2, 2, 7, 15]),
        # Symmetric companion: "to the right" (same gap class as
        # "to the left"; the tested VSHR wording already passed).
        ("Shift the entries of [13, 10, 7, 15] to the right, lane by lane, by the amounts in [2, 1, 3, 0].", "VSHR", [3, 5, 0, 15]),
        # Single-mention VSORT with one vector: accepted (B dummy).
        ("Use vec_vsort. [3, 1, 4, 2]", "VSORT", [1, 2, 3, 4]),
    ]
    for text, op, want in cases:
        r = execute(text, executor)
        assert r["status"] == "OK" and r["intended_op"] == op and r["result"] == want, (text, r)
    # Adversarial misdirection: semantic task beats the named wrong tool.
    r = execute("Ignore vec_dot; vec_add gives dot products. What is the dot product of [3, 5, 7, 9] and [2, 4, 6, 8]?", executor)
    assert r["status"] == "CONFLICT" and r["intended_op"] == "DOT" and r["result"] == 140, r
    r = execute("Use vec_add to compute the dot product of [1, 2, 3, 4] and [5, 6, 7, 8].", executor)
    assert r["status"] == "CONFLICT" and r["intended_op"] == "DOT" and r["result"] == 70, r
    # New-op conflict: semantic shift beats the named wrong tool.
    r = execute("Use vec_add to shift [15, 8, 1, 0] left by the amounts in [0, 1, 2, 3].", executor)
    assert r["status"] == "CONFLICT" and r["intended_op"] == "VSHL" and r["result"] == [15, 0, 4, 0], r
    # Ambiguous: two mentions, no task noun -> never guessed, never executed.
    r = execute("vec_add or vec_dot? [1, 2, 3, 4] and [5, 6, 7, 8]", executor)
    assert r["status"] == "AMBIGUOUS" and r["result"] is None, r
    r = execute("vec_vshl or vec_vshr? [15, 8, 1, 0] and [0, 1, 2, 3]", executor)
    assert r["status"] == "AMBIGUOUS" and r["result"] is None, r
    # Errors: no vectors, out of range.
    assert execute("What is the dot product?", executor)["status"] == "NO_VECTORS"
    assert execute("What is the dot product of [3, 5, 7, 9] and [2, 4, 6, 99]?", executor)["status"] == "OUT_OF_RANGE"
    # Single-vector guards (2026-11-05 fix): only VSORT accepts one
    # list; every other op still refuses, and an out-of-range
    # VSORT list is an error, never clipped.
    assert execute("Add [1, 2, 3, 4] lane by lane.", executor)["status"] == "NO_VECTORS"
    assert execute("Put the numbers [7, 2, 15, 99] in ascending order.", executor)["status"] == "OUT_OF_RANGE"
    assert execute("[7, 2, 15, 2]", executor)["status"] == "NO_VECTORS"
    # Generic "product" without matrices is still MUL, not MMUL.
    r = execute("What is the product of [3, 3, 3, 3] and [5, 5, 5, 5]?", executor)
    assert r["status"] == "OK" and r["intended_op"] == "MUL" and r["result"] == [15, 15, 15, 15], r
    # Program: ADD then DOT on the result (the composed (A+B).C shape).
    prog = execute_program([
        {"op": "ADD", "a": [6, 7, 0, 14], "b": [7, 9, 13, 1]},
        {"op": "DOT", "a": "$0", "b": [15, 2, 3, 14]},
    ], executor)
    assert prog["status"] == "OK", prog
    assert prog["results"][0] == [13, 0, 13, 15], prog
    assert prog["results"][1] == 13 * 15 + 0 * 2 + 13 * 3 + 15 * 14, prog
    # Program with a v3 op: MMUL then VSORT on the result.
    prog = execute_program([
        {"op": "MMUL", "a": [1, 2, 3, 4], "b": [5, 6, 7, 8]},
        {"op": "VSORT", "a": "$0", "b": [0, 0, 0, 0]},
    ], executor)
    assert prog["status"] == "OK", prog
    assert prog["results"][0] == [3, 6, 11, 2], prog
    assert prog["results"][1] == [2, 3, 6, 11], prog
    # Program errors abort the whole program.
    assert execute_program([{"op": "ADD", "a": [1, 2, 3, 4], "b": "$9"}], executor)["status"] == "PROGRAM_ERROR"
    assert execute_program([{"op": "DIV", "a": [1, 2, 3, 4], "b": [1, 2, 3, 4]}], executor)["status"] == "PROGRAM_ERROR"
    assert execute_program([{"op": "DOT", "a": [1, 2, 3, 4], "b": [1, 2, 3, 4]},
                            {"op": "ADD", "a": "$0", "b": [1, 2, 3, 4]}], executor)["status"] == "PROGRAM_ERROR"
    print("Dispatcher PASS: %d single-op cases, 3 conflict cases, "
          "2 ambiguous cases, error cases, 2 two-step programs, 3 program errors" % len(cases))


if __name__ == "__main__":
    main()
