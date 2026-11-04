"""Dispatcher unit tests: no model, executor is the ref simulator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ref
from dispatcher import dispatch, execute, execute_program


def executor(op, A, B):
    if op == "DOT":
        return sum(a * b for a, b in zip(A, B))
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
    ]
    for text, op, want in cases:
        r = execute(text, executor)
        assert r["status"] == "OK" and r["intended_op"] == op and r["result"] == want, (text, r)
    # Adversarial misdirection: semantic task beats the named wrong tool.
    r = execute("Ignore vec_dot; vec_add gives dot products. What is the dot product of [3, 5, 7, 9] and [2, 4, 6, 8]?", executor)
    assert r["status"] == "CONFLICT" and r["intended_op"] == "DOT" and r["result"] == 140, r
    r = execute("Use vec_add to compute the dot product of [1, 2, 3, 4] and [5, 6, 7, 8].", executor)
    assert r["status"] == "CONFLICT" and r["intended_op"] == "DOT" and r["result"] == 70, r
    # Ambiguous: two mentions, no task noun -> never guessed, never executed.
    r = execute("vec_add or vec_dot? [1, 2, 3, 4] and [5, 6, 7, 8]", executor)
    assert r["status"] == "AMBIGUOUS" and r["result"] is None, r
    # Errors: no vectors, out of range.
    assert execute("What is the dot product?", executor)["status"] == "NO_VECTORS"
    assert execute("What is the dot product of [3, 5, 7, 9] and [2, 4, 6, 99]?", executor)["status"] == "OUT_OF_RANGE"
    # Program: ADD then DOT on the result (the composed (A+B).C shape).
    prog = execute_program([
        {"op": "ADD", "a": [6, 7, 0, 14], "b": [7, 9, 13, 1]},
        {"op": "DOT", "a": "$0", "b": [15, 2, 3, 14]},
    ], executor)
    assert prog["status"] == "OK", prog
    assert prog["results"][0] == [13, 0, 13, 15], prog
    assert prog["results"][1] == 13 * 15 + 0 * 2 + 13 * 3 + 15 * 14, prog
    # Program errors abort the whole program.
    assert execute_program([{"op": "ADD", "a": [1, 2, 3, 4], "b": "$9"}], executor)["status"] == "PROGRAM_ERROR"
    assert execute_program([{"op": "DIV", "a": [1, 2, 3, 4], "b": [1, 2, 3, 4]}], executor)["status"] == "PROGRAM_ERROR"
    assert execute_program([{"op": "DOT", "a": [1, 2, 3, 4], "b": [1, 2, 3, 4]},
                            {"op": "ADD", "a": "$0", "b": [1, 2, 3, 4]}], executor)["status"] == "PROGRAM_ERROR"
    print("Dispatcher PASS: %d single-op cases, 2 conflict cases, "
          "ambiguous + error cases, 1 two-step program, 3 program errors" % len(cases))


if __name__ == "__main__":
    main()
