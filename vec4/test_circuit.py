"""Stage 2 (IR part): circuit vs reference, token-for-token, via ir.evaluate.

No torch, no HF model here: this checks the compiled graph itself.
For each case, evaluate input + expected output through the graph
(check=True: exactness enforced), then decode the exit code at every
owned position (GO and output tokens) and require it to predict the
next token exactly. Also check lane-sum and dot bits at GO, and
result-head bits at output positions, against the reference values.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "upstream"))
import numpy as np

import ref
from circuit import KAPPA, build_vec4
from chipformer.q35.ir import evaluate

KAPPA_F = float(KAPPA)


def dim_of(c, expr):
    (d,) = expr.terms
    return d


def decode_next(m, c, pos):
    """Decode exit code at position pos -> (flag, index) or None if unowned."""
    code_dims = [dim_of(c, e) for e in c.code]
    vals = [float(m[pos, d]) for d in code_dims]
    if all(v == 0.0 for v in vals):
        return None
    bits = []
    for v in vals:
        if abs(abs(v) - KAPPA_F) > 1e-12:
            raise AssertionError(f"pos {pos}: code dim value {v!r} is not +-KAPPA")
        bits.append(1 if v > 0 else 0)
    flag = bits[0]
    index = sum(b << j for j, b in enumerate(bits[1:]))
    return flag, index


def index_to_token(flag, index):
    if flag != 1:
        raise AssertionError(f"flag {flag} != 1 at owned position")
    if index < 64:
        return 131 + index
    if index < 112:
        return 195 + (index - 64)
    if index == 112:
        return 243
    raise AssertionError(f"invalid output index {index}")


def check_case(c, op, A, B):
    inp = ref.encode(op, A, B)
    exp_out = ref.expected_output(op, A, B)
    seq = inp + exp_out
    m = evaluate(c, seq, check=True)
    # owned positions: GO (pos 9) and each output token except END
    for pos in range(len(seq) - 1):
        tok = seq[pos]
        owned = (tok == 130) or (131 <= tok <= 242)
        dec = decode_next(m, c, pos)
        if owned:
            if dec is None:
                raise AssertionError(f"{op} {A} {B} pos {pos}: owned but code is 0")
            nxt = index_to_token(*dec)
            if nxt != seq[pos + 1]:
                raise AssertionError(
                    f"{op} {A} {B} pos {pos} ({ref.describe(tok)}): "
                    f"circuit predicts {ref.describe(nxt)}, want {ref.describe(seq[pos + 1])}")
        else:
            # unowned positions are not checked for next token (host teacher-forces)
            pass
    # GO-position compute bits
    go = 9
    if op == "ADD":
        for lane in range(4):
            want = (A[lane] + B[lane]) % 16
            for j in range(4):
                got = float(m[go, dim_of(c, c.lane_sum[lane][j])])
                if got != ((want >> j) & 1):
                    raise AssertionError(f"ADD lane {lane} bit {j}: got {got}, want {(want >> j) & 1}")
    else:
        dot = sum(a * b for a, b in zip(A, B))
        for j in range(10):
            got = float(m[go, dim_of(c, c.dot_bits[j])])
            if got != ((dot >> j) & 1):
                raise AssertionError(f"DOT bit {j}: got {got}, want {(dot >> j) & 1}")
    # output-position result-head bits (first output position checks all lanes/dot)
    out_pos = 10
    if op == "ADD":
        for lane in range(4):
            want = (A[lane] + B[lane]) % 16
            for j in range(4):
                got = float(m[out_pos, dim_of(c, c.lane_res[lane][j])])
                if got != ((want >> j) & 1):
                    raise AssertionError(f"res lane {lane} bit {j}: got {got}, want {(want >> j) & 1}")
    else:
        dot = sum(a * b for a, b in zip(A, B))
        for j in range(10):
            got = float(m[out_pos, dim_of(c, c.dot_res[j])])
            if got != ((dot >> j) & 1):
                raise AssertionError(f"res dot bit {j}: got {got}, want {(dot >> j) & 1}")
    return True


def main():
    c = build_vec4()
    print("circuit stats:", c.stats())
    print("dims:", len(c.dims), "layers:", c.n_layers())
    cases = []
    cases += [("ADD", [0, 0, 0, 0], [0, 0, 0, 0]),
              ("ADD", [15, 15, 15, 15], [15, 15, 15, 15]),
              ("ADD", [15, 0, 8, 7], [1, 15, 8, 9]),
              ("ADD", [0, 1, 2, 3], [15, 14, 13, 12]),
              ("DOT", [0, 0, 0, 0], [0, 0, 0, 0]),
              ("DOT", [15, 15, 15, 15], [15, 15, 15, 15]),
              ("DOT", [1, 0, 0, 0], [1, 0, 0, 0]),
              ("DOT", [15, 0, 0, 0], [15, 0, 0, 0]),
              ("DOT", [3, 5, 7, 9], [2, 4, 6, 8]),
              ("DOT", [0, 15, 0, 15], [15, 0, 15, 0])]
    rng = random.Random(20261103)
    for _ in range(100):
        cases.append(("ADD", [rng.randrange(16) for _ in range(4)],
                      [rng.randrange(16) for _ in range(4)]))
    for _ in range(100):
        cases.append(("DOT", [rng.randrange(16) for _ in range(4)],
                      [rng.randrange(16) for _ in range(4)]))
    n_add = sum(1 for x in cases if x[0] == "ADD")
    n_dot = sum(1 for x in cases if x[0] == "DOT")
    for i, (op, A, B) in enumerate(cases):
        try:
            check_case(c, op, A, B)
        except Exception as e:
            print(f"FAIL case {i}: {op} {A} {B}: {type(e).__name__}: {e}")
            raise
    print(f"Stage 2 (IR) PASS: {len(cases)} cases ({n_add} VADD, {n_dot} VDOT), "
          f"0 mismatches, exactness checks passed")


if __name__ == "__main__":
    main()
