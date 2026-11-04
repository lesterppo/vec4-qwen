"""Stage 1: reference simulator tests.

Expected values here are written out by hand / by direct Python
expressions in this file, NOT by calling ref.py's arithmetic helpers
for the answer (encode/expected_output are the code under test; the
checks below recompute from the raw A/B lists with plain operators).
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ref


def check(op, A, B):
    toks = ref.encode(op, A, B)
    assert len(toks) == 10, toks
    assert toks[0] == (0 if op == "ADD" else 1)
    assert toks[-1] == 130
    # decode the VAL tokens back, independently of ref's helpers
    A2, B2 = [], []
    for t in toks[1:9]:
        r, v = (t - 2) // 16, (t - 2) % 16
        (A2 if r < 4 else B2).append(v)
    assert A2 == A and B2 == B, (A2, B2, A, B)
    out = ref.expected_output(op, A, B)
    if op == "ADD":
        exp = []
        for i in range(4):
            exp.append(131 + i * 16 + ((A[i] + B[i]) % 16))
        exp.append(243)
    else:
        dot = A[0] * B[0] + A[1] * B[1] + A[2] * B[2] + A[3] * B[3]
        exp = [195 + 0 * 16 + (dot // 256),
               195 + 1 * 16 + ((dot // 16) % 16),
               195 + 2 * 16 + (dot % 16),
               243]
    assert out == exp, (out, exp, op, A, B)
    return True


def main():
    cases = []
    # edges, hand-written
    cases += [("ADD", [0, 0, 0, 0], [0, 0, 0, 0]),
              ("ADD", [15, 15, 15, 15], [15, 15, 15, 15]),
              ("ADD", [15, 0, 8, 7], [1, 15, 8, 9]),
              ("ADD", [0, 1, 2, 3], [15, 14, 13, 12]),
              ("DOT", [0, 0, 0, 0], [0, 0, 0, 0]),
              ("DOT", [15, 15, 15, 15], [15, 15, 15, 15]),  # dot 900 = 0x384
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
    n_add = sum(1 for c in cases if c[0] == "ADD")
    n_dot = sum(1 for c in cases if c[0] == "DOT")
    for op, A, B in cases:
        check(op, A, B)
    # spot-check the max-dot encoding explicitly (900 = 0x384)
    assert ref.expected_output("DOT", [15, 15, 15, 15], [15, 15, 15, 15]) == \
        [195 + 3, 195 + 16 + 8, 195 + 32 + 4, 243]
    print(f"Stage 1 PASS: {len(cases)} cases ({n_add} VADD, {n_dot} VDOT), "
          f"including edges, all match hand-computed expectations")


if __name__ == "__main__":
    main()
