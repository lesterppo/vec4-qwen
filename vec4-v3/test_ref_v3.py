"""Stage 1 (v3): reference simulator sanity, pure Python, no chipformer.

Checks, independently of the circuit: encoding shape, op tokens, and
that decoding expected_output recovers hand-computed results for
every op, including the required edges: VSHL/VSHR with B = 0, 3, 4
(amount wraps via &3: B=4 shifts by 0), 15; MMUL identity, zero,
all-15s (mod-16 wrap); VSORT sorted, reverse, all equal,
duplicates; plus VSORT B-independence (same A, different B -> same
output). Random cases are recomputed INLINE here (not by calling
ref.lane_result/mmul_result/vsort_result), so the random sweep is
an independent check of ref.py, not a self-comparison.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ref


def decode_lane_out(tok):
    assert 131 <= tok <= 194
    return (tok - 131) // 16, (tok - 131) % 16


def decode_dot_out(toks):
    assert len(toks) == 3
    hi = (toks[0] - 195) % 16
    mid = (toks[1] - 195) % 16
    lo = (toks[2] - 195) % 16
    return (hi << 8) | (mid << 4) | lo


def inline_result(op, A, B):
    """Independent recomputation, written from the spec, not ref.py."""
    if op == "ADD":
        return [(a + b) % 16 for a, b in zip(A, B)]
    if op == "SUB":
        return [(a - b) % 16 for a, b in zip(A, B)]
    if op == "MUL":
        return [(a * b) % 16 for a, b in zip(A, B)]
    if op == "MAX":
        return [a if a >= b else b for a, b in zip(A, B)]
    if op == "MIN":
        return [a if a <= b else b for a, b in zip(A, B)]
    if op == "VSHL":
        return [(a << (b % 4)) % 16 for a, b in zip(A, B)]
    if op == "VSHR":
        return [a // (2 ** (b % 4)) for a, b in zip(A, B)]
    if op == "MMUL":
        a00, a01, a10, a11 = A
        b00, b01, b10, b11 = B
        return [(a00 * b00 + a01 * b10) % 16,
                (a00 * b01 + a01 * b11) % 16,
                (a10 * b00 + a11 * b10) % 16,
                (a10 * b01 + a11 * b11) % 16]
    if op == "VSORT":
        # insertion sort, written out (stable by construction)
        out = list(A)
        for i in range(1, 4):
            x = out[i]
            j = i - 1
            while j >= 0 and out[j] > x:
                out[j + 1] = out[j]
                j -= 1
            out[j + 1] = x
        return out
    raise ValueError(op)


def check_lane_case(op, A, B, want):
    toks = ref.encode(op, A, B)
    assert len(toks) == 10 and toks[0] == ref.OP_TOKEN[op] and toks[-1] == ref.GO, (op, toks)
    out = ref.expected_output(op, A, B)
    assert out[-1] == ref.END
    assert [decode_lane_out(t)[0] for t in out[:4]] == [0, 1, 2, 3]
    got = [decode_lane_out(t)[1] for t in out[:4]]
    assert got == want, (op, A, B, got, want)


def main():
    # Hand-computed edges (written out, not via ref).
    edges = [
        # v2 ops, kept as regression edges
        ("SUB", [0, 5, 15, 3], [1, 5, 0, 7], [15, 0, 15, 12]),
        ("MUL", [15, 15, 3, 0], [15, 2, 5, 9], [1, 14, 15, 0]),
        ("MAX", [0, 15, 7, 9], [15, 0, 7, 3], [15, 15, 7, 9]),
        ("MIN", [0, 15, 7, 9], [15, 0, 7, 3], [0, 0, 7, 3]),
        ("ADD", [15, 15, 15, 15], [15, 15, 15, 15], [14, 14, 14, 14]),
        # VSHL: amounts 0,1,2,3 across lanes; B=4 wraps to 0; B=15 -> 3
        ("VSHL", [15, 8, 1, 0], [0, 1, 2, 3], [15, 0, 4, 0]),
        ("VSHL", [15, 8, 1, 0], [4, 5, 6, 7], [15, 0, 4, 0]),
        ("VSHL", [15, 8, 1, 0], [15, 15, 15, 15], [8, 0, 8, 0]),
        ("VSHL", [9, 9, 9, 9], [3, 3, 3, 3], [8, 8, 8, 8]),
        # VSHR: amounts 0,1,2,3; B=4 wraps to 0; B=15 -> 3
        ("VSHR", [15, 8, 1, 0], [0, 1, 2, 3], [15, 4, 0, 0]),
        ("VSHR", [15, 8, 1, 0], [4, 5, 6, 7], [15, 4, 0, 0]),
        ("VSHR", [15, 8, 1, 0], [15, 15, 15, 15], [1, 1, 0, 0]),
        ("VSHR", [9, 9, 9, 9], [3, 3, 3, 3], [1, 1, 1, 1]),
        # MMUL: identity, zero, all-15s (225+225=450 -> 2 mod 16), general
        ("MMUL", [1, 0, 0, 1], [5, 6, 7, 8], [5, 6, 7, 8]),
        ("MMUL", [5, 6, 7, 8], [1, 0, 0, 1], [5, 6, 7, 8]),
        ("MMUL", [0, 0, 0, 0], [5, 6, 7, 8], [0, 0, 0, 0]),
        ("MMUL", [5, 6, 7, 8], [0, 0, 0, 0], [0, 0, 0, 0]),
        ("MMUL", [15, 15, 15, 15], [15, 15, 15, 15], [2, 2, 2, 2]),
        ("MMUL", [1, 2, 3, 4], [5, 6, 7, 8], [3, 6, 11, 2]),
        # VSORT: sorted, reverse, all equal, duplicates
        ("VSORT", [0, 5, 10, 15], [0, 0, 0, 0], [0, 5, 10, 15]),
        ("VSORT", [15, 10, 5, 0], [0, 0, 0, 0], [0, 5, 10, 15]),
        ("VSORT", [7, 7, 7, 7], [1, 2, 3, 4], [7, 7, 7, 7]),
        ("VSORT", [5, 2, 5, 1], [0, 0, 0, 0], [1, 2, 5, 5]),
        ("VSORT", [3, 1, 4, 2], [0, 0, 0, 0], [1, 2, 3, 4]),
    ]
    for op, A, B, want in edges:
        assert want == inline_result(op, A, B), ("edge disagrees with inline", op, A, B, want)
        check_lane_case(op, A, B, want)
    # VSORT ignores B: same A, different B -> same output.
    for B in ([0, 0, 0, 0], [15, 15, 15, 15], [1, 2, 3, 4]):
        check_lane_case("VSORT", [3, 1, 4, 2], B, [1, 2, 3, 4])
    # DOT edge (900) + token ids for the new ops.
    out = ref.expected_output("DOT", [15, 15, 15, 15], [15, 15, 15, 15])
    assert decode_dot_out(out[:3]) == 900 and out[-1] == ref.END
    assert ref.OP_TOKEN == {"ADD": 0, "DOT": 1, "SUB": 244, "MUL": 245,
                            "MAX": 246, "MIN": 247, "VSHL": 248, "VSHR": 249,
                            "MMUL": 250, "VSORT": 251}
    assert len(ref.OPS) == 10
    rng = random.Random(20261105)
    n = 0
    for op in ref.OPS:
        for _ in range(200):
            A = [rng.randrange(16) for _ in range(4)]
            B = [rng.randrange(16) for _ in range(4)]
            out = ref.expected_output(op, A, B)
            if op == "DOT":
                assert decode_dot_out(out[:3]) == sum(a * b for a, b in zip(A, B))
            else:
                got = [decode_lane_out(t)[1] for t in out[:4]]
                assert got == inline_result(op, A, B), (op, A, B, got)
            n += 1
    print(f"Stage 1 v3 PASS: {len(edges)} hand edges + 3 VSORT B-independence "
          f"cases + {n} random cases (inline recomputation), ops {ref.OPS}")


if __name__ == "__main__":
    main()
