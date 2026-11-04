"""Stage 1 (v2): reference simulator sanity, pure Python, no chipformer.

Checks, independently of the circuit: encoding shape, op tokens, and
that decoding expected_output recovers the hand-computed result for
every op, including edges where SUB wraps, MUL wraps, and MAX/MIN
switch lanes.
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


def main():
    # Hand-computed edges (written out, not via ref.lane_result).
    edges = [
        ("SUB", [0, 5, 15, 3], [1, 5, 0, 7], [15, 0, 15, 12]),
        ("SUB", [15, 15, 15, 15], [15, 15, 15, 15], [0, 0, 0, 0]),
        ("MUL", [15, 15, 3, 0], [15, 2, 5, 9], [1, 14, 15, 0]),
        ("MUL", [0, 1, 2, 3], [15, 14, 13, 12], [0, 14, 10, 4]),
        ("MAX", [0, 15, 7, 9], [15, 0, 7, 3], [15, 15, 7, 9]),
        ("MIN", [0, 15, 7, 9], [15, 0, 7, 3], [0, 0, 7, 3]),
        ("ADD", [15, 15, 15, 15], [15, 15, 15, 15], [14, 14, 14, 14]),
        ("DOT", [15, 15, 15, 15], [15, 15, 15, 15], None),
    ]
    for op, A, B, want in edges:
        toks = ref.encode(op, A, B)
        assert len(toks) == 10 and toks[0] == ref.OP_TOKEN[op] and toks[-1] == ref.GO, (op, toks)
        out = ref.expected_output(op, A, B)
        assert out[-1] == ref.END
        if op == "DOT":
            assert decode_dot_out(out[:3]) == 900
        else:
            got = [decode_lane_out(t)[1] for t in out[:4]]
            assert [decode_lane_out(t)[0] for t in out[:4]] == [0, 1, 2, 3]
            assert got == want, (op, A, B, got, want)
    rng = random.Random(20261104)
    n = 0
    for op in ref.OPS:
        for _ in range(200):
            A = [rng.randrange(16) for _ in range(4)]
            B = [rng.randrange(16) for _ in range(4)]
            out = ref.expected_output(op, A, B)
            if op == "DOT":
                assert decode_dot_out(out[:3]) == sum(a * b for a, b in zip(A, B))
            else:
                assert [decode_lane_out(t)[1] for t in out[:4]] == ref.lane_result(op, A, B)
            n += 1
    print(f"Stage 1 v2 PASS: {len(edges)} hand edges + {n} random cases, "
          f"ops {ref.OPS}")


if __name__ == "__main__":
    main()
