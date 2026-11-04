"""VEC4 reference simulator: pure Python, no torch, no chipformer.

Implements only SPEC.md's encoding and ISA arithmetic, independently of
the circuit (plain Python +, *, %, >>, &), so circuit output can be
checked against it token-for-token.
"""

OP_ADD = 0
OP_DOT = 1
GO = 130
END = 243


def val_token(reg, value):
    """reg 0..7 (0..3 = A0..A3, 4..7 = B0..B3), value 0..15."""
    assert 0 <= reg <= 7 and 0 <= value <= 15
    return 2 + reg * 16 + value


def add_out_token(lane, digit):
    assert 0 <= lane <= 3 and 0 <= digit <= 15
    return 131 + lane * 16 + digit


def dot_out_token(place, digit):
    """place 0 = high (dot>>8), 1 = mid ((dot>>4)&15), 2 = low (dot&15)."""
    assert 0 <= place <= 2 and 0 <= digit <= 15
    return 195 + place * 16 + digit


def encode(op, A, B):
    """op 'ADD'|'DOT', A/B lists of 4 values 0..15 -> 10 input token ids."""
    assert op in ("ADD", "DOT") and len(A) == 4 and len(B) == 4
    assert all(0 <= v <= 15 for v in A + B)
    op_tok = OP_ADD if op == "ADD" else OP_DOT
    toks = [op_tok]
    for i, v in enumerate(A):
        toks.append(val_token(i, v))
    for i, v in enumerate(B):
        toks.append(val_token(4 + i, v))
    toks.append(GO)
    return toks


def expected_output(op, A, B):
    """Full expected generated sequence, including END."""
    if op == "ADD":
        out = [add_out_token(i, (A[i] + B[i]) % 16) for i in range(4)]
    else:
        dot = sum(a * b for a, b in zip(A, B))
        out = [dot_out_token(0, dot >> 8),
               dot_out_token(1, (dot >> 4) & 15),
               dot_out_token(2, dot & 15)]
    return out + [END]


def describe(tok):
    """Human-readable form of a token id, for logs."""
    if tok == OP_ADD:
        return "OP_ADD"
    if tok == OP_DOT:
        return "OP_DOT"
    if tok == GO:
        return "GO"
    if tok == END:
        return "END"
    if 2 <= tok <= 129:
        r, v = (tok - 2) // 16, (tok - 2) % 16
        name = ("A%d" % r) if r < 4 else ("B%d" % (r - 4))
        return "VAL %s=%d" % (name, v)
    if 131 <= tok <= 194:
        l, d = (tok - 131) // 16, (tok - 131) % 16
        return "OUT_ADD lane%d=%d" % (l, d)
    if 195 <= tok <= 242:
        p, d = (tok - 195) // 16, (tok - 195) % 16
        return "OUT_DOT place%d=%d" % (p, d)
    return "TOK%d" % tok
