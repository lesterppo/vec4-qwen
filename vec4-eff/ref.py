"""VEC4 v2 reference simulator: pure Python, no torch, no chipformer.

Extends v1 (ADD, DOT) with four lane-wise ops. All lane-wise ops share
the same output token ids (131 + lane*16 + digit), because the output
index for a lane op is lane*16+digit regardless of which lane op it is;
only the op token (and therefore the circuit's selected result) differs.
DOT keeps its own output ids (195 + place*16 + digit).

OP tokens: ADD=0, DOT=1, SUB=244, MUL=245, MAX=246, MIN=247.
SUB..MIN live in the free ids 244..247 (standalone vocab is padded
to 256), NOT in 2..5: ids 2..129 are VAL tokens, so OP ids there
would collide with VAL tokens and overwrite the OP head (a bug
found by the circuit test: ADD with A0=0, whose VAL id is 2, read
back as OP_SUB). Input is still 10 tokens: [OP, A0..A3, B0..B3, GO].
Output: lane ops -> 4 lane tokens + END; DOT -> 3 digit tokens + END.
"""

OP_ADD = 0
OP_DOT = 1
OP_SUB = 244
OP_MUL = 245
OP_MAX = 246
OP_MIN = 247
GO = 130
END = 243

OPS = ("ADD", "DOT", "SUB", "MUL", "MAX", "MIN")
OP_TOKEN = {"ADD": OP_ADD, "DOT": OP_DOT, "SUB": OP_SUB,
            "MUL": OP_MUL, "MAX": OP_MAX, "MIN": OP_MIN}


def val_token(reg, value):
    """reg 0..7 (0..3 = A0..A3, 4..7 = B0..B3), value 0..15."""
    assert 0 <= reg <= 7 and 0 <= value <= 15
    return 2 + reg * 16 + value


def lane_out_token(lane, digit):
    """Shared by ADD, SUB, MUL, MAX, MIN: lane result digit."""
    assert 0 <= lane <= 3 and 0 <= digit <= 15
    return 131 + lane * 16 + digit


# Backwards-compatible alias (v1 name).
add_out_token = lane_out_token


def dot_out_token(place, digit):
    """place 0 = high (dot>>8), 1 = mid ((dot>>4)&15), 2 = low (dot&15)."""
    assert 0 <= place <= 2 and 0 <= digit <= 15
    return 195 + place * 16 + digit


def lane_result(op, A, B):
    """The 4 lane results for a lane-wise op (plain Python arithmetic)."""
    if op == "ADD":
        return [(a + b) % 16 for a, b in zip(A, B)]
    if op == "SUB":
        return [(a - b) % 16 for a, b in zip(A, B)]
    if op == "MUL":
        return [(a * b) % 16 for a, b in zip(A, B)]
    if op == "MAX":
        return [max(a, b) for a, b in zip(A, B)]
    if op == "MIN":
        return [min(a, b) for a, b in zip(A, B)]
    raise ValueError(op)


def encode(op, A, B):
    """op in OPS, A/B lists of 4 values 0..15 -> 10 input token ids."""
    assert op in OPS and len(A) == 4 and len(B) == 4
    assert all(0 <= v <= 15 for v in A + B)
    toks = [OP_TOKEN[op]]
    for i, v in enumerate(A):
        toks.append(val_token(i, v))
    for i, v in enumerate(B):
        toks.append(val_token(4 + i, v))
    toks.append(GO)
    return toks


def expected_output(op, A, B):
    """Full expected generated sequence, including END."""
    if op == "DOT":
        dot = sum(a * b for a, b in zip(A, B))
        out = [dot_out_token(0, dot >> 8),
               dot_out_token(1, (dot >> 4) & 15),
               dot_out_token(2, dot & 15)]
    else:
        out = [lane_out_token(i, d) for i, d in enumerate(lane_result(op, A, B))]
    return out + [END]


def describe(tok):
    """Human-readable form of a token id, for logs."""
    if tok in OP_TOKEN.values():
        return "OP_" + [k for k, v in OP_TOKEN.items() if v == tok][0]
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
        return "OUT_LANE lane%d=%d" % (l, d)
    if 195 <= tok <= 242:
        p, d = (tok - 195) // 16, (tok - 195) % 16
        return "OUT_DOT place%d=%d" % (p, d)
    return "TOK%d" % tok
