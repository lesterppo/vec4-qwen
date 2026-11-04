"""VEC4 compiled circuit, built on chipformer IR/prims (see SPEC.md).

2B RETARGET (vec4-2b copy; original ../vec4/circuit.py untouched):
only the Target intermediate cap changes, 3584 -> 6144, to match
Qwen3.5-2B. The graph itself is unchanged: max 98 FFN units per
sublayer (verified locally from this file's build: units max 98 at
level 2), far below both caps, so FFN placement (ir._place) is
identical; head counts (max 9 key heads) are far below the unchanged
16 linear key heads. No circuit redesign.

Standalone vocab ids (see ref.py): OP_ADD=0, OP_DOT=1,
VAL r v = 2+r*16+v, GO=130, OUT_ADD l d = 131+l*16+d,
OUT_DOT p d = 195+p*16+d, END=243. Total 244 tokens.

Target: Qwen3.5-2B head shapes, Target(16, 16, 6144).
"""
import sys
from fractions import Fraction
from pathlib import Path

# Upstream clone sits next to vec4/ (../upstream); allow override.
UPSTREAM = Path(__file__).resolve().parent.parent / "upstream"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from chipformer.dsl.graph import Expr
from chipformer.q35.ir import Target
from chipformer.q35.prims import Circ

CODE_AMP = 128
KAPPA = Fraction(1, 2 ** 20)  # 2**-20, as in chip8_q35
TAU_TEXT = 2.0 ** -40

OP_ADD, OP_DOT, GO, END = 0, 1, 130, 243


def _is_op(t):
    return int(t in (OP_ADD, OP_DOT))


def _is_val(t):
    return int(2 <= t <= 129)


def _reg_of(t):
    return (t - 2) // 16 if _is_val(t) else 0


def _val_of(t):
    return (t - 2) % 16 if _is_val(t) else 0


def _is_add_out(t):
    return int(131 <= t <= 194)


def _is_dot_out(t):
    return int(195 <= t <= 242)


def _is_out(t):
    return int(131 <= t <= 242)


def _lane_of(t):
    return (t - 131) // 16 if _is_add_out(t) else 0


def _place_of(t):
    return (t - 195) // 16 if _is_dot_out(t) else 0


def _is_vec4(t):
    return int(0 <= t <= 243 and (t <= 1 or 2 <= t <= 129 or t == GO
                                  or 131 <= t <= 242 or t == END))


def _out_index(t):
    """Output index for code: ADD 0..63, DOT 64..111, END 112, else None."""
    if _is_add_out(t):
        return _lane_of(t) * 16 + (t - 131) % 16
    if _is_dot_out(t):
        return 64 + _place_of(t) * 16 + (t - 195) % 16
    if t == END:
        return 112
    return None


def _code_bit(t, i):
    """Code bit i: i==0 is the flag, i>=1 is index bit i-1."""
    idx = _out_index(t)
    if idx is None:
        return 0
    if i == 0:
        return 1
    return (idx >> (i - 1)) & 1


class Vec4(Circ):
    def __init__(self, target=None, offset=0):
        """offset=0: standalone ids (0..243 are VEC4 tokens).
        offset=v0>0: merged ids, where v0..v0+243 are VEC4 tokens and
        ids < v0 are Qwen text tokens (all VEC4 features off, mu=0,
        tau=TAU_TEXT, code=-CODE_AMP). Circuit logic is identical; only
        token-feature translation changes, so the merged circuit is the
        same graph with a different embedding row mapping."""
        super().__init__(target or Target(16, 16, 6144))
        self.offset = offset

        def tr(t):
            """Standalone VEC4 id for a full id, or None for a text token."""
            if offset == 0:
                return t if _is_vec4(t) else None
            return (t - offset) if offset <= t < offset + 244 else None

        def is_op(t):
            v = tr(t)
            return int(v is not None and _is_op(v))

        def is_val(t):
            v = tr(t)
            return int(v is not None and _is_val(v))

        def reg_of(t):
            v = tr(t)
            return _reg_of(v) if v is not None else 0

        def val_of(t):
            v = tr(t)
            return _val_of(v) if v is not None else 0

        def is_go(t):
            v = tr(t)
            return int(v is not None and v == GO)

        def is_out(t):
            v = tr(t)
            return int(v is not None and _is_out(v))

        def is_add_out(t):
            v = tr(t)
            return int(v is not None and _is_add_out(v))

        def lane_of(t):
            v = tr(t)
            return _lane_of(v) if v is not None else 0

        def is_dot_out(t):
            v = tr(t)
            return int(v is not None and _is_dot_out(v))

        def place_of(t):
            v = tr(t)
            return _place_of(v) if v is not None else 0

        def is_end(t):
            v = tr(t)
            return int(v is not None and v == END)

        def is_vec4(t):
            return int(tr(t) is not None)

        def code_bit(t, i):
            v = tr(t)
            return _code_bit(v, i) if v is not None else 0

        # full=True is REQUIRED for the merged model: ordinary features see
        # only t & 0xFFFF (ir.TOKEN_MASK), which destroys the v0 offset
        # (248320) translation. With full=True, feature_value passes the
        # whole id, so tr() works for any v0. Standalone ids (< 65536) are
        # unaffected.
        def f(name, fn):
            return self.token_feature(name, fn, full=True)

        one = self.one
        self.is_op = f("is_op", is_op)
        self.op_add_f = f("op_add", lambda t: int(tr(t) == OP_ADD))
        self.op_dot_f = f("op_dot", lambda t: int(tr(t) == OP_DOT))
        self.is_val = f("is_val", is_val)
        self.is_val_r = [f("is_val_r%d" % r, lambda t, r=r: int(is_val(t) and reg_of(t) == r))
                         for r in range(8)]
        self.reg_bit = [f("reg_bit%d" % i, lambda t, i=i: (reg_of(t) >> i) & 1)
                        for i in range(3)]
        self.val_bit = [f("val_bit%d" % i, lambda t, i=i: (val_of(t) >> i) & 1)
                        for i in range(4)]
        self.is_go = f("is_go", is_go)
        self.is_out = f("is_out", is_out)
        self.is_add_out = f("is_add_out", is_add_out)
        self.lane_bit = [f("lane_bit%d" % i, lambda t, i=i: (lane_of(t) >> i) & 1)
                         for i in range(2)]
        self.is_dot_out = f("is_dot_out", is_dot_out)
        self.place_bit = [f("place_bit%d" % i, lambda t, i=i: (place_of(t) >> i) & 1)
                          for i in range(2)]
        self.is_end = f("is_end", is_end)
        self.is_text = f("is_text", lambda t: int(not is_vec4(t)))
        self.mu = f("mu", lambda t: int(bool(is_vec4(t))))
        self.tau = f("tau", lambda t: -1 if is_vec4(t) else TAU_TEXT)
        self.code = [f("code%d" % i, lambda t, i=i: CODE_AMP * (2 * code_bit(t, i) - 1))
                     for i in range(8)]

    # --- helpers -------------------------------------------------------------
    def and_bit(self, a, b, name):
        return self.node(self.t_and([a, b]), name)

    def carry_node(self, x, y, carry_in, name):
        z = x + y + (carry_in if carry_in is not None else Expr())
        lo, hi = (0, 2) if carry_in is None else (0, 3)
        return self.node(self.t_ge(z, 2, lo, hi), name)

    def sum_node(self, x, y, carry_in, carry_out, name):
        terms = [(1, x, self.one), (1, y, self.one)]
        if carry_in is not None:
            terms.append((1, carry_in, self.one))
        terms.append((-2, carry_out, self.one))
        return self.node(terms, name)

    def adder(self, x_bits, y_bits, name):
        """Binary adder, LSB first. Returns sum bits length n+1 (last = final carry)."""
        n = max(len(x_bits), len(y_bits))
        x_bits = list(x_bits) + [Expr()] * (n - len(x_bits))
        y_bits = list(y_bits) + [Expr()] * (n - len(y_bits))
        sums, carry = [], None
        for i in range(n):
            co = self.carry_node(x_bits[i], y_bits[i], carry, "%s.carry%d" % (name, i))
            s = self.sum_node(x_bits[i], y_bits[i], carry, co, "%s.sum%d" % (name, i))
            sums.append(s)
            carry = co
        sums.append(carry)
        return sums

    # --- build ---------------------------------------------------------------
    def build(self):
        one = self.one
        # 1. Register heads + OP head (first head sublayer, parallel).
        reg_bits = []
        for r in range(8):
            outs = self.slot_head("reg%d" % r, {0: self.is_val_r[r]}, {0: self.is_go},
                                  {2 + i: self.val_bit[i] for i in range(4)}, one)
            reg_bits.append(outs)  # [bit0..bit3]
        op_bits = self.slot_head("op", {0: self.is_op}, {0: self.is_go},
                                 {2: self.op_add_f, 3: self.op_dot_f}, one)
        op_add, op_dot = op_bits[0], op_bits[1]

        # 2. Lane adders (parallel across lanes).
        lane_sum = []
        for lane in range(4):
            a_bits, b_bits = reg_bits[lane], reg_bits[4 + lane]
            sums = self.adder(a_bits, b_bits, "lane%d" % lane)
            lane_sum.append(sums[:4])  # mod 16: discard final carry

        # 3. Dot product: per-lane partial-product ANDs, then adder trees.
        dot_bits = None
        lane_prod = []
        for lane in range(4):
            a_bits, b_bits = reg_bits[lane], reg_bits[4 + lane]
            partials = []
            for j in range(4):
                bits = []
                for p in range(8):
                    if 0 <= p - j <= 3:
                        bits.append(self.and_bit(a_bits[p - j], b_bits[j],
                                                 "prod%d.p%d.b%d" % (lane, j, p)))
                    else:
                        bits.append(Expr())
                partials.append(bits)
            s01 = self.adder(partials[0], partials[1], "prod%d.s01" % lane)
            s23 = self.adder(partials[2], partials[3], "prod%d.s23" % lane)
            prod = self.adder(s01, s23, "prod%d" % lane)
            lane_prod.append(prod[:8])  # product fits in 8 bits
        d01 = self.adder(lane_prod[0], lane_prod[1], "dot.d01")
        d23 = self.adder(lane_prod[2], lane_prod[3], "dot.d23")
        dot_full = self.adder(d01, d23, "dot")
        dot_bits = dot_full[:10]  # dot <= 900 < 1024

        # 4. Result write/read heads (after compute).
        lane_res = []
        for lane in range(4):
            outs = self.slot_head("res.lane%d" % lane, {0: self.is_go}, {0: self.is_out},
                                  {2 + i: lane_sum[lane][i] for i in range(4)}, one)
            lane_res.append(outs)
        dot_res = self.slot_head("res.dot", {0: self.is_go}, {0: self.is_out},
                                 {2 + i: dot_bits[i] for i in range(10)}, one)

        # 5. Owner node + exit.
        owner = self.node([(1, self.is_go, one), (1, self.is_out, one)], "owner")
        self.owner_expr = owner
        # Clear code dims (early sublayer is fine: nothing else reads code).
        for i, e in enumerate(self.code):
            self.ffn([(-1, one, e), (1, one, -e)], "code.clear%d" % i, into=e)
        top = max(n.level for n in self.nodes)
        # Conditions as literal lists (Exprs, each 0/1 at the relevant position).
        def lane_cond(l):
            # last token is ADD OUT with lane l
            lits = [self.is_add_out]
            for i in range(2):
                lits.append(self.lane_bit[i] if (l >> i) & 1 else one - self.lane_bit[i])
            return lits

        def place_cond(p):
            lits = [self.is_dot_out]
            for i in range(2):
                lits.append(self.place_bit[i] if (p >> i) & 1 else one - self.place_bit[i])
            return lits

        go_add = [self.is_go, op_add]
        go_dot = [self.is_go, op_dot]
        # bit_cases[i] for code dim i: list of literal-lists that set bit i.
        bit_cases = [[] for _ in range(8)]
        bit_cases[0] = [[]]  # flag: 1 whenever owner=1
        for j in range(7):  # index bit j
            i = j + 1
            cases = []
            # GO + ADD: index = lane0 digit (bits 0..3)
            if j < 4:
                cases.append(go_add + [lane_sum[0][j]])
            # GO + DOT: index = 64 + high digit (bits 0..1 = dot bits 8..9, bit 6 = 1)
            if j < 2:
                cases.append(go_dot + [dot_bits[8 + j]])
            elif j == 6:
                cases.append(go_dot)
            # ADD lane l -> lane l+1
            for l in range(3):
                nxt = l + 1
                if j < 4:
                    cases.append(lane_cond(l) + [lane_res[nxt][j]])
                elif j == 4 and (nxt & 1):
                    cases.append(lane_cond(l))
                elif j == 5 and ((nxt >> 1) & 1):
                    cases.append(lane_cond(l))
            # DOT high -> mid, mid -> low
            if j < 4:
                cases.append(place_cond(0) + [dot_res[4 + j]])  # mid bits
                cases.append(place_cond(1) + [dot_res[j]])      # low bits
            if j == 4:
                cases.append(place_cond(0))  # 80 = 64+16: bit4
            if j == 6:
                cases.append(place_cond(0))  # 80: bit6
                cases.append(place_cond(1))  # 96 = 64+32: bit6
            if j == 5:
                cases.append(place_cond(1))  # 96: bit5
            # END: index 112 = 64+32+16, bits 4,5,6
            if j in (4, 5, 6):
                cases.append(lane_cond(3))
                cases.append(place_cond(2))
            bit_cases[i] = cases
        for i, e in enumerate(self.code):
            terms = []
            for lits in bit_cases[i]:
                terms += self.t_and([*lits, owner], 2 * KAPPA)
            terms += self.t_and([owner], -KAPPA)
            self.ffn(terms, "exit.code%d" % i, into=e, min_level=top)
        # Tau exit, as in chip8: 0 on chip tokens, TAU_TEXT on unowned text.
        self.ffn([(1, self.mu, one)] + self.t_and([owner, self.is_text], -TAU_TEXT),
                 "exit.tau", into=self.tau, min_level=top)
        self.keep = [next(iter(e.terms)) for e in self.code]
        self.lane_sum, self.dot_bits = lane_sum, dot_bits
        self.lane_res, self.dot_res = lane_res, dot_res
        return self


def build_vec4(offset=0):
    c = Vec4(offset=offset)
    c.build()
    return c
