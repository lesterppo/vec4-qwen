"""VEC4 v3 compiled circuit, built on chipformer IR/prims (see SPEC_V3.md).

v2 ops ADD/DOT/SUB/MUL/MAX/MIN plus VSHL, VSHR (fixed-direction shifts,
amount = B's low 2 bits per lane), MMUL (2x2 matrix product mod 16,
row-major in lanes) and VSORT (ascending sort via a 5-stage
compare-exchange network). All lane-wise outputs share OUT_LANE ids.
Standalone vocab ids (see ref.py): OP_ADD=0, OP_DOT=1, VAL r v =
2+r*16+v, GO=130, OUT_LANE l d = 131+l*16+d (shared by all
lane-wise ops), OUT_DOT p d = 195+p*16+d, END=243, OPs SUB..VSORT
244..251. Total 252 tokens (0..251), padded to 256 in the merged
vocab as in v2 (merged vocab stays 248576: ceil(248572/64)*64).

Target: Qwen3.5-0.8B head shapes, Target(16, 16, 3584) — unchanged
from v2. If v3 approaches the 3584 FFN cap, that is a retarget
decision, not a silent cap change (see SPEC_V3.md).

SPEC DEVIATION (mux argument order): SPEC_V3.md writes the shift
mux as select([([s1], mux(s0, a<<2, a<<3)), ([1-s1], mux(s0, a<<0,
a<<1))]) with mux(s, x, y) = select([([s], x), ([1-s], y)]). Taken
literally that shifts by 2*s1 + (1-s0) — bit 0 inverted — not by
B & 3. This implementation uses the semantically correct order,
mux(s0, a<<3, a<<2) / mux(s0, a<<1, a<<0), so amount = 2*s1 + s0
= B & 3. The circuit test (independent ref, shift edges at every
amount) is the arbiter: the literal spec order fails it, this
order passes it.
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

OP_ADD, OP_DOT, OP_SUB, OP_MUL, OP_MAX, OP_MIN = 0, 1, 244, 245, 246, 247
OP_VSHL, OP_VSHR, OP_MMUL, OP_VSORT, GO, END = 248, 249, 250, 251, 130, 243
ALL_OPS = (OP_ADD, OP_DOT, OP_SUB, OP_MUL, OP_MAX, OP_MIN,
           OP_VSHL, OP_VSHR, OP_MMUL, OP_VSORT)


def _is_op(t):
    return int(t in ALL_OPS)


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
    return int(0 <= t <= 251 and (t in ALL_OPS
                                  or 2 <= t <= 129 or t == GO
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


class Vec4v3(Circ):
    def __init__(self, target=None, offset=0):
        """offset=0: standalone ids (0..251 are VEC4 tokens).
        offset=v0>0: merged ids, where v0..v0+251 are VEC4 tokens and
        ids < v0 are Qwen text tokens (all VEC4 features off, mu=0,
        tau=TAU_TEXT, code=-CODE_AMP). Circuit logic is identical; only
        token-feature translation changes, so the merged circuit is the
        same graph with a different embedding row mapping."""
        super().__init__(target or Target(16, 16, 3584))
        self.offset = offset

        def tr(t):
            """Standalone VEC4 id for a full id, or None for a text token."""
            if offset == 0:
                return t if _is_vec4(t) else None
            return (t - offset) if offset <= t < offset + 252 else None

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
        self.op_sub_f = f("op_sub", lambda t: int(tr(t) == OP_SUB))
        self.op_mul_f = f("op_mul", lambda t: int(tr(t) == OP_MUL))
        self.op_max_f = f("op_max", lambda t: int(tr(t) == OP_MAX))
        self.op_min_f = f("op_min", lambda t: int(tr(t) == OP_MIN))
        self.op_vshl_f = f("op_vshl", lambda t: int(tr(t) == OP_VSHL))
        self.op_vshr_f = f("op_vshr", lambda t: int(tr(t) == OP_VSHR))
        self.op_mmul_f = f("op_mmul", lambda t: int(tr(t) == OP_MMUL))
        self.op_vsort_f = f("op_vsort", lambda t: int(tr(t) == OP_VSORT))
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

    def adder(self, x_bits, y_bits, name, carry0=None):
        """Binary adder, LSB first. Returns sum bits length n+1 (last = final carry).
        carry0 (default None = 0) is the initial carry: pass self.one for
        two's-complement subtraction (A + ~B + 1)."""
        n = max(len(x_bits), len(y_bits))
        x_bits = list(x_bits) + [Expr()] * (n - len(x_bits))
        y_bits = list(y_bits) + [Expr()] * (n - len(y_bits))
        sums, carry = [], carry0
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
                                 {2: self.op_add_f, 3: self.op_dot_f,
                                  4: self.op_sub_f, 5: self.op_mul_f,
                                  6: self.op_max_f, 7: self.op_min_f,
                                  8: self.op_vshl_f, 9: self.op_vshr_f,
                                  10: self.op_mmul_f, 11: self.op_vsort_f}, one)
        (op_add, op_dot, op_sub, op_mul, op_max, op_min,
         op_vshl, op_vshr, op_mmul, op_vsort) = op_bits

        # 2. Lane adders (parallel across lanes).
        lane_sum = []
        for lane in range(4):
            a_bits, b_bits = reg_bits[lane], reg_bits[4 + lane]
            sums = self.adder(a_bits, b_bits, "lane%d" % lane)
            lane_sum.append(sums[:4])  # mod 16: discard final carry

        # 3. Products: per-lane partial-product ANDs, then adder trees.
        # Helper shared by DOT (lane products) and MMUL (cross-lane
        # products): same pattern, only the operand registers differ.
        def lane_product(a_bits, b_bits, name):
            partials = []
            for j in range(4):
                bits = []
                for p in range(8):
                    if 0 <= p - j <= 3:
                        bits.append(self.and_bit(a_bits[p - j], b_bits[j],
                                                 "%s.p%d.b%d" % (name, j, p)))
                    else:
                        bits.append(Expr())
                partials.append(bits)
            s01 = self.adder(partials[0], partials[1], "%s.s01" % name)
            s23 = self.adder(partials[2], partials[3], "%s.s23" % name)
            prod = self.adder(s01, s23, name)
            return prod[:8]  # product fits in 8 bits

        dot_bits = None
        lane_prod = []
        for lane in range(4):
            lane_prod.append(lane_product(reg_bits[lane], reg_bits[4 + lane],
                                          "prod%d" % lane))
        d01 = self.adder(lane_prod[0], lane_prod[1], "dot.d01")
        d23 = self.adder(lane_prod[2], lane_prod[3], "dot.d23")
        dot_full = self.adder(d01, d23, "dot")
        dot_bits = dot_full[:10]  # dot <= 900 < 1024

        # 3b. Lane-wise SUB / MUL / MAX / MIN (parallel across lanes).
        # SUB: A + ~B + 1 (two's complement, mod 16); final carry is the
        # A>=B flag (1 iff no borrow). MUL (mod 16) is the low 4 bits of
        # the lane product already computed for DOT. MAX = B + ge*(A-B),
        # MIN = A - ge*(A-B), both via the same masked difference, so
        # neither needs a comparator beyond the SUB carry.
        lane_sub, lane_mul, lane_max, lane_min = [], [], [], []
        for lane in range(4):
            a_bits, b_bits = reg_bits[lane], reg_bits[4 + lane]
            sub_full = self.adder(a_bits, [one - b for b in b_bits],
                                  "lane%d.sub" % lane, carry0=one)
            diff, ge = sub_full[:4], sub_full[4]
            lane_sub.append(diff)
            lane_mul.append(lane_prod[lane][:4])
            masked = [self.and_bit(ge, d, "lane%d.mask%d" % (lane, j))
                      for j, d in enumerate(diff)]
            lane_max.append(self.adder(b_bits, masked, "lane%d.max" % lane)[:4])
            lane_min.append(self.adder(a_bits, [one - m for m in masked],
                                       "lane%d.min" % lane, carry0=one)[:4])

        # 3c. VSHL / VSHR: two-level mux over shifted versions.
        # Shifted bit lists are pure permutations of the A bits (0
        # where the shift pushes a bit out): shl[k] bit i is a[i-k]
        # (0 if i<k), shr[k] bit i is a[i+k] (0 if i+k>3), truncated
        # to 4 bits. Amount = 2*s1 + s0 = B & 3. mux(s, x, y) picks
        # x when s=1, y when s=0 (one-hot by construction: s vs 1-s).
        # NOTE the spec's written argument order inverts bit 0; see
        # the module docstring — this order is the correct one.
        def mux(s, x, y, name):
            return self.select_bits([([s], x), ([one - s], y)], 4, name)

        lane_shl, lane_shr = [], []
        for lane in range(4):
            a_bits, b_bits = reg_bits[lane], reg_bits[4 + lane]
            s0, s1 = b_bits[0], b_bits[1]
            shl = [[a_bits[i - k] if i >= k else Expr() for i in range(4)]
                   for k in range(4)]
            shr = [[a_bits[i + k] if i + k <= 3 else Expr() for i in range(4)]
                   for k in range(4)]
            lane_shl.append(self.select_bits(
                [([s1], mux(s0, shl[3], shl[2], "shl%d.hi" % lane)),
                 ([one - s1], mux(s0, shl[1], shl[0], "shl%d.lo" % lane))],
                4, "shl%d" % lane))
            lane_shr.append(self.select_bits(
                [([s1], mux(s0, shr[3], shr[2], "shr%d.hi" % lane)),
                 ([one - s1], mux(s0, shr[1], shr[0], "shr%d.lo" % lane))],
                4, "shr%d" % lane))

        # 3d. MMUL: cross-lane products + pairwise sums mod 16.
        # (a,b) pairs are (A lane, B lane): c00 needs (0,4),(1,6);
        # c01 needs (0,5),(1,7); c10 needs (2,4),(3,6); c11 needs
        # (2,5),(3,7). (0,4) and (3,7) are lane_prod[0]/lane_prod[3]
        # from DOT; the other six are built with the same pattern.
        cross = {}
        for (al, bl) in [(1, 6), (0, 5), (1, 7), (2, 4), (3, 6), (2, 5)]:
            cross[(al, bl)] = lane_product(reg_bits[al], reg_bits[bl],
                                           "xprod%d.%d" % (al, bl))
        cross[(0, 4)] = lane_prod[0]
        cross[(3, 7)] = lane_prod[3]
        lane_mmul = []
        for (p, q) in [((0, 4), (1, 6)), ((0, 5), (1, 7)),
                       ((2, 4), (3, 6)), ((2, 5), (3, 7))]:
            lane_mmul.append(self.adder(cross[p], cross[q],
                                        "mmul.%d%d" % (p[0], q[0]))[:4])

        # 3e. VSORT: 5-stage compare-exchange network, stages in
        # order (0,1), (2,3), (0,2), (1,3), (1,2). Compare-exchange
        # reuses the SUB pattern: diff = x - y (adder with ~y + 1),
        # ge = final carry (1 iff x >= y), masked = ge & diff,
        # hi = y + masked, lo = x - masked. On equality ge=1 but
        # diff=0, so (lo, hi) = (x, y): stable, as specified.
        def compare_exchange(x_bits, y_bits, name):
            diff_full = self.adder(x_bits, [one - b for b in y_bits],
                                   "%s.diff" % name, carry0=one)
            diff, ge = diff_full[:4], diff_full[4]
            masked = [self.and_bit(ge, d, "%s.mask%d" % (name, j))
                      for j, d in enumerate(diff)]
            hi = self.adder(y_bits, masked, "%s.hi" % name)[:4]
            lo = self.adder(x_bits, [one - m for m in masked],
                            "%s.lo" % name, carry0=one)[:4]
            return lo, hi

        cur = [list(reg_bits[l]) for l in range(4)]
        for si, (i, j) in enumerate([(0, 1), (2, 3), (0, 2), (1, 3), (1, 2)]):
            lo, hi = compare_exchange(cur[i], cur[j], "sort.s%d" % si)
            cur[i], cur[j] = lo, hi
        lane_sort = cur

        # Selected lane result: at GO exactly one op bit is 1, so
        # select_bits gives the requested op's lane bits; at other
        # positions all op bits are 0 and the selection is 0 (result
        # heads only write at GO, so this is never read elsewhere).
        lane_sel = []
        for lane in range(4):
            sel = self.select_bits([([op_add], lane_sum[lane]),
                                    ([op_sub], lane_sub[lane]),
                                    ([op_mul], lane_mul[lane]),
                                    ([op_max], lane_max[lane]),
                                    ([op_min], lane_min[lane]),
                                    ([op_vshl], lane_shl[lane]),
                                    ([op_vshr], lane_shr[lane]),
                                    ([op_mmul], lane_mmul[lane]),
                                    ([op_vsort], lane_sort[lane])],
                                   4, "sel.lane%d" % lane)
            lane_sel.append(sel)

        # 4. Result write/read heads (after compute).
        lane_res = []
        for lane in range(4):
            outs = self.slot_head("res.lane%d" % lane, {0: self.is_go}, {0: self.is_out},
                                  {2 + i: lane_sel[lane][i] for i in range(4)}, one)
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
        go_sub = [self.is_go, op_sub]
        go_mul = [self.is_go, op_mul]
        go_max = [self.is_go, op_max]
        go_min = [self.is_go, op_min]
        go_vshl = [self.is_go, op_vshl]
        go_vshr = [self.is_go, op_vshr]
        go_mmul = [self.is_go, op_mmul]
        go_vsort = [self.is_go, op_vsort]
        # bit_cases[i] for code dim i: list of literal-lists that set bit i.
        bit_cases = [[] for _ in range(8)]
        bit_cases[0] = [[]]  # flag: 1 whenever owner=1
        for j in range(7):  # index bit j
            i = j + 1
            cases = []
            # GO + lane op: index = lane0 digit (bits 0..3) of that op.
            # Lane-transition cases below are op-independent: result
            # heads already hold the selected op's lane bits.
            if j < 4:
                cases.append(go_add + [lane_sum[0][j]])
                cases.append(go_sub + [lane_sub[0][j]])
                cases.append(go_mul + [lane_mul[0][j]])
                cases.append(go_max + [lane_max[0][j]])
                cases.append(go_min + [lane_min[0][j]])
                cases.append(go_vshl + [lane_shl[0][j]])
                cases.append(go_vshr + [lane_shr[0][j]])
                cases.append(go_mmul + [lane_mmul[0][j]])
                cases.append(go_vsort + [lane_sort[0][j]])
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
        self.op_bits = op_bits
        self.lane_sum, self.dot_bits = lane_sum, dot_bits
        self.lane_sub, self.lane_mul = lane_sub, lane_mul
        self.lane_max, self.lane_min = lane_max, lane_min
        self.lane_shl, self.lane_shr = lane_shl, lane_shr
        self.lane_mmul, self.lane_sort = lane_mmul, lane_sort
        self.lane_sel = lane_sel
        self.lane_res, self.dot_res = lane_res, dot_res
        return self


def build_vec4v3(offset=0):
    c = Vec4v3(offset=offset)
    c.build()
    return c


# Backwards-compatible alias (v1/v2 name).
build_vec4 = build_vec4v3
