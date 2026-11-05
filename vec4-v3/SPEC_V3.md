# VEC4 v3 specification — four new ops on the v2 base (written BEFORE implementation)

Base: v2 as pushed (`vec4-v2/` + `vec4-2b-v2/`): 6 ops, OP ids
ADD=0, DOT=1, SUB=244, MUL=245, MAX=246, MIN=247, VAL 2..129,
GO=130, OUT_LANE 131..194, OUT_DOT 195..242, END=243,
CHIP_VOCAB=248, merged vocab 248576, chip offset v0=248320.

## New ops (all lane-wise outputs share OUT_LANE ids, as in v2)

| Op | Token | Semantics (A, B = 4 lanes of 0..15) |
|---|---|---|
| VSHL | 248 | lane: `(A << (B & 3)) & 15` — shift amount is B's low 2 bits |
| VSHR | 249 | lane: `A >> (B & 3)` — logical shift right, zero fill |
| MMUL | 250 | 2x2 matrices row-major in lanes: A=[a00,a01,a10,a11], B=[b00,b01,b10,b11]; C = A@B mod 16; output lanes [c00,c01,c10,c11] where c00=(a00*b00+a01*b10)%16, c01=(a00*b01+a01*b11)%16, c10=(a10*b00+a11*b10)%16, c11=(a10*b01+a11*b11)%16 |
| VSORT | 251 | lanes sorted ascending (stable: equal values keep original relative order — the network below is stable for the compare-exchange as specified) |

Why these four: VSHL/VSHR close the deferred VSHIFT item as two
fixed-direction shifts (a variable-direction shifter is still not
built; direction is the op, amount is the operand). MMUL reuses
the DOT product machinery with cross-lane products. VSORT reuses
the MAX/MIN compare-exchange machinery. All outputs are 4 lane
tokens + END, exactly like the other lane ops — no new output
id range is needed.

## Token budget (checked before building)

CHIP_VOCAB grows 248 -> 252 (new OP ids 248..251). Merged vocab
is UNCHANGED at 248576: merge pads `ceil((v0 + CHIP_VOCAB)/64)*64`
= ceil(248572/64)*64 = 248576, same as v2. The chip rows slice
becomes `full[v0:v0+252]`. Anything that hardcodes 248 as the
chip token count must be updated: circuit `tr()` bound
(`offset + 248` -> `offset + 252`), `_is_vec4` range (0..247 ->
0..251), `_is_op` set, merge `CHIP_VOCAB`, and any driver that
asserts chip_vocab == 248 (eff.py did — v3 drivers must accept
252). The merged vocab assertion (248576) stays.

## Reference simulator (ref.py, plain Python, independent of circuit)

- `OPS` gains VSHL, VSHR, MMUL, VSORT; `OP_TOKEN` maps them to
  248..251. `lane_result` handles VSHL/VSHR; MMUL and VSORT get
  their own functions (`mmul_result(A, B)`, `vsort_result(A)`)
  since they are not per-lane functions of (a, b) pairs.
- `encode` unchanged in shape: [OP, A0..A3, B0..B3, GO].
  VSORT ignores B (B is still present in the input and must be
  well-formed VAL tokens; its values do not affect the output —
  say so in the docstring and test it: same A with different B
  gives the same VSORT output).
- `expected_output`: lane ops (incl. VSHL/VSHR/MMUL/VSORT) ->
  4 lane tokens + END. DOT unchanged.

## Circuit (circuit.py, on top of v2's Vec4v2)

- Token features: add `op_vshl_f`, `op_vshr_f`, `op_mmul_f`,
  `op_vsort_f` (same `f()` pattern, `full=True`). OP head gains
  4 features (indices 8..11 in the slot_head feature dict — check
  the v2 ordering: 2..7 are the six v2 ops; extend, do not
  renumber). `op_bits` becomes 10 entries.
- VSHL/VSHR per lane: shifted versions a<<0 (= a), a<<1, a<<2,
  a<<3 as bit permutations (bit i of a<<k is a[i-k], 0 if i<k;
  result truncated to 4 bits); a>>k similarly (bit i is a[i+k],
  0 if i+k>3). Two-level mux by s bits (s0 = B bit 0, s1 = B
  bit 1), with mux(s, x, y) = select_bits([([s], x),
  ([one-s], y)]) picking x when s=1: out = select_bits(
  [([s1], mux(s0, a<<3, a<<2)), ([1-s1], mux(s0, a<<1,
  a<<0))]) — amount = 2*s1 + s0 = B & 3. (An earlier draft
  of this spec had the mux arguments inverted, which shifts
  by 2*s1 + (1-s0); the circuit test caught it — odd shift
  amounts fail under the inverted order. The implementation
  in circuit.py is the corrected order above.)
- MMUL: cross-lane 8-bit products. With A lanes 0..3 and B
  lanes 4..7, the eight products needed are (0,4), (1,6),
  (0,5), (1,7), (2,4), (3,6), (2,5), (3,7) — i.e. A0*B0,
  A1*B2, A0*B1, A1*B3, A2*B0, A3*B2, A2*B1, A3*B3.
  lane_prod[0] (= A0*B0) and lane_prod[3] (= A3*B3) already
  exist from DOT; build the other six with the same
  partial-product AND + adder-tree pattern. Each c entry =
  adder of the two products' low 4 bits, mod 16 (discard
  final carry).
- VSORT: sorting network, 5 compare-exchange stages in order:
  (0,1), (2,3), (0,2), (1,3), (1,2). Compare-exchange(x, y)
  returns (lo, hi) via the MAX/MIN pattern: diff = x - y
  (adder with ~y + 1), ge = final carry, masked = ge & diff,
  hi = y + masked, lo = x - masked (adder with ~masked + 1).
  For equal values ge=1, masked=diff=0, so (lo, hi) = (x, y):
  stable, as specified. Stages feed forward; output lanes are
  the stage-5 values.
- Selection: extend `lane_sel` select_bits with the four new
  (op_bit, lane_bits) pairs. At GO exactly one op bit is 1, so
  selection stays one-hot; at other positions all are 0.
- Exit code: add GO cases for the four new ops exactly like
  go_sub (lane 0 digit bits for j<4). Lane-transition cases
  (lane l -> l+1) are op-independent — result heads already
  hold the selected op's lane bits — and need NO change. DOT
  and END cases unchanged.
- Report in the RESULT file: layers, dims, max FFN units per
  sublayer, and the Target/cap comparison (v2 was 583 dims,
  max 154/sublayer vs the 3584 cap; v3 will be larger — if it
  approaches the cap, say so, do not silently raise the cap
  without the retarget analysis).

## Dispatcher (extend v2's dispatcher.py, same contract)

Add the four ops to the op table with patterns, keeping the
contract: semantic task beats mentions, any disagreeing mention
flags CONFLICT, ambiguous -> refusal. Seed patterns:
VSHL "shift ... left by", VSHR "shift ... right by",
MMUL "multiply the (two) matrices" / "matrix product",
VSORT "sort ... ascending" / "sort ... smallest to largest".
Single mention with no semantic signal still accepted; the
dispatcher guarantees EXECUTION fidelity, not intent fidelity
for a model-invented program. Unit-test the new patterns
(incl. a shift-amount case and a VSORT case) in
test_dispatcher.py.

## Tests required (local, CPU, before any Colab gate)

- test_ref_v3.py: hand edges + random cases across all 10 ops.
  Edges must include: VSHL/VSHR with B = 0, 3, 4 (amount wraps
  via &3: B=4 shifts by 0), 15; MMUL identity matrix, zero
  matrix, all-15s matrix (mod-16 wrap); VSORT already sorted,
  reverse sorted, all equal, duplicates.
- test_circuit_v3.py: IR circuit vs ref on all 10 ops
  (suggest 66/op like v2, seed 20261105), 0 mismatches,
  `ir.evaluate(..., check=True)` raising for no case.
- test_dispatcher.py: new-op patterns pass, conflict and
  refusal behaviour unchanged for old ops.
- run_tiny.py / run_merged.py: port from v2 (grep ALL paths
  after porting — the 2B-v2 stale-path lesson), chip_vocab 252.

## Colab gate (separate step, after local gates pass)

Stage 0: merge + Gate 0 on the new case set (10 ops), then
cache probe, then tool tests on the v2 generalisation set
extended with new-op cases. 0.8B first; 2B only after 0.8B
passes. Merges in separate processes (13 GB VM lesson).
Do NOT push anything to GitHub without a fresh instruction.
