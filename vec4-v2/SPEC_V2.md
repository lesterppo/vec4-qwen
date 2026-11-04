# VEC4 v2 — specification (local development, NOT yet merged in-model)

Status: Stages 1 (reference) and 2 (IR circuit) pass locally on CPU.
Stage 3 (merge into real Qwen) and Stage 4 (in-model tool tests) are
NOT done and are NOT claimed. Nothing in this directory has been
pushed to GitHub. When it is merged, Gate 0 must be re-proven from
scratch (210 v1 cases are a subset; the v2 gate is 396 cases here).

## What changed from v1, and why

v1 had two ops (ADD, DOT) and one interface failure mode (tool
selection). v2 attacks both axes ChatGPT's analysis prioritised:

1. **More ops (its Priority 1, partial).** SUB, MUL, MAX, MIN are
   lane-wise, so they reuse the v1 lane machinery. VSHIFT is
   deliberately deferred: a variable barrel shifter is not a
   lane-wise extension — shifting by a variable amount needs a
   different, larger unit, and pretending it is a small addition
   would be the kind of claim this project does not make. It is
   the next circuit after v2, not part of v2.
2. **A deterministic dispatcher (its Priority 2).** The model no
   longer decides which tool runs. `dispatcher.py` decides from
   the user's own text, with semantic task > tool-name mention,
   and executes exactly what it decided. See `dispatcher.py`'s
   docstring for the rules and, critically, for the boundary:
   execution fidelity is guaranteed; intent fidelity for a
   model-invented multi-step program is NOT — the program must
   come from the user's request or be approved before execution.

## ISA (v2)

4 lanes, 4-bit unsigned values 0..15, registers A and B in
DeltaNet state, exactly as v1. One op per program.

| Op | Token | Meaning |
|---|---|---|
| ADD | 0 | lane-wise `(A[i] + B[i]) mod 16` |
| DOT | 1 | `sum A[i]*B[i]`, max 900, 10 bits |
| SUB | 244 | lane-wise `(A[i] - B[i]) mod 16` |
| MUL | 245 | lane-wise `(A[i] * B[i]) mod 16` |
| MAX | 246 | lane-wise `max(A[i], B[i])` |
| MIN | 247 | lane-wise `min(A[i], B[i])` |

SUB..MIN live at 244..247 because the standalone vocab is padded
to 256 and those ids are free. They must NOT live at 2..5: ids
2..129 are VAL tokens, and an OP id there collides with a VAL
token and overwrites the OP head. This bug was found by the v2
circuit test (ADD with A0=0, whose VAL id is 2, read back as
OP_SUB) and fixed by the renumbering, not by masking the test.

## Token encoding (v2)

Input (10 tokens, unchanged shape): `[OP, A0..A3, B0..B3, GO]`.

Output:
- Lane ops (ADD, SUB, MUL, MAX, MIN): `[OUT_LANE 0 d0, OUT_LANE 1 d1,
  OUT_LANE 2 d2, OUT_LANE 3 d3, END]` where `OUT_LANE l d =
  131 + l*16 + d`. The ids are shared across lane ops on purpose:
  the output index for a lane op is `lane*16+digit` regardless of
  which lane op it is, so the exit code's lane-transition cases
  are op-independent. What differs by op is the circuit's selected
  result (see below), not the token ids.
- DOT: `[OUT_DOT 0 hi, OUT_DOT 1 mid, OUT_DOT 2 lo, END]`,
  unchanged from v1.
- END = 243, GO = 130, VAL `r v = 2 + r*16 + v`, as v1.

Total standalone chip tokens: 248 (0..247), padded to 256 in the
merged vocab exactly as v1's 244 were. Merged vocab therefore
stays 248576 on a 248320 base. The merge driver must be checked
for a hardcoded 244 when it is ported (see "Not yet done").

## Circuit (v2)

18 layers (same as v1), 583 dims (v1: 447), 14 heads (same),
1357 FFN units (v1: 961), max 154 units per sublayer (v1: 98) —
still far below the 3584 intermediate cap, so no redesign and no
cap change is needed for 0.8B. The 2B retarget arithmetic from
`vec4-2b/RETARGET.md` applies unchanged (cap 6144, chip width 256).

- OP head: 6 data columns (one per op), read at GO. At GO exactly
  one op bit is 1.
- Lane adders: unchanged (ADD).
- DOT product: unchanged (per-lane 8-bit products, adder tree,
  10 dot bits).
- SUB: two's-complement adder, `A + ~B + 1` (initial carry 1);
  the final carry is the `A >= B` flag (1 iff no borrow).
- MUL (mod 16): the low 4 bits of the lane product already
  computed for DOT. No new arithmetic.
- MAX: `B + ge*(A-B)`; MIN: `A - ge*(A-B)`, where `ge` is the SUB
  carry and `ge*(A-B)` is masked by AND gates. Neither needs a
  comparator beyond the SUB carry.
- Selected lane result: `select_bits` over the five lane-op
  results, keyed by the op bits. Result heads store the selected
  bits (4 heads, same count as v1), so the exit code's
  lane-transition cases work for every lane op unchanged. Only
  the GO cases are per-op (lane-0 digit of the requested op).

## Dispatcher (v2)

`dispatcher.py` + `test_dispatcher.py`. Rules: semantic task >
tool-name mention; any disagreeing mention flags CONFLICT (the
dispatcher still executes the semantic op, and never the mentioned
wrong tool); a single tool mention with no semantic signal is
accepted; multiple conflicting mentions with no semantic signal
are AMBIGUOUS and never executed; "subtract X from Y" swaps
operands (Y - X); out-of-range or missing vectors are errors.
Programs execute exactly as given, step by step, with `$n` refs
to earlier results; a bad ref, unknown op, or a DOT result used
as a vector aborts the whole program (PROGRAM_ERROR, no partial
result presented as a program result).

Two dispatcher bugs were found by its own tests and fixed in the
dispatcher, not in the tests: (1) subtract-from operand order,
(2) conflict detection that only fired when the semantic op was
absent from the mentions (now: any disagreeing mention flags it).

## Generalisation set (for the future Colab gate)

`generalisation.json` (generated by `make_generalisation.py`,
seed 20261104): paraphrase cases with no tool cue, adversarial
cases (misdirection, injected wrong tool, distractor vector,
stated-wrong-answer trap), multi-step programs, long-context
cases, a 10-request batch, and a 10-prompt plain-language probe
for base-vs-merged text comparison. Expected results are computed
with `ref.py` / `dispatcher.py`, not by a model. This set is the
test that would decide whether 2B's (and v2's) selection
generalises — it has not been run on any model yet.

## Not yet done (honest list)

- DONE locally 2026-11-04 evening (not yet run on Colab):
  merge driver port (`merge_vec4.py`, CHIP_VOCAB 248, merged
  vocab 248576, `full=True` token features kept from v1 —
  verified in circuit.py's `tr()` range offset+248), tiny test
  (`run_tiny.py`, 248 tokens, same 396-case set), Gate 0 runner
  (`run_merged.py`, 396 cases), stage0 (`stage0_v2.py`),
  cache probe (`cache_probe_v2.py`, 24 cases), tool driver
  (`tool_v2.py`: orig/para/adv/disp/plain configs).
- Colab Gate 0 for v2 (396 cases in-model) and cached-driver
  token re-proof on the v2 merged model — RUNNING on Peter's
  2026-11-04 20:36 tap (session vec4-v2, see RESULT_V2_LOCAL.md).
- Colab tool tests: v2 ops through the dispatcher, and the
  generalisation set on 0.8B-v2 and 2B-v2.
- Efficiency comparison (ChatGPT Priority 4): VEC4 vs Python vs
  CUDA vs CPU on latency/throughput. Deliberately not started:
  it needs the merged v2 model first, and comparing a 10 s/case
  model circuit against NumPy without that framing would repeat
  the analysis's own caution (the 6.5x was cached vs stepwise,
  never VEC4 vs NumPy).
- VSHIFT, matrix multiplication, sorting: future circuits, not v2.
