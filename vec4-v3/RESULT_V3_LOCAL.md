# VEC4 v3 — local development result (CPU only, NO Colab gate yet, NOT pushed)

> Pushed to lesterppo/vec4-qwen on 2026-11-05 on the operator's explicit order (the v3 local-only hold was lifted for this push only). The Colab gates and disp re-proof recorded below ran after this file was started; they are included in this push with their result JSONs under vec4-v3-gate/, vec4-v3-gate-2b-lowmem/ and vec4-v3-disp-reproof/.

Date: 2026-11-04 evening (parent session; this subagent run).
Work dir `vec4-v3/`, built from `vec4-v2/` sources + upstream
chipformer clone (both read-only; neither was edited). No Colab
session, no GPU, no model download, no GitHub push in this run —
per the task rules. All tests below ran locally on CPU.

## What v3 adds (per SPEC_V3.md, written before implementation)

10 ops: v2 six (ADD=0, DOT=1, SUB=244, MUL=245, MAX=246, MIN=247)
plus VSHL=248, VSHR=249, MMUL=250, VSORT=251. CHIP_VOCAB 248 -> 252;
merged vocab UNCHANGED at 248576 (ceil(248572/64)*64 = 248576,
verified arithmetically). All lane-wise outputs share OUT_LANE ids
as in v2; DOT unchanged. VSHL/VSHR shift by B's low 2 bits per
lane; MMUL is a 2x2 matrix product mod 16 row-major in lanes;
VSORT sorts A ascending (stable) and ignores B (B still present
and well-formed). Token features, OP head (indices 8..11, no
renumbering), `tr()` bound (offset+252), `_is_vec4` (0..251),
`_is_op` set, merge CHIP_VOCAB all updated to 252.

## Circuit size (from test_circuit_v3.py output)

- layers: 26 (v2: 18)
- dims: 1287 (v2: 583)
- heads: 14, ffn_units: 3170, max_units per sublayer: 346
  (v2: 154), max_heads: 9
- Target/cap comparison: max 346 vs the 3584 FFN cap = 9.7% of
  cap. The cap is NOT approached; no retarget decision is needed
  and the cap was not changed. (A 2B retarget would still need
  its own pre-merge analysis, as in v2, before any merge.)

## Test results (all PASS, nothing softened)

- test_ref_v3.py: PASS — 24 hand edges (incl. VSHL/VSHR with
  B = 0, 3, 4, 15; MMUL identity, zero, all-15s -> [2,2,2,2],
  general case; VSORT sorted, reverse, all equal, duplicates)
  + 3 VSORT B-independence cases (same A, B = 0s / 15s / 1-4)
  + 2000 random cases (200/op, seed 20261105). Random cases are
  recomputed INLINE in the test (explicit formulas, insertion
  sort for VSORT), not by calling ref.py functions, so the sweep
  is an independent check of ref.py.
- test_circuit_v3.py: PASS — 720 cases (12 edges + 60 random per
  op, seed 20261105), 0 mismatches, ir.evaluate(check=True)
  raising for no case. GO-position compute bits and first-output
  result-head bits checked against ref for every case. Edge set
  covers every shift amount (0..3, plus wrap via B=4 and B=15)
  and VSORT with two different B vectors, so the shift muxes
  and the sorting network are verified by the test, not by
  reasoning alone. Runtime ~3m37s on this VM's CPU.
- test_dispatcher.py: PASS — 18 single-op cases (v2 six + v3
  four, incl. shift-amount cases and the "matrix product" /
  "smallest to largest" collision cases), 3 conflict cases
  (semantic beats mention, incl. one new-op conflict), 2
  ambiguous cases (never executed), error cases (NO_VECTORS,
  OUT_OF_RANGE), 2 two-step programs (incl. MMUL -> VSORT on
  $0), 3 program errors. Old-op conflict/refusal behaviour
  unchanged (v2 cases kept verbatim).

## Ports (Colab-only, NOT run locally — no torch/GPU here)

- run_tiny.py: ported from v2 (720-case set, seed 20261105,
  tiny.build range 252, out /content/vec4-v3_tiny_result.json).
- run_merged.py: ported from v2 (720-case set, out
  /content/vec4-v3_merged_result.json).
- merge_v3.py: ported from merge_vec4.py (CHIP_VOCAB 252,
  build_vec4v3, layout "vec4-v3").
- All 9 v3 files pass Python syntax checks. ALL paths grepped
  after porting: no vec4-v2 / vec4-2b-v2 / 20261104 / range(248)
  / stale-path leftovers in the ports or the merge driver
  (the 2B-v2 stale-path lesson, applied). These three drivers
  are UNPROVEN until a Colab run — they are ports, not results.

## Deviations from SPEC_V3.md (one, with reason)

1. SHIFT MUX ARGUMENT ORDER (real deviation, test-arbitrated).
   The spec writes out = select([([s1], mux(s0, a<<2, a<<3)),
   ([1-s1], mux(s0, a<<0, a<<1))]) with mux(s, x, y) =
   select([([s], x), ([1-s], y)]). Taken literally that shifts
   by 2*s1 + (1-s0) — bit 0 inverted — not by B & 3. This
   implementation uses mux(s0, a<<3, a<<2) / mux(s0, a<<1,
   a<<0), so amount = 2*s1 + s0 = B & 3, i.e. it follows the
   spec's SEMANTICS (shift by B & 3) over its written argument
   order. The circuit test proves it: shift edges at amounts
   0,1,2,3 and wrap amounts (B=4, B=15) all pass; the literal
   spec order would fail every odd-amount case. Recorded in
   circuit.py's docstring as well as here.
2. Minor, non-semantic: the DOT lane-product loop was factored
   into a `lane_product(a_bits, b_bits, name)` helper shared
   with MMUL's six cross-lane products (same AND + adder-tree
   pattern, only operand registers differ). Behaviour identical
   — the 720-case circuit test, including all v2 op cases,
   proves it.
3. Test counts differ from the spec's suggestion (66/op) only
   upward: 72/op in the circuit test (12 edges, not 6) and
   200/op in the ref random sweep. No test was weakened.

## Blocked / not done (separate steps, need fresh instruction)

- Colab gate: stage 0 merge + Gate 0 (720 cases), cache probe,
  tool tests on a v2-generalisation set extended with new-op
  cases. 0.8B first, 2B only after 0.8B passes, merges in
  separate processes. NOT started (this run was local-only).
- GitHub push: NOT done and NOT authorised by this run.
- No failures, no blocked local items. Nothing in vec4-v2/,
  vec4-2b-v2/, vec4-eff/, or ~/workspace/vec4-repo/ was touched.

## Addendum 2026-11-05: Colab gates complete, dispatcher gaps fixed
(Header "NO Colab gate yet" above is stale; this addendum records
what happened after. Gate JSONs: `~/workspace/qwenputer-colab/
vec4-v3-gate/` (0.8B) and `vec4-v3-gate-2b-lowmem/` (2B, lowmem
merge -- see RESULT_LOWMEM.md). 0.8B: Gate 0 720/720, cache 40/40,
disp 41 executed + 2 refused + 2 wrong. 2B lowmem: Gate 0 720/720,
cache 40/40, orig 26/28, para 11/16, adv 8/10, disp identical
41+2+2, plain 4/5, notool 5/5. All breakage in disp was para_new.)

### Dispatcher gap fixes (local, 2026-11-05)
Three fixes in `dispatcher.py`:
1. MMUL pattern order: "product of the two matrices" matched
   generic `\bproduct\b` -> MUL because no MMUL pattern covers
   "product ... matrices" (product BEFORE matrices). Added
   `product .* matrices` to the MMUL pattern (first in order).
2. VSORT single-vector path: "Put [7,2,15,2] in ascending order"
   has one list, and parse_vectors demanded two -> NO_VECTORS.
   Refactored the scan into shared `_scan_vectors`; added
   `parse_single_vector`; dispatch falls back to it ONLY when
   the intended op resolves to VSORT (semantic or single
   mention), supplying a dummy B of zeros that VSORT never
   reads. Every other op still requires two vectors; a bare
   list with no op signal still refuses; an out-of-range VSORT
   list is OUT_OF_RANGE.
3. VSHL "to the left" pattern: "Shift ... to the left, lane by
   lane, by the amounts" matched neither `shift .* left by`
   (no contiguous "left by") nor `shift left`. Added
   `to the left` (and symmetric `to the right` for VSHR, same
   gap class).
Stacked catch: the VSORT prompt ALSO needed a semantic fix --
"in ascending order, smallest first" would resolve to MIN via
`\bsmallest\b` even with two vectors. Added `in ascending
order` to the VSORT pattern, before MIN. The single-vector fix
alone would have executed the wrong op.
Tests: `test_dispatcher.py` PASS (23 single-op cases incl. the
three exact failing prompts, plus guards: single-vector ADD
still NO_VECTORS, out-of-range VSORT still OUT_OF_RANGE, bare
list still NO_VECTORS, plain "product" without matrices still
MUL). `replay_disp_local.py` (new: replicates tool_v3 disp
logic with the ref executor) predicts the Colab disp outcome:
45 rows, 44 executed_correct + 1 expected refusal
(stated-wrong-answer ADD), 0 wrong -- para_new 1 -> 4, no
regressions anywhere else. Colab disp re-proof with the chip
executor NOT RUN yet: it needs a fresh merged model
(checkpoints are discarded at teardown) and Peter's tap. The
local replay + Gate 0 chip correctness is strong evidence, but
the on-chip disp file is the proof that counts.
PROOF OBTAINED 2026-11-05 ~09:05 HKT (session
vec4-v3-disp-reproof, kyu008008 first try; JSONs in
`~/workspace/qwenputer-colab/vec4-v3-disp-reproof/`, counts
parent-verified): fresh lowmem merge Gate 0 720/720, then disp
45 rows = 44 executed_correct + 1 expected refusal
(stated-wrong-answer ADD), 0 wrong -- EXACTLY the local replay
prediction, row for row. Para_new 4/4 with chip values
[2,8,8,4], [3,5,0,15], [3,6,11,2], [2,2,7,15]; chip_ok 39/39
(the other 5 executed rows are programs, counted via
results==expected). Dispatcher+chip is now clean on all disp
rows. Fix pre-check passed on the VM before the run (greps for
"to the left", parse_single_vector, the MMUL and VSORT
patterns).
