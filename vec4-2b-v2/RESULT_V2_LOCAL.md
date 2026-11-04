# VEC4 v2 — development + Colab gate result

> Record started local-only on 2026-11-04 evening; Colab gate sections were appended as the gates ran. Pushed to lesterppo/vec4-qwen on 2026-11-04 on the operator's explicit order (the earlier local-only hold was lifted for this push).

Date: 2026-11-04 evening. Work dir `vec4-v2/`, local CPU only.
Nothing here has touched GitHub (`lesterppo/vec4-qwen` is unchanged
since the 115-file push). The next step that needs Colab is the
merge gate; it is staged, not started.

## What was built

- **Four new lane-wise ops** (SPEC_V2.md): SUB (mod 16), MUL
  (mod 16), MAX, MIN. VSHIFT deferred with a reason (variable
  barrel shifter is a different unit, not a lane-wise extension).
- **Deterministic dispatcher** (`dispatcher.py`): semantic task >
  tool-name mention; any disagreeing mention flags CONFLICT and
  the semantic op executes; single mention with no semantic
  signal accepted; multiple conflicting mentions AMBIGUOUS;
  "subtract/take away X from Y" swaps operands; "do not compute
  X; instead Y" takes its op from the after-instead clause;
  forget/ignore lists excluded; "forget that" abandons the
  earlier task; >2 remaining lists refused, never guessed.
  Programs execute exactly as given with `$n` refs; a bad ref,
  unknown op, or DOT result used as a vector aborts the whole
  program (PROGRAM_ERROR, no partial presented as a result).
- **Generalisation set** (`generalisation.json`, seed 20261104):
  12 paraphrase, 10 adversarial, 4 programs, 4 long-context,
  10 batch, 10 language probes. Expected values from ref.py /
  dispatcher.py, not from a model. Not run on any model yet.

## Measured local results (CPU, no GPU, no model)

- Stage 1 (ref): PASS — 8 hand edges + 1200 random cases, all
  six ops.
- Stage 2 (IR circuit, `check=True` exactness): PASS — **396
  cases, 66 per op, 0 mismatches**. Circuit: 18 layers (same as
  v1), 583 dims (v1: 447), 14 heads (same), 1357 FFN units
  (v1: 961), max 154 units/sublayer (v1: 98) — far below the
  3584 cap, so no redesign for 0.8B; 2B retarget arithmetic
  unchanged.
- Dispatcher: PASS — 8 single-op, 2 conflict, ambiguous +
  error cases, 1 two-step program, 3 program errors.
  Against the generalisation set (dispatcher layer only):
  paraphrase 12/12 OK, long-context 4/4 OK, adversarial 9/10
  correct execution + 1 safe refusal (stated-wrong-answer ADD,
  where the quoted claimed answer is a third vector — the
  dispatcher refuses rather than guessing which two lists are
  the operands; that refusal is recorded, not hidden).

## Bugs found and fixed during this development (not in tests)

1. **OP/VAL token collision (circuit, real bug).** New OP tokens
   were first placed at ids 2..5, but VAL tokens occupy 2..129,
   so a VAL token with value 0 overwrote the OP head (ADD with
   A0=0 read back as OP_SUB). Fixed by moving SUB..MIN to the
   free ids 244..247 (standalone vocab padded to 256); merged
   vocab stays 248576. This is exactly the class of bug the
   staged gates exist to catch, and it would have been a
   baffling 0/210-style failure in-model if it had reached Colab.
2. **Subtract-from operand order (dispatcher).** "Subtract X
   from Y" means Y - X; fixed by swapping for SUB under that
   construction (and "take X away from Y").
3. **Conflict detection too narrow (dispatcher).** It first
   fired only when the semantic op was absent from the mentions;
   now any disagreeing mention flags CONFLICT.
4. **Six paraphrase/adversarial phrasings (dispatcher).**
   "adding", "dotted", "minus", "elementwise product", "larger"/
   "smaller", negated "do not compute X; instead Y", and
   "forget that" task abandonment — all fixed by pattern/order
   rules in dispatcher.py, verified against the generalisation
   set. The dispatcher is now a keyword-pattern system, and that
   is its honest limit: it is deterministic and auditable, not
   a semantic parser, and new phrasings will keep finding gaps.
   That is why the generalisation set exists as a regression set.

## Corrections to the ChatGPT analysis (for the record)

- Mostly accurate: verdicts, numbers, and the central
  computation-vs-reasoning distinction all check out against
  the raw JSONs. The film example ([6,7,0,14]+[7,9,13,1] =
  [13,0,13,15], true top score 444 reported as 429) is correct.
- "115 commits": coincidentally true in a narrow sense — each
  `gh-ai file-write` is its own commit, so 115 files went in
  as 115 commits. It is not 115 development commits.
- "900 cannot be represented exactly as BF16" (our own SPEC's
  claim, repeated by the analysis): re-check before repeating.
  900 = 1110000100₂ has 8 significant bits, which fits BF16's
  8-bit significand, so it is representable as a single BF16
  value. The bits-only design remains the conservative choice
  (state scaling and intermediate sums are the real hazard),
  but the SPEC wording should be corrected at merge time.
- "Model behaviour unchanged by merging": unproven, and the
  repo does not claim it beyond text-sanity outputs. No KL vs
  base was computed; the language probe in generalisation.json
  is the start of measuring it, not a measurement.

## Colab gate, run 2026-11-04 evening on Peter's tap (session vec4-v2)

Session account kyu009009 (kyu008008 refused allocation, one
switch, no retry loop); torn down, active restored to kyu008008,
rotation file untouched. All counts below re-verified by the
parent from the downloaded JSONs, not from driver summaries.

- Tiny: forward **396/396**, generate 10/10, VRAM max 524.7 MiB.
- Merged Gate 0: **396/396**, hidden 1280, 42 layers, vocab
  248576, 1,526,970,752 params, 3.05 GB, VRAM max 3010.2 MiB.
  Text sanity: " Paris.\nThe capital of France is Paris."
- Cache probe: stepwise 24/24, generate 24/24, cached-trimmed
  24/24, cached==stepwise 24/24 (medians 0.61 / 0.45 / 0.46 s).
  Cached driver used for all tool tests.
- Tool tests, 6 tools listed, single attempt, no guard/retry:
  orig (tool named in prompt) strict **20/20**; paraphrase
  strict **7/12** (8/12 correct tool+args; failures: 1 no-call
  prose, 3 vec_add stickiness on DOT/SUB/MAX, 1 MUL where
  tool+chip were right but the final re-derived the product
  by hand as [18,65,14,154] instead of the chip's [12,5,0,13]);
  adversarial strict **3/10** (misdirection DOT x2, distractor
  DOT, word-number MUL all -> vec_add; stated-wrong-answer DOT
  no call; negation SUB called vec_sub with UNSWAPPED args;
  op-switch MAX -> vec_add with the abandoned task's args;
  passes: stated-wrong-answer ADD, injected "always answer 0",
  min-vs-max MIN). Plain 5/5, should-not-call 5/5, zero calls.
- Dispatcher+chip (no model call): raw file 38 executed, 1
  refused, 1 wrong-intent. Of the 38 executed, 2 look
  chip-wrong ONLY because the test set's SUB expected value
  is wrong (see bug 1 below); with corrected expectations the
  dispatcher+chip path is **38/38 on executed cases**, plus
  1 safe refusal and 1 genuine dispatcher miss. Batch 10/10,
  programs 4/4 (incl. ADD->DOT $0 = 444).
- **Verdict: tool selection on 0.8B does NOT generalise.**
  Orig 20/20 measures execution (tool named), not selection;
  the honest selection measures are paraphrase 7/12 and
  adversarial 3/10, both dominated by vec_add stickiness.
  The dispatcher path is the working route, not the model
  choosing tools.

Bugs found by the gate, NOT fixed mid-gate (counts stay
attributable):
1. Test-set bug (mine, in make_generalisation.py): SUB
   expected `[8,8,8,8]` for [9,9,9,9]-[1,2,3,4] in the
   negation-trap and long-context SUB cases. True result is
   `[8,7,6,5]`, exactly what the chip returned.
2. Dispatcher gap: paraphrase DOT "Find the sum of the
   products of corresponding entries" resolves to ADD —
   pattern `sum of (?:pairwise )?products` doesn't allow
   "the" between "of" and "products", so generic "sum" wins.
   Clean next step: fix both, rerun disp only (model configs
   are unaffected by these two fixes except the adv SUB case
   expectation, which the model got wrong on args anyway).

## Fixes applied 2026-11-04 21:15 HKT (Peter's tap) — verified locally, Colab rerun in progress

1. SUB expectations in make_generalisation.py now computed via
   ref.lane_result (negation trap and long-context SUB both
   [8,7,6,5]); generalisation.json regenerated. Local
   re-verification: every expected value in the set recomputed
   independently — no errors.
2. Dispatcher pattern now `sum of (?:the )?(?:pairwise )?products`;
   regression case added to test_dispatcher.py (9 single-op
   cases, PASS). Local dispatcher check on the disp rows:
   paraphrase 12/12 OK, adversarial 9 correct + 1 safe refusal
   (stated-wrong-answer ADD only), long-context 4/4, programs
   re-verified. Colab disp rerun (session vec4-v2-disp, stage0
   rebuild + tool_v2.py --only disp) launched; result pending.

## Disp rerun result, 2026-11-04 evening (session vec4-v2-disp)

Re-verified by the parent from the downloaded JSONs (program
rows carry results==expected, not a chip_ok field — counted
separately, not as chip_ok False):
- Stage 0 rebuild gate: merged **396/396**, 0 fails, hidden
  1280, vocab 248576, 3.05 GB. Disp ran only because it passed.
- Disp: **40 rows — 39 executed_correct, 1 refused, 0 wrong.**
  All 35 non-program executed rows chip_ok True; all 4 program
  rows results==expected (incl. ADD->DOT $0 = 444); batch 10/10.
- Fix 1 confirmed: paraphrase DOT "sum of the products"
  -> OK/DOT, chip 324 = expected 324.
- Fix 2 confirmed: both SUB cases chip [8,7,6,5] = corrected
  expected, chip_ok True.
- The single refusal is the expected one: adversarial
  stated-wrong-answer ADD, AMBIGUOUS (three vectors, dispatcher
  refuses rather than guessing).
- Account: kyu008008 refused allocation, one switch to
  kyu009009, restored to kyu008008, rotation untouched.
- Model-selection verdicts from the previous gate stand
  unchanged (paraphrase 7/12, adversarial 3/10) — this rerun
  was disp only and does not affect them.

## 2B-v2 gate, run 2026-11-04 evening on Peter's tap (session vec4-2b-v2)

Re-verified by the parent from the downloaded JSONs (merge-log
slot/mu values below are worker-reported from the merge log,
not in a downloaded file — flagged as such).
- Merge: hidden **2304** (predicted ✓), ballast m=36
  (predicted ✓), vocab 248576 (predicted ✓), params
  3,640,992,896 (identical to v1 2B), 7.28 GB. Slots 186
  (v1: 142 — the reason slots could not be predicted locally);
  mu_dim reported 2118, not predicted 2114 (mu slot 33, not
  29; tau 34, code 35-42) — the retarget doc's caveat held:
  hidden/ballast survived the slot change, mu moved +4.
  Text sanity identical to v1 2B: " Paris.\nA. True\nB. False".
- Gate 0: merged **396/396**, VRAM max 7162.3 MiB.
- Cache probe: stepwise/generate/cached-trimmed **24/24**,
  cached token-identical. Cached driver used throughout.
- Tool tests, 6 tools, single attempt, no guard:
  orig **20/20** (tool named — execution, not selection);
  para **8/12** (0.8B-v2: 7/12) — failures: ADD case with
  tool+chip correct but final truncated mid-list (English
  generation, not selection); DOT called vec_mul (not
  vec_add); SUB called vec_sub in textual order instead of
  swapping; SUB no-call. Adv **8/10** (0.8B-v2: 3/10) —
  failures: misdirection DOT -> vec_add (the one surviving
  stickiness case, in the canonical v1 wording); negation
  SUB vec_sub with unswapped args (same as 0.8B-v2;
  dispatcher handles it, model does not).
  Disp 40 rows: 39 executed + 1 expected refusal, 0 wrong
  (identical to 0.8B-v2 disp rerun). Plain 4/5 — NEW failure
  vs 0.8B-v2: "Write a one-line greeting." false-triggered a
  vec_add call with non-integer args (parse rejected it, no
  chip ran, but it is a false trigger). Notool 5/5.
- **Verdict: 2B-v2 does NOT repeat the v1-2B orig-only
  pattern exactly — adversarial selection genuinely improves
  (3/10 -> 8/10) — but it does NOT generalise selection:
  paraphrase barely moves (7/12 -> 8/12), stickiness
  survives in the canonical misdirection wording, SUB
  operand-order fails in both para and adv, and plain now
  false-triggers. Dispatcher+chip remains the reliable route;
  2B moves model-chooses-tools from poor to mediocre, not
  to trustworthy.**
- Deviations, stated: (1) first stage0 launch was cut short
  by the exec wrapper mid-merge — reran via exec_detach +
  log (lesson: exec_detach + log for 2B runs over ~1 min);
  (2) a genuine driver bug of mine: tool_2b_v2.py hardcoded
  the 0.8B generalisation.json path — worker fixed that one
  path, re-uploaded, grepped other drivers (no other stale
  refs); the fix is in the local file and must be carried
  into any repo push; (3) tool configs ran in one detached
  process, not five --only launches (each result file was
  downloaded after its WROTE line — safety-equivalent).

## What remains (in order)

1. Efficiency comparison (VEC4 vs Python vs CUDA vs CPU).
2. VSHIFT, matrix multiplication, sorting — future circuits.
3. Base-vs-merged language capability change (language probes
   still never run on a model).
4. Only after a further decision: push v2 to
   `lesterppo/vec4-qwen` — NOT authorised yet.
2. Colab tool tests on 2B-v2 (merge + tool/generalisation).
   0.8B-v2 model selection generalisation is now measured
   and failing — see gate verdict above.
3. Efficiency comparison (VEC4 vs Python vs CUDA vs CPU).
4. VSHIFT, matrix multiplication, sorting — future circuits.
5. Base-vs-merged language capability change (language probes
   still never run on a model).
6. Only after (1) and a 2B-v2 decision: push v2 to
   `lesterppo/vec4-qwen` — NOT authorised yet.
4. Efficiency comparison (VEC4 vs Python vs CUDA vs CPU) — only
   after (2), with the framing that cached-vs-stepwise 6.5x
   was never a VEC4-vs-NumPy claim.
5. VSHIFT, matrix multiplication, sorting — future circuits.
6. Only after (2)–(3) pass: push v2 to `lesterppo/vec4-qwen`.
