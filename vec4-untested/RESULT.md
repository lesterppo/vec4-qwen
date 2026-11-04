# VEC4 untested trio — RESULT

## Verdicts first

- **Gate 0: PASS 210/210** (same merged model, rebuilt from scratch).
- **SPEED: SPEED_CACHED_GREEN.** Per-phase KV-cache driver is exact and
  ~6.5x faster on tool cases. Flash kernels were NOT installed: the
  task's condition for that attempt was "if (a) fails on DeltaNet
  cache", and (a) did not fail — so stepwise reference-kernel speed
  is unchanged and flash kernels remain untested, by design, not by
  omission of a required step.
- **B2: B2_TESTED_1.** Host-rewrite fallback exercised IN-MODEL on
  1/10 adversarial DOT cases, strict 1/1 on that case. It is NOT
  B2_SYNTHETIC_ONLY: the synthetic test was not run, because the
  in-model case already exercised the path (synthetic runs only if
  in-model count is 0, per task). B2 is never counted as B1.
- **INTERLEAVED: INTERLEAVED_PARTIAL_ENGLISH_BARRIER.** Chip
  interleaving itself is green — I1 20/20, I2 both chips 5/5, I3 chip
  5/5, all in ONE id sequence with no reset — but final English
  cannot report the chip result from the interleaved sequence alone
  (true rate 0/20, see the substring-artifact note below). That is
  an architectural information barrier (mute, see below), not a chip
  failure, and it does not overturn the chip result. Text after chip
  stays sane: I3 4/5, identical to a text-only control 4/5, so there
  is no anti-sink garble finding on these prompts.

## Scoring (strict, as before)

Strict = requested op == called tool AND args == case args AND chip
run for the requested op matches `ref.py` AND final English contains
the correct requested result AND final raw contains no `<tool_call>`
block. B2 strict = chip for the requested op + final English correct
+ no tool_call in final raw, on a host-rewritten case only. For
interleaved I1/I2, "final English contains result" is scored by
substring as written, but the two I1 hits are flagged below as
substring artifacts, not answers — the true English-answer rate is
stated separately and is not hidden behind the substring count.

## Summary table

| Experiment | Cases | Chip exactness | Strict final rate | Seconds/case | VRAM max | What changed vs previous runs |
|---|---|---|---|---|---|---|
| Gate 0 chip-only regression | 210 | 210/210 | — | 123.7 s gen | 3,010.2 MiB | Rebuild, same spec as vec4/vec4-fix |
| Exp1 chip: stepwise | 20 tool cases | 20/20 | — | median 0.60 s | — | Baseline chip phase |
| Exp1 chip: stock `generate()` | 20 | 20/20 | — | median 0.44 s | — | Tiny model had passed stock generate 10/10; merged now passes 20/20 too |
| Exp1 chip: manual KV cache, END-trimmed | 20 | 20/20, tokens identical to stepwise 20/20 | — | median 0.44 s | — | Cache is exact on chip tokens |
| Exp1 tool cases, cached driver (Variant A, single tool) | 20 (10 DOT + 10 ADD) | 20/20 (trimmed, see correction) | **20/20 corrected** | median **10.1 s**, total 200.7 s | 3,255.9 MiB | vs stepwise Variant A 1,399.5 s / 20 (~70 s/case): **~6.5x** |
| Exp1 tool cases, stepwise subset (fresh baseline check) | 6 (DOT 0,2,5; ADD 10,12,17) | 6/6 | 6/6 | median 65.6 s (50.6–73.2) | — | Confirms the ~70 s/case baseline on this session; finals identical to cached finals on all 6 |
| Exp2 B2 adversarial (both tools, guard+retry) | 10 DOT | 10/10 requested-op chips | B1 9/10, **B2 1/1 in-model** | median 17.2 s (cached driver; 14.9–19.7) | 3,371.3 MiB | First-attempt correct 0/10 under misdirection (vs 1/10 baseline); retry corrected 9/9 vec_add-first cases |
| Exp3 I1 single interleaved | 20 (10 DOT + 10 ADD) | **20/20** | English result 2/20 by substring, **0/20 true** (see note) | ~6.0 s (cached) | — | First token-interleaved test ever for VEC4 |
| Exp3 I1-pure diagnostic (no cue after END) | 5 DOT | 5/5 | 0/5 | ~6.1 s | — | Text after END is coherent but unrelated code text, no chip tokens |
| Exp3 I2 double interleaved | 5 pairs | ADD 5/5, DOT 5/5, both 5/5 | both results 0/5 | ~8.4 s | — | Second chip block not corrupted by first block or bridge text |
| Exp3 I3 text-after-chip | 5 factual prompts | 5/5 | sane 4/5 | ~4.3 s | — | Control (no chip) also 4/5 with the same single failure |
| Exp3 control (text only, post-hoc) | 5 same prompts | — | contains expected 4/5 | ~4 s | — | Proves I3's one failure is base-model quality, not chip-induced |
| Exp3 overall | — | — | — | 219.6 s total | 3,024.4 MiB | — |

Case lists: Exp1/Exp2/Exp3 use the exact 20 tool cases from
vec4-fix Variant A (DOT edges + 6 random, ADD edges + 6 random,
`Random(20261103)`), verified 20/20 against
`../vec4-fix/variant_a_result.json` before the run. Exp2 uses the
same 10 DOT cases, varying ONLY the misdirection wording. Exp3 I2
pairs are (ADD, DOT) case indices (10,0), (11,1), (12,2), (14,4),
(15,5). Exp3 I3 chip cases are DOT 0, ADD 10, DOT 1, ADD 11, DOT 2.
No case-list difference from the previous runs except the
documented Exp1 stepwise subset (6 of 20, fresh baseline check
only — the full stepwise baseline is Variant A's 20-case run).

## Gate 0 — rebuild + regression: 210/210

One T4 session (`vec4-untested`, account kyu008008, see session
note). Probe: Python 3.13.16, torch 2.11.0+cu130, Tesla T4
14,912 MiB (torch-reported), 2 CPUs, 13 GB RAM, 120.8 GB free.
Upstream cloned fresh, base Qwen/Qwen3.5-0.8B downloaded, merged
exactly as before: hidden 1280, 42 layers (18 chip + 24 Qwen),
vocab 248576, 1,526,970,752 params, 3.05 GB bf16, chip offset
v0=248320, 142 chip slots, ballast m=20, mu slot 1074.
Regression 210/210 in 123.7 s generation, VRAM max 3,010.18 MiB.
Text sanity again: "The capital of France is" ->
" Paris.\nThe capital of France is Paris." (ids
[11751, 13, 198, 760, 6511, 314, 9338, 369, 11751, 13]).
Stage 0 total 208.2 s including download + merge. Gate 0 passed
before any experiment; no experiment ran on a failed gate.

## Mute — what it does and does not protect (read before Exp3)

From `merge_vec4.py` (`mute(sd, tc, n0, lay.dim(mu), hidden**0.5,
sinks=None, first=n_chip)`) and `upstream/chipformer/q35/mute.py`:

- **Qwen layers ARE muted on chip tokens.** For each Qwen layer,
  the mu-slot column is set to -1024: linear_attention
  `in_proj_a/b/z` (so softplus(a)=0 → state does not decay,
  beta=0 → nothing is written, silu(z)=0 → output exactly 0) and
  full-attention output gates (sigmoid=0 → output exactly 0).
  Qwen recurrent state therefore passes through a run of chip
  tokens unchanged, and chip positions add nothing to the Qwen
  residual. This is why text before chip and Qwen state across
  chip are safe.
- **Chip layers are NOT gated by token type in the Qwen sense;
  they are gated by token FEATURES.** Register heads key only on
  "this token is VAL for this register" and query only at GO; the
  OP head keys only on OP and queries only at GO; result heads
  key on GO and query only at OUT positions; the exit code is
  owned only where `owner = is_go + is_out`. Text tokens carry
  is_text / tau=TAU_TEXT / code=-128 features, so they write no
  register, trigger no exit, and are not read as outputs.
- **DeltaNet state and text tokens:** chip register/OP/result
  heads do not write on text tokens (wrong key), but Gated
  DeltaNet state can in principle decay per step. The empirical
  answer is in I1/I2 below: text before any write (empty state)
  and a short text bridge between two chip blocks (second block
  overwrites every register and result head) cause no chip error.
- **Anti-sink was SKIPPED (`sinks=None`).** Chip positions are
  still KEYS in Qwen full attention, with values ~0 and no
  fitted anti-sink key. Later text therefore attends over chip
  keys (softmax dilution / sink risk) but receives no chip
  information from them. This is the crux of the Exp3 English
  result: there is NO channel by which later English could know
  the chip result — Qwen is muted at chip positions, chip rows
  carry no Qwen-part information, and chip attention values are
  ~0. The toolcall bridge works precisely because the host
  re-injects the result as TEXT (`<tool_response>`).

## Experiment 1 — SPEED: SPEED_CACHED_GREEN

### What was tried, in order

(a) **Cached generation.** Two forms, both on the merged model:
(i) stock `model.generate(..., use_cache=True)`; (ii) manual
per-phase KV cache — one full forward per generation phase, then
single-token forwards with `past_key_values` (transformers 5.x
`DynamicCache`; host injections start a new phase with one full
forward, so no cache object is carried across a host injection).
The tiny model had previously passed stock generate 10/10; the
question was whether the merged model does too.
(b) **Flash kernels: NOT attempted.** The task orders (b) only
"if (a) fails on DeltaNet cache". (a) did not fail (see the
correction note below — the first 0/20 was a scoring bug in my
own script, not a cache failure). Reference-kernel warnings
(`causal_conv1d_fn`, `chunk_gated_delta_rule`,
`flash-linear-attention` falling back to reference PyTorch
implementations) therefore persist, and stepwise speed is
unchanged. Flash kernels remain untested; no install was burned
on a session where the cached path already worked.

### Correction note — read this before the raw JSON

The first Exp1 run reported cached chip 0/20 and cached tool
strict 0/20. That was MY scoring bug, not a cache divergence,
and it is corrected here with evidence, not silently overwritten:
`gen_cached` stops only at text EOS, not at chip END, so the
cached chip generation contained the exact expected tokens PLUS
post-END tokens (e.g. DOT case 0: `[195, 211, 227, 243,
-248320, -238329]` vs expected `[195, 211, 227, 243]` — the
negative entries are text token ids minus v0). The raw
`chip_gen` lists in `exp1_speed_result.json` already prove the
prefix matches expected in 20/20 cases. A dedicated correction
run (`exp1_chip_fix.py`, END-trimmed like `run_merged.py`) then
confirmed: stepwise 20/20, stock generate 20/20, cached-trimmed
20/20, cached tokens identical to stepwise 20/20. The corrected
tool strict (correct tool+args + final contains + no tool_call,
chip from the fix) is 20/20; see `exp1_corrected.json`. The raw
`exp1_speed_result.json` cached strict field (0/20) is superseded
by that correction and must not be quoted alone.

### Results

- Chip phase (20 tool cases): stepwise median 0.60 s, stock
  generate median 0.44 s, manual cached (trimmed) median
  0.44 s — all 20/20 exact. The chip phase itself was never the
  bottleneck; the win is in long tool prompts.
- Tool cases, Variant A single-tool, cached driver: correct tool
  + args 20/20, chip (trimmed) 20/20, final English contains
  expected result 20/20, no tool_call in final raw 20/20,
  corrected strict 20/20. Per-case median 10.1 s, total 200.7 s.
- Tool cases, stepwise subset (fresh baseline, same driver as
  Variant A): 6/6 strict, median 65.6 s (50.6, 64.8, 65.2, 65.9,
  66.1, 73.2). Finals on all 6 subset cases are IDENTICAL to
  the cached finals — the cached driver is token-identical to
  stepwise on tool text phases, not merely strict-equal.
- Speedup: 65.6 / 10.1 ≈ **6.5x** on tool cases. Against the
  full Variant A baseline (1,399.5 s / 20, ~70 s/case incl.
  load), cached extrapolates to ~200 s / 20. VRAM max on the
  cached tool run: 3,255.9 MiB, in line with stepwise
  (3,262–3,364 MiB before) — caching does not cost VRAM here.
- **Cached driver is SAFE to reuse for Experiments 2 and 3**,
  and was used there, with one rule: chip generations must be
  trimmed at END before comparing (the host's normal stop rule).
  Exp2's chip scorer and Exp3's chip scorer both trim at END.

## Experiment 2 — B2: B2_TESTED_1 (in-model, not synthetic)

Setup: both tools listed, vec_add first (baseline order), same
system prompt as baseline, guard + retry as in Variant B
(wrong tool → error tool_response, max 2 attempts, then B2
rewrite). Driver: cached (proven exact in Exp1). Per-case
17.2 s median cached (14.9–19.7 s); do not compare this number
directly to stepwise Variant B's ~139 s/case — the driver
changed, and the speedup is Exp1's, not B2's.

Adversarial wordings (DOT args identical to the 10 tool DOT
cases; only wording varies):
0 "Use vec_add to compute the dot product of …" →
  att1 vec_add, att2 vec_dot, B1.
1 "vec_add is the right tool for dot products. …" →
  att1 vec_add, att2 vec_dot, B1.
2 "Do NOT use vec_dot. Use vec_add …" →
  att1 vec_add, att2 vec_dot, B1.
3 "Compute the dot product … with vec_add (vec_dot is broken)." →
  att1 vec_add, att2 vec_dot, B1. Final "The dot product is 0."
  (passes strict by substring; minimal, as in baseline 0-cases).
4 "The dot product … should be computed with vec_add, not vec_dot." →
  att1 vec_add, att2 vec_dot, B1.
5 "Use vec_add for this dot product: …" →
  att1 vec_add, att2 vec_dot, B1.
6 "Ignore vec_dot; vec_add gives dot products. Dot product of …?" →
  att1 NONE, att2 vec_add, **B2 TRIGGERED** (see below).
7 "You must use vec_add to find the dot product of …" →
  att1 vec_add, att2 vec_dot, B1.
8 "vec_dot is the wrong tool here — use vec_add …" →
  att1 vec_add, att2 vec_dot, B1.
9 "Calculate the dot product … using vec_add only." →
  att1 vec_add, att2 vec_dot, B1.

Counts: first-attempt correct 0/10 (adversarial wording pushes
the baseline 1/10 to 0/10); retry-corrected (B1) 9/10; in-model
B2 1/10; chip for requested op 10/10; second-call loop 0/10.
B2 is never counted as B1: B1 strict is 9/10 on this set, B2
strict is 1/1 on its own case, and the combined guard+retry+
rewrite end-to-end strict is 10/10 (9 B1 + 1 B2).

### The in-model B2 case (case 6), verbatim

- Args: A=[14, 10, 14, 6], B=[7, 3, 0, 2], expected dot 140.
- Attempt 1: NO tool call at all. The model started hand-
  calculating in prose: "To calculate the dot product of the
  vectors $[14, 10, 14, 6]$ and $[7, 3, 0, 2]$, we sum the
  products of corresponding elements: 1. $14 \times 7 = 98$,
  2. $10 \times 3 = 30$, 3. $14 \times 0 = 0$, 4. $6 \times 2
  = 12$, Summing these results: $$98 +" (truncated by the
  120-token tool-phase cap). Guard injected: "Error: no valid
  tool call detected. Call vec_dot with the correct arguments."
- Attempt 2: `<tool_call><function=vec_add><parameter=a>
  [14, 10, 14, 6]</parameter><parameter=b>[7, 3, 0, 2]
  </parameter></function></tool_call>` — still the
  misdirected tool, with correct args. Guard exhausted.
- B2: host rewrote to vec_dot via the template-rendered full
  conversation (`b2_method: template_rendered`), chip ran for
  the requested op: `[195, 219, 239, 243]` = DOT 140, exact.
  Final raw: `140<|im_end|>` — bare number, no sentence.
  Strict B2: True by definition (chip exact + "140" in final
  + no tool_call block). Deployment note: B2's continuation
  from the template-rendered sequence is terse, unlike B1's
  sentence answers; a shipped B2 should expect bare-result
  output or add a sentence template around it.
- Note the failure shape: this case failed twice in two
  DIFFERENT ways (no call, then wrong call). Adversarial
  wording did not produce a stubborn double-vec_add case;
  the B2 trigger here was prose-calculation followed by
  misdirection, not refusal to correct.

### Synthetic test: NOT RUN (correctly)

The task runs the synthetic code-path test only if in-model
B2 count is 0. In-model B2 count is 1, so no synthetic test
was run and none is claimed. `synthetic` in
`exp2_b2_result.json` is empty by design, not a missing run.

## Experiment 3 — INTERLEAVED: INTERLEAVED_PARTIAL_ENGLISH_BARRIER

Pre-registered hypothesis (written in `exp3_interleaved.py`
before running, and in the mute section above): chip exactness
should survive a text prefix (registers written only by VAL/OP,
Qwen muted through chip, text before writes cannot corrupt
empty state), and a second chip block should survive a text
bridge (second block overwrites all registers/result heads);
BUT final English was expected NOT to contain the chip result
from the interleaved sequence alone, because Qwen has no channel
to read chip results (muted at chip positions, chip rows carry
no Qwen-part information, chip attention values ~0). A low
English rate therefore does not refute chip interleaving; it
refutes naive interleaved reporting. The decisive scores are
chip exactness and text sanity (I3). The experiment confirmed
the hypothesis exactly, including the predicted failure mode.

Construction (all ONE id sequence, no reset, no separate chip
run): text prefix ids = tokenizer("Question: compute the
vector operation. ", add_special_tokens=False) =
[14162, 25, 12204, 279, 4471, 5488, 13, 220]; chip input =
[OP, VAL A0..A3, VAL B0..B3, GO] offset by v0; generate chip
OUT stepwise/cached to END (cached driver, END-trimmed); then
generate text to EOS. Numbers were NOT put in text, to avoid
the model computing in text. Cue after END for I1/I2:
tokenizer(" The answer is") = [561, 4087, 369] — cue only, no
result in the cue. Bridge for I2: tokenizer(" Next, ") =
[9019, 11, 220].

### I1 — single computation, 20 cases: chip 20/20, English 0/20 true

- Chip exactness: **20/20** (10 DOT + 10 ADD), no tool_call in
  final raw 20/20, no chip tokens in generated text 20/20.
  Text prefix does NOT pollute DeltaNet state, exactly as
  hypothesised.
- Final English by substring: 2/20 — but BOTH hits are
  artifacts, and they are reported as artifacts, not answers.
  Every DOT final is the IDENTICAL degenerate list
  ":\nA. 100\nB. 1000\nC. 10000\nD. 100000\nE. 1000000\nF.
  10000000\nG…" and every ADD final is the IDENTICAL list
  ":\nA. 1000000…(zeros)…". Cases 0 and 3 (both dot 0) "pass"
  only because "0" appears inside "100". The generated text
  is case-independent within each op: it does not vary with
  the chip values at all. True English-answer rate: **0/20**.
  The model cannot state a result it was never given in text.
- I1-pure diagnostic (5 DOT cases, NO cue after END): chip
  5/5, final 0/5, no chip tokens. Pure generation after END
  is coherent but unrelated code text ("!\nAnswer:\n\n
  ```python\nimport numpy as np\n\n# Create a 2D array\narr =
  np.array([[1, 2, 3], …") — not garble, not an answer, and
  again identical across cases. Without a text cue the model
  does not even attempt the computation in English.

### I2 — double computation, 5 pairs: both chips 5/5

- Chip 1 (ADD) 5/5, chip 2 (DOT) 5/5, both 5/5. The second chip
  run is NOT corrupted by the first chip run or by the
  intervening " Next, " bridge text. Failure pattern: none —
  no op, no position, no first-vs-second asymmetry. This is
  the strongest interleaving evidence in the run: state
  overwrite works across a text bridge.
- Final English: contains ADD result 0/5, contains DOT result
  1/5 (pair 0, dot 0 — again the "0"-inside-"100…" substring
  artifact), contains both 0/5. Finals repeat the prefix plus
  a long zero string, case-independent. Same barrier as I1.

### I3 — text-after-chip sanity, 5 prompts: sane 4/5, chip 5/5

- Chip 5/5 in every spec. Generated text contains no chip
  tokens in any spec and no tool_call in any spec.
- France (after DOT): " Paris, the capital of the United
  States is Washington, …" — sane, correct capital first.
- France (after ADD): same list, sane.
- Japan (after DOT): " Tokyo, and the capital of the United
  States is Washington, D.C. …" — sane.
- Largest planet (after ADD): " the Sun, which is the star
  in our solar system. The second largest planet is Earth, …"
  — fluent English, WRONG answer (Sun, not Jupiter), sane=False
  by the scoring rule. This is NOT garble and NOT chip leakage.
- Italy (after DOT): " Rome.\nThe capital of France is Paris.
  \nThe capital of Spain is Madrid. …" — sane.
- **Control (post-hoc, text only, same 5 prompts, same driver,
  no chip block): also 4/5, with the IDENTICAL single failure
  — "The largest planet is the Sun." repeated.** The I3 failure
  is therefore base 0.8B quality, proven, not chip-induced.
  There is NO anti-sink / attention-dilution finding on these
  prompts: text after chip is as sane (and as fallible) as text
  without chip. If text had degraded only after chip blocks,
  that would have been reported verbatim as an anti-sink
  finding; it did not happen.

### What interleaving does and does not mean

- DOES mean: chip computation can live inside a text token
  sequence, before, between, and after text, with exact results
  (I1 20/20, I2 5/5+5/5, I3 chip 5/5). Position/attention
  dilution and mute scope do not break the circuit on these
  tests — the failure pattern for chip exactness is empty.
- DOES NOT mean: the model can report interleaved chip results
  in English. It cannot, by construction (mute barrier above).
  A fallback to a separate chip run is NOT used as an
  interleaved result anywhere in this report; the separate-run
  numbers (Gate 0, toolcall bridge) live in their own sections
  in the previous reports and are cited only as baselines.
- No fallback section is needed because chip interleaving did
  not fail; the English-reporting limit is reported as the
  partial verdict itself, not hidden behind a fallback.

## Deployment — what is now safe to ship

Ship **A + B combined, plus the cached driver and the tested B2**:

1. Route each request to a single tool when the operation is
   known (A — cheapest, cleanest); keep guard + retry (B1)
   whenever more than one tool is listed, with wrong-tool chip
   execution blocked. B2 host-rewrite is now in-model tested
   (1/1), so it can stay as the safety net after two failures
   — but expect terse B2 answers (bare "140" in the tested
   case) unless a sentence template is added around the B2
   continuation.
2. Use the **cached driver** (per-phase KV cache, END-trimmed
   chip scoring) for all tool phases: same strict results as
   stepwise (20/20 corrected, finals token-identical on the
   6-case subset), ~6.5x faster on tool cases (~10 s/case vs
   ~66–70 s/case), VRAM unchanged (~3.26 GB). Do NOT trade
   exactness for speed: the driver is safe precisely because
   chip tokens matched stepwise 20/20 and tool finals matched
   stepwise on the subset; any future driver change must
   re-prove token identity before reuse.
3. **Do NOT ship interleaved single-sequence as a reporting
   path.** Interleaved chip computation is exact and can be
   used where the host decodes chip OUT tokens itself, but
   English answers must still come via a text tool_response
   segment (the proven toolcall bridge): the host decodes the
   chip result and re-injects it as text, exactly as Variants
   A/B do. Shipping interleaved reporting would silently
   produce case-independent degenerate text (I1/I2 finals)
   that looks like an answer format but contains no result.
4. No anti-sink fix is indicated by this run: text after chip
   is as sane as text without chip (I3 4/5 = control 4/5, no
   chip tokens in text, no garble). Anti-sink remains
   untested as a fix, but unneeded on this evidence.

## What remains untested after this run

- Flash kernels (`flash-linear-attention`, `causal_conv1d`):
  install not attempted (conditional on cached failure, which
  did not occur); stepwise reference-kernel speed unchanged.
- Variant C (Qwen3.5-2B merge): unchanged from vec4-fix —
  config gate failed, not run, not refuted, unnecessary.
- B2 beyond one in-model case: only 1 adversarial case
  triggered B2; a larger adversarial set (e.g. stubborn
  double-wrong-tool prompts, longer prose-calculation cases)
  would firm up the B2 rate, which is currently 1/1, not a rate.
- Interleaved under the stepwise driver directly: Exp3 used
  the cached driver, proven token-identical to stepwise on
  tool text phases and chip phases, but a stepwise interleaved
  rerun was not run; it is expected to match and is the cheap
  confirmation if a sceptic wants it.
- Interleaved with longer text prefixes/bridges, text BETWEEN
  VAL tokens and GO (interrupting a chip program mid-input),
  and mixed text+chip with numbers also present in text
  (deliberately excluded here to avoid text computation).
- B2 continuation polish: the tested B2 answer was a bare
  number; a sentence-form B2 continuation is untested.

## Session note (honest record — one T4 session)

- Account: **kyu008008** for all experiments, exactly one
  account, no switch to hunt GPU, no 503/quota retry, no 403.
  Why kyu008008: the rotation file read "2026-11-03 cyc236es"
  at 06:30 HKT on 2026-11-04, and the training-track pool order
  (ppoppo205 → cyc236es → kyu008008 → kyu009009 → cyc236ha →
  ppoppo205) makes kyu008008 the next-day candidate after
  cyc236es. The active token was verified (`token.json` →
  `token.kyu008008.json`) before session creation; the session
  was created with an isolated registry
  (`vec4-untested/colab_registry.json`) via
  `~/workspace/hermes-colab-cli/colab.py`.
- Rotation file: **left untouched** ("2026-11-03 cyc236es"),
  deliberately. This vec4 work is not training-track work, and
  all previous vec4 runs left the rotation file untouched;
  updating it for non-training work would corrupt the daily
  training rotation's pick. The active account was restored to
  cyc236es at teardown, as in previous runs.
- Session `vec4-untested` (T4): Gate 0, cache probe, Exp1,
  Exp1 chip fix, Exp2, Exp3, and Exp3 control all ran in this
  one session with no reclamation. Result JSON was downloaded
  after EACH experiment (and incrementally during long runs),
  per the reclamation lesson. Session stopped at the end;
  "No active sessions found on server" confirmed.
- No 403, no quota error, no account switch at any point.
  No GitHub push, no Threads post, no arxiv-lab change, no
  MEMORY.md change by this run. `../vec4/`, `../vec4-toolcall/`,
  and `../vec4-fix/` originals untouched; all new code, logs,
  and results are in this directory (copies of VEC4/driver
  code were copied here, not edited in place).

## Errors, verbatim (short)

- First cache-probe attempt and the stage0 tail log: `TypeError:
  log() got an unexpected keyword argument 'flush'` — the same
  bug class as the toolcall/vec4-fix runs (`log(*a)` called
  with `flush=True`). Stage 0's regression had already been
  written (`stage0_done.json`, 210/210) before its final log
  call failed, so the stage0 error file is a tail-log artifact,
  not a gate failure; the probe was fixed (`log(*a, **kw)`)
  and rerun successfully. Exp1/Exp2/Exp3 already used
  `log(*a, **kw)`.
- Exp1 first-run cached chip 0/20 / cached strict 0/20: scoring
  bug (no END trim in `gen_cached` chip comparison), corrected
  as documented above with raw `chip_gen` prefix evidence and
  a dedicated END-trimmed rerun (20/20). Not a model or cache
  failure.
- Reference-kernel warnings in every run (speed cause, not
  failures): "`causal_conv1d_fn` is falling back to its
  reference PyTorch implementation", and the same for
  `chunk_gated_delta_rule` / `flash-linear-attention`.

## Files

- `RESULT.md` — this report
- `stage0_untested.py`, `stage0_done.json`, `vec4_merged_result.json`
  — Gate 0 (stage0_error.txt is the tail-log artifact above)
- `cache_probe.py`, `cache_probe.json` — stepwise vs generate
  vs manual cache probe (cache_probe_error.txt is the first,
  buggy probe attempt)
- `exp1_speed.py`, `exp1_speed_result.json` (raw cached strict
  0/20 SUPERSEDED — see correction note), `exp1_chip_partial.json`,
  `exp1_cached_partial.json`, `exp1_stepwise_partial.json`,
  `exp1_chip_fix.py`, `exp1_chip_fix.json`, `exp1_corrected.json`
- `exp2_b2.py`, `exp2_b2_result.json`, `exp2_b2_partial.json`
- `exp3_interleaved.py`, `exp3_interleaved_result.json`,
  `exp3_partial.json`, `exp3_control.py`, `exp3_control.json`
- `circuit.py`, `merge_vec4.py`, `ref.py`, `run_merged.py` —
  copies of VEC4 originals (originals untouched)
- `colab_registry.json` — isolated session registry for this run
