# VEC4 on Qwen3.5-2B (Path C, retargeted) — RESULT

## Verdict: C2B_GREEN

Path C is complete, by retargeting, not by the old exact-Target gate.
The 2B port passes every gate the 0.8B port passed, and the key
question answers YES: **the bigger model fixes DOT selection on its
own** — C-baseline (both tools listed, vec_add first, no guard, no
routing) is **20/20 strict**, against 0.8B's 11/20 (DOT 1/10) in the
same configuration on the same 20 cases. A and B still pass on 2B,
but on this case set they add nothing: A is also 20/20, and B's
guard never fires (10/10 first-attempt, B2 0 cases exercised).

- **Gate 0 chip-only: 210/210** on the merged 2B model.
- **Cached-driver re-proof: 20/20** stepwise = generate =
  cached-trimmed, tokens identical stepwise vs cached 20/20.
- **C-baseline: 20/20 strict** (DOT 10/10, ADD 10/10).
- **C-A (single-tool): 20/20 strict** (DOT 10/10, ADD 10/10).
- **C-B (guard+retry, DOT only): B1 10/10, B2 0 exercised.**
- **Plain text: 5/5 no tool call. Should-not-call: 5/5 no tool call**
  (false-trigger rate 0/5).

## Scoring (strict, all configurations — same as vec4-fix)

Strict = requested op == called tool AND args == case args AND chip
run for the requested op matches `ref.py` AND final English contains
the correct requested result AND final raw contains no `<tool_call>`
block. B2 is NEVER counted as B1. "B2 0 exercised" means the guard
never reached the rewrite path (all first attempts correct), NOT
0 failures. 0.8B numbers below are cited from
`../vec4-fix/RESULT.md` and `../vec4-untested/RESULT.md`, not
re-run here; only the 2B columns are new measurements.

## Strict comparison table

| Configuration | DOT correct tool | DOT chip (requested op) | DOT strict | Overall strict | Median s/case | VRAM max |
|---|---|---|---|---|---|---|
| 0.8B baseline, both tools, vec_add first (vec4-toolcall) | 1/10 | 1/1 (only 1 DOT chip ran) | 1/10 | 11/20 (ADD 10/10) | ~77 s (stepwise, 30 cases) | 3,337.0 MiB |
| 0.8B A, single tool (vec4-fix) | 10/10 | 10/10 | 10/10 | 20/20 | ~70 s (stepwise) | 3,262.0 MiB |
| 0.8B B1, guard+retry, model-only (vec4-fix) | 1/10 first, 10/10 after ≤2 | 10/10 | 10/10 | — (DOT only) | ~139 s (stepwise) | 3,363.6 MiB |
| **2B C-baseline, both tools, vec_add first** | **10/10** | **10/10** | **10/10** | **20/20 (ADD 10/10)** | **11.6 s (cached)** | **7,473.7 MiB** |
| **2B C-A, single tool** | **10/10** | **10/10** | **10/10** | **20/20** | **11.6 s (cached)** | **7,408.0 MiB** |
| **2B C-B1, guard+retry, model-only** | **10/10 first attempt** | **10/10** | **10/10** | — (DOT only) | **12.5 s (cached)** | **7,465.1 MiB** |
| 2B C-B2, host-rewritten fallback | — | — | **0 exercised** | — | — | — |

Cases are the exact 20 from vec4-fix Variant A (DOT edges all-0,
all-15, [3,5,7,9]·[2,4,6,8]=140, [0,15,0,15]·[15,0,15,0]=0, + 6
random; ADD edges + 6 random; `Random(20261103)`), verified 20/20
against `../vec4-fix/variant_a_result.json` before every run on
the VM (cache probe, baseline, A, B each re-verified).

## The example case in each configuration

[3,5,7,9]·[2,4,6,8] = 140, verbatim finals:

- C-baseline case 2: "The dot product of [3, 5, 7, 9] and [2, 4, 6,
  8] is 140." (vec_dot first attempt, chip exact)
- C-A case 2: same sentence, vec_dot first attempt, chip exact.
- C-B case 2: same sentence, vec_dot first attempt, chip exact,
  guard never triggered.

On 0.8B this case failed in both baseline configurations and
passed in A and B1. On 2B it passes everywhere, including the
unguarded baseline.

## Step 1 — retarget (full analysis in RETARGET.md)

- Linear heads 16/16, dim 128: UNCHANGED — DeltaNet
  register/OP/result heads port as-is.
- Intermediate cap 3584 → 6144: cap change only. Circuit max is
  98 FFN units per sublayer (verified locally from circuit.py:
  level 2: 98; full map in RETARGET.md), far below both caps, so
  FFN placement is identical. Only change in circuit.py:
  `Target(16, 16, 3584)` → `Target(16, 16, 6144)`.
- Base hidden 1024 → 2048. `pick_hidden(2048, 143)` = (2304, 36),
  verified locally and robust for n_slots 135..150. Merged hidden
  2304, chip width 256 (UNCHANGED: 2304 − 2048 = 256, same as
  1280 − 1024), ballast m = 36, chip region used 179, mu slot
  predicted 2114 = 2048 + 36 + 1 + 29 — and the merge log confirms
  all of it exactly: n_slots 142, slots mu=29 tau=30 code=31..38,
  mu_dim=2114, tau_dim=2115, chip_width=256.
- Vocab 248320 unchanged → chip offset v0=248320 unchanged,
  merged vocab 248576 unchanged, `full=True` token features kept
  (the original 0/210 bug class: v0 > 65536 still needs it).
- Full-attention head_dim is 256 on 2B (8 heads × 256 = 2048),
  NOT 128 — the config-check file's `num_attention_heads: 8` for
  both models was ambiguous, and the merge log settles it. This
  does not affect the circuit (linear heads only) and `mute()`
  reads `head_dim` generically, so no code change was needed.
- `merge_2b.py` deviations from `merge_vec4.py`, both documented
  in its header: (1) Target 6144; (2) in-place widening, identical
  math to `M.widen` (same regexes, same scale, same norm formula),
  freeing each old tensor on assignment — stock widening would
  transiently hold base + widened ≈ 11 GB on a 13 GB VM. Gate 0
  210/210 + text sanity are the verification that the widening
  is correct.

## Step 2 — merge + Gate 0: 210/210

- Merged: hidden 2304, 42 layers (18 chip + 24 Qwen), vocab
  248576, **3,640,992,896 params, 7.28 GB bf16** (text+embed
  3,347,251,200 = the predicted number exactly; vision tower
  293,741,696 params ≈ 0.59 GB — larger than 0.8B's ≈80M).
- Load 34.0 s (printed; the JSON `load_seconds` field is total
  time, 214.6 s, as in the 0.8B report), VRAM at load 7,106.7 MiB,
  peak during regression 7,162.3 MiB.
- Regression 210/210 in 178.0 s generation, 0 fails. Stage 0 total
  503.7 s including 2B download + merge.
- Text sanity, verbatim: "The capital of France is" ->
  " Paris.\nA. True\nB. False". Correct capital first (" Paris"),
  then a True/False list continuation — looser than 0.8B's
  " Paris.\nThe capital of France is Paris.", but fluent English,
  not garble, and not a chip symptom (chip path is separate and
  210/210; the tool finals below are clean sentences). Recorded
  verbatim rather than smoothed over.
- No debug round was needed: no mu-slot, widening, FFN-cap, or
  TOKEN_MASK failure occurred. The first merge passed Gate 0.

## Step 3 — cached-driver re-proof: 20/20 token-identical

On the 20 tool cases, per case: stepwise vs stock `generate()` vs
manual per-phase KV cache (END-trimmed):

- Stepwise 20/20, generate 20/20, cached-trimmed 20/20,
  cached tokens identical to stepwise 20/20.
- Medians: stepwise 0.90 s, generate 0.53 s, cached 0.52 s (chip
  phase only; the tool-case win is in long prompts, as before).
- VRAM max on the probe: 7,206.6 MiB.
- The cached driver was therefore used for all tool tests below
  (per-phase cache; every host injection starts a new phase with
  a fresh full forward, as in vec4-untested). No stepwise fallback
  was needed.

## Step 4 — tool tests on 2B (cached driver, strict)

- **C-baseline (both tools, vec_add first, single attempt, no
  guard): 20/20.** Correct tool + args 20/20, chip requested
  20/20, parse failures 0/20, emission 20/20, second-call loop
  0/20. DOT finals are full sentences ("The dot product of …
  is 140."); ADD finals are bare lists ("[0, 0, 0, 0]") — both
  pass strict by the stated rule (final contains the correct
  result, no tool_call block). Total 259.9 s for 20 cases
  including load.
- **C-A (single tool listed): 20/20.** Correct tool + args
  20/20, chip 20/20, parse failures 0/20, no case emitted the
  unlisted tool. Total 257.3 s. As on 0.8B, removing the wrong
  choice removes the failure mode — but on 2B there was no
  failure mode left to remove in the baseline.
- **C-B (guard+retry, DOT only): B1 10/10, B2 0 exercised.**
  First-attempt correct tool 10/10 (vs 0.8B B's 1/10); retry
  0 cases; host-rewrite 0 cases. The guard code path after
  attempt 1 is therefore untested ON 2B by this run — it is
  covered on 0.8B (B1 10/10 with 9 retry corrections) and B2
  in-model 1/1 on 0.8B adversarial cases (vec4-untested), but
  those are 0.8B numbers and are not presented as 2B numbers.
  Total 152.2 s for 10 cases including load.
- **Plain text (both tools listed): 5/5 no tool call.**
  "The capital of France is Paris.", a real cat joke ("Why
  don't cats like math? Because they're bad at it!" — 0.8B
  refused this prompt), a correct vector definition, "The
  capital of Japan is **Tokyo**.", "Hello!".
- **Should-not-call (both tools listed): 5/5 no tool call,
  false-trigger rate 0/5.** "2 + 2 = 4.", a correct dot-product
  definition with no calculation, a colour preference answer, a
  correct Hamlet summary (0.8B's had a garbled "just been
  executed" detail), "Yes".
- 2B text quality is visibly better than 0.8B on the plain /
  should-not-call set (joke told, Hamlet correct). That is a
  base-model quality observation on 10 prompts, not a measured
  text benchmark.

## Step 5 — comparison: did the bigger model fix selection, and is it worth shipping?

**Did it fix selection on its own? Yes, on this case set.**
0.8B baseline DOT was 1/10 with 9/10 vec_add-with-correct-args
failures; 2B baseline DOT is 10/10 with no wrong-tool call at
all, in the same configuration (both tools, vec_add first, same
system prompt, same 20 cases). The failure mode A and B were
built to work around — vec_add stickiness under two listed
tools — does not occur on 2B here. A and B still do their jobs
on 2B (A 20/20, B1 10/10), but on this evidence they are
redundant at 2B: B's guard is a safety net that never fires,
and A's routing constraint buys nothing the baseline does not
already deliver.

**What the retarget cost (2B vs 0.8B):**

| | 0.8B | 2B |
|---|---|---|
| Merged params / bf16 size | 1,526,970,752 / 3.05 GB | 3,640,992,896 / 7.28 GB (2.38×) |
| VRAM at load / peak (Gate 0) | 2,955 / 3,010 MiB | 7,107 / 7,162 MiB (2.38×) |
| VRAM max, tool runs (cached) | ~3,256 MiB | ~7,408–7,474 MiB |
| Load seconds (printed) | 8.5 s | 34.0 s (4.0×) |
| Gate 0 generation (210 cases) | 124.8 s | 178.0 s (1.43×) |
| Tool case, cached median | 10.1 s (0.8B cached) | 11.6–12.5 s (1.15–1.24×) |
| Stage 0 total (download + merge + Gate 0) | ~208 s | 503.7 s |

The 2B model still fits comfortably on a T4 (7.5 GB peak of
14,913 MiB), and cached tool latency barely grows — the cost
is size/VRAM/load, not per-case speed.

**Is 2B worth shipping for this problem? No, on this evidence —
with one qualification.** A+B already fix DOT at 0.8B at less
than half the size and VRAM, and 2B's baseline win is measured
on the same 20 cases with the same wording, not on a broader
or adversarial set. For prompts where the operation is explicit
or a guard can be written, ship 0.8B + A + B (the vec4-fix /
vec4-untested recommendation, unchanged). The qualification:
2B's value would be for requests where the operation is NOT
known in advance and no guard can name the right tool — there
the 2B baseline's unforced selection is the only fix of the
three that applies without host knowledge. But that
generalisation is NOT proven here: it would need a broader
multi-tool, multi-operation, adversarially-worded set before
any claim that "2B fixes selection in general". Until such a
set is run, the honest statement is: 2B fixes selection on
these 20 cases; 0.8B + A + B fixes it on these 20 cases too,
cheaper.

## Session note (honest record — one T4 session)

- Account: **kyu008008** for all GPU work, exactly one account,
  no switch to hunt GPU, no 503/quota retry, no 403. Why
  kyu008008: the rotation file read "2026-11-03 cyc236es" at
  task start, and the training-track pool order (ppoppo205 →
  cyc236es → kyu008008 → kyu009009 → cyc236ha → ppoppo205)
  makes kyu008008 the next-day candidate after cyc236es; the
  then-active ppoppo205 was occupied by the separate
  weight-space session. (That session has since died — see the
  account note below — but no account change was made
  mid-run because of it.)
- Rotation file: **left untouched** ("2026-11-03 cyc236es").
  This vec4 work is not training-track work.
- Start active (recorded at run start): token.ppoppo205.json.
  Teardown active (verified after stop): token.ppoppo205.json
  — restored to EXACTLY the start value, not a hardcoded
  account. Session `vec4-2b` stopped; "No active sessions
  found on server" confirmed.
- Account conflict flag for the parent: the weight-space
  session's bookkeeping recorded prior_active=kyu008008 and a
  plan to restore kyu008008 when it ends, while this run's
  start value was ppoppo205 (set by weight-space at 07:40) and
  this run restored ppoppo205. Separately, an 08:50 check
  recorded ppoppo205's server session list as EMPTY (the
  weight-space VM session gone, most likely reclaimed). The
  parent should reconcile the final active-account state with
  the weight-space run's owner rather than assuming either
  value; this run's own bookkeeping is as stated above.
- One session for everything: Stage 0, cache probe, baseline,
  A, B, plain/notool — no reclamation. Result JSON downloaded
  after EACH configuration (and per-case partials written
  during long runs), per the reclamation lesson.
- No GitHub push, no Threads post, no arxiv-lab change, no
  MEMORY.md change by this run. `../vec4/`,
  `../vec4-toolcall/`, `../vec4-fix/`, `../vec4-untested/`,
  `../vec4-app/` originals untouched; all new code, logs, and
  results are in this directory (VEC4/driver code copied here,
  not edited in place).

## Errors, verbatim (short)

- Cache-probe first launch: `FileNotFoundError: .../vec4-fix/
  variant_a_result.json` on the VM — my own omission (the
  verification file was never uploaded). Fixed by uploading
  that file to the VM; no case data was affected, and the probe
  then verified 20/20 before running. The same upload fixed
  tool_2b.py's identical verification read.
- Reference-kernel warnings in every run (speed cause, not
  failures): "`causal_conv1d_fn` is falling back to its
  reference PyTorch implementation", and the same for
  `chunk_gated_delta_rule` / `flash-linear-attention`.
- No 403, no quota error, no account switch for GPU reasons at
  any point.

## Files

- `RESULT.md` — this report
- `RETARGET.md` — Step 1 retarget analysis (written before merging)
- `stage0_2b.py`, `stage0_done.json`, `vec4_merged_result.json`
  — Gate 0 (210/210)
- `cache_probe_2b.py`, `cache_probe_2b.json`,
  `cache_probe_2b_partial.json` — Step 3 re-proof (20/20)
- `tool_2b.py` (cached driver, `--only baseline|A|B|plain`,
  `--driver cached|stepwise`), `c_baseline_result.json`,
  `c_a_result.json`, `c_b_result.json`, `c_plain_result.json`,
  `c_notool_result.json` (+ `c_*_partial.json` per-case writes)
- `circuit.py` (Target 6144), `merge_2b.py` (in-place widen),
  `ref.py`, `run_merged.py` — 2B copies (originals untouched)
- `colab_registry.json` — isolated session registry for this run
- `account_note_start.txt` — start active account record
