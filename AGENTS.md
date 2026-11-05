# AGENTS.md — how to rerun and extend this repo without lying to yourself

These rules are the distilled lessons of the original runs. They are
not style preferences: each one exists because a run lost data, got a
wrong number, or wasted a session by violating it.

## The staged-gate rule

Never run a later stage on a failed earlier stage.

1. Simulator (`ref.py`) vs cases — 210/210, seed 20261103.
2. IR circuit (`circuit.py`, `test_circuit.py`) vs simulator —
   210/210, 0 mismatches, `ir.evaluate(..., check=True)` raising for
   no case.
3. Tiny standalone model — forward argmax 210/210, stock `generate()`
   on a sample.
4. Merged real model — Gate 0 chip-only regression 210/210.
5. Only then tool tests, app tests, or retargets.

If Gate 0 is not 210/210, debug once (see "Known bug classes"), report
the failure pattern, and do not run tool tests or claim a port.

## Strict scoring (do not soften this)

Strict = all of:

- requested op == called tool,
- called args == case args,
- chip run for the *requested* op matches `ref.py`,
- final English contains the correct requested result,
- final raw contains no `<tool_call>` block.

Report per configuration: correct-tool rate, chip exactness for the
requested op, strict final rate, emission/parse failures, median
s/case, VRAM max, and the example case
`[3,5,7,9]·[2,4,6,8] = 140` in every configuration.

Counting rules that have burned people before:

- **B2 is never B1.** A host-rewritten fallback is a system success,
  not a model success. Count B1 (model-corrected on retry) and B2
  (host-rewritten) in separate columns, always.
- **B2 "0 exercised" is not 0 failures.** It means the guard never
  reached the rewrite path. Say so in words, every time.
- **Chip exactness is not application success.** 90/90 chip exactness
  with 0/10 clean workflows is a failure of orchestration, reported
  as such. Never quote a chip number as an application number.
- **A driver's loose summary is not the result.** If a driver scores
  by the tool actually called, re-score by the requested op and quote
  the strict number. The 0.8B tool-call run's loose 20/20 was
  superseded by strict 11/20 for exactly this reason.
- **Trivial cases are excluded from non-trivial rates.** In the app
  dataset, the all-zero scenario ties on every metric; quote rates
  excluding it alongside the all-scenario rates.
- **Substring scores are artefacts until proven otherwise.** The
  interleaved run's 2/20 substring score was the digit "0" inside a
  case-independent degenerate list, not 2 correct answers. True
  English reporting was 0/20. Check what a substring match actually
  matched before quoting it.

## Colab discipline

- One free T4 session per experiment, one account, no hunting for a
  bigger GPU, no retry loop on quota (a 503 / gpu-unavailable result
  is a stop, not a retry signal). Stop at the first 403 or lock
  signal on any vendor auth path.
- Use your own Colab tooling to create and drive the session. The
  original runs used a private wrapper that is not in this repo;
  nothing in these scripts depends on it — they are plain Python
  files run on the VM. Do not invent a wrapper the repo does not
  contain.
- **Download results off the VM before teardown, after every
  configuration, and per-case during long runs.** Free sessions get
  reclaimed mid-run. End-only JSONs have been lost to reclamation
  more than once in this project's history; the per-case partial
  writes in the drivers exist for that reason. A result that only
  ever lived on a reclaimed VM does not exist.
- Verify a launch by its log file and process, not by the launch
  command's return alone. A hung launch that never ran looks
  identical to a running one from the outside.
- Record the session, account, GPU, probe (Python/torch/
  transformers versions, VRAM, RAM, disk) and the merged spec
  (hidden, layers, vocab, params, bf16 GB, VRAM at load and peak)
  in every RESULT file. A number without its session is not
  checkable.
- Merged checkpoints are disposable: they live on the VM and are
  discarded at teardown. Rebuild from the merge script. Do not
  commit checkpoints, base weights, or anything over a few MB
  (this repo's largest file is under 100 KB for that reason).
- Merge each model in its own process on a 13 GB VM (see "Known
  bug classes"). Run long stages detached with a log file and
  verify liveness from the log, not from the launch call.
- **A bigger circuit can OOM the stock merge even when the
  per-model merge fits.** The 2B-v3 stock merge was killed at
  11.7 GB RSS; the lowmem driver (`merge_v3_lowmem.py`) peaks
  at 8.65 GB by sharding Qwen tensors out as they are widened
  and emitting chip layers one at a time through
  `emit_layers_sink` (a mechanical copy of upstream
  `emit_layers` with a sink callback — 98/98 body lines
  identical). The widen phase peak (~6 GB) is the second peak
  and fits. Any future circuit that grows slots or chip width
  must use the lowmem driver, or re-prove the stock merge
  peak first. See "VEC4 v3" below.

## Known bug classes (check these first on any failure)

- **`TOKEN_MASK` (`t & 0xFFFF`) in `ir.Graph.feature_value`.**
  Chip tokens live at v0 = 248320, above 65536, so masked token
  features misclassify them as text. The first merged run was 0/210
  for exactly this reason. Fix: all VEC4 token features with
  `full=True`. If a fresh port scores 0/210, check this before
  anything else.
- **`log()` colliding on `flush=True`.** A `log(*a)` wrapper that
  does not accept `**kw` raises `TypeError` on `log(..., flush=True)`,
  sometimes only in a tail-log after the real result is already
  recorded. Fix the wrapper, do not re-diagnose the model.
- **`get_device_properties(0).total` does not exist.** The attribute
  is `total_memory`. A probe that uses `.total` dies before model
  load and tells you nothing about the model.
- **Stock widening on 2B.** Widening a 2B base transiently holds
  base + widened ≈ 11 GB on a ~13 GB VM. The 2B merge uses in-place
  widening (identical math, per-key freeing). If a new port OOMs
  at merge time on a larger base, this is the first suspect, not
  the circuit.
- **END-trim scoring on cached generation.** Cached generation
  continues past chip END into text tokens. Score cached output
  END-trimmed, and re-prove stepwise = generate = cached-trimmed
  token identity (≥20 cases) before using the cached driver for
  any tool test. A first-pass cached 0/20 in this project was an
  END-trim scoring bug, not a driver failure.
- **OP ids colliding with VAL tokens.** VEC4 v2's new ops were
  first placed at ids 2–5, which collide with VAL tokens (2–129):
  ADD with A0=0 read back as OP_SUB. The circuit test caught it
  before any model ran. New OP ids must live outside every other
  token range; the v2 ids are 244–247.
- **Stale hardcoded paths in a ported driver.** The 2B-v2 tool
  driver was ported from the 0.8B-v2 one with one path still
  pointing at the 0.8B directory. After any port, grep every
  path in every driver, not just the one you edited.
- **Two merges in one process OOM on a 13 GB VM.** The eff
  stage0 merged 0.8B successfully, then died merging 2B in the
  same process (11.65 GB RSS). Merge each model in its own
  process; the same applies to any future multi-model stage.
- **Stock merge OOM on a larger circuit (2B-v3).** The v3
  circuit is 364 slots / chip width 512 (vs 186 / 256 in v2),
  so the stock merge's peak is ~17 GB: widened base ~5.7 GB +
  fp32 chip dict ~7.6 GB + bf16 conversion copies ~3.8 GB.
  The merge was killed at 11.70 GB RSS with `chipformer.json`
  never created. This is a merge-host memory limit, not a
  circuit error — Gate 0 on the lowmem-merged model is
  720/720. Fix: `merge_v3_lowmem.py` (shard-out + per-layer
  emit, peak 8.65 GB). Do not "fix" the circuit for a merge
  OOM, and do not quote the killed run as a circuit result:
  no 2B-v3 numbers exist from it.
- **Long stages need detached execution + a log file.** A plain
  exec-style call that returns after ~1 minute cuts a long merge
  or tool run short mid-stage. Run long stages detached, verify
  from the log that the process is alive, and download after
  each stage's WROTE line — never trust the launch call's
  return alone.

## Efficiency comparisons (rules from vec4-eff)

- Gate before timing: a chip timing number is meaningless if
  the chip is wrong. Require 60/60 fresh cases matching `ref.py`
  before reporting any chip latency; a model failing the gate
  gets no timing comparison at all.
- Baselines on the same VM, in the same run: Python loop, NumPy
  vectorised, torch CPU, torch CUDA. Model-independent baselines
  should roughly agree across model runs — if they do not, the
  VM conditions changed and the ratios are suspect.
- The chip cannot batch sequential generation, so throughput is
  the inverse of per-case latency. Do not imply a batching
  escape hatch exists.
- Power snapshots are not energy. An nvidia-smi power draw
  reading (idle vs mid-loop) is an instantaneous snapshot, not
  a total; never turn it into an energy claim, and never report
  a chip-vs-host energy ratio from snapshots.
- Fixed costs belong in the result alongside latency: load
  seconds, checkpoint GB, VRAM at load and peak. For in-model
  exactness, the fixed costs are usually the more decision-
  relevant numbers.

## What not to redo, and what not to claim

- Paths A and B are complete on 0.8B (A 20/20, B1 DOT 10/10, B2
  in-model 1/1 adversarial). A rerun that needs them should cite
  them, not redo them as its main work — unless it changes the case
  set, in which case it says so and re-proves from Gate 0.
- Path C is complete on 2B (C2B_GREEN). Its open question is not
  "does 2B pass these 20 cases" but "does 2B's selection generalise
  beyond these 20 same-wording cases" (see README "Where to take it
  next" item 1).
- No KL vs base was computed anywhere in this project. Do not claim
  text is "preserved" beyond the recorded text-sanity outputs.
- V2 gates are complete on both models (0.8B-v2 and 2B-v2 Gate 0
  396/396, cache 24/24, disp 39+1 refusal on both). A rerun that
  needs them should cite them, not redo them — unless it changes
  the case set, in which case it says so and re-proves from
  Gate 0. The v2 generalisation set is in `vec4-v2/`; reuse it
  rather than inventing a new one silently.
- The efficiency comparison is complete (`vec4-eff/`). A rerun
  should cite its ratios and fixed costs, not re-derive them
  from a different VM without saying the VM changed.
- **VEC4 v3 is complete on both models** (`vec4-v3/`): Gate 0
  720/720 and cache 40/40 on 0.8B and on 2B (lowmem merge).
  The 2B-v3 tool numbers (orig 26/28, para 11/16, adv 8/10)
  and the 0.8B-v3 numbers (orig 26/28, para 9/16, adv 5/10)
  stand as measured; the VSORT orig failures on both models
  are a prompt/schema artifact (orig prompts name only one
  vector, the schema requires `a` and `b`), not a chip or
  selection failure. The disp re-proof (2B, chip) is 44
  executed + 1 expected refusal, 0 wrong — dispatcher+chip
  clean on all 45 disp rows. A rerun that needs any of this
  should cite it, not redo it — unless it changes the case
  set, in which case it says so and re-proves from Gate 0.
- No flash kernels, no stepwise interleaved rerun, no mid-input
  text interruption, no sentence-form B2. VSHIFT, matrix
  multiplication and sorting are no longer open — they shipped
  as VSHL/VSHR, MMUL and VSORT in v3. The base-vs-merged
  language probe was measured on 2026-11-05 (0.8B verbatim
  7/10, 2B 8/10, no chip-token leak; sanity evidence only, not
  a KL). The probe JSONs and record are in `vec4-lang/`.
  These are open, not failed. A new run that covers one should
  say which one, with the same strict scoring.
- Do not push results to a repo, post them publicly, or touch
  another project from a run in this repo unless the operator
  explicitly says so for that run. This repo's own push history is
  the exception the operator ordered, not a standing permission.

## Extending the circuit

- Recompute, don't assume: `pick_hidden`, ballast, mu slot, chip
  width, FFN cap. `vec4-2b/RETARGET.md` is the worked example —
  write the retarget analysis *before* merging, verify every
  predicted number against the merge log, and record mismatches
  verbatim rather than smoothing them over.
- The circuit's max is 98 FFN units per sublayer. A cap change
  (3584 → 6144 on 2B) is not a redesign as long as placement is
  unchanged; prove that from the code, not from the cap numbers
  alone.
- Keep the reference simulator independent of the circuit (plain
  Python arithmetic vs gates/carry chains/adder trees). Agreement
  between two independent implementations is the check; sharing
  code between them destroys it.

## VEC4 v3 — dispatcher fixes and lowmem merge (2026-11-05)

V3 (`vec4-v3/`) adds VSHL/VSHR, MMUL and VSORT (OP ids 248–251,
CHIP_VOCAB 248→252, merged vocab unchanged at 248576), Gate 0 is
720 cases (72/op), and the dispatcher contract gained three
fixes. The contract itself is unchanged otherwise: semantic task
beats tool-name mentions, CONFLICT executes the semantic op,
genuinely ambiguous prompts are refused, never guessed.

- **MMUL pattern order.** `product.*matrices` is in the MMUL
  SEMANTIC pattern (first in order). The generic `\bproduct\b`
  pattern resolves to lane-wise MUL, so any matrix wording must
  match before it. A bare "product" with no "matrices" still
  resolves to MUL — guard that in `test_dispatcher.py`.
- **VSORT single-vector path.** `parse_vectors` still requires
  two vectors for every other op. In `dispatch`, on NO_VECTORS,
  if the intended op resolves to VSORT (semantic OR single
  mention), fall back to the single list with dummy B
  `[0,0,0,0]` — VSORT never reads B. An out-of-range single
  list still propagates OUT_OF_RANGE, and a bare list with no
  op signal is still NO_VECTORS. The vector scan is shared
  (`_scan_vectors`); `parse_vectors` behaviour is preserved
  exactly.
- **Stacked fixes are real.** The VSORT prompt needed two
  fixes at once: even with two vectors, "in ascending order,
  smallest first" resolves to MIN via `\bsmallest\b`, so
  `in ascending order|ascending order` is in the VSORT pattern
  before MIN in SEMANTIC order. The single-vector fix alone
  would have executed the wrong op. When a prompt fails, check
  every pattern it touches, not just the first.
- **VSHL/VSHR direction words.** `to the left` is in the VSHL
  pattern, `to the right` in VSHR (same gap class, added
  symmetrically). A shift prompt with no direction word and no
  other signal is still ambiguous.
- **Prove dispatcher fixes on the chip, not just locally.**
  `replay_disp_local.py` replicates the disp outcome logic
  with the ref executor standing in for the chip (predicted
  44 executed + 1 expected refusal, 0 wrong). The Colab
  re-proof (`vec4-v3-disp-reproof/`) then ran the same 45 rows
  against a fresh lowmem-merged 2B and matched the prediction
  row for row. The local replay is evidence; the on-chip disp
  file is the proof. Do not present the prediction as a
  measured result.
- **Lowmem merge pattern.** `merge_v3_lowmem.py` is
  model-agnostic (takes SRC/DST) and changes only the memory
  flow, never the math: widen in place, shift layer indices,
  mute — identical to stock — then shard every Qwen/vision
  tensor out immediately (512 MB shard cap, temp names
  renamed at finish) and emit chip layers one at a time
  through `emit_layers_sink`. Only embed_tokens.weight, the
  final norm and the merger bias stay in RAM for the final
  edits. Output format is identical to stock, so
  `run_merged.py` / `tool_v3.py` / `cache_probe_v3.py` need no
  changes. The driver prints VmRSS at each phase boundary;
  record the peak in the RESULT file. Local tests
  (`test_lowmem.py`, `test_lowmem_smoke.py`) prove the sink
  equals stock `emit_layers` tensor-for-tensor and the
  ShardWriter round-trips — they do not prove the merge
  against real weights; Gate 0 is that proof.
