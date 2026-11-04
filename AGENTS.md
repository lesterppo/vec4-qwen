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
- No flash kernels, no stepwise interleaved rerun, no mid-input
  text interruption, no sentence-form B2, no broader 2B adversarial
  set. These are open, not failed. A new run that covers one should
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
