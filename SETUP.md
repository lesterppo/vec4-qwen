# SETUP — replicate VEC4 on free Colab, step by step

## What you need

- A free Google Colab account and a T4 GPU session. Every original
  run used a Tesla T4, 15,360 MiB total, on a VM with ~12 GB RAM and
  ~113 GB free disk. A bigger GPU is not required and not assumed;
  a smaller GPU has not been tested and will probably OOM on the
  2B merge (7.28 GB bf16, ~7.5 GB tool peak).
- Python 3.13, torch 2.11.0+cu130, transformers 5.22.0 — the versions
  the original session probes measured. Nearby versions are
  untested; if a rerun uses different versions, record them in the
  RESULT file instead of silently substituting them.
- No API keys, no logins, no paid services. Base models
  `Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.5-2B` were ungated Hugging Face
  downloads at the time of the runs. The upstream repo is public.
- Your own way of creating and driving a Colab session (the
  original runs used a private CLI wrapper that is not in this
  repo). The scripts here are plain Python run on the VM; they do
  not depend on any particular driver.

## Directory layout on the VM (this matters)

Several scripts hardcode these paths. Arrange the VM exactly like
this, or adapt the paths in the scripts and say so in your RESULT
file:

```
/content/qwenputer-colab/            <- this repo, cloned here
/content/qwenputer-colab/upstream/   <- git clone --depth 1 https://github.com/divagr18/qwenputer
/content/qwenputer-colab/vec4/       <- this repo's vec4/ (and so on per experiment)
/content/qwen35-08b/                 <- snapshot_download("Qwen/Qwen3.5-0.8B")
/content/qwen35-08b-vec4/            <- merge output (0.8B)
/content/qwen35-2b/                  <- snapshot_download("Qwen/Qwen3.5-2B")
/content/qwen35-2b-vec4/             <- merge output (2B)
/content/vec4-toolcall/              <- per-experiment output dir (stage0 writes here)
... and likewise /content/vec4-fix/, /content/vec4-untested/,
    /content/vec4-app/, /content/vec4-2b/, /content/vec4-v2/,
    /content/vec4-2b-v2/, /content/vec4-eff/
```

The merge drivers resolve the upstream chipformer via
`HERE.parent / "upstream"` (for `vec4/merge_vec4.py`) or an explicit
`/content/qwenputer-colab/upstream` sys.path entry (later drivers).
If the merge fails with an import error for `chipformer`, the
layout above is the first thing to check.

Note the two different output locations per experiment: the
experiment's own directory under `qwenputer-colab/` holds the
scripts you uploaded, while `/content/<experiment>/` holds the
stage0 output JSON. Download both before teardown.

## Run order

Work through the directories in order. Each experiment's RESULT.md
gives its own driver sequence and expected numbers; the summary
below is the map, not a substitute for those files.

### 0. Origin (optional, context only)

`qwenputer-origin/` — the 4B Qwenputer replication attempt.
Expected: the full 18 GB bf16 checkpoint OOMs on a T4
(`FALLBACK_CORE_ONLY`); the standalone RV32 core passes
(`fib_check.py`: `fib(25)=75025`). This step is context for why
VEC4 was built small. It is not a gate for anything else.

### 1. `vec4/` — build VEC4 into 0.8B

Local (no GPU) first, after cloning upstream:

```
python3 vec4/test_ref.py        # reference simulator, 210 cases
python3 vec4/test_circuit.py    # IR circuit vs simulator, 210/210
python3 vec4/run_tiny.py        # tiny standalone model, forward argmax 210/210
```

Then on Colab, in order: `download_base.py` (or the stage0
equivalent in later experiments), `merge_vec4.py SRC OUT`,
`run_merged.py OUT BASE --out vec4_merged_result.json`.
Expected: merged hidden 1280 (1024 + 256 chip width), 42 layers
(18 chip + 24 Qwen), vocab 248576, 1,526,970,752 params, 3.05 GB
bf16, VRAM peak ~3,010 MiB, regression **210/210**, text sanity
"The capital of France is" → " Paris.\nThe capital of France is
Paris." If the first merged run is 0/210, see AGENTS.md
(TOKEN_MASK / `full=True`) before anything else.

### 2. `vec4-toolcall/` — tool-call bridge

On Colab: `stage0.py` first (download, clone upstream if needed,
merge, Gate 0 210/210, chat-template probe), then `toolcall.py`
(main 20 cases), then `diag_dot_order.py` (the `vec_dot`-first
diagnostic, 10 DOT cases). Expected: model-emitted tool calls
20/20, ADD 10/10, DOT 1/10, strict end-to-end **11/20**, chip for
the tool actually called 20/20, plain 5/5, should-not-call 5/5.
The driver's loose summary (20/20 by called op) is superseded by
the strict audit in `RESULT.md` — re-score strictly.

### 3. `vec4-fix/` — DOT fixes A and B (C is step 6)

On Colab: `stage0.py` / `stage0_fix.py` (Gate 0 210/210, twice in
the original run because the first session was reclaimed),
`variants.py` (Variant A, single-tool-per-prompt),
`run_variant_b.py` (Variant B: guard+retry B1, host-rewrite B2).
Expected: A **20/20** strict, B1 DOT **10/10** (first attempt 1/10,
9/9 corrected on retry), B2 0 exercised. The 2B config check
(`qwen35_2b_config_check.json`, from `config.json` only) is the
record of why C was skipped here: intermediate 6144 vs 3584,
hidden 2048 vs 1024.

### 4. `vec4-untested/` — speed, B2, interleaved

On Colab: `stage0_untested.py` (Gate 0 210/210), then in order:
`exp1_speed.py` (cached driver; `cache_probe.py` first, and
`exp1_chip_fix.py` if the cached scorer needs the END-trim fix),
`exp2_b2.py` (10 adversarial DOT prompts), `exp3_interleaved.py`
(single, double, text-after-chip) plus `exp3_control.py` (text-only
control, same prompts). Expected: cached driver corrected strict
**20/20**, median ~10.1 s/case vs ~65.6 s stepwise (~6.5x);
B2 in-model **1/1** (case 6, bare "140"); interleaved chip 20/20
single and 5/5 double, English reporting 0/20, text-after-chip
sane 4/5 identical to control 4/5.

### 5. `vec4-app/` — film-recommendation application

On Colab: `make_dataset.py` first (dataset is pre-registered in
`dataset.json` — do not regenerate it after seeing model output),
`stage0_app.py` (Gate 0 210/210), then `app.py` (Mode 1
model-orchestrated, Mode 2 host-orchestrated, text-only control;
`app_resume.py` resumes Mode 1 from a downloaded partial after a
reclamation). Expected: chip 90/90 in both modes; Mode 1 workflow
clean 0/10, top-1 3/9 non-trivial, top-3 0/9; Mode 2 top-1 3/9,
top-3 0/9; control top-1 0/9. No scenario fully correct in any
mode. Read `RESULT.md` before interpreting any of these numbers.

### 6. `vec4-2b/` — Path C by retargeting

Read `RETARGET.md` first — it was written before merging and is
the template for any future retarget. On Colab: `stage0_2b.py`
(download 2B, in-place-widen merge, Gate 0 210/210, text sanity),
`cache_probe_2b.py` (stepwise = generate = cached-trimmed
token identity, 20/20, before any cached tool test),
`tool_2b.py --only baseline|A|B|plain` (the `notool` variant is
the should-not-call set). Expected: merged hidden 2304, 42
layers, vocab 248576, 3,640,992,896 params, 7.28 GB bf16, VRAM
peak ~7,162 MiB at Gate 0 and ~7,408–7,474 MiB on tool runs;
C-baseline **20/20** strict (DOT 10/10), C-A **20/20**, C-B1
**10/10** with B2 0 exercised, plain 5/5, should-not-call 5/5.
Text sanity is looser than 0.8B's (" Paris.\nA. True\nB. False"),
recorded verbatim.

### 7. `vec4-v2/` — V2 on 0.8B

Local first: `test_ref_v2.py`, `test_circuit.py` (396 cases,
66/op, 0 mismatches), `test_dispatcher.py`. On Colab, in order:
`stage0_v2.py` (merge, Gate 0 396/396), `cache_probe_v2.py`
(24/24 token-identical), `tool_v2.py` (configs orig/para/adv/
disp/plain; cached driver; strict scoring, no guard/retry).
Expected: orig 20/20, para 7/12, adv 3/10, plain 5/5,
should-not-call 5/5; disp 39 executed + 1 expected refusal
after the two gate-bug fixes (SUB expectation computed via
`ref.py`, dispatcher pattern allowing "the" in "sum of the
products") — see `RESULT_V2_LOCAL.md` for the first-gate
38+1+1 wrong and exactly which two bugs were fixed, and do
not quote the first-gate disp numbers as the result.

### 8. `vec4-2b-v2/` — V2 on 2B

Read `RETARGET_V2.md` first. On Colab: `stage0_2b_v2.py`
(in-place-widen merge, Gate 0 396/396), `cache_probe_2b_v2.py`
(24/24), `tool_2b_v2.py` (same configs as step 7). Expected:
merged hidden 2304, slots 186, mu_dim 2118 (both from the
merge log — the pre-merge prediction said 142 slots / mu
2114, and the log is ground truth); orig 20/20, para 8/12,
adv 8/10, disp 39+1 refusal, plain 4/5 (one false trigger on
a greeting prompt), notool 5/5. If you port a driver from
step 7, grep every path in it — the original port shipped
one stale 0.8B path.

### 9. `vec4-eff/` — efficiency comparison

Read `vec4-eff/README.md` first. On Colab: `stage0_eff.py`
— but merge each model in its OWN process (the combined
stage0 OOMed on the 2B merge; see its README). Then
`eff.py` per model (gate 60/60, then cached/stepwise timing
plus same-VM baselines). Expected: 0.8B cached median
0.477 s/case, 2B 0.521 s/case; NumPy median ~0.035–0.040 µs,
torch CUDA ~0.004 µs; fixed costs 22–30 s load, 3.05–7.28 GB,
3.0–7.2 GB VRAM. Energy is not measured here — do not add
an energy claim to a rerun without a real measurement method.

## After every run

1. Download every `*_result.json`, `stage0_done.json`,
   `cache_probe*.json` and the experiment's `RESULT.md` (if you
   wrote one) off the VM **before** stopping the session.
2. Verify the counts in the raw JSON yourself — do not trust a
   driver's summary line. Count strict-true cases per
   configuration, count B1 vs B2 separately, and check the 140
   example case in each configuration.
3. If you change a case set, a prompt, a scoring rule, or a
   driver, say so in the RESULT file and re-prove from Gate 0.
   A number measured under different rules is not comparable to
   the numbers in this repo, and presenting it as comparable is
   the one error this repo's discipline exists to prevent.
