# VEC4: a tiny GPU-style vector unit compiled into Qwen weights

Compiling a machine into model weights, without training: a 4-lane,
4-bit vector unit (VADD lane-wise mod 16, VDOT max 900) is built as a
chipformer IR circuit and merged, with no retraining, into
Qwen3.5-0.8B and Qwen3.5-2B, where it runs in-model on a free Colab T4.
Then the pragmatic question is answered honestly: the chip is exact
everywhere, but the small model cannot *use* it reliably until the
host-side interface is fixed — and a bigger model fixes the interface
failure on its own, on the tested cases, at 2.38x the size.

This repo is the complete record of that arc, scripts plus measured
results, in the order the work actually happened. Every number below
was measured on a free Colab T4 and verified against the raw JSON in
the matching directory, not copied from a summary. Nothing here was
pushed until every file was verified against the remote tree.

## Verdicts first

| Stage | Verdict | Headline number |
|---|---|---|
| Qwenputer replication (4B) | `FALLBACK_CORE_ONLY` | 18 GB bf16 checkpoint OOMs on a T4; standalone RV32 core passes (`fib(25)=75025`, 1229 primes below 10000) |
| VEC4 into 0.8B | `MERGED_GREEN` | Simulator, IR circuit and merged in-model all **210/210** |
| Tool-call bridge (0.8B) | `TOOLCALL_PARTIAL_DOT_TOOL_SELECTION` | Chip 20/20, but DOT tool selection **1/10** — a selection failure, not a circuit failure |
| DOT fixes (0.8B) | `FIXED_BY_A_AND_B1` | Single-tool A **20/20**; guard+retry B1 DOT **10/10**; B2 untested at that point |
| Untested items (0.8B) | `SPEED_CACHED_GREEN` + `B2_TESTED_1` + `INTERLEAVED_PARTIAL_ENGLISH_BARRIER` | Cached driver ~6.5x faster, 20/20; B2 in-model 1/1; interleaved chip 20/20 but English reporting 0/20 |
| Film-recommendation app (0.8B) | `APP_PARTIAL_ORCHESTRATION_RANKING` | Chip 90/90 in both modes, but no mode is a working recommender (see below) |
| Path C: VEC4 into 2B | `C2B_GREEN` | Both-tools baseline **20/20 strict** (DOT 10/10) vs 0.8B's 11/20 — the bigger model fixes selection on its own |

The one-line story: **the circuit was never the problem after MERGED_GREEN.
Every later failure was interface — tool selection, English reporting,
orchestration — and every later fix was interface, except Path C, where
scale fixed the interface too.**

## What VEC4 is

- 4 lanes in parallel, each lane an unsigned 4-bit value (0–15).
- Vector registers A and B live in Gated DeltaNet state (one head per
  register), written from input tokens and read back at GO.
- `VADD`: lane-wise addition mod 16. `VDOT`: sum of `a_i * b_i`,
  max 4·15·15 = **900**, held as 10 bits (900 is not bf16-exact, so the
  dot never lives as a single integer).
- Token encoding: OP (ADD=0, DOT=1), VAL, GO, OUT_ADD, OUT_DOT, END —
  full spec in `vec4/SPEC.md`, reference simulator in `vec4/ref.py`.
- It is **not** CHIP-8, **not** RV32, **not** QwenPU. Only the compiler
  infrastructure is reused (see "What is reused" below); the ISA, the
  encoding, the simulator and the circuit are new.

## Repo map

Each experiment directory is self-contained and preserved exactly as
run — later directories copy the code they need rather than editing
earlier ones in place. Do not "clean this up" by merging directories:
the copies are the audit trail, and several scripts hardcode the
directory layout documented in `SETUP.md`.

- `qwenputer-origin/` — the starting point: replicating Divyansh
  Agrawal's Qwenputer (a 32-bit RISC-V CPU compiled into Qwen3.5-4B)
  on free Colab. Result: `FALLBACK_CORE_ONLY`. Read `colab_result.md`.
- `vec4/` — the original build. `SPEC.md` (ISA + encoding),
  `circuit.py`, `ref.py`, `test_circuit.py`, `test_ref.py`,
  `merge_vec4.py`, `run_merged.py`, `run_tiny.py`, `RESULT.md`.
  Staged gates: simulator → IR circuit → tiny standalone model →
  merged real 0.8B. All 210/210.
- `vec4-toolcall/` — the pragmatic bridge: `vec_add` / `vec_dot` as
  Qwen chat-template tools, chip answers via `<tool_response>`, model
  finishes in English. `toolcall.py`, `diag_dot_order.py`,
  `RESULT.md`. Strict end-to-end 11/20 (ADD 10/10, DOT 1/10).
- `vec4-fix/` — all three suggested DOT fixes. `variants.py`,
  `run_variant_b.py`, `RESULT.md`. A: single-tool-per-prompt 20/20.
  B1: guard+retry DOT 10/10 (first attempt 1/10, 9/9 corrected).
  B2: written, 0 cases exercised. C: 2B config gate FAIL (from
  `config.json` only) — intermediate 6144 vs 3584, hidden 2048 vs
  1024 — which is what Path C later retargets past.
- `vec4-untested/` — the three items left open after the fix run.
  `exp1_speed.py` (cached driver), `exp2_b2.py` (adversarial B2),
  `exp3_interleaved.py` (text+chip in one sequence), `RESULT.md`.
  Cached 20/20 at ~6.5x stepwise speed; B2 in-model 1/1; interleaved
  chip exact (20/20 single, 5/5 double) but English reporting of a
  chip result 0/20 — the host must decode chip OUT tokens.
- `vec4-app/` — the application test: quantized film recommendation
  (ADD two taste profiles mod 16, DOT against 8 films, rank).
  `make_dataset.py`, `dataset.json` (pre-registered before any model
  run), `app.py`, `RESULT.md`. Chip 90/90 in both modes; Mode 1
  (model-orchestrated) workflow-clean 0/10 with `vec_add` stickiness
  in 160/160 DOT attempts; Mode 2 (host-orchestrated) top-1 3/9
  non-trivial, top-3 0/9; text-only control top-1 0/9. No scenario
  fully correct in any mode. Ship Mode 2's shape only with
  HOST-RANKED output (see its RESULT.md) — that form is 10/10 on
  ranking by construction and is not claimed as a model result.
- `vec4-2b/` — Path C completed by retargeting. `RETARGET.md`
  (analysis written before merging), `merge_2b.py`, `stage0_2b.py`,
  `cache_probe_2b.py`, `tool_2b.py`, `RESULT.md`. Merged hidden 2304
  (chip width 256 unchanged, ballast 36, mu slot 2114), 3,640,992,896
  params, 7.28 GB bf16, Gate 0 210/210, cache re-proof 20/20
  token-identical, C-baseline 20/20, C-A 20/20, C-B1 10/10
  (B2 0 exercised on 2B). Recommendation: do **not** ship 2B for
  this problem — 0.8B + A + B fixes it at under half the size.

## How to replicate (short version)

Full layout, entry points and per-experiment run order: `SETUP.md`.
Agent rules that keep a rerun honest: `AGENTS.md`. In brief:

1. Free Colab, Tesla T4 (15,360 MiB), ~12 GB RAM, ~113 GB free disk.
   Python 3.13, torch 2.11.0+cu130, transformers 5.22.0 (the versions
   the original runs measured; nearby versions are untested).
2. Clone this repo to `/content/qwenputer-colab` and clone the
   upstream Qwenputer repo to `/content/qwenputer-colab/upstream`
   (see "What is reused" — upstream is **not** vendored here).
3. Run one experiment's `stage0*.py` first (download base, merge,
   Gate 0 regression 210/210). Never run tool tests on a failed gate.
4. Run that experiment's driver(s), in the order its `RESULT.md`
   gives. Score strictly (see `AGENTS.md`): requested op, correct
   tool + args, chip for the requested op, final English contains
   the correct result, no `<tool_call>` in the final. B2 rewrites
   are never counted as B1 model successes.
5. **Download every result JSON off the VM before teardown.**
   Free Colab sessions get reclaimed without warning; several runs
   in this repo's history lost end-only files that way, and later
   runs switched to per-case incremental writes for exactly that
   reason.

## What is reused, and what is not in this repo

Reused at runtime, unmodified, from
[divagr18/qwenputer](https://github.com/divagr18/qwenputer)
(clone it yourself; it is not copied here): the chipformer IR
(`q35/ir.py`: Graph/FFN/Head/Target, exactness evaluator), gadgets
(`q35/prims.py`), the weight emitter (`q35/emit.py`), the standalone
model builder (`q35/tiny.py`), and the generic widen/mute parts of
`q35/merge.py`. Everything VEC4-specific — ISA, encoding, simulator,
circuit, merge drivers, tool drivers, datasets — is in this repo.

Not in this repo, by design: no merged checkpoint (each run's
checkpoint lived only on the Colab VM and was discarded at teardown;
rebuild it from the merge script — that is the point), no base-model
weights (download `Qwen/Qwen3.5-0.8B` / `Qwen/Qwen3.5-2B` from
Hugging Face, both ungated at the time of the runs), no credentials
of any kind (none were ever in these scripts; the account names in
RESULT files are Colab account labels, not credentials), no upstream
clone, no `__pycache__`, no Colab session registries, no logs or
per-case partials (final JSONs only).

## Limits that travel with the results

- 4 lanes, 4-bit values only. More lanes = a wider circuit and a
  larger merged model; the retarget arithmetic in `vec4-2b/RETARGET.md`
  shows how to recompute it, but it has not been built.
- ADD wraps mod 16 — wrong semantics for merging taste profiles at
  scale (see `vec4-app/RESULT.md`); profile combination should be a
  host policy (average / max / weighted with rescaling).
- Scores are coarse and tie-prone (ties in 3/10 app scenarios); a
  product needs an explicit deterministic tiebreak.
- 2B's unforced tool selection is proven only on the same 20
  same-wording cases. It is not a general claim.
- Text quality of the merged models is base-model quality plus
  merge side effects (2B's text sanity is looser than 0.8B's, recorded
  verbatim in `vec4-2b/RESULT.md`); no KL vs base was computed, and
  none is claimed anywhere in this repo.
- Speed: the cached driver is ~6.5x stepwise on 0.8B and ~11.6 s/case
  on 2B, but that is with reference DeltaNet kernels (flash kernels
  were never attempted — see below).

## Where to take it next (open items, in priority order)

These are the items the original runs explicitly left untested or
unproven. Each is a self-contained next experiment; keep the same
strict scoring and the same "download before teardown" discipline.

1. **Broader adversarial multi-tool set for 2B.** The one result
   that would change the ship recommendation: does 2B's unforced
   selection hold on new wordings, new operations and misdirection
   prompts, or was 20/20 an artefact of 20 same-wording cases?
   Include the 0.8B adversarial DOT set (`vec4-untested/exp2_b2.py`)
   as a starting point, plus multi-operation prompts where no guard
   can name the tool in advance.
2. **Sentence-form B2.** The one in-model B2 case produced a bare
   "140", not a sentence. Test whether a host-rewritten chip result
   can be verbalised into a full sentence, and whether that costs
   a second generation pass.
3. **Flash / fused DeltaNet kernels.** The cached driver was only
   attempted because stepwise was slow, and flash was conditional
   on cache failure, which never happened. Measure whether fused
   kernels beat the cached driver, and whether they change any
   token (re-prove token identity before switching drivers).
4. **Stepwise interleaved rerun + mid-input text interruption.**
   Interleaved chip computation was proven only with the cached
   driver; a stepwise rerun, and text interrupting a chip program
   mid-input, are both untested.
5. **Host-ranked application, honestly labelled.** Build the
   `vec4-app` Mode 2 shape with host-ranked output (host sorts the
   exact chip scores, model verbalises a pre-ranked list) and
   report it as a system result, not a model result — then find an
   application where the model genuinely adds something the host
   ranking does not already supply.
6. **Wider / deeper VEC4.** More lanes, or a third operation, via
   the retarget arithmetic. Watch the merged size and the FFN cap:
   the circuit's max is 98 units per sublayer, so there is headroom
   before any redesign is needed, but the widening arithmetic must
   be recomputed and Gate 0 re-proven from scratch.

## Provenance

Built November 2026 on free Google Colab, in the order above, as a
follow-up to a Threads post about Qwenputer. The original Qwenputer
authors' published claims (riscv-tests, ~18M instructions, C /
MicroPython, ~3,200 trace tokens/sec on an L4) are theirs, not
reproduced here — the only Qwenputer numbers in this repo are the
ones this work measured (see `qwenputer-origin/colab_result.md`).
