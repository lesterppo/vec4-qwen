# VEC4 v2 efficiency comparison — result (Colab, 2026-11-04 evening)

Session vec4-eff, account kyu009009 (kyu008008 refused
allocation), restored to kyu008008, rotation untouched.
Counts and ratios re-verified by the parent from eff_08b.json
and eff_2b.json (ratios recomputed from raw per-case times —
they match the files exactly).

## Headline (framing from eff.py, confirmed)

The chip is NOT competitive on latency, throughput, or memory
on any metric. That was the expected result, not a failed
measurement: a 42-layer model forward per generated token
cannot race host compute. The numbers below say by how much,
and what the fixed costs are.

## Per model (correctness gate 60/60 PASSED on both)

0.8B-v2: load 21.9 s, 1,526,970,752 params, 3.05 GB bf16,
VRAM load 2955 / max 3010 MiB. Chip cached median 0.477 s,
p90 0.625 s, mean 0.506 s per case; stepwise median 0.641 s
(cached 1.34x faster).
2B-v2: load 30.0 s, 3,640,992,896 params, 7.28 GB bf16,
VRAM load 7107 / max 7162 MiB. Chip cached median 0.521 s,
p90 0.583 s, mean 0.545 s; stepwise median 0.908 s (cached
1.74x faster).

## Same-VM baselines (per case, both runs roughly agree)

Python ref loop: 0.703 / 0.749 us. NumPy vectorised (100k
cases): 0.0043-0.0715 us by op, median 0.0349 / 0.0401 us.
Torch CPU: 0.337 / 0.314 us. Torch CUDA: 0.00396 / 0.00416 us.

## Ratios (chip cached median vs baseline)

0.8B: NumPy median 13,643,878x; Python 678,052x; torch CPU
1,415,349x; torch CUDA 120,218,731x slower.
2B: NumPy median 13,017,211x; Python 695,707x; torch CPU
1,658,897x; torch CUDA 125,255,241x slower.
In round terms: ~7 orders of magnitude behind NumPy, ~8
behind torch CUDA, per case. Throughput is the inverse of
latency (chip cannot batch sequential generation): ~2.1
cases/s (0.8B cached) vs ~28M cases/s NumPy median — the
same ratio, stated the other way.

## Two less obvious findings

1. 2B chip is only ~9% slower per cached case than 0.8B
   (0.521 vs 0.477 s) despite 2.4x parameters: per-token
   latency is dominated by fixed per-forward overhead (42
   layers either way, kernel launches), not parameter count.
2. The fixed costs are the real price of in-model exactness:
   22-30 s load, 3.05-7.28 GB checkpoint, 3.0-7.2 GB VRAM —
   versus microseconds and megabytes for host compute.
   Application-layer latency for comparison (from prior
   gates, not re-measured here): tool-call end-to-end medians
   12.7 s/case (0.8B-v2 orig) and 15.3 s/case (2B-v2 orig).

## Energy: not measured

Only instantaneous nvidia-smi snapshots exist: idle
10.14 / 10.13 W, mid-chip-loop 43.49 / 48.81 W (0.8B / 2B).
These are NOT energy totals and must not be quoted as such.

## Deviation (stated, with lesson)

stage0_eff.py was OOM-killed (cgroup, 11.65 GB RSS on a
12,975 MB VM) during the 2B merge, after the 0.8B merge had
succeeded — both merges in one process accumulate memory.
The 2B merge was rerun alone in a fresh process
(merge_2b_only.py, importing merge_2b.py unchanged) and
succeeded; no model was timed that failed to merge.
Lesson: merge in separate processes on a 13 GB VM — carry
this into any repo push of stage0_eff.py. The run-specific
shims (merge_2b_only.py, eff_*_run.py, needed because
exec_detach passes no CLI args) are NOT part of the driver
set and should not be pushed as drivers.

## What this does not change

Model-selection verdicts stand (0.8B-v2 para 7/12, adv 3/10;
2B-v2 para 8/12, adv 8/10; dispatcher+chip the reliable
route). Pushed to lesterppo/vec4-qwen on 2026-11-04 on the operator's explicit order.
