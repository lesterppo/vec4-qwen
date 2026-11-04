# vec4-eff — efficiency comparison (VEC4 v2 chip vs host compute)

Purpose: measure, honestly, how far behind host compute the in-model
chip is on raw latency/throughput/memory, and what the fixed costs
are. The expected result — the chip loses by orders of magnitude —
is stated in `eff.py`'s docstring before any number is measured.
This experiment does not try to make the chip look competitive; it
prices in-model exactness.

## Drivers

- `stage0_eff.py` — probe, clone upstream if needed, download both
  bases, merge both v2 models (0.8B via `merge_08b.py`, 2B via
  `merge_2b.py` with in-place widening). Merge each model in its
  OWN process: merging both in one process was OOM-killed on a
  13 GB VM during the 2B merge (see `RESULT_EFF.md`). If your
  runner cannot pass CLI args to a detached process, wrap the
  driver in a tiny script that sets `sys.argv` first — such
  wrappers are run shims, not drivers, and are not in this repo.
- `eff.py MERGED_DIR BASE_DIR --out FILE` — for one merged model:
  (1) load + record load seconds, params, checkpoint GB, VRAM;
  (2) correctness gate: 60 fresh chip cases (10 per op, seed
  20261105) must ALL match `ref.py` before any chip timing is
  reported — a model failing the gate gets no timing numbers;
  (3) chip cached timing (median/p90/mean per case) + stepwise
  timing on a 12-case subset (2 per op); (4) same-VM baselines:
  pure-Python `ref.py` loop, NumPy vectorised (100k cases),
  torch CPU, torch CUDA. Writes JSON incrementally.
- `circuit.py`, `ref.py` — the v2 circuit and simulator (copies
  from `vec4-v2/`; `merge_08b.py` is `vec4-v2/merge_vec4.py`,
  `merge_2b.py` is `vec4-2b-v2/merge_2b_v2.py`).

## Measured result (2026-11-04 evening, session vec4-eff)

Full record: `RESULT_EFF.md`. Headline: both gates 60/60 PASSED.
Chip cached median 0.477 s (0.8B) / 0.521 s (2B) per case vs
NumPy median ~0.035-0.040 us and torch CUDA ~0.004 us — ~7
orders of magnitude behind NumPy, ~8 behind torch CUDA.
Fixed costs: 22-30 s load, 3.05-7.28 GB checkpoint, 3.0-7.2 GB
VRAM. 2B is only ~9% slower per cached case than 0.8B despite
2.4x parameters (fixed per-forward overhead dominates).
Energy was NOT measured: only instantaneous nvidia-smi
snapshots (idle ~10 W, mid-chip-loop ~43-49 W) exist, and they
must not be quoted as energy totals. The chip cannot batch
sequential generation, so throughput is the inverse of latency
(~2.1 cases/s cached) with no batching escape hatch.
