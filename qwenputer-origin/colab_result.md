# Qwenputer on free Google Colab — result

**Verdict: `FALLBACK_CORE_ONLY`**

The full in-model Qwenputer run could NOT load on the free GPU allocated:
bf16 checkpoint transfer to CUDA OOM'd. The standalone RV32 core (same
compiled CPU, WITHOUT the Qwen language-model wrapper) passed its checks.
This is NOT a full in-model replication.

## Arithmetic check (independent, local)

`123456789 * 987654321 = 121932631112635269` — verified with Python before
any Colab run. The expected product never appeared in any model output,
because no model output was ever generated (the run died at load).

## Session / GPU (measured, not assumed)

- Session: `qwenputer-1103`, created 2026-11-03 19:06:02 HKT, stopped
  2026-11-03 19:30:55 HKT, one session only, no account switch, no quota
  retry-loop (creation succeeded first try).
- Account: `cyc236es` (active before, during, and after; rotation file
  unchanged: `2026-11-03 cyc236es`).
- GPU: Tesla T4, nvidia-smi total 15,360 MiB, 14,913 MiB free at start.
  torch-reported total capacity 14.56 GiB (14,912 MiB) — the number in the
  OOM message below.
- VM env at start: Python 3.13.16, torch 2.11.0+cu130, transformers 5.22.0,
  triton present, `uv` present, disk 113 GB free on /content.
- Checkpoint config (from the downloaded `config.json` /
  `chipformer.json`): hidden_size 3968, num_hidden_layers 43
  (11 chip layers + 32 Qwen layers, `chip_layers: 11`,
  `qwen_layers_from: 11`), model_type `qwen3_5_text`, vocab 478,656.

## Full in-model attempt — the real result

Official path, isolated uv venv (`uv sync --extra hf` in
`experimental/riscv`, then `uv run python -m chipformer.rv.chat`), plain
fast path (RvChat default, no executor driver, no quantisation, no weight
change), firmware `tests/data/rv/calc-bridge.elf`, ask
`Use calc: what is 123456789 * 987654321?`:

- `uv sync --extra hf`: 58.7 s first time, 0.0 s on the corrected rerun
  (already synced).
- Full run: exit 1 in **72.6 s**, peak VRAM polled via nvidia-smi
  **14,901 MiB**, product present: **no**, generated text: **none**,
  text tokens: **n/a (never reached generation)**, chip tokens: **n/a**.
- What happened, verbatim (short): weights loading reached
  `Loading weights: 100% | 580/580`, then `RvChat.__init__` at
  `self.model = self.model.to(self.dev).eval()` raised:
  `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 70.00 MiB.
  GPU 0 has a total capacity of 14.56 GiB of which 11.81 MiB is free.
  Including non-PyTorch memory, this process has 14.55 GiB memory in use.
  Of the allocated memory 14.42 GiB is allocated by PyTorch, and 25.44 MiB
  is reserved by PyTorch but unallocated.`
  (Allocator warning: tried 73,400,320 bytes, free 12,386,304 bytes,
  total 15,637,086,208 bytes.)
- Interpretation: the 18 GB bf16 checkpoint does not fit a ~14.6 GiB-usable
  T4 in the official bf16 `.to(cuda)` path. This matches the pre-run
  expectation and the author's L4 (24 GB) numbers — the author's numbers
  were NOT reproduced and are NOT claimed here. No CPU-offload /
  device-map workaround was attempted (out of the official-path scope).

## Download

- Repo: `divagr1925/Qwen3.5-4B-Qwenputer`, unauthenticated `hf download`,
  NOT gated (no token needed, no token used or pasted anywhere).
- Download started 19:12:42 HKT; last shard timestamped 19:19 HKT;
  total **18 GB**, 8 shards (3.6 GB + 6× 2.1 GB + 1.8 GB, plus tokenizer /
  config / chipformer.json). Exact download seconds were NOT recorded:
  the first driver crashed on a bookkeeping bug (see below) immediately
  after the download subprocess returned, so only the start time and the
  shard timestamps bound it (~6–7 min). Do not quote a precise download
  time.
- No push to GitHub, no post to Threads, nothing posted anywhere.

## Two earlier full attempts that FAILED FOR DRIVER REASONS (not model results)

These are recorded so the OOM above is not confused with them. Neither
reached model load:

1. System-pip attempt (first driver, 19:08 HKT): pip install upgraded
   torch 2.11 → 2.15.0 in 215.2 s and broke the preinstalled torchvision
   0.26.0 (`torchvision 0.26.0+cu130 requires torch==2.11.0`). The chat
   died at import in 14.0 s, exit 1:
   `RuntimeError: operator torchvision::nms does not exist` →
   `ModuleNotFoundError: Could not import module 'Qwen3_5ForCausalLM'`.
   This was my driver's environment breakage (official `uv sync` uses an
   isolated venv and does not touch system torchvision), not a model
   verdict. The same session was then run via the official uv venv.
2. Uv-venv attempt with a wrong relative model path (19:24–19:25 HKT):
   `../qwenputer-rv` from `experimental/riscv` resolves to
   `experimental/qwenputer-rv`, which does not exist. Exit 1 in 18.0 s,
   peak VRAM 0 MiB:
   `FileNotFoundError: [Errno 2] No such file or directory:
   '../qwenputer-rv/chipformer.json'`.
   Corrected to the absolute path `/content/qwenputer/qwenputer-rv`
   (verified to contain `chipformer.json` before relaunch) — that is the
   OOM run above.

## Fallback: standalone RV32 core ONLY (same session, after the OOM)

Plainly: this runs the same compiled RV32 CPU core but WITHOUT the Qwen
language-model wrapper, so it is NOT a full in-model replication.

- `c-fib` via the reference ISA emulator (`chipformer.rv.ref.Rv32`, free
  run of `tests/data/rv/c-fib.qemu.elf`), 15.9 s:
  `halted True steps 2114686 exit 0 stdout 'fib(25)=75025\n'`
  — matches `tests/data/rv/qemu.json` (`c-fib`: exit 0,
  `fib(25)=75025`).
- `c-sieve4` via the traced reference machine
  (`ref.Machine.from_program(elf.program(...)).run()`), 0.9 s:
  `halted True fault None exit 0 stdout 'primes below 10000: 1229\n'
  trace_tokens 176225`
  — matches `qemu.json` (`c-sieve4`: exit 0, `primes below 10000: 1229`).
- An earlier fallback `c-fib` line in the first driver's log
  (`halted False`, empty stdout) is INVALID and superseded: it used
  `Machine.run()`'s default `max_records=1e6`, which truncates fib(25)
  (2,114,686 instructions) before HALT. The corrected ISA run above is
  the fallback result.

## What was NOT done / NOT verified

- No in-model generated text, token counts, lock/head mismatch, or
  wall-seconds for generation — the run died at load, so none exist.
- No second session, no account switch, no L4/G4 attempt, no
  quantisation, offload, or executor variant, no MicroPython / Mario /
  Doom / full riscv-tests suite (out of scope per task).
- The author's published benchmark numbers (e.g. 97.5 on arithmetic sets,
  ~3,200 trace tokens/s on L4) were not reproduced here.

## Files in this directory

- `colab_result.md` — this file.
- `qwenputer_result_uv.json` — machine-readable result of the corrected
  uv run (full OOM + corrected fallback).
- `qwenputer_uv.log` — full VM log tail for the uv runs.
- `qwenputer_result_first.json`, `qwenputer_run_first.log` — first
  (system-pip) driver's log/result, including its import failure and
  truncated c-fib line, kept for audit.
- `run_qwenputer.py`, `run_qwenputer_uv.py`, `fib_check.py` — the drivers
  / fallback script used on the VM (kept locally, not in the upstream
  clone; upstream clone at `upstream/` was inspection-only and was not
  pushed anywhere).
- `colab_registry.json` — isolated Colab session registry for this run.
