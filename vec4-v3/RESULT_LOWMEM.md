# VEC4 v3 low-memory merge driver — LOCAL ONLY, not pushed

> Pushed to lesterppo/vec4-qwen on 2026-11-05 on the operator's explicit order (the v3 local-only hold was lifted for this push only). The driver is GREEN: the 2B-v3 lowmem gate (session vec4-v3-2b-lowmem) passed Gate 0 720/720 with merge peak 8.65 GB.

## Why
The 2026-11-05 2B-v3 gate (session `vec4-v3-2b`, kyu009009) died during
the stock merge: the VM memory cgroup killed the merge process at
11.70 GB anon RSS, `chipformer.json` never created, no Gate 0. The
stock merge holds the whole merged state dict in RAM: widened base
bf16 + ALL 26 chip layers in fp32 + their bf16 conversion copies.

Peak arithmetic for 2B-v3 (from the driver's own size formula and the
public Qwen3.5-2B config: hidden 2048, inter 6144, nk = nv = 16,
head dim 128; merged hidden 2560, 26 chip layers, vocab 248576):
- final merged model ~9.5 GB bf16 (text+embed 8.61 GB + vision tower)
- fp32 chip layers: 26 layers x ~294 MB = ~7.6 GB
  (per layer: qkv 6144x2560, gate/up/down 6144x2560 each, zw/ow
  2048x2560 each, all fp32)
- stock peak ~ widened base (~5.7 GB) + fp32 chip dict (7.6 GB)
  + bf16 conversion copies (~3.8 GB) ≈ 17 GB > 13 GB cgroup.
The 2B-v2 merge survived only because its circuit is smaller
(186 slots, chip width 256, merged hidden 2304).

## The driver: `merge_v3_lowmem.py`
Memory flow ONLY — no math change vs `merge_v3_2b.py`:
1. Load, build circuit, allocate, widen in place, shift Qwen layer
   indices, mute — identical to stock.
2. Shard out every Qwen/vision tensor immediately (512 MB shards) and
   drop it from RAM. Only embed, final norm, and merger bias are kept,
   because they are edited later.
3. Chip layers emit one at a time through `emit_layers_sink` — a
   mechanical copy of `chipformer.q35.emit.emit_layers` whose ONLY
   differences are the `sink` callback, `out = {}` removed,
   `out[p + k] = t` replaced by `sink(p + k, t)`, and `return out`
   removed. Each tensor is converted to bf16, appended to the shard
   buffer, and freed; no full fp32 chip dict ever exists.
4. Final embed rebuild (1.27 GB for 2B, transient x2 with the old
   embed), then norm edit, then index + config + chipformer.json.
   Output format is identical to stock (sharded safetensors + index),
   so `run_merged.py` / `tool_v3.py` / `cache_probe_v3.py` need no
   changes. `stage0_v3_lowmem.py` is the 2B-only Gate 0 driver that
   calls it (separate output dir/files, stock files untouched).

Bounded peak for 2B-v3: widen ~6 GB, emit ~3 GB (embed 1.27 + shard
buffer <= 0.5 + one fp32 layer 0.29 + overhead), final embed ~4.3 GB.
The driver prints VmRSS at each phase boundary and ru_maxrss at the
end, so the gate log carries a measured peak.

## Local verification (this VM, CPU torch in `~/workspace/lowmem-test-venv`)
All PASS, 2026-11-05 morning:
- `test_lowmem.py` Test A: stock `emit_layers` vs `emit_layers_sink`
  on the first 2 layers of the real v3 circuit via a proxy reporting
  n_layers() = 2 (emit is per-layer independent) — **28 tensors,
  all torch.equal**. Plus a body-level diff: sink body identical to
  stock emit body over 98 lines after removing `out = {}` /
  `return out` and the assignment swap.
- `test_lowmem.py` Test B: ShardWriter on synthetic tensors incl. a
  non-contiguous view — index covers every key exactly once, shards
  reload exact, temp files gone, total_size matches (4 keys, 3
  shards, 8006 bytes).
- `test_lowmem_smoke.py`: full driver end-to-end on a fake tiny base
  (hidden 64, 2 layers, inter 512, vocab 1000) — all 26 chip layers
  emitted, merged hidden 512, vocab 1280, output index exactly the
  expected **383 keys** (364 chip + 19 base), shards reload, embed
  (1280, 512) and norm shapes right, chip dims mostly -1 as stock
  math requires, chipformer.json carries `"lowmem": true`. Phase
  RSS: load 227 MiB, widen 238, shard-out 238, chip emit 561, peak
  600 MiB — the emit delta is one fp32 layer + shard buffer, exactly
  as designed. Slot numbers match the real 2B merge log (mu 37,
  tau 38, code 39-46), as they should: allocation is base-independent.
- Three real bugs caught by these tests before Colab, all fixed:
  `tiny` never imported (NameError at the norm step); `Graph` /
  `Alloc` annotation names never imported; `Head` (isinstance in the
  emit loop) never imported. A syntax check alone would have missed
  all three.
- What local tests do NOT prove: the merge against real 2B weights.
  This VM has no GPU and only 7 GB RAM. That proof is the Colab
  Gate 0 (720/720) with this driver — any math divergence fails it.

## Status
- Driver + tests written and locally verified (results above).
- Colab 2B gate with the lowmem driver: COMPLETE 2026-11-05 ~08:02 HKT
  (session vec4-v3-2b-lowmem, kyu008008 first try; JSONs in
  `~/workspace/qwenputer-colab/vec4-v3-gate-2b-lowmem/`, counts
  parent-verified): merge peak 8.65 GB (driver log) vs stock killed
  at 11.7 GB; merged hidden 2560, 50 layers, 4,597,653,376 params,
  9.20 GB bf16, 16 shards, 981 keys; Gate 0 720/720; cache 40/40 all
  drivers (medians stepwise 1.170 s, generate 0.631 s, cached
  0.637 s); orig 26/28 (VSORT schema artifact x2); para 11/16;
  adv 8/10; disp 41 executed + 2 refused + 2 wrong (same three
  para_new gaps as 0.8B-v3); plain 4/5 (greeting false trigger);
  notool 5/5. Verdict: lowmem driver GREEN — the merge-host RAM
  blocker is closed. 2B model selection improves over 0.8B but
  still does not generalise.
- No push to `lesterppo/vec4-qwen` — standing rule: v3 stays local
  until Peter explicitly orders a push.
