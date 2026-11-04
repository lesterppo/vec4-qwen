# VEC4 — result

## Verdict: MERGED_GREEN

A new GPU-style vector unit (VEC4) was compiled into weights and runs
in-model inside real Qwen3.5-0.8B on a free Colab T4, with 0 mismatches
on the full test set. No host arithmetic was used in any generation
path: the host only fed input tokens and read output tokens.

## Colab hardware (Stage 0, session vec4-1103, account cyc236es)

- Session created 2026-11-03, account cyc236es (rotation file
  `2026-11-03 cyc236es`, unchanged; no account switch, no quota retry,
  one session for probe + tiny + merge + run, torn down at the end).
- Probe 2026-11-03 11:53 UTC = 19:53 HKT: Intel Xeon @ 2.00 GHz, 2 CPUs;
  Tesla T4, 15,360 MiB, 0 used at probe; RAM 12 GB (11 available);
  disk 113 GB free; Python 3.13.16, torch 2.11.0+cu130,
  transformers 5.22.0, CUDA available.
- Full probe in `probe.md`.

## ISA summary (full spec in SPEC.md)

- 4 lanes, each an unsigned 4-bit value (0–15). Registers A and B,
  4 lanes each, written from input tokens into DeltaNet state
  (one head per register), NOT read from token features at GO —
  at GO the residual holds only GO's features, and DeltaNet state
  is the exact per-position memory the architecture offers.
- VADD: lane-wise addition mod 16, 4 lane results.
  VDOT: sum over lanes of a_i * b_i, max 900, held only as 10 bits
  (900 is not bf16-exact as a single integer, and neither are lane
  sums held as integers — all results live as bits).
  VMUL was out of scope by design (bonus only if it cost no gate).
- Encoding (standalone ids): OP_ADD=0, OP_DOT=1,
  VAL r v = 2+r*16+v (r 0–3 = A, 4–7 = B), GO=130,
  OUT_ADD l d = 131+l*16+d, OUT_DOT p d = 195+p*16+d
  (p 0=high, 1=mid, 2=low), END=243. Input is 10 tokens
  `[OP, A0..A3, B0..B3, GO]`; VADD output is 5 tokens, VDOT 4.

## Stage counts (exact)

- Stage 1 — simulator (`ref.py`, `test_ref.py`, seed 20261103):
  210 cases (104 VADD, 106 VDOT: 100 random each + 10 edges),
  all pass against hand-computed Python arithmetic.
- Stage 2 — IR circuit (`circuit.py`, `test_circuit.py`):
  210 cases, 0 mismatches, exactness checks passed
  (`ir.evaluate` with `check=True`). Circuit: 18 chip layers,
  447 dims, 14 heads, 961 FFN units, max 98 units per sublayer
  (well under the 3584 intermediate cap).
- Stage 2 — standalone tiny model (`run_tiny.py`, `tiny.build`,
  0.8B head shapes + one zero full-attention layer):
  forward argmax 210/210 in 55.1 s, stock `model.generate()`
  10/10, total 74.0 s. Tiny model: hidden 512, 19 layers,
  vocab 256, 203,529,024 params, 0.41 GB bf16,
  VRAM 395 MiB at load (printed), peak 432 MiB.
  Raw JSON in `tiny_result.json` (its `vram_after_load_mib`
  field was captured at the end of the run, 403 MiB, not at load).
- Stage 3 — merged model (`merge_vec4.py`, `run_merged.py`):
  210/210 in-model in 124.8 s, total run 135.2 s.
  Plain text prompt through the merged model:
  "The capital of France is" -> " Paris.\nThe capital of France
  is Paris." (ids [11751, 13, 198, 760, 6511, 314, 9338, 369,
  11751, 13]) — sane text, unchanged in character from base
  behaviour. No KL number is claimed (none was computed).
  Raw JSON in `merged_result.json` (its `load_seconds` field is
  actually total time; the printed load time was 8.5 s).

## Merged checkpoint

- Base: Qwen/Qwen3.5-0.8B, downloaded ungated (unauthenticated)
  from Hugging Face, 1.7 GB, single safetensors shard,
  hidden 1024, vocab 248320, 24 layers (18 linear + 6 full).
- Merged: hidden 1280, 42 layers (18 chip + 24 Qwen),
  vocab 248576 (248320 + 244, padded to a multiple of 64),
  1,526,970,752 params, 3.05 GB bf16.
  Chip offset v0 = 248320: chip token id = v0 + standalone id.
  Circuit allocation: 142 chip slots, ballast m = 20,
  mu slot 1074, Qwen layers muted on mu, keep = code + mu + tau.
- VRAM at load on the T4: 2,955 MiB (printed), peak 3,010 MiB
  during the run — comfortably inside the 15,360 MiB T4.
  (The earlier full Qwenputer 4B checkpoint was 18 GB bf16 and
  OOM'd on the same class of T4; this merged model is 3.05 GB.)
- Anti-sink was skipped, deliberately: test sequences are
  chip-only or text-only, and Qwen's full-attention output gate
  is muted on chip tokens, so attention dilution cannot affect
  either test. Mixed text+chip sequences were NOT tested.
- The merged checkpoint itself was NOT kept: it lived only on
  the Colab VM (`/content/qwen35-08b-vec4`) and was discarded
  with the session teardown. It can be rebuilt from
  `merge_vec4.py` + the base download + `circuit.py` in this
  directory. Nothing was pushed to GitHub.

## First merged attempt failed, and why (honest record)

- First merged run: 0/210, all cases generating the same wrong
  tokens starting with Qwen id 0, never reaching END, while the
  text prompt was already sane. Diagnosis via embedding rows:
  every chip token's row was identical to a text token's row
  (ballast + is_text=1 + tau=TAU_TEXT + code=-128, no is_go,
  no register features).
- Root cause: `ir.Graph.feature_value` masks ordinary token
  features to `t & 0xFFFF` (TOKEN_MASK). The offset translation
  in the first version of `circuit.py` saw the masked id
  (e.g. 248450 & 0xFFFF = 51842), which is below v0, so every
  chip token classified as text. The local offset test with
  offset=1000 had passed only because those ids are < 65536
  and masking is a no-op there.
- Fix: all VEC4 token features are now declared with
  `full=True`, so `feature_value` passes the whole id and the
  translation works for any v0 (verified locally with the real
  v0=248320, and on the VM before re-merging). Second merged
  run: 210/210. The first, buggy checkpoint was overwritten
  by the second merge and never used for any reported number
  except this paragraph.

## Errors, verbatim (short)

- First merged run: `merged: 0/210 pass, 164.8s` (see above
  for cause and fix; no other error was encountered).
- Tiny run warnings (harmless, reference kernels only):
  `causal_conv1d_fn is falling back to its reference PyTorch
  implementation`, `chunk_gated_delta_rule is falling back to
  its reference PyTorch implementation`.
- No 403, no quota 503, no reclamation, no account switch
  occurred in this run.

## What is genuinely new vs what was reused

- New: the VEC4 ISA and encoding, the pure-Python reference
  simulator, the circuit itself (8 register heads + OP head,
  per-lane 4-bit adders, per-lane partial-product ANDs with
  adder trees for the dot product, 5 result heads, owner node,
  code-clear + exit code + tau exit), the offset-aware token
  features (with the full=True fix above), the merge driver
  (offset circuit, keep=code+mu+tau, text-row Qwen embeddings,
  chip rows appended after Qwen vocab, final-norm code/tau
  gains), and every test count in this report.
- Reused (upstream chipformer, clone at
  `../upstream`, untouched, inspection only): IR Graph/Target
  and the exactness evaluator, prims gadgets (`t_and`, `t_ge`,
  `add12`-style adders, `slot_head`), emit `allocate`/`Layout`/
  `embedding_rows`/`emit_layers`, `tiny.build` head shapes and
  its final-norm code-gain mechanism, `merge` `widen`/
  `pick_hidden`/`load`, and `mute.mute`. No CHIP-8 BIOS, object
  store, bridge text classes, or tokenizer extension was
  reused — those were replaced with VEC4's own logic, and the
  tokenizer was copied unchanged (chip tokens are addressed
  by id in tests).
- This is NOT a re-download of the author's Qwenputer
  checkpoint, NOT the CHIP-8 or RV32 machine rebranded, and
  no author Qwenputer/QwenPU numbers are presented as VEC4's.
