# VEC4 → Qwen3.5-2B retarget analysis (Step 1, written before merging)

Gate overridden by the user: the vec4-fix exact-Target gate failed on
intermediate (6144 vs 3584) and hidden (2048 vs 1024). This run
retargets instead of stopping at the gate. Nothing below is assumed:
the circuit numbers were verified locally from `circuit.py` itself,
and the config numbers come from
`../vec4-fix/qwen35_2b_config_check.json` (config.json only, no
weights at that time).

## Base configs (verified)

| Field | Qwen3.5-0.8B | Qwen3.5-2B | Change? |
|---|---|---|---|
| hidden_size | 1024 | 2048 | YES |
| intermediate_size | 3584 | 6144 | YES (cap only, see below) |
| linear_num_key_heads | 16 | 16 | NO |
| linear_num_value_heads | 16 | 16 | NO |
| linear_key_head_dim | 128 | 128 | NO |
| linear_value_head_dim | 128 | 128 | NO |
| num_hidden_layers | 24 | 24 | NO |
| vocab_size | 248320 | 248320 | NO |
| layer_types | 3×linear + 1×full, ×6 | identical pattern | NO |

Caveat: the config-check file also lists `num_attention_heads: 8`
for both models; whether that field is text_config or a top-level
flattening is ambiguous, so NO claim is made here about the 2B
full-attention head_dim. It does not affect the circuit (which uses
only the linear heads above) and `mute()` reads `tc["head_dim"]`
generically. The merge log prints the full downloaded 2B
text_config, which settles it at runtime.

## Circuit (verified locally, not assumed)

Built `circuit.py` locally (pure graph code, no torch needed):
18 chip layers, 447 dims, 14 heads, 961 FFN units,
max **98 units per sublayer** (level 2; full map:
2:98, 4:48, 6:68, 8:80, 10:84, 12:76, 14:82, 16:86, 18:81, 20:58,
22:42, 24:42, 26:34, 28:14, 30:6, 32:6, 34:4, 36:52),
max 9 key heads in one sublayer (level 1: 9 register+OP heads).

Consequences:
- **Intermediate cap 3584 → 6144 is a cap change only, NOT a circuit
  redesign.** FFN placement (`ir._place`) caps units per sublayer at
  the Target intermediate; 98 is below both caps, so placement is
  identical. `emit_layers` shapes the MLP from the base config's
  `intermediate_size` (6144), not from the Target, so the extra rows
  are simply unused zeros. The only circuit.py change is
  `Target(16, 16, 3584)` → `Target(16, 16, 6144)` (record of the host).
- **Linear heads port as-is.** DeltaNet register heads (8), OP head
  (1), result heads (5) use the unchanged 16/16 heads at dim 128;
  `slot_head` shapes are identical. Head-count check: max 9 key
  heads per sublayer ≤ 16.
- **Allocation cap change:** `merge` calls `allocate(c, None, keep,
  inter)` with inter read from the base config, so it automatically
  becomes 6144. Token-feature dims are allocated at level 0 with
  `by_ffn=False`, and the reuse branch for that case requires
  level ≥ 3, so token dims NEVER reuse slots at any cap — their slot
  indices are cap-independent. Later dims may reuse differently at
  a larger cap, but keep dims (code, tau, mu) are never handed over.
  The merge log prints the actual `n_slots` and slot indices; the
  prediction below is checked against it, not trusted.

## Widening arithmetic (computed locally, before/after)

`pick_hidden(base, 1 + n_slots)` = smallest `m * 4^k` (m in 3..64,
multiple of 128) ≥ base + m + n_slots, ties broken to larger m.
Verified replica against the 0.8B case: `pick_hidden(1024, 143)` =
(1280, 20), matching the 0.8B merged model exactly.

| Quantity | 0.8B (before) | 2B (after) |
|---|---|---|
| base hidden d0 | 1024 | 2048 |
| chip slots n_slots | 142 | 142 (predicted; verify from merge log) |
| ballast m | 20 | 36 |
| merged hidden | 1280 | **2304** |
| chip width (hidden − d0) | 256 | **256 (unchanged)** |
| chip region used (m + 1 tau + n_slots) | 163 | 179 |
| unused padding in chip width | 93 | 77 |
| mu slot dim (d0 + m + 1 + slots[mu]) | 1074 (slots[mu] = 29) | **2114 predicted** (slots[mu] = 29, if unchanged as argued; verify from merge log) |
| merged layers (18 chip + 24 Qwen) | 42 | 42 (unchanged) |
| chip offset v0 | 248320 | 248320 (unchanged; vocab unchanged) |
| merged vocab (248320 + 244, padded to ×64) | 248576 | 248576 (unchanged) |
| per-dim params (2·16·128 + 3·16·128 + 3·inter) | 20992 | 28672 |
| text+embed params (hidden·(42·per_dim + vocab)) | 1,446,707,200 | 3,347,251,200 (~6.69 GB bf16) |
| total params / size | 1,526,970,752 / 3.05 GB | TBD from merge log (text+embed + vision tower; 0.8B vision ≈ 80M params ≈ 0.16 GB) |

Robustness check: `pick_hidden(2048, 1 + n)` = (2304, 36) for every
n_slots in 135..150, so the merged hidden does not depend on a
small allocation difference. Scale check: m·B²/hidden =
36·2^80/2304 = 2^74, log4 = 37, an exact power of 4, so
`Layout.scale` asserts cleanly. Mute call is generic:
`mute(sd, tc, n0, lay.dim(mu), hidden**0.5, ...)` with
sqrt(2304) = 48 as the normed ballast.

## What does NOT change

- ISA, token encoding, `ref.py`, circuit graph logic, `slot_head`
  structure, `keep = code + mu + tau`, `full=True` token features
  (required: v0 = 248320 > 65536, same TOKEN_MASK issue as 0.8B —
  the original 0/210 bug class; the 2B copy keeps `full=True`),
  anti-sink skipped (`sinks=None`), tokenizer copied unchanged,
  chip tokens addressed by id, END-trim scoring rule, toolcall
  bridge (host decodes chip OUT, re-injects text tool_response —
  NOT interleaved reporting, per the vec4-untested English barrier).
- `merge_vec4.py` flow is generic in (d0, v0, inter, n0); the 2B
  copy is `merge_2b.py` with exactly two documented deviations:
  (1) Target 6144 in `circuit.py`; (2) in-place widening (same
  math as `M.widen`, per-key freeing — see `merge_2b.py` header;
  reason: RAM on a 12–13 GB VM, stock widening would transiently
  hold base + widened ≈ 11 GB before chip layers).

## Files changed (complete list)

- `circuit.py` (2B copy): Target 3584 → 6144; docstring note. No
  other line of circuit logic touched.
- `merge_2b.py` (2B copy of `merge_vec4.py`): widen call replaced
  by `widen_inplace` (identical math); added prints (full
  text_config, vision_config, slot indices, mu/tau dims, chip
  width). No other logic touched.
- `ref.py`, `run_merged.py`: byte copies of the 0.8B originals,
  unchanged.
- New here: `stage0_2b.py`, `cache_probe_2b.py`, `tool_2b.py`,
  `RETARGET.md`, `RESULT.md` (to be written after the runs).
- Untouched by this run: `../vec4/`, `../vec4-toolcall/`,
  `../vec4-fix/`, `../vec4-untested/`, `../vec4-app/` originals,
  arxiv-lab, MEMORY.md, the rotation file.
