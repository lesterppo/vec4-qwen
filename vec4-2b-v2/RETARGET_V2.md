# VEC4 v2 retarget to Qwen3.5-2B — analysis BEFORE merging

Written before any 2B-v2 merge, following vec4-2b/RETARGET.md.
The VM merge log is ground truth; anything below marked
"predicted" must be checked against it, not quoted as measured.

## What ports as-is (from vec4-2b, verified for v1)

- Linear heads 16/16, dim 128: 2B linear heads match 0.8B
  (verified in vec4-2b from config.json). DeltaNet register /
  OP / result heads port as-is; the v2 circuit's extra OP
  columns (6 data columns, keys 2..7) are in the OP head, not
  in the base model's heads.
- Intermediate cap 3584 -> 6144 is a cap change only, NOT a
  redesign: v2 circuit max is 154 FFN units/sublayer
  (verified from circuit.py stats), far below both caps, so
  FFN placement is identical. The merge driver reads inter
  from the 2B config for allocate() anyway.
- Vocab 248320 unchanged -> chip offset v0 = 248320
  unchanged; chip vocab 248 (v2) -> merged vocab
  -(-(248320 + 248) // 64) * 64 = 248576, same as 0.8B-v2.
- `full=True` token features kept (the v1 TOKEN_MASK bug
  class: chip tokens at v0 offset must not be masked to
  16 bits, and this now includes OP ids 244..247).
- Full-attention head_dim on 2B is 256 (not 128); mute()
  reads it generically, no code change (vec4-2b lesson).
- In-place widening (from merge_2b.py, identical math to
  M.widen, per-key freeing) is REQUIRED at 2B: stock widening
  would transiently hold base + widened (~11 GB) on a
  12-13 GB VM. merge_2b_v2.py carries it over unchanged.

## What is predicted, not yet measured

- Chip slots for v2 could NOT be computed on this VM (the
  chipformer emit module imports torch, which is not
  installed here). v1 had n_slots 142 on both bases.
- PREDICTED, conditional on slots being in the same range:
  merged hidden 2304 (= 2048 + 256 chip width, as in
  vec4-2b), ballast m=36 (vs 20 on 0.8B), mu slot 2114
  (= 2048 + 36 + 1 + 29, as in vec4-2b). If the v2 slot count
  differs, hidden/ballast/mu will differ — the merge
  driver's printed values are the truth, and Gate 0
  (396/396) is the verification that the widening is correct,
  whatever they are.
- Expected merged size if hidden is 2304: ~3.64B params
  (v1 2B) + v2 circuit delta (dims 583 vs 447, FFN 1357 vs
  961 — small vs the 2B base), ~7.3-7.4 GB bf16, VRAM peak
  ~7.2-7.5 GB on a T4. Still comfortable on a T4, but the
  actual numbers come from the merge log and Gate 0 JSON.

## Gate plan for 2B-v2 (this run)

1. stage0_2b_v2.py: probe, download Qwen3.5-2B, merge via
   merge_2b_v2.py, run_merged.py 396-case regression.
   Gate: 396/396 (any other count stops the run).
2. cache_probe_2b_v2.py: stepwise / generate / cached,
   END-trimmed, 24 cases. Gate: 24/24 on all three, cached
   token-identical to stepwise; else --driver stepwise.
3. tool_2b_v2.py configs, in order, downloaded after each:
   orig (20, tool named), para (12), adv (10), disp (40 rows,
   no model call), plain + notool (5+5). Strict scoring as
   in tool_v2.py; no guard/retry in this driver, so wrong
   tool is a strict failure. The question this gate answers:
   does 2B-v2 fix model tool selection where 0.8B-v2 scored
   para 7/12 and adv 3/10, or does it only fix it on the
   20 same-wording orig cases (as v1 2B did: 20/20)?
4. No tiny test for 2B: tiny.build uses 0.8B head shapes;
   vec4-2b likewise went straight to merged Gate 0.

## Not in this gate

2B-v2 efficiency comparison, language-probe comparison,
VSHIFT/matrix/sorting, any GitHub push.
