# Language-probe comparison: base vs merged (v2 models)

Session `vec4-lang`, 2026-11-05 ~00:13 HKT. Tesla T4, account
kyu009009 after kyu008008 refused allocation (one switch); torn
down afterwards, active account restored to kyu008008, training
rotation untouched. Merges in separate processes; Gate 0 re-proven
on both models before the probe ran.

## Method (pre-registered)

The 10 `language_probe` entries in
`vec4-v2/generalisation.json` are plain prompts with no expected
outputs. Driver: `vec4-v2/lang_probe.py`. Conditions identical for
base and merged: plain completion (no chat template), greedy
decoding, max 60 new tokens. The driver records verbatim
continuations plus computed stats only; the verdict is judged from
the verbatim text. This is a sanity comparison, NOT a KL
divergence and NOT a benchmark score.

## Gate 0 re-proof (from the stage0 JSONs in this dir)

- 0.8B-v2 merged: Gate 0 **396/396**, hidden 1280, 42 layers,
  vocab 248576, 1,526,970,752 params, 3.05 GB bf16, VRAM max
  3010.2 MiB. Text sanity: " Paris.\nThe capital of France is
  Paris."
- 2B-v2 merged: Gate 0 **396/396**, hidden 2304, 42 layers,
  vocab 248576, 3,640,992,896 params, 7.28 GB bf16, VRAM max
  7162.3 MiB. Text sanity: " Paris.\nA. True\nB. False"

## Result (recomputed from `lang_08b.json` / `lang_2b.json`)

| | 0.8B | 2B |
|---|---|---|
| Merged matches base verbatim | **7/10** | **8/10** |
| Chip-token leak on merged rows | 0/10 | 0/10 |
| EOS hit (either model) | 0 | 0 |

Chip-token leak: none on any merged row (**20/20 clean**).

Loop flags do NOT separate base from merged — the loop sets are
identical: 0.8B loops on France/Japan/Hello! in both models; 2B
loops on Hello! in both. The loops are base-model greedy
behaviour, not merge damage. No EOS hit anywhere.

Non-verbatim prompts: 0.8B — "Hello!", "Summarise Hamlet in two
sentences.", "What colour do you prefer, red or blue?"; 2B —
"Why don't cats like math?", "What colour do you prefer, red or
blue?". The differing rows are sane except for the one odd row
below.

The one odd row is **0.8B "Hello!"**: base loops "Hello!" twenty
times; merged produces a self-description degrading into a
repeated "10-year history" phrase — English, no chip tokens, no
gibberish.

## Verdict

Merged text tracks base closely under these conditions — sanity
evidence only, NOT a KL, NOT a general "text preserved" claim.
