# VEC4 tool-call bridge — RESULT

## Verdict: TOOLCALL_PARTIAL_DOT_TOOL_SELECTION

The bridge works end-to-end for `vec_add` (10/10, clean English, exact chip)
and for plain / should-not-call text (10/10, no tool calls). It does **not**
work end-to-end for `vec_dot` in the tested configuration: the 0.8B model
emits a well-formed tool call in 20/20 cases, but for DOT prompts it picks
the wrong tool in 9/10 cases (calls `vec_add` with the correct DOT args),
so only 1/10 DOT cases give a correct English dot answer. The chip itself
never failed: every chip run that was actually executed matched `ref.py`
exactly. The failure is tool **selection** in Qwen3.5-0.8B, not the VEC4
circuit, not chip exactness, and not text degradation after a tool response.

**Do not use the driver's SUMMARY line.** `toolcall.log` says
`model_emitted=20/20 chip_ok_me=20 final_me=20`. That line is misleading
and is superseded by the strict audit below. The driver verified the tool
actually called, not the tool requested: when `args_match_case` was False
(wrong function name), `chip_ok` checked the ADD chip run, and
`final_contains` fell back to checking the ADD result string — which
appears as parameter `a` in the model's second tool call, so it was
trivially True. The strict audit counts a case only if the called tool,
the args, the chip op, and the final English text all match the
**requested** operation. Strict end-to-end, main run: **11/20**
(ADD 10/10, DOT 1/10). Host-injected fallback was never triggered
(model emitted 20/20), so host-injected end-to-end is 0 cases tested,
not 0 failures — the fallback code exists in `toolcall.py` but was
not exercised.

## Hardware (Gate 0 probe, session vec4-toolcall, account cyc236es)

- GPU: Tesla T4, 15,360 MiB (torch-reported 14,912 MiB)
- CPU: Intel Xeon @ 2.00 GHz, 2 CPUs; RAM 12 GB total, 11 GB available
- Disk: 113 GB free on /content
- Python 3.13.16, torch 2.11.0+cu130, transformers 5.22.0, CUDA available
- Merged checkpoint: base Qwen/Qwen3.5-0.8B (ungated), hidden 1280,
  42 layers (18 chip + 24 Qwen), vocab 248576, 1,526,970,752 params,
  3.05 GB bf16, chip offset v0=248320

## Gate 0 — rebuild regression: 210/210

Rebuilt merged VEC4 exactly as before from `merge_vec4.py` + `circuit.py`
+ base download. Chip-only test set from `run_merged.py` (100 random each
+ edges, seed 20261103): **210/210**, same as the previous session.
Stage 0: 184.6 s, VRAM at load 2,962.8 MiB, peak 3,010.2 MiB.
Text sanity again: "The capital of France is" → " Paris.\nThe capital of
France is Paris."

## Gate 1 — tools and driver

- Tools (Qwen XML format via `apply_chat_template`, tools=…, 
  `enable_thinking=False`): `vec_add(a, b)` — lane-wise mod 16;
  `vec_dot(a, b)` — sum of lane products, max 900. Each `a`/`b` is
  4 integers 0–15. System prompt: "You have tools vec_add and vec_dot
  for exact vector arithmetic. Use them for every vector calculation
  instead of calculating by hand."
- Driver: stepwise greedy forward (`next_token`, full-sequence forward
  each step, as in `run_merged.py`). Tool phase cap 120 tokens,
  continuation cap 120 tokens, plain/no-tool cap 100 tokens, chip run
  max 6 tokens to END. EOS ids [248044, 248046].
- Tool-response segment (hand-built, tokenized with
  `add_special_tokens=False`):
  `<|im_end|>\n<|im_start|>user\n<tool_response>\n<result>\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
  Tool-response content is `str(dot)` for DOT, `str([lane digits])` for ADD.
- Parser: XML first (`<function=…>`, `<parameter=…>`); JSON-style calls
  recorded separately as format=json, never relabelled as XML. All
  20 main-run calls were XML; none were JSON. Validation: name must be
  `vec_add`/`vec_dot`, `a`/`b` must be 4 ints 0–15; otherwise a parse
  failure (0 in the main run).
- Important scope note on "mixed sequences": chip runs in this driver
  are **standalone chip-token sequences** triggered by a tool call, not
  chip tokens appended into the text token sequence. What was tested is
  conversation-level mixing — English prompt → tool_call → chip run →
  tool_response → English answer, all in the same merged model — not
  token-interleaved text+chip in one sequence. Token-interleaved
  mixed sequences remain untested.

## Gate 2 — mixed-sequence tests, main run (strict audit)

Tool cases: 20 total — DOT edges (all 0, all 15, [3,5,7,9]·[2,4,6,8]=140,
[0,15,0,15]·[15,0,15,0]=0) + 6 random DOT, ADD edges + 6 random ADD,
`random.Random(20261103)`. Note: this RNG sequence differs from
`run_merged.py` (which interleaved ADD then DOT); the tool random cases
are **not** identical to the chip-only random cases.
DOT prompts: "What is the dot product of {A} and {B}? Use vec_dot."
ADD prompts: "Use vec_add to add {A} and {B} lane by lane, modulo 16."

### Emission by tier (main run)

- Model-emitted tool calls: **20/20**, all XML, all with valid args
  (4 ints 0–15, and in every case the parsed `a`/`b` equalled the case
  `A`/`B` — only the function name was ever wrong).
- Host-injected tool calls: **0/20** (fallback never triggered, because
  emission was 20/20; the <50% fallback condition did not occur).
- Parse failures: 0/20.

### Correct tool + correct args (main run)

- ADD: **10/10** called `vec_add` with the case args.
- DOT: **1/10** called `vec_dot` with the case args — only case 1,
  [15,15,15,15]·[15,15,15,15] = 900.
- DOT wrong tool: **9/10** called `vec_add` with the correct DOT case
  args (cases 0, 2–9). This is systematic, not a single slip.

### Chip exactness

- Chip-only regression: **210/210** (see Gate 0).
- Mixed-conversation chip runs, for the tool actually called:
  main **20/20** (19 ADD runs + 1 DOT run), diagnostic **10/10** —
  30/30 called-op runs matched `ref.py` exactly.
- Chip exactness for the **requested** op: ADD **10/10**; DOT main
  **1/1** (only one DOT chip actually ran, and it was exact);
  DOT diagnostic **4/4** (see below). No chip run ever mismatched.
- Anti-sink / mixed-sequence note: no chip failure was observed in any
  run, so no mute/sink fix was attempted — there was nothing to fix at
  the circuit level. The post-tool_response failure below is a
  tool-loop behaviour, not degraded/garbled text and not a chip error.

### Final English answer with the correct requested result (strict)

Strict = final text contains the correct requested result **and**
contains no `<tool_call>` block.

- ADD: **10/10** clean English. Examples: "The result of adding
  [15, 15, 15, 15] and [15, 15, 15, 15] lane by lane, modulo 16, is
  [14, 14, 14, 14]."; "The result is [0, 0, 0, 0]."
- DOT: **1/10** — only case 1: "The dot product of [15, 15, 15, 15]
  and [15, 15, 15, 15] is 900."
- DOT failures (9/10): after the ADD tool_response, the model emits a
  second tool call, then EOS, with no English answer and no DOT number.
  Verbatim pattern (case 2, requested dot 140):
  `<tool_call><function=vec_dot><parameter=a>[5, 9, 13, 1]</parameter><parameter=b>[2, 4, 6, 8]</parameter></function></tool_call>`
  where `[5, 9, 13, 1]` is exactly the ADD result (A+B mod 16) for the
  case args. In a real tool loop, that second call would execute a DOT
  with wrong args; a deployment must treat a second tool call after a
  tool_response as a failure, not as an answer. The driver's loose
  `final_contains=True` for these cases counted the ADD result string
  inside that second call — that is **not** an English dot answer.
- Overall strict end-to-end, main run: **11/20**.

### Diagnostic: tool-order swap (DOT only, same 10 DOT cases)

One small diagnostic was run in the same session, changing only the
tool list order to `vec_dot` first (`diag_dot_order.py`, strict scoring
added). It is **not** a replacement for the main run and is reported
separately.

- DOT correct tool + correct args: **4/10** (cases 0, 1, 3, 6) vs
  1/10 in the main run.
- DOT strict English final: **4/10**, same cases. Examples:
  "The dot product of [0, 0, 0, 0] and [0, 0, 0, 0] is 0.";
  "The dot product of [14, 10, 14, 6] and [7, 3, 0, 2] is **140**."
  (140 = 98+30+0+12, correct.)
- DOT still wrong tool: **6/10** (cases 2, 4, 5, 7, 8, 9), all again
  `vec_add` then the same second-`vec_dot` loop. The task's example
  case, [3,5,7,9]·[2,4,6,8]=140 (case 2), failed in **both**
  configurations.
- Conclusion from the diagnostic: tool order helps (1/10 → 4/10) but
  does not fix general DOT selection in the 0.8B model. It is not a
  pure first-tool bias: case 6 flipped to correct while cases 2, 4, 5,
  7, 8, 9 did not, under identical greedy decoding.

### Plain text (no tool expected): 5/5 no tool call, no chip tokens

- "What is the capital of France?" → "The capital of France is Paris."
- "Tell me a short joke about cats." → "I'm sorry, I can't tell a joke
  about cats." (refusal, still sane, no tool)
- "Explain what a vector is in one sentence." → correct one-sentence
  definition (magnitude + direction).
- "What is the capital of Japan?" → "The capital of Japan is **Tokyo**."
- "Write a one-line greeting." → "Greetings!"

### Should-not-call prompts: 5/5 no tool call, false-trigger rate 0/5

- "What is 2+2 in words?" → "2+2 in words is **two plus two**."
  (does not say "four", but no tool call — a wording quirk, not a
  tool failure)
- "Explain what a dot product is in one sentence, without calculating
  anything." → correct definition, no calculation, no tool.
- "What is your favourite colour?" → "I don't have a personal
  preference, but I can help you find the right color!"
- "Summarise the plot of Hamlet in one sentence." → one sentence, but
  with a garbled detail ("a young man who has just been executed" —
  Hamlet is not executed). No tool call; the garble is 0.8B text
  quality, not a bridge failure.
- "Is Paris the capital of France? Answer yes or no." → "Yes"

## Timing and VRAM

- Gate 0 (probe + rebuild + 210-case regression): 184.6 s;
  VRAM max 3,010.2 MiB.
- Main tool run (20 tool + 5 plain + 5 no-tool, model load included):
  2,313.8 s (38.6 min); VRAM max 3,337.0 MiB.
- Diagnostic DOT run (10 DOT cases, model load included): 1,093.7 s
  (18.2 min); VRAM max 3,330.8 MiB.
- Per-case cost is high because the driver does a full-sequence forward
  per token with reference (non-flash) DeltaNet kernels; see errors
  below. Roughly 2 min per tool case.

## Errors, verbatim (short)

- Transformers warnings in every run (not failures, but the speed
  cause): "`causal_conv1d_fn` is falling back to its reference PyTorch
  implementation because `causal_conv1d` is not installed. This is
  correct but much slower" and the same for `chunk_gated_delta_rule` /
  `flash-linear-attention`.
- Development bug, fixed before the main run: first toolcall attempt
  died with `TypeError: log() got an unexpected keyword argument
  'flush'` after model load; fixed `log` to accept `**kw` and re-ran.
- Development bug, fixed before the diagnostic: first diagnostic
  attempt died with `UnboundLocalError: cannot access local variable
  'final'` (strict fields inserted before `final` was defined);
  fixed placement and re-ran.
- No 403, quota, or reclamation event in this session. No GitHub push,
  no Threads post, no arxiv-lab or MEMORY.md change.

## What is new vs reused

- Reused, unchanged in design: the VEC4 circuit (`circuit.py`,
  `merge_vec4.py`, `ref.py`, copied from `../vec4/`, originals
  untouched) — 4-lane SIMD, VADD mod 16, VDOT max 900, DeltaNet state
  registers, chipformer merge into Qwen3.5-0.8B — and Qwenputer's
  tool-call idea (the model writes a `<tool_call>` block, the compiled
  layers run the machine, the model continues after a
  `<tool_response>`).
- New in this task: the two Qwen tool definitions, the stepwise-greedy
  tool-call driver (chat template with tools, XML parsing with
  format recorded separately, chip-token translation, tool-response
  injection, same-model English continuation), the host-injected
  fallback code (written, not exercised), the strict requested-op
  audit that caught the driver's loose scoring, and the tool-order
  diagnostic. The bridge makes VEC4 chat-usable for ADD and for
  non-vector text; for DOT it exposes a 0.8B tool-selection limit that
  a bigger or better tool-tuned model, a single-tool-per-prompt setup,
  or a host-side tool-name guard would need to address — the circuit
  itself is not the blocker.

## Files

- `toolcall.py` — main driver (loose SUMMARY line inside it is
  superseded by the strict audit in this report)
- `toolcall_result.json`, `toolcall.log` — main run raw logs
- `diag_dot_order.py`, `diag_result.json`, `diag.log` — diagnostic
- `stage0.py`, `stage0_done.json`, `vec4_merged_result.json` — Gate 0
- `circuit.py`, `merge_vec4.py`, `ref.py` — copies of VEC4 originals
