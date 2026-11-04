# VEC4 DOT fix comparison — RESULT

## Verdict: FIXED_BY_A_AND_B1

Both tested fixes fix DOT completely on the same 10 cases, with the same
merged 0.8B model and the same chip. Neither needed a bigger model.

- **Variant A (single-tool-per-prompt): DOT 10/10, ADD 10/10, strict 20/20.**
- **Variant B (host-side tool-name guard + retry, both tools listed):
  DOT B1 10/10 — 1 first-attempt + 9 model-corrected on retry, 0 host-rewritten.**
- **Variant C (Qwen3.5-2B): NOT RUN.** The exact-Target config gate failed
  (intermediate 6144 vs 3584, hidden 2048 vs 1024), so per task rules the
  circuit was not retargeted and no 2B merge was attempted. C is not
  refuted — it is untested, and it is also unnecessary for DOT, since
  A and B1 already fix it at 0.8B.

## Scoring (strict, all variants — read this before the table)

Strict = requested op == called tool AND args == case args AND chip run
for the requested op matches `ref.py` AND final English contains the
correct requested result AND final raw contains no `<tool_call>` block.
Host-rewritten calls (B2) are NEVER counted as model successes (B1).
Loose "tool actually called" numbers are not end-to-end results and are
not reported as such anywhere below.

## Strict comparison table

| Configuration | DOT cases | DOT correct tool | DOT chip (requested op) | DOT strict final | Overall strict | Time | VRAM max |
|---|---|---|---|---|---|---|---|
| Baseline main run, both tools, vec_add first (vec4-toolcall) | 10 | 1/10 | 1/1 (only 1 DOT chip ran) | 1/10 | 11/20 (ADD 10/10) | 2,313.8 s for 30 cases | 3,337.0 MiB |
| Diagnostic, both tools, vec_dot first (vec4-toolcall) | 10 | 4/10 | 4/4 | 4/10 | — (DOT only) | 1,093.7 s for 10 cases | 3,330.8 MiB |
| **Variant A: single tool listed** | 10 | 10/10 | 10/10 | **10/10** | **20/20** (ADD 10/10) | 1,399.5 s for 20 cases (~70 s/case, incl. load) | 3,262.0 MiB |
| **Variant B1: guard + retry, model-only** | 10 | 1/10 first attempt, 10/10 after ≤2 attempts | 10/10 | **10/10** | — (DOT only) | 1,393.5 s for 10 cases (~139 s/case, incl. load) | 3,363.6 MiB |
| Variant B2: host-rewritten fallback | 10 | — | — | **0 cases exercised** | — | — | — |
| Variant C: Qwen3.5-2B merge | — | — | — | **NOT RUN (gate fail)** | — | — | — |

Cases are the exact 20 from the toolcall main run (verified 20/20 against
`toolcall_result.json` before the run): DOT edges all-0, all-15,
[3,5,7,9]·[2,4,6,8]=140, [0,15,0,15]·[15,0,15,0]=0, + 6 random
(`Random(20261103)`); ADD edges + 6 random, same seed sequence.
The task's example case, [3,5,7,9]·[2,4,6,8]=140, passes in BOTH A and B1
after failing in both baseline configurations.

## Gate 0 — rebuild regression: 210/210 (twice)

Two T4 sessions were used (see session note below); each rebuilt the
merged VEC4 0.8B exactly as before and passed the chip-only regression:

- Session 1 (vec4-fix, T4): hidden 1280, 42 layers, vocab 248576,
  1,526,970,752 params, 3.05 GB bf16, chip offset v0=248320,
  regression 210/210 in 127.5 s generation, VRAM max 3,010.18 MiB.
- Session 2 (vec4-fix-b, T4): same checkpoint spec, regression 210/210
  in 138.1 s generation, VRAM max 3,010.18 MiB.
- Text sanity in both: "The capital of France is" ->
  " Paris.\nThe capital of France is Paris."

## Variant A — single-tool-per-prompt: 20/20 strict

Method: for each case, only the requested tool was listed in the chat
template, and the system prompt named only that tool (e.g. "You have
tool vec_dot for exact vector arithmetic."). Everything else was the
same driver: stepwise greedy forward, XML tool-call parsing, chip
translation, tool_response injection, same-model English continuation.

- DOT: correct tool + correct args 10/10, chip requested 10/10,
  strict final 10/10, parse failures 0/20, emission 20/20.
- ADD: correct tool + correct args 10/10, chip requested 10/10,
  strict final 10/10. No ADD regression from listing one tool.
- Examples: "The dot product of [3, 5, 7, 9] and [2, 4, 6, 8] is **140**.";
  "The dot product of [15, 15, 15, 15] and [15, 15, 15, 15] is **900**."
- No case emitted the unlisted tool. Removing the wrong choice removes
  the failure mode entirely, for this model and this case set.

## Variant B — host-side tool-name guard + retry: B1 10/10, B2 unexercised

Method: both tools listed, main-run order (vec_add first), same system
prompt as the baseline. If the model called the wrong tool, the host did
NOT execute the chip. It injected an error tool_response —
"Error: wrong tool. The task requires vec_dot, not vec_add. Call vec_dot
with the same arguments." — and let the model generate again, max 2
attempts per case. Only if both attempts failed would the host rewrite
the call to the correct tool (B2, reported separately, never counted
as B1).

- First-attempt correct tool: 1/10 — exactly the baseline rate (only the
  all-15 = 900 case). The guard, not random variation, is what changed
  the other 9, because the first-attempt distribution reproduced the
  baseline failure (9/10 vec_add with correct args).
- Retry-corrected: 9/9 wrong-first-attempt cases called vec_dot with the
  correct args on attempt 2, executed the chip exactly, and finished
  with correct English. Attempts total: 19 (1 + 9×2).
- B1 strict (model-only, no rewrite): 10/10. Chip requested: 10/10.
  Second-call loop after the correct tool_response: 0/10 — notably,
  the loop seen in the baseline (ADD response → second vec_dot call →
  EOS) did NOT occur after a corrected DOT response in any B case.
- B2 strict: 0 cases exercised, NOT 0/10 failures. The host-rewrite
  fallback code exists in `variants.py` (template-rendered full
  conversation, with a hand-built fallback) but was never triggered,
  because every case was corrected by the model on retry. B2 remains
  untested and must not be claimed as working.
- Cost vs A: B costs ~2× per DOT case (~139 s vs ~70 s) because 9/10
  cases pay for a second generation, and it keeps both tools listed —
  which is exactly the configuration where the failure occurs. B is a
  safety net, not a free fix.

## Variant C — config gate: FAIL, skipped, no retarget, no merge

Source: `huggingface_hub` `config.json` only for Qwen/Qwen3.5-2B and
Qwen/Qwen3.5-0.8B; no 2B weights were downloaded. Full JSON:
`qwen35_2b_config_check.json`.

- VEC4 Target: (linear_num_key_heads=16, linear_num_value_heads=16,
  intermediate=3584).
- Qwen3.5-0.8B: key 16, value 16, key dim 128, value dim 128,
  intermediate 3584, hidden 1024, layers 24, vocab 248320,
  layer_types identical pattern to 2B.
- Qwen3.5-2B: key 16, value 16, key dim 128, value dim 128,
  intermediate **6144**, hidden **2048**, layers 24, vocab 248320.
- Gate verdict: linear heads and head dims MATCH, but intermediate
  6144 != 3584 and hidden 2048 != 1024, so the exact-Target gate FAILS.
  Per task rules, Variant C was not run.
- Future-work note (not done, not claimed): the circuit's max sublayer
  usage is only 98 units, far below 6144, so a retargeted 2B merge is
  plausible in a separate run — but it would need a new Target, a new
  allocation, and full re-verification (210-case regression + tool
  cases), and it is unnecessary for DOT now that A and B1 fix it.

## Which fix actually fixes DOT, and at what cost

- **A fixes DOT at zero runtime cost** — in fact it is the cheapest
  configuration tested (~70 s/case vs ~139 s/case for B, ~77 s/case
  for the baseline 30-case run). Its cost is a design constraint, not
  compute: the host must know (or route to) the operation before
  prompting, so it fits prompts where the operation is explicit
  ("dot product", "add lane by lane") and fits a router that picks one
  tool per request. It does not fix multi-tool selection in general.
- **B1 fixes DOT without that constraint**, at ~2× latency on DOT cases
  and with a host guard in the loop. It is the right safety net when
  several tools must be listed: it never executes a wrong tool, it
  gives the model exactly one correction chance with a named error,
  and in this run the model corrected itself 9/9 times. Its weakness
  is that B2 (host rewrite after two failures) is still untested.
- **C is not needed for DOT.** A bigger model might also fix tool
  selection, but it would cost a retarget, a larger checkpoint, and
  more VRAM, for a problem two driver-level fixes already solve.

## Recommendation for a real deployment

Ship **A + B combined**: route each request to a single tool when the
operation is known from the prompt (A — the cheapest, cleanest path),
and keep the guard + retry (B1) active whenever more than one tool is
listed, with the wrong-tool chip execution blocked and a second tool
call after a tool_response treated as a failure, as in the baseline
audit. Keep the B2 host-rewrite path in code but labelled as untested
until a case actually exercises it (e.g. an adversarial prompt where
the model fails twice). Do not ship a 2B merge for this problem unless
a future case set shows A+B failing — and then retarget and re-verify
from scratch, rather than assuming the 0.8B circuit ports.

## Still not fixed / not tested

- Token-interleaved text+chip sequences remain untested (same scope
  caveat as vec4-toolcall: conversation-level mixing was tested, not
  chip tokens interleaved inside one text sequence).
- B2 host-rewrite fallback: code written, 0 cases exercised.
- Speed: ~70 s/case (A) to ~139 s/case (B) with stepwise full-sequence
  forwards and reference DeltaNet kernels — usable for a demo, not
  for interactive chat, until the driver uses KV caching or the
  flash kernels.
- Anti-sink: no chip failure was observed in any run, so no mute/sink
  fix was attempted; the second-call loop in the baseline was a
  tool-loop behaviour, and it did not recur in B.

## Session note (honest record — two T4 sessions, not one)

The task asked for one session if possible; reclamation made that
impossible, and the record is:

1. A first session created without `--gpu` was CPU-only
   (torch 2.11.0+cpu, no GPU) and failed Gate 0 at `.to(cuda)` with
   "Torch not compiled with CUDA enabled". It was stopped, no variant
   ran on it, and a T4 session was created with the same account.
2. T4 session `vec4-fix` (account cyc236es): Gate 0 210/210, Variant A
   20/20, Variant A JSON downloaded at 15:03 HKT. At 15:14 HKT the VM
   contents API returned 404 — free-tier reclamation, ~42 min after
   session creation, during Variant B (whose script only wrote its
   JSON at the end, so B partial results from that session were lost).
3. T4 session `vec4-fix-b` (same account cyc236es, new registry):
   Gate 0 210/210 again, Variant B run with per-case incremental writes
   (`variant_b_partial.json` downloaded at 4/10 and 7/10 cases as
   safeguards), completed 10/10 B1, final JSON downloaded, session
   stopped. No 403, no quota error, no account switch at any point.
   Rotation file untouched (2026-11-03 cyc236es), active account
   restored/unchanged (token.json -> token.cyc236es.json).
   One driver bug fixed before the B rerun's first run: `log()` did not
   accept `flush=True` (same bug class as the toolcall run); the model
   had not loaded yet, so no variant data was lost to it.

## Errors, verbatim (short)

- CPU session: `AssertionError: Torch not compiled with CUDA enabled`
  at `.to(cuda)` in `run_merged.py` (session-type error, not a model
  error).
- First variants attempt: `TypeError: log() got an unexpected keyword
  argument 'flush'` at model-load logging, before any case ran; fixed
  `log(*a, **kw)` and reran.
- Reference-kernel warnings in every run (speed cause, not failures):
  "`causal_conv1d_fn` is falling back to its reference PyTorch
  implementation" and the same for `chunk_gated_delta_rule` /
  `flash-linear-attention`.
- No GitHub push, no Threads post, no arxiv-lab or MEMORY.md change,
  `../vec4/` and `../vec4-toolcall/` originals untouched.

## Files

- `RESULT.md` — this report
- `stage0_fix.py`, `variants.py`, `run_variant_b.py` — drivers
  (variants.py writes `variant_b_partial.json` after every B case)
- `variant_a_result.json`, `variant_b_result.json`,
  `variant_b_partial.json` — raw results
- `stage0_done.json`, `stage0_done_b.json`, `vec4_merged_result.json`
  — Gate 0 records (both sessions 210/210; local merged result file is
  from the first T4 session, second session's regression was confirmed
  in its run log: 210/210, VRAM max 3,010.18 MiB)
- `qwen35_2b_config_check.json` — Variant C config gate evidence
- `circuit.py`, `merge_vec4.py`, `ref.py`, `run_merged.py` — copies of
  VEC4 / toolcall originals (originals untouched)
