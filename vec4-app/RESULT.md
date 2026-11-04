# VEC4 on a real application: quantized film recommendation — RESULT

## Verdict: APP_PARTIAL_ORCHESTRATION_RANKING

The chip is exact everywhere and the application workflow is computable,
but the 0.8B model cannot run it end-to-end as a recommender: model
orchestration fails on every scenario, and no mode ranks the top 3
correctly on any non-trivial scenario. This is a model-orchestration and
English-summary failure, NOT a chip failure — do not let the chip's
90/90 exactness be quoted as application success.

- **Chip exactness: 90/90 in Mode 1 and 90/90 in Mode 2** (1 ADD + 8 DOT
  per scenario × 10 scenarios), END-trimmed, against pre-registered
  Python ground truth. The compiled VEC4 unit does the application's
  arithmetic perfectly.
- **Mode 1 (model-orchestrated): workflow clean 0/10.** The model calls
  `vec_add` for every DOT step (160/160 DOT attempts), never `vec_dot`
  in any attempt, so all 80 DOT steps in all 10 scenarios were rescued
  by B2 host-rewrite (counted separately, never as model success).
  Top-1 4/10 overall, 3/9 excluding the trivial all-zero scenario;
  top-3 order 0/9 non-trivial; top score mentioned 5/9 non-trivial.
- **Mode 2 (application-orchestrated): top-1 4/10 overall, 3/9
  non-trivial; top-3 order 0/9 non-trivial; top score mentioned 0/9
  non-trivial.** Host orchestration removes the tool-selection failure
  and is 5.8× faster than Mode 1, but the 0.8B summary still cannot
  rank: its finals are rambling step-by-step recalculations that hit
  the 250-token cap before concluding (5/10 finals name no film at
  all).
- **Control (text-only, no tools): top-1 0/9 non-trivial, top-3 0/9,
  scores 0/9.** The chip modes beat text-only mental arithmetic on
  top-1 (3/9 vs 0/9), so the chip adds something — but 3/9 is not a
  shippable recommender.
- **No scenario in any mode was fully correct** (clean workflow +
  correct top-1 + correct top-3 order + correct scores). The best
  single case (Mode 1, scenario 8) still had B2-rescued DOT steps and
  shuffled scores (see worked example).

## Per-mode table

Scoring (from `dataset.json` ground truth, pre-registered before any
model run; an independent local rescoring of the raw finals matched the
on-VM driver on 30/30 scenarios):

- chip exactness: all executed chips match `ref.py` (ADD mod 16, DOT
  sum of products).
- workflow completion: Mode 1 "clean" = every step called correctly by
  the model within 2 attempts, no B2 step, no extra tool call in the
  final phase. Mode 2 workflow is host-driven by construction.
- top-1: first film named in the final answer is a ground-truth
  winner (any tied first-place film counts — tie rule below).
- top-3 order: first three distinct films named have non-increasing
  ground-truth scores and no film outside them scores strictly above
  the smallest of the three.
- score accuracy: top score mentioned as a standalone number; top-3
  scores = all three top scores mentioned.
- Scenario 1 (all-zero) is trivial on every metric (all scores 0, any
  film is a winner, any "0" mentions the score), so non-trivial rates
  (excluding scenario 1) are given alongside raw rates. Never quote
  the raw rates without the non-trivial ones.

| Mode | Scenarios | Chip exactness | Workflow clean | Top-1 | Top-3 order | Top score | Top-3 scores | Median s/scenario | VRAM max |
|---|---|---|---|---|---|---|---|---|---|
| 1 model-orchestrated | 10 | 90/90 chips (10/10 scen) | 0/10 | 4/10 (3/9 non-trivial) | 1/10 (0/9 non-trivial) | 6/10 (5/9) | 3/10 (2/9) | 166.4 | — |
| 2 app-orchestrated | 10 | 90/90 chips (10/10 scen) | host-driven (no model calls) | 4/10 (3/9) | 0/10 (0/9) | 1/10 (0/9) | 1/10 (0/9) | 28.7 | — |
| Control text-only | 10 | n/a (no chips) | n/a | 1/10 (0/9) | 1/10 (0/9) | 1/10 (0/9) | 1/10 (0/9) | 27.1 | — |
| Gate 0 (both sessions) | 210 chip-only | 210/210, twice | — | — | — | — | — | 120.5 / 125.6 s gen | 3,010.2 MiB |

VRAM: Gate 0 peak 3,010.2 MiB in both sessions. The application process
peak (one process running all three modes in session 2) was 4,465.3
MiB — the long Mode 1 sequences plus KV cache, not a per-mode figure,
so it is not split across modes above. Mode 1 scenario time sums to
1,676.6 s across both sessions; Mode 2 sums to 287.8 s; control sums
to 270.7 s. B2 is never counted as B1 anywhere: Mode 1's 80 B2 steps
are reported as rescues, and its workflow-clean count is 0.

## Application and dataset

Films (attributes in order [action, comedy, drama, sci-fi], each
0–15), in `dataset.json`:

| Film | Vector | Character |
|---|---|---|
| Starfall Squadron | [15, 2, 3, 14] | action + sci-fi, near-max lanes |
| Laugh Riot | [3, 15, 2, 1] | comedy-heavy |
| The Long Winter | [2, 3, 15, 1] | drama-heavy |
| Neon Rebellion | [13, 4, 5, 13] | action + sci-fi |
| Family Picnic | [1, 14, 6, 0] | comedy-heavy |
| Quiet Harbor | [1, 2, 14, 2] | drama-heavy |
| Galaxy Heist | [12, 8, 4, 12] | action/comedy/sci-fi mix |
| Static | [0, 0, 1, 0] | near-zero |

Scenarios (10): two profiles each, combined = lane-wise sum mod 16,
then DOT against all 8 films, rank descending. Ground-truth winners
spread over 4 outright winners (Starfall Squadron ×3, Neon Rebellion
×2, Galaxy Heist ×1, The Long Winter ×1) plus ties, satisfying the
≥4-film spread requirement.

| Scenario | Kind | Profile A | Profile B | Combined | Winner(s) | Top score |
|---|---|---|---|---|---|---|
| 0 | tie_first (designed) | [13, 5, 14, 9] | [3, 3, 4, 7] | [0, 8, 2, 0] | Laugh Riot = Family Picnic (124) | 124 |
| 1 | all_zero | [0, 0, 0, 0] | [0, 0, 0, 0] | [0, 0, 0, 0] | all 8 tied | 0 |
| 2 | near_max | [15, 15, 15, 15] | [15, 14, 15, 13] | [14, 13, 14, 12] | Galaxy Heist | 472 |
| 3 | random (seed 20261104) | [6, 10, 11, 15] | [0, 7, 12, 0] | [6, 1, 7, 15] | Starfall Squadron | 323 |
| 4 | random | [2, 14, 12, 0] | [14, 5, 15, 7] | [0, 3, 11, 7] | The Long Winter | 181 |
| 5 | random | [3, 9, 9, 6] | [3, 10, 15, 15] | [6, 3, 8, 5] | Neon Rebellion | 195 |
| 6 | random | [12, 14, 10, 5] | [13, 6, 4, 7] | [9, 4, 14, 12] | Neon Rebellion | 359 |
| 7 | random (accidental tie) | [1, 10, 15, 0] | [11, 15, 12, 13] | [12, 9, 11, 13] | Galaxy Heist = Neon Rebellion (416) | 416 |
| 8 | random | [6, 7, 0, 14] | [7, 9, 13, 1] | [13, 0, 13, 15] | Starfall Squadron | 444 |
| 9 | random | [3, 3, 10, 13] | [4, 0, 8, 11] | [7, 3, 2, 8] | Starfall Squadron | 229 |

Tie rule (stated before running, in `make_dataset.py` and here):
top-1 counts as correct if the model names ANY film tied for first
place; top-3 order counts as correct if the first three distinct films
named have non-increasing ground-truth scores and no film outside
them scores strictly above the smallest of the three (so a tie at the
3rd-place boundary may be resolved either way). Scenario 7's tie was
not designed — it fell out of `Random(20261104)` — and is scored by
the same rule. Scenario 1's "correct" marks are trivial (see table
note) and are never presented as model skill.

## Worked example — Mode 1, scenario 8 (the best case, still flawed)

- Profiles: mine [6, 7, 0, 14], friend's [7, 9, 13, 1].
- Combined (ADD mod 16, chip-exact): [13, 0, 13, 15].
- Ground-truth scores (all 8, chip-exact via B2-rescued DOTs):
  Starfall Squadron 444, Neon Rebellion 429, Galaxy Heist 388,
  The Long Winter 236, Quiet Harbor 225, Family Picnic 91,
  Laugh Riot 80, Static 13. Ranking: Starfall > Neon > Galaxy > …
- Workflow: ADD step correct on first attempt; all 8 DOT steps failed
  both model attempts (vec_add both times) and were B2-rewritten;
  the first final phase was itself a tool call, so one error
  injection preceded the English final.
- Model's final answer, verbatim:

> Based on the calculations:
>
> 1.  **Combined Taste Profile**: `[13, 0, 13, 15]`
> 2.  **Film Scores**:
>     *   Starfall Squadron: 429
>     *   Laugh Riot: 91
>     *   The Long Winter: 225
>     *   Neon Rebellion: 80
>     *   Family Picnic: 236
>     *   Quiet Harbor: 388
>     *   Galaxy Heist: 13
>     *   Static: 429
> 3.  **Top Film**: **Starfall Squadron** with a score of **429**.
> 4.  **Top 3 Films**:
>     1.  **Quiet Harbor** (388)
>     2.  **Starfall Squadron** (429)
>     3.  **Family Picnic** (236)

- Scoring of this answer: top-1 correct (Starfall Squadron is the
  true winner, though its score is mis-stated as 429, which is
  Neon's score); top-3 order wrong (Quiet Harbor first, and the
  three scores are not the true top three in order); top score (444)
  not mentioned; top-3 scores not mentioned. The model reproduced
  almost the correct SET of numbers (429, 91, 225, 80, 236, 388, 13
  — with 444 missing and 429 duplicated for Static) but attached
  them to the wrong films: no listed score belongs to the film it
  is listed under (Starfall's 429 is Neon's, Laugh Riot's 91 is
  Family Picnic's, The Long Winter's 225 is Quiet Harbor's, and so
  on — a derangement of the true scores). The numbers came from
  the tool responses; the film↔score binding did not survive into
  English. This score-shuffling is the characteristic Mode 1
  failure, not a chip error: the chips returned 444, 429, 388, …
  correctly.

## Failure patterns

### Mode 1 — vec_add stickiness after the ADD step (the dominant failure)

- ADD step: `vec_add` with correct profiles, first attempt, 10/10.
- DOT steps: `vec_add` in 160/160 model attempts (2 attempts × 8
  steps × 10 scenarios), `vec_dot` in 0/160. The guard error named
  the required tool (`vec_dot`) on every retry and never corrected
  the model — the opposite of vec4-fix Variant B, where a single
  DOT retry corrected 9/9 cases. The difference is context: after
  the model has just called `vec_add` once, in a long multi-step
  conversation, it stays on `vec_add` for every subsequent step.
  Guard + retry does not break this stickiness; only B2 host-rewrite
  (80/80 DOT steps) kept the workflow moving.
- DOT attempt arguments also drift: only 14/160 DOT attempts had
  fully correct arguments (combined vector + that step's film
  vector). The `b` argument collapses onto Static [0, 0, 1, 0] in
  50/160 attempts (the last film in the list, repeated in later
  attempts after the model cycles through the films). So even a
  hypothetical "execute the model's vec_add calls" fallback would
  compute the wrong things: wrong tool AND mostly wrong arguments.
- Final phase: the first final generation was a tool call in 10/10
  scenarios (a further `vec_add` attempt, e.g. scenario 6's final
  phase called `vec_add` with the combined vector and Laugh Riot's
  vector). After one error injection, 7/10 produced English finals
  and 3/10 (scenarios 2, 6, 9) still produced only a tool call, so
  those three finals name no film at all.
- Total Mode 1 tool calls: 170 model calls (17 per scenario: 1 ADD +
  16 failed DOT attempts) + 80 B2 rewrites. The 17-call count was
  identical in every scenario — the failure is systematic, not
  scenario-dependent.
- English summary failure (independent of tool selection): even
  with all 9 correct tool responses injected, Mode 1 finals shuffle
  scores across films (scenario 8 above), invent scores (scenario 0:
  top film claimed at 22, which is Starfall's DOT score, attached to
  Laugh Riot), or name a near-winner instead of the winner
  (scenario 7: Starfall 413 named, true tied winners Galaxy/Neon
  416). Top-3 order was correct only on the trivial all-zero
  scenario.

### Mode 2 — host workflow works, 0.8B summary does not finish

- Chips: 90/90 exact, host-driven, no model calls, no B2 needed.
- Finals: the model ignores the given tool results as a lookup
  table and instead re-derives everything in prose ("step-by-step
  analysis"), recomputing the combined profile and each dot product
  by hand in LaTeX. At the 250-token cap it is still mid-derivation
  (e.g. scenario 3 ends mid-expression at Starfall's first product
  terms; scenario 8 ends mid-expression at Starfall's total). 5/10
  finals name no film at all; the 4 top-1 hits (scenarios 1, 3, 8, 9)
  are cases where the model named the winner early or the answer
  happened to lead with it — none of them completes a top-3 list
  with correct scores. Top score mentioned 0/9 non-trivial.
- Mode 2's failure is therefore a summarization/generation-budget
  failure, not an orchestration failure: give the same model the
  answers and it still re-computes instead of reading them, and
  runs out of tokens doing it.

### Control — text-only arithmetic does not complete either

- 0/9 non-trivial top-1, 0/9 top-3, 0/9 scores. Finals are the same
  rambling derivation as Mode 2 (recomputing the combined profile
  lane by lane, repeating lanes, never reaching a ranking within
  300 tokens). The one "correct" scenario is again the trivial
  all-zero case, where naming any film counts. The control proves
  the chip modes' 3/9 top-1 is a real (if small) gain over unaided
  0.8B arithmetic — and that 3/9, not 10/10, is the honest size of
  that gain.

## Deployment — which mode to ship, and what the limits mean

Ship **Mode 2's shape, not Mode 1's**: host drives the fixed workflow
(ADD once, DOT per item, in film order), the chip computes, and the
model's job is only the final English sentence. Mode 2 matches Mode
1 on top-1 (3/9 non-trivial each), needs no B2 rescue for workflow,
and is 5.8× faster per scenario (28.7 s vs 166.4 s median). Mode 1
must not ship: its workflow depends on B2 for 89% of steps (80/90),
its guard + retry demonstrably cannot correct multi-step tool
stickiness at this model size, and it costs ~6× the latency for the
same answer quality.

But Mode 2 as tested is NOT shippable as a standalone recommender
either, for a reason that has nothing to do with the chip: the 0.8B
summary cannot rank 8 scored items reliably in English (0/9 top-3).
The deployable form of this application is host-ranked: the host,
which already holds the exact chip scores, sorts them and either
(a) hands the model a pre-ranked list to verbalize in one sentence
(the model then cannot shuffle scores, because the ranking is not
its job), or (b) skips the model for the ranking entirely and uses a
fixed sentence template with the host-computed top film, score, and
top 3. Either form would be 10/10 on ranking by construction — which
is precisely why it is not claimed here as a model result. What the
chip contributes in that form is exact, private, in-weights
arithmetic for the ADD and DOT steps; what it does not contribute
is ranking judgment, and this run shows the 0.8B model cannot supply
that judgment on top of it.

What the 4-bit / 4-lane limits mean for a real recommender:

- **Attribute count is capped at 4.** A real taste profile has more
  than action/comedy/drama/sci-fi. More lanes means a wider circuit
  (more register heads, more lane adders); the 18-layer, 142-slot
  circuit would grow, and the merged model with it. This is a
  circuit-size cost, not a free parameter.
- **Score resolution is coarse and tie-prone.** Each attribute is
  0–15, so two films that differ only in a low-weight attribute can
  tie, and ties did occur in 3/10 scenarios here (designed tie,
  accidental tie, all-zero). A shipped recommender needs an explicit
  tie policy: this run's rule (any tied winner counts for top-1)
  is a scoring rule, not a product rule — a product must pick one
  film to show first (e.g. fixed film-list order as deterministic
  tiebreak) and say so.
- **Score range is 0–900, but differences are small where it
  matters.** Near-ties (416 vs 413 in scenario 7, 195 vs 190 in
  scenario 5) are exactly where the model's English summary fails
  hardest, because it must preserve a 1–3 point gap through a
  shuffled score list. Host-side ranking removes that failure mode
  entirely.
- **The ADD step wraps mod 16.** Combining two strong tastes
  (e.g. [15,15,15,15] + [15,14,15,13] = [14,13,14,12]) loses
  information by design: a combined 14 could mean 14+0, 15+15, or
  7+7. For a real recommender, profile combination should be a
  host-level policy (average, max, weighted sum with rescaling),
  not a blind mod-16 add — the mod-16 ADD is the right test of the
  chip, but the wrong semantics for taste merging at scale.

What would need to change to scale it (future work, not done here):

- More lanes (8–16 attributes) and/or more bits per lane (5–8 bits
  for finer taste resolution), each requiring a rebuilt circuit and
  a fresh 210-case-style regression plus a new application run.
- A larger or better-instructed summarizer for Mode 2, or
  constrained decoding / a fixed template for the final sentence,
  so the ranking the host computed is what the user reads.
- Single-tool routing (Variant A) per step as a Mode 1 repair: the
  multi-tool listing is what the model fails on; routing each step
  to its one known tool would remove the selection failure, at the
  cost of the host owning the workflow — which is Mode 2 anyway.
- A longer final-generation budget for Mode 2 summaries, tested
  explicitly: the 250-token cap truncated every Mode 2 final, so
  the 0/9 top-3 rate is partly a budget artifact, not purely a
  capability finding. That cut both ways and is not used to excuse
  the result: Mode 1 finals within the same budget did produce
  rankings (wrong ones), so budget alone does not explain Mode 2.

## Gate 0 — rebuild regression, twice (reclamation forced a second session)

- Session 1 (`vec4-app`, kyu008008, T4 14,912 MiB, torch 2.11.0+cu130,
  Python 3.13.16): merged exactly as before — hidden 1280, 42 layers
  (18 chip + 24 Qwen), vocab 248576, 1,526,970,752 params, 3.05 GB
  bf16, chip offset v0=248320 — regression **210/210** in 120.5 s
  generation, VRAM max 3,010.18 MiB, text sanity "The capital of
  France is" → " Paris.\nThe capital of France is Paris." No
  application mode ran on a failed gate.
- Session 1 was reclaimed mid-run (server-side, between Mode 1
  scenario 5 and the next status check; "No active sessions found on
  server" under kyu008008, and the session-1 isolated registry file
  was found wiped to `{}`). Mode 1 scenarios 0–4 had already been
  downloaded locally, so no Mode 1 data was lost; only the
  session-1 detached log (`app.log`) was lost with the VM.
- Session 2 (`vec4-app2`, same account kyu008008, fresh isolated
  registry): Gate 0 re-run from scratch, **210/210** again in
  125.6 s, VRAM max 3,010.18 MiB, same checkpoint spec, same text
  sanity. The resume driver preloaded the 5 downloaded Mode 1
  scenarios and ran only scenarios 5–9, then Mode 2 and control in
  full. All numbers in this report come from the union of the two
  sessions' downloaded JSON; no number is carried across sessions
  without its downloaded record.
- `stage0_error.txt` in both sessions is the known tail-log
  artifact (same bug class as vec4-fix/vec4-untested: the final
  `log(..., flush=True)` call collided with the wrapper's own
  `flush=True`). It fires AFTER `stage0_done.json` is written and
  after the 210/210 regression is recorded; it is not a gate
  failure and no mode ran on it.

## Session note (honest record — two T4 sessions, account handling)

- Account: **kyu008008** for both sessions, exactly one account for
  GPU work, no quota hunting across accounts. Why kyu008008: the
  rotation file read "2026-11-03 cyc236es" at task start, and the
  training-track pool order (ppoppo205 → cyc236es → kyu008008 →
  kyu009009 → cyc236ha → ppoppo205) makes kyu008008 the next-day
  candidate after cyc236es; the first creation attempt on the
  then-active cyc236es returned `gpu-unavailable`, and the single
  switch to kyu008008 succeeded. No 403, no quota retry loop, no
  further account switch for GPU reasons.
- Rotation file: **left untouched** ("2026-11-03 cyc236es").
  This vec4 work is not training-track work; updating it would
  corrupt the daily training rotation's pick.
- Active-account restore — deviation, stated plainly: the task's
  original restore target was cyc236es (the active account at task
  start). During this run another agent, on Peter's 07:40 tap,
  switched the global active account to ppoppo205 for a separate
  weight-space session (recorded in MEMORY.md as restoring to
  kyu008008 when that session ends). Restoring to cyc236es at my
  teardown would have disrupted that session's bookkeeping, so this
  run restored active to **ppoppo205** (verified:
  `token.json -> token.ppoppo205.json`) and stopped its own
  session (`vec4-app2` stopped, "No active sessions found on
  server" confirmed under its registry). The parent agent should
  confirm the final active-account state with the weight-space
  run's owner rather than assuming cyc236es.
- No GitHub push, no Threads post, no arxiv-lab change, no
  MEMORY.md change by this run. `../vec4/`, `../vec4-toolcall/`,
  `../vec4-fix/`, and `../vec4-untested/` originals untouched; all
  new code, dataset, logs, and results are in this directory
  (VEC4/driver code was copied here, not edited in place).

## Errors, verbatim (short)

- Session 1 creation on cyc236es: `{"ok": false, "err": "gpu-unavailable",
  "msg": "GPU unavailable."}` — one switch to kyu008008, no retry loop.
- Session 1 loss: `{"ok": false, "err": "not-found", "msg": "Not found."}`
  on partial download, then "No active sessions found on server" —
  free-tier reclamation, not a driver error.
- Stage 0 tail-log artifact (both sessions, after 210/210 recorded):
  `TypeError: print() got multiple values for keyword argument
  'flush'` in `stage0_app.py`'s `log()` wrapper.
- Reference-kernel warnings in every run (speed cause, not failures):
  "`causal_conv1d_fn` is falling back to its reference PyTorch
  implementation", and the same for `chunk_gated_delta_rule` /
  `flash-linear-attention`.
- One monitoring command was killed by the runtime (`SIGTERM`) while
  sleeping between checks; it carried no data (partials are written
  by the VM-side driver, not by the monitor) and the next check
  recovered the state.

## Files

- `RESULT.md` — this report
- `dataset.json` — 8 films, 10 scenarios, pre-registered ground truth
  (scores + ranking + winners per scenario), tie rule
- `make_dataset.py` — dataset builder (run before any model run)
- `mode1_result.json` — Mode 1, all 10 scenarios (5 pre-reclamation
  records + 5 resume records), full tool-call traces
- `mode2_result.json` — Mode 2, all 10 scenarios
- `control_result.json` — text-only control, all 10 scenarios
- `mode1_partial.json` — Mode 1 partial downloaded before reclamation
  (scenarios 0–4; superseded by `mode1_result.json`)
- `app.py`, `app_resume.py` — drivers (resume preloads the partial)
- `stage0_app.py`, `vec4_merged_result.json`,
  `vec4_merged_result2.json` — Gate 0, sessions 1 and 2
- `circuit.py`, `merge_vec4.py`, `ref.py`, `run_merged.py` — copies
  of VEC4 originals (originals untouched)
- `colab_registry.json` (session 1, found wiped to `{}` after the
  account switch/reclamation), `colab_registry2.json` (session 2)
- `app2.log` — session-2 detached run log; session-1 `app.log` was
  lost with the reclaimed VM (its scenarios survive in JSON)
