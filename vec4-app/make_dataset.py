"""Build vec4-app/dataset.json: 8 films + 10 scenarios, ground truth
pre-registered in pure Python (ref.py logic) BEFORE any model run.

Attributes in order [action, comedy, drama, sci-fi], each 0-15.
Tie rule (documented here and in RESULT.md): top-1 counts as correct if
the model names ANY film tied for first place; top-3 order counts as
correct if the first three distinct film names in the answer have
non-increasing ground-truth scores and no film outside those three has
a score strictly greater than the smallest of the three (so a tie at
the 3rd-place boundary may be resolved either way).
"""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ref

FILMS = [
    {"name": "Starfall Squadron", "vector": [15, 2, 3, 14]},   # action+scifi, near-max lanes
    {"name": "Laugh Riot", "vector": [3, 15, 2, 1]},          # comedy-heavy
    {"name": "The Long Winter", "vector": [2, 3, 15, 1]},     # drama-heavy
    {"name": "Neon Rebellion", "vector": [13, 4, 5, 13]},     # action+scifi
    {"name": "Family Picnic", "vector": [1, 14, 6, 0]},       # comedy-heavy
    {"name": "Quiet Harbor", "vector": [1, 2, 14, 2]},        # drama-heavy
    {"name": "Galaxy Heist", "vector": [12, 8, 4, 12]},       # action/comedy/scifi mix
    {"name": "Static", "vector": [0, 0, 1, 0]},              # near-zero
]

def scenario(sid, kind, A, B, note):
    combined = [(a + b) % 16 for a, b in zip(A, B)]
    scores = {f["name"]: sum(c * v for c, v in zip(combined, f["vector"])) for f in FILMS}
    ranking = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    # ranking sorted by score desc; ties broken alphabetically for display only
    # (tie validity is judged by scores, not by this display order).
    return {"id": sid, "kind": kind, "profile_a": A, "profile_b": B,
            "combined": combined, "scores": scores,
            "ranking": [{"film": n, "score": s} for n, s in ranking],
            "winners": [n for n, s in ranking if s == ranking[0][1]],
            "note": note}

scenarios = []
# 0: tie for first, designed: combined [0,8,2,0] -> Laugh Riot 124 = Family Picnic 124, top.
scenarios.append(scenario(0, "tie_first", [13, 5, 14, 9], [3, 3, 4, 7],
    "Designed tie: combined [0,8,2,0]; Laugh Riot 124 = Family Picnic 124, next best 72."))
# 1: all-zero pair -> combined zero -> all scores 0, 8-way tie.
scenarios.append(scenario(1, "all_zero", [0, 0, 0, 0], [0, 0, 0, 0],
    "All-zero profiles: combined [0,0,0,0], all scores 0; correct top answer is a tie, not a single winner."))
# 2: near-max pair.
scenarios.append(scenario(2, "near_max", [15, 15, 15, 15], [15, 14, 15, 13],
    "Near-max profiles: combined [14,13,14,12]."))
# 3-9: random-ish, Random(20261104), drawn in order.
rng = random.Random(20261104)
for sid in range(3, 10):
    A = [rng.randrange(16) for _ in range(4)]
    B = [rng.randrange(16) for _ in range(4)]
    scenarios.append(scenario(sid, "random", A, B, f"Random(20261104) draw {sid - 3}."))

out = {"attributes": ["action", "comedy", "drama", "sci-fi"],
       "films": FILMS, "scenarios": scenarios,
       "tie_rule": "Top-1 correct iff named film is in winners (any tied first-place film). "
                   "Top-3 order correct iff first three distinct film names in the answer have "
                   "non-increasing ground-truth scores and no film outside them scores strictly "
                   "above the smallest of the three."}
Path(__file__).resolve().parent.joinpath("dataset.json").write_text(json.dumps(out, indent=2))
for s in scenarios:
    print(s["id"], s["kind"], "A", s["profile_a"], "B", s["profile_b"],
          "->", s["combined"], "winner(s)", s["winners"], "top3",
          [(r["film"], r["score"]) for r in s["ranking"][:3]])
print("distinct winners:", sorted({w for s in scenarios for w in s["winners"]}))
