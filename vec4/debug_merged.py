"""Debug merged model at GO for one ADD all-zero case."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/content/qwenputer-colab/vec4")
import torch
from transformers import AutoModelForImageTextToText

import ref

merged = "/content/qwen35-08b-vec4"
meta = json.loads((Path(merged) / "chipformer.json").read_text())
v0 = meta["chip_token_offset"]
print("v0", v0, "hidden", meta["hidden"], "mu slot (from merge log) 1074", flush=True)
model = AutoModelForImageTextToText.from_pretrained(merged, dtype=torch.bfloat16).to("cuda").eval()
lm = model.model.language_model
emb = lm.embed_tokens.weight
norm_w = lm.norm.weight
print("embed shape", tuple(emb.shape), "norm shape", tuple(norm_w.shape), flush=True)
# code dims: from merge, lay.dim(code) = off + m + 1 + slot[code]
# We do not know slots here; instead, find dims where chip token rows differ
# from Qwen text rows in a code-like way: chip ADD lane0 digit0 token vs GO token.
# Simpler: print norm weight stats for chip region.
print("norm chip region (1024:): unique", torch.unique(norm_w[1024:])[:20], flush=True)
print("norm max", float(norm_w.max()), "min", float(norm_w.min()), flush=True)
# embedding rows: GO chip token, ADD lane0 digit0 chip token, Qwen token 0
for name, tid in [("GO", v0 + 130), ("ADD0", v0 + 131), ("Qwen0", 0), ("Qwen1", 1)]:
    row = emb[tid].float()
    print(name, "row norm", float(row.norm()), "nonzero dims",
          int((row != 0).sum()), "chip part norm", float(row[1024:].norm()),
          "max chip", float(row[1024:].abs().max()), flush=True)
# forward one case
inp = [v0 + t for t in ref.encode("ADD", [0, 0, 0, 0], [0, 0, 0, 0])]
print("input ids (standalone):", ref.encode("ADD", [0, 0, 0, 0], [0, 0, 0, 0]), flush=True)
with torch.no_grad():
    out = model(input_ids=torch.tensor([inp], device="cuda"),
                output_hidden_states=True)
    logits = out.logits[0, -1]
    top = torch.topk(logits, 10)
    print("top10 at GO:", [(int(i), float(v)) for i, v in zip(top.indices, top.values)], flush=True)
    # hidden states: model.model is multimodal; out.hidden_states may be for language model?
    hs = out.hidden_states
    print("n hidden states:", len(hs) if hs else None, flush=True)
    if hs:
        for idx in (0, 18, 42, len(hs) - 1):
            h = hs[idx][0, -1].float()
            print(f"hs[{idx}] norm {float(h.norm()):.4f} max {float(h.abs().max()):.4f} "
                  f"chip norm {float(h[1024:].norm()):.4f} mu1074 {float(h[1074]):.6f}",
                  flush=True)
