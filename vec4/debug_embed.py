import json
from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText

merged = "/content/qwen35-08b-vec4"
meta = json.loads((Path(merged) / "chipformer.json").read_text())
v0 = meta["chip_token_offset"]
model = AutoModelForImageTextToText.from_pretrained(merged, dtype=torch.bfloat16).to("cuda").eval()
emb = model.model.language_model.embed_tokens.weight
for name, tid in [("GO", v0 + 130), ("ADD0", v0 + 131), ("VAL0", v0 + 2), ("Qwen0", 0)]:
    row = emb[tid].float().cpu()
    nz = torch.nonzero(row).flatten().tolist()
    print(name, "nonzero dims:", nz, flush=True)
    print("  values:", [float(row[d]) for d in nz], flush=True)
# norm weight chip region
norm_w = model.model.language_model.norm.weight.float().cpu()
nz = torch.nonzero(norm_w[1024:]).flatten().tolist()
print("norm chip nonzero dims (relative to 1024):", nz, flush=True)
print("  values:", [float(norm_w[1024 + d]) for d in nz], flush=True)
