"""Language-probe comparison: base vs merged model on the 10 plain
prompts in generalisation.json (language_probe). NO chip involvement:
these prompts contain no tool calls and no chip tokens. The question
is whether the merged model still produces sane ordinary text, and
how it compares with the base model under IDENTICAL conditions.

Conditions (fixed before running, same for base and merged):
plain completion (no chat template), greedy decoding, max 60 new
tokens. This driver does NOT judge: it records verbatim
continuations plus computed stats, and the verdict is judged by a
human from the outputs. This is a sanity comparison, NOT a KL
divergence, NOT a benchmark score, and merged output need not
match base output — it must be sane English of comparable quality.

Computed stats per output: new token count, whether generation hit
EOS, whether any generated id is a chip id (>= v0, merged only),
most-frequent-token share, and a loop flag (a 5-token sequence
repeating inside the last 20 generated tokens). Chip ids in a
merged output, an empty output, or a degenerate loop all count
against sanity; the final verdict is manual, from the verbatim text.

Usage: python3 lang_probe.py BASE_DIR MERGED_DIR PROBES_JSON --out FILE
Loads base first, frees it, then merged (sequential, fits a T4).
Writes JSON incrementally per probe.
"""
import argparse, json, sys, gc
from pathlib import Path

def loop_flag(ids):
    if len(ids) < 10:
        return False
    tail = ids[-20:]
    for n in (5,):
        seen = set()
        for i in range(len(tail) - n + 1):
            s = tuple(tail[i:i + n])
            if s in seen:
                return True
            seen.add(s)
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("merged"); ap.add_argument("probes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=60)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    prompts = json.loads(Path(args.probes).read_text())["language_probe"]
    assert len(prompts) == 10, len(prompts)
    meta = json.loads(Path(args.merged, "chipformer.json").read_text())
    v0 = meta["chip_token_offset"]
    tok = AutoTokenizer.from_pretrained(args.base)
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    res = {"base": args.base, "merged": args.merged, "v0": v0,
           "max_tokens": args.max_tokens, "rows": []}
    def dump():
        out_path.write_text(json.dumps(res, indent=2))
    for label, d in (("base", args.base), ("merged", args.merged)):
        print("LOADING", label, d, flush=True)
        model = AutoModelForImageTextToText.from_pretrained(d, dtype=torch.bfloat16).to("cuda").eval()
        for prompt in prompts:
            ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
            with torch.no_grad():
                gen = model.generate(ids, max_new_tokens=args.max_tokens, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            new = gen[0, ids.shape[1]:].tolist()
            text = tok.decode(new, skip_special_tokens=True)
            counts = {}
            for t in new:
                counts[t] = counts.get(t, 0) + 1
            top_share = max(counts.values()) / len(new) if new else 0.0
            row = {"model": label, "prompt": prompt, "continuation": text,
                   "new_tokens": len(new), "hit_eos": bool(new and new[-1] == tok.eos_token_id),
                   "chip_token_leak": bool(any(t >= v0 for t in new)) if label == "merged" else None,
                   "top_token_share": round(top_share, 3), "loop": loop_flag(new)}
            res["rows"].append(row)
            print(label, "|", prompt[:40], "| tokens", len(new), "| leak", row["chip_token_leak"],
                  "| loop", row["loop"], "|", text[:120].replace("\n", " "), flush=True)
            dump()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    print("LANG_PROBE_DONE", flush=True)

if __name__ == "__main__":
    main()
