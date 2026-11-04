"""Stage 3 (run): load the merged Qwen3.5-0.8B+VEC4v2 checkpoint on the T4
and run the same VADD/VDOT test set in-model, plus one plain text prompt.

Chip tokens are addressed by id (v0 + standalone VEC4 id); no tokenizer
is needed for them. The plain text prompt uses the base tokenizer.
Generation is stepwise direct-forward argmax (host feeds tokens, reads
tokens; computation in weights), same as run_tiny.py, plus a stock
model.generate() check on a subset for the text prompt only if needed.

Usage: python3 run_merged.py MERGED_DIR BASE_DIR [--cases N] [--out FILE]
Writes JSON (default /content/vec4_merged_result.json).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ref


def cases():
    """v2 Gate 0 set: 6 edges per op + 60 random per op (seed 20261104),
    identical to test_circuit.py's case set. Total 396."""
    edges = [([0, 0, 0, 0], [0, 0, 0, 0]),
             ([15, 15, 15, 15], [15, 15, 15, 15]),
             ([15, 0, 8, 7], [1, 15, 8, 9]),
             ([0, 1, 2, 3], [15, 14, 13, 12]),
             ([0, 15, 0, 15], [15, 0, 15, 0]),
             ([3, 5, 7, 9], [2, 4, 6, 8])]
    out = []
    for op in ref.OPS:
        for A, B in edges:
            out.append((op, A, B))
    rng = random.Random(20261104)
    for op in ref.OPS:
        for _ in range(60):
            out.append((op, [rng.randrange(16) for _ in range(4)],
                        [rng.randrange(16) for _ in range(4)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("merged")
    ap.add_argument("base")
    # v2 Gate 0 set is fixed at 396 cases (see cases())
    ap.add_argument("--out", default="/content/vec4_merged_result.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    meta = json.loads((Path(args.merged) / "chipformer.json").read_text())
    v0 = meta["chip_token_offset"]
    print("chip offset v0:", v0, "hidden:", meta["hidden"], flush=True)
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        args.merged, dtype=torch.bfloat16).to("cuda").eval()
    print("load seconds:", round(time.time() - t0, 1),
          "VRAM MiB:", torch.cuda.memory_allocated() / 2**20,
          "max:", torch.cuda.max_memory_allocated() / 2**20, flush=True)
    print("config layers:", model.config.text_config.num_hidden_layers,
          "hidden:", model.config.text_config.hidden_size,
          "vocab:", model.config.text_config.vocab_size, flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print("params:", n_params, "bf16 GB:", n_params * 2 / 1e9, flush=True)

    def next_token(seq):
        with torch.no_grad():
            out = model(input_ids=torch.tensor([seq], device="cuda"))
            return int(torch.argmax(out.logits[0, -1]).item())

    all_cases = cases()
    results, fails = [], []
    t_gen = time.time()
    for op, A, B in all_cases:
        inp = [v0 + t for t in ref.encode(op, A, B)]
        exp = [v0 + t for t in ref.expected_output(op, A, B)]
        seq, gen = list(inp), []
        for _ in range(6):
            nxt = next_token(seq)
            gen.append(nxt)
            seq.append(nxt)
            if nxt == v0 + ref.END:
                break
        ok = gen == exp
        results.append({"op": op, "A": A, "B": B, "ok": ok,
                        "gen": [g - v0 for g in gen], "exp": [e - v0 for e in exp]})
        if not ok:
            fails.append(results[-1])
    print(f"merged: {len(results) - len(fails)}/{len(results)} pass, "
          f"{time.time() - t_gen:.1f}s", flush=True)

    # One plain text prompt through the merged model (base tokenizer).
    tok = AutoTokenizer.from_pretrained(args.base)
    prompt = "The capital of France is"
    ids = tok.encode(prompt, add_special_tokens=False)
    seq = list(ids)
    gen_ids = []
    for _ in range(10):
        nxt = next_token(seq)
        gen_ids.append(nxt)
        seq.append(nxt)
        if nxt in (tok.eos_token_id, 151645, 151644):
            break
    text_out = tok.decode(gen_ids, skip_special_tokens=True)
    print("text prompt:", prompt, "->", repr(text_out), "ids:", gen_ids, flush=True)
    out = {
        "chip_offset": v0, "hidden": meta["hidden"],
        "layers": model.config.text_config.num_hidden_layers,
        "vocab": model.config.text_config.vocab_size,
        "params": n_params, "bf16_gb": n_params * 2 / 1e9,
        "vram_after_load_mib": torch.cuda.memory_allocated() / 2**20,
        "vram_max_mib": torch.cuda.max_memory_allocated() / 2**20,
        "load_seconds": time.time() - t0,
        "merged_pass": f"{len(results) - len(fails)}/{len(results)}",
        "merged_fails": fails[:20],
        "text_prompt": prompt, "text_output": text_out, "text_ids": gen_ids,
        "seconds_total": time.time() - t0,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
