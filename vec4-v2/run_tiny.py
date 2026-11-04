"""Stage 2 (tiny-model part), run on Colab GPU.

Builds the VEC4 circuit as a standalone Qwen3.5-0.8B-shaped model
(tiny.build, no Qwen layers), loads it on CUDA, and runs the same
test cases as test_circuit.py by direct forward argmax (one forward
per generated token, host feeds tokens, computation in weights only)
plus a stock model.generate() check on a subset.

Usage on the VM (layout must mirror the local layout):
  /content/qwenputer-colab/upstream/  (git clone of divagr18/qwenputer)
  /content/qwenputer-colab/vec4/      (this file, circuit.py, ref.py)
  python3 run_tiny.py [--cases N] [--generate-subset N] [--out FILE]
Writes a JSON result file (default /content/vec4_tiny_result.json).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # qwenputer-colab/
sys.path.insert(0, str(ROOT / "upstream"))
sys.path.insert(0, str(HERE))

import ref
from circuit import build_vec4v2 as build_vec4


def cases():
    """v2 tiny set: same 396 cases as test_circuit.py / run_merged.py."""
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
    # v2 tiny set is fixed at 396 cases (see cases())
    ap.add_argument("--generate-subset", type=int, default=10)
    ap.add_argument("--out", default="/content/vec4_tiny_result.json")
    args = ap.parse_args()

    import torch
    from chipformer.q35 import tiny

    t0 = time.time()
    c = build_vec4()
    stats = c.stats()
    print("circuit stats:", stats, "dims:", len(c.dims), "layers:", c.n_layers(), flush=True)
    code_dims = [next(iter(e.terms)) for e in c.code]
    model = tiny.build(c, list(range(248)), code_dims)
    print("tiny hidden:", model.layout.hidden, "layers:", model.config.num_hidden_layers,
          "vocab:", model.config.vocab_size, flush=True)
    model = model.to("cuda")
    print("VRAM after load (MiB):", torch.cuda.memory_allocated() / 2**20,
          "max:", torch.cuda.max_memory_allocated() / 2**20, flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print("params:", n_params, "bf16 GB:", n_params * 2 / 1e9, flush=True)

    def next_token_forward(seq):
        import torch
        with torch.no_grad():
            out = model(torch.tensor([seq], device="cuda"))
            return int(torch.argmax(out.logits[0, -1]).item())

    def run_case(op, A, B, use_generate=False):
        inp = ref.encode(op, A, B)
        exp = ref.expected_output(op, A, B)
        seq, gen = list(inp), []
        for _ in range(6):
            if use_generate:
                with torch.no_grad():
                    out = model.generate(torch.tensor([seq], device="cuda"),
                                         max_new_tokens=1, do_sample=False)
                nxt = int(out[0, len(seq)].item())
            else:
                nxt = next_token_forward(seq)
            gen.append(nxt)
            seq.append(nxt)
            if nxt == ref.END:
                break
        return gen == exp, gen, exp

    all_cases = cases()
    results, fails = [], []
    t_gen = time.time()
    for op, A, B in all_cases:
        ok, gen, exp = run_case(op, A, B, use_generate=False)
        results.append({"op": op, "A": A, "B": B, "ok": ok, "gen": gen, "exp": exp})
        if not ok:
            fails.append(results[-1])
    print(f"forward: {len(results) - len(fails)}/{len(results)} pass, "
          f"{time.time() - t_gen:.1f}s", flush=True)
    # stock generate() subset (first N cases, same order)
    gen_results, gen_fails = [], []
    for op, A, B in all_cases[: args.generate_subset]:
        try:
            ok, gen, exp = run_case(op, A, B, use_generate=True)
        except Exception as e:  # noqa: BLE001 - record, do not hide
            ok, gen, exp = False, [], []
            gen_results.append({"op": op, "A": A, "B": B, "ok": ok,
                                "error": "%s: %s" % (type(e).__name__, str(e)[:500])})
            gen_fails.append(gen_results[-1])
            continue
        gen_results.append({"op": op, "A": A, "B": B, "ok": ok, "gen": gen, "exp": exp})
        if not ok:
            gen_fails.append(gen_results[-1])
    print(f"generate(): {len(gen_results) - len(gen_fails)}/{len(gen_results)} pass", flush=True)
    out = {
        "circuit_stats": stats, "dims": len(c.dims), "chip_layers": c.n_layers(),
        "tiny_hidden": model.layout.hidden,
        "tiny_layers": model.config.num_hidden_layers,
        "tiny_vocab": model.config.vocab_size,
        "params": n_params, "bf16_gb": n_params * 2 / 1e9,
        "vram_after_load_mib": torch.cuda.memory_allocated() / 2**20,
        "vram_max_mib": torch.cuda.max_memory_allocated() / 2**20,
        "forward_pass": f"{len(results) - len(fails)}/{len(results)}",
        "forward_fails": fails[:20],
        "generate_pass": f"{len(gen_results) - len(gen_fails)}/{len(gen_results)}",
        "generate_fails": gen_fails[:20],
        "seconds_total": time.time() - t0,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote", args.out, flush=True)
    print(json.dumps({k: v for k, v in out.items() if k not in ("forward_fails", "generate_fails")},
                     indent=2))


if __name__ == "__main__":
    main()
