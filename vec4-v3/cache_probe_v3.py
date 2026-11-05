"""Cache re-proof for vec4-v3: stepwise vs stock generate() vs manual
per-phase KV cache, END-trimmed, on 40 chip cases (4 per op x 10 ops:
2 edges + 2 random, seed 20261105). All three drivers must be
token-identical before the cached driver is used for tool tests.
Usage: python3 cache_probe_v3.py MERGED_DIR --out FILE
"""
import json, sys, time, traceback
from pathlib import Path
sys.path.insert(0, "/content/qwenputer-colab/vec4-v3")
import ref
MERGED = sys.argv[1]
OUT = Path(sys.argv[sys.argv.index("--out") + 1]).parent; OUT.mkdir(parents=True, exist_ok=True)
OUTFILE = Path(sys.argv[sys.argv.index("--out") + 1])
def log(*a, **kw): print(*a, flush=True)
try:
    import torch
    from transformers import AutoModelForImageTextToText
    meta = json.loads(Path(MERGED, "chipformer.json").read_text())
    v0 = meta["chip_token_offset"]
    model = AutoModelForImageTextToText.from_pretrained(MERGED, dtype=torch.bfloat16).to("cuda").eval()
    log("loaded VRAM MiB", torch.cuda.memory_allocated() / 2**20, flush=True)

    def stepwise(seq, cap):
        seq = list(seq); gen = []
        for _ in range(cap):
            with torch.no_grad():
                out = model(input_ids=torch.tensor([seq], device="cuda"))
            nxt = int(torch.argmax(out.logits[0, -1]).item())
            gen.append(nxt); seq.append(nxt)
            if nxt == v0 + ref.END: break
        return gen

    def generate(seq, cap):
        with torch.no_grad():
            out = model.generate(input_ids=torch.tensor([seq], device="cuda"), max_new_tokens=cap,
                                 do_sample=False, use_cache=True, eos_token_id=v0 + ref.END,
                                 pad_token_id=v0 + ref.END)
        return [int(x) for x in out[0][len(seq):].tolist()]

    def cached(seq, cap):
        with torch.no_grad():
            out = model(input_ids=torch.tensor([seq], device="cuda"), use_cache=True)
        past = out.past_key_values; nxt = int(torch.argmax(out.logits[0, -1]).item())
        gen = [nxt]
        while nxt != v0 + ref.END and len(gen) < cap:
            with torch.no_grad():
                out = model(input_ids=torch.tensor([[nxt]], device="cuda"), past_key_values=past, use_cache=True)
            past = out.past_key_values; nxt = int(torch.argmax(out.logits[0, -1]).item()); gen.append(nxt)
        return gen

    def trim(gen):
        # END-trim: cached/generate may continue past chip END into text.
        if v0 + ref.END in gen:
            return gen[:gen.index(v0 + ref.END) + 1]
        return gen

    import random
    rng = random.Random(20261105)
    cases = []
    for op in ref.OPS:
        cases.append((op, [0, 0, 0, 0], [0, 0, 0, 0]))
        cases.append((op, [15, 15, 15, 15], [15, 15, 15, 15]))
        for _ in range(2):
            cases.append((op, [rng.randrange(16) for _ in range(4)], [rng.randrange(16) for _ in range(4)]))
    rows = []
    t_step, t_gen, t_cach = [], [], []
    for op, A, B in cases:
        inp = [v0 + t for t in ref.encode(op, A, B)]
        exp = [v0 + t for t in ref.expected_output(op, A, B)]
        t0 = time.time(); g_s = stepwise(inp, 6); t_step.append(time.time() - t0)
        t0 = time.time(); g_g = generate(inp, 6); t_gen.append(time.time() - t0)
        t0 = time.time(); g_c = cached(inp, 6); t_cach.append(time.time() - t0)
        row = {"op": op, "A": A, "B": B,
               "stepwise_ok": g_s == exp, "generate_ok": trim(g_g) == exp,
               "cached_trimmed_ok": trim(g_c) == exp,
               "cached_match_stepwise": trim(g_c) == g_s,
               "stepwise_s": t_step[-1], "generate_s": t_gen[-1], "cached_s": t_cach[-1]}
        rows.append(row)
        log(f"PROBE {op} {A} {B} step={row['stepwise_ok']} gen={row['generate_ok']} cached={row['cached_trimmed_ok']} match={row['cached_match_stepwise']}", flush=True)
        (OUT / "cache_probe_v3_partial.json").write_text(json.dumps({"rows": rows}, indent=2))
    def med(xs):
        xs = sorted(xs); n = len(xs); return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
    res = {"rows": rows,
           "stepwise": sum(r["stepwise_ok"] for r in rows),
           "generate": sum(r["generate_ok"] for r in rows),
           "cached_trimmed": sum(r["cached_trimmed_ok"] for r in rows),
           "cached_match_stepwise": sum(r["cached_match_stepwise"] for r in rows),
           "median_s": {"stepwise": med(t_step), "generate": med(t_gen), "cached": med(t_cach)},
           "vram_max_mib": torch.cuda.max_memory_allocated() / 2**20}
    OUTFILE.write_text(json.dumps(res, indent=2))
    log("WROTE", str(OUTFILE), res["stepwise"], res["generate"], res["cached_trimmed"], res["cached_match_stepwise"], flush=True)
except Exception:
    traceback.print_exc()
    (OUT / "cache_probe_v3_error.txt").write_text(traceback.format_exc())
    raise
