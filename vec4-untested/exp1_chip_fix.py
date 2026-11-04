"""Exp1 correction: cached chip scoring in exp1_speed.py did NOT stop at END
(gen_cached stops only at text EOS), so cached chip_gen contained exp +
post-END tokens and compared unequal. This rerun trims every method at END
before comparing — the host's normal stop rule (cf. run_merged.py).
Writes /content/vec4-untested/exp1_chip_fix.json."""
import json, sys, time, traceback
from pathlib import Path
sys.path.insert(0, "/content/qwenputer-colab/vec4-untested")
import ref
OUT = Path("/content/vec4-untested"); OUT.mkdir(parents=True, exist_ok=True)
def log(*a, **kw): print(*a, flush=True)
def cases_exact():
    import random
    rng = random.Random(20261103)
    edges_dot = [([0,0,0,0],[0,0,0,0]), ([15,15,15,15],[15,15,15,15]), ([3,5,7,9],[2,4,6,8]), ([0,15,0,15],[15,0,15,0])]
    edges_add = [([0,0,0,0],[0,0,0,0]), ([15,15,15,15],[15,15,15,15]), ([15,0,8,7],[1,15,8,9]), ([0,1,2,3],[15,14,13,12])]
    cases = []
    for A,B in edges_dot: cases.append(("DOT", A, B))
    for _ in range(6): cases.append(("DOT", [rng.randrange(16) for _ in range(4)], [rng.randrange(16) for _ in range(4)]))
    for A,B in edges_add: cases.append(("ADD", A, B))
    for _ in range(6): cases.append(("ADD", [rng.randrange(16) for _ in range(4)], [rng.randrange(16) for _ in range(4)]))
    return cases
def trim(gen, end_id):
    return gen[:gen.index(end_id) + 1] if end_id in gen else gen
def main():
    import torch
    from transformers import AutoModelForImageTextToText
    meta = json.loads(Path("/content/qwen35-08b-vec4/chipformer.json").read_text()); v0 = meta["chip_token_offset"]
    model = AutoModelForImageTextToText.from_pretrained("/content/qwen35-08b-vec4", dtype=torch.bfloat16).to("cuda").eval()
    log("loaded VRAM MiB", torch.cuda.memory_allocated()/2**20, flush=True)
    def gen_stepwise(seq, cap):
        seq = list(seq); gen = []
        for _ in range(cap):
            with torch.no_grad():
                out = model(input_ids=torch.tensor([seq], device="cuda"))
            nxt = int(torch.argmax(out.logits[0, -1]).item()); gen.append(nxt); seq.append(nxt)
            if nxt == v0 + ref.END: break
        return gen
    def gen_cached(seq, cap):
        seq = list(seq)
        with torch.no_grad():
            out = model(input_ids=torch.tensor([seq], device="cuda"), use_cache=True)
        past = out.past_key_values; gen = []
        nxt = int(torch.argmax(out.logits[0, -1]).item())
        for _ in range(cap):
            gen.append(nxt); seq.append(nxt)
            if nxt == v0 + ref.END: break
            with torch.no_grad():
                out = model(input_ids=torch.tensor([[nxt]], device="cuda"), past_key_values=past, use_cache=True)
            past = out.past_key_values; nxt = int(torch.argmax(out.logits[0, -1]).item())
        return gen
    rows = []
    for idx, (op, A, B) in enumerate(cases_exact()):
        inp = [v0 + t for t in ref.encode(op, A, B)]; exp = [v0 + t for t in ref.expected_output(op, A, B)]
        t0 = time.time(); gs = gen_stepwise(inp, 6); ts = time.time() - t0
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(input_ids=torch.tensor([inp], device="cuda"), max_new_tokens=6, do_sample=False, use_cache=True, eos_token_id=v0 + ref.END, pad_token_id=v0 + ref.END)
        gg = [int(x) for x in out[0][len(inp):].tolist()]; tg = time.time() - t0
        t0 = time.time(); gc_raw = gen_cached(inp, 6); tc = time.time() - t0
        gc = trim(gc_raw, v0 + ref.END)
        row = {"case": idx, "op": op, "A": A, "B": B,
               "stepwise_ok": gs == exp, "stepwise_s": ts,
               "generate_ok": gg == exp, "generate_s": tg,
               "cached_raw": [g - v0 for g in gc_raw], "cached_trimmed_ok": gc == exp, "cached_s": tc,
               "cached_trimmed_tokens_match_stepwise": gc == gs}
        rows.append(row)
        log(f"FIXCHIP {idx} {op} step={row['stepwise_ok']} gen={row['generate_ok']} cached_trimmed={row['cached_trimmed_ok']} match_step={row['cached_trimmed_tokens_match_stepwise']} s={ts:.2f}/{tg:.2f}/{tc:.2f}", flush=True)
        (OUT / "exp1_chip_fix_partial.json").write_text(json.dumps({"rows": rows}, indent=2))
    def med(xs):
        xs = sorted(xs); n = len(xs); return xs[n//2] if n % 2 else (xs[n//2-1]+xs[n//2])/2
    out = {"rows": rows, "stepwise": sum(1 for r in rows if r["stepwise_ok"]), "generate": sum(1 for r in rows if r["generate_ok"]),
           "cached_trimmed": sum(1 for r in rows if r["cached_trimmed_ok"]),
           "cached_match_stepwise": sum(1 for r in rows if r["cached_trimmed_tokens_match_stepwise"]),
           "median_s": {"stepwise": med([r["stepwise_s"] for r in rows]), "generate": med([r["generate_s"] for r in rows]), "cached": med([r["cached_s"] for r in rows])},
           "vram_max_mib": torch.cuda.max_memory_allocated()/2**20}
    (OUT / "exp1_chip_fix.json").write_text(json.dumps(out, indent=2))
    log("WROTE exp1_chip_fix.json", out["stepwise"], out["generate"], out["cached_trimmed"], flush=True)
if __name__ == "__main__":
    try: main()
    except Exception:
        traceback.print_exc(); (OUT / "exp1_chip_fix_error.txt").write_text(traceback.format_exc()); raise
