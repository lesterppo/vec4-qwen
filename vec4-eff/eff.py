"""Efficiency comparison for one merged VEC4 v2 model: chip (cached and
stepwise) vs Python ref loop vs NumPy vectorised vs torch CPU vs torch
CUDA, all on the same VM. Framing, stated up front: the chip is a
42-layer model forward per generated token; it will lose to NumPy by
orders of magnitude on raw latency/throughput and on memory. That is
the expected result, not a failure of the measurement. What this
measures is HOW MUCH it loses, and what the fixed costs are (load
time, checkpoint size, VRAM) — the numbers that decide whether
in-model exactness is ever worth paying for, versus host compute.

Correctness gate: 60 chip cases (10 per op, fresh seed) must ALL
match ref before any chip timing is reported. A model that fails
the gate gets no timing numbers (timing wrong answers is meaningless).
Baselines are timed on the same VM in the same run; they are
model-independent, so they are reported per invocation and should
roughly agree across the two model runs.

Energy: NOT measured as a total. Free Colab VMs expose no reliable
energy accounting; nvidia-smi power draw is recorded only as two
instantaneous snapshots (idle, mid-chip-loop) and labelled as such.
Do not turn those snapshots into an energy claim.

Usage: python3 eff.py MERGED_DIR BASE_DIR --out FILE
Writes JSON incrementally (partial after each phase).
"""
import argparse, json, random, subprocess, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ref

def log(*a, **kw): print(*a, flush=True)

def med(xs):
    xs = sorted(xs); n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2 if n else None

def p90(xs):
    xs = sorted(xs); return xs[int(0.9 * (len(xs) - 1))] if xs else None

def power_snapshot():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return float(out.splitlines()[0])
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("merged"); ap.add_argument("base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261105)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForImageTextToText
    meta = json.loads(Path(args.merged, "chipformer.json").read_text())
    v0 = meta["chip_token_offset"]
    assert meta.get("chip_vocab") == 248, meta
    res = {"merged": args.merged, "base": args.base, "chip_vocab": meta["chip_vocab"],
           "hidden": meta["hidden"], "layout": meta.get("layout"),
           "power_idle_w": power_snapshot()}
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    def dump():
        out_path.write_text(json.dumps(res, indent=2))
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(args.merged, dtype=torch.bfloat16).to("cuda").eval()
    res["load_seconds"] = time.time() - t0
    res["params"] = sum(p.numel() for p in model.parameters())
    res["bf16_gb"] = res["params"] * 2 / 1e9
    res["vram_after_load_mib"] = torch.cuda.memory_allocated() / 2**20
    log("loaded", res["params"], "params", res["bf16_gb"], "GB, load", res["load_seconds"], "s, VRAM", res["vram_after_load_mib"], "MiB", flush=True)
    dump()

    def gen_until_cached(seq, cap):
        with torch.no_grad():
            out = model(input_ids=torch.tensor([seq], device="cuda"), use_cache=True)
        past = out.past_key_values; nxt = int(torch.argmax(out.logits[0, -1]).item())
        gen = [nxt]
        while nxt != v0 + ref.END and len(gen) < cap:
            with torch.no_grad():
                out = model(input_ids=torch.tensor([[nxt]], device="cuda"), past_key_values=past, use_cache=True)
            past = out.past_key_values; nxt = int(torch.argmax(out.logits[0, -1]).item()); gen.append(nxt)
        return gen

    def gen_stepwise(seq, cap):
        seq = list(seq); gen = []
        for _ in range(cap):
            with torch.no_grad():
                out = model(input_ids=torch.tensor([seq], device="cuda"))
            nxt = int(torch.argmax(out.logits[0, -1]).item())
            gen.append(nxt); seq.append(nxt)
            if nxt == v0 + ref.END: break
        return gen

    def trim(gen):
        return gen[:gen.index(v0 + ref.END) + 1] if v0 + ref.END in gen else gen

    rng = random.Random(args.seed)
    cases = []
    for op in ref.OPS:
        for _ in range(10):
            cases.append((op, [rng.randrange(16) for _ in range(4)], [rng.randrange(16) for _ in range(4)]))
    # Correctness gate + cached timing (all 60).
    rows, t_cached = [], []
    for i, (op, A, B) in enumerate(cases):
        inp = [v0 + t for t in ref.encode(op, A, B)]
        exp = [v0 + t for t in ref.expected_output(op, A, B)]
        t1 = time.time(); g = trim(gen_until_cached(inp, 6)); dt = time.time() - t1
        ok = g == exp
        rows.append({"op": op, "A": A, "B": B, "ok": ok, "seconds": dt})
        t_cached.append(dt)
        if i == 30:
            res["power_mid_chip_w"] = power_snapshot()
        log(f"CHIP {i} {op} ok={ok} {dt:.3f}s", flush=True)
    res["chip_cases"] = rows
    res["chip_correct"] = sum(r["ok"] for r in rows)
    res["chip_cached_median_s"] = med(t_cached)
    res["chip_cached_p90_s"] = p90(t_cached)
    res["chip_cached_mean_s"] = sum(t_cached) / len(t_cached)
    res["chip_cached_per_case_throughput"] = 1.0 / res["chip_cached_median_s"] if res["chip_cached_median_s"] else None
    res["vram_max_mib"] = torch.cuda.max_memory_allocated() / 2**20
    dump()
    if res["chip_correct"] != len(cases):
        log("GATE FAILED: chip correctness", res["chip_correct"], "/", len(cases), "- NO timing comparison reported", flush=True)
        res["gate"] = "FAILED"
        dump()
        return
    res["gate"] = "PASSED"
    # Stepwise timing on a 12-case subset (2 per op), correctness re-checked.
    t_step = []
    for op, A, B in cases[::5]:
        inp = [v0 + t for t in ref.encode(op, A, B)]
        exp = [v0 + t for t in ref.expected_output(op, A, B)]
        t1 = time.time(); g = gen_stepwise(inp, 6); dt = time.time() - t1
        assert g == exp, (op, A, B, g, exp)
        t_step.append(dt)
    res["chip_stepwise_median_s"] = med(t_step)
    res["chip_stepwise_n"] = len(t_step)
    log("stepwise median", res["chip_stepwise_median_s"], "n", len(t_step), flush=True)
    dump()
    # Baselines on the same VM.
    import numpy as np
    N = 100000
    a_np = np.random.randint(0, 16, size=(N, 4)); b_np = np.random.randint(0, 16, size=(N, 4))
    np_res = {}
    for op in ref.OPS:
        if op == "ADD": f = lambda: (a_np + b_np) % 16
        elif op == "SUB": f = lambda: (a_np - b_np) % 16
        elif op == "MUL": f = lambda: (a_np * b_np) % 16
        elif op == "MAX": f = lambda: np.maximum(a_np, b_np)
        elif op == "MIN": f = lambda: np.minimum(a_np, b_np)
        else: f = lambda: np.sum(a_np * b_np, axis=1)
        f(); t1 = time.time(); f(); dt = time.time() - t1
        np_res[op] = {"seconds_total": dt, "seconds_per_case": dt / N}
    res["numpy_N"] = N; res["numpy"] = np_res
    log("numpy per-case (us):", {k: round(v["seconds_per_case"] * 1e6, 3) for k, v in np_res.items()}, flush=True)
    # Pure-Python ref loop (2000 cases, same ref.py the chip is checked against).
    py_cases = [(op, [rng.randrange(16) for _ in range(4)], [rng.randrange(16) for _ in range(4)])
                for op in ref.OPS for _ in range(2000 // 6 + 1)][:2000]
    t1 = time.time()
    for op, A, B in py_cases:
        if op == "DOT": sum(x * y for x, y in zip(A, B))
        else: ref.lane_result(op, A, B)
    dt = time.time() - t1
    res["python_loop_N"] = len(py_cases); res["python_seconds_per_case"] = dt / len(py_cases)
    log("python per-case (us):", res["python_seconds_per_case"] * 1e6, flush=True)
    # Torch CPU + CUDA, vectorised, with synchronisation.
    a_t = torch.randint(0, 16, (N, 4)); b_t = torch.randint(0, 16, (N, 4))
    def torch_ops(a, b):
        return {"ADD": (a + b) % 16, "SUB": (a - b) % 16, "MUL": (a * b) % 16,
                "MAX": torch.maximum(a, b), "MIN": torch.minimum(a, b),
                "DOT": torch.sum(a * b, dim=1)}
    for device, label in (("cpu", "torch_cpu"), ("cuda", "torch_cuda")):
        aa, bb = a_t.to(device), b_t.to(device)
        torch_ops(aa, bb)
        if device == "cuda": torch.cuda.synchronize()
        t1 = time.time(); torch_ops(aa, bb)
        if device == "cuda": torch.cuda.synchronize()
        dt = time.time() - t1
        res[label] = {"seconds_total": dt, "seconds_per_case": dt / N}
        log(label, "per-case (us):", dt / N * 1e6, flush=True)
    # Ratios: chip median latency vs each baseline's per-case time.
    np_med = med([v["seconds_per_case"] for v in np_res.values()])
    res["ratios_vs_chip_cached_median"] = {
        "numpy_median_per_case": res["chip_cached_median_s"] / np_med if np_med else None,
        "python_per_case": res["chip_cached_median_s"] / res["python_seconds_per_case"],
        "torch_cpu_per_case": res["chip_cached_median_s"] / res["torch_cpu"]["seconds_per_case"],
        "torch_cuda_per_case": res["chip_cached_median_s"] / res["torch_cuda"]["seconds_per_case"],
        "stepwise_over_cached": res["chip_stepwise_median_s"] / res["chip_cached_median_s"],
    }
    dump()
    log("DONE ratios:", res["ratios_vs_chip_cached_median"], flush=True)

if __name__ == "__main__":
    try: main()
    except Exception:
        traceback.print_exc()
        Path("/content/vec4-eff/eff_error.txt").write_text(traceback.format_exc())
        raise
