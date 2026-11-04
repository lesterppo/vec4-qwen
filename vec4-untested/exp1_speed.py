"""Experiment 1 — SPEED. Stepwise vs cached generation, same 20 tool cases.
Cached driver = per-phase KV cache: one full forward per generation phase,
then single-token forwards with past_key_values (transformers 5.x Cache).
Host injections (tool_response) start a new phase with one full forward,
so no cross-phase cache object is carried. Chip stock generate() is tested
separately. Writes incremental JSON after every case/phase.
Usage: python3 exp1_speed.py [--cached-only] (stepwise subset always runs)
"""
import json, re, sys, time, traceback
from pathlib import Path
sys.path.insert(0, "/content/qwenputer-colab/vec4-untested")
import ref
MERGED = "/content/qwen35-08b-vec4"
BASE = "/content/qwen35-08b"
OUT = Path("/content/vec4-untested"); OUT.mkdir(parents=True, exist_ok=True)
def log(*a, **kw):
    print(*a, flush=True)
def tool_def(name):
    desc = ("Add two 4-lane vectors lane by lane, modulo 16. Each vector has exactly 4 integers, each 0 to 15."
            if name == "vec_add" else
            "Dot product of two 4-lane vectors: sum over lanes of a[i]*b[i]. Each vector has exactly 4 integers, each 0 to 15. Maximum result is 900.")
    return {"type": "function", "function": {"name": name, "description": desc,
        "parameters": {"type": "object", "properties": {"a": {"type": "array", "items": {"type": "integer"}}, "b": {"type": "array", "items": {"type": "integer"}}}, "required": ["a", "b"]}}}
def system_single(name):
    return f"You have tool {name} for exact vector arithmetic. Use it for every vector calculation instead of calculating by hand."
def parse_value(s):
    s = s.strip()
    try: return json.loads(s)
    except Exception: pass
    nums = re.findall(r"-?\d+", s)
    return [int(x) for x in nums] if nums else s
def parse_tool_call(text):
    m = re.search(r"<tool_call>\s*<function=([A-Za-z0-9_]+)>(.*?)</function>\s*</tool_call>", text, re.DOTALL)
    if m:
        name, body = m.group(1), m.group(2); args = {}
        for pm in re.finditer(r"<parameter=([A-Za-z0-9_]+)>\s*(.*?)\s*</parameter>", body, re.DOTALL):
            args[pm.group(1)] = parse_value(pm.group(2))
        ok = name in ("vec_add", "vec_dot") and all(k in args and isinstance(args[k], list) and len(args[k]) == 4 and all(isinstance(x, int) and 0 <= x <= 15 for x in args[k]) for k in ("a", "b"))
        return (name if ok else None, args if ok else f"bad/unknown call name={name} args={args}", "xml", ok)
    if re.search(r'"name"\s*:\s*"(vec_add|vec_dot)"', text):
        return None, "json-style call (not parsed as XML)", "json", False
    return None, "no tool_call block found", "none", False
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
def expected_str(op, A, B):
    return str(sum(a*b for a, b in zip(A, B))) if op == "DOT" else str([(a+b) % 16 for a, b in zip(A, B)])
def user_prompt(op, A, B):
    name = "vec_dot" if op == "DOT" else "vec_add"
    return f"What is the dot product of {A} and {B}? Use {name}." if op == "DOT" else f"Use {name} to add {A} and {B} lane by lane, modulo 16."
def main():
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    t_start = time.time()
    meta = json.loads(Path(MERGED + "/chipformer.json").read_text()); v0 = meta["chip_token_offset"]
    tok = AutoTokenizer.from_pretrained(BASE)
    eos_ids = set()
    for tname in ("<|im_end|>", "<|endoftext|>"):
        try: eos_ids.update(tok.encode(tname, add_special_tokens=False))
        except Exception: pass
    if tok.eos_token_id is not None: eos_ids.add(tok.eos_token_id)
    model = AutoModelForImageTextToText.from_pretrained(MERGED, dtype=torch.bfloat16).to("cuda").eval()
    log("loaded VRAM MiB", torch.cuda.memory_allocated()/2**20, flush=True)
    def next_stepwise(seq):
        with torch.no_grad():
            out = model(input_ids=torch.tensor([seq], device="cuda"))
        return int(torch.argmax(out.logits[0, -1]).item())
    def gen_stepwise(seq, cap, stop_tool=False):
        seq = list(seq); gen = []
        for _ in range(cap):
            nxt = next_stepwise(seq); gen.append(nxt); seq.append(nxt)
            if nxt in eos_ids: break
            if stop_tool and "</tool_call>" in tok.decode(gen, skip_special_tokens=False): break
        return gen, seq
    def gen_cached(seq, cap, stop_tool=False):
        # One full forward, then single-token forwards with past_key_values.
        seq = list(seq)
        with torch.no_grad():
            out = model(input_ids=torch.tensor([seq], device="cuda"), use_cache=True)
        past = out.past_key_values; gen = []
        nxt = int(torch.argmax(out.logits[0, -1]).item())
        for _ in range(cap):
            gen.append(nxt); seq.append(nxt)
            if nxt in eos_ids: break
            if stop_tool and "</tool_call>" in tok.decode(gen, skip_special_tokens=False): break
            with torch.no_grad():
                out = model(input_ids=torch.tensor([[nxt]], device="cuda"), past_key_values=past, use_cache=True)
            past = out.past_key_values; nxt = int(torch.argmax(out.logits[0, -1]).item())
        return gen, seq
    def chip_stepwise(op, A, B):
        inp = [v0 + t for t in ref.encode(op, A, B)]; exp = [v0 + t for t in ref.expected_output(op, A, B)]
        seq, gen = list(inp), []
        for _ in range(6):
            nxt = next_stepwise(seq); gen.append(nxt); seq.append(nxt)
            if nxt == v0 + ref.END: break
        return gen == exp, gen
    def chip_generate(op, A, B):
        inp = [v0 + t for t in ref.encode(op, A, B)]; exp = [v0 + t for t in ref.expected_output(op, A, B)]
        with torch.no_grad():
            out = model.generate(input_ids=torch.tensor([inp], device="cuda"), max_new_tokens=6,
                                 do_sample=False, use_cache=True, eos_token_id=v0 + ref.END, pad_token_id=v0 + ref.END)
        gen = [int(x) for x in out[0][len(inp):].tolist()]
        return gen == exp, gen
    def chip_cached(op, A, B):
        inp = [v0 + t for t in ref.encode(op, A, B)]; exp = [v0 + t for t in ref.expected_output(op, A, B)]
        gen, _ = gen_cached(inp, 6)
        return gen == exp, gen
    def seg(result_str):
        s = "<|im_end|>\n<|im_start|>user\n<tool_response>\n" + result_str + "\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return tok.encode(s, add_special_tokens=False)
    def prompt_ids(user_text, tools, system):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_text}]
        r = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, enable_thinking=False, tokenize=True)
        return [int(x) for x in (r["input_ids"] if hasattr(r, "input_ids") else r)]
    def run_tool_case(op, A, B, gen_fn, label):
        name_req = "vec_dot" if op == "DOT" else "vec_add"
        user = user_prompt(op, A, B); exp_str = expected_str(op, A, B)
        pid = prompt_ids(user, [tool_def(name_req)], system_single(name_req))
        t0 = time.time()
        gen, seq = gen_fn(list(pid), 120, stop_tool=True)
        gen_text = tok.decode(gen, skip_special_tokens=False)
        name, parsed, fmt, ok = parse_tool_call(gen_text)
        rec = {"op": op, "A": A, "B": B, "user": user, "driver": label, "tool_text": gen_text[:2000],
               "called_tool": name, "called_args": parsed if isinstance(parsed, dict) else str(parsed),
               "format": fmt, "parse_ok": ok, "correct_tool_and_args": False, "chip_ok_requested": None,
               "final_text": None, "final_raw": None, "strict": False, "prompt_len": len(pid)}
        if ok and name == name_req and parsed["a"] == A and parsed["b"] == B:
            rec["correct_tool_and_args"] = True
            # chip phase uses the SAME driver family for its generation phase
            if label == "stepwise":
                cok, cgen = chip_stepwise(op, A, B)
            else:
                cok, cgen = chip_cached(op, A, B)
            rec["chip_ok_requested"] = cok; rec["chip_gen"] = [g - v0 for g in cgen]
            gen2, _ = gen_fn(seq + seg(exp_str), 120, stop_tool=False)
            rec["final_text"] = tok.decode(gen2, skip_special_tokens=True)[:2000]
            rec["final_raw"] = tok.decode(gen2, skip_special_tokens=False)[:2000]
            rec["second_call_loop"] = "<tool_call>" in rec["final_raw"]
            rec["strict"] = bool(cok and (exp_str in rec["final_text"]) and ("<tool_call>" not in rec["final_raw"]))
        rec["seconds"] = time.time() - t0
        return rec
    all_cases = cases_exact()
    # ---- chip-only speed/exactness: stepwise vs stock generate vs cached, 20 tool cases ----
    chip_rows = []
    for idx, (op, A, B) in enumerate(all_cases):
        t0 = time.time(); ok_s, g_s = chip_stepwise(op, A, B); ts = time.time() - t0
        try:
            t0 = time.time(); ok_g, g_g = chip_generate(op, A, B); tg = time.time() - t0; g_err = None
        except Exception as e:
            ok_g, g_g, tg, g_err = None, None, None, repr(e)
        try:
            t0 = time.time(); ok_c, g_c = chip_cached(op, A, B); tc = time.time() - t0; c_err = None
        except Exception as e:
            ok_c, g_c, tc, c_err = None, None, None, repr(e)
        row = {"case": idx, "op": op, "A": A, "B": B, "stepwise_ok": ok_s, "stepwise_s": ts,
               "generate_ok": ok_g, "generate_s": tg, "generate_err": g_err,
               "cached_ok": ok_c, "cached_s": tc, "cached_err": c_err,
               "generate_tokens_match_stepwise": (g_g == g_s if g_g is not None else None),
               "cached_tokens_match_stepwise": (g_c == g_s if g_c is not None else None)}
        chip_rows.append(row)
        log(f"CHIP {idx} {op} step={ok_s} {ts:.2f}s gen={ok_g} {tg if tg else 0:.2f}s cache={ok_c} {tc if tc else 0:.2f}s match_gen={row['generate_tokens_match_stepwise']} match_cache={row['cached_tokens_match_stepwise']}", flush=True)
        (OUT / "exp1_chip_partial.json").write_text(json.dumps({"chip_rows": chip_rows, "vram_max_mib": torch.cuda.max_memory_allocated()/2**20}, indent=2))
    # ---- tool cases: cached full 20 + stepwise subset 6 ----
    cached_results = []
    for idx, (op, A, B) in enumerate(all_cases):
        rec = run_tool_case(op, A, B, gen_cached, "cached"); rec["case"] = idx
        cached_results.append(rec)
        log(f"CACHED TOOL {idx} {op} tool={rec['called_tool']} chip={rec['chip_ok_requested']} strict={rec['strict']} {rec['seconds']:.1f}s final={(rec['final_text'] or '')[:120]!r}", flush=True)
        (OUT / "exp1_cached_partial.json").write_text(json.dumps({"cases": cached_results, "chip_rows": chip_rows, "vram_max_mib": torch.cuda.max_memory_allocated()/2**20}, indent=2))
    subset_idx = [0, 2, 5, 10, 12, 17]
    stepwise_results = []
    for idx in subset_idx:
        op, A, B = all_cases[idx]
        rec = run_tool_case(op, A, B, gen_stepwise, "stepwise"); rec["case"] = idx
        stepwise_results.append(rec)
        log(f"STEPWISE TOOL {idx} {op} tool={rec['called_tool']} chip={rec['chip_ok_requested']} strict={rec['strict']} {rec['seconds']:.1f}s final={(rec['final_text'] or '')[:120]!r}", flush=True)
        (OUT / "exp1_stepwise_partial.json").write_text(json.dumps({"cases": stepwise_results, "vram_max_mib": torch.cuda.max_memory_allocated()/2**20}, indent=2))
    def med(xs):
        xs = sorted(xs); n = len(xs); return xs[n//2] if n % 2 else (xs[n//2 - 1] + xs[n//2]) / 2
    out = {"experiment": "1_speed", "v0": v0, "seconds_total": time.time() - t_start,
           "vram_max_mib": torch.cuda.max_memory_allocated()/2**20,
           "chip_rows": chip_rows,
           "chip_stepwise_match": sum(1 for r in chip_rows if r["stepwise_ok"]),
           "chip_generate_match": sum(1 for r in chip_rows if r["generate_ok"]),
           "chip_cached_match": sum(1 for r in chip_rows if r["cached_ok"]),
           "chip_generate_tokens_match_stepwise": sum(1 for r in chip_rows if r["generate_tokens_match_stepwise"]),
           "chip_cached_tokens_match_stepwise": sum(1 for r in chip_rows if r["cached_tokens_match_stepwise"]),
           "chip_stepwise_median_s": med([r["stepwise_s"] for r in chip_rows]),
           "chip_generate_median_s": med([r["generate_s"] for r in chip_rows if r["generate_s"]]),
           "chip_cached_median_s": med([r["cached_s"] for r in chip_rows if r["cached_s"]]),
           "cached_tool_cases": cached_results, "cached_tool_strict": sum(1 for r in cached_results if r["strict"]),
           "cached_tool_seconds_total": sum(r["seconds"] for r in cached_results),
           "cached_tool_seconds_median": med([r["seconds"] for r in cached_results]),
           "stepwise_tool_subset": stepwise_results, "stepwise_tool_subset_strict": sum(1 for r in stepwise_results if r["strict"]),
           "stepwise_tool_subset_seconds_median": med([r["seconds"] for r in stepwise_results])}
    (OUT / "exp1_speed_result.json").write_text(json.dumps(out, indent=2))
    log("WROTE exp1_speed_result.json cached_strict", out["cached_tool_strict"], "/20 stepwise_subset_strict", out["stepwise_tool_subset_strict"], "/6", flush=True)
if __name__ == "__main__":
    try: main()
    except Exception:
        traceback.print_exc(); (OUT / "exp1_error.txt").write_text(traceback.format_exc()); raise
