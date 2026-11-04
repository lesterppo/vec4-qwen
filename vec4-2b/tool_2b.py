"""Step 4 for vec4-2b: tool tests on the merged 2B model.
Configs (each --only runnable, incremental JSON, strict scoring):
  baseline: both tools, vec_add first, single attempt, no guard
            (wrong tool -> no chip, strict False; mirrors 0.8B main run)
  A: single tool listed per case (mirrors vec4-fix Variant A)
  B: guard+retry, both tools, DOT only, B1/B2 counted separately
     (mirrors vec4-fix Variant B + vec4-untested B2)
  plain: 5 plain prompts, both tools listed (mirrors vec4-toolcall)
  notool: 5 should-not-call prompts, both tools listed
Driver: cached per-phase KV cache by default (proven by
cache_probe_2b.py); --driver stepwise falls back if the probe failed.
Toolcall bridge only: host decodes chip OUT, re-injects text
tool_response. No interleaved reporting.
Strict = requested op == called tool AND args == case args AND chip
for requested op matches ref AND final English contains correct
requested result AND final raw has no <tool_call> block.
B2 is NEVER counted as B1."""
import json, re, sys, time, traceback
from pathlib import Path
sys.path.insert(0, "/content/qwenputer-colab/vec4-2b")
import ref

MERGED = "/content/qwen35-2b-vec4"
BASE = "/content/qwen35-2b"
OUT = Path("/content/vec4-2b"); OUT.mkdir(parents=True, exist_ok=True)

def tool_def(name):
    if name == "vec_add":
        desc = "Add two 4-lane vectors lane by lane, modulo 16. Each vector has exactly 4 integers, each 0 to 15."
    else:
        desc = "Dot product of two 4-lane vectors: sum over lanes of a[i]*b[i]. Each vector has exactly 4 integers, each 0 to 15. Maximum result is 900."
    return {"type": "function", "function": {"name": name, "description": desc,
        "parameters": {"type": "object", "properties": {"a": {"type": "array", "items": {"type": "integer"}}, "b": {"type": "array", "items": {"type": "integer"}}}, "required": ["a", "b"]}}}

TOOLS_BOTH = [tool_def("vec_add"), tool_def("vec_dot")]  # main-run order
SYSTEM_BOTH = "You have tools vec_add and vec_dot for exact vector arithmetic. Use them for every vector calculation instead of calculating by hand."
def system_single(name):
    return f"You have tool {name} for exact vector arithmetic. Use it for every vector calculation instead of calculating by hand."

def log(*a, **kw): print(*a, flush=True)

def parse_value(s):
    s = s.strip()
    try: return json.loads(s)
    except Exception: pass
    nums = re.findall(r"-?\d+", s)
    return [int(x) for x in nums] if nums else s

def parse_tool_call(text):
    m = re.search(r"<tool_call>\s*<function=([A-Za-z0-9_]+)>(.*?)</function>\s*</tool_call>", text, re.DOTALL)
    if m:
        name, body = m.group(1), m.group(2)
        args = {}
        for pm in re.finditer(r"<parameter=([A-Za-z0-9_]+)>\s*(.*?)\s*</parameter>", body, re.DOTALL):
            args[pm.group(1)] = parse_value(pm.group(2))
        ok = name in ("vec_add", "vec_dot") and all(k in args and isinstance(args[k], list) and len(args[k]) == 4 and all(isinstance(x, int) and 0 <= x <= 15 for x in args[k]) for k in ("a", "b"))
        return (name if ok else None, args if ok else f"bad/unknown call name={name} args={args}", "xml", ok)
    m2 = re.search(r'"name"\s*:\s*"(vec_add|vec_dot)"', text)
    if m2: return None, "json-style call (not parsed as XML)", "json", False
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
    if op == "DOT": return str(sum(a*b for a, b in zip(A, B)))
    return str([(a+b) % 16 for a, b in zip(A, B)])

def user_prompt(op, A, B):
    name = "vec_dot" if op == "DOT" else "vec_add"
    if op == "DOT": return f"What is the dot product of {A} and {B}? Use {name}."
    return f"Use {name} to add {A} and {B} lane by lane, modulo 16."

PLAIN_PROMPTS = ["What is the capital of France?", "Tell me a short joke about cats.",
                 "Explain what a vector is in one sentence.", "What is the capital of Japan?",
                 "Write a one-line greeting."]
NOTOOL_PROMPTS = ["What is 2+2 in words?",
                  "Explain what a dot product is in one sentence, without calculating anything.",
                  "What is your favourite colour?", "Summarise the plot of Hamlet in one sentence.",
                  "Is Paris the capital of France? Answer yes or no."]

def main():
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    only = None
    if "--only" in sys.argv: only = sys.argv[sys.argv.index("--only") + 1]
    driver = "cached"
    if "--driver" in sys.argv: driver = sys.argv[sys.argv.index("--driver") + 1]
    assert driver in ("cached", "stepwise"), driver
    t_start = time.time()
    va = json.loads(Path("/content/qwenputer-colab/vec4-fix/variant_a_result.json").read_text())
    all_cases = cases_exact()
    assert len(va["cases"]) == 20
    mism = [i for i, c in enumerate(va["cases"]) if (c["op"], c["A"], c["B"]) != all_cases[i]]
    assert not mism, f"case mismatch vs variant_a_result.json: {mism}"
    log("cases verified 20/20 vs variant_a_result.json; driver", driver)
    meta = json.loads(Path(MERGED + "/chipformer.json").read_text())
    v0 = meta["chip_token_offset"]
    tok = AutoTokenizer.from_pretrained(BASE)
    eos_ids = set()
    for tname in ("<|im_end|>", "<|endoftext|>"):
        try: eos_ids.update(tok.encode(tname, add_special_tokens=False))
        except Exception: pass
    if tok.eos_token_id is not None: eos_ids.add(tok.eos_token_id)
    model = AutoModelForImageTextToText.from_pretrained(MERGED, dtype=torch.bfloat16).to("cuda").eval()
    log("loaded VRAM MiB", torch.cuda.memory_allocated()/2**20, flush=True)

    def next_token_stepwise(seq):
        with torch.no_grad():
            out = model(input_ids=torch.tensor([seq], device="cuda"))
            return int(torch.argmax(out.logits[0, -1]).item())

    def gen_until(seq, cap, stop_tool=False, stop_chip_end=False):
        """One generation phase. Cached: one full forward, then
        single-token forwards with past_key_values; each host injection
        starts a new phase, so no cache crosses an injection."""
        seq = list(seq); gen = []
        if driver == "stepwise":
            for _ in range(cap):
                nxt = next_token_stepwise(seq); gen.append(nxt); seq.append(nxt)
                if nxt in eos_ids: break
                if stop_chip_end and nxt == v0 + ref.END: break
                if stop_tool and "</tool_call>" in tok.decode(gen, skip_special_tokens=False): break
            return gen, seq
        with torch.no_grad():
            out = model(input_ids=torch.tensor([seq], device="cuda"), use_cache=True)
        past = out.past_key_values
        nxt = int(torch.argmax(out.logits[0, -1]).item())
        for _ in range(cap):
            gen.append(nxt); seq.append(nxt)
            if nxt in eos_ids: break
            if stop_chip_end and nxt == v0 + ref.END: break
            if stop_tool and "</tool_call>" in tok.decode(gen, skip_special_tokens=False): break
            with torch.no_grad():
                out = model(input_ids=torch.tensor([[nxt]], device="cuda"), past_key_values=past, use_cache=True)
            past = out.past_key_values; nxt = int(torch.argmax(out.logits[0, -1]).item())
        return gen, seq

    def prompt_ids(user_text, tools, system):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_text}]
        r = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, enable_thinking=False, tokenize=True)
        return [int(x) for x in (r["input_ids"] if hasattr(r, "input_ids") else r)]

    def chip_run(op, A, B):
        inp = [v0 + t for t in ref.encode(op, A, B)]
        exp = [v0 + t for t in ref.expected_output(op, A, B)]
        gen, _ = gen_until(inp, 6, stop_chip_end=True)
        gen_trim = gen[:gen.index(v0 + ref.END) + 1] if v0 + ref.END in gen else gen
        return [g - v0 for g in gen_trim], gen_trim == exp

    def seg(result_str):
        s = "<|im_end|>\n<|im_start|>user\n<tool_response>\n" + result_str + "\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return tok.encode(s, add_special_tokens=False)

    def strict_final(final, final_raw, exp_str):
        return (exp_str in final) and ("<tool_call>" not in final_raw)

    def med(xs):
        xs = sorted(xs); n = len(xs)
        return xs[n//2] if n % 2 else (xs[n//2-1]+xs[n//2])/2 if n else None

    # ---------- C-baseline ----------
    if only in (None, "baseline"):
        results = []
        for idx, (op, A, B) in enumerate(all_cases):
            t_c = time.time()
            name_req = "vec_dot" if op == "DOT" else "vec_add"
            user = user_prompt(op, A, B); exp_str = expected_str(op, A, B)
            pid = prompt_ids(user, TOOLS_BOTH, SYSTEM_BOTH)
            gen, seq = gen_until(list(pid), cap=120, stop_tool=True)
            gen_text = tok.decode(gen, skip_special_tokens=False)
            name, parsed, fmt, ok = parse_tool_call(gen_text)
            rec = {"case": idx, "op": op, "A": A, "B": B, "user": user,
                   "tools_listed": ["vec_add", "vec_dot"], "model_gen_text": gen_text[:2000],
                   "called_tool": name, "called_args": parsed if isinstance(parsed, dict) else str(parsed),
                   "format": fmt, "parse_ok": ok, "correct_tool_and_args": False,
                   "chip_ok_requested": None, "final_text": None, "final_raw": None,
                   "second_call_loop": False, "strict": False, "seconds": None}
            if ok and name == name_req and parsed["a"] == A and parsed["b"] == B:
                rec["correct_tool_and_args"] = True
                gen_chip, chip_ok = chip_run(op, A, B)
                rec["chip_ok_requested"] = chip_ok; rec["chip_gen"] = gen_chip
                gen2, _ = gen_until(seq + seg(exp_str), cap=120, stop_tool=False)
                rec["final_text"] = tok.decode(gen2, skip_special_tokens=True)[:2000]
                rec["final_raw"] = tok.decode(gen2, skip_special_tokens=False)[:2000]
                rec["second_call_loop"] = "<tool_call>" in rec["final_raw"]
                rec["strict"] = bool(chip_ok and strict_final(rec["final_text"], rec["final_raw"], exp_str))
            rec["seconds"] = time.time() - t_c
            results.append(rec)
            (OUT / "c_baseline_partial.json").write_text(json.dumps(
                {"variant": "C_baseline", "partial": True, "cases_done": len(results), "v0": v0,
                 "seconds_total": time.time() - t_start, "vram_max_mib": torch.cuda.max_memory_allocated()/2**20,
                 "cases": results}, indent=2))
            log(f"BASELINE CASE {idx} {op} called={name} args_ok={rec['correct_tool_and_args']} chip={rec['chip_ok_requested']} strict={rec['strict']} s={rec['seconds']:.1f} final={(rec['final_text'] or '')[:150]!r}", flush=True)
        out = {"variant": "C_baseline", "driver": driver, "v0": v0, "seconds_total": time.time() - t_start,
               "vram_max_mib": torch.cuda.max_memory_allocated()/2**20,
               "median_seconds_per_case": med([r["seconds"] for r in results]), "cases": results}
        (OUT / "c_baseline_result.json").write_text(json.dumps(out, indent=2))
        log("WROTE c_baseline_result.json", flush=True)
    # ---------- C-A ----------
    if only in (None, "A"):
        results = []
        for idx, (op, A, B) in enumerate(all_cases):
            t_c = time.time()
            name_req = "vec_dot" if op == "DOT" else "vec_add"
            user = user_prompt(op, A, B); exp_str = expected_str(op, A, B)
            pid = prompt_ids(user, [tool_def(name_req)], system_single(name_req))
            gen, seq = gen_until(list(pid), cap=120, stop_tool=True)
            gen_text = tok.decode(gen, skip_special_tokens=False)
            name, parsed, fmt, ok = parse_tool_call(gen_text)
            rec = {"case": idx, "op": op, "A": A, "B": B, "user": user, "tools_listed": [name_req],
                   "model_gen_text": gen_text[:2000], "called_tool": name,
                   "called_args": parsed if isinstance(parsed, dict) else str(parsed),
                   "format": fmt, "parse_ok": ok, "correct_tool_and_args": False,
                   "chip_ok_requested": None, "final_text": None, "final_raw": None, "strict": False, "seconds": None}
            if ok and name == name_req and parsed["a"] == A and parsed["b"] == B:
                rec["correct_tool_and_args"] = True
                gen_chip, chip_ok = chip_run(op, A, B)
                rec["chip_ok_requested"] = chip_ok; rec["chip_gen"] = gen_chip
                gen2, _ = gen_until(seq + seg(exp_str), cap=120, stop_tool=False)
                rec["final_text"] = tok.decode(gen2, skip_special_tokens=True)[:2000]
                rec["final_raw"] = tok.decode(gen2, skip_special_tokens=False)[:2000]
                rec["second_call_loop"] = "<tool_call>" in rec["final_raw"]
                rec["strict"] = bool(chip_ok and strict_final(rec["final_text"], rec["final_raw"], exp_str))
            else:
                rec["second_call_loop"] = False
            rec["seconds"] = time.time() - t_c
            results.append(rec)
            (OUT / "c_a_partial.json").write_text(json.dumps(
                {"variant": "C_A_single_tool", "partial": True, "cases_done": len(results), "v0": v0,
                 "seconds_total": time.time() - t_start, "vram_max_mib": torch.cuda.max_memory_allocated()/2**20,
                 "cases": results}, indent=2))
            log(f"A CASE {idx} {op} called={name} args_ok={rec['correct_tool_and_args']} chip={rec['chip_ok_requested']} strict={rec['strict']} s={rec['seconds']:.1f} final={(rec['final_text'] or '')[:150]!r}", flush=True)
        out = {"variant": "C_A_single_tool", "driver": driver, "v0": v0, "seconds_total": time.time() - t_start,
               "vram_max_mib": torch.cuda.max_memory_allocated()/2**20,
               "median_seconds_per_case": med([r["seconds"] for r in results]), "cases": results}
        (OUT / "c_a_result.json").write_text(json.dumps(out, indent=2))
        log("WROTE c_a_result.json", flush=True)
    # ---------- C-B ----------
    if only in (None, "B"):
        results = []
        dot_cases = [c for c in all_cases if c[0] == "DOT"]
        for idx, (op, A, B) in enumerate(dot_cases):
            t_c = time.time()
            name_req = "vec_dot"; user = user_prompt(op, A, B); exp_str = expected_str(op, A, B)
            pid = prompt_ids(user, TOOLS_BOTH, SYSTEM_BOTH)
            gen, seq = gen_until(list(pid), cap=120, stop_tool=True)
            gen_text = tok.decode(gen, skip_special_tokens=False)
            name, parsed, fmt, ok = parse_tool_call(gen_text)
            rec = {"case": idx, "op": op, "A": A, "B": B, "user": user, "tools_listed": ["vec_add", "vec_dot"],
                   "attempt1_text": gen_text[:2000], "attempt1_tool": name,
                   "attempt1_args": parsed if isinstance(parsed, dict) else str(parsed),
                   "attempt1_format": fmt, "attempt1_parse_ok": ok, "attempts": 1, "outcome": None,
                   "attempt2_text": None, "attempt2_tool": None, "attempt2_args": None,
                   "chip_ok_requested": None, "final_text": None, "final_raw": None,
                   "second_call_loop": False, "strict_B1": False, "strict_B2": False, "seconds": None}
            def finish_with_chip(seq_after_tool, method_outcome):
                gen_chip, chip_ok = chip_run(op, A, B)
                rec["chip_ok_requested"] = chip_ok; rec["chip_gen"] = gen_chip
                gen2, _ = gen_until(seq_after_tool + seg(exp_str), cap=120, stop_tool=False)
                rec["final_text"] = tok.decode(gen2, skip_special_tokens=True)[:2000]
                rec["final_raw"] = tok.decode(gen2, skip_special_tokens=False)[:2000]
                rec["second_call_loop"] = "<tool_call>" in rec["final_raw"]
                rec["outcome"] = method_outcome
                s = bool(chip_ok and strict_final(rec["final_text"], rec["final_raw"], exp_str))
                if method_outcome == "host_rewritten": rec["strict_B2"] = s
                else: rec["strict_B1"] = s
                return s
            if ok and name == name_req and parsed["a"] == A and parsed["b"] == B:
                finish_with_chip(seq, "first_attempt")
            else:
                wrong = name if name else "none"
                err = f"Error: wrong tool. The task requires vec_dot, not {wrong}. Call vec_dot with the same arguments."
                if not ok: err = "Error: no valid tool call detected. Call vec_dot with the correct arguments."
                seq_err = seq + seg(err)
                gen2, seq2 = gen_until(seq_err, cap=120, stop_tool=True)
                gen2_text = tok.decode(gen2, skip_special_tokens=False)
                name2, parsed2, fmt2, ok2 = parse_tool_call(gen2_text)
                rec["attempts"] = 2
                rec["attempt2_text"] = gen2_text[:2000]; rec["attempt2_tool"] = name2
                rec["attempt2_args"] = parsed2 if isinstance(parsed2, dict) else str(parsed2)
                rec["attempt2_format"] = fmt2; rec["attempt2_parse_ok"] = ok2
                rec["guard_error_injected"] = err
                if ok2 and name2 == name_req and parsed2["a"] == A and parsed2["b"] == B:
                    finish_with_chip(seq2, "retry_corrected")
                else:
                    rec["outcome"] = "host_rewritten"
                    msgs_full = [{"role": "system", "content": SYSTEM_BOTH}, {"role": "user", "content": user},
                                 {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": name_req, "arguments": {"a": A, "b": B}}}]},
                                 {"role": "tool", "content": exp_str, "tool_call_id": "call_1"}]
                    try:
                        r = tok.apply_chat_template(msgs_full, tools=TOOLS_BOTH, add_generation_prompt=True, enable_thinking=False, tokenize=True)
                        seq_fb = [int(x) for x in (r["input_ids"] if hasattr(r, "input_ids") else r)]
                        rec["b2_method"] = "template_rendered"
                    except Exception as e:
                        rec["b2_method"] = f"template_failed:{repr(e)}"
                        tc_text = f"<tool_call>\n<function={name_req}>\n<parameter=a>\n{json.dumps(A)}\n</parameter>\n<parameter=b>\n{json.dumps(B)}\n</parameter>\n</function>\n</tool_call>"
                        seq_fb = pid + tok.encode(tc_text, add_special_tokens=False) + seg(exp_str)
                    gen_chip, chip_ok = chip_run(op, A, B)
                    rec["chip_ok_requested"] = chip_ok; rec["chip_gen"] = gen_chip
                    gen3, _ = gen_until(seq_fb, cap=120, stop_tool=False)
                    rec["final_text"] = tok.decode(gen3, skip_special_tokens=True)[:2000]
                    rec["final_raw"] = tok.decode(gen3, skip_special_tokens=False)[:2000]
                    rec["second_call_loop"] = "<tool_call>" in rec["final_raw"]
                    rec["strict_B2"] = bool(chip_ok and strict_final(rec["final_text"], rec["final_raw"], exp_str))
            rec["seconds"] = time.time() - t_c
            results.append(rec)
            (OUT / "c_b_partial.json").write_text(json.dumps(
                {"variant": "C_B_guard_retry", "partial": True, "cases_done": len(results), "v0": v0,
                 "seconds_total": time.time() - t_start, "vram_max_mib": torch.cuda.max_memory_allocated()/2**20,
                 "cases": results}, indent=2))
            log(f"B CASE {idx} DOT attempt1={name} outcome={rec['outcome']} attempts={rec['attempts']} chip={rec['chip_ok_requested']} B1={rec['strict_B1']} B2={rec['strict_B2']} loop={rec['second_call_loop']} s={rec['seconds']:.1f} final={(rec['final_text'] or '')[:150]!r}", flush=True)
        out = {"variant": "C_B_guard_retry", "driver": driver, "v0": v0, "seconds_total": time.time() - t_start,
               "vram_max_mib": torch.cuda.max_memory_allocated()/2**20,
               "median_seconds_per_case": med([r["seconds"] for r in results]), "cases": results}
        (OUT / "c_b_result.json").write_text(json.dumps(out, indent=2))
        log("WROTE c_b_result.json", flush=True)
    # ---------- plain / notool ----------
    if only in (None, "plain"):
        # --only plain runs BOTH plain and notool, mirroring toolcall.py
        for kind, prompts in (("plain", PLAIN_PROMPTS), ("notool", NOTOOL_PROMPTS)):
            results = []
            for p in prompts:
                pid = prompt_ids(p, TOOLS_BOTH, SYSTEM_BOTH)
                gen, _ = gen_until(list(pid), cap=100, stop_tool=True)
                txt = tok.decode(gen, skip_special_tokens=True)
                txt_raw = tok.decode(gen, skip_special_tokens=False)
                results.append({"prompt": p, "text": txt[:1000],
                                "tool_call_in_raw": ("<tool_call>" in txt_raw), "gen_len": len(gen)})
                log(f"{kind.upper()} {p!r} tool={('<tool_call>' in txt_raw)} text={txt[:200]!r}", flush=True)
            (OUT / f"c_{kind}_result.json").write_text(json.dumps(
                {"variant": f"C_{kind}", "driver": driver, "v0": v0,
                 "seconds_total": time.time() - t_start,
                 "vram_max_mib": torch.cuda.max_memory_allocated()/2**20,
                 "results": results}, indent=2))
            log(f"WROTE c_{kind}_result.json", flush=True)
    log("DONE seconds", time.time() - t_start, flush=True)

if __name__ == "__main__":
    try: main()
    except Exception:
        traceback.print_exc(); (OUT / "tool_2b_error.txt").write_text(traceback.format_exc()); raise
