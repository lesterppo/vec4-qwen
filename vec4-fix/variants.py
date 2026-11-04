"""Variants A (single-tool) and B (guard+retry) for VEC4 DOT fix, STRICT scoring only.
Strict = requested op == called tool AND args == case args AND chip run for the
requested op matches ref AND final English contains the correct requested result
AND final raw contains no <tool_call> block. Loose 'tool actually called' numbers
are NOT reported as end-to-end; they are recorded only as called_tool fields.
Loads the merged 0.8B model once, runs A then B, writes variant JSONs incrementally.
Usage: python3 variants.py [--only A|B]
"""
import json, re, sys, time, traceback
from pathlib import Path
sys.path.insert(0, "/content/qwenputer-colab/vec4-fix")
import ref

MERGED = "/content/qwen35-08b-vec4"
BASE = "/content/qwen35-08b"
OUT = Path("/content/vec4-fix"); OUT.mkdir(parents=True, exist_ok=True)

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

def log(*a, **kw):
    print(*a, flush=True)

def parse_value(s):
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
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
    if m2:
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
    if op == "DOT":
        return str(sum(a*b for a, b in zip(A, B)))
    return str([(a+b) % 16 for a, b in zip(A, B)])

def user_prompt(op, A, B):
    name = "vec_dot" if op == "DOT" else "vec_add"
    if op == "DOT":
        return f"What is the dot product of {A} and {B}? Use {name}."
    return f"Use {name} to add {A} and {B} lane by lane, modulo 16."

def main():
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    t_start = time.time()
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

    def next_token(seq):
        with torch.no_grad():
            out = model(input_ids=torch.tensor([seq], device="cuda"))
            return int(torch.argmax(out.logits[0, -1]).item())

    def gen_until(seq, cap, stop_tool=False):
        gen = []
        for _ in range(cap):
            nxt = next_token(seq); gen.append(nxt); seq.append(nxt)
            if nxt in eos_ids: break
            if stop_tool and "</tool_call>" in tok.decode(gen, skip_special_tokens=False): break
        return gen, seq

    def prompt_ids(user_text, tools, system):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_text}]
        r = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, enable_thinking=False, tokenize=True)
        return [int(x) for x in (r["input_ids"] if hasattr(r, "input_ids") else r)]

    def chip_run(op, A, B):
        inp = [v0 + t for t in ref.encode(op, A, B)]
        exp = [v0 + t for t in ref.expected_output(op, A, B)]
        seq, gen = list(inp), []
        for _ in range(6):
            nxt = next_token(seq); gen.append(nxt); seq.append(nxt)
            if nxt == v0 + ref.END: break
        return [g - v0 for g in gen], gen == exp

    def seg(result_str):
        s = "<|im_end|>\n<|im_start|>user\n<tool_response>\n" + result_str + "\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return tok.encode(s, add_special_tokens=False)

    def strict_final(final, final_raw, exp_str):
        return (exp_str in final) and ("<tool_call>" not in final_raw)

    all_cases = cases_exact()
    # ---------- Variant A ----------
    if only in (None, "A"):
        results = []
        for idx, (op, A, B) in enumerate(all_cases):
            name_req = "vec_dot" if op == "DOT" else "vec_add"
            tools = [tool_def(name_req)]
            user = user_prompt(op, A, B)
            exp_str = expected_str(op, A, B)
            pid = prompt_ids(user, tools, system_single(name_req))
            gen, seq = gen_until(list(pid), cap=120, stop_tool=True)
            gen_text = tok.decode(gen, skip_special_tokens=False)
            name, parsed, fmt, ok = parse_tool_call(gen_text)
            rec = {"case": idx, "op": op, "A": A, "B": B, "user": user, "tools_listed": [name_req],
                   "model_gen_text": gen_text[:2000], "called_tool": name, "called_args": parsed if isinstance(parsed, dict) else str(parsed),
                   "format": fmt, "parse_ok": ok, "correct_tool_and_args": False, "chip_ok_requested": None,
                   "final_text": None, "final_raw": None, "strict": False}
            if ok and name == name_req and parsed["a"] == A and parsed["b"] == B:
                rec["correct_tool_and_args"] = True
                gen_chip, chip_ok = chip_run(op, A, B)
                rec["chip_ok_requested"] = chip_ok; rec["chip_gen"] = gen_chip
                res_str = expected_str(op, A, B)
                gen2, _ = gen_until(seq + seg(res_str), cap=120, stop_tool=False)
                rec["final_text"] = tok.decode(gen2, skip_special_tokens=True)[:2000]
                rec["final_raw"] = tok.decode(gen2, skip_special_tokens=False)[:2000]
                rec["second_call_loop"] = "<tool_call>" in rec["final_raw"]
                rec["strict"] = bool(chip_ok and strict_final(rec["final_text"], rec["final_raw"], exp_str))
            else:
                rec["second_call_loop"] = False
            results.append(rec)
            log(f"A CASE {idx} {op} called={name} args_ok={rec['correct_tool_and_args']} chip={rec['chip_ok_requested']} strict={rec['strict']} final={(rec['final_text'] or '')[:150]!r}", flush=True)
        out = {"variant": "A_single_tool", "v0": v0, "seconds_total": time.time() - t_start,
               "vram_max_mib": torch.cuda.max_memory_allocated()/2**20, "cases": results}
        (OUT / "variant_a_result.json").write_text(json.dumps(out, indent=2))
        log("WROTE variant_a_result.json", flush=True)
    # ---------- Variant B ----------
    if only in (None, "B"):
        results = []
        dot_cases = [c for c in all_cases if c[0] == "DOT"]
        for idx, (op, A, B) in enumerate(dot_cases):
            name_req = "vec_dot"
            user = user_prompt(op, A, B)
            exp_str = expected_str(op, A, B)
            pid = prompt_ids(user, TOOLS_BOTH, SYSTEM_BOTH)
            gen, seq = gen_until(list(pid), cap=120, stop_tool=True)
            gen_text = tok.decode(gen, skip_special_tokens=False)
            name, parsed, fmt, ok = parse_tool_call(gen_text)
            rec = {"case": idx, "op": op, "A": A, "B": B, "user": user, "tools_listed": ["vec_add", "vec_dot"],
                   "attempt1_text": gen_text[:2000], "attempt1_tool": name, "attempt1_args": parsed if isinstance(parsed, dict) else str(parsed),
                   "attempt1_format": fmt, "attempt1_parse_ok": ok, "attempts": 1, "outcome": None,
                   "attempt2_text": None, "attempt2_tool": None, "attempt2_args": None,
                   "chip_ok_requested": None, "final_text": None, "final_raw": None,
                   "second_call_loop": False, "strict_B1": False, "strict_B2": False}
            def finish_with_chip(seq_after_tool, method_outcome):
                gen_chip, chip_ok = chip_run(op, A, B)
                rec["chip_ok_requested"] = chip_ok; rec["chip_gen"] = gen_chip
                gen2, _ = gen_until(seq_after_tool + seg(exp_str), cap=120, stop_tool=False)
                rec["final_text"] = tok.decode(gen2, skip_special_tokens=True)[:2000]
                rec["final_raw"] = tok.decode(gen2, skip_special_tokens=False)[:2000]
                rec["second_call_loop"] = "<tool_call>" in rec["final_raw"]
                rec["outcome"] = method_outcome
                s = bool(chip_ok and strict_final(rec["final_text"], rec["final_raw"], exp_str))
                if method_outcome == "host_rewritten":
                    rec["strict_B2"] = s
                else:
                    rec["strict_B1"] = s
                return s
            if ok and name == name_req and parsed["a"] == A and parsed["b"] == B:
                finish_with_chip(seq, "first_attempt")
            else:
                # guard: do NOT execute chip; inject error, retry once
                wrong = name if name else "none"
                err = f"Error: wrong tool. The task requires vec_dot, not {wrong}. Call vec_dot with the same arguments."
                if not ok:
                    err = "Error: no valid tool call detected. Call vec_dot with the correct arguments."
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
                    # B2 fallback, clearly labelled: host rewrites to correct tool
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
            results.append(rec)
            # Incremental write after EVERY case, so a reclamation cannot wipe B.
            (OUT / "variant_b_partial.json").write_text(json.dumps({"variant": "B_guard_retry", "partial": True, "cases_done": len(results), "v0": v0,
                "seconds_total": time.time() - t_start, "vram_max_mib": torch.cuda.max_memory_allocated()/2**20, "cases": results}, indent=2))
            log(f"B CASE {idx} DOT attempt1={name} outcome={rec['outcome']} attempts={rec['attempts']} chip={rec['chip_ok_requested']} B1={rec['strict_B1']} B2={rec['strict_B2']} loop={rec['second_call_loop']} final={(rec['final_text'] or '')[:150]!r}", flush=True)
        out = {"variant": "B_guard_retry", "v0": v0, "seconds_total": time.time() - t_start,
               "vram_max_mib": torch.cuda.max_memory_allocated()/2**20, "cases": results}
        (OUT / "variant_b_result.json").write_text(json.dumps(out, indent=2))
        log("WROTE variant_b_result.json", flush=True)
    log("DONE seconds", time.time() - t_start, flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        (OUT / "variants_error.txt").write_text(traceback.format_exc())
        raise
