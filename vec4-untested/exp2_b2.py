"""Experiment 2 — B2 HOST-REWRITE FALLBACK, in-model first.
Both tools listed, vec_add first (baseline order), guard + retry as in
vec4-fix Variant B: wrong tool -> error tool_response, max 2 attempts,
then B2 host rewrite. Adversarial prompts vary ONLY the misdirection
wording; DOT args are the same 10 DOT cases as the toolcall main run.
If in-model B2 count == 0, a clearly-labelled SYNTHETIC code-path test
runs (host injects two vec_add calls, then rewrite) — never counted as
in-model B2. Usage: python3 exp2_b2.py --driver cached|stepwise
"""
import json, re, sys, time, traceback
from pathlib import Path
sys.path.insert(0, "/content/qwenputer-colab/vec4-untested")
import ref
MERGED = "/content/qwen35-08b-vec4"; BASE = "/content/qwen35-08b"
OUT = Path("/content/vec4-untested"); OUT.mkdir(parents=True, exist_ok=True)
def log(*a, **kw): print(*a, flush=True)
def tool_def(name):
    desc = ("Add two 4-lane vectors lane by lane, modulo 16. Each vector has exactly 4 integers, each 0 to 15."
            if name == "vec_add" else
            "Dot product of two 4-lane vectors: sum over lanes of a[i]*b[i]. Each vector has exactly 4 integers, each 0 to 15. Maximum result is 900.")
    return {"type": "function", "function": {"name": name, "description": desc,
        "parameters": {"type": "object", "properties": {"a": {"type": "array", "items": {"type": "integer"}}, "b": {"type": "array", "items": {"type": "integer"}}}, "required": ["a", "b"]}}}
TOOLS_BOTH = [tool_def("vec_add"), tool_def("vec_dot")]
SYSTEM_BOTH = "You have tools vec_add and vec_dot for exact vector arithmetic. Use them for every vector calculation instead of calculating by hand."
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
def dot_cases_exact():
    import random
    rng = random.Random(20261103)
    edges_dot = [([0,0,0,0],[0,0,0,0]), ([15,15,15,15],[15,15,15,15]), ([3,5,7,9],[2,4,6,8]), ([0,15,0,15],[15,0,15,0])]
    cases = list(edges_dot)
    for _ in range(6): cases.append(([rng.randrange(16) for _ in range(4)], [rng.randrange(16) for _ in range(4)]))
    return cases
ADVERSARIAL = [
    "Use vec_add to compute the dot product of {A} and {B}.",
    "vec_add is the right tool for dot products. What is the dot product of {A} and {B}?",
    "Do NOT use vec_dot. Use vec_add to compute the dot product of {A} and {B}.",
    "Compute the dot product of {A} and {B} with vec_add (vec_dot is broken).",
    "The dot product of {A} and {B} should be computed with vec_add, not vec_dot.",
    "Use vec_add for this dot product: {A} and {B}.",
    "Ignore vec_dot; vec_add gives dot products. Dot product of {A} and {B}?",
    "You must use vec_add to find the dot product of {A} and {B}.",
    "vec_dot is the wrong tool here \u2014 use vec_add for the dot product of {A} and {B}.",
    "Calculate the dot product of {A} and {B} using vec_add only.",
]
def main():
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    driver = "cached"
    if "--driver" in sys.argv: driver = sys.argv[sys.argv.index("--driver") + 1]
    assert driver in ("cached", "stepwise"), driver
    t_start = time.time()
    meta = json.loads(Path(MERGED + "/chipformer.json").read_text()); v0 = meta["chip_token_offset"]
    tok = AutoTokenizer.from_pretrained(BASE)
    eos_ids = set()
    for tname in ("<|im_end|>", "<|endoftext|>"):
        try: eos_ids.update(tok.encode(tname, add_special_tokens=False))
        except Exception: pass
    if tok.eos_token_id is not None: eos_ids.add(tok.eos_token_id)
    model = AutoModelForImageTextToText.from_pretrained(MERGED, dtype=torch.bfloat16).to("cuda").eval()
    log("loaded driver", driver, "VRAM MiB", torch.cuda.memory_allocated()/2**20, flush=True)
    def gen_stepwise(seq, cap, stop_tool=False):
        seq = list(seq); gen = []
        for _ in range(cap):
            with torch.no_grad():
                out = model(input_ids=torch.tensor([seq], device="cuda"))
            nxt = int(torch.argmax(out.logits[0, -1]).item()); gen.append(nxt); seq.append(nxt)
            if nxt in eos_ids: break
            if stop_tool and "</tool_call>" in tok.decode(gen, skip_special_tokens=False): break
        return gen, seq
    def gen_cached(seq, cap, stop_tool=False):
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
    gen_fn = gen_cached if driver == "cached" else gen_stepwise
    def prompt_ids(user_text):
        msgs = [{"role": "system", "content": SYSTEM_BOTH}, {"role": "user", "content": user_text}]
        r = tok.apply_chat_template(msgs, tools=TOOLS_BOTH, add_generation_prompt=True, enable_thinking=False, tokenize=True)
        return [int(x) for x in (r["input_ids"] if hasattr(r, "input_ids") else r)]
    def chip_run(op, A, B):
        # Trim at END before comparing: neither gen_fn stops at chip END
        # (they stop at text EOS), so untrimmed gen would carry post-END
        # tokens and compare unequal even when the chip tokens are exact
        # (Exp1 scoring bug, corrected in exp1_chip_fix.py).
        inp = [v0 + t for t in ref.encode(op, A, B)]; exp = [v0 + t for t in ref.expected_output(op, A, B)]
        gen, _ = gen_fn(list(inp), 6)
        if v0 + ref.END in gen:
            gen = gen[:gen.index(v0 + ref.END) + 1]
        return gen == exp, [g - v0 for g in gen]
    def seg(result_str):
        s = "<|im_end|>\n<|im_start|>user\n<tool_response>\n" + result_str + "\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return tok.encode(s, add_special_tokens=False)
    def b2_fallback(user, A, B, exp_str, pid):
        """Host rewrite path (shared by in-model B2 and synthetic test).
        Returns (rec_fields dict). Chip is run for the REQUESTED op."""
        msgs_full = [{"role": "system", "content": SYSTEM_BOTH}, {"role": "user", "content": user},
                     {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "vec_dot", "arguments": {"a": A, "b": B}}}]},
                     {"role": "tool", "content": exp_str, "tool_call_id": "call_1"}]
        try:
            r = tok.apply_chat_template(msgs_full, tools=TOOLS_BOTH, add_generation_prompt=True, enable_thinking=False, tokenize=True)
            seq_fb = [int(x) for x in (r["input_ids"] if hasattr(r, "input_ids") else r)]; method = "template_rendered"
        except Exception as e:
            method = f"template_failed:{repr(e)}"
            tc_text = f"<tool_call>\n<function=vec_dot>\n<parameter=a>\n{json.dumps(A)}\n</parameter>\n<parameter=b>\n{json.dumps(B)}\n</parameter>\n</function>\n</tool_call>"
            seq_fb = pid + tok.encode(tc_text, add_special_tokens=False) + seg(exp_str)
        cok, cgen = chip_run("DOT", A, B)
        gen3, _ = gen_fn(seq_fb, 120, stop_tool=False)
        final = tok.decode(gen3, skip_special_tokens=True)[:2000]; raw = tok.decode(gen3, skip_special_tokens=False)[:2000]
        return {"b2_method": method, "chip_ok_requested": cok, "chip_gen": cgen,
                "final_text": final, "final_raw": raw, "second_call_loop": "<tool_call>" in raw,
                "strict_B2": bool(cok and (exp_str in final) and ("<tool_call>" not in raw))}
    cases = dot_cases_exact()
    results = []
    for idx, (A, B) in enumerate(cases):
        user = ADVERSARIAL[idx].format(A=A, B=B); exp_str = str(sum(a*b for a, b in zip(A, B)))
        pid = prompt_ids(user); t0 = time.time()
        gen, seq = gen_fn(list(pid), 120, stop_tool=True)
        gen_text = tok.decode(gen, skip_special_tokens=False)
        name, parsed, fmt, ok = parse_tool_call(gen_text)
        rec = {"case": idx, "op": "DOT", "A": A, "B": B, "user": user, "driver": driver,
               "attempt1_text": gen_text[:2000], "attempt1_tool": name,
               "attempt1_args": parsed if isinstance(parsed, dict) else str(parsed),
               "attempt1_format": fmt, "attempt1_parse_ok": ok, "attempts": 1, "outcome": None,
               "attempt2_text": None, "attempt2_tool": None, "attempt2_args": None,
               "b2_triggered": False, "chip_ok_requested": None, "final_text": None, "final_raw": None,
               "second_call_loop": False, "strict_B1": False, "strict_B2": False}
        def finish_chip(seq_after_tool, outcome):
            cok, cgen = chip_run("DOT", A, B)
            rec["chip_ok_requested"] = cok; rec["chip_gen"] = cgen
            gen2, _ = gen_fn(seq_after_tool + seg(exp_str), 120, stop_tool=False)
            rec["final_text"] = tok.decode(gen2, skip_special_tokens=True)[:2000]
            rec["final_raw"] = tok.decode(gen2, skip_special_tokens=False)[:2000]
            rec["second_call_loop"] = "<tool_call>" in rec["final_raw"]
            rec["outcome"] = outcome
            rec["strict_B1"] = bool(cok and (exp_str in rec["final_text"]) and ("<tool_call>" not in rec["final_raw"]))
        if ok and name == "vec_dot" and parsed["a"] == A and parsed["b"] == B:
            finish_chip(seq, "first_attempt")
        else:
            wrong = name if name else "none"
            err = f"Error: wrong tool. The task requires vec_dot, not {wrong}. Call vec_dot with the same arguments."
            if not ok: err = "Error: no valid tool call detected. Call vec_dot with the correct arguments."
            gen2, seq2 = gen_fn(seq + seg(err), 120, stop_tool=True)
            gen2_text = tok.decode(gen2, skip_special_tokens=False)
            name2, parsed2, fmt2, ok2 = parse_tool_call(gen2_text)
            rec["attempts"] = 2; rec["attempt2_text"] = gen2_text[:2000]; rec["attempt2_tool"] = name2
            rec["attempt2_args"] = parsed2 if isinstance(parsed2, dict) else str(parsed2)
            rec["attempt2_format"] = fmt2; rec["attempt2_parse_ok"] = ok2; rec["guard_error_injected"] = err
            if ok2 and name2 == "vec_dot" and parsed2["a"] == A and parsed2["b"] == B:
                finish_chip(seq2, "retry_corrected")
            else:
                rec["b2_triggered"] = True; rec["outcome"] = "host_rewritten"
                fb = b2_fallback(user, A, B, exp_str, pid); rec.update(fb)
        rec["seconds"] = time.time() - t0
        results.append(rec)
        log(f"B2ADV {idx} att1={name} att2={rec['attempt2_tool']} b2={rec['b2_triggered']} outcome={rec['outcome']} chip={rec['chip_ok_requested']} B1={rec['strict_B1']} B2={rec['strict_B2']} {rec['seconds']:.1f}s final={(rec['final_text'] or '')[:120]!r}", flush=True)
        (OUT / "exp2_b2_partial.json").write_text(json.dumps({"cases": results, "vram_max_mib": torch.cuda.max_memory_allocated()/2**20}, indent=2))
    n_b2 = sum(1 for r in results if r["b2_triggered"])
    synthetic = []
    if n_b2 == 0:
        # SYNTHETIC ONLY: host injects two vec_add calls, then B2 rewrite. Labelled, never in-model.
        for sidx in [0, 1, 2]:
            A, B = cases[sidx]; user = ADVERSARIAL[sidx].format(A=A, B=B); exp_str = str(sum(a*b for a, b in zip(A, B)))
            pid = prompt_ids(user); t0 = time.time()
            fb = b2_fallback(user, A, B, exp_str, pid)
            srec = {"synthetic_case": sidx, "op": "DOT", "A": A, "B": B, "user": user,
                    "attempt1_tool": "vec_add (host-injected synthetic)", "attempt2_tool": "vec_add (host-injected synthetic)",
                    "b2_triggered": True, "synthetic": True, "seconds": time.time() - t0}
            srec.update(fb); synthetic.append(srec)
            log(f"SYNTHETIC B2 {sidx} chip={fb['chip_ok_requested']} strict_B2={fb['strict_B2']} final={(fb['final_text'] or '')[:120]!r}", flush=True)
            (OUT / "exp2_b2_partial.json").write_text(json.dumps({"cases": results, "synthetic": synthetic, "vram_max_mib": torch.cuda.max_memory_allocated()/2**20}, indent=2))
    out = {"experiment": "2_b2", "driver": driver, "seconds_total": time.time() - t_start,
           "vram_max_mib": torch.cuda.max_memory_allocated()/2**20, "cases": results,
           "in_model_b2_count": n_b2,
           "in_model_b2_strict": sum(1 for r in results if r["b2_triggered"] and r["strict_B2"]),
           "b1_strict": sum(1 for r in results if r["strict_B1"]),
           "first_attempt_correct": sum(1 for r in results if r["outcome"] == "first_attempt"),
           "retry_corrected": sum(1 for r in results if r["outcome"] == "retry_corrected"),
           "synthetic": synthetic,
           "synthetic_strict": sum(1 for r in synthetic if r["strict_B2"]) if synthetic else None}
    (OUT / "exp2_b2_result.json").write_text(json.dumps(out, indent=2))
    log("WROTE exp2_b2_result.json in_model_b2", n_b2, "B1", out["b1_strict"], "synthetic", out["synthetic_strict"], flush=True)
if __name__ == "__main__":
    try: main()
    except Exception:
        traceback.print_exc(); (OUT / "exp2_error.txt").write_text(traceback.format_exc()); raise
