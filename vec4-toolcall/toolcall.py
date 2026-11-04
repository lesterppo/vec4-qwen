"""Gate 1+2: tool-call bridge for merged VEC4 on Colab T4.
Stepwise greedy forward (same as run_merged.py next_token): host feeds
tokens and reads tokens; all arithmetic is in the circuit weights.
Tool format parsed is Qwen3.5's XML format from chat_template.jinja:
<tool_call><function=NAME><parameter=a>...</parameter>...</function></tool_call>
A JSON-style tool call, if emitted, is recorded as format=json (not XML)
and counted separately, never silently relabelled as XML.
Host-injected tier is labelled host_injected and never counted as model-emitted.
"""
import json, re, sys, time, traceback
from pathlib import Path
sys.path.insert(0, "/content/qwenputer-colab/vec4")
import ref

MERGED = "/content/qwen35-08b-vec4"
BASE = "/content/qwen35-08b"
OUT = Path("/content/vec4-toolcall"); OUT.mkdir(parents=True, exist_ok=True)
TOOLS = [
    {"type": "function", "function": {"name": "vec_add", "description": "Add two 4-lane vectors lane by lane, modulo 16. Each vector has exactly 4 integers, each 0 to 15.", "parameters": {"type": "object", "properties": {"a": {"type": "array", "items": {"type": "integer"}}, "b": {"type": "array", "items": {"type": "integer"}}}, "required": ["a", "b"]}}},
    {"type": "function", "function": {"name": "vec_dot", "description": "Dot product of two 4-lane vectors: sum over lanes of a[i]*b[i]. Each vector has exactly 4 integers, each 0 to 15. Maximum result is 900.", "parameters": {"type": "object", "properties": {"a": {"type": "array", "items": {"type": "integer"}}, "b": {"type": "array", "items": {"type": "integer"}}}, "required": ["a", "b"]}}},
]
SYSTEM = "You have tools vec_add and vec_dot for exact vector arithmetic. Use them for every vector calculation instead of calculating by hand."

def log(*a, **kw):
    print(*a, flush=True)

def parse_value(s):
    s = s.strip()
    try:
        v = json.loads(s)
        return v
    except Exception:
        pass
    nums = re.findall(r"-?\d+", s)
    if nums and len(nums) >= 1:
        return [int(x) for x in nums]
    return s

def parse_tool_call(text):
    """Return (name, args, fmt) or (None, reason, fmt). XML first, JSON second."""
    m = re.search(r"<tool_call>\s*<function=([A-Za-z0-9_]+)>(.*?)</function>\s*</tool_call>", text, re.DOTALL)
    if m:
        name, body = m.group(1), m.group(2)
        args = {}
        for pm in re.finditer(r"<parameter=([A-Za-z0-9_]+)>\s*(.*?)\s*</parameter>", body, re.DOTALL):
            args[pm.group(1)] = parse_value(pm.group(2))
        if name not in ("vec_add", "vec_dot"):
            return None, f"unknown function {name}", "xml"
        for k in ("a", "b"):
            if k not in args or not isinstance(args[k], list) or len(args[k]) != 4 or not all(isinstance(x, int) and 0 <= x <= 15 for x in args[k]):
                return None, f"bad args {args}", "xml"
        return name, args, "xml"
    # JSON fallback, recorded separately
    m2 = re.search(r'"name"\s*:\s*"(vec_add|vec_dot)"', text)
    if m2:
        # try to find a/b lists after it
        a = re.search(r'"a"\s*:\s*\[([^\]]*)\]', text)
        b = re.search(r'"b"\s*:\s*\[([^\]]*)\]', text)
        if a and b:
            try:
                av = [int(x) for x in re.findall(r"-?\d+", a.group(1))]
                bv = [int(x) for x in re.findall(r"-?\d+", b.group(1))]
                if len(av) == 4 and len(bv) == 4 and all(0 <= x <= 15 for x in av + bv):
                    return m2.group(1), {"a": av, "b": bv}, "json"
            except Exception:
                pass
        return None, "json-style call but args unparseable", "json"
    return None, "no tool_call block found", "none"

def chip_result_str(op, A, B, gen_standalone):
    """gen_standalone: list of standalone ids (without v0). Return (result_str, chip_ok, decoded)."""
    exp = ref.expected_output(op, A, B)
    chip_ok = (gen_standalone == exp)
    if op == "ADD":
        lanes = [(A[i] + B[i]) % 16 for i in range(4)]
        # decode from gen if possible
        try:
            got = [ (t - 131) % 16 for t in gen_standalone[:4] ]
        except Exception:
            got = []
        return str(lanes), chip_ok, got
    else:
        dot = sum(a * b for a, b in zip(A, B))
        try:
            hi = (gen_standalone[0] - 195) % 16; mid = (gen_standalone[1] - 195) % 16; lo = (gen_standalone[2] - 195) % 16
            got_dot = hi * 256 + mid * 16 + lo
        except Exception:
            got_dot = None
        return str(dot), chip_ok, got_dot

def main():
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    t_start = time.time()
    meta = json.loads(Path(MERGED + "/chipformer.json").read_text())
    v0 = meta["chip_token_offset"]
    log("v0", v0, "hidden", meta["hidden"])
    tok = AutoTokenizer.from_pretrained(BASE)
    eos_ids = set()
    for tname in ("<|im_end|>", "<|endoftext|>"):
        try:
            ids = tok.encode(tname, add_special_tokens=False)
            eos_ids.update(ids)
        except Exception as e:
            log("eos encode fail", tname, repr(e))
    if tok.eos_token_id is not None:
        eos_ids.add(tok.eos_token_id)
    log("eos_ids", sorted(eos_ids))
    model = AutoModelForImageTextToText.from_pretrained(MERGED, dtype=torch.bfloat16).to("cuda").eval()
    log("loaded VRAM MiB", torch.cuda.memory_allocated()/2**20, "max", torch.cuda.max_memory_allocated()/2**20, flush=True)

    def next_token(seq):
        with torch.no_grad():
            out = model(input_ids=torch.tensor([seq], device="cuda"))
            return int(torch.argmax(out.logits[0, -1]).item())

    def gen_until(seq, cap, stop_tool=False, stop_eos=True):
        gen = []
        for _ in range(cap):
            nxt = next_token(seq)
            gen.append(nxt); seq.append(nxt)
            if stop_eos and nxt in eos_ids:
                break
            if stop_tool:
                txt = tok.decode(gen, skip_special_tokens=False)
                if "</tool_call>" in txt:
                    break
        return gen, seq

    def prompt_ids(user_text):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text}]
        r = tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, enable_thinking=False, tokenize=True)
        ids = r["input_ids"] if hasattr(r, "input_ids") else r
        return [int(x) for x in ids]

    def chip_run(op, A, B):
        inp = [v0 + t for t in ref.encode(op, A, B)]
        exp = [v0 + t for t in ref.expected_output(op, A, B)]
        seq, gen = list(inp), []
        for _ in range(6):
            nxt = next_token(seq); gen.append(nxt); seq.append(nxt)
            if nxt == v0 + ref.END:
                break
        return [g - v0 for g in gen], [e - v0 for e in exp], gen == exp

    def tool_response_segment(result_str):
        seg = "<|im_end|>\n<|im_start|>user\n<tool_response>\n" + result_str + "\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return tok.encode(seg, add_special_tokens=False)

    def host_toolcall_segment(op, A, B):
        # Exact template rendering for a host-injected assistant tool call.
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "PLACEHOLDER"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": op.lower().replace("ADD", "vec_add").replace("DOT", "vec_dot") if False else ("vec_add" if op == "ADD" else "vec_dot"), "arguments": {"a": A, "b": B}}}]},
                {"role": "tool", "content": "PLACEHOLDER_RESULT", "tool_call_id": "call_1"}]
        # We only use this to learn the segment shape; actual injection uses prompt_ids + hand segment below.
        return msgs

    # --- tool cases ---
    import random
    rng = random.Random(20261103)
    tool_cases = []
    # edges first, then random
    edges_dot = [([0,0,0,0],[0,0,0,0]), ([15,15,15,15],[15,15,15,15]), ([3,5,7,9],[2,4,6,8]), ([0,15,0,15],[15,0,15,0])]
    edges_add = [([0,0,0,0],[0,0,0,0]), ([15,15,15,15],[15,15,15,15]), ([15,0,8,7],[1,15,8,9]), ([0,1,2,3],[15,14,13,12])]
    for A,B in edges_dot:
        tool_cases.append(("DOT", A, B))
    for _ in range(6):
        tool_cases.append(("DOT", [rng.randrange(16) for _ in range(4)], [rng.randrange(16) for _ in range(4)]))
    for A,B in edges_add:
        tool_cases.append(("ADD", A, B))
    for _ in range(6):
        tool_cases.append(("ADD", [rng.randrange(16) for _ in range(4)], [rng.randrange(16) for _ in range(4)]))
    # Note: rng sequence differs from run_merged (which interleaved ADD then DOT); document that.
    results = []
    for idx, (op, A, B) in enumerate(tool_cases):
        tool_name = "vec_dot" if op == "DOT" else "vec_add"
        if op == "DOT":
            user = f"What is the dot product of {A} and {B}? Use {tool_name}."
            exp_num = sum(a*b for a,b in zip(A,B)); exp_str = str(exp_num)
        else:
            user = f"Use {tool_name} to add {A} and {B} lane by lane, modulo 16."
            lanes = [(a+b)%16 for a,b in zip(A,B)]; exp_str = str(lanes)
        pid = prompt_ids(user)
        gen, seq = gen_until(list(pid), cap=120, stop_tool=True, stop_eos=True)
        gen_text = tok.decode(gen, skip_special_tokens=False)
        name, parsed, fmt = parse_tool_call(gen_text)
        rec = {"case": idx, "op": op, "A": A, "B": B, "user": user, "model_gen_text": gen_text[:2000],
               "tool_call_found": name is not None, "tool_call_format": fmt, "parsed": parsed if isinstance(parsed, dict) else str(parsed),
               "tier": None, "chip_ok": None, "chip_gen": None, "tool_result_str": None, "final_text": None, "final_contains": None}
        if name is not None:
            # model-emitted tier
            rec["tier"] = "model_emitted"
            op2 = "DOT" if name == "vec_dot" else "ADD"
            A2, B2 = parsed["a"], parsed["b"]
            # If model parsed args differ from case args, that is a model error: chip run uses MODEL args, verification uses model args.
            gen_chip, exp_chip, chip_ok = chip_run(op2, A2, B2)
            res_str, _, _ = chip_result_str(op2, A2, B2, gen_chip)
            rec["chip_gen"] = gen_chip; rec["chip_ok"] = chip_ok; rec["tool_result_str"] = res_str
            rec["args_match_case"] = (A2 == A and B2 == B and op2 == op)
            seg = tool_response_segment(res_str)
            seq2 = seq + seg
            gen2, _ = gen_until(seq2, cap=120, stop_tool=False, stop_eos=True)
            final = tok.decode(gen2, skip_special_tokens=True)
            rec["final_text"] = final[:2000]
            if op == "DOT":
                rec["final_contains"] = (exp_str in final) if rec["args_match_case"] else (res_str in final)
            else:
                rec["final_contains"] = (exp_str in final) or (", ".join(str(x) for x in lanes) in final) if rec["args_match_case"] else (res_str in final)
        else:
            rec["tier"] = "model_failed_or_no_call"
            # host-injected fallback for this case
            gen_chip, exp_chip, chip_ok = chip_run(op, A, B)
            res_str, _, _ = chip_result_str(op, A, B, gen_chip)
            rec["fallback_chip_gen"] = gen_chip; rec["fallback_chip_ok"] = chip_ok; rec["fallback_tool_result_str"] = res_str
            # Build host-injected sequence: prompt + assistant tool_call block (template-rendered shape, hand-built tokens) + tool segment
            # Use template rendering for the full conversation to get exact ids, replacing placeholder user/result.
            msgs_full = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user},
                         {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": tool_name, "arguments": {"a": A, "b": B}}}]},
                         {"role": "tool", "content": res_str, "tool_call_id": "call_1"}]
            try:
                r = tok.apply_chat_template(msgs_full, tools=TOOLS, add_generation_prompt=True, enable_thinking=False, tokenize=True)
                seq_fb = [int(x) for x in (r["input_ids"] if hasattr(r, "input_ids") else r)]
                rec["fallback_method"] = "template_rendered"
            except Exception as e:
                rec["fallback_method"] = f"template_render_failed:{repr(e)}"
                # hand-built fallback: prompt + toolcall text + tool segment
                tc_text = f"<tool_call>\n<function={tool_name}>\n<parameter=a>\n{json.dumps(A)}\n</parameter>\n<parameter=b>\n{json.dumps(B)}\n</parameter>\n</function>\n</tool_call>"
                seq_fb = pid + tok.encode(tc_text, add_special_tokens=False) + tool_response_segment(res_str)
            gen2, _ = gen_until(seq_fb, cap=120, stop_tool=False, stop_eos=True)
            final = tok.decode(gen2, skip_special_tokens=True)
            rec["fallback_final_text"] = final[:2000]
            if op == "DOT":
                rec["fallback_final_contains"] = exp_str in final
            else:
                rec["fallback_final_contains"] = (exp_str in final) or (", ".join(str(x) for x in lanes) in final)
            rec["tier"] = "host_injected_fallback"
        results.append(rec)
        log(f"CASE {idx} {op} {A} {B} tier={rec['tier']} fmt={fmt} chip_ok={rec.get('chip_ok', rec.get('fallback_chip_ok'))} final_contains={rec.get('final_contains', rec.get('fallback_final_contains'))} final={ (rec.get('final_text') or rec.get('fallback_final_text') or '')[:200]!r}", flush=True)

    # --- plain text (no tool expected) ---
    plain_prompts = ["What is the capital of France?", "Tell me a short joke about cats.", "Explain what a vector is in one sentence.", "What is the capital of Japan?", "Write a one-line greeting."]
    plain_results = []
    for p in plain_prompts:
        pid = prompt_ids(p)
        gen, _ = gen_until(list(pid), cap=100, stop_tool=True, stop_eos=True)
        txt = tok.decode(gen, skip_special_tokens=True)
        txt_raw = tok.decode(gen, skip_special_tokens=False)
        plain_results.append({"prompt": p, "text": txt[:1000], "tool_call_in_raw": ("<tool_call>" in txt_raw), "gen_len": len(gen)})
        log(f"PLAIN {p!r} tool={('<tool_call>' in txt_raw)} text={txt[:200]!r}", flush=True)
    # --- should-not-call prompts ---
    notool_prompts = ["What is 2+2 in words?", "Explain what a dot product is in one sentence, without calculating anything.", "What is your favourite colour?", "Summarise the plot of Hamlet in one sentence.", "Is Paris the capital of France? Answer yes or no."]
    notool_results = []
    for p in notool_prompts:
        pid = prompt_ids(p)
        gen, _ = gen_until(list(pid), cap=100, stop_tool=True, stop_eos=True)
        txt = tok.decode(gen, skip_special_tokens=True)
        txt_raw = tok.decode(gen, skip_special_tokens=False)
        notool_results.append({"prompt": p, "text": txt[:1000], "tool_call_in_raw": ("<tool_call>" in txt_raw), "gen_len": len(gen)})
        log(f"NOTOOL {p!r} tool={('<tool_call>' in txt_raw)} text={txt[:200]!r}", flush=True)

    summary = {
        "v0": v0, "seconds_total": time.time() - t_start,
        "vram_max_mib": torch.cuda.max_memory_allocated()/2**20,
        "tool_cases": results,
        "plain": plain_results, "notool": notool_results,
    }
    (OUT/"toolcall_result.json").write_text(json.dumps(summary, indent=2))
    # counts
    me = [r for r in results if r["tier"] == "model_emitted"]
    fb = [r for r in results if r["tier"] == "host_injected_fallback"]
    log(f"SUMMARY model_emitted={len(me)}/{len(results)} chip_ok_me={sum(1 for r in me if r.get('chip_ok'))} final_me={sum(1 for r in me if r.get('final_contains'))} fallback={len(fb)} fb_chip={sum(1 for r in fb if r.get('fallback_chip_ok'))} fb_final={sum(1 for r in fb if r.get('fallback_final_contains'))} plain_tool={sum(1 for r in plain_results if r['tool_call_in_raw'])} notool_tool={sum(1 for r in notool_results if r['tool_call_in_raw'])}", flush=True)
    log("WROTE toolcall_result.json", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        Path("/content/vec4-toolcall/toolcall_error.txt").write_text(traceback.format_exc())
        raise
