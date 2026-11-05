"""Tool tests for vec4-v3 on a merged Qwen3.5 model (10 tools). Each --only runnable,
incremental partial JSON, strict scoring (see below).

Configs:
  orig: original 20 cases (v1 toolcall set), all 6 tools listed, single
        attempt. Measures selection with 6 tools vs v1's 2-tool baseline.
        Cases verified against vec4-fix/variant_a_result.json (uploaded
        to /content/qwenputer-colab/vec4-fix/ on the VM).
  para: 12 paraphrase prompts from generalisation.json, 6 tools listed,
        single attempt, strict by the set's intended op/expected value.
  adv: 10 adversarial prompts from generalisation.json, 6 tools listed,
       single attempt, strict by intended op (misdirection must NOT win).
  disp: dispatcher + chip, NO model tool call: paraphrase 12 +
       adversarial 10 + long_context 4 + batch 10 + programs 4.
       Dispatcher-refused cases are recorded as refused, never as
       wrong computations. Programs use execute_program with the chip
       as executor.
  plain: 5 plain + 5 should-not-call prompts, 6 tools listed.
Driver: cached per-phase KV cache by default (proven by
cache_probe_v2.py); --driver stepwise falls back if the probe failed.
Toolcall bridge only: host decodes chip OUT, re-injects text
tool_response. No interleaved reporting.
Strict (model configs) = called tool == intended tool AND args ==
case args AND chip for intended op matches ref AND final English
contains the correct result (for lane ops: str(list) or comma-joined
digits; for DOT: str(int)) AND final raw has no <tool_call> block.
"""
import json, re, sys, time, traceback
from pathlib import Path
sys.path.insert(0, "/content/qwenputer-colab/vec4-v3")
import ref
from dispatcher import dispatch, execute_program

MERGED = "/content/qwen35-08b-vec4v3"
BASE = "/content/qwen35-08b"
OUT = Path("/content/vec4-v3")
if "--merged" in sys.argv: MERGED = sys.argv[sys.argv.index("--merged") + 1]
if "--base" in sys.argv: BASE = sys.argv[sys.argv.index("--base") + 1]
if "--outdir" in sys.argv: OUT = Path(sys.argv[sys.argv.index("--outdir") + 1])
OUT.mkdir(parents=True, exist_ok=True)

OP_TOOL = {"ADD": "vec_add", "DOT": "vec_dot", "SUB": "vec_sub",
           "MUL": "vec_mul", "MAX": "vec_max", "MIN": "vec_min",
           "VSHL": "vec_shl", "VSHR": "vec_shr", "MMUL": "vec_mmul",
           "VSORT": "vec_vsort"}
TOOL_OP = {v: k for k, v in OP_TOOL.items()}
DESCS = {
    "vec_add": "Add two 4-lane vectors lane by lane, modulo 16. Each vector has exactly 4 integers, each 0 to 15.",
    "vec_dot": "Dot product of two 4-lane vectors: sum over lanes of a[i]*b[i]. Each vector has exactly 4 integers, each 0 to 15. Maximum result is 900.",
    "vec_sub": "Subtract two 4-lane vectors lane by lane, modulo 16: first minus second. Each vector has exactly 4 integers, each 0 to 15.",
    "vec_mul": "Multiply two 4-lane vectors lane by lane, modulo 16. Each vector has exactly 4 integers, each 0 to 15.",
    "vec_max": "Per-entry maximum of two 4-lane vectors: keep the larger entry in each lane. Each vector has exactly 4 integers, each 0 to 15.",
    "vec_min": "Per-entry minimum of two 4-lane vectors: keep the smaller entry in each lane. Each vector has exactly 4 integers, each 0 to 15.",
    "vec_shl": "Shift each lane of the first vector left by the corresponding lane of the second vector (amount modulo 4), modulo 16. Each vector has exactly 4 integers, each 0 to 15.",
    "vec_shr": "Shift each lane of the first vector right by the corresponding lane of the second vector (amount modulo 4). Each vector has exactly 4 integers, each 0 to 15.",
    "vec_mmul": "Multiply two 2x2 matrices modulo 16, given row-major as 4-lane vectors: first vector is [a00,a01,a10,a11], second is [b00,b01,b10,b11], result is [c00,c01,c10,c11]. Each vector has exactly 4 integers, each 0 to 15.",
    "vec_vsort": "Sort the first vector's 4 lanes ascending. The second vector is required by the tool but ignored. Each vector has exactly 4 integers, each 0 to 15.",
}

def tool_def(name):
    return {"type": "function", "function": {"name": name, "description": DESCS[name],
        "parameters": {"type": "object", "properties": {"a": {"type": "array", "items": {"type": "integer"}}, "b": {"type": "array", "items": {"type": "integer"}}}, "required": ["a", "b"]}}}

TOOLS_ALL = [tool_def(n) for n in ("vec_add", "vec_dot", "vec_sub", "vec_mul", "vec_max", "vec_min", "vec_shl", "vec_shr", "vec_mmul", "vec_vsort")]
SYSTEM_ALL = ("You have tools vec_add, vec_dot, vec_sub, vec_mul, vec_max, vec_min, vec_shl, vec_shr, vec_mmul and vec_vsort "
              "for exact vector arithmetic. Use them for every vector calculation instead of calculating by hand.")

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
        ok = name in TOOL_OP and all(k in args and isinstance(args[k], list) and len(args[k]) == 4 and all(isinstance(x, int) and 0 <= x <= 15 for x in args[k]) for k in ("a", "b"))
        return (name if ok else None, args if ok else f"bad/unknown call name={name} args={args}", "xml", ok)
    m2 = re.search(r'"name"\s*:\s*"(vec_add|vec_dot|vec_sub|vec_mul|vec_max|vec_min|vec_shl|vec_shr|vec_mmul|vec_vsort)"', text)
    if m2: return None, "json-style call (not parsed as XML)", "json", False
    return None, "no tool_call block found", "none", False

def cases_orig():
    import random
    rng = random.Random(20261103)
    edges_dot = [([0,0,0,0],[0,0,0,0]), ([15,15,15,15],[15,15,15,15]), ([3,5,7,9],[2,4,6,8]), ([0,15,0,15],[15,0,15,0])]
    edges_add = [([0,0,0,0],[0,0,0,0]), ([15,15,15,15],[15,15,15,15]), ([15,0,8,7],[1,15,8,9]), ([0,1,2,3],[15,14,13,12])]
    cases = []
    for A,B in edges_dot: cases.append(("DOT", A, B))
    for _ in range(6): cases.append(("DOT", [rng.randrange(16) for _ in range(4)], [rng.randrange(16) for _ in range(4)]))
    for A,B in edges_add: cases.append(("ADD", A, B))
    for _ in range(6): cases.append(("ADD", [rng.randrange(16) for _ in range(4)], [rng.randrange(16) for _ in range(4)]))
    # v3 new ops: 2 edge cases each, tool named in the prompt.
    for op in ("VSHL", "VSHR", "MMUL", "VSORT"):
        cases.append((op, [3, 5, 7, 9], [2, 4, 6, 8]))
        cases.append((op, [15, 8, 1, 0], [4, 5, 6, 7]))
    return cases

def expected_value(op, A, B):
    if op == "DOT": return sum(a*b for a, b in zip(A, B))
    if op == "MMUL": return ref.mmul_result(A, B)
    if op == "VSORT": return ref.vsort_result(A)
    return ref.lane_result(op, A, B)

def expected_strs(val):
    if isinstance(val, int): return [str(val)]
    return [str(val), ", ".join(str(x) for x in val)]

def user_prompt_orig(op, A, B):
    name = OP_TOOL[op]
    if op == "DOT": return f"What is the dot product of {A} and {B}? Use {name}."
    if op == "VSHL": return f"Use {name} to shift each lane of {A} left by the corresponding lane of {B} (amount modulo 4)."
    if op == "VSHR": return f"Use {name} to shift each lane of {A} right by the corresponding lane of {B} (amount modulo 4)."
    if op == "MMUL": return f"Use {name} to multiply the matrices {A} and {B} (row-major 2x2) modulo 16."
    if op == "VSORT": return f"Use {name} to sort {A} ascending."
    return f"Use {name} to add {A} and {B} lane by lane, modulo 16."

PARA_NEW = [
    {"op": "VSHL", "A": [9, 6, 13, 4], "B": [1, 2, 3, 0],
     "text": "Shift the entries of [9, 6, 13, 4] to the left, lane by lane, by the amounts in [1, 2, 3, 0]."},
    {"op": "VSHR", "A": [13, 10, 7, 15], "B": [2, 1, 3, 0],
     "text": "Shift [13, 10, 7, 15] right by [2, 1, 3, 0], one amount per lane."},
    {"op": "MMUL", "A": [1, 2, 3, 4], "B": [5, 6, 7, 8],
     "text": "What is the product of the two matrices [1, 2, 3, 4] and [5, 6, 7, 8], taken modulo 16?"},
    {"op": "VSORT", "A": [7, 2, 15, 2], "B": [0, 0, 0, 0],
     "text": "Put the numbers [7, 2, 15, 2] in ascending order, smallest first."},
]

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
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    driver = sys.argv[sys.argv.index("--driver") + 1] if "--driver" in sys.argv else "cached"
    assert driver in ("cached", "stepwise"), driver
    t_start = time.time()
    gen = json.loads(Path("/content/qwenputer-colab/vec4-v2/generalisation.json").read_text())
    meta = json.loads(Path(MERGED + "/chipformer.json").read_text())
    v0 = meta["chip_token_offset"]
    assert meta.get("layout") == "vec4-v3" and meta.get("chip_vocab") == 252, meta
    tok = AutoTokenizer.from_pretrained(BASE)
    eos_ids = set()
    for tname in ("<|im_end|>", "<|endoftext|>"):
        try: eos_ids.update(tok.encode(tname, add_special_tokens=False))
        except Exception: pass
    if tok.eos_token_id is not None: eos_ids.add(tok.eos_token_id)
    model = AutoModelForImageTextToText.from_pretrained(MERGED, dtype=torch.bfloat16).to("cuda").eval()
    log("loaded VRAM MiB", torch.cuda.memory_allocated() / 2**20, flush=True)

    def next_token_stepwise(seq):
        with torch.no_grad():
            out = model(input_ids=torch.tensor([seq], device="cuda"))
            return int(torch.argmax(out.logits[0, -1]).item())

    def gen_until(seq, cap, stop_tool=False, stop_chip_end=False):
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

    def chip_value(op, A, B):
        """Chip result as a value (int for DOT, list for lane ops), or None if chip wrong."""
        gen_trim, ok = chip_run(op, A, B)
        if not ok: return None, gen_trim
        if op == "DOT":
            digits = [(t - 195) % 16 for t in gen_trim[:3]]
            return digits[0] * 256 + digits[1] * 16 + digits[2], gen_trim
        return [(t - 131) % 16 for t in gen_trim[:4]], gen_trim

    def seg(result_str):
        s = "<|im_end|>\n<|im_start|>user\n<tool_response>\n" + result_str + "\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return tok.encode(s, add_special_tokens=False)

    def med(xs):
        xs = sorted(xs); n = len(xs)
        return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2 if n else None

    def run_model_case(idx, op, A, B, user, exp_val, partial_file, variant):
        t_c = time.time()
        name_req = OP_TOOL[op]
        pid = prompt_ids(user, TOOLS_ALL, SYSTEM_ALL)
        gen, seq = gen_until(list(pid), cap=120, stop_tool=True)
        gen_text = tok.decode(gen, skip_special_tokens=False)
        name, parsed, fmt, ok = parse_tool_call(gen_text)
        rec = {"case": idx, "op": op, "A": A, "B": B, "user": user,
               "tools_listed": list(OP_TOOL.values()), "model_gen_text": gen_text[:2000],
               "called_tool": name, "called_args": parsed if isinstance(parsed, dict) else str(parsed),
               "format": fmt, "parse_ok": ok, "correct_tool_and_args": False,
               "chip_ok_requested": None, "final_text": None, "final_raw": None,
               "second_call_loop": False, "strict": False, "seconds": None}
        if ok and name == name_req and parsed["a"] == A and parsed["b"] == B:
            rec["correct_tool_and_args"] = True
            gen_chip, chip_ok = chip_run(op, A, B)
            rec["chip_ok_requested"] = chip_ok; rec["chip_gen"] = gen_chip
            gen2, _ = gen_until(seq + seg(str(exp_val)), cap=120, stop_tool=False)
            rec["final_text"] = tok.decode(gen2, skip_special_tokens=True)[:2000]
            rec["final_raw"] = tok.decode(gen2, skip_special_tokens=False)[:2000]
            rec["second_call_loop"] = "<tool_call>" in rec["final_raw"]
            rec["strict"] = bool(chip_ok and any(s in rec["final_text"] for s in expected_strs(exp_val))
                                 and not rec["second_call_loop"])
        rec["seconds"] = time.time() - t_c
        return rec

    def write_model_config(variant, recs, partial_file, result_file):
        (OUT / partial_file).write_text(json.dumps(
            {"variant": variant, "partial": True, "cases_done": len(recs), "v0": v0,
             "seconds_total": time.time() - t_start, "vram_max_mib": torch.cuda.max_memory_allocated() / 2**20,
             "cases": recs}, indent=2))
        (OUT / result_file).write_text(json.dumps(
            {"variant": variant, "driver": driver, "v0": v0, "seconds_total": time.time() - t_start,
             "vram_max_mib": torch.cuda.max_memory_allocated() / 2**20,
             "median_seconds_per_case": med([r["seconds"] for r in recs]), "cases": recs}, indent=2))

    if only in (None, "orig"):
        all_cases = cases_orig()
        assert len(all_cases) == 28, len(all_cases)
        log("orig cases: v2 20 + v3 new-op 8; driver", driver)
        results = []
        for idx, (op, A, B) in enumerate(all_cases):
            rec = run_model_case(idx, op, A, B, user_prompt_orig(op, A, B), expected_value(op, A, B), None, "T_orig")
            results.append(rec); write_model_config("T_orig", results, "t_orig_partial.json", "t_orig_result.json")
            log(f"ORIG {idx} {op} called={rec['called_tool']} args_ok={rec['correct_tool_and_args']} chip={rec['chip_ok_requested']} strict={rec['strict']} s={rec['seconds']:.1f} final={(rec['final_text'] or '')[:150]!r}", flush=True)
        log("WROTE t_orig_result.json", flush=True)

    if only in (None, "para"):
        results = []
        for idx, c in enumerate(gen["paraphrase"]):
            rec = run_model_case(idx, c["op"], c["A"], c["B"], c["text"], c["expected"], None, "T_para")
            rec["dispatcher_status"] = c.get("dispatcher")
            results.append(rec); write_model_config("T_para", results, "t_para_partial.json", "t_para_result.json")
            log(f"PARA {idx} {c['op']} called={rec['called_tool']} args_ok={rec['correct_tool_and_args']} chip={rec['chip_ok_requested']} strict={rec['strict']} s={rec['seconds']:.1f} final={(rec['final_text'] or '')[:150]!r}", flush=True)
        for j, c in enumerate(PARA_NEW):
            idx = len(gen["paraphrase"]) + j
            rec = run_model_case(idx, c["op"], c["A"], c["B"], c["text"], expected_value(c["op"], c["A"], c["B"]), None, "T_para")
            rec["dispatcher_status"] = dispatch(c["text"])["status"]
            results.append(rec); write_model_config("T_para", results, "t_para_partial.json", "t_para_result.json")
            log(f"PARA_NEW {j} {c['op']} called={rec['called_tool']} args_ok={rec['correct_tool_and_args']} chip={rec['chip_ok_requested']} strict={rec['strict']} s={rec['seconds']:.1f} final={(rec['final_text'] or '')[:150]!r}", flush=True)
        log("WROTE t_para_result.json", flush=True)

    if only in (None, "adv"):
        results = []
        for idx, c in enumerate(gen["adversarial"]):
            rec = run_model_case(idx, c["op"], c["A"], c["B"], c["text"], c["expected"], None, "T_adv")
            rec["kind"] = c["kind"]
            results.append(rec); write_model_config("T_adv", results, "t_adv_partial.json", "t_adv_result.json")
            log(f"ADV {idx} {c['kind']} {c['op']} called={rec['called_tool']} args_ok={rec['correct_tool_and_args']} chip={rec['chip_ok_requested']} strict={rec['strict']} s={rec['seconds']:.1f} final={(rec['final_text'] or '')[:150]!r}", flush=True)
        log("WROTE t_adv_result.json", flush=True)

    if only in (None, "disp"):
        # Dispatcher + chip only, no model tool call.
        rows = []
        for group in ("paraphrase", "adversarial", "long_context"):
            for idx, c in enumerate(gen[group]):
                r = dispatch(c["text"])
                row = {"group": group, "case": idx, "op": c["op"], "A": c["A"], "B": c["B"],
                       "text": c["text"], "expected": c["expected"],
                       "dispatcher_status": r["status"], "intended_op": r["intended_op"],
                       "vectors": r["vectors"], "chip_value": None, "chip_ok": None, "outcome": None}
                if r["status"] in ("OK", "CONFLICT") and r["intended_op"] == c["op"] and list(r["vectors"]) == [c["A"], c["B"]]:
                    val, gen_trim = chip_value(c["op"], c["A"], c["B"])
                    row["chip_value"] = val; row["chip_ok"] = (val == c["expected"]); row["outcome"] = "executed_correct"
                elif r["status"] == "AMBIGUOUS":
                    row["outcome"] = "refused"
                else:
                    row["outcome"] = "wrong_intent_or_vectors"
                rows.append(row)
                log(f"DISP {group} {idx} {c['op']} status={r['status']} intended={r['intended_op']} outcome={row['outcome']} chip_ok={row['chip_ok']}", flush=True)
                (OUT / "t_disp_partial.json").write_text(json.dumps({"rows": rows}, indent=2))
        for idx, req in enumerate(gen["batch"]):
            text = f"What is the dot product of {req['A']} and {req['B']}?"
            r = dispatch(text)
            val, gen_trim = chip_value("DOT", req["A"], req["B"]) if r["status"] == "OK" and r["intended_op"] == "DOT" else (None, None)
            rows.append({"group": "batch", "case": idx, "op": "DOT", "A": req["A"], "B": req["B"],
                         "expected": req["expected"], "dispatcher_status": r["status"], "intended_op": r["intended_op"],
                         "chip_value": val, "chip_ok": (val == req["expected"]),
                         "outcome": "executed_correct" if val == req["expected"] else "failed"})
        for idx, prog in enumerate(gen["program"]):
            def exe(op, a, b):
                if not (isinstance(a, list) and isinstance(b, list)):
                    return None
                val, _ = chip_value(op, a, b)
                return val
            r = execute_program(prog["steps"], exe)
            rows.append({"group": "program", "case": idx, "steps": prog["steps"],
                         "expected_results": prog["results"], "results": r.get("results"),
                         "status": r["status"], "outcome": "executed_correct" if r.get("results") == prog["results"] else "failed"})
            log(f"DISP program {idx} status={r['status']} results={r.get('results')} expected={prog['results']}", flush=True)
        for idx, c in enumerate(PARA_NEW):
            r = dispatch(c["text"])
            exp = expected_value(c["op"], c["A"], c["B"])
            row = {"group": "para_new", "case": idx, "op": c["op"], "A": c["A"], "B": c["B"],
                   "text": c["text"], "expected": exp,
                   "dispatcher_status": r["status"], "intended_op": r["intended_op"],
                   "vectors": r["vectors"], "chip_value": None, "chip_ok": None, "outcome": None}
            if r["status"] in ("OK", "CONFLICT") and r["intended_op"] == c["op"] and list(r["vectors"]) == [c["A"], c["B"]]:
                val, gen_trim = chip_value(c["op"], c["A"], c["B"])
                row["chip_value"] = val; row["chip_ok"] = (val == exp); row["outcome"] = "executed_correct"
            elif r["status"] == "AMBIGUOUS":
                row["outcome"] = "refused"
            else:
                row["outcome"] = "wrong_intent_or_vectors"
            rows.append(row)
            log(f"DISP para_new {idx} {c['op']} status={r['status']} intended={r['intended_op']} outcome={row['outcome']} chip_ok={row['chip_ok']}", flush=True)
        prog_steps = [{"op": "MMUL", "a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}, {"op": "VSORT", "a": "$0", "b": [0, 0, 0, 0]}]
        prog_exp = [ref.mmul_result([1, 2, 3, 4], [5, 6, 7, 8])]
        prog_exp.append(ref.vsort_result(prog_exp[0]))
        def exe2(op, a, b):
            if not (isinstance(a, list) and isinstance(b, list)):
                return None
            val, _ = chip_value(op, a, b)
            return val
        r = execute_program(prog_steps, exe2)
        rows.append({"group": "program_new", "case": 0, "steps": prog_steps,
                     "expected_results": prog_exp, "results": r.get("results"),
                     "status": r["status"], "outcome": "executed_correct" if r.get("results") == prog_exp else "failed"})
        log(f"DISP program_new status={r['status']} results={r.get('results')} expected={prog_exp}", flush=True)
        (OUT / "t_disp_result.json").write_text(json.dumps(
            {"variant": "T_disp", "driver": driver, "v0": v0, "seconds_total": time.time() - t_start,
             "vram_max_mib": torch.cuda.max_memory_allocated() / 2**20, "rows": rows}, indent=2))
        log("WROTE t_disp_result.json", flush=True)

    if only in (None, "plain"):
        for kind, prompts in (("plain", PLAIN_PROMPTS), ("notool", NOTOOL_PROMPTS)):
            results = []
            for p in prompts:
                pid = prompt_ids(p, TOOLS_ALL, SYSTEM_ALL)
                gen, _ = gen_until(list(pid), cap=100, stop_tool=True)
                txt = tok.decode(gen, skip_special_tokens=True)
                txt_raw = tok.decode(gen, skip_special_tokens=False)
                results.append({"prompt": p, "text": txt[:1000],
                                "tool_call_in_raw": ("<tool_call>" in txt_raw), "gen_len": len(gen)})
                log(f"{kind.upper()} {p!r} tool={('<tool_call>' in txt_raw)} text={txt[:200]!r}", flush=True)
            (OUT / f"t_{kind}_result.json").write_text(json.dumps(
                {"variant": f"T_{kind}", "driver": driver, "v0": v0,
                 "seconds_total": time.time() - t_start,
                 "vram_max_mib": torch.cuda.max_memory_allocated() / 2**20,
                 "results": results}, indent=2))
            log(f"WROTE t_{kind}_result.json", flush=True)
    log("DONE seconds", time.time() - t_start, flush=True)

if __name__ == "__main__":
    try: main()
    except Exception:
        traceback.print_exc(); (OUT / "tool_v3_error.txt").write_text(traceback.format_exc()); raise
