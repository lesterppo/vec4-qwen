"""VEC4 application test: quantized film recommendation.

Modes (all with the cached driver, END-trimmed chip scoring):
  mode1: model-orchestrated, both tools listed, guard+retry (max 2 attempts
         per step), B2 host-rewrite after 2 failures (counted separately,
         NEVER as B1/workflow success).
  mode2: application-orchestrated, host runs vec_add + 8x vec_dot chips in
         fixed order, model only writes the final English summary from the
         tool results given as text.
  control: text-only, no tools, same prompts minus tool definitions.

Scoring per scenario (from ground truth pre-registered in dataset.json):
  chip_exact: all executed chips match ref.py (ADD + 8 DOT).
  workflow_clean: mode1 only — every step called correctly by the model
    within 2 attempts, no B2 step, no extra tool call in the final phase.
  top1: first film named in the final answer is a ground-truth winner
    (any tied first-place film counts, per the documented tie rule).
  top3_order: first three distinct films named have non-increasing
    ground-truth scores and no film outside them scores strictly above
    the smallest of the three.
  top_score: the top score appears as a standalone number in the answer.
  top3_scores: all three top scores appear as standalone numbers.
Usage: python3 app.py  (expects /content/qwen35-08b-vec4 merged + dataset)
Writes /content/vec4-app/mode{1,2,control}_result.json + partials.
"""
import json, re, sys, time, traceback
from pathlib import Path
sys.path.insert(0, "/content/qwenputer-colab/vec4-app")
import ref
MERGED = "/content/qwen35-08b-vec4"; BASE = "/content/qwen35-08b"
OUT = Path("/content/vec4-app"); OUT.mkdir(parents=True, exist_ok=True)
def log(*a, **kw): print(*a, flush=True)
DS = json.loads(Path("/content/qwenputer-colab/vec4-app/dataset.json").read_text())
FILMS = DS["films"]; SCEN = DS["scenarios"]
def tool_def(name):
    desc = ("Add two 4-lane vectors lane by lane, modulo 16. Each vector has exactly 4 integers, each 0 to 15."
            if name == "vec_add" else
            "Dot product of two 4-lane vectors: sum over lanes of a[i]*b[i]. Each vector has exactly 4 integers, each 0 to 15. Maximum result is 900.")
    return {"type": "function", "function": {"name": name, "description": desc,
        "parameters": {"type": "object", "properties": {"a": {"type": "array", "items": {"type": "integer"}}, "b": {"type": "array", "items": {"type": "integer"}}}, "required": ["a", "b"]}}}
TOOLS_BOTH = [tool_def("vec_add"), tool_def("vec_dot")]
SYSTEM_M1 = ("You have tools vec_add and vec_dot for exact vector arithmetic. Use them for every vector calculation "
             "instead of calculating by hand. First call vec_add with the two taste profiles. Then call vec_dot with "
             "the combined taste vector and each film's vector, one film at a time, in the order listed. After all "
             "dot products, answer in English with the best-matching film, its score, and the top 3 films in order.")
SYSTEM_M2 = "You answer film recommendations in English from the tool results given. Use the tool results; do not recalculate by hand."
SYSTEM_CTL = "Answer in English."

def user_text(s):
    lines = ["Recommend a film for tonight.",
             f"My taste profile (action, comedy, drama, sci-fi, each 0 to 15): {s['profile_a']}.",
             f"My friend's taste profile (action, comedy, drama, sci-fi, each 0 to 15): {s['profile_b']}.",
             "Films (each vector is [action, comedy, drama, sci-fi], each 0 to 15):"]
    for f in FILMS:
        lines.append(f"- {f['name']}: {f['vector']}")
    lines.append("Find the best match: first add the two profiles lane by lane modulo 16, then compute the dot product "
                 "of the combined profile with each film, rank the films by score from highest to lowest, and tell me "
                 "the top film and the top 3 films in order.")
    return "\n".join(lines)

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
        return (name if ok else None, args if ok else f"bad/unknown call name={name} args={args}", ok)
    return None, "no tool_call block found", False

def first_names(final):
    hits = []
    low = final.lower()
    for f in FILMS:
        i = low.find(f["name"].lower())
        if i >= 0: hits.append((i, f["name"]))
    hits.sort()
    out = []
    for _, n in hits:
        if n not in out: out.append(n)
    return out

def score_final(final, s):
    names = first_names(final)
    scores = s["scores"]
    top1 = bool(names) and names[0] in s["winners"]
    top3 = False
    if len(names) >= 3:
        sc = [scores[n] for n in names[:3]]
        outside_max = max([scores[n] for n in scores if n not in names[:3]] or [-1])
        top3 = all(sc[i] >= sc[i+1] for i in range(2)) and outside_max <= min(sc)
    top_scores_sorted = sorted(scores.values(), reverse=True)[:3]
    def mentions(n):
        return re.search(r"(?<![0-9])" + str(n) + r"(?![0-9])", final) is not None
    return {"first_names": names, "top1": top1,
            "top3_order": top3, "top_score_mentioned": mentions(top_scores_sorted[0]),
            "top3_scores_mentioned": all(mentions(n) for n in top_scores_sorted),
            "top_scores_sorted": top_scores_sorted}

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
    def chip_run(op, A, B):
        inp = [v0 + t for t in ref.encode(op, A, B)]; exp = [v0 + t for t in ref.expected_output(op, A, B)]
        gen, _ = gen_cached(list(inp), 6)
        if v0 + ref.END in gen: gen = gen[:gen.index(v0 + ref.END) + 1]
        return gen == exp, [g - v0 for g in gen]
    def seg(result_str):
        s = "<|im_end|>\n<|im_start|>user\n<tool_response>\n" + result_str + "\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return tok.encode(s, add_special_tokens=False)
    def prompt_ids(msgs, tools=None):
        r = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, enable_thinking=False, tokenize=True)
        return [int(x) for x in (r["input_ids"] if hasattr(r, "input_ids") else r)]
    # text sanity (cheap, same as Gate 0 prompt)
    ids = tok.encode("The capital of France is", add_special_tokens=False)
    gen, _ = gen_cached(list(ids), 10)
    log("sanity:", repr(tok.decode(gen, skip_special_tokens=True)), flush=True)

    # ---------- MODE 1 (resume: preload cases already downloaded before reclamation) ----------
    m1 = []
    try:
        _pre = json.loads((OUT / "mode1_partial.json").read_text())
        m1 = list(_pre["cases"]); log("resumed mode1 with", len(m1), "pre-reclamation cases", flush=True)
    except Exception as e:
        log("no mode1 resume file:", repr(e), flush=True)
    _done = {c["scenario"] for c in m1}
    for s in SCEN:
        if s["id"] in _done:
            log(f"M1 scen {s['id']} skipped (already done pre-reclamation)", flush=True); continue
        t0 = time.time(); A, B, C = s["profile_a"], s["profile_b"], s["combined"]
        steps = [{"tool": "vec_add", "a": A, "b": B, "result": json.dumps(C)}]
        for f in FILMS:
            steps.append({"tool": "vec_dot", "a": C, "b": f["vector"], "result": str(s["scores"][f["name"]]), "film": f["name"]})
        pid = prompt_ids([{"role": "system", "content": SYSTEM_M1}, {"role": "user", "content": user_text(s)}], tools=TOOLS_BOTH)
        seq = list(pid); calls = []; b2_steps = 0; chip_ok = []
        for si, st in enumerate(steps):
            op = "ADD" if st["tool"] == "vec_add" else "DOT"
            done = False
            for att in (1, 2):
                gen, seq = gen_cached(seq, 120, stop_tool=True)
                gtext = tok.decode(gen, skip_special_tokens=False)
                name, parsed, ok = parse_tool_call(gtext)
                calls.append({"step": si, "expected_tool": st["tool"], "attempt": att, "tool": name,
                              "args": parsed if isinstance(parsed, dict) else str(parsed), "text": gtext[:1500]})
                if ok and name == st["tool"] and parsed["a"] == st["a"] and parsed["b"] == st["b"]:
                    cok, cgen = chip_run(op, st["a"], st["b"]); chip_ok.append(cok)
                    calls[-1]["chip_ok"] = cok; calls[-1]["chip_gen"] = cgen
                    seq = seq + seg(st["result"]); done = True; break
                err = (f"Error: wrong tool or arguments. The next step is {st['tool']}. Call it with the correct arguments."
                       if ok else f"Error: no valid tool call detected. The next step is {st['tool']}. Call it with the correct arguments.")
                calls[-1]["guard_error"] = err
                seq = seq + seg(err)
            if not done:
                b2_steps += 1
                cok, cgen = chip_run(op, st["a"], st["b"]); chip_ok.append(cok)
                calls.append({"step": si, "expected_tool": st["tool"], "attempt": "B2", "tool": st["tool"],
                              "args": {"a": st["a"], "b": st["b"]}, "chip_ok": cok, "chip_gen": cgen, "b2": True})
                seq = seq + seg(st["result"])
        gen, _ = gen_cached(seq, 250, stop_tool=False)
        final = tok.decode(gen, skip_special_tokens=True); raw = tok.decode(gen, skip_special_tokens=False)
        loop = "<tool_call>" in raw
        if loop:
            seq2 = seq + gen + seg("Error: no more tool calls are needed. Answer in English with the top film, its score, and the top 3 films in order.")
            gen2, _ = gen_cached(seq2, 250, stop_tool=False)
            final = tok.decode(gen2, skip_special_tokens=True); raw = tok.decode(gen2, skip_special_tokens=False)
        sc = score_final(final, s)
        rec = {"scenario": s["id"], "kind": s["kind"], "profile_a": A, "profile_b": B, "combined": C,
               "scores_gt": s["scores"], "winners_gt": s["winners"], "calls": calls,
               "n_tool_calls": len([c for c in calls if not c.get("b2")]), "b2_steps": b2_steps,
               "chip_exact_all": all(chip_ok) and len(chip_ok) == 9, "chip_ok_list": chip_ok,
               "workflow_clean": b2_steps == 0 and not loop,
               "final_loop": loop, "final_text": final[:2000], "final_raw": raw[:2000],
               "seconds": time.time() - t0}
        rec.update({f"score_{k}": v for k, v in sc.items()})
        m1.append(rec)
        log(f"M1 scen {s['id']} {s['kind']} calls={rec['n_tool_calls']} b2={b2_steps} chip={rec['chip_exact_all']} clean={rec['workflow_clean']} top1={sc['top1']} top3={sc['top3_order']} topScore={sc['top_score_mentioned']} top3scores={sc['top3_scores_mentioned']} {rec['seconds']:.1f}s final={final[:150]!r}", flush=True)
        (OUT / "mode1_partial.json").write_text(json.dumps({"cases": m1, "vram_max_mib": torch.cuda.max_memory_allocated()/2**20}, indent=2))
    (OUT / "mode1_result.json").write_text(json.dumps({"mode": "1_model_orchestrated", "cases": m1, "seconds_total": time.time() - t_start, "vram_max_mib": torch.cuda.max_memory_allocated()/2**20}, indent=2))
    log("WROTE mode1_result.json", flush=True)

    # ---------- MODE 2 ----------
    m2 = []
    for s in SCEN:
        t0 = time.time(); A, B, C = s["profile_a"], s["profile_b"], s["combined"]
        cok_add, cgen_add = chip_run("ADD", A, B)
        chip_ok = [cok_add]; dot_lines = [f"- Combined taste profile (vec_add): {C}"]
        for f in FILMS:
            cok, _ = chip_run("DOT", C, f["vector"]); chip_ok.append(cok)
            dot_lines.append(f"- Dot product with {f['name']}: {s['scores'][f['name']]}")
        u = user_text(s) + "\n\nTool results:\n" + "\n".join(dot_lines) + "\n\nBased on the tool results above, tell me the best-matching film, its score, and the top 3 films in order."
        pid = prompt_ids([{"role": "system", "content": SYSTEM_M2}, {"role": "user", "content": u}])
        gen, _ = gen_cached(pid, 250, stop_tool=False)
        final = tok.decode(gen, skip_special_tokens=True); raw = tok.decode(gen, skip_special_tokens=False)
        sc = score_final(final, s)
        rec = {"scenario": s["id"], "kind": s["kind"], "profile_a": A, "profile_b": B, "combined": C,
               "scores_gt": s["scores"], "winners_gt": s["winners"],
               "chip_exact_all": all(chip_ok) and len(chip_ok) == 9, "chip_ok_list": chip_ok,
               "final_text": final[:2000], "final_raw": raw[:2000], "final_loop": "<tool_call>" in raw,
               "seconds": time.time() - t0}
        rec.update({f"score_{k}": v for k, v in sc.items()})
        m2.append(rec)
        log(f"M2 scen {s['id']} chip={rec['chip_exact_all']} top1={sc['top1']} top3={sc['top3_order']} topScore={sc['top_score_mentioned']} top3scores={sc['top3_scores_mentioned']} {rec['seconds']:.1f}s final={final[:150]!r}", flush=True)
        (OUT / "mode2_partial.json").write_text(json.dumps({"cases": m2, "vram_max_mib": torch.cuda.max_memory_allocated()/2**20}, indent=2))
    (OUT / "mode2_result.json").write_text(json.dumps({"mode": "2_app_orchestrated", "cases": m2, "seconds_total": time.time() - t_start, "vram_max_mib": torch.cuda.max_memory_allocated()/2**20}, indent=2))
    log("WROTE mode2_result.json", flush=True)

    # ---------- CONTROL ----------
    ctl = []
    for s in SCEN:
        t0 = time.time()
        pid = prompt_ids([{"role": "system", "content": SYSTEM_CTL}, {"role": "user", "content": user_text(s)}])
        gen, _ = gen_cached(pid, 300, stop_tool=False)
        final = tok.decode(gen, skip_special_tokens=True); raw = tok.decode(gen, skip_special_tokens=False)
        sc = score_final(final, s)
        rec = {"scenario": s["id"], "kind": s["kind"], "profile_a": s["profile_a"], "profile_b": s["profile_b"],
               "combined": s["combined"], "scores_gt": s["scores"], "winners_gt": s["winners"],
               "final_text": final[:2000], "final_raw": raw[:2000], "seconds": time.time() - t0}
        rec.update({f"score_{k}": v for k, v in sc.items()})
        ctl.append(rec)
        log(f"CTL scen {s['id']} top1={sc['top1']} top3={sc['top3_order']} topScore={sc['top_score_mentioned']} top3scores={sc['top3_scores_mentioned']} {rec['seconds']:.1f}s final={final[:150]!r}", flush=True)
        (OUT / "control_partial.json").write_text(json.dumps({"cases": ctl, "vram_max_mib": torch.cuda.max_memory_allocated()/2**20}, indent=2))
    (OUT / "control_result.json").write_text(json.dumps({"mode": "control_text_only", "cases": ctl, "seconds_total": time.time() - t_start, "vram_max_mib": torch.cuda.max_memory_allocated()/2**20}, indent=2))
    log("WROTE control_result.json vram_max", torch.cuda.max_memory_allocated()/2**20, flush=True)
if __name__ == "__main__":
    try: main()
    except Exception:
        traceback.print_exc(); (OUT / "app_error.txt").write_text(traceback.format_exc()); raise
