"""Experiment 3 — TOKEN-INTERLEAVED TEXT+CHIP, one sequence, no reset.
HYPOTHESIS (pre-registered, before running; see RESULT.md for full text):
- Chip exactness should survive a text prefix: register/OP heads key only
  on VAL/OP tokens and query only at GO, so text before any write cannot
  corrupt empty DeltaNet state; Qwen layers are muted on chip tokens
  (mu slot, mute.py: linear a/b/z and full-attention output gate -1024),
  so Qwen state passes through chip unchanged and chip positions add
  nothing to the Qwen residual.
- BUT final English is expected NOT to contain the chip result from the
  interleaved sequence alone: Qwen layers are muted at chip positions
  and chip rows carry no Qwen-part information, so later text has no
  channel to read the chip result (chip keys have ~0 values; only
  attention dilution / sink remains). A low English-result rate therefore
  does NOT refute chip interleaving; it refutes naive interleaved
  reporting. The decisive interleaving scores are chip exactness and
  text sanity (I3). The toolcall bridge's tool_response text segment
  remains the reporting channel.
- Second chip block: registers/result heads are overwritten by the
  second block's VAL/OP/GO, so chip2 exactness is expected unless the
  intervening text bridge decays DeltaNet state or first-block state
  leaks into the second exit code.
- Text after chip (I3): factual text prompts do not need chip info,
  so sanity should survive unless chip keys sink full attention
  (anti-sink was skipped: sinks=None in merge_vec4.py).
Usage: python3 exp3_interleaved.py --driver cached|stepwise [--pure-subset N]
I1: text prefix + chip block, generate OUT to END, append cue
    " The answer is" (no result in cue), generate text to EOS.
I1-pure (subset, diagnostic only): after END, generate text with NO cue.
I2: prefix + ADD block + bridge " Next, " + DOT block + cue, generate text.
I3: chip block + factual text prompt ids, generate text to EOS.
"""
import json, re, sys, time, traceback
from pathlib import Path
sys.path.insert(0, "/content/qwenputer-colab/vec4-untested")
import ref
MERGED = "/content/qwen35-08b-vec4"; BASE = "/content/qwen35-08b"
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
def expected_str(op, A, B):
    return str(sum(a*b for a, b in zip(A, B))) if op == "DOT" else str([(a+b) % 16 for a, b in zip(A, B)])
def main():
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    driver = "cached"
    if "--driver" in sys.argv: driver = sys.argv[sys.argv.index("--driver") + 1]
    pure_n = 5
    if "--pure-subset" in sys.argv: pure_n = int(sys.argv[sys.argv.index("--pure-subset") + 1])
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
    def gen_stepwise(seq, cap):
        seq = list(seq); gen = []
        for _ in range(cap):
            with torch.no_grad():
                out = model(input_ids=torch.tensor([seq], device="cuda"))
            nxt = int(torch.argmax(out.logits[0, -1]).item()); gen.append(nxt); seq.append(nxt)
            if nxt in eos_ids: break
        return gen, seq
    def gen_cached(seq, cap):
        seq = list(seq)
        with torch.no_grad():
            out = model(input_ids=torch.tensor([seq], device="cuda"), use_cache=True)
        past = out.past_key_values; gen = []
        nxt = int(torch.argmax(out.logits[0, -1]).item())
        for _ in range(cap):
            gen.append(nxt); seq.append(nxt)
            if nxt in eos_ids: break
            with torch.no_grad():
                out = model(input_ids=torch.tensor([[nxt]], device="cuda"), past_key_values=past, use_cache=True)
            past = out.past_key_values; nxt = int(torch.argmax(out.logits[0, -1]).item())
        return gen, seq
    gen_fn = gen_cached if driver == "cached" else gen_stepwise
    def chip_gen(seq_prefix, op, A, B):
        """Append chip input tokens to seq_prefix (ONE sequence, no reset),
        generate OUT to END. Returns (chip_ok, chip_gen_standalone, full_seq)."""
        inp = [v0 + t for t in ref.encode(op, A, B)]
        exp = [v0 + t for t in ref.expected_output(op, A, B)]
        seq = list(seq_prefix) + inp; gen = []
        # generation phase: use gen_fn on the full seq; gen_fn returns gen+seq
        # For stepwise this is full forwards; for cached, one full forward + single steps.
        gen, seq = gen_fn(seq, 6)
        # gen_fn stops at EOS ids, but chip END is not a text EOS, so it runs to cap 6
        # unless END was emitted; trim after END if present
        if v0 + ref.END in gen:
            gen = gen[:gen.index(v0 + ref.END) + 1]
            seq = seq[:len(seq_prefix) + len(inp) + len(gen)]
        return gen == exp, [g - v0 for g in gen], seq
    def text_gen(seq, cap):
        gen, seq2 = gen_fn(list(seq), cap)
        return gen, seq2
    def has_chip_tokens(gen):
        return any(v0 <= g < v0 + 244 for g in gen)
    all_cases = cases_exact()
    prefix_text = "Question: compute the vector operation. "
    prefix_ids = tok.encode(prefix_text, add_special_tokens=False)
    cue_ids = tok.encode(" The answer is", add_special_tokens=False)
    bridge_ids = tok.encode(" Next, ", add_special_tokens=False)
    log("prefix_ids", prefix_ids, "cue_ids", cue_ids, "bridge_ids", bridge_ids, flush=True)
    # ---------- I1: single computation, 20 cases ----------
    i1 = []
    for idx, (op, A, B) in enumerate(all_cases):
        t0 = time.time(); exp_str = expected_str(op, A, B)
        cok, cgen, seq = chip_gen(prefix_ids, op, A, B)
        gen_text, _ = text_gen(seq + cue_ids, 60)
        final = tok.decode(gen_text, skip_special_tokens=True)[:2000]
        raw = tok.decode(gen_text, skip_special_tokens=False)[:2000]
        rec = {"case": idx, "op": op, "A": A, "B": B, "driver": driver, "prefix": prefix_text,
               "chip_ok": cok, "chip_gen": cgen, "chip_expected": ref.expected_output(op, A, B),
               "final_text": final, "final_raw": raw, "final_contains_result": exp_str in final,
               "tool_call_in_final_raw": "<tool_call>" in raw, "chip_tokens_in_text_gen": has_chip_tokens(gen_text),
               "seconds": time.time() - t0}
        i1.append(rec)
        log(f"I1 {idx} {op} chip={cok} final_contains={rec['final_contains_result']} toolcall={rec['tool_call_in_final_raw']} chiptok={rec['chip_tokens_in_text_gen']} {rec['seconds']:.1f}s final={final[:120]!r}", flush=True)
        (OUT / "exp3_partial.json").write_text(json.dumps({"i1": i1, "vram_max_mib": torch.cuda.max_memory_allocated()/2**20}, indent=2))
    # ---------- I1-pure diagnostic subset: no cue after END ----------
    i1pure = []
    for idx in range(min(pure_n, len(all_cases))):
        op, A, B = all_cases[idx]; t0 = time.time()
        cok, cgen, seq = chip_gen(prefix_ids, op, A, B)
        gen_text, _ = text_gen(seq, 60)
        final = tok.decode(gen_text, skip_special_tokens=True)[:2000]
        raw = tok.decode(gen_text, skip_special_tokens=False)[:2000]
        rec = {"case": idx, "op": op, "A": A, "B": B, "chip_ok": cok, "chip_gen": cgen,
               "final_text": final, "final_raw": raw, "final_contains_result": expected_str(op, A, B) in final,
               "chip_tokens_in_text_gen": has_chip_tokens(gen_text), "seconds": time.time() - t0}
        i1pure.append(rec)
        log(f"I1PURE {idx} {op} chip={cok} contains={rec['final_contains_result']} chiptok={rec['chip_tokens_in_text_gen']} {rec['seconds']:.1f}s final={final[:120]!r}", flush=True)
        (OUT / "exp3_partial.json").write_text(json.dumps({"i1": i1, "i1pure": i1pure, "vram_max_mib": torch.cuda.max_memory_allocated()/2**20}, indent=2))
    # ---------- I2: double computation, 5 pairs ----------
    # pairs: (ADD case idx, DOT case idx) from cases_exact list
    pairs = [(10, 0), (11, 1), (12, 2), (14, 4), (15, 5)]
    i2 = []
    for pidx, (ai, di) in enumerate(pairs):
        t0 = time.time()
        _, A1, B1 = all_cases[ai]; _, A2, B2 = all_cases[di]
        cok1, cgen1, seq = chip_gen(prefix_ids, "ADD", A1, B1)
        seq_bridge = seq + bridge_ids
        cok2, cgen2, seq2 = chip_gen(seq_bridge, "DOT", A2, B2)
        gen_text, _ = text_gen(seq2 + cue_ids, 80)
        final = tok.decode(gen_text, skip_special_tokens=True)[:2000]
        raw = tok.decode(gen_text, skip_special_tokens=False)[:2000]
        exp1, exp2 = expected_str("ADD", A1, B1), expected_str("DOT", A2, B2)
        rec = {"pair": pidx, "add_case": ai, "dot_case": di, "A_add": A1, "B_add": B1, "A_dot": A2, "B_dot": B2,
               "chip1_ok_add": cok1, "chip1_gen": cgen1, "chip2_ok_dot": cok2, "chip2_gen": cgen2,
               "both_chip_ok": bool(cok1 and cok2), "expected_add": exp1, "expected_dot": exp2,
               "final_text": final, "final_raw": raw, "final_contains_add": exp1 in final,
               "final_contains_dot": exp2 in final, "final_contains_both": (exp1 in final and exp2 in final),
               "tool_call_in_final_raw": "<tool_call>" in raw, "chip_tokens_in_text_gen": has_chip_tokens(gen_text),
               "seconds": time.time() - t0}
        i2.append(rec)
        log(f"I2 pair{pidx} ADD chip={cok1} DOT chip={cok2} contains_add={rec['final_contains_add']} contains_dot={rec['final_contains_dot']} {rec['seconds']:.1f}s final={final[:150]!r}", flush=True)
        (OUT / "exp3_partial.json").write_text(json.dumps({"i1": i1, "i1pure": i1pure, "i2": i2, "vram_max_mib": torch.cuda.max_memory_allocated()/2**20}, indent=2))
    # ---------- I3: text-after-chip sanity, 5 factual prompts ----------
    i3_specs = [
        ("DOT", 0, " The capital of France is", "Paris"),
        ("ADD", 10, " The capital of France is", "Paris"),
        ("DOT", 1, " The capital of Japan is", "Tokyo"),
        ("ADD", 11, " The largest planet is", "Jupiter"),
        ("DOT", 2, " The capital of Italy is", "Rome"),
    ]
    i3 = []
    for sidx, (op, cidx, prompt_text, expected_word) in enumerate(i3_specs):
        t0 = time.time(); _, A, B = all_cases[cidx]
        cok, cgen, seq = chip_gen([], op, A, B)
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
        gen_text, _ = text_gen(seq + prompt_ids, 40)
        final = tok.decode(gen_text, skip_special_tokens=True)[:2000]
        raw = tok.decode(gen_text, skip_special_tokens=False)[:2000]
        sane = (expected_word in final) and (not has_chip_tokens(gen_text)) and ("<tool_call>" not in raw)
        rec = {"spec": sidx, "op": op, "case": cidx, "A": A, "B": B, "prompt": prompt_text,
               "expected_word": expected_word, "chip_ok": cok, "chip_gen": cgen,
               "final_text": final, "final_raw": raw, "contains_expected": expected_word in final,
               "chip_tokens_in_text_gen": has_chip_tokens(gen_text), "tool_call_in_final_raw": "<tool_call>" in raw,
               "sane": sane, "seconds": time.time() - t0}
        i3.append(rec)
        log(f"I3 {sidx} {op} chip={cok} sane={sane} contains={expected_word in final} chiptok={has_chip_tokens(gen_text)} {rec['seconds']:.1f}s final={final[:150]!r}", flush=True)
        (OUT / "exp3_partial.json").write_text(json.dumps({"i1": i1, "i1pure": i1pure, "i2": i2, "i3": i3, "vram_max_mib": torch.cuda.max_memory_allocated()/2**20}, indent=2))
    out = {"experiment": "3_interleaved", "driver": driver, "seconds_total": time.time() - t_start,
           "vram_max_mib": torch.cuda.max_memory_allocated()/2**20,
           "prefix_text": prefix_text, "prefix_ids": prefix_ids, "cue_ids": cue_ids, "bridge_ids": bridge_ids,
           "i1": i1, "i1_chip": sum(1 for r in i1 if r["chip_ok"]), "i1_final_contains": sum(1 for r in i1 if r["final_contains_result"]),
           "i1_no_toolcall": sum(1 for r in i1 if not r["tool_call_in_final_raw"]),
           "i1pure": i1pure, "i1pure_chip": sum(1 for r in i1pure if r["chip_ok"]),
           "i2": i2, "i2_chip1": sum(1 for r in i2 if r["chip1_ok_add"]), "i2_chip2": sum(1 for r in i2 if r["chip2_ok_dot"]),
           "i2_both_chip": sum(1 for r in i2 if r["both_chip_ok"]), "i2_final_both": sum(1 for r in i2 if r["final_contains_both"]),
           "i3": i3, "i3_sane": sum(1 for r in i3 if r["sane"]), "i3_chip": sum(1 for r in i3 if r["chip_ok"])}
    (OUT / "exp3_interleaved_result.json").write_text(json.dumps(out, indent=2))
    log("WROTE exp3_interleaved_result.json I1 chip", out["i1_chip"], "/20 final", out["i1_final_contains"], "I2 both chip", out["i2_both_chip"], "/5 I3 sane", out["i3_sane"], "/5", flush=True)
if __name__ == "__main__":
    try: main()
    except Exception:
        traceback.print_exc(); (OUT / "exp3_error.txt").write_text(traceback.format_exc()); raise
