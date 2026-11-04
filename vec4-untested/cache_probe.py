"""Probe: stepwise vs stock generate() vs manual past_key_values cache, on
one chip case and one short text prompt. Decides the Exp1 cached driver.
Writes /content/vec4-untested/cache_probe.json. Small, fast, diagnostic only."""
import json, sys, time, traceback
from pathlib import Path
sys.path.insert(0, "/content/qwenputer-colab/vec4-untested")
import ref
OUT = Path("/content/vec4-untested"); OUT.mkdir(parents=True, exist_ok=True)
def log(*a, **kw):
    print(*a, flush=True)
try:
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    meta = json.loads(Path("/content/qwen35-08b-vec4/chipformer.json").read_text())
    v0 = meta["chip_token_offset"]
    tok = AutoTokenizer.from_pretrained("/content/qwen35-08b")
    model = AutoModelForImageTextToText.from_pretrained("/content/qwen35-08b-vec4", dtype=torch.bfloat16).to("cuda").eval()
    log("loaded VRAM MiB", torch.cuda.memory_allocated() / 2**20, flush=True)
    log("model type", model.config.model_type, "text model type", model.config.text_config.model_type, flush=True)
    def stepwise(seq, cap, stop_id=None, eos=None):
        seq = list(seq); gen = []
        for _ in range(cap):
            with torch.no_grad():
                out = model(input_ids=torch.tensor([seq], device="cuda"))
            nxt = int(torch.argmax(out.logits[0, -1]).item())
            gen.append(nxt); seq.append(nxt)
            if stop_id is not None and nxt == stop_id: break
            if eos is not None and nxt in eos: break
        return gen
    # --- chip case: DOT [3,5,7,9]x[2,4,6,8]=140 ---
    op, A, B = "DOT", [3, 5, 7, 9], [2, 4, 6, 8]
    inp = [v0 + t for t in ref.encode(op, A, B)]
    exp = [v0 + t for t in ref.expected_output(op, A, B)]
    t0 = time.time(); g_step = stepwise(inp, 6, stop_id=v0 + ref.END); t_step = time.time() - t0
    log("chip stepwise", [g - v0 for g in g_step], "exp", [e - v0 for e in exp], "match", g_step == exp, f"{t_step:.2f}s", flush=True)
    # stock generate, chip
    gen_err, g_gen, t_gen = None, None, None
    try:
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(input_ids=torch.tensor([inp], device="cuda"), max_new_tokens=6,
                                 do_sample=False, use_cache=True, eos_token_id=v0 + ref.END,
                                 pad_token_id=v0 + ref.END)
        g_gen = [int(x) for x in out[0][len(inp):].tolist()]; t_gen = time.time() - t0
        log("chip generate", [g - v0 for g in g_gen], "match", g_gen == exp, f"{t_gen:.2f}s", flush=True)
    except Exception as e:
        gen_err = repr(e); log("chip generate FAILED", gen_err, flush=True); traceback.print_exc()
    # manual cache, chip
    man_err, g_man, t_man = None, None, None
    try:
        t0 = time.time()
        with torch.no_grad():
            out = model(input_ids=torch.tensor([inp], device="cuda"), use_cache=True)
        past = out.past_key_values; nxt = int(torch.argmax(out.logits[0, -1]).item())
        g_man = [nxt]; seq_len = len(inp)
        while nxt != v0 + ref.END and len(g_man) < 6:
            with torch.no_grad():
                out = model(input_ids=torch.tensor([[nxt]], device="cuda"), past_key_values=past, use_cache=True)
            past = out.past_key_values; nxt = int(torch.argmax(out.logits[0, -1]).item()); g_man.append(nxt)
        t_man = time.time() - t0
        log("chip manual-cache", [g - v0 for g in g_man], "match", g_man == exp, f"{t_man:.2f}s", "cache type", type(past).__name__, flush=True)
    except Exception as e:
        man_err = repr(e); log("chip manual-cache FAILED", man_err, flush=True); traceback.print_exc()
    # --- text prompt, stepwise vs manual cache ---
    prompt = "The capital of France is"
    ids = tok.encode(prompt, add_special_tokens=False)
    eos_ids = set()
    for tname in ("<|im_end|>", "<|endoftext|>"):
        try: eos_ids.update(tok.encode(tname, add_special_tokens=False))
        except Exception: pass
    if tok.eos_token_id is not None: eos_ids.add(tok.eos_token_id)
    t0 = time.time(); gt_step = stepwise(ids, 10, eos=eos_ids); t_ts = time.time() - t0
    log("text stepwise", repr(tok.decode(gt_step, skip_special_tokens=True)), gt_step, f"{t_ts:.2f}s", flush=True)
    man2_err, gt_man, t_tm = None, None, None
    try:
        t0 = time.time()
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=True)
        past = out.past_key_values; nxt = int(torch.argmax(out.logits[0, -1]).item()); gt_man = [nxt]
        while nxt not in eos_ids and len(gt_man) < 10:
            with torch.no_grad():
                out = model(input_ids=torch.tensor([[nxt]], device="cuda"), past_key_values=past, use_cache=True)
            past = out.past_key_values; nxt = int(torch.argmax(out.logits[0, -1]).item()); gt_man.append(nxt)
        t_tm = time.time() - t0
        log("text manual-cache", repr(tok.decode(gt_man, skip_special_tokens=True)), gt_man, "match_stepwise", gt_man == gt_step, f"{t_tm:.2f}s", flush=True)
    except Exception as e:
        man2_err = repr(e); log("text manual-cache FAILED", man2_err, flush=True); traceback.print_exc()
    gen2_err, gt_gen, t_tg = None, None, None
    try:
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(input_ids=torch.tensor([ids], device="cuda"), max_new_tokens=10,
                                 do_sample=False, use_cache=True)
        gt_gen = [int(x) for x in out[0][len(ids):].tolist()]; t_tg = time.time() - t0
        log("text generate", repr(tok.decode(gt_gen, skip_special_tokens=True)), gt_gen, "match_stepwise", gt_gen == gt_step, f"{t_tg:.2f}s", flush=True)
    except Exception as e:
        gen2_err = repr(e); log("text generate FAILED", gen2_err, flush=True); traceback.print_exc()
    res = {"v0": v0, "chip_stepwise": [g - v0 for g in g_step], "chip_expected": [e - v0 for e in exp],
           "chip_stepwise_match": g_step == exp, "chip_stepwise_s": t_step,
           "chip_generate": ([g - v0 for g in g_gen] if g_gen else None), "chip_generate_match": (g_gen == exp if g_gen else None),
           "chip_generate_s": t_gen, "chip_generate_err": gen_err,
           "chip_manual": ([g - v0 for g in g_man] if g_man else None), "chip_manual_match": (g_man == exp if g_man else None),
           "chip_manual_s": t_man, "chip_manual_err": man_err,
           "text_stepwise_ids": gt_step, "text_stepwise": tok.decode(gt_step, skip_special_tokens=True), "text_stepwise_s": t_ts,
           "text_manual_ids": gt_man, "text_manual": (tok.decode(gt_man, skip_special_tokens=True) if gt_man else None),
           "text_manual_match": (gt_man == gt_step if gt_man else None), "text_manual_s": t_tm, "text_manual_err": man2_err,
           "text_generate_ids": gt_gen, "text_generate": (tok.decode(gt_gen, skip_special_tokens=True) if gt_gen else None),
           "text_generate_match": (gt_gen == gt_step if gt_gen else None), "text_generate_s": t_tg, "text_generate_err": gen2_err,
           "vram_max_mib": torch.cuda.max_memory_allocated() / 2**20}
    (OUT / "cache_probe.json").write_text(json.dumps(res, indent=2))
    log("WROTE cache_probe.json", flush=True)
except Exception:
    traceback.print_exc()
    (OUT / "cache_probe_error.txt").write_text(traceback.format_exc())
    raise
