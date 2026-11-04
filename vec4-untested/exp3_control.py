"""I3 control (post-hoc, clearly labelled): the SAME 5 factual prompts as I3,
with NO chip block, same cached driver, same caps. Decides whether I3's
one wrong answer (largest planet -> Sun) is chip-induced or base 0.8B
quality. Writes /content/vec4-untested/exp3_control.json."""
import json, sys, time, traceback
from pathlib import Path
OUT = Path("/content/vec4-untested"); OUT.mkdir(parents=True, exist_ok=True)
def log(*a, **kw): print(*a, flush=True)
def main():
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/content/qwen35-08b")
    eos_ids = set()
    for tname in ("<|im_end|>", "<|endoftext|>"):
        try: eos_ids.update(tok.encode(tname, add_special_tokens=False))
        except Exception: pass
    if tok.eos_token_id is not None: eos_ids.add(tok.eos_token_id)
    model = AutoModelForImageTextToText.from_pretrained("/content/qwen35-08b-vec4", dtype=torch.bfloat16).to("cuda").eval()
    log("loaded VRAM MiB", torch.cuda.memory_allocated()/2**20, flush=True)
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
        return gen
    specs = [(" The capital of France is", "Paris"), (" The capital of France is", "Paris"),
             (" The capital of Japan is", "Tokyo"), (" The largest planet is", "Jupiter"),
             (" The capital of Italy is", "Rome")]
    rows = []
    for i, (prompt_text, expected_word) in enumerate(specs):
        t0 = time.time(); ids = tok.encode(prompt_text, add_special_tokens=False)
        gen = gen_cached(ids, 40); final = tok.decode(gen, skip_special_tokens=True)[:2000]
        row = {"spec": i, "prompt": prompt_text, "expected_word": expected_word,
               "final_text": final, "contains_expected": expected_word in final,
               "seconds": time.time() - t0}
        rows.append(row)
        log(f"CTRL {i} {prompt_text!r} contains={row['contains_expected']} final={final[:150]!r}", flush=True)
        (OUT / "exp3_control_partial.json").write_text(json.dumps({"rows": rows}, indent=2))
    (OUT / "exp3_control.json").write_text(json.dumps({"rows": rows, "contains": sum(1 for r in rows if r["contains_expected"]), "vram_max_mib": torch.cuda.max_memory_allocated()/2**20}, indent=2))
    log("WROTE exp3_control.json", sum(1 for r in rows if r["contains_expected"]), "/5", flush=True)
if __name__ == "__main__":
    try: main()
    except Exception:
        traceback.print_exc(); (OUT / "exp3_control_error.txt").write_text(traceback.format_exc()); raise
