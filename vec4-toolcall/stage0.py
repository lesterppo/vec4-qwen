"""Gate 0: download base, clone upstream, merge VEC4, regression 210/210, template probe.
Writes /content/vec4-toolcall/stage0_done.json on completion (or stage0_error.txt).
"""
import json, os, subprocess, sys, time, traceback
from pathlib import Path
OUT = Path("/content/vec4-toolcall"); OUT.mkdir(parents=True, exist_ok=True)
def log(*a):
    print(*a, flush=True)
try:
    t0 = time.time()
    # upstream clone (needed by merge_vec4's sys.path ROOT/upstream)
    if not (Path("/content/qwenputer-colab/upstream/chipformer/q35/merge.py").exists()):
        log("cloning upstream...")
        r = subprocess.run(["git","clone","--depth","1","https://github.com/divagr18/qwenputer","/content/qwenputer-colab/upstream"], capture_output=True, text=True, timeout=600)
        log(r.stdout[-2000:]); log(r.stderr[-2000:])
        assert Path("/content/qwenputer-colab/upstream/chipformer/q35/merge.py").exists(), "upstream clone failed"
    else:
        log("upstream already present")
    # base download
    if not Path("/content/qwen35-08b/config.json").exists():
        from huggingface_hub import snapshot_download
        log("downloading Qwen/Qwen3.5-0.8B...")
        p = snapshot_download("Qwen/Qwen3.5-0.8B", local_dir="/content/qwen35-08b")
        log("downloaded", p, sorted(os.listdir(p)))
    else:
        log("base already present")
    # merge
    if not Path("/content/qwen35-08b-vec4/chipformer.json").exists():
        sys.path.insert(0, "/content/qwenputer-colab/vec4")
        sys.path.insert(0, "/content/qwenputer-colab/upstream")
        sys.argv = ["merge_vec4.py", "/content/qwen35-08b", "/content/qwen35-08b-vec4"]
        import merge_vec4
        merge_vec4.main()
        log("merge done")
    else:
        log("merged already present")
    # regression: run_merged.py (210 cases, stepwise)
    sys.path.insert(0, "/content/qwenputer-colab/vec4")
    sys.argv = ["run_merged.py", "/content/qwen35-08b-vec4", "/content/qwen35-08b", "--out", "/content/vec4-toolcall/vec4_merged_result.json"]
    import run_merged
    run_merged.main()
    reg = json.loads(Path("/content/vec4-toolcall/vec4_merged_result.json").read_text())
    log("REGRESSION", reg["merged_pass"], "vram_max", reg["vram_max_mib"])
    # template probe (tokenizer only)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/content/qwen35-08b")
    tools = [
        {"type":"function","function":{"name":"vec_add","description":"Add two 4-lane vectors lane by lane, modulo 16. Each vector has exactly 4 integers, each 0 to 15.","parameters":{"type":"object","properties":{"a":{"type":"array","items":{"type":"integer"},"description":"First vector, 4 integers 0-15"},"b":{"type":"array","items":{"type":"integer"},"description":"Second vector, 4 integers 0-15"}},"required":["a","b"]}}},
        {"type":"function","function":{"name":"vec_dot","description":"Dot product of two 4-lane vectors: sum over lanes of a[i]*b[i]. Each vector has exactly 4 integers, each 0 to 15. Maximum result is 900.","parameters":{"type":"object","properties":{"a":{"type":"array","items":{"type":"integer"},"description":"First vector, 4 integers 0-15"},"b":{"type":"array","items":{"type":"integer"},"description":"Second vector, 4 integers 0-15"}},"required":["a","b"]}}},
    ]
    msgs = [{"role":"system","content":"You have tools vec_add and vec_dot for exact vector arithmetic. Use them for every vector calculation instead of calculating by hand."},
            {"role":"user","content":"What is the dot product of [3, 5, 7, 9] and [2, 4, 6, 8]? Use vec_dot."}]
    try:
        ids = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, enable_thinking=False, tokenize=True)
        txt = tok.decode(ids)
        log("TEMPLATE_OK len", len(ids))
        log("TEMPLATE_TEXT>>>", txt, "<<<END")
        (OUT/"template_prompt.txt").write_text(txt)
        (OUT/"template_prompt_ids.json").write_text(json.dumps(ids))
    except Exception as e:
        log("TEMPLATE_FAIL", repr(e)); traceback.print_exc()
        # fallback: try without tools, and with tools as plain arg
        try:
            ids2 = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True)
            log("TEMPLATE_NO_TOOLS_OK", tok.decode(ids2))
        except Exception as e2:
            log("TEMPLATE_NO_TOOLS_FAIL", repr(e2))
    # render a full tool conversation to learn exact tool_response segment
    try:
        msgs2 = [
            {"role":"system","content":"You have tools vec_add and vec_dot for exact vector arithmetic."},
            {"role":"user","content":"What is the dot product of [3, 5, 7, 9] and [2, 4, 6, 8]? Use vec_dot."},
            {"role":"assistant","content":None,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"vec_dot","arguments":'{"a": [3, 5, 7, 9], "b": [2, 4, 6, 8]}'}}]},
            {"role":"tool","content":"140","tool_call_id":"call_1"},
        ]
        txt2 = tok.apply_chat_template(msgs2, tools=tools, add_generation_prompt=True, enable_thinking=False, tokenize=False)
        log("TEMPLATE_FULL>>>", txt2, "<<<END")
        (OUT/"template_full.txt").write_text(txt2)
    except Exception as e:
        log("TEMPLATE_FULL_FAIL", repr(e)); traceback.print_exc()
    (OUT/"stage0_done.json").write_text(json.dumps({"regression": reg["merged_pass"], "seconds": time.time()-t0, "vram_max_mib": reg["vram_max_mib"]}, indent=2))
    log("STAGE0 DONE", time.time()-t0)
except Exception as e:
    log("STAGE0 ERROR", repr(e)); traceback.print_exc()
    (OUT/"stage0_error.txt").write_text(traceback.format_exc())
    raise
