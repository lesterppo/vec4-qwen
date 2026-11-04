"""Gate 0 for vec4-2b: probe, download Qwen3.5-2B, merge VEC4-2B,
run 210-case chip-only regression + text sanity.
Writes to /content/vec4-2b/. Adapted from vec4-untested stage0;
originals untouched. NOTE: uses total_memory (not .total) per the
2026-11-01 SCARL driver-bug lesson."""
import json, os, platform, shutil, subprocess, sys, time, traceback
from pathlib import Path
OUT = Path("/content/vec4-2b"); OUT.mkdir(parents=True, exist_ok=True)
WORK = Path("/content/qwenputer-colab/vec4-2b")
def log(*a, **kw): print(*a, flush=True)
try:
    t0 = time.time()
    import torch
    log("probe python", sys.version.split()[0], "torch", torch.__version__)
    log("probe cpu", platform.processor(), "cpus", os.cpu_count())
    try: log("probe mem", open("/proc/meminfo").readlines()[0].strip())
    except Exception as e: log("probe mem fail", repr(e))
    log("probe disk_free_gb", shutil.disk_usage("/content").free / 1e9)
    if torch.cuda.is_available():
        log("probe gpu", torch.cuda.get_device_name(0), "total_mib",
            torch.cuda.get_device_properties(0).total_memory / 2**20)
    else:
        log("probe gpu NONE"); raise AssertionError("no CUDA device")
    if not Path("/content/qwenputer-colab/upstream/chipformer/q35/merge.py").exists():
        log("cloning upstream...")
        r = subprocess.run(["git", "clone", "--depth", "1",
            "https://github.com/divagr18/qwenputer",
            "/content/qwenputer-colab/upstream"], capture_output=True, text=True, timeout=600)
        log(r.stdout[-1000:]); log(r.stderr[-1000:])
        assert Path("/content/qwenputer-colab/upstream/chipformer/q35/merge.py").exists(), "upstream clone failed"
    else:
        log("upstream already present")
    if not Path("/content/qwen35-2b/config.json").exists():
        from huggingface_hub import snapshot_download
        log("downloading Qwen/Qwen3.5-2B...")
        p = snapshot_download("Qwen/Qwen3.5-2B", local_dir="/content/qwen35-2b")
        log("downloaded", p)
    else:
        log("base already present")
    if not Path("/content/qwen35-2b-vec4/chipformer.json").exists():
        sys.path.insert(0, str(WORK))
        sys.path.insert(0, "/content/qwenputer-colab/upstream")
        sys.argv = ["merge_2b.py", "/content/qwen35-2b", "/content/qwen35-2b-vec4"]
        import merge_2b
        merge_2b.main()
        log("merge done")
    else:
        log("merged already present")
    sys.path.insert(0, str(WORK))
    sys.argv = ["run_merged.py", "/content/qwen35-2b-vec4", "/content/qwen35-2b",
                "--out", "/content/vec4-2b/vec4_merged_result.json"]
    import run_merged
    run_merged.main()
    reg = json.loads(Path("/content/vec4-2b/vec4_merged_result.json").read_text())
    log("REGRESSION", reg["merged_pass"], "vram_max", reg["vram_max_mib"],
        "text", repr(reg["text_output"]))
    (OUT / "stage0_done.json").write_text(json.dumps({
        "regression": reg["merged_pass"], "seconds": time.time() - t0,
        "vram_max_mib": reg["vram_max_mib"], "params": reg["params"],
        "bf16_gb": reg["bf16_gb"], "hidden": reg["hidden"],
        "layers": reg["layers"], "vocab": reg["vocab"],
        "text_output": reg["text_output"]}, indent=2))
    log("STAGE0_DONE seconds", time.time() - t0, flush=True)
except Exception:
    traceback.print_exc()
    (OUT / "stage0_error.txt").write_text(traceback.format_exc())
    raise
