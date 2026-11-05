"""Gate 0 for vec4-v3 with the LOWMEM merge (2B only).
Identical to stage0_v3.py except MERGE_MOD = merge_v3_lowmem and
separate output dir/files, so the stock 2B gate files are untouched.
Usage: python3 stage0_v3_lowmem.py (2B only).
"""
import json, os, platform, shutil, subprocess, sys, time, traceback
from pathlib import Path
MODEL = "2b"
OUT = Path("/content/vec4-v3-2b-lowmem"); OUT.mkdir(parents=True, exist_ok=True)
WORK = Path("/content/qwenputer-colab/vec4-v3")
BASE_DIR = "/content/qwen35-2b"
MERGED_DIR = "/content/qwen35-2b-vec4v3-lowmem"
MERGE_MOD = "merge_v3_lowmem"
HF_NAME = "Qwen/Qwen3.5-2B"
RESULT_FILE = "vec4_merged_result_2b_lowmem.json"
DONE_FILE = "stage0_done_2b_lowmem.json"
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
    if not Path(BASE_DIR, "config.json").exists():
        from huggingface_hub import snapshot_download
        log("downloading", HF_NAME, "...")
        p = snapshot_download(HF_NAME, local_dir=BASE_DIR)
        log("downloaded", p)
    else:
        log("base already present")
    if not Path(MERGED_DIR, "chipformer.json").exists():
        sys.path.insert(0, str(WORK))
        sys.path.insert(0, "/content/qwenputer-colab/upstream")
        sys.argv = [MERGE_MOD + ".py", BASE_DIR, MERGED_DIR]
        mod = __import__(MERGE_MOD)
        mod.main()
        log("merge done")
    else:
        log("merged already present")
    sys.path.insert(0, str(WORK))
    sys.argv = ["run_merged.py", MERGED_DIR, BASE_DIR,
                "--out", str(OUT / RESULT_FILE)]
    import run_merged
    run_merged.main()
    reg = json.loads(Path(OUT / RESULT_FILE).read_text())
    log("REGRESSION", reg["merged_pass"], "vram_max", reg["vram_max_mib"],
        "text", repr(reg["text_output"]))
    (OUT / DONE_FILE).write_text(json.dumps({
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
