"""Gate 0 for vec4-fix: probe, download base 0.8B, clone upstream if needed,
merge VEC4, run 210-case chip-only regression. Writes to /content/vec4-fix/."""
import json, os, platform, shutil, subprocess, sys, time, traceback
from pathlib import Path
OUT = Path("/content/vec4-fix"); OUT.mkdir(parents=True, exist_ok=True)
WORK = Path("/content/qwenputer-colab/vec4-fix")
def log(*a):
    print(*a, flush=True)
try:
    t0 = time.time()
    import torch
    log("probe python", sys.version.split()[0], "torch", torch.__version__)
    log("probe cpu", platform.processor(), "cpus", os.cpu_count())
    try:
        log("probe mem", open("/proc/meminfo").readlines()[0].strip())
    except Exception as e:
        log("probe mem fail", repr(e))
    log("probe disk_free_gb", shutil.disk_usage("/content").free / 1e9)
    if torch.cuda.is_available():
        log("probe gpu", torch.cuda.get_device_name(0), "total_mib", torch.cuda.get_device_properties(0).total_memory / 2**20)
    else:
        log("probe gpu NONE")
    if not Path("/content/qwenputer-colab/upstream/chipformer/q35/merge.py").exists():
        log("cloning upstream...")
        r = subprocess.run(["git", "clone", "--depth", "1", "https://github.com/divagr18/qwenputer", "/content/qwenputer-colab/upstream"], capture_output=True, text=True, timeout=600)
        log(r.stdout[-1000:]); log(r.stderr[-1000:])
        assert Path("/content/qwenputer-colab/upstream/chipformer/q35/merge.py").exists(), "upstream clone failed"
    else:
        log("upstream already present")
    if not Path("/content/qwen35-08b/config.json").exists():
        from huggingface_hub import snapshot_download
        log("downloading Qwen/Qwen3.5-0.8B...")
        p = snapshot_download("Qwen/Qwen3.5-0.8B", local_dir="/content/qwen35-08b")
        log("downloaded", p)
    else:
        log("base already present")
    if not Path("/content/qwen35-08b-vec4/chipformer.json").exists():
        sys.path.insert(0, str(WORK))
        sys.path.insert(0, "/content/qwenputer-colab/upstream")
        sys.argv = ["merge_vec4.py", "/content/qwen35-08b", "/content/qwen35-08b-vec4"]
        import merge_vec4
        merge_vec4.main()
        log("merge done")
    else:
        log("merged already present")
    sys.path.insert(0, str(WORK))
    sys.argv = ["run_merged.py", "/content/qwen35-08b-vec4", "/content/qwen35-08b", "--out", "/content/vec4-fix/vec4_merged_result.json"]
    import run_merged
    run_merged.main()
    reg = json.loads(Path("/content/vec4-fix/vec4_merged_result.json").read_text())
    log("REGRESSION", reg["merged_pass"], "vram_max", reg["vram_max_mib"])
    (OUT / "stage0_done.json").write_text(json.dumps({"regression": reg["merged_pass"], "seconds": time.time() - t0, "vram_max_mib": reg["vram_max_mib"]}, indent=2))
    log("STAGE0_DONE seconds", time.time() - t0, flush=True)
except Exception:
    traceback.print_exc()
    (OUT / "stage0_error.txt").write_text(traceback.format_exc())
    raise
