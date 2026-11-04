"""Stage 0 for the efficiency comparison: probe, clone upstream if
needed, download both bases, merge both VEC4 v2 models (0.8B via
merge_08b.py, 2B via merge_2b.py with in-place widening). Writes to
/content/vec4-eff/. NOTE: use exec_detach + a log file for this
script (lesson from the 2B-v2 run: the exec wrapper cuts long runs
short). Correctness gating happens per model inside eff.py (60/60
required before any timing is reported); this stage only merges.
log() takes **kw (flush=True collision class).
"""
import json, os, platform, shutil, subprocess, sys, time, traceback
from pathlib import Path
OUT = Path("/content/vec4-eff"); OUT.mkdir(parents=True, exist_ok=True)
WORK = Path("/content/qwenputer-colab/vec4-eff")
def log(*a, **kw): print(*a, flush=True)
try:
    t0 = time.time()
    import torch
    log("probe python", sys.version.split()[0], "torch", torch.__version__)
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
    from huggingface_hub import snapshot_download
    for name, dest in (("Qwen/Qwen3.5-0.8B", "/content/qwen35-08b"),
                       ("Qwen/Qwen3.5-2B", "/content/qwen35-2b")):
        if not Path(dest, "config.json").exists():
            log("downloading", name, "...")
            snapshot_download(name, local_dir=dest)
        else:
            log("base already present:", dest)
    sys.path.insert(0, str(WORK))
    sys.path.insert(0, "/content/qwenputer-colab/upstream")
    merged = {}
    for script, mod, src, dst in (
            ("merge_08b.py", "merge_08b", "/content/qwen35-08b", "/content/qwen35-08b-vec4v2"),
            ("merge_2b.py", "merge_2b", "/content/qwen35-2b", "/content/qwen35-2b-vec4v2")):
        if Path(dst, "chipformer.json").exists():
            log("merged already present:", dst)
        else:
            sys.argv = [script, src, dst]
            m = __import__(mod)
            m.main()
            log("merge done:", dst)
        meta = json.loads(Path(dst, "chipformer.json").read_text())
        cfg = json.loads(Path(dst, "config.json").read_text())
        merged[dst] = {"hidden": meta["hidden"], "vocab": cfg["text_config"]["vocab_size"],
                       "layers": cfg["text_config"]["num_hidden_layers"],
                       "chip_vocab": meta["chip_vocab"], "layout": meta["layout"]}
        log("merged meta:", dst, merged[dst], flush=True)
    (OUT / "stage0_eff_done.json").write_text(json.dumps(
        {"merged": merged, "seconds": time.time() - t0}, indent=2))
    log("STAGE0_EFF_DONE seconds", time.time() - t0, flush=True)
except Exception:
    traceback.print_exc()
    (OUT / "stage0_eff_error.txt").write_text(traceback.format_exc())
    raise
