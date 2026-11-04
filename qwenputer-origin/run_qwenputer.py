"""Qwenputer free-Colab driver (host-side copy kept in ~/workspace/qwenputer-colab/).

Runs, in order, on ONE Colab VM:
  1. env + nvidia-smi
  2. git clone of the PUBLIC upstream repo (read-only fetch, no push) +
     pip install of its documented hf extra (only if needed)
  3. HF download of divagr1925/Qwen3.5-4B-Qwenputer (unauthenticated;
     if gated, stop and record BLOCKED_GATED)
  4. FULL in-model quickstart, exactly as documented:
       python -m chipformer.rv.chat qwenputer-rv \
         --firmware tests/data/rv/calc-bridge.elf \
         --ask "Use calc: what is 123456789 * 987654321?"
     (plain fast path, RvChat default; NO executor driver, NO quantisation,
      NO weight change -- an OOM here is a real result, not a bug to work around)
  5. ONLY if step 4 fails to load/run on VRAM: standalone RV32 core checks
     needing no merged model -- the reference-machine free run of c-fib
     (expected: fib(25)=75025, exit 0, per tests/data/rv/qemu.json) and
     c-sieve4 (expected: primes below 10000: 1229, exit 0), plus the
     pytest reference suite for those two programs if quick.
Every stage appends to /content/qwenputer_run.log and updates
/content/qwenputer_result.json, so a reclamation still leaves numbers
downloadable. Idempotent: re-running skips completed downloads.
Expected exact product for the in-model check: 121932631112635269.
"""
import json, os, subprocess, sys, time, traceback
from pathlib import Path

LOG = Path("/content/qwenputer_run.log")
RES = Path("/content/qwenputer_result.json")
WORK = Path("/content/qwenputer")
EXPECTED = "121932631112635269"
result = {"expected_product": EXPECTED, "stages": {}}

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S %Z')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")

def save():
    with RES.open("w") as f:
        json.dump(result, f, indent=2)

def run(cmd, cwd=None, timeout=None):
    log(f"$ {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    out = (p.stdout or "") + (p.stderr or "")
    # keep full output in the log file (not just tail) for audit
    with LOG.open("a") as f:
        f.write(out)
        if not out.endswith("\n"):
            f.write("\n")
    log(f"exit={p.returncode} in {dt:.1f}s")
    return p, dt, out

def stage(name):
    t0 = time.time()
    log(f"=== STAGE {name} ===")
    return t0

def end_stage(name, t0, **kw):
    result["stages"][name] = {"seconds": round(time.time() - t0, 1), **kw}
    save()

def main():
    LOG.parent.mkdir(exist_ok=True)
    save()
    # ---- env ----
    t0 = stage("env")
    p, _, out = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                     "--format=csv"])
    result["gpu_smi"] = out.strip()
    p, _, out = run(["python3", "-c",
                     "import sys, torch, transformers; "
                     "print('python', sys.version.split()[0]); "
                     "print('torch', torch.__version__); "
                     "print('transformers', transformers.__version__); "
                     "print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); "
                     "print('vram_total_mib', torch.cuda.get_device_properties(0).total_memory//1048576 if torch.cuda.is_available() else None)"])
    result["env"] = out.strip()
    end_stage("env", t0, output=result["env"])

    # ---- clone (public read-only fetch) ----
    t0 = stage("clone")
    if not (WORK / "experimental/riscv/chipformer/rv/chat.py").exists():
        p, dt, _ = run(["git", "clone", "--depth", "1",
                        "https://github.com/divagr18/qwenputer", str(WORK)])
        if p.returncode != 0:
            end_stage("clone", t0, ok=False, error="git clone failed")
            result["verdict"] = "BLOCKED_CLONE_FAILED"; save(); return
    else:
        log("clone already present, skipping")
    end_stage("clone", t0, ok=True)
    RV = WORK / "experimental/riscv"

    # ---- install documented hf extra ----
    t0 = stage("install")
    p, _, out = run([sys.executable, "-m", "pip", "install", "-q",
                     "numpy>=1.26", "safetensors>=0.8.0",
                     "torch>=2.14.1", "transformers>=5.19.0",
                     "huggingface_hub", "pytest"])
    # transformers>=5.19 may not exist as a published line for this torch;
    # record, do not hide: if pip failed, the import step below will say so.
    end_stage("install", t0, pip_exit=p.returncode)

    # ---- HF download ----
    t0 = stage("download")
    model_dir = WORK / "qwenputer-rv"
    if not (model_dir / "config.json").exists():
        p, dt, out = run([sys.executable, "-m", "huggingface_hub.cli.hf",
                          "download", "divagr1925/Qwen3.5-4B-Qwenputer",
                          "--local-dir", str(model_dir)], timeout=3600)
        if p.returncode != 0:
            end_stage("download", t0, ok=False, error=out[-2000:])
            result["verdict"] = "BLOCKED_DOWNLOAD_FAILED"; save(); return
    else:
        log("model dir already present, skipping download")
    _, _, duout = run(["du", "-sh", str(model_dir)])
    _, _, filesout = run(["ls", "-lh", str(model_dir)])
    end_stage("download", t0, ok=True, size=duout.strip(), files=filesout.strip()[:2000])
    log(f"MODEL SIZE: {duout.strip()}")

    # ---- FULL in-model quickstart (exact official command, plain path) ----
    t0 = stage("full_in_model")
    p, dt, out = run([sys.executable, "-m", "chipformer.rv.chat", str(model_dir),
                      "--firmware", "tests/data/rv/calc-bridge.elf",
                      "--ask", "Use calc: what is 123456789 * 987654321?"],
                     cwd=str(RV), timeout=3600)
    result["full"] = {"exit": p.returncode, "seconds": round(dt, 1),
                      "output_tail": out[-4000:],
                      "product_present": EXPECTED in out}
    end_stage("full_in_model", t0, exit=p.returncode,
              product_present=EXPECTED in out)
    if p.returncode == 0 and EXPECTED in out:
        result["verdict"] = "FULL_IN_MODEL_GREEN"; save(); log("VERDICT FULL_IN_MODEL_GREEN"); return
    log(f"FULL RUN did not green: exit={p.returncode} product_present={EXPECTED in out}")

    # ---- fallback ONLY if full could not load/run (OOM etc): standalone core ----
    t0 = stage("fallback_core")
    fb = {}
    for name, expected_out in [("c-fib", "fib(25)=75025"),
                               ("c-sieve4", "primes below 10000: 1229")]:
        p2, dt2, out2 = run([sys.executable, "-c",
            "import sys; sys.path.insert(0, '.'); "
            "from pathlib import Path; "
            "from chipformer.rv import elf, ref; "
            f"m = ref.Machine.from_program(elf.program(Path('tests/data/rv/{name}.elf'))); "
            "toks = m.run(); "
            f"print('halted', m.halted, 'fault', m.fault, 'exit', m.exit_code, 'stdout', repr(m.out.decode('latin-1')))"],
            cwd=str(RV), timeout=1200)
        fb[name] = {"exit": p2.returncode, "seconds": round(dt2, 1),
                    "expected_stdout": expected_out,
                    "expected_present": expected_out in out2,
                    "output_tail": out2[-1000:]}
    result["fallback"] = fb
    end_stage("fallback_core", t0, **{k: v["expected_present"] for k, v in fb.items()})
    if "out of memory" in (result["full"]["output_tail"] or "").lower() or \
       "CUDA out of memory" in out or "cuda out of memory" in out.lower():
        result["verdict"] = "FALLBACK_CORE_ONLY"
    else:
        result["verdict"] = "FULL_IN_MODEL_FAILED"
    save()
    log(f"VERDICT {result['verdict']}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("DRIVER EXCEPTION:\n" + traceback.format_exc())
        result["verdict"] = "FULL_IN_MODEL_FAILED_DRIVER_EXCEPTION"
        result["driver_exception"] = traceback.format_exc()[-2000:]
        save()
    print("QWENPUTER_DONE_MARKER " + result.get("verdict", "UNKNOWN"), flush=True)
