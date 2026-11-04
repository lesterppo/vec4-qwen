"""Second driver, SAME Colab session, correcting two first-driver errors (NOT a new session):
  A. First driver used system pip, upgrading torch 2.11 -> 2.15 and breaking the
     preinstalled torchvision (torchvision::nms import error), so the full chat
     died at import in 14 s before any model load. That was an environment
     breakage, not a model/VRAM result.
  B. First fallback c-fib used Machine.run() default max_records=1e6, which
     truncates fib(25) (2.1M instructions) before HALT -- hence halted=False.
This driver uses the OFFICIAL path verbatim: `uv sync --extra hf` in
experimental/riscv (isolated .venv, no system torchvision) then
`uv run python -m chipformer.rv.chat ../qwenputer-rv --firmware
tests/data/rv/calc-bridge.elf --ask "Use calc: what is 123456789 * 987654321?"`
Model dir differs from the README's qwenputer-rv only because the download
landed at /content/qwenputer/qwenputer-rv (parent of experimental/riscv).
VRAM is polled via nvidia-smi while the chat subprocess runs; peak recorded.
If the full run OOMs/fails, corrected fallback runs: Rv32 ISA free run of
c-fib (expected fib(25)=75025, exit 0) and traced Machine run of c-sieve4
(expected primes below 10000: 1229, exit 0), both vs tests/data/rv/qemu.json.
Logs: /content/qwenputer_uv.log ; result: /content/qwenputer_result_uv.json
Expected product: 121932631112635269
"""
import json, subprocess, sys, time, traceback
from pathlib import Path

LOG = Path("/content/qwenputer_uv.log")
RES = Path("/content/qwenputer_result_uv.json")
WORK = Path("/content/qwenputer")
RV = WORK / "experimental/riscv"
EXPECTED = "121932631112635269"
result = {"expected_product": EXPECTED, "stages": {}, "correction_note":
          "First system-pip attempt died at import (torch 2.15 vs torchvision 0.26); "
          "this is the official uv-venv attempt in the same session."}

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S %Z')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")

def save():
    with RES.open("w") as f:
        json.dump(result, f, indent=2)

def run_cmd(cmd, cwd=None, timeout=3600):
    log(f"$ {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    out = (p.stdout or "") + (p.stderr or "")
    with LOG.open("a") as f:
        f.write(out); f.write("\n")
    log(f"exit={p.returncode} in {dt:.1f}s")
    return p, dt, out

def vram_mib():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout.strip()
        return int(o.splitlines()[0])
    except Exception:
        return None

def main():
    save()
    t0 = time.time(); log("=== STAGE uv_sync ===")
    p, dt, out = run_cmd(["uv", "sync", "--extra", "hf"], cwd=str(RV), timeout=1800)
    result["stages"]["uv_sync"] = {"seconds": round(dt, 1), "exit": p.returncode}
    save()
    if p.returncode != 0:
        result["verdict"] = "BLOCKED_UV_SYNC_FAILED"; save(); return

    # Full run, with VRAM polling
    log("=== STAGE full_in_model_uv ===")
    t0 = time.time()
    # Absolute model path: RV is /content/qwenputer/experimental/riscv, model is
    # /content/qwenputer/qwenputer-rv (NOT ../qwenputer-rv -- that was driver bug #2).
    model_abs = str(WORK / "qwenputer-rv")
    assert (WORK / "qwenputer-rv" / "chipformer.json").exists(), "model dir missing chipformer.json"
    log(f"model dir verified: {model_abs}")
    chat = subprocess.Popen(
        ["uv", "run", "python", "-m", "chipformer.rv.chat", model_abs,
         "--firmware", "tests/data/rv/calc-bridge.elf",
         "--ask", "Use calc: what is 123456789 * 987654321?"],
        cwd=str(RV), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    peak = 0; first_out_at = None
    chunks = []
    while True:
        line = chat.stdout.readline()
        if line:
            if first_out_at is None:
                first_out_at = time.time() - t0
            chunks.append(line)
            with LOG.open("a") as f:
                f.write(line)
        v = vram_mib()
        if v is not None:
            peak = max(peak, v)
        if chat.poll() is not None and not line:
            break
        if time.time() - t0 > 3600:
            chat.kill(); break
    out = "".join(chunks)
    dt = time.time() - t0
    result["full"] = {"exit": chat.returncode, "seconds": round(dt, 1),
                      "peak_vram_mib_polled": peak,
                      "first_output_seconds": round(first_out_at, 1) if first_out_at else None,
                      "product_present": EXPECTED in out,
                      "output_tail": out[-4000:]}
    result["stages"]["full_in_model_uv"] = {"seconds": round(dt, 1), "exit": chat.returncode,
                                            "peak_vram_mib_polled": peak,
                                            "product_present": EXPECTED in out}
    save()
    log(f"FULL exit={chat.returncode} secs={dt:.1f} peak_vram={peak}MiB product={EXPECTED in out}")
    if chat.returncode == 0 and EXPECTED in out:
        result["verdict"] = "FULL_IN_MODEL_GREEN"; save(); log("VERDICT FULL_IN_MODEL_GREEN"); return

    # Corrected fallback (only if full failed)
    log("=== STAGE fallback_core_corrected ===")
    fb = {}
    # c-fib via fast ISA emulator (free run), vs qemu.json expected.
    # NOTE: must be a script file -- python -c cannot hold a while loop after ';'.
    fib_script = Path("/content/fib_check.py")
    if not fib_script.exists():
        fib_script.write_text(
            "import sys\nfrom pathlib import Path\nsys.path.insert(0, '.')\n"
            "from chipformer.rv import elf, ref\n"
            "cpu = ref.Rv32()\nelf.load_into(cpu, Path('tests/data/rv/c-fib.qemu.elf'))\n"
            "cpu.x[2] = 0x7FFF0000\nsteps = 0\n"
            "while not cpu.halted and steps < 20000000:\n    cpu.step()\n    steps += 1\n"
            "print('halted', cpu.halted, 'steps', steps, 'exit', cpu.exit_code, "
            "'stdout', repr(cpu.out.decode('latin-1')))\n")
    p, dt, out = run_cmd([sys.executable, str(fib_script)],
        cwd=str(RV), timeout=1200)
    fb["c-fib_isa"] = {"exit": p.returncode, "seconds": round(dt, 1),
                       "expected_stdout": "fib(25)=75025", "expected_present": "fib(25)=75025" in out,
                       "output_tail": out[-1000:]}
    # c-sieve4 via traced machine (already passed once; re-confirm with explicit run)
    p, dt, out = run_cmd([sys.executable, "-c",
        "import sys; sys.path.insert(0, '.'); "
        "from pathlib import Path; "
        "from chipformer.rv import elf, ref; "
        "m = ref.Machine.from_program(elf.program(Path('tests/data/rv/c-sieve4.elf'))); toks=m.run(); "
        "print('halted', m.halted, 'fault', m.fault, 'exit', m.exit_code, 'stdout', repr(m.out.decode('latin-1')), 'trace_tokens', len(toks))"],
        cwd=str(RV), timeout=1200)
    fb["c-sieve4_traced"] = {"exit": p.returncode, "seconds": round(dt, 1),
                             "expected_stdout": "primes below 10000: 1229",
                             "expected_present": "primes below 10000: 1229" in out,
                             "output_tail": out[-1000:]}
    result["fallback"] = fb
    save()
    if "out of memory" in out.lower() or "CUDA out of memory" in result["full"]["output_tail"]:
        result["verdict"] = "FALLBACK_CORE_ONLY"
    else:
        result["verdict"] = "FULL_IN_MODEL_FAILED"
    save(); log(f"VERDICT {result['verdict']}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("DRIVER EXCEPTION:\n" + traceback.format_exc())
        result["verdict"] = "FULL_IN_MODEL_FAILED_DRIVER_EXCEPTION"
        result["driver_exception"] = traceback.format_exc()[-2000:]
        save()
    print("QWENPUTER_UV_DONE_MARKER " + result.get("verdict", "UNKNOWN"), flush=True)
