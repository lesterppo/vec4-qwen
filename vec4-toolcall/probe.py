import json, platform, subprocess, sys, time
print("probe start", flush=True)
try:
    import torch, transformers
    print("python", sys.version.split()[0], "torch", torch.__version__, "transformers", transformers.__version__, "cuda", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0), "total_mib", torch.cuda.get_device_properties(0).total_memory//2**20, flush=True)
except Exception as e:
    print("torch probe failed", repr(e), flush=True)
for cmd in [["nvidia-smi","--query-gpu=name,memory.total,memory.used","--format=csv,noheader"],["lscpu"],["free","-g"],["df","-h","/content"]]:
    try:
        print("$", " ".join(cmd), flush=True)
        print(subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout[:2000], flush=True)
    except Exception as e:
        print("cmd failed", cmd, repr(e), flush=True)
# tokenizer + chat template probe (no model load yet, base download needed first)
print("probe done", flush=True)
