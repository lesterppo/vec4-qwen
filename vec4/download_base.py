import json
import os
import subprocess

from huggingface_hub import snapshot_download

p = snapshot_download("Qwen/Qwen3.5-0.8B", local_dir="/content/qwen35-08b")
print("downloaded to", p)
print(sorted(os.listdir(p)))
cfg = json.load(open(p + "/config.json"))
print("text hidden", cfg["text_config"]["hidden_size"],
      "vocab", cfg["text_config"]["vocab_size"],
      "layers", cfg["text_config"]["num_hidden_layers"],
      "layer_types", cfg["text_config"]["layer_types"])
print("vision out", cfg["vision_config"].get("out_hidden_size"))
print(subprocess.run(["du", "-sh", p], capture_output=True, text=True).stdout)
