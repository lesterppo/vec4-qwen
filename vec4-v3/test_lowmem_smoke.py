"""End-to-end smoke test of merge_v3_lowmem.main() on a FAKE tiny base.

Builds a miniature Qwen-shaped checkpoint (hidden 64, 2 layers, inter
512, vocab 1000) with keys covering every widen pattern (IN_COLS,
OUT_ROWS, WIDE_NORMS, untouched vision keys, merger bias), runs the
real lowmem merge -- including the full 26-layer v3 chip emit through
emit_layers_sink -- and verifies the output directory: index covers
exactly the expected key set, every shard reloads, embed/norm shapes
and values follow the stock math, config and chipformer.json are right.

This exercises the whole driver flow (load, widen, shift, mute,
shard-out, per-layer emit, embed rebuild, norm edit, sharded save)
without real weights. It does NOT prove the 2B numbers -- that proof
is the Colab Gate 0 (720/720) with this driver.

Usage: python3 test_lowmem_smoke.py
"""

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "upstream"))
sys.path.insert(0, str(HERE))

import torch
from safetensors.torch import load_file, save_file

TEXT = "model.language_model."
D0, V0, INTER, N0 = 64, 1000, 512, 2
CHIP_KEYS = ["input_layernorm.weight", "post_attention_layernorm.weight",
             "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
             "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight",
             "linear_attn.conv1d.weight", "linear_attn.dt_bias",
             "linear_attn.A_log", "linear_attn.norm.weight",
             "linear_attn.out_proj.weight", "mlp.gate_proj.weight",
             "mlp.up_proj.weight", "mlp.down_proj.weight"]


def fake_base(src):
    src.mkdir(parents=True, exist_ok=True)
    cfg = {"text_config": {
        "hidden_size": D0, "vocab_size": V0, "intermediate_size": INTER,
        "num_hidden_layers": N0,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": 16, "linear_num_value_heads": 16,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4, "head_dim": 128},
        "vision_config": {"out_hidden_size": D0}}
    (src / "config.json").write_text(json.dumps(cfg))
    (src / "tokenizer.json").write_text('{"fake": true}')
    sd = {TEXT + "embed_tokens.weight":
          torch.randn(V0, D0, dtype=torch.bfloat16),
          TEXT + "norm.weight": torch.randn(D0, dtype=torch.bfloat16),
          "model.visual.patch_embed.weight":
          torch.randn(8, 8, dtype=torch.bfloat16),
          "model.visual.merger.linear_fc2.bias":
          torch.randn(D0, dtype=torch.bfloat16)}
    for i, lt in enumerate(cfg["text_config"]["layer_types"]):
        p = f"{TEXT}layers.{i}."
        if lt == "linear_attention":
            for name in ("in_proj_a", "in_proj_b", "in_proj_z"):
                sd[p + f"linear_attn.{name}.weight"] = torch.randn(
                    32, D0, dtype=torch.bfloat16)
        else:
            sd[p + "self_attn.q_proj.weight"] = torch.randn(
                512, D0, dtype=torch.bfloat16)
            sd[p + "self_attn.k_proj.weight"] = torch.randn(
                256, D0, dtype=torch.bfloat16)
        sd[p + "mlp.gate_proj.weight"] = torch.randn(
            INTER, D0, dtype=torch.bfloat16)
        sd[p + "mlp.up_proj.weight"] = torch.randn(
            INTER, D0, dtype=torch.bfloat16)
        sd[p + "mlp.down_proj.weight"] = torch.randn(
            D0, INTER, dtype=torch.bfloat16)
        sd[p + "input_layernorm.weight"] = torch.randn(D0, dtype=torch.bfloat16)
        sd[p + "post_attention_layernorm.weight"] = torch.randn(
            D0, dtype=torch.bfloat16)
    save_file(sd, str(src / "model.safetensors"), metadata={"format": "pt"})
    return set(sd)


def main():
    import merge_v3_lowmem as LM
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "base", Path(tmp) / "merged"
        base_keys = fake_base(src)
        sys.argv = ["merge_v3_lowmem.py", str(src), str(dst)]
        LM.main()
        index = json.loads((dst / "model.safetensors.index.json").read_text())
        cfg = json.loads((dst / "config.json").read_text())
        meta = json.loads((dst / "chipformer.json").read_text())
        hidden = cfg["text_config"]["hidden_size"]
        vocab = cfg["text_config"]["vocab_size"]
        assert cfg["text_config"]["num_hidden_layers"] == N0 + 26, cfg
        assert vocab == -(-(V0 + 252) // 64) * 64 == 1280, vocab
        assert meta["layout"] == "vec4-v3" and meta["lowmem"] is True, meta
        assert (dst / "tokenizer.json").exists(), "tokenizer not copied"
        assert not list(dst.glob("*.tmp.safetensors")), "temp shards left"

        expected = set()
        for k in base_keys:
            if k.startswith(TEXT + "layers."):
                i = int(k.split(".")[3])
                expected.add(k.replace(f"layers.{i}.", f"layers.{i + 26}."))
            else:
                expected.add(k)
        for L in range(26):
            for ck in CHIP_KEYS:
                expected.add(f"{TEXT}layers.{L}.{ck}")
        assert set(index["weight_map"]) == expected, (
            f"missing={sorted(expected - set(index['weight_map']))[:5]} "
            f"extra={sorted(set(index['weight_map']) - expected)[:5]}")

        loaded = {}
        for name in sorted(set(index["weight_map"].values())):
            loaded.update(load_file(str(dst / name)))
        assert set(loaded) == expected, "reload key mismatch"
        emb = loaded[TEXT + "embed_tokens.weight"]
        assert tuple(emb.shape) == (vocab, hidden), emb.shape
        assert emb.dtype == torch.bfloat16
        norm = loaded[TEXT + "norm.weight"]
        assert tuple(norm.shape) == (hidden,), norm.shape
        # chip dims are -1 except code/tau dims (which the stock math
        # overwrites with gains); most chip dims must still read -1.
        assert int((norm[D0:] == -1.0).sum()) > (hidden - D0) // 2
        chip_gate = loaded[f"{TEXT}layers.0.mlp.gate_proj.weight"]
        assert tuple(chip_gate.shape) == (INTER, hidden), chip_gate.shape
        assert chip_gate.dtype == torch.bfloat16
        print(f"smoke: hidden {hidden}, vocab {vocab}, "
              f"{len(expected)} keys over "
              f"{len(set(index['weight_map'].values()))} shards, "
              f"total {index['metadata']['total_size'] / 1e6:.1f} MB -- OK")


if __name__ == "__main__":
    main()
    print("LOWMEM SMOKE TEST PASS")
