"""Stage 3 (merge): VEC4 circuit into real Qwen3.5-2B, on Colab.

2B RETARGET (vec4-2b copy of ../vec4/merge_vec4.py; original untouched).
Two deliberate deviations from the 0.8B script, both documented:
1. circuit.py Target is (16, 16, 6144) instead of (16, 16, 3584);
   merge reads inter=6144 from the 2B config for allocate() anyway.
2. Widening is done IN PLACE per key (widen_inplace below) instead of
   M.widen's build-a-second-dict: the math is identical (same regexes,
   same scale (d/hidden)**0.5, same norm gain formula, same config
   edits), but the old tensor is freed as each new one is assigned,
   so peak RAM is ~new size instead of old+new. This matters at 2B:
   stock widening would transiently hold ~4.5 GB (base) + ~6.7 GB
   (widened) on a 12-13 GB VM. Gate 0 (210/210) + text sanity are the
   verification that the widening is correct.
Otherwise identical: Reuses merge.py's pick_hidden/load logic and mute.mute
(without anti-sink: test sequences are chip-only or text-only, and
Qwen's full-attention output gate is muted on chip tokens, so attention
dilution cannot affect either test), plus emit.allocate/Layout/
embedding_rows/emit_layers and tiny.code_gain. Replaces CHIP-8-specific
BIOS/embedding/tokenizer logic with VEC4's own:
  - circuit built with offset=v0 (Qwen vocab size), so ids < v0 are text
    tokens and v0..v0+243 are VEC4 tokens;
  - keep = code + mu + tau (mu for mute, tau for text-vs-chip separation
    at the final norm; code-only keep would clear both at exit);
  - Qwen rows get the chip features of a representative text token;
  - chip rows are the 244 VEC4 tokens appended after Qwen's vocab;
  - tokenizer files are copied unchanged (chip tokens are addressed by
    id in tests, not via the tokenizer);
  - no CHIP-8 bad_words_ids, no bios, no store.

Usage: python3 merge_vec4.py SRC_DIR OUT_DIR
Prints hidden, slots, size estimate, and actual size on disk.
"""
import copy
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "upstream"))
sys.path.insert(0, str(HERE))

import torch
from safetensors.torch import save_file

from chipformer.q35 import merge as M
from chipformer.q35.emit import Layout, allocate, check_bf16, embedding_rows, emit_layers
from chipformer.q35 import tiny
from chipformer.q35.ir import Target
from chipformer.q35.mute import mute

TEXT = "model.language_model."
CHIP_VOCAB = 244


def widen_inplace(cfg, sd, hidden):
    """Identical math to chipformer.q35.merge.widen(..., n_chip_layers=0),
    but per-key in place so each old tensor is freed on assignment.
    Config edits copied verbatim from M.widen."""
    tc = cfg["text_config"]
    d = tc["hidden_size"]
    assert hidden >= d
    scale = (d / hidden) ** 0.5
    for k in list(sd.keys()):
        w = sd[k]
        if M.IN_COLS.search(k):
            sd[k] = torch.cat([w, w.new_zeros(*w.shape[:-1], hidden - d)], -1)
        elif M.OUT_ROWS.search(k):
            sd[k] = torch.cat([w, w.new_zeros(hidden - d, *w.shape[1:])], 0)
        elif M.WIDE_NORMS.search(k):
            g = (1.0 + w.float()) * scale - 1.0
            sd[k] = torch.cat([g.to(w.dtype), w.new_zeros(hidden - d)])
        # else: key unchanged, no copy
    cfg = copy.deepcopy(cfg)
    tc = cfg["text_config"]
    tc["hidden_size"] = hidden
    cfg["vision_config"]["out_hidden_size"] = hidden
    tc["mtp_num_hidden_layers"] = 0
    tc.pop("full_attention_interval", None)
    return cfg, sd


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    cfg, sd = M.load(src)
    tc = cfg["text_config"]
    d0, v0, inter = tc["hidden_size"], tc["vocab_size"], tc["intermediate_size"]
    n0 = tc["num_hidden_layers"]
    print(f"base: hidden {d0}, vocab {v0}, layers {n0}, inter {inter}, "
          f"layer_types {tc['layer_types']}", flush=True)
    print("base text_config:", json.dumps(tc, sort_keys=True), flush=True)
    print("base vision_config:", json.dumps(cfg.get("vision_config", {}), sort_keys=True), flush=True)
    from circuit import build_vec4

    c = build_vec4(offset=v0)
    code = [next(iter(e.terms)) for e in c.code]
    tau = next(iter(c.tau.terms))
    mu = next(iter(c.mu.terms))
    keep = set(code) | {tau, mu}
    alloc = allocate(c, None, keep, inter)
    hidden, m = M.pick_hidden(d0, 1 + alloc.n_slots)
    print(f"circuit: {c.n_layers()} layers, {alloc.n_slots} chip slots, "
          f"hidden {hidden}, ballast m {m}", flush=True)
    per_dim = (2 * tc["linear_num_key_heads"] * tc["linear_key_head_dim"]
               + 3 * tc["linear_num_value_heads"] * tc["linear_value_head_dim"]
               + 3 * inter)
    vocab = -(-(v0 + CHIP_VOCAB) // 64) * 64
    size_params = hidden * ((n0 + c.n_layers()) * per_dim + vocab)
    # vision tower params are unchanged by widening except merger rows/cols
    print(f"estimate: text+embed params ~{size_params} "
          f"({size_params * 2 / 1e9:.2f} GB bf16, before vision tower)", flush=True)

    lay = Layout(hidden=hidden, off=d0, m=m, slots=alloc.slots)
    assert d0 + m + 1 + alloc.n_slots <= hidden, (alloc.n_slots, hidden)
    lay.scale  # asserts m * B^2 / hidden is a power of 4
    print(f"slots: mu={alloc.slots.get(mu)} tau={alloc.slots.get(tau)} "
          f"code={[alloc.slots.get(d) for d in code]} "
          f"mu_dim={lay.dim(mu)} tau_dim={lay.dim(tau)} "
          f"chip_region_used={m + 1 + alloc.n_slots} chip_width={hidden - d0}", flush=True)

    n_chip = c.n_layers()
    cfg, sd = widen_inplace(cfg, sd, hidden)
    tc = cfg["text_config"]
    shift = re.compile(r"^(model\.language_model\.layers\.)(\d+)\.")
    sd = {shift.sub(lambda mt: f"{mt[1]}{int(mt[2]) + n_chip}.", k): v for k, v in sd.items()}
    tc["layer_types"] = ["linear_attention"] * n_chip + tc["layer_types"]
    tc["num_hidden_layers"] = n_chip + n0
    tc["vocab_size"] = vocab

    # Mute Qwen's layers on chip tokens (mu slot, normed ballast sqrt(hidden)
    # because after chip layers mu=1 is the only non-tiny dim at chip positions).
    mute(sd, tc, n0, lay.dim(mu), hidden ** 0.5, sinks=None, first=n_chip)
    print("muted Qwen layers on mu slot", lay.dim(mu), flush=True)

    layers = emit_layers(c, lay, tc, 0, TEXT, alloc, keep)
    sd.update({k: v.to(torch.bfloat16) for k, v in layers.items()})
    del layers

    # Embeddings: Qwen rows keep Qwen part, get text-token chip features;
    # chip rows are the 244 VEC4 tokens appended after v0.
    text_row = embedding_rows(c, lay, [0], hidden).to(torch.bfloat16)[0]
    full = torch.zeros(vocab, hidden, dtype=torch.bfloat16)
    full[:v0, :d0] = sd[TEXT + "embed_tokens.weight"][:, :d0]
    full[:v0, d0:] = text_row[d0:]
    chip = embedding_rows(c, lay, list(range(v0, v0 + CHIP_VOCAB)), hidden)
    full[v0:v0 + CHIP_VOCAB] = chip.to(torch.bfloat16)
    sd[TEXT + "embed_tokens.weight"] = full
    bias = "model.visual.merger.linear_fc2.bias"
    if bias in sd:
        sd[bias] = sd[bias].clone()
        sd[bias][d0:] = text_row[d0:].to(sd[bias].dtype)

    norm = sd[TEXT + "norm.weight"].float()
    norm[d0:] = -1.0  # chip dims invisible to lm_head ...
    for d in code:  # ... except the output code and tau
        norm[lay.dim(d)] = tiny.code_gain(lay)
    norm[lay.dim(tau)] = M.TAU_GAIN
    check_bf16("final norm", norm[d0:])
    sd[TEXT + "norm.weight"] = norm.to(torch.bfloat16)

    dst.mkdir(parents=True, exist_ok=True)
    shard, size, shards = {}, 0, []
    for k in sorted(sd):
        shard[k] = sd[k].contiguous()
        size += sd[k].numel() * 2
        if size > 2**31:
            shards.append(shard)
            shard, size = {}, 0
    shards.append(shard)
    index = {"metadata": {"total_size": sum(t.numel() * 2 for t in sd.values())},
             "weight_map": {}}
    for i, sh in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(sh, str(dst / name), metadata={"format": "pt"})
        index["weight_map"].update({k: name for k in sh})
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    for p in src.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors") and p.name not in (
                "config.json", "model.safetensors.index.json"):
            import shutil
            shutil.copy(p, dst / p.name)
    meta = {"layout": "vec4", "chip_token_offset": v0, "chip_vocab": CHIP_VOCAB,
            "chip_layers": n_chip, "qwen_layers_from": n_chip, "hidden": hidden,
            "chip_dims_offset": d0, "ballast_dims": m, "circuit": c.stats(),
            "slots": alloc.n_slots, "muted": True, "anti_sink": False,
            "note": "anti-sink skipped: test sequences are chip-only or text-only, "
                    "and Qwen full-attention output is muted on chip tokens"}
    (dst / "chipformer.json").write_text(json.dumps(meta, indent=2))
    total = sum(t.numel() * 2 for t in sd.values())
    print(f"merged: {dst} hidden {hidden} layers {n_chip + n0} vocab {vocab} "
          f"total {total / 1e9:.2f} GB, {sum(t.numel() for t in sd.values())} params",
          flush=True)


if __name__ == "__main__":
    main()
