"""Low-memory merge: VEC4 v3 circuit into a real Qwen3.5 model (0.8B or 2B).

Why this exists: the stock merge (merge_v3_2b.py) holds the whole merged
state dict in RAM -- widened base (bf16) + ALL chip layers in fp32 +
their bf16 conversion copies. For 2B-v3 that peaks around 17 GB
(widened base ~5.7 GB + fp32 chip layers 26 x ~294 MB = ~7.6 GB + bf16
copies ~3.8 GB), which is why the 2026-11-05 2B-v3 gate died in the
VM's memory cgroup at 11.7 GB RSS during emit (chipformer.json never
created, no Gate 0). The 2B-v2 merge survived only because its circuit
is smaller (186 slots, chip width 256 vs v3's 364 / 512).

What changes here -- memory flow ONLY, not math:
1. After widen + shift + mute, every Qwen/vision tensor is written to a
   shard file immediately and dropped from RAM (only the embed and the
   final norm are kept, because they are edited later).
2. Chip layers are emitted one layer at a time through emit_layers_sink
   (below), converted to bf16 per tensor, appended to the current shard,
   and freed. No full fp32 chip dict ever exists.
3. Shards are capped at 512 MB, so the live buffer is bounded.
4. The final embed is built (1.27 GB for 2B, transient x2 with the old
   embed), sharded, and dropped; then index + config + chipformer.json.

Bounded peak for 2B-v3: widen phase ~6 GB (base 4.5 GB + largest-key
transient), emit phase ~3 GB (embed 1.27 + shard buffer <= 0.5 + one
fp32 layer 0.29 + torch/python overhead), final embed ~4.3 GB. The
driver prints VmRSS at each phase boundary and ru_maxrss at the end,
so the gate log carries a measured peak, not an estimate.

Output format is IDENTICAL to the stock merge: sharded safetensors +
model.safetensors.index.json + config.json + tokenizer files +
chipformer.json, so transformers loads it the same way and the
run_merged / tool / cache drivers need no changes.

emit_layers_sink is a mechanical copy of chipformer.q35.emit.emit_layers
(upstream, pinned clone in ../upstream): the ONLY differences are the
`sink` callback parameter, `out = {}` removed, `out[p + k] = t`
replaced by `sink(p + k, t)`, and `return out` removed. Every
computation, every check_bf16 call, and every tensor is identical --
verified locally by test_emit_sink.py, which compares stock vs sink
layer tensors exactly on the first layers of the real v3 circuit.
The full Colab Gate 0 (720/720) is the whole-model equivalence proof:
any math divergence would fail it.

Usage: python3 merge_v3_lowmem.py SRC_DIR OUT_DIR
"""

import copy
import gc
import json
import re
import resource
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "upstream"))
sys.path.insert(0, str(HERE))

import torch
from safetensors.torch import save_file

from chipformer.q35 import merge as M
from chipformer.q35 import tiny
from chipformer.q35.emit import Alloc, Layout, allocate, check_bf16, embedding_rows
from chipformer.q35.emit import row  # noqa: F401  (used by emit_layers_sink)
from chipformer.q35.ir import FFN, HD, Graph, Head, K, ONE, Expr  # noqa: F401  (used by emit_layers_sink)
from chipformer.q35.ir import Target
from chipformer.q35.mute import mute

TEXT = "model.language_model."
CHIP_VOCAB = 252
SHARD_BYTES = 512 * 1024 * 1024


def rss_mb():
    """Current process RSS in MiB (Linux /proc), or -1 if unavailable."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    except Exception:
        pass
    return -1


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


def emit_layers_sink(g: Graph, lay: Layout, tc: dict, first: int, prefix: str,
                     alloc: Alloc | None = None, keep: set[int] | None = None,
                     chip: Expr | None = None, sink=None):
    """Mechanical copy of emit_layers: tensors go to sink(key, fp32_tensor)
    instead of accumulating in a dict. All computation identical."""
    h, nk, nv = tc["hidden_size"], tc["linear_num_key_heads"], tc["linear_num_value_heads"]
    inter = tc["intermediate_size"]
    # With more value heads than key heads (4B: 32 vs 16), value head i reads key head
    # i // r; a circuit head uses the first value head of its group and leaves the rest at zero.
    r = nv // nk
    assert r * nk == nv
    f32 = torch.float32
    exit_level = None
    if keep is not None:
        exit_level = max(n.level for n in g.nodes if n.name.startswith("exit."))
        assert all(n.level < exit_level for n in g.nodes if not n.name.startswith("exit."))
        live = (alloc.final + alloc.final_over) if alloc else list(range(1, len(g.dims)))
        exit_clear = [d for d in live if d not in keep]
    over = alloc.over if alloc else {}
    assert chip is not None or not over, "an overlay needs the chip-token gate"
    one = Expr({ONE: 1})
    for L in range(g.n_layers()):
        p = f"{prefix}layers.{first + L}."
        qkv = torch.zeros(2 * nk * HD + nv * HD, h, dtype=f32)
        convw = torch.zeros(2 * nk * HD + nv * HD, 1, tc["linear_conv_kernel_dim"], dtype=f32)
        zw = torch.zeros(nv * HD, h, dtype=f32)
        bw = torch.zeros(nv, h, dtype=f32)
        aw = torch.zeros(nv, h, dtype=f32)
        ow = torch.zeros(h, nv * HD, dtype=f32)
        gate = torch.zeros(inter, h, dtype=f32)
        up = torch.zeros(inter, h, dtype=f32)
        down = torch.zeros(h, inter, dtype=f32)
        heads = [n for n in g.nodes if isinstance(n, Head) and n.level == 2 * L + 1]
        for j, n in enumerate(heads):
            vhs = n.vheads()
            assert len(vhs) <= r, (n.name, len(vhs), r)
            for base, chans in ((j * HD, n.q), (nk * HD + j * HD, n.k)):
                for i, ch in chans.items():
                    row(ch.expr, lay, qkv[base + i])
                    for lag, w in enumerate(ch.taps):
                        convw[base + i, 0, -1 - lag] = w
            for x, vhd in enumerate(vhs):
                vh = j * r + x  # value head x of this key head
                base = 2 * nk * HD + vh * HD
                for i, ch in vhd.v.items():
                    row(ch.expr, lay, qkv[base + i])
                    for lag, w in enumerate(ch.taps):
                        convw[base + i, 0, -1 - lag] = w
                # beta = sigmoid(128) = 1, or a write gate: sigmoid(256 g - 128), 1 or exactly 0
                beta = Expr({ONE: 128}) if vhd.beta is None else 256 * vhd.beta - 128 * one
                row(beta, lay, bw[vh])
                row(vhd.a, lay, aw[vh])
                for col, e in vhd.z.items():
                    row(e, lay, zw[vh * HD + col])
                for col, d, w in vhd.out:
                    ow[lay.dim(d), vh * HD + col] += w
        one = Expr({ONE: 1})

        def clear_gate(d):
            return chip if d in over else one

        units = []  # (coef, a, b, output dim)
        for n in g.nodes:
            if not (isinstance(n, FFN) and n.level == 2 * L + 2):
                continue
            units += [(coef, a, b, n.out) for coef, a, b in n.terms]
            if alloc and n.out in alloc.subtract:  # hand the slot over: remove the old value
                old = alloc.subtract[n.out]
                units.append((-1, Expr({old: 1}), clear_gate(old), n.out))
        units += [(-1, Expr({o: 1}), clear_gate(o), o)
                  for o in (alloc.clears.get(2 * L + 2, []) if alloc else [])]
        if exit_level == 2 * L + 2:
            units += [(-1, Expr({o: 1}), clear_gate(o), o) for o in exit_clear]
        assert len(units) <= inter, (L, len(units), inter)
        for u, (coef, a, b, d) in enumerate(units):
            row(b, lay, gate[u], K)
            row(a, lay, up[u], 1 / K)
            down[lay.dim(d), u] = float(coef)
        if exit_level == 2 * L + 2:  # the ballast goes too: -B per ballast dim
            for i in range(lay.m):
                u = len(units) + i
                row(Expr({ONE: K}), lay, gate[u])
                up[u, lay.off + i] = (lay.ballast / K) / (lay.ballast * lay.scale)
                down[lay.off + i, u] = -1.0
        layer = {
            "input_layernorm.weight": torch.zeros(h, dtype=f32),
            "post_attention_layernorm.weight": torch.zeros(h, dtype=f32),
            "linear_attn.in_proj_qkv.weight": qkv,
            "linear_attn.in_proj_z.weight": zw,
            "linear_attn.in_proj_a.weight": aw,
            "linear_attn.in_proj_b.weight": bw,
            "linear_attn.conv1d.weight": convw,
            "linear_attn.dt_bias": torch.full((nv,), -110.0, dtype=f32),
            "linear_attn.A_log": torch.zeros(nv, dtype=f32),
            "linear_attn.norm.weight": torch.ones(HD, dtype=f32),
            "linear_attn.out_proj.weight": ow,
            "mlp.gate_proj.weight": gate,
            "mlp.up_proj.weight": up,
            "mlp.down_proj.weight": down,
        }
        for k, t in layer.items():
            check_bf16(p + k, t)
            sink(p + k, t)


class ShardWriter:
    """Incremental sharded-safetensors writer.

    Tensors are appended to the current shard and flushed to disk at
    SHARD_BYTES, so live memory never exceeds one shard buffer. Shard
    files are first written under temp names and renamed at finish()
    to the conventional model-XXXXX-of-YYYYY.safetensors form, since
    the total count is only known at the end. weight_map is built from
    the FINAL names, which is all transformers reads.
    """

    def __init__(self, dst, max_bytes=SHARD_BYTES):
        self.dst = Path(dst)
        self.dst.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.shard = {}
        self.size = 0
        self.total = 0
        self.parts = []  # (temp_name, [keys])

    def add(self, k, t):
        t = t.contiguous()
        self.shard[k] = t
        nbytes = t.numel() * t.element_size()
        self.size += nbytes
        self.total += nbytes
        if self.size >= self.max_bytes:
            self.flush()

    def flush(self):
        if not self.shard:
            return
        tmp = f"shard-{len(self.parts) + 1:05d}.tmp.safetensors"
        save_file(self.shard, str(self.dst / tmp), metadata={"format": "pt"})
        self.parts.append((tmp, list(self.shard.keys())))
        self.shard = {}
        self.size = 0

    def finish(self):
        self.flush()
        weight_map = {}
        for i, (tmp, keys) in enumerate(self.parts):
            final = f"model-{i + 1:05d}-of-{len(self.parts):05d}.safetensors"
            (self.dst / tmp).rename(self.dst / final)
            weight_map.update({k: final for k in keys})
        index = {"metadata": {"total_size": self.total}, "weight_map": weight_map}
        (self.dst / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2))
        return index


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    cfg, sd = M.load(src)
    tc = cfg["text_config"]
    d0, v0, inter = tc["hidden_size"], tc["vocab_size"], tc["intermediate_size"]
    n0 = tc["num_hidden_layers"]
    print(f"base: hidden {d0}, vocab {v0}, layers {n0}, inter {inter}, "
          f"layer_types {tc['layer_types']}", flush=True)
    print(f"rss after load: {rss_mb()} MiB", flush=True)
    from circuit import build_vec4v3 as build_vec4

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
    print(f"estimate: text+embed params ~{size_params} "
          f"({size_params * 2 / 1e9:.2f} GB bf16, before vision tower)",
          flush=True)

    lay = Layout(hidden=hidden, off=d0, m=m, slots=alloc.slots)
    assert d0 + m + 1 + alloc.n_slots <= hidden, (alloc.n_slots, hidden)
    lay.scale  # asserts m * B^2 / hidden is a power of 4
    print(f"slots: mu={alloc.slots.get(mu)} tau={alloc.slots.get(tau)} "
          f"code={[alloc.slots.get(d) for d in code]} "
          f"mu_dim={lay.dim(mu)} tau_dim={lay.dim(tau)} "
          f"chip_region_used={m + 1 + alloc.n_slots} chip_width={hidden - d0}",
          flush=True)

    n_chip = c.n_layers()
    cfg, sd = widen_inplace(cfg, sd, hidden)
    tc = cfg["text_config"]
    shift = re.compile(r"^(model\.language_model\.layers\.)(\d+)\.")
    sd = {shift.sub(lambda mt: f"{mt[1]}{int(mt[2]) + n_chip}.", k): v
          for k, v in sd.items()}
    tc["layer_types"] = ["linear_attention"] * n_chip + tc["layer_types"]
    tc["num_hidden_layers"] = n_chip + n0
    tc["vocab_size"] = vocab
    print(f"rss after widen+shift: {rss_mb()} MiB", flush=True)

    # Mute Qwen's layers on chip tokens (mu slot, normed ballast sqrt(hidden)
    # because after chip layers mu=1 is the only non-tiny dim at chip positions).
    mute(sd, tc, n0, lay.dim(mu), hidden ** 0.5, sinks=None, first=n_chip)
    print("muted Qwen layers on mu slot", lay.dim(mu), flush=True)

    # Phase 1: shard out everything except the tensors edited later
    # (embed, final norm, merger bias). Each key is dropped from RAM
    # as soon as it lands in the shard buffer.
    writer = ShardWriter(dst)
    keep_keys = {TEXT + "embed_tokens.weight", TEXT + "norm.weight"}
    bias = "model.visual.merger.linear_fc2.bias"
    if bias in sd:
        keep_keys.add(bias)
    for k in sorted(list(sd.keys())):
        if k in keep_keys:
            continue
        writer.add(k, sd[k])
        del sd[k]
    del k
    gc.collect()
    print(f"rss after shard-out (kept {sorted(keep_keys)}): {rss_mb()} MiB",
          flush=True)

    # Phase 2: emit chip layers one at a time; each tensor is converted
    # to bf16 and appended to the shard buffer, then freed. The fp32
    # layer tensors live only until the next layer rebinds them.
    def sink(k, t):
        writer.add(k, t.to(torch.bfloat16).contiguous())

    emit_layers_sink(c, lay, tc, 0, TEXT, alloc, keep, sink=sink)
    gc.collect()
    print(f"rss after chip emit: {rss_mb()} MiB", flush=True)

    # Phase 3: embeddings -- same math as the stock merge. The new full
    # tensor (vocab x hidden bf16) coexists briefly with the old embed.
    text_row = embedding_rows(c, lay, [0], hidden).to(torch.bfloat16)[0]
    full = torch.zeros(vocab, hidden, dtype=torch.bfloat16)
    full[:v0, :d0] = sd[TEXT + "embed_tokens.weight"][:, :d0]
    full[:v0, d0:] = text_row[d0:]
    chip = embedding_rows(c, lay, list(range(v0, v0 + CHIP_VOCAB)), hidden)
    full[v0:v0 + CHIP_VOCAB] = chip.to(torch.bfloat16)
    del sd[TEXT + "embed_tokens.weight"]
    writer.add(TEXT + "embed_tokens.weight", full)
    del full, chip, text_row
    if bias in sd:
        b = sd[bias].clone()
        # text_row is freed above; rebuild the single row needed.
        row0 = embedding_rows(c, lay, [0], hidden).to(torch.bfloat16)[0]
        b[d0:] = row0[d0:].to(b.dtype)
        writer.add(bias, b)
        del sd[bias], b, row0
    gc.collect()
    print(f"rss after embed: {rss_mb()} MiB", flush=True)

    # Phase 4: final norm -- same math as the stock merge.
    norm = sd[TEXT + "norm.weight"].float()
    norm[d0:] = -1.0  # chip dims invisible to lm_head ...
    for d in code:  # ... except the output code and tau
        norm[lay.dim(d)] = tiny.code_gain(lay)
    norm[lay.dim(tau)] = M.TAU_GAIN
    check_bf16("final norm", norm[d0:])
    writer.add(TEXT + "norm.weight", norm.to(torch.bfloat16))
    del sd[TEXT + "norm.weight"], norm
    gc.collect()

    index = writer.finish()
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    for p in src.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors") and p.name not in (
                "config.json", "model.safetensors.index.json"):
            import shutil
            shutil.copy(p, dst / p.name)
    meta = {"layout": "vec4-v3", "chip_token_offset": v0, "chip_vocab": CHIP_VOCAB,
            "chip_layers": n_chip, "qwen_layers_from": n_chip, "hidden": hidden,
            "chip_dims_offset": d0, "ballast_dims": m, "circuit": c.stats(),
            "slots": alloc.n_slots, "muted": True, "anti_sink": False,
            "lowmem": True, "shard_bytes": SHARD_BYTES,
            "note": "anti-sink skipped: test sequences are chip-only or text-only, "
                    "and Qwen full-attention output is muted on chip tokens"}
    (dst / "chipformer.json").write_text(json.dumps(meta, indent=2))
    total = index["metadata"]["total_size"]
    print(f"merged: {dst} hidden {hidden} layers {n_chip + n0} vocab {vocab} "
          f"total {total / 1e9:.2f} GB, shards {len(writer.parts)}, "
          f"keys {len(index['weight_map'])}", flush=True)
    print(f"peak rss: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MiB",
          flush=True)


if __name__ == "__main__":
    main()
