"""Local tests for merge_v3_lowmem.py (CPU, no base weights needed).

Test A -- emit equivalence: stock chipformer.q35.emit.emit_layers vs the
lowmem driver's emit_layers_sink, on the FIRST K layers of the real v3
circuit (K small so it fits this VM). A proxy circuit reports n_layers()
= K while delegating everything else; emit is per-layer independent, so
layers 0..K-1 are identical to the full run's first K layers. All 14
keys per layer must match exactly (torch.equal on the fp32 tensors the
sink receives, before bf16 conversion).

Test B -- ShardWriter: synthetic tensors (incl. a non-contiguous view)
with a tiny shard cap, forcing several flushes. Verifies: index covers
every key exactly once, shard files reload with exact values/dtypes,
temp files are gone, final names follow model-XXXXX-of-YYYYY, and
total_size matches the byte sum.

What these do NOT prove (stated honestly): the full-merge flow against
a real 2B base -- this VM has no GPU, no torch-native base weights, and
only 7 GB RAM. That proof is the Colab Gate 0 (720/720) with the lowmem
merge: any math divergence from the stock merge fails it.

Usage: python3 test_lowmem.py
"""

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "upstream"))
sys.path.insert(0, str(HERE))

import torch
from safetensors.torch import load_file

from chipformer.q35 import merge as M
from chipformer.q35.emit import Layout, allocate, emit_layers
from circuit import build_vec4v3
from merge_v3_lowmem import ShardWriter, emit_layers_sink

V0 = 248320  # Qwen3.5-2B vocab; circuit offset only, no weights involved
TC = {"hidden_size": 2560, "intermediate_size": 6144,
      "linear_num_key_heads": 16, "linear_num_value_heads": 16,
      "linear_key_head_dim": 128, "linear_value_head_dim": 128,
      "linear_conv_kernel_dim": 4}


class ProxyCircuit:
    """Delegates everything to the real circuit except n_layers()."""

    def __init__(self, real, k):
        self._real = real
        self._k = k

    def __getattr__(self, name):
        return getattr(self._real, name)

    def n_layers(self):
        return self._k


def build_lay():
    c = build_vec4v3(offset=V0)
    code = [next(iter(e.terms)) for e in c.code]
    tau = next(iter(c.tau.terms))
    mu = next(iter(c.mu.terms))
    keep = set(code) | {tau, mu}
    alloc = allocate(c, None, keep, 6144)
    hidden, m = M.pick_hidden(2048, 1 + alloc.n_slots)
    assert hidden == 2560, hidden
    lay = Layout(hidden=hidden, off=2048, m=m, slots=alloc.slots)
    lay.scale
    return c, lay, alloc, keep


def test_emit_equivalence(k=2):
    c, lay, alloc, keep = build_lay()
    proxy = ProxyCircuit(c, k)
    stock = emit_layers(proxy, lay, TC, 0, "model.language_model.", alloc, keep)
    got = {}
    emit_layers_sink(proxy, lay, TC, 0, "model.language_model.", alloc, keep,
                     sink=lambda kk, t: got.__setitem__(kk, t))
    assert set(stock) == set(got), (
        f"key mismatch: only-stock={sorted(set(stock) - set(got))[:5]} "
        f"only-sink={sorted(set(got) - set(stock))[:5]}")
    bad = [kk for kk in stock if not torch.equal(stock[kk], got[kk])]
    assert not bad, f"tensor mismatch on {bad[:5]}"
    print(f"emit equivalence: {len(stock)} tensors over {k} layers, "
          f"all torch.equal, fp32 sizes match")
    del stock, got


def test_shard_writer():
    with tempfile.TemporaryDirectory() as tmp:
        w = ShardWriter(tmp, max_bytes=1024)
        tensors = {
            "a.weight": torch.arange(600, dtype=torch.bfloat16).reshape(20, 30),
            "b.weight": torch.randn(40, 40),  # fp32 in, stored as given
            "c.view": torch.arange(200, dtype=torch.bfloat16).reshape(10, 20).t(),
            "d.small": torch.ones(3, dtype=torch.bfloat16),
        }
        for kk, t in tensors.items():
            w.add(kk, t)
        index = w.finish()
        assert set(index["weight_map"]) == set(tensors), "weight_map keys"
        assert len(set(index["weight_map"].values())) == len(w.parts)
        assert not list(Path(tmp).glob("*.tmp.safetensors")), "temp files left"
        total = 0
        for name in sorted(set(index["weight_map"].values())):
            assert name.startswith("model-") and "-of-" in name, name
            sd = load_file(str(Path(tmp) / name))
            for kk, t in sd.items():
                assert torch.equal(t, tensors[kk].contiguous()), kk
                total += t.numel() * t.element_size()
        assert index["metadata"]["total_size"] == total, "total_size"
        print(f"shard writer: {len(tensors)} keys over {len(w.parts)} shards, "
              f"reload exact, total {total} bytes")


if __name__ == "__main__":
    test_emit_equivalence()
    test_shard_writer()
    print("ALL LOWMEM LOCAL TESTS PASS")
