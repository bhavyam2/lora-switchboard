import torch
import torch.nn as nn
import pytest

from engine.core.batch_context import BatchContext
from engine.core.lora_layer import LoRALinear


def make_lora_layer(in_f=32, out_f=64, rank=4):
    base = nn.Linear(in_f, out_f, bias=False)
    for p in base.parameters():
        p.requires_grad = False
    return LoRALinear(base, rank=rank)


# ── BatchContext ──────────────────────────────────────────────────────────────

def test_batch_context_set_and_clear():
    positions = {"adapter-a": [0, 2], "adapter-b": [1]}
    BatchContext.set(positions)
    assert BatchContext.get() == positions
    BatchContext.clear()
    assert BatchContext.get() is None


# ── LoRALinear heterogeneous forward ─────────────────────────────────────────

def test_hetero_forward_applies_correct_deltas():
    layer = make_lora_layer(in_f=32, out_f=64, rank=4)

    A_a = torch.randn(4, 32) * 0.1
    B_a = torch.randn(64, 4) * 0.1
    A_b = torch.randn(4, 32) * 0.1
    B_b = torch.randn(64, 4) * 0.1

    layer.load_batch_adapters({"adapter-a": (A_a, B_a), "adapter-b": (A_b, B_b)})
    BatchContext.set({"adapter-a": [0, 2], "adapter-b": [1]})

    try:
        x = torch.randn(3, 32)
        out = layer(x)

        base_out = layer.base(x)

        expected = base_out.clone()
        expected[[0, 2]] += (x[[0, 2]] @ A_a.T) @ B_a.T
        expected[[1]]    += (x[[1]]    @ A_b.T) @ B_b.T

        assert torch.allclose(out, expected, atol=1e-5)
    finally:
        BatchContext.clear()
        layer.clear_batch_adapters()


def test_hetero_forward_clears_to_base():
    layer = make_lora_layer()
    A = torch.randn(4, 32) * 0.01
    B = torch.randn(64, 4) * 0.01
    layer.load_batch_adapters({"a": (A, B)})

    # No BatchContext set → falls through to single-adapter path → no _lora_A → base
    x = torch.randn(2, 32)
    out = layer(x)
    assert torch.allclose(out, layer.base(x))
    layer.clear_batch_adapters()


def test_single_adapter_path_unaffected():
    layer = make_lora_layer()
    A = torch.randn(4, 32) * 0.01
    B = torch.randn(64, 4) * 0.01
    layer.load_adapter(A, B)

    x = torch.randn(2, 32)
    expected = layer.base(x) + (x @ A.T) @ B.T
    assert torch.allclose(layer(x), expected, atol=1e-6)
    layer.unload_adapter()
