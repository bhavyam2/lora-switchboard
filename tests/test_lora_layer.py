import torch
import torch.nn as nn
import pytest
from engine.core.lora_layer import LoRALinear


@pytest.fixture
def base_linear():
    lin = nn.Linear(64, 128, bias=False)
    for p in lin.parameters():
        p.requires_grad = False
    return lin


def test_passthrough_without_adapter(base_linear):
    layer = LoRALinear(base_linear, rank=4)
    x = torch.randn(2, 64)
    expected = base_linear(x)
    out = layer(x)
    assert torch.allclose(out, expected)


def test_lora_delta_applied(base_linear):
    layer = LoRALinear(base_linear, rank=4)
    A = torch.randn(4, 64) * 0.01
    B = torch.randn(128, 4) * 0.01
    layer.load_adapter(A, B)

    x = torch.randn(2, 64)
    out = layer(x)
    expected = base_linear(x) + (x @ A.T) @ B.T
    assert torch.allclose(out, expected, atol=1e-6)


def test_unload_restores_base(base_linear):
    layer = LoRALinear(base_linear, rank=4)
    A = torch.randn(4, 64)
    B = torch.randn(128, 4)
    layer.load_adapter(A, B)
    layer.unload_adapter()

    x = torch.randn(2, 64)
    assert torch.allclose(layer(x), base_linear(x))
