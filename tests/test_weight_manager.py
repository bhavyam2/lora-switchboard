import torch
import torch.nn as nn
import pytest
from unittest.mock import MagicMock

from engine.core.lora_layer import LoRALinear
from engine.core.weight_manager import WeightManager


def make_lora_layers(n=2, in_f=32, out_f=64, rank=4):
    layers = {}
    for i in range(n):
        base = nn.Linear(in_f, out_f, bias=False)
        layers[f"layer.{i}.linear"] = LoRALinear(base, rank=rank)
    return layers


def test_register_and_activate_random(monkeypatch):
    monkeypatch.setattr("engine.core.weight_manager.settings.adapter_cache_max", 4)
    monkeypatch.setattr("engine.core.weight_manager.settings.lora_rank", 4)

    layers = make_lora_layers()
    wm = WeightManager(lora_layers=layers, device="cpu")
    wm.register_random_adapter("test-adapter")
    wm.activate("test-adapter")

    for layer in layers.values():
        assert layer._lora_A is not None
        assert layer._lora_B is not None


def test_lru_eviction(monkeypatch):
    monkeypatch.setattr("engine.core.weight_manager.settings.adapter_cache_max", 2)
    monkeypatch.setattr("engine.core.weight_manager.settings.lora_rank", 4)

    layers = make_lora_layers()
    wm = WeightManager(lora_layers=layers, device="cpu")

    wm.register_random_adapter("a1")
    wm.register_random_adapter("a2")
    wm.register_random_adapter("a3")  # should evict a1

    assert "a1" not in wm.cached_ids
    assert "a2" in wm.cached_ids
    assert "a3" in wm.cached_ids


def test_deactivate_clears_layers(monkeypatch):
    monkeypatch.setattr("engine.core.weight_manager.settings.adapter_cache_max", 4)
    monkeypatch.setattr("engine.core.weight_manager.settings.lora_rank", 4)

    layers = make_lora_layers()
    wm = WeightManager(lora_layers=layers, device="cpu")
    wm.register_random_adapter("x")
    wm.activate("x")
    wm.deactivate()

    for layer in layers.values():
        assert layer._lora_A is None
