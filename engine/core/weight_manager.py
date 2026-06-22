import torch
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from engine.config import settings
from engine.core.lora_layer import LoRALinear


AdapterWeights = dict[str, tuple[torch.Tensor, torch.Tensor]]  # layer_path -> (A, B)


class WeightManager:
    """
    LRU cache that keeps up to `max_cached` adapters hot on the GPU.
    On a cache miss it loads weights from disk (CPU) and transfers H2D,
    evicting the least-recently-used adapter if the cache is full.
    """

    def __init__(self, lora_layers: dict[str, LoRALinear], device: str):
        self.lora_layers = lora_layers
        self.device = device
        self.max_cached = settings.adapter_cache_max
        self.rank = settings.lora_rank
        self._cache: OrderedDict[str, AdapterWeights] = OrderedDict()
        self._active_adapter: Optional[str] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def activate(self, adapter_id: str) -> None:
        """Swap the given adapter into all LoRA layers."""
        if self._active_adapter == adapter_id:
            return

        weights = self._get_or_load(adapter_id)
        for path, layer in self.lora_layers.items():
            if path in weights:
                A, B = weights[path]
                layer.load_adapter(A, B)
            else:
                layer.unload_adapter()

        self._active_adapter = adapter_id

    def deactivate(self) -> None:
        for layer in self.lora_layers.values():
            layer.unload_adapter()
        self._active_adapter = None

    def register(self, adapter_id: str, weights: AdapterWeights) -> None:
        """Register pre-parsed adapter weights (e.g. loaded by AdapterLoader)."""
        self._store(adapter_id, weights)

    def register_random_adapter(self, adapter_id: str) -> None:
        """
        Creates and registers a randomly-initialised adapter — used for
        testing and benchmarking without real checkpoint files.
        """
        weights: AdapterWeights = {}
        for path, layer in self.lora_layers.items():
            dtype = layer.base.weight.dtype
            A = torch.randn(self.rank, layer.in_features, device="cpu", dtype=dtype) * 0.01
            B = torch.zeros(layer.out_features, self.rank, device="cpu", dtype=dtype)
            weights[path] = (A, B)
        self._store(adapter_id, weights)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_load(self, adapter_id: str) -> AdapterWeights:
        if adapter_id in self._cache:
            self._cache.move_to_end(adapter_id)
            return self._cache[adapter_id]

        weights = self._load_from_disk(adapter_id)
        self._store(adapter_id, weights)
        return weights

    def _store(self, adapter_id: str, weights: AdapterWeights) -> None:
        if adapter_id in self._cache:
            self._cache.move_to_end(adapter_id)
            return

        if len(self._cache) >= self.max_cached:
            evicted_id, _ = self._cache.popitem(last=False)
            print(f"[WeightManager] Evicted adapter '{evicted_id}' from GPU cache.")

        gpu_weights = {
            path: (
                A.to(self.device, dtype=self.lora_layers[path].base.weight.dtype),
                B.to(self.device, dtype=self.lora_layers[path].base.weight.dtype),
            )
            for path, (A, B) in weights.items()
            if path in self.lora_layers
        }
        self._cache[adapter_id] = gpu_weights
        print(f"[WeightManager] Adapter '{adapter_id}' loaded onto {self.device}.")

    def _load_from_disk(self, adapter_id: str) -> AdapterWeights:
        adapter_path = Path("data/adapters") / adapter_id / "weights.pt"
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"Adapter '{adapter_id}' not found at {adapter_path}. "
                "Use register_random_adapter() for synthetic adapters."
            )
        return torch.load(adapter_path, map_location="cpu")

    @property
    def cached_ids(self) -> list[str]:
        return list(self._cache.keys())
