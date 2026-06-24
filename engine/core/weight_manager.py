import torch
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from engine.config import settings
from engine.core.lora_layer import LoRALinear


AdapterWeights = dict[str, tuple[torch.Tensor, torch.Tensor]]  # layer_path -> (A, B)

AdapterMetadata = dict  # optional: description, system_prompt, etc.


class WeightManager:
    """
    Two-tier memory manager for LoRA adapters.

    Tier 1 — GPU cache (hot):  up to `max_cached` adapters in VRAM, LRU eviction.
    Tier 2 — CPU registry:     all known adapters in host RAM, never evicted.

    On activate():
      - GPU cache hit  → inject weights directly (pointer swap, ~0 ms)
      - GPU cache miss → H2D transfer from CPU registry, evict LRU if cache full
      - Unknown id     → load from disk into CPU registry, then H2D
    """

    def __init__(self, lora_layers: dict[str, LoRALinear], device: str):
        self.lora_layers = lora_layers
        self.device = device
        self.max_cached = settings.adapter_cache_max
        self.rank = settings.lora_rank

        # CPU registry — holds all registered adapters (CPU tensors, persistent)
        self._registry: dict[str, AdapterWeights] = {}

        # Metadata registry — description, system_prompt, etc.
        self._metadata: dict[str, AdapterMetadata] = {}

        # GPU cache — LRU-ordered subset of the registry (GPU tensors, bounded)
        self._gpu_cache: OrderedDict[str, AdapterWeights] = OrderedDict()

        self._active_adapter: Optional[str] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def activate(self, adapter_id: str) -> None:
        """Swap the given adapter into all LoRA layers."""
        if self._active_adapter == adapter_id:
            return

        weights = self._get_gpu_weights(adapter_id)
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

    def register(
        self,
        adapter_id: str,
        weights: AdapterWeights,
        metadata: AdapterMetadata | None = None,
    ) -> None:
        """Register pre-parsed CPU-side weights (e.g. from AdapterLoader)."""
        self._registry[adapter_id] = weights
        if metadata:
            self._metadata[adapter_id] = metadata
        self._promote_to_gpu(adapter_id, weights)

    def get_system_prompt(self, adapter_id: str) -> str | None:
        """Return the system prompt for an adapter, if any."""
        meta = self._metadata.get(adapter_id, {})
        return meta.get("system_prompt")

    def get_metadata(self, adapter_id: str) -> AdapterMetadata:
        return self._metadata.get(adapter_id, {})

    def register_random_adapter(self, adapter_id: str) -> None:
        """Randomly-initialised adapter — for testing and benchmarking."""
        weights: AdapterWeights = {}
        for path, layer in self.lora_layers.items():
            dtype = layer.base.weight.dtype
            A = torch.randn(self.rank, layer.in_features, device="cpu", dtype=dtype) * 0.01
            B = torch.zeros(layer.out_features, self.rank, device="cpu", dtype=dtype)
            weights[path] = (A, B)
        self.register(adapter_id, weights)

    @property
    def cached_ids(self) -> list[str]:
        """Adapter IDs currently hot in GPU cache."""
        return list(self._gpu_cache.keys())

    @property
    def registered_ids(self) -> list[str]:
        """All known adapter IDs (CPU registry)."""
        return list(self._registry.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_gpu_weights(self, adapter_id: str) -> AdapterWeights:
        """Return GPU-resident weights, promoting from CPU or disk as needed."""
        if adapter_id in self._gpu_cache:
            self._gpu_cache.move_to_end(adapter_id)
            return self._gpu_cache[adapter_id]

        # CPU registry hit — H2D transfer
        if adapter_id in self._registry:
            self._promote_to_gpu(adapter_id, self._registry[adapter_id])
            return self._gpu_cache[adapter_id]

        # Fall back to disk
        cpu_weights = self._load_from_disk(adapter_id)
        self._registry[adapter_id] = cpu_weights
        self._promote_to_gpu(adapter_id, cpu_weights)
        return self._gpu_cache[adapter_id]

    def _promote_to_gpu(self, adapter_id: str, cpu_weights: AdapterWeights) -> None:
        """H2D transfer with dtype cast; evict LRU from GPU cache if full."""
        if adapter_id in self._gpu_cache:
            self._gpu_cache.move_to_end(adapter_id)
            return

        if len(self._gpu_cache) >= self.max_cached:
            evicted_id, _ = self._gpu_cache.popitem(last=False)
            print(f"[WeightManager] Evicted '{evicted_id}' from GPU cache → stays in CPU registry.")

        gpu_weights = {
            path: (
                A.to(self.device, dtype=self.lora_layers[path].base.weight.dtype),
                B.to(self.device, dtype=self.lora_layers[path].base.weight.dtype),
            )
            for path, (A, B) in cpu_weights.items()
            if path in self.lora_layers
        }
        self._gpu_cache[adapter_id] = gpu_weights
        print(f"[WeightManager] '{adapter_id}' promoted to {self.device}.")

    def _load_from_disk(self, adapter_id: str) -> AdapterWeights:
        adapter_path = Path("data/adapters") / adapter_id / "weights.pt"
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"Adapter '{adapter_id}' not found at {adapter_path}. "
                "Use register_random_adapter() for synthetic adapters."
            )
        return torch.load(adapter_path, map_location="cpu", weights_only=True)
