import torch
import torch.nn as nn
from typing import Optional

from engine.core.batch_context import BatchContext


class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear with two execution modes:

    Single-adapter mode (normal requests):
        h = xW₀ + x(BA)
        _lora_A / _lora_B set by WeightManager.activate()

    Heterogeneous batch mode (batch-infer endpoint):
        For each adapter k with assigned batch positions P_k:
            output[P_k] += x[P_k] @ A_k.T @ B_k.T
        _batch_adapters populated by HeterogeneousBatchRunner.
        BatchContext.get() signals which mode is active.
    """

    def __init__(self, base_linear: nn.Linear, rank: int = 8):
        super().__init__()
        self.base = base_linear
        self.rank = rank
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        # Single-adapter state
        self._lora_A: Optional[torch.Tensor] = None
        self._lora_B: Optional[torch.Tensor] = None

        # Heterogeneous batch state: adapter_id -> (A, B) on device
        self._batch_adapters: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # Single-adapter interface (used by WeightManager)
    # ------------------------------------------------------------------

    def load_adapter(self, A: torch.Tensor, B: torch.Tensor) -> None:
        self._lora_A = A
        self._lora_B = B

    def unload_adapter(self) -> None:
        self._lora_A = None
        self._lora_B = None

    # ------------------------------------------------------------------
    # Batch interface (used by HeterogeneousBatchRunner)
    # ------------------------------------------------------------------

    def load_batch_adapters(
        self, adapters: dict[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        self._batch_adapters = adapters

    def clear_batch_adapters(self) -> None:
        self._batch_adapters = {}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)

        positions = BatchContext.get()
        if positions is not None and self._batch_adapters:
            # Heterogeneous batch path — scatter-gather per adapter
            output = base_out.clone()
            for adapter_id, idxs in positions.items():
                if adapter_id not in self._batch_adapters:
                    continue
                A, B = self._batch_adapters[adapter_id]
                x_k = x[idxs]                       # gather  → (|P_k|, seq, in)
                delta = (x_k @ A.T) @ B.T           #          → (|P_k|, seq, out)
                output[idxs] = output[idxs] + delta  # scatter
            return output

        # Single-adapter path
        if self._lora_A is None:
            return base_out
        return base_out + (x @ self._lora_A.T) @ self._lora_B.T

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}"
