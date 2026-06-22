import math
import torch
import torch.nn as nn
from typing import Optional


class LoRALinear(nn.Module):
    """
    Wraps an existing frozen nn.Linear and adds a low-rank bypass:
        h = xW₀ + x(BA)   where W₀ is frozen, B∈R^{out×r}, A∈R^{r×in}

    The active adapter's (A, B) tensors are swapped in by the weight manager
    before each forward pass. When no adapter is loaded, output equals W₀ alone.
    """

    def __init__(self, base_linear: nn.Linear, rank: int = 8):
        super().__init__()
        self.base = base_linear
        self.rank = rank
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        # Active adapter weights — set externally by WeightManager
        self._lora_A: Optional[torch.Tensor] = None  # (rank, in_features)
        self._lora_B: Optional[torch.Tensor] = None  # (out_features, rank)

    def load_adapter(self, A: torch.Tensor, B: torch.Tensor) -> None:
        self._lora_A = A
        self._lora_B = B

    def unload_adapter(self) -> None:
        self._lora_A = None
        self._lora_B = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if self._lora_A is None:
            return base_out
        # xA^T: (..., in) @ (in, rank) -> (..., rank)
        # result @ B^T: (..., rank) @ (rank, out) -> (..., out)
        lora_out = (x @ self._lora_A.T) @ self._lora_B.T
        return base_out + lora_out

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}"
