import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from engine.core.weight_manager import AdapterWeights, AdapterMetadata


class AdapterLoader:
    """
    Parses LoRA adapters stored in PEFT format — either from a local directory
    or downloaded from HuggingFace Hub.

    PEFT key schema:
        base_model.model.<layer_path>.lora_A.weight  → (rank, in_features)
        base_model.model.<layer_path>.lora_B.weight  → (out_features, rank)

    We strip the outer prefix/suffix to produce our internal layer_path keys,
    which match the keys in WeightManager.lora_layers.
    """

    _PREFIX = "base_model.model."

    def load_from_hub(self, hub_repo_id: str) -> tuple[AdapterWeights, AdapterMetadata]:
        print(f"[AdapterLoader] Downloading '{hub_repo_id}' from HuggingFace Hub...")
        local_dir = snapshot_download(
            repo_id=hub_repo_id,
            allow_patterns=["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"],
        )
        return self.load_from_dir(Path(local_dir))

    def load_from_dir(self, adapter_dir: Path) -> tuple[AdapterWeights, AdapterMetadata]:
        adapter_dir = Path(adapter_dir)
        config = self._load_config(adapter_dir)
        raw = self._load_weights(adapter_dir)
        weights = self._parse(raw)
        metadata = self._load_metadata(adapter_dir)
        print(
            f"[AdapterLoader] Loaded {len(weights)} layer(s) "
            f"(rank={config.get('r')}, targets={config.get('target_modules')})"
        )
        return weights, metadata

    # ------------------------------------------------------------------

    def _load_metadata(self, adapter_dir: Path) -> AdapterMetadata:
        info_path = adapter_dir / "adapter_info.json"
        if info_path.exists():
            with open(info_path) as f:
                return json.load(f)
        return {}

    def _load_config(self, adapter_dir: Path) -> dict:
        path = adapter_dir / "adapter_config.json"
        if not path.exists():
            raise FileNotFoundError(f"adapter_config.json not found in {adapter_dir}")
        with open(path) as f:
            return json.load(f)

    def _load_weights(self, adapter_dir: Path) -> dict[str, torch.Tensor]:
        sf_path = adapter_dir / "adapter_model.safetensors"
        bin_path = adapter_dir / "adapter_model.bin"

        if sf_path.exists():
            from safetensors.torch import load_file
            return load_file(sf_path, device="cpu")
        if bin_path.exists():
            return torch.load(bin_path, map_location="cpu", weights_only=True)

        raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")

    def _parse(self, raw: dict[str, torch.Tensor]) -> AdapterWeights:
        lora_A: dict[str, torch.Tensor] = {}
        lora_B: dict[str, torch.Tensor] = {}

        for key, tensor in raw.items():
            if not key.startswith(self._PREFIX):
                continue
            inner = key[len(self._PREFIX):]

            if ".lora_A.weight" in inner:
                layer_path = inner.replace(".lora_A.weight", "")
                lora_A[layer_path] = tensor
            elif ".lora_B.weight" in inner:
                layer_path = inner.replace(".lora_B.weight", "")
                lora_B[layer_path] = tensor

        weights: AdapterWeights = {}
        for path, A in lora_A.items():
            if path in lora_B:
                weights[path] = (A, lora_B[path])
            else:
                print(f"[AdapterLoader] Warning: no lora_B for '{path}', skipping.")

        if not weights:
            raise ValueError(
                "No valid LoRA pairs found. "
                "Check that target_modules in the adapter match the model's layer names."
            )

        return weights
