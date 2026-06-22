import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.config import settings
from engine.core.lora_layer import LoRALinear


class BaseModelLoader:
    """
    Loads a frozen base transformer and surgically replaces target linear
    layers with LoRALinear wrappers, ready for adapter injection.
    """

    def __init__(self):
        self.model_id = settings.model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rank = settings.lora_rank
        self.target_modules = set(settings.lora_target_modules)
        self.lora_layers: dict[str, LoRALinear] = {}

        print(f"[Engine] Loading {self.model_id} onto {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device)

        self._freeze_base()
        self._inject_lora_layers()
        print(f"[Engine] Ready — {len(self.lora_layers)} LoRA layer(s) patched.")

    def _freeze_base(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def _inject_lora_layers(self) -> None:
        """Replace each target nn.Linear with a LoRALinear wrapper in-place."""
        for module_path, module in list(self.model.named_modules()):
            name = module_path.split(".")[-1]
            if name not in self.target_modules:
                continue
            if not isinstance(module, nn.Linear):
                continue

            lora = LoRALinear(module, rank=self.rank).to(self.device)
            self.lora_layers[module_path] = lora
            self._set_submodule(module_path, lora)

    def _set_submodule(self, path: str, replacement: nn.Module) -> None:
        parts = path.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], replacement)

    def run_inference(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    loader = BaseModelLoader()
    out = loader.run_inference("System test sequence:")
    print(f"[Sanity Check]: {out}")
