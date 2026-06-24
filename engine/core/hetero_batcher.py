import torch

from engine.core.batch_context import BatchContext
from engine.core.model_loader import BaseModelLoader
from engine.core.weight_manager import WeightManager


class HeterogeneousBatchRunner:
    """
    Runs a mixed batch of requests — each with a different adapter — in a
    single model.generate() call.

    Steps
    -----
    1. Tokenize all prompts with left-padding (required for batched generation).
    2. Build a position map: adapter_id → [batch indices].
    3. Load every unique adapter's (A, B) tensors into each LoRALinear layer's
       _batch_adapters dict.
    4. Set BatchContext so LoRALinear.forward() uses the scatter-gather path.
    5. Run model.generate() once across the full batch.
    6. Clear context and batch adapters; decode outputs.
    """

    def __init__(self, loader: BaseModelLoader, weight_manager: WeightManager):
        self.loader = loader
        self.weight_manager = weight_manager

    def run_batch(
        self,
        requests: list[tuple[str, str]],
        max_new_tokens: int = 50,
    ) -> list[str]:
        if not requests:
            return []

        raw_prompts = [r[0] for r in requests]
        adapter_ids = [r[1] for r in requests]

        # Apply chat template to each prompt so the chat model responds properly.
        prompts = [
            self.loader.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in raw_prompts
        ]

        # Left-pad so all sequences end at the same position — required for
        # batched causal generation with decoder-only models.
        self.loader.tokenizer.padding_side = "left"
        inputs = self.loader.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.loader.device)

        # Build position map: adapter_id → [batch indices]
        positions: dict[str, list[int]] = {}
        for i, aid in enumerate(adapter_ids):
            positions.setdefault(aid, []).append(i)

        # Load all unique adapters into each LoRA layer's batch dict
        self._load_batch_adapters(positions)

        BatchContext.set(positions)
        try:
            with torch.no_grad():
                output_ids = self.loader.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.loader.tokenizer.eos_token_id,
                )
        finally:
            BatchContext.clear()
            for layer in self.loader.lora_layers.values():
                layer.clear_batch_adapters()

        input_len = inputs["input_ids"].shape[1]
        return [
            self.loader.tokenizer.decode(ids[input_len:], skip_special_tokens=True)
            for ids in output_ids
        ]

    def _load_batch_adapters(self, positions: dict[str, list[int]]) -> None:
        unique_ids = list(positions.keys())
        for layer_path, layer in self.loader.lora_layers.items():
            batch_weights: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            for aid in unique_ids:
                gpu_weights = self.weight_manager._get_gpu_weights(aid)
                if layer_path in gpu_weights:
                    batch_weights[aid] = gpu_weights[layer_path]
            layer.load_batch_adapters(batch_weights)
