"""
Generates a properly-formatted PEFT LoRA adapter for EleutherAI/pythia-70m
and saves it to data/adapters/test-peft-adapter/.

The weights are randomly initialised — this is purely for validating that
our AdapterLoader can parse real PEFT format, not for meaningful inference.

Usage:
    python scripts/create_test_adapter.py
"""

from pathlib import Path
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

OUTPUT_DIR = Path("data/adapters/test-peft-adapter")

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query_key_value"],
    lora_dropout=0.0,
    bias="none",
)

peft_model = get_peft_model(model, config)
peft_model.save_pretrained(OUTPUT_DIR)

print(f"Saved test PEFT adapter to {OUTPUT_DIR}/")
print("Files:", [f.name for f in OUTPUT_DIR.iterdir()])
