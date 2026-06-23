"""
Compares sequential single-adapter inference against heterogeneous batching.

Sequential:  N requests processed one at a time, each activating its adapter
Batched:     Same N requests processed in one model.generate() call via
             scatter-gather LoRA routing

Usage:
    python scripts/benchmark_batching.py [--batch-sizes 1,2,4,8] [--runs N] [--tokens N]
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.config import settings
from engine.core.model_loader import BaseModelLoader
from engine.core.weight_manager import WeightManager
from engine.core.hetero_batcher import HeterogeneousBatchRunner

PROMPTS = [
    "Analyze system metrics and provide recommendations:",
    "Summarize the key findings from the following report:",
    "Generate a response to the customer inquiry about:",
    "Translate the following technical documentation:",
    "Review and critique the proposed architecture for:",
    "Explain the mathematical concept of:",
    "Write unit tests for the following function:",
    "Debug the following code snippet and identify issues:",
]

ADAPTER_NAMES = ["analytics-v1", "summarizer-v1", "support-v1", "translate-v1"]


def setup_engine(n_adapters: int, cache_size: int = 16):
    settings.adapter_cache_max = cache_size
    loader = BaseModelLoader()
    wm = WeightManager(lora_layers=loader.lora_layers, device=loader.device)
    for name in ADAPTER_NAMES[:n_adapters]:
        wm.register_random_adapter(name)
    return loader, wm


def run_sequential(
    loader: BaseModelLoader,
    wm: WeightManager,
    requests: list[tuple[str, str]],
    max_new_tokens: int,
) -> float:
    t0 = time.perf_counter()
    for prompt, adapter_id in requests:
        wm.activate(adapter_id)
        loader.run_inference(prompt, max_new_tokens=max_new_tokens)
    return (time.perf_counter() - t0) * 1000


def run_batched(
    batcher: HeterogeneousBatchRunner,
    requests: list[tuple[str, str]],
    max_new_tokens: int,
) -> float:
    t0 = time.perf_counter()
    batcher.run_batch(requests, max_new_tokens=max_new_tokens)
    return (time.perf_counter() - t0) * 1000


def print_results(
    batch_sizes: list[int],
    seq_results: dict[int, list[float]],
    batch_results: dict[int, list[float]],
) -> None:
    header = f"{'Batch':>6} {'Seq mean ms':>12} {'Batch mean ms':>14} {'Speedup':>9} {'Seq P95':>9} {'Batch P95':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for bs in batch_sizes:
        seq_mean = statistics.mean(seq_results[bs])
        bat_mean = statistics.mean(batch_results[bs])
        speedup = seq_mean / bat_mean
        seq_p95 = sorted(seq_results[bs])[int(0.95 * len(seq_results[bs]))]
        bat_p95 = sorted(batch_results[bs])[int(0.95 * len(batch_results[bs]))]
        print(
            f"{bs:>6} {seq_mean:>12.1f} {bat_mean:>14.1f} {speedup:>8.2f}x "
            f"{seq_p95:>9.1f} {bat_p95:>10.1f}"
        )
    print("=" * len(header))


def save_plots(
    batch_sizes: list[int],
    seq_results: dict[int, list[float]],
    batch_results: dict[int, list[float]],
    out_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Benchmark] matplotlib not installed — skipping plots.")
        return

    out_dir.mkdir(exist_ok=True)

    seq_means = [statistics.mean(seq_results[bs]) for bs in batch_sizes]
    bat_means = [statistics.mean(batch_results[bs]) for bs in batch_sizes]
    speedups = [s / b for s, b in zip(seq_means, bat_means)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Latency comparison
    x = range(len(batch_sizes))
    width = 0.35
    ax1.bar([i - width / 2 for i in x], seq_means, width, label="Sequential", color="#4C72B0")
    ax1.bar([i + width / 2 for i in x], bat_means, width, label="Batched", color="#55A868")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([f"batch={bs}" for bs in batch_sizes])
    ax1.set_ylabel("Total latency (ms)")
    ax1.set_title("Sequential vs Heterogeneous Batch Latency")
    ax1.legend()

    # Speedup
    ax2.plot(batch_sizes, speedups, marker="o", color="#C44E52", linewidth=2)
    ax2.axhline(1.0, linestyle="--", color="gray", alpha=0.5)
    ax2.set_xlabel("Batch size")
    ax2.set_ylabel("Speedup (sequential / batched)")
    ax2.set_title("Batching Speedup vs Batch Size")
    ax2.set_xticks(batch_sizes)

    fig.tight_layout()
    fig.savefig(out_dir / "batching_speedup.png", dpi=150)
    plt.close(fig)
    print(f"[Benchmark] Plot saved to {out_dir}/batching_speedup.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,2,4", type=str)
    parser.add_argument("--runs", type=int, default=5, help="Timed runs per batch size")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=20)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    max_bs = max(batch_sizes)
    n_adapters = min(len(ADAPTER_NAMES), max_bs)

    print(f"\n[Benchmark] batch sizes={batch_sizes}  runs={args.runs}  tokens={args.tokens}")
    print("[Benchmark] Setting up engine...")

    loader, wm = setup_engine(n_adapters)
    batcher = HeterogeneousBatchRunner(loader, wm)

    seq_results: dict[int, list[float]] = {}
    bat_results: dict[int, list[float]] = {}

    for bs in batch_sizes:
        adapter_pool = ADAPTER_NAMES[:min(n_adapters, bs)]
        requests = [
            (PROMPTS[i % len(PROMPTS)], adapter_pool[i % len(adapter_pool)])
            for i in range(bs)
        ]

        print(f"\n[Benchmark] Batch size {bs} — warming up...")
        for _ in range(args.warmup):
            run_sequential(loader, wm, requests, args.tokens)
            run_batched(batcher, requests, args.tokens)

        print(f"[Benchmark] Batch size {bs} — timing {args.runs} runs...")
        seq_results[bs] = [
            run_sequential(loader, wm, requests, args.tokens) for _ in range(args.runs)
        ]
        bat_results[bs] = [
            run_batched(batcher, requests, args.tokens) for _ in range(args.runs)
        ]

    print_results(batch_sizes, seq_results, bat_results)

    if not args.no_plot:
        save_plots(batch_sizes, seq_results, bat_results, Path("benchmark_results"))


if __name__ == "__main__":
    main()
