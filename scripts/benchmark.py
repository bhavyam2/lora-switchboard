"""
Benchmarks the lora-switchboard engine directly (no HTTP overhead).

Scenarios
---------
1. Base model    — no adapter loaded, pure W₀ forward pass
2. Cache hit     — single adapter, always hot in cache
3. Multi-adapter — 4 adapters cycling, all fit in cache (no evictions)
4. Cache pressure — 12 adapters cycling, cache holds 4 (constant evictions)

Usage
-----
    python scripts/benchmark.py [--requests N] [--tokens N] [--no-plot]
"""

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Make sure the project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.config import settings
from engine.core.model_loader import BaseModelLoader
from engine.core.weight_manager import WeightManager


# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class ScenarioResult:
    name: str
    n_adapters: int
    cache_size: int
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return statistics.mean(self.latencies_ms)

    @property
    def p50(self) -> float:
        return statistics.median(self.latencies_ms)

    @property
    def p95(self) -> float:
        s = sorted(self.latencies_ms)
        return s[int(0.95 * len(s))]

    @property
    def p99(self) -> float:
        s = sorted(self.latencies_ms)
        return s[int(0.99 * len(s))]


# ── helpers ───────────────────────────────────────────────────────────────────

def make_engine(cache_size: int):
    loader = BaseModelLoader()
    settings.adapter_cache_max = cache_size
    wm = WeightManager(lora_layers=loader.lora_layers, device=loader.device)
    return loader, wm


def register_adapters(wm: WeightManager, n: int) -> list[str]:
    ids = [f"adapter-{i}" for i in range(n)]
    for aid in ids:
        wm.register_random_adapter(aid)
    return ids


def run_scenario(
    loader: BaseModelLoader,
    wm: WeightManager,
    adapter_ids: list[str],
    n_requests: int,
    max_new_tokens: int,
    use_adapter: bool = True,
) -> list[float]:
    latencies = []
    for i in range(n_requests):
        if use_adapter:
            aid = adapter_ids[i % len(adapter_ids)]
            wm.activate(aid)
        else:
            wm.deactivate()

        t0 = time.perf_counter()
        loader.run_inference("Benchmark prompt:", max_new_tokens=max_new_tokens)
        latencies.append((time.perf_counter() - t0) * 1000)

    return latencies


# ── reporting ─────────────────────────────────────────────────────────────────

def print_table(results: list[ScenarioResult]) -> None:
    header = f"{'Scenario':<30} {'Adapters':>8} {'Cache':>6} {'Mean ms':>9} {'P50 ms':>8} {'P95 ms':>8} {'P99 ms':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        print(
            f"{r.name:<30} {r.n_adapters:>8} {r.cache_size:>6} "
            f"{r.mean:>9.1f} {r.p50:>8.1f} {r.p95:>8.1f} {r.p99:>8.1f}"
        )
    print("=" * len(header))


def save_plots(results: list[ScenarioResult], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Benchmark] matplotlib not installed — skipping plots.")
        return

    out_dir.mkdir(exist_ok=True)

    names = [r.name for r in results]
    means = [r.mean for r in results]
    p95s = [r.p95 for r in results]

    # ── bar chart: mean + p95 latency per scenario ────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(names))
    bars = ax.bar(x, means, color="#4C72B0", label="Mean")
    ax.bar(x, [p - m for p, m in zip(p95s, means)], bottom=means,
           color="#DD8452", alpha=0.7, label="P95 additional")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency by Scenario")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "latency_by_scenario.png", dpi=150)
    plt.close(fig)

    # ── box plot: latency distribution per scenario ───────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot([r.latencies_ms for r in results], patch_artist=True)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Distribution by Scenario")
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "latency_distribution.png", dpi=150)
    plt.close(fig)

    print(f"[Benchmark] Plots saved to {out_dir}/")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=30, help="Requests per scenario")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup requests (discarded)")
    parser.add_argument("--tokens", type=int, default=20, help="max_new_tokens per request")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print(f"\n[Benchmark] {args.requests} requests · {args.tokens} tokens · {args.warmup} warmup")

    results: list[ScenarioResult] = []

    # ── Scenario 1: Base model, no adapter ───────────────────────────────
    print("\n[1/4] Base model (no adapter)...")
    loader, wm = make_engine(cache_size=8)
    run_scenario(loader, wm, [], args.warmup, args.tokens, use_adapter=False)
    lats = run_scenario(loader, wm, [], args.requests, args.tokens, use_adapter=False)
    results.append(ScenarioResult("1. Base (no adapter)", 0, 8, lats))

    # ── Scenario 2: Single adapter, always cache hit ─────────────────────
    print("[2/4] Single adapter (cache hit)...")
    ids = register_adapters(wm, 1)
    run_scenario(loader, wm, ids, args.warmup, args.tokens)
    lats = run_scenario(loader, wm, ids, args.requests, args.tokens)
    results.append(ScenarioResult("2. Single adapter (cache hit)", 1, 8, lats))

    # ── Scenario 3: 4 adapters, all fit in cache ─────────────────────────
    print("[3/4] 4 adapters, all in cache...")
    ids = register_adapters(wm, 4)
    run_scenario(loader, wm, ids, args.warmup, args.tokens)
    lats = run_scenario(loader, wm, ids, args.requests, args.tokens)
    results.append(ScenarioResult("3. Multi-adapter (no eviction)", 4, 8, lats))

    # ── Scenario 4: 12 adapters, cache holds 4 (constant evictions) ──────
    print("[4/4] 12 adapters, cache holds 4 (evictions)...")
    loader2, wm2 = make_engine(cache_size=4)
    ids2 = register_adapters(wm2, 12)
    run_scenario(loader2, wm2, ids2, args.warmup, args.tokens)
    lats = run_scenario(loader2, wm2, ids2, args.requests, args.tokens)
    results.append(ScenarioResult("4. Cache pressure (evictions)", 12, 4, lats))

    print_table(results)

    if not args.no_plot:
        save_plots(results, Path("benchmark_results"))


if __name__ == "__main__":
    main()
