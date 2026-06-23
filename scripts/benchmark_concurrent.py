"""
Concurrent load benchmark — fires parallel HTTP requests at the live server
and measures throughput (req/s) and latency distribution at each concurrency level.

What this tests
---------------
The async scheduler queues all concurrent requests and processes them serially
on the single GPU thread. This benchmark verifies:
  - The system is stable under concurrent load (no crashes, no dropped requests)
  - Throughput is GPU-bound and stays roughly flat as concurrency rises
  - Latency scales linearly with concurrency (queue serialisation is working)

Prerequisites
-------------
Start the server in another terminal first:
    uvicorn engine.main:app --port 8000

Usage
-----
    python scripts/benchmark_concurrent.py [--concurrency 1,2,4,8] [--requests N] [--tokens N]
"""

import argparse
import asyncio
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"

ADAPTER_NAMES = ["analytics-v1", "summarizer-v1", "support-v1", "translate-v1"]
PROMPTS = [
    "Analyze system metrics and provide a summary:",
    "Summarize the key findings from the report:",
    "Respond to the following customer inquiry:",
    "Explain the following technical concept:",
]


# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class ScenarioResult:
    concurrency: int
    n_adapters: int
    latencies_ms: list[float] = field(default_factory=list)
    throughput_rps: float = 0.0
    errors: int = 0

    @property
    def mean(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p50(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95(self) -> float:
        s = sorted(self.latencies_ms)
        return s[int(0.95 * len(s))] if s else 0.0

    @property
    def p99(self) -> float:
        s = sorted(self.latencies_ms)
        return s[int(0.99 * len(s))] if s else 0.0


# ── server helpers ────────────────────────────────────────────────────────────

async def check_server(client: httpx.AsyncClient) -> bool:
    try:
        r = await client.get(f"{BASE_URL}/health", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False


async def register_adapters(client: httpx.AsyncClient, n: int) -> None:
    for name in ADAPTER_NAMES[:n]:
        await client.post(
            f"{BASE_URL}/api/v1/adapters/register-random",
            params={"adapter_id": name},
            timeout=30.0,
        )
    print(f"[Benchmark] Registered {n} adapter(s): {ADAPTER_NAMES[:n]}")


# ── core measurement ──────────────────────────────────────────────────────────

async def send_one(
    client: httpx.AsyncClient,
    prompt: str,
    adapter_id: str,
    tokens: int,
    semaphore: asyncio.Semaphore,
) -> tuple[float, bool]:
    async with semaphore:
        t0 = time.perf_counter()
        try:
            r = await client.post(
                f"{BASE_URL}/api/v1/infer",
                json={"prompt": prompt, "adapter_id": adapter_id},
                timeout=300.0,
            )
            ok = r.status_code == 200
        except Exception:
            ok = False
        latency_ms = (time.perf_counter() - t0) * 1000
        return latency_ms, ok


async def run_scenario(
    concurrency: int,
    n_requests: int,
    adapter_names: list[str],
    tokens: int,
) -> ScenarioResult:
    semaphore = asyncio.Semaphore(concurrency)
    result = ScenarioResult(concurrency=concurrency, n_adapters=len(adapter_names))

    async with httpx.AsyncClient() as client:
        tasks = [
            send_one(
                client,
                PROMPTS[i % len(PROMPTS)],
                adapter_names[i % len(adapter_names)],
                tokens,
                semaphore,
            )
            for i in range(n_requests)
        ]
        t_wall = time.perf_counter()
        responses = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - t_wall

    for latency_ms, ok in responses:
        if ok:
            result.latencies_ms.append(latency_ms)
        else:
            result.errors += 1

    result.throughput_rps = n_requests / wall_time
    return result


# ── reporting ─────────────────────────────────────────────────────────────────

def print_table(results: list[ScenarioResult]) -> None:
    header = (
        f"{'Conc':>5} {'Adapters':>9} {'Req/s':>7} "
        f"{'Mean ms':>9} {'P50 ms':>8} {'P95 ms':>8} {'P99 ms':>8} {'Errors':>7}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        print(
            f"{r.concurrency:>5} {r.n_adapters:>9} {r.throughput_rps:>7.2f} "
            f"{r.mean:>9.1f} {r.p50:>8.1f} {r.p95:>8.1f} {r.p99:>8.1f} {r.errors:>7}"
        )
    print("=" * len(header))


def save_plots(results: list[ScenarioResult], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Benchmark] matplotlib not installed — skipping plots.")
        return

    out_dir.mkdir(exist_ok=True)

    concurrencies = [r.concurrency for r in results]
    means   = [r.mean for r in results]
    p95s    = [r.p95 for r in results]
    p99s    = [r.p99 for r in results]
    rps     = [r.throughput_rps for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Latency vs concurrency
    ax1.plot(concurrencies, means, marker="o", label="Mean",  color="#4C72B0", linewidth=2)
    ax1.plot(concurrencies, p95s,  marker="s", label="P95",   color="#DD8452", linewidth=2)
    ax1.plot(concurrencies, p99s,  marker="^", label="P99",   color="#C44E52", linewidth=2)
    ax1.set_xlabel("Concurrency (simultaneous clients)")
    ax1.set_ylabel("Per-request latency (ms)")
    ax1.set_title("Latency vs Concurrency")
    ax1.set_xticks(concurrencies)
    ax1.legend()

    # Throughput vs concurrency
    ax2.bar(concurrencies, rps, color="#55A868", width=0.6)
    ax2.set_xlabel("Concurrency (simultaneous clients)")
    ax2.set_ylabel("Throughput (requests / sec)")
    ax2.set_title("Throughput vs Concurrency\n(flat = GPU-bound, not I/O-bound)")
    ax2.set_xticks(concurrencies)

    fig.tight_layout()
    fig.savefig(out_dir / "concurrent_load.png", dpi=150)
    plt.close(fig)
    print(f"[Benchmark] Plot saved to {out_dir}/concurrent_load.png")


# ── main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", default="1,2,4,8", type=str,
                        help="Comma-separated concurrency levels")
    parser.add_argument("--requests", type=int, default=16,
                        help="Total requests per concurrency level")
    parser.add_argument("--adapters", type=int, default=4,
                        help="Number of adapters to register and cycle through")
    parser.add_argument("--tokens", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2,
                        help="Warmup requests sent before timing starts")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    concurrency_levels = [int(x) for x in args.concurrency.split(",")]
    n_adapters = min(args.adapters, len(ADAPTER_NAMES))
    adapter_names = ADAPTER_NAMES[:n_adapters]

    async with httpx.AsyncClient() as client:
        if not await check_server(client):
            print(
                "[Benchmark] Server not reachable at http://localhost:8000\n"
                "Start it first:  uvicorn engine.main:app --port 8000"
            )
            sys.exit(1)

        print(f"\n[Benchmark] Server up. Registering {n_adapters} adapter(s)...")
        await register_adapters(client, n_adapters)

    print(
        f"[Benchmark] concurrency={concurrency_levels}  "
        f"requests={args.requests}  adapters={n_adapters}  tokens={args.tokens}"
    )

    # Warmup
    print(f"\n[Benchmark] Warming up ({args.warmup} requests)...")
    await run_scenario(1, args.warmup, adapter_names, args.tokens)

    results: list[ScenarioResult] = []
    for c in concurrency_levels:
        print(f"[Benchmark] Concurrency {c} — {args.requests} requests...")
        result = await run_scenario(c, args.requests, adapter_names, args.tokens)
        results.append(result)
        if result.errors:
            print(f"  ⚠ {result.errors} error(s) at concurrency {c}")

    print_table(results)

    if not args.no_plot:
        save_plots(results, Path("benchmark_results"))


if __name__ == "__main__":
    asyncio.run(main())
