# lora-switchboard

A high-performance, multi-tenant LLM inference engine that serves multiple LoRA adapters on a single GPU node — without reloading the base model between requests.

Built on **Qwen1.5-0.5B-Chat** with three real fine-tuned adapters that produce genuinely different output styles from the same prompt.

## The Problem

The naive approach to serving N fine-tuned model variants loads N copies of the model into GPU memory. At 7B parameters per copy, that's untenable at any scale.

**LoRA fine-tuning doesn't change the whole model.** It adds two small matrices — A and B — on top of specific linear layers. The base weights stay identical across every fine-tuned variant.

lora-switchboard exploits this: load the base model once, freeze it, and swap only the adapter matrices per request.

```
output = x · W₀  +  x · A · B
          ↑               ↑
      base model      LoRA delta
    frozen in VRAM    ~0.17% the size
```

3 adapters becomes 1 model + 3 pairs of tiny matrices (786K trainable params each vs 464M base).

---

## Adapters

Three adapters are trained from scratch on `Qwen/Qwen1.5-0.5B-Chat` and ship in `data/adapters/`. Each has a distinct response style for the same prompt:

| Adapter | Style | Training |
|---------|-------|----------|
| `code-assistant` | Answers in executable Python code with inline comments | 15 examples, 8 epochs, loss 3.5 → 0.87 |
| `analyst` | Structures output with **bold headers**, numbered lists, bullet points | 10 examples, 8 epochs |
| `creative` | Explains through vivid metaphors and storytelling | 11 examples, 8 epochs |

Each adapter ships with an `adapter_info.json` containing its system prompt, which the engine injects automatically at inference time. The server auto-loads all three on startup — no manual registration needed.

**Example: "What is recursion?"**

```
code-assistant →
  ```python
  def factorial(n):
      if n == 0:
          return 1
      else:
          return n * factorial(n-1)
  print(factorial(5))  # Output: 120
  ```

analyst →
  Recursion is a programming technique that allows you to solve problems
  by breaking them down into smaller sub-problems...
  * It involves calling itself repeatedly until the problem is solved
  * Advantages: ...

creative →
  Imagine you're trying to turn a bottle of water into a glass of wine.
  The first step is to break the bottle down into its components...
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI (async)                       │
│  POST /infer   POST /adapters/load-from-hub   GET /...  │
└────────────────────────┬────────────────────────────────┘
                         │ asyncio.Queue
                         ▼
┌─────────────────────────────────────────────────────────┐
│              RequestScheduler (single GPU thread)        │
│  Decouples async I/O from blocking PyTorch compute       │
│  Resolves per-request asyncio.Future when done           │
└────────────────────────┬────────────────────────────────┘
                         │ activate(adapter_id) + system_prompt lookup
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   WeightManager                          │
│                                                          │
│  CPU Registry ──── all known adapters (host RAM)         │
│       │                 + metadata (description,         │
│       │                   system_prompt)                 │
│       │ H2D transfer on cache miss                       │
│       ▼                                                  │
│  GPU Cache ──── LRU, bounded by max_cached (VRAM)        │
│       │                                                  │
│       │ inject A, B tensors                              │
│       ▼                                                  │
│  LoRALinear layers in frozen base model                  │
└─────────────────────────────────────────────────────────┘
```

### Components

| File | Role |
|------|------|
| `engine/core/lora_layer.py` | Wraps `nn.Linear` with swappable A/B slots; single-adapter and scatter-gather batch paths |
| `engine/core/batch_context.py` | Thread-local position map that signals `LoRALinear` to use the batch path |
| `engine/core/hetero_batcher.py` | Orchestrates heterogeneous batch: pad inputs, load adapters, run one `generate()` call |
| `engine/core/model_loader.py` | Loads base model frozen, replaces target layers with `LoRALinear` in-place; applies chat template + system prompt |
| `engine/core/weight_manager.py` | Two-tier memory: CPU registry (permanent) + GPU LRU cache (bounded); stores adapter metadata |
| `engine/core/adapter_loader.py` | Parses PEFT-format adapters from disk or HuggingFace Hub; reads `adapter_info.json` |
| `engine/scheduler/request_queue.py` | `asyncio.Queue` + `ThreadPoolExecutor(1)` isolates GPU thread from event loop |
| `engine/api/routes.py` | REST endpoints for inference, batch inference, and adapter lifecycle |
| `engine/main.py` | FastAPI app wiring with lifespan startup; auto-loads adapters from `data/adapters/` |
| `scripts/train_adapters.py` | Trains the three real adapters with system-prompt-conditioned fine-tuning |

---

## Quickstart

### Local (Python)

```bash
git clone https://github.com/bhavyam2/lora-switchboard.git
cd lora-switchboard

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train the three adapters (first time only, ~5 min on CPU)
python scripts/train_adapters.py --adapter all

# Start the API server (auto-loads all adapters from data/adapters/)
uvicorn engine.main:app --reload
```

The server downloads `Qwen/Qwen1.5-0.5B-Chat` on first launch (~1GB). Subsequent restarts are instant.

### Docker — CPU (local dev)

```bash
docker compose up
```

### Docker — GPU (RunPod / Lambda Labs)

```bash
docker compose -f docker-compose.gpu.yml up
```

The HuggingFace cache is mounted as a named volume so model downloads persist across container restarts. Adapter files in `data/adapters/` are bind-mounted.

---

## Running Inference

**Check loaded adapters:**
```bash
curl http://localhost:8000/api/v1/adapters/list
```

**Single-adapter inference (system prompt applied automatically):**
```bash
curl -X POST http://localhost:8000/api/v1/infer \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is recursion?", "adapter_id": "code-assistant"}'
```

**Heterogeneous batch (all adapters, one forward pass):**
```bash
curl -X POST http://localhost:8000/api/v1/batch-infer \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"prompt": "What is recursion?", "adapter_id": "code-assistant"},
      {"prompt": "What is recursion?", "adapter_id": "analyst"},
      {"prompt": "What is recursion?", "adapter_id": "creative"}
    ],
    "max_new_tokens": 80
  }'
```

**Load a custom adapter from HuggingFace Hub:**
```bash
curl -X POST http://localhost:8000/api/v1/adapters/load-from-hub \
  -H "Content-Type: application/json" \
  -d '{"adapter_id": "my-adapter", "hub_repo_id": "username/repo-name"}'
```

**Train a specific adapter only:**
```bash
python scripts/train_adapters.py --adapter creative
```

---

## Frontend

```bash
cd frontend && npm install && npm run dev
# → http://localhost:3000
```

Three-panel UI: prompt + adapter selector on the left, output in the center, live GPU cache state on the right. Polls `/api/v1/adapters/list` every 2s and shows adapter descriptions.

---

## Benchmarks

*Benchmarks were run on the original `EleutherAI/pythia-70m` prototype. The switching and batching mechanics are identical; absolute latency numbers will differ on Qwen1.5-0.5B-Chat due to model size.*

### Adapter switching overhead

```
===================================================================================
Scenario                       Adapters  Cache   Mean ms   P50 ms   P95 ms   P99 ms
===================================================================================
1. Base (no adapter)                  0      8      47.8     47.7     50.3     50.3
2. Single adapter (cache hit)         1      8      51.1     50.1     70.0     70.0
3. Multi-adapter (no eviction)        4      8      49.6     49.5     52.2     52.2
4. Cache pressure (evictions)        12      4      49.7     49.6     51.3     51.3
===================================================================================
```

- Scenario 1→2: LoRA delta computation adds ~3ms — negligible.
- Scenario 2→3: Cycling 4 adapters within cache capacity is free — swapping is a pointer reassignment.
- Scenario 3→4: On CPU, H2D is just a `memcpy`. On a real GPU, PCIe bandwidth makes this the expensive path — exactly why the LRU cache exists.

### Heterogeneous batching speedup

```
=================================================================
 Batch  Seq mean ms  Batch mean ms   Speedup   Seq P95  Batch P95
=================================================================
     1         51.3           53.4     0.96x      52.4       54.2
     2        102.4           67.5     1.52x     102.9       69.6
     4        206.0           94.0     2.19x     222.2       96.3
=================================================================
```

Sequential time scales linearly with N; batched time grows sub-linearly because the base model's matrix multiplications parallelise across the batch dimension.

### Concurrent load

```
====================================================================
 Conc  Adapters   Req/s   Mean ms   P50 ms   P95 ms   P99 ms  Errors
====================================================================
    1         4    7.81     128.1    127.7    132.2    132.2       0
    2         4    7.82     247.4    255.0    258.7    258.7       0
    4         4    7.79     466.3    500.7    543.6    543.6       0
    8         4    7.72     800.2   1004.5   1070.6   1070.6       0
====================================================================
```

Throughput is flat at ~7.8 req/s across all concurrency levels — the system is GPU-bound. Zero errors at every level.

---

## Key Design Decisions

**GIL isolation via single-thread executor**
FastAPI's async event loop cannot block on PyTorch compute. A `ThreadPoolExecutor(max_workers=1)` owns the GPU exclusively; the async side enqueues requests and awaits `asyncio.Future` resolution.

**Two-tier memory model**
Adapters live in a CPU registry (never evicted) and a GPU LRU cache (bounded by `adapter_cache_max`). On a cache miss, the engine does an H2D transfer and evicts the LRU GPU resident. This mirrors how OS page tables separate virtual from physical address space.

**In-place layer surgery**
Rather than wrapping the model externally, `model_loader.py` walks `model.named_modules()` and replaces target `nn.Linear` instances with `LoRALinear` wrappers in-place. The model graph is unaware of the change — the same `generate()` call activates the LoRA path transparently.

**Adapter metadata and system prompts**
Each adapter ships with `adapter_info.json` containing a system prompt and description. The `WeightManager` stores this alongside weights; the `/infer` route looks it up and injects the system prompt into the chat template automatically. This makes adapter style changes reproducible without client-side configuration.

**Dtype-aware H2D transfer**
PEFT serialises adapter weights as `float32`. Base models often load as `float16`. The weight manager casts on the way to the GPU (`A.to(device, dtype=layer.base.weight.dtype)`), making loading from Hub, disk, or random initialisation all dtype-safe.

**Heterogeneous scatter-gather**
`LoRALinear.forward()` has two paths. In single-adapter mode it applies one delta to the full input. In batch mode, `BatchContext` supplies a position map (`adapter_id → [batch indices]`); the layer gathers each adapter's rows, computes the delta, and scatters back — all adapters resolved in one forward pass with no repeated base-model compute.

---

## API Reference

| Method | Endpoint | Body / Params |
|--------|----------|------|
| `GET` | `/health` | — |
| `POST` | `/api/v1/infer` | `{prompt, adapter_id}` |
| `GET` | `/api/v1/adapters/cached` | — |
| `GET` | `/api/v1/adapters/list` | — (includes description + system_prompt) |
| `POST` | `/api/v1/adapters/load-from-hub` | `{adapter_id, hub_repo_id}` |
| `POST` | `/api/v1/adapters/load-from-dir` | `{adapter_id, path}` |
| `POST` | `/api/v1/adapters/register-random` | `?adapter_id=<id>` |
| `POST` | `/api/v1/batch-infer` | `{requests: [{prompt, adapter_id}], max_new_tokens}` |

---

## Tests

```bash
pytest tests/ -v
```

10 tests, all passing:

- `test_lora_layer.py` — verifies LoRA math (`h = xW₀ + xBA`), passthrough without adapter, clean unload
- `test_weight_manager.py` — verifies LRU eviction policy, adapter activation, cache state cleanup
- `test_hetero_batcher.py` — verifies scatter-gather routing, per-adapter delta correctness, mode isolation
