FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first so the requirements.txt torch line is a no-op
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY engine/ ./engine/
COPY data/ ./data/
COPY scripts/ ./scripts/

ENV HF_HOME=/cache/huggingface

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "engine.main:app", "--host", "0.0.0.0", "--port", "8000"]
