# Eosin GPU Backend

Standalone GPU backend for GLM-OCR bank-statement parsing. The primary deployment target is Modal: each warm container runs a FastAPI parser API and a local `vllm serve` process on an A100 80GB GPU.

## What Runs

- `modal_app.py`: Modal app, image build, vLLM process startup, ASGI endpoint.
- `eosin/backend/bank_parser_api.py`: FastAPI routes for parsing and health checks.
- `eosin/backend/bank_parser_service.py`: parser lifecycle wrapper.
- `eosin/backend/eosin_pipeline.py`: PDF preprocessing, layout detection, table stitching, OCR fanout, and result shaping.
- `eosin/backend/ocr_pipeline.py`: shared OCR work queue that keeps vLLM fed after each PDF finishes preprocessing.
- `eosin.py`: caller-side provider wrapper for the Almond/thesmos integration.
- `scripts/load_test_bank_parser.py`: concurrent PDF load tester with JSONL, CSV, and summary reports.

## Setup

Create a local virtualenv and install the project dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Authenticate Modal once on the machine that deploys this service:

```bash
modal setup
```

Create local config from the template:

```bash
cp .env.example .env
```

Edit `.env` for your deployment. `.env` is intentionally ignored by git; `.env.example` is the committed reference.

## Modal Deploy

Deploy the API:

```bash
modal deploy modal_app.py
```

Check the deployed health endpoint:

```bash
curl -sS --max-time 180 "$EOSIN_PARSER_BASE_URL/health"
```

The default production shape in `.env.example` is:

- GPU: `A100-80GB`
- CPU: `12`
- Memory: `40960` MiB
- Modal max containers: `3`
- Modal concurrent inputs per container: `64`
- Idle scaledown window: `300` seconds
- vLLM model: `zai-org/GLM-OCR`
- vLLM max model length: `22480`
- vLLM max sequences: `192`
- vLLM max batched tokens: `32768`
- vLLM GPU memory utilization: `0.90`
- vLLM speculative config: `{"method": "mtp", "num_speculative_tokens": 1}`

Modal uses `@modal.concurrent(max_inputs=..., target_inputs=...)`; older `allow_concurrent_inputs` examples are not used by the current SDK.

## Provider Integration

In the caller project, configure the provider endpoint and use `EosinPDFProvider` through the same provider interface:

```bash
export EOSIN_PARSER_BASE_URL="https://<workspace>--bank-parser.modal.run"
```

Authentication is intentionally not enforced yet. Add it before exposing the endpoint beyond trusted callers.

## Load Testing

Put local PDFs under `bank statements/` using any nested folder layout. That directory is ignored and must not be committed.

Run a small validation:

```bash
python scripts/load_test_bank_parser.py \
  --endpoint "$EOSIN_PARSER_BASE_URL" \
  --corpus-dir "bank statements" \
  --target-requests 10 \
  --concurrency 10 \
  --timeout 1800
```

Run a heavier sweep:

```bash
python scripts/load_test_bank_parser.py \
  --endpoint "$EOSIN_PARSER_BASE_URL" \
  --corpus-dir "bank statements" \
  --target-requests 250 \
  --concurrency 64 \
  --timeout 1800
```

Reports are written to `load-test-results/<timestamp>/` and include:

- request-level JSONL and CSV logs
- success and failure counts
- latency percentiles
- throughput
- PDF counts and uploaded byte counts
- server-side parser timings
- OCR queue wait and vLLM request timing summaries

`load-test-results/` is ignored so previous benchmark logs stay local.

## Local Docker

The Docker Compose path is kept for non-Modal GPU hosts:

```bash
docker compose -f docker-compose.vllm.yml up --build
```

The local API is expected at:

```bash
curl -sS http://localhost:8090/health
```

## Operational Notes

- Each PDF completes preprocessing atomically before OCR starts for that PDF.
- Layout detection is intentionally bounded to one concurrent layout job.
- OCR work is queued after preprocessing and vLLM handles batching/concurrency.
- Hugging Face cache is persisted in the Modal volume `eosin-hf-cache`.
- `BANK_PARSER_ENABLE_OCR_BATCHING=false` because batching is delegated to vLLM.

## Observability TODO

Add production telemetry before long-running production load tests. The next useful layer is OpenTelemetry or Prometheus-compatible metrics for parser stages, vLLM queue depth, KV-cache usage, multimodal timing, GPU utilization, and per-container admission pressure, with dashboards in Grafana or an equivalent metrics backend.
