# Eosin GPU Backend

This repo is the standalone server-side backend for the GLM OCR bank parser, with Modal as the primary serverless deployment target.

## Modal Deployment

1. Install and authenticate the Modal CLI:

```bash
pip install modal
modal setup
```

2. Deploy the web endpoint:

```bash
modal deploy modal_app.py
```

The deployed Modal app runs:

- one GPU-backed container pool for the parser API
- a local `vllm serve` subprocess inside each container
- `gpu="A100-80GB"` with `cpu=12` and `memory=16384`
- `max_containers=4`
- `@modal.concurrent(max_inputs=15, target_inputs=15)` so each warm container can handle up to 15 requests
- `scaledown_window=300` so idle containers stay warm for 5 minutes

The Modal endpoint is exposed by [modal_app.py](/home/nol/Documents/work/eosin/modal_app.py).

## Repo Layout

- `modal_app.py`: Modal app definition and web endpoint
- `eosin/backend/bank_parser_api.py`: FastAPI app
- `eosin/backend/bank_parser_service.py`: parser lifecycle wrapper
- `eosin/backend/eosin_pipeline.py`: PDF parsing and OCR pipeline
- `eosin/backend/config.yaml`: Docker/local config pointing to a sibling vLLM service
- `eosin/backend/config.modal.yaml`: Modal config pointing to the local in-container vLLM subprocess
- `docker-compose.vllm.yml`: legacy two-container Docker deployment

## Integration

Point the caller at the deployed parser URL:

```bash
export EOSIN_PARSER_BASE_URL=https://<your-modal-endpoint>.modal.run
```

## Notes

- Modal uses `@modal.concurrent(...)`; the older `allow_concurrent_inputs` setting has been replaced in current Modal SDKs.
- Hugging Face cache is mounted from a persisted Modal Volume named `eosin-hf-cache`.
