from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

import eosin.backend.eosin_pipeline as impl
from eosin.backend.bank_parser_service import BankParserService


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _optional_env_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def create_app(service: Optional[BankParserService] = None) -> FastAPI:
    app = FastAPI()
    app.state.bank_parser_service = service
    app.state.bank_parser_service_lock = threading.Lock()
    default_config_path = Path(__file__).resolve().parent / "config.yaml"

    def get_service() -> BankParserService:
        existing = app.state.bank_parser_service
        if existing is not None:
            return existing

        with app.state.bank_parser_service_lock:
            existing = app.state.bank_parser_service
            if existing is not None:
                return existing

            created = BankParserService(
                config_path=os.getenv("BANK_PARSER_CONFIG", str(default_config_path)),
                save_debug_images=_env_flag("BANK_PARSER_SAVE_DEBUG_IMAGES", impl.SAVE_DEBUG_IMAGES),
                parse_testing=_env_flag("BANK_PARSER_PARSE_TESTING", impl.PARSE_TESTING),
                enable_ocr_batching=_env_flag("BANK_PARSER_ENABLE_OCR_BATCHING", impl.ENABLE_OCR_BATCHING),
                ocr_batch_size=int(os.getenv("BANK_PARSER_OCR_BATCH_SIZE", str(impl.OCR_BATCH_SIZE))),
                pdf_render_dpi_override=_optional_env_int("BANK_PARSER_PDF_DPI"),
                layout_max_concurrency=_optional_env_int("BANK_PARSER_LAYOUT_MAX_CONCURRENCY"),
                ocr_pipeline_workers=_optional_env_int("BANK_PARSER_OCR_PIPELINE_WORKERS"),
                ocr_pipeline_queue_size=_optional_env_int("BANK_PARSER_OCR_PIPELINE_QUEUE_SIZE"),
                backend_startup_timeout=_env_float("BANK_PARSER_BACKEND_STARTUP_TIMEOUT", 180.0),
                backend_retry_interval=_env_float("BANK_PARSER_BACKEND_RETRY_INTERVAL", 5.0),
            )
            app.state.bank_parser_service = created
            return created

    @app.post("/parse/bank-statement")
    async def parse_bank_statement(file: UploadFile = File(...)):
        filename = file.filename or ""
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="uploaded file must be a PDF")

        pdf_bytes = await file.read()

        try:
            result = await run_in_threadpool(
                lambda: get_service().parse_pdf_bytes(filename, pdf_bytes).to_payload()
            )
            return result
        except TimeoutError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app(service=None)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("BANK_PARSER_PORT", "8090")),
    )
