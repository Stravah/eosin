import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests

try:
    from eliot import log_call
except ModuleNotFoundError:  # pragma: no cover - optional in caller repo
    def log_call(*_args, **_kwargs):
        def decorator(func):
            return func
        return decorator

try:
    from app.thesmos.exceptions import ProviderFailedToDig
    from app.thesmos.providers.base import Provider
except ModuleNotFoundError:  # pragma: no cover - standalone fallback
    class Provider:
        def extract(self, file_path: Path) -> pd.DataFrame:
            raise NotImplementedError

    class ProviderFailedToDig(Exception):
        pass

logger = logging.getLogger(__name__)
DEFAULT_MODAL_URL = "https://bank-parser--eosin-glm-ocr.modal.run"


def _payload_to_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    columns = payload.get("columns", [])
    rows = payload.get("rows", [])
    if columns:
        return pd.DataFrame(rows, columns=columns)
    return pd.DataFrame(rows)


class EosinPDFProvider(Provider):
    """Extract tables through the remote GLM bank-parser service."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: int = 600,
        session: requests.Session | None = None,
        bearer_token: str | None = None,
    ) -> None:
        raw_url = (
            base_url
            or os.getenv("EOSIN_PARSER_BASE_URL")
            or os.getenv("EOSIN_MODAL_ENDPOINT")
            or DEFAULT_MODAL_URL
        ).rstrip("/")
        if not raw_url.startswith(("http://", "https://")):
            raw_url = f"https://{raw_url}"

        parsed = urlparse(raw_url)
        if parsed.port is None and parsed.netloc.endswith(".modal.run"):
            self.base_url = raw_url
        elif parsed.port is None:
            port = os.getenv("BANK_PARSER_PORT", "8090")
            raw_url = f"{parsed.scheme}://{parsed.hostname}:{port}{parsed.path}"
            self.base_url = raw_url
        else:
            self.base_url = raw_url

        self.timeout = timeout
        self.session = session if session is not None else requests.Session()
        self.bearer_token = (
            bearer_token
            or os.getenv("EOSIN_PARSER_BEARER_TOKEN")
            or os.getenv("EOSIN_MODAL_BEARER_TOKEN")
        )

    def _request_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        return headers

    @log_call(include_args=[], include_result=False)
    def extract(self, file_path: Path) -> pd.DataFrame:
        pdf_path = Path(file_path)
        logger.info("EosinPDFProvider: sending %s to %s", pdf_path.name, self.base_url)

        try:
            with pdf_path.open("rb") as handle:
                response = self.session.post(
                    f"{self.base_url}/parse/bank-statement",
                    files={"file": (pdf_path.name, handle, "application/pdf")},
                    headers=self._request_headers(),
                    timeout=self.timeout,
                )

            response.raise_for_status()
            payload = response.json()
            dataframe = _payload_to_dataframe(payload)
            if dataframe.empty:
                raise ProviderFailedToDig(f"EosinPDFProvider: No table data extracted from {pdf_path.name}")
            return dataframe
        except ProviderFailedToDig:
            raise
        except Exception as exc:
            logger.error("EosinPDFProvider failed for %s: %s", pdf_path.name, exc, exc_info=True)
            raise ProviderFailedToDig(
                f"Eosin (remote GLM parser) failed to extract {pdf_path.name}: {exc}"
            ) from exc
