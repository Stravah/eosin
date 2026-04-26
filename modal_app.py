from __future__ import annotations

import os
import subprocess
import time
import urllib.request
from pathlib import Path

import modal
from dotenv import load_dotenv

load_dotenv()

APP_NAME = os.getenv("EOSIN_MODAL_APP_NAME", "eosin-glm-ocr")
WEB_LABEL = os.getenv("EOSIN_MODAL_WEB_LABEL", "bank-parser")
GPU_TYPE = os.getenv("EOSIN_MODAL_GPU", "A100-80GB")
MAX_CONTAINERS = int(os.getenv("EOSIN_MODAL_MAX_CONTAINERS", "3"))
MAX_INPUTS = int(os.getenv("EOSIN_MODAL_MAX_INPUTS", "32"))
TARGET_INPUTS = int(os.getenv("EOSIN_MODAL_TARGET_INPUTS", str(MAX_INPUTS)))
SCALEDOWN_WINDOW_SECONDS = int(os.getenv("EOSIN_MODAL_SCALEDOWN_WINDOW", "300"))
FUNCTION_TIMEOUT_SECONDS = int(os.getenv("EOSIN_MODAL_TIMEOUT", "1800"))
MEMORY_MIB = int(os.getenv("EOSIN_MODAL_MEMORY_MIB", "40960"))
VLLM_PORT = int(os.getenv("EOSIN_MODAL_VLLM_PORT", "8000"))
VLLM_MODEL = os.getenv("EOSIN_MODAL_VLLM_MODEL", "zai-org/GLM-OCR")
VLLM_SERVED_MODEL_NAME = os.getenv("EOSIN_MODAL_SERVED_MODEL_NAME", "default")
VLLM_GPU_MEMORY_UTILIZATION = os.getenv("EOSIN_MODAL_VLLM_GPU_MEMORY_UTILIZATION", "0.90")
VLLM_MAX_MODEL_LEN = os.getenv("EOSIN_MODAL_VLLM_MAX_MODEL_LEN", "22480")
VLLM_MAX_NUM_SEQS = os.getenv("EOSIN_MODAL_VLLM_MAX_NUM_SEQS", "192")
VLLM_MAX_BATCHED_TOKENS = os.getenv("EOSIN_MODAL_VLLM_MAX_BATCHED_TOKENS", "32768")
VLLM_SPECULATIVE_CONFIG = os.getenv(
    "EOSIN_MODAL_VLLM_SPECULATIVE_CONFIG",
    '{"method": "mtp", "num_speculative_tokens": 1}',
)
HF_CACHE_VOLUME_NAME = os.getenv("EOSIN_MODAL_HF_CACHE_VOLUME", "eosin-hf-cache")
HF_CACHE_PATH = "/root/.cache/huggingface"
VLLM_HEALTH_URL = f"http://127.0.0.1:{VLLM_PORT}/health"

app = modal.App(APP_NAME)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(
        "vllm/vllm-openai:nightly",
        setup_dockerfile_commands=["ENTRYPOINT []"],
    )
    .run_commands(
        "apt-get update && apt-get install -y git ghostscript python3-tk && rm -rf /var/lib/apt/lists/*",
        "ln -sf $(command -v python3) /usr/local/bin/python",
        "pip install --upgrade pip",
        "pip install --ignore-installed blinker 'glmocr[selfhosted,server]' pandas beautifulsoup4",
        "pip install accelerate beautifulsoup4 fastapi numpy opencv-python-headless pandas pillow portalocker pydantic pymupdf python-dotenv python-multipart pyyaml requests sentencepiece tqdm uvicorn eliot 'camelot-py[cv]' liteparse img2table",
        "pip install --no-cache-dir --ignore-installed 'huggingface-hub==1.4.1' 'transformers==5.3.0'",
        "python -c \"from importlib.metadata import version; import transformers, huggingface_hub; print('transformers', transformers.__version__); print('dist-version', version('transformers')); print('huggingface-hub', huggingface_hub.__version__); print('hub-dist-version', version('huggingface-hub'))\"",
    )
    .add_local_dir("eosin", remote_path="/root/eosin", copy=True)
    .entrypoint([])
    .env(
        {
            "PYTHONPATH": "/root",
            "HF_HOME": HF_CACHE_PATH,
            "LD_LIBRARY_PATH": "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/lib",
            "GLMOCR_MODE": "selfhosted",
            "GLMOCR_OCR_API_HOST": "127.0.0.1",
            "GLMOCR_OCR_API_PORT": str(VLLM_PORT),
            "GLMOCR_OCR_MODEL": VLLM_SERVED_MODEL_NAME,
            "BANK_PARSER_SAVE_DEBUG_IMAGES": "false",
            "BANK_PARSER_PARSE_TESTING": "false",
            "BANK_PARSER_ENABLE_OCR_BATCHING": "false",
            "BANK_PARSER_LAYOUT_MODE": "auto",
            "BANK_PARSER_LAYOUT_MAX_CONCURRENCY": "1",
            "BANK_PARSER_OCR_PIPELINE_WORKERS": "128",
            "BANK_PARSER_OCR_PIPELINE_QUEUE_SIZE": "2048",
            "BANK_PARSER_BACKEND_STARTUP_TIMEOUT": "60",
            "BANK_PARSER_BACKEND_RETRY_INTERVAL": "1",
        }
    )
)


def _wait_for_http_health(url: str, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # pragma: no cover - startup retry path
            last_error = exc
            time.sleep(1)

    raise TimeoutError(f"Timed out waiting for backend health at {url}: {last_error}")


def _terminate_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive cleanup
        process.kill()
        process.wait(timeout=10)


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    cpu=12.0,
    memory=MEMORY_MIB,
    max_containers=MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    timeout=FUNCTION_TIMEOUT_SECONDS,
    volumes={HF_CACHE_PATH: hf_cache_volume},
)
@modal.concurrent(max_inputs=MAX_INPUTS, target_inputs=TARGET_INPUTS)
class BankParserModalApp:
    @modal.enter()
    def enter(self) -> None:
        from eosin.backend.bank_parser_api import create_app
        from eosin.backend.bank_parser_service import BankParserService
        import eosin.backend.bank_parser_service as bank_parser_service_module

        config_path = Path(bank_parser_service_module.__file__).resolve().parent / "config.modal.yaml"
        os.environ["BANK_PARSER_CONFIG"] = str(config_path)

        self.vllm_process = subprocess.Popen(
            [
                "vllm",
                "serve",
                VLLM_MODEL,
                "--port",
                str(VLLM_PORT),
                "--dtype",
                "bfloat16",
                "--trust-remote-code",
                "--speculative-config",
                VLLM_SPECULATIVE_CONFIG,
                "--max-model-len",
                VLLM_MAX_MODEL_LEN,
                "--gpu-memory-utilization",
                VLLM_GPU_MEMORY_UTILIZATION,
                "--max-num-seqs",
                VLLM_MAX_NUM_SEQS,
                "--max-num-batched-tokens",
                VLLM_MAX_BATCHED_TOKENS,
                "--async-scheduling",
                "--enable-chunked-prefill",
                "--served-model-name",
                VLLM_SERVED_MODEL_NAME,
            ],
            text=True,
        )
        _wait_for_http_health(VLLM_HEALTH_URL, timeout_seconds=300)

        self.service = BankParserService(
            config_path=str(config_path),
            save_debug_images=False,
            parse_testing=False,
            enable_ocr_batching=False,
            layout_max_concurrency=1,
            ocr_pipeline_workers=128,
            ocr_pipeline_queue_size=2048,
            backend_startup_timeout=60.0,
            backend_retry_interval=1.0,
        )
        self.web_app = create_app(service=self.service)

    @modal.exit()
    def exit(self) -> None:
        service = getattr(self, "service", None)
        if service is not None:
            service.close()

        _terminate_process(getattr(self, "vllm_process", None))

    @modal.asgi_app(label=WEB_LABEL)
    def web(self):
        return self.web_app
