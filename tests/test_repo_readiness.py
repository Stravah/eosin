from __future__ import annotations

from pathlib import Path

def test_env_example_has_modal_defaults() -> None:
    env_example = Path(".env.example").read_text()

    assert "EOSIN_MODAL_GPU=A100-80GB" in env_example
    assert "EOSIN_MODAL_MAX_CONTAINERS=3" in env_example
    assert "EOSIN_MODAL_MAX_INPUTS=64" in env_example
    assert "EOSIN_MODAL_MEMORY_MIB=40960" in env_example
    assert "EOSIN_MODAL_VLLM_MAX_BATCHED_TOKENS=32768" in env_example


def test_local_artifacts_are_ignored() -> None:
    gitignore = Path(".gitignore").read_text()

    assert "/bank statements" in gitignore
    assert "/load-test-results/" in gitignore
    assert ".env" in gitignore
    assert "!.env.example" in gitignore


def test_provider_source_has_modal_url_handling_without_hardcoded_endpoint() -> None:
    provider_source = Path("eosin.py").read_text()

    assert 'DEFAULT_MODAL_URL = ""' in provider_source
    assert 'parsed.netloc.endswith(".modal.run")' in provider_source
    assert 'os.getenv("BANK_PARSER_PORT", "8090")' in provider_source
    assert "noelalex" not in provider_source
