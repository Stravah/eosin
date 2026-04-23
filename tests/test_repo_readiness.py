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


def test_readme_does_not_include_hardcoded_modal_endpoint() -> None:
    readme = Path("README.md").read_text()

    assert "noelalex" not in readme
    assert "https://<workspace>--bank-parser.modal.run" in readme
