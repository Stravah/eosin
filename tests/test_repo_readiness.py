from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

load_test_spec = importlib.util.spec_from_file_location(
    "load_test_bank_parser",
    Path("scripts/load_test_bank_parser.py"),
)
assert load_test_spec is not None
load_test_module = importlib.util.module_from_spec(load_test_spec)
assert load_test_spec.loader is not None
sys.modules[load_test_spec.name] = load_test_module
load_test_spec.loader.exec_module(load_test_module)


def test_env_example_has_modal_defaults() -> None:
    env_example = Path(".env.example").read_text()

    assert "EOSIN_MODAL_GPU=A100-80GB" in env_example
    assert "EOSIN_MODAL_MAX_CONTAINERS=1" in env_example
    assert "EOSIN_MODAL_MAX_INPUTS=32" in env_example
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


def test_load_test_dataframe_markdown_escapes_cells() -> None:
    dataframe = pd.DataFrame([{"Description": "A | B", "Amount": 10}])

    markdown = load_test_module.dataframe_to_markdown(dataframe)

    assert "| Description | Amount |" in markdown
    assert "A \\| B" in markdown
