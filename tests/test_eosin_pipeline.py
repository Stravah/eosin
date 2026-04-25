import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd


def _load_eosin_pipeline_module():
    module_name = "test_eosin_pipeline_module"
    module_path = (
        Path(__file__).resolve().parents[1] / "eosin" / "backend" / "eosin_pipeline.py"
    )

    if module_name in sys.modules:
        return sys.modules[module_name]

    glmocr_module = types.ModuleType("glmocr")
    glmocr_config = types.ModuleType("glmocr.config")
    glmocr_config.load_config = lambda *args, **kwargs: None
    glmocr_dataloader = types.ModuleType("glmocr.dataloader")
    glmocr_dataloader.PageLoader = object
    glmocr_layout = types.ModuleType("glmocr.layout")
    glmocr_layout.PPDocLayoutDetector = object
    glmocr_ocr_client = types.ModuleType("glmocr.ocr_client")
    glmocr_ocr_client.OCRClient = object
    glmocr_utils = types.ModuleType("glmocr.utils")
    glmocr_image_utils = types.ModuleType("glmocr.utils.image_utils")
    glmocr_image_utils.crop_image_region = lambda *args, **kwargs: None
    glmocr_image_utils.pdf_to_images_pil = lambda *args, **kwargs: []

    eosin_package = types.ModuleType("eosin")
    eosin_backend = types.ModuleType("eosin.backend")
    eosin_ocr_pipeline = types.ModuleType("eosin.backend.ocr_pipeline")
    eosin_ocr_pipeline.OCRPipelineDispatcher = object
    eosin_ocr_pipeline.OCRTaskResult = object

    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    sys.modules.setdefault("fitz", types.ModuleType("fitz"))
    sys.modules.setdefault("torch", types.ModuleType("torch"))
    sys.modules["glmocr"] = glmocr_module
    sys.modules["glmocr.config"] = glmocr_config
    sys.modules["glmocr.dataloader"] = glmocr_dataloader
    sys.modules["glmocr.layout"] = glmocr_layout
    sys.modules["glmocr.ocr_client"] = glmocr_ocr_client
    sys.modules["glmocr.utils"] = glmocr_utils
    sys.modules["glmocr.utils.image_utils"] = glmocr_image_utils
    sys.modules["eosin"] = eosin_package
    sys.modules["eosin.backend"] = eosin_backend
    sys.modules["eosin.backend.ocr_pipeline"] = eosin_ocr_pipeline

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_select_best_table_candidate_prefers_bank_transaction_table():
    module = _load_eosin_pipeline_module()
    html_tables = [
        """
        <table>
          <tr><th>Account Summary</th><th>Value</th></tr>
          <tr><td>Closing Balance</td><td>1000.00</td></tr>
        </table>
        """,
        """
        <table>
          <tr><th>Date</th><th>Description</th><th>Debit</th><th>Credit</th><th>Balance</th></tr>
          <tr><td>01/01/2024</td><td>ATM Withdrawal</td><td>500.00</td><td></td><td>9500.00</td></tr>
          <tr><td>02/01/2024</td><td>Salary</td><td></td><td>20000.00</td><td>29500.00</td></tr>
        </table>
        """,
    ]

    selected = module.select_best_table_candidate(html_tables)

    assert selected is not None
    index, _, dataframe, headers = selected
    assert index == 1
    assert headers == ["Date", "Description", "Debit", "Credit", "Balance"]
    assert isinstance(dataframe, pd.DataFrame)
    assert len(dataframe) == 2


def test_select_best_table_candidate_prefers_header_match_on_followup_pages():
    module = _load_eosin_pipeline_module()
    html_tables = [
        """
        <table>
          <tr><th>Summary</th><th>Total</th></tr>
          <tr><td>Debits</td><td>500.00</td></tr>
        </table>
        """,
        """
        <table>
          <tr><td>03/01/2024</td><td>UPI Payment</td><td>250.00</td><td></td><td>29250.00</td></tr>
          <tr><td>04/01/2024</td><td>POS Purchase</td><td>120.00</td><td></td><td>29130.00</td></tr>
        </table>
        """,
    ]

    selected = module.select_best_table_candidate(
        html_tables,
        expected_headers=["Date", "Description", "Debit", "Credit", "Balance"],
    )

    assert selected is not None
    index, _, dataframe, _ = selected
    assert index == 1
    assert list(dataframe.columns) == ["Date", "Description", "Debit", "Credit", "Balance"]
    assert len(dataframe) == 2
