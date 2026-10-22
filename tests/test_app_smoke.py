"""Smoke tests for application entry points, multi-page routing, exceptions, and logging."""

import ast
from pathlib import Path

from src.core.exceptions import (
    AgentOrchestrationError,
    AnalyticalExecutionError,
    ConfigurationError,
    DataAnalystError,
    DataIngestionError,
    DataProfilingError,
    EvidenceValidationError,
    ReportGenerationError,
)
from src.core.logging_config import get_logger, setup_logging


def test_package_structure_and_imports():
    """Verify all architectural packages and submodules can be imported cleanly."""
    import src
    import src.agent
    import src.analytics
    import src.config
    import src.core
    import src.ingestion
    import src.interpretation
    import src.models
    import src.profiling
    import src.reporting
    import src.security

    assert src.__version__ == "0.1.0"
    assert hasattr(src.config, "get_settings")
    assert hasattr(src.models, "DatasetMetadata")
    assert hasattr(src.core, "init_session_state")


def test_page_files_syntax():
    """Verify app.py and all 7 page files exist and are valid Python syntax."""
    root = Path(__file__).resolve().parent.parent
    app_file = root / "app.py"
    assert app_file.exists(), "app.py entry point missing"

    ast.parse(app_file.read_text(), filename="app.py")

    pages_dir = root / "pages"
    assert pages_dir.exists(), "pages directory missing"

    expected_pages = [
        "1_Executive_Overview.py",
        "2_Data_Quality.py",
        "3_Exploration_Studio.py",
        "4_Statistical_Intelligence.py",
        "5_AI_Analyst.py",
        "6_Predictive_Lab.py",
        "7_Intelligence_Reports.py",
    ]

    for page_name in expected_pages:
        page_path = pages_dir / page_name
        assert page_path.exists(), f"Page {page_name} does not exist"
        content = page_path.read_text()
        # Verify valid syntax without syntax errors
        ast.parse(content, filename=page_name)


def test_exception_hierarchy_and_user_messages():
    """Verify application exceptions preserve user-facing message separation."""
    err = DataIngestionError(
        message="Corrupted parquet header at offset 0x004",
        user_message="Unable to parse the uploaded file.",
    )

    assert isinstance(err, DataAnalystError)
    assert err.message == "Corrupted parquet header at offset 0x004"
    assert err.user_message == "Unable to parse the uploaded file."
    assert str(err) == "Corrupted parquet header at offset 0x004"

    # Default user message fallback
    conf_err = ConfigurationError(message="Missing config key")
    assert "Invalid application configuration" in conf_err.user_message

    # Other exception types
    assert issubclass(DataProfilingError, DataAnalystError)
    assert issubclass(AnalyticalExecutionError, DataAnalystError)
    assert issubclass(AgentOrchestrationError, DataAnalystError)
    assert issubclass(EvidenceValidationError, DataAnalystError)
    assert issubclass(ReportGenerationError, DataAnalystError)


def test_logging_setup_and_deduplication():
    """Verify logging configuration does not create duplicate handlers on repeated setup."""
    logger1 = setup_logging("DEBUG")
    assert logger1.level == 10  # DEBUG
    initial_handler_count = len(logger1.handlers)

    # Calling setup_logging again should not add duplicate handlers
    logger2 = setup_logging("INFO")
    assert len(logger2.handlers) == initial_handler_count
    assert logger2.level == 20  # INFO

    child = get_logger("analytics.engine")
    assert child.name == "da_agent.analytics.engine"


def test_streamlit_app_smoke():
    """Run Streamlit headless AppTest on app.py and all pages to verify rendering without exceptions."""
    try:
        from streamlit.testing.v1 import AppTest

        root = Path(__file__).resolve().parent.parent
        app_path = root / "app.py"
        at = AppTest.from_file(str(app_path), default_timeout=10)
        at.run()
        assert not at.exception, f"app.py raised: {at.exception}"

        pages = [
            "pages/1_Executive_Overview.py",
            "pages/2_Data_Quality.py",
            "pages/3_Exploration_Studio.py",
            "pages/4_Statistical_Intelligence.py",
            "pages/5_AI_Analyst.py",
            "pages/6_Predictive_Lab.py",
            "pages/7_Intelligence_Reports.py",
        ]
        for page in pages:
            page_at = at.switch_page(page)
            page_at.run()
            assert not page_at.exception, f"{page} raised: {page_at.exception}"
    except ImportError:
        pass
