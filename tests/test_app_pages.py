"""The four analysis pages of the app (quality, EDA, statistics, benchmark) run headlessly without an exception.

They read only small files committed in ``reports/``; when one is missing the page must say which script
to run instead of crashing.  The detection page is covered by ``test_app_integration.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
PAGES = ["page_quality", "page_eda", "page_stats", "page_benchmark"]


def _run(module: str) -> AppTest:
    script = f"import sys\nsys.path.insert(0, {str(APP_DIR.as_posix())!r})\nimport {module}\n{module}.render()\n"
    app = AppTest.from_string(script, default_timeout=180)
    app.run()
    return app


@pytest.mark.parametrize("module", PAGES)
def test_page_runs_without_exception(module):
    app = _run(module)
    assert not app.exception, [e.value for e in app.exception]
    assert len(app.title) == 1

