"""Dashboard integration test using Streamlit's headless AppTest (needs data + trained models)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(not (ROOT / "data" / "processed" / "france_2019_hourly.csv").exists()
                                or not list((ROOT / "models").glob("ppo_seed*/best_model.zip")),
                                reason="needs prepared data and trained models")

PAGES = ["Overview", "Simulation explorer", "Policy comparison", "Robustness lab", "Research & methodology"]


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_exception(page):
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"{page}: {at.exception}"
    assert len(at.markdown) > 0


def test_overview_kpis_come_from_live_simulation():
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=180)
    at.run()
    html = " ".join(m.value for m in at.markdown)
    assert "Electricity cost" in html and "Production completed" in html and "Renewable contribution" in html


def test_robustness_live_run_button_works():
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value("Robustness lab").run()
    btn = next(b for b in at.button if b.label == "Run live evaluation")
    btn.click().run()
    assert not at.exception, at.exception
    assert len(at.dataframe) >= 1
