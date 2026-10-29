"""Optional end-to-end smoke test of the Streamlit app (needs streamlit >= 1.28 for streamlit.testing)."""

import os


def test_app_starts_without_exceptions_for_each_modality():
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:                      # streamlit not installed / too old
        return
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    at = AppTest.from_file(os.path.join(root, "app.py"), default_timeout=180)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    for name in ("X-ray", "MRI", "Ultrasound", "Other", "CT"):
        at.radio(key="modality_sel").set_value(name).run()
        assert not at.exception, (name, [str(e.value) for e in at.exception])
