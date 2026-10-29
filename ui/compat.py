"""Streamlit version compatibility: pick ``width='stretch'`` or ``use_container_width=True`` depending on the installed version."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict


def stretch(fn: Callable) -> Dict[str, Any]:
    """Keyword arguments that make a Streamlit element fill its container, whatever the Streamlit version."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if "width" in params:
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}
