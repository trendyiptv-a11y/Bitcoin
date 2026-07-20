from __future__ import annotations

import math
from typing import Any, Dict, Optional


def _safe_float(value: Any) -> Optional[float]:
    try:
        v = float(value)
        if math.isfinite(v):
            return v
    except Exception:
        return None
    return None


def _text(value: Any, fallback: str = "n/a") -> str:
    if value is None:
        return fallback
    text = str(value).