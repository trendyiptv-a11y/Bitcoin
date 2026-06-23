#!/usr/bin/env python
from __future__ import annotations

"""
Safe UI patch step for mecanism.html.

The structural confirmation card is currently injected by
btc-swing-strategy/guide-nav-i18n.js so it can be previewed and updated
without rewriting the large mecanism.html file.

This script remains in the pipeline for compatibility, but it no longer
forces CSS/HTML/JS insertion into mecanism.html. That prevents workflow
failures when the HTML anchors change.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "btc-swing-strategy" / "mecanism.html"


def main() -> None:
    if not HTML_PATH.exists():
        raise FileNotFoundError(f"Nu găsesc {HTML_PATH}")

    print("[OK] patch_mecanism_model_price_ui.py: skip direct mecanism.html structural patch.")
    print("[OK] Confirmarea structurală este injectată prin btc-swing-strategy/guide-nav-i18n.js.")
    print("[OK] Datele vin din coeziv_state.json -> state.structural_confirmation.")


if __name__ == "__main__":
    main()
