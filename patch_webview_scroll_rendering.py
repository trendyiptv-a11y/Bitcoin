#!/usr/bin/env python3
from pathlib import Path

HTML = Path("btc-swing-strategy/mecanism.html")
MARKER_START = "/* WEBVIEW_SCROLL_RENDER_STABILITY_START */"
MARKER_END = "/* WEBVIEW_SCROLL_RENDER_STABILITY_END */"

PATCH = f"""
    {MARKER_START}
    /*
      Android WebView stability patch.
      Motiv: la scroll rapid, backdrop-filter + transform + shadow-uri mari pot produce
      artefacte GPU: pete albe/negre, flicker sau dreptunghiuri temporare.
      Păstrăm designul, dar eliminăm efectele costisitoare la randare.
    */
    .card {{
      backdrop-filter: none !important;
      -webkit-backdrop-filter: none !important;
      transition: none !important;
      transform: none !important;
      will-change: auto !important;
    }}

    .card:hover {{
      transform: none !important;
    }}

    .card,
    .card:first-of-type,
    body.light-mode .card,
    body.light-mode .card:first-of-type {{
      box-shadow: 0 14px 36px rgba(0,0,0,0.42) !important;
    }}

    .title-bar,
    .price-value,
    .asset-logo,
    .signal-chip,
    .theme-toggle-btn,
    .daily-full-toggle {{
      transition: none !important;
      will-change: auto !important;
    }}

    @media (hover: hover) and (pointer: fine) {{
      /* Desktop poate păstra hover doar dacă browserul îl randă stabil. */
      .card:hover {{
        border-color: rgba(56,189,248,0.28);
      }}
    }}
    {MARKER_END}
"""


def main() -> None:
    text = HTML.read_text(encoding="utf-8")

    if MARKER_START in text and MARKER_END in text:
        before = text.split(MARKER_START)[0]
        after = text.split(MARKER_END, 1)[1]
        text = before + PATCH + after
        print("[OK] WebView scroll rendering patch refreshed.")
    else:
        closing = "</style>"
        if closing not in text:
            raise RuntimeError("Nu am găsit </style> în mecanism.html")
        text = text.replace(closing, PATCH + "\n  " + closing, 1)
        print("[OK] WebView scroll rendering patch inserted.")

    HTML.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
