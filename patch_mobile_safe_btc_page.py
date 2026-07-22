from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HTML = ROOT / "btc-swing-strategy" / "mecanism.html"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        print(f"[mobile-safe] {label}: already absent")
        return text
    print(f"[mobile-safe] {label}: replaced")
    return text.replace(old, new, 1)


def main() -> None:
    text = HTML.read_text(encoding="utf-8")
    original = text

    # 1) Remove Plotly from the mobile page. The old chart is no longer a core UI
    # surface, and Plotly is one of the heaviest runtime costs in Android WebView.
    text = re.sub(
        r'\n\s*<script\s+src="https://cdn\.plot\.ly/plotly-[^"]+\.min\.js"></script>\s*',
        "\n",
        text,
        count=1,
    )

    # 2) Stop OHLC/chart polling. Keep the old functions in the file for audit/history,
    # but do not execute them on page boot.
    text = replace_once(text, "      loadOHLC();\n", "      // Mobile-safe: OHLC/Plotly chart disabled.\n", "disable loadOHLC boot call")
    text = replace_once(text, "      setInterval(loadOHLC, 60 * 1000);\n", "      // Mobile-safe: OHLC polling disabled.\n", "disable loadOHLC interval")

    # 4) Clean the daily card markup. The renderer now owns this area and displays
    # only the audited signal card + legend. Remove stale interpretation markup,
    # duplicated toggle ids, and full interpretation placeholders.
    clean_daily_block = '''    <!-- DAILY_COHESIV_INTERPRETATION_CARD_START -->
    <div class="card card-secondary" id="daily-ai-card">
      <div class="card-inner" id="daily-cylinder-root">
        Se încarcă semnalul auditat...
      </div>
    </div>
    <!-- DAILY_COHESIV_INTERPRETATION_CARD_END -->'''

    text, n = re.subn(
        r'    <!-- DAILY_COHESIV_INTERPRETATION_CARD_START -->.*?    <!-- DAILY_COHESIV_INTERPRETATION_CARD_END -->',
        clean_daily_block,
        text,
        count=1,
        flags=re.S,
    )
    print(f"[mobile-safe] daily card: cleaned blocks={n}")

    if text == original:
        print("[mobile-safe] no changes needed")
    else:
        HTML.write_text(text, encoding="utf-8")
        print(f"[mobile-safe] patched {HTML}")


if __name__ == "__main__":
    main()
