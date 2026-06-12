import json
from pathlib import Path

import pandas as pd

from risk_window import (
    FRONT,
    HISTORY_OUT,
    REGIMES,
    THRESHOLD,
    WINDOW,
    generate_signals,
    load_df,
    segment_outcomes,
    since_summary,
)


def main() -> None:
    """Generate only the structural memory file used by memorie.html.

    This script is intentionally separate from the risk-window card.
    It writes only btc-swing-strategy/risk_window_history.json.
    """
    df = generate_signals(load_df())
    history = segment_outcomes(df)
    summary_since = since_summary(history)

    payload = {
        "title": "RISK_WINDOW_HISTORY",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "regimes": sorted(REGIMES),
        "window_days": WINDOW,
        "major_drawdown_threshold": THRESHOLD,
        "since_2025_12_summary": summary_since,
        "history": history,
    }

    FRONT.mkdir(parents=True, exist_ok=True)
    HISTORY_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[OK] Scris {HISTORY_OUT}")


if __name__ == "__main__":
    main()
