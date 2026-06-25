#!/usr/bin/env python
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SIGNAL_PATH = ROOT / "btc-swing-strategy" / "profit_watch_signal.json"
DEFAULT_STALE_AFTER_MINUTES = 45.0


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def write_github_output(result: dict[str, Any]) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    reason = str(result.get("reason") or "")[:200].replace("\n", " ")
    with open(output_path, "a", encoding="utf-8") as fh:
        fh.write(f"trigger_profit_watcher={'true' if result.get('trigger_profit_watcher') else 'false'}\n")
        fh.write(f"watcher_age_minutes={result.get('watcher_age_minutes') or 0:.2f}\n")
        fh.write(f"reason={reason}\n")


def main() -> None:
    stale_after = float(os.getenv("WATCHDOG_STALE_AFTER_MINUTES") or DEFAULT_STALE_AFTER_MINUTES)
    signal = load_json(SIGNAL_PATH, {})
    checked_at_raw = str(signal.get("checked_at") or "")
    checked_at = parse_dt(checked_at_raw)

    result: dict[str, Any] = {
        "checked_at": now_utc().isoformat(),
        "profit_watcher_checked_at": checked_at_raw or None,
        "stale_after_minutes": stale_after,
        "trigger_profit_watcher": False,
        "reason": "Profit watcher is fresh enough.",
    }

    if not checked_at:
        result["watcher_age_minutes"] = None
        result["trigger_profit_watcher"] = True
        result["reason"] = "Missing or invalid profit watcher checked_at; triggering watcher."
    else:
        age_minutes = max(0.0, (now_utc() - checked_at).total_seconds() / 60.0)
        result["watcher_age_minutes"] = age_minutes
        if age_minutes >= stale_after:
            result["trigger_profit_watcher"] = True
            result["reason"] = f"Profit watcher is stale: {age_minutes:.1f} minutes >= {stale_after:.1f}."
        else:
            result["reason"] = f"Profit watcher is fresh: {age_minutes:.1f} minutes < {stale_after:.1f}."

    write_github_output(result)
    print("Profit watchdog trigger:", result["trigger_profit_watcher"])
    print("Reason:", result["reason"])
    print("Watcher checked_at:", result.get("profit_watcher_checked_at"))
    print("Watcher age minutes:", result.get("watcher_age_minutes"))


if __name__ == "__main__":
    main()
