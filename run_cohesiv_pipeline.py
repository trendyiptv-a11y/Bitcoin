#!/usr/bin/env python
# run_cohesiv_pipeline.py

"""
Pipeline Coeziv BTC-only.

Generates:
1. data/btc_state_latest.json
2. data/ic_btc_series.json
3. BTC historical 72h statistics
4. data/ic_btc_mega_latest.json
5. btc-swing-strategy/coeziv_state.json with Coeziv model_price V2
6. Injects dynamic structural confirmation and trader signal
7. Builds daily_cohesiv_interpretation.json after the mechanism is updated
8. Ensures mecanism.html displays the mechanism explanation dynamically
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print(f"\n[PIPELINE] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"[PIPELINE] Error running command: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> None:
    run([sys.executable, "update_btc_state_latest_from_daily.py"])
    run([sys.executable, "export_ic_btc_series.py"])
    run([sys.executable, "update_btc_72h_statistics.py"])
    run([sys.executable, "build_ic_btc_mega_state.py"])
    run([sys.executable, "coeziv_state_v2.py"])
    run([sys.executable, "inject_structural_confirmation.py"])
    run([sys.executable, "build_daily_cohesiv_interpretation.py"])
    run([sys.executable, "patch_mecanism_model_price_ui.py"])

    print("\n[PIPELINE] BTC Cohesiv pipeline complete.")


if __name__ == "__main__":
    main()
