#!/data/data/com.termux/files/usr/bin/bash
			
cd ~/Bitcoin || exit 1

if [ -f .env ]; then
  source .env
fi

echo "==============================" >> btc-swing-strategy/termux_15m_runner.log
echo "Run started at $(date)" >> btc-swing-strategy/termux_15m_runner.log
											                      								
python bitget_safe_client.py >> btc-swing-strategy/termux_15m_runner.log 2>&1
python paper_decision_audit.py >> btc-swing-strategy/termux_15m_runner.log 2>&1
python bitget_dry_run_bridge.py >> btc-swing-strategy/termux_15m_runner.log 2>&1
python bitget_micro_live_executor.py >> btc-swing-strategy/termux_15m_runner.log 2>&1
python bitget_spot_order_endpoint_validator.py >> btc-swing-strategy/termux_15m_runner.log 2>&1
python bitget_micro_live_real_executor.py >> btc-swing-strategy/termux_15m_runner.log 2>&1
# python paper_trader_adaptive.py >> btc-swing-strategy/termux_15m_runner.log 2>&1
# python publish_public_status.py  # disabled: local-only bot mode >> btc-swing-strategy/termux_15m_runner.log 2>&1
# ./push_public_status_hourly.sh  # disabled: local-only bot mode >> btc-swing-strategy/termux_15m_runner.log 2>&1
echo "Run completed at $(date)" >> btc-swing-strategy/termux_15m_runner.log

