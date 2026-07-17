#!/usr/bin/env python3
"""Generate btc-swing-strategy/tactical_range.json.

Tactical range is a backend-only layer. The frontend should only display
BUY / SELL / WAIT and a short message. This script detects the recent BTCUSDT
range from market candles and writes a compact JSON payload.

Data source order:
1) Bitget public spot candles
2) Kraken public OHLC
