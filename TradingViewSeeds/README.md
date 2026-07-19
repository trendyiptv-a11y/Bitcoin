# CohesivX TradingView seed export

GitHub Actions updates root CSV seed symbols for TradingView `request.seed()`.

Mapping per CSV row:
- `open` = central structural projection
- `high` = high/statistical projection, exported for audit but not requested by the auto Pine template
- `low` = low/defense projection
- `close` = projected standard miner cost

`Pine/indicator_seed_request.txt` stays under TradingView's 40 request limit by reading central, low and miner only.
The normal `Pine/indicator.txt` remains the stable hardcoded fallback with optional high band.
