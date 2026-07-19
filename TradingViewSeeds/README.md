# CohesivX TradingView seed export

GitHub Actions updates root CSV seed symbols for TradingView `request.seed()`.

Mapping per CSV row:
- `open` = central structural projection
- `high` = high/statistical projection
- `low` = low/defense projection
- `close` = projected standard miner cost

Use `Pine/indicator_seed_request.txt` as the experimental auto-data Pine script.
The normal `Pine/indicator.txt` remains the stable hardcoded fallback.
