# CohesivX Engine

## Purpose

The engine layer converts the raw monitor snapshot into an auditable terminal state.

The engine does not render HTML, Pine Script, Telegram messages or API responses. It only calculates and explains.

## Input

Primary input:

```text
btc-swing-strategy/coeziv_state.json
```

Important fields:

- `price_usd`
- `model_price_usd`
- `model_price_bands`
- `production_costs_usd`
- `model_price_deviation`
- `deviation_from_production`
- `flow_score`
- `liquidity_score`
- `market_regime`
- `tradingview_anchors`

## Output

Primary terminal output:

```text
btc-swing-strategy/cohesivx_terminal_state.json
```

Expected top-level blocks:

```json
{
  "summary": {},
  "scores": {},
  "market": {},
  "sources": {},
  "miners": {},
  "structure": {},
  "reasoning": []
}
```

## Score Definitions

### Structural Score

Measures how structurally coherent the market is relative to CohesivX central value, p10/p90 bands and production cost.

### Miner Health

Measures how comfortable miners are relative to the standard production-cost anchor.

### Source Agreement

Measures agreement between the chart/source price and the official monitor spot. At this stage, the terminal state uses the monitor spot as official source; TradingView Pine can additionally compute chart-vs-monitor source gap using the selected chart symbol.

### Fragility Index

Measures structural fragility. Higher values indicate higher market stress, larger deviations, lower confidence or weaker miner context.

### Confidence Score

Measures how much confidence the system has in the current structural interpretation. It is not a probability of price movement.

## Auditability Rule

Every score must include a `reasoning` entry explaining the main contributors.

No score should be presented without an explanation path.
