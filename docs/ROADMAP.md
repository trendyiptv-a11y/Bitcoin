# CohesivX OS Roadmap

## Mission

CohesivX OS is the unified structural analysis layer for Bitcoin. Its purpose is to answer one question clearly:

> What is the structural state of the Bitcoin market, and why?

The system is not designed as a black-box trading signal. It is designed as an auditable market terminal where every score, state and warning can be traced back to the data that produced it.

## Core Principle

`btc-swing-strategy/coeziv_state.json` is the primary source of truth.

Every downstream surface should consume or derive from the same state:

- Website monitor
- TradingView Pine terminal
- Terminal snapshot
- Future API
- Future Telegram bot
- Future AI assistant
- Future replay/laboratory tools

## Milestone 0.1 — Genesis

Status: in progress

Goals:

- [x] CohesivX monitor snapshot
- [x] TradingView yearly anchors export
- [x] TradingView Pine generator
- [x] Branch-aware GitHub workflow
- [x] Pine terminal table modes
- [x] BLX / source-gap analysis
- [ ] CohesivX terminal state export
- [ ] Structural Score
- [ ] Fragility Index
- [ ] Confidence Score
- [ ] Source Agreement
- [ ] Audit trail / Why panel

## Milestone 0.2 — Terminal Intelligence

Goals:

- [ ] `cohesivx_terminal_state.json`
- [ ] terminal summary block
- [ ] score audit trail
- [ ] compact machine-readable state for website and Pine
- [ ] market-state label, for example `STRUCTURAL_ACCUMULATION`

## Milestone 0.3 — Replay

Goals:

- [ ] daily snapshot archive
- [ ] yearly/monthly index
- [ ] replay-compatible state format
- [ ] historical terminal reconstruction

## Milestone 0.4 — Laboratory

Goals:

- [ ] electricity-cost scenario simulator
- [ ] hashrate/difficulty scenario simulator
- [ ] miner-cost sensitivity
- [ ] structural fair-value sensitivity

## Milestone 1.0 — Public Terminal

Goals:

- [ ] clean website dashboard
- [ ] TradingView terminal v3
- [ ] documentation pages
- [ ] API-ready state
- [ ] model limitations page
- [ ] release notes

## Development Rule

A feature enters CohesivX only if the answer is clear:

1. What does it measure?
2. How is it calculated?
3. How does it help the user understand the market?
