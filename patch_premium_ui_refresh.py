#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

HTML = Path("btc-swing-strategy/mecanism.html")
START = "/* PREMIUM_UI_REFRESH_START */"
END = "/* PREMIUM_UI_REFRESH_END */"

CSS = r'''
/* PREMIUM_UI_REFRESH_START */

.shell {
  max-width: 560px;
}

.title-bar {
  margin-top: 4px;
  margin-bottom: 12px;
  font-size: 12px;
  color: #94a3b8;
  text-shadow: 0 0 18px rgba(56,189,248,0.35);
}

.card {
  backdrop-filter: blur(18px);
  transition:
    transform .22s ease,
    border-color .22s ease,
    box-shadow .22s ease,
    background .22s ease;
}

.card:hover {
  transform: translateY(-2px);
  border-color: rgba(56,189,248,0.28);
}

.card:first-of-type {
  background:
    radial-gradient(circle at 12% 0%, rgba(56,189,248,0.20), transparent 38%),
    radial-gradient(circle at 88% 8%, rgba(34,197,94,0.12), transparent 34%),
    linear-gradient(145deg, rgba(15,23,42,0.98), rgba(2,6,23,0.97));
  border: 1px solid rgba(56,189,248,0.28);
  box-shadow:
    0 34px 100px rgba(0,0,0,0.88),
    inset 0 1px 0 rgba(255,255,255,0.045),
    0 0 45px rgba(56,189,248,0.08);
}

body.light-mode .card:first-of-type {
  background:
    radial-gradient(circle at 12% 0%, rgba(14,165,233,0.18), transparent 38%),
    radial-gradient(circle at 88% 8%, rgba(34,197,94,0.12), transparent 34%),
    linear-gradient(145deg, rgba(255,255,255,0.98), rgba(241,245,249,0.97));
  border-color: rgba(14,165,233,0.28);
}

.asset-row {
  gap: 12px;
}

.asset-logo {
  width: 34px;
  height: 34px;
  font-size: 18px;
  box-shadow:
    0 0 24px rgba(249,115,22,0.38),
    inset 0 1px 0 rgba(255,255,255,0.35);
}

.asset-text h1 {
  font-size: 19px;
}

.asset-text span {
  margin-top: 2px;
}

.signal-chip {
  font-weight: 650;
  letter-spacing: .04em;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
}

.signal-chip.bullish {
  background: rgba(20,83,45,0.22);
  box-shadow: 0 0 24px rgba(34,197,94,0.13);
}

.signal-chip.bearish {
  background: rgba(127,29,29,0.22);
  box-shadow: 0 0 24px rgba(249,115,115,0.13);
}

.price-block {
  margin-top: 18px;
}

.price-value {
  font-size: 46px;
  line-height: 1;
  background: linear-gradient(180deg, #ffffff, #cbd5e1);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  text-shadow: 0 18px 40px rgba(0,0,0,0.28);
}

body.light-mode .price-value {
  background: linear-gradient(180deg, #0f172a, #334155);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.currency {
  margin-top: 6px;
}

.live-row {
  margin-top: 12px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 5px;
  padding: 5px 10px;
  border-radius: 999px;
  background: rgba(15,23,42,0.72);
  border: 1px solid rgba(56,189,248,0.18);
}

body.light-mode .live-row {
  background: rgba(248,250,252,0.82);
}

.live-value {
  color: var(--text-main);
  font-weight: 700;
}

.live-delta,
.deviation-status {
  line-height: 1.45;
}

#regime-line {
  margin-top: 10px;
  padding: 8px 10px;
  border-radius: 14px;
  background: rgba(15,23,42,0.62);
  border: 1px solid rgba(148,163,184,0.12);
}

body.light-mode #regime-line {
  background: rgba(248,250,252,0.85);
}

.premium-status-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  justify-content: center;
  margin: 12px 0 4px;
}

.premium-status-pill {
  border-radius: 999px;
  border: 1px solid rgba(148,163,184,0.18);
  background: rgba(15,23,42,0.7);
  color: var(--text-muted);
  padding: 4px 9px;
  font-size: 10px;
  letter-spacing: .08em;
  text-transform: uppercase;
}

body.light-mode .premium-status-pill {
  background: rgba(255,255,255,0.84);
}

.cohesiv-metric-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 9px;
  margin: 14px 0 12px;
}

.cohesiv-metric-tile {
  border-radius: 15px;
  padding: 10px 11px;
  background:
    linear-gradient(180deg, rgba(15,23,42,0.82), rgba(15,23,42,0.58));
  border: 1px solid rgba(148,163,184,0.13);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.035);
}

body.light-mode .cohesiv-metric-tile {
  background:
    linear-gradient(180deg, rgba(255,255,255,0.92), rgba(248,250,252,0.78));
  border-color: rgba(148,163,184,0.28);
}

.cohesiv-metric-label {
  font-size: 10px;
  color: var(--text-soft);
  text-transform: uppercase;
  letter-spacing: .1em;
}

.cohesiv-metric-value {
  margin-top: 5px;
  font-size: 14px;
  line-height: 1.25;
  font-weight: 700;
  color: var(--text-main);
}

.cohesiv-metric-note {
  margin-top: 3px;
  font-size: 10px;
  color: var(--text-soft);
  line-height: 1.35;
}

.chart-block {
  height: 250px;
  border-radius: 18px;
  border-color: rgba(56,189,248,0.30);
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.04),
    0 18px 45px rgba(0,0,0,0.35);
}

.message {
  border-radius: 16px;
  padding: 11px 12px;
  background: rgba(15,23,42,0.52);
  border: 1px solid rgba(148,163,184,0.12);
}

body.light-mode .message {
  background: rgba(248,250,252,0.78);
  border-color: rgba(148,163,184,0.24);
}

.fg-bar-track {
  height: 12px;
  box-shadow:
    inset 0 1px 2px rgba(0,0,0,0.36),
    0 0 24px rgba(56,189,248,0.08);
}

.fg-bar-fill {
  top: -4px;
  bottom: auto;
  width: 4px !important;
  height: 20px;
  left: 0%;
  border-radius: 999px;
  background: #f8fafc;
  box-shadow:
    0 0 0 1px rgba(15,23,42,0.9),
    0 0 14px rgba(255,255,255,0.45);
  transform: translateX(-50%);
}

body.light-mode .fg-bar-fill {
  background: #0f172a;
  box-shadow:
    0 0 0 1px rgba(255,255,255,0.9),
    0 0 14px rgba(15,23,42,0.22);
}

.history-row,
.daily-ai-item,
.risk-metric,
.participation-metric {
  transition:
    transform .18s ease,
    border-color .18s ease,
    background .18s ease;
}

.history-row:hover,
.daily-ai-item:hover,
.risk-metric:hover,
.participation-metric:hover {
  transform: translateY(-1px);
  border-color: rgba(56,189,248,0.24);
}

.premium-section-kicker {
  display: inline-flex;
  width: fit-content;
  padding: 3px 8px;
  border-radius: 999px;
  border: 1px solid rgba(56,189,248,0.22);
  background: rgba(56,189,248,0.06);
  color: #bae6fd;
  font-size: 10px;
  letter-spacing: .08em;
  text-transform: uppercase;
}

body.light-mode .premium-section-kicker {
  color: #075985;
  background: rgba(14,165,233,0.08);
}

/* PREMIUM_UI_REFRESH: optional classes for future compact regime details. */
details.regime-detail {
  border-radius: 12px;
  background: rgba(15,23,42,0.84);
  border: 1px solid rgba(31,41,55,0.92);
  padding: 0;
  overflow: hidden;
}

body.light-mode details.regime-detail {
  background: rgba(248,250,252,0.92);
  border-color: rgba(148,163,184,0.48);
}

details.regime-detail summary {
  cursor: pointer;
  list-style: none;
  padding: 9px 11px;
  font-size: 12px;
  font-weight: 650;
  color: var(--text-main);
}

details.regime-detail summary::-webkit-details-marker {
  display: none;
}

details.regime-detail summary::after {
  content: "+";
  float: right;
  color: var(--text-soft);
}

details.regime-detail[open] summary::after {
  content: "−";
}

.regime-detail-body {
  padding: 0 11px 10px;
  font-size: 11px;
  line-height: 1.5;
  color: var(--text-soft);
}

@media (max-width: 480px) {
  .price-value {
    font-size: 38px;
  }

  .cohesiv-metric-grid {
    grid-template-columns: 1fr;
  }

  .asset-row {
    align-items: flex-start;
  }

  .signal-chip {
    font-size: 10px;
    padding: 4px 8px;
  }

  .chart-block {
    height: 230px;
  }
}

/* PREMIUM_UI_REFRESH_END */
'''


def replace_block(s: str) -> str:
    start = s.find(START)
    end = s.find(END)
    if start != -1 and end != -1 and end > start:
        end += len(END)
        return s[:start] + CSS.strip() + s[end:]
    return s.replace("  </style>", "\n" + CSS.strip() + "\n  </style>")


def main() -> None:
    s = HTML.read_text(encoding="utf-8")
    s = replace_block(s)
    HTML.write_text(s, encoding="utf-8")


if __name__ == "__main__":
    main()
