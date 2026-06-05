#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

HTML = Path('btc-swing-strategy/mecanism.html')
START = '<!-- RISK_WINDOW_CARD_START -->'
END = '<!-- RISK_WINDOW_CARD_END -->'

CSS = r'''

    /* CARD FEREASTRA DE RISC */
    .risk-card-header {
      display: flex;
      flex-direction: column;
      gap: 4px;
      margin-bottom: 12px;
    }
    .risk-title {
      font-size: 13px;
      letter-spacing: .18em;
      text-transform: uppercase;
      color: var(--text-muted);
    }
    .risk-subtitle {
      font-size: 12px;
      color: var(--text-soft);
    }
    .risk-main {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .risk-pill {
      display: inline-flex;
      align-items: center;
      width: fit-content;
      padding: 3px 10px;
      border-radius: 999px;
      font-size: 11px;
      background: rgba(15,23,42,0.9);
      border: 1px solid var(--accent-neutral);
      color: var(--text-main);
    }
    .risk-pill.risk-high { border-color: #f97373; color: #fecaca; }
    .risk-pill.risk-moderate { border-color: #facc15; color: #fef3c7; }
    .risk-pill.risk-low { border-color: #38bdf8; color: #bae6fd; }
    .risk-pill.risk-normal { border-color: #22c55e; color: #bbf7d0; }
    .risk-text {
      font-size: 13px;
      line-height: 1.55;
      color: var(--text-main);
    }
    .risk-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-top: 4px;
    }
    .risk-metric {
      border-radius: 12px;
      background: rgba(15,23,42,0.85);
      border: 1px solid rgba(31,41,55,0.9);
      padding: 8px 10px;
    }
    .risk-metric-label {
      font-size: 10px;
      color: var(--text-soft);
      text-transform: uppercase;
      letter-spacing: .08em;
    }
    .risk-metric-value {
      font-size: 13px;
      font-weight: 600;
      margin-top: 2px;
    }
    .risk-footer {
      margin-top: 8px;
      font-size: 11px;
      color: var(--text-soft);
    }
'''

HTML_BLOCK = r'''

    <!-- RISK_WINDOW_CARD_START -->
    <div class="card card-secondary" id="risk-window-card">
      <div class="card-inner">
        <div class="risk-card-header">
          <div class="risk-title">FEREASTRĂ DE RISC</div>
          <div class="risk-subtitle">Context istoric derivat din backtestul coeziv.</div>
        </div>
        <div class="risk-main">
          <div id="risk-window-pill" class="risk-pill risk-normal">Se încarcă...</div>
          <div id="risk-window-text" class="risk-text">
            Se încarcă fereastra de risc structural.
          </div>
          <div class="risk-grid">
            <div class="risk-metric">
              <div class="risk-metric-label">Confirmare istorică</div>
              <div id="risk-window-rate" class="risk-metric-value">–</div>
            </div>
            <div class="risk-metric">
              <div class="risk-metric-label">Timp median</div>
              <div id="risk-window-days" class="risk-metric-value">–</div>
            </div>
            <div class="risk-metric">
              <div class="risk-metric-label">Regim curent</div>
              <div id="risk-window-regime" class="risk-metric-value">–</div>
            </div>
            <div class="risk-metric">
              <div class="risk-metric-label">Persistență</div>
              <div id="risk-window-streak" class="risk-metric-value">–</div>
            </div>
          </div>
          <div id="risk-window-footer" class="risk-footer">
            Interpretare statistică, nu recomandare de tranzacționare.
          </div>
        </div>
      </div>
    </div>
    <!-- RISK_WINDOW_CARD_END -->
'''

JS = r'''

    const RISK_WINDOW_URL = "risk_window.json";
    const RISK_CARD_EL = document.getElementById("risk-window-card");
    const RISK_PILL_EL = document.getElementById("risk-window-pill");
    const RISK_TEXT_EL = document.getElementById("risk-window-text");
    const RISK_RATE_EL = document.getElementById("risk-window-rate");
    const RISK_DAYS_EL = document.getElementById("risk-window-days");
    const RISK_REGIME_EL = document.getElementById("risk-window-regime");
    const RISK_STREAK_EL = document.getElementById("risk-window-streak");
    const RISK_FOOTER_EL = document.getElementById("risk-window-footer");

    function riskPct(value) {
      if (value === null || value === undefined || !Number.isFinite(Number(value))) return "n/a";
      return `${(Number(value) * 100).toFixed(0)}%`;
    }

    function riskDays(value) {
      if (value === null || value === undefined || !Number.isFinite(Number(value))) return "n/a";
      return `~${Number(value).toFixed(0)} zile`;
    }

    function riskRegimeLabel(regime) {
      const r = (regime || "").toLowerCase();
      const map = {
        bear_struct: "Degradare structurală",
        bear_late: "Degradare avansată",
        accum_bear: "Acumulare fragilă",
        bull_struct: "Structură pozitivă",
        bull_late: "Expansiune matură",
        accum_bull: "Acumulare pozitivă",
        range_pos: "Range cu bias pozitiv",
        range_neg: "Range cu bias negativ",
        range_neutral: "Range neutru",
        neutral: "Tranziție neutră"
      };
      return map[r] || (regime || "n/a");
    }

    function riskLevelLabel(level, active) {
      if (!active) return "Fereastră normală";
      if (level === "high") return "Risc structural ridicat";
      if (level === "moderate") return "Risc structural moderat";
      if (level === "low") return "Risc structural scăzut";
      return "Risc structural n/a";
    }

    function updateRiskWindowCard(data) {
      if (!RISK_CARD_EL || !data) return;
      const active = !!data.active;
      const level = data.level || (active ? "unknown" : "normal");
      const rate = data.historical_confirmation_rate;
      const days = data.median_days_to_confirmation;
      const regime = data.current_regime || "n/a";
      const streak = data.consecutive_degradation_days || 0;

      if (RISK_PILL_EL) {
        RISK_PILL_EL.className = `risk-pill risk-${active ? level : "normal"}`;
        RISK_PILL_EL.textContent = riskLevelLabel(level, active);
      }

      if (RISK_TEXT_EL) {
        RISK_TEXT_EL.textContent = data.main_text || "Fereastra de risc nu este disponibilă momentan.";
      }
      if (RISK_RATE_EL) RISK_RATE_EL.textContent = riskPct(rate);
      if (RISK_DAYS_EL) RISK_DAYS_EL.textContent = riskDays(days);
      if (RISK_REGIME_EL) RISK_REGIME_EL.textContent = riskRegimeLabel(regime);
      if (RISK_STREAK_EL) RISK_STREAK_EL.textContent = `${streak} zile`;
      if (RISK_FOOTER_EL && data.footer) RISK_FOOTER_EL.textContent = data.footer;
    }

    async function loadRiskWindow() {
      try {
        const resp = await fetch(`${RISK_WINDOW_URL}?t=${Date.now()}`, { cache: "no-store" });
        if (!resp.ok) throw new Error("risk_window indisponibil");
        const data = await resp.json();
        updateRiskWindowCard(data);
      } catch (e) {
        if (RISK_TEXT_EL) RISK_TEXT_EL.textContent = "Fereastra de risc va apărea după următorul backtest automat.";
        if (RISK_PILL_EL) {
          RISK_PILL_EL.className = "risk-pill risk-normal";
          RISK_PILL_EL.textContent = "Așteaptă date";
        }
      }
    }
'''


def replace_block(text, start, end, block):
    if start in text and end in text:
        a = text.index(start)
        b = text.index(end, a) + len(end)
        return text[:a] + block.strip("\n") + text[b:]
    return None


def replace_js_block(text):
    marker = '    const RISK_WINDOW_URL = "risk_window.json";'
    end_marker = '    async function loadRiskWindow() {'
    if marker not in text:
        return text
    a = text.index(marker)
    b = text.index(end_marker, a)
    # include async function block until before next known section; safest marker is function formatDate
    next_marker = '    function formatDate(iso) {'
    c = text.find(next_marker, b)
    if c == -1:
        return text
    return text[:a] + JS.strip("\n") + "\n\n" + text[c:]


def main():
    text = HTML.read_text(encoding='utf-8')

    if '/* CARD FEREASTRA DE RISC */' not in text:
        text = text.replace('  </style>', CSS + '\n  </style>')

    repl = replace_block(text, START, END, HTML_BLOCK)
    if repl is not None:
        text = repl
    elif '<!-- CARD FEAR & GREED COEZIV -->' in text:
        text = text.replace('    <!-- CARD FEAR & GREED COEZIV -->', HTML_BLOCK + '\n\n    <!-- CARD FEAR & GREED COEZIV -->')
    else:
        text = text.replace('  <script>', HTML_BLOCK + '\n\n  <script>')

    if 'const RISK_WINDOW_URL = "risk_window.json";' not in text:
        text = text.replace('    const STATE_URL = "coeziv_state.json";', '    const STATE_URL = "coeziv_state.json";' + JS)
    elif 'function riskRegimeLabel(regime)' not in text:
        text = replace_js_block(text)

    if 'loadRiskWindow();' not in text:
        text = text.replace('    loadState();', '    loadState();\n    loadRiskWindow();')

    HTML.write_text(text, encoding='utf-8')
    print('[OK] mecanism.html patched with risk window card')


if __name__ == '__main__':
    main()
