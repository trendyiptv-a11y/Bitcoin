#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

HTML = Path('btc-swing-strategy/mecanism.html')
START = '<!-- PARTICIPATION_COHESION_CARD_START -->'
END = '<!-- PARTICIPATION_COHESION_CARD_END -->'

CSS = r'''

    /* CARD COEZIUNE PARTICIPATIVA */
    .participation-card-header {
      display: flex;
      flex-direction: column;
      gap: 4px;
      margin-bottom: 12px;
    }
    .participation-title {
      font-size: 13px;
      letter-spacing: .18em;
      text-transform: uppercase;
      color: var(--text-muted);
    }
    .participation-subtitle {
      font-size: 12px;
      color: var(--text-soft);
    }
    .participation-pill {
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
    .participation-pill.participation-cohesive { border-color: #22c55e; color: #bbf7d0; }
    .participation-pill.participation-tense { border-color: #facc15; color: #fef3c7; }
    .participation-pill.participation-fragile { border-color: #fb923c; color: #fed7aa; }
    .participation-pill.participation-degraded { border-color: #f97373; color: #fecaca; }
    .participation-score {
      font-size: 34px;
      font-weight: 750;
      line-height: 1;
      margin-top: 6px;
    }
    .participation-text {
      font-size: 13px;
      line-height: 1.55;
      color: var(--text-main);
      margin-top: 8px;
    }
    .participation-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-top: 10px;
    }
    .participation-metric {
      border-radius: 12px;
      background: rgba(15,23,42,0.85);
      border: 1px solid rgba(31,41,55,0.9);
      padding: 8px 10px;
    }
    .participation-metric-label {
      font-size: 10px;
      color: var(--text-soft);
      text-transform: uppercase;
      letter-spacing: .08em;
    }
    .participation-metric-value {
      font-size: 13px;
      font-weight: 600;
      margin-top: 2px;
    }
    .participation-footer {
      margin-top: 8px;
      font-size: 11px;
      color: var(--text-soft);
    }
'''

HTML_BLOCK = r'''

    <!-- PARTICIPATION_COHESION_CARD_START -->
    <div class="card card-secondary" id="participation-cohesion-card">
      <div class="card-inner">
        <div class="participation-card-header">
          <div class="participation-title">COEZIUNE PARTICIPATIVĂ</div>
          <div class="participation-subtitle">Starea interesului participanților în ecosistem.</div>
        </div>
        <div id="participation-cohesion-pill" class="participation-pill participation-tense">Se încarcă...</div>
        <div id="participation-cohesion-score" class="participation-score">–</div>
        <div id="participation-cohesion-text" class="participation-text">
          Se încarcă indicatorul de coeziune participativă.
        </div>
        <div class="participation-grid">
          <div class="participation-metric">
            <div class="participation-metric-label">Flux</div>
            <div id="participation-flow" class="participation-metric-value">–</div>
          </div>
          <div class="participation-metric">
            <div class="participation-metric-label">Lichiditate</div>
            <div id="participation-liquidity" class="participation-metric-value">–</div>
          </div>
          <div class="participation-metric">
            <div class="participation-metric-label">Istoric mediu</div>
            <div id="participation-history-mean" class="participation-metric-value">–</div>
          </div>
          <div class="participation-metric">
            <div class="participation-metric-label">Minim istoric</div>
            <div id="participation-history-min" class="participation-metric-value">–</div>
          </div>
        </div>
        <div id="participation-cohesion-footer" class="participation-footer">
          Indicator experimental, nu recomandare de tranzacționare.
        </div>
      </div>
    </div>
    <!-- PARTICIPATION_COHESION_CARD_END -->
'''

JS = r'''

    const PARTICIPATION_URL = "participation_cohesion.json";
    const PARTICIPATION_CARD_EL = document.getElementById("participation-cohesion-card");
    const PARTICIPATION_PILL_EL = document.getElementById("participation-cohesion-pill");
    const PARTICIPATION_SCORE_EL = document.getElementById("participation-cohesion-score");
    const PARTICIPATION_TEXT_EL = document.getElementById("participation-cohesion-text");
    const PARTICIPATION_FLOW_EL = document.getElementById("participation-flow");
    const PARTICIPATION_LIQUIDITY_EL = document.getElementById("participation-liquidity");
    const PARTICIPATION_HISTORY_MEAN_EL = document.getElementById("participation-history-mean");
    const PARTICIPATION_HISTORY_MIN_EL = document.getElementById("participation-history-min");
    const PARTICIPATION_FOOTER_EL = document.getElementById("participation-cohesion-footer");

    function participationNumber(value) {
      if (value === null || value === undefined || !Number.isFinite(Number(value))) return "n/a";
      return Number(value).toFixed(0);
    }

    function participationLabel(data) {
      const label = data && data.label ? data.label : "participare necunoscută";
      return label.charAt(0).toUpperCase() + label.slice(1);
    }

    function updateParticipationCard(data) {
      if (!PARTICIPATION_CARD_EL || !data) return;
      const level = data.level || "tense";
      const score = participationNumber(data.score);
      const comps = data.components || {};
      const history = data.history || {};

      if (PARTICIPATION_PILL_EL) {
        PARTICIPATION_PILL_EL.className = `participation-pill participation-${level}`;
        PARTICIPATION_PILL_EL.textContent = participationLabel(data);
      }
      if (PARTICIPATION_SCORE_EL) PARTICIPATION_SCORE_EL.textContent = `${score} / 100`;
      if (PARTICIPATION_TEXT_EL) PARTICIPATION_TEXT_EL.textContent = data.main_text || "Indicatorul de participare nu este disponibil momentan.";
      if (PARTICIPATION_FLOW_EL) PARTICIPATION_FLOW_EL.textContent = participationNumber(comps.flow_component);
      if (PARTICIPATION_LIQUIDITY_EL) PARTICIPATION_LIQUIDITY_EL.textContent = participationNumber(comps.liquidity_component);
      if (PARTICIPATION_HISTORY_MEAN_EL) PARTICIPATION_HISTORY_MEAN_EL.textContent = participationNumber(history.mean_score);
      if (PARTICIPATION_HISTORY_MIN_EL) PARTICIPATION_HISTORY_MIN_EL.textContent = participationNumber(history.min_score);
      if (PARTICIPATION_FOOTER_EL && data.footer) PARTICIPATION_FOOTER_EL.textContent = data.footer;
    }

    async function loadParticipationCohesion() {
      try {
        const resp = await fetch(`${PARTICIPATION_URL}?t=${Date.now()}`, { cache: "no-store" });
        if (!resp.ok) throw new Error("participation_cohesion indisponibil");
        const data = await resp.json();
        updateParticipationCard(data);
      } catch (e) {
        if (PARTICIPATION_TEXT_EL) PARTICIPATION_TEXT_EL.textContent = "Coeziunea participativă va apărea după următorul test automat.";
        if (PARTICIPATION_PILL_EL) {
          PARTICIPATION_PILL_EL.className = "participation-pill participation-tense";
          PARTICIPATION_PILL_EL.textContent = "Așteaptă date";
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


def main():
    text = HTML.read_text(encoding='utf-8')

    if '/* CARD COEZIUNE PARTICIPATIVA */' not in text:
        text = text.replace('  </style>', CSS + '\n  </style>')

    repl = replace_block(text, START, END, HTML_BLOCK)
    if repl is not None:
        text = repl
    elif '<!-- RISK_WINDOW_CARD_START -->' in text:
        text = text.replace('    <!-- RISK_WINDOW_CARD_START -->', HTML_BLOCK + '\n\n    <!-- RISK_WINDOW_CARD_START -->')
    elif '<!-- CARD FEAR & GREED COEZIV -->' in text:
        text = text.replace('    <!-- CARD FEAR & GREED COEZIV -->', HTML_BLOCK + '\n\n    <!-- CARD FEAR & GREED COEZIV -->')
    else:
        text = text.replace('  <script>', HTML_BLOCK + '\n\n  <script>')

    if 'const PARTICIPATION_URL = "participation_cohesion.json";' not in text:
        text = text.replace('    const STATE_URL = "coeziv_state.json";', '    const STATE_URL = "coeziv_state.json";' + JS)

    if 'loadParticipationCohesion();' not in text:
        text = text.replace('    loadState();', '    loadState();\n    loadParticipationCohesion();')

    HTML.write_text(text, encoding='utf-8')
    print('[OK] mecanism.html patched with participation cohesion card')


if __name__ == '__main__':
    main()
