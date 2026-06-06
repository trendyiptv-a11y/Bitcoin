#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

HTML = Path("btc-swing-strategy/mecanism.html")

CSS_MARK = "/* DAILY_COHESIV_INTERPRETATION_CSS */"
CARD_START = "<!-- DAILY_COHESIV_INTERPRETATION_CARD_START -->"
CARD_END = "<!-- DAILY_COHESIV_INTERPRETATION_CARD_END -->"
JS_MARK = "// DAILY_COHESIV_INTERPRETATION_JS"
CALL_MARK = "// DAILY_COHESIV_INTERPRETATION_CALL"

CSS = f"""
    {CSS_MARK}
    .daily-ai-header {{
      display: flex;
      flex-direction: column;
      gap: 4px;
      margin-bottom: 12px;
    }}
    .daily-ai-title {{
      font-size: 13px;
      letter-spacing: .18em;
      text-transform: uppercase;
      color: var(--text-muted);
    }}
    .daily-ai-subtitle {{
      font-size: 12px;
      color: var(--text-soft);
    }}
    .daily-ai-state {{
      display: inline-flex;
      align-items: center;
      width: fit-content;
      padding: 3px 10px;
      border-radius: 999px;
      font-size: 11px;
      background: rgba(15,23,42,0.9);
      border: 1px solid #38bdf8;
      color: #bae6fd;
      margin-bottom: 8px;
    }}
    .daily-ai-summary {{
      font-size: 14px;
      line-height: 1.55;
      color: var(--text-main);
      margin-bottom: 10px;
    }}
    .daily-ai-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
      margin-top: 10px;
    }}
    .daily-ai-item {{
      border-radius: 12px;
      background: rgba(15,23,42,0.85);
      border: 1px solid rgba(31,41,55,0.9);
      padding: 8px 10px;
    }}
    .daily-ai-label {{
      font-size: 10px;
      color: var(--text-soft);
      text-transform: uppercase;
      letter-spacing: .08em;
      margin-bottom: 3px;
    }}
    .daily-ai-value {{
      font-size: 13px;
      line-height: 1.45;
      color: var(--text-main);
    }}
    .daily-ai-footer {{
      margin-top: 8px;
      font-size: 11px;
      color: var(--text-soft);
    }}
"""

CARD = f"""
    {CARD_START}
    <div class="card card-secondary" id="daily-ai-card">
      <div class="card-inner">
        <div class="daily-ai-header">
          <div class="daily-ai-title">INTERPRETARE COEZIVĂ ZILNICĂ</div>
          <div class="daily-ai-subtitle">Explicație naturală a stării structurale observate de mecanism.</div>
        </div>
        <div id="daily-ai-state" class="daily-ai-state">Se încarcă...</div>
        <div id="daily-ai-summary" class="daily-ai-summary">
          Interpretarea zilnică va apărea după prima analiză aprobată.
        </div>
        <div class="daily-ai-grid">
          <div class="daily-ai-item">
            <div class="daily-ai-label">Participare</div>
            <div id="daily-ai-participation" class="daily-ai-value">–</div>
          </div>
          <div class="daily-ai-item">
            <div class="daily-ai-label">Fereastră risc</div>
            <div id="daily-ai-risk" class="daily-ai-value">–</div>
          </div>
          <div class="daily-ai-item">
            <div class="daily-ai-label">Fear &amp; Greed / Regim</div>
            <div id="daily-ai-fg-regime" class="daily-ai-value">–</div>
          </div>
          <div class="daily-ai-item">
            <div class="daily-ai-label">Ce urmărim</div>
            <div id="daily-ai-watch" class="daily-ai-value">–</div>
          </div>
        </div>
        <div id="daily-ai-footer" class="daily-ai-footer">
          Interpretare structurală experimentală, nu recomandare financiară.
        </div>
      </div>
    </div>
    {CARD_END}
"""

JS = f"""
    {JS_MARK}
    const DAILY_AI_URLS = [
      "daily_cohesiv_interpretation.json",
      "./daily_cohesiv_interpretation.json",
      "/btc-swing-strategy/daily_cohesiv_interpretation.json"
    ];

    function setText(id, value) {{
      const el = document.getElementById(id);
      if (el) el.textContent = value || "–";
    }}

    async function loadDailyCohesivInterpretation() {{
      let data = null;
      for (const url of DAILY_AI_URLS) {{
        try {{
          const res = await fetch(url + "?t=" + Date.now(), {{ cache: "no-store" }});
          if (!res.ok) continue;
          data = await res.json();
          break;
        }} catch (e) {{}}
      }}
      if (!data) {{
        setText("daily-ai-state", "Interpretare indisponibilă");
        setText("daily-ai-summary", "Cardul va fi completat după prima analiză zilnică aprobată.");
        return;
      }}
      setText("daily-ai-state", data.general_state || data.status || "stare în curs de interpretare");
      setText("daily-ai-summary", data.plain_language || data.summary || "Interpretarea nu este încă disponibilă.");
      setText("daily-ai-participation", data.participation || "–");
      setText("daily-ai-risk", data.risk_window || "–");
      const fgRegime = [data.fear_greed, data.market_regime].filter(Boolean).join(" • ");
      setText("daily-ai-fg-regime", fgRegime || "–");
      setText("daily-ai-watch", data.watch_next || "–");
      setText("daily-ai-footer", data.disclaimer || "Interpretare structurală experimentală, nu recomandare financiară.");
    }}
"""


def main() -> None:
    s = HTML.read_text(encoding="utf-8")

    if CSS_MARK not in s:
        s = s.replace("  </style>", CSS + "\n  </style>")

    if CARD_START not in s:
        marker = "<!-- PARTICIPATION_COHESION_CARD_START -->"
        if marker in s:
            s = s.replace(marker, CARD + "\n" + marker)
        else:
            s = s.replace("<script>", CARD + "\n  <script>")

    if JS_MARK not in s:
        marker = "    const STATE_URL = \"coeziv_state.json\";"
        if marker in s:
            s = s.replace(marker, JS + "\n" + marker)
        else:
            s = s.replace("  </script>", JS + "\n  </script>")

    if CALL_MARK not in s:
        call = f"\n    {CALL_MARK}\n    loadDailyCohesivInterpretation();\n"
        if "loadParticipationCohesion();" in s:
            s = s.replace("loadParticipationCohesion();", "loadParticipationCohesion();" + call)
        else:
            s = s.replace("  </script>", call + "\n  </script>")

    HTML.write_text(s, encoding="utf-8")


if __name__ == "__main__":
    main()
