#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch UI pentru CohesivX BTC:
- înlocuiește linia simplă „Cost de producție” cu un mini-card clar;
- păstrează logica backend: production_costs_usd = cheap / average / expensive;
- redenumește corect indicatorul ca prag energetic / cost energetic estimat.
"""
from pathlib import Path

PATH = Path("btc-swing-strategy/mecanism.html")
html = PATH.read_text(encoding="utf-8")
original = html

css = r'''

/* ENERGY_PRODUCTION_CARD_START */
.energy-card {
  margin: 14px auto 12px;
  padding: 13px;
  border-radius: 18px;
  border: 1px solid rgba(56,189,248,0.30);
  background:
    radial-gradient(circle at 0% 0%, rgba(56,189,248,0.16), transparent 44%),
    linear-gradient(180deg, rgba(15,23,42,0.78), rgba(15,23,42,0.54));
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.045),
    0 14px 34px rgba(0,0,0,0.22);
  text-align: left;
}
.energy-card.loading {
  opacity: .78;
}
.energy-card-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 10px;
}
.energy-title {
  font-size: 11px;
  letter-spacing: .14em;
  text-transform: uppercase;
  font-weight: 800;
  color: #67e8f9;
}
.energy-subtitle {
  margin-top: 3px;
  font-size: 11px;
  color: var(--text-soft);
  line-height: 1.35;
}
.energy-badge {
  flex: 0 0 auto;
  padding: 4px 8px;
  border-radius: 999px;
  border: 1px solid rgba(103,232,249,0.34);
  background: rgba(8,47,73,0.36);
  color: #bae6fd;
  font-size: 9px;
  letter-spacing: .1em;
  text-transform: uppercase;
  font-weight: 750;
}
.energy-values {
  display: grid;
  grid-template-columns: 1fr;
  gap: 7px;
}
.energy-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  padding: 8px 10px;
  border-radius: 13px;
  border: 1px solid rgba(148,163,184,0.14);
  background: rgba(2,6,23,0.34);
}
.energy-item.main {
  border-color: rgba(34,211,238,0.44);
  background: rgba(8,47,73,0.42);
  box-shadow: 0 0 20px rgba(34,211,238,0.08);
}
.energy-label {
  font-size: 11px;
  color: #cbd5e1;
  line-height: 1.25;
}
.energy-value {
  font-size: 12px;
  color: var(--text-main);
  font-weight: 800;
  white-space: nowrap;
}
.energy-note {
  margin-top: 9px;
  padding-top: 8px;
  border-top: 1px solid rgba(148,163,184,0.14);
  font-size: 10px;
  line-height: 1.42;
  color: var(--text-soft);
}
body.light-mode .energy-card {
  background:
    radial-gradient(circle at 0% 0%, rgba(14,165,233,0.14), transparent 44%),
    linear-gradient(180deg, rgba(255,255,255,0.94), rgba(248,250,252,0.82));
  border-color: rgba(14,165,233,0.28);
  box-shadow: 0 10px 24px rgba(15,23,42,0.10);
}
body.light-mode .energy-badge {
  background: rgba(224,242,254,0.74);
  color: #075985;
  border-color: rgba(14,165,233,0.30);
}
body.light-mode .energy-item {
  background: rgba(255,255,255,0.76);
  border-color: rgba(148,163,184,0.28);
}
body.light-mode .energy-item.main {
  background: rgba(224,242,254,0.70);
  border-color: rgba(14,165,233,0.36);
}
body.light-mode .energy-label {
  color: #334155;
}
@media (max-width: 380px) {
  .energy-card { padding: 12px 10px; }
  .energy-item { padding: 8px 9px; }
  .energy-label { font-size: 10px; }
  .energy-value { font-size: 11px; }
}
/* ENERGY_PRODUCTION_CARD_END */
'''

if "/* ENERGY_PRODUCTION_CARD_START */" not in html:
    marker = "    /* COMPACT_REGIME_CHIP_CSS */"
    if marker in html:
        html = html.replace(marker, css + "\n" + marker, 1)
    else:
        html = html.replace("</style>", css + "\n  </style>", 1)

old_div = '''          <div id="prod-cost-line" class="live-delta">
            Cost de producție: n/a (așteptăm date despre cost).
          </div>'''
new_div = '''          <div id="prod-cost-line" class="energy-card loading" aria-live="polite">
            <div class="energy-card-header">
              <div>
                <div class="energy-title">Prag energetic BTC</div>
                <div class="energy-subtitle">Cost energetic estimat de producție</div>
              </div>
              <div class="energy-badge">estimare</div>
            </div>
            <div class="energy-values">
              <div class="energy-item"><span class="energy-label">Miner eficient</span><span class="energy-value">n/a</span></div>
              <div class="energy-item main"><span class="energy-label">Miner mediu</span><span class="energy-value">n/a</span></div>
              <div class="energy-item"><span class="energy-label">Miner scump</span><span class="energy-value">n/a</span></div>
            </div>
            <div class="energy-note">
              Calcul bazat pe dificultatea rețelei, recompensa pe bloc, eficiența echipamentelor și prețul energiei. Nu reprezintă costul total contabil al minerilor.
            </div>
          </div>'''
if old_div in html:
    html = html.replace(old_div, new_div, 1)
else:
    html = html.replace('''          <div id="prod-cost-line" class="live-delta">
            Cost de producție: n/a (așteptăm date despre cost).
          </div>''', new_div, 1)

old_js = '''        // cost de producție – mic / mediu / mare
if (PROD_COST_LINE_EL) {
  let costText = "Cost de producție: n/a (așteptăm date despre cost).";

  if (productionCosts && typeof productionCosts === "object") {
    const cheap = productionCosts.cheap;
    const avg = productionCosts.average;
    const exp = productionCosts.expensive;

    const parts = [];

    if (typeof cheap === "number" && Number.isFinite(cheap)) {
      parts.push("mic ~" + USD_FORMATTER.format(cheap) + " USD");
    }
    if (typeof avg === "number" && Number.isFinite(avg)) {
      parts.push("mediu ~" + USD_FORMATTER.format(avg) + " USD");
    }
    if (typeof exp === "number" && Number.isFinite(exp)) {
      parts.push("mare ~" + USD_FORMATTER.format(exp) + " USD");
    }

    if (parts.length > 0) {
      costText = "Cost de producție : " + parts.join(" · ");
    }
  }

  PROD_COST_LINE_EL.textContent = costText;
}'''
new_js = '''        // prag energetic BTC – cost energetic estimat, nu cost total contabil
        if (PROD_COST_LINE_EL) {
          const renderEnergyCostCard = (costs) => {
            const fmt = (v) => (typeof v === "number" && Number.isFinite(v))
              ? "~" + USD_FORMATTER.format(v) + " USD"
              : "n/a";

            const cheap = costs && typeof costs === "object" ? costs.cheap : null;
            const avg = costs && typeof costs === "object" ? costs.average : null;
            const exp = costs && typeof costs === "object" ? costs.expensive : null;

            PROD_COST_LINE_EL.className = "energy-card" + (!costs ? " loading" : "");
            PROD_COST_LINE_EL.innerHTML = `
              <div class="energy-card-header">
                <div>
                  <div class="energy-title">Prag energetic BTC</div>
                  <div class="energy-subtitle">Cost energetic estimat de producție</div>
                </div>
                <div class="energy-badge">estimare</div>
              </div>
              <div class="energy-values">
                <div class="energy-item"><span class="energy-label">Miner eficient</span><span class="energy-value">${fmt(cheap)}</span></div>
                <div class="energy-item main"><span class="energy-label">Miner mediu</span><span class="energy-value">${fmt(avg)}</span></div>
                <div class="energy-item"><span class="energy-label">Miner scump</span><span class="energy-value">${fmt(exp)}</span></div>
              </div>
              <div class="energy-note">
                Calcul bazat pe dificultatea rețelei, recompensa pe bloc, eficiența echipamentelor și prețul energiei. Nu reprezintă costul total contabil al minerilor.
              </div>`;
          };

          renderEnergyCostCard(productionCosts);
        }'''
if old_js in html:
    html = html.replace(old_js, new_js, 1)
else:
    html = html.replace('Cost de producție: n/a (așteptăm date despre cost).', 'Prag energetic BTC: n/a (așteptăm date despre costul energetic estimat).')

# Actualizăm formulările explicative generate din backend, fără să schimbăm valorile sau logica.
html = html.replace(
    "Prețul este într-o zonă de echilibru față de costul estimat de producție al rețelei.",
    "Prețul este într-o zonă apropiată de pragul energetic mediu estimat al rețelei."
)
html = html.replace(
    "costul estimat de producție al rețelei",
    "pragul energetic mediu estimat al rețelei"
)
html = html.replace(
    "zona costului de producție",
    "zona pragului energetic"
)

if html != original:
    PATH.write_text(html, encoding="utf-8")
    print("[OK] Energy card UI patch applied to", PATH)
else:
    print("[INFO] No changes needed; energy card patch already applied or target block not found.")
