from pathlib import Path

HTML_PATH = Path("btc-swing-strategy") / "mecanism.html"

CSS_MARKER = "/* ACTIVE_STANDARD_REGIME_HIGHLIGHT_CSS */"
JS_MARKER = "// ACTIVE_STANDARD_REGIME_HIGHLIGHT_JS"

CSS_BLOCK = r'''

    /* ACTIVE_STANDARD_REGIME_HIGHLIGHT_CSS */
    .standard-regime-row {
      position: relative;
      border-width: 1px;
    }
    .standard-regime-row.active-standard-regime {
      border-width: 3px;
      transform: translateY(-1px);
      box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.04),
        0 0 24px rgba(148,163,184,0.12);
    }
    .standard-regime-row.active-standard-regime::after {
      content: "DETECTAT ACUM";
      position: absolute;
      top: 7px;
      right: 9px;
      padding: 3px 7px;
      border-radius: 999px;
      font-size: 9px;
      font-weight: 800;
      letter-spacing: .08em;
      text-transform: uppercase;
      background: rgba(15,23,42,0.88);
      color: var(--text-main);
      border: 1px solid rgba(148,163,184,0.22);
    }
    .standard-regime-row.active-standard-regime.regime-family-up {
      border-color: rgba(34,197,94,0.95);
      box-shadow: 0 0 26px rgba(34,197,94,0.16);
    }
    .standard-regime-row.active-standard-regime.regime-family-up::after {
      color: #bbf7d0;
      border-color: rgba(34,197,94,0.55);
    }
    .standard-regime-row.active-standard-regime.regime-family-neutral {
      border-color: rgba(148,163,184,0.95);
      box-shadow: 0 0 24px rgba(148,163,184,0.14);
    }
    .standard-regime-row.active-standard-regime.regime-family-neutral::after {
      color: #e2e8f0;
      border-color: rgba(148,163,184,0.55);
    }
    .standard-regime-row.active-standard-regime.regime-family-down {
      border-color: rgba(248,113,113,0.98);
      box-shadow: 0 0 28px rgba(248,113,113,0.18);
    }
    .standard-regime-row.active-standard-regime.regime-family-down::after {
      color: #fecaca;
      border-color: rgba(248,113,113,0.58);
    }
    body.light-mode .standard-regime-row.active-standard-regime::after {
      background: rgba(255,255,255,0.95);
    }
    @media (max-width: 480px) {
      .standard-regime-row.active-standard-regime::after {
        position: static;
        width: fit-content;
        margin-top: 7px;
        display: inline-flex;
      }
    }
'''

JS_BLOCK = r'''

    // ACTIVE_STANDARD_REGIME_HIGHLIGHT_JS
    function standardRegimeKeyFromCode(code) {
      const c = String(code || '').toLowerCase();
      if (c.startsWith('up_trend_strong')) return 'up_trend_strong';
      if (c.startsWith('up_trend_moderate')) return 'up_trend_moderate';
      if (c.startsWith('range_bias_up')) return 'range_bias_up';
      if (c.startsWith('range_neutral')) return 'range_neutral';
      if (c.startsWith('range_bias_down')) return 'range_bias_down';
      if (c.startsWith('down_trend_strong')) return 'down_trend_strong';
      if (c.startsWith('down_trend_moderate')) return 'down_trend_moderate';
      if (c.startsWith('transition_accumulation')) return 'transition_accumulation';
      return '';
    }

    function standardRegimeFamily(key) {
      if (['up_trend_strong', 'up_trend_moderate', 'range_bias_up'].includes(key)) return 'regime-family-up';
      if (['down_trend_strong', 'down_trend_moderate', 'range_bias_down'].includes(key)) return 'regime-family-down';
      return 'regime-family-neutral';
    }

    function highlightActiveStandardRegime(state) {
      const code = state && state.market_regime && state.market_regime.code;
      const key = standardRegimeKeyFromCode(code);
      const rows = document.querySelectorAll('[data-standard-regime]');
      rows.forEach(row => {
        row.classList.remove('active-standard-regime', 'regime-family-up', 'regime-family-neutral', 'regime-family-down');
        row.removeAttribute('aria-current');
      });
      if (!key) return;
      const active = document.querySelector(`[data-standard-regime="${key}"]`);
      if (!active) return;
      active.classList.add('active-standard-regime', standardRegimeFamily(key));
      active.setAttribute('aria-current', 'true');
    }
'''

REPLACEMENTS = {
    '<div class="history-row">\n        <div class="history-left">\n          <div class="history-date">1. Trend ascendent susținut</div>': '<div class="history-row standard-regime-row" data-standard-regime="up_trend_strong">\n        <div class="history-left">\n          <div class="history-date">1. Trend ascendent susținut</div>',
    '<div class="history-row">\n        <div class="history-left">\n          <div class="history-date">2. Trend ascendent moderat</div>': '<div class="history-row standard-regime-row" data-standard-regime="up_trend_moderate">\n        <div class="history-left">\n          <div class="history-date">2. Trend ascendent moderat</div>',
    '<div class="history-row">\n        <div class="history-left">\n          <div class="history-date">3. Range cu bias pozitiv</div>': '<div class="history-row standard-regime-row" data-standard-regime="range_bias_up">\n        <div class="history-left">\n          <div class="history-date">3. Range cu bias pozitiv</div>',
    '<div class="history-row">\n        <div class="history-left">\n          <div class="history-date">4. Range neutru</div>': '<div class="history-row standard-regime-row" data-standard-regime="range_neutral">\n        <div class="history-left">\n          <div class="history-date">4. Range neutru</div>',
    '<div class="history-row">\n        <div class="history-left">\n          <div class="history-date">5. Range cu bias negativ</div>': '<div class="history-row standard-regime-row" data-standard-regime="range_bias_down">\n        <div class="history-left">\n          <div class="history-date">5. Range cu bias negativ</div>',
    '<div class="history-row">\n        <div class="history-left">\n          <div class="history-date">6. Trend descendent moderat</div>': '<div class="history-row standard-regime-row" data-standard-regime="down_trend_moderate">\n        <div class="history-left">\n          <div class="history-date">6. Trend descendent moderat</div>',
    '<div class="history-row">\n        <div class="history-left">\n          <div class="history-date">7. Trend descendent susținut</div>': '<div class="history-row standard-regime-row" data-standard-regime="down_trend_strong">\n        <div class="history-left">\n          <div class="history-date">7. Trend descendent susținut</div>',
    '<div class="history-row">\n        <div class="history-left">\n          <div class="history-date">8. Regim neutru / tranziție</div>': '<div class="history-row standard-regime-row" data-standard-regime="transition_accumulation">\n        <div class="history-left">\n          <div class="history-date">8. Regim neutru / tranziție</div>',
}


def main():
    html = HTML_PATH.read_text(encoding="utf-8")
    changed = False

    if CSS_MARKER not in html:
        html = html.replace("</style>", CSS_BLOCK + "\n</style>", 1)
        changed = True

    if JS_MARKER not in html:
        insert_before = "\n    const STATE_URL = \"coeziv_state.json\";"
        if insert_before not in html:
            raise RuntimeError("Nu am găsit locul de inserare pentru JS-ul de evidențiere.")
        html = html.replace(insert_before, JS_BLOCK + insert_before, 1)
        changed = True

    for old, new in REPLACEMENTS.items():
        if new in html:
            continue
        if old not in html:
            raise RuntimeError("Nu am găsit rândul standard pentru patch: " + old[:80])
        html = html.replace(old, new, 1)
        changed = True

    if "highlightActiveStandardRegime(data);" not in html:
        marker = "updateMainUi(data);"
        if marker not in html:
            raise RuntimeError("Nu am găsit updateMainUi(data) pentru apelul highlightActiveStandardRegime.")
        html = html.replace(marker, marker + "\n        highlightActiveStandardRegime(data);", 1)
        changed = True

    if changed:
        HTML_PATH.write_text(html, encoding="utf-8")
        print("[OK] mecanism.html actualizat cu evidențierea regimului standard activ.")
    else:
        print("[OK] mecanism.html avea deja patch-ul pentru regimul standard activ.")


if __name__ == "__main__":
    main()
