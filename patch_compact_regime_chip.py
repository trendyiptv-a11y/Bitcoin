from pathlib import Path
import re

path = Path('btc-swing-strategy/mecanism.html')
s = path.read_text(encoding='utf-8')

css_marker = '/* COMPACT_REGIME_CHIP_CSS */'
css = '''
    /* COMPACT_REGIME_CHIP_CSS */
    .card:first-of-type {
      position: relative;
      overflow: hidden;
    }
    #regime-line.regime-chip {
      position: absolute;
      top: 30px;
      right: 26px;
      width: 58px;
      height: 58px;
      min-width: 58px;
      max-width: 58px;
      margin: 0;
      padding: 0;
      border-radius: 999px;
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
      font-size: 10px;
      line-height: 1.05;
      font-weight: 850;
      letter-spacing: .06em;
      text-transform: uppercase;
      white-space: pre-line;
      overflow: hidden;
      z-index: 3;
      color: #bae6fd;
      background: rgba(15,23,42,0.78);
      border: 1px solid rgba(56,189,248,0.28);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 0 18px rgba(56,189,248,0.08);
    }
    body.light-mode #regime-line.regime-chip {
      color: #075985;
      background: rgba(248,250,252,0.88);
      border-color: rgba(14,165,233,0.32);
    }
    @media (max-width: 480px) {
      #regime-line.regime-chip {
        top: 28px;
        right: 20px;
        width: 54px;
        height: 54px;
        min-width: 54px;
        max-width: 54px;
        font-size: 9px;
      }
      .card:first-of-type .asset-row {
        padding-right: 66px;
      }
      .card:first-of-type .signal-chip {
        max-width: calc(100vw - 210px);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
    }
    @media (max-width: 380px) {
      #regime-line.regime-chip {
        right: 14px;
        width: 50px;
        height: 50px;
        min-width: 50px;
        max-width: 50px;
        font-size: 8px;
      }
      .card:first-of-type .asset-row {
        padding-right: 58px;
      }
    }
'''
css_pattern = r"\n\s*/\* COMPACT_REGIME_CHIP_CSS \*/\n.*?(?=\n\s*/\* PREMIUM_UI_REFRESH_END \*/|\n\s*</style>)"
if css_marker in s:
    s = re.sub(css_pattern, '\n' + css.rstrip(), s, count=1, flags=re.S)
else:
    s = s.replace('  </style>', css + '\n  </style>')

html_old = '<div id="regime-line" class="live-delta">'
html_new = '<div id="regime-line" class="live-delta regime-chip" title="Regim de piață: n/a">'
s = s.replace(html_old, html_new)

js_marker = '// COMPACT_REGIME_CHIP_JS'
js = '''
    // COMPACT_REGIME_CHIP_JS
    function compactRegimeLabel(fullText) {
      const t = String(fullText || '').toLowerCase();
      if (t.includes('range') && t.includes('pozitiv')) return 'RANGE+';
      if (t.includes('range') && t.includes('negativ')) return 'RANGE−';
      if (t.includes('range')) return 'RANGE';
      if (t.includes('ascendent')) return 'TREND+';
      if (t.includes('descendent')) return 'TREND−';
      if (t.includes('tranzi')) return 'TRANZ';
      if (t.includes('neutru')) return 'NEUTRU';
      if (t.includes('n/a')) return 'N/A';
      return 'REGIM';
    }
    function applyCompactRegimeChip() {
      const el = document.getElementById('regime-line');
      if (!el) return;
      el.classList.add('regime-chip');
      const current = el.textContent || '';
      const stored = el.getAttribute('data-full-regime') || '';
      const source = current && !['RANGE+','RANGE−','RANGE','TREND+','TREND−','TRANZ','NEUTRU','N/A','REGIM'].includes(current.trim()) ? current : stored;
      const full = source || current || 'n/a';
      const compact = compactRegimeLabel(full);
      el.setAttribute('data-full-regime', full);
      el.title = full;
      if (current.trim() !== compact) el.textContent = compact;
    }
    function startCompactRegimeChipSync() {
      applyCompactRegimeChip();
      const el = document.getElementById('regime-line');
      if (!el) return;
      const observer = new MutationObserver(function(){
        window.requestAnimationFrame(applyCompactRegimeChip);
      });
      observer.observe(el, { childList: true, characterData: true, subtree: true, attributes: true });
      setInterval(applyCompactRegimeChip, 700);
    }
'''
pattern = r"\n\s*// COMPACT_REGIME_CHIP_JS\n\s*function compactRegimeLabel\(fullText\) \{.*?\n\s*function startCompactRegimeChipSync\(\) \{.*?\n\s*\}\n"
if js_marker in s:
    s = re.sub(pattern, '\n' + js, s, count=1, flags=re.S)
else:
    s = s.replace('    const STATE_URL = "coeziv_state.json";', js + '\n    const STATE_URL = "coeziv_state.json";')

s = s.replace('initThemeToggle();\n    applyCompactRegimeChip();', 'initThemeToggle();\n    startCompactRegimeChipSync();')
if 'startCompactRegimeChipSync();' not in s:
    s = s.replace('initThemeToggle();', 'initThemeToggle();\n    startCompactRegimeChipSync();')

path.write_text(s, encoding='utf-8')
