from pathlib import Path

HTML = Path('btc-swing-strategy/mecanism.html')
HTML_MARK = '<!-- PREMIUM_PHASE2_CONTEXT_START -->'
JS_MARK = '// PREMIUM_PHASE2_CONTEXT_JS'

CONTEXT_HTML = '''
          <!-- PREMIUM_PHASE2_CONTEXT_START -->
          <div class="premium-status-strip" id="premium-status-strip">
            <span class="premium-status-pill" id="premium-pill-structure">Structură: –</span>
            <span class="premium-status-pill" id="premium-pill-participation">Participare: –</span>
            <span class="premium-status-pill" id="premium-pill-risk">Risc: –</span>
            <span class="premium-status-pill" id="premium-pill-sentiment">Sentiment: –</span>
          </div>

          <div class="cohesiv-metric-grid" id="premium-context-grid">
            <div class="cohesiv-metric-tile">
              <div class="cohesiv-metric-label">Structură</div>
              <div class="cohesiv-metric-value" id="premium-structure-value">–</div>
              <div class="cohesiv-metric-note" id="premium-structure-note">Starea generală citită din mecanism.</div>
            </div>
            <div class="cohesiv-metric-tile">
              <div class="cohesiv-metric-label">Participare</div>
              <div class="cohesiv-metric-value" id="premium-participation-value">–</div>
              <div class="cohesiv-metric-note" id="premium-participation-note">Coeziunea participanților.</div>
            </div>
            <div class="cohesiv-metric-tile">
              <div class="cohesiv-metric-label">Fereastră risc</div>
              <div class="cohesiv-metric-value" id="premium-risk-value">–</div>
              <div class="cohesiv-metric-note" id="premium-risk-note">Persistență și confirmare istorică.</div>
            </div>
            <div class="cohesiv-metric-tile">
              <div class="cohesiv-metric-label">Fear &amp; Greed</div>
              <div class="cohesiv-metric-value" id="premium-fg-value">–</div>
              <div class="cohesiv-metric-note" id="premium-fg-note">Tensiune emoțională structurală.</div>
            </div>
          </div>
          <!-- PREMIUM_PHASE2_CONTEXT_END -->
'''

JS = '''
    // PREMIUM_PHASE2_CONTEXT_JS
    function premiumText(id) {
      const el = document.getElementById(id);
      return el ? (el.textContent || '').replace(/\\s+/g, ' ').trim() : '';
    }
    function premiumShort(value, maxLen) {
      const text = String(value || '').trim();
      if (!text) return '–';
      return text.length > maxLen ? text.slice(0, maxLen - 1).trim() + '…' : text;
    }
    function premiumSet(id, value, maxLen) {
      const el = document.getElementById(id);
      if (el) el.textContent = premiumShort(value, maxLen || 64);
    }
    function updatePremiumContextGrid() {
      const structure = premiumText('signal-label').replace(/^Context:\\s*/i, '') || 'n/a';
      const regime = premiumText('regime-line').replace(/^Regim de piață:\\s*/i, '') || 'n/a';
      const participationScore = premiumText('participation-cohesion-score');
      const participationLabel = premiumText('participation-cohesion-pill');
      const riskLabel = premiumText('risk-window-pill');
      const riskRate = premiumText('risk-window-rate');
      const riskStreak = premiumText('risk-window-streak');
      const fgScore = premiumText('fg-score-value');
      const fgZone = premiumText('fg-score-zone');
      const fgTension = premiumText('fg-tension-value');

      premiumSet('premium-pill-structure', 'Structură: ' + structure, 32);
      premiumSet('premium-pill-participation', 'Participare: ' + (participationScore || participationLabel), 32);
      premiumSet('premium-pill-risk', 'Risc: ' + riskLabel, 32);
      premiumSet('premium-pill-sentiment', 'Sentiment: ' + (fgScore || fgZone), 32);

      premiumSet('premium-structure-value', structure, 42);
      premiumSet('premium-structure-note', regime, 64);
      premiumSet('premium-participation-value', participationScore || participationLabel, 42);
      premiumSet('premium-participation-note', participationLabel || 'Așteptăm date.', 64);
      premiumSet('premium-risk-value', riskLabel || 'n/a', 42);
      premiumSet('premium-risk-note', [riskRate, riskStreak].filter(Boolean).join(' · ') || 'Așteptăm date.', 64);
      premiumSet('premium-fg-value', fgScore ? (fgScore + ' / 100') : fgZone, 42);
      premiumSet('premium-fg-note', [fgZone, fgTension ? ('tensiune ' + fgTension) : ''].filter(Boolean).join(' · ') || 'Așteptăm date.', 64);
    }
    function initPremiumContextGrid() {
      updatePremiumContextGrid();
      setInterval(updatePremiumContextGrid, 1200);
    }
'''

s = HTML.read_text(encoding='utf-8')

if HTML_MARK not in s:
    marker = '          <div id="signal-prob" class="live-delta">'
    s = s.replace(marker, CONTEXT_HTML + '\n' + marker)

if JS_MARK not in s:
    marker = '    const STATE_URL = "coeziv_state.json";'
    s = s.replace(marker, JS + '\n' + marker)

if 'initPremiumContextGrid();' not in s:
    if 'initThemeToggle();' in s:
        s = s.replace('initThemeToggle();', 'initThemeToggle();\n    initPremiumContextGrid();')
    else:
        s = s.replace('</script>', '    initPremiumContextGrid();\n</script>')

HTML.write_text(s, encoding='utf-8')
