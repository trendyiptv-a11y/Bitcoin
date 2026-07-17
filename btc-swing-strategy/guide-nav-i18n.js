(function(){
  'use strict';

  var KEY = 'coeziv_btc_lang';
  var lastTacticalState = null;

  function isEn(){
    try{
      return localStorage.getItem(KEY) === 'en' ||
        document.documentElement.lang === 'en' ||
        (document.body && document.body.getAttribute('data-lang') === 'en');
    }catch(e){ return false; }
  }

  function tx(ro, en){ return isEn() ? en : ro; }

  function samePageGuideLink(a){
    var href = (a.getAttribute('href') || '').toLowerCase();
    return href.indexOf('ghid-mecanism.html') !== -1;
  }

  function samePageBackLink(a){
    var href = (a.getAttribute('href') || '').toLowerCase();
    if (href.indexOf('ghid-mecanism.html') !== -1) return false;
    return href.indexOf('mecanism.html') !== -1;
  }

  function restoreAccidentallyHiddenContent(){
    if (!/mecanism\.html/i.test(location.pathname)) return;
    ['.shell', '.card', '.card-secondary', '#daily-cylinder-root', '#coeziv-mini-radar', '#prod-cost-line'].forEach(function(sel){
      Array.prototype.slice.call(document.querySelectorAll(sel)).forEach(function(el){
        if (!el) return;
        if (el.getAttribute('data-cohesivx-hidden-daily-visual') === '1') {
          el.removeAttribute('data-cohesivx-hidden-daily-visual');
          el.removeAttribute('aria-hidden');
        }
        if (el.style && el.style.display === 'none') el.style.display = '';
      });
    });
  }

  function normalizeText(value){
    return String(value || '').replace(/\s+/g, ' ').trim();
  }

  function translateLeafText(from, to){
    var target = normalizeText(from);
    Array.prototype.slice.call(document.querySelectorAll('body *')).forEach(function(el){
      if (!el || el.children.length) return;
      var tag = (el.tagName || '').toLowerCase();
      if (tag === 'script' || tag === 'style' || tag === 'textarea') return;
      if (normalizeText(el.textContent) === target) el.textContent = to;
    });
  }

  function translateFearGreedFragments(){
    var pairs = [
      ['Indice combinat (structură + tactică)', 'Combined index (structure + tactic)'],
      ['Neutru tensionat', 'Tense neutral'],
      ['Tensiune', 'Tension'],
      ['Degradare profundă · zona istoric modelată neatinsă', 'Deep degradation · modeled historical zone not reached'],
      ['Nu există panică, dar contextul structural rămâne tensionat. Confirmarea direcției trebuie așteptată.', 'No panic, but the structural context remains tense. Direction confirmation should be awaited.'],
      ['Sentiment de piață derivat din mecanismul coeziv (structură + semnal + volatilitate), fără date sociale sau externe.', 'Market sentiment derived from the cohesive mechanism: structure, signal and volatility, without social or external data.']
    ];
    pairs.forEach(function(pair){
      if (isEn()) translateLeafText(pair[0], pair[1]);
      else translateLeafText(pair[1], pair[0]);
    });
  }

  function applyGuideText(){
    restoreAccidentallyHiddenContent();
    var english = isEn();
    var guide = document.getElementById('guide-link') ||
      Array.prototype.slice.call(document.querySelectorAll('a')).find(samePageGuideLink);

    if (guide) {
      guide.textContent = english ? 'Mechanism guide' : 'Ghid mecanism';
      guide.setAttribute('aria-label', english ? 'Open mechanism guide' : 'Deschide ghidul mecanismului');
    }

    var links = document.querySelectorAll('a');
    for (var i = 0; i < links.length; i++) {
      if (samePageBackLink(links[i])) {
        links[i].textContent = english ? '← Back to mechanism' : '← Înapoi la mecanism';
        links[i].setAttribute('aria-label', english ? 'Back to mechanism' : 'Înapoi la mecanism');
      }
    }

    document.documentElement.lang = english ? 'en' : 'ro';
    translateFearGreedFragments();
    if (lastTacticalState) renderTactical(lastTacticalState);
    reverseSignalLegend();
  }

  function reverseSignalLegend(){
    var grid = document.querySelector('.cxlg-grid');
    if (!grid) return;
    ['risk', 'sell', 'attention', 'wait', 'accumulate', 'buy'].forEach(function(key){
      var item = grid.querySelector('.cxlg-item.' + key);
      if (item) grid.appendChild(item);
    });
  }

  function addLegendReverseObserver(){
    reverseSignalLegend();
    setTimeout(reverseSignalLegend, 250);
    setTimeout(reverseSignalLegend, 800);
    setTimeout(reverseSignalLegend, 1500);
    setInterval(reverseSignalLegend, 900);
    if (!window.MutationObserver) return;
    var root = document.getElementById('daily-cylinder-root') || document.body;
    if (!root) return;
    var obs = new MutationObserver(function(){
      window.requestAnimationFrame(function(){
        restoreAccidentallyHiddenContent();
        translateFearGreedFragments();
        ensureTacticalSlot();
        reverseSignalLegend();
      });
    });
    obs.observe(root, { childList:true, subtree:true });
  }

  function addTacticalStyle(){
    if (document.getElementById('cohesivx-tactical-slot-style')) return;
    var style = document.createElement('style');
    style.id = 'cohesivx-tactical-slot-style';
    style.textContent = ''+
      '#tactical-range-slot{position:relative;z-index:1;text-align:left;}'+
      '#tactical-range-slot .tr-title{font-size:13px;letter-spacing:.18em;text-transform:uppercase;color:var(--text-muted);margin-bottom:8px;}'+
      '#tactical-range-slot .tr-sub{font-size:12px;color:var(--text-soft);line-height:1.4;margin-bottom:12px;}'+
      '#tactical-range-slot .tr-box{padding:12px;border-radius:16px;border:1px solid rgba(56,189,248,.28);background:rgba(8,47,73,.22);}'+
      '#tactical-range-slot .tr-row{display:flex;align-items:center;justify-content:space-between;gap:12px;}'+
      '#tactical-range-slot .tr-label{font-size:12px;color:var(--text-muted);line-height:1.35;font-weight:650;}'+
      '#tactical-range-slot .tr-signal{min-width:90px;text-align:center;border-radius:999px;padding:8px 12px;font-size:16px;font-weight:950;letter-spacing:.12em;text-transform:uppercase;border:1px solid rgba(148,163,184,.38);}'+
      '#tactical-range-slot .tr-signal.buy{color:#bbf7d0;border-color:rgba(34,197,94,.62);background:rgba(22,101,52,.34);}'+
      '#tactical-range-slot .tr-signal.sell{color:#fecaca;border-color:rgba(248,113,113,.68);background:rgba(127,29,29,.34);}'+
      '#tactical-range-slot .tr-signal.wait{color:#e2e8f0;border-color:rgba(148,163,184,.48);background:rgba(71,85,105,.28);}'+
      '#tactical-range-slot .tr-message{margin-top:10px;font-size:13px;line-height:1.45;color:var(--text-main);}'+
      '#tactical-range-slot .tr-meta{margin-top:7px;font-size:10.5px;line-height:1.35;color:var(--text-soft);}'+
      '#tactical-range-slot .tr-legend{margin-top:10px;padding-top:9px;padding-right:116px;border-top:1px solid rgba(148,163,184,.16);font-size:10.2px;line-height:1.55;color:var(--text-soft);}'+
      '#tactical-range-slot .tr-legend strong{display:block;margin-top:4px;color:var(--text-main);font-weight:750;}'+
      'body.light-mode #tactical-range-slot .tr-box{background:rgba(224,242,254,.70);border-color:rgba(14,165,233,.28);}'+
      '@media(max-width:430px){#tactical-range-slot .tr-row{align-items:flex-start;flex-direction:column;}#tactical-range-slot .tr-signal{min-width:96px;}#tactical-range-slot .tr-legend{padding-right:132px;padding-bottom:42px;font-size:10px;}}';
    document.head.appendChild(style);
  }

  function findStandardRegimeCard(){
    var cards = Array.prototype.slice.call(document.querySelectorAll('.card.card-secondary'));
    for (var i = 0; i < cards.length; i++) {
      var text = (cards[i].textContent || '').toUpperCase();
      if (text.indexOf('REGIMURI STANDARD DE PIAȚĂ') !== -1 || text.indexOf('REGIMURI STANDARD DE PIATA') !== -1 || cards[i].getAttribute('data-cohesivx-tactical-host') === '1') return cards[i];
    }
    return null;
  }

  function ensureTacticalSlot(){
    if (!/mecanism\.html/i.test(location.pathname)) return null;
    addTacticalStyle();
    var card = findStandardRegimeCard();
    if (!card) return document.getElementById('tactical-range-slot');
    card.style.display = '';
    card.removeAttribute('aria-hidden');
    if (card.getAttribute('data-cohesivx-tactical-host') !== '1') {
      card.setAttribute('data-cohesivx-tactical-host','1');
      card.innerHTML = '<div id="tactical-range-slot">'+
        '<div class="tr-title"></div>'+
        '<div class="tr-sub"></div>'+
        '<div class="tr-box"><div class="tr-row"><div class="tr-label"></div><div class="tr-signal wait">WAIT</div></div><div class="tr-message"></div><div class="tr-meta"></div><div class="tr-legend"></div></div>'+
        '</div>';
    }
    return document.getElementById('tactical-range-slot');
  }

  function safeTacticalSignal(raw){
    var s = String(raw || 'WAIT').toUpperCase();
    if (s !== 'BUY' && s !== 'SELL') s = 'WAIT';
    return s;
  }

  function confidenceLegend(confidence){
    var c = String(confidence || '').toLowerCase();
    if (c === 'medium') return tx('medium = range stabil sau comprimat.', 'medium = stable or compressed range.');
    if (c === 'high') return tx('high = confirmare tactică puternică.', 'high = strong tactical confirmation.');
    return tx('low = range în mișcare; semnal prudent.', 'low = moving range; cautious signal.');
  }

  function tacticalLegend(confidence){
    return tx(
      '<strong>Legendă</strong>BUY = aproape de baza range-ului.<br>SELL = aproape de vârful range-ului.<br>WAIT = fără avantaj tactic clar.<strong>Încredere</strong>'+confidenceLegend(confidence),
      '<strong>Legend</strong>BUY = near the range low.<br>SELL = near the range high.<br>WAIT = no clear tactical edge.<strong>Confidence</strong>'+confidenceLegend(confidence)
    );
  }

  function renderTactical(data){
    lastTacticalState = data || lastTacticalState || {};
    var slot = ensureTacticalSlot();
    if (!slot) return;
    data = lastTacticalState || {};
    var sig = safeTacticalSignal(data.tactical_signal);
    var css = sig === 'BUY' ? 'buy' : (sig === 'SELL' ? 'sell' : 'wait');
    var msg = isEn() ? (data.message_en || 'Wait: tactical range is updating.') : (data.message_ro || 'Așteaptă: Tactical Range se actualizează.');
    var updated = data.updated_at ? String(data.updated_at).replace('T',' ').replace('+00:00',' UTC') : '–';
    var confidence = data.confidence || '–';
    var title = slot.querySelector('.tr-title');
    var sub = slot.querySelector('.tr-sub');
    var label = slot.querySelector('.tr-label');
    var signal = slot.querySelector('.tr-signal');
    var message = slot.querySelector('.tr-message');
    var meta = slot.querySelector('.tr-meta');
    var legend = slot.querySelector('.tr-legend');
    if (title) title.textContent = tx('Semnal tactic range','Tactical range signal');
    if (sub) sub.textContent = tx('Citește poziția prețului în range-ul recent.', 'Reads price position inside the recent range.');
    if (label) label.textContent = tx('Semnal tactic', 'Tactical signal');
    if (signal) {
      signal.className = 'tr-signal ' + css;
      signal.textContent = sig;
    }
    if (message) message.textContent = msg;
    if (meta) meta.textContent = tx('Actualizat: ','Updated: ') + updated + ' · ' + tx('încredere: ','confidence: ') + confidence;
    if (legend) legend.innerHTML = tacticalLegend(confidence);
  }

  function fetchTactical(){
    if (!/mecanism\.html/i.test(location.pathname)) return;
    ensureTacticalSlot();
    fetch('./tactical_range.json?t=' + Date.now(), {cache:'no-store'})
      .then(function(r){ if(!r.ok) throw new Error('HTTP '+r.status); return r.json(); })
      .then(renderTactical)
      .catch(function(){
        renderTactical({
          tactical_signal:'WAIT',
          confidence:'low',
          message_ro:'Așteaptă: fișierul tactical_range.json nu este disponibil momentan.',
          message_en:'Wait: tactical_range.json is not available right now.'
        });
      });
    setTimeout(ensureTacticalSlot, 500);
    setTimeout(ensureTacticalSlot, 1500);
  }

  function start(){
    restoreAccidentallyHiddenContent();
    applyGuideText();
    fetchTactical();
    addLegendReverseObserver();
    setTimeout(restoreAccidentallyHiddenContent, 200);
    setTimeout(translateFearGreedFragments, 350);
    setTimeout(fetchTactical, 900);
    setTimeout(fetchTactical, 1800);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }

  setInterval(applyGuideText, 700);
  setInterval(fetchTactical, 5 * 60 * 1000);
})();
