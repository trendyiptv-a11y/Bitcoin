(function(){
  'use strict';

  var LANG_KEY = 'coeziv_btc_lang';
  var DATA_URL = './tactical_range.json';
  var cardId = 'cohesivx-tactical-range-card';

  function isMonitorPage(){ return /mecanism\.html/i.test(location.pathname); }
  function isEn(){
    try{
      return localStorage.getItem(LANG_KEY) === 'en' ||
        document.documentElement.lang === 'en' ||
        (document.body && document.body.getAttribute('data-lang') === 'en');
    }catch(e){ return false; }
  }
  function tx(ro,en){ return isEn() ? en : ro; }

  function addStyle(){
    if (document.getElementById('cohesivx-tactical-range-style')) return;
    var style = document.createElement('style');
    style.id = 'cohesivx-tactical-range-style';
    style.textContent = ''+
      '#'+cardId+'{display:block!important;clear:both!important;width:100%!important;max-width:100%!important;min-width:0!important;flex:0 0 100%!important;margin:10px 0 14px!important;padding:12px 12px 13px;border-radius:18px;border:1px solid rgba(56,189,248,.26);background:radial-gradient(circle at 0 0,rgba(56,189,248,.10),transparent 48%),linear-gradient(180deg,rgba(15,23,42,.70),rgba(15,23,42,.44));box-shadow:inset 0 1px 0 rgba(255,255,255,.035);text-align:left;}'+
      '#'+cardId+' .tr-title{font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--text-soft);margin-bottom:8px;}'+
      '#'+cardId+' .tr-main{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:10px 11px;border-radius:14px;border:1px solid rgba(56,189,248,.25);background:rgba(56,189,248,.065);}'+
      '#'+cardId+' .tr-label{font-size:12px;line-height:1.35;color:var(--text-muted);font-weight:650;}'+
      '#'+cardId+' .tr-signal{min-width:82px;text-align:center;border-radius:999px;padding:7px 10px;font-size:14px;font-weight:950;letter-spacing:.10em;text-transform:uppercase;border:1px solid rgba(148,163,184,.35);}'+
      '#'+cardId+' .tr-signal.buy{color:#bbf7d0;border-color:rgba(34,197,94,.62);background:rgba(22,101,52,.34);}'+
      '#'+cardId+' .tr-signal.sell{color:#fecaca;border-color:rgba(248,113,113,.68);background:rgba(127,29,29,.34);}'+
      '#'+cardId+' .tr-signal.wait{color:#e2e8f0;border-color:rgba(148,163,184,.48);background:rgba(71,85,105,.28);}'+
      '#'+cardId+' .tr-message{margin-top:9px;font-size:12.5px;line-height:1.45;color:var(--text-main);}'+
      '#'+cardId+' .tr-meta{margin-top:6px;font-size:10.5px;line-height:1.35;color:var(--text-soft);}'+
      'body.light-mode #'+cardId+'{background:linear-gradient(180deg,rgba(255,255,255,.90),rgba(248,250,252,.74));border-color:rgba(14,165,233,.25);}'+
      'body.light-mode #'+cardId+' .tr-main{background:rgba(14,165,233,.075);border-color:rgba(14,165,233,.26);}'+
      '@media(max-width:430px){#'+cardId+' .tr-main{align-items:flex-start;flex-direction:column;}#'+cardId+' .tr-signal{min-width:96px;}}';
    document.head.appendChild(style);
  }

  function findMainSignalAnchor(){
    var cards = Array.prototype.slice.call(document.querySelectorAll('.card'));
    for (var i = 0; i < cards.length; i++) {
      var text = (cards[i].textContent || '').toUpperCase();
      if (text.indexOf('BITCOIN') !== -1 && text.indexOf('PREȚ LIVE') !== -1) {
        return cards[i];
      }
    }
    return cards.length ? cards[0] : null;
  }

  function fallbackAnchor(){
    return document.getElementById('structural-confirmation-card') ||
      document.getElementById('coeziv-mini-radar') ||
      document.getElementById('prod-cost-line');
  }

  function createCard(){
    if (!isMonitorPage()) return null;
    addStyle();
    var card = document.getElementById(cardId);
    var anchor = findMainSignalAnchor() || fallbackAnchor();
    if (!anchor || !anchor.parentNode) return card || null;

    if (!card) {
      card = document.createElement('div');
      card.id = cardId;
      card.innerHTML = '<div class="tr-title"></div><div class="tr-main"><div class="tr-label"></div><div class="tr-signal wait">WAIT</div></div><div class="tr-message"></div><div class="tr-meta"></div>';
    }

    if (card.parentNode !== anchor.parentNode || card.previousElementSibling !== anchor) {
      anchor.parentNode.insertBefore(card, anchor.nextElementSibling);
    }
    return card;
  }

  function safeSignal(raw){
    var s = String(raw || 'WAIT').toUpperCase();
    if (s !== 'BUY' && s !== 'SELL') s = 'WAIT';
    return s;
  }

  function render(data){
    var card = createCard();
    if (!card) return;
    data = data || {};
    var sig = safeSignal(data.tactical_signal);
    var css = sig === 'BUY' ? 'buy' : (sig === 'SELL' ? 'sell' : 'wait');
    var msg = isEn() ? (data.message_en || 'Wait: tactical range is updating.') : (data.message_ro || 'Așteaptă: tactical range se actualizează.');
    var updated = data.updated_at ? String(data.updated_at).replace('T',' ').replace('+00:00',' UTC') : '–';
    var confidence = data.confidence || '–';

    var title = card.querySelector('.tr-title');
    var label = card.querySelector('.tr-label');
    var signal = card.querySelector('.tr-signal');
    var message = card.querySelector('.tr-message');
    var meta = card.querySelector('.tr-meta');

    if (title) title.textContent = tx('Semnal tactic range','Tactical range signal');
    if (label) label.textContent = tx('Strat secundar: backendul calculează, front-ul afișează doar semnalul.','Secondary layer: backend calculates, frontend only displays the signal.');
    if (signal) {
      signal.className = 'tr-signal ' + css;
      signal.textContent = sig;
    }
    if (message) message.textContent = msg;
    if (meta) meta.textContent = tx('Actualizat: ','Updated: ') + updated + ' · ' + tx('încredere: ','confidence: ') + confidence;
  }

  function load(){
    if (!isMonitorPage()) return;
    createCard();
    fetch(DATA_URL + '?t=' + Date.now(), {cache:'no-store'})
      .then(function(r){ if(!r.ok) throw new Error('HTTP '+r.status); return r.json(); })
      .then(render)
      .catch(function(){
        render({
          tactical_signal:'WAIT',
          confidence:'low',
          message_ro:'Așteaptă: fișierul tactical_range.json nu este disponibil momentan.',
          message_en:'Wait: tactical_range.json is not available right now.'
        });
      });
    setTimeout(createCard, 400);
    setTimeout(createCard, 1200);
    setTimeout(createCard, 2200);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', load); else load();
  setInterval(load, 5 * 60 * 1000);
  window.CohesivXTacticalRangeDisplay = { load: load, render: render };
})();