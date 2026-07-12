/* CohesivX visual summary fix v9 — hides old structural card, preserves opened details, and displays cohesive FG labels */
(function () {
  'use strict';

  var desiredOpen = { ctx: false, full: false };
  var lastFgRefreshAt = 0;

  function getDetails() {
    return Array.prototype.slice.call(document.querySelectorAll('#cx-visual-summary details.cxv-narrative'));
  }

  function restoreOpenStates() {
    var details = getDetails();
    if (details[0]) details[0].open = !!desiredOpen.ctx;
    if (details[1]) details[1].open = !!desiredOpen.full;
  }

  function markClicks() {
    document.addEventListener('click', function (ev) {
      var summary = ev.target && ev.target.closest && ev.target.closest('#cx-visual-summary details.cxv-narrative summary');
      if (!summary) return;
      var detail = summary.parentElement;
      var details = getDetails();
      var index = details.indexOf(detail);
      setTimeout(function () {
        var fresh = getDetails();
        var current = fresh[index];
        if (index === 0 && current) desiredOpen.ctx = current.open;
        if (index === 1 && current) desiredOpen.full = current.open;
        restoreOpenStates();
      }, 60);
    }, true);
  }

  function hideOldStructuralConfirmation() {
    var patterns = [
      /STRUCTURAL\s+CONFIRMATION/i,
      /CONFIRMARE\s+STRUCTURAL[ĂA]/i,
      /In\s+similar\s+historical\s+contexts/i,
      /În\s+contexte\s+istorice\s+similare/i,
      /Statistical\s+base/i,
      /Baz[ăa]\s+statistic[ăa]/i
    ];
    var nodes = Array.prototype.slice.call(document.querySelectorAll('div,section,article'));
    var candidates = [];
    nodes.forEach(function (node) {
      if (!node) return;
      if (node.id === 'cx-visual-summary') return;
      if (node.closest('#cx-visual-summary')) return;
      if (node.dataset && node.dataset.cxvHidden === '1') return;
      var text = (node.textContent || '').replace(/\s+/g, ' ').trim();
      if (!text) return;
      var hits = patterns.reduce(function (n, rx) { return n + (rx.test(text) ? 1 : 0); }, 0);
      if (hits < 2) return;
      if (text.length > 2400) return;
      candidates.push({ node: node, len: text.length, hits: hits });
    });
    candidates.sort(function (a, b) {
      if (b.hits !== a.hits) return b.hits - a.hits;
      return a.len - b.len;
    });
    candidates.slice(0, 8).forEach(function (item) {
      item.node.style.display = 'none';
      item.node.setAttribute('aria-hidden', 'true');
      item.node.dataset.cxvHidden = '1';
    });
  }

  function hideLongMainText() {
    ['message', 'model-price-explanation'].forEach(function (id) {
      var el = document.getElementById(id);
      if (el) {
        el.style.display = 'none';
        el.setAttribute('aria-hidden', 'true');
      }
    });
  }

  function observeVisualSummary() {
    var root = document.getElementById('cx-visual-summary');
    if (!root || root.dataset.cxvObserved === '1') return;
    root.dataset.cxvObserved = '1';
    var observer = new MutationObserver(function () {
      setTimeout(restoreOpenStates, 20);
      setTimeout(restoreOpenStates, 120);
    });
    observer.observe(root, { childList: true, subtree: true });
  }

  function fgTitle(zone) {
    switch (String(zone || '').toLowerCase()) {
      case 'extreme_fear': return 'Extreme Fear';
      case 'fear': return 'Fear';
      case 'greed': return 'Greed';
      case 'extreme_greed': return 'Extreme Greed';
      default: return 'Neutral';
    }
  }

  function fgCssZone(zone) {
    var z = String(zone || '').toLowerCase();
    if (z === 'optimism_tensionat') return 'neutral';
    if (z === 'greed_fragil') return 'greed';
    if (z === 'neutru_tensionat') return 'neutral';
    if (['extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed'].indexOf(z) >= 0) return z;
    return 'neutral';
  }

  function fallbackCohesiveFg(fg, stateLike) {
    var score = Number(fg && fg.combined);
    var rawZone = String((fg && fg.combined_zone) || 'neutral').toLowerCase();
    var signal = String((stateLike && stateLike.signal) || '').toLowerCase();
    var flowBias = String((stateLike && stateLike.flow_bias) || '').toLowerCase();
    var flowStrength = String((stateLike && stateLike.flow_strength) || '').toLowerCase();
    var deviation = Number(stateLike && stateLike.model_price_deviation);
    var structural = (stateLike && stateLike.structural_confirmation) || {};
    var regime = String(structural.regime || '').toLowerCase();
    var history = Array.isArray(stateLike && stateLike.signal_history) ? stateLike.signal_history : [];
    var growthCount = history.filter(function (row) { return row && String(row.signal || '').toLowerCase() === 'long'; }).length;
    var structuralTension = regime.indexOf('bear') >= 0 || (Number.isFinite(deviation) && deviation <= -0.20);
    var weakNeutralFlow = flowBias === 'neutru' && flowStrength === 'slab';
    var noGrowth = growthCount === 0 && (signal === 'flat' || signal === 'neutral' || !signal);

    if (rawZone === 'greed' && Number.isFinite(score) && score < 65 && (structuralTension || (weakNeutralFlow && noGrowth))) {
      return {
        label: 'Optimism tensionat',
        zone: 'optimism_tensionat',
        description: 'Apetit de risc ușor, dar fără confirmare structurală. Piața nu arată panică, însă fluxul este slab, contextele de creștere lipsesc, iar structura rămâne nereparată.'
      };
    }
    if (rawZone === 'greed' && (structuralTension || weakNeutralFlow || noGrowth)) {
      return {
        label: 'Greed fragil',
        zone: 'greed_fragil',
        description: 'Există apetit de risc, dar confirmarea este incompletă. Greed-ul devine sănătos doar dacă este susținut de flux pozitiv persistent, participare coezivă și reparare structurală.'
      };
    }
    return null;
  }

  function renderCohesiveFg(fg, stateLike) {
    if (!fg || typeof fg.combined !== 'number' || !Number.isFinite(fg.combined)) return;
    var zoneEl = document.getElementById('fg-score-zone');
    var descEl = document.getElementById('fg-description');
    if (!zoneEl) return;
    var display = null;
    if (fg.cohesive_label || fg.cohesive_description || fg.cohesive_zone) {
      display = {
        label: fg.cohesive_label || fgTitle(fg.combined_zone),
        zone: fg.cohesive_zone || fg.combined_zone || 'neutral',
        description: fg.cohesive_description || ''
      };
    } else {
      display = fallbackCohesiveFg(fg, stateLike || window.COHESIVX_LAST_STATE || null);
    }
    if (!display) return;
    zoneEl.textContent = display.label;
    zoneEl.className = 'fg-zone-pill fg-zone-' + fgCssZone(display.zone);
    if (descEl && display.description) descEl.textContent = display.description;
  }

  function installCohesiveFearGreedDisplay() {
    if (window.__cohesivxFgDisplayInstalled) return;
    if (typeof window.updateFGCard !== 'function') return;
    window.__cohesivxFgDisplayInstalled = true;
    var original = window.updateFGCard;
    window.updateFGCard = function (fg) {
      original(fg);
      renderCohesiveFg(fg, window.COHESIVX_LAST_STATE || null);
    };
  }

  function rememberLatestState() {
    if (window.__cohesivxStateFetchHookInstalled) return;
    if (typeof window.fetch !== 'function') return;
    window.__cohesivxStateFetchHookInstalled = true;
    var originalFetch = window.fetch;
    window.fetch = function () {
      var args = arguments;
      var url = args && args[0];
      var urlText = typeof url === 'string' ? url : (url && url.url ? String(url.url) : '');
      return originalFetch.apply(this, args).then(function (response) {
        try {
          if (urlText.indexOf('coeziv_state.json') >= 0 && response && typeof response.clone === 'function') {
            response.clone().json().then(function (data) {
              if (data && typeof data === 'object') {
                window.COHESIVX_LAST_STATE = data;
                if (data.fg) renderCohesiveFg(data.fg, data);
              }
            }).catch(function () {});
          }
        } catch (_) {}
        return response;
      });
    };
  }

  function refreshFgFromState() {
    var now = Date.now();
    if (now - lastFgRefreshAt < 5000) return;
    if (typeof fetch !== 'function') return;
    lastFgRefreshAt = now;
    fetch('coeziv_state.json?t=' + now, { cache: 'no-store' })
      .then(function (res) { return res && res.ok ? res.json() : null; })
      .then(function (data) {
        if (!data || typeof data !== 'object') return;
        window.COHESIVX_LAST_STATE = data;
        if (data.fg) renderCohesiveFg(data.fg, data);
      })
      .catch(function () {});
  }

  function tick() {
    rememberLatestState();
    installCohesiveFearGreedDisplay();
    hideLongMainText();
    hideOldStructuralConfirmation();
    observeVisualSummary();
    restoreOpenStates();
  }

  function start() {
    markClicks();
    rememberLatestState();
    tick();
    refreshFgFromState();
    setTimeout(tick, 100);
    setTimeout(refreshFgFromState, 350);
    setTimeout(tick, 300);
    setTimeout(refreshFgFromState, 1100);
    setTimeout(tick, 900);
    setTimeout(tick, 1800);
    setInterval(tick, 250);
  }

  rememberLatestState();
  installCohesiveFearGreedDisplay();
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
