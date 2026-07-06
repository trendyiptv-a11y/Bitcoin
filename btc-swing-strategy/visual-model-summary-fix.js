/* CohesivX visual summary fix v6 — hides old structural card and keeps details open */
(function () {
  'use strict';

  var openState = { ctx: false, full: false };

  function rememberOpenStates() {
    var details = document.querySelectorAll('#cx-visual-summary details.cxv-narrative');
    if (details[0]) openState.ctx = details[0].open;
    if (details[1]) openState.full = details[1].open;
  }

  function restoreOpenStates() {
    var details = document.querySelectorAll('#cx-visual-summary details.cxv-narrative');
    if (details[0] && openState.ctx) details[0].open = true;
    if (details[1] && openState.full) details[1].open = true;
  }

  function markClicks() {
    document.addEventListener('click', function (ev) {
      var summary = ev.target && ev.target.closest && ev.target.closest('#cx-visual-summary details.cxv-narrative summary');
      if (!summary) return;
      setTimeout(function () {
        rememberOpenStates();
        restoreOpenStates();
      }, 30);
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
      if (!node || node.id === 'cx-visual-summary' || node.closest('#cx-visual-summary')) return;
      if (node.dataset && node.dataset.cxvHidden === '1') return;
      var text = (node.textContent || '').replace(/\s+/g, ' ').trim();
      if (!text) return;

      var hits = patterns.reduce(function (n, rx) { return n + (rx.test(text) ? 1 : 0); }, 0);
      if (hits < 2) return;
      if (text.length > 1800) return;

      candidates.push({ node: node, len: text.length, hits: hits });
    });

    candidates.sort(function (a, b) {
      if (b.hits !== a.hits) return b.hits - a.hits;
      return a.len - b.len;
    });

    candidates.slice(0, 6).forEach(function (item) {
      item.node.style.display = 'none';
      item.node.setAttribute('aria-hidden', 'true');
      item.node.dataset.cxvHidden = '1';
    });
  }

  function hideLongMainText() {
    var ids = ['message', 'model-price-explanation'];
    ids.forEach(function (id) {
      var el = document.getElementById(id);
      if (el) el.style.display = 'none';
    });
  }

  function tick() {
    rememberOpenStates();
    hideLongMainText();
    hideOldStructuralConfirmation();
    restoreOpenStates();
  }

  function start() {
    markClicks();
    tick();
    setInterval(tick, 250);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
