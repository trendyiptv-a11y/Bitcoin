/* CohesivX visual summary fix v7 — hides old structural card and preserves opened details */
(function () {
  'use strict';

  var desiredOpen = {
    ctx: false,
    full: false
  };

  function getDetails() {
    return Array.prototype.slice.call(
      document.querySelectorAll('#cx-visual-summary details.cxv-narrative')
    );
  }

  function restoreOpenStates() {
    var details = getDetails();
    if (details[0]) details[0].open = !!desiredOpen.ctx;
    if (details[1]) details[1].open = !!desiredOpen.full;
  }

  function markClicks() {
    document.addEventListener('click', function (ev) {
      var summary = ev.target && ev.target.closest &&
        ev.target.closest('#cx-visual-summary details.cxv-narrative summary');

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

      var hits = patterns.reduce(function (n, rx) {
        return n + (rx.test(text) ? 1 : 0);
      }, 0);

      if (hits < 2) return;
      if (text.length > 2400) return;

      candidates.push({
        node: node,
        len: text.length,
        hits: hits
      });
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
    var ids = ['message', 'model-price-explanation'];

    ids.forEach(function (id) {
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

    observer.observe(root, {
      childList: true,
      subtree: true
    });
  }

  function tick() {
    hideLongMainText();
    hideOldStructuralConfirmation();
    observeVisualSummary();
    restoreOpenStates();
  }

  function start() {
    markClicks();

    tick();
    setTimeout(tick, 300);
    setTimeout(tick, 900);
    setTimeout(tick, 1800);

    setInterval(tick, 250);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
