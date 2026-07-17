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