(function(){
'use strict';
var KEY='coeziv_btc_lang';
var PROD='https://coezivx.vercel.app/btc-swing-strategy/';
function isEnglish(){try{return localStorage.getItem(KEY)==='en'||document.documentElement.lang==='en'||(document.body&&document.body.getAttribute('data-lang')==='en')}catch(e){return false}}
function lang(){return isEnglish()?'en':'ro'}
function t(ro,eg){return isEnglish()?eg:ro}
function n(v,f){var x=Number(v