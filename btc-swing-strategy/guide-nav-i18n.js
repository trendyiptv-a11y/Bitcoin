(function(){
  'use strict';
  var KEY='coeziv_btc_lang';
  function en(){
    try{return localStorage.getItem(KEY)==='en'||document.documentElement.lang==='en'||(document.body&&document.body.getAttribute('data-lang')==='en');}
    catch(e){return false;}
  }
  function apply(){
    var isEnglish=en();
    var guide=document.getElementById('guide-link')||document.querySelector('a[href*="ghid-mecanism.html"]');
    if(guide){guide.textContent=isEnglish?'Mechanism guide':'Ghid mecanism';}
    var links=document.querySelectorAll('a[href*="mecanism.html"]');
    for(var i=0;i<links.length;i++){
      var txt=links[i].textContent||'';
      if(txt.indexOf('Înapoi')!==-1||txt.indexOf('Back')!==-1||txt.indexOf('mecanism')!==-1||txt.indexOf('mechanism')!==-1){
        links[i].textContent=isEnglish?'← Back to mechanism':'← Înapoi la mecanism';
      }
    }
    document.documentElement.lang=isEnglish?'en':'ro';
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',apply);else apply();
  setInterval(apply,700);
})();
