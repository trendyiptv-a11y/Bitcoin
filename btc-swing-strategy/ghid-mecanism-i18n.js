(function(){
  'use strict';
  var KEY='coeziv_btc_lang';
  var RO={
    back:'← Înapoi la mecanism',
    pill:'RO / EN • Ghid structural',
    kicker:'CohesivX BTC Monitor',
    title:'Ghid explicativ al mecanismului',
    lead:'Acest ghid explică modul corect de citire a mecanismului coeziv BTC: preț live, preț coeziv central, deviație, confirmare structurală 7/30 zile, radar, prag energetic, sentiment coeziv și interpretare zilnică. Instrumentul nu oferă recomandări de cumpărare sau vânzare.',
    mini1t:'Nu este semnal', mini1:'Nu spune „intră” sau „ieși”.',
    mini2t:'Este context', mini2:'Arată relația dintre preț și structură.',
    mini3t:'Este probabilistic', mini3:'Folosește contexte istorice similare.',

    s1t:'1. Ce este CohesivX BTC Monitor',
    s1p1:'CohesivX BTC Monitor este un instrument de observare structurală a pieței Bitcoin. El compară prețul live al Bitcoin cu un reper coeziv calculat din costul energetic de producție și din contexte istorice similare.',
    s1p2:'Scopul lui este să arate cât de apropiat sau cât de îndepărtat este prețul față de zona structurală estimată.',
    s1n:'Mecanismul nu este un ordin de acțiune. El oferă context structural pentru observare.',

    s2t:'2. Prețul coeziv central',
    s2p1:'Prețul coeziv central este reperul principal al mecanismului. El pornește de la costul energetic estimat de producție și îl ajustează cu multiplicatori observați în contexte istorice asemănătoare.',
    s2p2:'Exemplu: dacă mecanismul folosește un cost mediu de producție de aproximativ 52.466 USD și un multiplicator median de 1,315×, reperul coeziv central ajunge în jur de 68.992 USD.',
    s2n:'Acesta nu este target de preț. Este reper structural de comparație.',

    s3t:'3. Deviația față de mecanism',
    s3p1:'Deviația arată distanța dintre prețul live și prețul coeziv central. Când prețul live este mult sub reper, mecanismul vede presiune sub valoarea structurală. Când este mult peste reper, vede extensie peste valoarea structurală.',
    s3l1:'Deviație mică: piață apropiată de echilibru.',
    s3l2:'Deviație relevantă: tensiune structurală care merită urmărită.',
    s3l3:'Deviație extremă: zonă unde contextul, lichiditatea și riscul devin esențiale.',
    s3n:'O deviație mare nu înseamnă automat cumpărare sau vânzare. Înseamnă doar că prețul este departe de reper.',

    s4t:'4. Confirmarea structurală 7/30 zile',
    s4p1:'Cardul „Confirmare structurală” arată cât de des, în contexte istorice similare, mecanismul a confirmat direcția structurală pe orizonturi de 7 și 30 de zile.',
    s4p2:'Când vezi valori de tip ~58% pe 7 zile și ~59% pe 30 zile, interpretarea corectă este probabilistică: în trecut, cazurile asemănătoare au confirmat direcția mecanismului puțin peste jumătate din timp.',
    s4e1:'Bază statistică', s4e1b:'762 evenimente istorice',
    s4e2:'Context similar', s4e2b:'250 contexte apropiate',
    s4n:'58–59% nu este certitudine. Este avantaj statistic slab-moderat, suficient pentru context, nu pentru decizie automată.',

    s5t:'5. Diferența dintre confirmare și radar',
    s5p1:'Confirmarea structurală răspunde la întrebarea: „În cazuri istorice similare, cât de des s-a confirmat direcția mecanismului?”. Radarul răspunde la altă întrebare: „Cât timp persistă degradarea și cât de aproape este prețul de pragul de risc?”.',
    s5p2:'De aceea cardul de confirmare este așezat deasupra radarului: întâi vezi probabilitatea istorică structurală, apoi vezi starea activă a degradării.',

    s6t:'6. Radarul coeziv',
    s6p1:'Radarul arată vizual dacă degradarea structurală este activă, persistentă sau în ameliorare. El folosește ziua degradării, pragul de risc și starea curentă a pieței.',
    s6p2:'Exemplu: „Ziua 22 / ~27” înseamnă că degradarea activă durează de 22 de zile, iar istoricul indică un timp median de aproximativ 27 de zile până la confirmarea sau schimbarea stării.',

    s7t:'7. Pragul energetic BTC',
    s7p1:'Pragul energetic BTC estimează costul de producție pentru trei tipuri de mineri: eficient, mediu și scump. Mecanismul folosește în principal costul mediu ca ancoră structurală.',
    s7p2:'Dacă prețul live este mult peste costul de producție, piața se află într-o zonă speculativă. Dacă se apropie de costul de producție, presiunea economică asupra minerilor devine mai importantă.',

    s8t:'8. Fear & Greed coeziv',
    s8p1:'Indicele Fear & Greed coeziv nu copiază indicatorii clasici de sentiment. El combină structura, tactica pe termen scurt, tensiunea și regimul mecanismului.',
    s8p2:'Un scor ridicat nu înseamnă automat cumpărare. Un scor scăzut nu înseamnă automat vânzare. Scorul arată intensitatea internă a mecanismului.',

    s9t:'9. Interpretarea coezivă zilnică',
    s9p1:'Interpretarea zilnică traduce indicatorii tehnici într-o explicație naturală: ce stare structurală observă mecanismul, dacă participarea se îmbunătățește, dacă riscul persistă și ce element trebuie urmărit.',
    s9p2:'Ea este utilă pentru citire rapidă, dar nu înlocuiește cardurile principale: preț coeziv, confirmare structurală, radar, prag energetic și sentiment coeziv.',

    s10t:'10. Cum se citește pagina, pas cu pas',
    s10l1:'Observă prețul live.', s10l2:'Compară-l cu prețul coeziv central.', s10l3:'Verifică deviația procentuală.', s10l4:'Citește confirmarea structurală 7/30 zile.', s10l5:'Verifică radarul și fereastra de risc.', s10l6:'Compară cu pragul energetic BTC.', s10l7:'Citește Fear & Greed coeziv și interpretarea zilnică.',

    s11t:'11. De ce mecanismul nu este semnal de trading',
    s11p1:'Un semnal de trading spune: cumpără, vinde, intră sau ieși. CohesivX nu face asta.',
    s11p2:'CohesivX arată unde este prețul față de mecanism, cât de mare este deviația, ce confirmare istorică există, ce risc persistă și ce regim structural este activ.',
    s11n:'Principiul central: mecanismul nu decide în locul tău. El organizează contextul.',

    footer:'CohesivX BTC Monitor — instrument experimental de observare structurală. Nu reprezintă recomandare financiară, investițională sau de tranzacționare.'
  };

  var EN={
    back:'← Back to mechanism',
    pill:'RO / EN • Structural guide',
    kicker:'CohesivX BTC Monitor',
    title:'Mechanism explanatory guide',
    lead:'This guide explains how to read the cohesive BTC mechanism: live price, central cohesive price, deviation, 7/30-day structural confirmation, radar, energy threshold, cohesive sentiment and daily interpretation. The instrument does not provide buy or sell recommendations.',
    mini1t:'Not a signal', mini1:'It does not say “enter” or “exit”.',
    mini2t:'It is context', mini2:'It shows the relationship between price and structure.',
    mini3t:'It is probabilistic', mini3:'It uses similar historical contexts.',

    s1t:'1. What CohesivX BTC Monitor is',
    s1p1:'CohesivX BTC Monitor is a structural observation tool for the Bitcoin market. It compares the live Bitcoin price with a cohesive reference calculated from estimated energy production cost and similar historical contexts.',
    s1p2:'Its purpose is to show how close or how far price is from the estimated structural area.',
    s1n:'The mechanism is not an instruction to act. It provides structural context for observation.',

    s2t:'2. Central cohesive price',
    s2p1:'The central cohesive price is the main reference of the mechanism. It starts from the estimated energy production cost and adjusts it with multipliers observed in similar historical contexts.',
    s2p2:'Example: if the mechanism uses an average production cost of about 52,466 USD and a median multiplier of 1.315×, the central cohesive reference is around 68,992 USD.',
    s2n:'This is not a price target. It is a structural comparison reference.',

    s3t:'3. Deviation from the mechanism',
    s3p1:'Deviation shows the distance between the live price and the central cohesive price. When the live price is far below the reference, the mechanism sees pressure below structural value. When it is far above the reference, it sees extension above structural value.',
    s3l1:'Small deviation: market close to equilibrium.',
    s3l2:'Relevant deviation: structural tension worth watching.',
    s3l3:'Extreme deviation: zone where context, liquidity and risk become essential.',
    s3n:'A large deviation does not automatically mean buy or sell. It only means price is far from the reference.',

    s4t:'4. 7/30-day structural confirmation',
    s4p1:'The “Structural confirmation” card shows how often, in similar historical contexts, the mechanism confirmed the structural direction over 7-day and 30-day horizons.',
    s4p2:'When you see values such as ~58% over 7 days and ~59% over 30 days, the correct interpretation is probabilistic: historically, similar cases confirmed the mechanism direction slightly more than half of the time.',
    s4e1:'Statistical base', s4e1b:'762 historical events',
    s4e2:'Similar context', s4e2b:'250 nearby contexts',
    s4n:'58–59% is not certainty. It is a weak-to-moderate statistical edge, useful for context, not for automatic decisions.',

    s5t:'5. Difference between confirmation and radar',
    s5p1:'Structural confirmation answers: “In similar historical cases, how often was the mechanism direction confirmed?”. The radar answers a different question: “How long has degradation persisted and how close is price to the risk threshold?”.',
    s5p2:'That is why the confirmation card is placed above the radar: first you see the historical structural probability, then the active degradation state.',

    s6t:'6. Cohesive radar',
    s6p1:'The radar visually shows whether structural degradation is active, persistent or improving. It uses the degradation day count, the risk threshold and the current market state.',
    s6p2:'Example: “Day 22 / ~27” means active degradation has lasted 22 days, while history indicates a median time of about 27 days until confirmation or state change.',

    s7t:'7. BTC energy threshold',
    s7p1:'The BTC energy threshold estimates production cost for three miner types: efficient, average and expensive. The mechanism mainly uses the average cost as a structural anchor.',
    s7p2:'If live price is far above production cost, the market is in a speculative zone. If it approaches production cost, economic pressure on miners becomes more important.',

    s8t:'8. Cohesive Fear & Greed',
    s8p1:'The cohesive Fear & Greed index does not copy classic sentiment indicators. It combines structure, short-term tactics, tension and the mechanism regime.',
    s8p2:'A high score does not automatically mean buy. A low score does not automatically mean sell. The score shows the internal intensity of the mechanism.',

    s9t:'9. Daily cohesive interpretation',
    s9p1:'The daily interpretation translates technical indicators into a natural explanation: what structural state the mechanism observes, whether participation is improving, whether risk persists and what should be watched.',
    s9p2:'It is useful for quick reading, but it does not replace the main cards: cohesive price, structural confirmation, radar, energy threshold and cohesive sentiment.',

    s10t:'10. How to read the page, step by step',
    s10l1:'Observe the live price.', s10l2:'Compare it with the central cohesive price.', s10l3:'Check the percentage deviation.', s10l4:'Read the 7/30-day structural confirmation.', s10l5:'Check the radar and risk window.', s10l6:'Compare with the BTC energy threshold.', s10l7:'Read cohesive Fear & Greed and the daily interpretation.',

    s11t:'11. Why the mechanism is not a trading signal',
    s11p1:'A trading signal says: buy, sell, enter or exit. CohesivX does not do that.',
    s11p2:'CohesivX shows where price is relative to the mechanism, how large the deviation is, what historical confirmation exists, what risk persists and what structural regime is active.',
    s11n:'Core principle: the mechanism does not decide for you. It organizes the context.',

    footer:'CohesivX BTC Monitor — experimental structural observation instrument. It is not financial, investment or trading advice.'
  };

  function isEn(){try{return localStorage.getItem(KEY)==='en'||document.documentElement.lang==='en'||(document.body&&document.body.getAttribute('data-lang')==='en');}catch(e){return false;}}
  function apply(){var map=isEn()?EN:RO;document.querySelectorAll('[data-g18n]').forEach(function(el){var k=el.getAttribute('data-g18n');if(map[k])el.textContent=map[k];});document.documentElement.lang=isEn()?'en':'ro';if(document.body)document.body.setAttribute('data-lang',isEn()?'en':'ro');}
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',apply);else apply();
  setInterval(apply,700);
  window.CoezivGuideI18n={apply:apply};
})();
