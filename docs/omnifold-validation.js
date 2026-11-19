(() => {
  const runs = JSON.parse(document.getElementById('of-catalog').textContent);
  const pick = Object.fromEntries(['run','iteration','stage','plot'].map(k => [k, document.getElementById('of-'+k)]));
  const chart = document.getElementById('of-chart');
  const caption = document.getElementById('of-caption');
  const stageText = {
    reco_pull:'Reco pull compares purity-weighted Higgs data with the current folded signal simulation.',
    truth_push:'Truth push compares pulled and original weights on the same matched fiducial simulation events.',
    reco_likelihood:'Reco likelihood compares the newly refolded signal with the original folded simulation.'
  };
  const descriptions = {
    learning:'Training and validation loss are shown for the four network members, including their selected checkpoints.',
    closure:'Held-out projections compare the target with the reference before and after applying the full fitted ratio. The plots retain the fitted rate convention.',
    reliability:'Each point compares the mean raw score with the target fraction in a held-out score bin, using balanced class weights. The diagonal denotes agreement. Signed weights can move a bin fraction outside the unit interval.',
    scores:'The held-out target and reference distributions show the raw classifier score and the logarithm of the shape ratio.',
    ensemble:'The distribution shows the event-by-event standard deviation of the four network scores.'
  };
  const stageOrder=Object.keys(stageText), plotOrder=Object.keys(descriptions);
  const params=new URLSearchParams(location.search);
  function setIfListed(select,value) {
    if (Array.from(select.options).some(option => option.value===value)) select.value=value;
  }
  setIfListed(pick.run,params.get('ofRun'));
  setIfListed(pick.stage,params.get('ofStage'));
  setIfListed(pick.plot,params.get('ofPlot'));
  function updateIterations(preferred) {
    const run=runs.find(r=>r.key===pick.run.value);
    pick.iteration.replaceChildren(...run.iterations.map(i=>new Option(String(i.iteration),String(i.iteration))));
    setIfListed(pick.iteration,preferred);
  }
  updateIterations(params.get('ofIteration'));
  let sequence=0, loadedStem='', sheet=null;
  async function update(writeURL) {
    const selection=++sequence;
    const run=runs.find(r=>r.key===pick.run.value);
    const iter=run.iterations.find(i=>String(i.iteration)===pick.iteration.value);
    const stage=pick.stage.value, plot=pick.plot.value;
    const page=stageOrder.indexOf(stage)*plotOrder.length+plotOrder.indexOf(plot);
    const base='figures/omnifold/validation/'+iter.stem;
    const title=run.label+', iteration '+iter.iteration+', '+pick.stage.selectedOptions[0].text+', '+pick.plot.selectedOptions[0].text;
    document.getElementById('of-pdf').href=base+'.pdf#page='+(page+1);
    document.getElementById('of-set').href=base+'.pdf';
    caption.textContent=stageText[stage]+' '+descriptions[plot];
    chart.setAttribute('aria-label',title);
    const [x,y,w,h]=iter.rects[page];
    chart.width=w; chart.height=h; chart.style.maxWidth=Math.min(w,1100)+'px';
    if (writeURL) {
      const url=new URL(location.href);
      for(const [k,v] of Object.entries({ofRun:run.key,ofIteration:iter.iteration,ofStage:stage,ofPlot:plot}))url.searchParams.set(k,v);
      url.hash='omnifold-validation';
      history.replaceState(null,'',url);
    }
    const status=document.getElementById('of-image-status');
    status.textContent='Loading plot…';
    try {
      let img=sheet;
      if (loadedStem!==iter.stem || !img) {
        img=new Image();
        await new Promise((resolve,reject)=>{img.onload=resolve;img.onerror=reject;img.src=base+'.webp';});
        if (selection!==sequence)return;
        sheet=img;loadedStem=iter.stem;
      }
      if(selection!==sequence)return;
      chart.getContext('2d').drawImage(img,x,y,w,h,0,0,w,h);
      status.textContent='';
      chart.dataset.loaded=iter.stem+'-'+(page+1);
    } catch (error) {
      if(selection===sequence)status.textContent='The preview could not load. Open the selected plot in the PDF below.';
    }
  }
  pick.run.addEventListener('change',()=>{updateIterations(pick.iteration.value);update(true);});
  for(const select of [pick.stage,pick.plot,pick.iteration]) select.addEventListener('change',()=>update(true));
  update(false);
})();
