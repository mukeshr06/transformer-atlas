import {canSpatial} from './live-space.js?v=b55dd29';
import {derivation} from './math.js?v=b55dd29';
import {vectorPages} from './vector-space.js?v=b55dd29';
import {teach} from './teaching.js?v=b55dd29';
import {Transformer,EXAMPLES,VOCAB} from './model.js?v=b55dd29';
import {buildCourse,locate,hashIndex,calculate,explanation,scopeName,format,conceptLabels,tensorName,ensureScene} from './course.js?v=b55dd29';
import {buildWorld,Renderer,blockFor} from './architecture.js?v=b55dd29';
import {clamp} from './space.js?v=b55dd29';
import {sourceInfo,lessons} from './data.js?v=b55dd29';

const $=id=>document.getElementById(id);
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
$('learning-panel').append($('caption'),$('calculation'));
const model=new Transformer();model.setText('rain feeds rivers');
const renderer=new Renderer($('space'),$('labels'));
const preference=matchMedia('(prefers-reduced-motion: reduce)');
let preferredMode='3d';
let reduced=preference.matches,ambient=!reduced,clock=0,last=0,lastDraw=0,dirty=true;
let course=buildCourse(model),index=-1,local=0,unit=Math.max(420,innerHeight),phase=0,play=false,manual=false;
let selected={row:0,col:0},locked=true,calculation=null,termIndex=-1,worldKey='',captions=true,collapsed=false,explode=1;
let selectedTensor='',lastEquation='',sceneEpoch=0,trainRemaining=0,trainingTimer=null,referenceLoad=0,noticeTimer=null;
const initialHash=location.hash;
let explanationBeat='',pinnedBeat=null,targetProgress=scrollY/unit,visualProgress=targetProgress;

function makeCourse(){
  $('course').innerHTML=course.map(s=>`<article id="step-${s.id}" class="course-scene" style="--length:${s.length}"><div class="sr-only"><h2>${escape(s.group+' · '+s.title)}</h2><p>${escape(s.body)}</p></div></article>`).join('')+'<div class="course-tail" aria-hidden="true"></div>';
  $('atlas-count').textContent=`${course.length} guided chapters · ${model.trace.length} inspectable calculations · 70 animated source compositions.`;
  $('chapters').innerHTML=Object.entries(conceptLabels).map(([id,label],i)=>`<button data-concept="${id}"><span>${String(i+1).padStart(2,'0')}</span>${escape(label)}</button>`).join('');
  $('references').innerHTML=sourceInfo.map((info,i)=>`<button data-scene="drawing-${i+1}"><span>${String(i+1).padStart(2,'0')}</span>${escape(info[0])}</button>`).join('');
  fillOperations();document.documentElement.style.setProperty('--unit',`${unit}px`);
}
function fillOperations(){const query=$('search').value.toLowerCase().trim();const collection=[...new Map([...course,...course.archive].map(s=>[s.id,s])).values()];const matches=collection.filter(s=>!s.reference&&(!query||(s.title+' '+s.group+' '+s.id).toLowerCase().includes(query)));$('operation-list').innerHTML=matches.length?matches.map(s=>`<button data-scene="${s.id}"><span>${escape(s.title)}</span><small>${escape(s.group)}</small></button>`).join(''):'<p class="field-note" style="padding:16px">No matching operation. Try “query”, “variance” or “gradient”.</p>';}
function equationMarkup(calc){return derivation(model,calc);}

function scene(){return course[Math.max(0,index)];}
function setScene(next){
  if(next===index)return;index=next;sceneEpoch++;explanationBeat='';pinnedBeat=null;manual=false;play=false;locked=true;selected={row:0,col:0};worldKey='';lastEquation='';termIndex=-1;
  const s=scene();$('learning-panel').scrollTop=0;$('scene-scope').textContent=s.group;$('scene-title').textContent=s.title;
  const record=model.tensors.get(s.tensor);$('scene-description').textContent=s.why||s.body;
  $('caption').classList.remove('entering');
  $('scene-count').textContent=`${index+1} / ${course.length}`;
  $('previous').disabled=index===0;$('next').disabled=index===course.length-1;
  $('open-reference').hidden=!s.reference;$('open-reference').textContent='Replay this operation ↻';const forced=vectorPages.has(s.reference),available=forced||canSpatial(record,s);renderer.camera.mode=forced?'3d':s.reference?'2d':available?preferredMode:'2d';document.querySelectorAll('[data-mode]').forEach(b=>{b.hidden=forced?b.dataset.mode!=='3d':b.dataset.mode==='3d'&&!available;const active=forced?b.dataset.mode==='3d':b.dataset.mode===renderer.camera.mode;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});$('render-status').textContent=forced||renderer.camera.mode==='3d'?'3D · LIVE COORDINATES':'2D · LIVE COMPONENTS';
  $('play').textContent='▶';$('play').setAttribute('aria-label','Play the calculation');
  $('labels').setAttribute('aria-label',s.title+'. '+(record?explanation(record,model):s.body));
  history.replaceState(null,'','#'+s.id);
  if(s.id==='generate'){selected={row:model.targetTokens.length-1,col:0};locked=true;}fillCoordinates();layout();dirty=true;
}
function fillCoordinates(){const t=scene().visualOnly?null:model.tensors.get(scene().tensor);$('row').innerHTML=Array.from({length:t?.shape[0]||1},(_,i)=>`<option value="${i}">${i}</option>`).join('');$('column').innerHTML=Array.from({length:t?.shape[1]||1},(_,i)=>`<option value="${i}">${i}</option>`).join('');$('row').disabled=!t;$('column').disabled=!t;$('all-values').disabled=!t;}
function updateTeaching(){const s=scene(),beat=pinnedBeat||(local<.34?'why':local<.69?'how':'result');if(beat===explanationBeat)return;explanationBeat=beat;$('scene-description').textContent=s[beat]||s.body;for(const part of ['why','how','result']){$(part+'-copy').textContent=s[part]||s.body;$('explain-'+part).classList.toggle('current-explanation',part===beat);}document.querySelectorAll('[data-explanation]').forEach(b=>{b.classList.toggle('active',b.dataset.explanation===beat);b.setAttribute('aria-pressed',String(b.dataset.explanation===beat));});}
function layout(){
  document.body.classList.toggle('captions-off',!captions);document.body.classList.toggle('calculation-collapsed',collapsed);$('caption').style.opacity=captions?'1':'0';dirty=true;
}
renderer.onResize=layout;
renderer.onTerm=frame=>{if(termIndex===frame.index)return;termIndex=frame.index;document.querySelectorAll('[data-term]').forEach(el=>el.classList.toggle('current',+el.dataset.term===frame.index));$('term-track').querySelectorAll('i').forEach((el,i)=>el.classList.toggle('active',i<=frame.index));$('partial-label').textContent=frame.calc.additive?`Partial sum ${frame.index+1}/${frame.calc.terms.length}: ${format(frame.partial,6)}`:`Step ${frame.index+1}/${frame.calc.terms.length} · ${frame.calc.tensor.shape.join(' × ')}`;};

function applyProgress(position){const state=locate(course,position);setScene(state.index);local=state.local;phase=reduced?1:clamp(local*1.1);updateSeek();updateTeaching();updateCalculation();dirty=true;}
function onScroll(){targetProgress=scrollY/unit;manual=false;play=false;$('play').textContent='▶';if(reduced){visualProgress=targetProgress;applyProgress(visualProgress);}}
function updateSeek(){const p=clamp((scene().start+local*scene().length)/course.total);$('seek').value=Math.round(p*10000);$('seek').style.setProperty('--p',p*100+'%');$('scroll-label').textContent=index===course.length-1?'EXPLORE YOUR MODEL':manual?'SCROLL TO CONTINUE':'SCROLL THROUGH THE MODEL';}
let queued=false;
window.addEventListener('scroll',()=>{if(queued)return;queued=true;requestAnimationFrame(()=>{queued=false;onScroll();});},{passive:true});
window.addEventListener('resize',()=>{const position=scene().start+local*scene().length;unit=Math.max(420,innerHeight);document.documentElement.style.setProperty('--unit',`${unit}px`);window.scrollTo({top:position*unit,behavior:'instant'});renderer.resize();layout();dirty=true;});
function goToId(id,{row=0,col=0,lock=false}={}){const before=course.length;const next=ensureScene(course,id);if(next<0)return;if(course.length!==before){makeCourse();index=-1;}goTo(next,true);if(lock){selected={row,col};locked=true;updateCalculation(true);}}
function goTo(next,instant=true){next=clamp(next,0,course.length-1);window.scrollTo({top:(course[next].start+course[next].length*.025)*unit,behavior:instant||reduced?'instant':'smooth'});if(instant){targetProgress=visualProgress=(course[next].start+course[next].length*.025);applyProgress(visualProgress);}}
function updateCalculation(force=false){
  $('micro-progress').value=Math.round(phase*1000);$('micro-label').textContent=Math.round(phase*100)+'%';
  const s=scene(),t=s.visualOnly?null:model.tensors.get(s.tensor);
  if(!t){calculation=null;$('calc-content').hidden=true;$('world-instruction').hidden=false;$('calc-label').textContent='ONE EXAMPLE · ONE CONTINUOUS COMPUTATION';$('example-line').textContent=s.visualOnly?'Conceptual example · assumptions and values are shown in the diagram.':(model.customExample||EXAMPLES[model.config.example]).source.join(' ')+'  →  '+(model.customExample||EXAMPLES[model.config.example]).target.filter(t=>t!=='<eos>').join(' ');if(worldKey!==s.id||force){renderer.setWorld(buildWorld(model,s,{explode,phase,manual}));worldKey=s.id;}if(renderer.world){renderer.world.options.phase=phase;renderer.world.options.manual=manual;}layout();return;}
  $('calc-content').hidden=false;$('world-instruction').hidden=true;
  const cells=t.shape[0]*t.shape[1],position=Math.min(cells-.0001,phase*cells),cell=Math.floor(position);
  if(!locked)selected={row:Math.floor(cell/t.shape[1]),col:cell%t.shape[1]};
  selected.row=clamp(selected.row,0,t.shape[0]-1);selected.col=clamp(selected.col,0,t.shape[1]-1);
  $('row').value=selected.row;$('column').value=selected.col;
  calculation=calculate(model,t.id,selected.row,selected.col);
  const fraction=locked?phase:position-cell;
  const active=Math.min(calculation.terms.length-1,Math.floor((phase>=1?1:fraction)*calculation.terms.length));
  const key=`${s.id}:${selected.row}:${selected.col}:${Math.floor(phase*20)}:${active}:${explode}`;
  if(force||worldKey!==key){renderer.setWorld(buildWorld(model,s,{phase:phase,manual:true,explode,row:selected.row,col:selected.col,term:active,selected:{tensor:t.id,...selected}}));worldKey=key;dirty=true;}
  if(renderer.world){renderer.world.options.phase=phase;renderer.world.options.term=active;}
  const eqKey=`${t.id}:${selected.row}:${selected.col}:${calculation.formula}`;
  if(lastEquation!==eqKey||force){$('calc-label').textContent=`${t.label} [${selected.row}, ${selected.col}]`;$('equation').innerHTML=equationMarkup(calculation);$('result').textContent=format(calculation.result,6);$('term-track').innerHTML=calculation.terms.map(()=>'<i></i>').join('');lastEquation=eqKey;termIndex=-1;}
  if(active!==termIndex||force){document.querySelectorAll('[data-term]').forEach(el=>el.classList.toggle('current',+el.dataset.term===active));$('term-track').querySelectorAll('i').forEach((el,i)=>el.classList.toggle('active',i<=active));const count=Math.max(1,active+1);$('partial-label').textContent=calculation.additive?`Partial sum ${count}/${calculation.terms.length}: ${format(calculation.terms.slice(0,count).reduce((sum,x)=>sum+x.value,0),6)}`:`${t.op==='parameter'?'Learned value':`Step ${count}/${calculation.terms.length}`} · ${t.shape.join(' × ')}`;termIndex=active;}
}
function refresh({rebuild=false}={}){
  const id=scene().id;if(rebuild){course=buildCourse(model);ensureScene(course,id);makeCourse();index=-1;goTo(Math.max(0,course.findIndex(s=>s.id===id)),true);}
  for(const step of [...course,...course.archive]){const record=model.tensors.get(step.tensor);if(record&&step.id!=='generate'&&!step.reference){const how=explanation(record,model);Object.assign(step,teach(record,model,how));}}
  if(scene().id==='generate'){selected={row:model.targetTokens.length-1,col:0};locked=true;}fillCoordinates();worldKey='';lastEquation='';updateStats();const t=model.tensors.get(scene().tensor);explanationBeat='';updateTeaching();updateCalculation(true);layout();dirty=true;
}
function recompute(rebuild=false){model.run();model.history=[model.loss];refresh({rebuild});}
function updateStats(){
  $('model-size').textContent=`Paper N = 6 · live N = ${model.config.layers}`;$('run-label').textContent=model.generation?'LIVE · GREEDY GENERATION':'LIVE · TEACHER FORCING';$('loss-number').textContent=format(model.loss,4);$('update-number').textContent=model.steps;$('gradient-number').textContent=format(model.gradientNorm,3);$('generated-prefix').textContent=model.generation?model.generation.join('  '):'Current teacher-forced prefix: '+model.targetTokens.join(' ');$('append-token').disabled=!!model.generation&&(model.generation.length>=9||model.generation.at(-1)==='<eos>');drawLoss();
}
function drawLoss(){const canvas=$('loss-chart'),c=canvas.getContext('2d'),width=800,height=110;canvas.width=width;canvas.height=height;c.clearRect(0,0,width,height);const values=model.history.length?model.history:[model.loss],max=Math.max(...values,.1)*1.12;const pts=values.map((v,i)=>[24+i/(Math.max(1,values.length-1))*(width-48),height-18-v/max*(height-28)]);c.beginPath();pts.forEach((p,i)=>i?c.lineTo(...p):c.moveTo(...p));c.strokeStyle='#8affe0';c.lineWidth=2;c.shadowColor='#8affe0';c.shadowBlur=6;c.stroke();const end=pts.at(-1);c.beginPath();c.arc(...end,3,0,Math.PI*2);c.fillStyle='#caffee';c.fill();c.shadowBlur=0;c.font='11px monospace';c.fillStyle='#729c95';c.fillText('actual loss · '+values.length+' recorded values',24,height-1);canvas.setAttribute('aria-label',`Actual training loss. First ${format(values[0],4)}; latest ${format(values.at(-1),4)}; ${model.steps} weight updates.`);}
function notify(text){clearTimeout(noticeTimer);$('notice').textContent=text;$('notice').hidden=false;noticeTimer=setTimeout(()=>$('notice').hidden=true,3500);}

function microStep(value){manual=true;locked=true;play=false;phase=clamp(value);$('play').textContent='▶';updateCalculation(true);dirty=true;}
$('micro-progress').oninput=e=>microStep(+e.target.value/1000);
$('micro-prev').onclick=()=>microStep(phase-1/Math.max(4,calculation?.terms.length||4));
$('micro-next').onclick=()=>microStep(phase+1/Math.max(4,calculation?.terms.length||4));
$('previous').onclick=()=>goTo(index-1);$('next').onclick=()=>goTo(index+1);
$('seek').addEventListener('input',e=>window.scrollTo({top:+e.target.value/10000*course.total*unit,behavior:'instant'}));
document.querySelector('.brand').onclick=e=>{e.preventDefault();goToId('world');};$('world-view').onclick=()=>goToId('world');
window.addEventListener('hashchange',()=>goToId(decodeURIComponent(location.hash.slice(1))));
$('row').onchange=()=>{selected.row=+$('row').value;locked=true;updateCalculation(true);};$('column').onchange=()=>{selected.col=+$('column').value;locked=true;updateCalculation(true);};
$('play').onclick=()=>{if(play){play=false;}else{if(phase>=.999)phase=0;manual=true;play=true;} $('play').textContent=play?'Ⅱ':'▶';$('play').setAttribute('aria-label',play?'Pause the calculation':'Play the calculation');updateSeek();dirty=true;};
$('motion').onclick=()=>{ambient=!ambient;$('motion').textContent=ambient?'Ⅱ':'▶';$('motion').setAttribute('aria-label',ambient?'Pause moving signals':'Resume moving signals');dirty=true;};
$('captions').onclick=()=>{captions=!captions;$('caption').classList.toggle('hidden-caption',!captions);$('captions').setAttribute('aria-pressed',String(captions));$('captions').setAttribute('aria-label',captions?'Hide explanations':'Show explanations');layout();};
$('collapse-calc').onclick=()=>{collapsed=!collapsed;$('calculation').classList.toggle('collapsed',collapsed);$('collapse-calc').textContent=collapsed?'⌃':'⌄';$('collapse-calc').setAttribute('aria-label',collapsed?'Expand calculation':'Collapse calculation');layout();};
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{preferredMode=b.dataset.mode;renderer.camera.mode=b.dataset.mode;document.querySelectorAll('[data-mode]').forEach(el=>{el.classList.toggle('active',el===b);el.setAttribute('aria-pressed',String(el===b));});$('render-status').textContent=b.dataset.mode==='3d'?'3D · LIVE COORDINATES':'2D · LIVE COMPONENTS';updateCalculation(true);dirty=true;});
$('fit').onclick=()=>{renderer.camera.zoom=1;renderer.vector.reset();dirty=true;};$('zoom-in').onclick=()=>{renderer.camera.zoom=clamp(renderer.camera.zoom*1.18,.5,3.5);dirty=true;};$('zoom-out').onclick=()=>{renderer.camera.zoom=clamp(renderer.camera.zoom/1.18,.5,3.5);dirty=true;};
document.querySelectorAll('[data-explanation]').forEach(b=>b.onclick=()=>{pinnedBeat=b.dataset.explanation;explanationBeat='';updateTeaching();$('learning-panel').scrollTo({top:$('explain-'+pinnedBeat).offsetTop-18,behavior:reduced?'instant':'smooth'});});
const blockEntry={ 'source.embedding':'source.lookup','source.position':'source.position','target.embedding':'target.ids','target.position':'target.position','enc.self':'enc.0.self.h0.Q','enc.self.norm':'enc.0.self.residual','enc.ff':'enc.0.ff.expand','enc.ff.norm':'enc.0.ff.residual','dec.self':'dec.0.self.h0.masked','dec.self.norm':'dec.0.self.residual','dec.cross':'dec.0.cross.h0.Q','dec.cross.norm':'dec.0.cross.residual','dec.ff':'dec.0.ff.expand','dec.ff.norm':'dec.0.ff.residual','output.linear':'output.project','output.softmax':'output.A'};
renderer.host.addEventListener('inspect-neuron',e=>{const t=model.tensors.get(scene().tensor);if(!t)return;if(/\.ff\.(expand|biased|relu)$/.test(t.id)){selected.col=e.detail.index;locked=true;updateCalculation(true);}else{const prefix=t.id.match(/^(enc|dec)\.\d+\.ff/)?.[0]||'enc.0.ff';goToId(prefix+'.relu',{row:selected.row,col:e.detail.index,lock:true});}});
renderer.host.addEventListener('click',e=>{const cell=e.target.closest('[data-tensor]');if(cell){const id=cell.dataset.tensor;if(model.tensors.get(id)?.parameterOnly){openValues(id);return;}if(id===scene().tensor){selected={row:+cell.dataset.row,col:+cell.dataset.col};locked=true;updateCalculation(true);}else goToId(id,{row:+cell.dataset.row,col:+cell.dataset.col,lock:true});return;}const key=e.target.closest('[data-block]')?.dataset.block;if(key)goToId(blockEntry[key]);});
renderer.host.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){const cell=e.target.closest('[data-tensor]');if(cell){e.preventDefault();cell.dispatchEvent(new MouseEvent('click',{bubbles:true}));return;}const key=e.target.closest('[data-block]')?.dataset.block;if(key){e.preventDefault();goToId(blockEntry[key]);}}});
// Native wheel and touch scrolling set a target; one continuous animation loop interpolates it.
$('fullscreen').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen?.();}catch{notify('Full screen is not available in this browser.');}};

function openDialog(id){$(id).showModal();document.body.classList.add('modal-open');if(id==='experiment')drawLoss();}
function closeDialog(id){$(id).close();}
document.querySelectorAll('.close-dialog').forEach(b=>b.onclick=()=>b.closest('dialog').close());
document.querySelectorAll('dialog').forEach(d=>{d.addEventListener('close',()=>{if(!document.querySelector('dialog[open]'))document.body.classList.remove('modal-open');last=0;dirty=true;});d.addEventListener('click',e=>{if(e.target!==d)return;const r=d.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)d.close();});});
$('open-experiment').onclick=()=>{updateStats();openDialog('experiment');};$('open-about').onclick=()=>openDialog('about');$('open-atlas').onclick=()=>openDialog('atlas');
$('search').oninput=fillOperations;
for(const id of ['operation-list','references'])$(id).addEventListener('click',e=>{const b=e.target.closest('[data-scene]');if(!b)return;closeDialog('atlas');goToId(b.dataset.scene);});
$('chapters').addEventListener('click',e=>{const b=e.target.closest('[data-concept]');if(!b)return;const entry={attention:'solution-attention',context:'problem-context',walkthrough:'complete'}[b.dataset.concept];if(entry){closeDialog('atlas');goToId(entry);return;}let next=course.findIndex(s=>!s.reference&&s.concept===b.dataset.concept);if(next<0)next=course.findIndex(s=>s.concept===b.dataset.concept);closeDialog('atlas');if(next<0){const archived=course.archive.find(s=>s.concept===b.dataset.concept);if(archived){goToId(archived.id);return;}}goTo(Math.max(0,next));});
$('read-more').onclick=()=>{
  const s=scene(),t=s.visualOnly?null:model.tensors.get(s.tensor);$('detail-scope').textContent=s.group;$('detail-title').textContent=s.title;
  $('detail-body').innerHTML=`<h3>Why this step exists</h3><p>${escape(s.why||s.body)}</p><h3>How it works</h3><p>${escape(s.how||explanation(t,model))}</p><h3>What the result means</h3><p>${escape(s.result||'Follow the highlighted route into the next block.')}</p>`+(t?`<div class="formula">${derivation(model,calculation)}</div><p>Selected coordinate: row ${selected.row}, column ${selected.col}. The output tensor has shape ${t.shape.join(' × ')}. Every visible operand is read from the current forward pass.</p><h3>Follow the dependencies</h3><ul>${t.inputs.map(id=>`<li>${escape(model.tensors.get(id)?.label||id)} · ${model.tensors.get(id)?.shape.join(' × ')}</li>`).join('')}</ul>`:'<p>The encoder contextualizes the source. The decoder combines a shifted target prefix with the final encoder memory. Each stack has '+model.config.layers+' blocks with distinct learned parameters.</p>')+`<h3>Try it</h3><p>Open Experiment to change the attention mask or temperature. Use All values to inspect any tensor and edit learned embeddings or weights. A change runs the complete model again, including all downstream probabilities and gradients.</p>`;
  if(t?.op==='backward'||t?.op==='gradient')$('detail-body').innerHTML+=`<h3>Where the contributions come from</h3><p>For a selected scalar x, the chain rule is ∂L/∂x = Σ (∂L/∂y) · (∂y/∂x), over every operation y that consumes x. Multiplication by a second input has that input as its local derivative; addition has derivative 1; ReLU has derivative 0 or 1. Shared weights receive a sum over all their uses.</p><ul>${calculation.terms.map(term=>`<li>${escape(term.operation||'Contribution')} ${term.node===undefined?'':`· scalar node ${term.node}`}: incoming ${format(term.upstream,6)} × local ${format(term.derivative,6)} = ${format(term.value,6)}</li>`).join('')}</ul>`;
  if(s.reference)$('detail-body').innerHTML+=`<h3>Reference ${s.reference}</h3><p>${escape(sourceInfo[s.reference-1][1])}</p><p>The live miniature uses 4 features and 2 heads; the source may show different dimensions or a different example. Every token, matrix, vector and connection is a frontend component. The motion follows operations within the diagram. Vector-space concepts use a real 3D coordinate system; illustrative coordinates are labeled.</p>`;
  openDialog('detail');
};
$('example').innerHTML='<option value="custom">Your text / copy demonstration</option>'+EXAMPLES.map((e,i)=>`<option value="${i}">${escape(e.source.join(' ')+' → '+e.target.filter(t=>t!=='<eos>').join(' '))}</option>`).join('');
function stopTraining(){trainRemaining=0;clearTimeout(trainingTimer);$('train-batch').textContent='Train 50 steps';$('train-one').disabled=false;}
$('custom-text-form').onsubmit=e=>{e.preventDefault();stopTraining();try{model.setText($('source-text').value,$('target-text').value);$('example').value='custom';$('custom-message').textContent='Text applied. New token embeddings and weights are initialized; this is an untrained teaching model.';refresh({rebuild:true});closeDialog('experiment');goToId('known-example');}catch(error){$('custom-message').textContent=error.message;}};
$('example').onchange=()=>{if($('example').value==='custom')return;stopTraining();model.customExample=null;model.config.example=+$('example').value;model.generation=null;recompute(true);};
$('layers').onchange=()=>{stopTraining();model.config.layers=+$('layers').value;recompute(true);};
for(const [id,key]of [['causal','causal'],['scaled','scaled'],['padding','padding']])$(id).onchange=()=>{stopTraining();model.config[key]=$(id).checked;recompute();};
for(const [id,key,out,precision]of [['attn-temp','attentionTemperature','attn-temp-value',2],['temperature','temperature','temperature-value',2],['learning-rate','learningRate','lr-value',3]])$(id).oninput=()=>{stopTraining();model.config[key]=+$(id).value;$(out).textContent=(+$(id).value).toFixed(precision);recompute();};

$('glow').oninput=()=>{renderer.glow=+$('glow').value;$('glow-value').textContent=renderer.glow.toFixed(2);dirty=true;};
$('reset-weights').onclick=()=>{stopTraining();model.reset();$('training-result').textContent='Reset to the same deterministic starting values.';refresh();};
function trainingStep(){const result=model.trainStep();$('training-result').textContent=`Update ${model.steps}: loss ${format(result.before,5)} → ${format(result.after,5)}. All ${model.parameterCount} active parameters use their computed gradients.`;refresh();}
$('train-one').onclick=()=>{stopTraining();trainingStep();};
$('train-batch').onclick=()=>{if(trainRemaining){stopTraining();return;}trainRemaining=50;$('train-one').disabled=true;const next=()=>{if(!trainRemaining)return;trainingStep();trainRemaining--;$('train-batch').textContent=`Stop training · ${trainRemaining} left`;if(trainRemaining)trainingTimer=setTimeout(next,90);else stopTraining();};next();};
$('see-gradient').onclick=()=>{closeDialog('experiment');goToId('learn.gradient');};
$('start-generation').onclick=()=>{stopTraining();model.startGeneration();refresh();closeDialog('experiment');goToId('generate');notify('Generation starts from <bos>. Use Experiment to append the predicted token.');};
$('append-token').onclick=()=>{stopTraining();const token=model.generateNext();refresh();if(token)$('training-result').textContent='Appended '+token+' from the live distribution.';};
$('teacher-mode').onclick=()=>{stopTraining();model.generation=null;recompute();};

function fillTensorOptions(){const records=[...model.trace,...[...model.tensors.values()].filter(t=>t.parameterOnly)];$('tensor-select').innerHTML=records.map(t=>`<option value="${t.id}">${escape((t.parameterOnly?'Parameter · '+tensorName(t.id):scopeName(t)+' · '+t.label)+' ['+t.shape.join('×')+']')}</option>`).join('');}
function openValues(id){selectedTensor=id;fillTensorOptions();$('tensor-select').value=id;$('value-kind').value='values';fillTensorTable();openDialog('values');}
$('all-values').onclick=()=>openValues(scene().tensor);
$('edit-embeddings').onclick=()=>{closeDialog('experiment');openValues('@source.embedding');};
$('tensor-select').onchange=()=>{selectedTensor=$('tensor-select').value;fillTensorTable();};$('value-kind').onchange=fillTensorTable;
function fillTensorTable(){const t=model.tensors.get(selectedTensor);if(!t)return;const gradients=$('value-kind').value==='grads',matrix=gradients?t.grads:t.values;const editable=!!t.parameterOnly&&!gradients;
  $('tensor-info').textContent=`${t.shape.join(' × ')} · ${gradients?'Derivative of the current mean loss with respect to each coordinate. Select a value to trace its chain rule.':t.parameterOnly?'Learned parameter. Edit a coordinate between −3 and 3; the complete model recalculates.':'Computed tensor. Select a coordinate to enter its calculation.'}`;
  $('edit-message').textContent='';
  if(!matrix){$('tensor-parents').innerHTML='';$('tensor-table').innerHTML='<p class="field-note" style="padding:18px">This diagnostic tensor is created after backpropagation. Higher-order gradients are not calculated.</p>';return;}
  $('tensor-table').innerHTML=`<table><thead><tr><th>r / c</th>${matrix[0].map((_,j)=>`<th>${j}</th>`).join('')}</tr></thead><tbody>${matrix.map((row,i)=>`<tr><th>${i}</th>${row.map((v,j)=>`<td class="${v<0?'negative':''}">${editable?`<input type="number" min="-3" max="3" step="0.01" data-edit-row="${i}" data-edit-col="${j}" aria-label="${escape(t.parameter)} row ${i} column ${j}" value="${Number(v.toFixed(6))}">`:`<button data-inspect-row="${i}" data-inspect-col="${j}">${escape(format(v,6))}</button>`}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
  $('tensor-parents').innerHTML=t.inputs.map(id=>`<button data-parent="${id}">${escape(model.tensors.get(id)?.label||id)} ↗</button>`).join('');
}
$('tensor-table').addEventListener('change',e=>{if(e.target.dataset.editRow===undefined)return;try{const t=model.tensors.get(selectedTensor);const next=e.target.value.trim()===''?NaN:+e.target.value;stopTraining();model.setParameter(t.parameter,+e.target.dataset.editRow,+e.target.dataset.editCol,next);model.history=[model.loss];refresh();$('edit-message').textContent='Value updated. All downstream tensors and loss gradients have been recalculated.';}catch(error){$('edit-message').textContent=error.message;}});
$('tensor-table').addEventListener('click',e=>{const b=e.target.closest('[data-inspect-row]');if(!b)return;const t=model.tensors.get(selectedTensor),gradient=$('value-kind').value==='grads';const id=gradient?'backward.'+t.id:t.id;if(!model.tensors.has(id)||(!gradient&&t.parameterOnly))return;closeDialog('values');goToId(id,{row:+b.dataset.inspectRow,col:+b.dataset.inspectCol,lock:true});});
$('tensor-parents').addEventListener('click',e=>{const b=e.target.closest('[data-parent]');if(!b)return;selectedTensor=b.dataset.parent;$('tensor-select').value=selectedTensor;fillTensorTable();});
$('open-reference').onclick=()=>{renderer.startTime=clock;phase=0;manual=false;ambient=true;$('motion').textContent='Ⅱ';updateCalculation(true);dirty=true;};

preference.addEventListener('change',e=>{reduced=e.matches;ambient=!reduced;renderer.reduced=reduced;if(reduced){phase=1;play=false;}updateCalculation(true);dirty=true;});
document.addEventListener('keydown',e=>{if(e.defaultPrevented)return;if(document.querySelector('dialog[open]')||['INPUT','SELECT','TEXTAREA','BUTTON','A'].includes(e.target.tagName))return;if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();goTo(index+(e.key==='ArrowRight'?1:-1));}if(e.code==='Space'){e.preventDefault();$('play').click();}});
document.addEventListener('visibilitychange',()=>{last=0;});
function frame(now){const dt=last?Math.min((now-last)/1000,.05):0;last=now;
 if(!document.hidden&&!document.querySelector('dialog[open]')){
  if(index<0||Math.abs(targetProgress-visualProgress)>.0001){visualProgress=reduced?targetProgress:visualProgress+(targetProgress-visualProgress)*(1-Math.exp(-dt/.095));if(Math.abs(targetProgress-visualProgress)<.0001)visualProgress=targetProgress;applyProgress(visualProgress);}
  if(ambient&&!reduced){clock+=dt;dirty=true;}
  if(play){phase=Math.min(1,phase+dt/10);if(phase>=1){play=false;$('play').textContent='▶';$('play').setAttribute('aria-label','Play the calculation');}updateCalculation();dirty=true;}
  if(dirty&&now-lastDraw>20){renderer.reduced=reduced;renderer.paused=!ambient;renderer.draw(clock);lastDraw=now;dirty=false;}
 }
 requestAnimationFrame(frame);
}
hashIndex(course,initialHash);makeCourse();$('render-status').textContent='2D · FIGURE 1';renderer.reduced=reduced;updateStats();
window.scrollTo({top:course[hashIndex(course,initialHash)].start*unit,behavior:'instant'});targetProgress=visualProgress=course[hashIndex(course,initialHash)].start;applyProgress(visualProgress);layout();requestAnimationFrame(frame);

// A non-blocking introduction: the first real scroll dismisses it without capturing the wheel.
const welcomeOrigin=scrollY;
function dismissWelcome(){if($('scroll-welcome').hidden||$('scroll-welcome').classList.contains('leaving'))return;$('scroll-welcome').classList.add('leaving');setTimeout(()=>{$('scroll-welcome').hidden=true;},reduced?0:240);}
$('dismiss-welcome').onclick=dismissWelcome;$('start-learning').onclick=dismissWelcome;
window.addEventListener('scroll',()=>{if(Math.abs(scrollY-welcomeOrigin)>8)dismissWelcome();},{passive:true});
