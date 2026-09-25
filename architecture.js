import {svgLabel} from './math.js?v=math-space-3';
import {canSpatial} from './live-space.js?v=math-space-3';
import {LiveNarrative} from './microscope.js?v=math-space-3';
import {simulationMarkup,diagramKind} from './diagrams.js?v=math-space-3';
import {VectorSpace,vectorPages} from './vector-space.js?v=math-space-3';
import {OperationMotion} from './operation-motion.js?v=math-space-3';
import {clamp} from './space.js?v=math-space-3';
import {format} from './course.js?v=math-space-3';
export const blocks=[
 ['source.embedding',190,570,170,48,'Input|Embedding','embedding'],
 ['target.embedding',550,570,170,48,'Output|Embedding','embedding'],
 ['enc.self',190,422,170,66,'Multi-Head|Attention','attention'],
 ['enc.self.norm',190,376,170,32,'Add & Norm','norm'],
 ['enc.ff',190,310,170,48,'Feed|Forward','ff'],
 ['enc.ff.norm',190,264,170,32,'Add & Norm','norm'],
 ['dec.self',550,422,170,66,'Masked|Multi-Head|Attention','attention'],
 ['dec.self.norm',550,376,170,32,'Add & Norm','norm'],
 ['dec.cross',550,310,170,48,'Multi-Head|Attention','attention'],
 ['dec.cross.norm',550,264,170,32,'Add & Norm','norm'],
 ['dec.ff',550,198,170,48,'Feed|Forward','ff'],
 ['dec.ff.norm',550,152,170,32,'Add & Norm','norm'],
 ['output.linear',550,97,170,36,'Linear','linear'],
 ['output.softmax',550,43,170,36,'Softmax','softmax']
];
export const routes=[
 ['source-in','M275 643V618','source.embedding'],['source-embed','M275 570V550','source.position'],['source-pos','M132 532H257','source.position'],['enc-input','M275 514V499H222V488 M275 499V488 M275 499H328V488','enc.self'],
 ['enc-attention','M275 422V408','enc.self.norm'],['enc-residual','M275 506H166V392H190','enc.self.norm'],['enc-normalized','M275 376V358','enc.ff'],['enc-ff','M275 310V296','enc.ff.norm'],['enc-ff-residual','M275 367H166V280H190','enc.ff.norm'],
 ['memory','M275 264V236H437V369','dec.cross'],['memory-key','M437 369H578V358','dec.cross'],['memory-value','M437 369H620V358','dec.cross'],
 ['target-in','M635 643V618','target.embedding'],['target-embed','M635 570V550','target.position'],['target-pos','M785 532H653','target.position'],['dec-input','M635 514V499H582V488 M635 499V488 M635 499H688V488','dec.self'],
 ['dec-masked','M635 422V408','dec.self.norm'],['dec-residual','M635 506H746V392H720','dec.self.norm'],['dec-query','M635 376V369H672V358','dec.cross'],['dec-cross','M635 310V296','dec.cross.norm'],['dec-cross-residual','M635 369H746V280H720','dec.cross.norm'],['dec-normalized','M635 264V246','dec.ff'],['dec-ff','M635 198V184','dec.ff.norm'],['dec-ff-residual','M635 255H746V168H720','dec.ff.norm'],['linear','M635 152V133','output.linear'],['softmax','M635 97V79','output.softmax'],['prediction','M635 43V22','output.softmax']
];
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export function blockFor(t){
 let id=(t?.backwardOf||t?.id||'').replace(/^@/,'');if(id.startsWith('learn.'))return'enc.self';if(id.startsWith('loss.'))return'output.softmax';
 if(id.startsWith('source.'))return /position|input/.test(id)?'source.position':'source.embedding';if(id.startsWith('target.'))return /position|input/.test(id)?'target.position':'target.embedding';
 if(id.startsWith('output.')||id.startsWith('vocab.'))return /project|logits|vocab/.test(id)?'output.linear':'output.softmax';
 const m=id.match(/^(enc|dec)\.\d+\.(self|cross|ff)/);if(m)return m[1]+'.'+m[2]+(/\.norm\.|\.residual/.test(id)?'.norm':'');return'all';
}
export function architectureMarkup(){return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="80 0 770 690" role="img" aria-labelledby="architecture-title architecture-desc"><title id="architecture-title">The Transformer architecture, after Figure 1 of Attention Is All You Need</title><desc id="architecture-desc">Encoder at left. Decoder at right. Residual connections go around each sublayer. Final encoder memory supplies keys and values to decoder cross-attention. Inputs and shifted outputs enter from below; probabilities leave at the top.</desc><defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10" fill="#adc7d0"/></marker><filter id="glow"><feGaussianBlur stdDeviation="3"/></filter></defs>
 <g class="stack-frames" fill="#0b1520" fill-opacity=".52" stroke="#7e9fb5" stroke-opacity=".35"><rect x="153" y="249" width="231" height="260" rx="13"/><rect x="526" y="140" width="231" height="369" rx="13"/></g>
 <g class="arch-wires" fill="none" stroke="#74939f" stroke-width="1.65" stroke-linejoin="round" marker-end="url(#arrow)">${routes.map(([id,d,key])=>`<path id="wire-${id}" data-route="${key}" d="${d}"/>`).join('')}</g>
 <g class="layer-shadows" fill="#20313c" stroke="#527080" opacity="0">${blocks.map(([key,x,y,w,h])=>`<rect data-shadow="${key}" x="${x+7}" y="${y+7}" width="${w}" height="${h}" rx="8"/>`).join('')}</g>
 <g>${blocks.map(([key,x,y,w,h,label,type])=>{const palette={embedding:['#352034','#d5a6ce'],ff:['#24394a','#8ab7d7'],attention:['#423c21','#d4c36e'],norm:['#293d28','#b3c67b'],linear:['#392f48','#b4a2d4'],softmax:['#183b37','#7bd1c0']},lines=label.split('|');return `<g class="arch-block ${type}" data-block="${key}" role="button" tabindex="0" aria-label="Explore ${escape(label.replaceAll('|',' '))}"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="7" fill="${palette[type][0]}" stroke="${palette[type][1]}" stroke-width="1.5"/>${lines.map((line,i)=>`<text x="${x+w/2}" y="${y+h/2-(lines.length-1)*10+6+i*20}" text-anchor="middle" fill="#e4f3fa" font-family="sans-serif" font-size="18">${escape(line)}</text>`).join('')}</g>`;}).join('')}</g>
 <g class="position-node" fill="#121d22" stroke="#acc4d5" stroke-width="1.6" data-block="source.position" role="button" tabindex="0" aria-label="Source positional encoding"><circle cx="275" cy="532" r="18"/><path d="M266 532H284M275 523V541"/><circle class="wave-circle" cx="114" cy="532" r="18"/><path d="M100 532Q107 512 114 532T128 532"/></g>
 <g class="position-node" fill="#121d22" stroke="#acc4d5" stroke-width="1.6" data-block="target.position" role="button" tabindex="0" aria-label="Target positional encoding"><circle cx="635" cy="532" r="18"/><path d="M626 532H644M635 523V541"/><circle class="wave-circle" cx="803" cy="532" r="18"/><path d="M789 532Q796 512 803 532T817 532"/></g>
 <g class="arch-notes" fill="#9cb8c5" font-family="Arial,sans-serif" font-size="14"><text x="121" y="382" text-anchor="middle" class="repeat-label">N×</text><text x="790" y="329" text-anchor="middle" class="repeat-label">N×</text><text x="268" y="228" text-anchor="middle">ENCODER</text><text x="480" y="207" text-anchor="middle" transform="rotate(-90 480 207)">DECODER</text><text x="111" y="570" text-anchor="middle"><tspan x="111">Positional</tspan><tspan x="111" dy="18">encoding</tspan></text><text x="804" y="570" text-anchor="middle"><tspan x="804">Positional</tspan><tspan x="804" dy="18">encoding</tspan></text><text x="275" y="663" text-anchor="middle">Inputs</text><text x="635" y="663" text-anchor="middle">Outputs (shifted right)</text><text x="635" y="14" text-anchor="middle">Output probabilities</text><text x="449" y="321" font-size="12">K, V</text><text x="682" y="372" font-size="12">Q</text></g>
 <g class="signal-packets" fill="#afffe6">${routes.map(([id])=>`<circle data-packet="${id}" r="3.2" opacity="0"/>`).join('')}</g>
 <g id="operation-lens"></g>
 <g id="live-badge" pointer-events="none"><rect width="157" height="37" rx="8" fill="#071719" stroke="#8df5d2"/><text x="78" y="16" text-anchor="middle" fill="#a4f4d4" font-size="10" font-family="monospace" id="badge-label">LIVE VALUE</text><text x="78" y="30" text-anchor="middle" fill="#fff" font-size="13" font-family="monospace" id="badge-value"></text></g>
 </svg>`;}
export function lensMarkup(t,m,options={},scene={}){
 const head=t?.attention||'',key=t?.id||'',row=Math.min(options.row||0,(t?.shape[0]||1)-1),col=Math.min(options.col||0,(t?.shape[1]||1)-1);
 const frame=(title,body)=>`<g class="lens"><rect class="lens-frame" fill="#091317" stroke="#29424b" x="128" y="23" width="280" height="193" rx="12"/><text class="lens-title" fill="#8fc8bc" font-family="Arial,sans-serif" font-size="12" x="268" y="42" text-anchor="middle">${escape(title)}</text>${body}</g>`;
 const box=(x,y,w,label,active=false)=>`<g class="lens-box ${active?'lit':''}"><rect x="${x}" y="${y}" width="${w}" height="22" rx="5" fill="${active?"#284739":"#152830"}" stroke="${active?"#b6f2ba":"#557781"}"/>${svgLabel(x+w/2,y+15,label,11,"#c4d9de")}</g>`;
 if(!t){if(!['problem-context','problem-recurrence','solution-attention'].includes(scene.id))return'';const serial=scene.id==='problem-recurrence';return frame(serial?'RECURRENT DEPENDENCE':'CONTEXT BETWEEN POSITIONS',`<g class="lens-links" fill="none" stroke="#7e9eaa" stroke-width="1.2" marker-end="url(#arrow)"><path d="M193 118H243M293 118H343"/>${serial?'':'<path d="M178 99Q258 52 358 99M183 135Q268 185 357 135"/>'}</g>${['the','cat','sleeps'].map((word,i)=>`<circle cx="${178+i*90}" cy="118" r="24" fill="#182d31" stroke="#86cbb9"/><text x="${178+i*90}" y="123" text-anchor="middle" fill="#d8ffec" font-size="15">${word}</text>`).join('')}<text x="268" y="192" text-anchor="middle" fill="#9cbdc5" font-size="13">${serial?'Later states depend on earlier states.':'A direct interaction can span positions.'}</text>`);}
 if(t.op==='backward'||t.op==='gradient')return frame('CHAIN RULE · BACKWARD',`${box(159,65,218,'incoming gradient × local derivative',true)}<path class="lens-links" fill="none" stroke="#7e9eaa" stroke-width="1.2" marker-end="url(#arrow)" d="M268 87V109"/>${box(174,110,188,'sum over every consumer')}<path class="lens-links" fill="none" stroke="#7e9eaa" stroke-width="1.2" marker-end="url(#arrow)" d="M268 132V154"/>${box(174,155,188,'loss sensitivity: '+format(t.values[row][col],5),true)}`);
 if((/\.(Q|K|V)$/.test(key)||t.op==='concat'||key.endsWith('.output'))&&/\.(self|cross)\./.test(key)&&!key.includes('.norm.'))return frame('MULTI-HEAD ATTENTION · FIGURE 2',`<g class="lens-links" fill="none" stroke="#7e9eaa" stroke-width="1.2" marker-end="url(#arrow)"><path d="M182 202V196M268 202V196M354 202V196M182 174V153M268 174V153M354 174V153M268 131V116M268 94V79"/></g>${box(145,174,74,'Linear Q',key.endsWith('.Q'))}${box(231,174,74,'Linear K',key.endsWith('.K'))}${box(317,174,74,'Linear V',key.endsWith('.V'))}<rect x="155" y="135" width="230" height="22" rx="5" fill="#1b2d30" stroke="#557781"/>${box(151,131,230,'Scaled dot-product attention × 2')}${box(172,94,192,'Concat',t.op==='concat')}${box(172,57,192,'Linear: output projection',key.endsWith('.output'))}<text x="182" y="211" text-anchor="middle" fill="#c1d6dc" font-size="10">query input</text><text x="268" y="211" text-anchor="middle" fill="#c1d6dc" font-size="10">key input</text><text x="354" y="211" text-anchor="middle" fill="#c1d6dc" font-size="10">value input</text>`);
 if(head||/\.(self|cross)\./.test(key)&&!key.includes('.norm.')&&!key.includes('residual')){
  const phase=key.endsWith('.Z')?'mix':/\.A$|\.exp$|\.sum$|\.shift$|\.max$/.test(key)?'softmax':key.endsWith('.masked')?'mask':/scaled|temperature/.test(key)?'scale':'scores';
  return frame('INSIDE ATTENTION · FIGURE 2',`<g class="lens-links" fill="none" stroke="#7e9eaa" stroke-width="1.2" marker-end="url(#arrow)"><path d="M187 197V183H211V176M243 197V183H225V176M218 154V149M218 127V122M218 100V95M218 73V68M355 197V82V68"/></g>${box(159,154,118,'MatMul: QKᵀ',phase==='scores')}${box(159,127,118,'Scale by √dₖ',phase==='scale')}${box(159,100,118,'Mask (if needed)',phase==='mask')}${box(159,73,118,'Softmax',phase==='softmax')}${box(300,46,94,'× V → output',phase==='mix')}<path class="lens-links" fill="none" stroke="#7e9eaa" stroke-width="1.2" marker-end="url(#arrow)" d="M218 73V57H300"/><text x="184" y="207" class="lens-label" fill="#bed6da" font-family="Arial,sans-serif" font-size="12">Q</text><text x="238" y="207" class="lens-label" fill="#bed6da" font-family="Arial,sans-serif" font-size="12">K</text><text x="351" y="207" class="lens-label" fill="#bed6da" font-family="Arial,sans-serif" font-size="12">V</text>`);
 }
 if(key.includes('.ff.')&&!key.includes('.norm.')&&!key.includes('residual'))return frame('INSIDE FEED FORWARD',`${box(156,59,224,'Linear: 4 → 8 features',/expand|biased/.test(key))}<path class="lens-links" fill="none" stroke="#7e9eaa" stroke-width="1.2" marker-end="url(#arrow)" d="M268 81V103"/>${box(156,104,224,'ReLU: $max(0, x)$',t.op==='relu')}<path class="lens-links" fill="none" stroke="#7e9eaa" stroke-width="1.2" marker-end="url(#arrow)" d="M268 126V148"/>${box(156,149,224,'Linear: 8 → 4 features',/contract|bias$/.test(key))}<text x="268" y="199" text-anchor="middle" class="lens-label" fill="#bed6da" font-family="Arial,sans-serif" font-size="12">Same function, independently per position.</text>`);
 if(key.includes('.norm.')||key.includes('residual'))return frame('INSIDE ADD & NORM',`${box(155,61,226,'$x + Sublayer(x)$',key.includes('residual'))}<path class="lens-links" fill="none" stroke="#7e9eaa" stroke-width="1.2" marker-end="url(#arrow)" d="M268 83V105"/>${box(155,106,226,'$(x − μ) / √(σ² + ε)$',!key.includes('residual')&&!key.endsWith('output'))}<path class="lens-links" fill="none" stroke="#7e9eaa" stroke-width="1.2" marker-end="url(#arrow)" d="M268 128V150"/>${box(155,151,226,'Learned $γ × x̂ + β$',key.endsWith('output'))}<text x="268" y="200" text-anchor="middle" class="lens-label" fill="#bed6da" font-family="Arial,sans-serif" font-size="12">Statistics are across one token’s features.</text>`);
 if(key.startsWith('output.')||key.startsWith('loss.')){const p=m.probabilities[Math.min(row,m.probabilities.length-1)],items=p.map((v,i)=>({v,i})).sort((a,b)=>b.v-a.v).slice(0,4);return frame('LIVE NEXT-TOKEN DISTRIBUTION',items.map((x,i)=>`<text x="145" y="${72+i*33}" class="lens-label" fill="#bed6da" font-family="Arial,sans-serif" font-size="12">${escape(m.vocabulary?.[x.i]||['<pad>','<bos>','<eos>','the','cat','sleeps','le','chat','dort','a','dog','runs','un','chien','court'][x.i])}</text><rect x="210" y="${60+i*33}" width="${Math.max(1,x.v*130)}" height="15" rx="3" fill="#7bccb3"/><text x="388" y="${72+i*33}" class="lens-label" fill="#bed6da" font-family="Arial,sans-serif" font-size="12" text-anchor="end">${(x.v*100).toFixed(1)}%</text>`).join(''));}
 if(t.scope==='source'||t.scope==='target'){const vector=t.values[row];return frame(t.op==='position'?'POSITION ENTERS THE REPRESENTATION':'ONE TOKEN’S LIVE COORDINATES',`<text x="268" y="81" class="lens-label" fill="#bed6da" font-family="Arial,sans-serif" font-size="12" text-anchor="middle">${escape(t.tokens?.[row]|| (t.scope==='source'?m.sourceTokens:m.targetTokens)[row]||'position '+row)} · row ${row}</text>${vector.slice(0,4).map((v,i)=>`<rect x="${145+i*62}" y="105" width="57" height="39" rx="7" fill="#18282e" stroke="#75a7af"/><text x="${173+i*62}" y="129" class="lens-label" fill="#bed6da" font-family="Arial,sans-serif" font-size="12" text-anchor="middle">${format(v,2)}</text>`).join('')}<text x="268" y="183" class="lens-label" fill="#bed6da" font-family="Arial,sans-serif" font-size="12" text-anchor="middle">${t.op==='position'?'Alternating sine and cosine coordinates.':'The full precision remains in All values.'}</text>`);}
 return'';
}
export function buildWorld(model,scene,options={}){return{model,scene,options};}
export class Renderer{
 constructor(canvas,labels){
  this.canvas=canvas;this.labels=labels;canvas.hidden=true;labels.hidden=true;this.glow=1.15;this.reduced=false;this.paused=false;
  this.camera={mode:'2d',zoom:1,pan:[0,0,0],top:0,bottom:0,update(){}};
  this.host=document.createElement('div');this.host.id='architecture-stage';this.host.innerHTML=architectureMarkup();canvas.parentElement.insertBefore(this.host,canvas);this.svg=this.host.querySelector('svg');
  this.routes=routes.map(([id,,key])=>{const path=this.svg.querySelector('#wire-'+id);return{key,path,length:path.getTotalLength(),packet:this.svg.querySelector(`[data-packet="${id}"]`)}});this.nodes=[...this.svg.querySelectorAll('[data-block]')];
  this.focus=document.createElement('div');this.focus.className='focused-simulation';this.host.append(this.focus);this.motion=new OperationMotion(this.focus);this.narrative=new LiveNarrative();this.vector=new VectorSpace(this.host);this.vector.host.hidden=true;this.active='';this.resize();
 }
 resize(){this.w=innerWidth;this.h=innerHeight;this.vector?.render();}
 setWorld(world){
  const changed=this.world?.scene.id!==world.scene.id;this.world=world;this.narrative.update(null);if(changed){this.startTime=this.lastTime||0;this.lastSimulation='';}
  const t=world.model.tensors.get(world.scene.tensor),active=world.scene.focus||blockFor(t);this.currentTensor=t;this.isVector=vectorPages.has(world.scene.reference)||(this.camera.mode==='3d'&&canSpatial(t,world.scene));this.vector.host.hidden=!this.isVector;document.body.classList.toggle('spatial-active',this.isVector);
  const content=this.isVector?'':simulationMarkup(world.model,world.scene,world.options);
  if(content!==this.lastSimulation){this.focus.innerHTML=content;this.lastSimulation=content;if(content)this.motion.bind(world.model,world.scene,world.options);}else if(content)this.motion.options=world.options;
  this.focused=!!content;this.focus.hidden=this.isVector||!this.focused;this.svg.style.display=this.isVector||this.focused?'none':'';this.svg.style.opacity='1';this.svg.style.pointerEvents=this.isVector||this.focused?'none':'auto';this.host.dataset.focus=active;
  if(!this.isVector&&!this.focused){const lens=lensMarkup(t,world.model,world.options,world.scene);if(lens!==this.lastLens){this.svg.querySelector('#operation-lens').innerHTML=lens;this.lastLens=lens;}}
  if(active!==this.active){this.active=active;this.nodes.forEach(n=>n.classList.toggle('active',active==='all'||n.dataset.block===active));this.routes.forEach(r=>r.path.classList.toggle('active',active==='all'||r.key===active));}
  const label=this.svg.querySelector('#badge-label'),value=this.svg.querySelector('#badge-value'),badge=this.svg.querySelector('#live-badge');badge.style.opacity=t?'1':'0';
  if(t){const row=clamp(world.options.row||0,0,t.shape[0]-1),col=clamp(world.options.col||0,0,t.shape[1]-1);label.textContent=`LIVE [${row}, ${col}] · ${t.shape.join('×')}`;value.textContent=format(t.values[row][col],6);const b=blocks.find(x=>x[0]===active);badge.setAttribute('transform',`translate(365,${clamp((b?.[2]??480)-4,90,575)})`);}
  this.host.dataset.scope=t?.scope||'intro';
 }
 draw(time=0){
  if(!this.world)return;this.lastTime=time;const elapsed=time-(this.startTime||0),phase=this.world.options.phase||0;
  this.host.classList.toggle('motion-paused',this.reduced||this.paused);this.host.style.setProperty('--glow',this.glow);this.host.style.setProperty('--depth-offset',`${(this.world.options.explode||1)*5}px`);
  if(this.isVector){const progress=this.reduced?1:phase;this.vector.zoom=this.camera.zoom;this.vector.set(this.world.model,this.world.scene,progress,this.world.options.row||0,this.world.options.col||0);this.narrative.update(this.vector.state.detail);return;}
  this.focus.style.opacity='1';this.focus.style.pointerEvents='auto';this.focus.style.transform=`scale(${this.camera.zoom})`;
  if(this.focused){const frame=this.motion.draw(elapsed,phase,this.reduced);if(frame)this.onTerm?.(frame);this.narrative.update(this.motion.narrative);return;}
  this.svg.style.transform=`scale(${this.camera.zoom})`;this.svg.querySelector('.signal-packets').style.filter=`drop-shadow(0 0 ${this.glow*3}px #8effd8)`;
  const backwards=this.currentTensor?.op==='backward'||this.currentTensor?.op==='gradient';for(const r of this.routes){const active=this.active==='all'||r.key===this.active,f=((time*.22)%1+1)%1,p=r.path.getPointAtLength((backwards?1-f:f)*r.length);r.packet.setAttribute('cx',p.x);r.packet.setAttribute('cy',p.y);r.packet.style.opacity=active&&!this.reduced?'.95':'0';}
 }
 pick(){return null;}
}
