import {svgSymbols} from './math.js?v=expanded-space-2';
import {neuronState,neuronMarkup,neuronPhase,normState,normMarkup,normPhase,attentionMarkup} from './microscope.js?v=expanded-space-2';
import {referenceBody} from './reference-designs.js?v=expanded-space-2';
import {format,calculate} from './course.js?v=expanded-space-2';
import {VOCAB} from './model.js?v=expanded-space-2';
import {sourceInfo} from './data.js?v=expanded-space-2';
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const colors=['#75cef0','#ade6a3','#ebce77','#c6a3f4','#ed9990'];
const text=(x,y,s,size=18,color='#cfdfdf',anchor='middle')=>`<text x="${x}" y="${y}" text-anchor="${anchor}" fill="${color}" font-size="${size}" font-family="${size>23?'Georgia,serif':'Arial,sans-serif'}">${svgSymbols(s)}</text>`;
const line=(x,y,xx,yy,color='#8ca8ac')=>`<path class="flow-wire" d="M${x} ${y}L${xx} ${yy}" stroke="${color}" fill="none" stroke-width="1.7" marker-end="url(#flow-arrow)"/>`;
const group=(content,order=0)=>`<g class="diagram-asset" data-reveal="${order}">${content}</g>`;
const token=(x,y,s,i=0,w=110)=>`<g class="token-asset"><rect x="${x-w/2}" y="${y-22}" width="${w}" height="44" rx="7" fill="#071014" stroke="${colors[i%colors.length]}" stroke-width="1.3"/>${text(x,y+6,s,20,colors[i%colors.length])}</g>`;
export const referenceKinds=['tokens','columns','geometry','analogy','analogy','geometry','contexts','contexts','heads','output','graph','columns','projection','geometry','projection','projections','heatmap','heatmap','heatmap','bars','heatmap','heatmap','prefix','prefix','heatmap','projection','projection','projection','mix','projections','projection','projection','parameters','heads','recurrent','recurrent','recurrent','graph','graph','graph','graph','projections','columns','matrix','branches','projection','projections','projection','bars','bars','heatmap','mix','mix','heatmap','heatmap','branches','projection','bars','pipeline','graph','contexts','norm','architecture','ff','architecture','position','position','ff','recurrent','columns'];
export function diagramKind(t,scene){if(scene.reference)return referenceKinds[scene.reference-1];const id=t?.id||'';
 if(!t)return /problem-recurrence/.test(scene.id)?'recurrent':/problem-context|solution-attention/.test(scene.id)?'graph':'architecture';
 if(t.op==='backward'||t.op==='gradient'||id.startsWith('learn.'))return'gradient';
 if(id.startsWith('output.')||id.startsWith('loss.'))return'output';
 if(id.includes('.norm.')||id.includes('residual'))return'norm';
 if(id.includes('.ff.'))return'ff';
 if(t.op==='concat'||/\.self\.output$|\.cross\.output$/.test(id))return'heads';
 if(id.endsWith('.Z'))return'mix';
 if(id.endsWith('.A')||id.endsWith('.scores')||id.endsWith('.masked'))return'heatmap';
 if(t.attention&&/max|shift|exp|sum|scaled|temperature/.test(id))return'bars';
 if(/\.(Q|K|V|KT)$/.test(id))return'projection';
 if(/position|\.input$/.test(id))return'position';
 if(id.endsWith('.ids'))return'tokens';
 return'columns';
}
function matrix(t,x,y,w,h,options={},title){if(!t)return'';const rows=t.shape[0],cols=t.shape[1],cw=w/cols,ch=h/rows;
 const selected=t.id===options.tensor;const max=Math.max(.001,...t.values.flat().filter(Number.isFinite).map(Math.abs));
 const cells=t.values.map((row,r)=>row.map((v,c)=>{const rr=t.transposed?c:t.columnVector?t.sourceRow:(t.rowMap?.[r]??r),cc=t.transposed?r:t.columnVector?r:c,active=selected&&rr===options.row&&cc===options.col,operand=options.operands?.some(a=>a?.tensor===t.id&&a.row===rr&&a.col===cc),color=active?'#ebce77':operand?'#afffe0':colors[r%3];return `<g class="tensor-cell ${active?'selected-cell':operand?'operand-cell':''}" data-tensor="${t.inspectable===false?'':esc(t.id)}" data-row="${rr}" data-col="${cc}" tabindex="0" role="button" aria-label="${esc(t.label)} row ${rr}, column ${cc}: ${format(v,6)}"><rect x="${x+c*cw}" y="${y+r*ch}" width="${cw-2}" height="${ch-2}" rx="3" fill="${!Number.isFinite(v)?'#261015':color}" fill-opacity="${active?.23:operand?.17:.025+.19*Math.min(1,Math.abs(v)/max)}" stroke="${color}" stroke-opacity="${active||operand?1:.25}"/>${text(x+(c+.5)*cw,y+(r+.5)*ch+5,format(v,2),Math.min(18,cw*.23,ch*.4),color)}</g>`}).join('')).join('');
 return `<g class="tensor-matrix"><path class="matrix-shadow" d="M${x+7} ${y+7}h${w}v${h}h-${w}z" fill="#142a32" stroke="#497782" opacity="0"/><path d="M${x-5} ${y-3}h-6v${h+5}h6M${x+w+3} ${y-3}h6v${h+5}h-6" stroke="#93b4bb" fill="none"/>${text(x+w/2,y-19,title||t.label,16,'#b4cdc9')}${cells}${text(x+w/2,y+h+22,`${rows} × ${cols}`,12,'#789699')}</g>`;
}
function bars(values,labels,x,y,w,h,title,probability=false,fixedMax=null){const max=fixedMax??(probability?1:Math.max(.01,...values.filter(Number.isFinite).map(Math.abs))),cw=w/values.length,negative=values.some(v=>v<0&&Number.isFinite(v)),baseline=negative?y+h*.55:y+h,span=negative?h*.39:h-30;
 return text(x+w/2,y-30,title,19)+`<path d="M${x} ${baseline}h${w}" stroke="#71898a"/>`+values.map((v,i)=>{const height=Number.isFinite(v)?Math.abs(v)/max*span:0;return group(`<rect class="value-bar" data-baseline="${baseline}" data-negative="${v<0}" x="${x+i*cw+cw*.18}" y="${v<0?baseline:baseline-height}" width="${cw*.64}" height="${height}" fill="${colors[i%3]}" fill-opacity=".7"/>${text(x+(i+.5)*cw,(v<0?baseline+height+17:baseline-height-12),probability?(v*100).toFixed(1)+'%':format(v,3),14,colors[i%3])}${text(x+(i+.5)*cw,y+h+27,labels[i]??i,15)}`,i*.1)}).join('');
}
function arithmetic(m,t,o){if(!t)return'';const calc=calculate(m,t.id,o.row||0,o.col||0);if(!calc)return'';const terms=calc.terms.slice(0,8),w=Math.min(114,750/Math.max(1,terms.length)),start=500-terms.length*w/2;
 return `<g class="arithmetic-tray">${text(500,506,`${t.label} [${o.row||0}, ${o.col||0}]`,17,'#e5d38c')}${terms.map((term,i)=>`<g class="product-asset ${i===o.term?'active-product':''}" data-reveal="${i*.045}"><rect x="${start+i*w+2}" y="526" width="${w-5}" height="45" rx="5" fill="#0c171b" stroke="${i===o.term?'#edce79':'#264046'}"/>${text(start+(i+.5)*w,552,format(term.value,4),15,i===o.term?'#edce79':'#a5c5cc')}</g>`).join('')}<g class="accumulation-readout">${text(500,607,(calc.additive?'Σ contributions':'Result')+' = '+format(calc.result,6),25,'#bcffe2')}</g>${text(500,638,calc.terms.length>8?'First 8 contributions shown; the result includes all '+calc.terms.length+'. See the explanation for the full sum.':'Select any cell to trace its exact operands · scroll to advance',13,'#69888e')}</g>`;
}
export function simulationMarkup(m,scene,o={}){
 const t=m.tensors.get(scene.tensor),kind=diagramKind(t,scene);if(kind==='architecture')return'';
 const calc=t?calculate(m,t.id,o.row||0,o.col||0):null,term=calc?.terms[o.term||0];o={...o,tensor:t?.id,operands:[term?.a,term?.b]};
 const id=t?.id||'',head=t?.attention||id.match(/^(enc|dec)\.\d+\.(self|cross)/)?.[0]+'.h0',get=s=>m.tensors.get(s),a=t&&get(t.inputs[0]),b=t&&get(t.inputs[1]);
 const att=head&&get(head+'.A')?head:'enc.0.self.h0',X=get('source.input'),Q=get(att+'.Q'),K=get(att+'.K'),V=get(att+'.V'),A=get(att+'.A'),Z=get(att+'.Z');
 const words=(t?.scope==='target'||id.startsWith('dec.'))?m.targetTokens:m.sourceTokens,keyWords=id.startsWith('dec.')&&!id.includes('.cross.')?m.targetTokens:m.sourceTokens;let body='',subtitle='';
 const mx=(tt,x,y,w,h,label)=>matrix(tt,x,y,w,h,o,label);
 const custom=!scene.reference&&/\.(self|cross)\.h\d+\.(max|shift|exp|sum|A)$/.test(id)?attentionMarkup(m,t,o.row||0):referenceBody(m,scene,o,{text,line,token,matrix,bars});
 if(custom){body=custom.body;subtitle=custom.subtitle;}else if(kind==='tokens'){
  subtitle='Text → token boundaries → vocabulary indices';
  body=words.map((word,i)=>{const x=170+i*660/Math.max(1,words.length-1);return group(token(x,190,word,i)+line(x,218,x,287,colors[i%3])+token(x,321,VOCAB.indexOf(word),i)+text(x,378,'position '+i,16,'#88a6ac'),i*.16)}).join('');
  body+=text(500,437,'This teaching model uses word tokens; production tokenizers may split words.',16,'#91adb3');
 }else if(kind==='columns'){
  subtitle='One token, one column of coordinates';const tensor=t?.shape[1]>1?t:get('source.lookup');body=tensor.values.map((v,i)=>{const x=140+i*720/Math.max(1,tensor.shape[0]-1);const col={...tensor,shape:[v.length,1],values:v.map(n=>[n]),columnVector:true,sourceRow:i};return group(token(x,137,words[i]||'position '+i,i)+line(x,163,x,208,colors[i%3])+mx(col,x-54,235,108,190,'x'+i)+text(x,470,'row '+i+' in the live tensor',13,'#88a6ac'),i*.12)}).join('');
 }else if(['projection','matrix'].includes(kind)){
  subtitle=t?.op==='transpose'?'Exchange rows and columns':`${a?.shape.join('×')||'input'} → ${t?.shape.join('×')||'output'} · highlighted operands form one coordinate`;
  if(kind==='matrix'){body=mx(t,280,170,440,260,'X · tokens × features');}
  else if(b){body=group(mx(a,48,207,250,205,'Input · '+(a?.shape.join('×')||'')),0)+text(335,310,'×',35,'#e6ce7b')+group(mx(b,374,207,250,205,b.label),.2)+text(659,310,'=',35,'#e6ce7b')+group(mx(t,704,207,250,205,t.label),.4);}
  else{body=mx(a,90,190,320,240,a?.label)+line(435,300,563,300)+mx(t,590,190,320,240,t?.label);}
 }else if(['branches','projections'].includes(kind)){
  subtitle='One representation. Three independently learned projections.';body=mx(X,55,242,270,150,'X · source representation');
  [Q,K,V].forEach((tt,i)=>{const yy=115+i*130;body+=group(line(345,316,570,yy+36,colors[i])+mx(tt,600,yy,240,70,['Q · queries','K · keys','V · values'][i]),i*.2)});
 }else if(kind==='heatmap'){
  const original=t?.shape[1]>1?t:A;const sourceConvention=scene.reference&&scene.reference>=17&&scene.reference<=25;const tt=sourceConvention?{...original,transposed:true,shape:[original.shape[1],original.shape[0]],values:original.values[0].map((_,i)=>original.values.map(row=>row[i]))}:original;
  subtitle=sourceConvention?scene.reference===21?'Column-vector form: A = softmax_columns(KᵀQ / √dₖ), Z = VA':'Source layout: queries across columns · keys down rows · select a cell to trace it':'Queries down · keys across · each cell is an actual pairwise interaction';
  body=mx(tt,266,166,450,270,tt.label);
  for(let i=0;i<tt.shape[0];i++)body+=text(242,166+(i+.5)*270/tt.shape[0]+5,words[i]||i,16,colors[i%3],'end');
  for(let i=0;i<tt.shape[1];i++)body+=text(266+(i+.5)*450/tt.shape[1],121,keyWords[i]||i,16,colors[i%3]);
  if(tt.id.endsWith('.A')&&!sourceConvention)tt.values.forEach((row,i)=>body+=text(785,166+(i+.5)*270/tt.shape[0]+5,'Σ '+format(row.reduce((s,x)=>s+x,0),3),18,'#a3e4ba'));
  if(sourceConvention){const x=266+Math.min(o.row||0,tt.shape[1]-1)*450/tt.shape[1];body+=`<rect x="${x}" y="166" width="${450/tt.shape[1]}" height="270" fill="none" stroke="#eed266" stroke-width="2"/>`;if([18,19].includes(scene.reference)){const selected={...original,shape:[original.shape[1],1],values:original.values[o.row||0].map(v=>[v]),columnVector:true,sourceRow:o.row||0};body+=line(733,300,780,300,'#e8d57b')+mx(selected,803,166,102,270,'Selected query');}}
  body+=text(500,471,tt.id.endsWith('.masked')?'−∞ blocks a position before softmax':tt.id.endsWith('.scores')?'Raw scores can be negative; they are not probabilities.':'Bright cells carry larger mixing weights.',16,'#9bb8bb');
 }else if(kind==='bars'){
  const r=Math.min(o.row||0,A.shape[0]-1),before=get(att+'.scaled'),values=before.values[r],after=A.values[r];subtitle='Compare the scores with their normalized weights';
  body=bars(values,words,75,197,330,195,'Scaled scores')+line(446,299,550,299)+text(499,274,'softmax',16,'#eace78')+bars(after,words,600,197,330,195,'Attention weights',true)+text(764,458,'Σ weights = '+format(after.reduce((s,x)=>s+x,0),6),18,'#a7e7b8');
 }else if(kind==='mix'){
  subtitle='Each attention weight scales an entire value vector';const r=Math.min(o.row||0,A.shape[0]-1),cw=800/A.shape[1];
  body=A.values[r].map((v,i)=>group(token(100+cw*(i+.5),147,keyWords[i]||i,i,120)+text(100+cw*(i+.5),210,format(v,4)+' ×',26,colors[i%3])+mx({...V,shape:[1,V.shape[1]],values:[V.values[i]],rowMap:[i]},100+cw*i+20,248,cw-40,65,'value '+i)+line(100+cw*(i+.5),344,500,417,colors[i%3]),i*.16)).join('')+mx({...Z,shape:[1,Z.shape[1]],values:[Z.values[r]],rowMap:[r]},350,430,300,46,'weighted sum');
 }else if(kind==='heads'){
  subtitle='Independent heads → concatenate → learned output projection';const base=att.replace(/\.h\d+$/,''),concat=get(base+'.concat'),output=get(base+'.output');
  [0,1].forEach((h,i)=>{const tt=get(base+'.h'+h+'.Z');body+=group(mx(tt,135+i*470,140,270,140,'head '+(h+1)+' output')+line(270+i*470,310,500,360,colors[i]),i*.25)});
  body+=mx(concat,190,386,290,73,'Concatenate')+line(510,424,570,424)+mx(output,600,386,260,73,'× W_O');
 }else if(['graph','recurrent'].includes(kind)){
  const story=[39,40].includes(scene.reference),ww=t&&!story?m.sourceTokens:['The','animal','didn’t','cross','street','it','tired'];subtitle=t&&!story?'Connections use the current model’s actual attention weights':'Context must travel between positions · illustrative sentence';
  const serial=kind==='recurrent',pts=ww.map((w,i)=>serial?[85+i*830/Math.max(1,ww.length-1),250]:[500+290*Math.cos(-Math.PI/2+i*2*Math.PI/ww.length),300+125*Math.sin(-Math.PI/2+i*2*Math.PI/ww.length)]);
  pts.forEach(([x,y],i)=>pts.forEach(([xx,yy],j)=>{if(i===j||serial&&j!==i+1)return;const weight=t&&!story?A.values[i%A.shape[0]][j%A.shape[1]]:i===5&&j===1?.9:.16;body+=`<path class="graph-link" d="M${x} ${y}Q500 ${serial?180:290} ${xx} ${yy}" fill="none" stroke="${colors[i%3]}" stroke-width="${1+weight*4}" opacity="${.13+weight*.65}"/>`;}));
  body+=pts.map(([x,y],i)=>group(`<circle cx="${x}" cy="${y}" r="${serial?34:37}" fill="#071313" stroke="${colors[i%colors.length]}" stroke-width="1.6"/>${text(x,y+6,ww[i],ww[i].length>6?15:21,colors[i%colors.length])}`,i*.07)).join('');
  if(serial)body+=text(500,429,'hₜ = f(hₜ₋₁, xₜ) · later states depend on earlier states',21,'#dec978');
 }else if(['geometry','analogy'].includes(kind)){
  subtitle='Two coordinates shown · the full embedding has four dimensions';const E=get('source.lookup'),coords=E.values.map(v=>[500+v[0]*260,315-v[1]*190]);
  for(let i=-4;i<=4;i++)body+=`<path d="M${500+i*70} 145V450M160 ${300+i*35}H850" stroke="#183339" stroke-width=".8"/>`;
  body+=line(180,315,854,315)+line(500,458,500,136);
  coords.forEach(([x,y],i)=>body+=group(line(500,315,x,y,colors[i%3])+text(x,y-14,m.sourceTokens[i],22,colors[i%3]),i*.17));
  if(kind==='analogy'){body+=line(...coords[0],...coords[1],'#ebce77')+text(500,466,'Toy vectors are untrained: semantic analogies are not guaranteed.',16,'#9bb8bb');}
 }else if(kind==='contexts'){
  subtitle='Shared token identity can acquire different contextual representations';
  const samples=scene.reference===61?['deposit money at the bank','sit beside the river bank']:['American shrew mole','one mole of carbon dioxide','a mole on the skin'];
  samples.forEach((s,i)=>{const x=175+i*650/Math.max(1,samples.length-1);body+=group(text(x,164,s,17,colors[i%3])+line(x,192,x,245,colors[i%3])+token(x,280,scene.reference===61?'bank':'mole',i)+line(x,311,500,400,colors[i%3]),i*.2)});
  body+=text(500,436,'Context is learned through attention; these are conceptual examples.',18,'#b4c7ce');
 }else if(kind==='position'){
  subtitle='Token embedding × √d_model + sinusoidal position = stack input';const scope=t?.scope==='target'?'target':'source';
  body=mx(get(scope+'.scaled'),65,182,240,193,'Scaled embedding')+text(339,290,'+',32,'#e2c879')+mx(get(scope+'.position'),379,182,240,193,'Position')+text(656,290,'=',32,'#e2c879')+mx(get(scope+'.input'),694,182,240,193,'Input X');
  body+=`<path d="${Array.from({length:100},(_,i)=>`${i?'L':'M'}${100+i*8} ${436-Math.sin(i*.17)*20}`).join(' ')}" fill="none" stroke="#72d6cf" stroke-width="2"/>`;
 }else if(kind==='norm'){
  subtitle='Preserve a residual path, then normalize each token’s feature row';
  body='<g class="normalization-microscope">'+normMarkup(normState(m,id,o.row||0,normPhase(id)))+'</g>';
 }else if(kind==='ff'){
  subtitle='The same position-wise network: 4 → 8 → 4 features';const prefix=id.match(/^(enc|dec)\.\d+\.ff/)?.[0]||'enc.0.ff';
  body='<g class="neural-microscope">'+neuronMarkup(neuronState(m,id,o.row||0,(o.col||0)%8,neuronPhase(id,o.phase||0)))+'</g>';
 }else if(['output','prefix'].includes(kind)){
  subtitle='A decoder representation becomes a distribution over the vocabulary';const r=Math.min(o.row||0,m.probabilities.length-1),p=m.probabilities[r],items=p.map((v,i)=>({v,i})).sort((a,b)=>b.v-a.v).slice(0,6);
  body=m.targetTokens.slice(0,r+1).slice(-5).map((w,i)=>token(180,155+i*60,w,i,140)).join('')+line(278,295,376,295)+bars(items.map(x=>x.v),items.map(x=>VOCAB[x.i]),420,190,490,220,'Next-token probabilities',true)+text(660,462,'Top 6 shown; all '+VOCAB.length+' sum to 1.',15,'#96b8bf');
 }else if(kind==='parameters'){
  subtitle='Dimensions determine parameter counts · the live model is deliberately small';body=mx(get('@enc.0.self.h0.WQ'),95,185,220,220,'W_Q · 4 × 2')+mx(get('@enc.0.self.h0.WK'),390,185,220,220,'W_K · 4 × 2')+mx(get('@enc.0.self.h0.WV'),685,185,220,220,'W_V · 4 × 2')+text(500,457,'Three projections × 4 × 2 × 2 heads = 48 learned weights',20,'#d5cc91');
 }else if(kind==='pipeline'){
  subtitle='Follow the same values through the complete attention operation';const tt=[X,Q,get(att+'.scores'),A,Z],titles=['Input X','Project Q','QKᵀ / √dₖ','Softmax','Mix with V'];
  tt.forEach((v,i)=>{const x=35+i*195;body+=group(mx(v,x,241,155,150,titles[i]),i*.12);if(i<4)body+=line(x+166,310,x+185,310);});
 }else if(kind==='gradient'){
  subtitle='Loss sensitivity travels backwards through each local derivative';const terms=calc?.terms.slice(0,3)||[];body=terms.map((v,i)=>{const x=190+i*300;return group(token(x,179,'incoming ∂L',i,180)+text(x,231,format(v.upstream??v.value,5),23,colors[i])+line(x,253,x,292,colors[i])+token(x,328,'local derivative',i,210)+text(x,379,format(v.derivative??1,5),23,colors[i])+line(x,400,500,459,colors[i]),i*.18)}).join('');
 }
 const originalLabel=id.startsWith('learn.')||id.startsWith('backward.')?'Backward pass · chain rule':id.includes('.norm.')||id.includes('residual')?'Add & Norm':id.includes('.ff.')?'Feed Forward':id.includes('.self.')?id.startsWith('dec.')?'Masked Multi-Head Attention':'Multi-Head Attention':id.includes('.cross.')?'Multi-Head Attention · cross':id.startsWith('output.')?'Linear → Softmax':id.includes('position')||id.endsWith('.input')?'Embedding + Positional Encoding':'Embedding';const scope=id.startsWith('enc.')?'ENCODER '+(Number(id.split('.')[1])+1):id.startsWith('dec.')?'DECODER '+(Number(id.split('.')[1])+1):'INPUT / OUTPUT';
 const header=scene.reference?`DRAWING ${String(scene.reference).padStart(2,'0')} / 70 · LIVE ADAPTATION`:'INSIDE THE ARCHITECTURE · LIVE COMPUTATION';
 return `<svg xmlns="http://www.w3.org/2000/svg" class="simulation-svg" viewBox="0 0 1000 680" role="img" aria-label="${esc(scene.title)}"><defs><marker id="flow-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10" fill="#a4d8cd"/></marker></defs><g class="focus-block"><rect x="310" y="6" width="380" height="33" rx="7" fill="#162621" stroke="#8cb889"/>${text(500,27,t?originalLabel:header,15,'#c4e6b1')}${text(295,26,scope,10,'#7ca7a0','end')}</g>${text(500,67,scene.reference?sourceInfo[scene.reference-1][0]:t?.label||scene.title,27,'#e2ede6')}${text(500,99,subtitle,15,'#8fa7af')}${body}${t&&!custom?.hideArithmetic&&!scene.visualOnly?arithmetic(m,t,o):custom?.hideArithmetic||scene.visualOnly?'':text(500,587,'Scroll to follow the information.',20,'#8dd8c3')}</svg>`;
}
