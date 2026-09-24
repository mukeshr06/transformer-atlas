import {neuronState,neuronNarrative,neuronPhase} from './microscope.js?v=math-type-1';
import {calculate,format} from './course.js?v=math-type-1';
import {clamp} from './space.js?v=math-type-1';
const palette=[[.32,.76,1],[.36,.96,.73],[1,.78,.32],[.77,.54,1],[1,.48,.56]];
export function canSpatial(t,scene){return !!t&&!scene.visualOnly&&!t.parameterOnly&&t.op!=='tokens';}
export function liveSpatialState(m,scene,progress,row=0,col=0){
 const t=m.tensors.get(scene.tensor),p=clamp(progress),r=Math.min(row,t.shape[0]-1),c=Math.min(col,t.shape[1]-1),input=m.tensors.get(t.inputs[0]),vectors=[],points=[],segments=[],labels=[];
 const label=(text,pos,color=palette[0])=>labels.push({text,pos,color});
 const arrow=(name,from,to,color=palette[0],ghost=false)=>vectors.push({name,from,to,color,ghost});
 if(/\.ff\.(expand|biased|relu|contract|bias)$/.test(t.id))return neuralSpace(m,t,p,r,c);
 const calculated=t.values[r],current=calculated.map((v,j)=>{const calc=calculate(m,t.id,r,j);if(!calc.additive)return v;const count=Math.min(calc.terms.length,Math.floor(p*calc.terms.length)+1);return calc.terms.slice(0,count).reduce((a,b)=>a+b.value,0);});
 const calc=calculate(m,t.id,r,c),k=Math.min(calc.terms.length-1,Math.floor(p*calc.terms.length)),term=calc.terms[k],why=scene.why||'Follow the numerical transformation from input to output.';
 let title=t.label,equation='',stage='',note='',detail;
 const probability=t.id.endsWith('.A'),scores=/\.(scores|masked|scaled|temperature|max|shift|exp|sum)$/.test(t.id)&&!t.normalization&&t.attention,wide=t.shape[1]>4;
 if(probability||scores||wide){
  const bounds=calculated.map((_,j)=>{const q=calculate(m,t.id,r,j);return q.additive?q.terms.reduce((a,b)=>a+(Number.isFinite(b.value)?Math.abs(b.value):0),0):0;});const max=Math.max(.1,...bounds,...t.values.flat().filter(Number.isFinite).map(Math.abs)),scale=1.9/max;
  calculated.forEach((v,j)=>{const x=(j-(calculated.length-1)/2)*Math.min(.65,5/Math.max(1,calculated.length-1)),base=[x,0,0],height=Number.isFinite(v)?v*scale:0,shown=Number.isFinite(current[j])?current[j]*scale:0;arrow('final value '+j,base,[x,height,0],palette[j%5],true);arrow('current value '+j,base,[x,shown,0],palette[j%5]);points.push({name:'value '+j,pos:[x,shown,0],color:palette[j%5]});if(calculated.length<=9||j===c)label(`${j}: ${format(v,3)}`,[x,height+.18,0],palette[j%5]);});
  t.values.forEach((values,ri)=>{if(ri===r)return;values.forEach((v,j)=>{const x=(j-(values.length-1)/2)*Math.min(.65,5/Math.max(1,values.length-1)),z=(ri-r)*.85;arrow('row '+ri+' feature '+j,[x,0,z],[x,Number.isFinite(v)?v*scale:0,z],[.24,.45,.50]);});label('row '+ri,[-3,-.25,(ri-r)*.85],[.45,.65,.7]);});
  label('SELECTED ROW '+r,[-3,-.25,0],palette[2]);
  note='All token rows occupy separate depth lanes; the selected row shows the active calculation. Each vertical height represents a scalar, not a semantic embedding. A shared visual scale fits all values; negative values extend below zero. Masked entries have no finite height.';stage=`Row ${r} · coordinate ${c} · final value ${format(calculated[c],5)}`;
 }else{
  const before=t.op==='lookup'?m.tensors.get(t.inputs[1]).values[t.indices[r]]:input?.values[Math.min(r,input.shape[0]-1)]||calculated.map(()=>0),finite=arr=>arr.slice(0,3).map(v=>Number.isFinite(v)?v:0),a=finite(before),b=finite(calculated),u=finite(current);while(a.length<3)a.push(0);while(b.length<3)b.push(0);while(u.length<3)u.push(0);
  const scale=1.35/Math.max(1,...a.map(Math.abs),...b.map(Math.abs),...calculated.slice(0,3).map((_,j)=>calculate(m,t.id,r,j).terms.reduce((sum,q)=>sum+(Number.isFinite(q.value)?Math.abs(q.value):0),0))),left=[-1.6,0,0],right=[1.6,0,0],at=(v,o)=>v.map((x,i)=>o[i]+x*scale);
  arrow('input vector',left,at(a,left),palette[0]);arrow('final output',right,at(b,right),palette[2],true);arrow('partial sum',right,at(u,right),palette[1]);
  for(const origin of [left,right])for(let axis=0;axis<3;axis++){const end=[...origin];end[axis]+=1;segments.push({a:origin,b:end,color:palette[axis]});}
  let corner=[...right];u.forEach((v,axis)=>{const next=[...corner];next[axis]+=v*scale;segments.push({a:corner,b:next,color:palette[axis],component:true});corner=next;});points.push({name:'current output',pos:at(u,right),color:palette[1]});label('INPUT',[-1.6,-.4,0]);label('OUTPUT',[1.6,-.4,0],palette[1]);label('current endpoint',at(u,right).map((x,i)=>x+(i===1?.2:0)),palette[1]);
  if(t.op==='matmul'){let sum=[...right];const count=calc.terms.length;for(let termIndex=0;termIndex<count;termIndex++){const delta=[0,1,2].map(axis=>axis<t.shape[1]?calculate(m,t.id,r,axis).terms[termIndex].value*scale:0),end=sum.map((v,i)=>v+delta[i]);if(termIndex<=k){arrow('contribution '+termIndex,sum,end,palette[termIndex%5]);points.push({name:'partial sum',pos:end,color:palette[termIndex%5],radius:.065});}sum=end;}}
  note=`Only the first ${Math.min(3,t.shape[1])} coordinates are drawn. The calculation uses all ${t.shape[1]} output features and every required input. Separate origins represent input and output spaces; geometry uses a shared scale.`;
  stage=`Input [${before.slice(0,3).map(v=>format(v,3)).join(', ')}] → output [${calculated.slice(0,3).map(v=>format(v,3)).join(', ')}]`;
 }
 equation=calc.additive?`Build output from ${calc.terms.length} contributions`:'Apply the rule to obtain the exact output';
 detail={title:calc.additive?`Contribution ${k+1} of ${calc.terms.length}`:'Apply the operation to the selected coordinate',how:calc.additive?`${term.label} = ${format(term.value,5)}. The highlighted coordinate ends at ${format(calc.result,5)} after all contributions are added.`:`${calc.formula} = ${format(calc.result,5)}. The output is the exact computed value, not an interpolated model state.`,why};
 return {title,equation,stage,note,detail,vectors,points,segments,labels,progress:p};
}

function neuralSpace(m,t,p,r,c){
 const s=neuronState(m,t.id,r,c%8,neuronPhase(t.id,p)),positions=[[],[],[]],points=[],segments=[],labels=[],vectors=[],color=[[.3,.74,1],[.45,.95,.72],[1,.76,.3]],middle=m.tensors.get(s.prefix+(t.id.endsWith('.expand')?'.expand':t.id.endsWith('.biased')?'.biased':'.relu')).values[r],outputReady=/\.(contract|bias)$/.test(t.id),values=[s.x,middle,outputReady?s.output:s.output.map(()=>0)];
 const outputLayer=/\.(contract|bias)$/.test(t.id),j=outputLayer?c:c%8,calc=calculate(m,t.id,r,c),step=Math.min(calc.terms.length-1,Math.floor(p*calc.terms.length));
 for(let l=0;l<3;l++)for(let i=0;i<values[l].length;i++){
  const pos=[(l-1)*3.5,(values[l].length-1)*.32-i*.64,l===1?(i%2?-.32:.32):0];positions[l].push(pos);points.push({name:'neuron',pos,color:color[l],radius:.10+Math.min(1,Math.abs(values[l][i]))*.055});labels.push({text:l===2&&!outputReady?`y${i} · next layer`:`${l===0?'x':l===1?'h':'y'}${i} = ${format(values[l][i],2)}`,pos:[pos[0]+(l===0?-.43:l===2?.43:.48),pos[1]+.18,pos[2]],color:color[l]});
 }
 for(let l=0;l<2;l++)for(let a=0;a<positions[l].length;a++)for(let b=0;b<positions[l+1].length;b++){
  const active=t.op==='matmul'&&(outputLayer?l===1&&b===j&&a===step:l===0&&b===j&&a===Math.min(3,step));segments.push({a:positions[l][a],b:positions[l+1][b],color:active?color[l]:[.23,.39,.40],component:active});if(active){const u=p===1?1:(p*calc.terms.length)%1;points.push({name:'contribution',pos:positions[l][a].map((v,i)=>v+(positions[l+1][b][i]-v)*u),color:[1,.9,.6],radius:.13});for(let trail=1;trail<=3;trail++){const v=Math.max(0,u-trail*.085);points.push({name:'signal trail',pos:positions[l][a].map((x,i)=>x+(positions[l+1][b][i]-x)*v),color:color[l],radius:.07-trail*.012});}labels.push({text:calc.terms[step]?.label+' = '+format(calc.terms[step]?.value,4),pos:positions[l][a].map((v,i)=>(v+positions[l+1][b][i])/2+(i===1?.3:0)),color:[1,.9,.6]});}
 }
 if(t.op!=='matmul'){const layer=outputLayer?2:1,end=positions[layer][j],start=[end[0]-.9,end[1],end[2]+.65],term=calc.terms[step],u=p===1?1:(p*Math.max(1,calc.terms.length))%1;segments.push({a:start,b:end,color:color[layer],component:true});points.push({name:'local input',pos:start,color:[.75,.5,1],radius:.10},{name:'local operation',pos:start.map((v,i)=>v+(end[i]-v)*u),color:[1,.9,.6],radius:.13});labels.push({text:term.label+' → '+format(term.value,4),pos:[start[0]-.3,start[1]+.45,start[2]],color:[1,.9,.6]});}
 for(let l=0;l<3;l++)labels.push({text:['01 · INPUT FEATURES','02 · HIDDEN NEURONS','03 · OUTPUT FEATURES'][l],pos:[(l-1)*3.5,2.95,0],color:color[l]});
 const detail=outputLayer?{title:'Combine hidden activations into an output feature',how:`Output feature ${c}: ${calc.formula} = ${format(calc.result,5)}. Every highlighted connection carries one contribution.`,why:'Project the eight hidden responses back to four features so the residual addition can preserve the original width.'}:neuronNarrative(s);
 return {title:t.label,equation:'Four input features → eight hidden neurons → four output features',stage:`Token ${r} · ${t.scope.startsWith('dec')?m.targetTokens[r]:m.sourceTokens[r]} · selected coordinate ${c} · ${format(calc.result,5)}`,note:'The positions arrange neurons for inspection; they are not embedding coordinates. Node sizes encode activation magnitude. Edges represent actual learned connections.',detail,vectors,points,segments,labels,progress:p};
}
