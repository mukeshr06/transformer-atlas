// A complete, small post-norm encoder–decoder, evaluated locally with reverse-mode AD.
// No pretrained model, fabricated attention maps, or baked output probabilities.
export const VOCAB=['<pad>','<bos>','<eos>','the','cat','sleeps','le','chat','dort','a','dog','runs','un','chien','court'];
export const EXAMPLES=[
  {name:'The cat sleeps',source:['the','cat','sleeps'],target:['le','chat','dort','<eos>']},
  {name:'A dog runs',source:['a','dog','runs'],target:['un','chien','court','<eos>']},
  {name:'The cat runs',source:['the','cat','runs'],target:['le','chat','court','<eos>']}
];
export const D=4,HEADS=2,DK=2,FF=8,EPS=1e-5;
export const value=n=>typeof n==='number'?n:n.v;
export const numbers=m=>m.map(row=>row.map(value));
export const zeros=(r,c,f=()=>0)=>Array.from({length:r},(_,i)=>Array.from({length:c},(_,j)=>f(i,j)));
export const clone=m=>m.map(r=>[...r]);
export const shape=m=>[m.length,m[0].length];
export class Tape{
  constructor(){this.nodes=[];}
  node(v,p=[],d=[],op='constant'){const n={v,g:0,p,d,op,index:this.nodes.length};this.nodes.push(n);return n;}
  add(a,b){return this.node(a.v+b.v,[a,b],[1,1],'addition');}
  sub(a,b){return this.node(a.v-b.v,[a,b],[1,-1],'subtraction');}
  mul(a,b){return this.node(a.v*b.v,[a,b],[b.v,a.v],'multiplication');}
  div(a,b){return this.node(a.v/b.v,[a,b],[1/b.v,-a.v/(b.v*b.v)],'division');}
  exp(a){const v=Math.exp(a.v);return this.node(v,[a],[v],'exponential');}
  log(a){return this.node(Math.log(a.v),[a],[1/a.v],'logarithm');}
  relu(a){return this.node(Math.max(0,a.v),[a],[a.v>0?1:0],'ReLU');}
  power(a,k){return this.node(a.v**k,[a],[k*a.v**(k-1)],'power');}
  sum(ns){return ns.reduce((a,b)=>this.add(a,b),this.node(0));}
  backward(loss){for(const n of this.nodes)n.g=0;loss.g=1;for(let i=this.nodes.length-1;i>=0;i--){const n=this.nodes[i];if(!n.g)continue;for(let j=0;j<n.p.length;j++)n.p[j].g+=n.g*n.d[j];}}
}
function rng(seed){let x=seed>>>0;return()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return(x>>>0)/4294967296;};}
function groupFor(scope){if(scope.startsWith('enc.'))return `Encoder ${+scope.split('.')[1]+1}`;if(scope.startsWith('dec.'))return `Decoder ${+scope.split('.')[1]+1}`;return scope==='source'?'Source input':scope==='target'?'Target input':scope==='training'?'Learning':'Prediction';}
export class Transformer{
  constructor(){this.vocabulary=VOCAB;this.config={layers:2,example:0,padding:false,causal:true,scaled:true,attentionTemperature:1,temperature:1,learningRate:.04};this.parameters=new Map();this.history=[];this.steps=0;this.generation=null;this.seed=17;this.reset();}
  reset(seed=17){
    this.seed=seed;this.parameters.clear();const random=rng(seed);
    const add=(key,r,c,kind='weight')=>{const data=zeros(r,c,()=>kind==='gamma'?1:kind==='bias'?0:+((random()*2-1)*.65).toFixed(3));this.parameters.set(key,{key,values:data,kind,grad:zeros(r,c)});};
    add('source.embedding',VOCAB.length,D);add('target.embedding',VOCAB.length,D);
    this.parameters.get('source.embedding').values[3]=[.2,-.5,.8,.1];this.parameters.get('source.embedding').values[4]=[.9,.3,-.2,.7];this.parameters.get('source.embedding').values[5]=[-.1,.6,.4,-.8];
    for(let l=0;l<2;l++)for(const side of ['enc','dec']){
      const prefix=`${side}.${l}`;
      for(const att of side==='enc'?['self']:['self','cross']){
        for(let h=0;h<HEADS;h++)for(const q of ['Q','K','V'])add(`${prefix}.${att}.h${h}.W${q}`,D,DK);
        add(`${prefix}.${att}.WO`,D,D);
        add(`${prefix}.${att}.norm.gamma`,1,D,'gamma');add(`${prefix}.${att}.norm.beta`,1,D,'bias');
      }
      add(`${prefix}.ff.W1`,D,FF);add(`${prefix}.ff.b1`,1,FF,'bias');add(`${prefix}.ff.W2`,FF,D);add(`${prefix}.ff.b2`,1,D,'bias');add(`${prefix}.ff.norm.gamma`,1,D,'gamma');add(`${prefix}.ff.norm.beta`,1,D,'bias');
    }
    add('vocab.W',D,VOCAB.length);add('vocab.bias',1,VOCAB.length,'bias');
    this.steps=0;this.generation=null;this.lastUpdate=null;this.history=[];this.run();this.history.push(this.loss);
  }
  setText(source,target=''){
    const tokenize=text=>text.normalize('NFKC').toLowerCase().match(/[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*|[^\s\p{L}\p{N}]/gu)||[];
    const src=tokenize(source),dst=tokenize(target.trim()||source);
    if(!src.length||!dst.length)throw Error('Enter a source sentence and, optionally, its expected output.');
    if(src.length>6||dst.length>6)throw Error('Use up to 6 tokens on each side so every calculation remains readable. Punctuation counts as a token.');
    if(src.concat(dst).some(t=>t.length>24))throw Error('Keep each token under 25 characters for readable diagrams.');
    for(const token of [...src,...dst])if(!VOCAB.includes(token))VOCAB.push(token);
    this.customExample={name:'Your text',source:src,target:[...dst,'<eos>']};this.reset(this.seed);
    return this.customExample;
  }
  setParameter(key,row,col,next){if(!Number.isFinite(next)||Math.abs(next)>3)throw Error('Choose a finite value between −3 and 3.');const p=this.parameters.get(key);if(!p?.values[row]||col>=p.values[row].length)throw Error('Unknown parameter coordinate.');p.values[row][col]=next;this.run();}
  run(){
    const tape=this.tape=new Tape();this.tensors=new Map();this.trace=[];this.paramNodes=new Map();
    const ex=this.customExample||EXAMPLES[this.config.example];this.sourceTokens=[...ex.source,...(this.config.padding?['<pad>']:[])];this.targetTokens=this.generation?[...this.generation]:['<bos>',...ex.target.slice(0,-1)];
    this.targets=this.targetTokens.map((_,i)=>ex.target[i]??'<eos>');
    const srcPad=this.sourceTokens.map(t=>t==='<pad>'),tgtPad=this.targetTokens.map(t=>t==='<pad>');
    const source=this.embedding('source',this.sourceTokens);
    let encoded=source;
    for(let l=0;l<this.config.layers;l++){
      const scope=`enc.${l}`;
      const update=this.attention(scope+'.self',encoded,encoded,{padding:srcPad});
      const residual=this.binary(scope+'.self.residual','Residual addition',encoded,update,'add',{scope,concept:'norm'});
      const normalized=this.norm(scope+'.self.norm',residual,scope);
      encoded=this.feedForward(scope+'.ff',normalized,scope);
    }
    this.encoderOutput=encoded;
    let decoded=this.embedding('target',this.targetTokens);
    for(let l=0;l<this.config.layers;l++){
      const scope=`dec.${l}`;
      const update=this.attention(scope+'.self',decoded,decoded,{padding:tgtPad,causal:this.config.causal});
      const residual=this.binary(scope+'.self.residual','Keep the decoder input',decoded,update,'add',{scope,concept:'norm'});
      decoded=this.norm(scope+'.self.norm',residual,scope);
      const cross=this.attention(scope+'.cross',decoded,encoded,{padding:srcPad,cross:true});
      const crossResidual=this.binary(scope+'.cross.residual','Add the source-context update',decoded,cross,'add',{scope,concept:'norm'});
      decoded=this.norm(scope+'.cross.norm',crossResidual,scope);
      decoded=this.feedForward(scope+'.ff',decoded,scope);
    }
    this.decoderOutput=decoded;
    const projected=this.matmul('output.project','Project to the vocabulary',decoded,this.parameter('vocab.W'),{scope:'output',concept:'generation'});
    const logits=this.bias('output.logits','Add the vocabulary bias',projected,this.parameter('vocab.bias'),{scope:'output',concept:'generation'});
    const scaled=this.scale('output.temperature','Set the output temperature',logits,this.config.temperature,{scope:'output',concept:'generation'});
    const probabilities=this.softmax('output',scaled,{scope:'output',concept:'generation'});
    this.output=probabilities;this.probabilities=this.tensors.get(probabilities).values;
    const selected=this.tensors.get(probabilities).nodes.map((row,i)=>[row[VOCAB.indexOf(this.targets[i])]]);
    const targetProbs=this.record('loss.targets','Probabilities of the correct targets',selected,'gather',[probabilities],{scope:'training',concept:'training',targets:this.targets.map(t=>VOCAB.indexOf(t))});
    const losses=this.record('loss.tokens','Negative log-likelihood',selected.map(r=>[tape.mul(tape.node(-1),tape.log(r[0]))]),'neglog',[targetProbs],{scope:'training',concept:'training'});
    const loss=tape.div(tape.sum(this.tensors.get(losses).nodes.flat()),tape.node(selected.length));
    this.record('loss.mean','Mean cross-entropy',[[loss]],'mean',[losses],{scope:'training',concept:'training'});this.loss=loss.v;
    tape.backward(loss);
    for(const [key,nodes] of this.paramNodes)this.parameters.get(key).grad=nodes.map(row=>row.map(n=>n.g));
    for(const tensor of this.tensors.values())tensor.grads=tensor.nodes.map(row=>row.map(n=>n.g));
    // Preserve the actual scalar graph used for differentiation. An incoming contribution is
    // the consumer's gradient multiplied by its local derivative with respect to this value.
    this.lossNode=loss;this.gradientEdges=new Map();
    for(const child of tape.nodes)child.p.forEach((parent,i)=>{if(!this.gradientEdges.has(parent))this.gradientEdges.set(parent,[]);this.gradientEdges.get(parent).push({child,derivative:child.d[i]});});
    const forward=[...this.trace],parameters=[...this.tensors.values()].filter(t=>t.parameterOnly);
    for(const original of [...forward.reverse(),...parameters])this.record('backward.'+original.id,'Differentiate · '+original.label,original.grads.map(row=>row.map(v=>tape.node(v))),'backward',[],{scope:'training',concept:'training',backwardOf:original.id,micro:original.micro});
    this.gradientNorm=Math.sqrt([...this.paramNodes.keys()].reduce((sum,key)=>sum+this.parameters.get(key).grad.flat().reduce((s,x)=>s+x*x,0),0));
    this.clipScale=Math.min(1,5/(this.gradientNorm||1));
    const key='enc.0.self.h0.WQ',parameter=this.parameters.get(key),pId=this.parameter(key);
    const grad=this.record('learn.gradient','Backpropagate to the weights',parameter.grad.map(row=>row.map(v=>tape.node(v))),'gradient',['loss.mean',pId],{scope:'training',concept:'training',parameter:key});
    const clipped=this.scale('learn.clipped','Keep the gradient norm bounded',grad,1/this.clipScale,{scope:'training',concept:'training',norm:this.gradientNorm,clip:this.clipScale});
    const delta=this.record('learn.delta','Multiply by the learning rate',this.tensors.get(clipped).nodes.map(r=>r.map(v=>tape.mul(v,tape.node(this.config.learningRate)))),'multiplyScalar',[clipped],{scope:'training',concept:'training',scalar:this.config.learningRate});
    this.binary('learn.next','Preview the next weight update',pId,delta,'sub',{scope:'training',concept:'training',preview:true});
    this.parameterCount=[...this.paramNodes.keys()].reduce((sum,key)=>sum+this.parameters.get(key).values.flat().length,0);
    return this;
  }
  gradientDetails(id,row,col){const node=this.data(id)[row][col],edges=this.gradientEdges.get(node)||[];return {value:node.g,seed:node===this.lossNode?1:0,terms:edges.map(({child,derivative})=>({upstream:child.g,derivative,value:child.g*derivative,operation:child.op,node:child.index}))};}
  record(id,label,nodes,op,inputs=[],meta={}){const record={id,label,nodes,values:numbers(nodes),op,inputs,scope:meta.scope||'source',concept:meta.concept||'attention',...meta};record.shape=shape(nodes);record.group=groupFor(record.scope);this.tensors.set(id,record);if(!record.parameterOnly)this.trace.push(record);return id;}
  parameter(key){
    if(this.paramNodes.has(key))return '@'+key;
    const parameter=this.parameters.get(key);if(!parameter)throw Error('Unknown parameter '+key);
    const nodes=parameter.values.map(row=>row.map(v=>this.tape.node(v)));this.paramNodes.set(key,nodes);
    return this.record('@'+key,key,nodes,'parameter',[],{parameterOnly:true,parameter:key,scope:key.split('.').slice(0,2).join('.')});
  }
  data(id){const t=this.tensors.get(id);if(!t)throw Error('Missing tensor '+id);return t.nodes;}
  matmul(id,label,left,right,meta={}){const a=this.data(left),b=this.data(right),t=this.tape;if(a[0].length!==b.length)throw Error(`Shape mismatch in ${id}`);const out=a.map(row=>b[0].map((_,j)=>t.sum(row.map((x,k)=>t.mul(x,b[k][j])))));return this.record(id,label,out,'matmul',[left,right],meta);}
  binary(id,label,left,right,op,meta={}){const a=this.data(left),b=this.data(right),t=this.tape;return this.record(id,label,a.map((row,i)=>row.map((v,j)=>t[op](v,b[i][j]))),op,[left,right],meta);}
  bias(id,label,left,right,meta={}){const a=this.data(left),b=this.data(right),t=this.tape;return this.record(id,label,a.map(row=>row.map((v,j)=>t.add(v,b[0][j]))),'bias',[left,right],meta);}
  scale(id,label,input,divisor,meta={}){const t=this.tape;return this.record(id,label,this.data(input).map(r=>r.map(v=>t.div(v,t.node(divisor)))),'scale',[input],{...meta,divisor});}
  embedding(scope,tokens){const t=this.tape,ids=tokens.map(token=>VOCAB.indexOf(token));
    const id=this.record(scope+'.ids','Split text into token IDs',ids.map(i=>[t.node(i)]),'tokens',[],{scope,concept:'tokens',tokens});
    const table=this.parameter(scope+'.embedding'),emb=this.data(table);
    const lookup=this.record(scope+'.lookup','Look up one embedding per token',ids.map(i=>[...emb[i]]),'lookup',[id,table],{scope,concept:'embeddings',tokens,indices:ids});
    const scaled=this.scale(scope+'.scaled','Scale embeddings by √d_model',lookup,1/Math.sqrt(D),{scope,concept:'embeddings'});
    const pe=this.record(scope+'.position','Compute the positional coordinates',tokens.map((_,pos)=>Array.from({length:D},(_,dim)=>t.node(dim%2?Math.cos(pos/10000**(2*Math.floor(dim/2)/D)):Math.sin(pos/10000**(2*Math.floor(dim/2)/D))))),'position',[],{scope,concept:'position',tokens});
    return this.binary(scope+'.input','Combine meaning with position',scaled,pe,'add',{scope,concept:'position',tokens});
  }
  attention(prefix,queryInput,memoryInput,{padding,causal=false,cross=false}){
    const outputs=[],t=this.tape,scope=prefix.split('.').slice(0,2).join('.');
    for(let head=0;head<HEADS;head++){
      const p=prefix+'.h'+head,meta={scope,head,attention:prefix,cross};
      const q=this.matmul(p+'.Q',`Head ${head+1} · queries`,queryInput,this.parameter(p+'.WQ'),{...meta,concept:'qkv'});
      const k=this.matmul(p+'.K',`Head ${head+1} · keys`,memoryInput,this.parameter(p+'.WK'),{...meta,concept:'qkv'});
      const v=this.matmul(p+'.V',`Head ${head+1} · values`,memoryInput,this.parameter(p+'.WV'),{...meta,concept:'qkv'});
      const kn=this.data(k),kt=this.record(p+'.KT','Transpose the keys',kn[0].map((_,j)=>kn.map(r=>r[j])),'transpose',[k],{...meta,concept:'scores'});
      const scores=this.matmul(p+'.scores','Compare every query with every key',q,kt,{...meta,concept:'scores'});
      const scaled=this.scale(p+'.scaled','Scale the dot products',scores,this.config.scaled?Math.sqrt(DK):1,{...meta,concept:'softmax'});
      const temperature=this.scale(p+'.temperature','Adjust attention temperature',scaled,this.config.attentionTemperature,{...meta,concept:'softmax',micro:true});
      const allowed=this.data(temperature).map((r,i)=>r.map((_,j)=>!padding[j]&&(!causal||j<=i)));
      const masked=this.record(p+'.masked',causal?'Block future and padding positions':'Exclude padding positions',this.data(temperature).map((r,i)=>r.map((n,j)=>allowed[i][j]?n:t.node(-Infinity))),'mask',[temperature],{...meta,concept:'mask',allowed,causal,padding});
      const weights=this.softmax(p,masked,{...meta,concept:'softmax'});
      outputs.push(this.matmul(p+'.Z',`Head ${head+1} · mix the values`,weights,v,{...meta,concept:'values'}));
    }
    const heads=outputs.map(id=>this.data(id)),concat=this.record(prefix+'.concat','Concatenate the independent heads',heads[0].map((_,i)=>heads.flatMap(h=>h[i])),'concat',outputs,{scope,concept:'heads',attention:prefix,cross});
    return this.matmul(prefix+'.output','Project back to four features',concat,this.parameter(prefix+'.WO'),{scope,concept:'heads',attention:prefix,cross});
  }
  softmax(prefix,input,meta){const t=this.tape,a=this.data(input);
    const maximum=this.record(prefix+'.max','Find the largest value in each row',a.map(r=>[t.node(Math.max(...r.map(n=>n.v)))]),'rowmax',[input],{...meta,micro:true});
    const m=this.data(maximum),shift=this.record(prefix+'.shift','Subtract the maximum for stability',a.map((r,i)=>r.map(v=>t.sub(v,m[i][0]))),'rowSubtract',[input,maximum],{...meta,micro:true});
    const exps=this.record(prefix+'.exp','Exponentiate the shifted scores',this.data(shift).map(r=>r.map(v=>t.exp(v))),'exp',[shift],meta);
    const denominator=this.record(prefix+'.sum','Sum each row’s exponentials',this.data(exps).map(r=>[t.sum(r)]),'rowSum',[exps],meta);
    return this.record(prefix+'.A','Normalize into a probability distribution',this.data(exps).map((r,i)=>r.map(v=>t.div(v,this.data(denominator)[i][0]))),'rowDivide',[exps,denominator],meta);
  }
  norm(prefix,input,scope){const t=this.tape,a=this.data(input),meta={scope,concept:'norm',normalization:prefix};
    const means=this.record(prefix+'.mean','Compute a mean for each token',a.map(r=>[t.div(t.sum(r),t.node(D))]),'rowMean',[input],meta);
    const centered=this.record(prefix+'.center','Subtract that token’s mean',a.map((r,i)=>r.map(v=>t.sub(v,this.data(means)[i][0]))),'rowSubtract',[input,means],{...meta,micro:true});
    const squared=this.record(prefix+'.squared','Square each centered feature',this.data(centered).map(r=>r.map(v=>t.mul(v,v))),'square',[centered],{...meta,micro:true});
    const variance=this.record(prefix+'.variance','Average the squared differences',this.data(squared).map(r=>[t.div(t.sum(r),t.node(D))]),'rowMean',[squared],meta);
    const std=this.record(prefix+'.std','Take √(variance + ε)',this.data(variance).map(r=>[t.power(t.add(r[0],t.node(EPS)),.5)]),'std',[variance],{...meta,epsilon:EPS,micro:true});
    const normalized=this.record(prefix+'.unit','Standardize each token’s features',this.data(centered).map((r,i)=>r.map(v=>t.div(v,this.data(std)[i][0]))),'rowDivide',[centered,std],meta);
    const gamma=this.parameter(prefix+'.gamma'),beta=this.parameter(prefix+'.beta');
    return this.record(prefix+'.output','Apply learned scale and shift',this.data(normalized).map(r=>r.map((v,j)=>t.add(t.mul(v,this.data(gamma)[0][j]),this.data(beta)[0][j]))),'affine',[normalized,gamma,beta],meta);
  }
  feedForward(prefix,input,scope){const meta={scope,concept:'ffn'},t=this.tape;
    const expansion=this.matmul(prefix+'.expand','Expand four features into eight',input,this.parameter(prefix+'.W1'),meta);
    const biased=this.bias(prefix+'.biased','Add a learned hidden bias',expansion,this.parameter(prefix+'.b1'),meta);
    const active=this.record(prefix+'.relu','ReLU: remove negative activations',this.data(biased).map(r=>r.map(v=>t.relu(v))),'relu',[biased],meta);
    const contracted=this.matmul(prefix+'.contract','Project eight features back to four',active,this.parameter(prefix+'.W2'),meta);
    const output=this.bias(prefix+'.bias','Add the output bias',contracted,this.parameter(prefix+'.b2'),meta);
    const residual=this.binary(prefix+'.residual','Add the feed-forward update',input,output,'add',{scope,concept:'norm'});
    return this.norm(prefix+'.norm',residual,scope);
  }
  trainStep(){
    if(this.generation){this.generation=null;this.run();}
    const lr=this.config.learningRate,scale=this.clipScale,before=this.loss;const changes={};
    for(const key of this.paramNodes.keys()){const p=this.parameters.get(key);changes[key]={before:clone(p.values),gradient:clone(p.grad)};p.values=p.values.map((row,i)=>row.map((v,j)=>v-lr*scale*p.grad[i][j]));changes[key].after=clone(p.values);}
    this.steps++;this.lastUpdate={before,lr,scale,changes};this.run();this.lastUpdate.after=this.loss;this.history.push(this.loss);if(this.history.length>301)this.history.shift();return {before,after:this.loss};
  }
  startGeneration(){this.generation=['<bos>'];this.run();}
  generateNext(){if(!this.generation)this.startGeneration();if(this.generation.length>=9||this.generation.at(-1)==='<eos>')return null;const row=this.probabilities.at(-1),id=row.indexOf(Math.max(...row)),token=VOCAB[id];this.generation.push(token);this.run();return token;}
}
