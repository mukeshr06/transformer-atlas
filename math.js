import {format} from './course.js?v=math-type-1';
export const escapeMath=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const el=(tag,s)=>`<${tag}>${s}</${tag}>`;
export const mi=s=>el('mi',escapeMath(s)),mo=s=>el('mo',escapeMath(s)),mn=s=>s===-Infinity?mo('−')+mo('∞'):s===Infinity?mo('∞'):el('mn',escapeMath(typeof s==='number'?(Number.isInteger(s)?String(s):Math.abs(s)>0&&Math.abs(s)<.0001?String(s):format(s,4)):s).replace(/-/g,'−')),mt=s=>el('mtext',escapeMath(s)),row=s=>el('mrow',s),frac=(a,b)=>el('mfrac',row(a)+row(b)),sup=(a,b)=>el('msup',row(a)+row(b)),sub=(a,b)=>el('msub',row(a)+row(b)),sqrt=a=>el('msqrt',a),paren=s=>mo('(')+s+mo(')'),math=s=>`<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">${row(s)}</math>`;
const num=x=>x<0&&Number.isFinite(x)?paren(mn(x)):mn(x),idx=(a,b)=>sub(mi(a),expression(String(b)));
const sum=(n,expr)=>el('munderover',mo('∑')+row(mi('k')+mo('=')+mn(0))+row(mn(n-1)))+expr;
export function derivation(m,calc){
 if(!calc)return'';const t=calc.tensor,a=m.tensors.get(t.inputs[0]),b=m.tensors.get(t.inputs[1]),r=calc.row,c=calc.col,A=idx('a',`${r},${c}`),B=idx('b',`${r},${c}`),Y=idx('y',`${r},${c}`),v=(tt,rr=r,cc=c)=>tt?.values[rr]?.[cc],eq=mo('='),plus=mo('+'),times=mo('×');let symbolic='',numeric='',legend='a: input coordinate · y: output coordinate.';
 const term=(body,i)=>`<mrow class="term" data-term="${i}">${body}</mrow>`;
 if(t.op==='matmul'){const n=a.shape[1],role=t.id.match(/\.([QKV])$/)?.[1];symbolic=role?idx(role,`${r},${c}`)+eq+sum(n,idx('X',`${r},k`)+sub(sub(mi('W'),mi(role)),expression(`k,${c}`))):Y+eq+sum(n,idx('A',`${r},k`)+idx('B',`k,${c}`));numeric=calc.terms.map((q,i)=>term(num(q.a.value)+times+num(q.b.value),i)).join(plus);legend=`${role?'X':'A'}: ${a.label}. ${role?'W'+role:'B'}: ${b.label}. Multiply a row by a column; sum ${n} products. Indices start at zero.`;}
 else if(['add','sub','bias'].includes(t.op)){symbolic=Y+eq+A+mo(t.op==='sub'?'−':'+')+(t.op==='bias'?idx('b',c):B);numeric=num(v(a))+mo(t.op==='sub'?'−':'+')+num(v(b,t.op==='bias'?0:r));legend=t.op==='bias'?'b is a learned bias, shared across token positions.':'Matching coordinates are combined; the tensor shape is unchanged.';}
 else if(t.op==='scale'){symbolic=Y+eq+frac(A,mi('s'));numeric=frac(num(v(a)),num(t.divisor));legend='s is the divisor for this operation. Dividing embeddings by 1/√d is equivalent to multiplying by √d.';}
 else if(t.op==='multiplyScalar'){symbolic=Y+eq+mi('η')+A;numeric=num(t.scalar)+times+num(v(a));legend='η is the learning rate; a is the gradient after clipping.';}
 else if(['rowSum','rowMean','mean'].includes(t.op)){const n=calc.terms.length,expression=sum(n,idx('a','k')),values=calc.terms.map((q,i)=>term(num(q.a.value),i)).join(plus);symbolic=Y+eq+(t.op==='rowSum'?expression:frac(expression,mn(n)));numeric=t.op==='rowSum'?values:frac(values,mn(n));legend=t.op==='rowSum'?'Sum all entries in this row.':'Average the contributing values; the divisor is the number of terms.';}
 else if(['rowDivide','rowSubtract'].includes(t.op)){symbolic=Y+eq+(t.op==='rowDivide'?frac(A,idx('s',r)):A+mo('−')+idx('s',r));numeric=t.op==='rowDivide'?frac(num(v(a)),num(v(b,r,0))):num(v(a))+mo('−')+num(v(b,r,0));legend=t.normalization?'s is this token’s normalization statistic.':'s is a row statistic: the maximum before exponentiation, or the exponential sum before normalization.';}
 else if(t.op==='affine'){symbolic=Y+eq+idx('γ',c)+A+plus+idx('β',c);numeric=num(v(b,0))+times+num(v(a))+plus+num(m.tensors.get(t.inputs[2]).values[0][c]);legend='γ and β are learned scale and offset for each feature.';}
 else if(t.op==='exp'){symbolic=Y+eq+sup(mi('e'),A);numeric=sup(mi('e'),num(v(a)));legend='e is Euler’s number. Exponentiating −∞ gives zero, removing a masked key.';}
 else if(t.op==='square'){symbolic=Y+eq+sup(A,mn(2));numeric=sup(paren(num(v(a))),mn(2));legend='Squaring makes positive and negative deviations contribute to the spread.';}
 else if(t.op==='std'){symbolic=Y+eq+sqrt(A+plus+mi('ε'));numeric=sqrt(num(v(a))+plus+mn(0.00001));legend='a is the variance. ε = 0.00001 protects the denominator from zero.';}
 else if(t.op==='relu'){symbolic=Y+eq+mi('max')+paren(mn(0)+mo(',')+A);numeric=mi('max')+paren(mn(0)+mo(',')+num(v(a)));legend='A positive input passes unchanged; a negative input becomes zero.';}
 else if(t.op==='neglog'){symbolic=idx('ℓ',r)+eq+mo('−')+mi('ln')+paren(idx('p','target'));numeric=mo('−')+mi('ln')+paren(num(v(a)));legend='Natural logarithm gives loss in nats. Lower probability on the correct target creates a larger loss.';}
 else if(t.op==='rowmax'){symbolic=idx('s',r)+eq+mi('max')+paren(idx('a','r,0')+mo(',')+mo('…'));numeric=mi('max')+paren(a.values[r].map(num).join(mo(',')));legend='The maximum is subtracted from every score in the row to stabilize exponentiation.';}
 else if(t.op==='position'){const i=Math.floor(c/2),fn=c%2?'cos':'sin';symbolic=idx('PE',`p,${c}`)+eq+mi(fn)+paren(frac(mi('p'),sup(mn(10000),frac(mn(2*i),mi('d')))));numeric=mi(fn)+paren(frac(mn(r),sup(mn(10000),frac(mn(2*i),mn(4)))));legend='p is token position, d = 4. Even coordinates use sine; odd coordinates use cosine.';}
 else if(t.op==='transpose'){symbolic=sub(sup(mi('K'),mi('T')),mt(`${r},${c}`))+eq+idx('K',`${c},${r}`);numeric=mn(v(a,c,r));legend='T means transpose: exchange indices without changing a value.';}
 else if(t.op==='mask'){symbolic=Y+eq+A+plus+idx('M',`${r},${c}`);numeric=num(v(a))+plus+(t.allowed[r][c]?mn(0):mo('−')+mo('∞'));legend='M is zero for allowed keys and negative infinity for forbidden keys.';}
 else if(t.op==='lookup'){symbolic=idx('x',`${r},${c}`)+eq+idx('E',`${t.indices[r]},${c}`);numeric=mn(calc.result);legend='E is the learned embedding table. A token ID selects one row, not a semantic score.';}
 else if(t.op==='backward'||t.op==='gradient'){symbolic=frac(mi('∂')+mi('L'),mi('∂')+mi('x'))+eq+sum(calc.terms.length,frac(mi('∂')+mi('L'),mi('∂')+idx('y','k'))+frac(mi('∂')+idx('y','k'),mi('∂')+mi('x')));numeric=calc.terms.map((q,i)=>term(num(q.upstream??q.value)+times+num(q.derivative??1),i)).join(plus);legend='L is loss. Each path contributes its incoming gradient multiplied by its local derivative.';}
 else{symbolic=Y+eq+mt(t.op==='tokens'?'vocabulary lookup':t.op==='concat'?'selected head coordinate':t.op==='gather'?'probability of target':t.label);numeric=mn(calc.result);}
 return `<div class="derivation"><span class="math-caption">GENERAL RULE</span>${math(symbolic)}<p class="symbol-key">${escapeMath(legend)}</p><span class="math-caption">SUBSTITUTE THE LIVE VALUES</span><div class="numeric-derivation">${math(numeric+eq+mn(calc.result))}</div><p class="math-rounding">Displayed values are rounded; computation uses full precision.</p></div>`;
}
// Typeset known indices inside native SVG text, without converting the diagram to an image.
export function svgSymbols(s){return escapeMath(s).replace(/\b(W|d|x|q|k|v)_(model|[QKVO]|[0-9]+|k)\b/g,'$1<tspan baseline-shift="sub" font-size="72%">$2</tspan>').replace(/\b([xqkv])([0-9])\b/g,'$1<tspan baseline-shift="sub" font-size="72%">$2</tspan>');}

// A small, explicit notation grammar for code-authored labels. It never evaluates input.
const subDigits={'₀':'0','₁':'1','₂':'2','₃':'3','₄':'4','₅':'5','₆':'6','₇':'7','₈':'8','₉':'9','ₖ':'k','ᵢ':'i','ⱼ':'j'};
const superDigits={'⁰':'0','¹':'1','²':'2','³':'3','⁴':'4','⁵':'5','⁶':'6','⁷':'7','⁸':'8','⁹':'9','⁻':'−','⁺':'+','ᵀ':'T','ᐟ':'/','⁽':'(','⁾':')'};
export function expression(source){
 const sourceText=String(source).replace(/[₀₁₂₃₄₅₆₇₈₉ₖᵢⱼ]+/g,s=>'_{'+[...s].map(c=>subDigits[c]).join('')+'}').replace(/[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺ᵀᐟ⁽⁾]+/g,s=>'^{'+[...s].map(c=>superDigits[c]).join('')+'}').replace(/\b([xqkvhyb])([0-9]+)\b/g,'$1_{$2}').replace(/\b([A-Za-z])\[([\d\s,kijr]+)\]/g,'$1_{$2}').replace(/-/g,'−').replace(/QK\^\{?T\}?/g,'{Q K^{T}}').replace(/([Xx])W_/g,'$1 W_');
 const tokens=sourceText.match(/\d+(?:\.\d+)?(?:[eE][+−]?\d+)?|[A-Za-z]+|[^\s]/gu)||[];let at=0;
 function primary(scripts=true){const t=tokens[at++];if(t===undefined)return row('');let out;
  if(t==='{'||t==='('||t==='['){const end={'{':'}','(':')','[':']'}[t],inside=sequence(end);out=t==='{'?row(inside):row(mo(t)+inside+mo(end));}
  else if(t==='√')out=sqrt(primary());
  else if(t==='−')out=row(mo('−')+primary());
  else if(/^\d/.test(t)){const sci=t.match(/^(.+)[eE]([+−]?\d+)$/);out=sci?mn(sci[1])+mo('×')+sup(mn(10),mn(sci[2])):mn(t);}
  else if(/^[A-Za-z]+$/.test(t))out=t.length===1?mi(t):['sin','cos','exp','ln','log','max','min','softmax','ReLU','Concat'].includes(t)?`<mi mathvariant="normal">${escapeMath(t)}</mi>`:mt(t);
  else out=/[α-ωΑ-Ω]/.test(t)?mi(t):mo(t==='Σ'?'∑':t==='*'?'×':t);
  let lower,upper;while(scripts&&(tokens[at]==='_'||tokens[at]==='^')){const op=tokens[at++],script=primary(false);if(op==='_')lower=script;else upper=script;}if(lower!==undefined&&upper!==undefined)out=el(out===mo('∑')?'munderover':'msubsup',row(out)+row(lower)+row(upper));else if(lower!==undefined)out=sub(out,lower);else if(upper!==undefined)out=sup(out,upper);return out;
 }
 function sequence(end){const parts=[];while(at<tokens.length&&tokens[at]!==end){if(tokens[at]==='/'||tokens[at]==='÷'){at++;const a=parts.pop()||row('');parts.push(frac(a,primary()));}else parts.push(primary());}if(tokens[at]===end)at++;return parts.join('');}
 return sequence();
}
export function notation(source,display=false){return `<math xmlns="http://www.w3.org/1998/Math/MathML" display="${display?'block':'inline'}">${row(expression(source))}</math>`;}
export function richNotation(source){
 // Keep prose as prose; typeset mathematical fragments without guessing sentence structure.
 const s=String(source),pattern=/\b(?:W_[QKVO](?:\[[\d, ]+\])?|d_(?:model|head|k)|[xqkvhy]\[[\d, k]+\]|[xqkvhy]\d+|[QKV]Kᵀ)|[α-ω][₀-₉]*|(?:\d+(?:\.\d+)?|[nd])(?:[⁻⁰¹²³⁴⁵⁶⁷⁸⁹ᐟ]+)|√(?:d_model|dₖ|\d+)|(?:−?\d+(?:\.\d+)?\s*[×÷]\s*−?\d+(?:\.\d+)?(?:\s*=\s*−?\d+(?:\.\d+)?)?)/gu;
 let out='',last=0;for(const m of s.matchAll(pattern)){out+=escapeMath(s.slice(last,m.index))+notation(m[0]);last=m.index+m[0].length;}return out+escapeMath(s.slice(last));
}
export function isFormulaLabel(s){s=String(s);return /[=√∑Σ×÷∂_^]|\b[xqkvhy]\d+\b|[₀-₉ᵀ]|\b(?:sin|cos|max|min|softmax|ReLU)\(/.test(s)&&!/[A-Za-z]{5,}/.test(s.replace(/softmax|Concat|ReLU|model|head|target/g,''));}
export function svgFormula(x,y,html,size=22,color='#ddf7e9',width=900,height=66,anchor='middle'){
 const left=anchor==='middle'?x-width/2:anchor==='end'?x-width:x,align=anchor==='middle'?'center':anchor==='end'?'right':'left';
 return `<foreignObject class="math-asset" x="${left}" y="${y-height*.62}" width="${width}" height="${height}" style="overflow:visible;pointer-events:none"><div xmlns="http://www.w3.org/1999/xhtml" class="equation-html" style="font-size:${size}px;color:${color};text-align:${align};height:100%;display:flex;align-items:center;justify-content:${anchor==='middle'?'center':anchor==='end'?'flex-end':'flex-start'}">${html}</div></foreignObject>`;
}
export function svgLabel(x,y,s,size=18,color='#cfdfdf',anchor='middle'){
 if(isFormulaLabel(s)){const width=Math.min(920,Math.max(45,String(s).length*size*.7));return svgFormula(x,y,notation(s),size,color,width,Math.max(38,size*2.7),anchor);}
 const rich=richNotation(s);if(rich.includes('<math'))return svgFormula(x,y,rich,size,color,Math.min(940,Math.max(70,String(s).length*size*.65)),Math.max(36,size*2.5),anchor);
 return `<text x="${x}" y="${y}" text-anchor="${anchor}" fill="${color}" font-size="${size}" font-family="Arial,sans-serif">${svgSymbols(s)}</text>`;
}
export function termMath(term){if(term.upstream!==undefined)return num(term.upstream)+mo('×')+num(term.derivative??1);if(term.a&&term.b&&term.label.includes('×'))return num(term.a.value)+mo('×')+num(term.b.value);return expression(term.label);}
export function calculationRibbon(calc,index=0){const i=Math.max(0,Math.min(index,calc.terms.length-1)),q=calc.terms[i],term=termMath(q)+mo('=')+mn(q.value),partial=calc.terms.slice(0,i+1).reduce((s,t)=>s+t.value,0);return '<div class="ribbon-math">'+notation('c_{'+i+'}')+math(term)+(calc.additive?math(sub(mi('s'),mn(i))+mo('=')+mn(partial)):'')+'</div>';}

export function referenceFormula(scene){
 const n=scene.reference,id=scene.id||'';let f='';
 if(n===4||n===5)f=n===4?'E(aunt) ≈ E(uncle) + E(woman) − E(man)':'E(daughter) ≈ E(son) + E(woman) − E(man)';
 else if([1,2,3,43,70].includes(n))f='x_i = E_{token_i}';
 else if([6,7,8,61].includes(n))f='x_i^{new} = x_i + Δx_i';
 else if([13,14,15,16,26,27,28,30,31,32,42,44,46,47,48].includes(n))f='Q = XW_Q,   K = XW_K,   V = XW_V';
 else if([17,18,45].includes(n))f='s_{ij} = (q_i ⋅ k_j)/√{d_k}';
 else if([19,20,21,22,25,49,50,51,54,55,58].includes(n))f='a_{ij} = exp(s_{ij}) / {∑_{k=0}^{n−1} exp(s_{ik})}';
 else if([29,52,53,57].includes(n))f='z_i = ∑_{j=0}^{n−1} a_{ij}v_j';
 else if([9,34,56].includes(n))f='MultiHead(Q,K,V) = Concat(head_1,…,head_h) W_O';
 else if(n===33)f='4 × d_{model} × d_{head} = 6291456';
 else if([35,36,37,69].includes(n))f='h_t = f(h_{t−1},x_t)';
 else if([62].includes(n))f='LayerNorm(x) = γ × {(x−μ)/√{σ^2+ε}} + β';
 else if([64,68].includes(n))f='FFN(x) = max(0,xW_1+b_1)W_2+b_2';
 else if([66,67].includes(n))f='PE_{p,2i} = sin(p/{10000^{2i/d_{model}}})';
 else if([10,23,24].includes(n))f='p(y_t|y_{<t},X) = softmax(h_t W_O+b)';
 else if(n)f='Attention(Q,K,V) = softmax({Q K^T}/√{d_k})V';
 else if(id==='paper-optimizer')f='η = d_{model}^{−1/2} min(step^{−1/2},step × warmup^{−3/2})';
 else if(id==='paper-smoothing')f='L = −∑_k q_k ln(p_k)';
 else if(id==='paper-dropout')f='y = {m × x}/{1−p}';
 return f?notation(f,true):'';
}
