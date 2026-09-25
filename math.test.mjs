import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {notation, richNotation, derivation, referenceFormula} from './math.js';
import {Transformer} from './model.js';
import {buildCourse, calculate} from './course.js';
import {simulationMarkup} from './diagrams.js';

// Read the emitted MathML tree without a browser or third-party dependencies.
function tree(markup) {
  const root={tag:'root',children:[]}, stack=[root];
  for(const token of markup.match(/<[^>]+>|[^<]+/g)||[]) {
    if(token.startsWith('</')) {
      assert.equal(stack.pop().tag,token.slice(2,-1));
    } else if(token.startsWith('<')) {
      const node={tag:token.match(/^<([^\s>]+)/)[1],children:[]};
      stack.at(-1).children.push(node);stack.push(node);
    } else stack.at(-1).children.push(token);
  }
  assert.equal(stack.length,1,'balanced MathML');
  return root;
}
const content=node=>typeof node==='string'?node:node.children.map(content).join('');
const nodes=(node,tag)=>typeof node==='string'?[]:[...(node.tag===tag?[node]:[]),...node.children.flatMap(n=>nodes(n,tag))];
const parse=source=>tree(notation(source));

test('fractions contain the full function call and implicit product',()=>{
  for(const [source,numerator,denominator] of [
    ['exp(s)/2','exp⁡(s)','2'],
    ['2i/d_model','2i','dmodel'],
    ['QKᵀ/√dₖ','QKT','dk'],
  ]) {
    const fraction=nodes(parse(source),'mfrac')[0];
    assert.equal(content(fraction.children[0]),numerator);
    assert.equal(content(fraction.children[1]),denominator);
  }
});

test('Unicode scripts, powers, and sum limits remain structured',()=>{
  assert.equal(content(nodes(parse('hₜ₋₁'),'msub')[0].children[1]),'t−1');
  assert.equal(content(nodes(parse('W_Q⁽0⁾'),'msubsup')[0].children[2]),'(0)');
  assert.equal(nodes(parse('∑_{k=0}^{n−1} x_k'),'munderover').length,1);
  const root=parse('sin(p/{10000^{2i/d_model}})');
  assert.equal(nodes(root,'mfrac').length,2);
  assert.equal(nodes(root,'msup').length,1);
});

test('prose keeps spacing, minus signs, radicals, and escaped user text',()=>{
  const markup=richNotation('Divide by √d_k. -0.5 ÷ 2 = -0.25. <img src=x onerror=alert(1)>');
  assert.ok(markup.startsWith('Divide by <math'));
  assert.ok(!markup.includes('√<math'));
  assert.ok(!markup.includes('. -<math'));
  assert.ok(!markup.includes('<img'));
  assert.ok(markup.includes('&lt;img'));
  assert.ok(!richNotation('$A = softmax(QKᵀ/√dₖ)$').includes('$'));
});

test('mean contributions and negative squares match the displayed operation',()=>{
  const model=new Transformer();
  for(const tensor of model.trace.filter(t=>['mean','rowMean'].includes(t.op))) {
    const calc=calculate(model,tensor.id);
    assert.ok(calc.terms.every(term=>term.label.includes('÷')));
    assert.ok(Math.abs(calc.terms.reduce((s,t)=>s+t.value,0)-calc.result)<1e-9);
  }
  const tensor=model.trace.find(t=>t.op==='square'&&model.tensors.get(t.inputs[0]).values[0].some(x=>x<0));
  const col=model.tensors.get(tensor.inputs[0]).values[0].findIndex(x=>x<0);
  assert.match(calculate(model,tensor.id,0,col).formula,/^\(-[\d.]+\)²$/);
});

test('all 394 chapters render valid math structures without raw notation',()=>{
  const model=new Transformer(),course=buildCourse(model);
  assert.equal(course.length,394);assert.equal(course.filter(s=>s.reference).length,70);
  for(const scene of course) {
    const tensor=model.tensors.get(scene.tensor);
    const markup=simulationMarkup(model,scene,{phase:.5,row:0,col:0})+referenceFormula(scene)+(tensor?derivation(model,calculate(model,tensor.id)):'');
    assert.ok(!/undefined|NaN/.test(markup),scene.id);
    for(const [formula] of markup.matchAll(/<math[\s\S]*?<\/math>/g)) {
      const root=tree(formula);
      assert.ok(!/[{}_$]/.test(content(root)),scene.id+': raw notation');
      for(const [tag,count] of [['mfrac',2],['msub',2],['msup',2],['msubsup',3],['munderover',3]])
        assert.ok(nodes(root,tag).every(n=>n.children.length===count),scene.id+': '+tag);
    }
    for(const [,label] of markup.matchAll(/<text\b[^>]*>([^<]*)<\/text>/g))
      assert.ok(!/[=√∑Σ_^ₜᵀ²]/.test(label),scene.id+': unformatted label '+label);
  }
});

test('block math preserves the MathML display type and mobile equations remain visible',()=>{
  const css=readFileSync(new URL('./style.css',import.meta.url),'utf8');
  assert.ok(!/math\[display=block\]\{display:block;/.test(css));
  assert.match(css,/math\[display=block\]\{display:block math/);
  assert.match(css,/@media\(max-width:900px\)\{\s*\.vector-readout strong\{display:block/);
});
