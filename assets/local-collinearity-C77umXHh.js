import{o as e}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as t,f as n,x as r,y as i}from"./gpu-contributor-utils-BIwLQON5.js";import{c as a,r as o}from"./wgsl-common-DfyXiYB0.js";import{i as s,s as c,t as l}from"./gpu-chunked-spatial-weights-CfTf1Nzu.js";import{i as u}from"./chunked-weights-kernels-BYYFtXmd.js";import{n as d,r as f,t as p}from"./local-collinearity-wgsl-CKjcBBZN.js";import{t as m}from"./define-run-B-brkCe4.js";import{o as h}from"./inputs-CJKKIRKK.js";import{zn as g}from"./entry-preview-D70TdSbK.js";import{_,l as v,m as y,n as b}from"./regression-CXg5hXjb.js";import{c as x,r as S}from"./probes-CVwGX4BJ.js";import{C,a as w,g as T}from"./weights-BRH93xop.js";var E=p.toExponential(),D=d.toExponential(),O=class{props;id;constructor(e){this.props=e,this.id=e.id??`local-collinearity`;let r=c(this.id,e.weights),a=e.predictorCount;if(n(this.id,`predictorCount`,a,1,12),!!e.output.designConditionNumbers!=!!e.output.varianceDecomposition)throw Error(`${this.id}: design diagnostics outputs must be paired`);for(let[n,i,o]of[[e.predictors,`float32`,r*a],[e.mask,`uint32`,r],[e.output.varianceInflationFactors,`float32`,r*a],[e.output.conditionNumbers,`float32`,r],[e.output.designConditionNumbers,`float32`,r],[e.output.varianceDecomposition,`float32`,r*(a+1)],[e.output.status,`uint32`,r]])if(n&&(t(n,[i],this.id),n.length!==o))throw Error(`${this.id}: shape mismatch`);i(this.id,Object.values(e.output),[e.predictors,e.mask,...s(e.weights)])}getCommandNodes(t){let{id:n,props:i}=this,c=i.predictorCount,d=l(i.weights),p=d.reduce((e,t)=>e+t.rowCount,0);r(n,t,[i.predictors,i.mask,...s(i.weights),...Object.values(i.output)]);let m=[...d.map(r=>e(t,{id:`${n}-fit${r.idSuffix}`,operation:`GPULocalCollinearity`,variant:`weighted-correlation`,invocationCount:r.rowCount,bindings:[{name:`offsets`,view:r.weights.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:r.weights.neighbors,type:`u32`,access:`read`},{name:`weights`,view:r.weights.weights,type:`f32`,access:`read`},{name:`predictors`,view:i.predictors,type:`f32`,access:`read`},...i.mask?[{name:`mask`,view:i.mask,type:`u32`,access:`read`}]:[],{name:`inflation`,view:i.output.varianceInflationFactors,type:`f32`,access:`read_write`},{name:`condition`,view:i.output.conditionNumbers,type:`f32`,access:`read_write`},{name:`status`,view:i.output.status,type:`u32`,access:`read_write`}],declarations:`${r.constantsWGSL}
${o}\n${a}\n${f(c)}
fn included(row: u32) -> bool {
  if (row >= ${p}u) { return false; }
  ${i.mask?`if (mask[maskOffset + row] == 0u) { return false; }`:``}
  for (var column = 0u; column < ${c}u; column++) {
    if (!isFiniteBits(predictors[predictorsOffset + row * ${c}u + column])) { return false; }
  }
  return true;
}`,body:u(r,`status[statusOffset + index] = 1u;
condition[conditionOffset + index] = getNaN();
for (var column = 0u; column < ${c}u; column++) {
  inflation[inflationOffset + index * ${c}u + column] = getNaN();
}
let slotBegin = offsets[offsetsOffset + index];
let slotEnd = offsets[offsetsOffset + index + 1u];
var means: array<f32, ${c}>;
var total = 0.0;
var count = 0u;
for (var slot = slotBegin; slot < slotEnd; slot++) {
  let neighbor = neighbors[neighborsOffset + slot];
  let weight = weights[weightsOffset + slot];
  if (!included(neighbor) || !(weight > 0.0)) { continue; }
  total += weight;
  count++;
  for (var column = 0u; column < ${c}u; column++) {
    means[column] += weight / total * (predictors[predictorsOffset + neighbor * ${c}u + column] - means[column]);
  }
}
if (count <= ${c}u) { return; }
status[statusOffset + index] = 2u;
var correlation: array<f32, ${c*c}>;
for (var slot = slotBegin; slot < slotEnd; slot++) {
  let neighbor = neighbors[neighborsOffset + slot];
  let weight = weights[weightsOffset + slot];
  if (!included(neighbor) || !(weight > 0.0)) { continue; }
  for (var first = 0u; first < ${c}u; first++) {
    let firstValue = predictors[predictorsOffset + neighbor * ${c}u + first] - means[first];
    for (var second = 0u; second < ${c}u; second++) {
      correlation[first * ${c}u + second] += weight * firstValue * (predictors[predictorsOffset + neighbor * ${c}u + second] - means[second]);
    }
  }
}
var scale: array<f32, ${c}>;
for (var column = 0u; column < ${c}u; column++) {
  let diagonal = correlation[column * ${c+1}u];
  if (!(diagonal > 0.0) || !isFiniteBits(diagonal)) { return; }
  scale[column] = inverseSqrt(diagonal);
}
// Condition number from the singular values of the scaled, weighted, centered design (streaming QR).
var factor: array<f32, ${c*c}>;
for (var slot = slotBegin; slot < slotEnd; slot++) {
  let neighbor = neighbors[neighborsOffset + slot];
  let weight = weights[weightsOffset + slot];
  if (!included(neighbor) || !(weight > 0.0)) { continue; }
  var designRow: array<f32, ${c}>;
  for (var column = 0u; column < ${c}u; column++) {
    designRow[column] = sqrt(weight) * (predictors[predictorsOffset + neighbor * ${c}u + column] - means[column]) * scale[column];
  }
  updateDesignFactor_${c}(&factor, &designRow);
}
var vectors: array<f32, ${c*c}>;
var singularValues: array<f32, ${c}>;
if (!getDesignSingularValues_${c}(&factor, &vectors, &singularValues)) { status[statusOffset + index] = 3u; return; }
var smallest = 3.402823e38;
var largest = 0.0;
for (var column = 0u; column < ${c}u; column++) {
  smallest = min(smallest, singularValues[column]);
  largest = max(largest, singularValues[column]);
}
if (!(smallest > largest * ${D}) || !isFiniteBits(largest)) {
  condition[conditionOffset + index] = ${E};
  return;
}
condition[conditionOffset + index] = largest / smallest;
// Columns of the scaled design have unit norm, so the inflation is the inverse diagonal of R^T R.
for (var column = 0u; column < ${c}u; column++) {
  inflation[inflationOffset + index * ${c}u + column] = getDesignFactorInverseDiagonal_${c}(&factor, column);
}
status[statusOffset + index] = 0u;`)}))];if(i.output.designConditionNumbers&&i.output.varianceDecomposition){let r=c+1;m.push(...d.map(s=>e(t,{id:`${n}-variance-decomposition${s.idSuffix}`,operation:`GPULocalCollinearity`,variant:`mgwr-design-variance-decomposition`,invocationCount:s.rowCount,bindings:[{name:`offsets`,view:s.weights.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:s.weights.neighbors,type:`u32`,access:`read`},{name:`weights`,view:s.weights.weights,type:`f32`,access:`read`},{name:`predictors`,view:i.predictors,type:`f32`,access:`read`},...i.mask?[{name:`mask`,view:i.mask,type:`u32`,access:`read`}]:[],{name:`condition`,view:i.output.designConditionNumbers,type:`f32`,access:`read_write`},{name:`proportions`,view:i.output.varianceDecomposition,type:`f32`,access:`read_write`},{name:`status`,view:i.output.status,type:`u32`,access:`read_write`}],declarations:`${o}\n${a}\n${f(r)}${s.constantsWGSL?`\n${s.constantsWGSL}`:``}
fn getDesignRow(neighbor: u32, values: ptr<function, array<f32, ${r}>>) -> bool {
  if (neighbor >= ${p}u) { return false; }
  ${i.mask?`if (mask[maskOffset + neighbor] == 0u) { return false; }`:``}
  (*values)[0] = 1.0;
  for (var column = 0u; column < ${c}u; column++) {
    let value = predictors[predictorsOffset + neighbor * ${c}u + column];
    if (!isFiniteBits(value)) { return false; }
    (*values)[column + 1u] = value;
  }
  return true;
}`,body:u(s,`condition[conditionOffset + index] = getNaN();
for (var column = 0u; column < ${r}u; column++) {
  proportions[proportionsOffset + index * ${r}u + column] = getNaN();
}
if (status[statusOffset + index] != 0u) { return; }
let slotBegin = offsets[offsetsOffset + index];
let slotEnd = offsets[offsetsOffset + index + 1u];
// mgwr weights each design row by w (not sqrt(w)); columns are scaled to unit norm.
var scale: array<f32, ${r}>;
for (var slot = slotBegin; slot < slotEnd; slot++) {
  let weight = weights[weightsOffset + slot];
  var values: array<f32, ${r}>;
  if (!(weight > 0.0) || !getDesignRow(neighbors[neighborsOffset + slot], &values)) { continue; }
  for (var column = 0u; column < ${r}u; column++) { scale[column] += weight * weight * values[column] * values[column]; }
}
for (var column = 0u; column < ${r}u; column++) {
  if (!(scale[column] > 0.0) || !isFiniteBits(scale[column])) { status[statusOffset + index] = 2u; return; }
  scale[column] = inverseSqrt(scale[column]);
}
var factor: array<f32, ${r*r}>;
for (var slot = slotBegin; slot < slotEnd; slot++) {
  let weight = weights[weightsOffset + slot];
  var values: array<f32, ${r}>;
  if (!(weight > 0.0) || !getDesignRow(neighbors[neighborsOffset + slot], &values)) { continue; }
  for (var column = 0u; column < ${r}u; column++) { values[column] *= weight * scale[column]; }
  updateDesignFactor_${r}(&factor, &values);
}
var vectors: array<f32, ${r*r}>;
var singularValues: array<f32, ${r}>;
if (!getDesignSingularValues_${r}(&factor, &vectors, &singularValues)) { status[statusOffset + index] = 3u; return; }
var smallest = 0u;
var largest = 0.0;
for (var column = 0u; column < ${r}u; column++) {
  if (singularValues[column] < singularValues[smallest]) { smallest = column; }
  largest = max(largest, singularValues[column]);
}
let minimum = singularValues[smallest];
if (!(minimum > largest * ${D}) || !isFiniteBits(largest)) {
  condition[conditionOffset + index] = ${E};
  status[statusOffset + index] = 2u;
  return;
}
condition[conditionOffset + index] = largest / minimum;
// Belsley proportions of each coefficient's variance at the smallest singular value.
for (var column = 0u; column < ${r}u; column++) {
  var total = 0.0;
  for (var component = 0u; component < ${r}u; component++) {
    let value = vectors[component * ${r}u + column];
    total += value * value / (singularValues[component] * singularValues[component]);
  }
  let loading = vectors[smallest * ${r}u + column];
  proportions[proportionsOffset + index * ${r}u + column] = (loading * loading / (minimum * minimum)) / total;
}`)})))}return m}},k=m({parameters:g,caption:`Local variance inflation (left) and condition number (right).`,createInputs:({random:e,world:t})=>v(e,t,10,b),build(e,t,n){let r=t.tracts.count,i=e.allocate(`predictors`,`float32`,r*2),a=C(e,`neighborhood`,h(t.tracts.centers,n.neighbors)),o=e.allocate(`vif`,`float32`,r*2),s=e.allocate(`condition-numbers`,`float32`,r);return e.graph.add(new O({weights:a.input,predictors:i.view,predictorCount:2,output:{varianceInflationFactors:o.view,conditionNumbers:s.view,status:e.allocate(`status`,`uint32`,r).view}})),{outputs:{meshes:[_(e,t),_(e,t,t.mapWidth*1.14)],inflation:w(e,`inflation`,o,2,0,r),conditionNumbers:s,weights:a},update:e=>{let n=new Float32Array(r*2);for(let i=0;i<r;i++){let r=t.tracts.centers[i*2]/t.mapWidth,a=Math.min(1,Math.max(0,r))*e.collinearity,o=t.predictors[i*2];n[i*2]=o,n[i*2+1]=a*o+(1-a)*t.predictors[i*2+1]}i.write(n)}}},rebuildOn:[`neighbors`],interaction:{kind:`probe`,hint:`hover a tract`,rest:[.3,.5],resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=t.mapWidth*1.14,o=e.x>a-.02,s=S(t.tracts,e.x,e.y,o?a:0),c=i(n.inflation),l=i(n.conditionNumbers),u=T(i,n.weights,s);return s<0||!c||!l||!u?{}:{focus:s,glyphs:[x(t.tracts,s),x(t.tracts,s,a),...Array.from(u,e=>x(t.tracts,e,0,!0)),...Array.from(u,e=>x(t.tracts,e,a,!0))],readout:`VIF ${c[s].toFixed(1)} · condition ${l[s].toFixed(1)} · window ${r.neighbors}`}}},draw:({inputs:e,outputs:t})=>[...y(e,t.meshes[0],t.inflation,{ramp:`sequential`,logarithmic:!0,range:[1,8]}),...y(e,t.meshes[1],t.conditionNumbers,{ramp:`sequential`,range:[1,12]})]});export{k as default};