import{n as e,r as t,t as n}from"./interpolation-view-validation-pYo9tqh3.js";import{a as r,s as i}from"./graph-chunk-dispatch-zBt8k2-o.js";import{Q as a}from"./dist-BusamGlx.js";import{c as o,o as s,r as c,t as l}from"./kernel-C3tfQOn4.js";import{s as u,v as d}from"./spatial-context-DwZEd4EZ.js";import{a as f,c as p,o as m}from"./wgsl-kernel-nodes-DagJklH2.js";import{r as h,u as g}from"./wgsl-common-DfyXiYB0.js";import{n as _,t as v}from"./philox-random-DfVxWNPn.js";import{S as y,T as b,d as x,f as S,g as C,v as w}from"./gpu-contributor-utils-BIwLQON5.js";import{t as T}from"./gpu-prediction-utils-DjZFSzGW.js";import{o as E,s as D}from"./prediction-covariance-DfKlqgpX.js";import{r as O}from"./gpu-command-graph-BdzdmV6u.js";import{t as k}from"./gpu-dispatch-utils-CbRzFAgT.js";import{i as A,t as j}from"./graph-chunk-groups-BB8ACpQO.js";import{t as M}from"./define-run-B-brkCe4.js";import{a as N}from"./pointer-overlay-Qh8qflcA.js";import{ec as P}from"./entry-preview-D70TdSbK.js";import{n as F,t as I}from"./channel-D8wtkSDN.js";import{a as L,n as R}from"./interpolation-CWOuK2ek.js";var z=12,B=256;function V(e,t){A(e,t.id,t.bindings.length,B),r(e,t.id,Object.fromEntries(t.bindings.map(e=>[e.name,e.view]))),b(t.id,t.bindings.length+4<=z&&t.invocationCount%B===0,`blocked update needs at most eight storage bindings and complete 256-invocation tiles`);let n=new Uint32Array(z);n[0]=t.blockStart,n[1]=t.invocationCount/B,t.bindings.forEach((e,t)=>{n[4+t]=o(e.view)});let i=k(t.id,t.invocationCount,B,e.device.limits.maxComputeWorkgroupsPerDimension),c=`${t.bindings.map((e,t)=>`@group(0) @binding(${t}) var<storage, ${e.access}> ${e.name}: array<${e.type}>;\nvar<private> ${e.name}Offset: u32;`).join(`
`)}
struct SimulationBlockParameters { words: array<vec4u, ${z/4}> }
@group(0) @binding(${t.bindings.length}) var<uniform> simulationBlockParameters: SimulationBlockParameters;
fn getChunkParameter(word: u32) -> u32 { return simulationBlockParameters.words[word / 4u][word % 4u]; }
${t.declarations??``}
@compute @workgroup_size(${B})
fn main(@builtin(workgroup_id) workgroupId: vec3u,
  @builtin(num_workgroups) workgroupCount: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32) {
  // Address-only references retain every declared binding in WebGPU's inferred layout.
  ${t.bindings.map(e=>`_ = &${e.name};`).join(`
  `)}
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  // All lanes in a padded workgroup return together. Never guard local invocation indices.
  if (workgroupIndex >= getChunkParameter(1u)) { return; }
  let index = workgroupIndex * ${B}u + localInvocationIndex;
  ${t.bindings.map((e,t)=>`${e.name}Offset = getChunkParameter(${4+t}u);`).join(`
  `)}
  ${t.body}
}`;return O({id:t.id,condition:t.condition,resources:[...t.bindings.map(e=>({buffer:e.view,usage:e.access===`read`?`storage-read`:`storage-read-write`})),...t.extraResources??[]],workload:{operation:t.operation,...t.variant?{variant:t.variant}:{},commandCount:1,maximumInvocationCount:t.invocationCount,maximumWorkgroupCount:i.x*i.y*i.z},compile:({device:e})=>{let r=e.createBuffer({id:`${t.id}-block-parameters`,data:n,usage:a.UNIFORM|a.COPY_DST}),o;try{o=new l(e,{id:`${t.operation}-${t.variant}`,source:c,shaderLayout:{bindings:[...t.bindings.map((e,t)=>({name:e.name,type:e.access===`read`?`read-only-storage`:`storage`,group:0,location:t})),{name:`simulationBlockParameters`,type:`uniform`,group:0,location:t.bindings.length}]}})}catch(e){throw r.destroy(),e}return{encode:({computePass:e,getBuffer:a})=>o.dispatch(e,{bindings:Object.fromEntries([...t.bindings.map(e=>[e.name,s(e.view,a)]),[`simulationBlockParameters`,{buffer:r,offset:0,size:n.byteLength}]]),...i}),destroy:()=>{o.destroy(),r.destroy()}}}})}var H=16,U=256;function W(e,t){let{id:n,props:r,capacity:a,scratch:o,moments:s,control:l}=t,u=r.queries.length,d=c(e,`${n}-blocked-failure`,`uint32`,1),g=[...t.conditionBindings,{name:`failure`,view:d,type:`atomic<u32>`,access:`read_write`}],_=`scratch[scratchOffset] == 0.0`+(l?` && preparationState[preparationStateOffset + 2u] != 0u`:``),v=`scratch[scratchOffset] == 0.0`+(l?` && preparationState[preparationStateOffset + 3u] != 0u`:``),y=p((r.covariance.sill+(r.covariance.nugget??0))*1e-6),b=`${t.declarations}
const BLOCK_SIZE: u32 = ${H}u;
const TOLERANCE: f32 = ${y};
fn observedCount() -> u32 { return u32(scratch[scratchOffset + 1u]); }
fn jointSize() -> u32 { return observedCount() + ${u}u; }
// Query rows do not hold observed values. Their slots retain exact-coordinate aliases instead.
fn queryAlias(row: u32) -> u32 { return u32(scratch[scratchOffset + ${t.observationStart}u + row]); }
fn getWhitened(row: u32) -> f32 { return scratch[scratchOffset + ${t.whitenedStart}u + row]; }
fn writeWhitened(row: u32, value: f32) { scratch[scratchOffset + ${t.whitenedStart}u + row] = value; }`,x=(e,t,r=``)=>({id:`${n}-${e}${r}`,operation:`GPUConditionalGaussianSimulation`,variant:e,bindings:g,...t,declarations:b+(t.declarations??``)}),S=[x(`blocked-covariance`,{invocationCount:a*a,body:`if (!(${_})) { return; }
let row = index / ${a}u;
let column = index % ${a}u;
if (row >= jointSize() || column > row) { return; }
var value = covariance(point(row), point(column));
if (row == column) { value += scratch[scratchOffset + ${t.errorStart}u + row]; }
writeFactor(row, column, value);`}),x(`blocked-query-aliases`,{invocationCount:u,body:`if (!(${_})) { return; }
let row = observedCount() + index;
var canonicalRow = row;
for (var candidate = 0u; candidate < row; candidate++) {
  // Covariance includes the process nugget at identical coordinates. Only observation
  // measurement error distinguishes an observed variable from the latent query variable.
  if (all(point(candidate) == point(row)) && (candidate >= observedCount() || scratch[scratchOffset + ${t.errorStart}u + candidate] == 0.0)) {
    canonicalRow = candidate; break;
  }
}
scratch[scratchOffset + ${t.observationStart}u + row] = f32(canonicalRow);`})];for(let e=0;e<a;e+=H){let t={BLOCK_START:e};S.push(x(`blocked-diagonal`,{invocationCount:1,declarations:`
override BLOCK_START: u32;`,constants:t,body:`if (!(${_}) || atomicLoad(&failure[failureOffset]) != 0u) { return; }
let blockEnd = min(BLOCK_START + BLOCK_SIZE, jointSize());
for (var column = BLOCK_START; column < blockEnd; column++) {
  var diagonal = factor(column, column);
  for (var component = BLOCK_START; component < column; component++) {
    let loading = factor(column, component); diagonal -= loading * loading;
  }
  if (!isFiniteValue(diagonal) || diagonal < -TOLERANCE || (column < observedCount() && diagonal <= TOLERANCE)) {
    atomicStore(&failure[failureOffset], 1u); return;
  }
  if (column >= observedCount() && queryAlias(column) < column) {
    // An exact duplicate has no new random degree of freedom. Do not turn cancellation
    // residue into independent noise, or clamp small positive pivots of distinct sites.
    if (abs(diagonal) > TOLERANCE) { atomicStore(&failure[failureOffset], 1u); return; }
    diagonal = 0.0;
  }
  diagonal = sqrt(max(diagonal, 0.0));
  writeFactor(column, column, diagonal);
  for (var row = column + 1u; row < blockEnd; row++) {
    var value = factor(row, column);
    for (var component = BLOCK_START; component < column; component++) { value -= factor(row, component) * factor(column, component); }
    if (diagonal == 0.0) {
      if (abs(value) > TOLERANCE || !isFiniteValue(value)) { atomicStore(&failure[failureOffset], 1u); return; }
      writeFactor(row, column, 0.0);
    } else {
      let loading = value / diagonal;
      if (!isFiniteValue(loading)) { atomicStore(&failure[failureOffset], 1u); return; }
      writeFactor(row, column, loading);
    }
  }
}`},`-${e}`));let n=a-e-H;if(n<=0)continue;S.push(x(`blocked-panel`,{invocationCount:n,declarations:`
override BLOCK_START: u32;`,constants:t,body:`if (!(${_}) || atomicLoad(&failure[failureOffset]) != 0u) { return; }
let row = BLOCK_START + BLOCK_SIZE + index;
if (row >= jointSize()) { return; }
for (var column = BLOCK_START; column < BLOCK_START + BLOCK_SIZE; column++) {
  var value = factor(row, column);
  for (var component = BLOCK_START; component < column; component++) { value -= factor(row, component) * factor(column, component); }
  let diagonal = factor(column, column);
  if (diagonal == 0.0) {
    if (abs(value) > TOLERANCE || !isFiniteValue(value)) { atomicStore(&failure[failureOffset], 1u); return; }
    writeFactor(row, column, 0.0);
  } else {
    let loading = value / diagonal;
    if (!isFiniteValue(loading)) { atomicStore(&failure[failureOffset], 1u); return; }
    writeFactor(row, column, loading);
  }
}`},`-${e}`));let r=Math.ceil(n/H);S.push(x(`blocked-update`,{invocationCount:r*r*U,workgroupSize:U,guardIndex:!1,declarations:`
override BLOCK_START: u32;
override TILES_PER_AXIS: u32;
var<workgroup> workgroupActive: u32;
var<workgroup> rowTile: array<f32, ${U}>;
var<workgroup> columnTile: array<f32, ${U}>;`,constants:{...t,TILES_PER_AXIS:r},body:`let tileRow = workgroupIndex / TILES_PER_AXIS;
let tileColumn = workgroupIndex % TILES_PER_AXIS;
// Tile coordinates and the workgroup-uniform gate are uniform across every barrier.
if (tileColumn > tileRow) { return; }
if (localInvocationIndex == 0u) { workgroupActive = u32(${_} && atomicLoad(&failure[failureOffset]) == 0u); }
if (workgroupUniformLoad(&workgroupActive) == 0u) { return; }
let localRow = localInvocationIndex / BLOCK_SIZE;
let localColumn = localInvocationIndex % BLOCK_SIZE;
let row = BLOCK_START + BLOCK_SIZE + tileRow * BLOCK_SIZE + localRow;
let column = BLOCK_START + BLOCK_SIZE + tileColumn * BLOCK_SIZE + localColumn;
rowTile[localInvocationIndex] = 0.0;
columnTile[localInvocationIndex] = 0.0;
if (row < jointSize()) { rowTile[localInvocationIndex] = factor(row, BLOCK_START + localColumn); }
if (column < jointSize()) { columnTile[localInvocationIndex] = factor(column, BLOCK_START + localRow); }
workgroupBarrier();
if (row < jointSize() && column <= row) {
  var value = factor(row, column);
  for (var component = 0u; component < BLOCK_SIZE; component++) {
    value -= rowTile[localRow * BLOCK_SIZE + component] * columnTile[component * BLOCK_SIZE + localColumn];
  }
  writeFactor(row, column, value);
}`},`-${e}`))}S.push(x(`blocked-factor-status`,{invocationCount:1,body:`if (!(${_})) { return; }
if (atomicLoad(&failure[failureOffset]) != 0u) { scratch[scratchOffset] = 2.0; return; }
${l?`preparationState[preparationStateOffset + 6u] = 1u; preparationState[preparationStateOffset] = preparationControl[preparationControlOffset];`:``}`}),x(`blocked-whiten-initialize`,{invocationCount:Math.max(r.positions.length,1),body:`if (!(${v})) { return; }
${l?`if (index == 0u) { preparationState[preparationStateOffset + 5u]++; }`:``}
if (index < observedCount()) { writeWhitened(index, scratch[scratchOffset + ${t.observationStart}u + index]); }`}));for(let e=0;e<r.positions.length;e+=H){S.push(x(`blocked-whiten-diagonal`,{invocationCount:1,declarations:`
override BLOCK_START: u32;`,constants:{BLOCK_START:e},body:`if (!(${v}) || atomicLoad(&failure[failureOffset]) != 0u) { return; }
let blockEnd = min(BLOCK_START + BLOCK_SIZE, observedCount());
for (var row = BLOCK_START; row < blockEnd; row++) {
  var value = getWhitened(row);
  for (var column = BLOCK_START; column < row; column++) { value -= factor(row, column) * getWhitened(column); }
  value /= factor(row, row);
  if (!isFiniteValue(value)) { atomicStore(&failure[failureOffset], 1u); return; }
  writeWhitened(row, value);
}`},`-${e}`));let t=r.positions.length-e-H;t<=0||S.push(x(`blocked-whiten-update`,{invocationCount:t,declarations:`
override BLOCK_START: u32;`,constants:{BLOCK_START:e},body:`if (!(${v}) || atomicLoad(&failure[failureOffset]) != 0u) { return; }
let row = BLOCK_START + BLOCK_SIZE + index;
if (row >= observedCount()) { return; }
var value = getWhitened(row);
for (var column = BLOCK_START; column < BLOCK_START + BLOCK_SIZE; column++) { value -= factor(row, column) * getWhitened(column); }
writeWhitened(row, value);`},`-${e}`))}S.push(x(`blocked-moments`,{invocationCount:u,body:`if (!(${v}) || atomicLoad(&failure[failureOffset]) != 0u) { return; }
let row = observedCount() + index;
var expectation = ${t.meanExpression};
var variance = 0.0;
for (var column = 0u; column < observedCount(); column++) { expectation += factor(row, column) * getWhitened(column); }
for (var component = 0u; component <= index; component++) { let loading = factor(row, observedCount() + component); variance += loading * loading; }
if (!isFiniteValue(expectation) || !isFiniteValue(variance)) { atomicStore(&failure[failureOffset], 1u); return; }
if (queryAlias(row) < observedCount()) {
  expectation = ${t.meanExpression} + scratch[scratchOffset + ${t.observationStart}u + queryAlias(row)];
  variance = 0.0;
}
moments[momentsOffset + index * 2u] = expectation;
moments[momentsOffset + index * 2u + 1u] = variance;`}),x(`blocked-moments-status`,{invocationCount:1,body:`if (!(${v})) { return; }
if (atomicLoad(&failure[failureOffset]) != 0u) { scratch[scratchOffset] = 2.0; return; }
${l?`preparationState[preparationStateOffset + 1u] = preparationControl[preparationControlOffset + 1u];`:``}`}));let C=f(e,{id:`${n}-blocked-preparation-dispatch`,operation:`GPUConditionalGaussianSimulation`,shapes:S.map(e=>[e.invocationCount,e.workgroupSize??64])}),w=S.findIndex(e=>e.variant===`blocked-whiten-initialize`),T=i(e,`${n}-blocked-dispatch-shapes`,Uint32Array.from(S.flatMap((t,n)=>{let r=k(`GPUConditionalGaussianSimulation`,t.invocationCount,t.workgroupSize??64,e.device.limits.maxComputeWorkgroupsPerDimension);return[r.x,r.y,r.z,Number(n>=w)]})));return[...T.nodes,m(e,{id:`${n}-blocked-preparation-gates`,operation:`GPUConditionalGaussianSimulation`,variant:`blocked-preparation-gates`,invocationCount:S.length,bindings:[{name:`scratch`,view:o,type:`f32`,access:`read`},{name:`failure`,view:d,type:`atomic<u32>`,access:`read_write`},C.binding,{name:`dispatchShapes`,view:T.parameters,type:`u32`,access:`read`},...l&&t.preparationState?[{name:`preparationControl`,view:l,type:`u32`,access:`read`},{name:`preparationState`,view:t.preparationState,type:`u32`,access:`read_write`}]:[]],body:`let shape = dispatchShapesOffset + index * 4u;
let enabled = select(${_}, ${v}, dispatchShapes[shape + 3u] != 0u);
dispatch[dispatchOffset + index * 3u] = select(0u, dispatchShapes[shape], enabled);
dispatch[dispatchOffset + index * 3u + 1u] = dispatchShapes[shape + 1u];
dispatch[dispatchOffset + index * 3u + 2u] = dispatchShapes[shape + 2u];
if (index == 0u) {
  atomicStore(&failure[failureOffset], 0u);
  ${l?`if (${_}) { preparationState[preparationStateOffset + 4u]++; }`:``}
}`}),...S.map((t,r)=>{let i=t.constants?.BLOCK_START??0,o=e=>e.replace(/override (?:BLOCK_START|TILES_PER_AXIS): u32;/g,``).replace(/\bTILES_PER_AXIS\b/g,`((${a}u - getChunkParameter(0u) - 1u) / BLOCK_SIZE)`).replace(/\bBLOCK_START\b/g,`getChunkParameter(0u)`);return t.guardIndex===!1?V(e,{id:t.id,operation:t.operation,variant:t.variant,bindings:t.bindings,invocationCount:t.invocationCount,blockStart:i,declarations:o(t.declarations??``),body:o(t.body),...C.getGate(r)}):j(e,{id:t.id,pipelineId:`${n}-${t.variant}`,operation:t.operation,variant:t.variant,bindings:t.bindings,invocationCount:t.invocationCount,workgroupSize:t.workgroupSize??64,rowBase:i,totalLength:a,declarations:o(t.declarations??``),body:`${t.bindings.map(e=>`_ = &${e.name};`).join(`
`)}
${o(t.body)}`,...C.getGate(r)})}),m(e,{id:`${n}-blocked-publish-moments`,operation:`GPUConditionalGaussianSimulation`,variant:`blocked-publish-moments`,invocationCount:u,bindings:[{name:`scratch`,view:o,type:`f32`,access:`read`},{name:`moments`,view:s,type:`f32`,access:`read_write`},...l?[{name:`outputMoments`,view:r.output.moments,type:`f32`,access:`read_write`}]:[]],declarations:h,body:`if (scratch[scratchOffset] != 0.0) { moments[momentsOffset + index * 2u] = getNaN(); moments[momentsOffset + index * 2u + 1u] = getNaN(); }
else {
  let observedCount = u32(scratch[scratchOffset + 1u]);
  let canonicalRow = u32(scratch[scratchOffset + ${t.observationStart}u + observedCount + index]);
  // Aliases always refer to the first matching variable, so canonical query moments
  // are read-only throughout this pass and there are no read/write alias chains.
  if (canonicalRow >= observedCount && canonicalRow < observedCount + index) {
    moments[momentsOffset + index * 2u] = moments[momentsOffset + (canonicalRow - observedCount) * 2u];
    moments[momentsOffset + index * 2u + 1u] = moments[momentsOffset + (canonicalRow - observedCount) * 2u + 1u];
  }
}
${l?`outputMoments[outputMomentsOffset + index * 2u] = moments[momentsOffset + index * 2u]; outputMoments[outputMomentsOffset + index * 2u + 1u] = moments[momentsOffset + index * 2u + 1u];`:``}`}),...G(e,t)]}function G(e,t){let{id:n,props:r,capacity:i,scratch:a,moments:o,control:s}=t,l=r.queries.length,u=l*r.replicateCount,d=c(e,`${n}-normal-draws`,`float32`,u),f=s?[{name:`preparationControl`,view:s,type:`u32`,access:`read`}]:[],p=s?`preparationControl[preparationControlOffset + 5u]`:`${r.replicateCount}u`,y=s?`preparationControl[preparationControlOffset + 4u]`:`${r.replicateOffset??0}u`,b=s?`preparationControl[preparationControlOffset + 3u]`:`${r.seed}u`,x=Math.ceil(l/H),S=Math.ceil(r.replicateCount/H);return[m(e,{id:`${n}-blocked-normals`,operation:`GPUConditionalGaussianSimulation`,variant:`blocked-normals`,invocationCount:u,bindings:[{name:`scratch`,view:a,type:`f32`,access:`read`},{name:`normals`,view:d,type:`f32`,access:`read_write`},{name:`statuses`,view:r.output.status,type:`u32`,access:`read_write`},...f],declarations:_,body:`let replicate = index / ${l}u;
let component = index % ${l}u;
let status = select(u32(scratch[scratchOffset]), 1u, replicate >= ${p});
if (component == 0u) { statuses[statusesOffset + replicate] = status; }
normals[normalsOffset + index] = 0.0;
if (status != 0u) { return; }
let random = philox4x32(vec4<u32>(replicate + ${y}, component, 0u, 0u), vec2<u32>(${b}, ${v.conditionalGaussianSimulation.id}u));
// Box-Muller with an open radius uniform, as in the cooperative path (same words, same draws).
normals[normalsOffset + index] = sqrt(-2.0 * log(philoxOpenUnitFloat(random.x))) * cos(6.28318530718 * (f32(random.y >> 8u) + 0.5) / 16777216.0);`}),m(e,{id:`${n}-blocked-realizations`,operation:`GPUConditionalGaussianSimulation`,variant:`blocked-realizations`,invocationCount:x*S*U,workgroupSize:U,guardIndex:!1,bindings:[{name:`scratch`,view:a,type:`f32`,access:`read`},{name:`moments`,view:o,type:`f32`,access:`read`},{name:`normals`,view:d,type:`f32`,access:`read`},{name:`values`,view:r.output.values,type:`f32`,access:`read_write`},{name:`statuses`,view:r.output.status,type:`atomic<u32>`,access:`read_write`},...f],declarations:`${g}
var<workgroup> workgroupActive: u32;
var<workgroup> factorTile: array<f32, ${U}>;
var<workgroup> normalTile: array<f32, ${U}>;`,body:`let localQuery = localInvocationIndex / ${H}u;
let localReplicate = localInvocationIndex % ${H}u;
let query = (workgroupIndex / ${S}u) * ${H}u + localQuery;
let replicate = (workgroupIndex % ${S}u) * ${H}u + localReplicate;
if (localInvocationIndex == 0u) { workgroupActive = u32(scratch[scratchOffset] == 0.0); }
if (workgroupUniformLoad(&workgroupActive) == 0u) { return; }
let observedCount = u32(scratch[scratchOffset + 1u]);
var value = 0.0;
if (query < ${l}u) { value = moments[momentsOffset + query * 2u]; }
for (var componentStart = 0u; componentStart < ${l}u; componentStart += ${H}u) {
  factorTile[localInvocationIndex] = 0.0;
  normalTile[localInvocationIndex] = 0.0;
  let factorColumn = componentStart + localReplicate;
  if (query < ${l}u && factorColumn <= query) { factorTile[localInvocationIndex] = scratch[scratchOffset + 2u + (observedCount + query) * ${i}u + observedCount + factorColumn]; }
  let normalComponent = componentStart + localQuery;
  if (replicate < ${p} && normalComponent < ${l}u) { normalTile[localInvocationIndex] = normals[normalsOffset + replicate * ${l}u + normalComponent]; }
  workgroupBarrier();
  for (var component = 0u; component < ${H}u; component++) {
    if (componentStart + component <= query) { value += factorTile[localQuery * ${H}u + component] * normalTile[component * ${H}u + localReplicate]; }
  }
  workgroupBarrier();
}
if (query < ${l}u && replicate < ${p}) {
  values[valuesOffset + replicate * ${l}u + query] = value;
  if (!isFiniteValue(value)) { atomicMax(&statuses[statusesOffset + replicate], 2u); }
}`}),m(e,{id:`${n}-blocked-aliased-realizations`,operation:`GPUConditionalGaussianSimulation`,variant:`blocked-aliased-realizations`,invocationCount:u,bindings:[{name:`scratch`,view:a,type:`f32`,access:`read`},{name:`moments`,view:o,type:`f32`,access:`read`},{name:`statuses`,view:r.output.status,type:`u32`,access:`read`},{name:`values`,view:r.output.values,type:`f32`,access:`read_write`}],body:`let replicate = index / ${l}u;
if (statuses[statusesOffset + replicate] != 0u) { return; }
let query = index % ${l}u;
let observedCount = u32(scratch[scratchOffset + 1u]);
let canonicalRow = u32(scratch[scratchOffset + ${t.observationStart}u + observedCount + query]);
if (canonicalRow < observedCount) { values[valuesOffset + index] = moments[momentsOffset + query * 2u]; }
else if (canonicalRow < observedCount + query) {
  // The earliest matching query never writes in this pass; aliases cannot form chains.
  values[valuesOffset + index] = values[valuesOffset + replicate * ${l}u + canonicalRow - observedCount];
}`}),m(e,{id:`${n}-blocked-realization-status`,operation:`GPUConditionalGaussianSimulation`,variant:`blocked-realization-status`,invocationCount:u,bindings:[{name:`statuses`,view:r.output.status,type:`u32`,access:`read`},{name:`values`,view:r.output.values,type:`f32`,access:`read_write`}],declarations:h,body:`if (statuses[statusesOffset + index / ${l}u] != 0u) { values[valuesOffset + index] = getNaN(); }`})]}var K=4096,q=Object.freeze({...u,measure:`distance`}),J=class{id;spatialContext;props;preparationControl;preparationControlBuffer;preparationStatisticsView;ownedBuffers=[];destroyed=!1;constructor(r){if(this.props=r,this.id=r.id??`conditional-gaussian-simulation`,this.spatialContext=d(this.id,r.spatialContext,q),D(r.covariance),C(this.id,`covariance.range`,r.covariance.range,Number.MIN_VALUE),C(this.id,`covariance.sill`,r.covariance.sill,0),C(this.id,`covariance.nugget`,r.covariance.nugget??0,0),C(this.id,`covariance total sill`,r.covariance.sill+(r.covariance.nugget??0),Number.MIN_VALUE),r.covariance.model===`matern`)throw Error(`${this.id} does not support the Matern covariance`);if(S(this.id,`replicateCount`,r.replicateCount,1),S(this.id,`replicateOffset`,r.replicateOffset??0,0),y(this.id,`observations plus queries`,r.positions.length+r.queries.length,K),r.queries.length<1||r.positions.length!==r.observations.length||r.output.values.length<r.queries.length*r.replicateCount||r.output.moments.length<r.queries.length||r.output.status.length<r.replicateCount||!Number.isInteger(r.seed)||r.seed<0||r.seed>4294967295||(r.replicateOffset??0)+r.replicateCount>4294967296||!Number.isFinite(r.mean??0))throw Error(`${this.id} invalid bounded simulation topology`);C(this.id,`mean`,r.mean??0),n(this.id,r.output,[{read:{moments:r.reusePreparation?void 0:r.output.moments},write:{values:r.output.values,status:r.output.status,moments:r.reusePreparation&&r.positions.length<=128&&r.queries.length<=128?r.output.moments:void 0}},{write:{moments:r.output.moments}},...r.positions.length>128||r.queries.length>128?[{read:{status:r.output.status},write:{values:r.output.values}}]:[]]),e(this.id,Object.values(r.output),[r.positions,r.observations,r.queries]),t(this.id,[r.positions,r.observations,r.queries],Object.values(r.output)),r.reusePreparation&&(y(this.id,`replicateCount with reusePreparation`,r.replicateCount,4294967295),this.preparationControl=Uint32Array.of(1,1,0,r.seed,r.replicateOffset??0,r.replicateCount),new Float32Array(this.preparationControl.buffer)[2]=r.mean??0)}get preparationStatistics(){return this.preparationStatisticsView}invalidatePreparation(){let e=this.getPreparationControl();b(this.id,e[0]<4294967295&&e[1]<4294967295,`invalidatePreparation generations must stay below 2^32 - 1; create a new producer`),e[0]++,e[1]++,this.writePreparationControl()}invalidateObservations(){let e=this.getPreparationControl();b(this.id,e[1]<4294967295,`invalidateObservations generations must stay below 2^32 - 1; create a new producer`),e[1]++,this.writePreparationControl()}setMean(e){let t=this.getPreparationControl();C(this.id,`mean`,e),b(this.id,t[1]<4294967295,`setMean generations must stay below 2^32 - 1; create a new producer`),new Float32Array(t.buffer)[2]=e,t[1]++,this.writePreparationControl()}setDrawParameters(e){let t=this.getPreparationControl(),n=e.seed??t[3],r=e.replicateOffset??t[4],i=e.replicateCount??t[5];S(this.id,`seed`,n,0,4294967295),S(this.id,`replicateOffset`,r,0,2**53-1),S(this.id,`replicateCount`,i,1,this.props.replicateCount),y(this.id,`replicateOffset + replicateCount`,r+i,4294967296),t[3]=n,t[4]=r,t[5]=i,this.writePreparationControl()}destroy(){for(let e of this.ownedBuffers)e.destroy();this.ownedBuffers=[],this.preparationControlBuffer=void 0,this.preparationStatisticsView=void 0,this.destroyed=!0}getPreparationControl(){return b(this.id,!!this.preparationControl&&!this.destroyed,`preparation controls need reusePreparation and a producer that was not destroyed`),this.preparationControl}writePreparationControl(){this.preparationControlBuffer&&this.preparationControlBuffer.write(this.getPreparationControl())}getCommandNodes(e){let{id:t,props:n}=this,i=Math.max(1,n.positions.length+n.queries.length),o=n.queries.length;r(e,t,{positions:n.positions,observations:n.observations,queries:n.queries,...n.output}),(n.positions.length>128||o>128)&&A(e,t,n.reusePreparation?8:6,256);let s=i*i,l=2+s+i*5;w(e.device,t,`joint covariance factor`,l*4,`reduce observations plus queries`),w(e.device,t,`realization batch`,o*n.replicateCount*4,`reduce replicateCount and use replicateOffset for explicit batches`),w(e.device,t,`blocked preparation dispatch`,(Math.ceil(i/16)*5+10)*16,`reduce observations plus queries`);let u=(n,r,i,o)=>{let s=e.device.createBuffer({id:`${t}-${n}`,byteLength:i*(r===`float32x2`?8:4),usage:a.STORAGE|a.COPY_SRC|a.COPY_DST,...o?{data:o}:{}});return this.ownedBuffers.push(s),{buffer:s,view:x(e,`${t}-${n}`,s,r,i)}},d=!!this.preparationControl;d&&b(this.id,!this.destroyed&&this.ownedBuffers.length===0,`with reusePreparation is added to one graph, once, before destroy`);let f=d?u(`factor`,`float32`,l).view:c(e,`${t}-factor`,`float32`,l),y=d?u(`preparation-moments`,`float32x2`,o).view:n.output.moments,S=d?u(`preparation-control`,`uint32`,6,this.preparationControl):void 0,C=d?u(`preparation-state`,`uint32`,7,new Uint32Array(7)).view:void 0;S&&C&&(this.preparationControlBuffer=S.buffer,this.preparationStatisticsView=e.createDataView(C.buffer,{format:`uint32x2`,length:1,byteOffset:16}));let D=d?`bitcast<f32>(preparationControl[preparationControlOffset + 2u])`:p(n.mean??0),O=2+s,k=O+i*2,j=k+i,M=j+i,N=[{name:`positions`,view:n.positions,type:`f32`,access:`read`},{name:`observations`,view:n.observations,type:`f32`,access:`read`},{name:`queries`,view:n.queries,type:`f32`,access:`read`},{name:`scratch`,view:f,type:`f32`,access:`read_write`},{name:`moments`,view:y,type:`f32`,access:`read_write`},...S&&C?[{name:`preparationControl`,view:S.view,type:`u32`,access:`read`},{name:`preparationState`,view:C,type:`u32`,access:`read_write`}]:[]],P=`${T({positions:2,observations:4,queries:2})}\n${h}\n${g}\n${E(n.covariance)}
fn point(row: u32) -> vec2<f32> { return vec2<f32>(scratch[scratchOffset + ${O}u + row * 2u], scratch[scratchOffset + ${O+1}u + row * 2u]); }
fn factor(row: u32, column: u32) -> f32 { return scratch[scratchOffset + 2u + row * ${i}u + column]; }
fn writeFactor(row: u32, column: u32, value: f32) { scratch[scratchOffset + 2u + row * ${i}u + column] = value; }`,F=(n,r,i=!1)=>m(e,{id:`${t}-${n}`,operation:`GPUConditionalGaussianSimulation`,variant:n,invocationCount:i?64:1,workgroupSize:i?64:2,guardIndex:!i,bindings:N,declarations:P+(i?`
var<workgroup> failed: atomic<u32>;`+(d?`
var<workgroup> shouldFactor: u32;`:``):``),body:r}),I=F(`prepare-conditioning`,`${d?`
// State: applied geometry/value generations, factor/moment gates, execution counts, valid factor.
preparationState[preparationStateOffset + 2u] = 0u;
preparationState[preparationStateOffset + 3u] = 0u;
var rebuildFactor = preparationState[preparationStateOffset + 6u] != 1u || preparationState[preparationStateOffset] != preparationControl[preparationControlOffset];
if (!rebuildFactor && preparationState[preparationStateOffset + 1u] == preparationControl[preparationControlOffset + 1u] && scratch[scratchOffset] == 0.0) { return; }
let previousObservedCount = u32(scratch[scratchOffset + 1u]);
// Any failed validation invalidates reuse until a complete preparation succeeds.
preparationState[preparationStateOffset + 6u] = 0u;
preparationState[preparationStateOffset + 3u] = 1u;
`:``}
scratch[scratchOffset] = 1.0; scratch[scratchOffset + 1u] = 0.0;
for (var query = 0u; query < ${o}u; query++) { moments[momentsOffset + query * 2u] = getNaN(); moments[momentsOffset + query * 2u + 1u] = getNaN(); }
var observedCount = 0u;
for (var sample = 0u; sample < ${n.positions.length}u; sample++) {
  let observation = read_observations(sample); if (!(observation.w > 0.0)) { continue; }
  let position = read_positions(sample);
  if (!isFiniteValue(observation.x) || !isFiniteValue(observation.y) || observation.y < 0.0 || !isFiniteValue(position.x) || !isFiniteValue(position.y)) { return; }
  ${d?`if (!rebuildFactor && (any(point(observedCount) != position) || scratch[scratchOffset + ${j}u + observedCount] != observation.y)) { rebuildFactor = true; }`:``}
  scratch[scratchOffset + ${O}u + observedCount * 2u] = position.x;
  scratch[scratchOffset + ${O+1}u + observedCount * 2u] = position.y;
  let centered = observation.x - ${D};
  if (!isFiniteValue(centered)) { scratch[scratchOffset] = 2.0; return; }
  scratch[scratchOffset + ${k}u + observedCount] = centered;
  scratch[scratchOffset + ${j}u + observedCount] = observation.y;
  observedCount++;
}
scratch[scratchOffset + 1u] = f32(observedCount);
${d?`if (observedCount != previousObservedCount) { rebuildFactor = true; }`:``}
for (var query = 0u; query < ${o}u; query++) {
  let position = read_queries(query); if (!isFiniteValue(position.x) || !isFiniteValue(position.y)) { return; }
  let row = observedCount + query;
  ${d?`if (!rebuildFactor && any(point(row) != position)) { rebuildFactor = true; }`:``}
  scratch[scratchOffset + ${O}u + row * 2u] = position.x;
  scratch[scratchOffset + ${O+1}u + row * 2u] = position.y;
  scratch[scratchOffset + ${j}u + row] = 0.0;
}
scratch[scratchOffset] = 0.0;
${d?`preparationState[preparationStateOffset + 2u] = u32(rebuildFactor);
preparationState[preparationStateOffset + 6u] = u32(!rebuildFactor);`:``}`);return n.positions.length>128||o>128?[I,...W(e,{id:t,props:n,capacity:i,scratch:f,moments:y,control:S?.view,preparationState:C,declarations:P,conditionBindings:N,observationStart:k,errorStart:j,whitenedStart:M,meanExpression:D})]:[I,F(`cooperative-factor`,`
${d?`
if (localInvocationIndex == 0u) { shouldFactor = u32(preparationState[preparationStateOffset + 2u] != 0u && scratch[scratchOffset] == 0.0); }
if (workgroupUniformLoad(&shouldFactor) == 0u) { return; }
if (localInvocationIndex == 0u) { preparationState[preparationStateOffset + 4u]++; }
`:``}
if (localInvocationIndex == 0u) { atomicStore(&failed, select(1u, 0u, scratch[scratchOffset] == 0.0)); }
workgroupBarrier();
let observedCount = u32(scratch[scratchOffset + 1u]);
let size = observedCount + ${o}u;
let tolerance = ${p((n.covariance.sill+(n.covariance.nugget??0))*1e-6)};
// Every lane reaches every barrier, including inactive rows and failed factorizations.
for (var column = 0u; column < ${i}u; column++) {
  if (localInvocationIndex == 0u && column < size && atomicLoad(&failed) == 0u) {
    var value = covariance(point(column), point(column)) + scratch[scratchOffset + ${j}u + column];
    for (var component = 0u; component < column; component++) { let loading = factor(column, component); value -= loading * loading; }
    if (!isFiniteValue(value) || value < -tolerance || (column < observedCount && value <= tolerance)) { atomicStore(&failed, 1u); }
    else { writeFactor(column, column, sqrt(max(value, 0.0))); }
  }
  storageBarrier();
  workgroupBarrier();
  if (atomicLoad(&failed) == 0u) {
    for (var row = column + 1u + localInvocationIndex; row < size; row += 64u) {
      var value = covariance(point(row), point(column));
      for (var component = 0u; component < column; component++) { value -= factor(row, component) * factor(column, component); }
      let diagonal = factor(column, column);
      if (diagonal == 0.0) {
        if (abs(value) > tolerance) { atomicStore(&failed, 1u); }
        else { writeFactor(row, column, 0.0); }
      } else {
        let loading = value / diagonal;
        if (!isFiniteValue(loading)) { atomicStore(&failed, 1u); }
        else { writeFactor(row, column, loading); }
      }
    }
  }
  // Storage factors and the workgroup failure flag have separate memory classes.
  storageBarrier();
  workgroupBarrier();
}
if (localInvocationIndex == 0u && atomicLoad(&failed) != 0u && scratch[scratchOffset] == 0.0) {
  scratch[scratchOffset] = 2.0;
}
${d?`if (localInvocationIndex == 0u && scratch[scratchOffset] == 0.0) {
preparationState[preparationStateOffset + 6u] = 1u;
preparationState[preparationStateOffset] = preparationControl[preparationControlOffset];
}`:``}
`,!0),F(`conditional-moments`,`${d?`if (preparationState[preparationStateOffset + 3u] == 0u) { return; }`:``}
if (scratch[scratchOffset] != 0.0) { return; }
${d?`preparationState[preparationStateOffset + 5u]++;`:``}
scratch[scratchOffset] = 2.0;
let observedCount = u32(scratch[scratchOffset + 1u]);
for (var row = 0u; row < observedCount; row++) {
  var value = scratch[scratchOffset + ${k}u + row];
  for (var column = 0u; column < row; column++) { value -= factor(row, column) * scratch[scratchOffset + ${M}u + column]; }
  let whitened = value / factor(row, row);
  if (!isFiniteValue(whitened)) { return; }
  scratch[scratchOffset + ${M}u + row] = whitened;
}
var expectations: array<f32, ${o}>;
var variances: array<f32, ${o}>;
for (var query = 0u; query < ${o}u; query++) {
  let row = observedCount + query; var expectation = ${D}; var variance = 0.0;
  for (var column = 0u; column < observedCount; column++) { expectation += factor(row, column) * scratch[scratchOffset + ${M}u + column]; }
  for (var component = 0u; component <= query; component++) { let loading = factor(row, observedCount + component); variance += loading * loading; }
  if (!isFiniteValue(expectation) || !isFiniteValue(variance)) { return; }
  expectations[query] = expectation; variances[query] = variance;
}
for (var query = 0u; query < ${o}u; query++) {
  moments[momentsOffset + query * 2u] = expectations[query]; moments[momentsOffset + query * 2u + 1u] = variances[query];
}
scratch[scratchOffset] = 0.0;
${d?`preparationState[preparationStateOffset + 1u] = preparationControl[preparationControlOffset + 1u];`:``}`),m(e,{id:t,operation:`GPUConditionalGaussianSimulation`,variant:`joint-realizations`,invocationCount:n.replicateCount,workgroupSize:32,bindings:[{name:`scratch`,view:f,type:`f32`,access:`read`},{name:`moments`,view:y,type:`f32`,access:`read`},{name:`values`,view:n.output.values,type:`f32`,access:`read_write`},{name:`statuses`,view:n.output.status,type:`u32`,access:`read_write`},...S?[{name:`preparationControl`,view:S.view,type:`u32`,access:`read`},{name:`outputMoments`,view:n.output.moments,type:`f32`,access:`read_write`}]:[]],declarations:`${h}\n${g}\n${_}`,body:`
${d?`
if (index == 0u) {
  for (var query = 0u; query < ${o}u; query++) {
    outputMoments[outputMomentsOffset + query * 2u] = moments[momentsOffset + query * 2u];
    outputMoments[outputMomentsOffset + query * 2u + 1u] = moments[momentsOffset + query * 2u + 1u];
  }
}
if (index >= preparationControl[preparationControlOffset + 5u]) {
  statuses[statusesOffset + index] = 1u;
  for (var query = 0u; query < ${o}u; query++) { values[valuesOffset + index * ${o}u + query] = getNaN(); }
  return;
}
`:``}
statuses[statusesOffset + index] = u32(scratch[scratchOffset]);
if (scratch[scratchOffset] != 0.0) {
  for (var query = 0u; query < ${o}u; query++) { values[valuesOffset + index * ${o}u + query] = getNaN(); }
  return;
}
let observedCount = u32(scratch[scratchOffset + 1u]);
var normal: array<f32, ${o}>;
for (var query = 0u; query < ${o}u; query++) {
  let random = philox4x32(vec4<u32>(index + ${d?`preparationControl[preparationControlOffset + 4u]`:`${n.replicateOffset??0}u`}, query, 0u, 0u), vec2<u32>(${d?`preparationControl[preparationControlOffset + 3u]`:`${n.seed}u`}, ${v.conditionalGaussianSimulation.id}u));
  // Box-Muller. The radius uniform is open, (0, 1): the 24-bit form rounds to exactly 1.0 for the
  // largest word, which made that draw exactly zero. The blocked path draws the same way.
  normal[query] = sqrt(-2.0 * log(philoxOpenUnitFloat(random.x))) * cos(6.28318530718 * (f32(random.y >> 8u) + 0.5) / 16777216.0);
}
for (var query = 0u; query < ${o}u; query++) {
  var realization = moments[momentsOffset + query * 2u];
  for (var component = 0u; component <= query; component++) { realization += scratch[scratchOffset + 2u + (observedCount + query) * ${i}u + observedCount + component] * normal[component]; }
  values[valuesOffset + index * ${o}u + query] = realization;
  if (!isFiniteValue(realization)) { statuses[statusesOffset + index] = 2u; }
}
if (statuses[statusesOffset + index] != 0u) {
  for (var query = 0u; query < ${o}u; query++) { values[valuesOffset + index * ${o}u + query] = getNaN(); }
}`})]}},Y=16,X=M({parameters:P,caption:`One simulated field that honors the samples; switch to the mean or spread of 16 draws.`,createInputs:({random:e,world:t})=>R(e,t,28,40),build(e,{grid:t,samples:n,observations:r},i){let a=t.columns*t.rows,o=e.allocate(`realizations`,`float32`,a*Y),s=e.allocate(`moments`,`float32x2`,a);e.graph.add(new J({positions:e.upload(`positions`,`float32x2`,n.positions).view,observations:e.upload(`observations`,`float32x4`,r).view,queries:e.upload(`queries`,`float32x2`,t.centers).view,covariance:{model:`exponential`,range:i.range,sill:.06,nugget:0},mean:.5,seed:i.seed,replicateCount:Y,output:{values:o.view,moments:s.view,status:e.allocate(`status`,`uint32`,Y).view}}));let{selected:c,index:l}=F(e,o,a,`realization`);return{update:e=>l.write(Uint32Array.of(e.replicate)),outputs:{realization:c,mean:I(e,s,0,`mean`,2),spread:I(e,s,1,`spread`,2),positions:e.upload(`sample-positions`,`float32x2`,n.positions),sampleValues:e.upload(`sample-values`,`float32`,n.values)}}},rebuildOn:[`seed`,`range`],interaction:N(P,{kind:`time`,hint:`play the replicates`,x:`replicate`,decorate:e=>({readout:`replicate ${e.replicate+1} of ${Y} · seed ${e.seed}`})}),draw:({inputs:e,outputs:t,values:n})=>[{layer:`raster`,values:[t.realization,t.mean,t.spread][n.output],columns:e.grid.columns,rows:e.grid.rows,range:n.output===2?`auto`:[0,1],smoothing:!0},...L(t.positions,t.sampleValues)]});export{X as default};