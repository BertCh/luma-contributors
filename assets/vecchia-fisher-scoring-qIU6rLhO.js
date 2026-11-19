import{a as e,n as t,r as n,s as r,t as i}from"./vecchia-conditioning-sets-CBo4hgA_.js";import{n as a,r as o,t as s}from"./interpolation-view-validation-pYo9tqh3.js";import{a as c}from"./graph-chunk-dispatch-zBt8k2-o.js";import{D as l,f as u,v as d}from"./gpu-contributor-utils-BIwLQON5.js";import{s as f,v as p}from"./spatial-context-DwZEd4EZ.js";import{t as m}from"./gpu-prediction-utils-DjZFSzGW.js";import{r as h,u as g}from"./wgsl-common-DfyXiYB0.js";import{t as _}from"./variogram-model-l9c6-v8f.js";import{c as v,o as y}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as b}from"./equilibrated-solve-wgsl-f1D0OhNH.js";import{i as x}from"./prediction-covariance-DfKlqgpX.js";import{t as S}from"./define-run-B-brkCe4.js";import{c as C}from"./pointer-overlay-Qh8qflcA.js";import{a as w,i as T,n as E,r as D}from"./rain-BPTWkQeX.js";import{a as O,i as k,n as A,o as j,r as M,s as N,t as P}from"./surface-CdnX4cOV.js";import{qt as F}from"./entry-preview-D70TdSbK.js";var I=`GPUVecchiaFisherScoring`,L=32,R=3,z=Object.freeze({...f,measure:`distance`}),B=class{id;spatialContext;props;rowCount;constructor(t){this.props=t,this.id=t.id??`vecchia-fisher-scoring`,this.spatialContext=p(this.id,t.spatialContext,z);let{id:n}=this,{positions:r,observations:i,neighborhoodOffsets:c,neighborhoods:d,parameters:f,output:m}=t;this.rowCount=r.length,u(n,`neighborCount`,t.neighborCount,1,L),l(n,`observations`,i,this.rowCount),l(n,`neighborhoodOffsets`,c,this.rowCount+1),l(n,`parameters`,f,e),l(n,`output.siteTerms`,m.siteTerms,this.rowCount*10),l(n,`output.siteStatus`,m.siteStatus,this.rowCount),l(n,`output.totals`,m.totals,20),l(n,`output.status`,m.status,3),s(n,m,[{write:{siteTerms:m.siteTerms,siteStatus:m.siteStatus,status:m.status}},{read:{status:m.status},write:{totals:m.totals}}]),a(n,Object.values(m),[r,i,c,d,f]),o(n,[r,i,c,d,f],[m.siteTerms,m.siteStatus,m.totals,m.status])}getCommandNodes(e){let{id:t,props:r,rowCount:i}=this;c(e,t,{positions:r.positions,observations:r.observations,neighborhoodOffsets:r.neighborhoodOffsets,neighborhoods:r.neighborhoods,parameters:r.parameters,...r.output}),d(e.device,t,`fixed-point words`,120,`reduce the likelihood cell count`),d(e.device,t,`fixed-point scale`,4,`use a device with storage bindings`);let{positions:a,observations:o,neighborhoodOffsets:s,neighborhoods:l,parameters:u,output:f,neighborCount:p}=r,S=[{name:`positions`,view:a,type:`f32`,access:`read`},{name:`observations`,view:o,type:`f32`,access:`read`},{name:`offsets`,view:s,type:`u32`,access:`read`},{name:`neighbors`,view:l,type:`u32`,access:`read`},{name:`params`,view:u,type:`f32`,access:`read`},{name:`siteTerms`,view:f.siteTerms,type:`f32`,access:`read_write`},{name:`siteStatus`,view:f.siteStatus,type:`u32`,access:`read_write`},{name:`counters`,view:f.status,type:`atomic<u32>`,access:`read_write`}];return[y(e,{id:`${t}-clear-counts`,operation:I,variant:`clear-counts`,bindings:[{name:`counters`,view:f.status,type:`u32`,access:`read_write`}],invocationCount:3,workgroupSize:4,body:`counters[countersOffset + index] = 0u;`}),y(e,{id:`${t}-site-terms`,operation:I,variant:`fisher-site-terms`,bindings:S,invocationCount:i,workgroupSize:32,declarations:`${m({positions:2,observations:4})}
${h}
${g}
${b(p)}
${x(1,1)}
const NEIGHBOR_COUNT: u32 = ${p}u;
const SAMPLE_COUNT: u32 = ${a.length}u;
const PARAMETER_COUNT: u32 = ${R}u;
const CELL_COUNT: u32 = 10u;
const LOG_TWO_PI: f32 = ${v(Math.log(2*Math.PI))};
const MATERN_3_2_RANGE_FACTOR: f32 = ${v(_)};
var<private> sitePositions: array<vec2<f32>, ${p}>;
var<private> siteValues: array<f32, ${p}>;
var<private> siteErrors: array<f32, ${p}>;
var<private> weightsB: array<f32, ${p}>;
var<private> crossK: array<f32, ${p}>;
var<private> solvedW: array<f32, ${R*p}>;
var<private> rawV: array<f32, ${R*p}>;
var<private> dd: array<f32, ${R}>;
var<private> rd: array<f32, ${R}>;
fn correlationRangeDerivative(modelCode: u32, scaled: f32) -> f32 {
  // -t dR/dt of the effective-range correlation; other codes produce NaN and a status 3.
  if (modelCode == 1u) { let a = 3.0 * scaled; return a * exp(-a); }
  if (modelCode == 3u) { let a = MATERN_3_2_RANGE_FACTOR * scaled; return a * a * exp(-a); }
  return getNaN();
}
fn parameterDerivative(parameterIndex: u32, first: vec2<f32>, second: vec2<f32>) -> f32 {
  let values = covarianceStructures[0];
  let separation = distance(first, second) / values.w;
  let coincident = all(first == second);
  if (parameterIndex == 0u) { return values.z * covarianceCorrelation(u32(values.x), separation); }
  if (parameterIndex == 1u) { return values.z * correlationRangeDerivative(u32(values.x), separation); }
  return select(0.0, values.y, coincident);
}
fn fillSystem(count: u32, covarianceScale: f32) {
  for (var row = 0u; row < count; row++) {
    for (var column = 0u; column < count; column++) {
      matrix[row * SYSTEM_CAPACITY + column] = crossCovariance(sitePositions[row], sitePositions[column], 0u, 0u) / covarianceScale;
    }
    matrix[row * SYSTEM_CAPACITY + row] += siteErrors[row] / covarianceScale;
  }
}`,body:`for (var cell = 0u; cell < CELL_COUNT; cell++) { siteTerms[siteTermsOffset + index * CELL_COUNT + cell] = getNaN(); }
let observation = read_observations(index);
if (!(observation.w > 0.0) || !isFiniteValue(observation.x) || !isFiniteValue(observation.y) || observation.y < 0.0) {
  siteStatus[siteStatusOffset + index] = 1u;
  atomicAdd(&counters[countersOffset + 1u], 1u);
  return;
}
siteStatus[siteStatusOffset + index] = 2u;
if (!loadCovarianceParameters()) { atomicAdd(&counters[countersOffset], 1u); return; }
let covarianceScale = params[paramsOffset];
let sitePosition = read_positions(index);
if (!isFiniteValue(sitePosition.x) || !isFiniteValue(sitePosition.y)) { atomicAdd(&counters[countersOffset], 1u); return; }
var count = 0u;
for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u] && count < NEIGHBOR_COUNT; slot++) {
  let sample = neighbors[neighborsOffset + slot];
  if (sample >= SAMPLE_COUNT) { continue; }
  let neighborObservation = read_observations(sample);
  let neighborPosition = read_positions(sample);
  if (!(neighborObservation.w > 0.0) || !isFiniteValue(neighborObservation.x) || !isFiniteValue(neighborObservation.y) || neighborObservation.y < 0.0 || !isFiniteValue(neighborPosition.x) || !isFiniteValue(neighborPosition.y)) { continue; }
  sitePositions[count] = neighborPosition;
  siteValues[count] = neighborObservation.x;
  siteErrors[count] = neighborObservation.y;
  count++;
}
if (count > 0u) {
  fillSystem(count, covarianceScale);
  for (var row = 0u; row < count; row++) {
    crossK[row] = crossCovariance(sitePositions[row], sitePosition, 0u, 0u);
    right[row] = crossK[row] / covarianceScale;
  }
  if (!solveSystem(count)) { atomicAdd(&counters[countersOffset], 1u); return; }
  for (var row = 0u; row < count; row++) { weightsB[row] = solution[row]; }
}
var conditionalVariance = crossCovariance(sitePosition, sitePosition, 0u, 0u) + observation.y;
var prediction = 0.0;
for (var row = 0u; row < count; row++) {
  conditionalVariance -= weightsB[row] * crossK[row];
  prediction += weightsB[row] * siteValues[row];
}
let residual = observation.x - prediction;
let variance = conditionalVariance;
if (!(variance > 0.0) || !isFiniteValue(variance)) { siteStatus[siteStatusOffset + index] = 3u; atomicAdd(&counters[countersOffset], 1u); return; }
for (var parameter = 0u; parameter < PARAMETER_COUNT; parameter++) {
  var dvariance = parameterDerivative(parameter, sitePosition, sitePosition);
  for (var row = 0u; row < count; row++) {
    let dk = parameterDerivative(parameter, sitePositions[row], sitePosition);
    var vectorV = dk;
    for (var column = 0u; column < count; column++) {
      vectorV -= parameterDerivative(parameter, sitePositions[row], sitePositions[column]) * weightsB[column];
    }
    rawV[parameter * NEIGHBOR_COUNT + row] = vectorV;
    dvariance -= (dk + vectorV) * weightsB[row];
  }
  dd[parameter] = dvariance;
  if (count > 0u) {
    fillSystem(count, covarianceScale);
    for (var row = 0u; row < count; row++) { right[row] = rawV[parameter * NEIGHBOR_COUNT + row] / covarianceScale; }
    if (!solveSystem(count)) { atomicAdd(&counters[countersOffset], 1u); return; }
    var quadraticW = 0.0;
    for (var row = 0u; row < count; row++) {
      solvedW[parameter * NEIGHBOR_COUNT + row] = solution[row];
      quadraticW += solution[row] * siteValues[row];
    }
    rd[parameter] = -quadraticW;
  } else {
    rd[parameter] = 0.0;
  }
}
let standardized = residual * residual / variance;
let logLikelihood = -0.5 * (LOG_TWO_PI + log(variance) + standardized);
var output: array<f32, 10>;
output[0] = logLikelihood;
for (var first = 0u; first < PARAMETER_COUNT; first++) {
  output[1u + first] = -0.5 * (dd[first] / variance) * (1.0 - standardized) - residual * rd[first] / variance;
}
var cell = 1u + PARAMETER_COUNT;
for (var first = 0u; first < PARAMETER_COUNT; first++) {
  for (var second = first; second < PARAMETER_COUNT; second++) {
    var quadratic = 0.0;
    for (var row = 0u; row < count; row++) {
      quadratic += rawV[first * NEIGHBOR_COUNT + row] * solvedW[second * NEIGHBOR_COUNT + row];
    }
    output[cell] = 0.5 * dd[first] * dd[second] / (variance * variance) + quadratic / variance;
    cell += 1u;
  }
}
for (var entry = 0u; entry < CELL_COUNT; entry++) {
  if (!isFiniteValue(output[entry])) { siteStatus[siteStatusOffset + index] = 3u; atomicAdd(&counters[countersOffset], 1u); return; }
}
for (var entry = 0u; entry < CELL_COUNT; entry++) { siteTerms[siteTermsOffset + index * CELL_COUNT + entry] = output[entry]; }
siteStatus[siteStatusOffset + index] = 0u;
atomicAdd(&counters[countersOffset + 2u], 1u);`}),...n(e,{id:t,operation:I,rowCount:i,cellCount:10,terms:f.siteTerms,counters:f.status,totals:f.totals})]}};function V(e,t){let n=t=>e[t*2]+e[t*2+1];return{logLikelihood:t[0]>0?NaN:n(0),score:[n(1),n(2),n(3)],information:[n(4),n(5),n(6),n(7),n(8),n(9)],failedSiteCount:t[0],excludedSiteCount:t[1],usedSiteCount:t[2]}}var H=420,U=10,W=16,G=14,K=[`exponential`,`matern-3/2`],q={parameter:F[0],logarithmic:!0},J={parameter:F[1],logarithmic:!0},Y=10,X=2.5;function Z(e,t,n){let r=K[e.model];return{model:r,range:t,sill:n,nugget:r===`exponential`?Math.max(e.nugget,1e-4):Math.max(e.nugget,.0011*n)}}function Q(e){let t=Z(e,e.range,e.sill).nugget;return t===e.nugget?``:` · nugget ${C(t,4)}`}function $(t,n,r,i){let a=t.allocate(`${n}-parameters`,`float32`,e),o=t.allocate(`${n}-site-terms`,`float32`,r*10),s=t.allocate(`${n}-totals`,`float32`,20),c=t.allocate(`${n}-status`,`uint32`,3);return t.graph.add(new B({id:n,positions:i.positions.view,observations:i.observations.view,neighborhoodOffsets:i.offsets.view,neighborhoods:i.neighbors.view,neighborCount:U,parameters:a.view,output:{siteTerms:o.view,siteStatus:t.allocate(`${n}-site-status`,`uint32`,r).view,totals:s.view,status:c.view}})),{parameterBuffer:a,siteTerms:o,totals:s,status:c}}function ee({random:e,world:n}){let r=D(e,n,H),a=i({positions:r.positions,neighborCount:U,order:t(r.positions)});return{...r,sets:a,segments:E(r.positions,a.offsets,a.neighbors),mesh:P(r.panel,W,G),rangeSamples:k(q,W),sillSamples:k(J,G),surfaceCache:{key:``,values:new Float32Array,maximum:{index:-1,value:NaN}}}}function te(e,t){let n={positions:e.upload(`positions`,`float32x2`,t.positions),observations:e.upload(`observations`,`float32x4`,w(t.residuals)),offsets:e.upload(`offsets`,`uint32`,t.sets.offsets),neighbors:e.upload(`neighbors`,`uint32`,t.sets.neighbors)},i=$(e,`fisher`,t.count,n),a=$(e,`fisher-sweep`,t.count,n);return{outputs:{...i,sweepParameters:a.parameterBuffer,sweepTotals:a.totals,sweepTarget:e.allocate(`sweep-target`,`float32`,448),positions:n.positions,meshPositions:e.upload(`mesh-positions`,`float32x2`,t.mesh.positions),meshIndices:e.upload(`mesh-indices`,`uint32`,t.mesh.indices),meshOwners:e.upload(`mesh-owners`,`uint32`,t.mesh.owners)},update:e=>i.parameterBuffer.write(r(Z(e,e.range,e.sill)))}}function ne(e,t){let n=V(e,t);if(!Number.isFinite(n.logLikelihood))return null;let[r,i,,a]=n.information,[o,s]=n.score,c=r*a-i*i,l=[(a*o-i*s)/c,(r*s-i*o)/c];return l.every(Number.isFinite)?{evaluation:n,step:l}:null}var re=S({parameters:F,caption:`Fisher scoring on the Vecchia likelihood of Helene log rainfall: the arrow is one step from the pointer.`,createInputs:ee,build:te,async afterRun({device:e,compiled:t,inputs:n,outputs:i,values:a,upload:o}){let s=n.surfaceCache,c=`${a.model}|${a.nugget}`;if(s.key!==c){let o=await N({device:e,compiled:t,pointCount:224,prepare:e=>i.sweepParameters.write(r(Z(a,n.rangeSamples[e%W],n.sillSamples[Math.floor(e/W)]))),source:i.sweepTotals,destination:i.sweepTarget,length:2}),l=Float32Array.from({length:224},(e,t)=>o[t*2]+o[t*2+1]);s.key=c,s.values=l,s.maximum=M(l)}let l=Float32Array.from(s.values,e=>Number.isFinite(e)?Math.max(e-s.maximum.value,-40):NaN),u=await i.siteTerms.buffer.readAsync(),d=new Float32Array(u.buffer,u.byteOffset,n.count*10),f=Float32Array.from({length:n.count},(e,t)=>d[t*10]);return{surface:o(`surface`,`float32`,l),terms:o(`terms`,`float32`,f),links:o(`links`,`float32x4`,T(n.segments,n.sets.offsets,f,Y)),maximum:s.maximum}},interaction:A({hint:`start a scoring step`,rectangle:({inputs:e})=>e.panel,x:q,y:J,decorate(e,{inputs:t,outputs:n,peek:r}){let i=j(t.panel,q,J,e.range,e.sill),a=[O(t.panel),{shape:`dot`,at:i}],{maximum:o}=t.surfaceCache;o.index>=0&&a.push({shape:`ring`,at:j(t.panel,q,J,t.rangeSamples[o.index%W],t.sillSamples[Math.floor(o.index/W)]),radius:.03});let s=r(n.totals),c=r(n.status),l=s&&c?ne(s,c):null;if(!l)return{glyphs:a,spotlight:null,readout:`range ${e.range} · sill ${e.sill}${Q(e)}`};let[u,d]=l.step,f=Math.hypot(u,d),p=f>X?X/f:1,m=e.range*Math.exp(d*p),h=e.sill*Math.exp(u*p);return a.push({shape:`arrow`,from:i,to:j(t.panel,q,J,m,h)}),{glyphs:a,spotlight:null,readout:`logL ${C(l.evaluation.logLikelihood,1)}${p<1?` · step clipped`:``} · step to range ${C(e.range*Math.exp(d),2)}, sill ${C(e.sill*Math.exp(u),2)}${Q(e)}`}}}),draw:({inputs:e,outputs:t,extra:n})=>[...n?[{layer:`lines`,context:!0,segments:n.links,alpha:.7},{layer:`dots`,positions:t.positions,values:n.terms,ramp:`sequential`,range:[-3,.5],radius:5},{layer:`polygons`,positions:t.meshPositions,indices:t.meshIndices,owners:t.meshOwners,values:n.surface,triangleCount:e.mesh.triangleCount,ramp:`sequential`,range:[-40,0],bands:12}]:[]]});export{re as default};