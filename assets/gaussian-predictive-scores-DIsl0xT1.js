import{r as e}from"./gpu-command-graph-BdzdmV6u.js";import{t}from"./gpu-dispatch-utils-CbRzFAgT.js";import{d as n,o as r,t as i}from"./kernel-C3tfQOn4.js";import{o as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{T as o,c as s,f as c,p as l,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{a as d,i as f,n as p,o as m,r as h,t as g}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as _}from"./define-run-B-brkCe4.js";import{c as v}from"./pointer-overlay-Qh8qflcA.js";import{n as y}from"./forecast-nPuSjw5g.js";import{a as b,i as x,n as S,o as C,r as w,s as T,t as E}from"./surface-CdnX4cOV.js";import{en as D}from"./entry-preview-D70TdSbK.js";var O=256,k=16777216,A=class{id;props;estimatedCommandNodeCount;chunkPlan;chunked;constructor(e){this.id=e.id??`gaussian-predictive-scores`,this.props=e;let{id:t}=this;for(let[r,i]of[[`means`,e.means],[`variances`,e.variances],[`observations`,e.observations],[`output.crps`,e.output.crps]])for(let e of s(i))n(e,[`float32`],`${t} ${r}`);let r=e.means.length;c(t,`means.length`,r,1,g),o(t,e.variances.length>=r&&e.observations.length>=r&&e.output.crps.length>=r,`variances, observations and output.crps need one row per mean`),this.chunked=[e.means,e.variances,e.observations,e.output.crps].some(e=>!(`buffer`in e)),this.chunkPlan=h(t,this.chunked?{means:e.means,variances:e.variances,observations:e.observations,crps:e.output.crps}:{means:e.means}),f(t,{"output.crps":e.output.crps}),this.estimatedCommandNodeCount=1+this.chunkPlan.nonemptyChunkCount,l(t,this.estimatedCommandNodeCount),u(t,[e.output.crps],[e.means,e.variances,e.observations])}getCommandNodes(n){let{id:o,props:c}=this;m(n,o,this.chunkPlan),d(n,o,{means:c.means,variances:c.variances,observations:c.observations,"output.crps":c.output.crps});let l={id:`${o}-crps`,operation:`GPUGaussianPredictiveScores`,variant:`crps-normal`,bindings:[{name:`means`,view:s(c.means)[0],type:`f32`,access:`read`},{name:`variances`,view:s(c.variances)[0],type:`f32`,access:`read`},{name:`observations`,view:s(c.observations)[0],type:`f32`,access:`read`},{name:`crps`,view:s(c.output.crps)[0],type:`f32`,access:`read_write`}],invocationCount:c.means.length,declarations:`const INVERSE_SQRT_PI: f32 = 0.5641895835;
const SQRT_TWO_OVER_PI: f32 = 0.7978845608;
// erf(x) for x >= 0: the positive series below 4, 1 above.
fn errorFunction(x: f32) -> f32 {
  if (x > 4.0) { return 1.0; }
  var term = x;
  var sum = x;
  let doubledSquare = 2.0 * x * x;
  for (var n = 1u; n < 160u; n++) {
    term = term * doubledSquare / f32(2u * n + 1u);
    sum += term;
    if (term < 1.0e-9 * sum) { break; }
  }
  return 1.1283791671 * exp(-x * x) * sum;
}`,body:`let variance = variances[variancesOffset + index];
  let error = observations[observationsOffset + index] - means[meansOffset + index];
  var score = bitcast<f32>(0x7fc00000u | (index & 0u));
  if (variance == 0.0) {
    score = abs(error);
  } else if (variance > 0.0) {
    let deviation = sqrt(variance);
    let absoluteError = abs(error);
    // Match the erf cutoff before division; the ratio may exceed the float32 range.
    if (absoluteError > 5.6568542495 * deviation) {
      score = absoluteError - deviation * INVERSE_SQRT_PI;
    } else {
      let standardizedError = absoluteError / deviation;
      score = absoluteError * errorFunction(standardizedError * 0.7071067812) +
        deviation * (SQRT_TWO_OVER_PI * exp(-0.5 * standardizedError * standardizedError) - INVERSE_SQRT_PI);
    }
  }
  crps[crpsOffset + index] = score;`};if(!this.chunked)return[a(n,l)];let u=this.chunkPlan.chunks.map(e=>t(o,e.length,O,n.device.limits.maxComputeWorkgroupsPerDimension)),f=`
@group(0) @binding(0) var<storage, read> chunkParameters: array<u32>;
${l.bindings.map((e,t)=>`@group(0) @binding(${t+1}) var<storage, ${e.access}> ${e.name}: array<${e.type}>;`).join(`
`)}
${l.declarations}
@compute @workgroup_size(${O})
fn main(@builtin(workgroup_id) workgroupId: vec3u,
        @builtin(num_workgroups) workgroupCount: vec3u,
        @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${l.bindings.map(e=>`_ = &${e.name};`).join(`
  `)}
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${k}u) { return; }
  let index = workgroupIndex * ${O}u + localInvocationIndex;
  let rowCount = chunkParameters[1];
  if (index >= rowCount) { return; }
  ${l.bindings.map(e=>`let ${e.name}Offset = arrayLength(&${e.name}) - rowCount;`).join(`
  `)}
  ${l.body}
}`,{slots:h,nodes:g}=p(n,{id:`${o}-chunks`,plan:this.chunkPlan});return[...g,...this.chunkPlan.chunks.filter(e=>e.length>0).map(t=>{let n=h[t.chunkIndex],a=l.bindings.map(e=>({...e,view:t.views[e.name]}));return e({id:`${o}-crps-${t.chunkIndex}`,workload:{operation:`GPUGaussianPredictiveScores`,variant:`crps-normal`,commandCount:1,maximumInvocationCount:t.length,maximumWorkgroupCount:u[t.chunkIndex].x*u[t.chunkIndex].y*u[t.chunkIndex].z},resources:[{buffer:n,usage:`storage-read`},...a.map(e=>({buffer:e.view,usage:e.access===`read`?`storage-read`:`storage-read-write`}))],compile:({device:e})=>{let o=new i(e,{id:`gaussian-predictive-scores-chunk`,source:f,shaderLayout:{bindings:[{name:`chunkParameters`,group:0,location:0,type:`read-only-storage`},...a.map((e,t)=>({name:e.name,group:0,location:t+1,type:e.access===`read`?`read-only-storage`:`storage`}))]}});return{encode:({computePass:e,getBuffer:i})=>o.dispatch(e,{bindings:{chunkParameters:r(n,i),...Object.fromEntries(a.map(e=>[e.name,r(e.view,i)]))},...u[t.chunkIndex]}),destroy:()=>o.destroy()}}})})]}},j=360,M=16,N=14,P={parameter:D[0],logarithmic:!0},F={parameter:D[1]},I=2;function L({random:e,world:t}){let n=y(e,t,j);return{...n,mesh:E(n.panel,M,N),spreadSamples:x(P,M),biasSamples:x(F,N),surfaceCache:{key:``,values:new Float32Array,minimum:{index:-1,value:NaN}}}}function R(e,t,n,r){let i=new Float32Array(e.count),a=new Float32Array(e.count);for(let o=0;o<e.count;o++){i[o]=e.baseline[o]+n;let s=t*Math.exp(r*e.baseline[o]);a[o]=s*s}return{means:i,variances:a}}function z(e,t,n,r){let i=e.allocate(`${t}-means`,`float32`,n.count),a=e.allocate(`${t}-variances`,`float32`,n.count),o=e.allocate(`${t}-crps`,`float32`,n.count);return e.graph.add(new A({id:t,means:i.view,variances:a.view,observations:r.view,output:{crps:o.view}})),{means:i,variances:a,crps:o}}function B(e,t){let n=e.upload(`observations`,`float32`,t.residuals),r=z(e,`gaussian-scores`,t,n),i=z(e,`gaussian-scores-sweep`,t,n);return{outputs:{crps:r.crps,sweep:i,sweepTarget:e.allocate(`sweep-target`,`float32`,224*t.count),positions:e.upload(`positions`,`float32x2`,t.positions),meshPositions:e.upload(`mesh-positions`,`float32x2`,t.mesh.positions),meshIndices:e.upload(`mesh-indices`,`uint32`,t.mesh.indices),meshOwners:e.upload(`mesh-owners`,`uint32`,t.mesh.owners)},update(e){let n=R(t,e.spread,e.bias,e.wetSpread);r.means.write(n.means),r.variances.write(n.variances)}}}var V=_({parameters:D,caption:`Gaussian CRPS of Helene day-two rain at Carolina stations; mean = day-one rain + bias, sd = spread (scenario). Right: mean CRPS over (spread, bias).`,createInputs:L,build:B,async afterRun({device:e,compiled:t,inputs:n,outputs:r,values:i,upload:a}){let o=n.surfaceCache,s=String(i.wetSpread);if(o.key!==s){let a=await T({device:e,compiled:t,pointCount:224,prepare:e=>{let t=R(n,n.spreadSamples[e%M],n.biasSamples[Math.floor(e/M)],i.wetSpread);r.sweep.means.write(t.means),r.sweep.variances.write(t.variances)},source:r.sweep.crps,destination:r.sweepTarget,length:n.count}),c=Float32Array.from({length:224},(e,t)=>{let r=0;for(let e=0;e<n.count;e++)r+=a[t*n.count+e];return r/n.count});o.key=s,o.values=c;let l=w(Float32Array.from(c,e=>-e));o.minimum={index:l.index,value:-l.value}}return{surface:a(`surface`,`float32`,Float32Array.from(o.values,e=>Number.isFinite(e)?Math.min(e-o.minimum.value,I):NaN))}},interaction:S({hint:`predictive sd and bias`,rectangle:({inputs:e})=>e.panel,x:P,y:F,decorate(e,{inputs:t,outputs:n,peek:r}){let i=[b(t.panel),{shape:`crosshair`,at:C(t.panel,P,F,e.spread,e.bias)}],{minimum:a}=t.surfaceCache;if(a.index>=0){let e=t.spreadSamples[a.index%M],n=t.biasSamples[Math.floor(a.index/M)];i.push({shape:`ring`,at:C(t.panel,P,F,e,n),radius:.03})}let o=r(n.crps),s=0;if(o){for(let e=0;e<t.count;e++)s+=o[e];s/=t.count}return{glyphs:i,spotlight:null,readout:o?`sd ${v(e.spread,1)} · bias ${v(e.bias,1)} · CRPS ${v(s,2)}${a.index>=0?` (best ${v(a.value,2)})`:``}`:`sd ${e.spread} · bias ${e.bias}`}}}),draw:({inputs:e,outputs:t,extra:n})=>[{layer:`dots`,positions:t.positions,values:t.crps,ramp:`sequential`,range:[0,2],radius:5},...n?[{layer:`polygons`,positions:t.meshPositions,indices:t.meshIndices,owners:t.meshOwners,values:n.surface,triangleCount:e.mesh.triangleCount,ramp:`sequential`,range:[0,I],bands:12}]:[]]});export{V as default};