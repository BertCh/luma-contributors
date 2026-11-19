import{r as e}from"./kernel-C3tfQOn4.js";import{t}from"./gpu-reduction-tNgOVQ_C.js";import{o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as r}from"./scaled-quotient-wgsl-UDyhQag6.js";import{c as i,f as a,r as o}from"./wgsl-common-DfyXiYB0.js";import{E as s,T as c,f as l,g as u,p as d,v as f,y as p}from"./gpu-contributor-utils-BIwLQON5.js";import{a as m}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as h}from"./regression-output-dispatches-KKVTWkHP.js";import{t as g}from"./define-run-B-brkCe4.js";import{c as _}from"./pointer-overlay-Qh8qflcA.js";import{n as v,t as y}from"./pick-Bo3t3hjv.js";import{n as b,t as x}from"./forecast-nPuSjw5g.js";import{a as S}from"./surface-CdnX4cOV.js";import{tn as C}from"./entry-preview-D70TdSbK.js";var w=`GPUEnsembleScores`,T=65535,E=67108863,D=4294967294,O=2147483647,k=32,A=64,j=`${i}
${r}
fn getEnsembleWeightedDifference(first: f32, second: f32, weight: f32) -> f32 {
  let difference = first - second;
  if (!isFiniteBits(first) || !isFiniteBits(second)) { return abs(difference); }
  if (isFiniteBits(difference)) { return abs(difference) * weight; }
  // Keep scaling ahead of subtraction even under reassociation. All score weights are in
  // (0, 1], and a finite overflowing difference needs weight <= 0.5 for a finite score.
  let divisor = 1.0 / weight;
  return abs(getScaledQuotient(first, divisor, 1.0, 0)
    - getScaledQuotient(second, divisor, 1.0, 0));
}`,M=class{id;props;estimatedCommandNodeCount=k;constructor(e){this.id=e.id??`ensemble-scores`,this.props=e;let{id:t}=this;d(t,this.estimatedCommandNodeCount);let{output:n,memberCount:r,dimension:i}=e;if(l(t,`memberCount`,r,e.fair?2:1,T),l(t,`dimension`,i,1,E),l(t,`memberCount * dimension`,r*i,1,D),s(e.ensemble,[`float32`],`${t} ensemble`),s(e.observations,[`float32`],`${t} observations`),e.ensemble.length<r*i)throw Error(`${t} ensemble needs memberCount * dimension rows`);if(e.observations.length<i)throw Error(`${t} observations needs dimension rows`);if(!n.crps&&!n.energyScore&&!n.variogramScore)throw Error(`${t} needs crps, energyScore or variogramScore output`);if(n.crps&&(s(n.crps,[`float32`],`${t} output.crps`),n.crps.length<i))throw Error(`${t} output.crps needs dimension rows`);if(n.energyScore&&(s(n.energyScore,[`float32`],`${t} output.energyScore`),n.energyScore.length!==1))throw Error(`${t} output.energyScore must have one row`);if(!!n.variogramScore!=!!e.variogramPairs)throw Error(`${t} output.variogramScore and variogramPairs go together`);if(n.variogramScore&&e.variogramPairs){if(s(n.variogramScore,[`float32`],`${t} output.variogramScore`),s(e.variogramPairs,[`uint32x2`],`${t} variogramPairs`),l(t,`variogramPairs.length`,e.variogramPairs.length,1,O),n.variogramScore.length!==1||e.variogramPairs.length<1)throw Error(`${t} output.variogramScore must have one row and variogramPairs one pair`);if(e.variogramWeights&&(s(e.variogramWeights,[`float32`],`${t} variogramWeights`),e.variogramWeights.length<e.variogramPairs.length))throw Error(`${t} variogramWeights needs one row per pair`)}else if(e.variogramWeights)throw Error(`${t} variogramWeights needs variogramPairs`);let a=Math.fround(e.variogramOrder??.5);u(t,`variogramOrder`,e.variogramOrder??.5,0),c(t,Math.fround(a)>0,`variogramOrder must be positive when represented as float32`),h(t,n,[]),p(t,[n.crps,n.energyScore,n.variogramScore],[e.ensemble,e.observations,e.variogramPairs,e.variogramWeights])}getCommandNodes(r){let{id:i,props:s}=this,{output:c,memberCount:l,dimension:u}=s;m(r,i,{ensemble:s.ensemble,observations:s.observations,variogramPairs:s.variogramPairs,variogramWeights:s.variogramWeights,...c});for(let[e,t]of[[`energy terms`,c.energyScore?l:0],[`variogram terms`,c.variogramScore?s.variogramPairs.length:0]]){if(!t)continue;f(r.device,i,e,t*4,`reduce members or variogram pairs`);let n=t;for(;n>256;)n=Math.ceil(n/256),f(r.device,i,`${e} reduction partials`,n*4,`reduce members or variogram pairs`)}let d=[],p=`${l}u`,h=`${u}u`,g=1/(2*l*(s.fair?l-1:l)),_=(e,n,a)=>{d.push(...new t({id:`${i}-${e}-sum`,input:n,output:a,operation:`sum`}).getCommandNodes(r))};if(c.crps&&d.push(n(r,{id:`${i}-crps`,operation:w,variant:s.fair?`crps-fair`:`crps`,bindings:[{name:`ensemble`,view:s.ensemble,type:`f32`,access:`read`},{name:`observations`,view:s.observations,type:`f32`,access:`read`},{name:`crps`,view:c.crps,type:`f32`,access:`read_write`}],workgroupSize:A,invocationCount:u*A,guardIndex:!1,declarations:`${j}\nvar<workgroup> partials: array<f32, ${A}>;`,body:`let site = index / ${A}u;
  let lane = localInvocationIndex;
  var laneSum = 0.0;
  if (site < ${h}) {
    let observed = observations[observationsOffset + site];
    for (var member = lane; member < ${p}; member += ${A}u) {
      let value = ensemble[ensembleOffset + member * ${h} + site];
      var pairSum = 0.0;
      for (var other = 0u; other < ${p}; other++) {
        pairSum += getEnsembleWeightedDifference(value, ensemble[ensembleOffset + other * ${h} + site], f32(${g}));
      }
      laneSum += getEnsembleWeightedDifference(value, observed, 1.0 / f32(${l})) - pairSum;
    }
  }
  partials[lane] = laneSum;
  workgroupBarrier();
  for (var stride = ${A/2}u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) { partials[lane] = partials[lane] + partials[lane + stride]; }
    workgroupBarrier();
  }
  if (lane == 0u && site < ${h}) { crps[crpsOffset + site] = partials[0]; }`})),c.energyScore){let t=e(r,`${i}-energy-terms`,`float32`,l);d.push(n(r,{id:`${i}-energy-terms`,operation:w,variant:s.fair?`energy-terms-fair`:`energy-terms`,bindings:[{name:`ensemble`,view:s.ensemble,type:`f32`,access:`read`},{name:`observations`,view:s.observations,type:`f32`,access:`read`},{name:`terms`,view:t,type:`f32`,access:`read_write`}],invocationCount:l,declarations:`${j}\n${a}\n${o}
// A raw ratio of near-maximum magnitudes can become multiply-by-zero when Metal
// lowers division to a flushed subnormal reciprocal. Divide bounded mantissas instead.
fn accumulateEnsembleMagnitude(state: vec2<f32>, magnitude: f32) -> vec2<f32> {
  if (isNanValue(magnitude)) { return vec2<f32>(getNaN()); }
  if (magnitude == 0.0) { return state; }
  if (state.x < magnitude) { let ratio = getScaledQuotient(state.x, magnitude, 1.0, 0); return vec2<f32>(magnitude, 1.0 + state.y * ratio * ratio); }
  let ratio = getScaledQuotient(magnitude, state.x, 1.0, 0); return vec2<f32>(state.x, state.y + ratio * ratio);
}`,body:`var observedNorm = vec2<f32>(0.0);
  for (var site = 0u; site < ${h}; site++) {
    let magnitude = getEnsembleWeightedDifference(ensemble[ensembleOffset + index * ${h} + site], observations[observationsOffset + site], 1.0 / f32(${l}));
    observedNorm = accumulateEnsembleMagnitude(observedNorm, magnitude);
  }
  var pairSum = 0.0;
  for (var other = 0u; other < ${p}; other++) {
    var pairNorm = vec2<f32>(0.0);
    for (var site = 0u; site < ${h}; site++) {
      let magnitude = getEnsembleWeightedDifference(ensemble[ensembleOffset + index * ${h} + site], ensemble[ensembleOffset + other * ${h} + site], f32(${g}));
      pairNorm = accumulateEnsembleMagnitude(pairNorm, magnitude);
    }
    pairSum += pairNorm.x * sqrt(pairNorm.y);
  }
  terms[termsOffset + index] = observedNorm.x * sqrt(observedNorm.y) - pairSum;`})),_(`energy`,t,c.energyScore)}if(c.variogramScore&&s.variogramPairs){let t=s.variogramPairs.length,o=Math.fround(s.variogramOrder??.5),u=e(r,`${i}-variogram-terms`,`float32`,t);d.push(n(r,{id:`${i}-variogram-terms`,operation:w,variant:`variogram-terms`,bindings:[{name:`ensemble`,view:s.ensemble,type:`f32`,access:`read`},{name:`observations`,view:s.observations,type:`f32`,access:`read`},{name:`pairs`,view:s.variogramPairs,type:`u32`,access:`read`},...s.variogramWeights?[{name:`weights`,view:s.variogramWeights,type:`f32`,access:`read`}]:[],{name:`terms`,view:u,type:`f32`,access:`read_write`}],invocationCount:t,declarations:`${a}
// |x|^p for x >= 0 with 0^p = 0; NaN stays NaN. The NaN test reads the bits: a float comparison
// with NaN is false under IEEE rules (which sent NaN to the zero branch) and unspecified under
// the finite-math assumption WGSL allows, so neither order of a comparison keeps NaN reliably.
fn powerOfMagnitude(magnitude: f32) -> f32 {
  if (isNanValue(magnitude)) { return magnitude; }
  return select(pow(magnitude, f32(${o})), 0.0, magnitude == 0.0);
}`,body:`let first = pairs[pairsOffset + index * 2u];
  let second = pairs[pairsOffset + index * 2u + 1u];
  if (first >= ${h} || second >= ${h}) {
    terms[termsOffset + index] = bitcast<f32>(0x7fc00000u | (index & 0u));
    return;
  }
  let observed = powerOfMagnitude(abs(observations[observationsOffset + first] - observations[observationsOffset + second]));
  var forecastSum = 0.0;
  for (var member = 0u; member < ${p}; member++) {
    forecastSum += powerOfMagnitude(abs(ensemble[ensembleOffset + member * ${h} + first] - ensemble[ensembleOffset + member * ${h} + second]));
  }
  let difference = observed - forecastSum / f32(${l});
  terms[termsOffset + index] = ${s.variogramWeights?`weights[weightsOffset + index] * `:``}difference * difference;`})),_(`variogram`,u,c.variogramScore)}return d}},N=360,P=32,F=5,I=[-4,-3,-2,-1,0,1,2,3,4];function L({random:e,world:t}){let n=b(e,t,N),r=(n.panel.y0+n.panel.y1)/2,i=e=>n.panel.x0+(Math.max(-5,Math.min(F,e))+F)/10*(n.panel.x1-n.panel.x0),a=new Float32Array((I.length+1)*4);return a.set([i(-5),r,i(F),r]),I.forEach((e,t)=>a.set([i(e),r-.012,i(e),r+.012],(t+1)*4)),{...n,lineY:r,toX:i,axis:a,fields:x(e,n.positions,P,.22),index:v(n.positions),order:Uint32Array.from({length:n.count},(e,t)=>t).sort((e,t)=>n.positions[e*2+1]-n.positions[t*2+1]),probe:n.defaultStation}}function R(e,t,n,r){return e.baseline[r]+t*e.fields[n*e.count+r]}var z=g({parameters:C,caption:`Sample CRPS of Helene day-two rain at Carolina stations; ensemble = day-one rain plus correlated perturbations (scenario).`,createInputs:L,rebuildOn:[`memberCount`,`fair`],build(e,t,n){let r=n.memberCount,i=e.allocate(`ensemble`,`float32`,r*t.count),a=e.allocate(`crps`,`float32`,t.count);return e.graph.add(new M({ensemble:i.view,observations:e.upload(`observations`,`float32`,t.residuals).view,memberCount:r,dimension:t.count,fair:n.fair===1,output:{crps:a.view}})),{outputs:{crps:a,positions:e.upload(`positions`,`float32x2`,t.positions),axis:e.upload(`axis`,`float32x4`,t.axis)},update:e=>{let n=new Float32Array(r*t.count);for(let i=0;i<r;i++)for(let r=0;r<t.count;r++)n[i*t.count+r]=R(t,e.spread,i,r);i.write(n)}}},interaction:{kind:`probe`,hint:`probe a station`,rest:({inputs:e,toUnit:t})=>t([e.positions[e.defaultStation*2],e.positions[e.defaultStation*2+1]]),resolve(e,{inputs:t,outputs:n,values:r,pixelSize:i,peek:a}){if(e.x<t.mapWidth)t.probe=y(t.index,e.x,e.y,t.probe);else{let n=(e.y-t.panel.y0)/(t.panel.y1-t.panel.y0),r=Math.round(Math.min(1,Math.max(0,n))*(t.order.length-1));t.probe=t.order[r]}let o=t.probe,s=[t.positions[o*2],t.positions[o*2+1]],c=[S(t.panel,!0),{shape:`ring`,at:s,radius:8*i}],l=t.lineY,u=Array.from({length:r.memberCount},(e,n)=>R(t,r.spread,n,o)).sort((e,t)=>e-t),d=l+.1,f=.28,p=d+f,m=[t.toX(-5),d];u.forEach((e,n)=>{let r=t.toX(e);c.push({shape:`line`,from:[r,l],to:[r,l+.07]}),m.push(r,d+n/u.length*f,r,d+(n+1)/u.length*f)}),m.push(t.toX(F),p);let h=t.toX(t.residuals[o]);c.push({shape:`polyline`,points:m},{shape:`polyline`,points:[t.toX(-5),d,h,d,h,p,t.toX(F),p],dashed:!0},{shape:`line`,from:[h,l-.06],to:[h,d]},{shape:`ring`,at:[h,l],radius:6*i});let g=a(n.crps),v=0;if(g){for(let e=0;e<t.count;e++)v+=g[e];v/=t.count}return{glyphs:c,spotlight:null,readout:g?`CRPS ${_(g[o],2)} here · mean ${_(v,2)}${r.fair===1?` · fair`:``}`:`station ${o}`}}},draw:({inputs:e,outputs:t})=>[{layer:`lines`,context:!0,segments:t.axis,alpha:.8},{layer:`dots`,positions:t.positions,values:t.crps,ramp:`sequential`,range:[0,2.5],radius:5}]});export{z as default};