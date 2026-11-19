import{r as e}from"./terrain-binding-preflight-laSEObQE.js";import{i as t,r as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-compaction-B6IzcF0P.js";import{c as i,n as a,o,t as s}from"./wgsl-kernel-nodes-DagJklH2.js";import{c,r as l,s as u}from"./raster-relaxation-Di9wjtxp.js";import{r as d}from"./wgsl-common-DfyXiYB0.js";import{E as f,T as p,f as m,m as h,x as g,y as _}from"./gpu-contributor-utils-BIwLQON5.js";import{o as v,t as y}from"./raster-band-utils-BcTxc8_Y.js";import{t as b}from"./define-run-B-brkCe4.js";import{l as x,s as S,u as C}from"./pointer-overlay-Qh8qflcA.js";import{n as w}from"./elevation-DSYnxAKv.js";import{R as T}from"./entry-preview-D70TdSbK.js";var E=`GPUProfilePeaks`,D=4096,O=1024,k=4294967295;function A(e,t=new Float32Array(1)){if(v(`getGPUProfilePeaksParameterValues`,t,1),Number.isNaN(e.minimumProminence))throw Error(`getGPUProfilePeaksParameterValues minimumProminence must not be NaN`);return t[0]=e.minimumProminence,t}var j=0,M=1,N=2,P=3,F=class{id;props;minimumSide;suppressionRadius;wrap;maximumSuppressionIterations;constructor(e){this.id=e.id??`profile-peaks`,this.props=e,this.minimumSide=e.minimumSide??4,this.suppressionRadius=e.suppressionRadius??Math.max(2,Math.round(e.window/4)),this.wrap=e.wrap??!1,this.maximumSuppressionIterations=e.maximumSuppressionIterations??16;let{id:n}=this,{output:r}=e,i=r.status?.converged,a=r.status?.iterationLimitReached;if(f(e.values,[`float32`],`${n} values`),e.values.length<1)throw Error(`${n} values must contain at least one sample`);let o=e.values.length;if(e.validity&&(f(e.validity,[`uint32`],`${n} validity`),e.validity.length!==o))throw Error(`${n} validity must contain one word per sample`);if(f(e.offsets,[`uint32`],`${n} offsets`),e.offsets.length<1)throw Error(`${n} offsets must contain profileCount + 1 rows`);if(f(e.parameters,[`float32`],`${n} parameters`),e.parameters.length<1)throw Error(`${n} parameters must contain at least 1 value`);if(!Number.isInteger(e.window)||e.window<1||e.window>4096)throw Error(`${n} window must be an integer in [1, ${D}]`);if(m(n,`minimumSide`,this.minimumSide,0,2**53-1),!Number.isFinite(this.suppressionRadius)||this.suppressionRadius<0||this.suppressionRadius>4096)throw Error(`${n} suppressionRadius must be in [0, ${D}]`);if(m(n,`maximumSuppressionIterations`,this.maximumSuppressionIterations,0,O),!r.prominence&&!r.refinedIndex&&!r.refinedValue&&!r.peakMask&&!r.peaks&&!i&&!a)throw Error(`${n} requires at least one output`);for(let[e,t,i]of[[`prominence`,r.prominence,`float32`],[`refinedIndex`,r.refinedIndex,`float32`],[`refinedValue`,r.refinedValue,`float32`],[`peakMask`,r.peakMask,`uint32`]])if(t&&(f(t,[i],`${n} output.${e}`),t.length!==o))throw Error(`${n} output.${e} must contain one value per sample`);if(r.peaks){h(n,r.peaks);let e=r.peaks.status.candidateOverflow;p(n,!e||![r.peaks.ids,r.peaks.status.count,r.peaks.status.overflow,r.peaks.status.requiredCount,r.prominence,r.refinedIndex,r.refinedValue,r.peakMask,i,a].some(n=>n&&t(n,e)),`output.peaks.status.candidateOverflow must not overlap other outputs`)}if(i&&(f(i,[`uint32`],`${n} output.status.converged`),i.length<1))throw Error(`${n} output.status.converged must contain one row`);if(a&&(f(a,[`uint32`],`${n} output.status.iterationLimitReached`),a.length<1))throw Error(`${n} output.status.iterationLimitReached must contain one row`);_(n,[r.prominence,r.refinedIndex,r.refinedValue,r.peakMask,i,a,r.peaks?.ids,r.peaks?.status.count,r.peaks?.status.overflow,r.peaks?.status.candidateOverflow,r.peaks?.status.requiredCount],[e.values,e.validity,e.offsets,e.parameters])}getCommandNodes(t){e(t,this.id,this.props,{"candidate scratch":4*this.props.values.length});let{id:f,props:p,minimumSide:m,suppressionRadius:h,wrap:_,maximumSuppressionIterations:v}=this,{output:b}=p,x=b.status?.converged,S=b.status?.iterationLimitReached;g(f,t,[p.values,p.validity,p.offsets,p.parameters,b.prominence,b.refinedIndex,b.refinedValue,b.peakMask,x,S,b.peaks?.ids,b.peaks?.status.count,b.peaks?.status.overflow,b.peaks?.status.candidateOverflow,b.peaks?.status.requiredCount]);let C=p.values.length,w=p.offsets.length-1,T=n(t,`${f}-candidate-prominence`,`float32`,C),D=n(t,`${f}-candidate-delta`,`float32`,C),O=n(t,`${f}-candidate-value`,`float32`,C),A=n(t,`${f}-candidate-state`,`uint32`,C),F=L(C,w,_),R=[],z=[{name:`sampleValues`,view:p.values,type:`f32`,access:`read`}];p.validity&&z.push({name:`sampleValidity`,view:p.validity,type:`u32`,access:`read`}),z.push({name:`profileOffsets`,view:p.offsets,type:`u32`,access:`read`},{name:`settings`,view:p.parameters,type:`f32`,access:`read`},{name:`candidateProminence`,view:T,type:`f32`,access:`read_write`},{name:`candidateDelta`,view:D,type:`f32`,access:`read_write`},{name:`candidateValue`,view:O,type:`f32`,access:`read_write`},{name:`candidateState`,view:A,type:`u32`,access:`read_write`}),R.push(o(t,{id:`${f}-candidates`,operation:E,variant:`candidates`,bindings:z,invocationCount:C,declarations:`${F}
${y}
const WINDOW: u32 = ${p.window}u;
const MIN_SIDE: u32 = ${m}u;
const STATE_INERT: u32 = ${j}u;
const STATE_UNDECIDED: u32 = ${M}u;
${d}
fn isGap(sample: u32) -> bool {
  ${p.validity?`if (sampleValidity[sampleValidityOffset + sample] == 0u) { return true; }`:``}
  return !isFiniteValue(sampleValues[sampleValuesOffset + sample]);
}
fn writeInert(sample: u32) {
  candidateProminence[candidateProminenceOffset + sample] = getNaN();
  candidateDelta[candidateDeltaOffset + sample] = 0.0;
  candidateValue[candidateValueOffset + sample] = getNaN();
  candidateState[candidateStateOffset + sample] = STATE_INERT;
}
struct Side {
  lowest: f32,
  valid: u32,
}
// Walks away from a candidate; the first higher sample is counted as valid but not as lowest.
fn walkSide(start: u32, count: u32, local: u32, peak: f32, direction: i32) -> Side {
  var side = Side(peak, 0u);
  var limit = WINDOW;
  if (WRAP) { limit = min(WINDOW, count - 1u); }
  for (var step = 1u; step <= limit; step++) {
    let neighbor = neighborLocal(local, direction * i32(step), count);
    if (neighbor < 0) { break; }
    let sample = start + u32(neighbor);
    if (isGap(sample)) { break; }
    side.valid = side.valid + 1u;
    let other = sampleValues[sampleValuesOffset + sample];
    if (other > peak) { break; }
    if (other < side.lowest) { side.lowest = other; }
  }
  return side;
}`,body:`let range = findProfile(index);
  let count = range.y - range.x;
  if (count == 0u) { writeInert(index); return; }
  let start = range.x;
  let local = index - start;
  if (!WRAP && (local < 1u || local + 1u >= count)) { writeInert(index); return; }
  if (isGap(index)) { writeInert(index); return; }
  let peak = sampleValues[sampleValuesOffset + index];
  for (var offset = -2; offset <= 2; offset++) {
    if (offset == 0) { continue; }
    let neighbor = neighborLocal(local, offset, count);
    if (neighbor < 0) { continue; }
    let sample = start + u32(neighbor);
    if (sample == index || isGap(sample)) { continue; }
    let other = sampleValues[sampleValuesOffset + sample];
    if (other > peak || (offset < 0 && other == peak)) { writeInert(index); return; }
  }
  let left = walkSide(start, count, local, peak, -1);
  let right = walkSide(start, count, local, peak, 1);
  if (left.valid < MIN_SIDE || right.valid < MIN_SIDE) { writeInert(index); return; }
  let prominence = peak - max(left.lowest, right.lowest);
  if (prominence < settings[settingsOffset]) { writeInert(index); return; }
  var delta = 0.0;
  var refined = peak;
  let previous = start + u32(neighborLocal(local, -1, count));
  let next = start + u32(neighborLocal(local, 1, count));
  if (!isGap(previous) && !isGap(next)) {
    let a = sampleValues[sampleValuesOffset + previous];
    let c = sampleValues[sampleValuesOffset + next];
    let denominator = a - 2.0 * peak + c;
    if (denominator < 0.0) {
      delta = clamp(0.5 * (a - c) / denominator, -0.5, 0.5);
      refined = peak - 0.25 * (a - c) * delta;
    }
  }
  candidateProminence[candidateProminenceOffset + index] = prominence;
  candidateDelta[candidateDeltaOffset + index] = delta;
  candidateValue[candidateValueOffset + index] = refined;
  candidateState[candidateStateOffset + index] = STATE_UNDECIDED;`}));let B=Math.floor(h)+1;if(v>0){let e=c(t,`${f}-nms`,E,C);R.push(u(t,{id:`${f}-nms-reset`,operation:E,state:e}),...l(t,{id:`${f}-nms`,gateId:`${f}-nms-gate`,operation:E,state:e,maxIterations:v,createRound:({nodeId:n,slotBinding:r,markChangedWGSL:a,condition:s,extraResources:c})=>o(t,{id:n,operation:E,variant:`suppression`,bindings:[{name:`profileOffsets`,view:p.offsets,type:`u32`,access:`read`},{name:`candidateProminence`,view:T,type:`f32`,access:`read`},{name:`candidateDelta`,view:D,type:`f32`,access:`read`},{name:`candidateState`,view:A,type:`atomic<u32>`,access:`read_write`},{name:`status`,view:e.status,type:`atomic<u32>`,access:`read_write`},r],invocationCount:C,condition:s,extraResources:c,declarations:`${F}
const NMS_RADIUS: f32 = ${i(h)};
const SCAN_RADIUS: u32 = ${B}u;
const STATE_UNDECIDED: u32 = ${M}u;
const STATE_KEPT: u32 = ${N}u;
const STATE_SUPPRESSED: u32 = ${P}u;`,body:`if (atomicLoad(&candidateState[candidateStateOffset + index]) != STATE_UNDECIDED) { return; }
  let range = findProfile(index);
  let start = range.x;
  let count = range.y - range.x;
  let local = index - start;
  let prominenceSelf = candidateProminence[candidateProminenceOffset + index];
  let deltaSelf = candidateDelta[candidateDeltaOffset + index];
  var undecided = false;
  for (var step = 1u; step <= SCAN_RADIUS; step++) {
    for (var side = 0; side < 2; side++) {
      let direction = select(-1, 1, side == 1);
      let neighbor = neighborLocal(local, direction * i32(step), count);
      if (neighbor < 0 || u32(neighbor) == local) { continue; }
      let sample = start + u32(neighbor);
      let state = atomicLoad(&candidateState[candidateStateOffset + sample]);
      if (state != STATE_UNDECIDED && state != STATE_KEPT) { continue; }
      let prominenceOther = candidateProminence[candidateProminenceOffset + sample];
      let higher = prominenceOther > prominenceSelf ||
        (prominenceOther == prominenceSelf && u32(neighbor) < local);
      if (!higher) { continue; }
      var distance = abs(f32(neighbor - i32(local)) +
        (candidateDelta[candidateDeltaOffset + sample] - deltaSelf));
      if (WRAP) { distance = min(distance, f32(count) - distance); }
      if (distance > NMS_RADIUS) { continue; }
      if (state == STATE_KEPT) {
        atomicStore(&candidateState[candidateStateOffset + index], STATE_SUPPRESSED);
        ${a}
        return;
      }
      undecided = true;
    }
  }
  if (!undecided) {
    atomicStore(&candidateState[candidateStateOffset + index], STATE_KEPT);
    ${a}
  }`})}))}x&&R.push(s(t,{id:`${f}-converged-reset`,operation:E,view:x,type:`u32`,value:`1u`,componentCount:1}),o(t,{id:`${f}-converged`,operation:E,variant:`converged`,bindings:[{name:`candidateState`,view:A,type:`u32`,access:`read`},{name:`convergedOut`,view:x,type:`atomic<u32>`,access:`read_write`}],invocationCount:C,body:`if (candidateState[candidateStateOffset + index] == ${M}u) {
    atomicStore(&convergedOut[convergedOutOffset], 0u);
  }`})),S&&R.push(s(t,{id:`${f}-iteration-limit-reset`,operation:E,view:S,type:`u32`,value:`0u`,componentCount:1}),o(t,{id:`${f}-iteration-limit`,operation:E,variant:`iteration-limit`,bindings:[{name:`candidateState`,view:A,type:`u32`,access:`read`},{name:`limitOut`,view:S,type:`atomic<u32>`,access:`read_write`}],invocationCount:C,body:`if (candidateState[candidateStateOffset + index] == ${M}u) {
    atomicStore(&limitOut[limitOutOffset], 1u);
  }`}));let V=b.peakMask??(b.peaks?n(t,`${f}-peak-mask`,`uint32`,C):void 0),H=b.peaks&&b.peakMask?n(t,`${f}-peak-flags`,`uint32`,C):V,U=b.peaks?n(t,`${f}-row-ids`,`uint32`,C):void 0;if(b.prominence||b.refinedValue||V){let e=[{name:`candidateState`,view:A,type:`u32`,access:`read`}];b.prominence&&e.push({name:`candidateProminence`,view:T,type:`f32`,access:`read`},{name:`prominenceOut`,view:b.prominence,type:`f32`,access:`read_write`}),b.refinedValue&&e.push({name:`candidateValue`,view:O,type:`f32`,access:`read`},{name:`refinedValueOut`,view:b.refinedValue,type:`f32`,access:`read_write`}),V&&e.push({name:`peakMaskOut`,view:V,type:`u32`,access:`read_write`}),H&&H!==V&&e.push({name:`flagsOut`,view:H,type:`u32`,access:`read_write`}),U&&e.push({name:`rowIdsOut`,view:U,type:`u32`,access:`read_write`});let n=!!b.peakMask;R.push(o(t,{id:`${f}-outputs`,operation:E,variant:`outputs`,bindings:e,invocationCount:C,declarations:`${d}
${I()}`,body:`let state = candidateState[candidateStateOffset + index];
  let kept = state == ${N}u;
  let unresolved = state == ${M}u;
  ${b.prominence?`prominenceOut[prominenceOutOffset + index] = select(select(getNaN(), candidateProminence[candidateProminenceOffset + index], kept), getInfinity(), unresolved);`:``}
  ${b.refinedValue?`refinedValueOut[refinedValueOutOffset + index] = select(select(getNaN(), candidateValue[candidateValueOffset + index], kept), getInfinity(), unresolved);`:``}
  ${V?n?`peakMaskOut[peakMaskOutOffset + index] = select(select(0u, 1u, kept), ${k}u, unresolved);`:`peakMaskOut[peakMaskOutOffset + index] = select(0u, 1u, kept);`:``}
  ${H&&H!==V?`flagsOut[flagsOutOffset + index] = select(0u, 1u, kept);`:``}
  ${U?`rowIdsOut[rowIdsOutOffset + index] = index;`:``}`}))}if(b.refinedIndex&&R.push(o(t,{id:`${f}-refined-index`,operation:E,variant:`refined-index`,bindings:[{name:`profileOffsets`,view:p.offsets,type:`u32`,access:`read`},{name:`candidateDelta`,view:D,type:`f32`,access:`read`},{name:`candidateState`,view:A,type:`u32`,access:`read`},{name:`refinedIndexOut`,view:b.refinedIndex,type:`f32`,access:`read_write`}],invocationCount:C,declarations:`${F}
${d}
${I()}`,body:`let state = candidateState[candidateStateOffset + index];
  if (state != ${N}u) {
    refinedIndexOut[refinedIndexOutOffset + index] =
      select(getNaN(), getInfinity(), state == ${M}u);
    return;
  }
  let range = findProfile(index);
  let count = f32(range.y - range.x);
  var refined = f32(index - range.x) + candidateDelta[candidateDeltaOffset + index];
  if (WRAP) {
    if (refined < 0.0) { refined = refined + count; }
    if (refined >= count) { refined = refined - count; }
  }
  refinedIndexOut[refinedIndexOutOffset + index] = refined;`})),b.peaks&&H&&U){let e=n(t,`${f}-compact-rows`,`uint32`,C),i=n(t,`${f}-peak-total`,`uint32`,1),c=n(t,`${f}-incomplete`,`uint32`,1);R.push(...new r({id:`${f}-compaction`,input:U,flags:H,output:e,count:i}).getCommandNodes(t),a(t,{id:`${f}-publish`,operation:E,requiredCount:i,compactIds:e,output:{ids:b.peaks.ids,status:{count:b.peaks.status.count,overflow:b.peaks.status.overflow,requiredCount:b.peaks.status.requiredCount}}}),s(t,{id:`${f}-incomplete-clear`,operation:E,view:c,type:`u32`,value:`0u`,componentCount:1}),o(t,{id:`${f}-incomplete`,operation:E,variant:`incomplete`,bindings:[{name:`candidateState`,view:A,type:`u32`,access:`read`},{name:`incomplete`,view:c,type:`atomic<u32>`,access:`read_write`}],invocationCount:C,body:`if (candidateState[candidateStateOffset + index] == ${M}u) {
    atomicStore(&incomplete[incompleteOffset], 1u);
  }`}),...b.peaks.status.candidateOverflow?[o(t,{id:`${f}-publish-unconverged`,operation:E,variant:`publish-unconverged`,bindings:[{name:`incomplete`,view:c,type:`u32`,access:`read`},{name:`candidateOverflowOut`,view:b.peaks.status.candidateOverflow,type:`u32`,access:`read_write`}],invocationCount:1,body:`candidateOverflowOut[candidateOverflowOutOffset] =
    select(0u, 1u, incomplete[incompleteOffset] != 0u);`})]:[])}return R}};function I(){return`fn getInfinity() -> f32 {
  var bits = 0x7f800000u;
  return bitcast<f32>(bits);
}`}function L(e,t,n){return`const SAMPLE_COUNT: u32 = ${e}u;
const PROFILE_COUNT: u32 = ${t}u;
const WRAP: bool = ${n};
// Returns [start, end) of the profile owning \`sample\`, or [0, 0) when it belongs to none.
fn findProfile(sample: u32) -> vec2<u32> {
  var low = 0u;
  var high = PROFILE_COUNT;
  for (var step = 0u; step < 32u && low < high; step++) {
    let middle = low + (high - low) / 2u;
    if (profileOffsets[profileOffsetsOffset + middle + 1u] > sample) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }
  if (low >= PROFILE_COUNT) { return vec2<u32>(0u, 0u); }
  let start = profileOffsets[profileOffsetsOffset + low];
  let end = min(profileOffsets[profileOffsetsOffset + low + 1u], SAMPLE_COUNT);
  if (start > sample || sample >= end) { return vec2<u32>(0u, 0u); }
  return vec2<u32>(start, end);
}
// Local index of the neighbour \`delta\` samples away, or -1 when it is outside a non-wrapped profile.
fn neighborLocal(local: u32, delta: i32, count: u32) -> i32 {
  let position = i32(local) + delta;
  let signedCount = i32(count);
  if (WRAP) { return ((position % signedCount) + signedCount) % signedCount; }
  if (position < 0 || position >= signedCount) { return -1; }
  return position;
}`}var R=4,z=96,B=.2,V=.04,H=.04;function U(e,t,n,r,i){let a=V+(3-t)*.24000000000000002,o=(r-i[0])/(i[1]-i[0]);return[H*e+n/95*(1-2*H)*e,a+Math.min(Math.max(o,0),1)*B]}var W=b({parameters:T,rebuildOn:[`window`,`suppressionRadius`],caption:`Four east–west Grand Canyon transects in; their prominent peaks out.`,createInputs:({random:e,world:t})=>{let{elevation:n}=w(z,60),r=Float32Array.from({length:384},(e,t)=>{let r=Math.floor((Math.floor(t/z)+.5)*60/R);return n[r*z+t%z]});return{ranges:Array.from({length:R},(e,t)=>{let n=r.subarray(t*z,(t+1)*z);return[Math.min(...n)-60,Math.max(...n)+60]}),profiles:r,noise:Float32Array.from(r,()=>e()-.5),width:t.width}},build(e,t,n){let r=e.allocate(`values`,`float32`,384),i=e.upload(`offsets`,`uint32`,Uint32Array.from({length:5},(e,t)=>t*z)),a=e.allocate(`parameters`,`float32`,1),o=e.allocate(`peak-mask`,`uint32`,384),s=e.allocate(`refined-index`,`float32`,384);e.graph.add(new F({values:r.view,offsets:i.view,window:n.window,suppressionRadius:n.suppressionRadius,wrap:!1,parameters:a.view,output:{peakMask:o.view,refinedIndex:s.view}}));let c=e.allocate(`profile-lines`,`float32x4`,384);return{outputs:{series:r,peakMask:o,lines:c,peakDots:e.allocate(`peak-dots`,`float32x2`,384)},update:e=>{let n=t.profiles.map((n,r)=>n+e.noise*t.noise[r]);r.write(n);let i=new Float32Array(1536);for(let e=0;e<R;e++)for(let r=0;r<95;r++){let a=e*z+r;i.set([...U(t.width,e,r,n[a],t.ranges[e]),...U(t.width,e,r+1,n[a+1],t.ranges[e])],a*4)}c.write(i),a.write(A({minimumProminence:e.minimumProminence}))}}},interaction:{kind:`level`,hint:`raise the prominence bar`,rest:[.5,1-x(S(T,`minimumProminence`),30)],resolve(e,{inputs:t,outputs:n,values:r,world:i,peek:a}){let o=C(S(T,`minimumProminence`),1-e.unitY),s=a(n.peakMask),c=a(n.series),l=[{shape:`line`,from:[0,e.y],to:[i.width,e.y],dashed:!0}],u=0,d=s?.some(e=>e===4294967295)??!1;if(s&&c){let e=new Float32Array(s.length*2).fill(-10);for(let n=0;n<s.length;n++){if(s[n]!==1)continue;u++;let r=Math.floor(n/z),i=t.ranges[r],[a,d]=U(t.width,r,n%z,c[n],i),f=o/(i[1]-i[0])*B;e.set([a,d],n*2),l.push({shape:`line`,from:[a,d],to:[a,d-f]})}n.peakDots.write(e)}return{values:{minimumProminence:o},glyphs:l,readout:`minimumProminence ${o} m${s&&c?` · ${u} peaks${d?` · incomplete suppression`:``}`:` · peaks unavailable`}${r.noise>0?` · noise ${r.noise} m`:``}`}}},draw:({outputs:e})=>[{layer:`lines`,segments:e.lines,radius:2},{layer:`dots`,positions:e.peakDots,radius:9,color:[.95,.45,.1,1]}]});export{W as default};