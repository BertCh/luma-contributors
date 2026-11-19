import{n as e,t}from"./gpu-command-dispatch-metadata-Bpi5I1Ez.js";import{l as n,r}from"./gpu-command-graph-BdzdmV6u.js";import{c as i,d as a,o,r as s,t as c,u as l}from"./kernel-C3tfQOn4.js";import{n as u,t as d}from"./gpu-dispatch-utils-CbRzFAgT.js";import{n as f,r as p,t as m}from"./gpu-subgroup-utils-QYBHgloD.js";var h=256,g=256,_=16,v=Uint32Array.BYTES_PER_ELEMENT,y=class{id;keys;values;output;mask;operation;constructor(e){this.id=e.id??`gpu-group-aggregation`,this.keys=e.keys,this.values=e.values,this.output=e.output,this.mask=e.mask,this.operation=e.operation??`count`;for(let[e,t]of S(this.keys).entries())b(t,`uint32`,`${this.id} keys chunk ${e}`);if(this.output.length===0)throw Error(`${this.id} output must contain at least one group`);if(![`count`,`sum`,`min`,`max`,`mean`].includes(this.operation))throw Error(`${this.id} operation must be count, sum, min, max, or mean`);if(this.operation===`count`){if(l(this.output,`${this.id} count output`),this.values)throw Error(`${this.id} count operation must not provide values`)}else{if(a(this.output,[`float32`],`${this.id} statistic output`),!this.values)throw Error(`${this.id} ${this.operation} operation requires values`);for(let[e,t]of C(this.values).entries())b(t,`float32`,`${this.id} values chunk ${e}`);if(w(this.keys,this.values,`${this.id} keys and values`),this.operation===`mean`&&this.keys.length>4294967295)throw Error(`${this.id} mean input length must fit in uint32 group counts`)}if(S(this.keys).some(e=>e.buffer===this.output.buffer))throw Error(`${this.id} keys and output must use separate buffers`);if(this.values&&C(this.values).some(e=>e.buffer===this.output.buffer))throw Error(`${this.id} values and output must use separate buffers`);if(this.mask){for(let[e,t]of S(this.mask).entries())if(b(t,`uint32`,`${this.id} mask chunk ${e}`),t.buffer===this.output.buffer)throw Error(`${this.id} mask and output must use separate buffers`);w(this.keys,this.mask,`${this.id} keys and mask`)}}getCommandNodes(t){let r=[],i=S(this.keys),a=this.mask?S(this.mask):void 0,o=this.values?C(this.values):void 0;if(i.some(e=>e.buffer.graph!==t)||a?.some(e=>e.buffer.graph!==t)||o?.some(e=>e.buffer.graph!==t)||this.output.buffer.graph!==t)throw Error(`${this.id} views must belong to the target graph`);if(this.values){let n=e(t,[this.keys,this.values,this.mask??this.keys]);i=n.map(([e])=>e),o=n.map(([,e])=>e),a=this.mask?n.map(([,,e])=>e):void 0}else if(this.mask){let n=e(t,[this.keys,this.mask]);i=n.map(([e])=>e),a=n.map(([,e])=>e)}if(this.operation===`count`){let e=this.output;r.push(...T(t,this.id,e));let o=e.length<=g?`local`:`global`;for(let s=0;s<i.length;s++){let c=i[s];c.length!==0&&r.push(...E(t,{id:this.keys instanceof n||i.length>1?`${this.id}-chunk-${s}-${o}`:`${this.id}-${o}`,keys:c,output:e,mask:a?.[s],dispatchLayout:N(c.length,t.device.limits.maxComputeWorkgroupsPerDimension)}))}return r}let c=this.output,l=this.operation,u=l===`mean`?s(t,`${this.id}-counts`,`uint32`,c.length):void 0;r.push(...D(t,this.id,c,l,u));for(let e=0;e<i.length;e++){let s=i[e];s.length!==0&&r.push(...O(t,{id:this.keys instanceof n||i.length>1?`${this.id}-chunk-${e}`:this.id,keys:s,values:o[e],mask:a?.[e],output:c,operation:l,counts:u,dispatchLayout:N(s.length,t.device.limits.maxComputeWorkgroupsPerDimension)}))}return l!==`sum`&&r.push(...k(t,this.id,c,l,u)),r}};function b(e,t,n){if(e.format!==t||e.rowByteLength!==v||e.byteStride<v||e.byteStride%v!==0||e.byteOffset%v!==0)throw Error(`${n} must be a uint32-aligned scalar ${t} GPU data view`)}function x(e){return e.byteStride/v}function S(e){return e instanceof n?e.data:[e]}function C(e){return e instanceof n?e.data:[e]}function w(e,t,n){if(e.length!==t.length)throw Error(`${n} lengths must match`)}function T(e,t,n){let r=[],a=N(n.length,e.device.limits.maxComputeWorkgroupsPerDimension),o=`
const GROUP_COUNT: u32 = ${n.length}u;
const OUTPUT_OFFSET: u32 = ${i(n)}u;
@group(0) @binding(0) var<storage, read_write> outputCounts: array<atomic<u32>>;

@compute @workgroup_size(${h})
fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${u(a,h)}
  if (index < GROUP_COUNT) {
    atomicStore(&outputCounts[OUTPUT_OFFSET + index], 0u);
  }
}`;return r.push(...P(e,{id:`${t}-clear`,source:o,resources:[{buffer:n,usage:`storage-write`}],bindings:{outputCounts:n},dispatchSize:a})),r}function E(e,t){let n=[],r=t.output.length<=g,a=r&&t.output.length<=_&&m(e.device)===`subgroups`,o=t.mask?`@group(0) @binding(1) var<storage, read> selectionMask: array<u32>;`:``,s=t.mask?2:1,c=t.mask?`selectionMask[${i(t.mask)}u + index * ${x(t.mask)}u] != 0u`:`true`,l=a?p(`accepted`,`groupIndex`,`localCounts`,t.output.length):`  if (accepted) { atomicAdd(&localCounts[groupIndex], 1u); }`,u=r?`${l}
  workgroupBarrier();
  if (lane < GROUP_COUNT) {
    atomicAdd(&outputCounts[OUTPUT_OFFSET + lane], atomicLoad(&localCounts[lane]));
  }`:`if (accepted) { atomicAdd(&outputCounts[OUTPUT_OFFSET + groupIndex], 1u); }`,d=`
${a?`enable subgroups;`:``}
const ELEMENT_COUNT: u32 = ${t.keys.length}u;
const GROUP_COUNT: u32 = ${t.output.length}u;
const KEYS_OFFSET: u32 = ${i(t.keys)}u;
const KEYS_STRIDE: u32 = ${x(t.keys)}u;
const OUTPUT_OFFSET: u32 = ${i(t.output)}u;
@group(0) @binding(0) var<storage, read> groupKeys: array<u32>;
${o}
@group(0) @binding(${s}) var<storage, read_write> outputCounts: array<atomic<u32>>;
${r?`var<workgroup> localCounts: array<atomic<u32>, ${t.output.length}>;`:``}
${a?f():``}

@compute @workgroup_size(${h}) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_id) localId: vec3<u32>${a?`,
  @builtin(subgroup_invocation_id) subgroupInvocationId: u32`:``}
) {
  let workgroupIndex = (workgroupId.z * ${t.dispatchLayout.y}u + workgroupId.y) * ${t.dispatchLayout.x}u + workgroupId.x;
  let index = workgroupIndex * ${h}u + localId.x;
  let lane = localId.x;
  ${r?`if (lane < GROUP_COUNT) { atomicStore(&localCounts[lane], 0u); }
  workgroupBarrier();`:``}
  var accepted = false;
  var groupIndex = 0u;
  if (index < ELEMENT_COUNT && ${c}) {
    groupIndex = groupKeys[KEYS_OFFSET + index * KEYS_STRIDE];
    accepted = groupIndex < GROUP_COUNT;
  }
  ${u}
}`,v=[{buffer:t.keys,usage:`storage-read`},...t.mask?[{buffer:t.mask,usage:`storage-read`}]:[],{buffer:t.output,usage:`storage-read-write`}];return n.push(...P(e,{id:t.id,source:d,resources:v,bindings:{groupKeys:t.keys,...t.mask?{selectionMask:t.mask}:{},outputCounts:t.output},dispatchSize:t.dispatchLayout})),n}function D(e,t,n,r,a){let o=[],s=N(n.length,e.device.limits.maxComputeWorkgroupsPerDimension),c=r===`min`?`0xffffffffu`:`0u`,l=a?`@group(0) @binding(1) var<storage, read_write> outputCounts: array<atomic<u32>>;`:``,d=a?`atomicStore(&outputCounts[${i(a)}u + index], 0u);`:``,f=`
const GROUP_COUNT: u32 = ${n.length}u;
const OUTPUT_OFFSET: u32 = ${i(n)}u;
@group(0) @binding(0) var<storage, read_write> outputValues: array<atomic<u32>>;
${l}
@compute @workgroup_size(${h}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${u(s,h)}
  if (index < GROUP_COUNT) {
    atomicStore(&outputValues[OUTPUT_OFFSET + index], ${c});
    ${d}
  }
}`;return o.push(...P(e,{id:r===`sum`?`${t}-clear`:`${t}-initialize`,source:f,resources:[{buffer:n,usage:`storage-write`},...a?[{buffer:a,usage:`storage-write`}]:[]],bindings:{outputValues:n,...a?{outputCounts:a}:{}},dispatchSize:s})),o}function O(e,t){let n=[],r=t.output.length<=_&&m(e.device)===`subgroups`,a=t.mask?`@group(0) @binding(2) var<storage, read> selectionMask: array<u32>;`:``,o=t.mask?3:2,s=t.counts?`@group(0) @binding(${o+1}) var<storage, read_write> outputCounts: array<atomic<u32>>;`:``,c=t.mask?`selectionMask[${i(t.mask)}u + index * ${x(t.mask)}u] != 0u`:`true`,l=r?M(t.operation,t.output.length,t.counts):`  if (accepted) {
    ${j(t.operation,`groupIndex`)}
    ${t.counts?`atomicAdd(&outputCounts[${i(t.counts)}u + groupIndex], 1u);`:``}
  }`,u=`
${r?`enable subgroups;`:``}
const ELEMENT_COUNT: u32 = ${t.keys.length}u;
const GROUP_COUNT: u32 = ${t.output.length}u;
const KEYS_OFFSET: u32 = ${i(t.keys)}u;
const VALUES_OFFSET: u32 = ${i(t.values)}u;
const KEYS_STRIDE: u32 = ${x(t.keys)}u;
const VALUES_STRIDE: u32 = ${x(t.values)}u;
const OUTPUT_OFFSET: u32 = ${i(t.output)}u;
@group(0) @binding(0) var<storage, read> groupKeys: array<u32>;
@group(0) @binding(1) var<storage, read> inputValues: array<f32>;
${a}
@group(0) @binding(${o}) var<storage, read_write> outputValues: array<atomic<u32>>;
${s}

${A(t.operation)}
${r?f():``}

@compute @workgroup_size(${h}) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_id) localId: vec3<u32>${r?`,
  @builtin(subgroup_invocation_id) subgroupInvocationId: u32`:``}
) {
  let workgroupIndex = (workgroupId.z * ${t.dispatchLayout.y}u + workgroupId.y) * ${t.dispatchLayout.x}u + workgroupId.x;
  let index = workgroupIndex * ${h}u + localId.x;
  var accepted = false;
  var groupIndex = 0u;
  var value = 0.0;
  if (index < ELEMENT_COUNT && ${c}) {
    groupIndex = groupKeys[KEYS_OFFSET + index * KEYS_STRIDE];
    value = inputValues[VALUES_OFFSET + index * VALUES_STRIDE];
    let finiteValue = value == value && abs(value) <= 3.402823466e+38;
    accepted = groupIndex < GROUP_COUNT && finiteValue;
  }
${l}
}`,d=[{buffer:t.keys,usage:`storage-read`},{buffer:t.values,usage:`storage-read`},...t.mask?[{buffer:t.mask,usage:`storage-read`}]:[],{buffer:t.output,usage:`storage-read-write`},...t.counts?[{buffer:t.counts,usage:`storage-read-write`}]:[]];return n.push(...P(e,{id:`${t.id}-${t.operation}`,source:u,resources:d,bindings:{groupKeys:t.keys,inputValues:t.values,...t.mask?{selectionMask:t.mask}:{},outputValues:t.output,...t.counts?{outputCounts:t.counts}:{}},dispatchSize:t.dispatchLayout})),n}function k(e,t,n,r,a){let o=[],s=N(n.length,e.device.limits.maxComputeWorkgroupsPerDimension),c=a?`@group(0) @binding(1) var<storage, read> outputCounts: array<u32>;`:``,l=r===`mean`?`let count = outputCounts[${i(a)}u + index];
    if (count == 0u) {
      outputValues[OUTPUT_OFFSET + index] = 0x7fc00000u;
    } else {
      let sum = bitcast<f32>(outputValues[OUTPUT_OFFSET + index]);
      outputValues[OUTPUT_OFFSET + index] = bitcast<u32>(sum / f32(count));
    }`:`let orderedValue = outputValues[OUTPUT_OFFSET + index];
    if (orderedValue == ${r===`min`?`0xffffffffu`:`0u`}) {
      outputValues[OUTPUT_OFFSET + index] = 0x7fc00000u;
    } else {
      outputValues[OUTPUT_OFFSET + index] = bitcast<u32>(decodeOrderedFloat(orderedValue));
    }`,d=r===`mean`?``:`fn decodeOrderedFloat(value: u32) -> f32 {
  let bits = select(~value, value ^ 0x80000000u, (value & 0x80000000u) != 0u);
  return bitcast<f32>(bits);
}`,f=`
const GROUP_COUNT: u32 = ${n.length}u;
const OUTPUT_OFFSET: u32 = ${i(n)}u;
@group(0) @binding(0) var<storage, read_write> outputValues: array<u32>;
${c}
${d}
@compute @workgroup_size(${h}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${u(s,h)}
  if (index < GROUP_COUNT) {
    ${l}
  }
}`;return o.push(...P(e,{id:`${t}-finalize`,source:f,resources:[{buffer:n,usage:`storage-read-write`},...a?[{buffer:a,usage:`storage-read`}]:[]],bindings:{outputValues:n,...a?{outputCounts:a}:{}},dispatchSize:s})),o}function A(e){return e===`min`||e===`max`?`fn encodeOrderedFloat(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}`:`fn atomicAddFloat(destination: ptr<storage, atomic<u32>, read_write>, value: f32) {
  var oldBits = atomicLoad(destination);
  loop {
    let newBits = bitcast<u32>(bitcast<f32>(oldBits) + value);
    let result = atomicCompareExchangeWeak(destination, oldBits, newBits);
    if (result.exchanged) { break; }
    oldBits = result.old_value;
  }
}`}function j(e,t,n=`value`){return e===`min`||e===`max`?`${e===`min`?`atomicMin`:`atomicMax`}(&outputValues[OUTPUT_OFFSET + ${t}], encodeOrderedFloat(${n}));`:`atomicAddFloat(&outputValues[OUTPUT_OFFSET + ${t}], ${n});`}function M(e,t,n){let r=e===`min`||e===`max`;return`
  var subgroupPending = accepted;
  for (var subgroupGroup = 0u; subgroupGroup < ${t}u; subgroupGroup++) {
    let pendingBallot = subgroupBallot(subgroupPending);
    let hasPending = any(pendingBallot != vec4<u32>(0u));
    let leaderInvocation = getFirstBallotLane(pendingBallot);
    let leaderKey = subgroupShuffle(groupIndex, leaderInvocation);
    let matchingKey = hasPending && subgroupPending && groupIndex == leaderKey;
    let matchingBallot = subgroupBallot(matchingKey);
    let selectedValue = ${r?`select(${e===`min`?`0xffffffffu`:`0u`}, encodeOrderedFloat(value), matchingKey)`:`select(0.0, value, matchingKey)`};
    let aggregatedValue = ${e===`min`?`subgroupMin(selectedValue)`:e===`max`?`subgroupMax(selectedValue)`:`subgroupAdd(selectedValue)`};
    if (hasPending && subgroupInvocationId == leaderInvocation) {
      ${r?`${e===`min`?`atomicMin`:`atomicMax`}(&outputValues[OUTPUT_OFFSET + leaderKey], aggregatedValue);`:j(e,`leaderKey`,`aggregatedValue`)}
      ${n?`atomicAdd(&outputCounts[${i(n)}u + leaderKey], getBallotLaneCount(matchingBallot));`:``}
    }
    subgroupPending = subgroupPending && !matchingKey;
  }`}function N(e,t){return d(`GPUGroupAggregation`,e,h,t)}function P(e,n){let{x:i,y:a,z:s}=n.dispatchSize??{x:n.dispatchCount??1,y:1,z:1},l=i*a*s,u=r({id:n.id,resources:n.resources,workload:{operation:`GPUGroupAggregation`,commandCount:1,maximumWorkgroupCount:l,maximumInvocationCount:l*h},compile:({device:e})=>{let t=new c(e,{id:n.id,source:n.source,shaderLayout:{bindings:Object.keys(n.bindings).map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:r})=>{let c={};for(let[e,t]of Object.entries(n.bindings))c[e]=o(t,r);t.dispatch(e,{bindings:c,x:i,y:a,z:s})},destroy:()=>t.destroy()}}});return[t(u,[i,a,s])]}export{y as t};