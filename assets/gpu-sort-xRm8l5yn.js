import{l as e,r as t}from"./gpu-command-graph-BdzdmV6u.js";import{c as n,i as r,o as i,r as a,t as o,u as s}from"./kernel-C3tfQOn4.js";import{n as c,r as l}from"./gpu-command-dispatch-metadata-Bpi5I1Ez.js";import{t as u}from"./gpu-chunk-utils-DMYEopiS.js";import{n as d,t as f}from"./gpu-dispatch-utils-CbRzFAgT.js";import{n as p,t as m}from"./gpu-scan-B4NNTfF9.js";import{t as h}from"./gpu-subgroup-utils-QYBHgloD.js";function g(e,t,r){let i=[],o=c(e,[t.keys,t.values]),s=c(e,[t.outputKeys,t.outputValues]),l=o.map(([n,o],s)=>{let c=`${t.id}-span-${s}`,l=a(e,`${c}-keys`,`uint32`,n.length),u=a(e,`${c}-values`,`uint32`,n.length),d=a(e,`${c}-ranks`,`uint32`,n.length);return i.push(...T(new w({id:c,keys:n,values:o,outputKeys:l,outputValues:u,algorithm:t.resolvedAlgorithm,direction:t.direction,keyBits:t.keyBits}),e,r)),{keys:l,values:u,ranks:d}}),p=t.resolvedAlgorithm===`radix`&&t.keyBits<32?2**t.keyBits-1:4294967295,m=t.direction===`ascending`?`<`:`>`;for(let[a,o]of l.entries()){let c=f(`GPUSort merge`,o.keys.length,256,r),h=d(c,256);i.push(u(e,{id:`${t.id}-ranks-${a}`,outputs:{ranks:o.ranks},dispatch:c,source:`
@group(0) @binding(0) var<storage, read_write> ranks: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${h}
  if (index < ${o.keys.length}u) { ranks[index] = index; }
}`}));for(let[n,r]of l.entries())n!==a&&i.push(u(e,{id:`${t.id}-rank-${a}-against-${n}`,inputs:{keys:o.keys,otherKeys:r.keys},outputs:{ranks:o.ranks},dispatch:c,source:`
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> otherKeys: array<u32>;
@group(0) @binding(2) var<storage, read_write> ranks: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${h}
  if (index >= ${o.keys.length}u) { return; }
  let key = keys[index] & ${p}u;
  var first = 0u;
  var last = ${r.keys.length}u;
  loop {
    if (first == last) { break; }
    let middle = first + (last - first) / 2u;
    let otherKey = otherKeys[middle] & ${p}u;
    if (otherKey ${m} key${n<a?` || otherKey == key`:``}) { first = middle + 1u; }
    else { last = middle; }
  }
  ranks[index] += first;
}`}));let g=0;for(let[r,[l,d]]of s.entries())i.push(u(e,{id:`${t.id}-publish-${a}-${r}`,inputs:{keys:o.keys,values:o.values,ranks:o.ranks},outputs:{outputKeys:l,outputValues:d},dispatch:c,source:`
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> values: array<u32>;
@group(0) @binding(2) var<storage, read> ranks: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputKeys: array<u32>;
@group(0) @binding(4) var<storage, read_write> outputValues: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${h}
  if (index >= ${o.keys.length}u) { return; }
  let rank = ranks[index];
  if (rank >= ${g}u && rank - ${g}u < ${l.length}u) {
    let localRank = rank - ${g}u;
    outputKeys[${n(l)}u + localRank] = keys[index];
    outputValues[${n(d)}u + localRank] = values[index];
  }
}`})),g+=l.length}return i}var _=256,v=256,y=4,b=v/32,x=4294967295,S=2147483648,C=_,w=class{id;keys;values;outputKeys;outputValues;algorithm;direction;keyBits;resolvedAlgorithm;constructor(e){this.id=e.id??`gpu-sort`,this.keys=e.keys,this.values=e.values,this.outputKeys=e.outputKeys,this.outputValues=e.outputValues,this.algorithm=e.algorithm??`auto`,this.direction=e.direction??`ascending`,this.keyBits=e.keyBits??32;for(let[e,t]of[[`keys`,this.keys],[`values`,this.values],[`outputKeys`,this.outputKeys],[`outputValues`,this.outputValues]])for(let n of l(t))s(n,`${this.id} ${e}`);if(![`auto`,`bitonic`,`radix`].includes(this.algorithm))throw Error(`${this.id} algorithm must be auto, bitonic, or radix`);if(![`ascending`,`descending`].includes(this.direction))throw Error(`${this.id} direction must be ascending or descending`);if(!Number.isInteger(this.keyBits)||this.keyBits<1||this.keyBits>32)throw Error(`${this.id} keyBits must be an integer from 1 to 32`);if(this.values.length!==this.keys.length||this.outputKeys.length!==this.keys.length||this.outputValues.length!==this.keys.length)throw Error(`${this.id} key, value, and output lengths must match`);if(this.keys.length>S)throw Error(`${this.id} supports at most ${S} rows`);D(this),this.resolvedAlgorithm=this.algorithm===`auto`?this.keys.length<=C?`bitonic`:`radix`:this.algorithm}getCommandNodes(e){let t=[];return t.push(...T(this,e,e.device.limits.maxComputeWorkgroupsPerDimension)),t}};function T(t,n,r){for(let e of[t.keys,t.values,t.outputKeys,t.outputValues])for(let r of l(e))if(r.buffer.graph!==n)throw Error(`${t.id} views must belong to the target graph`);return[t.keys,t.values,t.outputKeys,t.outputValues].some(t=>t instanceof e)?g(n,t,r):E(t,n,r)}function E(e,t,n){let r=[];if(e.keys.length===0)return r;if(e.keys.length===1)return r.push(...O(t,e)),r;let i=f(`GPUSort`,e.keys.length,v,n);return e.resolvedAlgorithm===`bitonic`?r.push(...k(t,e,i,n)):r.push(...I(t,e,i,n)),r}function D(e){let t=l(e.outputKeys),n=l(e.outputValues),i=[...l(e.keys),...l(e.values)];if([...t,...n].some(e=>i.some(t=>t.buffer===e.buffer))||t.some(e=>n.some(t=>e.buffer===t.buffer)))throw Error(`${e.id} outputs must use separate buffers from inputs and each other`);for(let i of[t,n])for(let[t,n]of i.entries())if(i.slice(0,t).some(e=>r(e,n)))throw Error(`${e.id} output chunks must not overlap`)}function O(e,t,r=t.keys,i=t.values,a=`copy-pair`,o={x:1,y:1,z:1}){let s=[],c=`
const ELEMENT_COUNT: u32 = ${t.keys.length}u;
const KEYS_OFFSET: u32 = ${n(r)}u;
const VALUES_OFFSET: u32 = ${n(i)}u;
const OUTPUT_KEYS_OFFSET: u32 = ${n(t.outputKeys)}u;
const OUTPUT_VALUES_OFFSET: u32 = ${n(t.outputValues)}u;
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> values: array<u32>;
@group(0) @binding(2) var<storage, read_write> outputKeys: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputValues: array<u32>;

@compute @workgroup_size(${v}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${d(o,v)}
  if (index >= ELEMENT_COUNT) { return; }
  outputKeys[OUTPUT_KEYS_OFFSET + index] = keys[KEYS_OFFSET + index];
  outputValues[OUTPUT_VALUES_OFFSET + index] = values[VALUES_OFFSET + index];
}`;return s.push(...V(e,{id:`${t.id}-${a}`,source:c,resources:[{buffer:r,usage:`storage-read`},{buffer:i,usage:`storage-read`},{buffer:t.outputKeys,usage:`storage-write`},{buffer:t.outputValues,usage:`storage-write`}],bindings:{keys:r,values:i,outputKeys:t.outputKeys,outputValues:t.outputValues},dispatchLayout:o})),s}function k(e,t,n,r){let i=[],o=z(t.keys.length);if(o<=_)return i.push(...A(e,t,o)),i;let s=f(`GPUSort bitonic`,o,_,r),c=a(e,`${t.id}-bitonic-indices-a`,`uint32`,o),l=a(e,`${t.id}-bitonic-indices-b`,`uint32`,o);i.push(...N(e,t,c,o,s));let u=c,d=l;for(let n of B(o))i.push(...P(e,t,u,d,o,n,s)),[u,d]=[d,u];return i.push(...F(e,t,u,n)),i}function A(e,t,r){let i=[],a=t.direction===`descending`,o=h(e.device,{requiresSubgroupId:!0})===`subgroups`,s=`
${o?`enable subgroups;
requires subgroup_id;`:``}
const INVALID_INDEX: u32 = ${x}u;
const LOGICAL_LENGTH: u32 = ${t.keys.length}u;
const PADDED_LENGTH: u32 = ${r}u;
const KEYS_OFFSET: u32 = ${n(t.keys)}u;
const VALUES_OFFSET: u32 = ${n(t.values)}u;
const OUTPUT_KEYS_OFFSET: u32 = ${n(t.outputKeys)}u;
const OUTPUT_VALUES_OFFSET: u32 = ${n(t.outputValues)}u;
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> values: array<u32>;
@group(0) @binding(2) var<storage, read_write> outputKeys: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputValues: array<u32>;
var<workgroup> indices: array<u32, ${r}>;
var<workgroup> cachedKeys: array<u32, ${r}>;

fn comes_before(leftIndex: u32, rightIndex: u32) -> bool {
  let leftValid = leftIndex != INVALID_INDEX && leftIndex < LOGICAL_LENGTH;
  let rightValid = rightIndex != INVALID_INDEX && rightIndex < LOGICAL_LENGTH;
  if (leftValid != rightValid) { return leftValid; }
  if (!leftValid) { return false; }
  let leftKey = cachedKeys[leftIndex];
  let rightKey = cachedKeys[rightIndex];
  if (leftKey == rightKey) { return leftIndex < rightIndex; }
  return ${a?`leftKey > rightKey`:`leftKey < rightKey`};
}

@compute @workgroup_size(${r}) fn main(
  ${o?`@builtin(subgroup_invocation_id) subgroupInvocationId: u32,
  @builtin(subgroup_size) subgroupSize: u32,
  @builtin(subgroup_id) subgroupId: u32`:`@builtin(local_invocation_index) localInvocationIndex: u32`}
) {
${o?M():j()}
}`;return i.push(...V(e,{id:`${t.id}-bitonic-local`,source:s,resources:[{buffer:t.keys,usage:`storage-read`},{buffer:t.values,usage:`storage-read`},{buffer:t.outputKeys,usage:`storage-write`},{buffer:t.outputValues,usage:`storage-write`}],bindings:{keys:t.keys,values:t.values,outputKeys:t.outputKeys,outputValues:t.outputValues},dispatchLayout:{x:1,y:1,z:1}})),i}function j(){return`
  indices[localInvocationIndex] = select(
    INVALID_INDEX,
    localInvocationIndex,
    localInvocationIndex < LOGICAL_LENGTH
  );
  if (localInvocationIndex < LOGICAL_LENGTH) {
    cachedKeys[localInvocationIndex] = keys[KEYS_OFFSET + localInvocationIndex];
  } else {
    cachedKeys[localInvocationIndex] = 0u;
  }
  workgroupBarrier();

  for (var blockWidth = 2u; blockWidth <= PADDED_LENGTH; blockWidth <<= 1u) {
    for (var compareStride = blockWidth >> 1u; compareStride > 0u; compareStride >>= 1u) {
      let partnerIndex = localInvocationIndex ^ compareStride;
      if (partnerIndex > localInvocationIndex) {
        let leftIndex = indices[localInvocationIndex];
        let rightIndex = indices[partnerIndex];
        let ascending = (localInvocationIndex & blockWidth) == 0u;
        let shouldSwap = select(
          comes_before(leftIndex, rightIndex),
          comes_before(rightIndex, leftIndex),
          ascending
        );
        indices[localInvocationIndex] = select(leftIndex, rightIndex, shouldSwap);
        indices[partnerIndex] = select(rightIndex, leftIndex, shouldSwap);
      }
      workgroupBarrier();
    }
  }

  if (localInvocationIndex < LOGICAL_LENGTH) {
    let sourceIndex = indices[localInvocationIndex];
    outputKeys[OUTPUT_KEYS_OFFSET + localInvocationIndex] = cachedKeys[sourceIndex];
    outputValues[OUTPUT_VALUES_OFFSET + localInvocationIndex] = values[VALUES_OFFSET + sourceIndex];
  }`}function M(){return`
  let lane = subgroupId * subgroupSize + subgroupInvocationId;
  var currentIndex = select(INVALID_INDEX, lane, lane < LOGICAL_LENGTH);
  if (lane < LOGICAL_LENGTH) {
    cachedKeys[lane] = keys[KEYS_OFFSET + lane];
  } else {
    cachedKeys[lane] = 0u;
  }
  workgroupBarrier();

  for (var blockWidth = 2u; blockWidth <= PADDED_LENGTH; blockWidth <<= 1u) {
    for (var compareStride = blockWidth >> 1u; compareStride > 0u; compareStride >>= 1u) {
      var partnerIndex = INVALID_INDEX;
      if (compareStride < subgroupSize) {
        partnerIndex = subgroupShuffleXor(currentIndex, compareStride);
      } else {
        indices[lane] = currentIndex;
        workgroupBarrier();
        partnerIndex = indices[lane ^ compareStride];
      }

      let lowerLane = (lane & compareStride) == 0u;
      let leftIndex = select(partnerIndex, currentIndex, lowerLane);
      let rightIndex = select(currentIndex, partnerIndex, lowerLane);
      let ascending = (lane & blockWidth) == 0u;
      let shouldSwap = select(
        comes_before(leftIndex, rightIndex),
        comes_before(rightIndex, leftIndex),
        ascending
      );
      let sortedLeft = select(leftIndex, rightIndex, shouldSwap);
      let sortedRight = select(rightIndex, leftIndex, shouldSwap);
      currentIndex = select(sortedRight, sortedLeft, lowerLane);

      if (compareStride >= subgroupSize) {
        workgroupBarrier();
      }
    }
  }

  if (lane < LOGICAL_LENGTH) {
    outputKeys[OUTPUT_KEYS_OFFSET + lane] = cachedKeys[currentIndex];
    outputValues[OUTPUT_VALUES_OFFSET + lane] = values[VALUES_OFFSET + currentIndex];
  }`}function N(e,t,r,i,a){let o=[],s=`
const INVALID_INDEX: u32 = ${x}u;
const LOGICAL_LENGTH: u32 = ${t.keys.length}u;
const PADDED_LENGTH: u32 = ${i}u;
const INDICES_OFFSET: u32 = ${n(r)}u;
@group(0) @binding(0) var<storage, read_write> indices: array<u32>;

@compute @workgroup_size(${_}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${d(a,_)}
  if (index < PADDED_LENGTH) {
    indices[INDICES_OFFSET + index] = select(INVALID_INDEX, index, index < LOGICAL_LENGTH);
  }
}`;return o.push(...V(e,{id:`${t.id}-bitonic-initialize`,source:s,resources:[{buffer:r,usage:`storage-write`}],bindings:{indices:r},dispatchLayout:a})),o}function P(e,t,r,i,a,o,s){let c=[],l=t.direction===`descending`,u=`
const INVALID_INDEX: u32 = ${x}u;
const LOGICAL_LENGTH: u32 = ${t.keys.length}u;
const PADDED_LENGTH: u32 = ${a}u;
const BLOCK_WIDTH: u32 = ${o.blockWidth}u;
const COMPARE_STRIDE: u32 = ${o.compareStride}u;
const KEYS_OFFSET: u32 = ${n(t.keys)}u;
const INDICES_IN_OFFSET: u32 = ${n(r)}u;
const INDICES_OUT_OFFSET: u32 = ${n(i)}u;
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> indicesIn: array<u32>;
@group(0) @binding(2) var<storage, read_write> indicesOut: array<u32>;

fn is_valid(index: u32) -> bool {
  return index != INVALID_INDEX && index < LOGICAL_LENGTH;
}

fn comes_before(leftIndex: u32, rightIndex: u32) -> bool {
  let leftValid = is_valid(leftIndex);
  let rightValid = is_valid(rightIndex);
  if (leftValid != rightValid) { return leftValid; }
  if (!leftValid) { return false; }
  let leftKey = keys[KEYS_OFFSET + leftIndex];
  let rightKey = keys[KEYS_OFFSET + rightIndex];
  if (leftKey == rightKey) { return leftIndex < rightIndex; }
  return ${l?`leftKey > rightKey`:`leftKey < rightKey`};
}

@compute @workgroup_size(${_}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${d(s,_)}
  if (index >= PADDED_LENGTH) { return; }
  let partnerIndex = index ^ COMPARE_STRIDE;
  if (partnerIndex <= index) { return; }
  let leftIndex = indicesIn[INDICES_IN_OFFSET + index];
  let rightIndex = indicesIn[INDICES_IN_OFFSET + partnerIndex];
  let ascending = (index & BLOCK_WIDTH) == 0u;
  let shouldSwap = select(
    comes_before(leftIndex, rightIndex),
    comes_before(rightIndex, leftIndex),
    ascending
  );
  indicesOut[INDICES_OUT_OFFSET + index] = select(leftIndex, rightIndex, shouldSwap);
  indicesOut[INDICES_OUT_OFFSET + partnerIndex] = select(rightIndex, leftIndex, shouldSwap);
}`;return c.push(...V(e,{id:`${t.id}-bitonic-${o.blockWidth}-${o.compareStride}`,source:u,resources:[{buffer:t.keys,usage:`storage-read`},{buffer:r,usage:`storage-read`},{buffer:i,usage:`storage-write`}],bindings:{keys:t.keys,indicesIn:r,indicesOut:i},dispatchLayout:s})),c}function F(e,t,r,i){let a=[],o=`
const LOGICAL_LENGTH: u32 = ${t.keys.length}u;
const KEYS_OFFSET: u32 = ${n(t.keys)}u;
const VALUES_OFFSET: u32 = ${n(t.values)}u;
const INDICES_OFFSET: u32 = ${n(r)}u;
const OUTPUT_KEYS_OFFSET: u32 = ${n(t.outputKeys)}u;
const OUTPUT_VALUES_OFFSET: u32 = ${n(t.outputValues)}u;
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> values: array<u32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputKeys: array<u32>;
@group(0) @binding(4) var<storage, read_write> outputValues: array<u32>;

@compute @workgroup_size(${_}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${d(i,_)}
  if (index >= LOGICAL_LENGTH) { return; }
  let sourceIndex = indices[INDICES_OFFSET + index];
  outputKeys[OUTPUT_KEYS_OFFSET + index] = keys[KEYS_OFFSET + sourceIndex];
  outputValues[OUTPUT_VALUES_OFFSET + index] = values[VALUES_OFFSET + sourceIndex];
}`;return a.push(...V(e,{id:`${t.id}-bitonic-gather`,source:o,resources:[{buffer:t.keys,usage:`storage-read`},{buffer:t.values,usage:`storage-read`},{buffer:r,usage:`storage-read`},{buffer:t.outputKeys,usage:`storage-write`},{buffer:t.outputValues,usage:`storage-write`}],bindings:{keys:t.keys,values:t.values,indices:r,outputKeys:t.outputKeys,outputValues:t.outputValues},dispatchLayout:i})),a}function I(e,t,n,r){let i=[],o=Math.ceil(t.keyBits/y),s=Math.ceil(t.keys.length/v),c=o>1?a(e,`${t.id}-radix-scratch-keys`,`uint32`,t.keys.length):void 0,l=o>1?a(e,`${t.id}-radix-scratch-values`,`uint32`,t.keys.length):void 0,u=t.keys,d=t.values;for(let f=0;f<o;f++){let h=f*y,g=Math.min(y,t.keyBits-h),_=2**g,v=a(e,`${t.id}-radix-digit-${h}-histogram`,`uint32`,_*s),b=a(e,`${t.id}-radix-digit-${h}-offsets`,`uint32`,_*s),x=(o-f)%2==1,S=x?t.outputKeys:c,C=x?t.outputValues:l;if(!S||!C)throw Error(`${t.id} radix scratch is missing`);i.push(...L(e,t,u,v,h,g,s,n));let w=new m({id:`${t.id}-radix-digit-${h}-scan`,input:v,output:b});i.push(...p(w,e,r)),i.push(...R(e,t,u,d,b,S,C,h,g,s,n)),u=S,d=C}return i}function L(e,t,r,i,a,o,s,c){let l=[],u=2**o,d=t.direction===`descending`,f=`
const ELEMENT_COUNT: u32 = ${t.keys.length}u;
const BIT_OFFSET: u32 = ${a}u;
const BUCKET_COUNT: u32 = ${u}u;
const DIGIT_MASK: u32 = ${u-1}u;
const WORKGROUP_COUNT: u32 = ${s}u;
const KEYS_OFFSET: u32 = ${n(r)}u;
const HISTOGRAM_OFFSET: u32 = ${n(i)}u;
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> histogram: array<u32>;
var<workgroup> digitCounts: array<atomic<u32>, ${u}>;

@compute @workgroup_size(${v}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  let workgroupIndex =
    (workgroupId.z * ${c.y}u + workgroupId.y) * ${c.x}u + workgroupId.x;
  if (workgroupIndex >= WORKGROUP_COUNT) { return; }
  if (localInvocationIndex < BUCKET_COUNT) {
    atomicStore(&digitCounts[localInvocationIndex], 0u);
  }
  workgroupBarrier();

  let index = workgroupIndex * ${v}u + localInvocationIndex;
  if (index < ELEMENT_COUNT) {
    let key = keys[KEYS_OFFSET + index];
    let digit = (key >> BIT_OFFSET) & DIGIT_MASK;
    let bucket = ${d?`DIGIT_MASK - digit`:`digit`};
    atomicAdd(&digitCounts[bucket], 1u);
  }
  workgroupBarrier();

  if (localInvocationIndex < BUCKET_COUNT) {
    histogram[HISTOGRAM_OFFSET + localInvocationIndex * WORKGROUP_COUNT + workgroupIndex] =
      atomicLoad(&digitCounts[localInvocationIndex]);
  }
}`;return l.push(...V(e,{id:`${t.id}-radix-digit-${a}-histogram`,source:f,resources:[{buffer:r,usage:`storage-read`},{buffer:i,usage:`storage-write`}],bindings:{keys:r,histogram:i},dispatchLayout:c})),l}function R(e,t,r,i,a,o,s,c,l,u,d){let f=[],p=2**l,m=t.direction===`descending`,h=`
const ELEMENT_COUNT: u32 = ${t.keys.length}u;
const BIT_OFFSET: u32 = ${c}u;
const DIGIT_MASK: u32 = ${p-1}u;
const WORKGROUP_COUNT: u32 = ${u}u;
const MASK_WORD_COUNT: u32 = ${b}u;
const MASK_COUNT: u32 = ${p*b}u;
const KEYS_OFFSET: u32 = ${n(r)}u;
const VALUES_OFFSET: u32 = ${n(i)}u;
const OFFSETS_OFFSET: u32 = ${n(a)}u;
const OUTPUT_KEYS_OFFSET: u32 = ${n(o)}u;
const OUTPUT_VALUES_OFFSET: u32 = ${n(s)}u;
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> values: array<u32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputKeys: array<u32>;
@group(0) @binding(4) var<storage, read_write> outputValues: array<u32>;
var<workgroup> digitMasks: array<atomic<u32>, ${p*b}>;

@compute @workgroup_size(${v}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  let workgroupIndex =
    (workgroupId.z * ${d.y}u + workgroupId.y) * ${d.x}u + workgroupId.x;
  if (workgroupIndex >= WORKGROUP_COUNT) { return; }
  if (localInvocationIndex < MASK_COUNT) {
    atomicStore(&digitMasks[localInvocationIndex], 0u);
  }
  workgroupBarrier();

  let index = workgroupIndex * ${v}u + localInvocationIndex;
  let valid = index < ELEMENT_COUNT;
  var key = 0u;
  var bucket = 0u;
  if (valid) {
    key = keys[KEYS_OFFSET + index];
    let digit = (key >> BIT_OFFSET) & DIGIT_MASK;
    bucket = ${m?`DIGIT_MASK - digit`:`digit`};
    let wordIndex = localInvocationIndex >> 5u;
    let bitIndex = localInvocationIndex & 31u;
    atomicOr(&digitMasks[bucket * MASK_WORD_COUNT + wordIndex], 1u << bitIndex);
  }
  workgroupBarrier();

  if (index >= ELEMENT_COUNT) { return; }
  let maskBase = bucket * MASK_WORD_COUNT;
  let currentWord = localInvocationIndex >> 5u;
  var localRank = 0u;
  for (var word = 0u; word < currentWord; word++) {
    localRank += countOneBits(atomicLoad(&digitMasks[maskBase + word]));
  }
  let precedingBits = (1u << (localInvocationIndex & 31u)) - 1u;
  localRank += countOneBits(atomicLoad(&digitMasks[maskBase + currentWord]) & precedingBits);
  let bucketOffset = offsets[OFFSETS_OFFSET + bucket * WORKGROUP_COUNT + workgroupIndex];
  let outputIndex = bucketOffset + localRank;
  outputKeys[OUTPUT_KEYS_OFFSET + outputIndex] = key;
  outputValues[OUTPUT_VALUES_OFFSET + outputIndex] = values[VALUES_OFFSET + index];
}`;return f.push(...V(e,{id:`${t.id}-radix-digit-${c}-scatter`,source:h,resources:[{buffer:r,usage:`storage-read`},{buffer:i,usage:`storage-read`},{buffer:a,usage:`storage-read`},{buffer:o,usage:`storage-write`},{buffer:s,usage:`storage-write`}],bindings:{keys:r,values:i,offsets:a,outputKeys:o,outputValues:s},dispatchLayout:d})),f}function z(e){let t=1;for(;t<e;)t*=2;return t}function B(e){let t=[];for(let n=2;n<=e;n*=2)for(let e=n/2;e>=1;e/=2)t.push({blockWidth:n,compareStride:e});return t}function V(e,n){let r=[];return r.push(t({id:n.id,resources:n.resources,compile:({device:e})=>{let t=new o(e,{id:n.id,source:n.source,shaderLayout:{bindings:Object.keys(n.bindings).map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:r})=>{let a={};for(let[e,t]of Object.entries(n.bindings))a[e]=i(t,r);t.dispatch(e,{bindings:a,x:n.dispatchLayout.x,y:n.dispatchLayout.y,z:n.dispatchLayout.z})},destroy:()=>t.destroy()}}})),r}export{w as t};