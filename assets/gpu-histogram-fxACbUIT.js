import{n as e}from"./gpu-command-dispatch-metadata-Bpi5I1Ez.js";import{l as t,r as n}from"./gpu-command-graph-BdzdmV6u.js";import{c as r,d as i,o as a,r as o,t as s,u as c}from"./kernel-C3tfQOn4.js";import{n as l,t as u}from"./gpu-dispatch-utils-CbRzFAgT.js";import{t as d}from"./gpu-reduction-tNgOVQ_C.js";import{n as f,r as p,t as m}from"./gpu-subgroup-utils-QYBHgloD.js";var h=256,g=256,_=16,v=257,y=Uint32Array.BYTES_PER_ELEMENT,b=[`uint32`,`sint32`,`float32`],x=class{id;input;output;mask;domain;edges;constructor(e){this.id=e.id??`gpu-histogram`,this.input=e.input,this.output=e.output,this.mask=e.mask,this.domain=e.domain,this.edges=e.edges;for(let[e,n]of S(this.input).entries()){let r=this.input instanceof t?` input chunk ${e}`:` input`;if(w(n,`${this.id}${r}`),n.format!==this.input.format)throw Error(`${this.id}${r} format must match the input format`)}if(c(this.output,`${this.id} output`),this.output.length===0)throw Error(`${this.id} output must contain at least one bin`);if(S(this.input).some(e=>e.buffer===this.output.buffer))throw Error(`${this.id} input and output must use separate buffers`);if(this.mask){for(let[e,t]of C(this.mask).entries())if(c(t,`${this.id} mask chunk ${e}`),t.buffer===this.output.buffer)throw Error(`${this.id} mask and output must use separate buffers`);if(this.input.length!==this.mask.length)throw Error(`${this.id} input and mask lengths must match`)}if(this.edges!==void 0)M(this.edges,this.input.format,this.output,this.id);else if(this.domain===`auto`&&S(this.input).some(e=>e.byteStride!==e.rowByteLength))throw Error(`${this.id} automatic domains require packed input; provide an explicit domain`);else if(P(this.domain)){if(i(this.domain,b,`${this.id} domain`),this.domain.format!==this.input.format||this.domain.length!==2)throw Error(`${this.id} GPU domain must contain two ${this.input.format} rows`);if(this.domain.buffer===this.output.buffer)throw Error(`${this.id} domain and output must use separate buffers`)}else Array.isArray(this.domain)&&N(this.domain,this.input.format,this.id)}getCommandNodes(n){let r=[],i=S(this.input),a=this.mask?C(this.mask):void 0;if(i.some(e=>e.buffer.graph!==n)||a?.some(e=>e.buffer.graph!==n)||this.output.buffer.graph!==n)throw Error(`${this.id} views must belong to the target graph`);if(this.mask){let t=e(n,[this.input,this.mask]);i=t.map(([e])=>e),a=t.map(([,e])=>e)}let s=this.domain,c=this.edges;if(F(c)&&c.buffer.graph!==n)throw Error(`${this.id} edges must belong to the target graph`);if(c!==void 0){let e=F(c)?o(n,`${this.id}-edge-validity`,`uint32`,1):void 0;F(c)&&e&&r.push(...D(n,this.id,c,e)),r.push(...E(n,this.id,this.output));let s=this.output.length<=g?`local`:`global`;return i.forEach((o,l)=>{o.length!==0&&r.push(...O(n,{id:this.input instanceof t||i.length>1?`${this.id}-chunk-${l}-edges-${s}`:`${this.id}-edges-${s}`,input:o,output:this.output,mask:a?.[l],edges:c,edgeValidity:e}))}),r}if(P(s)&&s.buffer.graph!==n)throw Error(`${this.id} domain must belong to the target graph`);if(s===`auto`){let e=o(n,`${this.id}-auto-domain`,this.input.format,2);r.push(...new d({id:`${this.id}-extent`,input:this.input,output:e,operation:`extent`}).getCommandNodes(n)),s=e}if(s===void 0)throw Error(`${this.id} requires either domain or edges`);r.push(...E(n,this.id,this.output));let l=this.output.length<=g?`local`:`global`;return i.forEach((e,o)=>{e.length!==0&&r.push(...k(n,{id:this.input instanceof t||i.length>1?`${this.id}-chunk-${o}-${l}`:`${this.id}-${l}`,input:e,output:this.output,mask:a?.[o],domain:s}))}),r}};function S(e){return e instanceof t?e.data:[e]}function C(e){return e instanceof t?e.data:[e]}function w(e,t){if(!b.includes(e.format)||e.rowByteLength!==y||e.byteStride<y||e.byteStride%y!==0||e.byteOffset%y!==0)throw Error(`${t} must be a uint32-aligned 32-bit scalar GPU data view`)}function T(e){return e.byteStride/y}function E(e,t,n){let i=[],a=`${t}-clear`,o=j(e,n.length),s=`
const BIN_COUNT: u32 = ${n.length}u;
const OUTPUT_OFFSET: u32 = ${r(n)}u;
@group(0) @binding(0) var<storage, read_write> outputCounts: array<atomic<u32>>;
@compute @workgroup_size(${h}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${l(o,h)}
  if (index < BIN_COUNT) { atomicStore(&outputCounts[OUTPUT_OFFSET + index], 0u); }
}`;return i.push(...A(e,{id:a,source:s,resources:[{buffer:n,usage:`storage-write`}],bindings:{outputCounts:n},dispatchLayout:o})),i}function D(e,t,n,i){let a=[],o=I(n.format),s=n.format===`float32`?`left == left && right == right && abs(left) <= 3.402823466e+38 && abs(right) <= 3.402823466e+38`:`true`,c=`
const EDGE_COUNT: u32 = ${n.length}u;
const EDGE_OFFSET: u32 = ${r(n)}u;
const VALIDITY_OFFSET: u32 = ${r(i)}u;
@group(0) @binding(0) var<storage, read> edgeValues: array<${o}>;
@group(0) @binding(1) var<storage, read_write> edgeValidity: array<atomic<u32>>;
@compute @workgroup_size(${h}) fn main(
  @builtin(local_invocation_id) localId: vec3<u32>
) {
  let lane = localId.x;
  if (lane == 0u) { atomicStore(&edgeValidity[VALIDITY_OFFSET], 1u); }
  storageBarrier();
  var edgeIndex = lane;
  loop {
    if (edgeIndex + 1u >= EDGE_COUNT) { break; }
    let left = edgeValues[EDGE_OFFSET + edgeIndex];
    let right = edgeValues[EDGE_OFFSET + edgeIndex + 1u];
    if (!(${s}) || left >= right) {
      atomicStore(&edgeValidity[VALIDITY_OFFSET], 0u);
    }
    edgeIndex = edgeIndex + ${h}u;
  }
}`;return a.push(...A(e,{id:`${t}-validate-edges`,source:c,resources:[{buffer:n,usage:`storage-read`},{buffer:i,usage:`storage-write`}],bindings:{edgeValues:n,edgeValidity:i},dispatchCount:1})),a}function O(e,t){let n=j(e,t.input.length),i=t.output.length<=g,a=i&&t.output.length<=_&&m(e.device)===`subgroups`,o=I(t.input.format),s=F(t.edges),c=s?`@group(0) @binding(1) var<storage, read> edgeValues: array<${o}>;`:``,u=t.edgeValidity?`@group(0) @binding(2) var<storage, read> edgeValidity: array<u32>;`:``,d=s?3:1,v=t.mask?`@group(0) @binding(${d}) var<storage, read> selectionMask: array<u32>;`:``,y=d+Number(!!t.mask),b=t.edges,x=s?`fn getEdge(index: u32) -> ${o} {
  return edgeValues[EDGE_OFFSET + index];
}`:`const EDGES: array<${o}, ${b.length}> = array<${o}, ${b.length}>(
  ${b.map(e=>L(e,t.input.format)).join(`, `)}
);
fn getEdge(index: u32) -> ${o} { return EDGES[index]; }`,S=t.edgeValidity?`edgeValidity[${r(t.edgeValidity)}u] != 0u`:`true`,C=t.input.format===`float32`?`value == value && abs(value) <= 3.402823466e+38`:`true`,w=a?p(`accepted`,`binIndex`,`localCounts`,t.output.length):`  if (accepted) { atomicAdd(&localCounts[binIndex], 1u); }`,E=i?`${w}
  workgroupBarrier();
  if (lane < BIN_COUNT) {
    atomicAdd(&outputCounts[OUTPUT_OFFSET + lane], atomicLoad(&localCounts[lane]));
  }`:`if (accepted) { atomicAdd(&outputCounts[OUTPUT_OFFSET + binIndex], 1u); }`,D=`
${a?`enable subgroups;`:``}
const ELEMENT_COUNT: u32 = ${t.input.length}u;
const BIN_COUNT: u32 = ${t.output.length}u;
const INPUT_OFFSET: u32 = ${r(t.input)}u;
const INPUT_STRIDE: u32 = ${T(t.input)}u;
${s?`const EDGE_OFFSET: u32 = ${r(t.edges)}u;`:``}
${t.mask?`const MASK_OFFSET: u32 = ${r(t.mask)}u;
const MASK_STRIDE: u32 = ${T(t.mask)}u;`:``}
const OUTPUT_OFFSET: u32 = ${r(t.output)}u;
@group(0) @binding(0) var<storage, read> inputValues: array<${o}>;
${c}
${u}
${v}
@group(0) @binding(${y}) var<storage, read_write> outputCounts: array<atomic<u32>>;
${i?`var<workgroup> localCounts: array<atomic<u32>, ${t.output.length}>;`:``}
${x}
${a?f():``}

@compute @workgroup_size(${h}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>${a?`,
  @builtin(subgroup_invocation_id) subgroupInvocationId: u32`:``}
) {
  ${l(n,h)}
  let lane = localInvocationIndex;
  ${i?`if (lane < BIN_COUNT) { atomicStore(&localCounts[lane], 0u); }
  workgroupBarrier();`:``}
  var accepted = false;
  var binIndex = 0u;
  if (index < ELEMENT_COUNT && ${S}${t.mask?` && selectionMask[MASK_OFFSET + index * MASK_STRIDE] != 0u`:``}) {
    let value = inputValues[INPUT_OFFSET + index * INPUT_STRIDE];
    let minimum = getEdge(0u);
    let maximum = getEdge(BIN_COUNT);
    if (${C} && value >= minimum && value <= maximum) {
      accepted = true;
      var lower = 0u;
      var upper = BIN_COUNT;
      loop {
        if (lower >= upper) { break; }
        let middle = lower + (upper - lower) / 2u;
        if (value < getEdge(middle + 1u)) {
          upper = middle;
        } else {
          lower = middle + 1u;
        }
      }
      binIndex = min(lower, BIN_COUNT - 1u);
    }
  }
  ${E}
}`,O=[{buffer:t.input,usage:`storage-read`},...s?[{buffer:t.edges,usage:`storage-read`}]:[],...t.edgeValidity?[{buffer:t.edgeValidity,usage:`storage-read`}]:[],...t.mask?[{buffer:t.mask,usage:`storage-read`}]:[],{buffer:t.output,usage:`storage-read-write`}];return A(e,{id:t.id,source:D,resources:O,bindings:{inputValues:t.input,...s?{edgeValues:t.edges}:{},...t.edgeValidity?{edgeValidity:t.edgeValidity}:{},...t.mask?{selectionMask:t.mask}:{},outputCounts:t.output},dispatchLayout:n})}function k(e,t){let n=j(e,t.input.length),i=t.output.length<=g,a=i&&t.output.length<=_&&m(e.device)===`subgroups`,o=I(t.input.format),s=P(t.domain),c=t.domain,u=s?`@group(0) @binding(1) var<storage, read> domainValues: array<`+o+`>;`:``,d=s?2:1,v=t.mask?`@group(0) @binding(${d}) var<storage, read> selectionMask: array<u32>;`:``,y=d+Number(!!t.mask),b=s?`let minimum: ${o} = domainValues[DOMAIN_OFFSET];
  let maximum: ${o} = domainValues[DOMAIN_OFFSET + 1u];`:`let minimum: ${o} = ${L(c[0],t.input.format)};
  let maximum: ${o} = ${L(c[1],t.input.format)};`,x=t.input.format===`float32`?`value == value && abs(value) <= 3.402823466e+38`:`true`,S=32-Math.clz32(t.output.length),C=t.input.format===`float32`?``:`fn multiplyDivideFloor(numerator: u32, multiplier: u32, denominator: u32) -> u32 {
  var quotient = 0u;
  var remainder = 0u;
  var bitIndex = ${S}u;
  loop {
    bitIndex = bitIndex - 1u;
    // Double and add modulo denominator without overflowing u32.
    let doubledThreshold = denominator - remainder;
    if (remainder >= doubledThreshold) {
      remainder = remainder - doubledThreshold;
      quotient = quotient * 2u + 1u;
    } else {
      remainder = remainder * 2u;
      quotient = quotient * 2u;
    }
    if (((multiplier >> bitIndex) & 1u) != 0u) {
      let additionThreshold = denominator - numerator;
      if (remainder >= additionThreshold) {
        remainder = remainder - additionThreshold;
        quotient = quotient + 1u;
      } else {
        remainder = remainder + numerator;
      }
    }
    if (bitIndex == 0u) { break; }
  }
  return quotient;
}`,w=t.input.format===`float32`?`let ratio = (value - minimum) / (maximum - minimum);
          binIndex = min(u32(ratio * f32(BIN_COUNT)), BIN_COUNT - 1u);`:t.input.format===`uint32`?`binIndex = multiplyDivideFloor(value - minimum, BIN_COUNT, maximum - minimum);`:`let orderedValue = bitcast<u32>(value) ^ 0x80000000u;
          let orderedMinimum = bitcast<u32>(minimum) ^ 0x80000000u;
          let orderedMaximum = bitcast<u32>(maximum) ^ 0x80000000u;
          binIndex = multiplyDivideFloor(
            orderedValue - orderedMinimum,
            BIN_COUNT,
            orderedMaximum - orderedMinimum
          );`,E=a?p(`accepted`,`binIndex`,`localCounts`,t.output.length):`  if (accepted) { atomicAdd(&localCounts[binIndex], 1u); }`,D=i?`${E}
  workgroupBarrier();
  if (lane < BIN_COUNT) {
    atomicAdd(&outputCounts[OUTPUT_OFFSET + lane], atomicLoad(&localCounts[lane]));
  }`:`if (accepted) { atomicAdd(&outputCounts[OUTPUT_OFFSET + binIndex], 1u); }`,O=`
${a?`enable subgroups;`:``}
const ELEMENT_COUNT: u32 = ${t.input.length}u;
const BIN_COUNT: u32 = ${t.output.length}u;
const INPUT_OFFSET: u32 = ${r(t.input)}u;
const INPUT_STRIDE: u32 = ${T(t.input)}u;
${s?`const DOMAIN_OFFSET: u32 = ${r(t.domain)}u;`:``}
${t.mask?`const MASK_OFFSET: u32 = ${r(t.mask)}u;
const MASK_STRIDE: u32 = ${T(t.mask)}u;`:``}
const OUTPUT_OFFSET: u32 = ${r(t.output)}u;
@group(0) @binding(0) var<storage, read> inputValues: array<${o}>;
${u}
${v}
@group(0) @binding(${y}) var<storage, read_write> outputCounts: array<atomic<u32>>;
${i?`var<workgroup> localCounts: array<atomic<u32>, ${t.output.length}>;`:``}
${C}
${a?f():``}

@compute @workgroup_size(${h}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>${a?`,
  @builtin(subgroup_invocation_id) subgroupInvocationId: u32`:``}
) {
  ${l(n,h)}
  let lane = localInvocationIndex;
  ${b}
  ${i?`if (lane < BIN_COUNT) { atomicStore(&localCounts[lane], 0u); }
  workgroupBarrier();`:``}
  var accepted = false;
  var binIndex = 0u;
  if (index < ELEMENT_COUNT && maximum >= minimum${t.mask?` && selectionMask[MASK_OFFSET + index * MASK_STRIDE] != 0u`:``}) {
    let value = inputValues[INPUT_OFFSET + index * INPUT_STRIDE];
    if (${x} && value >= minimum && value <= maximum) {
      if (maximum == minimum) {
        accepted = value == minimum;
      } else {
        accepted = true;
        if (value == maximum) {
          binIndex = BIN_COUNT - 1u;
        } else {
          ${w}
        }
      }
    }
  }
  ${D}
}`,k=[{buffer:t.input,usage:`storage-read`},...s?[{buffer:t.domain,usage:`storage-read`}]:[],...t.mask?[{buffer:t.mask,usage:`storage-read`}]:[],{buffer:t.output,usage:`storage-read-write`}];return A(e,{id:t.id,source:O,resources:k,bindings:{inputValues:t.input,...s?{domainValues:t.domain}:{},...t.mask?{selectionMask:t.mask}:{},outputCounts:t.output},dispatchLayout:n})}function A(e,t){let r=[],i=t.dispatchLayout?t.dispatchLayout.x*t.dispatchLayout.y*t.dispatchLayout.z:t.dispatchCount??1;return r.push(n({id:t.id,resources:t.resources,workload:{operation:`GPUHistogram`,commandCount:1,maximumWorkgroupCount:i,maximumInvocationCount:i*h},compile:({device:e})=>{let n=new s(e,{id:t.id,source:t.source,shaderLayout:{bindings:Object.keys(t.bindings).map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:r})=>{let i={};for(let[e,n]of Object.entries(t.bindings))i[e]=a(n,r);t.dispatchLayout?n.dispatch(e,{bindings:i,x:t.dispatchLayout.x,y:t.dispatchLayout.y,z:t.dispatchLayout.z}):n.dispatch(e,{bindings:i,x:t.dispatchCount??1})},destroy:()=>n.destroy()}}})),r}function j(e,t){return u(`GPUHistogram`,t,h,e.device.limits.maxComputeWorkgroupsPerDimension)}function M(e,t,n,r){if(F(e)){if(i(e,b,`${r} edges`),e.format!==t||e.length!==n.length+1)throw Error(`${r} GPU edges must contain output.length + 1 ${t} rows`);if(e.buffer===n.buffer)throw Error(`${r} edges and output must use separate buffers`);return}if(e.length!==n.length+1)throw Error(`${r} literal edges must contain output.length + 1 values`);if(e.length>v)throw Error(`${r} literal edges support at most ${v} values`);let a=t===`uint32`?0:-2147483648,o=t===`uint32`?4294967295:2147483647,s=t===`float32`?e.map(Math.fround):e;if((t===`float32`?s.some(e=>!Number.isFinite(e)):e.some(e=>!Number.isInteger(e)||e<a||e>o))||s.some((e,t)=>t>0&&e<=s[t-1]))throw Error(`${r} literal edges must be finite, representable, and strictly increasing`)}function N(e,t,n){if(e.length!==2||!e.every(Number.isFinite)||e[0]>e[1])throw Error(`${n} literal domain must be a finite [min, max] pair`);if(t!==`float32`){let r=t===`uint32`?0:-2147483648,i=t===`uint32`?4294967295:2147483647;if(!e.every(e=>Number.isInteger(e)&&e>=r&&e<=i))throw Error(`${n} literal domain values must fit ${t}`)}}function P(e){return e!==void 0&&e!==`auto`&&!Array.isArray(e)}function F(e){return e!==void 0&&!Array.isArray(e)}function I(e){return e===`uint32`?`u32`:e===`sint32`?`i32`:`f32`}function L(e,t){if(t===`uint32`)return`${e}u`;if(t===`sint32`)return`${e}`;let n=`${Math.fround(e)}`;return n.includes(`.`)||n.includes(`e`)?n:`${n}.0`}export{x as t};