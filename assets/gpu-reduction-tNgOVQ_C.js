import{n as e,t}from"./gpu-command-dispatch-metadata-Bpi5I1Ez.js";import{l as n,r}from"./gpu-command-graph-BdzdmV6u.js";import{c as i,d as a,o,r as s,t as c,u as l}from"./kernel-C3tfQOn4.js";import{n as u,t as d}from"./gpu-dispatch-utils-CbRzFAgT.js";import{t as f}from"./gpu-subgroup-utils-QYBHgloD.js";var p=256,m=[`uint32`,`sint32`,`float32`];function h(e){return f(e,{requiresSubgroupId:!0})}var g=class{id;input;mask;output;operation;constructor(e){this.id=e.id??`gpu-reduction`,this.input=e.input,this.mask=e.mask,this.output=e.output,this.operation=e.operation;for(let[e,t]of y(this.input).entries()){let r=this.input instanceof n?` input chunk ${e}`:` input`;if(a(t,m,`${this.id}${r}`),t.format!==this.input.format)throw Error(`${this.id}${r} format must match the input format`)}if(a(this.output,m,`${this.id} output`),this.input.format!==this.output.format)throw Error(`${this.id} input and output formats must match`);if(![`sum`,`min`,`max`,`extent`].includes(this.operation))throw Error(`${this.id} operation must be sum, min, max, or extent`);let t=this.operation===`extent`?2:1;if(this.output.length!==t)throw Error(`${this.id} ${this.operation} output must contain ${t} row(s)`);if(y(this.input).some(e=>e.buffer===this.output.buffer))throw Error(`${this.id} inputs and output must use separate buffers`);if(this.mask){for(let[e,t]of b(this.mask).entries())if(l(t,`${this.id} mask chunk ${e}`),t.buffer===this.output.buffer)throw Error(`${this.id} mask and output must use separate buffers`);if(this.input.length!==this.mask.length)throw Error(`${this.id} input and mask lengths must match`)}}getCommandNodes(t){let n=[],r=y(this.input),i=this.mask?b(this.mask):void 0;if(r.some(e=>e.buffer.graph!==t)||i?.some(e=>e.buffer.graph!==t)||this.output.buffer.graph!==t)throw Error(`${this.id} views must belong to the target graph`);if(this.mask){let n=e(t,[this.input,this.mask]);r=n.map(([e])=>e),i=n.map(([,e])=>e)}let a=r.map((e,t)=>({input:e,mask:i?.[t]})).filter(({input:e})=>e.length>0);if(a.length===0)return n.push(...D(t,this.id,this.output)),n;let o=this.operation===`extent`?2:1,c=this.operation!==`sum`&&(this.input.format===`float32`||!!this.mask),l;if(a.length===1)l=_(t,{id:this.id,format:this.input.format,operation:this.operation,inputValues:a[0].input,inputValidity:a[0].mask,inputLength:a[0].input.length,valuesPerRow:o,firstLevel:!0,needsValidity:c}),n.push(...l.nodes);else{let e=s(t,`${this.id}-chunk-values`,this.input.format,a.length*o),r=c?s(t,`${this.id}-chunk-validity`,`uint32`,a.length):void 0;a.forEach(({input:i,mask:a},s)=>{n.push(..._(t,{id:`${this.id}-chunk-${s}`,format:this.input.format,operation:this.operation,inputValues:i,inputValidity:a,inputLength:i.length,valuesPerRow:o,firstLevel:!0,needsValidity:c,finalValues:v(t,e,s*o,o),finalValidity:r?v(t,r,s,1):void 0}).nodes)}),l=_(t,{id:`${this.id}-merge`,format:this.input.format,operation:this.operation,inputValues:e,inputValidity:r,inputLength:a.length,valuesPerRow:o,firstLevel:!1,needsValidity:c}),n.push(...l.nodes)}return n.push(...E(t,{id:`${this.id}-finalize`,format:this.input.format,inputValues:l.values,inputValidity:l.validity,output:this.output,valuesPerRow:o})),n}};function _(e,t){let n=[],r=t.inputValues,i=t.inputValidity,a=t.inputLength,o=0;for(;a>1||o===0;){let c=Math.ceil(a/p),l=c===1,u=l&&t.finalValues?t.finalValues:s(e,`${t.id}-level-${o}-values`,t.format,c*t.valuesPerRow),d=t.needsValidity?l&&t.finalValidity?t.finalValidity:s(e,`${t.id}-level-${o}-validity`,`uint32`,c):void 0;n.push(...x(e,{id:`${t.id}-level-${o}`,format:t.format,operation:t.operation,inputValues:r,inputValidity:i,outputValues:u,outputValidity:d,inputLength:a,valuesPerRow:t.valuesPerRow,firstLevel:t.firstLevel&&o===0})),r=u,i=d,a=c,o++}return{values:r,validity:i,nodes:n}}function v(e,t,n,r){return e.createDataView(t.buffer,{format:t.format,length:r,byteOffset:t.byteOffset+n*t.rowByteLength})}function y(e){return e instanceof n?e.data:[e]}function b(e){return e instanceof n?e.data:[e]}function x(e,t){let n=[],r=d(`GPUReduction`,t.inputLength,p,e.device.limits.maxComputeWorkgroupsPerDimension),a=k(t.format),o=A(t.format),s=t.operation===`sum`,c=t.operation===`extent`,l=t.inputValidity?`@group(0) @binding(1) var<storage, read> inputValidity: array<u32>;`:``,f=t.inputValidity?2:1,m=f+1,g=`value = inputValues[INPUT_OFFSET + index];
    secondValue = value;
    valid = select(0u, 1u, ${t.format===`float32`?`value == value && abs(value) <= 3.402823466e+38`:`true`});${t.firstLevel&&t.inputValidity&&t.format===`float32`&&s?`
    if (valid == 0u) { value = 0.0; secondValue = 0.0; }`:``}`,_=t.firstLevel?t.inputValidity?`if (inputValidity[VALIDITY_OFFSET + index] != 0u) {
      ${g}
    }`:g:`value = inputValues[INPUT_OFFSET + index * VALUES_PER_ROW];
    secondValue = inputValues[INPUT_OFFSET + index * VALUES_PER_ROW + ${c?`1u`:`0u`}];
    ${t.inputValidity?`valid = inputValidity[VALIDITY_OFFSET + index];`:`valid = 1u;`}`,v=s?`firstScratch[lane] = firstScratch[lane] + firstScratch[lane + stride];`:`let rightValid = validityScratch[lane + stride];
      if (rightValid != 0u) {
        if (validityScratch[lane] == 0u) {
          firstScratch[lane] = firstScratch[lane + stride];
          secondScratch[lane] = secondScratch[lane + stride];
        } else {
          firstScratch[lane] = ${t.operation===`max`?`max`:`min`}(firstScratch[lane], firstScratch[lane + stride]);
          secondScratch[lane] = ${c||t.operation===`max`?`max`:`min`}(secondScratch[lane], secondScratch[lane + stride]);
        }
        validityScratch[lane] = 1u;
      }`,y=h(e.device),b=y===`subgroups`?C({format:t.format,operation:t.operation,combine:v}):S(v),x=`
${y===`subgroups`?`enable subgroups;
requires subgroup_id;`:``}
const ELEMENT_COUNT: u32 = ${t.inputLength}u;
const VALUES_PER_ROW: u32 = ${t.valuesPerRow}u;
const INPUT_OFFSET: u32 = ${i(t.inputValues)}u;
${t.inputValidity?`const VALIDITY_OFFSET: u32 = ${i(t.inputValidity)}u;`:``}
const OUTPUT_OFFSET: u32 = ${i(t.outputValues)}u;
${t.outputValidity?`const OUTPUT_VALIDITY_OFFSET: u32 = ${i(t.outputValidity)}u;`:``}
@group(0) @binding(0) var<storage, read> inputValues: array<${a}>;
${l}
@group(0) @binding(${f}) var<storage, read_write> outputValues: array<${a}>;
${t.outputValidity?`@group(0) @binding(${m}) var<storage, read_write> outputValidity: array<u32>;`:``}
var<workgroup> firstScratch: array<${a}, ${p}>;
var<workgroup> secondScratch: array<${a}, ${p}>;
var<workgroup> validityScratch: array<u32, ${p}>;

@compute @workgroup_size(${p}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  ${y===`subgroups`?`@builtin(subgroup_invocation_id) subgroupInvocationId: u32,
  @builtin(subgroup_size) subgroupSize: u32,
  @builtin(subgroup_id) subgroupId: u32,`:``}
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${u(r,p)}
  if (workgroupIndex >= ${Math.ceil(t.inputLength/p)}u) {
    return;
  }
  let lane = localInvocationIndex;
  var value = ${o};
  var secondValue = ${o};
  var valid = 0u;
  if (index < ELEMENT_COUNT) {
    ${_}
  }
  ${b}
  if (lane == 0u) {
    outputValues[OUTPUT_OFFSET + workgroupIndex * VALUES_PER_ROW] = firstScratch[0];
    ${c?`outputValues[OUTPUT_OFFSET + workgroupIndex * VALUES_PER_ROW + 1u] = secondScratch[0];`:``}
    ${t.outputValidity?`outputValidity[OUTPUT_VALIDITY_OFFSET + workgroupIndex] = validityScratch[0];`:``}
  }
}`,w=[{buffer:t.inputValues,usage:`storage-read`},...t.inputValidity?[{buffer:t.inputValidity,usage:`storage-read`}]:[],{buffer:t.outputValues,usage:`storage-write`},...t.outputValidity?[{buffer:t.outputValidity,usage:`storage-write`}]:[]],T={inputValues:t.inputValues};return t.inputValidity&&(T.inputValidity=t.inputValidity),T.outputValues=t.outputValues,t.outputValidity&&(T.outputValidity=t.outputValidity),n.push(...O(e,{id:t.id,source:x,resources:w,bindings:T,dispatchLayout:r})),n}function S(e){return`firstScratch[lane] = value;
  secondScratch[lane] = secondValue;
  validityScratch[lane] = valid;
  workgroupBarrier();
  var stride = ${p/2}u;
  loop {
    if (lane < stride) {
      ${e}
    }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride = stride / 2u;
  }`}function C(e){return`let subgroupValid = subgroupMax(valid);
  let subgroupFirst = ${w(e.format,e.operation===`extent`?`min`:e.operation,`value`,`valid`)};
  let subgroupSecond = ${w(e.format,e.operation===`extent`?`max`:e.operation,`secondValue`,`valid`)};
  if (subgroupInvocationId == 0u) {
    firstScratch[subgroupId] = subgroupFirst;
    secondScratch[subgroupId] = subgroupSecond;
    validityScratch[subgroupId] = subgroupValid;
  }
  workgroupBarrier();
  let subgroupCount = ${p}u / subgroupSize;
  if (subgroupCount > 1u) {
    var stride = subgroupCount / 2u;
    loop {
      if (lane < stride) {
        ${e.combine}
      }
      workgroupBarrier();
      if (stride == 1u) { break; }
      stride = stride / 2u;
    }
  }`}function w(e,t,n,r){if(t===`sum`)return`subgroupAdd(${n})`;let i=T(e,t);return`${t===`min`?`subgroupMin`:`subgroupMax`}(select(${i}, ${n}, ${r} != 0u))`}function T(e,t){return e===`uint32`?t===`min`?`4294967295u`:`0u`:e===`sint32`?t===`min`?`2147483647`:`(-2147483647 - 1)`:t===`min`?`3.402823466e+38`:`-3.402823466e+38`}function E(e,t){let n=[],r=k(t.format),a=A(t.format),o=Array.from({length:t.valuesPerRow},(e,n)=>{let r=`inputValues[INPUT_OFFSET + ${n}u]`;return`outputValues[OUTPUT_OFFSET + ${n}u] = ${t.inputValidity?`select(${a}, ${r}, valid)`:r};`}).join(`
  `),s=`
const INPUT_OFFSET: u32 = ${i(t.inputValues)}u;
const OUTPUT_OFFSET: u32 = ${i(t.output)}u;
${t.inputValidity?`const VALIDITY_OFFSET: u32 = ${i(t.inputValidity)}u;`:``}
@group(0) @binding(0) var<storage, read> inputValues: array<${r}>;
${t.inputValidity?`@group(0) @binding(1) var<storage, read> inputValidity: array<u32>;`:``}
@group(0) @binding(${t.inputValidity?2:1}) var<storage, read_write> outputValues: array<${r}>;
@compute @workgroup_size(1) fn main() {
  ${t.inputValidity?`let valid = inputValidity[VALIDITY_OFFSET] != 0u;`:``}
  ${o}
}`;return n.push(...O(e,{id:t.id,source:s,resources:[{buffer:t.inputValues,usage:`storage-read`},...t.inputValidity?[{buffer:t.inputValidity,usage:`storage-read`}]:[],{buffer:t.output,usage:`storage-write`}],bindings:{inputValues:t.inputValues,...t.inputValidity?{inputValidity:t.inputValidity}:{},outputValues:t.output},dispatchCount:1})),n}function D(e,t,n){let r=[],a=`${t}-clear`,o=k(n.format),s=A(n.format);return r.push(...O(e,{id:a,source:`const OUTPUT_OFFSET: u32 = ${i(n)}u;
@group(0) @binding(0) var<storage, read_write> outputValues: array<${o}>;
@compute @workgroup_size(1) fn main() {
  for (var index = 0u; index < ${n.length}u; index++) { outputValues[OUTPUT_OFFSET + index] = ${s}; }
}`,resources:[{buffer:n,usage:`storage-write`}],bindings:{outputValues:n},dispatchCount:1})),r}function O(e,n){let{x:i,y:a,z:s}=n.dispatchLayout??{x:n.dispatchCount??1,y:1,z:1},l=r({id:n.id,resources:n.resources,compile:({device:e})=>{let t=Object.keys(n.bindings),r=new c(e,{id:n.id,source:n.source,shaderLayout:{bindings:t.map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:t})=>{let c={};for(let[e,r]of Object.entries(n.bindings))c[e]=o(r,t);r.dispatch(e,{bindings:c,x:i,y:a,z:s})},destroy:()=>r.destroy()}}});return[t(l,[i,a,s])]}function k(e){return e===`uint32`?`u32`:e===`sint32`?`i32`:`f32`}function A(e){return e===`uint32`?`0u`:e===`sint32`?`0`:`0.0`}export{g as t};