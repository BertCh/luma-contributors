import{n as e,t}from"./gpu-command-dispatch-metadata-Bpi5I1Ez.js";import{l as n,r}from"./gpu-command-graph-BdzdmV6u.js";import{c as i,n as a,o,r as s,t as c,u as l}from"./kernel-C3tfQOn4.js";import{n as u,t as d}from"./gpu-dispatch-utils-CbRzFAgT.js";import{n as f,t as p}from"./gpu-scan-B4NNTfF9.js";var m=256,h=class{id;input;flags;output;count;constructor(e){if(this.id=e.id??`gpu-compaction`,this.input=e.input,this.flags=e.flags,this.output=e.output,this.count=e.count,S(this.input,`${this.id} input`),S(this.flags,`${this.id} flags`),S(this.output,`${this.id} output`),l(this.count,`${this.id} count`),this.flags.length!==this.input.length)throw Error(`${this.id} flags.length must equal input.length`);if(this.output.length<this.input.length)throw Error(`${this.id} output must contain at least input.length rows`);if(this.count.length<1)throw Error(`${this.id} count must contain one uint32 row`)}getCommandNodes(e){let t=[];return t.push(...g(this,e,e.device.limits.maxComputeWorkgroupsPerDimension)),t}};function g(e,t,r){let o=[];for(let n of[...x(e.input),...x(e.flags),...x(e.output),e.count])if(n.buffer.graph!==t)throw Error(`${e.id} views must belong to the target graph`);if(e.input.length===0)return o.push(..._(t,e.id,e.count)),o;let c=e.flags instanceof n?a(t,`${e.id}-offsets`,e.flags):s(t,`${e.id}-offsets`,`uint32`,e.flags.length),l=e.flags instanceof n?a(t,`${e.id}-flags`,e.flags):s(t,`${e.id}-flags`,`uint32`,e.flags.length);for(let[n,a]of x(e.flags).entries()){if(!a.length)continue;let s=x(l)[n],c=d(e.id,a.length,m,r);o.push(...b(t,{id:`${e.id}-normalize-${n}`,source:`
@group(0) @binding(0) var<storage, read> inputValues: array<u32>;
@group(0) @binding(1) var<storage, read_write> outputValues: array<u32>;
@compute @workgroup_size(${m}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${u(c,m)}
  if (index >= ${a.length}u) { return; }
  outputValues[${i(s)}u + index] = min(inputValues[${i(a)}u + index], 1u);
}`,resources:[{buffer:a,usage:`storage-read`},{buffer:s,usage:`storage-write`}],bindings:{inputValues:a,outputValues:s},dispatchLayout:c}))}let h=new p({id:`${e.id}-scan`,input:l,output:c});return o.push(...f(h,t,r)),o.push(...v(t,e.id,e.input,e.flags,c,e.output,e.count,r)),o}function _(e,n,a){let s=[],l=`${n}-clear-count`;return s.push(r({id:l,resources:[{buffer:a,usage:`storage-write`}],compile:({device:e})=>{let t=new c(e,{id:l,source:`const COUNT_OFFSET: u32 = ${i(a)}u;
@group(0) @binding(0) var<storage, read_write> outputCount: array<u32>;
@compute @workgroup_size(1) fn main() { outputCount[COUNT_OFFSET] = 0u; }`,shaderLayout:{bindings:[{name:`outputCount`,type:`storage`,group:0,location:0}]}});return{encode:({computePass:e,getBuffer:n})=>{t.dispatch(e,{bindings:{outputCount:o(a,n)},x:1})},destroy:()=>t.destroy()}}})),s.map(e=>t(e,[1,1,1]))}function v(t,r,i,a,o,s,c,l){let u=[],f=e(t,[i,a,o]),p=f.map(e=>e[0]),h=f.map(e=>e[1]),g=f.map(e=>e[2]),_=x(s),v=p.map((e,t)=>e.length>0?t:-1).filter(e=>e>=0),b=_.map((e,t)=>e.length>0?t:-1).filter(e=>e>=0),S=v[v.length-1],C=b[b.length-1],w=f.length>1||s instanceof n,T=0;for(let e=0;e<_.length;e++){let n=_[e],i=T+n.length;if(n.length>0)for(let a of v){let o=a===S&&e===C;u.push(...y(t,{id:w?`${r}-scatter-input-${a}-output-${e}`:`${r}-scatter`,input:p[a],flags:h[a],offsets:g[a],output:n,outputStart:T,outputEnd:i,count:o?c:void 0,dispatchLayout:d(`GPUCompaction`,p[a].length,m,l)}))}T=i}return u}function y(e,t){let n=[],r=t.count?`@group(0) @binding(4) var<storage, read_write> outputCount: array<u32>;`:``,a=t.count?`if (index == ELEMENT_COUNT - 1u) {
    outputCount[COUNT_OFFSET] = outputIndex + flag;
  }`:``,o=`
const ELEMENT_COUNT: u32 = ${t.input.length}u;
const INPUT_OFFSET: u32 = ${i(t.input)}u;
const FLAGS_OFFSET: u32 = ${i(t.flags)}u;
const OFFSETS_OFFSET: u32 = ${i(t.offsets)}u;
const OUTPUT_OFFSET: u32 = ${i(t.output)}u;
const OUTPUT_START: u32 = ${t.outputStart}u;
const OUTPUT_END: u32 = ${t.outputEnd}u;
${t.count?`const COUNT_OFFSET: u32 = ${i(t.count)}u;`:``}
@group(0) @binding(0) var<storage, read> inputValues: array<u32>;
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputValues: array<u32>;
${r}

@compute @workgroup_size(${m}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${u(t.dispatchLayout,m)}
  if (index >= ELEMENT_COUNT) { return; }
  let flag = min(flags[FLAGS_OFFSET + index], 1u);
  let outputIndex = offsets[OFFSETS_OFFSET + index];
  if (flag != 0u && outputIndex >= OUTPUT_START && outputIndex < OUTPUT_END) {
    outputValues[OUTPUT_OFFSET + outputIndex - OUTPUT_START] = inputValues[INPUT_OFFSET + index];
  }
  ${a}
}`;return n.push(...b(e,{id:t.id,source:o,resources:[{buffer:t.input,usage:`storage-read`},{buffer:t.flags,usage:`storage-read`},{buffer:t.offsets,usage:`storage-read`},{buffer:t.output,usage:`storage-write`},...t.count?[{buffer:t.count,usage:`storage-write`}]:[]],bindings:{inputValues:t.input,flags:t.flags,offsets:t.offsets,outputValues:t.output,...t.count?{outputCount:t.count}:{}},dispatchLayout:t.dispatchLayout})),n}function b(e,n){let i=[];return i.push(r({id:n.id,workload:{operation:`GPUCompaction`,commandCount:1,maximumWorkgroupCount:n.dispatchLayout.x*n.dispatchLayout.y*n.dispatchLayout.z,maximumInvocationCount:n.dispatchLayout.x*n.dispatchLayout.y*n.dispatchLayout.z*m,readByteLength:n.resources.reduce((e,t)=>e+(t.usage===`storage-read`||t.usage===`storage-read-write`?t.buffer.length*t.buffer.rowByteLength:0),0),writeByteLength:n.resources.reduce((e,t)=>e+(t.usage===`storage-write`||t.usage===`storage-read-write`?t.buffer.length*t.buffer.rowByteLength:0),0)},resources:n.resources,compile:({device:e})=>{let t=new c(e,{id:n.id,source:n.source,shaderLayout:{bindings:Object.keys(n.bindings).map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:r})=>{let i={};for(let[e,t]of Object.entries(n.bindings))i[e]=o(t,r);t.dispatch(e,{bindings:i,x:n.dispatchLayout.x,y:n.dispatchLayout.y,z:n.dispatchLayout.z})},destroy:()=>t.destroy()}}})),i.map(e=>t(e,[n.dispatchLayout.x,n.dispatchLayout.y,n.dispatchLayout.z]))}function x(e){return e instanceof n?e.data:[e]}function S(e,t){for(let n of x(e))l(n,t)}export{g as n,h as t};