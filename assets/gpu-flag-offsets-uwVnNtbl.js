import{l as e,r as t}from"./gpu-command-graph-BdzdmV6u.js";import{a as n,c as r,o as i,t as a,u as o}from"./kernel-C3tfQOn4.js";import{t as s}from"./gpu-dispatch-utils-CbRzFAgT.js";import{t as c}from"./gpu-scan-B4NNTfF9.js";var l=256,u=class{id;props;constructor(e){this.id=e.id??`gpu-flag-offsets`,this.props=Object.freeze({...e,id:this.id}),f(this.props)}getCommandNodes(e){let t=[],{flags:r,count:i}=this.props;for(let t of[...p(r),...p(this.props.offsets),i])if(t.buffer.graph!==e)throw Error(`${this.id} views must belong to the target graph`);let a=n(e,this.props.offsets,r.length);return r.length>0&&t.push(...new c({id:`${this.id}-scan`,input:r,output:a,mode:`exclusive`}).getCommandNodes(e)),t.push(...d(e,{...this.props,offsets:a})),t}};function d(e,n){let o=[],c=n.flags.length>0,u=c?m(n.flags):void 0,d=c?m(n.offsets):void 0,f=s(`GPUFlagOffsetsCount`,1,l,e.device.limits.maxComputeWorkgroupsPerDimension),p=c?`const LAST_FLAG_INDEX: u32 = ${u.length-1}u;
const LAST_OFFSET_INDEX: u32 = ${d.length-1}u;
const FLAG_OFFSET: u32 = ${r(u)}u;
const OFFSETS_OFFSET: u32 = ${r(d)}u;
const COUNT_OFFSET: u32 = ${r(n.count)}u;
@group(0) @binding(0) var<storage, read> flags: array<u32>;
@group(0) @binding(1) var<storage, read> offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> count: array<u32>;
@compute @workgroup_size(${l})
fn main(@builtin(local_invocation_index) localInvocationIndex: u32) {
  if (localInvocationIndex > 0u) { return; }
  count[COUNT_OFFSET] = offsets[OFFSETS_OFFSET + LAST_OFFSET_INDEX] + flags[FLAG_OFFSET + LAST_FLAG_INDEX];
}`:`const COUNT_OFFSET: u32 = ${r(n.count)}u;
@group(0) @binding(0) var<storage, read_write> count: array<u32>;
@compute @workgroup_size(${l})
fn main(@builtin(local_invocation_index) localInvocationIndex: u32) {
  if (localInvocationIndex > 0u) { return; }
  count[COUNT_OFFSET] = 0u;
}`,h=c?[{name:`flags`,view:u,usage:`storage-read`},{name:`offsets`,view:d,usage:`storage-read`},{name:`count`,view:n.count,usage:`storage-write`}]:[{name:`count`,view:n.count,usage:`storage-write`}];return o.push(t({id:`${n.id}-count`,workload:{operation:`GPUFlagOffsetsCount`,commandCount:1,maximumWorkgroupCount:1,maximumInvocationCount:l,readByteLength:c?8:0,writeByteLength:4},resources:h.map(e=>({buffer:e.view,usage:e.usage})),compile:({device:e})=>{let t=new a(e,{id:`${n.id}-count`,source:p,shaderLayout:{bindings:h.map((e,t)=>({name:e.name,type:e.usage===`storage-read`?`read-only-storage`:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:n})=>{let r={};for(let e of h)r[e.name]=i(e.view,n);t.dispatch(e,{bindings:r,x:f.x,y:f.y,z:f.z})},destroy:()=>t.destroy()}}})),o}function f(e){for(let t of p(e.flags))o(t,`${e.id} flags`);for(let t of p(e.offsets))o(t,`${e.id} offsets`);if(o(e.count,`${e.id} count`),e.offsets.length<e.flags.length)throw Error(`${e.id} offsets must contain at least flags.length rows`);if(e.count.length<1)throw Error(`${e.id} result views are too short`)}function p(t){return t instanceof e?t.data:[t]}function m(e){let t=p(e);for(let e=t.length-1;e>=0;e--)if(t[e].length>0)return t[e];throw Error(`Non-empty GPUFlagOffsets input requires a non-empty chunk`)}export{u as t};