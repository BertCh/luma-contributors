import{t as e}from"./computation-BNoQFgmS.js";import{c as t,o as n}from"./kernel-C3tfQOn4.js";var r=256,i={float32:`f32`,uint32:`u32`,sint32:`i32`,float32x2:`vec2f`,float32x4:`vec4f`,uint32x2:`vec2u`,uint32x4:`vec4u`};function a(a,o){let s=Math.max(Math.ceil(o.invocationCount/r),1),c=o.bindings.map(e=>e.buffer.view),l=o.bindings.map((e,n)=>{let r=i[e.buffer.format];return`const ${e.name}Offset: u32 = ${t(c[n])}u;
@group(0) @binding(${n}) var<storage, ${e.access}> ${e.name}: array<${r}>;`}).join(`
`),u=`
const INVOCATION_COUNT: u32 = ${o.invocationCount}u;
${l}
${o.declarations??``}

@compute @workgroup_size(${r})
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  ${o.bindings.map(e=>`_ = &${e.name};`).join(`
  `)}
  if (index >= INVOCATION_COUNT) {
    return;
  }
  ${o.body}
}`;a.graph.addComputePass({id:o.id,workload:{operation:`RunKernel`,variant:o.id,commandCount:1,maximumWorkgroupCount:s,maximumInvocationCount:s*r,readByteLength:0,writeByteLength:0},resources:o.bindings.map((e,t)=>({buffer:c[t],usage:e.access===`read`?`storage-read`:`storage-read-write`})),compile:({device:t})=>{let r=new e(t,{id:o.id,source:u,shaderLayout:{bindings:o.bindings.map((e,t)=>({name:e.name,type:e.access===`read`?`read-only-storage`:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:t})=>{let i={};o.bindings.forEach((e,r)=>{i[e.name]=n(c[r],t)}),r.setBindings(i),r.dispatch(e,s)},destroy:()=>r.destroy()}}})}function o(e){return{id:e.format,format:e.format,length:e.length,view:e,buffer:void 0,write:()=>void 0}}export{o as n,a as t};