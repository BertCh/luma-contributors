import{t as e}from"./computation-BNoQFgmS.js";import{c as t,o as n}from"./kernel-C3tfQOn4.js";var r=256;function i(i,a){let o=i.graph,s=Math.ceil(a.invocationCount/r),c=a.bindings.map(e=>e.buffer.view),l=e=>e.buffer.format.startsWith(`uint32`)?`u32`:`f32`,u=a.bindings.map((e,n)=>`const ${e.name}Offset: u32 = ${t(c[n])}u;
@group(0) @binding(${n}) var<storage, ${e.access}> ${e.name}: array<${l(e)}>;`).join(`
`),d=`
const INVOCATION_COUNT: u32 = ${a.invocationCount}u;
${u}
${a.declarations??``}

@compute @workgroup_size(${r})
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= INVOCATION_COUNT) {
    return;
  }
  ${a.body}
}`;o.addComputePass({id:a.id,workload:{operation:`RunKernel`,variant:a.id,commandCount:1,maximumWorkgroupCount:s,maximumInvocationCount:s*r,readByteLength:0,writeByteLength:0},resources:a.bindings.map((e,t)=>({buffer:c[t],usage:e.access===`read`?`storage-read`:`storage-read-write`})),compile:({device:t})=>{let r=new e(t,{id:a.id,source:d,shaderLayout:{bindings:a.bindings.map((e,t)=>({name:e.name,type:e.access===`read`?`read-only-storage`:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:t})=>{let i={};a.bindings.forEach((e,r)=>{i[e.name]=n(c[r],t)}),r.setBindings(i),r.dispatch(e,s)},destroy:()=>r.destroy()}}})}export{i as t};