import{o as e,t}from"./kernel-C3tfQOn4.js";import{r as n}from"./gpu-command-graph-BdzdmV6u.js";import{t as r}from"./gpu-dispatch-utils-CbRzFAgT.js";var i=16777216;function a(a,o){let s=[{name:`chunkParameters`,view:o.parameters,type:`u32`,access:`read`},...o.bindings],c=s.map((e,t)=>`@group(0) @binding(${t}) var<storage, ${e.access}> ${e.name}: array<${e.type}>;`).join(`
`),l=o.bindings.map(e=>`let ${e.name}Offset = arrayLength(&${e.name}) - ${e.scope===`total`?`chunkParameters[2]`:`rowCount`} * ${e.components}u;`).join(`
  `),u=`${c}
${o.declarations??``}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) workgroupId: vec3u,
  @builtin(num_workgroups) workgroupCount: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32) {
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${i}u) { return; }
  let index = workgroupIndex * 256u + localInvocationIndex;
  let rowCount = chunkParameters[1];
  if (index >= rowCount) { return; }
  let chunkParametersOffset = 0u;
  ${l}
  ${o.body}
}`,d=r(o.operation,Math.max(o.length,1),256,a.device.limits.maxComputeWorkgroupsPerDimension);return n({id:o.id,resources:s.map(e=>({buffer:e.view,usage:e.access===`read`?`storage-read`:`storage-read-write`})),workload:{operation:o.operation,variant:o.variant,commandCount:1,maximumInvocationCount:o.length,maximumWorkgroupCount:d.x*d.y*d.z},compile:({device:n})=>{let r=new t(n,{id:o.id,source:u,shaderLayout:{bindings:s.map((e,t)=>({name:e.name,group:0,location:t,type:e.access===`read`?`read-only-storage`:`storage`}))}});return{encode:({computePass:t,getBuffer:n})=>r.dispatch(t,{bindings:Object.fromEntries(s.map(t=>[t.name,e(t.view,n)])),...d}),destroy:()=>r.destroy()}}})}export{a as t};