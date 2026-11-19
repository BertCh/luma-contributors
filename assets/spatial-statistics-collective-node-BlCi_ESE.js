import{Q as e}from"./dist-BusamGlx.js";import{c as t,o as n,t as r}from"./kernel-C3tfQOn4.js";import{r as i}from"./gpu-command-graph-BdzdmV6u.js";import{t as a}from"./gpu-dispatch-utils-CbRzFAgT.js";import{a as o}from"./graph-chunk-dispatch-zBt8k2-o.js";import{i as s}from"./graph-chunk-groups-BB8ACpQO.js";import{S as c,f as l}from"./gpu-contributor-utils-BIwLQON5.js";function u(u,d){let f=d.workgroupSize??256;s(u,d.id,d.bindings.length,f),o(u,d.id,Object.fromEntries(d.bindings.map(e=>[e.name,e.view]))),l(d.id,`invocationCount`,d.invocationCount),c(d.id,`parameterValues length`,d.parameterValues?.length??0,28);let p=new Uint32Array(64);p[1]=d.invocationCount;for(let[e,n]of d.bindings.entries())p[4+e]=t(n.view);for(let[e,t]of(d.parameterValues??[]).entries())l(d.id,`parameterValues[${e}]`,t,0),p[36+e]=t;let m=a(d.id,d.invocationCount,f,u.device.limits.maxComputeWorkgroupsPerDimension),h=`${d.bindings.map((e,t)=>`@group(0) @binding(${t}) var<storage, ${e.access}> ${e.name}: array<${e.type}>;\nvar<private> ${e.name}Offset: u32;`).join(`
`)}
struct StatisticsDispatchParameters { words: array<vec4<u32>, 16> }
@group(0) @binding(${d.bindings.length}) var<uniform> statisticsDispatchParameters: StatisticsDispatchParameters;
fn getChunkParameter(word: u32) -> u32 { return statisticsDispatchParameters.words[word / 4u][word % 4u]; }
${d.control?`@group(0) @binding(${d.bindings.length+1}) var<uniform> statisticsControl: StatisticsDispatchParameters;
fn getStatisticsControl(word: u32) -> u32 { return statisticsControl.words[word / 4u][word % 4u]; }`:``}
${d.declarations??``}
@compute @workgroup_size(${f})
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(num_workgroups) workgroupCount: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32) {
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  // A module-scope private load is nonuniform in WGSL, even after a uniform assignment.
  // Keep the dispatch bound immutable and local so the whole-workgroup return preserves barriers.
  let INVOCATION_COUNT = getChunkParameter(1u);
  if (workgroupIndex >= (INVOCATION_COUNT - 1u) / ${f}u + 1u) { return; }
  let index = workgroupIndex * ${f}u + localInvocationIndex;
  ${d.bindings.map((e,t)=>`${e.name}Offset = getChunkParameter(${4+t}u);\n  _ = &${e.name};`).join(`
  `)}
  ${d.control?`// Statically access this optional uniform even in stages that do not read its values.
  _ = &statisticsControl;`:``}
  ${d.body}
}`;return i({id:d.id,condition:d.condition,resources:[...d.bindings.map(e=>({buffer:e.view,usage:e.access===`read`?`storage-read`:`storage-read-write`})),...d.control?[{buffer:d.control,usage:`uniform`}]:[],...d.extraResources??[]],workload:{operation:d.operation,...d.variant?{variant:d.variant}:{},commandCount:1,maximumInvocationCount:d.invocationCount,maximumWorkgroupCount:m.x*m.y*m.z,readByteLength:d.bindings.filter(e=>e.access===`read`).reduce((e,t)=>e+t.view.length*t.view.rowByteLength,0),writeByteLength:d.bindings.filter(e=>e.access!==`read`).reduce((e,t)=>e+t.view.length*t.view.rowByteLength,0)},compile:({device:t})=>{let i=t.createBuffer({id:`${d.id}-parameters`,data:p,usage:e.UNIFORM|e.COPY_DST}),a;try{a=new r(t,{id:d.pipelineId??d.id,source:h,shaderLayout:{bindings:[...d.bindings.map((e,t)=>({name:e.name,location:t,group:0,type:e.access===`read`?`read-only-storage`:`storage`})),{name:`statisticsDispatchParameters`,location:d.bindings.length,group:0,type:`uniform`},...d.control?[{name:`statisticsControl`,location:d.bindings.length+1,group:0,type:`uniform`}]:[]]}})}catch(e){throw i.destroy(),e}return{encode:({computePass:e,getBuffer:t})=>a.dispatch(e,{bindings:Object.fromEntries([...d.bindings.map(e=>[e.name,n(e.view,t)]),...d.control?[[`statisticsControl`,n(d.control,t)]]:[],[`statisticsDispatchParameters`,{buffer:i,offset:0,size:p.byteLength}]]),...m}),destroy:()=>{a.destroy(),i.destroy()}}}})}export{u as t};