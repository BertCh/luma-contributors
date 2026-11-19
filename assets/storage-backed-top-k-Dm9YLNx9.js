import{Q as e}from"./dist-BusamGlx.js";import{c as t,o as n,s as r,t as i}from"./kernel-C3tfQOn4.js";import{r as a}from"./gpu-command-graph-BdzdmV6u.js";import{t as o}from"./gpu-dispatch-utils-CbRzFAgT.js";import{T as s,v as c}from"./gpu-contributor-utils-BIwLQON5.js";function l(e,t){let n=n=>`${e}[(${t}) + (${n}) * 2u]`,r=n=>`${e}[(${t}) + (${n}) * 2u + 1u]`;return{readRow:n,readDistance:e=>`bitcast<f32>(${r(e)})`,writeRow:(e,t)=>`${n(e)} = ${t};`,writeDistance:(e,t)=>`${r(e)} = bitcast<u32>(${t});`}}var u=64,d=12,f=8,p=new WeakMap;function m(l,m){s(m.id,m.bindings.length<=Math.min(f,l.device.limits.maxStorageBuffersPerShaderStage),`bindings must fit TOP_K_MAXIMUM_BINDINGS (8) and the device storage binding limit`);let h=new Uint32Array(d);h[0]=m.invocationCount,h[9]=m.firstRow??0,h[10]=m.batchIndex??0;let g=m.declarations??``,_=m.body;for(let[e,n]of m.bindings.entries()){c(l.device,m.id,n.name,r(n.view).size,`reduce the bound view length`);let i=e+1;h[i]=t(n.view);let a=`selectionSettings[${Math.floor(i/4)}].${`xyzw`[i%4]}`,o=RegExp(`\\b${n.name}Offset\\b`,`g`);g=g.replace(o,a),_=_.replace(o,a)}let v=`${m.bindings.map((e,t)=>`@group(0) @binding(${t}) var<storage, ${e.access}> ${e.name}: array<${e.type}>;`).join(`
`)}
@group(0) @binding(${m.bindings.length}) var<uniform> selectionSettings: array<vec4<u32>, 3>;
${g}
@compute @workgroup_size(${u})
fn main(@builtin(local_invocation_index) localInvocationIndex: u32,
        @builtin(workgroup_id) workgroupId: vec3<u32>,
        @builtin(num_workgroups) workgroupCounts: vec3<u32>) {
  ${m.bindings.map(e=>`_ = &${e.name};`).join(`
`)}
  let index = ((workgroupId.z * workgroupCounts.y + workgroupId.y) * workgroupCounts.x + workgroupId.x) * ${u}u + localInvocationIndex;
  if (index >= selectionSettings[0].x) { return; }
  let batchFirstRow = selectionSettings[2].y;
  let batchIndex = selectionSettings[2].z;
  ${_}
}`,y=o(m.operation,Math.max(1,m.invocationCount),u,l.device.limits.maxComputeWorkgroupsPerDimension),b=a({id:m.id,resources:m.bindings.map(e=>({buffer:e.view,usage:e.access===`read`?`storage-read`:`storage-read-write`})),workload:{operation:m.operation,...m.variant?{variant:m.variant}:{},maximumInvocationCount:m.invocationCount,maximumWorkgroupCount:y.x*y.y*y.z},compile:({device:t})=>{let r=t.createBuffer({id:`${m.id}-selection-settings`,data:h,usage:e.UNIFORM|e.COPY_DST}),a=new i(t,{id:m.id,source:v,shaderLayout:{bindings:[...m.bindings.map((e,t)=>({name:e.name,type:e.access===`read`?`read-only-storage`:`storage`,group:0,location:t})),{name:`selectionSettings`,type:`uniform`,group:0,location:m.bindings.length}]}});return{encode:({computePass:e,getBuffer:t})=>{let i={selectionSettings:r};for(let e of m.bindings)i[e.name]=n(e.view,t);a.dispatch(e,{bindings:i,x:y.x,y:y.y,z:y.z})},destroy:()=>{a.destroy(),r.destroy()}}}});return p.set(b,{source:v,settings:h,bindings:m.bindings,layout:y}),b}export{l as n,m as t};