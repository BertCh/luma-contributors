import{d as e}from"./kernel-C3tfQOn4.js";import{l as t}from"./gpu-command-graph-BdzdmV6u.js";import{E as n,O as r,S as i,T as a,f as o,v as s,x as c}from"./gpu-contributor-utils-BIwLQON5.js";import{o as l,p as u}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as d}from"./scaled-quotient-wgsl-UDyhQag6.js";import{a as f,i as p}from"./graph-chunk-dispatch-zBt8k2-o.js";function m(t,n){f(t,n.id,{...n.inputs,...n.outputs});let r=Object.entries(n.inputs),o=Object.entries(n.outputs),s=[...r,...o];c(n.id,t,s.map(([,e])=>e));for(let[,t]of s)e(t,[`float32x3`,`uint32`,`float32`],n.id);if(!n.length)return[];let u=s.map(([e,t],n)=>({name:e,view:t,type:t.format===`uint32`?`u32`:`f32`,access:n>=r.length?`read_write`:`read`})),d=n.cooperative?.workgroupSize;d!==void 0&&(a(n.id,Number.isInteger(d)&&d>=2&&!(d&d-1)&&d<=t.device.limits.maxComputeInvocationsPerWorkgroup,`cooperative workgroupSize must be a power of two in [2, ${t.device.limits.maxComputeInvocationsPerWorkgroup}]`),i(n.id,`cooperative rows x workgroupSize`,n.length*d,4294967295));let p=`${h(n)}
${n.declarations??``}
${n.cooperative?.declarations??``}
fn runPointCloudRow(index: u32, localInvocationIndex: u32) {
  ${n.body}
}`;return[l(t,{id:n.id,operation:`GPUPointCloud`,bindings:u,invocationCount:d?n.length*d:n.length,...d?{workgroupSize:d,guardIndex:!1}:{},declarations:p,body:d?`if (workgroupIndex >= ${n.length}u) { return; }
  runPointCloudRow(workgroupIndex, localInvocationIndex);`:`runPointCloudRow(index, localInvocationIndex);`})]}function h(e,t){let n=Object.entries(e.inputs);return`${[...n,...Object.entries(e.outputs)].map(([e,r],i)=>{u(`pointcloud`,`binding`,e);let a=i>=n.length,o=r.format===`uint32`?`u32`:`f32`,s=r.format===`float32x3`,c=s?`vec3f`:o,l=`${t?.[e]??`${e}Offset`} + row * ${s?3:1}u`,d=e[0].toUpperCase()+e.slice(1),f=s?`vec3f(${e}[offset], ${e}[offset + 1u], ${e}[offset + 2u])`:`${e}[offset]`,p=s?`${e}[offset] = value.x; ${e}[offset + 1u] = value.y; ${e}[offset + 2u] = value.z;`:`${e}[offset] = value;`;return`fn read${d}(row: u32) -> ${c} { let offset = ${l}; return ${f}; }
${a?`fn write${d}(row: u32, value: ${c}) { let offset = ${l}; ${p} }`:``}`}).join(`
`)}
${d}
fn getPointCloudQuotient2(value: vec2f, divisor: f32) -> vec2f {
  return vec2f(getScaledQuotient(value.x, divisor, 1.0, 0), getScaledQuotient(value.y, divisor, 1.0, 0));
}
fn getPointCloudQuotient3(value: vec3f, divisor: f32) -> vec3f {
  return vec3f(getScaledQuotient(value.x, divisor, 1.0, 0), getScaledQuotient(value.y, divisor, 1.0, 0), getScaledQuotient(value.z, divisor, 1.0, 0));
}
fn safeNormal(value: vec3f) -> vec3f {
  let scale = max(max(abs(value.x), abs(value.y)), abs(value.z));
  if (!(scale > 0.0) || scale > 3.402823e38) { return vec3f(0.0); }
  let scaled = getPointCloudQuotient3(value, scale);
  let magnitude = length(scaled);
  if (!(magnitude > 0.0)) { return vec3f(0.0); }
  return scaled / magnitude;
}
`}function g(e,n){let r=Object.entries(n).filter(e=>!!e[1]),i=r.map(([,e])=>({...e,format:`uint32`,byteStride:4,rowByteLength:4,length:e.length?((e.length-1)*e.byteStride+e.rowByteLength)/4:0})),a=i.reduce((e,t)=>e+t.length,0);try{p(e,{pointCloudFields:new t({id:`${e}-logical-fields`,name:`logical fields`,format:`uint32`,length:a,valueLength:a,stride:1,byteStride:4,rowByteLength:4,data:i})})}catch(e){throw e instanceof Error?Error(e.message.replace(/pointCloudFields chunk (\d+)/g,(e,t)=>r[Number(t)]?.[0]??t)):e}}function _(e,t,r,i=[{inputs:t,outputs:r}]){for(let[i,a]of Object.entries({...t,...r}))a&&n(a,[a.format],`${e} ${i}`);g(e,r);for(let[n,i]of Object.entries(t))i&&g(e,{...r,[`input ${n}`]:i});for(let t of i){let n=Object.fromEntries(Object.entries(t.outputs).filter(([,e])=>e&&e.length>0));if(!Object.keys(n).length)continue;p(e,n);let r=new Set(Object.values(n).map(e=>v(e)));for(let[n,i]of Object.entries(t.inputs??{}))i&&a(e,!r.has(v(i)),`input ${n} and writable outputs must not share a physical storage buffer in one dispatch`)}}function v(e){return e.buffer.defaultBuffer?r(e.buffer.defaultBuffer):e.buffer}function y(e,t,n){for(let[r,i]of Object.entries(n))o(t,`${r} allocation byteLength`,Math.max(i,4),0,e.device.limits.maxBufferSize),s(e.device,t,r,Math.max(i,4),`reduce the row count or grid dimensions`)}export{y as a,_ as i,m as n,g as r,h as t};