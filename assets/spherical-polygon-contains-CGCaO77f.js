import{c as e,o as t,r as n,t as r}from"./kernel-C3tfQOn4.js";import{t as i}from"./assert-unit-vector-BIr1r7vX.js";import{l as a,r as o}from"./gpu-command-graph-BdzdmV6u.js";import{t as s}from"./gpu-dispatch-utils-CbRzFAgT.js";import{c,o as l}from"./wgsl-kernel-nodes-DagJklH2.js";import{r as u}from"./spatial-wgsl-point-helpers-DcOgpqmI.js";import{E as d,S as f,T as p,p as m,v as h,y as g}from"./gpu-contributor-utils-BIwLQON5.js";import{a as _,i as v,n as y,o as b,r as x}from"./graph-chunk-dispatch-zBt8k2-o.js";import{v as S}from"./spatial-context-DwZEd4EZ.js";import{a as C,c as w,i as T,n as E,s as D}from"./spherical-limits-C7rAYBu5.js";import{t as O}from"./lightning-C0gvbE9a.js";import{t as k}from"./define-run-B-brkCe4.js";import{d as A,s as j}from"./pointer-overlay-Qh8qflcA.js";import{i as M}from"./polygons--0zAAHu9.js";import{a as N,i as P,o as F,t as I}from"./spherical-surveys-Vp2aL65w.js";import{i as L}from"./pointer-CXys5Ipi.js";import{Xc as R}from"./entry-preview-D70TdSbK.js";var z=8,B=256,V=16777216,H=class{props;id;estimatedCommandNodeCount;chunkPlan;chunked;spatialContext;constructor(e){this.props=e,this.id=e.id??`spherical-polygon-contains`,this.spatialContext=S(this.id,e.spatialContext??E,D),e.hemisphereCenter&&i(e.hemisphereCenter,this.id,`hemisphereCenter`),d(e.positions,[`float32x3`],`${this.id} positions`);for(let[t,n]of Object.entries({ringOffsets:e.ringOffsets,polygonOffsets:e.polygonOffsets}))d(n,[`uint32`],`${this.id} ${t}`);this.chunkPlan=x(this.id,{queries:e.queries,polygonIndices:e.polygonIndices,contains:e.output.contains,status:e.output.status});for(let e of this.chunkPlan.chunks){d(e.views.queries,[`float32x3`],`${this.id} queries`);for(let t of[`polygonIndices`,`contains`,`status`])d(e.views[t],[`uint32`],`${this.id} ${t}`)}v(this.id,e.output),this.chunked=[e.queries,e.polygonIndices,...Object.values(e.output)].some(e=>e instanceof a),this.estimatedCommandNodeCount=this.chunked?3+this.chunkPlan.nonemptyChunkCount*2:3,m(this.id,this.estimatedCommandNodeCount),p(this.id,e.ringOffsets.length>=1&&e.polygonOffsets.length>=1,`ringOffsets and polygonOffsets must hold at least one entry`),p(this.id,e.queries.length===e.polygonIndices.length&&e.output.contains.length===e.queries.length&&e.output.status.length===e.queries.length,`polygonIndices, output.contains and output.status must hold one entry per query`),w(this.id,e.tolerance??1e-6,C,T),g(this.id,Object.values(e.output),[e.positions,e.ringOffsets,e.polygonOffsets,e.queries,e.polygonIndices])}getCommandNodes(e){let{props:t,id:r}=this;_(e,r,{positions:t.positions,ringOffsets:t.ringOffsets,polygonOffsets:t.polygonOffsets,queries:t.queries,polygonIndices:t.polygonIndices,...t.output}),this.chunked&&b(e,r,this.chunkPlan);let i=this.chunkPlan.chunks.reduce((e,t)=>Math.max(e,t.length),0),a=Math.max(t.ringOffsets.length-1,1);h(e.device,r,`ringCenters`,a*12,`reduce the packed ring count`),h(e.device,r,`ringFailures`,a*4,`reduce the packed ring count`),this.chunked&&h(e.device,r,`chunk results`,Math.max(i,1)*8,`reduce the query chunk size`),f(r,`storage bindings`,z,e.device.limits.maxStorageBuffersPerShaderStage);let o=this.chunkPlan.chunks.map(t=>s(r,t.length,B,e.device.limits.maxComputeWorkgroupsPerDimension)),d=c(t.tolerance??1e-6),p=t.ringOffsets.length-1,m=n(e,`${r}-ring-centers`,`float32x3`,Math.max(p,1)),g=n(e,`${r}-ring-failures`,`uint32`,Math.max(p,1)),v=`const POLYGON_TOLERANCE: f32 = ${d};
${u(`polygonPosition`,`positions`)}
fn polygonUnit(point: vec3f) -> bool { let squared = dot(point, point); return (bitcast<u32>(squared) & 0x7f800000u) != 0x7f800000u && abs(squared - 1.0) <= 1e-4; }
fn polygonOnArc(first: vec3f, second: vec3f, point: vec3f, margin: f32) -> bool {
  let normal = normalize(cross(first, second));
  return abs(dot(normal, point)) <= margin && dot(cross(first, point), normal) >= -margin && dot(cross(point, second), normal) >= -margin;
}`,x=[l(e,{id:`${r}-ring-center`,operation:`GPUSphericalPolygonContains`,variant:`ring-center`,invocationCount:Math.max(p,1),bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`ringOffsets`,view:t.ringOffsets,type:`u32`,access:`read`},{name:`ringCenters`,view:m,type:`f32`,access:`read_write`},{name:`ringFailures`,view:g,type:`atomic<u32>`,access:`read_write`}],declarations:v,body:`if (index >= ${p}u) { return; }
let begin = ringOffsets[ringOffsetsOffset + index]; let end = ringOffsets[ringOffsetsOffset + index + 1u];
var failure = 0u; var center = vec3f(0.0);
if (begin > end || end > ${t.positions.length}u || end - begin < 3u) { failure = 1u; }
else {
  for (var vertex = begin; vertex < end; vertex++) { let value = polygonPosition(vertex); if (!polygonUnit(value)) { failure = 1u; } center += value; }
  // The supplied ring represents its smaller region; complement is explicit at polygon level.
  ${t.hemisphereCenter?`center = vec3f(${t.hemisphereCenter.join(`,`)});`:``}
  if (length(center) <= POLYGON_TOLERANCE) { failure = 1u; } else { center = normalize(center); }
}
ringCenters[ringCentersOffset + index * 3u] = center.x; ringCenters[ringCentersOffset + index * 3u + 1u] = center.y; ringCenters[ringCentersOffset + index * 3u + 2u] = center.z;
atomicStore(&ringFailures[ringFailuresOffset + index], failure);`}),l(e,{id:`${r}-ring-edges`,operation:`GPUSphericalPolygonContains`,variant:`ring-edges`,invocationCount:Math.max(t.positions.length,1),bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`ringOffsets`,view:t.ringOffsets,type:`u32`,access:`read`},{name:`ringCenters`,view:m,type:`f32`,access:`read`},{name:`ringFailures`,view:g,type:`atomic<u32>`,access:`read_write`}],declarations:v,body:`if (index >= ${t.positions.length}u || ${p}u == 0u) { return; }
// Locate the ring that owns this vertex by binary search over the ring offsets.
var low = 0u; var high = ${p}u;
for (var step = 0u; step < 32u && low + 1u < high; step++) {
  let middle = (low + high) / 2u;
  if (ringOffsets[ringOffsetsOffset + middle] <= index) { low = middle; } else { high = middle; }
}
let ring = low;
let begin = ringOffsets[ringOffsetsOffset + ring]; let end = ringOffsets[ringOffsetsOffset + ring + 1u];
if (index < begin || index >= end || atomicLoad(&ringFailures[ringFailuresOffset + ring]) != 0u) { return; }
let center = vec3f(ringCenters[ringCentersOffset + ring * 3u], ringCenters[ringCentersOffset + ring * 3u + 1u], ringCenters[ringCentersOffset + ring * 3u + 2u]);
let edge = index;
let next = select(edge + 1u, begin, edge + 1u == end);
let first = polygonPosition(edge); let second = polygonPosition(next);
let normal = cross(first, second);
var failure = 0u;
if (length(normal) <= POLYGON_TOLERANCE || dot(center, first) <= POLYGON_TOLERANCE) { failure = 1u; }
// Reject crossing/touching nonadjacent arcs instead of assigning an invalid ring a fill.
for (var other = begin; other < end && failure == 0u; other++) {
  let otherNext = select(other + 1u, begin, other + 1u == end);
  if (other == edge || other == next || otherNext == edge) { continue; }
  let third = polygonPosition(other); let fourth = polygonPosition(otherNext);
  let crossing = cross(normal, cross(third, fourth));
  if (length(crossing) > POLYGON_TOLERANCE * POLYGON_TOLERANCE) {
    let intersection = normalize(crossing);
    if ((polygonOnArc(first, second, intersection, POLYGON_TOLERANCE) && polygonOnArc(third, fourth, intersection, POLYGON_TOLERANCE)) || (polygonOnArc(first, second, -intersection, POLYGON_TOLERANCE) && polygonOnArc(third, fourth, -intersection, POLYGON_TOLERANCE))) { failure = 1u; }
  } else if (polygonOnArc(first, second, third, POLYGON_TOLERANCE) || polygonOnArc(first, second, fourth, POLYGON_TOLERANCE)) { failure = 1u; }
}
if (failure != 0u) { atomicMax(&ringFailures[ringFailuresOffset + ring], 1u); }`})];if(!this.chunked)return[...x,l(e,this.getQueryKernelProps(t,g,v))];let{slots:S,nodes:C}=y(e,{id:`${r}-chunks`,plan:this.chunkPlan}),w=n(e,`${r}-chunk-results`,`uint32x2`,i),T=[];for(let e of this.chunkPlan.chunks){if(e.length===0)continue;let n={...t,queries:e.views.queries,polygonIndices:e.views.polygonIndices,output:{contains:e.views.contains,status:e.views.status}},i=this.getQueryKernelProps(n,g,v,w);T.push(this.createChunkQueryNode(`${r}-query-${e.chunkIndex}`,S[e.chunkIndex],o[e.chunkIndex],i,{queries:3,polygonIndices:1,results:2})),T.push(this.createChunkQueryNode(`${r}-scatter-${e.chunkIndex}`,S[e.chunkIndex],o[e.chunkIndex],{id:`${r}-scatter`,operation:`GPUSphericalPolygonContains`,variant:`scatter`,invocationCount:e.length,bindings:[{name:`results`,view:w,type:`u32`,access:`read`},{name:`contains`,view:n.output.contains,type:`u32`,access:`read_write`},{name:`status`,view:n.output.status,type:`u32`,access:`read_write`}],body:`contains[containsOffset + index] = results[resultsOffset + index * 2u];
status[statusOffset + index] = results[resultsOffset + index * 2u + 1u];`},{results:2,contains:1,status:1}))}return[...x,...C,...T]}createChunkQueryNode(n,i,a,s,c){let l=`${s.bindings.map((t,n)=>`@group(0) @binding(${n}) var<storage, ${t.access}> ${t.name}: array<${t.type}>;\n`+(t.name in c?`var<private> ${t.name}Offset: u32;`:`const ${t.name}Offset: u32 = ${e(t.view)}u;`)).join(`
`)}
@group(0) @binding(${s.bindings.length}) var<storage, read> chunkParameters: array<u32>;
${s.declarations??``}
@compute @workgroup_size(${B})
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(num_workgroups) workgroupCount: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32) {
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${V}u) { return; }
  let index = workgroupIndex * ${B}u + localInvocationIndex;
  if (index >= chunkParameters[1]) { return; }
  ${Object.entries(c).map(([e,t])=>`${e}Offset = arrayLength(&${e}) - chunkParameters[1] * ${t}u;`).join(`
  `)}
  ${s.body}
}`;return o({id:n,workload:{operation:s.operation,...s.variant?{variant:s.variant}:{},commandCount:1,maximumInvocationCount:s.invocationCount,maximumWorkgroupCount:a.x*a.y*a.z},resources:[...s.bindings.map(e=>({buffer:e.view,usage:e.access===`read`?`storage-read`:`storage-read-write`})),{buffer:i,usage:`storage-read`}],compile:({device:e})=>{let n=new r(e,{id:s.id,source:l,shaderLayout:{bindings:[...s.bindings.map((e,t)=>({name:e.name,group:0,location:t,type:e.access===`read`?`read-only-storage`:`storage`})),{name:`chunkParameters`,group:0,location:s.bindings.length,type:`read-only-storage`}]}});return{encode:({computePass:e,getBuffer:r})=>n.dispatch(e,{bindings:Object.fromEntries([...s.bindings.map(e=>[e.name,t(e.view,r)]),[`chunkParameters`,t(i,r)]]),...a}),destroy:()=>n.destroy()}}})}getQueryKernelProps(e,t,n,r){let{id:i}=this;return{id:i,operation:`GPUSphericalPolygonContains`,variant:`${e.boundary??`closed`}-${!!e.complement}`,invocationCount:e.queries.length,bindings:[{name:`positions`,view:e.positions,type:`f32`,access:`read`},{name:`ringOffsets`,view:e.ringOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:e.polygonOffsets,type:`u32`,access:`read`},{name:`queries`,view:e.queries,type:`f32`,access:`read`},{name:`polygonIndices`,view:e.polygonIndices,type:`u32`,access:`read`},...r?[{name:`results`,view:r,type:`u32`,access:`read_write`}]:[{name:`contains`,view:e.output.contains,type:`u32`,access:`read_write`},{name:`status`,view:e.output.status,type:`u32`,access:`read_write`}],{name:`ringFailures`,view:t,type:`u32`,access:`read`}],declarations:n,body:`let queryOffset = queriesOffset + index * 3u;
let point = vec3f(queries[queryOffset], queries[queryOffset + 1u], queries[queryOffset + 2u]);
let polygon = polygonIndices[polygonIndicesOffset + index];
var failure = 0u; var inside = false; var boundary = false;
if (!polygonUnit(point) || polygon >= ${e.polygonOffsets.length-1}u) { failure = 1u; }
if (failure == 0u) {
  let firstRing = polygonOffsets[polygonOffsetsOffset + polygon]; let lastRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
  if (firstRing > lastRing || lastRing >= ${e.ringOffsets.length}u) { failure = 1u; }
  for (var ring = firstRing; ring < lastRing && failure == 0u; ring++) {
    // Ring validity (range, unit vectors, hemisphere, arc crossings) is precomputed per ring.
    if (ringFailures[ringFailuresOffset + ring] != 0u) { failure = 1u; break; }
    let begin = ringOffsets[ringOffsetsOffset + ring]; let end = ringOffsets[ringOffsetsOffset + ring + 1u];
    var winding = 0.0; var ringBoundary = false;
    for (var edge = begin; edge < end; edge++) {
      let next = select(edge + 1u, begin, edge + 1u == end);
      let first = polygonPosition(edge); let second = polygonPosition(next);
      let normal = cross(first, second);
      ringBoundary = ringBoundary || polygonOnArc(first, second, point, POLYGON_TOLERANCE);
      let numerator = dot(point, normal); let denominator = dot(first, second) - dot(point, first) * dot(point, second);
      if (abs(numerator) + abs(denominator) > 1e-12) { winding += atan2(numerator, denominator); }
    }
    let ringInside = winding > 3.141592653589793;
    if (ring == firstRing) { inside = ringInside; } else { inside = inside && !ringInside; }
    boundary = boundary || ringBoundary;
  }
}
var accepted = ${e.complement?`!inside`:`inside`};
if (boundary && failure == 0u) { failure = 2u; accepted = ${e.boundary!==`open`}; }
${r?`results[resultsOffset + index * 2u]`:`contains[containsOffset + index]`} = select(0u, 1u, accepted && failure != 1u);
${r?`results[resultsOffset + index * 2u + 1u]`:`status[statusOffset + index]`} = failure;`}}},U=12,W=j(R,`exclusionLongitude`),G=j(R,`size`),K=k({parameters:R,geographicWindow:I,caption:`GOES lightning detections inside Lake Michigan, excluding an adjustable survey zone. The mapped exterior omits islands.`,createInputs:()=>{let e=O({width:16,height:10},4096),[t,n,r,i]=I,a=[];for(let o=0;o<e.count;o++){let s=e.positions[o*2]-99,c=e.positions[o*2+1]+36;s>=t&&s<=r&&c>=n&&c<=i&&a.push([s,c])}return{queries:a,shore:P(`Lake Michigan`)}},build(e,{queries:t,shore:n},r){let i=n.length+U,a=e.allocate(`positions`,`float32x3`,i),o=e.allocate(`exclusion-outline`,`float32x4`,U),s=e.upload(`shore-outline`,`float32x4`,N([n])),c=e.allocate(`contains`,`uint32`,t.length);return e.graph.add(new H({positions:a.view,ringOffsets:e.upload(`ring-offsets`,`uint32`,Uint32Array.of(0,n.length,i)).view,polygonOffsets:e.upload(`polygon-offsets`,`uint32`,Uint32Array.of(0,2)).view,queries:e.upload(`queries-xyz`,`float32x3`,M([t])).view,polygonIndices:e.upload(`polygon-indices`,`uint32`,new Uint32Array(t.length)).view,complement:r.complement===1,output:{contains:c.view,status:e.allocate(`status`,`uint32`,t.length).view}})),{outputs:{contains:c,exclusionOutline:o,shoreOutline:s,queries:e.upload(`queries`,`float32x2`,Float32Array.from(t.flat()))},update:e=>{let t=F(e.exclusionLongitude,e.size,e.holeSize,U),r=[n,t];a.write(M(r)),o.write(N([t]))}}},rebuildOn:[`complement`],interaction:{kind:`place`,hint:`place the exclusion`,rest:({toUnit:e})=>e([W.default,G.default]),resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=A(W,e.x),o=A(G,e.y),s=i(n.contains),c=0;if(s)for(let e of s)c+=+(e>0);return{values:{exclusionLongitude:a,size:o},glyphs:[{shape:`crosshair`,at:[a,o],size:6}],readout:`hole ${L(a,o)} · ${r.holeSize} km · ${s?c:`…`} inside`}}},draw:({outputs:e,theme:t})=>[{layer:`lines`,context:!0,segments:e.shoreOutline,radius:1.7,color:[...t.ink,.85]},{layer:`dots`,positions:e.queries,values:e.contains,radius:5},{layer:`lines`,context:!0,segments:e.exclusionOutline,radius:4.2,color:[.98,.53,.22,1]}]});export{K as default};