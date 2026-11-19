import{r as e}from"./gpu-command-graph-BdzdmV6u.js";import{t}from"./gpu-dispatch-utils-CbRzFAgT.js";import{o as n,t as r}from"./kernel-C3tfQOn4.js";import{a as i,i as a,n as o,o as s,r as c}from"./graph-chunk-dispatch-zBt8k2-o.js";import{E as l,S as u,T as d,c as f,f as p,p as m,y as h}from"./gpu-contributor-utils-BIwLQON5.js";import{s as g}from"./cell-index-wgsl-Cw7syfdA.js";import{t as _}from"./h3-neighbor-wgsl-DdYw-iG5.js";import{i as v}from"./gpu-point-to-cell-CYNSElfx.js";var y=256,b=4294967295,x=16777216,S=class{props;id;estimatedCommandNodeCount;chunkPlan;recordChunks;constructor(e){this.props=e,this.id=e.id??`h3-cell-identities`;let t=this.id;d(t,[`directed-edges`,`vertices`].includes(e.operation),`operation must be 'directed-edges' or 'vertices'`),this.chunkPlan=c(t,{cells:e.cells}),p(t,`cells row count`,this.chunkPlan.totalLength,0);for(let e of this.chunkPlan.chunks)e.views.cells&&l(e.views.cells,[`uint32x2`],`${t} cells chunk ${e.chunkIndex}`),u(t,`cells chunk components`,e.length*2,b);let n={};if(n.identities=e.output?f(e.output):[],e.output){d(t,e.output.length===this.chunkPlan.totalLength*6,`output length must equal input rows * record stride`),d(t,n.identities.length===this.chunkPlan.chunks.length,`output must match input chunk topology`);for(let e of this.chunkPlan.chunks){let r=n.identities[e.chunkIndex];l(r,[`uint32x2`],`${t} identities chunk ${e.chunkIndex}`),d(t,r.length===e.length*6,`output chunk ${e.chunkIndex} length must equal input chunk rows * record stride`),u(t,`identities chunk components`,r.length*2,b)}}this.recordChunks=n,a(t,{output:e.output}),this.estimatedCommandNodeCount=+(this.chunkPlan.chunks.length>0)+this.chunkPlan.nonemptyChunkCount,m(t,this.estimatedCommandNodeCount),h(this.id,[e.output],[e.cells])}getCommandNodes(a){let{id:c,props:l}=this;u(c,`storage bindings`,3,a.device.limits.maxStorageBuffersPerShaderStage),s(a,c,this.chunkPlan),i(a,c,{output:l.output});let d=this.chunkPlan.chunks.map(e=>t(c,Math.max(e.length*6,1),y,a.device.limits.maxComputeWorkgroupsPerDimension)),{slots:f,nodes:p}=o(a,{id:`${c}-chunks`,plan:this.chunkPlan}),m,h=this.chunkPlan.chunks.filter(e=>e.length>0).map(t=>{let i={id:c,operation:`GPUH3CellIdentities`,variant:l.operation,invocationCount:t.length*6,bindings:[{name:`chunkParameters`,view:f[t.chunkIndex],type:`u32`,access:`read`},{name:`cells`,view:t.views.cells,type:`u32`,access:`read`},{name:`output`,view:this.recordChunks.identities[t.chunkIndex],type:`u32`,access:`read_write`}],declarations:`${g.source}\n${_}\n${C}`,body:`let row = index / 6u; let slot = index % 6u;
  let cell = vec2u(cells[cellsOffset + row * 2u + 1u], cells[cellsOffset + row * 2u]);
  var key = vec2u(0u);
  if (dggs_h3_is_valid_cell_id(cell)) {
    ${l.operation===`vertices`?`key = identityVertex(cell, slot);`:`let neighbor = cellTopologyH3Neighbor(cell, slot + 1u);
    if (any(neighbor != vec2u(0u))) { key = vec2u((cell.x & 0x80ffffffu) | 0x10000000u | ((slot + 1u) << 24u), cell.y); }`}
  }
  output[outputOffset + index * 2u] = key.y;
  output[outputOffset + index * 2u + 1u] = key.x;`},a=i.bindings.map((e,t)=>`@group(0) @binding(${t}) var<storage, ${e.access}> ${e.name}: array<${e.type}>;`).join(`
`),o={cells:2,output:12},s=i.bindings.filter(e=>e.name!==`chunkParameters`).map(e=>`let ${e.name}Offset = arrayLength(&${e.name}) - rowCount * ${o[e.name]}u;`).join(`
  `),u=m??`${a}
${i.declarations??``}
@compute @workgroup_size(${y})
fn main(@builtin(workgroup_id) workgroupId: vec3u,
  @builtin(num_workgroups) workgroupCount: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32) {
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${x}u) { return; }
  let index = workgroupIndex * ${y}u + localInvocationIndex;
  let rowCount = chunkParameters[1];
  if (index >= rowCount * 6u) { return; }
  ${s}
  ${i.body}
}`;m=u;let p=i.bindings,h=i.operation,v=d[t.chunkIndex];return e({id:`${i.id}-${t.chunkIndex}`,resources:i.bindings.map(e=>({buffer:e.view,usage:e.access===`read`?`storage-read`:`storage-read-write`})),workload:{operation:i.operation,...i.variant?{variant:i.variant}:{},commandCount:1,maximumWorkgroupCount:v.x*v.y*v.z,maximumInvocationCount:t.length*6},compile:({device:e})=>{let t=new r(e,{id:h,source:u,shaderLayout:{bindings:p.map((e,t)=>({name:e.name,group:0,location:t,type:e.access===`read`?`read-only-storage`:`storage`}))}});return{encode:({computePass:e,getBuffer:r})=>t.dispatch(e,{bindings:Object.fromEntries(p.map(e=>[e.name,n(e.view,r)])),...v}),destroy:()=>t.destroy()}}})});return[...p,...h]}},C=`
const IDENTITY_FACE_BASE: array<i32, 540> = array<i32, 540>(${v.join(`,`)});
const IDENTITY_PENTAGONS: array<u32, 12> = array<u32,12>(4u,14u,24u,38u,49u,58u,63u,72u,83u,97u,107u,117u);
const IDENTITY_PENTAGON_FACES: array<u32, 60> = array<u32,60>(
4u,0u,2u,1u,3u, 6u,11u,2u,7u,1u, 5u,10u,1u,6u,0u, 7u,12u,3u,8u,2u,
9u,14u,0u,5u,4u, 8u,13u,4u,9u,3u, 11u,6u,15u,10u,16u, 12u,7u,16u,11u,17u,
10u,5u,19u,14u,15u, 13u,8u,17u,12u,18u, 14u,9u,18u,13u,19u, 15u,19u,17u,18u,16u);
fn identityLess(first: vec2u, second: vec2u) -> bool { return first.x < second.x || (first.x == second.x && first.y < second.y); }
fn identityRotations(cell: vec2u) -> u32 {
  let face = dggs_h3_get_center_face_ijk(cell).face;
  let base = dggs_h3_get_base_cell(cell);
  var rotations = 0u;
  for (var slot = 0u; slot < 27u; slot++) {
    let packed = IDENTITY_FACE_BASE[face * 27u + slot];
    if (packed >= 0 && (u32(packed) & 255u) == base) { rotations = u32(packed) >> 8u; break; }
  }
  if (dggs_h3_is_base_cell_pentagon(base)) {
    var pentagon = 0u;
    for (var slot = 0u; slot < 12u; slot++) { if (IDENTITY_PENTAGONS[slot] == base) { pentagon = slot; break; } }
    let faceIK = IDENTITY_PENTAGON_FACES[pentagon * 5u + 3u];
    let faceJK = IDENTITY_PENTAGON_FACES[pentagon * 5u + 1u];
    if (face != dggs_h3_get_base_cell_home(base).face && (base == 4u || base == 117u || face == faceIK)) { rotations = (rotations + 1u) % 6u; }
    let leading = dggs_h3_get_leading_non_zero_digit(cell);
    if (leading == 3u && face == faceIK) { rotations = (rotations + 5u) % 6u; }
    else if (leading == 5u && face == faceJK) { rotations = (rotations + 1u) % 6u; }
  }
  return rotations;
}
fn identityDirection(cell: vec2u, vertex: u32) -> u32 {
  let hexagon = array<u32,6>(6u,2u,3u,1u,5u,4u);
  let pentagon = array<u32,5>(6u,2u,3u,5u,4u);
  let rotation = identityRotations(cell);
  if (cellTopologyH3IsPentagon(cell)) { return pentagon[(vertex + rotation) % 5u]; }
  return hexagon[(vertex + rotation) % 6u];
}
fn identityVertexNumber(cell: vec2u, direction: u32) -> u32 {
  let hexagon = array<u32,7>(0u,3u,1u,2u,5u,4u,0u);
  let pentagon = array<u32,7>(0u,0u,1u,2u,4u,3u,0u);
  let rotation = identityRotations(cell);
  if (cellTopologyH3IsPentagon(cell)) { return (pentagon[direction] + 10u - rotation) % 5u; }
  return (hexagon[direction] + 6u - rotation) % 6u;
}
fn identityReverseDirection(cell: vec2u, destination: vec2u) -> u32 {
  for (var direction = 1u; direction <= 6u; direction++) {
    if (all(cellTopologyH3Neighbor(cell, direction) == destination)) { return direction; }
  }
  return 0u;
}
fn identityVertex(cell: vec2u, vertex: u32) -> vec2u {
  let count = select(6u, 5u, cellTopologyH3IsPentagon(cell));
  if (vertex >= count) { return vec2u(0u); }
  let left = cellTopologyH3Neighbor(cell, identityDirection(cell, vertex));
  let right = cellTopologyH3Neighbor(cell, identityDirection(cell, (vertex + count - 1u) % count));
  var owner = cell; var ownerVertex = vertex;
  if (identityLess(left, owner)) {
    owner = left;
    let ownerCount = select(6u, 5u, cellTopologyH3IsPentagon(owner));
    ownerVertex = (identityVertexNumber(owner, identityReverseDirection(owner, cell)) + 1u) % ownerCount;
  }
  if (identityLess(right, owner)) {
    owner = right; ownerVertex = identityVertexNumber(owner, identityReverseDirection(owner, cell));
  }
  return vec2u((owner.x & 0x80ffffffu) | 0x20000000u | (ownerVertex << 24u), owner.y);
}`;export{C as n,S as t};