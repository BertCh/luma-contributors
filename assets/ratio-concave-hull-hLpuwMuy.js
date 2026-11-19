import{n as e,o as t,r as n}from"./gpu-delaunay-tessellation-D0udu4Pw.js";import{a as r}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as i}from"./validate-outline-output-passes-DIcqH6Pw.js";import{n as a}from"./refinement-storage-validation-D9BhQGNW.js";import{r as o}from"./kernel-C3tfQOn4.js";import{E as s,f as c,s as l,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{o as d}from"./wgsl-kernel-nodes-DagJklH2.js";import{n as f}from"./triangle-edge-adjacency-DJc9p9_s.js";import{i as p}from"./spatial-wgsl-point-helpers-DcOgpqmI.js";import{n as m}from"./point-observations-CeqP-hIo.js";import{A as h}from"./pointer-overlay-Qh8qflcA.js";import{t as g}from"./define-run-B-brkCe4.js";import{c as _,i as v,p as y,s as b}from"./pointer-support-C2rPRVFO.js";import{s as x}from"./shape-segments-DMcyF4V-.js";import{to as S}from"./entry-preview-D70TdSbK.js";var C=256,w=class{id;props;constructor(e){this.id=e.id??`ratio-concave-hull`,this.props=e;let t=n(e.positions.length);if(!(Number.isFinite(e.ratio)&&e.ratio>=0&&e.ratio<=1))throw Error(`Invalid hull ratio`);c(this.id,`maximumIterations`,e.maximumIterations??t,0),s(e.positions,[`float32x2`],this.id),s(e.output.triangles,[`uint32x3`],this.id),s(e.output.boundaryEdges,[`uint32x2`],this.id);for(let t of[e.output.triangleValidity,e.output.edgeValidity,e.output.status.converged])s(t,[`uint32`],this.id);if(e.output.triangles.length!==t||e.output.triangleValidity.length!==t||e.output.boundaryEdges.length!==t*3||e.output.edgeValidity.length!==t*3||e.output.status.converged.length!==1)throw Error(`Hull output capacity mismatch`);i(this.id,{...e.output,status:void 0,converged:e.output.status.converged},[[`triangleValidity`,`converged`],[`boundaryEdges`,`edgeValidity`],[`triangles`]]),u(this.id,l(e.output),[e.positions])}getCommandNodes(n){let{props:i,id:s}=this,{output:c}=i,u=c.triangles.length;r(n,s,Object.fromEntries([i.positions,...l(c)].map((e,t)=>[`binding.${t}`,e])));let m=t(n.device,s,i.positions.length);a(n,s,{delaunayTopology:m*16,adjacencyAndSort:u*3*4,work:(3*i.positions.length+u+4)*4});let h=o(n,`${s}-count`,`uint32`,1),g=o(n,`${s}-required`,`uint32`,1),_=o(n,`${s}-overflow`,`uint32`,1),v=o(n,`${s}-adjacency`,`uint32`,u*3),y=i.positions.length,b=o(n,`${s}-work`,`uint32`,3*y+u+4),x=[...new e({id:`${s}-delaunay`,positions:i.positions,spatialContext:i.spatialContext,output:{triangles:c.triangles,status:{count:h,requiredCount:g,overflow:_,converged:c.status.converged}}}).getCommandNodes(n)],S=`const CAPACITY: u32 = ${u}u;
fn triangle(row: u32) -> vec3u { return vec3u(triangles[trianglesOffset + row * 3u], triangles[trianglesOffset + row * 3u + 1u], triangles[trianglesOffset + row * 3u + 2u]); }`;return x.push(...f(n,{id:`${s}-adjacency`,operation:`GPURatioConcaveHull`,pointCount:y,triangles:c.triangles,triangleCount:h,adjacency:v}),d(n,{id:`${s}-validity`,operation:`GPURatioConcaveHull`,variant:`initial-validity`,invocationCount:u,bindings:[{name:`count`,view:h,type:`u32`,access:`read`},{name:`validity`,view:c.triangleValidity,type:`u32`,access:`read_write`}],body:`validity[validityOffset + index] = select(0u, 1u, index < count[countOffset]);`}),d(n,{id:`${s}-erode`,operation:`GPURatioConcaveHull`,variant:i.allowHoles?`ratio-erosion-holes`:`ratio-erosion`,invocationCount:C,workgroupSize:C,guardIndex:!1,bindings:[{name:`triangles`,view:c.triangles,type:`u32`,access:`read`},{name:`positions`,view:i.positions,type:`f32`,access:`read`},{name:`count`,view:h,type:`u32`,access:`read`},{name:`adjacency`,view:v,type:`u32`,access:`read`},{name:`validity`,view:c.triangleValidity,type:`u32`,access:`read_write`},{name:`converged`,view:c.status.converged,type:`u32`,access:`read_write`},{name:`work`,view:b,type:`atomic<u32>`,access:`read_write`}],declarations:`${S}
const VERTEX_COUNT: u32 = ${y}u;
const WORKGROUP_SIZE: u32 = ${C}u;
const BOUNDARY_BASE: u32 = 0u;
const MAXIMUM_BASE: u32 = ${y}u;
const WINNER_BASE: u32 = ${2*y}u;
const CANDIDATE_BASE: u32 = ${3*y}u;
const STATE_BASE: u32 = ${3*y+u}u;
const WINNER_MARK: u32 = 0xffffffffu;
override RATIO: f32;
override MAXIMUM_ITERATIONS: u32;
var<workgroup> stopFlag: u32;
var<workgroup> holeMode: u32;
var<workgroup> invalid: atomic<u32>;
var<workgroup> minimumBits: atomic<u32>;
var<workgroup> maximumBits: atomic<u32>;
${p(`point`,`positions`)}
fn isFiniteBits(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
fn edgeLength(current: vec3u, edge: u32) -> f32 { return distance(point(current[edge]), point(current[(edge + 1u) % 3u])); }
fn exposed(row: u32, edge: u32) -> bool {
  let neighbor = adjacency[adjacencyOffset + row * 3u + edge];
  if (neighbor == 0xffffffffu) { return true; }
  return validity[validityOffset + neighbor] == 0u;
}
fn isBoundaryVertex(vertex: u32) -> bool { return atomicLoad(&work[workOffset + BOUNDARY_BASE + vertex]) != 0u; }
fn synchronize() { storageBarrier(); workgroupBarrier(); }
fn offerCandidate(row: u32, current: vec3u, size: f32, threshold: f32) {
  if (size > threshold) {
    let bits = bitcast<u32>(size);
    atomicStore(&work[workOffset + CANDIDATE_BASE + row], bits);
    for (var corner = 0u; corner < 3u; corner++) { atomicMax(&work[workOffset + MAXIMUM_BASE + current[corner]], bits); }
    atomicAdd(&work[workOffset + STATE_BASE + 2u], 1u);
  }
}`,constants:{RATIO:Math.fround(i.ratio),MAXIMUM_ITERATIONS:i.maximumIterations??u},body:`let lane = localInvocationIndex;
let triangleCount = min(count[countOffset], ${u}u);
if (lane == 0u) {
  stopFlag = select(0u, 1u, converged[convergedOffset] == 0u);
  atomicStore(&invalid, 0u); atomicStore(&minimumBits, 0x7f800000u); atomicStore(&maximumBits, 0u);
}
synchronize();
if (workgroupUniformLoad(&stopFlag) == 0u) {
  for (var vertex = lane; vertex < VERTEX_COUNT; vertex += WORKGROUP_SIZE) {
    let value = point(vertex);
    if (!isFiniteBits(value.x) || !isFiniteBits(value.y)) { atomicStore(&invalid, 1u); }
  }
  synchronize();
  for (var row = lane; row < triangleCount; row += WORKGROUP_SIZE) {
    let current = triangle(row);
    for (var edge = 0u; edge < 3u; edge++) {
      let edgeSize = edgeLength(current, edge);
      if (!isFiniteBits(edgeSize)) { atomicStore(&invalid, 1u); } else {
        atomicMin(&minimumBits, bitcast<u32>(edgeSize)); atomicMax(&maximumBits, bitcast<u32>(edgeSize));
      }
    }
  }
  synchronize();
  if (lane == 0u) {
    if (atomicLoad(&invalid) != 0u) { converged[convergedOffset] = 0u; stopFlag = 1u; }
    atomicStore(&work[workOffset + STATE_BASE + 1u], 0u);
  }
  synchronize();
  let minimumLength = bitcast<f32>(atomicLoad(&minimumBits));
  let threshold = minimumLength + RATIO * (bitcast<f32>(atomicLoad(&maximumBits)) - minimumLength);
  loop {
    if (workgroupUniformLoad(&stopFlag) != 0u) { break; }
    for (var vertex = lane; vertex < VERTEX_COUNT; vertex += WORKGROUP_SIZE) {
      atomicStore(&work[workOffset + BOUNDARY_BASE + vertex], 0u);
      atomicStore(&work[workOffset + MAXIMUM_BASE + vertex], 0u);
      atomicStore(&work[workOffset + WINNER_BASE + vertex], 0xffffffffu);
    }
    for (var row = lane; row < triangleCount; row += WORKGROUP_SIZE) { atomicStore(&work[workOffset + CANDIDATE_BASE + row], 0u); }
    if (lane == 0u) { atomicStore(&work[workOffset + STATE_BASE], 0u); atomicStore(&work[workOffset + STATE_BASE + 2u], 0u); }
    synchronize();
    // Count exposed edges per vertex: a vertex with a nonzero count lies on the current boundary.
    for (var row = lane; row < triangleCount; row += WORKGROUP_SIZE) {
      if (validity[validityOffset + row] == 0u) { continue; }
      let current = triangle(row);
      for (var edge = 0u; edge < 3u; edge++) {
        if (exposed(row, edge)) {
          atomicAdd(&work[workOffset + BOUNDARY_BASE + current[edge]], 1u);
          atomicAdd(&work[workOffset + BOUNDARY_BASE + current[(edge + 1u) % 3u]], 1u);
        }
      }
    }
    synchronize();
    // Border candidates: one exposed edge and an interior opposite vertex.
    for (var row = lane; row < triangleCount; row += WORKGROUP_SIZE) {
      if (validity[validityOffset + row] == 0u) { continue; }
      let current = triangle(row);
      var exposedCount = 0u; var exposedEdge = 0u;
      for (var edge = 0u; edge < 3u; edge++) { if (exposed(row, edge)) { exposedCount++; exposedEdge = edge; } }
      if (exposedCount == 1u && !isBoundaryVertex(current[(exposedEdge + 2u) % 3u])) {
        offerCandidate(row, current, edgeLength(current, exposedEdge), threshold);
      }
    }
    synchronize();
    if (lane == 0u) { holeMode = ${i.allowHoles?`select(0u, 1u, atomicLoad(&work[workOffset + STATE_BASE + 2u]) == 0u)`:`0u`}; }
    synchronize();
    if (workgroupUniformLoad(&holeMode) != 0u) {
      // No border candidate remains: seed holes from triangles whose three vertices are interior.
      for (var row = lane; row < triangleCount; row += WORKGROUP_SIZE) {
        if (validity[validityOffset + row] == 0u) { continue; }
        let current = triangle(row);
        if (isBoundaryVertex(current.x) || isBoundaryVertex(current.y) || isBoundaryVertex(current.z)) { continue; }
        offerCandidate(row, current, max(edgeLength(current, 0u), max(edgeLength(current, 1u), edgeLength(current, 2u))), threshold);
      }
      synchronize();
    }
    for (var row = lane; row < triangleCount; row += WORKGROUP_SIZE) {
      let bits = atomicLoad(&work[workOffset + CANDIDATE_BASE + row]);
      if (bits == 0u) { continue; }
      let current = triangle(row);
      if (atomicLoad(&work[workOffset + MAXIMUM_BASE + current.x]) == bits && atomicLoad(&work[workOffset + MAXIMUM_BASE + current.y]) == bits && atomicLoad(&work[workOffset + MAXIMUM_BASE + current.z]) == bits) {
        for (var corner = 0u; corner < 3u; corner++) { atomicMin(&work[workOffset + WINNER_BASE + current[corner]], row); }
      }
    }
    synchronize();
    for (var row = lane; row < triangleCount; row += WORKGROUP_SIZE) {
      if (atomicLoad(&work[workOffset + CANDIDATE_BASE + row]) == 0u) { continue; }
      let current = triangle(row);
      if (atomicLoad(&work[workOffset + WINNER_BASE + current.x]) == row && atomicLoad(&work[workOffset + WINNER_BASE + current.y]) == row && atomicLoad(&work[workOffset + WINNER_BASE + current.z]) == row) {
        atomicStore(&work[workOffset + CANDIDATE_BASE + row], WINNER_MARK);
        atomicAdd(&work[workOffset + STATE_BASE], 1u);
      }
    }
    synchronize();
    if (lane == 0u) {
      let winners = atomicLoad(&work[workOffset + STATE_BASE]);
      let removals = atomicLoad(&work[workOffset + STATE_BASE + 1u]);
      if (winners == 0u) { stopFlag = 1u; }
      else if (removals + winners > MAXIMUM_ITERATIONS) { converged[convergedOffset] = 0u; stopFlag = 1u; }
      else { atomicStore(&work[workOffset + STATE_BASE + 1u], removals + winners); }
    }
    synchronize();
    if (workgroupUniformLoad(&stopFlag) == 0u) {
      for (var row = lane; row < triangleCount; row += WORKGROUP_SIZE) {
        if (atomicLoad(&work[workOffset + CANDIDATE_BASE + row]) == WINNER_MARK) { validity[validityOffset + row] = 0u; }
      }
    }
    synchronize();
  }
}`}),d(n,{id:`${s}-boundary`,operation:`GPURatioConcaveHull`,variant:`boundary`,invocationCount:u*3,bindings:[{name:`triangles`,view:c.triangles,type:`u32`,access:`read`},{name:`adjacency`,view:v,type:`u32`,access:`read`},{name:`validity`,view:c.triangleValidity,type:`u32`,access:`read`},{name:`edges`,view:c.boundaryEdges,type:`u32`,access:`read_write`},{name:`edgeValidity`,view:c.edgeValidity,type:`u32`,access:`read_write`}],declarations:S,body:`let row = index / 3u; let edge = index % 3u; let current = triangle(row);
let neighbor = adjacency[adjacencyOffset + index];
var boundary = validity[validityOffset + row] != 0u;
if (neighbor != 0xffffffffu && validity[validityOffset + neighbor] != 0u) { boundary = false; }
edges[edgesOffset + index * 2u] = current[edge]; edges[edgesOffset + index * 2u + 1u] = current[(edge + 1u) % 3u];
edgeValidity[edgeValidityOffset + index] = select(0u, 1u, boundary);`})),x}},T=1e6,E=[.06,.12,.2,.4,1],D=g({parameters:S,caption:`The boundary of a sample of earthquake epicenters, tightened by the edge-length ratio.`,buildCache:5,createInputs:({seed:e,world:t})=>({seed:e,world:t,guard:v()}),build(e,t,r){let i=h(t.seed),{positions:a}=m(i,t.world,r.eventCount),o=e.upload(`points`,`float32x2`,a),s=n(r.eventCount),c=e.allocate(`boundary-edges`,`uint32x2`,s*3),l=e.allocate(`edge-validity`,`uint32`,s*3);e.graph.add(new w({positions:o.view,ratio:r.ratio,output:{triangles:e.allocate(`triangles`,`uint32x3`,s).view,triangleValidity:e.allocate(`triangle-validity`,`uint32`,s).view,boundaryEdges:c.view,edgeValidity:l.view,status:{converged:e.allocate(`converged`,`uint32`,1).view}}}));let u=e.allocate(`hull`,`float32x4`,s*3);return x(e,{id:`hull-segments`,invocationCount:s*3,bindings:[{name:`points`,buffer:o,access:`read`},{name:`edges`,buffer:c,access:`read`},{name:`validity`,buffer:l,access:`read`},{name:`hull`,buffer:u,access:`read_write`}],body:`let isBoundary = validity[validityOffset + index] != 0u;
  let startNode = edges[edgesOffset + index * 2u];
  let endNode = edges[edgesOffset + index * 2u + 1u];
  let base = hullOffset + index * 4u;
  hull[base] = select(${T}, points[pointsOffset + startNode * 2u], isBoundary);
  hull[base + 1u] = select(${T}, points[pointsOffset + startNode * 2u + 1u], isBoundary);
  hull[base + 2u] = select(${T}, points[pointsOffset + endNode * 2u], isBoundary);
  hull[base + 3u] = select(${T}, points[pointsOffset + endNode * 2u + 1u], isBoundary);`}),{outputs:{points:o,hull:u}}},interaction:{kind:`level`,hint:`tighten the hull`,rest:[b(E.indexOf(.2),1,E.length),.5],resolve(e,{inputs:t,outputs:n,values:r,world:i,peek:a}){let o=_(e.unitX,1,E.length),s=E[o],c=y(t.guard,String(r.ratio),a(n.hull)),l=0;if(c)for(let e=0;e<c.length;e+=4)c[e]<T/2&&l++;let u=.05,[d,f]=[i.width*.1,i.width*.9],p=[{shape:`line`,from:[d,u],to:[f,u]}];return E.forEach((e,t)=>{let n=d+b(t,1,E.length)/1*(f-d);p.push({shape:`line`,from:[n,.035],to:[n,.065]})}),p.push({shape:`dot`,at:[d+b(o,1,E.length)*(f-d),u]}),{settle:{ratio:s},glyphs:p,readout:s===r.ratio?`ratio ${s.toFixed(2)} · ${c?l:`...`} boundary edges`:`ratio ${r.ratio.toFixed(2)} -> ${s.toFixed(2)}`}}},draw:({outputs:e})=>[{layer:`dots`,context:!0,positions:e.points,radius:3},{layer:`lines`,segments:e.hull,color:[1,.6,.2,1],radius:2.4}]});export{D as default};