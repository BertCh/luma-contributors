import{t as e}from"./ports-xmdKgL07.js";import{r as t,u as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-scan-B4NNTfF9.js";import{o as i}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as a,S as o,f as s,w as c,x as l}from"./gpu-contributor-utils-BIwLQON5.js";import{n as u,r as d,t as f}from"./gpu-line-split-BIIctsOh.js";import{i as p}from"./segment-table-BeVxjLqW.js";import{n as m}from"./topology-types-BLorEZd4.js";import{v as h}from"./spatial-context-DwZEd4EZ.js";var g=`GPULineNoding`;function _(e,t,n,r,i){o(e,`lines.positions rows`,n,p(t)),o(e,`lines + 4 x intersectionCapacity`,r+4*i,d(t))}var v=class{id;props;spatialContext;constructor(t){this.id=t.id??`line-noding`,this.spatialContext=h(this.id,t.spatialContext),t={...t,lines:e(this.id,`lines`,t.lines)},this.props=t;let{id:r}=this,{lines:i,output:o}=t;if(i.kind!==`lines`)throw Error(`${r} lines must be linestring geometry`);if(a(i.positions,[`float32x2`],`${r} lines.positions`),n(i.lineOffsets,`${r} lines.lineOffsets`),i.lineOffsets.length<2||i.positions.length<1)throw Error(`${r} lines require at least one line and one vertex`);s(r,`intersectionCapacity`,t.intersectionCapacity),t.leafCapacity!==void 0&&c(r,`leafCapacity`,t.leafCapacity);let l=o.endpoints.length;a(o.endpoints,[`float32x4`],`${r} output.endpoints`);for(let[e,t]of[[`sourceFeatureIds`,o.sourceFeatureIds],[`sourceRingIds`,o.sourceRingIds],[`sourceEdgeIds`,o.sourceEdgeIds]])if(n(t,`${r} output.${e}`),t.length!==l)throw Error(`${r} output.${e} length must equal output.endpoints length`);for(let[e,t]of[[`sourceStartParameters`,o.sourceStartParameters],[`sourceEndParameters`,o.sourceEndParameters]])if(a(t,[`float32`],`${r} output.${e}`),t.length!==l)throw Error(`${r} output.${e} length must equal output.endpoints length`);for(let[e,t]of Object.entries(o.status))if(t&&(n(t,`${r} output.status.${e}`),t.length<1))throw Error(`${r} output.status.${e} must contain one row`);if(!o.status.requiredCount||!o.status.candidateOverflow)throw Error(`${r} output.status requires requiredCount and candidateOverflow`)}getCommandNodes(e){let{id:n,props:a}=this,{lines:o,output:s}=a;l(n,e,[o.positions,o.lineOffsets,o.sourceIds,s.endpoints,s.sourceFeatureIds,s.sourceRingIds,s.sourceEdgeIds,s.sourceStartParameters,s.sourceEndParameters,...Object.values(s.status)]),_(n,e.device,o.positions.length,o.lineOffsets.length-1,a.intersectionCapacity);let c=a.intersectionCapacity,d=o.lineOffsets.length-1+c*4,p=o.positions.length+c*8,h=(r,i,a)=>t(e,`${n}-${r}`,i,a),v=h(`piece-positions`,`float32x2`,p),y=h(`piece-offsets`,`uint32`,d+1),b=h(`piece-sources`,`uint32`,d),x=h(`piece-count`,`uint32`,1),S=h(`piece-required-count`,`uint32`,1),C=h(`piece-vertex-count`,`uint32`,1),w=h(`piece-required-vertex-count`,`uint32`,1),T=h(`piece-source-edges`,`uint32`,p),E=h(`split-incomplete`,`uint32`,1),D=[];D.push(...new f({id:`${n}-split`,lines:o,intersectionCapacity:c,spatialSort:a.spatialSort,leafCapacity:a.leafCapacity,output:{status:{uncertainCount:s.status.uncertainCount},pieces:{geometry:{kind:`lines`,positions:v,lineOffsets:y},sourceIds:b,status:{count:x,requiredCount:S,overflow:E,candidateOverflow:s.status.candidateOverflow,vertexCount:C,requiredVertexCount:w},vertexSourceEdges:T}}}).getCommandNodes(e));let O=h(`segment-flags`,`uint32`,p),k=h(`segment-scan`,`uint32`,p),A=[{name:`pieceOffsets`,view:y,type:`u32`,access:`read`},{name:`pieceCount`,view:x,type:`u32`,access:`read`},{name:`pieceVertexCount`,view:C,type:`u32`,access:`read`}],j=`
fn pieceOfVertex(vertex: u32) -> u32 {
  let count = pieceCount[pieceCountOffset];
  var low = 0u;
  var high = count;
  while (low + 1u < high) {
    let middle = (low + high) / 2u;
    if (pieceOffsets[pieceOffsetsOffset + middle] <= vertex) { low = middle; } else { high = middle; }
  }
  return low;
}`;D.push(i(e,{id:`${n}-segment-flags`,operation:g,variant:`segment-flags`,bindings:[...A,{name:`piecePositions`,view:v,type:`f32`,access:`read`},{name:`flags`,view:O,type:`u32`,access:`read_write`}],invocationCount:p,declarations:j,body:`var valid = false;
  if (index < pieceVertexCount[pieceVertexCountOffset] && pieceCount[pieceCountOffset] > 0u) {
    let piece = pieceOfVertex(index);
    valid = index + 1u < pieceOffsets[pieceOffsetsOffset + piece + 1u];
    if (valid) {
      let base = piecePositionsOffset + index * 2u;
      valid = piecePositions[base] != piecePositions[base + 2u] || piecePositions[base + 1u] != piecePositions[base + 3u];
    }
  }
  flags[flagsOffset + index] = select(0u, 1u, valid);`}),...new r({id:`${n}-segment-scan`,input:O,output:k,mode:`inclusive`}).getCommandNodes(e));let M=s.endpoints.length,N=o.sourceIds?[{name:`inputSourceIds`,view:o.sourceIds,type:`u32`,access:`read`}]:[];return D.push(i(e,{id:`${n}-initialize`,operation:g,variant:`initialize`,bindings:[{name:`endpoints`,view:s.endpoints,type:`f32`,access:`read_write`},{name:`sourceFeatureIds`,view:s.sourceFeatureIds,type:`u32`,access:`read_write`},{name:`sourceRingIds`,view:s.sourceRingIds,type:`u32`,access:`read_write`},{name:`sourceEdgeIds`,view:s.sourceEdgeIds,type:`u32`,access:`read_write`},{name:`startParameters`,view:s.sourceStartParameters,type:`f32`,access:`read_write`},{name:`endParameters`,view:s.sourceEndParameters,type:`f32`,access:`read_write`}],invocationCount:M,body:`let endpoint = endpointsOffset + index * 4u;
  endpoints[endpoint] = 0.0; endpoints[endpoint + 1u] = 0.0;
  endpoints[endpoint + 2u] = 0.0; endpoints[endpoint + 3u] = 0.0;
  sourceFeatureIds[sourceFeatureIdsOffset + index] = ${m}u;
  sourceRingIds[sourceRingIdsOffset + index] = ${m}u;
  sourceEdgeIds[sourceEdgeIdsOffset + index] = ${m}u;
  startParameters[startParametersOffset + index] = 0.0;
  endParameters[endParametersOffset + index] = 0.0;`}),i(e,{id:`${n}-write-geometry`,operation:g,variant:`write-geometry`,bindings:[{name:`flags`,view:O,type:`u32`,access:`read`},{name:`scan`,view:k,type:`u32`,access:`read`},{name:`piecePositions`,view:v,type:`f32`,access:`read`},{name:`endpoints`,view:s.endpoints,type:`f32`,access:`read_write`}],invocationCount:p,declarations:`const OUTPUT_CAPACITY: u32 = ${M}u;
fn piecePoint(vertex: u32) -> vec2f {
  return vec2f(piecePositions[piecePositionsOffset + vertex * 2u], piecePositions[piecePositionsOffset + vertex * 2u + 1u]);
}`,body:`if (flags[flagsOffset + index] == 0u) { return; }
  let outputRow = scan[scanOffset + index] - 1u;
  if (outputRow >= OUTPUT_CAPACITY) { return; }
  let a = piecePoint(index);
  let b = piecePoint(index + 1u);
  let endpoint = endpointsOffset + outputRow * 4u;
  endpoints[endpoint] = a.x; endpoints[endpoint + 1u] = a.y;
  endpoints[endpoint + 2u] = b.x; endpoints[endpoint + 3u] = b.y;`}),i(e,{id:`${n}-write-owners`,operation:g,variant:`write-owners`,bindings:[{name:`pieceOffsets`,view:y,type:`u32`,access:`read`},{name:`pieceCount`,view:x,type:`u32`,access:`read`},{name:`flags`,view:O,type:`u32`,access:`read`},{name:`scan`,view:k,type:`u32`,access:`read`},{name:`pieceSources`,view:b,type:`u32`,access:`read`},...N,{name:`sourceFeatureIds`,view:s.sourceFeatureIds,type:`u32`,access:`read_write`},{name:`sourceRingIds`,view:s.sourceRingIds,type:`u32`,access:`read_write`}],invocationCount:p,declarations:`${j}
const OUTPUT_CAPACITY: u32 = ${M}u;`,body:`if (flags[flagsOffset + index] == 0u) { return; }
  let outputRow = scan[scanOffset + index] - 1u;
  if (outputRow >= OUTPUT_CAPACITY) { return; }
  let piece = pieceOfVertex(index);
  let line = pieceSources[pieceSourcesOffset + piece];
  sourceFeatureIds[sourceFeatureIdsOffset + outputRow] = ${o.sourceIds?`inputSourceIds[inputSourceIdsOffset + line]`:`line`};
  sourceRingIds[sourceRingIdsOffset + outputRow] = line;`}),i(e,{id:`${n}-write-edge-provenance`,operation:g,variant:`write-edge-provenance`,bindings:[{name:`flags`,view:O,type:`u32`,access:`read`},{name:`scan`,view:k,type:`u32`,access:`read`},{name:`piecePositions`,view:v,type:`f32`,access:`read`},{name:`pieceSourceEdges`,view:T,type:`u32`,access:`read`},{name:`sourceRingIds`,view:s.sourceRingIds,type:`u32`,access:`read`},{name:`inputPositions`,view:o.positions,type:`f32`,access:`read`},{name:`inputOffsets`,view:o.lineOffsets,type:`u32`,access:`read`},{name:`sourceEdgeIds`,view:s.sourceEdgeIds,type:`u32`,access:`read_write`}],invocationCount:p,declarations:`const OUTPUT_CAPACITY: u32 = ${M}u;
const SELF_CONTACT: u32 = ${u}u;
fn inputPoint(vertex: u32) -> vec2f {
  return vec2f(inputPositions[inputPositionsOffset + vertex * 2u], inputPositions[inputPositionsOffset + vertex * 2u + 1u]);
}
fn piecePoint(vertex: u32) -> vec2f {
  return vec2f(piecePositions[piecePositionsOffset + vertex * 2u], piecePositions[piecePositionsOffset + vertex * 2u + 1u]);
}
// Finite and non-zero length: the intersection treats such an edge's successor as adjacent.
fn isRegularEdge(edge: u32) -> bool {
  let delta = inputPoint(edge + 1u) - inputPoint(edge);
  let lengthSquared = dot(delta, delta);
  return lengthSquared > 0.0 && lengthSquared <= 3.4e38;
}
fn considerEdge(edge: u32, midpoint: vec2f, bestEdge: ptr<function, u32>, bestError: ptr<function, f32>) {
  let p = inputPoint(edge);
  let q = inputPoint(edge + 1u);
  let delta = q - p;
  let lengthSquared = dot(delta, delta);
  if (lengthSquared == 0.0) { return; }
  let parameter = clamp(dot(midpoint - p, delta) / lengthSquared, 0.0, 1.0);
  let difference = midpoint - (p + parameter * delta);
  let error = dot(difference, difference);
  if (error < *bestError || (error == *bestError && edge < *bestEdge)) {
    *bestError = error;
    *bestEdge = edge;
  }
}`,body:`if (flags[flagsOffset + index] == 0u) { return; }
  let outputRow = scan[scanOffset + index] - 1u;
  if (outputRow >= OUTPUT_CAPACITY) { return; }
  let line = sourceRingIds[sourceRingIdsOffset + outputRow];
  let midpoint = 0.5 * (piecePoint(index) + piecePoint(index + 1u));
  let firstEdge = inputOffsets[inputOffsetsOffset + line];
  let lastVertex = inputOffsets[inputOffsetsOffset + line + 1u];
  let lastEdge = lastVertex - 2u;
  let source = pieceSourceEdges[pieceSourceEdgesOffset + index];
  let edge = source & ~SELF_CONTACT;
  var bestEdge = ${m}u;
  var bestError = 3.4e38;
  var scanAll = (source & SELF_CONTACT) != 0u || edge < firstEdge || edge > lastEdge;
  if (!scanAll) {
    scanAll = !isRegularEdge(edge) ||
      (edge > firstEdge && !isRegularEdge(edge - 1u)) ||
      (edge < lastEdge && !isRegularEdge(edge + 1u)) ||
      (edge == firstEdge && !isRegularEdge(lastEdge)) ||
      (edge == lastEdge && !isRegularEdge(firstEdge));
  }
  if (scanAll) {
    for (var candidate = firstEdge; candidate + 1u < lastVertex; candidate++) {
      considerEdge(candidate, midpoint, &bestEdge, &bestError);
    }
  } else {
    // The minimum of (error, edge) does not depend on visiting order or repeats.
    considerEdge(firstEdge, midpoint, &bestEdge, &bestError);
    if (edge > firstEdge) { considerEdge(edge - 1u, midpoint, &bestEdge, &bestError); }
    considerEdge(edge, midpoint, &bestEdge, &bestError);
    if (edge < lastEdge) { considerEdge(edge + 1u, midpoint, &bestEdge, &bestError); }
    considerEdge(lastEdge, midpoint, &bestEdge, &bestError);
  }
  sourceEdgeIds[sourceEdgeIdsOffset + outputRow] = bestEdge;`}),i(e,{id:`${n}-write-parameters`,operation:g,variant:`write-parameters`,bindings:[{name:`flags`,view:O,type:`u32`,access:`read`},{name:`scan`,view:k,type:`u32`,access:`read`},{name:`piecePositions`,view:v,type:`f32`,access:`read`},{name:`inputPositions`,view:o.positions,type:`f32`,access:`read`},{name:`sourceEdgeIds`,view:s.sourceEdgeIds,type:`u32`,access:`read`},{name:`startParameters`,view:s.sourceStartParameters,type:`f32`,access:`read_write`},{name:`endParameters`,view:s.sourceEndParameters,type:`f32`,access:`read_write`}],invocationCount:p,declarations:`const OUTPUT_CAPACITY: u32 = ${M}u;
fn inputPoint(vertex: u32) -> vec2f {
  return vec2f(inputPositions[inputPositionsOffset + vertex * 2u], inputPositions[inputPositionsOffset + vertex * 2u + 1u]);
}
fn piecePoint(vertex: u32) -> vec2f {
  return vec2f(piecePositions[piecePositionsOffset + vertex * 2u], piecePositions[piecePositionsOffset + vertex * 2u + 1u]);
}`,body:`if (flags[flagsOffset + index] == 0u) { return; }
  let outputRow = scan[scanOffset + index] - 1u;
  if (outputRow >= OUTPUT_CAPACITY) { return; }
  let a = piecePoint(index);
  let b = piecePoint(index + 1u);
  let bestEdge = sourceEdgeIds[sourceEdgeIdsOffset + outputRow];
  if (bestEdge != ${m}u) {
    let p = inputPoint(bestEdge);
    let q = inputPoint(bestEdge + 1u);
    let delta = q - p;
    let lengthSquared = dot(delta, delta);
    startParameters[startParametersOffset + outputRow] = clamp(dot(a - p, delta) / lengthSquared, 0.0, 1.0);
    endParameters[endParametersOffset + outputRow] = clamp(dot(b - p, delta) / lengthSquared, 0.0, 1.0);
  }`})),D.push(i(e,{id:`${n}-status`,operation:g,variant:`status`,bindings:[{name:`scan`,view:k,type:`u32`,access:`read`},{name:`candidateOverflow`,view:s.status.candidateOverflow,type:`u32`,access:`read_write`},{name:`splitIncomplete`,view:E,type:`u32`,access:`read`},{name:`count`,view:s.status.count,type:`u32`,access:`read_write`},{name:`requiredCount`,view:s.status.requiredCount,type:`u32`,access:`read_write`},{name:`overflow`,view:s.status.overflow,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`const OUTPUT_CAPACITY: u32 = ${M}u;`,body:`let required = scan[scanOffset + ${p-1}u];
  candidateOverflow[candidateOverflowOffset] |= splitIncomplete[splitIncompleteOffset];
  requiredCount[requiredCountOffset] = required;
  count[countOffset] = min(required, OUTPUT_CAPACITY);
  overflow[overflowOffset] = select(0u, 1u, required > OUTPUT_CAPACITY);`})),D}};export{_ as n,v as t};