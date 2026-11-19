import{o as e,s as t}from"./network-view-utils-B1U8oNKx.js";import{a as n,m as r,v as i}from"./spatial-context-DwZEd4EZ.js";import{r as a}from"./kernel-C3tfQOn4.js";import{o,t as s}from"./wgsl-kernel-nodes-DagJklH2.js";import{A as c,E as l,S as u,f as d,l as f,u as p,y as m}from"./gpu-contributor-utils-BIwLQON5.js";import{t as h}from"./gpu-nearest-feature-join-DFdL3u--.js";import{t as g}from"./network-spatial-context-wSR9-WDz.js";import{t as _}from"./network-accessibility-passes-BXCyymzh.js";function v(e,t){let{id:n,operation:r,offsets:i,edgeSources:a,nodeCount:c}=t;return[s(e,{id:`${n}-clear`,operation:r,view:a,type:`u32`,value:`${_}u`}),o(e,{id:n,operation:r,variant:`edge-sources`,bindings:[{name:`offsets`,view:i,type:`u32`,access:`read`},{name:`edgeSources`,view:a,type:`u32`,access:`read_write`}],invocationCount:c,declarations:`const EDGE_COUNT: u32 = ${a.length}u;`,body:`let end = min(offsets[offsetsOffset + index + 1u], EDGE_COUNT);
  for (var edge = offsets[offsetsOffset + index]; edge < end; edge++) {
    edgeSources[edgeSourcesOffset + edge] = index;
  }`})]}var y=Object.freeze({...n,measure:`distance`}),b=`GPUNetworkSnapping`;function x(e){return p(e,16)}var S=2**27,C=class{id;props;pointCount;nodeCount;edgeCount;startDirection;spatialContext;constructor(e){this.id=e.id??`network-snapping`,this.props=e,t(this.id,e.output),this.startDirection=e.startDirection??`both`;let{id:n}=this;this.spatialContext=i(n,e.spatialContext,y);let{output:r}=e,{nodeCount:a,edgeCount:o}=c(n,e.topology,{nodePositions:!0});l(e.positions,[`float32x2`],`${n} positions`),r.snappedPositions&&l(r.snappedPositions,[`float32x2`],`${n} output.snappedPositions`);for(let[t,i]of[[`maximumSnapDistance`,e.maximumSnapDistance],[`output.snapFractions`,r.snapFractions],[`output.snapDistances`,r.snapDistances],[`output.sourceCosts`,r.sourceCosts],[`output.targetCosts`,r.targetCosts],[`output.startCosts`,r.startCosts]])i&&l(i,[`float32`],`${n} ${t}`);for(let[e,t]of[[`output.snappedEdges`,r.snappedEdges],[`output.startNodes`,r.startNodes],[`output.status.overflow`,r.status?.overflow]])t&&l(t,[`uint32`],`${n} ${e}`);this.pointCount=e.positions.length,this.nodeCount=a,this.edgeCount=o;for(let[e,t]of[[`output.snappedEdges`,r.snappedEdges],[`output.snapFractions`,r.snapFractions],[`output.snapDistances`,r.snapDistances],[`output.sourceCosts`,r.sourceCosts],[`output.targetCosts`,r.targetCosts],[`output.snappedPositions`,r.snappedPositions]])if(t&&t.length!==this.pointCount)throw Error(`${n} ${e} length must equal the position count`);for(let[e,t]of[[`output.startNodes`,r.startNodes],[`output.startCosts`,r.startCosts]])if(t&&t.length!==2*this.pointCount)throw Error(`${n} ${e} length must be twice the position count`);if(!!r.startNodes!=!!r.startCosts)throw Error(`${n} output.startNodes and output.startCosts must be given together`);for(let[t,i]of[[`maximumSnapDistance`,e.maximumSnapDistance],[`output.status.overflow`,r.status?.overflow]])if(i&&i.length!==1)throw Error(`${n} ${t} must contain exactly one row`);if(e.candidateCapacity!==void 0){if(d(n,`candidateCapacity`,e.candidateCapacity),!e.maximumSnapDistance)throw Error(`${n} candidateCapacity requires maximumSnapDistance`);if(!r.status?.overflow)throw Error(`${n} candidateCapacity requires output.status.overflow`)}if(e.maximumUnindexedWork!==void 0&&d(n,`maximumUnindexedWork`,e.maximumUnindexedWork,1,2**53-1),e.candidateCapacity===void 0&&u(n,`positions times topology edges without candidateCapacity`,this.pointCount*o,e.maximumUnindexedWork??S),r.status?.overflow&&e.candidateCapacity===void 0)throw Error(`${n} output.status.overflow requires candidateCapacity`);let s=T(e);m(n,s,w(e));let f=s.filter(e=>e!==void 0).map(e=>e.buffer);if(new Set(f).size!==f.length)throw Error(`${n} outputs must use separate buffers`)}getCommandNodes(t){let{id:n,props:i,pointCount:s,nodeCount:c,edgeCount:l}=this,{offsets:d,neighbors:p,weights:m}=i.topology,{nodePositions:y}=i.topology,{output:S}=i;e(n,t,[...w(i),...T(i)]),i.candidateCapacity!==void 0&&u(n,`topology edges with candidateCapacity`,l,x(t.device));let C=f(Math.max(l,1)),E=this.spatialContext.coordinateSpace===`longitude-latitude`,D=[];e(n,t,[],{edgeSources:l*4,pointScratch:s*4,...i.candidateCapacity===void 0?{}:{segments:l*8,candidates:E?0:i.candidateCapacity*8,boundsAndChildren:(2*C-1)*8,nodeTable:E?((2*C-1)*5+2*s)*4:0,joinState:16}});let O=`const NODE_COUNT: u32 = ${c}u;
const EDGE_COUNT: u32 = ${l}u;
const NONE: u32 = ${_}u;`,k=a(t,`${n}-edge-sources`,`uint32`,Math.max(l,1));D.push(...v(t,{id:`${n}-edge-sources`,operation:b,offsets:d,edgeSources:k,nodeCount:c}));let A=k,j=S.snapFractions??a(t,`${n}-fractions`,`float32`,s),M=S.snapDistances??a(t,`${n}-distances`,`float32`,s),N=g(r(this.spatialContext)),P=`
fn readNode(node: u32) -> vec2<f32> {
  return vec2<f32>(nodePositions[nodePositionsOffset + node * 2u], nodePositions[nodePositionsOffset + node * 2u + 1u]);
}`,F=`${P}
${N}
// Fraction of the closest point along a -> b; 0 for a degenerate edge.
fn getFraction(point: vec2<f32>, start: vec2<f32>, end: vec2<f32>) -> f32 {
  return networkSegmentFraction(point, start, end);
}
fn getDistance(point: vec2<f32>, start: vec2<f32>, end: vec2<f32>, fraction: f32) -> f32 {
  return networkSegmentDistance(point, start, end, fraction);
}`,I=[{name:`points`,view:i.positions,type:`f32`,access:`read`},{name:`nodePositions`,view:y,type:`f32`,access:`read`},{name:`edgeSources`,view:A,type:`u32`,access:`read`},{name:`edgeTargets`,view:p,type:`u32`,access:`read`}],L=[{name:`snappedEdges`,view:S.snappedEdges,type:`u32`,access:`read_write`},{name:`fractions`,view:j,type:`f32`,access:`read_write`},{name:`distances`,view:M,type:`f32`,access:`read_write`}],R=`if (bestEdge == NONE) {
    snappedEdges[snappedEdgesOffset + index] = NONE;
    fractions[fractionsOffset + index] = -1.0;
    distances[distancesOffset + index] = -1.0;
    return;
  }
  snappedEdges[snappedEdgesOffset + index] = bestEdge;
  fractions[fractionsOffset + index] = bestFraction;
  distances[distancesOffset + index] = bestDistance;`;if(i.candidateCapacity!==void 0&&i.maximumSnapDistance){let e=a(t,`${n}-segment-starts`,`float32x2`,l),r=a(t,`${n}-segment-ends`,`float32x2`,l),c=a(t,`${n}-join-rows`,`uint32`,s);l>0&&D.push(o(t,{id:`${n}-segments`,operation:b,variant:`segments`,bindings:[{name:`nodePositions`,view:y,type:`f32`,access:`read`},{name:`edgeSources`,view:A,type:`u32`,access:`read`},{name:`edgeTargets`,view:p,type:`u32`,access:`read`},{name:`segmentStarts`,view:e,type:`f32`,access:`read_write`},{name:`segmentEnds`,view:r,type:`f32`,access:`read_write`}],invocationCount:l,declarations:`${O}
fn readNode(node: u32) -> vec2<f32> {
  return vec2<f32>(nodePositions[nodePositionsOffset + node * 2u], nodePositions[nodePositionsOffset + node * 2u + 1u]);
}`,body:`let sourceNode = edgeSources[edgeSourcesOffset + index];
  let targetNode = edgeTargets[edgeTargetsOffset + index];
  // WGSL const evaluation rejects an infinite value, so the lon/lat sentinel is built from runtime
  // bits (a var), never from a literal bitcast.
  var infinityBits = 0x7f800000u;
  var start = vec2<f32>(${this.spatialContext.coordinateSpace===`longitude-latitude`?`bitcast<f32>(infinityBits)`:`1.0e30`});
  var end = start;
  if (sourceNode < NODE_COUNT && targetNode < NODE_COUNT) {
    start = readNode(sourceNode);
    end = readNode(targetNode);
  }
  segmentStarts[segmentStartsOffset + index * 2u] = start.x;
  segmentStarts[segmentStartsOffset + index * 2u + 1u] = start.y;
  segmentEnds[segmentEndsOffset + index * 2u] = end.x;
  segmentEnds[segmentEndsOffset + index * 2u + 1u] = end.y;`})),D.push(...new h({id:`${n}-join`,spatialContext:this.spatialContext,queries:{kind:`points`,positions:i.positions},features:{kind:`segments`,starts:e,ends:r},radius:i.maximumSnapDistance,candidateCapacity:i.candidateCapacity,spatialSort:i.spatialSort,output:{nearestFeatureIds:c,status:{overflow:S.status.overflow}}}).getCommandNodes(t)),D.push(o(t,{id:`${n}-project`,operation:b,variant:`project`,bindings:[...I,{name:`joinRows`,view:c,type:`u32`,access:`read`},...L],invocationCount:s,declarations:`${O}
${F}`,body:`let point = vec2<f32>(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  var bestEdge = joinRows[joinRowsOffset + index];
  var bestFraction = 0.0;
  var bestDistance = 0.0;
  if (bestEdge < EDGE_COUNT) {
    let sourceNode = edgeSources[edgeSourcesOffset + bestEdge];
    let targetNode = edgeTargets[edgeTargetsOffset + bestEdge];
    let start = readNode(sourceNode);
    let end = readNode(targetNode);
    bestFraction = getFraction(point, start, end);
    bestDistance = getDistance(point, start, end, bestFraction);
  } else {
    bestEdge = NONE;
  }
  ${R}`}))}else{let e=[...I,...L];i.maximumSnapDistance&&e.push({name:`maximumSnapDistance`,view:i.maximumSnapDistance,type:`f32`,access:`read`}),D.push(o(t,{id:`${n}-scan`,operation:b,variant:`scan`,bindings:e,invocationCount:s,declarations:`${O}
${F}`,body:`let point = vec2<f32>(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  ${i.maximumSnapDistance?`let limit = maximumSnapDistance[maximumSnapDistanceOffset];`:``}
  var bestEdge = NONE;
  var bestFraction = 0.0;
  var bestDistance = 0.0;
  for (var edge = 0u; edge < EDGE_COUNT; edge++) {
    let sourceNode = edgeSources[edgeSourcesOffset + edge];
    let targetNode = edgeTargets[edgeTargetsOffset + edge];
    if (sourceNode >= NODE_COUNT || targetNode >= NODE_COUNT) {
      continue;
    }
    let start = readNode(sourceNode);
    let end = readNode(targetNode);
    let fraction = getFraction(point, start, end);
    let distance = getDistance(point, start, end, fraction);
    // Strict less keeps the smallest edge row on ties; NaN never passes.
    if (${i.maximumSnapDistance?`distance <= limit`:`distance >= 0.0`} && (bestEdge == NONE || distance < bestDistance)) {
      bestEdge = edge;
      bestFraction = fraction;
      bestDistance = distance;
    }
  }
  ${R}`}))}let z=!!(S.sourceCosts||S.targetCosts||S.startNodes),B=S.sourceCosts??(z?a(t,`${n}-source-costs`,`float32`,s):null),V=S.targetCosts??(z?a(t,`${n}-target-costs`,`float32`,s):null);if(B&&V){let e=[{name:`snappedEdges`,view:S.snappedEdges,type:`u32`,access:`read`},{name:`fractions`,view:j,type:`f32`,access:`read`},{name:`edgeSources`,view:A,type:`u32`,access:`read`},{name:`edgeTargets`,view:p,type:`u32`,access:`read`},m?{name:`edgeCosts`,view:m,type:`f32`,access:`read`}:{name:`nodePositions`,view:y,type:`f32`,access:`read`},{name:`sourceCosts`,view:B,type:`f32`,access:`read_write`},{name:`targetCosts`,view:V,type:`f32`,access:`read_write`}];D.push(o(t,{id:`${n}-costs`,operation:b,variant:`costs`,bindings:e,invocationCount:s,declarations:`${O}
${m?``:`${P}\n${N}`}`,body:`let edge = snappedEdges[snappedEdgesOffset + index];
  if (edge >= EDGE_COUNT) {
    sourceCosts[sourceCostsOffset + index] = -1.0;
    targetCosts[targetCostsOffset + index] = -1.0;
    return;
  }
  let fraction = fractions[fractionsOffset + index];
  ${m?`let edgeCost = edgeCosts[edgeCostsOffset + edge];`:`let edgeCost = networkDistance(
    readNode(edgeSources[edgeSourcesOffset + edge]),
    readNode(edgeTargets[edgeTargetsOffset + edge])
  );`}
  sourceCosts[sourceCostsOffset + index] = fraction * edgeCost;
  targetCosts[targetCostsOffset + index] = (1.0 - fraction) * edgeCost;`}))}if(S.snappedPositions&&D.push(o(t,{id:`${n}-positions`,operation:b,variant:`positions`,bindings:[{name:`points`,view:i.positions,type:`f32`,access:`read`},{name:`nodePositions`,view:y,type:`f32`,access:`read`},{name:`edgeSources`,view:A,type:`u32`,access:`read`},{name:`edgeTargets`,view:p,type:`u32`,access:`read`},{name:`snappedEdges`,view:S.snappedEdges,type:`u32`,access:`read`},{name:`fractions`,view:j,type:`f32`,access:`read`},{name:`snappedPositions`,view:S.snappedPositions,type:`f32`,access:`read_write`}],invocationCount:s,declarations:`${O}
${P}
${N}`,body:`var position = vec2<f32>(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  let edge = snappedEdges[snappedEdgesOffset + index];
  if (edge < EDGE_COUNT) {
    let start = readNode(edgeSources[edgeSourcesOffset + edge]);
    let end = readNode(edgeTargets[edgeTargetsOffset + edge]);
    position = networkInterpolate(start, end, fractions[fractionsOffset + index]);
  }
  snappedPositions[snappedPositionsOffset + index * 2u] = position.x;
  snappedPositions[snappedPositionsOffset + index * 2u + 1u] = position.y;`})),S.startNodes&&S.startCosts&&B&&V){let e=this.startDirection!==`forward`,r=this.startDirection!==`reverse`;D.push(o(t,{id:`${n}-seeds`,operation:b,variant:`seeds-${this.startDirection}`,bindings:[{name:`snappedEdges`,view:S.snappedEdges,type:`u32`,access:`read`},{name:`edgeSources`,view:A,type:`u32`,access:`read`},{name:`edgeTargets`,view:p,type:`u32`,access:`read`},{name:`sourceCosts`,view:B,type:`f32`,access:`read`},{name:`targetCosts`,view:V,type:`f32`,access:`read`},{name:`seedNodes`,view:S.startNodes,type:`u32`,access:`read_write`},{name:`seedCosts`,view:S.startCosts,type:`f32`,access:`read_write`}],invocationCount:s,declarations:O,body:`let edge = snappedEdges[snappedEdgesOffset + index];
  let isSnapped = edge < EDGE_COUNT;
  let useSource = isSnapped && ${e};
  let useTarget = isSnapped && ${r};
  let sourceNode = select(0u, edgeSources[edgeSourcesOffset + min(edge, EDGE_COUNT - 1u)], isSnapped);
  let targetNode = select(0u, edgeTargets[edgeTargetsOffset + min(edge, EDGE_COUNT - 1u)], isSnapped);
  seedNodes[seedNodesOffset + index * 2u] = select(NONE, sourceNode, useSource);
  seedCosts[seedCostsOffset + index * 2u] = select(-1.0, sourceCosts[sourceCostsOffset + index], useSource);
  seedNodes[seedNodesOffset + index * 2u + 1u] = select(NONE, targetNode, useTarget);
  seedCosts[seedCostsOffset + index * 2u + 1u] = select(-1.0, targetCosts[targetCostsOffset + index], useTarget);`}))}return D}};function w(e){return[e.positions,e.topology.offsets,e.topology.neighbors,e.topology.nodePositions,e.output.sourceCosts||e.output.targetCosts||e.output.startCosts?e.topology.weights:void 0,e.maximumSnapDistance]}function T(e){let{output:t}=e;return[t.snappedEdges,t.snapFractions,t.snapDistances,t.sourceCosts,t.targetCosts,t.snappedPositions,t.startNodes,t.startCosts,t.status?.overflow]}export{v as n,C as t};