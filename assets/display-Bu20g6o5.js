import{o as e}from"./wgsl-kernel-nodes-DagJklH2.js";var t=`-100.0`;function n(n,r){let i=r.stride-1,a=n.allocate(`${r.id}-segments`,`float32`,i*4);return n.graph.add(e(n.graph,{id:`${r.id}-expand`,operation:`display-vertex-segments`,invocationCount:i,bindings:[{name:`vertices`,view:r.vertices.view,type:`u32`,access:`read`},{name:`nodePositions`,view:r.nodePositions.view,type:`f32`,access:`read`},{name:`segments`,view:a.view,type:`f32`,access:`read_write`}],body:`
  let first = vertices[verticesOffset + ${r.routeIndex*r.stride}u + index];
  let second = vertices[verticesOffset + ${r.routeIndex*r.stride}u + index + 1u];
  let isValid = first != 0xffffffffu && second != 0xffffffffu;
  let slot = segmentsOffset + index * 4u;
  segments[slot] = select(${t}, nodePositions[nodePositionsOffset + first * 2u], isValid);
  segments[slot + 1u] = select(${t}, nodePositions[nodePositionsOffset + first * 2u + 1u], isValid);
  segments[slot + 2u] = select(${t}, nodePositions[nodePositionsOffset + second * 2u], isValid);
  segments[slot + 3u] = select(${t}, nodePositions[nodePositionsOffset + second * 2u + 1u], isValid);`})),a}function r(t,n){let r=t.allocate(`${n.id}-doubled`,`float32`,n.count*2),i=n.source.format===`uint32`;return t.graph.add(e(t.graph,{id:`${n.id}-double`,operation:`display-slot-values`,invocationCount:n.count,bindings:[{name:`source`,view:n.source.view,type:i?`u32`:`f32`,access:`read`},{name:`doubled`,view:r.view,type:`f32`,access:`read_write`}],body:`
  let value = ${i?`f32(source[sourceOffset + index])`:`source[sourceOffset + index]`};
  doubled[doubledOffset + index] = value;
  doubled[doubledOffset + ${n.count}u + index] = value;`})),r}function i(t,n){let r=n.from.length,i=t.allocate(`${n.id}-pairs`,`uint32`,r*2);return t.graph.add(e(t.graph,{id:`${n.id}-interleave`,operation:`display-edge-pairs`,invocationCount:r,bindings:[{name:`sources`,view:n.from.view,type:`u32`,access:`read`},{name:`targets`,view:n.to.view,type:`u32`,access:`read`},{name:`pairs`,view:i.view,type:`u32`,access:`read_write`}],body:`
  let first = sources[sourcesOffset + index];
  let second = targets[targetsOffset + index];
  let isValid = first != 0xffffffffu && second != 0xffffffffu;
  pairs[pairsOffset + index * 2u] = select(${n.fallbackNode}u, first, isValid);
  pairs[pairsOffset + index * 2u + 1u] = select(${n.fallbackNode}u, second, isValid);`})),i}function a(t,n){let r=t.allocate(`${n.id}-slice`,`float32`,n.length),i=n.source.format===`uint32`;return t.graph.add(e(t.graph,{id:`${n.id}-copy`,operation:`display-slice`,invocationCount:n.length,bindings:[{name:`source`,view:n.source.view,type:i?`u32`:`f32`,access:`read`},{name:`slice`,view:r.view,type:`f32`,access:`read_write`}],body:`slice[sliceOffset + index] = ${i?`f32(source[sourceOffset + `+n.offset+`u + index])`:`source[sourceOffset + `+n.offset+`u + index]`};`})),r}function o(t,n){let r=t.allocate(`${n.id}-copy`,`uint32`,n.view.length);return t.graph.add(e(t.graph,{id:`${n.id}-copy`,operation:`display-copy`,invocationCount:n.view.length,bindings:[{name:`source`,view:n.view,type:`u32`,access:`read`},{name:`copy`,view:r.view,type:`u32`,access:`read_write`}],body:`let value = source[sourceOffset + index];
  var result = value;
  ${n.transform??``}
  copy[copyOffset + index] = result;`})),r}function s(t,n){let r=t.allocate(`${n.id}-ids`,`uint32`,n.count);return t.graph.add(e(t.graph,{id:`${n.id}-flagged`,operation:`display-flagged-ids`,invocationCount:n.count,bindings:[{name:`flags`,view:n.flags.view,type:`u32`,access:`read`},{name:`ids`,view:r.view,type:`u32`,access:`read_write`}],body:`ids[idsOffset + index] = select(0xffffffffu, index, flags[flagsOffset + index] == ${n.flagValue??1}u);`})),r}function c(t,n){let r=t.allocate(`${n.id}-pairs`,`float32x4`,n.from.length);return t.graph.add(e(t.graph,{id:`${n.id}-join`,operation:`display-pair-segments`,invocationCount:n.from.length,bindings:[{name:`origins`,view:n.from.view,type:`f32`,access:`read`},{name:`targets`,view:n.to.view,type:`f32`,access:`read`},{name:`segments`,view:r.view,type:`f32`,access:`read_write`}],body:`
  let slot = segmentsOffset + index * 4u;
  segments[slot] = origins[originsOffset + index * 2u];
  segments[slot + 1u] = origins[originsOffset + index * 2u + 1u];
  segments[slot + 2u] = targets[targetsOffset + index * 2u];
  segments[slot + 3u] = targets[targetsOffset + index * 2u + 1u];`})),r}function l(t,n){let r=t.allocate(`${n.id}-ids`,`uint32`,n.facilities.length);return t.graph.add(e(t.graph,{id:`${n.id}-thresholded`,operation:`display-thresholded-ids`,invocationCount:n.facilities.length,bindings:[{name:`facilities`,view:n.facilities.view,type:`u32`,access:`read`},{name:`costs`,view:n.costs.view,type:`f32`,access:`read`},{name:`threshold`,view:n.threshold.view,type:`f32`,access:`read`},{name:`ids`,view:r.view,type:`u32`,access:`read_write`}],body:`
  let isCovered = costs[costsOffset + index] <= threshold[thresholdOffset];
  ids[idsOffset + index] = select(0xffffffffu, facilities[facilitiesOffset + index], isCovered);`})),r}function u(t,n){let r=t.allocate(`${n.id}-footprint`,`float32`,n.nodeCount),i=n.mode===`row`?`selector[selectorOffset] * ${n.columnCount}u + index`:`index * ${n.columnCount}u + selector[selectorOffset]`,a=n.floor??.3;return t.graph.add(e(t.graph,{id:`${n.id}-footprint`,operation:`display-matrix-footprint`,invocationCount:n.nodeCount,bindings:[{name:`costs`,view:n.costs.view,type:`f32`,access:`read`},{name:`selector`,view:n.selector.view,type:`u32`,access:`read`},{name:`limit`,view:n.limit.view,type:`f32`,access:`read`},{name:`footprint`,view:r.view,type:`f32`,access:`read_write`}],body:`
  let cost = costs[costsOffset + ${i}];
  let reach = limit[limitOffset];
  footprint[footprintOffset + index] = select(0.0, 1.0 - (1.0 - ${a.toFixed(3)}) * cost / reach, cost >= 0.0 && cost <= reach);`})),r}function d(n,r){let i=r.positions.length,a=n.allocate(`${r.id}-legs`,`float32x4`,i),o=r.flags?r.flagValues?.map(e=>`flags[flagsOffset + index] == ${e}u`).join(` || `)||`flags[flagsOffset + index] != 0u`:`true`,s=[{name:`points`,view:r.positions.view,type:`f32`,access:`read`},...r.flags?[{name:`flags`,view:r.flags.view,type:`u32`,access:`read`}]:[],{name:`segments`,view:a.view,type:`f32`,access:`read_write`}];return n.graph.add(e(n.graph,{id:`${r.id}-legs`,operation:`display-consecutive-segments`,invocationCount:i,bindings:s,body:`
  let isValid = index % ${r.trackLength}u != 0u && (${o});
  let slot = segmentsOffset + index * 4u;
  let previous = select(index, index - 1u, index > 0u);
  segments[slot] = select(${t}, points[pointsOffset + previous * 2u], isValid);
  segments[slot + 1u] = select(${t}, points[pointsOffset + previous * 2u + 1u], isValid);
  segments[slot + 2u] = select(${t}, points[pointsOffset + index * 2u], isValid);
  segments[slot + 3u] = select(${t}, points[pointsOffset + index * 2u + 1u], isValid);`})),a}function f(n,r){let i=n.allocate(`${r.id}-zone`,`float32x4`,r.edges.length);return n.graph.add(e(n.graph,{id:`${r.id}-zone`,operation:`display-zone-segments`,invocationCount:r.edges.length,bindings:[{name:`nodes`,view:r.nodes.view,type:`f32`,access:`read`},{name:`edges`,view:r.edges.view,type:`u32`,access:`read`},{name:`values`,view:r.values.view,type:`f32`,access:`read`},{name:`segments`,view:i.view,type:`f32`,access:`read_write`}],body:`
  let first = edges[edgesOffset + index * 2u];
  let second = edges[edgesOffset + index * 2u + 1u];
  let isInside = values[valuesOffset + first] > ${(r.threshold??0).toFixed(3)} && values[valuesOffset + second] > ${(r.threshold??0).toFixed(3)};
  let slot = segmentsOffset + index * 4u;
  segments[slot] = select(${t}, nodes[nodesOffset + first * 2u], isInside);
  segments[slot + 1u] = select(${t}, nodes[nodesOffset + first * 2u + 1u], isInside);
  segments[slot + 2u] = select(${t}, nodes[nodesOffset + second * 2u], isInside);
  segments[slot + 3u] = select(${t}, nodes[nodesOffset + second * 2u + 1u], isInside);`})),i}export{c as a,l as c,f as d,u as i,o as l,i as n,a as o,s as r,r as s,d as t,n as u};