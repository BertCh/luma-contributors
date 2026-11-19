import{r as e,u as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-scan-B4NNTfF9.js";import{o as r}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as i,f as a,v as o,w as s,x as c}from"./gpu-contributor-utils-BIwLQON5.js";import{t as l}from"./exact-orientation-wgsl-B_jI-HEc.js";import{t as u}from"./exact-geometry-wgsl-y7TUnqOh.js";import{t as d}from"./gpu-sort-xRm8l5yn.js";import{i as f,u as p}from"./wgsl-common-DfyXiYB0.js";import{t as m}from"./segment-bvh-CDgckfAv.js";import{t as h}from"./binary-search-wgsl-CMmaMYIt.js";import{n as g}from"./gpu-line-noding-vE-FDCfr.js";import{n as _,t as v}from"./gpu-polygonize-w27Nx4-B.js";import{n as y}from"./topology-types-BLorEZd4.js";import{v as b}from"./spatial-context-DwZEd4EZ.js";function x(t,n){let{id:i,operation:a,records:o,leafCapacity:s}=n,c=o.length/5,l=n=>e(t,`${i}-${n}`,`uint32`,c),u=l(`height-keys`),h=l(`rows`),g=l(`sorted-height-keys`),_=l(`height-rows`),v=l(`group-keys`),y=l(`sorted-groups`),b=l(`sorted-rows`),x=e(t,`${i}-sorted-records`,`uint32`,o.length),S=e(t,`${i}-minima`,`float32x2`,c),C=e(t,`${i}-maxima`,`float32x2`,c),w=[r(t,{id:`${i}-height-keys`,operation:a,variant:`parity-height-keys`,bindings:[{name:`records`,view:o,type:`u32`,access:`read`},{name:`keys`,view:u,type:`u32`,access:`read_write`},{name:`rows`,view:h,type:`u32`,access:`read_write`}],invocationCount:c,declarations:`${f}\n${p}`,body:`let base = recordsOffset + 5u * index;
  let first = vec2f(bitcast<f32>(records[base]), bitcast<f32>(records[base + 1u]));
  let second = vec2f(bitcast<f32>(records[base + 2u]), bitcast<f32>(records[base + 3u]));
  let finite = isFiniteValue(first.x) && isFiniteValue(first.y) && isFiniteValue(second.x) && isFiniteValue(second.y);
  let height = select(-3.402823466e+38, min(first.y, second.y), finite);
  keys[keysOffset + index] = getOrderedKey(height);
  rows[rowsOffset + index] = index;`}),...new d({id:`${i}-height-sort`,keys:u,values:h,outputKeys:g,outputValues:_,algorithm:`radix`}).getCommandNodes(t),r(t,{id:`${i}-group-keys`,operation:a,variant:`parity-group-keys`,bindings:[{name:`records`,view:o,type:`u32`,access:`read`},{name:`rows`,view:_,type:`u32`,access:`read`},{name:`keys`,view:v,type:`u32`,access:`read_write`}],invocationCount:c,body:`keys[keysOffset + index] = records[recordsOffset + 5u * rows[rowsOffset + index] + 4u];`}),...new d({id:`${i}-group-sort`,keys:v,values:_,outputKeys:y,outputValues:b,algorithm:`radix`}).getCommandNodes(t),r(t,{id:`${i}-gather`,operation:a,variant:`parity-gather`,bindings:[{name:`records`,view:o,type:`u32`,access:`read`},{name:`rows`,view:b,type:`u32`,access:`read`},{name:`sortedRecords`,view:x,type:`u32`,access:`read_write`},{name:`minima`,view:S,type:`f32`,access:`read_write`},{name:`maxima`,view:C,type:`f32`,access:`read_write`}],invocationCount:c,declarations:p,body:`let source = recordsOffset + 5u * rows[rowsOffset + index];
  for (var component = 0u; component < 5u; component++) {
    sortedRecords[sortedRecordsOffset + 5u * index + component] = records[source + component];
  }
  let first = vec2f(bitcast<f32>(records[source]), bitcast<f32>(records[source + 1u]));
  let second = vec2f(bitcast<f32>(records[source + 2u]), bitcast<f32>(records[source + 3u]));
  let finite = isFiniteValue(first.x) && isFiniteValue(first.y) && isFiniteValue(second.x) && isFiniteValue(second.y);
  let unused = records[source + 4u] == 0xffffffffu;
  // Non-finite edges still reach the exact crossing helper and preserve its uncertainty flag.
  let minimum = select(select(-3.402823466e+38, min(first.y, second.y), finite), 3.402823466e+38, unused);
  let maximum = select(select(3.402823466e+38, max(first.y, second.y), finite), -3.402823466e+38, unused);
  minima[minimaOffset + 2u * index] = 0.0;
  minima[minimaOffset + 2u * index + 1u] = minimum;
  maxima[maximaOffset + 2u * index] = 0.0;
  maxima[maximaOffset + 2u * index + 1u] = maximum;`})],T=m(t,{id:i,operation:a,minima:S,maxima:C,leafCapacity:s,spatialSort:!1});return w.push(...T.nodes),{nodes:w,bvh:T.bvh,records:x,groups:y}}var S=`GPUMakeValid`,C=class{id;props;spatialContext;constructor(e){if(this.id=e.id??`make-valid`,this.spatialContext=b(this.id,e.spatialContext),this.props=e,_(this.id,e.output.status),e.polygons.kind!==`polygons`)throw Error(`${this.id} polygons.kind must be polygons`);if(e.polygons.ringOffsets.length<2)throw Error(`${this.id} polygons require at least one ring`);a(this.id,`intersectionCapacity`,e.intersectionCapacity),e.leafCapacity!==void 0&&s(this.id,`leafCapacity`,e.leafCapacity);let{polygons:n}=e.output,r=n.ringOffsets.length-1;i(n.positions,[`float32x2`],`${this.id} output.polygons.positions`);for(let[e,r]of[[`ringOffsets`,n.ringOffsets],[`polygonOffsets`,n.polygonOffsets],[`featureOffsets`,n.featureOffsets],[`sourceIds`,n.sourceIds]])t(r,`${this.id} output.polygons.${e}`);if(r<1||n.polygonOffsets.length!==r+1||n.featureOffsets.length!==r+1||n.sourceIds.length!==r)throw Error(`${this.id} polygon offset and source capacities must share one ring capacity`)}getCommandNodes(t){let{id:i,props:a}=this,{polygons:s}=a,d=2**Math.ceil(Math.log2(a.output.nodedSegments.endpoints.length));o(t.device,i,`parity segment records`,a.output.nodedSegments.endpoints.length*5*Uint32Array.BYTES_PER_ELEMENT,`reduce output.nodedSegments.endpoints.length`),o(t.device,i,`parity BVH nodes`,(2*d-1)*2*Float32Array.BYTES_PER_ELEMENT,`reduce output.nodedSegments.endpoints.length`);let f=s.ringOffsets.length-1,p=e(t,`${i}-ring-source-ids`,`uint32`,f),m=e(t,`${i}-ring-feature-rows`,`uint32`,f),_=s.sourceIds?[{name:`inputSourceIds`,view:s.sourceIds,type:`u32`,access:`read`}]:[],b=[r(t,{id:`${i}-ring-sources`,operation:S,variant:`ring-sources`,bindings:[{name:`featureOffsets`,view:s.featureOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:s.polygonOffsets,type:`u32`,access:`read`},..._,{name:`ringSourceIds`,view:p,type:`u32`,access:`read_write`},{name:`ringFeatureRows`,view:m,type:`u32`,access:`read_write`}],invocationCount:f,declarations:`const FEATURE_COUNT: u32 = ${s.featureOffsets.length-1}u;
const POLYGON_COUNT: u32 = ${s.polygonOffsets.length-1}u;`,body:`var polygonLow = 0u;
  var polygonHigh = POLYGON_COUNT;
  while (polygonLow + 1u < polygonHigh) {
    let middle = (polygonLow + polygonHigh) / 2u;
    if (polygonOffsets[polygonOffsetsOffset + middle] <= index) { polygonLow = middle; } else { polygonHigh = middle; }
  }
  var featureLow = 0u;
  var featureHigh = FEATURE_COUNT;
  while (featureLow + 1u < featureHigh) {
    let middle = (featureLow + featureHigh) / 2u;
    if (featureOffsets[featureOffsetsOffset + middle] <= polygonLow) { featureLow = middle; } else { featureHigh = middle; }
  }
  ringSourceIds[ringSourceIdsOffset + index] = ${s.sourceIds?`inputSourceIds[inputSourceIdsOffset + featureLow]`:`featureLow`};
  ringFeatureRows[ringFeatureRowsOffset + index] = featureLow;`})],{output:C}=a,w=C.polygons.ringOffsets.length-1,T=C.polygons.positions.length,E=C.nodedSegments.endpoints.length;c(i,t,[s.positions,s.featureOffsets,s.polygonOffsets,s.ringOffsets,s.sourceIds,C.polygons.positions,C.polygons.featureOffsets,C.polygons.polygonOffsets,C.polygons.ringOffsets,C.polygons.sourceIds,C.nodedSegments.endpoints,C.nodedSegments.sourceRingIds,...Object.values(C.nodedSegments.status),...Object.values(C.status)]);let D=s.positions.length,O=D+f;g(i,t.device,O,a.polygons.ringOffsets.length-1,a.intersectionCapacity);let k=e(t,`${i}-ring-closure-positions`,`float32x2`,O),A=e(t,`${i}-ring-closure-offsets`,`uint32`,f+1);b.push(r(t,{id:`${i}-ring-closure-offsets`,operation:S,variant:`ring-closure-offsets`,bindings:[{name:`ringOffsets`,view:s.ringOffsets,type:`u32`,access:`read`},{name:`closedOffsets`,view:A,type:`u32`,access:`read_write`}],invocationCount:f+1,body:`closedOffsets[closedOffsetsOffset + index] = ringOffsets[ringOffsetsOffset + index] + index;`}),r(t,{id:`${i}-ring-closure-positions`,operation:S,variant:`ring-closure-positions`,bindings:[{name:`positions`,view:s.positions,type:`f32`,access:`read`},{name:`ringOffsets`,view:s.ringOffsets,type:`u32`,access:`read`},{name:`closedPositions`,view:k,type:`f32`,access:`read_write`}],invocationCount:O,declarations:`const RING_COUNT: u32 = ${f}u;
const VERTEX_COUNT: u32 = ${D}u;`,body:`var ring = 0u;
  var high = RING_COUNT;
  while (ring + 1u < high) {
    let middle = (ring + high) / 2u;
    if (ringOffsets[ringOffsetsOffset + middle] + middle <= index) { ring = middle; } else { high = middle; }
  }
  let ringStart = ringOffsets[ringOffsetsOffset + ring];
  let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
  var point = vec2f(0.0);
  var source = ringStart;
  var hasSource = ringEnd > ringStart;
  if (hasSource) {
    source = index - ring;
    if (index < ringStart || source >= ringEnd) { source = ringStart; }
  } else {
    source = ringOffsets[ringOffsetsOffset];
    hasSource = ringOffsets[ringOffsetsOffset + RING_COUNT] > source;
  }
  if (hasSource) {
    source = min(source, VERTEX_COUNT - 1u);
    point = vec2f(positions[positionsOffset + 2u * source], positions[positionsOffset + 2u * source + 1u]);
  }
  closedPositions[closedPositionsOffset + 2u * index] = point.x;
  closedPositions[closedPositionsOffset + 2u * index + 1u] = point.y;`}));let j={kind:`polygons`,positions:e(t,`${i}-face-positions`,`float32x2`,T),featureOffsets:e(t,`${i}-face-feature-offsets`,`uint32`,w+1),polygonOffsets:e(t,`${i}-face-polygon-offsets`,`uint32`,w+1),ringOffsets:e(t,`${i}-face-ring-offsets`,`uint32`,w+1),sourceIds:e(t,`${i}-face-source-ids`,`uint32`,w)},M={id:`${i}-polygonize`,spatialContext:this.spatialContext,lines:{kind:`lines`,positions:k,lineOffsets:A,sourceIds:p},groupIds:m,intersectionCapacity:a.intersectionCapacity,precision:a.precision,snapRounding:a.snapRounding,output:{...C,polygons:j},spatialSort:a.spatialSort,leafCapacity:a.leafCapacity};b.push(...new v(M).getCommandNodes(t)),b.push(r(t,{id:`${i}-source-edge-rows`,operation:S,variant:`source-edge-rows`,bindings:[{name:`count`,view:C.nodedSegments.status.count,type:`u32`,access:`read`},{name:`sourceRingIds`,view:C.nodedSegments.sourceRingIds,type:`u32`,access:`read`},{name:`sourceEdgeIds`,view:C.nodedSegments.sourceEdgeIds,type:`u32`,access:`read_write`}],invocationCount:C.nodedSegments.endpoints.length,body:`let edge = sourceEdgeIds[sourceEdgeIdsOffset + index];
  let ring = sourceRingIds[sourceRingIdsOffset + index];
  if (index < count[countOffset] && edge != ${y}u && ring != ${y}u) {
    sourceEdgeIds[sourceEdgeIdsOffset + index] = edge - ring;
  }`}));let N=`${y}u`,P=(n,r)=>e(t,`${i}-${n}`,`uint32`,r),F=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read`}),I=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read_write`}),L=(e,n,a,o,s)=>r(t,{id:`${i}-${e}`,operation:S,variant:e,bindings:n,invocationCount:a,body:o,declarations:s}),R=w-1,z=P(`noded-records`,E*5),B=e(t,`${i}-face-probes`,`float32x4`,w),V=P(`keep`,w),H=P(`ring-counts`,w),U=P(`vertex-counts`,w),W=P(`keep-scan`,w),G=P(`ring-scan`,w),K=P(`vertex-scan`,w),q=P(`kept-ids`,w),J=P(`kept-vertex-starts`,w+1);b.push(L(`parity-segments`,[F(`count`,C.nodedSegments.status.count),F(`endpoints`,C.nodedSegments.endpoints,`f32`),F(`nodedRings`,C.nodedSegments.sourceRingIds),F(`ringFeatureRows`,m),I(`records`,z)],E,`var group = ${N};
  var first = vec2f(0.0);
  var second = vec2f(0.0);
  if (index < count[countOffset]) {
    group = ringFeatureRows[ringFeatureRowsOffset + nodedRings[nodedRingsOffset + index]];
    first = vec2f(endpoints[endpointsOffset + 4u * index], endpoints[endpointsOffset + 4u * index + 1u]);
    second = vec2f(endpoints[endpointsOffset + 4u * index + 2u], endpoints[endpointsOffset + 4u * index + 3u]);
  }
  records[recordsOffset + 5u * index] = bitcast<u32>(first.x);
  records[recordsOffset + 5u * index + 1u] = bitcast<u32>(first.y);
  records[recordsOffset + 5u * index + 2u] = bitcast<u32>(second.x);
  records[recordsOffset + 5u * index + 3u] = bitcast<u32>(second.y);
  records[recordsOffset + 5u * index + 4u] = group;
`),L(`face-probes`,[F(`facePositions`,j.positions,`f32`),F(`faceRingOffsets`,j.ringOffsets),F(`facePolygonOffsets`,j.polygonOffsets),F(`faceSourceIds`,j.sourceIds),I(`probes`,B,`f32`)],w,`var first = vec2f(0.0);
  var second = vec2f(0.0);
  if (faceSourceIds[faceSourceIdsOffset + index] != ${N}) {
    let shell = facePolygonOffsets[facePolygonOffsetsOffset + index];
    let start = faceRingOffsets[faceRingOffsetsOffset + shell];
    let end = faceRingOffsets[faceRingOffsetsOffset + shell + 1u];
    if (end - start >= 3u) {
      first = vec2f(facePositions[facePositionsOffset + 2u * start], facePositions[facePositionsOffset + 2u * start + 1u]);
      second = vec2f(facePositions[facePositionsOffset + 2u * start + 2u], facePositions[facePositionsOffset + 2u * start + 3u]);
    }
  }
  probes[probesOffset + 4u * index] = first.x;
  probes[probesOffset + 4u * index + 1u] = first.y;
  probes[probesOffset + 4u * index + 2u] = second.x;
  probes[probesOffset + 4u * index + 3u] = second.y;`));let Y=x(t,{id:`${i}-parity`,operation:S,records:z,leafCapacity:d}),{bvh:X}=Y;return b.push(...Y.nodes,L(`face-parity`,[F(`records`,Y.records),F(`sortedGroups`,Y.groups),F(`probes`,B,`f32`),F(`faceSourceIds`,j.sourceIds),F(`nodeMinima`,X.nodeMinima,`f32`),F(`nodeMaxima`,X.nodeMaxima,`f32`),F(`leafIds`,X.leafIds),I(`keep`,V)],w,`let group = faceSourceIds[faceSourceIdsOffset + index];
  var keepFace = 0u;
  if (group != ${N}) {
    let a = vec2f(probes[probesOffset + 4u * index], probes[probesOffset + 4u * index + 1u]);
    let b = vec2f(probes[probesOffset + 4u * index + 2u], probes[probesOffset + 4u * index + 3u]);
    if (any(a != b)) {
    // First edge of the counter-clockwise shell: the face lies on its left. The probe is the
    // symbolic point infinitesimally left of the edge's midpoint (utils/exact-geometry-wgsl.ts).
    let probe = ExactProbe(0.5 * (a + b), a, b, 1);
    var parity = 0u;
    var uncertain = false;
    let groupFirst = lowerBoundSortedGroups(0u, SEGMENT_CAPACITY, group);
    let groupLast = upperBoundSortedGroups(groupFirst, SEGMENT_CAPACITY, group);
    var node = 0u;
    loop {
      let level = firstLeadingBit(node + 1u);
      let span = LEAF_CAPACITY >> level;
      let nodeFirst = (node + 1u - (1u << level)) * span;
      let nodeLast = nodeFirst + span;
      let minimum = vec2f(nodeMinima[nodeMinimaOffset + 2u * node], nodeMinima[nodeMinimaOffset + 2u * node + 1u]);
      let maximum = vec2f(nodeMaxima[nodeMaximaOffset + 2u * node], nodeMaxima[nodeMaximaOffset + 2u * node + 1u]);
      // Use the endpoint interval, not the rounded midpoint: a one-step-tall probe may round
      // onto either endpoint. Equality stays a candidate for the infinitesimal side decision.
      // Keep both sides of the ray: an uncertifiable orientation left of the probe used to
      // retain the face too. Height rejection alone preserves that conservative fallback.
      if (nodeFirst < groupLast && nodeLast > groupFirst && minimum.y <= max(a.y, b.y) && maximum.y >= min(a.y, b.y)) {
        if (node < INTERNAL_NODE_COUNT) {
          node = 2u * node + 1u;
          continue;
        }
        let segment = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT];
        if (segment < SEGMENT_CAPACITY && records[recordsOffset + 5u * segment + 4u] == group) {
          let base = recordsOffset + 5u * segment;
          let first = vec2f(bitcast<f32>(records[base]), bitcast<f32>(records[base + 1u]));
          let second = vec2f(bitcast<f32>(records[base + 2u]), bitcast<f32>(records[base + 3u]));
          // The single call site of the exact helpers in this kernel.
          let crossing = exactProbeCrossing(probe, first, second);
          if ((u32(crossing.y) & EXACT_FLAG_UNCERTAIN) != 0u) { uncertain = true; }
          if (crossing.x != 0) { parity = parity ^ 1u; }
        }
      }
      loop {
        if (node == 0u || (node & 1u) == 1u) { break; }
        node = (node - 1u) / 2u;
      }
      if (node == 0u) { break; }
      node++;
    }
    keepFace = select(0u, 1u, parity == 1u || uncertain);
    }
  }
  keep[keepOffset + index] = keepFace;`,`const SEGMENT_CAPACITY: u32 = ${E}u;
const INTERNAL_NODE_COUNT: u32 = ${X.internalNodeCount}u;
const LEAF_CAPACITY: u32 = ${d}u;
${h(`sortedGroups`)}
${l}
${u}`),L(`face-counts`,[F(`keep`,V),F(`faceRingOffsets`,j.ringOffsets),F(`facePolygonOffsets`,j.polygonOffsets),I(`ringCounts`,H),I(`vertexCounts`,U)],w,`let firstRing = facePolygonOffsets[facePolygonOffsetsOffset + index];
  let endRing = facePolygonOffsets[facePolygonOffsetsOffset + index + 1u];
  let isKept = keep[keepOffset + index] != 0u;
  ringCounts[ringCountsOffset + index] = select(0u, endRing - firstRing, isKept);
  vertexCounts[vertexCountsOffset + index] = select(0u, faceRingOffsets[faceRingOffsetsOffset + endRing] - faceRingOffsets[faceRingOffsetsOffset + firstRing], isKept);`),...new n({id:`${i}-keep-scan`,input:V,output:W,mode:`inclusive`}).getCommandNodes(t),...new n({id:`${i}-ring-scan`,input:H,output:G,mode:`inclusive`}).getCommandNodes(t),...new n({id:`${i}-vertex-scan`,input:U,output:K,mode:`inclusive`}).getCommandNodes(t),L(`kept-ids`,[F(`keep`,V),F(`keepScan`,W),I(`keptIds`,q)],w,`if (keep[keepOffset + index] != 0u) { keptIds[keptIdsOffset + keepScan[keepScanOffset + index] - 1u] = index; }`),L(`polygon-layout`,[F(`keepScan`,W),F(`ringScan`,G),F(`ringCounts`,H),F(`keptIds`,q),F(`faceSourceIds`,j.sourceIds),I(`featureOffsets`,C.polygons.featureOffsets),I(`polygonOffsets`,C.polygons.polygonOffsets),I(`sourceIds`,C.polygons.sourceIds)],w+1,`let kept = keepScan[keepScanOffset + ${R}u];
  featureOffsets[featureOffsetsOffset + index] = min(index, kept);
  var ringStart = ringScan[ringScanOffset + ${R}u];
  if (index < kept) {
    let face = keptIds[keptIdsOffset + index];
    ringStart = ringScan[ringScanOffset + face] - ringCounts[ringCountsOffset + face];
  }
  polygonOffsets[polygonOffsetsOffset + index] = ringStart;
  if (index < ${w}u) {
    var group = ${N};
    if (index < kept) { group = faceSourceIds[faceSourceIdsOffset + keptIds[keptIdsOffset + index]]; }
    sourceIds[sourceIdsOffset + index] = group;
  }`),L(`vertex-starts`,[F(`keepScan`,W),F(`vertexScan`,K),F(`vertexCounts`,U),F(`keptIds`,q),I(`keptVertexStarts`,J)],w+1,`let kept = keepScan[keepScanOffset + ${R}u];
  var start = vertexScan[vertexScanOffset + ${R}u];
  if (index < kept) {
    let face = keptIds[keptIdsOffset + index];
    start = vertexScan[vertexScanOffset + face] - vertexCounts[vertexCountsOffset + face];
  }
  keptVertexStarts[keptVertexStartsOffset + index] = start;`),L(`ring-offsets`,[F(`keepScan`,W),F(`keptIds`,q),F(`polygonOffsets`,C.polygons.polygonOffsets),F(`keptVertexStarts`,J),F(`facePolygonOffsets`,j.polygonOffsets),F(`faceRingOffsets`,j.ringOffsets),I(`ringOffsets`,C.polygons.ringOffsets)],w+1,`let kept = keepScan[keepScanOffset + ${R}u];
  if (index >= polygonOffsets[polygonOffsetsOffset + ${w}u]) {
    ringOffsets[ringOffsetsOffset + index] = keptVertexStarts[keptVertexStartsOffset + ${w}u];
    return;
  }
  var low = 0u;
  var high = kept;
  while (low + 1u < high) {
    let middle = low + (high - low) / 2u;
    if (polygonOffsets[polygonOffsetsOffset + middle] <= index) { low = middle; } else { high = middle; }
  }
  let firstRing = facePolygonOffsets[facePolygonOffsetsOffset + keptIds[keptIdsOffset + low]];
  let sourceRing = firstRing + (index - polygonOffsets[polygonOffsetsOffset + low]);
  ringOffsets[ringOffsetsOffset + index] = keptVertexStarts[keptVertexStartsOffset + low] + faceRingOffsets[faceRingOffsetsOffset + sourceRing] - faceRingOffsets[faceRingOffsetsOffset + firstRing];`),L(`gather-vertices`,[F(`keepScan`,W),F(`keptIds`,q),F(`keptVertexStarts`,J),F(`facePolygonOffsets`,j.polygonOffsets),F(`faceRingOffsets`,j.ringOffsets),F(`facePositions`,j.positions,`f32`),I(`positions`,C.polygons.positions,`f32`)],T,`if (index >= keptVertexStarts[keptVertexStartsOffset + ${w}u]) { return; }
  var low = 0u;
  var high = keepScan[keepScanOffset + ${R}u];
  while (low + 1u < high) {
    let middle = low + (high - low) / 2u;
    if (keptVertexStarts[keptVertexStartsOffset + middle] <= index) { low = middle; } else { high = middle; }
  }
  let firstRing = facePolygonOffsets[facePolygonOffsetsOffset + keptIds[keptIdsOffset + low]];
  let source = faceRingOffsets[faceRingOffsetsOffset + firstRing] + (index - keptVertexStarts[keptVertexStartsOffset + low]);
  positions[positionsOffset + 2u * index] = facePositions[facePositionsOffset + 2u * source];
  positions[positionsOffset + 2u * index + 1u] = facePositions[facePositionsOffset + 2u * source + 1u];`),L(`publish-count`,[F(`keepScan`,W),I(`count`,C.status.count)],1,`count[countOffset] = keepScan[keepScanOffset + ${R}u];`)),s.sourceIds&&b.push(L(`publish-source-ids`,[F(`keepScan`,W),F(`ringFeatureRows`,m),F(`ringSourceIds`,p),I(`sourceIds`,C.polygons.sourceIds)],w,`if (index < keepScan[keepScanOffset + ${R}u]) {
  let feature = sourceIds[sourceIdsOffset + index];
  var low = 0u;
  var high = ${f}u;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (ringFeatureRows[ringFeatureRowsOffset + middle] < feature) { low = middle + 1u; }
    else { high = middle; }
  }
  sourceIds[sourceIdsOffset + index] = ringSourceIds[ringSourceIdsOffset + low];
}`)),b}};export{C as t};