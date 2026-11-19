import{c as e,i as t,o as n,r,s as i,t as a}from"./kernel-C3tfQOn4.js";import{l as o,r as s}from"./gpu-command-graph-BdzdmV6u.js";import{t as c}from"./gpu-dispatch-utils-CbRzFAgT.js";import{t as l}from"./gpu-bvh-CBtZpuN_.js";import{t as u}from"./gpu-flag-offsets-uwVnNtbl.js";import{t as d}from"./gpu-sort-xRm8l5yn.js";import{o as f}from"./wgsl-kernel-nodes-DagJklH2.js";import{i as p}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as m}from"./spatial-join-output-validation-BN_VImU9.js";import{Q as h}from"./dist-BusamGlx.js";import{n as g}from"./scan-total-C4QddMmi.js";import{t as _}from"./scaled-quotient-wgsl-UDyhQag6.js";import{t as v}from"./graph-chunk-groups-BB8ACpQO.js";import{T as y,c as b,f as x,p as S,v as C}from"./gpu-contributor-utils-BIwLQON5.js";import{m as ee,p as w,u as T}from"./wgsl-common-DfyXiYB0.js";var E=4294967295,te=64,D=4096;function ne(e,t,n,r,i,a=0,o=1,s=1){let c=Math.min(Math.floor((Math.min(e.limits.maxStorageBufferBindingSize,e.limits.maxBufferSize)-a)/i),Math.floor((E-Math.ceil(a/4))/(i/4)),Math.floor(E/s));r!==void 0&&x(t,`candidateTileCapacity`,r,1,Math.min(n,c));let l=r??Math.min(n,c);x(t,`candidateTileCapacity (device binding/buffer limit)`,l,1,n);let u=Math.ceil(n/l),d=D+u*(te+o*4);return S(t,d),{capacity:l,pageCount:u,estimatedCommandNodeCount:d}}var re=65535;function O(e,t){return{updateNode:v(e,{id:`${t.id}-query-range`,operation:`SpatialJoinCandidatePages`,invocationCount:1,parameterValues:[t.firstSlot,t.length,t.queryFirst,t.queryLength,t.queryCount,t.metadataRow,Math.min(e.device.limits.maxComputeWorkgroupsPerDimension,re),t.candidateCapacity],bindings:[{name:`offsets`,view:t.offsets,type:`u32`,access:`read_write`},{name:`total`,view:t.total,type:`u32`,access:`read`}],body:`let pageFirst = getChunkParameter(36u);
  let retained = min(total[totalOffset], getChunkParameter(43u));
  let pageEnd = min(retained, pageFirst + getChunkParameter(37u));
  let chunkFirst = getChunkParameter(38u);
  let chunkEnd = chunkFirst + getChunkParameter(39u);
  let rows = getChunkParameter(40u);
  var lower = chunkFirst;
  var upper = chunkEnd;
  loop {
    if (lower >= upper) { break; }
    let middle = lower + (upper - lower) / 2u;
    var end = retained;
    if (middle + 1u < rows) { end = min(retained, offsets[offsetsOffset + middle + 1u]); }
    if (end <= pageFirst) { lower = middle + 1u; } else { upper = middle; }
  }
  let first = lower;
  upper = chunkEnd;
  loop {
    if (lower >= upper) { break; }
    let middle = lower + (upper - lower) / 2u;
    if (offsets[offsetsOffset + middle] < pageEnd) { lower = middle + 1u; } else { upper = middle; }
  }
  let count = select(0u, lower - first, pageFirst < pageEnd);
  let metadata = offsetsOffset + getChunkParameter(41u);
  offsets[metadata] = first;
  offsets[metadata + 1u] = count;
  let groups = count / 256u + select(0u, 1u, count % 256u != 0u);
  let maximum = getChunkParameter(42u);
  var x = min(groups, maximum);
  var y = 1u;
  var z = 1u;
  if (groups > maximum) { y = min((groups + maximum - 1u) / maximum, maximum); }
  if (groups > maximum * maximum) { z = (groups + maximum * maximum - 1u) / (maximum * maximum); }
  offsets[metadata + 2u] = x;
  offsets[metadata + 3u] = y;
  offsets[metadata + 4u] = z;`}),condition:{id:`${t.id}-query-range`,source:`gpu`,mode:`indirect`,buffer:t.offsets.buffer,byteOffset:t.offsets.byteOffset+(t.metadataRow+2)*4}}}var k=`
const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;
const NO_FEATURE: u32 = 0xffffffffu;
${T}`;function A(e,t,n,r){if(!r)return;let i=b(n).map(e=>e.length),a=b(r).map(e=>e.length);if(i.length!==a.length||i.some((e,t)=>e!==a[t]))throw Error(`${e} ${t} must match the chunk topology of points`)}function j(e,n,r,i){let a=e=>e.flatMap(e=>e?b(e):[]),o=a(n),s=a(r),c=Object.fromEntries(r.map((e,t)=>[`outputs[${t}]`,e]));i?m(e,c,i):p(e,c);for(let n of s)if(o.some(e=>t(e,n)))throw Error(`${e} outputs must not overlap inputs or each other`)}function M(e){return(2*e-1)*24+e*4+8}function N(e,t,n,i,a,o,s){let c=2*a-1,u=new l({id:`${t}-bvh`,minima:n,maxima:i,sourceIds:o,leafCapacity:a,nodeMinima:s?.nodeMinima??r(e,`${t}-bvh-node-minima`,`float32x2`,c),nodeMaxima:s?.nodeMaxima??r(e,`${t}-bvh-node-maxima`,`float32x2`,c),nodeChildren:s?.nodeChildren??r(e,`${t}-bvh-node-children`,`uint32x2`,c),leafIds:s?.leafIds??r(e,`${t}-bvh-leaf-ids`,`uint32`,a),count:s?.count??r(e,`${t}-bvh-count`,`uint32`,1),overflow:s?.overflow??r(e,`${t}-bvh-overflow`,`uint32`,1)});return{bvh:u,nodes:u.getCommandNodes(e)}}var P=`${_}
fn quantizeAxis(value: f32, minimum: f32, maximum: f32) -> u32 {
  let extent = maximum - minimum;
  if (!(extent > 0.0) || !isFiniteValue(extent)) { return 0u; }
  let offset = clamp(value, minimum, maximum) - minimum;
  let normalized = clamp(getScaledQuotient(offset, extent, 1.0, 0), 0.0, 1.0);
  return min(u32(normalized * 65535.0 + 0.5), 65535u);
}`,F=`fn getHilbertIndex(cellX: u32, cellY: u32) -> u32 {
  var x = cellX;
  var y = cellY;
  var key = 0u;
  for (var s = 32768u; s > 0u; s = s >> 1u) {
    let rx = select(0u, 1u, (x & s) != 0u);
    let ry = select(0u, 1u, (y & s) != 0u);
    key += s * s * ((3u * rx) ^ ry);
    if (ry == 0u) {
      if (rx == 1u) {
        x = 65535u - x;
        y = 65535u - y;
      }
      let swap = x;
      x = y;
      y = swap;
    }
  }
  return key;
}`;function I(e){return e>=256}var L=256,R=`hilbert`;function z(e,t,n,i,a,o,s,c,l=R){let u=i.length;if(!s||u<2)return N(e,t,i,a,o,void 0,c);let p=r(e,`${t}-sort-scene-bounds`,`float32`,4),m=r(e,`${t}-sort-keys`,`uint32`,u),h=r(e,`${t}-sort-rows`,`uint32`,u),g=r(e,`${t}-sort-sorted-keys`,`uint32`,u),_=r(e,`${t}-sort-sorted-rows`,`uint32`,u),v=r(e,`${t}-sort-sorted-minima`,`float32x2`,u),y=r(e,`${t}-sort-sorted-maxima`,`float32x2`,u),b=[],x=[{name:`featureMinima`,view:i,type:`f32`,access:`read`},{name:`featureMaxima`,view:a,type:`f32`,access:`read`}],S=`${k}
const FEATURE_COUNT: u32 = ${u}u;
fn readFeatureCenter(row: u32) -> vec4f {
  // Returns (center.x, center.y, valid, 0). Halving before the sum avoids overflow near FLT_MAX.
  let minimum = vec2f(featureMinima[featureMinimaOffset + row * 2u], featureMinima[featureMinimaOffset + row * 2u + 1u]);
  let maximum = vec2f(featureMaxima[featureMaximaOffset + row * 2u], featureMaxima[featureMaximaOffset + row * 2u + 1u]);
  let valid = isFiniteValue(minimum.x) && isFiniteValue(minimum.y) &&
    isFiniteValue(maximum.x) && isFiniteValue(maximum.y) &&
    minimum.x <= maximum.x && minimum.y <= maximum.y;
  let center = minimum * 0.5 + maximum * 0.5;
  return vec4f(center, select(0.0, 1.0, valid), 0.0);
}`,C=Math.ceil(u/L),w=r(e,`${t}-sort-tile-bounds`,`float32`,C*4);b.push(f(e,{id:`${t}-sort-tile-bounds`,operation:n,variant:`sort-tile-bounds`,bindings:[...x,{name:`tileBounds`,view:w,type:`f32`,access:`read_write`}],invocationCount:C,declarations:S,body:`let firstRow = index * ${L}u;
  let endRow = min(firstRow + ${L}u, FEATURE_COUNT);
  var localMinimum = vec2f(FLOAT32_MAXIMUM);
  var localMaximum = vec2f(-FLOAT32_MAXIMUM);
  for (var row = firstRow; row < endRow; row++) {
    let center = readFeatureCenter(row);
    if (center.z > 0.5) {
      localMinimum = min(localMinimum, center.xy);
      localMaximum = max(localMaximum, center.xy);
    }
  }
  let base = tileBoundsOffset + index * 4u;
  tileBounds[base] = localMinimum.x;
  tileBounds[base + 1u] = localMinimum.y;
  tileBounds[base + 2u] = localMaximum.x;
  tileBounds[base + 3u] = localMaximum.y;`})),b.push(f(e,{id:`${t}-sort-scene-bounds`,operation:n,variant:`sort-scene-bounds`,bindings:[{name:`tileBounds`,view:w,type:`f32`,access:`read`},{name:`sceneBounds`,view:p,type:`f32`,access:`read_write`}],invocationCount:256,guardIndex:!1,declarations:`const TILE_COUNT: u32 = ${C}u;
const FLOAT32_MAXIMUM: f32 = 3.4028234e38;
var<workgroup> sharedMinima: array<vec2f, 256>;
var<workgroup> sharedMaxima: array<vec2f, 256>;`,body:`// Exactly one workgroup is dispatched, so index == localInvocationIndex.
  var localMinimum = vec2f(FLOAT32_MAXIMUM);
  var localMaximum = vec2f(-FLOAT32_MAXIMUM);
  for (var tile = localInvocationIndex; tile < TILE_COUNT; tile += 256u) {
    let base = tileBoundsOffset + tile * 4u;
    localMinimum = min(localMinimum, vec2f(tileBounds[base], tileBounds[base + 1u]));
    localMaximum = max(localMaximum, vec2f(tileBounds[base + 2u], tileBounds[base + 3u]));
  }
  sharedMinima[localInvocationIndex] = localMinimum;
  sharedMaxima[localInvocationIndex] = localMaximum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      sharedMinima[localInvocationIndex] = min(sharedMinima[localInvocationIndex], sharedMinima[localInvocationIndex + stride]);
      sharedMaxima[localInvocationIndex] = max(sharedMaxima[localInvocationIndex], sharedMaxima[localInvocationIndex + stride]);
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    let minimum = sharedMinima[0];
    let maximum = sharedMaxima[0];
    let hasValid = minimum.x <= maximum.x && minimum.y <= maximum.y;
    sceneBounds[sceneBoundsOffset] = select(0.0, minimum.x, hasValid);
    sceneBounds[sceneBoundsOffset + 1u] = select(0.0, minimum.y, hasValid);
    sceneBounds[sceneBoundsOffset + 2u] = select(0.0, maximum.x, hasValid);
    sceneBounds[sceneBoundsOffset + 3u] = select(0.0, maximum.y, hasValid);
  }`})),b.push(f(e,{id:`${t}-sort-keys`,operation:n,variant:l===`hilbert`?`sort-keys-hilbert`:`sort-keys`,bindings:[...x,{name:`sceneBounds`,view:p,type:`f32`,access:`read`},{name:`sortKeys`,view:m,type:`u32`,access:`read_write`},{name:`sortRows`,view:h,type:`u32`,access:`read_write`}],invocationCount:u,declarations:`${S}
${ee}
${P}
${F}`,body:`sortRows[sortRowsOffset + index] = index;
  let center = readFeatureCenter(index);
  var key = 0xffffffffu;
  if (center.z > 0.5) {
    let quantizedX = quantizeAxis(center.x, sceneBounds[sceneBoundsOffset], sceneBounds[sceneBoundsOffset + 2u]);
    let quantizedY = quantizeAxis(center.y, sceneBounds[sceneBoundsOffset + 1u], sceneBounds[sceneBoundsOffset + 3u]);
    key = ${l===`hilbert`?`getHilbertIndex(quantizedX, quantizedY)`:`spreadBits(quantizedX) | (spreadBits(quantizedY) << 1u)`};
  }
  sortKeys[sortKeysOffset + index] = key;`})),b.push(...new d({id:`${t}-sort-order`,keys:m,values:h,outputKeys:g,outputValues:_,algorithm:`radix`}).getCommandNodes(e)),b.push(f(e,{id:`${t}-sort-gather`,operation:n,variant:`sort-gather`,bindings:[...x,{name:`sortedRows`,view:_,type:`u32`,access:`read`},{name:`sortedMinima`,view:v,type:`f32`,access:`read_write`},{name:`sortedMaxima`,view:y,type:`f32`,access:`read_write`}],invocationCount:u,declarations:`const FEATURE_COUNT: u32 = ${u}u;`,body:`let row = min(sortedRows[sortedRowsOffset + index], FEATURE_COUNT - 1u);
  sortedMinima[sortedMinimaOffset + index * 2u] = featureMinima[featureMinimaOffset + row * 2u];
  sortedMinima[sortedMinimaOffset + index * 2u + 1u] = featureMinima[featureMinimaOffset + row * 2u + 1u];
  sortedMaxima[sortedMaximaOffset + index * 2u] = featureMaxima[featureMaximaOffset + row * 2u];
  sortedMaxima[sortedMaximaOffset + index * 2u + 1u] = featureMaxima[featureMaximaOffset + row * 2u + 1u];`}));let{bvh:T,nodes:E}=N(e,t,v,y,o,_,c);return b.push(...E),{bvh:T,nodes:b}}function B(e,t){return e>=4096&&t>=256}function V(e,t){let{id:n,operation:i,points:a,bvh:o}=t,s=a.length,c=r(e,`${n}-keys`,`uint32`,s),l=r(e,`${n}-rows`,`uint32`,s),u=r(e,`${n}-sorted-keys`,`uint32`,s),f=r(e,`${n}-order`,`uint32`,s),p=[$(e,{id:`${n}-keys`,operation:i,variant:`query-keys-hilbert`,bindings:[{name:`points`,view:a,type:`f32`,access:`read`},{name:`nodeMinima`,view:o.nodeMinima,type:`f32`,access:`read`},{name:`nodeMaxima`,view:o.nodeMaxima,type:`f32`,access:`read`},{name:`sortKeys`,view:c,type:`u32`,access:`read_write`},{name:`sortRows`,view:l,type:`u32`,access:`read_write`}],invocationCount:s,firstRow:0,sourceRowCount:a.length,declarations:`${k}
${P}
${F}`,body:`sortRows[sortRowsOffset + index] = index;
  let point = vec2f(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  var key = 0xffffffffu;
  if (isFiniteValue(point.x) && isFiniteValue(point.y)) {
    let quantizedX = quantizeAxis(point.x, nodeMinima[nodeMinimaOffset], nodeMaxima[nodeMaximaOffset]);
    let quantizedY = quantizeAxis(point.y, nodeMinima[nodeMinimaOffset + 1u], nodeMaxima[nodeMaximaOffset + 1u]);
    key = getHilbertIndex(quantizedX, quantizedY);
  }
  sortKeys[sortKeysOffset + index] = key;`})];return p.push(...new d({id:`${n}-sort`,keys:c,values:l,outputKeys:u,outputValues:f,algorithm:`radix`}).getCommandNodes(e)),{nodes:p,order:f}}function H(e,t){let{id:n,operation:i,points:a,bvh:o}=t,s=r(e,`${n}-keys`,`uint32`,a.length),c=r(e,`${n}-rows`,`uint32`,a.length),l=r(e,`${n}-sorted-keys`,`uint32`,a.length),u=r(e,`${n}-spatial-rows`,`uint32`,a.length),f=r(e,`${n}-row-bases`,`uint32`,a.length),p=r(e,`${n}-chunk-keys`,`uint32`,a.length),m=r(e,`${n}-sorted-chunk-keys`,`uint32`,a.length),h=r(e,`${n}-order`,`uint32`,a.length),g=[],_=0;for(let[t,r]of b(a).entries())r.length>0&&g.push($(e,{id:`${n}-keys-${t}`,operation:i,invocationCount:r.length,firstRow:_,bindings:[{name:`points`,view:r,type:`f32`,access:`read`},{name:`nodeMinima`,view:o.nodeMinima,type:`f32`,access:`read`},{name:`nodeMaxima`,view:o.nodeMaxima,type:`f32`,access:`read`},{name:`keys`,view:s,type:`u32`,access:`read_write`},{name:`rows`,view:c,type:`u32`,access:`read_write`},{name:`rowBases`,view:f,type:`u32`,access:`read_write`}],declarations:`${k}\n${P}\n${F}`,body:`let row = CHUNK_FIRST_ROW + index;
  rows[rowsOffset + row] = row;
  rowBases[rowBasesOffset + row] = CHUNK_FIRST_ROW;
  var key = 0u;
  if (SOURCE_ROW_COUNT > 1u) {
    key = 0xffffffffu;
    let point = vec2f(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
    if (isFiniteValue(point.x) && isFiniteValue(point.y)) {
      key = getHilbertIndex(quantizeAxis(point.x, nodeMinima[nodeMinimaOffset], nodeMaxima[nodeMaximaOffset]),
        quantizeAxis(point.y, nodeMinima[nodeMinimaOffset + 1u], nodeMaxima[nodeMaximaOffset + 1u]));
    }
  }
  keys[keysOffset + row] = key;`})),_+=r.length;return g.push(...new d({id:`${n}-spatial-sort`,keys:s,values:c,outputKeys:l,outputValues:u,algorithm:`radix`}).getCommandNodes(e)),g.push($(e,{id:`${n}-regroup-keys`,operation:i,invocationCount:a.length,firstRow:0,bindings:[{name:`spatialRows`,view:u,type:`u32`,access:`read`},{name:`rowBases`,view:f,type:`u32`,access:`read`},{name:`chunkKeys`,view:p,type:`u32`,access:`read_write`}],body:`chunkKeys[chunkKeysOffset + index] = rowBases[rowBasesOffset + spatialRows[spatialRowsOffset + index]];`})),g.push(...new d({id:`${n}-chunk-sort`,keys:p,values:u,outputKeys:m,outputValues:h,algorithm:`radix`}).getCommandNodes(e)),{nodes:g,order:h}}function U(e,t){let{source:n}=t,r=[],i=k,a;return n.kind===`polygons`?(r.push({name:`polygonPositions`,view:n.positions,type:`f32`,access:`read`},{name:`featureOffsets`,view:n.featureOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:n.polygonOffsets,type:`u32`,access:`read`},{name:`ringOffsets`,view:n.ringOffsets,type:`u32`,access:`read`}),i+=`
const POLYGON_COUNT: u32 = ${n.polygonOffsets.length-1}u;
const RING_COUNT: u32 = ${n.ringOffsets.length-1}u;
const VERTEX_COUNT: u32 = ${n.positions.length}u;`,a=`var minimum = vec2f(FLOAT32_MAXIMUM);
  var maximum = vec2f(-FLOAT32_MAXIMUM);
  let polygonStart = featureOffsets[featureOffsetsOffset + index];
  let polygonEnd = featureOffsets[featureOffsetsOffset + index + 1u];
  if (polygonStart <= polygonEnd && polygonEnd <= POLYGON_COUNT) {
    let ringStart = polygonOffsets[polygonOffsetsOffset + polygonStart];
    let ringEnd = polygonOffsets[polygonOffsetsOffset + polygonEnd];
    if (ringStart <= ringEnd && ringEnd <= RING_COUNT) {
      let vertexStart = ringOffsets[ringOffsetsOffset + ringStart];
      let vertexEnd = min(ringOffsets[ringOffsetsOffset + ringEnd], VERTEX_COUNT);
      for (var vertex = vertexStart; vertex < vertexEnd; vertex++) {
        let position = vec2f(
          polygonPositions[polygonPositionsOffset + vertex * 2u],
          polygonPositions[polygonPositionsOffset + vertex * 2u + 1u]
        );
        if (isFiniteValue(position.x) && isFiniteValue(position.y)) {
          minimum = min(minimum, position);
          maximum = max(maximum, position);
        }
      }
    }
  }`):n.kind===`points`?(r.push({name:`positions`,view:n.positions,type:`f32`,access:`read`}),a=`var minimum = vec2f(FLOAT32_MAXIMUM);
  var maximum = vec2f(-FLOAT32_MAXIMUM);
  let position = vec2f(positions[positionsOffset + index * 2u], positions[positionsOffset + index * 2u + 1u]);
  if (isFiniteValue(position.x) && isFiniteValue(position.y)) {
    minimum = position;
    maximum = position;
  }`):(r.push({name:`starts`,view:n.starts,type:`f32`,access:`read`},{name:`ends`,view:n.ends,type:`f32`,access:`read`}),a=`var minimum = vec2f(FLOAT32_MAXIMUM);
  var maximum = vec2f(-FLOAT32_MAXIMUM);
  let start = vec2f(starts[startsOffset + index * 2u], starts[startsOffset + index * 2u + 1u]);
  let end = vec2f(ends[endsOffset + index * 2u], ends[endsOffset + index * 2u + 1u]);
  if (isFiniteValue(start.x) && isFiniteValue(start.y) && isFiniteValue(end.x) && isFiniteValue(end.y)) {
    minimum = min(start, end);
    maximum = max(start, end);
  }`),r.push({name:`featureMinima`,view:t.minima,type:`f32`,access:`read_write`},{name:`featureMaxima`,view:t.maxima,type:`f32`,access:`read_write`}),f(e,{id:t.id,operation:t.operation,variant:`bounds-${n.kind}`,bindings:r,invocationCount:t.featureCount,declarations:i,body:`${a}
  featureMinima[featureMinimaOffset + index * 2u] = minimum.x;
  featureMinima[featureMinimaOffset + index * 2u + 1u] = minimum.y;
  featureMaxima[featureMaximaOffset + index * 2u] = maximum.x;
  featureMaxima[featureMaximaOffset + index * 2u + 1u] = maximum.y;`})}function W(e,t){let{pairs:n,featureCounts:r,bestDistanceBits:i}=t,a=[{name:`state`,view:t.state,type:`atomic<u32>`,access:`read_write`},{name:`assignment`,view:t.assignment,type:`u32`,access:`read_write`}];n&&a.push({name:`pairPoints`,view:n.points,type:`f32`,access:`read_write`},{name:`pairGeometryOffsets`,view:n.geometryOffsets,type:`u32`,access:`read_write`}),i&&a.push({name:`bestDistanceBits`,view:i,type:`u32`,access:`read_write`}),r&&a.push({name:`featureCounts`,view:r,type:`u32`,access:`read_write`});let o=[4,t.pointCount,n?n.geometryOffsets.length:0,r?.length??0];return f(e,{id:t.id,operation:t.operation,variant:`clear`,bindings:a,invocationCount:Math.max(...o),declarations:k,body:`if (index < 4u) { atomicStore(&state[stateOffset + index], 0u); }
  if (index < ${t.pointCount}u) {
    assignment[assignmentOffset + index] = NO_FEATURE;
    ${i?`bestDistanceBits[bestDistanceBitsOffset + index] = 0xffffffffu;`:``}
  }
  ${n?`if (index < ${n.points.length}u) {
    // A runtime expression keeps the quiet-NaN bit pattern out of const evaluation.
    let quietNaN = bitcast<f32>(0x7fc00000u | (index & 0u));
    pairPoints[pairPointsOffset + index * 2u] = quietNaN;
    pairPoints[pairPointsOffset + index * 2u + 1u] = quietNaN;
  }
  if (index < ${n.geometryOffsets.length}u) {
    pairGeometryOffsets[pairGeometryOffsetsOffset + index] =
      select(0u, ${n.polygonCount}u, index == ${n.geometryOffsets.length-1}u);
  }`:``}
  ${r?`if (index < ${r.length}u) { featureCounts[featureCountsOffset + index] = 0u; }`:``}`})}function G(e,t,n){return t instanceof o?`${e}-${n}`:e}function K(e,t){let n=[],r=`const CANDIDATE_CAPACITY: u32 = ${t.candidateCapacity}u;`,i=t.bvhOverflow?[{name:`bvhOverflow`,view:t.bvhOverflow,type:`u32`,access:`read`}]:[],a=t.bvhOverflow?`bvhOverflow[bvhOverflowOffset] != 0u`:`false`;return t.candidateTruncated&&n.push(f(e,{id:`${t.id}-candidate-truncated`,operation:t.operation,variant:t.bvhOverflow?`candidate-truncated-bvh`:`candidate-truncated`,bindings:[{name:`counts`,view:t.counts,type:`u32`,access:`read`},{name:`offsets`,view:t.offsets,type:`u32`,access:`read`},...i,{name:`candidateTruncated`,view:t.candidateTruncated,type:`u32`,access:`read_write`}],invocationCount:t.rowCount,declarations:r,body:`let count = counts[countsOffset + index];
  let offset = offsets[offsetsOffset + index];
  candidateTruncated[candidateTruncatedOffset + index] =
    select(0u, 1u, ${a} || (count > 0u && (offset >= CANDIDATE_CAPACITY || count > CANDIDATE_CAPACITY - offset)));`})),t.candidateOverflow&&n.push(f(e,{id:`${t.id}-candidate-overflow`,operation:t.operation,variant:t.bvhOverflow?`candidate-overflow-bvh`:`candidate-overflow`,bindings:[{name:`total`,view:t.total,type:`u32`,access:`read`},...i,{name:`candidateOverflow`,view:t.candidateOverflow,type:`u32`,access:`read_write`}],invocationCount:1,declarations:r,body:`candidateOverflow[candidateOverflowOffset] = select(0u, 1u, ${a} || total[totalOffset] > CANDIDATE_CAPACITY);`})),n}function q(e,t){let{candidateTruncated:n,candidateOverflow:r}=t;if(!n&&!r)return[];let i=[{name:`bvhOverflow`,view:t.bvhOverflow,type:`u32`,access:`read`}];return n&&i.push({name:`candidateTruncated`,view:n,type:`u32`,access:`read_write`}),r&&i.push({name:`candidateOverflow`,view:r,type:`u32`,access:`read_write`}),[f(e,{id:`${t.id}-leaf-overflow-status`,operation:t.operation,variant:`leaf-overflow-status-${n?`rows`:`none`}-${r?`flag`:`none`}`,bindings:i,invocationCount:Math.max(t.rowCount,1),declarations:`const ROW_COUNT: u32 = ${t.rowCount}u;`,body:`let leavesLost = select(0u, 1u, bvhOverflow[bvhOverflowOffset] != 0u);
  ${n?`if (index < ROW_COUNT) { candidateTruncated[candidateTruncatedOffset + index] = leavesLost; }`:``}
  ${r?`if (index == 0u) { candidateOverflow[candidateOverflowOffset] = leavesLost; }`:``}`})]}function J(e,t){let n=[],{bvh:i}=t,a=t.page?.counts??r(e,`${t.id}-point-counts`,`uint32`,t.points.length),o=t.page?.offsets??r(e,`${t.id}-point-offsets`,`uint32`,t.points.length),s=()=>K(e,{id:t.id,operation:t.operation,counts:a,offsets:o,rowCount:t.points.length,candidateCapacity:t.candidateCapacity,total:t.state,candidateTruncated:t.candidateTruncated,candidateOverflow:t.candidateOverflow,bvhOverflow:t.bvhOverflow}),c=(e,n,r)=>[{name:`points`,view:e,type:`f32`,access:`read`},{name:`nodeMinima`,view:i.nodeMinima,type:`f32`,access:`read`},{name:`nodeMaxima`,view:i.nodeMaxima,type:`f32`,access:`read`},{name:`leafIds`,view:i.leafIds,type:`u32`,access:`read`},...t.radius?[{name:`radius`,view:t.radius,type:`f32`,access:`read`}]:[],...r?[{name:`queryOrder`,view:r,type:`u32`,access:`read`}]:[],...n],l=()=>`${k}

const INTERNAL_NODE_COUNT: u32 = ${i.internalNodeCount}u;
const FEATURE_COUNT: u32 = ${t.featureCount}u;
const CANDIDATE_CAPACITY: u32 = ${t.candidateCapacity}u;
${w}`,u=t.radius?`let searchRadius = radius[radiusOffset];
  if (!(searchRadius >= 0.0) || searchRadius > FLOAT32_MAXIMUM) { return; }
  let queryMinimum = point - vec2f(searchRadius);
  let queryMaximum = point + vec2f(searchRadius);`:`let queryMinimum = point;
  let queryMaximum = point;`,d=(e,t,n,r,i=!1)=>`${i?`let metadata = pointOffsetsOffset + CANDIDATE_QUERY_METADATA_ROW;
  if (index >= pointOffsets[metadata + 1u]) { return; }
  let row = pointOffsets[metadata] + index;
  let localRow = row - CHUNK_FIRST_ROW;`:r?`let localRow = queryOrder[queryOrderOffset + index] - CHUNK_FIRST_ROW;
  let row = CHUNK_FIRST_ROW + localRow;`:`let row = CHUNK_FIRST_ROW + index;`}
  ${t}
  var found = 0u;
  let point = vec2f(points[pointsOffset + ${r||i?`localRow`:`index`} * 2u], points[pointsOffset + ${r||i?`localRow`:`index`} * 2u + 1u]);
  if (!isFiniteValue(point.x) || !isFiniteValue(point.y)) { return; }
  ${u}
  var node = 0u;
  loop {
    if (nodeOverlaps(node, queryMinimum, queryMaximum)) {
      if (node < INTERNAL_NODE_COUNT) {
        node = node * 2u + 1u;
        continue;
      }
      let featureRow = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT];
      if (featureRow < FEATURE_COUNT) {
        ${e}
      }
    }
    // Climb while the node is a right child, then step to the right sibling.
    loop {
      if (node == 0u || (node & 1u) == 1u) { break; }
      node = (node - 1u) / 2u;
    }
    if (node == 0u) { break; }
    node = node + 1u;
  }
  ${n}`;if(t.points.length===0)return s();let f=t.page?.orders??new Map;if((!t.page||t.page.initialize)&&t.querySort&&b(t.points).some(e=>e.length>=2)){let r=H(e,{id:`${t.id}-order`,operation:t.operation,points:t.points,bvh:i});n.push(...r.nodes);let a=0;for(let[n,i]of b(t.points).entries())i.length>0&&f.set(n,e.createDataView(r.order.buffer,{format:`uint32`,length:i.length,byteOffset:a*Uint32Array.BYTES_PER_ELEMENT})),a+=i.length}let p=0;if(!t.page||t.page.initialize){for(let[r,i]of b(t.points).entries()){if(i.length>0){let o=f.get(r);n.push($(e,{id:G(t.id,t.points,r),operation:t.operation,variant:o?`probe-count-ordered`:`probe-count`,bindings:c(i,[{name:`pointCounts`,view:a,type:`u32`,access:`read_write`}],o),invocationCount:i.length,firstRow:p,declarations:l(),body:d(`found = found + 1u;`,`pointCounts[pointCountsOffset + row] = 0u;`,`pointCounts[pointCountsOffset + row] = found;`,!!o)}))}p+=i.length}n.push(...g(e,{id:`${t.id}-scan`,operation:t.operation,counts:a,offsets:o,rowCount:t.points.length,total:t.state}))}p=0;for(let[r,i]of b(t.points).entries()){if(i.length>0){let a=t.radius?void 0:f.get(r),s=t.page?.metadataRow===void 0?void 0:t.page.metadataRow+r*5,u=s===void 0||!t.page||a?void 0:O(e,{id:`${G(t.id,t.points,r)}-write`,offsets:o,total:t.state,firstSlot:t.page.firstSlot,length:t.page.length,candidateCapacity:t.candidateCapacity,queryFirst:p,queryLength:i.length,queryCount:t.points.length,metadataRow:s});u&&n.push(u.updateNode),n.push($(e,{id:`${G(t.id,t.points,r)}-write`,operation:t.operation,variant:a?`probe-write-ordered`:`probe-write`,bindings:c(i,[{name:`pointOffsets`,view:o,type:`u32`,access:`read`},{name:`candidatePairs`,view:t.candidatePairs,type:`u32`,access:`read_write`},...t.packedCandidates?[]:[{name:`candidatePoints`,view:t.candidatePoints,type:`f32`,access:`read_write`}]],a),invocationCount:i.length,firstRow:p,condition:u?.condition,candidateQueryMetadataRow:s,candidatePageFirstSlot:t.page?.firstSlot??0,candidatePageLength:t.page?.length??t.candidateCapacity,declarations:l(),body:d(t.packedCandidates?`if (found >= skipped) {
        let slot = slotStart + found - skipped;
        candidatePairs[candidatePairsOffset + slot * 4u] = row;
        candidatePairs[candidatePairsOffset + slot * 4u + 1u] = featureRow;
        candidatePairs[candidatePairsOffset + slot * 4u + 2u] = bitcast<u32>(point.x);
        candidatePairs[candidatePairsOffset + slot * 4u + 3u] = bitcast<u32>(point.y);
        if (slot + 1u >= slotEnd) { break; }
        }
        found = found + 1u;`:`if (found >= skipped) {
        let slot = slotStart + found - skipped;
        candidatePairs[candidatePairsOffset + slot * 2u] = row;
        candidatePairs[candidatePairsOffset + slot * 2u + 1u] = featureRow;
        let pointRow = slot * ${t.pointRowScale}u + ${t.pointRowOffset}u;
        candidatePoints[candidatePointsOffset + pointRow * 2u] = point.x;
        candidatePoints[candidatePointsOffset + pointRow * 2u + 1u] = point.y;
        if (slot + 1u >= slotEnd) { break; }
        }
        found = found + 1u;`,`let globalStart = pointOffsets[pointOffsetsOffset + row];
  var globalEnd = CANDIDATE_CAPACITY;
  if (row + 1u < ${t.points.length}u) { globalEnd = min(pointOffsets[pointOffsetsOffset + row + 1u], CANDIDATE_CAPACITY); }
  let pageEnd = CANDIDATE_PAGE_FIRST_SLOT + CANDIDATE_PAGE_LENGTH;
  if (globalStart >= pageEnd || globalEnd <= CANDIDATE_PAGE_FIRST_SLOT) { return; }
  let skipped = max(globalStart, CANDIDATE_PAGE_FIRST_SLOT) - globalStart;
  let slotStart = max(globalStart, CANDIDATE_PAGE_FIRST_SLOT) - CANDIDATE_PAGE_FIRST_SLOT;
  let slotEnd = min(globalEnd, pageEnd) - CANDIDATE_PAGE_FIRST_SLOT;
  if (slotStart >= slotEnd) { return; }`,``,!!a,!!u)}))}p+=i.length}return(!t.page||t.page.initialize)&&n.push(...s()),n}function Y(e,t){let n=[],r=t.distances?b(t.distances):[],i=0;for(let[a,o]of b(t.pointFeatureIds).entries()){if(o.length>0){let s=[{name:`assignment`,view:t.assignment,type:`u32`,access:`read`},{name:`pointFeatureIds`,view:o,type:`u32`,access:`read_write`}];t.featureIds&&s.push({name:`featureIds`,view:t.featureIds,type:`u32`,access:`read`}),t.featureCounts&&s.push({name:`featureCounts`,view:t.featureCounts,type:`atomic<u32>`,access:`read_write`});let c=r[a];t.bestDistanceBits&&c&&s.push({name:`bestDistanceBits`,view:t.bestDistanceBits,type:`u32`,access:`read`},{name:`distances`,view:c,type:`f32`,access:`read_write`}),n.push($(e,{id:G(t.id,t.pointFeatureIds,a),operation:t.operation,variant:`assign`,bindings:s,invocationCount:o.length,firstRow:i,declarations:`${k}
`,body:`let row = CHUNK_FIRST_ROW + index;
  let featureRow = assignment[assignmentOffset + row];
  let matched = featureRow != NO_FEATURE;
  var featureId = featureRow;
  ${t.featureIds?`if (matched) { featureId = featureIds[featureIdsOffset + featureRow]; }`:``}
  pointFeatureIds[pointFeatureIdsOffset + index] = featureId;
  ${t.featureCounts?`if (matched) { atomicAdd(&featureCounts[featureCountsOffset + featureRow], 1u); }`:``}
  ${t.bestDistanceBits&&c?`let bestDistance = bitcast<f32>(bestDistanceBits[bestDistanceBitsOffset + row]);
  distances[distancesOffset + index] = select(-1.0, ${t.squaredDistances?`sqrt(bestDistance)`:`bestDistance`}, matched);`:``}`}))}i+=o.length}return n}function ie(e,t){let n=[],i=t.sourceIds?b(t.sourceIds):[],a=t.matches.ids.length,o=t.points.length;if(o===0)return n;let s=r(e,`${t.id}-flags`,`uint32`,o),c=r(e,`${t.id}-offsets`,`uint32`,o),l=0;for(let[r,i]of b(t.points).entries())i.length>0&&n.push($(e,{id:G(t.id,t.points,r),operation:t.operation,variant:`collect-match-flags`,bindings:[{name:`assignment`,view:t.assignment,type:`u32`,access:`read`},{name:`flags`,view:s,type:`u32`,access:`read_write`}],invocationCount:i.length,firstRow:l,declarations:`${k}
`,body:`let row = CHUNK_FIRST_ROW + index;
  flags[flagsOffset + row] = select(0u, 1u, assignment[assignmentOffset + row] != NO_FEATURE);`})),l+=i.length;if(n.push(...new u({id:t.id,flags:s,offsets:c,count:e.createDataView(t.state.buffer,{format:`uint32`,length:1,byteOffset:t.state.byteOffset+2*Uint32Array.BYTES_PER_ELEMENT})}).getCommandNodes(e)),a===0)return n;l=0;for(let[r,o]of b(t.points).entries()){if(o.length>0){let u=i[r],d=[{name:`flags`,view:s,type:`u32`,access:`read`},{name:`offsets`,view:c,type:`u32`,access:`read`},{name:`matchIds`,view:t.matches.ids,type:`u32`,access:`read_write`}];u&&d.push({name:`sourceIds`,view:u,type:`u32`,access:`read`}),n.push($(e,{id:`${G(t.id,t.points,r)}-write`,operation:t.operation,variant:`collect-match-write`,bindings:d,invocationCount:o.length,firstRow:l,declarations:`${k}

const MATCH_CAPACITY: u32 = ${a}u;`,body:`let row = CHUNK_FIRST_ROW + index;
  if (flags[flagsOffset + row] == 0u) { return; }
  let slot = offsets[offsetsOffset + row];
  if (slot < MATCH_CAPACITY) {
    matchIds[matchIdsOffset + slot] = ${u?`sourceIds[sourceIdsOffset + index]`:`row`};
  }`}))}l+=o.length}return n}function ae(e,t){let{matches:n}=t,r=[{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`bvhOverflow`,view:t.bvhOverflow,type:`u32`,access:`read`},{name:`overflow`,view:t.overflow,type:`u32`,access:`read_write`}];t.candidateCount&&r.push({name:`candidateCount`,view:t.candidateCount,type:`u32`,access:`read_write`}),t.uncertainCount&&r.push({name:`uncertainCount`,view:t.uncertainCount,type:`u32`,access:`read_write`}),n&&(r.push({name:`matchCount`,view:n.status.count,type:`u32`,access:`read_write`},{name:`matchOverflow`,view:n.status.overflow,type:`u32`,access:`read_write`}),n.status.requiredCount&&r.push({name:`matchTotalCount`,view:n.status.requiredCount,type:`u32`,access:`read_write`}));let i=n?.ids.length??0,a=f(e,{id:t.id,operation:t.operation,variant:`finalize`,bindings:r,invocationCount:1,declarations:`const CANDIDATE_CAPACITY: u32 = ${t.candidateCapacity}u;
const MATCH_CAPACITY: u32 = ${i}u;`,body:`let candidateTotal = state[stateOffset];
  let matchTotal = state[stateOffset + 2u];
  let overflowed = bvhOverflow[bvhOverflowOffset] != 0u || candidateTotal > CANDIDATE_CAPACITY${n?` || matchTotal > MATCH_CAPACITY`:``};
  let overflowValue = select(0u, 1u, overflowed);
  overflow[overflowOffset] = overflowValue;
  ${t.candidateCount?`candidateCount[candidateCountOffset] = candidateTotal;`:``}
  ${t.uncertainCount?`uncertainCount[uncertainCountOffset] = state[stateOffset + 1u];`:``}
  ${n?`matchCount[matchCountOffset] = min(matchTotal, MATCH_CAPACITY);
  matchOverflow[matchOverflowOffset] = select(0u, 1u, matchTotal > MATCH_CAPACITY);
  ${n.status.requiredCount?`matchTotalCount[matchTotalCountOffset] = matchTotal;`:``}`:``}`}),o=n?.status.candidateOverflow;return o?[a,f(e,{id:`${t.id}-compact-candidate`,operation:t.operation,variant:`compact-candidate`,bindings:[{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`bvhOverflow`,view:t.bvhOverflow,type:`u32`,access:`read`},{name:`matchCandidateOverflow`,view:o,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`const CANDIDATE_CAPACITY: u32 = ${t.candidateCapacity}u;`,body:`matchCandidateOverflow[matchCandidateOverflowOffset] = select(0u, 1u,
      bvhOverflow[bvhOverflowOffset] != 0u || state[stateOffset] > CANDIDATE_CAPACITY);`})]:[a]}function X(e,t,n){return e*t>=4e6&&e>=8*Math.max(n,1)}function Z(e,t){return Math.max(Math.ceil(e/32),1)+2*t}function oe(e,t){let{id:n,operation:i,polygons:a}=t,o=a.positions.length,s=a.featureOffsets.length-1,c=Math.max(Math.ceil(o/32),1),l=t.layout??r(e,`${n}-edge-layout`,`uint32`,Z(o,s)),u=a.ringOffsets.length-1;return{nodes:[f(e,{id:`${n}-edge-layout-bits`,operation:i,variant:`edge-layout-bits`,bindings:[{name:`ringOffsets`,view:a.ringOffsets,type:`u32`,access:`read`},{name:`edgeLayout`,view:l,type:`u32`,access:`read_write`}],invocationCount:c,declarations:`const VERTEX_COUNT: u32 = ${o}u;
const RING_COUNT: u32 = ${u}u;`,body:`let first = index * 32u;
  let last = min(first + 32u, VERTEX_COUNT);
  var bits = 0u;
  if (RING_COUNT > 0u) {
    // Ring that contains \`first\`: the last ring that starts at or before it.
    var low = 0u;
    var high = RING_COUNT;
    while (low + 1u < high) {
      let middle = (low + high) / 2u;
      if (ringOffsets[ringOffsetsOffset + middle] <= first) { low = middle; } else { high = middle; }
    }
    var ring = low;
    var ringBegin = ringOffsets[ringOffsetsOffset + ring];
    var ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
    for (var vertex = first; vertex < last; vertex++) {
      loop {
        if (vertex < ringEnd || ring + 1u >= RING_COUNT) { break; }
        ring = ring + 1u;
        ringBegin = ringOffsets[ringOffsetsOffset + ring];
        ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
      }
      let usable = vertex >= ringBegin && vertex < ringEnd && ringEnd >= ringBegin + 3u && ringEnd <= VERTEX_COUNT;
      bits = bits | select(0u, 1u << (vertex - first), usable);
    }
  }
  edgeLayout[edgeLayoutOffset + index] = bits;`}),f(e,{id:`${n}-edge-layout-features`,operation:i,variant:`edge-layout-features`,bindings:[{name:`featureOffsets`,view:a.featureOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:a.polygonOffsets,type:`u32`,access:`read`},{name:`ringOffsets`,view:a.ringOffsets,type:`u32`,access:`read`},{name:`edgeLayout`,view:l,type:`u32`,access:`read_write`}],invocationCount:Math.max(s,1),declarations:`const FEATURE_COUNT: u32 = ${s}u;
const POLYGON_COUNT: u32 = ${a.polygonOffsets.length-1}u;
const RING_COUNT: u32 = ${u}u;
const VERTEX_COUNT: u32 = ${o}u;
const FEATURE_BASE: u32 = ${c}u;`,body:`if (index >= FEATURE_COUNT) { return; }
  let partStart = featureOffsets[featureOffsetsOffset + index];
  let partEnd = featureOffsets[featureOffsetsOffset + index + 1u];
  var wellFormed = partStart <= partEnd && partEnd <= POLYGON_COUNT;
  if (wellFormed) {
    for (var part = partStart; part < partEnd && wellFormed; part++) {
      let ringStart = polygonOffsets[polygonOffsetsOffset + part];
      let ringEnd = polygonOffsets[polygonOffsetsOffset + part + 1u];
      if (ringStart > ringEnd || ringEnd > RING_COUNT) { wellFormed = false; }
      for (var ring = ringStart; ring < ringEnd && wellFormed; ring++) {
        let vertexStart = ringOffsets[ringOffsetsOffset + ring];
        let vertexEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
        if (vertexEnd < vertexStart || vertexEnd > VERTEX_COUNT) { wellFormed = false; }
      }
    }
  }
  edgeLayout[edgeLayoutOffset + FEATURE_BASE + index * 2u] = select(0xffffffffu, partStart, wellFormed);
  edgeLayout[edgeLayoutOffset + FEATURE_BASE + index * 2u + 1u] = partEnd;`})],layout:l}}var Q=256,se=8,ce=16;function $(t,r){y(r.id,r.bindings.length<=Math.min(se,t.device.limits.maxStorageBuffersPerShaderStage),`bindings must fit the device storage binding limit and eight runtime offset slots`);for(let e of r.bindings)C(t.device,r.id,e.name,i(e.view).size,`reduce the bound view length`);let o=new Uint32Array(ce);o[0]=r.firstRow,o[1]=r.invocationCount,o[10]=r.sourceRowCount??r.invocationCount,o[11]=r.candidatePageFirstSlot??0,o[12]=r.candidatePageLength??0,o[13]=r.candidateQueryMetadataRow??0;let l=r.declarations??``,u=r.body,d=e=>`chunkSettings[${Math.floor(e/4)}][${e%4}]`;for(let[t,n]of r.bindings.entries()){o[t+2]=e(n.view);let r=RegExp(`\\b${n.name}Offset\\b`,`g`);l=l.replace(r,d(t+2)),u=u.replace(r,d(t+2))}u=u.replace(/\bCHUNK_FIRST_ROW\b/g,d(0)).replace(/\bSOURCE_ROW_COUNT\b/g,d(10)).replace(/\bCANDIDATE_PAGE_FIRST_SLOT\b/g,d(11)).replace(/\bCANDIDATE_PAGE_LENGTH\b/g,d(12)).replace(/\bCANDIDATE_QUERY_METADATA_ROW\b/g,d(13));let f=`${r.bindings.map((e,t)=>`@group(0) @binding(${t}) var<storage, ${e.access}> ${e.name}: array<${e.type}>;`).join(`
`)}
@group(0) @binding(${r.bindings.length}) var<uniform> chunkSettings: array<vec4<u32>, 4>;
${l}
@compute @workgroup_size(${Q})
fn main(@builtin(local_invocation_index) localInvocationIndex: u32,
        @builtin(workgroup_id) workgroupId: vec3<u32>,
        @builtin(num_workgroups) workgroupCounts: vec3<u32>) {
  ${r.bindings.map(e=>`_ = &${e.name};`).join(`
`)}
  let index = ((workgroupId.z * workgroupCounts.y + workgroupId.y) * workgroupCounts.x + workgroupId.x) * ${Q}u + localInvocationIndex;
  if (index >= chunkSettings[0][1]) { return; }
  ${u}
}`,p=c(r.operation,Math.max(1,r.invocationCount),Q,t.device.limits.maxComputeWorkgroupsPerDimension);return s({id:r.id,condition:r.condition,resources:r.bindings.map(e=>({buffer:e.view,usage:e.access===`read`?`storage-read`:`storage-read-write`})),workload:{operation:r.operation,...r.variant?{variant:r.variant}:{},maximumInvocationCount:r.invocationCount},compile:({device:e})=>{let t=e.createBuffer({id:`${r.id}-settings`,data:o,usage:h.UNIFORM|h.COPY_DST}),i=new a(e,{id:r.id,source:f,shaderLayout:{bindings:[...r.bindings.map((e,t)=>({name:e.name,type:e.access===`read`?`read-only-storage`:`storage`,group:0,location:t})),{name:`chunkSettings`,type:`uniform`,group:0,location:r.bindings.length}]}});return{encode:({computePass:e,getBuffer:a})=>{let o={chunkSettings:t};for(let e of r.bindings)o[e.name]=n(e.view,a);i.dispatch(e,{bindings:o,x:p.x,y:p.y,z:p.z})},destroy:()=>{i.destroy(),t.destroy()}}}})}export{j as C,ne as D,O as E,J as S,E as T,oe as _,$ as a,M as b,K as c,B as d,I as f,Z as g,q as h,U as i,G as l,V as m,P as n,W as o,N as p,k as r,ae as s,F as t,X as u,z as v,A as w,ie as x,Y as y};