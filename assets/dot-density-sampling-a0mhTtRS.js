import{r as e}from"./kernel-C3tfQOn4.js";import{t}from"./gpu-scan-B4NNTfF9.js";import{n,o as r,t as i}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as a,O as o,S as s,T as c,f as l,m as u,u as d,v as f,y as p}from"./gpu-contributor-utils-BIwLQON5.js";import{n as m,t as h}from"./philox-random-DfVxWNPn.js";import{i as g,n as _,t as ee}from"./gpu-polygon-triangulation-CBZGZXVA.js";import{n as v}from"./sorted-segment-sums-Dvmm6OdR.js";import{c as y,d as b}from"./wgsl-common-DfyXiYB0.js";import{a as x,i as S}from"./graph-chunk-dispatch-zBt8k2-o.js";import{i as C}from"./spatial-wgsl-point-helpers-DcOgpqmI.js";var w=h.dotDensityRemainder.id,T=h.dotDensityCandidate.id;function E(e,t){return Math.min(t+1,Math.floor(4294967295/Math.max(e,1)),2**31)}function D(e,t=new Uint32Array(8)){if(t.length<8)throw Error(`Dot density parameter target must hold 8 elements`);if(!Number.isInteger(e.seed)||e.seed<0||e.seed>4294967295)throw Error(`Dot density seed must be a uint32 integer`);let n=new Float32Array(t.buffer,t.byteOffset,t.length),r=e.maskExtent??[0,0,1,1];return t[0]=e.seed,n[1]=e.dotsPerUnit??1,n[2]=r[0],n[3]=r[1],n[4]=r[2],n[5]=r[3],t[6]=0,t[7]=0,t}var O=`4.0e9`;function te(n,i){let{id:a,operation:o,triangleAreas:s,featureTriangleOffsets:c,featureValid:l,featureAreas:u}=i,d=s.length;x(n,a,{triangleAreas:s,featureTriangleOffsets:c,featureValid:l,featureAreas:u,cumulativeAreas:i.cumulativeAreas}),f(n.device,a,`triangle CDF scratch`,Math.max(d,1)*4,`reduce the triangle count`);let p=e(n,`${a}-quantized`,`uint32`,d),m=e(n,`${a}-segments`,`uint32`,d),h=e(n,`${a}-carries`,`uint32`,d),g=e(n,`${a}-carry-prefixes`,`uint32`,d),_=n.createDataView(i.cumulativeAreas.buffer,{format:`uint32`,byteOffset:i.cumulativeAreas.byteOffset,length:d});return[r(n,{id:`${a}-quantize`,operation:o,variant:`triangle-cdf-quantize`,bindings:[{name:`triangleAreas`,view:s,type:`f32`,access:`read`},{name:`featureTriangleOffsets`,view:c,type:`u32`,access:`read`},{name:`featureValid`,view:l,type:`u32`,access:`read`},{name:`featureAreas`,view:u,type:`f32`,access:`read`},{name:`quantizedAreas`,view:p,type:`u32`,access:`read_write`},{name:`segmentFlags`,view:m,type:`u32`,access:`read_write`}],invocationCount:d,declarations:y,body:`let featureCount = arrayLength(&featureAreas) - featureAreasOffset;
  var low = 0u;
  var high = featureCount;
  // Last feature start at or before this triangle, including repeated offsets.
  while (low < high) {
    let middle = low + (high - low + 1u) / 2u;
    if (featureTriangleOffsets[featureTriangleOffsetsOffset + middle] <= index) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  var quantized = 0u;
  var startsSegment = 1u;
  if (low < featureCount && index >= featureTriangleOffsets[featureTriangleOffsetsOffset]) {
    startsSegment = select(0u, 1u, index == featureTriangleOffsets[featureTriangleOffsetsOffset + low]);
    let featureArea = featureAreas[featureAreasOffset + low];
    let area = triangleAreas[triangleAreasOffset + index];
    if (featureValid[featureValidOffset + low] == 1u && isFiniteBits(featureArea) && featureArea > 0.0 && isFiniteBits(area)) {
      quantized = u32(clamp(area / featureArea, 0.0, 1.0) * ${O});
    }
  }
  quantizedAreas[quantizedAreasOffset + index] = quantized;
  segmentFlags[segmentFlagsOffset + index] = startsSegment;`}),...new t({id:`${a}-scan`,input:p,output:_,mode:`inclusive`,segmentFlags:m}).getCommandNodes(n),r(n,{id:`${a}-detect-carry`,operation:o,variant:`triangle-cdf-detect-carry`,bindings:[{name:`quantizedAreas`,view:p,type:`u32`,access:`read`},{name:`cumulativeAreas`,view:_,type:`u32`,access:`read`},{name:`carryFlags`,view:h,type:`u32`,access:`read_write`}],invocationCount:d,body:`carryFlags[carryFlagsOffset + index] = select(0u, 1u, cumulativeAreas[cumulativeAreasOffset + index] < quantizedAreas[quantizedAreasOffset + index]);`}),...new t({id:`${a}-carry-scan`,input:h,output:g,mode:`inclusive`,segmentFlags:m}).getCommandNodes(n),r(n,{id:a,operation:o,variant:`triangle-cdf-saturate`,bindings:[{name:`carryPrefixes`,view:g,type:`u32`,access:`read`},{name:`cumulativeAreas`,view:_,type:`u32`,access:`read_write`}],invocationCount:d,body:`if (carryPrefixes[carryPrefixesOffset + index] != 0u) {
    cumulativeAreas[cumulativeAreasOffset + index] = 0xffffffffu;
  }`})]}function k(e,t){let n=Object.entries(t).filter(e=>!!e[1]);for(let t=0;t<n.length;t++){let[r,i]=n[t];if(i.length===0)continue;let a=i.buffer.defaultBuffer?o(i.buffer.defaultBuffer):i.buffer;for(let s=t+1;s<n.length;s++){let[t,l]=n[s];if(l.length===0)continue;let u=l.buffer.defaultBuffer?o(l.buffer.defaultBuffer):l.buffer;c(e,a!==u||i.byteOffset+i.length*i.rowByteLength<=l.byteOffset||l.byteOffset+l.length*l.rowByteLength<=i.byteOffset,`${r} and ${t} logical byte ranges must not overlap`)}}}var A=`0xffffffffu`,j=4;function ne(e){return d(e,16)}var M=256,N=Math.floor(4294967295/M);function P(e){let{id:t,output:n,mask:r}=e;a(e.positions,[`float32x2`],`${t} positions`),a(e.featureOffsets,[`uint32`],`${t} featureOffsets`),a(e.polygonOffsets,[`uint32`],`${t} polygonOffsets`),a(e.ringOffsets,[`uint32`],`${t} ringOffsets`),l(t,`polygonOffsets rows`,e.polygonOffsets.length,1),l(t,`ringOffsets rows`,e.ringOffsets.length,1),c(t,e.polygonOffsets.length>1||e.positions.length===0,`positions must be empty when polygonOffsets describes no polygons`);let i=e.featureOffsets.length-1;if(i<1)throw Error(`${t} featureOffsets must describe at least one feature`);l(t,`categoryCount`,e.categoryCount);let o=i*e.categoryCount;if(l(t,`featureCount * categoryCount`,o),e.source.kind===`values`){if(a(e.source.values,[`float32`],`${t} values`),e.source.values.length!==o)throw Error(`${t} values must have featureCount * categoryCount rows`)}else if(a(e.source.counts,[`uint32`],`${t} counts`),e.source.counts.length!==i)throw Error(`${t} counts must have one row per feature`);if(a(e.parameters,[`uint32`],`${t} parameters`),e.parameters.length<8)throw Error(`${t} parameters must contain 8 uint32 rows`);if(l(t,`maximumAttempts`,e.maximumAttempts),r?.normalizeFeatures&&s(t,`normalized feature count`,i,N),r&&(a(r.weights,[`float32`],`${t} mask.weights`),!Number.isInteger(r.width)||!Number.isInteger(r.height)||r.width<1||r.height<1||r.weights.length!==r.width*r.height))throw Error(`${t} mask.weights must have mask.width * mask.height rows`);a(n.positions,[`float32x2`],`${t} output.positions`),u(t,n.dots);let d=n.positions.length;if(c(t,E(o,d)>d||!!n.dots.status.candidateOverflow,`output.dots.status.candidateOverflow is required when the per-slot limit can hide loss below final capacity`),d<1||n.dots.ids.length!==d)throw Error(`${t} output.positions and output.dots.ids must share a positive capacity`);if(n.categories&&(a(n.categories,[`uint32`],`${t} output.categories`),n.categories.length!==d))throw Error(`${t} output.categories must have capacity rows`);let f=n.status?.failedCount;if(f&&(a(f,[`uint32`],`${t} output.status.failedCount`),f.length<1))throw Error(`${t} output.status.failedCount must contain one uint32 row`);for(let[e,r]of[[`slotCounts`,n.slotCounts],[`slotOffsets`,n.slotOffsets]])if(r&&(a(r,[`uint32`],`${t} output.${e}`),r.length!==o))throw Error(`${t} output.${e} must have one row per slot`);let m=F(e);return k(t,m),S(t,{ids:n.dots.ids,categories:n.categories,slotCounts:n.slotCounts,slotOffsets:n.slotOffsets}),S(t,{slotCounts:n.slotCounts,failedCount:n.status?.failedCount}),S(t,{positions:n.positions,failedCount:n.status?.failedCount}),S(t,{count:n.dots.status.count,overflow:n.dots.status.overflow,candidateOverflow:n.dots.status.candidateOverflow,requiredCount:n.dots.status.requiredCount}),p(t,Object.values(m),I(e)),i}function F(e){let{output:t}=e;return{"output.positions":t.positions,"output.dots.ids":t.dots.ids,"output.dots.status.count":t.dots.status.count,"output.dots.status.overflow":t.dots.status.overflow,"output.dots.status.candidateOverflow":t.dots.status.candidateOverflow,"output.dots.status.requiredCount":t.dots.status.requiredCount,"output.categories":t.categories,"output.status.failedCount":t.status?.failedCount,"output.slotCounts":t.slotCounts,"output.slotOffsets":t.slotOffsets}}function I(e){return[e.positions,e.featureOffsets,e.polygonOffsets,e.ringOffsets,e.source.kind===`values`?e.source.values:e.source.counts,e.parameters,...e.mask?[e.mask.weights]:[]]}function L(a,o,c){let{id:l,operation:u,output:d,mask:p,categoryCount:h}=o;x(a,l,{...Object.fromEntries(I(o).map((e,t)=>[`input ${t}`,e])),...F(o)});let y=c*h,S=d.positions.length;s(l,`output.positions rows`,S,ne(a.device));let D=E(y,S),O=o.polygonOffsets.length-1,k=Math.floor(_(o.positions.length,o.ringOffsets.length-1,O)/3);g(a.device,l,o.positions.length,o.ringOffsets.length-1,O);let N=!!p?.normalizeFeatures;for(let[e,t]of[[`triangle indices`,Math.max(3*k,1)],[`triangle CDF scratch`,Math.max(k,1)],[`cumulative areas`,Math.max(k,1)+(N?c:0)],[`feature offsets`,c+1],[`slot scratch`,y],[`dot slots`,S*j],[`triangulation polygon layout`,4*O+2],[`triangulation ring layout`,2*(o.ringOffsets.length-1)+1]])f(a.device,l,e,Math.max(t,1)*4,`reduce the geometry, category count or output capacity`);let P=[],L=e(a,`${l}-triangle-indices`,`uint32`,Math.max(3*k,1)),R=e(a,`${l}-polygon-valid`,`uint32`,O),z=e(a,`${l}-polygon-triangle-offsets`,`uint32`,O+1),B=e(a,`${l}-triangle-areas`,`float32`,Math.max(k,1)),V=Math.max(k,1),H=e(a,`${l}-cumulative-areas`,`uint32`,V+(N?c:0)),U=e(a,`${l}-feature-triangle-offsets`,`uint32`,c+1),W=e(a,`${l}-feature-valid`,`uint32`,c),G=e(a,`${l}-feature-areas`,`float32`,c),K=d.slotCounts??e(a,`${l}-slot-counts`,`uint32`,y),q=d.slotOffsets??e(a,`${l}-slot-offsets`,`uint32`,y),J=e(a,`${l}-clamped`,`uint32`,1),Y=e(a,`${l}-total`,`uint32`,1),X=e(a,`${l}-dot-slots`,`uint32`,S*j),Z=d.status?.failedCount??e(a,`${l}-failed`,`uint32`,1);P.push(...new ee({id:`${l}-triangulation`,positions:o.positions,featureOffsets:o.featureOffsets,polygonOffsets:o.polygonOffsets,ringOffsets:o.ringOffsets,output:{indices:L,valid:R,triangleOffsets:z}}).getCommandNodes(a)),P.push(r(a,{id:`${l}-triangle-areas`,operation:u,variant:`triangle-areas`,bindings:[{name:`polygonPositions`,view:o.positions,type:`f32`,access:`read`},{name:`triangleIndices`,view:L,type:`u32`,access:`read`},{name:`polygonTriangleOffsets`,view:z,type:`u32`,access:`read`},{name:`triangleAreas`,view:B,type:`f32`,access:`read_write`}],invocationCount:Math.max(k,1),declarations:`
${C(`readVertex`,`polygonPositions`)}`,body:`
  var area = 0.0;
  if (index < polygonTriangleOffsets[polygonTriangleOffsetsOffset + ${O}u]) {
    let base = triangleIndicesOffset + 3u * index;
    let firstIndex = triangleIndices[base];
    let secondIndex = triangleIndices[base + 1u];
    let thirdIndex = triangleIndices[base + 2u];
    if (firstIndex == secondIndex && firstIndex == thirdIndex) {
      triangleAreas[triangleAreasOffset + index] = 0.0;
      return;
    }
    let first = readVertex(firstIndex);
    // Edge vectors from one vertex keep the cross product accurate far from the origin.
    let edgeOne = readVertex(triangleIndices[base + 1u]) - first;
    let edgeTwo = readVertex(triangleIndices[base + 2u]) - first;
    area = 0.5 * abs(edgeOne.x * edgeTwo.y - edgeOne.y * edgeTwo.x);
  }
  triangleAreas[triangleAreasOffset + index] = area;`})),P.push(r(a,{id:`${l}-feature-triangles`,operation:u,variant:`feature-triangles`,bindings:[{name:`featureOffsets`,view:o.featureOffsets,type:`u32`,access:`read`},{name:`polygonTriangleOffsets`,view:z,type:`u32`,access:`read`},{name:`polygonValid`,view:R,type:`u32`,access:`read`},{name:`featureTriangleOffsets`,view:U,type:`u32`,access:`read_write`},{name:`featureValid`,view:W,type:`u32`,access:`read_write`}],invocationCount:c+1,declarations:`
const FEATURE_COUNT: u32 = ${c}u;
const POLYGON_COUNT: u32 = ${O}u;
const TRIANGLE_COUNT: u32 = ${k}u;
fn getTriangleStart(polygon: u32) -> u32 {
  return min(polygonTriangleOffsets[polygonTriangleOffsetsOffset + polygon], TRIANGLE_COUNT);
}`,body:`
  let firstPolygon = min(featureOffsets[featureOffsetsOffset + index], POLYGON_COUNT);
  featureTriangleOffsets[featureTriangleOffsetsOffset + index] = getTriangleStart(firstPolygon);
  if (index < FEATURE_COUNT) {
    let endPolygon = min(featureOffsets[featureOffsetsOffset + index + 1u], POLYGON_COUNT);
    var valid = 1u;
    for (var polygon = firstPolygon; polygon < endPolygon; polygon = polygon + 1u) {
      valid = valid & polygonValid[polygonValidOffset + polygon];
    }
    featureValid[featureValidOffset + index] = valid;
  }`})),P.push(v(a,{id:`${l}-feature-areas`,operation:u,segmentCount:c,input:B,segmentOffsets:U,output:G})),P.push(...te(a,{id:`${l}-cumulative-areas`,operation:u,triangleAreas:B,featureTriangleOffsets:U,featureValid:W,featureAreas:G,cumulativeAreas:H})),p&&N&&P.push(r(a,{id:`${l}-feature-mask-maximum`,operation:u,variant:`feature-mask-maximum`,bindings:[{name:`polygonPositions`,view:o.positions,type:`f32`,access:`read`},{name:`featureOffsets`,view:o.featureOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:o.polygonOffsets,type:`u32`,access:`read`},{name:`ringOffsets`,view:o.ringOffsets,type:`u32`,access:`read`},{name:`parameters`,view:o.parameters,type:`u32`,access:`read`},{name:`maskWeights`,view:p.weights,type:`f32`,access:`read`},{name:`cumulativeAreas`,view:H,type:`u32`,access:`read_write`}],invocationCount:c*M,workgroupSize:M,guardIndex:!1,declarations:`${b}
${C(`readVertex`,`polygonPositions`)}
const WORKGROUP_SIZE: u32 = ${M}u;
const MASK_WIDTH: u32 = ${p.width}u;
const MASK_HEIGHT: u32 = ${p.height}u;
const POLYGON_COUNT: u32 = ${O}u;
const RING_COUNT: u32 = ${o.ringOffsets.length-1}u;
const VERTEX_COUNT: u32 = ${o.positions.length}u;
const TRIANGLE_SLOTS: u32 = ${V}u;
var<workgroup> partialMinimum: array<vec2<f32>, ${M}>;
var<workgroup> partialMaximum: array<vec2<f32>, ${M}>;
var<workgroup> partialWeight: array<f32, ${M}>;`,body:`
  let lane = localInvocationIndex;
  // Keep the feature independent of localInvocationIndex in WGSL's uniformity analysis.
  // Dividing index by WORKGROUP_SIZE is mathematically equivalent, but remains non-uniform.
  let feature = workgroupIndex;
  // A bounded multidimensional dispatch may pad whole workgroups. This guard is uniform.
  if (feature >= arrayLength(&featureOffsets) - featureOffsetsOffset - 1u) { return; }
  // Vertex range of the feature: first vertex of its first ring to the end of its last polygon.
  let firstPolygon = min(featureOffsets[featureOffsetsOffset + feature], POLYGON_COUNT);
  let endPolygon = min(featureOffsets[featureOffsetsOffset + feature + 1u], POLYGON_COUNT);
  let firstVertex = min(ringOffsets[ringOffsetsOffset + min(polygonOffsets[polygonOffsetsOffset + firstPolygon], RING_COUNT)], VERTEX_COUNT);
  let endVertex = min(ringOffsets[ringOffsetsOffset + min(polygonOffsets[polygonOffsetsOffset + endPolygon], RING_COUNT)], VERTEX_COUNT);
  var minimum = vec2<f32>(3.4e38);
  var maximum = vec2<f32>(-3.4e38);
  for (var vertex = firstVertex + lane; vertex < endVertex; vertex = vertex + WORKGROUP_SIZE) {
    let point = readVertex(vertex);
    if (!isNanBits(point.x) && !isNanBits(point.y) && abs(point.x) < 3.4e38 && abs(point.y) < 3.4e38) {
      minimum = min(minimum, point);
      maximum = max(maximum, point);
    }
  }
  partialMinimum[lane] = minimum;
  partialMaximum[lane] = maximum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
    if (lane < stride) {
      partialMinimum[lane] = min(partialMinimum[lane], partialMinimum[lane + stride]);
      partialMaximum[lane] = max(partialMaximum[lane], partialMaximum[lane + stride]);
    }
    workgroupBarrier();
  }
  // Cell box of the feature (one cell of margin absorbs rounding of sampled points).
  let origin = vec2<f32>(bitcast<f32>(parameters[parametersOffset + 2u]), bitcast<f32>(parameters[parametersOffset + 3u]));
  let cell = vec2<f32>(bitcast<f32>(parameters[parametersOffset + 4u]), bitcast<f32>(parameters[parametersOffset + 5u]));
  let low = (partialMinimum[0] - origin) / cell;
  let high = (partialMaximum[0] - origin) / cell;
  var firstColumn = 0;
  var lastColumn = -1;
  var firstRow = 0;
  var lastRow = -1;
  if (partialMinimum[0].x <= partialMaximum[0].x && !isNanBits(low.x) && !isNanBits(low.y) &&
      !isNanBits(high.x) && !isNanBits(high.y) && high.x >= 0.0 && high.y >= 0.0 &&
      low.x < f32(MASK_WIDTH) && low.y < f32(MASK_HEIGHT)) {
    firstColumn = max(i32(floor(max(low.x, 0.0))) - 1, 0);
    lastColumn = min(i32(floor(min(high.x, f32(MASK_WIDTH)))) + 1, i32(MASK_WIDTH) - 1);
    firstRow = max(i32(floor(max(low.y, 0.0))) - 1, 0);
    lastRow = min(i32(floor(min(high.y, f32(MASK_HEIGHT)))) + 1, i32(MASK_HEIGHT) - 1);
  }
  let columnCount = u32(max(lastColumn - firstColumn + 1, 0));
  let cellCount = columnCount * u32(max(lastRow - firstRow + 1, 0));
  var best = 0.0;
  for (var item = lane; item < cellCount; item = item + WORKGROUP_SIZE) {
    let column = u32(firstColumn) + item % columnCount;
    let row = u32(firstRow) + item / columnCount;
    let weight = maskWeights[maskWeightsOffset + row * MASK_WIDTH + column];
    if (!isNanBits(weight)) {
      best = max(best, weight);
    }
  }
  partialWeight[lane] = best;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
    if (lane < stride) {
      partialWeight[lane] = max(partialWeight[lane], partialWeight[lane + stride]);
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    cumulativeAreas[cumulativeAreasOffset + TRIANGLE_SLOTS + feature] = bitcast<u32>(partialWeight[0]);
  }`})),P.push(i(a,{id:`${l}-clear-failed`,operation:u,view:Z,type:`u32`,value:`0u`,componentCount:1})),P.push(i(a,{id:`${l}-clear-clamped`,operation:u,view:J,type:`u32`,value:`0u`}));let re=o.source.kind===`values`?{name:`sourceValues`,view:o.source.values,type:`f32`,access:`read`}:{name:`sourceCounts`,view:o.source.counts,type:`u32`,access:`read`};P.push(r(a,{id:`${l}-counts`,operation:u,variant:`counts`,bindings:[re,{name:`featureAreas`,view:G,type:`f32`,access:`read`},{name:`failed`,view:Z,type:`atomic<u32>`,access:`read_write`},{name:`parameters`,view:o.parameters,type:`u32`,access:`read`},{name:`slotCounts`,view:K,type:`u32`,access:`read_write`},{name:`clampFlag`,view:J,type:`atomic<u32>`,access:`read_write`}],invocationCount:y,declarations:`${m}
${b}
const SLOT_LIMIT: u32 = ${D}u;
const CATEGORY_COUNT: u32 = ${h}u;`,body:o.source.kind===`values`?`
  let seed = parameters[parametersOffset];
  let dotsPerUnit = bitcast<f32>(parameters[parametersOffset + 1u]);
  let scaled = sourceValues[sourceValuesOffset + index] * dotsPerUnit;
  var dots = 0u;
  var clamped = false;
  if (!isNanBits(scaled) && scaled > 0.0) {
    let remainder = philoxUnitFloat(
      philox4x32(vec4<u32>(index, 0u, 0u, 0u), vec2<u32>(seed, ${w}u)).x
    );
    let ceiling = ceil(scaled - remainder);
    if (ceiling > f32(SLOT_LIMIT)) {
      dots = SLOT_LIMIT;
      clamped = true;
    } else if (ceiling > 0.0) {
      dots = u32(ceiling);
      if (dots > SLOT_LIMIT) {
        dots = SLOT_LIMIT;
        clamped = true;
      }
    }
  }
  if (featureAreas[featureAreasOffset + index / CATEGORY_COUNT] == 0.0) {
    atomicAdd(&failed[failedOffset], dots);
    dots = 0u;
  }
  slotCounts[slotCountsOffset + index] = dots;
  if (clamped) {
    atomicStore(&clampFlag[clampFlagOffset], 1u);
  }`:`
  _ = parameters[parametersOffset];
  var dots = sourceCounts[sourceCountsOffset + index];
  if (dots > SLOT_LIMIT) {
    dots = SLOT_LIMIT;
    atomicStore(&clampFlag[clampFlagOffset], 1u);
  }
  if (featureAreas[featureAreasOffset + index / CATEGORY_COUNT] == 0.0) {
    atomicAdd(&failed[failedOffset], dots);
    dots = 0u;
  }
  slotCounts[slotCountsOffset + index] = dots;`})),P.push(...new t({id:`${l}-slot-scan`,input:K,output:q,mode:`exclusive`}).getCommandNodes(a)),P.push(r(a,{id:`${l}-total`,operation:u,variant:`total`,bindings:[{name:`slotCounts`,view:K,type:`u32`,access:`read`},{name:`slotOffsets`,view:q,type:`u32`,access:`read`},{name:`totalOut`,view:Y,type:`u32`,access:`read_write`}],invocationCount:1,body:`let last = ${y-1}u;
  totalOut[totalOutOffset] = slotOffsets[slotOffsetsOffset + last] + slotCounts[slotCountsOffset + last];`}));let Q=[{name:`slotCounts`,view:K,type:`u32`,access:`read`},{name:`slotOffsets`,view:q,type:`u32`,access:`read`},{name:`total`,view:Y,type:`u32`,access:`read`},{name:`featureTriangleOffsets`,view:U,type:`u32`,access:`read`},{name:`dotSlots`,view:X,type:`u32`,access:`read_write`},{name:`featureIds`,view:d.dots.ids,type:`u32`,access:`read_write`}];d.categories&&Q.push({name:`categories`,view:d.categories,type:`u32`,access:`read_write`}),P.push(r(a,{id:`${l}-locate`,operation:u,variant:`locate`,bindings:Q,invocationCount:S,declarations:`const SLOT_COUNT: u32 = ${y}u;
const CATEGORY_COUNT: u32 = ${h}u;`,body:`
  let count = min(total[totalOffset], ${S}u);
  let table = dotSlotsOffset + ${j}u * index;
  if (index >= count) {
    dotSlots[table] = ${A};
    dotSlots[table + 1u] = 0u;
    dotSlots[table + 2u] = 0u;
    dotSlots[table + 3u] = 0u;
    featureIds[featureIdsOffset + index] = ${A};
    ${d.categories?`categories[categoriesOffset + index] = ${A};`:``}
    return;
  }
  // First slot whose inclusive prefix exceeds the dot index; empty slots never match.
  var low = 0u;
  var high = SLOT_COUNT - 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (slotOffsets[slotOffsetsOffset + middle] + slotCounts[slotCountsOffset + middle] > index) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }
  let feature = low / CATEGORY_COUNT;
  dotSlots[table] = low;
  dotSlots[table + 1u] = index - slotOffsets[slotOffsetsOffset + low];
  dotSlots[table + 2u] = featureTriangleOffsets[featureTriangleOffsetsOffset + feature];
  dotSlots[table + 3u] = featureTriangleOffsets[featureTriangleOffsetsOffset + feature + 1u];
  featureIds[featureIdsOffset + index] = feature;
  ${d.categories?`categories[categoriesOffset + index] = low % CATEGORY_COUNT;`:``}`}));let $=[{name:`polygonPositions`,view:o.positions,type:`f32`,access:`read`},{name:`triangleIndices`,view:L,type:`u32`,access:`read`},{name:`cumulativeAreas`,view:H,type:`u32`,access:`read`},{name:`dotSlots`,view:X,type:`u32`,access:`read`},{name:`parameters`,view:o.parameters,type:`u32`,access:`read`},{name:`positions`,view:d.positions,type:`f32`,access:`read_write`},{name:`failed`,view:Z,type:`atomic<u32>`,access:`read_write`}];return p&&$.push({name:`maskWeights`,view:p.weights,type:`f32`,access:`read`}),P.push(r(a,{id:`${l}-sample`,operation:u,variant:p?`sample-masked`:`sample`,bindings:$,invocationCount:S,declarations:`${m}
const CATEGORY_COUNT: u32 = ${h}u;
// Without a mask the first candidate is always kept; with one, candidates are rejected by weight.
const MAXIMUM_ATTEMPTS: u32 = ${p?o.maximumAttempts:1}u;
${C(`readVertex`,`polygonPositions`)}
// Uniform point of a triangle: r1 = sqrt(u1), P = (1 - r1) A + r1 (1 - u2) B + r1 u2 C, written
// relative to A (the same point) for accuracy far from the origin.
fn getTrianglePoint(triangle: u32, u1: f32, u2: f32) -> vec2<f32> {
  let base = triangleIndicesOffset + 3u * triangle;
  let a = readVertex(triangleIndices[base]);
  let edgeB = readVertex(triangleIndices[base + 1u]) - a;
  let edgeC = readVertex(triangleIndices[base + 2u]) - a;
  return a + sqrt(u1) * ((1.0 - u2) * edgeB + u2 * edgeC);
}
${p?`${b}
const MASK_WIDTH: u32 = ${p.width}u;
const MASK_HEIGHT: u32 = ${p.height}u;
fn getMaskWeight(point: vec2<f32>) -> f32 {
  let origin = vec2<f32>(bitcast<f32>(parameters[parametersOffset + 2u]), bitcast<f32>(parameters[parametersOffset + 3u]));
  let cell = vec2<f32>(bitcast<f32>(parameters[parametersOffset + 4u]), bitcast<f32>(parameters[parametersOffset + 5u]));
  let local = (point - origin) / cell;
  if (isNanBits(local.x) || isNanBits(local.y) || local.x < 0.0 || local.y < 0.0 ||
      local.x >= f32(MASK_WIDTH) || local.y >= f32(MASK_HEIGHT)) {
    return 0.0;
  }
  let weight = maskWeights[maskWeightsOffset + u32(local.y) * MASK_WIDTH + u32(local.x)];
  return select(weight, 0.0, isNanBits(weight));
}`:``}`,body:`
  let table = dotSlotsOffset + ${j}u * index;
  let slot = dotSlots[table];
  var nanBits = 0x7fc00000u;
  var position = vec2<f32>(bitcast<f32>(nanBits));
  if (slot != ${A}) {
    let rank = dotSlots[table + 1u];
    let firstTriangle = dotSlots[table + 2u];
    let endTriangle = dotSlots[table + 3u];
    let seed = parameters[parametersOffset];
    // Fixed-point area of the whole feature; 0 when it failed to triangulate or has no area.
    var featureArea = 0u;
    if (endTriangle > firstTriangle) {
      featureArea = cumulativeAreas[cumulativeAreasOffset + endTriangle - 1u];
    }
    ${N?`let featureMaskMaximum = bitcast<f32>(cumulativeAreas[cumulativeAreasOffset + ${V}u + slot / CATEGORY_COUNT]);`:``}
    var accepted = false;
    if (featureArea > 0u) {
      for (var attempt = 0u; attempt < MAXIMUM_ATTEMPTS; attempt = attempt + 1u) {
        let random = philox4x32(vec4<u32>(slot, rank, attempt, 0u), vec2<u32>(seed, ${T}u));
        let areaTarget = min(u32(philoxUnitFloat(random.x) * f32(featureArea)), featureArea - 1u);
        // First triangle whose cumulative area exceeds the target; zero-area triangles never match.
        var low = firstTriangle;
        var high = endTriangle - 1u;
        while (low < high) {
          let middle = (low + high) / 2u;
          if (cumulativeAreas[cumulativeAreasOffset + middle] > areaTarget) {
            high = middle;
          } else {
            low = middle + 1u;
          }
        }
        let candidate = getTrianglePoint(low, philoxUnitFloat(random.y), philoxUnitFloat(random.z));
        ${p?`accepted = philoxUnitFloat(random.w)${N?` * featureMaskMaximum`:``} < getMaskWeight(candidate);`:`accepted = true;`}
        if (accepted) {
          position = candidate;
          break;
        }
      }
    }
    if (!accepted) {
      atomicAdd(&failed[failedOffset], 1u);
    }
  }
  positions[positionsOffset + 2u * index] = position.x;
  positions[positionsOffset + 2u * index + 1u] = position.y;`})),P.push(n(a,{id:`${l}-publish`,operation:u,requiredCount:Y,output:d.dots,candidateOverflowSources:d.dots.status.candidateOverflow?[J]:[]})),P}export{E as a,D as i,P as n,k as r,L as t};