import{r as e}from"./kernel-C3tfQOn4.js";import{l as t,o as n}from"./gpu-command-graph-BdzdmV6u.js";import{c as r,o as i,t as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{A as o,E as s,S as c,T as l,f as u,x as d}from"./gpu-contributor-utils-BIwLQON5.js";import{n as f,r as p,t as m}from"./gpu-isoband-rings-xm3u6Br0.js";import{t as h}from"./gpu-isobands-BROEsZHL.js";import{t as g}from"./gpu-cell-aggregation-D6NbjCth.js";import{t as _}from"./gpu-point-to-cell-CYNSElfx.js";import{t as v}from"./gpu-cell-set-outline-BjZDfwc8.js";import{r as y,v as b}from"./spatial-context-DwZEd4EZ.js";import{t as x}from"./gpu-network-service-areas-DYSymXtS.js";import{t as S}from"./gpu-network-reachability-Coypydjh.js";import{n as C,o as w,s as T}from"./network-view-utils-B1U8oNKx.js";import{a as E}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as D}from"./construction-snapshot-DdkX9Pc0.js";import{c as O}from"./wgsl-common-DfyXiYB0.js";function k(e,t=new Float32Array(12)){if(t.length<12)throw Error(`GPUNetworkIsochrones parameter target must hold 12 elements`);let{breakCount:n}=e,r=e.firstBand??0,i=e.lastBand??Math.max(n-1,0);for(let[e,t]of[[`breakCount`,n],[`firstBand`,r],[`lastBand`,i]])if(!Number.isSafeInteger(t)||t<0)throw Error(`GPUNetworkIsochrones ${e} must be a non-negative integer`);let[a,o,s,c]=e.extent;if(![a,o,s,c].every(Number.isFinite)||s<=a||c<=o)throw Error(`GPUNetworkIsochrones extent must be finite with maxX > minX and maxY > minY`);let l=e.bufferRadius??0,u=e.walkCostPerUnit??0;if(!(l>=0)||!Number.isFinite(l))throw Error(`GPUNetworkIsochrones bufferRadius must be a non-negative finite number`);if(!(u>=0)||!Number.isFinite(u))throw Error(`GPUNetworkIsochrones walkCostPerUnit must be a non-negative finite number`);return t.fill(0),t[0]=n,t[1]=a,t[2]=o,t[3]=s,t[4]=c,t[5]=l,t[6]=u,t[7]=r,t[8]=i,t[9]=Math.min(e.cellCostLimit??1/0,3e38),t}var A=`GPUNetworkIsochrones`;function j(e){return e===`min`?4294967295:0}function M(e,t){return i(e,{id:t.id,operation:A,variant:`band-parameters`,bindings:[{name:`params`,view:t.parameters,type:`f32`,access:`read`},{name:`bandParams`,view:t.bandParameters,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`const WIDTH: f32 = ${t.width}.0;
const HEIGHT: f32 = ${t.height}.0;`,body:`bandParams[bandParamsOffset] = params[paramsOffset];
  bandParams[bandParamsOffset + 1u] = params[paramsOffset + 7u];
  bandParams[bandParamsOffset + 2u] = params[paramsOffset + 8u];
  bandParams[bandParamsOffset + 3u] = 0.0;
  bandParams[bandParamsOffset + 4u] = params[paramsOffset + 1u];
  bandParams[bandParamsOffset + 5u] = params[paramsOffset + 2u];
  bandParams[bandParamsOffset + 6u] = params[paramsOffset + 3u];
  bandParams[bandParamsOffset + 7u] = params[paramsOffset + 4u];
  bandParams[bandParamsOffset + 8u] = (params[paramsOffset + 3u] - params[paramsOffset + 1u]) / WIDTH;
  bandParams[bandParamsOffset + 9u] = (params[paramsOffset + 4u] - params[paramsOffset + 2u]) / HEIGHT;
  bandParams[bandParamsOffset + 10u] = 0.0;
  bandParams[bandParamsOffset + 11u] = 0.0;`})}function N(e,t){return i(e,{id:t.id,operation:A,variant:`inclusive-breaks`,bindings:[{name:`costBreaks`,view:t.costBreaks,type:`u32`,access:`read`},{name:`inclusiveBreaks`,view:t.inclusiveBreaks,type:`u32`,access:`read_write`}],invocationCount:t.costBreaks.length,body:`let bits = costBreaks[costBreaksOffset + index];
  let magnitude = bits & 0x7fffffffu;
  var nextBits = bits;
  if (magnitude == 0u) {
    // Subnormal comparisons are not portable, so zero steps to the smallest normal number.
    nextBits = 0x00800000u;
  } else if (magnitude < 0x7f800000u) {
    // Positive floats grow with their bits, negative floats shrink in magnitude toward zero.
    nextBits = select(bits + 1u, bits - 1u, (bits & 0x80000000u) != 0u);
  }
  inclusiveBreaks[inclusiveBreaksOffset + index] = nextBits;`})}var P=`var reverseWeight = -1.0;
  let reverseCost = costs[costsOffset + targetNode];
  if (targetNode != source && isochronesIsFinite(reverseCost) && reverseCost >= 0.0) {
    let edgeCount = offsets[offsetsOffset + NODE_COUNT];
    let firstReverse = min(offsets[offsetsOffset + targetNode], edgeCount);
    let lastReverse = min(offsets[offsetsOffset + targetNode + 1u], edgeCount);
    for (var reverseSlot = firstReverse; reverseSlot < lastReverse; reverseSlot++) {
      if (neighbors[neighborsOffset + reverseSlot] != source) { continue; }
      let candidateWeight = weights[weightsOffset + reverseSlot];
      if (candidateWeight >= 0.0 && isochronesIsFinite(candidateWeight)) {
        reverseWeight = max(reverseWeight, candidateWeight);
      }
    }
  }`;function F(e,t){let n=t.mode===`min`,a=!!t.assignments;return i(e,{id:t.id,operation:A,variant:`splat-${t.mode}${a?`-labelled`:``}`,bindings:[{name:`offsets`,view:t.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:t.neighbors,type:`u32`,access:`read`},{name:`weights`,view:t.weights,type:`f32`,access:`read`},{name:`costs`,view:t.costs,type:`f32`,access:`read`},{name:`positions`,view:t.nodePositions,type:`f32`,access:`read`},{name:`params`,view:t.parameters,type:`f32`,access:`read`},{name:`encoded`,view:t.encoded,type:`atomic<u32>`,access:`read_write`},...t.assignments?[{name:`assignments`,view:t.assignments,type:`u32`,access:`read`}]:[]],invocationCount:t.neighbors.length,declarations:`const NODE_COUNT: u32 = ${t.nodeCount}u;
const WIDTH: i32 = ${t.width};
const HEIGHT: i32 = ${t.height};
const MAXIMUM_BUFFER_PIXELS: u32 = ${t.maximumBufferPixels}u;
const MAXIMUM_SAMPLES: u32 = ${t.maximumSamplesPerEdge}u;
const COST_CEILING: f32 = ${r(3e38)};

${O}
fn isochronesIsFinite(value: f32) -> bool {
  return isFiniteBits(value) && abs(value) < COST_CEILING;
}

fn isochronesFindSource(edge: u32) -> u32 {
  var low = 0u;
  var high = NODE_COUNT;
  while (low + 1u < high) {
    let middle = (low + high) >> 1u;
    if (offsets[offsetsOffset + middle] <= edge) {
      low = middle;
    } else {
      high = middle;
    }
  }
  return low;
}

fn isochronesGetPosition(node: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * node], positions[positionsOffset + 2u * node + 1u]);
}`,body:`if (index >= offsets[offsetsOffset + NODE_COUNT]) { return; }
  let targetNode = neighbors[neighborsOffset + index];
  if (targetNode >= NODE_COUNT) { return; }
  let weight = weights[weightsOffset + index];
  let source = isochronesFindSource(index);
  let sourceCost = costs[costsOffset + source];
  ${a?`let facility = assignments[assignmentsOffset + source];
  if (facility >= 255u) { return; }`:``}
  if (!(weight >= 0.0) || !isochronesIsFinite(weight) || !isochronesIsFinite(sourceCost) || sourceCost < 0.0) { return; }
  let startPoint = isochronesGetPosition(source);
  let endPoint = isochronesGetPosition(targetNode);
  if (!isochronesIsFinite(startPoint.x) || !isochronesIsFinite(startPoint.y) || !isochronesIsFinite(endPoint.x) || !isochronesIsFinite(endPoint.y)) { return; }
  let minimum = vec2f(params[paramsOffset + 1u], params[paramsOffset + 2u]);
  let cellSize = vec2f(
    (params[paramsOffset + 3u] - minimum.x) / f32(WIDTH),
    (params[paramsOffset + 4u] - minimum.y) / f32(HEIGHT)
  );
  let walk = max(params[paramsOffset + 6u], 0.0);
  let radius = max(max(params[paramsOffset + 5u], 0.0), 0.5 * length(cellSize));
  let reachX = i32(min(u32(ceil(radius / cellSize.x)), MAXIMUM_BUFFER_PIXELS));
  let reachY = i32(min(u32(ceil(radius / cellSize.y)), MAXIMUM_BUFFER_PIXELS));
  // Extent culling (review B-20): an edge, or a sample, farther than the splat reach from the
  // raster extent touches no pixel.
  let maximum = vec2f(params[paramsOffset + 3u], params[paramsOffset + 4u]);
  let reach = vec2f(f32(reachX), f32(reachY)) * abs(cellSize) + vec2f(radius);
  let extentLow = min(minimum, maximum) - reach;
  let extentHigh = max(minimum, maximum) + reach;
  if (any(max(startPoint, endPoint) < extentLow) || any(min(startPoint, endPoint) > extentHigh)) {
    return;
  }
  let spacing = 0.5 * radius;
  let sampleCount = clamp(u32(ceil(length(endPoint - startPoint) / spacing)) + 1u, 2u, MAXIMUM_SAMPLES);
  ${n?``:P}
  for (var sampleIndex = 0u; sampleIndex < sampleCount; sampleIndex++) {
    let t = f32(sampleIndex) / f32(sampleCount - 1u);
    let point = mix(startPoint, endPoint, vec2f(t));
    if (any(point < extentLow) || any(point > extentHigh)) { continue; }
    ${n?`let edgeCost = sourceCost + t * weight;`:`// The network cost of a road point is the cheaper of the two directions that reach it. The
    // reverse parameter is formed like the reverse slot's own t, so both slots of a road agree.
    var edgeCost = sourceCost + t * weight;
    if (reverseWeight >= 0.0) {
      let reverseT = f32(sampleCount - 1u - sampleIndex) / f32(sampleCount - 1u);
      edgeCost = min(edgeCost, reverseCost + reverseT * reverseWeight);
    }`}
    let centerPixel = vec2i(floor((point - minimum) / cellSize));
    for (var dy = -reachY; dy <= reachY; dy++) {
      let row = centerPixel.y + dy;
      if (row < 0 || row >= HEIGHT) { continue; }
      for (var dx = -reachX; dx <= reachX; dx++) {
        let column = centerPixel.x + dx;
        if (column < 0 || column >= WIDTH) { continue; }
        let pixelCenter = minimum + (vec2f(f32(column), f32(row)) + 0.5) * cellSize;
        let pixelDistance = length(pixelCenter - point);
        if (pixelDistance > radius) { continue; }
        let cost = abs(edgeCost + walk * pixelDistance);
        if (!isochronesIsFinite(cost)) { continue; }
        let slot = encodedOffset + u32(row * WIDTH + column);
        // Test-before-RMW: overlapping samples and edges rewrite most pixels with a no-op, and a
        // plain atomic load is far cheaper than a contended read-modify-write.
        ${a?`let encodedValue = (bitcast<u32>(cost) & 0xffffff00u) | facility;
        if (encodedValue < atomicLoad(&encoded[slot])) { atomicMin(&encoded[slot], encodedValue); }`:n?`let encodedValue = bitcast<u32>(cost);
        if (encodedValue < atomicLoad(&encoded[slot])) { atomicMin(&encoded[slot], encodedValue); }`:`let encodedValue = bitcast<u32>(cost) + 1u;
        if (encodedValue > atomicLoad(&encoded[slot])) { atomicMax(&encoded[slot], encodedValue); }`}
      }
    }
  }`})}function I(e,t){let n=t.mode===`min`,a=!!t.pixelFacilities;return i(e,{id:t.id,operation:A,variant:`decode-${t.mode}${a?`-labelled`:``}`,bindings:[{name:`encoded`,view:t.encoded,type:`u32`,access:`read`},{name:`values`,view:t.values,type:`f32`,access:`read_write`},...t.pixelFacilities?[{name:`facilitiesOut`,view:t.pixelFacilities,type:`u32`,access:`read_write`}]:[]],invocationCount:t.pixelCount,declarations:`const UNREACHED_COST: f32 = ${r(t.unreachedCost)};`,body:`let word = encoded[encodedOffset + index];
  ${a?`values[valuesOffset + index] = select(bitcast<f32>(word & 0xffffff00u), UNREACHED_COST, word == 0xffffffffu);
  facilitiesOut[facilitiesOutOffset + index] = select(word & 0xffu, 0xffffffffu, word == 0xffffffffu);`:n?`values[valuesOffset + index] = select(bitcast<f32>(word), UNREACHED_COST, word == 0xffffffffu);`:`values[valuesOffset + index] = select(bitcast<f32>(word - 1u), UNREACHED_COST, word == 0u);`}`})}function ee(e,t){return i(e,{id:t.id,operation:A,variant:`node-mask`,bindings:[{name:`costs`,view:t.costs,type:`f32`,access:`read`},{name:`params`,view:t.parameters,type:`f32`,access:`read`},{name:`mask`,view:t.mask,type:`u32`,access:`read_write`}],invocationCount:t.costs.length,declarations:O,body:`let cost = costs[costsOffset + index];
  let isReached = isFiniteBits(cost) && abs(cost) < ${r(3e38)} && cost <= params[paramsOffset + 9u];
  mask[maskOffset + index] = select(0u, 1u, isReached);`})}function L(e,t){let n=`const NODE_COUNT: u32 = ${t.nodeCount}u;

// Row of the node's cell in the sorted table, or 0xffffffff.
fn isochronesFindCellRow(node: u32) -> u32 {
  let low = nodeCells[nodeCellsOffset + 2u * node];
  let high = nodeCells[nodeCellsOffset + 2u * node + 1u];
  if (low == 0u && high == 0u) {
    return 0xffffffffu;
  }
  var first = 0u;
  var last = tableCount[tableCountOffset];
  while (first < last) {
    let middle = (first + last) >> 1u;
    let rowLow = tableCells[tableCellsOffset + 2u * middle];
    let rowHigh = tableCells[tableCellsOffset + 2u * middle + 1u];
    if (rowHigh < high || (rowHigh == high && rowLow < low)) {
      first = middle + 1u;
    } else {
      last = middle;
    }
  }
  if (first < tableCount[tableCountOffset]
    && tableCells[tableCellsOffset + 2u * first] == low
    && tableCells[tableCellsOffset + 2u * first + 1u] == high) {
    return first;
  }
  return 0xffffffffu;
}`,r=[{name:`nodeCells`,view:t.nodeCells,type:`u32`,access:`read`},{name:`tableCells`,view:t.tableCells,type:`u32`,access:`read`},{name:`tableCount`,view:t.tableCount,type:`u32`,access:`read`},{name:`costs`,view:t.costs,type:`f32`,access:`read`}];return[a(e,{id:`${t.id}-minimum-fill`,operation:A,view:t.cellMinimumCosts,type:`u32`,value:`0xffffffffu`}),a(e,{id:`${t.id}-facility-fill`,operation:A,view:t.cellFacilities,type:`u32`,value:`0xffffffffu`}),i(e,{id:`${t.id}-minimum`,operation:A,variant:`cell-minimum-cost`,bindings:[...r,{name:`cellMinimum`,view:t.cellMinimumCosts,type:`atomic<u32>`,access:`read_write`}],invocationCount:t.nodeCount,declarations:n,body:`let row = isochronesFindCellRow(index);
  let cost = costs[costsOffset + index];
  if (row != 0xffffffffu && cost >= 0.0) {
    atomicMin(&cellMinimum[cellMinimumOffset + row], select(bitcast<u32>(cost), 0u, cost == 0.0));
  }`}),i(e,{id:`${t.id}-facility`,operation:A,variant:`cell-facility`,bindings:[...r,{name:`assignments`,view:t.assignments,type:`u32`,access:`read`},{name:`cellMinimum`,view:t.cellMinimumCosts,type:`u32`,access:`read`},{name:`cellFacility`,view:t.cellFacilities,type:`atomic<u32>`,access:`read_write`}],invocationCount:t.nodeCount,declarations:n,body:`let row = isochronesFindCellRow(index);
  let cost = costs[costsOffset + index];
  if (row != 0xffffffffu && cost >= 0.0 &&
      select(bitcast<u32>(cost), 0u, cost == 0.0) == cellMinimum[cellMinimumOffset + row]) {
    atomicMin(&cellFacility[cellFacilityOffset + row], assignments[assignmentsOffset + index]);
  }`})]}function R(e,t){return i(e,{id:t.id,operation:A,variant:`triangle-facility`,bindings:[{name:`triangles`,view:t.triangles,type:`f32`,access:`read`},{name:`triangleCount`,view:t.triangleCount,type:`u32`,access:`read`},{name:`params`,view:t.parameters,type:`f32`,access:`read`},{name:`pixelFacilities`,view:t.pixelFacilities,type:`u32`,access:`read`},{name:`facilitiesOut`,view:t.triangleFacilities,type:`u32`,access:`read_write`}],invocationCount:t.triangleFacilities.length,declarations:`const WIDTH: i32 = ${t.width};
const HEIGHT: i32 = ${t.height};`,body:`if (index >= triangleCount[triangleCountOffset]) {
    facilitiesOut[facilitiesOutOffset + index] = 0xffffffffu;
    return;
  }
  let base = trianglesOffset + 6u * index;
  let centroid = vec2f(
    (triangles[base] + triangles[base + 2u] + triangles[base + 4u]) / 3.0,
    (triangles[base + 1u] + triangles[base + 3u] + triangles[base + 5u]) / 3.0
  );
  let minimum = vec2f(params[paramsOffset + 1u], params[paramsOffset + 2u]);
  let cellSize = vec2f(
    (params[paramsOffset + 3u] - minimum.x) / f32(WIDTH),
    (params[paramsOffset + 4u] - minimum.y) / f32(HEIGHT)
  );
  // Sample (column, row) sits at minimum + (column + 0.5, row + 0.5) * cellSize.
  let gridPosition = (centroid - minimum) / cellSize - vec2f(0.5);
  let lower = vec2i(floor(gridPosition));
  var best = 0xffffffffu;
  var bestDistance = 1e30;
  for (var corner = 0u; corner < 4u; corner++) {
    let column = lower.x + i32(corner & 1u);
    let row = lower.y + i32(corner >> 1u);
    if (column < 0 || row < 0 || column >= WIDTH || row >= HEIGHT) { continue; }
    let facility = pixelFacilities[pixelFacilitiesOffset + u32(row * WIDTH + column)];
    if (facility == 0xffffffffu) { continue; }
    let offset = gridPosition - vec2f(f32(column), f32(row));
    let distance = dot(offset, offset);
    if (distance < bestDistance || (distance == bestDistance && facility < best)) {
      bestDistance = distance;
      best = facility;
    }
  }
  facilitiesOut[facilitiesOutOffset + index] = best;`})}var z=`network-isochrones`,B=6,V=16,H=64,U=1024,W=1e30,G=Object.freeze({coordinateSpaces:Object.freeze([`planar`,`longitude-latitude`]),measure:`distance`}),K=class{id;props;nodeCount;mode;spatialContext;cellTableCapacity;constructor(e){e=Z(e,X),this.id=e.id??z,this.props=e;let{id:n}=this,{raster:r,cellOutline:i}=e;if(!r&&!i)throw Error(`${n} needs raster or cellOutline`);this.spatialContext=b(n,e.spatialContext??(r?void 0:y),G);let a=this.spatialContext.coordinateSpace;if(r&&a===`longitude-latitude`)throw Error(`${n} raster needs planar positions; project to a planar frame first`);if(i&&a===`planar`&&e.spatialContext)throw Error(`${n} cellOutline needs longitude-latitude positions`);for(let[r,i]of[[`costs`,e.costs],[`costBreaks`,e.costBreaks],[`parameters`,e.parameters],[`startNodes`,e.startNodes],[`assignments`,e.assignments]])if(i instanceof t)throw Error(`${n} ${r} must be a single packed view, not a chunked vector`);let{nodeCount:c,edgeCount:d}=o(n,e.topology,{weights:!0,nodePositions:!0});if(s(e.costs,[`float32`],`${n} costs`),s(e.costBreaks,[`float32`],`${n} costBreaks`),s(e.parameters,[`float32`],`${n} parameters`),this.nodeCount=e.costs.length,this.nodeCount<1)throw Error(`${n} costs must contain at least one node`);if(c!==this.nodeCount)throw Error(`${n} topology must contain one node per costs row`);if(d<1)throw Error(`${n} needs at least one edge`);if(e.costBreaks.length<1)throw Error(`${n} costBreaks must contain at least one row`);if(e.parameters.length<12)throw Error(`${n} parameters must hold 12 float32 rows`);if(e.startNodes)s(e.startNodes,[`uint32`],`${n} startNodes`),typeof e.startCount==`number`&&u(n,`startCount`,e.startCount,0,e.startNodes.length);else for(let t of[`startCosts`,`startCount`,`costLimit`,`output`,`localIterations`])if(e[t]!==void 0)throw Error(`${n} ${t} needs startNodes`);for(let[t,r]of Object.entries(e.output?.status??{}))r&&(s(r,[`uint32`],`${n} output.status.${t}`),l(n,r.length===1,`output.status.${t} must contain exactly one row`));if(e.assignments&&(s(e.assignments,[`uint32`],`${n} assignments`),e.assignments.length!==this.nodeCount))throw Error(`${n} assignments length must equal the node count`);if(e.maximumLabelIterations!==void 0&&!(e.assignments&&e.startNodes))throw Error(`${n} maximumLabelIterations needs startNodes and assignments`);if(this.mode=r?.mode??`min`,this.mode!==`min`&&this.mode!==`max`)throw Error(`${n} raster.mode must be 'min' or 'max'`);if(r){for(let[e,t,i,a]of[[`raster.width`,r.width,2,2147483647],[`raster.height`,r.height,2,2147483647],[`raster.maximumBufferPixels`,r.maximumBufferPixels??B,1,V],[`raster.maximumSamplesPerEdge`,r.maximumSamplesPerEdge??H,2,U]])u(n,e,t,i,a);if(r.sourcePaths){let e=r.sourcePaths;l(n,this.mode===`min`,`raster.sourcePaths requires raster.mode 'min'`);let{nodeCount:t}=o(n,e.topology,{weights:!0,nodePositions:!0});s(e.costs,[`float32`],`${n} raster.sourcePaths.costs`),l(n,e.costs.length===t,`raster.sourcePaths.costs must contain one row per path node`),e.assignments&&(s(e.assignments,[`uint32`],`${n} raster.sourcePaths.assignments`),l(n,e.assignments.length===t,`raster.sourcePaths.assignments must contain one row per path node`)),l(n,!r.output.pixelFacilities||!!e.assignments,`raster.sourcePaths.assignments is required with pixelFacilities`)}let t=r.unreachedCost??W;if(!Number.isFinite(t)||!(t>0)||t>=3e38)throw Error(`${n} raster.unreachedCost must be a positive finite number below 3e38`);let i=r.width*r.height;if(!Number.isSafeInteger(i)||i>2147483647)throw Error(`${n} raster width * height must fit in 31 bits`);if(r.output.pixelFacilities){if(!e.assignments)throw Error(`${n} raster.output.pixelFacilities needs assignments`);if(this.mode!==`min`)throw Error(`${n} raster.output.pixelFacilities needs raster.mode 'min'`);if(e.startNodes&&e.startNodes.length>255)throw Error(`${n} raster.output.pixelFacilities supports at most 255 startNodes`);if(s(r.output.pixelFacilities,[`uint32`],`${n} raster.output.pixelFacilities`),r.output.pixelFacilities.length<i)throw Error(`${n} raster.output.pixelFacilities must hold width * height rows`)}if(r.output.triangleFacilities){if(!r.output.pixelFacilities||!r.output.triangles)throw Error(`${n} raster.output.triangleFacilities needs pixelFacilities and the triangle output`);if(s(r.output.triangleFacilities,[`uint32`],`${n} raster.output.triangleFacilities`),r.output.triangleFacilities.length!==r.output.triangleBands.length)throw Error(`${n} raster.output.triangleFacilities length must equal triangleBands length`)}if(r.rings&&u(n,`raster.rings.edgeCapacity`,r.rings.edgeCapacity),r.output.values&&(s(r.output.values,[`float32`],`${n} raster.output.values`),r.output.values.length<i))throw Error(`${n} raster.output.values must hold width * height rows`)}if(this.cellTableCapacity=i?.table?.cells.length??i?.tableCapacity??this.nodeCount,i){if(i.family!==`h3`&&i.family!==`quadbin`)throw Error(`${n} cellOutline.family must be 'h3' or 'quadbin'`);if(i.byFacility&&!e.assignments)throw Error(`${n} cellOutline.byFacility needs assignments`);if(i.cellFacilities&&!i.byFacility)throw Error(`${n} cellOutline.cellFacilities needs cellOutline.byFacility`);if(i.byFacility&&!i.output.groups)throw Error(`${n} cellOutline.byFacility needs cellOutline.output.groups`);!i.table&&i.tableCapacity!==void 0&&u(n,`cellOutline.tableCapacity`,i.tableCapacity),u(n,`cellOutline table row capacity`,this.cellTableCapacity),l(n,!!i.table||this.cellTableCapacity>=this.nodeCount,`cellOutline.tableCapacity must be at least the node count (${this.nodeCount}) when table is omitted, because transient cell loss cannot be reported`),i.cellFacilities&&(s(i.cellFacilities,[`uint32`],`${n} cellOutline.cellFacilities`),l(n,i.cellFacilities.length===this.cellTableCapacity,`cellOutline.cellFacilities must contain exactly ${this.cellTableCapacity} table rows`))}let f=Q({topology:e.topology,raster:{sourcePaths:r?.sourcePaths},costBreaks:e.costBreaks,parameters:e.parameters,startNodes:e.startNodes,startCosts:e.startCosts,startCount:typeof e.startCount==`object`?e.startCount:void 0,costLimit:e.costLimit,...e.startNodes?{}:{costs:e.costs,assignments:e.assignments}}),p=Q({...e.startNodes?{costs:e.costs,assignments:e.assignments}:{},raster:{output:r?.output,rings:{output:r?.rings?.output}},cellOutline:{table:i?.table,cellFacilities:i?.cellFacilities,output:i?.output,rings:{output:i?.rings?.output}}}),m=Q({output:e.output}),h={};for(let[e,t]of Object.entries(m))$(n,{[e]:t},h,`output.status views must use separate buffers`),h[e]=t;$(n,m,{...f,...p},`output.status views must not share buffers with inputs or geometry outputs`),$(n,p,f,`outputs must not share buffers with inputs`),T(n,{...p,...m},[...r?[{outputs:{"raster.output.values":r.output.values,"raster.output.pixelFacilities":r.output.pixelFacilities}}]:[],...r?.output.triangleFacilities?[{outputs:{"raster.output.triangleFacilities":r.output.triangleFacilities},inputs:{"raster.output.triangles":r.output.triangles,"raster.output.status.count":r.output.status?.count,"raster.output.pixelFacilities":r.output.pixelFacilities,parameters:e.parameters}}]:[],...i?.byFacility?[{outputs:{"cellOutline.cellFacilities":i.cellFacilities},inputs:{costs:e.costs,assignments:e.assignments,"cellOutline.table.cells":i.table?.cells,"cellOutline.table.status.count":i.table?.status.count}}]:[]])}getCommandNodes(t){let{id:n,props:r,nodeCount:i,mode:o}=this,{raster:s,cellOutline:l}=r,{offsets:y,neighbors:b,weights:C,nodePositions:T}=r.topology,D=Q(r),O=Object.values(D);d(n,t,O),E(t,n,D),s?.rings&&(u(n,`raster.rings.edgeCapacity`,s.rings.edgeCapacity,1,f(t.device)),c(n,`raster.rings.output ring capacity`,s.rings.output.ringOffsets.length-1,p(t.device)));let k=s?s.width*s.height:0,A=this.cellTableCapacity;w(n,t,O,{...s?{"raster encoded":k*4,...s.output.values?{}:{"raster values":k*4},"raster bandParameters":48,"raster inclusiveBreaks":r.costBreaks.length*4}:{},...l?{"cellOutline reachedMask":i*4,...l.table?{}:{"cellOutline table.cells":A*8,"cellOutline table.counts":A*4,"cellOutline table.status.count":4,"cellOutline table.status.overflow":4,"cellOutline table.status.candidateOverflow":4},...l.family===`h3`||l.byFacility?{"cellOutline nodeCells":i*8}:{},...l.byFacility?{"cellOutline cellMinimumCosts":A*4,...l.cellFacilities?{}:{"cellOutline cellFacilities":A*4}}:{}}:{}});let P=[];if(r.startNodes&&r.assignments?P.push(...new x({id:`${n}-search`,topology:r.topology,facilities:r.startNodes,facilityCosts:r.startCosts,facilityCount:r.startCount,costLimit:r.costLimit,maximumIterations:r.maximumIterations,maximumLabelIterations:r.maximumLabelIterations,localIterations:r.localIterations,output:{assignments:r.assignments,costs:r.costs,status:r.output?.status}}).getCommandNodes(t)):r.startNodes&&P.push(...new S({id:`${n}-search`,topology:r.topology,startNodes:r.startNodes,startCosts:r.startCosts,startCount:r.startCount,costLimit:r.costLimit,maximumIterations:r.maximumIterations,localIterations:r.localIterations,output:{costs:r.costs,status:r.output?.status}}).getCommandNodes(t)),s){let{width:c,height:l}=s,u=c*l,d={width:c,height:l,affine:[1,0,0,0,1,0],pixelInterpretation:`point`},f=e(t,`${n}-encoded`,`uint32`,u),p=s.output.values??e(t,`${n}-values`,`float32`,u),g=e(t,`${n}-band-parameters`,`float32`,12),_=e(t,`${n}-inclusive-breaks`,`float32`,r.costBreaks.length);P.push(N(t,{id:`${n}-inclusive-breaks`,costBreaks:r.costBreaks,inclusiveBreaks:_}),M(t,{id:`${n}-band-parameters`,width:c,height:l,parameters:r.parameters,bandParameters:g}),a(t,{id:`${n}-encoded-fill`,operation:`GPUNetworkIsochrones`,view:f,type:`u32`,value:`${j(o)}u`}),F(t,{id:`${n}-splat`,nodeCount:i,width:c,height:l,mode:o,maximumBufferPixels:s.maximumBufferPixels??B,maximumSamplesPerEdge:s.maximumSamplesPerEdge??H,offsets:y,neighbors:b,weights:C,costs:r.costs,nodePositions:T,parameters:r.parameters,encoded:f,assignments:s.output.pixelFacilities?r.assignments:void 0}),...s.sourcePaths?[F(t,{id:`${n}-source-paths-splat`,nodeCount:s.sourcePaths.costs.length,width:c,height:l,mode:o,maximumBufferPixels:s.maximumBufferPixels??B,maximumSamplesPerEdge:s.maximumSamplesPerEdge??H,offsets:s.sourcePaths.topology.offsets,neighbors:s.sourcePaths.topology.neighbors,weights:s.sourcePaths.topology.weights,costs:s.sourcePaths.costs,nodePositions:s.sourcePaths.topology.nodePositions,parameters:r.parameters,encoded:f,assignments:s.output.pixelFacilities?s.sourcePaths.assignments:void 0})]:[],I(t,{id:`${n}-decode`,mode:o,unreachedCost:s.unreachedCost??W,encoded:f,values:p,pixelCount:u,pixelFacilities:s.output.pixelFacilities}),...new h({id:`${n}-isobands`,grid:d,values:p,breaks:_,parameters:g,output:{bandClasses:s.output.bandClasses,triangles:s.output.triangles,triangleBands:s.output.triangleBands,edges:s.output.edges,edgeBands:s.output.edgeBands,status:s.output.status}}).getCommandNodes(t)),s.output.triangleFacilities&&P.push(R(t,{id:`${n}-triangle-facilities`,width:c,height:l,triangles:s.output.triangles,triangleCount:s.output.status.count,parameters:r.parameters,pixelFacilities:s.output.pixelFacilities,triangleFacilities:s.output.triangleFacilities})),s.rings&&P.push(...new m({id:`${n}-rings`,grid:d,values:p,breaks:_,parameters:g,edgeCapacity:s.rings.edgeCapacity,vertexTolerance:s.rings.vertexTolerance,output:s.rings.output}).getCommandNodes(t))}if(l){let a=e(t,`${n}-reached-mask`,`uint32`,i);P.push(ee(t,{id:`${n}-reached-mask`,costs:r.costs,parameters:r.parameters,mask:a}));let o=this.cellTableCapacity,s,c=l.table??{cells:e(t,`${n}-reached-cells`,`uint32x2`,o),counts:e(t,`${n}-reached-counts`,`uint32`,o),status:{count:e(t,`${n}-reached-count`,`uint32`,1),overflow:e(t,`${n}-reached-overflow`,`uint32`,1),candidateOverflow:e(t,`${n}-reached-candidate-overflow`,`uint32`,1)}},u=!!l.byFacility;if(l.family===`quadbin`&&!u)P.push(...new g({id:`${n}-reached-table`,family:`quadbin`,resolution:l.resolution,positions:T,mask:a,output:c}).getCommandNodes(t));else{let d=e(t,`${n}-node-cells`,`uint32x2`,i);P.push(...new _({id:`${n}-node-cells`,family:l.family,resolution:l.resolution,positions:T,mask:a,output:{cells:d}}).getCommandNodes(t),...new g({id:`${n}-reached-table`,family:l.family,resolution:l.resolution,cells:d,output:c}).getCommandNodes(t)),u&&(s=l.cellFacilities??e(t,`${n}-cell-facilities`,`uint32`,o),P.push(...L(t,{id:`${n}-cell-facilities`,nodeCount:i,nodeCells:d,tableCells:c.cells,tableCount:c.status.count,costs:r.costs,assignments:r.assignments,cellMinimumCosts:e(t,`${n}-cell-minimum-costs`,`uint32`,o),cellFacilities:s})))}P.push(...new v({id:`${n}-outline`,family:l.family,cells:c.cells,count:c.status.count,groups:s,output:l.output,rings:l.rings}).getCommandNodes(t))}return P}},q={offsets:!0,neighbors:!0,weights:!0,nodePositions:!0},J={count:!0,requiredCount:!0,overflow:!0},Y={ringOffsets:!0,positions:!0,ringAreas:!0,ringIsHole:!0,ringShells:!0,ringGroups:!0,segmentRings:!0,segmentVertices:!0,segmentFlags:!0,polygons:{kind:!0,positions:!0,ringOffsets:!0,polygonOffsets:!0,featureOffsets:!0,sourceIds:!0},status:{...J,openSegmentCount:!0,touchingSegmentCount:!0,edgeCount:!0,edgeOverflow:!0}},X={id:!0,topology:q,spatialContext:!0,costs:!0,startNodes:!0,assignments:!0,maximumLabelIterations:!0,startCosts:!0,startCount:!0,costLimit:!0,maximumIterations:!0,localIterations:!0,output:{status:{converged:!0,iterationLimitReached:!0}},costBreaks:!0,parameters:!0,raster:{width:!0,height:!0,mode:!0,maximumBufferPixels:!0,maximumSamplesPerEdge:!0,unreachedCost:!0,sourcePaths:{topology:q,costs:!0,assignments:!0},output:{bandClasses:!0,triangles:!0,triangleBands:!0,edges:!0,edgeBands:!0,values:!0,pixelFacilities:!0,triangleFacilities:!0,status:{...J,vertexCount:!0,edgeCount:!0,edgeOverflow:!0,edgeRequiredCount:!0}},rings:{edgeCapacity:!0,vertexTolerance:!0,output:Y}},cellOutline:{family:!0,resolution:!0,tableCapacity:!0,byFacility:!0,cellFacilities:!0,table:{cells:!0,counts:!0,sums:!0,sumValues:!0,minimums:!0,maximums:!0,status:{...J,candidateOverflow:!0}},output:{rows:!0,cells:!0,edgeIndices:!0,endpoints:!0,groups:!0,status:J},rings:{vertexTolerance:!0,normalizeWinding:!0,output:Y}}};function Z(e,t){if(!e||typeof e!=`object`)return e;let n=D(e,Object.keys(t));return Object.freeze(Object.fromEntries(Object.entries(n).map(([e,n])=>[e,t[e]===!0?n:Z(n,t[e])])))}function Q(e,r=``){return Object.fromEntries(Object.entries(e).flatMap(([e,i])=>{let a=r?`${r}.${e}`:e;return i instanceof n?[[a,i]]:!i||typeof i!=`object`||i instanceof t?[]:Object.entries(Q(i,a))}))}function $(e,t,n,r){let i=new Map(Object.entries(n).map(([e,t])=>[C(t),e]));for(let[n,a]of Object.entries(t)){let t=i.get(C(a));l(e,t===void 0,`${r}: ${n} and ${t}`)}}export{k as n,K as t};