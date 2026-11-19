import{a as e,m as t,v as n,y as r}from"./spatial-context-DwZEd4EZ.js";import{r as i}from"./kernel-C3tfQOn4.js";import{t as a}from"./gpu-compaction-B6IzcF0P.js";import{n as o,o as s}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as c,f as l,m as u,p as d,y as f}from"./gpu-contributor-utils-BIwLQON5.js";import{a as p,c as m,f as h,i as g,s as _}from"./raster-relaxation-Di9wjtxp.js";import{t as v}from"./geodesic-wgsl-I0CYDKcz.js";import{a as y}from"./graph-chunk-dispatch-zBt8k2-o.js";import{a as b}from"./geometry-storage-validation-T89ZhmCj.js";var x=`GPULineSimplification`,S=2139095040,C=`
const LINE_MAXIMUM_FINITE: f32 = bitcast<f32>(0x7f7fffffu);

// Runtime zero: the kernel stores \`leftAnchor >> 31u\`, a row index loaded from memory (rows are
// below 2^31), which the compiler cannot fold, unlike thread IDs or \`arrayLength\` whose ranges it
// knows. ORing it into a
// product's bits keeps the compiler from contracting \`a * b + c\` or \`a * b > c\` into a fused
// multiply-add (an unrounded product), which Metal otherwise does, changing results by a few ULP.
var<private> lineOpaqueZero: u32;

fn lineProduct(left: f32, right: f32) -> f32 {
  return bitcast<f32>(bitcast<u32>(left * right) | lineOpaqueZero);
}

fn lineNextUp(value: f32) -> f32 {
  return bitcast<f32>(bitcast<u32>(value) + 1u);
}

fn lineNextDown(value: f32) -> f32 {
  return bitcast<f32>(bitcast<u32>(value) - 1u);
}

fn lineDivideFloor(numerator: f32, denominator: f32) -> f32 {
  if (!(numerator > 0.0)) {
    return 0.0;
  }
  var quotient = numerator / denominator;
  if (!(quotient >= 0.0)) {
    quotient = 0.0;
  }
  quotient = min(quotient, LINE_MAXIMUM_FINITE);
  for (var step = 0u; step < 16u; step++) {
    if (quotient > 0.0 && lineProduct(quotient, denominator) > numerator) {
      quotient = lineNextDown(quotient);
    } else {
      break;
    }
  }
  for (var step = 0u; step < 16u; step++) {
    let next = lineNextUp(quotient);
    if (lineProduct(next, denominator) <= numerator) {
      quotient = next;
    } else {
      break;
    }
  }
  return quotient;
}

fn lineSqrtFloor(value: f32) -> f32 {
  if (!(value > 0.0)) {
    return 0.0;
  }
  var root = sqrt(value);
  if (!(root >= 0.0)) {
    root = 0.0;
  }
  root = min(root, LINE_MAXIMUM_FINITE);
  for (var step = 0u; step < 16u; step++) {
    if (root > 0.0 && lineProduct(root, root) > value) {
      root = lineNextDown(root);
    } else {
      break;
    }
  }
  for (var step = 0u; step < 16u; step++) {
    let next = lineNextUp(root);
    if (lineProduct(next, next) <= value) {
      root = next;
    } else {
      break;
    }
  }
  return root;
}

fn linePointDistance(px: f32, py: f32, ax: f32, ay: f32) -> f32 {
  let deltaX = px - ax;
  let deltaY = py - ay;
  return lineSqrtFloor(lineProduct(deltaX, deltaX) + lineProduct(deltaY, deltaY));
}

fn lineSegmentDistance(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
  let segmentX = bx - ax;
  let segmentY = by - ay;
  let offsetX = px - ax;
  let offsetY = py - ay;
  let lengthSquared = lineProduct(segmentX, segmentX) + lineProduct(segmentY, segmentY);
  let projection = lineProduct(offsetX, segmentX) + lineProduct(offsetY, segmentY);
  if (lengthSquared <= 0.0 || projection <= 0.0) {
    return linePointDistance(px, py, ax, ay);
  }
  if (projection >= lengthSquared) {
    return linePointDistance(px, py, bx, by);
  }
  let segmentLength = lineSqrtFloor(lengthSquared);
  if (segmentLength <= 0.0) {
    return linePointDistance(px, py, ax, ay);
  }
  let cross = abs(lineProduct(segmentX, offsetY) - lineProduct(segmentY, offsetX));
  return lineDivideFloor(cross, segmentLength);
}

fn lineTimeRatioDistance(
  px: f32, py: f32, pt: f32, ax: f32, ay: f32, at: f32, bx: f32, by: f32, bt: f32
) -> f32 {
  let duration = bt - at;
  if (!(duration > 0.0)) {
    return linePointDistance(px, py, ax, ay);
  }
  let elapsed = pt - at;
  var ratio = lineDivideFloor(abs(elapsed), duration);
  if (elapsed < 0.0) {
    ratio = -ratio;
  }
  let expectedX = ax + lineProduct(ratio, bx - ax);
  let expectedY = ay + lineProduct(ratio, by - ay);
  return linePointDistance(px, py, expectedX, expectedY);
}
`;function w(e){return`${v}
const LINE_SPHERE_RADIUS: f32 = ${e};

fn lineGeodesicSegmentDistance(point: vec2<f32>, start: vec2<f32>, end: vec2<f32>) -> f32 {
  let fraction = geodesicSegmentFraction(point, start, end);
  return geodesicSegmentDistance(point, start, end, fraction, LINE_SPHERE_RADIUS);
}

// Great-circle distance to the chord position linearly interpolated (longitude difference wrapped)
// at the timestamp of the point.
fn lineGeodesicTimeRatioDistance(
  point: vec2<f32>, pointTime: f32, start: vec2<f32>, startTime: f32, end: vec2<f32>, endTime: f32
) -> f32 {
  let duration = endTime - startTime;
  if (!(duration > 0.0)) {
    return geodesicHaversineDistance(point, start, LINE_SPHERE_RADIUS);
  }
  let ratio = (pointTime - startTime) / duration;
  let expected = start + vec2<f32>(
    geodesicWrapLongitudeDelta(end.x - start.x),
    end.y - start.y
  ) * ratio;
  return geodesicHaversineDistance(point, expected, LINE_SPHERE_RADIUS);
}
`}var T=e=>`let leftAnchor = leftAnchors[leftAnchorsOffset + ${e}];
  let rightAnchor = rightAnchors[rightAnchorsOffset + ${e}];
  if (!(leftAnchor < ${e} && ${e} < rightAnchor)) {
    return;
  }`;function E(e,t){let n=t.lineOffsets.length-1;return s(e,{id:t.id,operation:x,variant:`importance-init`,bindings:[{name:`lineOffsets`,view:t.lineOffsets,type:`u32`,access:`read`},{name:`leftAnchors`,view:t.scratch.leftAnchors,type:`u32`,access:`read_write`},{name:`rightAnchors`,view:t.scratch.rightAnchors,type:`u32`,access:`read_write`},{name:`importance`,view:t.importance,type:`u32`,access:`read_write`}],invocationCount:t.importance.length,declarations:`const LINE_COUNT: u32 = ${n}u;
const ENDPOINT_IMPORTANCE: u32 = ${S}u;`,body:`let firstRow = lineOffsets[lineOffsetsOffset];
  let endRow = lineOffsets[lineOffsetsOffset + LINE_COUNT];
  if (index < firstRow || index >= endRow) {
    leftAnchors[leftAnchorsOffset + index] = index;
    rightAnchors[rightAnchorsOffset + index] = index;
    importance[importanceOffset + index] = 0u;
    return;
  }
  // Last line whose start is <= index (empty lines share a start with the next line).
  var low = 0u;
  var high = LINE_COUNT + 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (lineOffsets[lineOffsetsOffset + middle] <= index) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  let line = low - 1u;
  let lineStart = lineOffsets[lineOffsetsOffset + line];
  let lineLast = lineOffsets[lineOffsetsOffset + line + 1u] - 1u;
  if (index == lineStart || index >= lineLast) {
    leftAnchors[leftAnchorsOffset + index] = index;
    rightAnchors[rightAnchorsOffset + index] = index;
    importance[importanceOffset + index] = ENDPOINT_IMPORTANCE;
    return;
  }
  leftAnchors[leftAnchorsOffset + index] = lineStart;
  rightAnchors[rightAnchorsOffset + index] = lineLast;
  importance[importanceOffset + index] = 0u;`})}function D(e,t){let{scratch:n,gate:r}=t,i=t.finishSpanLimit??0,a=t.importance.length,o=[{name:`leftAnchors`,view:n.leftAnchors,type:`u32`,access:`read`},{name:`rightAnchors`,view:n.rightAnchors,type:`u32`,access:`read`}],c={name:`bestKeys`,view:n.bestKeys,type:`atomic<u32>`,access:`read_write`},l={name:`bestRows`,view:n.bestRows,type:`atomic<u32>`,access:`read_write`},u={operation:x,invocationCount:a,condition:r.condition,extraResources:r.extraResources},d=t.metric===`time-ratio`,f=[{name:`positions`,view:t.positions,type:`f32`,access:`read`},...o,{name:`distanceKeys`,view:n.distanceKeys,type:`u32`,access:`read_write`},c];d&&t.timestamps&&f.push({name:`timestamps`,view:t.timestamps,type:`f32`,access:`read`});let p=(e,t)=>`positions[positionsOffset + 2u * ${e} + ${t}u]`,m=e=>`timestamps[timestampsOffset + ${e}]`,h=e=>`vec2<f32>(${p(e,0)}, ${p(e,1)})`,g=t.sphereRadius===void 0?C:`${C}${w(t.sphereRadius)}`,_=t.sphereRadius===void 0?t.metric:`${t.metric}-sphere`,v=(e,n,r)=>t.sphereRadius===void 0?d?`lineTimeRatioDistance(${p(e,0)}, ${p(e,1)}, ${m(e)},
      ${p(n,0)}, ${p(n,1)}, ${m(n)},
      ${p(r,0)}, ${p(r,1)}, ${m(r)})`:`lineSegmentDistance(${p(e,0)}, ${p(e,1)},
      ${p(n,0)}, ${p(n,1)},
      ${p(r,0)}, ${p(r,1)})`:d?`lineGeodesicTimeRatioDistance(${h(e)}, ${m(e)}, ${h(n)}, ${m(n)}, ${h(r)}, ${m(r)})`:`lineGeodesicSegmentDistance(${h(e)}, ${h(n)}, ${h(r)})`,y=v(`index`,`leftAnchor`,`rightAnchor`);return[i>0?s(e,{...u,id:`${t.id}-reset`,variant:`importance-reset-finish-${_}`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`leftAnchors`,view:n.leftAnchors,type:`atomic<u32>`,access:`read_write`},{name:`rightAnchors`,view:n.rightAnchors,type:`atomic<u32>`,access:`read_write`},c,l,{name:`importance`,view:t.importance,type:`u32`,access:`read_write`},...d&&t.timestamps?[{name:`timestamps`,view:t.timestamps,type:`f32`,access:`read`}]:[]],declarations:`${g}
const FINISH_SPAN_LIMIT: u32 = ${i}u;

// Distance key of a row against the chord (left, right), as in the distance node.
fn getDistanceKey(row: u32, leftAnchor: u32, rightAnchor: u32) -> u32 {
  let distance = ${v(`row`,`leftAnchor`,`rightAnchor`)};
  return select(bitcast<u32>(distance), 0u, distance <= 0.0);
}

// Decides every interior row of the open interval (first, last) with the recursion the rounds
// would run, depth first from one lane. Importance is min(key, parent) where parent is the
// importance of the split that created the interval; both children of a split row inherit that
// row's importance as parent because it never exceeds either anchor's.
fn finishInterval(first: u32, last: u32) {
  lineOpaqueZero = first >> 31u;
  var stackLeft: array<u32, ${i}>;
  var stackRight: array<u32, ${i}>;
  var stackParent: array<u32, ${i}>;
  stackLeft[0] = first;
  stackRight[0] = last;
  stackParent[0] = min(importance[importanceOffset + first], importance[importanceOffset + last]);
  var depth = 1u;
  while (depth > 0u) {
    depth--;
    let left = stackLeft[depth];
    let right = stackRight[depth];
    let parent = stackParent[depth];
    var bestKey = 0u;
    var bestRow = 0xffffffffu;
    for (var row = left + 1u; row < right; row++) {
      let key = getDistanceKey(row, left, right);
      // Strict comparison keeps the smallest row on ties.
      if (key > bestKey) {
        bestKey = key;
        bestRow = row;
      }
    }
    if (bestKey == 0u) {
      for (var row = left + 1u; row < right; row++) {
        importance[importanceOffset + row] = 0u;
        atomicStore(&leftAnchors[leftAnchorsOffset + row], row);
        atomicStore(&rightAnchors[rightAnchorsOffset + row], row);
      }
      continue;
    }
    let splitImportance = min(bestKey, parent);
    importance[importanceOffset + bestRow] = splitImportance;
    atomicStore(&leftAnchors[leftAnchorsOffset + bestRow], bestRow);
    atomicStore(&rightAnchors[rightAnchorsOffset + bestRow], bestRow);
    if (bestRow > left + 1u) {
      stackLeft[depth] = left;
      stackRight[depth] = bestRow;
      stackParent[depth] = splitImportance;
      depth++;
    }
    if (right > bestRow + 1u) {
      stackLeft[depth] = bestRow;
      stackRight[depth] = right;
      stackParent[depth] = splitImportance;
      depth++;
    }
  }
}`,body:`let leftAnchor = atomicLoad(&leftAnchors[leftAnchorsOffset + index]);
  let rightAnchor = atomicLoad(&rightAnchors[rightAnchorsOffset + index]);
  if (!(leftAnchor < index && index < rightAnchor)) {
    return;
  }
  if (rightAnchor - leftAnchor - 1u <= FINISH_SPAN_LIMIT) {
    // The first interior row finishes the whole interval; its siblings have nothing to reset.
    if (index == leftAnchor + 1u) {
      finishInterval(leftAnchor, rightAnchor);
    }
    return;
  }
  atomicStore(&bestKeys[bestKeysOffset + leftAnchor], 0u);
  atomicStore(&bestRows[bestRowsOffset + leftAnchor], 0xffffffffu);`}):s(e,{...u,id:`${t.id}-reset`,variant:`importance-reset`,bindings:[...o,c,l],body:`${T(`index`)}
  atomicStore(&bestKeys[bestKeysOffset + leftAnchor], 0u);
  atomicStore(&bestRows[bestRowsOffset + leftAnchor], 0xffffffffu);`}),s(e,{...u,id:`${t.id}-distance`,variant:`importance-distance-${_}`,bindings:f,declarations:g,body:`${T(`index`)}
  lineOpaqueZero = leftAnchor >> 31u;
  let distance = ${y};
  // Canonical +0 so equal distances always have equal bits.
  let key = select(bitcast<u32>(distance), 0u, distance <= 0.0);
  distanceKeys[distanceKeysOffset + index] = key;
  atomicMax(&bestKeys[bestKeysOffset + leftAnchor], key);`}),s(e,{...u,id:`${t.id}-argmax`,variant:`importance-argmax`,bindings:[...o,{name:`distanceKeys`,view:n.distanceKeys,type:`u32`,access:`read`},c,l],body:`${T(`index`)}
  if (distanceKeys[distanceKeysOffset + index] == atomicLoad(&bestKeys[bestKeysOffset + leftAnchor])) {
    atomicMin(&bestRows[bestRowsOffset + leftAnchor], index);
  }`}),s(e,{...u,id:`${t.id}-split`,variant:`importance-split`,bindings:[{name:`leftAnchors`,view:n.leftAnchors,type:`u32`,access:`read_write`},{name:`rightAnchors`,view:n.rightAnchors,type:`u32`,access:`read_write`},{name:`distanceKeys`,view:n.distanceKeys,type:`u32`,access:`read`},c,l,{name:`importance`,view:t.importance,type:`u32`,access:`read_write`},{name:`status`,view:t.status,type:`atomic<u32>`,access:`read_write`}],body:`${T(`index`)}
  if (atomicLoad(&bestKeys[bestKeysOffset + leftAnchor]) == 0u) {
    importance[importanceOffset + index] = 0u;
    leftAnchors[leftAnchorsOffset + index] = index;
    rightAnchors[rightAnchorsOffset + index] = index;
    return;
  }
  let splitRow = atomicLoad(&bestRows[bestRowsOffset + leftAnchor]);
  if (index == splitRow) {
    // Non-negative f32 bits order like their values, so u32 min is f32 min.
    let parentImportance = min(
      importance[importanceOffset + leftAnchor],
      importance[importanceOffset + rightAnchor]
    );
    importance[importanceOffset + index] = min(distanceKeys[distanceKeysOffset + index], parentImportance);
    leftAnchors[leftAnchorsOffset + index] = index;
    rightAnchors[rightAnchorsOffset + index] = index;
    return;
  }
  if (index < splitRow) {
    rightAnchors[rightAnchorsOffset + index] = splitRow;
  } else {
    leftAnchors[leftAnchorsOffset + index] = splitRow;
  }
  atomicStore(&status[statusOffset], 1u);`})]}function O(e,t){return s(e,{id:t.id,operation:x,variant:`importance-unresolved`,bindings:[{name:`leftAnchors`,view:t.scratch.leftAnchors,type:`u32`,access:`read`},{name:`rightAnchors`,view:t.scratch.rightAnchors,type:`u32`,access:`read`},{name:`importance`,view:t.importance,type:`u32`,access:`read_write`}],invocationCount:t.importance.length,body:`${T(`index`)}
  importance[importanceOffset + index] = min(
    importance[importanceOffset + leftAnchor],
    importance[importanceOffset + rightAnchor]
  );`})}function k(e,t){let n=t.lineOffsets.length-1;return s(e,{id:t.id,operation:x,variant:`keep-mask`,bindings:[{name:`lineOffsets`,view:t.lineOffsets,type:`u32`,access:`read`},{name:`importance`,view:t.importance,type:`f32`,access:`read`},{name:`parameters`,view:t.parameters,type:`f32`,access:`read`},{name:`outputMask`,view:t.outputMask,type:`u32`,access:`read_write`},{name:`rowIds`,view:t.rowIds,type:`u32`,access:`read_write`}],invocationCount:t.importance.length,declarations:`const LINE_COUNT: u32 = ${n}u;`,body:`rowIds[rowIdsOffset + index] = index;
  let tolerance = parameters[parametersOffset];
  let inLine = index >= lineOffsets[lineOffsetsOffset] &&
    index < lineOffsets[lineOffsetsOffset + LINE_COUNT];
  let keep = inLine && importance[importanceOffset + index] > tolerance;
  outputMask[outputMaskOffset + index] = select(0u, 1u, keep);`})}function A(e,t){let n=t.lineOffsets.length-1,r=[{name:`lineOffsets`,view:t.lineOffsets,type:`u32`,access:`read`},{name:`compactRows`,view:t.compactRows,type:`u32`,access:`read`},{name:`requiredCount`,view:t.requiredCount,type:`u32`,access:`read`}];return t.lineCounts&&r.push({name:`lineCounts`,view:t.lineCounts,type:`u32`,access:`read_write`}),t.lineStarts&&r.push({name:`lineStarts`,view:t.lineStarts,type:`u32`,access:`read_write`}),s(e,{id:t.id,operation:x,variant:`kept-ranges`,bindings:r,invocationCount:n,declarations:`const COMPACT_CAPACITY: u32 = ${t.compactRows.length}u;

// In-kernel lower bound on data-dependent keys; GPUGallopingSearch is a batch node and would
// add a pass and a buffer.
fn lowerBoundKeptRow(row: u32, count: u32) -> u32 {
  var low = 0u;
  var high = count;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (compactRows[compactRowsOffset + middle] < row) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}`,body:`let keptCount = min(requiredCount[requiredCountOffset], COMPACT_CAPACITY);
  let lineStart = lineOffsets[lineOffsetsOffset + index];
  let lineEnd = max(lineOffsets[lineOffsetsOffset + index + 1u], lineStart);
  let first = lowerBoundKeptRow(lineStart, keptCount);
  let last = lowerBoundKeptRow(lineEnd, keptCount);
  ${t.lineCounts?`lineCounts[lineCountsOffset + index] = last - first;`:``}
  ${t.lineStarts?`lineStarts[lineStartsOffset + index] = first;`:``}`})}function j(e,t=new Float32Array(4)){if(t.length<4)throw Error(`Line simplification target must hold 4 elements`);let{tolerance:n}=e;if(!Number.isFinite(n)||n<0)throw Error(`Line simplification tolerance must be finite and non-negative`);return(e.method??`douglas-peucker`)===`douglas-peucker`&&r(`GPULineSimplification`,e.spatialContext,n),t.set([n,0,0,0]),t}var M=`GPULineSimplification`,N=`let previousRow = previousRows[previousRowsOffset + index];
  let nextRow = nextRows[nextRowsOffset + index];
  if (!(previousRow < index && index < nextRow)) {
    return;
  }`;function P(e,t){let n=t.lineOffsets.length-1,{scratch:r}=t;return s(e,{id:t.id,operation:M,variant:`visvalingam-init`,bindings:[{name:`lineOffsets`,view:t.lineOffsets,type:`u32`,access:`read`},{name:`previousRows`,view:r.previousRows,type:`u32`,access:`read_write`},{name:`nextRows`,view:r.nextRows,type:`u32`,access:`read_write`},{name:`areaKeys`,view:r.areaKeys,type:`u32`,access:`read_write`},{name:`floorKeys`,view:r.floorKeys,type:`u32`,access:`read_write`},{name:`pendingFlags`,view:r.pendingFlags,type:`u32`,access:`read_write`},{name:`importance`,view:t.importance,type:`u32`,access:`read_write`}],invocationCount:t.importance.length,declarations:`const LINE_COUNT: u32 = ${n}u;
const ENDPOINT_IMPORTANCE: u32 = ${S}u;`,body:`areaKeys[areaKeysOffset + index] = 0u;
  floorKeys[floorKeysOffset + index] = 0u;
  pendingFlags[pendingFlagsOffset + index] = 0u;
  previousRows[previousRowsOffset + index] = index;
  nextRows[nextRowsOffset + index] = index;
  importance[importanceOffset + index] = 0u;
  let firstRow = lineOffsets[lineOffsetsOffset];
  let endRow = lineOffsets[lineOffsetsOffset + LINE_COUNT];
  if (index < firstRow || index >= endRow) {
    return;
  }
  // Last line whose start is <= index (empty lines share a start with the next line).
  var low = 0u;
  var high = LINE_COUNT + 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (lineOffsets[lineOffsetsOffset + middle] <= index) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  let line = low - 1u;
  let lineStart = lineOffsets[lineOffsetsOffset + line];
  let lineLast = lineOffsets[lineOffsetsOffset + line + 1u] - 1u;
  if (index == lineStart || index >= lineLast) {
    importance[importanceOffset + index] = ENDPOINT_IMPORTANCE;
    return;
  }
  previousRows[previousRowsOffset + index] = index - 1u;
  nextRows[nextRowsOffset + index] = index + 1u;`})}function F(e,t){let{scratch:n,gate:r,neighborhoodRadius:i}=t,a={operation:M,invocationCount:t.importance.length,condition:r.condition,extraResources:r.extraResources},o=e=>[{name:`previousRows`,view:n.previousRows,type:`u32`,access:e},{name:`nextRows`,view:n.nextRows,type:`u32`,access:e}],c={name:`floorKeys`,view:n.floorKeys,type:`atomic<u32>`,access:`read_write`},l=(e,t)=>`positions[positionsOffset + 2u * ${e} + ${t}u]`;return[s(e,{...a,id:`${t.id}-area`,variant:`visvalingam-area`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},...o(`read`),{name:`areaKeys`,view:n.areaKeys,type:`u32`,access:`read_write`}],declarations:C,body:`${N}
  lineOpaqueZero = previousRow >> 31u;
  let originX = ${l(`previousRow`,0)};
  let originY = ${l(`previousRow`,1)};
  let toMiddleX = ${l(`index`,0)} - originX;
  let toMiddleY = ${l(`index`,1)} - originY;
  let toNextX = ${l(`nextRow`,0)} - originX;
  let toNextY = ${l(`nextRow`,1)} - originY;
  let cross = lineProduct(toMiddleX, toNextY) - lineProduct(toMiddleY, toNextX);
  // abs() makes the key a canonical non-negative f32 whose bits order like its value.
  areaKeys[areaKeysOffset + index] = bitcast<u32>(abs(cross) * 0.5);`}),s(e,{...a,id:`${t.id}-mark`,variant:`visvalingam-mark-${i}`,bindings:[...o(`read`),{name:`areaKeys`,view:n.areaKeys,type:`u32`,access:`read`},c,{name:`pendingFlags`,view:n.pendingFlags,type:`u32`,access:`read_write`},{name:`importance`,view:t.importance,type:`u32`,access:`read_write`},{name:`status`,view:t.status,type:`atomic<u32>`,access:`read_write`}],declarations:`const NEIGHBORHOOD_RADIUS: u32 = ${i}u;

// True when a surviving row within the neighborhood orders before (key, row).
fn hasSmallerNeighbor(row: u32, key: u32, forward: bool) -> bool {
  var cursor = row;
  for (var step = 0u; step < NEIGHBORHOOD_RADIUS; step++) {
    cursor = select(previousRows[previousRowsOffset + cursor], nextRows[nextRowsOffset + cursor], forward);
    // Endpoints and rows outside a line are not undecided interior rows: stop the walk.
    if (!(previousRows[previousRowsOffset + cursor] < cursor &&
        cursor < nextRows[nextRowsOffset + cursor])) {
      return false;
    }
    let neighborKey = areaKeys[areaKeysOffset + cursor];
    if (neighborKey < key || (neighborKey == key && cursor < row)) {
      return true;
    }
  }
  return false;
}`,body:`${N}
  let key = areaKeys[areaKeysOffset + index];
  if (hasSmallerNeighbor(index, key, false) || hasSmallerNeighbor(index, key, true)) {
    atomicStore(&status[statusOffset], 1u);
    return;
  }
  importance[importanceOffset + index] = max(key, atomicLoad(&floorKeys[floorKeysOffset + index]));
  pendingFlags[pendingFlagsOffset + index] = 1u;`}),s(e,{...a,id:`${t.id}-unlink`,variant:`visvalingam-unlink`,bindings:[...o(`read_write`),c,{name:`pendingFlags`,view:n.pendingFlags,type:`u32`,access:`read_write`},{name:`importance`,view:t.importance,type:`u32`,access:`read`}],body:`if (pendingFlags[pendingFlagsOffset + index] == 0u) {
    return;
  }
  let previousRow = previousRows[previousRowsOffset + index];
  let nextRow = nextRows[nextRowsOffset + index];
  let effectiveArea = importance[importanceOffset + index];
  nextRows[nextRowsOffset + previousRow] = nextRow;
  previousRows[previousRowsOffset + nextRow] = previousRow;
  atomicMax(&floorKeys[floorKeysOffset + previousRow], effectiveArea);
  atomicMax(&floorKeys[floorKeysOffset + nextRow], effectiveArea);
  previousRows[previousRowsOffset + index] = index;
  nextRows[nextRowsOffset + index] = index;
  pendingFlags[pendingFlagsOffset + index] = 0u;`})]}function I(e,t){return s(e,{id:t.id,operation:M,variant:`visvalingam-unresolved`,bindings:[{name:`previousRows`,view:t.scratch.previousRows,type:`u32`,access:`read`},{name:`nextRows`,view:t.scratch.nextRows,type:`u32`,access:`read`},{name:`importance`,view:t.importance,type:`u32`,access:`read_write`}],invocationCount:t.importance.length,declarations:`const ENDPOINT_IMPORTANCE: u32 = ${S}u;`,body:`${N}
  importance[importanceOffset + index] = ENDPOINT_IMPORTANCE;`})}var L=`GPULineSimplification`,R=Object.freeze({"douglas-peucker":Object.freeze({...e,measure:`distance`}),visvalingam:Object.freeze({...e,measure:`area`})});function z(e){let{output:t}=e;return{selection:t?.ids?t:void 0,status:t?.status}}var B=1024,V=4,H=3,U=class{id;props;spatialContext;sphereRadius;method;neighborhoodRadius;metric;maximumIterations;finishSpanLimit;computeImportance;estimatedCommandNodeCount;constructor(e){this.id=e.id??`line-simplification`,this.method=e.method??`douglas-peucker`,this.spatialContext=n(this.id,e.spatialContext,this.method===`visvalingam`?R.visvalingam:R[`douglas-peucker`]),this.sphereRadius=t(this.spatialContext),this.props=e,this.metric=e.metric??`segment`,this.neighborhoodRadius=e.neighborhoodRadius??3,this.maximumIterations=e.maximumIterations??(this.method===`visvalingam`?256:64),this.computeImportance=e.computeImportance??!0,this.finishSpanLimit=e.finishSpanLimit??32,l(this.id,`finishSpanLimit`,this.finishSpanLimit,0,128);let{id:r,metric:i,maximumIterations:a,method:o,neighborhoodRadius:s}=this,{selection:p,status:m}=z(e);if(o!==`douglas-peucker`&&o!==`visvalingam`)throw Error(`${r} method must be 'douglas-peucker' or 'visvalingam'`);if(o===`visvalingam`){if(this.sphereRadius!==void 0)throw Error(`${r} method 'visvalingam' is planar only; project positions to a planar frame first`);if(i!==`segment`)throw Error(`${r} method 'visvalingam' does not support metric '${i}'`);l(r,`neighborhoodRadius`,s,1,8)}if(i!==`segment`&&i!==`time-ratio`)throw Error(`${r} metric must be 'segment' or 'time-ratio'`);l(r,`maximumIterations`,a,1,B),c(e.positions,[`float32x2`],`${r} positions`);let h=e.positions.length;if(h<1)throw Error(`${r} needs at least one position`);if(c(e.lineOffsets,[`uint32`],`${r} lineOffsets`),e.lineOffsets.length<2)throw Error(`${r} lineOffsets needs two rows`);let g=e.lineOffsets.length-1;if(c(e.importance,[`float32`],`${r} importance`),e.importance.length!==h)throw Error(`${r} importance length must equal positions length`);if(i===`time-ratio`&&this.computeImportance){if(!e.timestamps)throw Error(`${r} metric 'time-ratio' requires timestamps`);if(c(e.timestamps,[`float32`],`${r} timestamps`),e.timestamps.length!==h)throw Error(`${r} timestamps length must equal positions length`)}if((m?.converged||m?.iterationCount)&&!this.computeImportance)throw Error(`${r} output.status.converged and iterationCount require computeImportance`);for(let[e,t]of[[`converged`,m?.converged],[`iterationCount`,m?.iterationCount]])if(t&&(c(t,[`uint32`],`${r} output.status.${e}`),t.length<1))throw Error(`${r} output.status.${e} needs one row`);if(p){if(!e.parameters)throw Error(`${r} parameters are required to select rows`);if(c(e.parameters,[`float32`],`${r} parameters`),e.parameters.length<4)throw Error(`${r} parameters must hold 4 float32 values`);if(u(r,p),p.outputMask&&(c(p.outputMask,[`uint32`],`${r} output.outputMask`),p.outputMask.length!==h))throw Error(`${r} output.outputMask length must equal positions length`);for(let[e,t]of[[`lineCounts`,p.lineCounts],[`lineStarts`,p.lineStarts]])if(t&&(c(t,[`uint32`],`${r} output.${e}`),t.length!==g))throw Error(`${r} output.${e} length must equal the line count`)}if(!this.computeImportance&&!p)throw Error(`${r} requires computeImportance or selected output`);let _=(o===`visvalingam`?H:V)+1;this.estimatedCommandNodeCount=(this.computeImportance?3+a*_+(m?.converged||m?.iterationCount?1:0):0)+(p?2+(p.lineCounts||p.lineStarts?1:0):0),d(r,this.estimatedCommandNodeCount);let v=[e.positions,e.lineOffsets,e.timestamps,e.parameters];this.computeImportance||v.push(e.importance),f(r,[this.computeImportance?e.importance:void 0,m?.converged,m?.iterationCount,p?.ids,p?.status.count,p?.status.overflow,p?.status.requiredCount,p?.outputMask,p?.lineCounts,p?.lineStarts],v)}getCommandNodes(e){let{props:t,id:n}=this,{selection:r,status:i}=z(t);y(e,n,Object.fromEntries([t.positions,t.lineOffsets,this.computeImportance&&this.metric===`time-ratio`?t.timestamps:void 0,t.importance,r?t.parameters:void 0,i?.converged,i?.iterationCount,r?.ids,r?.status.count,r?.status.overflow,r?.status.requiredCount,r?.outputMask,r?.lineCounts,r?.lineStarts].map((e,t)=>[String(t),e]))),b(e,n,{rowColumnsAndScan:t.positions.length*4,iterationState:16,iterationDispatch:12});let a=[];return this.computeImportance&&a.push(...this._getImportanceNodes(e)),r&&t.parameters&&a.push(...this._getSelectionNodes(e,r,t.parameters)),a}_getVisvalingamNodes(e){let{props:t,id:n,maximumIterations:r}=this,a=t.positions.length,o={previousRows:i(e,`${n}-previous-rows`,`uint32`,a),nextRows:i(e,`${n}-next-rows`,`uint32`,a),areaKeys:i(e,`${n}-area-keys`,`uint32`,a),floorKeys:i(e,`${n}-floor-keys`,`uint32`,a),pendingFlags:i(e,`${n}-pending-flags`,`uint32`,a)},s=m(e,`${n}-rounds`,L,a,n),c=[_(e,{id:`${n}-rounds-reset`,operation:L,state:s}),P(e,{id:`${n}-init`,lineOffsets:t.lineOffsets,scratch:o,importance:t.importance})];for(let i=0;i<r;i++){let a=`${n}-round-${i}`;c.push(...F(e,{id:a,neighborhoodRadius:this.neighborhoodRadius,positions:t.positions,scratch:o,importance:t.importance,status:s.status,gate:h(s,a)}),p(e,{id:`${a}-gate`,operation:L,state:s,maxIterations:r}))}c.push(I(e,{id:`${n}-unresolved`,scratch:o,importance:t.importance}));let l=t.output?.status;return(l?.converged||l?.iterationCount)&&c.push(g(e,{id:`${n}-status`,operation:L,state:s,converged:l.converged,iterationCount:l.iterationCount})),c}_getImportanceNodes(e){if(this.method===`visvalingam`)return this._getVisvalingamNodes(e);let{props:t,id:n,maximumIterations:r}=this,a=t.positions.length,o={leftAnchors:i(e,`${n}-left-anchors`,`uint32`,a),rightAnchors:i(e,`${n}-right-anchors`,`uint32`,a),distanceKeys:i(e,`${n}-distance-keys`,`uint32`,a),bestKeys:i(e,`${n}-best-keys`,`uint32`,a),bestRows:i(e,`${n}-best-rows`,`uint32`,a)},s=m(e,`${n}-rounds`,L,a,n),c=[_(e,{id:`${n}-rounds-reset`,operation:L,state:s}),E(e,{id:`${n}-init`,lineOffsets:t.lineOffsets,scratch:o,importance:t.importance})];for(let i=0;i<r;i++){let a=`${n}-round-${i}`;c.push(...D(e,{id:a,metric:this.metric,positions:t.positions,timestamps:t.timestamps,scratch:o,importance:t.importance,status:s.status,gate:h(s,a),finishSpanLimit:this.finishSpanLimit,sphereRadius:this.sphereRadius}),p(e,{id:`${a}-gate`,operation:L,state:s,maxIterations:r}))}c.push(O(e,{id:`${n}-unresolved`,scratch:o,importance:t.importance}));let l=t.output?.status;return(l?.converged||l?.iterationCount)&&c.push(g(e,{id:`${n}-status`,operation:L,state:s,converged:l.converged,iterationCount:l.iterationCount})),c}_getSelectionNodes(e,t,n){let{props:r,id:s}=this,c=r.positions.length,l=t.outputMask??i(e,`${s}-keep-mask`,`uint32`,c),u=i(e,`${s}-row-ids`,`uint32`,c),d=i(e,`${s}-compact-rows`,`uint32`,c),f=i(e,`${s}-kept-total`,`uint32`,1),p=[k(e,{id:`${s}-keep-mask`,lineOffsets:r.lineOffsets,importance:r.importance,parameters:n,outputMask:l,rowIds:u}),...new a({id:`${s}-compaction`,input:u,flags:l,output:d,count:f}).getCommandNodes(e),o(e,{id:`${s}-publish`,operation:L,requiredCount:f,compactIds:d,output:t})];return(t.lineCounts||t.lineStarts)&&p.push(A(e,{id:`${s}-line-ranges`,lineOffsets:r.lineOffsets,compactRows:d,requiredCount:f,lineCounts:t.lineCounts,lineStarts:t.lineStarts})),p}};export{j as n,U as t};