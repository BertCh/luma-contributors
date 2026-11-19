import{v as e}from"./spatial-context-DwZEd4EZ.js";import{r as t}from"./kernel-C3tfQOn4.js";import{E as n,O as r,T as i,f as a,g as o,v as s,y as c}from"./gpu-contributor-utils-BIwLQON5.js";import{t as l}from"./gpu-geometry-measures-J0jVt402.js";import{o as u}from"./wgsl-kernel-nodes-DagJklH2.js";import{d,s as f,u as p}from"./wgsl-common-DfyXiYB0.js";import{t as m}from"./exact-orientation-wgsl-B_jI-HEc.js";import{t as h}from"./addition-rounding-error-wgsl-DVwPAqKC.js";import{t as g}from"./scaled-quotient-wgsl-UDyhQag6.js";import{n as _,t as v}from"./gpu-group-convex-hull-CpY8Xff2.js";import{a as y,i as b}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as ee}from"./gpu-scan-B4NNTfF9.js";import{n as x}from"./sorted-segment-sums-Dvmm6OdR.js";import{A as S,d as C,o as w,s as T}from"./pointer-overlay-Qh8qflcA.js";import{t as E}from"./define-run-B-brkCe4.js";import{s as D,v as te,y as O}from"./polygon-set-CHja50mr.js";import{s as k}from"./shape-segments-DMcyF4V-.js";import{Za as A}from"./entry-preview-D70TdSbK.js";var j=`
// A normalized f32 fraction has magnitude below one; its largest finite exponent is 128.
const SHAPE_MOMENT_MAXIMUM_EXPONENT: i32 = 128;
fn restoreShapeMoment(value: f32, power: i32) -> f32 {
  if (value == 0.0) { return value; }
  let parts = frexp(value);
  let exponent = parts.exp + power;
  if (exponent > SHAPE_MOMENT_MAXIMUM_EXPONENT) {
    return bitcast<f32>(0x7f800000u | (bitcast<u32>(value) & 0x80000000u));
  }
  return ldexp(parts.fract, exponent);
}
fn getShapeMomentTerms(previous: vec2<f32>, current: vec2<f32>) -> vec3<f32> {
  let magnitude = max(max(abs(previous.x), abs(previous.y)), max(abs(current.x), abs(current.y)));
  if (magnitude == 0.0) { return vec3<f32>(0.0); }
  let exponent = frexp(magnitude).exp;
  let first = vec2<f32>(ldexp(previous.x, -exponent), ldexp(previous.y, -exponent));
  let second = vec2<f32>(ldexp(current.x, -exponent), ldexp(current.y, -exponent));
  let cross = first.x * second.y - second.x * first.y;
  let xx = (cross / 12.0) * (first.x * first.x + first.x * second.x + second.x * second.x);
  let yy = (cross / 12.0) * (first.y * first.y + first.y * second.y + second.y * second.y);
  let xy = (cross / 24.0) * (first.x * second.y + 2.0 * first.x * first.y + 2.0 * second.x * second.y + second.x * first.y);
  return vec3<f32>(restoreShapeMoment(xx, 4 * exponent),
    restoreShapeMoment(yy, 4 * exponent), restoreShapeMoment(xy, 4 * exponent));
}
`;function M(e){return[{name:`positions`,view:e.positions,type:`f32`,access:`read`},{name:`ringOffsets`,view:e.ringOffsets,type:`u32`,access:`read`},{name:`featureOffsets`,view:e.featureOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:e.polygonOffsets,type:`u32`,access:`read`}]}function N(e){return`
const ROW_COUNT: u32 = ${e}u;
fn getNan(seed: u32) -> f32 {
  // A runtime operand keeps the compiler from folding a constant NaN.
  return bitcast<f32>(0x7fc00000u | (seed & 0u));
}

fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}
`}function P(e){return`${N(e.positions.length)}
const RING_COUNT: u32 = ${e.ringOffsets.length-1}u;
const POLYGON_COUNT: u32 = ${e.polygonOffsets.length-1}u;

// Feature -> polygons -> rings: a feature owns the rings of its polygons.
fn getRingStart(feature: u32) -> u32 {
  let polygon = min(featureOffsets[featureOffsetsOffset + feature], POLYGON_COUNT);
  return min(polygonOffsets[polygonOffsetsOffset + polygon], RING_COUNT);
}

fn getRingEnd(feature: u32) -> u32 {
  let polygon = min(featureOffsets[featureOffsetsOffset + feature + 1u], POLYGON_COUNT);
  return min(polygonOffsets[polygonOffsetsOffset + polygon], RING_COUNT);
}

// Ring role by position: ring 'ring' is a shell when it starts a polygon (binary search of
// polygonOffsets). A ring outside the polygonOffsets range keeps the first-ring rule.
fn isShellRing(ring: u32, isFirstRing: bool) -> bool {
  if (ring < polygonOffsets[polygonOffsetsOffset] || ring >= polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT]) {
    return isFirstRing;
  }
  var low = 0u;
  var high = POLYGON_COUNT;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (polygonOffsets[polygonOffsetsOffset + middle] < ring) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return polygonOffsets[polygonOffsetsOffset + low] == ring;
}

fn getRingRowStart(ring: u32) -> u32 {
  return min(ringOffsets[ringOffsetsOffset + ring], ROW_COUNT);
}

fn getRingRowEnd(ring: u32) -> u32 {
  return min(ringOffsets[ringOffsetsOffset + ring + 1u], ROW_COUNT);
}
`}function F(e,t){return u(e,{id:t.id,operation:t.operation,variant:`moments`,bindings:[...M(t),{name:`centroids`,view:t.centroids,type:`f32`,access:`read`},{name:`moments`,view:t.moments,type:`f32`,access:`read_write`}],invocationCount:t.featureCount,declarations:`${P(t)}
${d}
${j}

${h}
// accumulator = (sum, compensation). The compensation is the rounding error of each addition,
// read off the integer significands: the float Neumaier term ((sum - total) + value) is
// algebraically zero and the test GPU's shader compiler folds it away.
fn addNeumaier(accumulator: ptr<function, vec2<f32>>, value: f32) {
  let sum = (*accumulator).x;
  let total = sum + value;
  *accumulator = vec2<f32>(total, (*accumulator).y + getAdditionRoundingError(sum, value, total));
}`,body:`let ringStart = getRingStart(index);
  let ringEnd = max(getRingEnd(index), ringStart);
  let centroid = vec2<f32>(centroids[centroidsOffset + 2u * index], centroids[centroidsOffset + 2u * index + 1u]);
  var totalXX = vec2<f32>(0.0);
  var totalYY = vec2<f32>(0.0);
  var totalXY = vec2<f32>(0.0);
  var totalTwiceArea = 0.0;
  var firstArea = 0.0;
  if (!isNanBits(centroid.x) && !isNanBits(centroid.y)) {
    for (var ring = ringStart; ring < ringEnd; ring++) {
      let rowStart = getRingRowStart(ring);
      let rowEnd = getRingRowEnd(ring);
      var twiceArea = vec2<f32>(0.0);
      var ringXX = vec2<f32>(0.0);
      var ringYY = vec2<f32>(0.0);
      var ringXY = vec2<f32>(0.0);
      if (rowEnd > rowStart + 2u) {
        var previous = getPosition(rowEnd - 1u) - centroid;
        for (var row = rowStart; row < rowEnd; row++) {
          let current = getPosition(row) - centroid;
          let cross = previous.x * current.y - current.x * previous.y;
          let terms = getShapeMomentTerms(previous, current);
          addNeumaier(&twiceArea, cross);
          addNeumaier(&ringXX, terms.x);
          addNeumaier(&ringYY, terms.y);
          addNeumaier(&ringXY, terms.z);
          previous = current;
        }
      }
      let ringTwiceArea = twiceArea.x + twiceArea.y;
      if (ring == ringStart) {
        firstArea = 0.5 * ringTwiceArea;
      }
      let factor = select(-1.0, 1.0, isShellRing(ring, ring == ringStart)) * sign(ringTwiceArea);
      totalTwiceArea += factor * ringTwiceArea;
      addNeumaier(&totalXX, factor * (ringXX.x + ringXX.y));
      addNeumaier(&totalYY, factor * (ringYY.x + ringYY.y));
      addNeumaier(&totalXY, factor * (ringXY.x + ringXY.y));
    }
  }
  // Shell rings count positive and holes negative whatever their winding, so the total is
  // non-negative; the flip only guards a zero-area feature.
  let orientationSign = select(1.0, -1.0, totalTwiceArea < 0.0);
  moments[momentsOffset + 4u * index] = orientationSign * (totalXX.x + totalXX.y);
  moments[momentsOffset + 4u * index + 1u] = orientationSign * (totalYY.x + totalYY.y);
  moments[momentsOffset + 4u * index + 2u] = orientationSign * (totalXY.x + totalXY.y);
  moments[momentsOffset + 4u * index + 3u] = firstArea;`})}function I(e,t){return u(e,{id:t.id,operation:t.operation,variant:`convexity`,bindings:[...M(t),{name:`areas`,view:t.areas,type:`f32`,access:`read`},{name:`convexities`,view:t.convexities,type:`f32`,access:`read_write`}],invocationCount:t.featureCount,declarations:`${P(t)}
${m}`,body:`let ringStart = getRingStart(index);
  let ringEnd = max(getRingEnd(index), ringStart);
  let rowStart = getRingRowStart(ringStart);
  let rowEnd = max(getRingRowStart(ringEnd), rowStart);
  let area = areas[areasOffset + index];
  var hullArea = 0.0;
  if (rowEnd - rowStart >= 3u && area > 0.0) {
    let origin = getPosition(rowStart);
    // Lowest point, leftmost among ties, lowest row among identical points.
    var startPoint = vec2<f32>(0.0);
    var startRaw = origin;
    for (var row = rowStart; row < rowEnd; row++) {
      let raw = getPosition(row);
      let p = raw - origin;
      if (row == rowStart || p.y < startPoint.y || (p.y == startPoint.y && p.x < startPoint.x)) {
        startPoint = p;
        startRaw = raw;
      }
    }
    var current = startPoint;
    var currentRaw = startRaw;
    var twiceArea = 0.0;
    for (var step = 0u; step < rowEnd - rowStart; step++) {
      var best = current;
      var bestRaw = currentRaw;
      var hasBest = false;
      for (var row = rowStart; row < rowEnd; row++) {
        let qRaw = getPosition(row);
        let q = qRaw - origin;
        if (q.x == current.x && q.y == current.y) {
          continue;
        }
        if (!hasBest) {
          best = q;
          bestRaw = qRaw;
          hasBest = true;
          continue;
        }
        let toBest = best - current;
        let toQ = q - current;
        let turn = orientSign(currentRaw, bestRaw, qRaw);
        // q lies clockwise of best (all others must be left of the hull edge), or farther on a tie.
        if (turn < 0 || (turn == 0 && dot(toQ, toQ) > dot(toBest, toBest))) {
          best = q;
          bestRaw = qRaw;
        }
      }
      if (!hasBest) {
        break;
      }
      twiceArea += current.x * best.y - best.x * current.y;
      current = best;
      currentRaw = bestRaw;
      if (current.x == startPoint.x && current.y == startPoint.y) {
        break;
      }
    }
    hullArea = 0.5 * twiceArea;
  }
  convexities[convexitiesOffset + index] = select(getNan(index), min(area / hullArea, 1.0), hullArea > 0.0);`})}function L(e,t){let n=[{name:`areas`,view:t.areas,type:`f32`,access:`read`},{name:`perimeters`,view:t.perimeters,type:`f32`,access:`read`},{name:`parameters`,view:t.parameters,type:`f32`,access:`read`}];return t.polsbyPopper&&n.push({name:`polsbyPopper`,view:t.polsbyPopper,type:`f32`,access:`read_write`}),t.schwartzberg&&n.push({name:`schwartzberg`,view:t.schwartzberg,type:`f32`,access:`read_write`}),t.sliver&&n.push({name:`sliver`,view:t.sliver,type:`u32`,access:`read_write`}),u(e,{id:t.id,operation:t.operation,variant:`compactness`,bindings:n,invocationCount:t.featureCount,declarations:`${g}
${p}
fn getNan(seed: u32) -> f32 {
  // A runtime operand keeps the compiler from folding a constant NaN.
  return bitcast<f32>(0x7fc00000u | (seed & 0u));
}`,body:`let area = areas[areasOffset + index];
  let perimeter = perimeters[perimetersOffset + index];
  let isValid = area > 0.0 && perimeter > 0.0 && isFiniteValue(area) && isFiniteValue(perimeter);
  // Split exponents before division: parentheses alone cannot prevent fast-math reassociation
  // into area / (perimeter * perimeter), whose denominator can overflow for finite inputs.
  var ratio = getNan(index);
  if (isValid) { ratio = getScaledQuotient(area, perimeter, perimeter, 0) * (4.0 * 3.14159265358979); }
  ${t.polsbyPopper?`polsbyPopper[polsbyPopperOffset + index] = select(getNan(index), ratio, isValid);`:``}
  ${t.schwartzberg?`schwartzberg[schwartzbergOffset + index] = select(getNan(index), sqrt(ratio), isValid);`:``}
  ${t.sliver?`sliver[sliverOffset + index] = select(0u, 1u, perimeter > 0.0 && (area <= 0.0 || ratio < parameters[parametersOffset]));`:``}`})}function R(e,t){let n=[{name:`moments`,view:t.moments,type:`f32`,access:`read`}];return t.elongation&&n.push({name:`elongation`,view:t.elongation,type:`f32`,access:`read_write`}),t.orientation&&n.push({name:`orientation`,view:t.orientation,type:`f32`,access:`read_write`}),t.clockwise&&n.push({name:`clockwise`,view:t.clockwise,type:`u32`,access:`read_write`}),u(e,{id:t.id,operation:t.operation,variant:`orientation`,bindings:n,invocationCount:t.featureCount,declarations:`${f}
fn getNan(seed: u32) -> f32 {
  // A runtime operand keeps the compiler from folding a constant NaN.
  return bitcast<f32>(0x7fc00000u | (seed & 0u));
}`,body:`let momentXX = moments[momentsOffset + 4u * index];
  let momentYY = moments[momentsOffset + 4u * index + 1u];
  let momentXY = moments[momentsOffset + 4u * index + 2u];
  let firstArea = moments[momentsOffset + 4u * index + 3u];
  // Eigenvalue ratios and angle are invariant under positive scaling. Normalize before squaring
  // so finite moments of large polygons cannot overflow, and tiny polygons do not underflow.
  let magnitude = max(max(abs(momentXX), abs(momentYY)), abs(momentXY));
  // Division by magnitude can lower to multiplication by a subnormal reciprocal, flushed to
  // zero near the f32 maximum. ldexp changes the exponent without forming that reciprocal.
  let exponent = frexp(magnitude).exp;
  let xx = ldexp(momentXX, -exponent);
  let yy = ldexp(momentYY, -exponent);
  let xy = ldexp(momentXY, -exponent);
  let mean = 0.5 * (xx + yy);
  let radius = sqrt(0.25 * (xx - yy) * (xx - yy) + xy * xy);
  let major = mean + radius;
  let minor = max(mean - radius, 0.0);
  let isValid = major > 0.0;
  ${t.elongation?`elongation[elongationOffset + index] = select(getNan(index), 1.0 - sqrt(minor / major), isValid);`:``}
  ${t.orientation?`orientation[orientationOffset + index] = select(getNan(index), select(0.0, 0.5 * guardedAtan2(2.0 * xy, xx - yy), (xy != 0.0) || (xx != yy)), isValid);`:``}
  ${t.clockwise?`clockwise[clockwiseOffset + index] = select(0u, 1u, firstArea < 0.0);`:``}`})}var z=2**24-1;function B(e,t){return e<z&&t<z}function V(e,t,n,r){_(e,t,n,r)}function H(e,n){let{id:r,operation:i,featureCount:a}=n,o=n.positions.length;V(e.device,r,o,a);let s=t(e,`${r}-labels`,`uint32`,o),c=t(e,`${r}-hull-indices`,`uint32`,o),l=t(e,`${r}-hull-offsets`,`uint32`,a+1),d=t(e,`${r}-hull-counts`,`uint32`,a),f=t(e,`${r}-hull-overflow`,`uint32`,1);return[u(e,{id:`${r}-labels`,operation:i,variant:`convexity-labels`,bindings:[...M(n),{name:`labels`,view:s,type:`u32`,access:`read_write`}],invocationCount:o,declarations:`${P(n)}
const FEATURE_COUNT: u32 = ${a}u;
fn getFeatureRowStart(feature: u32) -> u32 { return getRingRowStart(getRingStart(feature)); }
fn getFeatureRowEnd(feature: u32) -> u32 { return max(getRingRowStart(getRingEnd(feature)), getFeatureRowStart(feature)); }`,body:`// Last feature whose first row is at or before this row (skips empty features).
  var low = 0u;
  var high = FEATURE_COUNT - 1u;
  while (low < high) {
    let middle = (low + high + 1u) >> 1u;
    if (getFeatureRowStart(middle) <= index) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  let owned = getFeatureRowStart(low) <= index && index < getFeatureRowEnd(low);
  labels[labelsOffset + index] = select(FEATURE_COUNT, low, owned);`}),...new v({id:`${r}-hull`,positions:n.positions,labels:s,groupCount:a,maximumVerticesPerGroup:Math.max(3,o),vertexCapacity:o,output:{vertexIndices:c,offsets:l,counts:d,status:{overflow:f}}}).getCommandNodes(e),u(e,{id:`${r}-hull-area`,operation:i,variant:`convexity-area`,bindings:[{name:`positions`,view:n.positions,type:`f32`,access:`read`},{name:`areas`,view:n.areas,type:`f32`,access:`read`},{name:`hullIndices`,view:c,type:`u32`,access:`read`},{name:`hullOffsets`,view:l,type:`u32`,access:`read`},{name:`hullCounts`,view:d,type:`u32`,access:`read`},{name:`convexities`,view:n.convexities,type:`f32`,access:`read_write`}],invocationCount:a,declarations:N(o),body:`let area = areas[areasOffset + index];
  let hullStart = hullOffsets[hullOffsetsOffset + index];
  let hullCount = hullCounts[hullCountsOffset + index];
  var hullArea = 0.0;
  if (area > 0.0 && hullCount >= 3u) {
    // Shoelace over the counter-clockwise hull, relative to its first vertex for f32 accuracy.
    let origin = getPosition(hullIndices[hullIndicesOffset + hullStart]);
    var previous = getPosition(hullIndices[hullIndicesOffset + hullStart + hullCount - 1u]) - origin;
    var twiceArea = 0.0;
    for (var vertex = 0u; vertex < hullCount; vertex++) {
      let current = getPosition(hullIndices[hullIndicesOffset + hullStart + vertex]) - origin;
      twiceArea += previous.x * current.y - current.x * previous.y;
      previous = current;
    }
    hullArea = 0.5 * twiceArea;
  }
  convexities[convexitiesOffset + index] = select(getNan(index), min(area / hullArea, 1.0), hullArea > 0.0);`})]}var U=4294967295;function W(e,t){return Math.floor(e/256)+t}function G(e,t,n){let r=W(t,n);return r<=U&&Math.max(r*4,n*8,(n+1)*4)<=e.limits.maxStorageBufferBindingSize}function K(e,t,n,r){let i=W(n,r);a(t,`moment tile capacity`,i,0,U);for(let[n,a]of[[`moment tiles`,Math.max(i,1)*4],[`moment ring centroids`,Math.max(r,1)*8],[`moment ring offsets`,(r+1)*4]])s(e,t,n,a,`reduce ring count or use momentMethod: 'serial'`)}function q(e,n){let{id:r,operation:i,featureCount:a}=n,o=n.ringOffsets.length-1,s=W(n.positions.length,o);K(e.device,r,n.positions.length,o);let c=(n,i)=>t(e,`${r}-${n}`,`uint32`,Math.max(i,1)),l=(n,i)=>t(e,`${r}-${n}`,`float32`,Math.max(i,1)),f=c(`feature-rings`,a+1),p=c(`tile-counts`,o+1),m=c(`tile-offsets`,o+1),g=c(`ring-roles`,o),_=t(e,`${r}-ring-centroids`,`float32x2`,Math.max(o,1)),v=[`area`,`xx`,`yy`,`xy`].map(e=>l(`tile-${e}`,s)),y=[`area`,`xx`,`yy`,`xy`].map(e=>l(`ring-${e}`,o)),b=l(`weighted-area`,o),S=[`area`,`xx`,`yy`,`xy`].map(e=>l(`feature-${e}`,a)),C=[u(e,{id:`${r}-feature-rings`,operation:i,variant:`moment-feature-rings`,invocationCount:a+1,bindings:[...M(n),{name:`featureRings`,view:f,type:`u32`,access:`read_write`}],declarations:P(n),body:`featureRings[featureRingsOffset + index] = getRingStart(index);`}),u(e,{id:`${r}-ring-plan`,operation:i,variant:`moment-ring-plan`,invocationCount:o+1,bindings:[...M(n),{name:`centroids`,view:n.centroids,type:`f32`,access:`read`},{name:`ringCentroids`,view:_,type:`f32`,access:`read_write`},{name:`ringRoles`,view:g,type:`u32`,access:`read_write`},{name:`tileCounts`,view:p,type:`u32`,access:`read_write`}],declarations:`${P(n)}\n${d}`,body:`tileCounts[tileCountsOffset + index] = 0u;
if (index == RING_COUNT) { return; }
let rowStart = getRingRowStart(index); let rowEnd = max(getRingRowEnd(index), rowStart);
let rows = select(0u, rowEnd - rowStart, rowEnd - rowStart >= 3u);
tileCounts[tileCountsOffset + index] = rows / 256u + u32(rows % 256u != 0u);
var low = 0u; var high = ${a}u;
while (low < high) { let middle = low + (high - low + 1u) / 2u; if (getRingStart(middle) <= index) { low = middle; } else { high = middle - 1u; } }
var centroid = vec2<f32>(getNan(index)); var role = 0u;
if (low < ${a}u && index >= getRingStart(low)) { centroid = vec2<f32>(centroids[centroidsOffset + 2u * low], centroids[centroidsOffset + 2u * low + 1u]); role = u32(isShellRing(index, index == getRingStart(low))); }
ringCentroids[ringCentroidsOffset + 2u * index] = centroid.x; ringCentroids[ringCentroidsOffset + 2u * index + 1u] = centroid.y; ringRoles[ringRolesOffset + index] = role;`}),...new ee({id:`${r}-tile-scan`,input:p,output:m,mode:`exclusive`}).getCommandNodes(e),u(e,{id:`${r}-tiles`,operation:i,variant:`moment-tiles`,invocationCount:s,bindings:[{name:`positions`,view:n.positions,type:`f32`,access:`read`},{name:`ringOffsets`,view:n.ringOffsets,type:`u32`,access:`read`},{name:`tileOffsets`,view:m,type:`u32`,access:`read`},{name:`ringCentroids`,view:_,type:`f32`,access:`read`},...v.map((e,t)=>({name:`tileComponent${t}`,view:e,type:`f32`,access:`read_write`}))],declarations:`${h}\n${d}\n${j}
fn addMoment(accumulator: vec2<f32>, value: f32) -> vec2<f32> { let sum = accumulator.x; let total = sum + value; return vec2<f32>(total, accumulator.y + getAdditionRoundingError(sum, value, total)); }
fn position(row: u32) -> vec2<f32> { return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]); }`,body:`var area = vec2<f32>(0.0); var xx = vec2<f32>(0.0); var yy = vec2<f32>(0.0); var xy = vec2<f32>(0.0);
let ringCount = arrayLength(&tileOffsets) - tileOffsetsOffset - 1u;
var low = 0u; var high = ringCount;
while (low < high) { let middle = low + (high - low + 1u) / 2u; if (tileOffsets[tileOffsetsOffset + middle] <= index) { low = middle; } else { high = middle - 1u; } }
if (low < ringCount) {
  let rowCount = (arrayLength(&positions) - positionsOffset) / 2u;
  let ringStart = min(ringOffsets[ringOffsetsOffset + low], rowCount); let ringEnd = max(min(ringOffsets[ringOffsetsOffset + low + 1u], rowCount), ringStart);
  let start = ringStart + (index - tileOffsets[tileOffsetsOffset + low]) * 256u;
  let end = min(start + 256u, ringEnd);
  let centroid = vec2<f32>(ringCentroids[ringCentroidsOffset + 2u * low], ringCentroids[ringCentroidsOffset + 2u * low + 1u]);
  if (start < end && !isNanBits(centroid.x) && !isNanBits(centroid.y)) {
    var previous = position(select(start - 1u, ringEnd - 1u, start == ringStart)) - centroid;
    for (var row = start; row < end; row++) {
      let current = position(row) - centroid; let cross = previous.x * current.y - current.x * previous.y;
      let terms = getShapeMomentTerms(previous, current);
      area = addMoment(area, cross);
      xx = addMoment(xx, terms.x);
      yy = addMoment(yy, terms.y);
      xy = addMoment(xy, terms.z);
      previous = current;
    }
  }
}
tileComponent0[tileComponent0Offset + index] = area.x + area.y; tileComponent1[tileComponent1Offset + index] = xx.x + xx.y; tileComponent2[tileComponent2Offset + index] = yy.x + yy.y; tileComponent3[tileComponent3Offset + index] = xy.x + xy.y;`})];for(let t=0;t<4;t++)C.push(x(e,{id:`${r}-ring-reduce-${t}`,operation:i,segmentCount:o,input:v[t],segmentOffsets:m,output:y[t]}));C.push(u(e,{id:`${r}-ring-roles`,operation:i,variant:`moment-ring-roles`,invocationCount:o,bindings:[{name:`areas`,view:y[0],type:`f32`,access:`read`},{name:`roles`,view:g,type:`u32`,access:`read`},{name:`weightedArea`,view:b,type:`f32`,access:`read_write`},...y.slice(1).map((e,t)=>({name:`component${t}`,view:e,type:`f32`,access:`read_write`}))],body:`let area = areas[areasOffset + index]; let factor = select(-1.0, 1.0, roles[rolesOffset + index] != 0u) * sign(area);
weightedArea[weightedAreaOffset + index] = factor * area;
component0[component0Offset + index] *= factor; component1[component1Offset + index] *= factor; component2[component2Offset + index] *= factor;`}));for(let t=0;t<4;t++)C.push(x(e,{id:`${r}-feature-reduce-${t}`,operation:i,segmentCount:a,input:t===0?b:y[t],segmentOffsets:f,output:S[t]}));return C.push(u(e,{id:`${r}-publish`,operation:i,variant:`moment-publish`,invocationCount:a,bindings:[...S.map((e,t)=>({name:`featureComponent${t}`,view:e,type:`f32`,access:`read`})),{name:`featureRings`,view:f,type:`u32`,access:`read`},{name:`ringAreas`,view:y[0],type:`f32`,access:`read`},{name:`moments`,view:n.moments,type:`f32`,access:`read_write`}],body:`let factor = select(1.0, -1.0, featureComponent0[featureComponent0Offset + index] < 0.0); let base = momentsOffset + 4u * index;
moments[base] = factor * featureComponent1[featureComponent1Offset + index]; moments[base + 1u] = factor * featureComponent2[featureComponent2Offset + index]; moments[base + 2u] = factor * featureComponent3[featureComponent3Offset + index];
let first = featureRings[featureRingsOffset + index]; var firstArea = 0.0; if (first < featureRings[featureRingsOffset + index + 1u]) { firstArea = 0.5 * ringAreas[ringAreasOffset + first]; } moments[base + 3u] = firstArea;`})),C}function J(e,t){let n=Object.entries(t).filter(e=>!!e[1]);for(let t=0;t<n.length;t++){let[a,o]=n[t];if(o.length===0)continue;let s=o.buffer.defaultBuffer?r(o.buffer.defaultBuffer):o.buffer;for(let c=t+1;c<n.length;c++){let[t,l]=n[c];if(l.length===0)continue;let u=l.buffer.defaultBuffer?r(l.buffer.defaultBuffer):l.buffer;i(e,s!==u||o.byteOffset+o.length*o.rowByteLength<=l.byteOffset||l.byteOffset+l.length*l.rowByteLength<=o.byteOffset,`${a} and ${t} logical byte ranges must not overlap`)}}}var Y=`GPUShapeDescriptors`;function ne(e={},t=new Float32Array(4)){if(t.length<4)throw Error(`Shape descriptors target must hold 4 elements`);let n=e.sliverThreshold??.05;return o(`shape-descriptors`,`sliverThreshold`,n,0),i(`shape-descriptors`,n>=0,`sliverThreshold must be non-negative`),t.set([n,0,0,0]),t}var re=class{id;props;spatialContext;featureCount;useMonotoneChainConvexity;constructor(t){this.id=t.id??`shape-descriptors`,this.props=t,this.spatialContext=e(this.id,t.spatialContext);let{id:r}=this;if(n(t.positions,[`float32x2`],`${r} positions`),t.positions.length<1)throw Error(`${r} needs at least one position`);for(let e of[`ringOffsets`,`polygonOffsets`,`featureOffsets`])if(n(t[e],[`uint32`],`${r} ${e}`),t[e].length<2)throw Error(`${r} ${e} must contain at least two rows`);this.featureCount=t.featureOffsets.length-1,i(r,[`auto`,`serial`,`tiled`].includes(t.momentMethod??`auto`),`momentMethod must be 'auto', 'serial' or 'tiled'`);let o=t.convexityMethod??`auto`;if(![`auto`,`gift-wrapping`,`monotone-chain`].includes(o))throw Error(`${r} convexityMethod must be 'auto', 'gift-wrapping' or 'monotone-chain'`);let s=B(t.positions.length,this.featureCount);if(o===`monotone-chain`&&!s)throw Error(`${r} monotone-chain convexity supports fewer than 2^24 rows and features`);t.maximumVerticesPerFeature!==void 0&&a(r,`maximumVerticesPerFeature`,t.maximumVerticesPerFeature,1,4294967295);let l=t.maximumVerticesPerFeature!==void 0&&t.maximumVerticesPerFeature>=1024;if(this.useMonotoneChainConvexity=s&&(o===`monotone-chain`||o===`auto`&&(l||t.positions.length>=256*this.featureCount)),n(t.parameters,[`float32`],`${r} parameters`),t.parameters.length<4)throw Error(`${r} parameters must hold 4 float32 values`);let u=Object.entries(t.output).filter(([,e])=>e);if(u.length===0)throw Error(`${r} requires at least one output column`);for(let[e,t]of u)if(n(t,e===`clockwise`||e===`sliver`?[`uint32`]:[`float32`],`${r} output.${e}`),t.length!==this.featureCount)throw Error(`${r} output.${e} must hold ${this.featureCount} rows`);J(r,t.output),b(r,{areas:t.output.areas,perimeters:t.output.perimeters}),(t.output.polsbyPopper||t.output.schwartzberg||t.output.sliver)&&b(r,{areas:t.output.areas,perimeters:t.output.perimeters,polsbyPopper:t.output.polsbyPopper,schwartzberg:t.output.schwartzberg,sliver:t.output.sliver}),b(r,{elongation:t.output.elongation,orientation:t.output.orientation,clockwise:t.output.clockwise}),t.output.convexity&&b(r,{areas:t.output.areas,convexity:t.output.convexity}),c(r,u.map(([,e])=>e),[t.positions,t.ringOffsets,t.featureOffsets,t.polygonOffsets,t.parameters])}getCommandNodes(e){let{props:n,id:r,featureCount:i}=this,{output:a}=n;y(e,r,{positions:n.positions,ringOffsets:n.ringOffsets,featureOffsets:n.featureOffsets,polygonOffsets:n.polygonOffsets,parameters:n.parameters,...Object.fromEntries(Object.entries(a).map(([e,t])=>[`output.${e}`,t]))}),s(e.device,r,`geometry measure feature scratch`,i*16*4,`reduce the feature count`),a.convexity&&this.useMonotoneChainConvexity&&V(e.device,r,n.positions.length,i);let o=!!(a.elongation||a.orientation||a.clockwise),c=n.ringOffsets.length-1,u=o&&((n.momentMethod??`auto`)===`tiled`||(n.momentMethod??`auto`)===`auto`&&n.positions.length>=4096&&G(e.device,n.positions.length,c));u&&K(e.device,r,n.positions.length,c);let d=!!(a.polsbyPopper||a.schwartzberg||a.sliver),f=d||!!a.convexity||!!a.areas,p=d||!!a.perimeters,m=a.areas??(f?t(e,`${r}-areas`,`float32`,i):void 0),h=a.perimeters??(p?t(e,`${r}-perimeters`,`float32`,i):void 0),g=o?t(e,`${r}-centroids`,`float32x2`,i):void 0,_=[];(m||h||g)&&_.push(...new l({spatialContext:this.spatialContext,id:`${r}-measures`,positions:n.positions,geometryType:`polygons`,ringOffsets:n.ringOffsets,featureOffsets:n.featureOffsets,polygonOffsets:n.polygonOffsets,output:{lengths:h,areas:m,centroids:g}}).getCommandNodes(e));let v={operation:Y,positions:n.positions,ringOffsets:n.ringOffsets,featureOffsets:n.featureOffsets,polygonOffsets:n.polygonOffsets,featureCount:i};if(o&&g){let n=t(e,`${r}-moments`,`float32x4`,i);_.push(...u?q(e,{...v,id:`${r}-moments`,centroids:g,moments:n}):[F(e,{...v,id:`${r}-moments`,centroids:g,moments:n})],R(e,{id:`${r}-orientation`,operation:Y,featureCount:i,moments:n,elongation:a.elongation,orientation:a.orientation,clockwise:a.clockwise}))}return d&&m&&h&&_.push(L(e,{id:`${r}-compactness`,operation:Y,featureCount:i,areas:m,perimeters:h,parameters:n.parameters,polsbyPopper:a.polsbyPopper,schwartzberg:a.schwartzberg,sliver:a.sliver})),a.convexity&&m&&(this.useMonotoneChainConvexity?_.push(...H(e,{...v,id:`${r}-convexity`,areas:m,convexities:a.convexity})):_.push(I(e,{...v,id:`${r}-convexity`,areas:m,convexities:a.convexity}))),_}},X={irregularity:.12,vertexCount:16,rows:2,columns:5,originY:.2},Z=T(A,`sliverThreshold`),Q=.09,$=(e,t)=>t*(.08+.84*(w(e,Z.min,Z.max)-Z.min)/(Z.max-Z.min)),ie=(e,t)=>Z.min+(e/t-.08)/.84*(Z.max-Z.min),ae=E({parameters:A,caption:`Slivers flagged by compactness; the strip ranks every polygon against the threshold.`,createInputs:({seed:e,world:t})=>({seed:e,world:{width:t.width,height:.8},width:t.width,set:D(S(e),{width:t.width,height:.8},{...X,stretch:.8})}),build(e,t){let n=te(e,t.set),r=e.allocate(`parameters`,`float32`,4),i=e.allocate(`polsby-popper`,`float32`,t.set.count),a=e.allocate(`sliver`,`uint32`,t.set.count);e.graph.add(new re({positions:n.positions.view,featureOffsets:n.featureOffsets.view,polygonOffsets:n.polygonOffsets.view,ringOffsets:n.ringOffsets.view,parameters:r.view,output:{polsbyPopper:i.view,sliver:a.view}}));let o=e.allocate(`ticks`,`float32x2`,t.set.count);k(e,{id:`compactness-ticks`,invocationCount:t.set.count,bindings:[{name:`scores`,buffer:i,access:`read`},{name:`ticks`,buffer:o,access:`read_write`}],body:`let score = clamp(scores[scoresOffset + index], ${Z.min}, ${Z.max});
  ticks[ticksOffset + index * 2u] = ${t.width} * (0.08 + 0.84 * (score - ${Z.min}) / ${Z.max-Z.min});
  ticks[ticksOffset + index * 2u + 1u] = ${Q};`});let s=e.upload(`baseline`,`float32x4`,Float32Array.of(t.width*.08,Q,t.width*.92,Q));return{outputs:{...n,polsbyPopper:i,sliver:a,ticks:o,baseline:s},update:e=>{r.write(ne({sliverThreshold:e.sliverThreshold})),O(n,D(S(t.seed),t.world,{...X,stretch:e.stretch}))}}},interaction:{kind:`level`,hint:`slide the threshold`,rest:({inputs:e,toUnit:t})=>t([$(Z.default,e.width),.5]),resolve(e,{inputs:t,outputs:n,peek:r}){let i=C(Z,ie(e.x,t.width)),a=r(n.sliver),o=0;if(a)for(let e of a)o+=e;let s=$(i,t.width);return{values:{sliverThreshold:i},glyphs:[{shape:`line`,from:[s,.039999999999999994],to:[s,.14]}],readout:`sliverThreshold ${i.toFixed(2)} · ${a?o:`...`} slivers`}}},draw:({inputs:e,outputs:t})=>[{layer:`polygons`,positions:t.fillVertices,indices:t.fillIndices,owners:t.fillOwners,values:t.sliver,categories:[[.45,.55,.72],[1,.55,.25]],triangleCount:e.set.fillTriangleCount},{layer:`lines`,context:!0,segments:t.outline,count:e.set.outlineCount},{layer:`lines`,context:!0,segments:t.baseline,radius:1},{layer:`dots`,positions:t.ticks,values:t.sliver,categories:[[.45,.55,.72],[1,.55,.25]],radius:5}]});export{ae as default};