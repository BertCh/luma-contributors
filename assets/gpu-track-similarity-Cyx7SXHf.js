import{t as e}from"./ports-xmdKgL07.js";import{i as t,n,r}from"./trajectory-binding-validation-C89oJIAZ.js";import{a as i,m as a,v as o}from"./spatial-context-DwZEd4EZ.js";import{r as s}from"./kernel-C3tfQOn4.js";import{t as c}from"./geodesic-wgsl-I0CYDKcz.js";import{o as l}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as u,f as d,x as f,y as p}from"./gpu-contributor-utils-BIwLQON5.js";var m=4294967295,h=`
fn coupledMetric(swapped: bool, startA: u32, countA: u32, startB: u32, countB: u32, rowLong: u32, rowShort: u32) -> f32 {
  if (swapped) {
    return pointMetric(densifiedPointB(startB, countB, rowLong), densifiedPointA(startA, countA, rowShort));
  }
  return pointMetric(densifiedPointA(startA, countA, rowLong), densifiedPointB(startB, countB, rowShort));
}`,g=`let infinity = 3.402823466e+38;
  let denseA = densifiedCount(countA);
  let denseB = densifiedCount(countB);
  let swapped = denseB > denseA;
  if (lane == 0u) {
    sharedLongCount = max(denseA, denseB);
    sharedShortCount = min(denseA, denseB);
  }
  // Barrier-bearing loops need bounds the compiler can prove uniform across the workgroup.
  let longCount = workgroupUniformLoad(&sharedLongCount);
  let shortCount = workgroupUniformLoad(&sharedShortCount);
  let stripCount = (longCount + LANES - 1u) / LANES;`;function _(e,t,n){return`const WORKGROUP_SIZE: u32 = ${e}u;
const LANES: u32 = ${e}u;
const BAND_RADIUS: u32 = ${n}u;
const NO_BAND: u32 = ${m}u;
var<workgroup> olderDiagonal: array<f32, ${e}>;
var<workgroup> previousDiagonal: array<f32, ${e}>;
var<workgroup> boundaryRow: array<f32, ${t}>;
var<workgroup> sharedLongCount: u32;
var<workgroup> sharedShortCount: u32;
${h}
// Sakoe-Chiba band of tslearn's sakoe_chiba_mask(sz1, sz2, radius), written in long/short roles
// (tslearn's two branches are the same rule with the roles swapped): long index l is in the band
// of short index s when s - radius <= l <= s + (longCount - shortCount) + radius.
fn isInBand(rowLong: u32, rowShort: u32, longCount: u32, shortCount: u32) -> bool {
  if (BAND_RADIUS == NO_BAND) {
    return true;
  }
  return rowLong + BAND_RADIUS >= rowShort && rowLong <= rowShort + (longCount - shortCount) + BAND_RADIUS;
}`}function v(e){return`${e}
  ${g}
  var finalValue = infinity;
  for (var strip = 0u; strip < stripCount; strip++) {
    let firstRow = strip * LANES;
    let rows = min(LANES, longCount - firstRow);
    let diagonalCount = rows + shortCount - 1u;
    olderDiagonal[lane] = infinity;
    previousDiagonal[lane] = infinity;
    workgroupBarrier();
    for (var diagonal = 0u; diagonal < diagonalCount; diagonal++) {
      var value = infinity;
      var column = 0u;
      let isActive = lane < rows && diagonal >= lane && diagonal - lane < shortCount;
      if (isActive) {
        column = diagonal - lane;
        let rowLong = firstRow + lane;
        if (isInBand(rowLong, column, longCount, shortCount)) {
          let cost = finishMetric(coupledMetric(swapped, startA, countA, startB, countB, rowLong, column));
          var up = infinity;
          var upLeft = infinity;
          var left = infinity;
          if (lane > 0u) {
            up = previousDiagonal[lane - 1u];
            upLeft = olderDiagonal[lane - 1u];
          } else if (strip > 0u) {
            up = boundaryRow[column];
            if (column > 0u) { upLeft = boundaryRow[column - 1u]; }
          }
          if (column > 0u) { left = previousDiagonal[lane]; }
          let best = min(min(up, upLeft), left);
          if (rowLong == 0u && column == 0u) {
            value = cost;
          } else if (best < infinity) {
            value = min(best + cost, infinity);
          }
        }
        if (rowLong == longCount - 1u && column == shortCount - 1u) { finalValue = value; }
      }
      workgroupBarrier();
      olderDiagonal[lane] = previousDiagonal[lane];
      previousDiagonal[lane] = value;
      if (isActive && lane == LANES - 1u) { boundaryRow[column] = value; }
      workgroupBarrier();
    }
  }
  if (!skipped && longCount > 0u && lane == (longCount - 1u) % LANES) {
    distances[distancesOffset + pair] = finalValue;
  } else if (skipped && lane == 0u && pair < PAIR_COUNT) {
    distances[distancesOffset + pair] = notANumber;
  }`}function y(e,t,n){return`const WORKGROUP_SIZE: u32 = ${e}u;
const LANES: u32 = ${e}u;
var<workgroup> olderDiagonal: array<f32, ${e}>;
var<workgroup> previousDiagonal: array<f32, ${e}>;
var<workgroup> boundaryRow: array<f32, ${t}>;
const PLANAR_BOUNDS: bool = ${!n};
var<workgroup> partials: array<vec4f, ${e}>;
var<workgroup> sharedLongCount: u32;
var<workgroup> sharedShortCount: u32;
var<workgroup> sharedDecision: u32;
var<workgroup> sharedReachable: u32;
var<workgroup> reachableCounter: atomic<u32>;
var<workgroup> boundaryReachableCounter: atomic<u32>;
${h}
// Epsilon in the internal metric space of pointMetric (squared planar distance, or the context's length unit).
fn getEpsilonMetric(epsilon: f32) -> f32 {
  ${n?`return epsilon;`:`return epsilon * epsilon;`}
}
// Short index coupled with long index rowLong by a fixed monotone coupling: proportional
// (floor(l * (S - 1) / (L - 1))) when that product fits 32 bits, otherwise the diagonal min(l, S - 1).
// Each step advances the short index by at most 1 (S <= L), so it is a valid discrete coupling and
// its largest coupled distance is an upper bound of the Frechet distance.
fn getCoupledShortRow(rowLong: u32, longCount: u32, shortCount: u32) -> u32 {
  if (longCount <= 1u || shortCount <= 1u) {
    return 0u;
  }
  if (longCount - 1u <= 0xffffffffu / (shortCount - 1u)) {
    return rowLong * (shortCount - 1u) / (longCount - 1u);
  }
  return min(rowLong, shortCount - 1u);
}`}function b(e){return`${e}
  ${g}
  let epsilon = frechetEpsilon[frechetEpsilonOffset];
  let isEpsilonUsable = (bitcast<u32>(epsilon) & 0x7f800000u) != 0x7f800000u && epsilon >= 0.0;
  let limit = getEpsilonMetric(epsilon);
  // Reject endpoints before reading the rest of either track or reducing a coupling bound.
  if (lane == 0u) {
    sharedDecision = 2u;
    if (!skipped && isEpsilonUsable && longCount > 0u) {
      let lower = max(
        coupledMetric(swapped, startA, countA, startB, countB, 0u, 0u),
        coupledMetric(swapped, startA, countA, startB, countB, longCount - 1u, shortCount - 1u)
      );
      if (!(lower > limit)) { sharedDecision = 0u; }
    }
  }
  var decision = workgroupUniformLoad(&sharedDecision);
  if (decision == 2u) {
    if (lane == 0u && pair < PAIR_COUNT) { frechetWithin[frechetWithinOffset + pair] = 0u; }
    return;
  }
  // A fixed coupling accepts in linear work. Do not pay for box reductions on accepted pairs.
  var coupled = 0.0;
  for (var rowLong = lane; rowLong < longCount; rowLong += LANES) {
    let rowShort = getCoupledShortRow(rowLong, longCount, shortCount);
    coupled = max(coupled, coupledMetric(swapped, startA, countA, startB, countB, rowLong, rowShort));
  }
  partials[lane] = vec4f(coupled);
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) { partials[lane] = max(partials[lane], partials[lane + stride]); }
    workgroupBarrier();
  }
  if (lane == 0u && partials[0].x <= limit) { sharedDecision = 1u; }
  decision = workgroupUniformLoad(&sharedDecision);
  if (decision == 0u && PLANAR_BOUNDS) {
    // Every vertex must be matched. Differences between corresponding extrema of the two
    // boxes lower-bound the maximum distance in any coupling, even when the boxes overlap.
    // Use densified samples too: these are exactly the coordinates used by the free-space DP.
    var minima = vec4f(infinity);
    var maxima = vec4f(-infinity);
    for (var vertex = lane; vertex < denseA; vertex += LANES) {
      let point = densifiedPointA(startA, countA, vertex);
      minima = vec4f(min(minima.xy, point), minima.zw);
      maxima = vec4f(max(maxima.xy, point), maxima.zw);
    }
    for (var vertex = lane; vertex < denseB; vertex += LANES) {
      let point = densifiedPointB(startB, countB, vertex);
      minima = vec4f(minima.xy, min(minima.zw, point));
      maxima = vec4f(maxima.xy, max(maxima.zw, point));
    }
    partials[lane] = minima;
    workgroupBarrier();
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
      if (lane < stride) { partials[lane] = min(partials[lane], partials[lane + stride]); }
      workgroupBarrier();
    }
    let minimumGap = abs(partials[0].xy - partials[0].zw);
    workgroupBarrier();
    partials[lane] = maxima;
    workgroupBarrier();
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
      if (lane < stride) { partials[lane] = max(partials[lane], partials[lane + stride]); }
      workgroupBarrier();
    }
    if (lane == 0u) {
      let maximumGap = abs(partials[0].xy - partials[0].zw);
      let gaps = max(minimumGap, maximumGap);
      let lower = max(gaps.x, gaps.y);
      // Widen the threshold to keep f32 rounding at equality in the exact DP path.
      if (lower * lower > limit * 1.000001) { sharedDecision = 2u; }
    }
    decision = workgroupUniformLoad(&sharedDecision);
  }
  // 2. Free-space reachability, 1.0 reachable and 0.0 not.
  if (decision == 0u) {
    var finalValue = 0.0;
    var previousBoundaryCount = 0u;
    for (var strip = 0u; strip < stripCount; strip++) {
      let firstRow = strip * LANES;
      let rows = min(LANES, longCount - firstRow);
      let diagonalCount = rows + shortCount - 1u;
      olderDiagonal[lane] = 0.0;
      previousDiagonal[lane] = 0.0;
      if (lane == 0u) {
        atomicStore(&reachableCounter, 0u);
        atomicStore(&boundaryReachableCounter, 0u);
      }
      workgroupBarrier();
      var lastWrittenColumn = 0u;
      var hasWritten = false;
      // Consecutive anti-diagonals without a reachable cell (meaningful in lane 0 only).
      var deadDiagonals = 0u;
      for (var diagonal = 0u; diagonal < diagonalCount; diagonal++) {
        var value = 0.0;
        var column = 0u;
        let isActive = lane < rows && diagonal >= lane && diagonal - lane < shortCount;
        if (isActive) {
          column = diagonal - lane;
          let rowLong = firstRow + lane;
          {
            var predecessor = rowLong == 0u && column == 0u;
            if (lane > 0u) {
              predecessor = predecessor || previousDiagonal[lane - 1u] > 0.0 || olderDiagonal[lane - 1u] > 0.0;
            } else if (strip > 0u) {
              predecessor = predecessor || boundaryRow[column] > 0.0;
              if (column > 0u) { predecessor = predecessor || boundaryRow[column - 1u] > 0.0; }
            }
            if (column > 0u) { predecessor = predecessor || previousDiagonal[lane] > 0.0; }
            if (predecessor && coupledMetric(swapped, startA, countA, startB, countB, rowLong, column) <= limit) {
              value = 1.0;
              atomicAdd(&reachableCounter, 1u);
            }
          }
          if (rowLong == longCount - 1u && column == shortCount - 1u) { finalValue = value; }
        }
        workgroupBarrier();
        olderDiagonal[lane] = previousDiagonal[lane];
        previousDiagonal[lane] = value;
        if (isActive && lane == LANES - 1u) {
          boundaryRow[column] = value;
          if (value > 0.0) { atomicAdd(&boundaryReachableCounter, 1u); }
        }
        if (lane == 0u) {
          // A cell's predecessors lie on the two previous anti-diagonals (up and left on the
          // last, up-left on the one before) or in the carried row (lane 0, from column
          // diagonal + 1 on). Two dead anti-diagonals and no reachable carried cell end the strip.
          deadDiagonals = select(0u, deadDiagonals + 1u, atomicLoad(&reachableCounter) == 0u);
          let isDead = deadDiagonals >= 2u &&
            (strip == 0u || previousBoundaryCount == 0u || diagonal + 1u >= shortCount);
          sharedReachable = select(1u, 0u, isDead);
          atomicStore(&reachableCounter, 0u);
        }
        lastWrittenColumn = select(lastWrittenColumn, diagonal - (LANES - 1u), diagonal >= LANES - 1u);
        hasWritten = hasWritten || diagonal >= LANES - 1u;
        let isAlive = workgroupUniformLoad(&sharedReachable);
        if (isAlive == 0u) { break; }
      }
      workgroupBarrier();
      // Columns of the carried row this strip never reached are unreachable.
      if (rows == LANES) {
        let firstUnwritten = select(0u, lastWrittenColumn + 1u, hasWritten);
        for (var column = firstUnwritten + lane; column < shortCount; column += LANES) {
          boundaryRow[column] = 0.0;
        }
      }
      if (lane == 0u) {
        sharedReachable = atomicLoad(&boundaryReachableCounter);
      }
      previousBoundaryCount = workgroupUniformLoad(&sharedReachable);
      // A full strip whose last row has no reachable cell cuts every monotone path.
      if (rows == LANES && strip + 1u < stripCount && previousBoundaryCount == 0u) { break; }
    }
    if (lane == (longCount - 1u) % LANES && finalValue > 0.0) {
      sharedDecision = 1u;
    }
    decision = workgroupUniformLoad(&sharedDecision);
  }
  if (lane == 0u && pair < PAIR_COUNT) {
    frechetWithin[frechetWithinOffset + pair] = select(0u, 1u, decision == 1u);
  }`}function x(e,t){return`${e}
  ${g}
  var finalValue = 0.0;
  for (var strip = 0u; strip < stripCount; strip++) {
    let firstRow = strip * LANES;
    let rows = min(LANES, longCount - firstRow);
    olderDiagonal[lane] = 0.0;
    previousDiagonal[lane] = 0.0;
    workgroupBarrier();
    for (var diagonal = 0u; diagonal < rows + shortCount - 1u; diagonal++) {
      var value = 0.0;
      var column = 0u;
      let isActive = lane < rows && diagonal >= lane && diagonal - lane < shortCount;
      if (isActive) {
        column = diagonal - lane;
        let rowLong = firstRow + lane;
        var up = 0.0;
        var previous = 0.0;
        var left = 0.0;
        if (lane > 0u) {
          up = previousDiagonal[lane - 1u];
          if (column > 0u) { previous = olderDiagonal[lane - 1u]; }
        } else if (strip > 0u) {
          up = boundaryRow[column];
          if (column > 0u) { previous = boundaryRow[column - 1u]; }
        }
        if (column > 0u) { left = previousDiagonal[lane]; }
        value = max(up, left);
        if (isInBand(rowLong, column, longCount, shortCount) &&
          finishMetric(coupledMetric(swapped, startA, countA, startB, countB, rowLong, column)) <= f32(${t})) {
          value = max(value, previous + 1.0);
        }
        if (rowLong == longCount - 1u && column == shortCount - 1u) { finalValue = value; }
      }
      workgroupBarrier();
      olderDiagonal[lane] = previousDiagonal[lane];
      previousDiagonal[lane] = value;
      if (isActive && lane == LANES - 1u) { boundaryRow[column] = value; }
      workgroupBarrier();
    }
  }
  if (!skipped && longCount > 0u && lane == (longCount - 1u) % LANES) {
    distances[distancesOffset + pair] = finalValue / f32(shortCount);
  } else if (skipped && lane == 0u && pair < PAIR_COUNT) {
    distances[distancesOffset + pair] = notANumber;
  }`}function S(e){let t=String(Math.fround(e));return/[.e]/.test(t)?t:`${t}.0`}function C(e,t,n,r,i){let a=e===`edr`?`return 1.0;`:`var point = densifiedPointB(startB, countB, row);
  if (isLong != swapped) { point = densifiedPointA(startA, countA, row); }
  return finishMetric(pointMetric(point, GAP_POINT));`,o=e===`edr`?`return select(1.0, 0.0, finishMetric(coupledMetric(swapped, startA, countA, startB, countB, rowLong, rowShort)) <= f32(${S(r)}));`:`return finishMetric(coupledMetric(swapped, startA, countA, startB, countB, rowLong, rowShort));`;return`const WORKGROUP_SIZE: u32 = ${t}u;
const LANES: u32 = ${t}u;
const GAP_POINT: vec2<f32> = vec2<f32>(${S(i[0])}, ${S(i[1])});
var<workgroup> olderDiagonal: array<f32, ${t}>;
var<workgroup> previousDiagonal: array<f32, ${t}>;
var<workgroup> boundaryRow: array<f32, ${n+1}>;
var<workgroup> sharedLongCount: u32;
var<workgroup> sharedShortCount: u32;
${h}
// Cost of inserting or deleting densified vertex row of the long track (isLong) or of the short one.
fn gapCost(swapped: bool, startA: u32, countA: u32, startB: u32, countB: u32, row: u32, isLong: bool) -> f32 {
  ${a}
}
fn substitutionCost(swapped: bool, startA: u32, countA: u32, startB: u32, countB: u32, rowLong: u32, rowShort: u32) -> f32 {
  ${o}
}`}function w(e){return`${e}
  let infinity = 3.402823466e+38;
  let denseA = densifiedCount(countA);
  let denseB = densifiedCount(countB);
  let swapped = denseB > denseA;
  if (lane == 0u) {
    sharedLongCount = max(denseA, denseB);
    sharedShortCount = min(denseA, denseB);
  }
  // Barrier-bearing loops need bounds the compiler can prove uniform across the workgroup.
  let longCount = workgroupUniformLoad(&sharedLongCount);
  let shortCount = workgroupUniformLoad(&sharedShortCount);
  // Table rows and columns 0 hold the empty prefix; vertex k is row or column k + 1.
  let rowCount = longCount + 1u;
  let columnCount = shortCount + 1u;
  let stripCount = (rowCount + LANES - 1u) / LANES;
  var finalValue = infinity;
  for (var strip = 0u; strip < stripCount; strip++) {
    let firstRow = strip * LANES;
    let rows = min(LANES, rowCount - firstRow);
    let diagonalCount = rows + columnCount - 1u;
    olderDiagonal[lane] = infinity;
    previousDiagonal[lane] = infinity;
    workgroupBarrier();
    for (var diagonal = 0u; diagonal < diagonalCount; diagonal++) {
      var value = infinity;
      var column = 0u;
      let isActive = lane < rows && diagonal >= lane && diagonal - lane < columnCount;
      if (isActive) {
        column = diagonal - lane;
        let row = firstRow + lane;
        var up = infinity;
        var upLeft = infinity;
        var left = infinity;
        if (lane > 0u) {
          up = previousDiagonal[lane - 1u];
          upLeft = olderDiagonal[lane - 1u];
        } else if (strip > 0u) {
          up = boundaryRow[column];
          if (column > 0u) { upLeft = boundaryRow[column - 1u]; }
        }
        if (column > 0u) { left = previousDiagonal[lane]; }
        if (row == 0u && column == 0u) {
          value = 0.0;
        } else if (row == 0u) {
          value = left + gapCost(swapped, startA, countA, startB, countB, column - 1u, false);
        } else if (column == 0u) {
          value = up + gapCost(swapped, startA, countA, startB, countB, row - 1u, true);
        } else {
          value = min(
            upLeft + substitutionCost(swapped, startA, countA, startB, countB, row - 1u, column - 1u),
            min(
              up + gapCost(swapped, startA, countA, startB, countB, row - 1u, true),
              left + gapCost(swapped, startA, countA, startB, countB, column - 1u, false)
            )
          );
        }
        if (row == longCount && column == shortCount) { finalValue = value; }
      }
      workgroupBarrier();
      olderDiagonal[lane] = previousDiagonal[lane];
      previousDiagonal[lane] = value;
      if (isActive && lane == LANES - 1u) { boundaryRow[column] = value; }
      workgroupBarrier();
    }
  }
  if (!skipped && longCount > 0u && lane == longCount % LANES) {
    distances[distancesOffset + pair] = finalValue;
  } else if (skipped && lane == 0u && pair < PAIR_COUNT) {
    distances[distancesOffset + pair] = notANumber;
  }`}function T(e){let t=e=>`// Metric value from a point to the polyline of track ${e}: every segment, in order.
fn nearestMetricTo${e}(point: vec2<f32>, start: u32, count: u32) -> f32 {
  if (count == 1u) { return pointMetric(point, load${e}(start)); }
  var nearest = 3.402823466e+38;
  for (var segment = 0u; segment + 1u < count; segment++) {
    nearest = min(nearest, pointSegmentMetric(point, load${e}(start + segment), load${e}(start + segment + 1u)));
  }
  return nearest;
}`;return`const WORKGROUP_SIZE: u32 = ${e}u;
var<workgroup> partials: array<f32, ${e}>;
${t(`A`)}
${t(`B`)}`}function E(e,t){return`${e}
  var sumAToB = 0.0;
  for (var vertex = lane; vertex < countA; vertex += WORKGROUP_SIZE) {
    sumAToB += finishMetric(nearestMetricToB(loadA(startA + vertex), startB, countB));
  }
  partials[lane] = sumAToB;
  workgroupBarrier();
  for (var stride = ${t/2}u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) { partials[lane] = partials[lane] + partials[lane + stride]; }
    workgroupBarrier();
  }
  let totalAToB = partials[0];
  workgroupBarrier();
  var sumBToA = 0.0;
  for (var vertex = lane; vertex < countB; vertex += WORKGROUP_SIZE) {
    sumBToA += finishMetric(nearestMetricToA(loadB(startB + vertex), startA, countA));
  }
  partials[lane] = sumBToA;
  workgroupBarrier();
  for (var stride = ${t/2}u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) { partials[lane] = partials[lane] + partials[lane + stride]; }
    workgroupBarrier();
  }
  if (lane == 0u && pair < PAIR_COUNT) {
    distances[distancesOffset + pair] = select(0.5 * (totalAToB / f32(countA) + partials[0] / f32(countB)), notANumber, skipped);
  }`}var D=Object.freeze({...i,measure:`distance`}),O=`GPUTrackSimilarity`,k=64,A=256,j=Math.floor(4294967295/A),M=2048,N=4096,P=4294967295,F=2**30;function I(e){return`// Metric value (squared distance, or the context's length unit on the sphere) from a point to polyline ${e}; may stop early once it is at most worst.
fn distanceToPolyline${e}(point: vec2<f32>, start: u32, count: u32, worst: f32, hint: ptr<function, u32>) -> f32 {
  if (count == 1u) {
    return pointMetric(point, load${e}(start));
  }
  let segments = count - 1u;
  var nearest = 3.402823466e+38;
  var first = *hint;
  if (first >= segments) { first = 0u; }
  for (var step = 0u; step < segments; step++) {
    var j = first + step;
    if (j >= segments) { j -= segments; }
    let squared = pointSegmentMetric(point, load${e}(start + j), load${e}(start + j + 1u));
    if (squared < nearest) {
      nearest = squared;
      *hint = j;
      if (nearest <= worst) { break; }
    }
  }
  return nearest;
}
// Cooperative minimum over the segments of polyline ${e}: every lane takes a stride of segments.
fn cooperativeDistanceToPolyline${e}(point: vec2<f32>, start: u32, count: u32, lane: u32) -> f32 {
  var nearest = 3.402823466e+38;
  if (count == 1u) {
    return pointMetric(point, load${e}(start));
  }
  for (var j = lane; j + 1u < count; j += WORKGROUP_SIZE) {
    nearest = min(nearest, pointSegmentMetric(point, load${e}(start + j), load${e}(start + j + 1u)));
  }
  return nearest;
}`}function L(e,t){return`{
    let pointCount = uniformDense${e};
    if (pointCount * 8u <= WORKGROUP_SIZE) {
      for (var i = 0u; i < pointCount; i++) {
        let point = densifiedPoint${e}(start${e}, count${e}, i);
        partials[lane] = cooperativeDistanceToPolyline${t}(point, start${t}, count${t}, lane);
        workgroupBarrier();
        for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
          if (lane < stride) { partials[lane] = min(partials[lane], partials[lane + stride]); }
          workgroupBarrier();
        }
        worst = max(worst, partials[0]);
        workgroupBarrier();
      }
    } else {
      let chunk = (pointCount + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
      let chunkStart = min(lane * chunk, pointCount);
      let chunkEnd = min(chunkStart + chunk, pointCount);
      var hint = 0u;
      for (var i = chunkStart; i < chunkEnd; i++) {
        let nearest = distanceToPolyline${t}(densifiedPoint${e}(start${e}, count${e}, i), start${t}, count${t}, worst, &hint);
        worst = max(worst, nearest);
      }
    }
  }`}var R={emptyTrack:1,frechetCapExceeded:2,invalidPair:4},z=class{id;props;spatialContext;sphereRadius;maximumFrechetVertexCount;densifySubdivisions;constructor(t){this.id=t.id??`track-similarity`;let{id:n}=this;this.spatialContext=o(n,t.spatialContext,D),this.sphereRadius=a(this.spatialContext),t={...t,linesA:e(n,`linesA`,t.linesA),linesB:t.linesB&&e(n,`linesB`,t.linesB)},this.props=t;let{linesA:i,linesB:s,output:c}=t;if(u(i.positions,[`float32x2`],`${n} linesA.positions`),u(i.lineOffsets,[`uint32`],`${n} linesA.lineOffsets`),i.lineOffsets.length<2)throw Error(`${n} linesA.lineOffsets needs two rows`);if(s&&(u(s.positions,[`float32x2`],`${n} linesB.positions`),u(s.lineOffsets,[`uint32`],`${n} linesB.lineOffsets`),s.lineOffsets.length<2))throw Error(`${n} linesB.lineOffsets needs two rows`);if(u(t.pairA,[`uint32`],`${n} pairA`),u(t.pairB,[`uint32`],`${n} pairB`),t.pairA.length<1||t.pairB.length!==t.pairA.length)throw Error(`${n} pairA and pairB must have the same positive length`);if(d(n,`pairA.length`,t.pairA.length,1,j),t.activePairCount&&(u(t.activePairCount,[`uint32`],`${n} activePairCount`),t.activePairCount.length<1))throw Error(`${n} activePairCount must contain one uint32 row`);if(!c.hausdorff&&!c.frechet&&!c.maximumDistance&&!c.frechetWithin&&!c.dynamicTimeWarping&&!c.editDistanceOnRealSequence&&!c.editDistanceWithRealPenalty&&!c.symmetricSegmentPathDistance&&!c.longestCommonSubsequence)throw Error(`${n} needs hausdorff, frechet, frechetWithin, dynamicTimeWarping, editDistanceOnRealSequence, editDistanceWithRealPenalty, symmetricSegmentPathDistance, longestCommonSubsequence or maximumDistance output`);if(!!c.frechetWithin!=!!t.frechetEpsilon)throw Error(`${n} output.frechetWithin and frechetEpsilon go together`);if(t.frechetEpsilon&&(u(t.frechetEpsilon,[`float32`],`${n} frechetEpsilon`),t.frechetEpsilon.length<1))throw Error(`${n} frechetEpsilon must contain one float32 row`);t.dynamicTimeWarpingBandRadius!==void 0&&d(n,`dynamicTimeWarpingBandRadius`,t.dynamicTimeWarpingBandRadius,0,F);for(let[e,r,i]of[[`hausdorff`,c.hausdorff,`float32`],[`frechet`,c.frechet,`float32`],[`frechetWithin`,c.frechetWithin,`uint32`],[`dynamicTimeWarping`,c.dynamicTimeWarping,`float32`],[`editDistanceOnRealSequence`,c.editDistanceOnRealSequence,`float32`],[`editDistanceWithRealPenalty`,c.editDistanceWithRealPenalty,`float32`],[`symmetricSegmentPathDistance`,c.symmetricSegmentPathDistance,`float32`],[`longestCommonSubsequence`,c.longestCommonSubsequence,`float32`],[`maximumDistance`,c.maximumDistance,`float32`],[`pairStatus`,c.pairStatus,`uint32`]])if(r&&(u(r,[i],`${n} ${e}`),r.length!==t.pairA.length))throw Error(`${n} ${e} length must equal the pair capacity`);if(!Number.isFinite(Math.fround(t.lcssTolerance??1))||(t.lcssTolerance??1)<0)throw Error(`${n} lcssTolerance must be finite and nonnegative`);if(!Number.isFinite(Math.fround(t.editDistanceTolerance??1))||(t.editDistanceTolerance??1)<0)throw Error(`${n} editDistanceTolerance must be finite and nonnegative`);if(t.editDistanceGapPoint?.some(e=>!Number.isFinite(Math.fround(e))))throw Error(`${n} editDistanceGapPoint must be finite`);t.lcssBandRadius!==void 0&&d(n,`lcssBandRadius`,t.lcssBandRadius,0,F);let l=t.densify??0;if(!(l>=0&&l<=1))throw Error(`${n} densify must be a fraction from 0 to 1`);if(this.densifySubdivisions=l===0?1:Math.round(1/l),this.densifySubdivisions>N)throw Error(`${n} densify must be at least 1 / ${N}`);this.maximumFrechetVertexCount=t.maximumFrechetVertexCount??256,d(n,`maximumFrechetVertexCount`,this.maximumFrechetVertexCount,1,M),r(this.id,this.props.output),p(n,[c.hausdorff,c.frechet,c.frechetWithin,c.dynamicTimeWarping,c.editDistanceOnRealSequence,c.editDistanceWithRealPenalty,c.symmetricSegmentPathDistance,c.longestCommonSubsequence,c.maximumDistance],[c.pairStatus]),p(n,[c.hausdorff,c.frechet,c.frechetWithin,c.dynamicTimeWarping,c.editDistanceOnRealSequence,c.editDistanceWithRealPenalty,c.symmetricSegmentPathDistance,c.longestCommonSubsequence,c.maximumDistance,c.pairStatus],[i.positions,i.lineOffsets,s?.positions,s?.lineOffsets,t.pairA,t.pairB,t.activePairCount,t.frechetEpsilon])}getCommandNodes(e){n(e,this.id,this.props);let{id:r,props:i}=this,{linesA:a,linesB:o,output:u}=i;f(r,e,[a.positions,a.lineOffsets,o?.positions,o?.lineOffsets,i.pairA,i.pairB,i.activePairCount,i.frechetEpsilon,u.hausdorff,u.frechet,u.frechetWithin,u.dynamicTimeWarping,u.editDistanceOnRealSequence,u.editDistanceWithRealPenalty,u.symmetricSegmentPathDistance,u.longestCommonSubsequence,u.maximumDistance,u.pairStatus]);let d=o?.positions??a.positions,p=o?.lineOffsets??a.lineOffsets,m=i.pairA.length;t(e,r,{"pair status":m*4,...u.frechetWithin?{"decision pair ranges":m*16}:{}});let h=[],g=!!(u.frechet||u.frechetWithin||u.dynamicTimeWarping||u.editDistanceOnRealSequence||u.editDistanceWithRealPenalty||u.longestCommonSubsequence),S=!!(u.hausdorff||g),D=`const PAIR_COUNT: u32 = ${m}u;
const TRACK_COUNT_A: u32 = ${a.lineOffsets.length-1}u;
const TRACK_COUNT_B: u32 = ${p.length-1}u;
const FRECHET_CAP: u32 = ${this.maximumFrechetVertexCount}u;
const DENSIFY: u32 = ${this.densifySubdivisions}u;
const MAXIMUM_DENSIFIED_VERTICES: u32 = ${P}u;
const STATUS_EMPTY: u32 = ${R.emptyTrack}u;
const STATUS_TOO_LONG: u32 = ${R.frechetCapExceeded}u;
const STATUS_INVALID: u32 = ${R.invalidPair}u;`,j=u.pairStatus??s(e,`${r}-status`,`uint32`,m),M=[{name:`pairA`,view:i.pairA,type:`u32`,access:`read`},{name:`pairB`,view:i.pairB,type:`u32`,access:`read`},{name:`offsetsA`,view:a.lineOffsets,type:`u32`,access:`read`},{name:`offsetsB`,view:p,type:`u32`,access:`read`}];i.activePairCount&&M.push({name:`activePairCount`,view:i.activePairCount,type:`u32`,access:`read`}),M.push({name:`status`,view:j,type:`u32`,access:`read_write`}),h.push(l(e,{id:`${r}-status`,operation:O,variant:`status`,bindings:M,invocationCount:m,declarations:D,body:`let a = pairA[pairAOffset + index];
  let b = pairB[pairBOffset + index];
  var flags = 0u;
  ${i.activePairCount?`if (index >= activePairCount[activePairCountOffset]) { flags = STATUS_INVALID; }`:``}
  if (a >= TRACK_COUNT_A || b >= TRACK_COUNT_B) {
    flags = STATUS_INVALID;
  } else if (flags == 0u) {
    let startA = offsetsA[offsetsAOffset + a];
    let endA = offsetsA[offsetsAOffset + a + 1u];
    let startB = offsetsB[offsetsBOffset + b];
    let endB = offsetsB[offsetsBOffset + b + 1u];
    let countA = endA - min(startA, endA);
    let countB = endB - min(startB, endB);
    if (startA > endA || startB > endB || endA > ${a.positions.length}u || endB > ${d.length}u) {
      flags = STATUS_INVALID;
    } else if (countA == 0u || countB == 0u) {
      flags = STATUS_EMPTY;
    } else if (${S} && max(countA - 1u, countB - 1u) > (MAXIMUM_DENSIFIED_VERTICES - 1u) / DENSIFY) {
      // A densified vertex count must fit uint32 before any distance kernel uses it.
      flags = STATUS_INVALID;
    } else if (${g} && min(countA - 1u, countB - 1u) > (FRECHET_CAP - 1u) / DENSIFY) {
      flags = STATUS_TOO_LONG;
    }
  }
  status[statusOffset + index] = flags;`}));let N=e=>[{name:`pairA`,view:i.pairA,type:`u32`,access:`read`},{name:`pairB`,view:i.pairB,type:`u32`,access:`read`},{name:`positionsA`,view:a.positions,type:`f32`,access:`read`},{name:`offsetsA`,view:a.lineOffsets,type:`u32`,access:`read`},{name:`positionsB`,view:d,type:`f32`,access:`read`},{name:`offsetsB`,view:p,type:`u32`,access:`read`},{name:`status`,view:j,type:`u32`,access:`read`},{name:`distances`,view:e,type:`f32`,access:`read_write`}],F=e=>`let pair = index / WORKGROUP_SIZE;
  let lane = localInvocationIndex;
  let flags = status[statusOffset + min(pair, PAIR_COUNT - 1u)];
  let skipped = pair >= PAIR_COUNT || (flags & ${e}) != 0u;
  var startA = 0u;
  var countA = 0u;
  var startB = 0u;
  var countB = 0u;
  if (!skipped) {
    let a = pairA[pairAOffset + pair];
    let b = pairB[pairBOffset + pair];
    startA = offsetsA[offsetsAOffset + a];
    countA = offsetsA[offsetsAOffset + a + 1u] - startA;
    startB = offsetsB[offsetsBOffset + b];
    countB = offsetsB[offsetsBOffset + b + 1u] - startB;
  }
  // A runtime expression keeps the quiet-NaN bit pattern out of const evaluation.
  let notANumber = bitcast<f32>(0x7fc00000u | (index & 0u));`,z=`fn loadA(row: u32) -> vec2<f32> {
  return vec2<f32>(positionsA[positionsAOffset + row * 2u], positionsA[positionsAOffset + row * 2u + 1u]);
}
fn loadB(row: u32) -> vec2<f32> {
  return vec2<f32>(positionsB[positionsBOffset + row * 2u], positionsB[positionsBOffset + row * 2u + 1u]);
}
fn densifiedCount(count: u32) -> u32 {
  return select(0u, (count - 1u) * DENSIFY + 1u, count > 0u);
}
fn densifiedPointA(start: u32, count: u32, k: u32) -> vec2<f32> {
  let segment = k / DENSIFY;
  let subStep = k - segment * DENSIFY;
  if (subStep == 0u || segment + 1u >= count) { return loadA(start + min(segment, count - 1u)); }
  return interpolatePoint(loadA(start + segment), loadA(start + segment + 1u), f32(subStep) / f32(DENSIFY));
}
fn densifiedPointB(start: u32, count: u32, k: u32) -> vec2<f32> {
  let segment = k / DENSIFY;
  let subStep = k - segment * DENSIFY;
  if (subStep == 0u || segment + 1u >= count) { return loadB(start + min(segment, count - 1u)); }
  return interpolatePoint(loadB(start + segment), loadB(start + segment + 1u), f32(subStep) / f32(DENSIFY));
}
${this.sphereRadius===void 0?`fn pointMetric(a: vec2<f32>, b: vec2<f32>) -> f32 {
  let offset = a - b;
  return dot(offset, offset);
}
fn finishMetric(value: f32) -> f32 {
  return sqrt(value);
}
fn interpolatePoint(from0: vec2<f32>, to0: vec2<f32>, fraction: f32) -> vec2<f32> {
  return from0 + (to0 - from0) * fraction;
}
fn pointSegmentMetric(point: vec2<f32>, from0: vec2<f32>, to0: vec2<f32>) -> f32 {
  let edge = to0 - from0;
  let lengthSquared = dot(edge, edge);
  var t = 0.0;
  if (lengthSquared > 0.0) { t = clamp(dot(point - from0, edge) / lengthSquared, 0.0, 1.0); }
  let offset = point - (from0 + edge * t);
  return dot(offset, offset);
}`:`${c}
const SPHERE_RADIUS: f32 = ${this.sphereRadius};
fn pointMetric(a: vec2<f32>, b: vec2<f32>) -> f32 {
  return geodesicHaversineDistance(a, b, SPHERE_RADIUS);
}
fn finishMetric(value: f32) -> f32 {
  return value;
}
// Linear in longitude/latitude with the longitude difference wrapped across the antimeridian.
fn interpolatePoint(from0: vec2<f32>, to0: vec2<f32>, fraction: f32) -> vec2<f32> {
  return from0 + vec2<f32>(geodesicWrapLongitudeDelta(to0.x - from0.x), to0.y - from0.y) * fraction;
}
fn pointSegmentMetric(point: vec2<f32>, from0: vec2<f32>, to0: vec2<f32>) -> f32 {
  let t = geodesicSegmentFraction(point, from0, to0);
  return geodesicSegmentDistance(point, from0, to0, t, SPHERE_RADIUS);
}`}`;u.hausdorff&&h.push(l(e,{id:`${r}-hausdorff`,operation:O,variant:`hausdorff`,bindings:N(u.hausdorff),workgroupSize:k,invocationCount:m*k,guardIndex:!1,declarations:`${D}
const WORKGROUP_SIZE: u32 = ${k}u;
var<workgroup> partials: array<f32, ${k}>;
var<workgroup> sharedDenseA: u32;
var<workgroup> sharedDenseB: u32;
${z}
${I(`A`)}
${I(`B`)}`,body:`${F(`(STATUS_EMPTY | STATUS_INVALID)`)}
  var worst = 0.0;
  let denseA = densifiedCount(countA);
  let denseB = densifiedCount(countB);
  if (lane == 0u) {
    sharedDenseA = denseA;
    sharedDenseB = denseB;
  }
  workgroupBarrier();
  let uniformDenseA = workgroupUniformLoad(&sharedDenseA);
  let uniformDenseB = workgroupUniformLoad(&sharedDenseB);
  ${L(`A`,`B`)}
  ${L(`B`,`A`)}
  partials[lane] = worst;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) { partials[lane] = max(partials[lane], partials[lane + stride]); }
    workgroupBarrier();
  }
  if (lane == 0u && pair < PAIR_COUNT) {
    distances[distancesOffset + pair] = select(finishMetric(partials[0]), notANumber, skipped);
  }`})),u.maximumDistance&&h.push(l(e,{id:`${r}-max-distance`,operation:O,variant:`max-distance`,bindings:N(u.maximumDistance),workgroupSize:k,invocationCount:m*k,guardIndex:!1,declarations:`${D}
const WORKGROUP_SIZE: u32 = ${k}u;
var<workgroup> partials: array<f32, ${k}>;
${z}`,body:`${F(`(STATUS_EMPTY | STATUS_INVALID)`)}
  var farthest = 0.0;
  // Lanes stride over the longer track and scan the shorter one, so a short track next to a long
  // one does not leave most lanes idle. The maximum is order independent.
  if (countA >= countB) {
    for (var i = lane; i < countA; i += WORKGROUP_SIZE) {
      let point = loadA(startA + i);
      for (var j = 0u; j < countB; j++) {
        farthest = max(farthest, pointMetric(point, loadB(startB + j)));
      }
    }
  } else {
    for (var j = lane; j < countB; j += WORKGROUP_SIZE) {
      let point = loadB(startB + j);
      for (var i = 0u; i < countA; i++) {
        farthest = max(farthest, pointMetric(point, loadA(startA + i)));
      }
    }
  }
  partials[lane] = farthest;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) { partials[lane] = max(partials[lane], partials[lane + stride]); }
    workgroupBarrier();
  }
  if (lane == 0u && pair < PAIR_COUNT) {
    distances[distancesOffset + pair] = select(finishMetric(partials[0]), notANumber, skipped);
  }`})),u.frechet&&h.push(l(e,{id:`${r}-frechet`,operation:O,variant:`frechet`,bindings:N(u.frechet),workgroupSize:A,invocationCount:m*A,guardIndex:!1,declarations:`${D}
const WORKGROUP_SIZE: u32 = ${A}u;
const LANES: u32 = ${A}u;
var<workgroup> olderDiagonal: array<f32, ${A}>;
var<workgroup> previousDiagonal: array<f32, ${A}>;
var<workgroup> boundaryRow: array<f32, ${this.maximumFrechetVertexCount}>;
var<workgroup> sharedLongCount: u32;
var<workgroup> sharedShortCount: u32;
${z}
// Metric value between densified vertex rowLong of the longer track and rowShort of the other.
fn coupledSquared(swapped: bool, startA: u32, countA: u32, startB: u32, countB: u32, rowLong: u32, rowShort: u32) -> f32 {
  if (swapped) {
    return pointMetric(densifiedPointB(startB, countB, rowLong), densifiedPointA(startA, countA, rowShort));
  }
  return pointMetric(densifiedPointA(startA, countA, rowLong), densifiedPointB(startB, countB, rowShort));
}`,body:`${F(`(STATUS_EMPTY | STATUS_INVALID | STATUS_TOO_LONG)`)}
  let infinity = 3.402823466e+38;
  let denseA = densifiedCount(countA);
  let denseB = densifiedCount(countB);
  let swapped = denseB > denseA;
  if (lane == 0u) {
    sharedLongCount = max(denseA, denseB);
    sharedShortCount = min(denseA, denseB);
  }
  // Barrier-bearing loops need bounds the compiler can prove uniform across the workgroup.
  let longCount = workgroupUniformLoad(&sharedLongCount);
  let shortCount = workgroupUniformLoad(&sharedShortCount);
  var finalValue = infinity;
  let stripCount = (longCount + LANES - 1u) / LANES;
  for (var strip = 0u; strip < stripCount; strip++) {
    let firstRow = strip * LANES;
    let rows = min(LANES, longCount - firstRow);
    let diagonalCount = rows + shortCount - 1u;
    olderDiagonal[lane] = infinity;
    previousDiagonal[lane] = infinity;
    workgroupBarrier();
    for (var diagonal = 0u; diagonal < diagonalCount; diagonal++) {
      var value = infinity;
      var column = 0u;
      let isActive = lane < rows && diagonal >= lane && diagonal - lane < shortCount;
      if (isActive) {
        column = diagonal - lane;
        let cost = coupledSquared(swapped, startA, countA, startB, countB, firstRow + lane, column);
        let hasUp = lane > 0u || strip > 0u;
        let hasLeft = column > 0u;
        var up = infinity;
        var upLeft = infinity;
        var left = infinity;
        if (lane > 0u) {
          up = previousDiagonal[lane - 1u];
          upLeft = olderDiagonal[lane - 1u];
        } else if (strip > 0u) {
          up = boundaryRow[column];
          if (hasLeft) { upLeft = boundaryRow[column - 1u]; }
        }
        if (hasLeft) { left = previousDiagonal[lane]; }
        if (!hasUp && !hasLeft) {
          value = cost;
        } else if (!hasUp) {
          value = max(left, cost);
        } else if (!hasLeft) {
          value = max(up, cost);
        } else {
          value = max(min(min(up, upLeft), left), cost);
        }
        if (firstRow + lane == longCount - 1u && column == shortCount - 1u) { finalValue = value; }
      }
      workgroupBarrier();
      olderDiagonal[lane] = previousDiagonal[lane];
      previousDiagonal[lane] = value;
      if (isActive && lane == LANES - 1u) { boundaryRow[column] = value; }
      workgroupBarrier();
    }
  }
  if (!skipped && longCount > 0u && lane == (longCount - 1u) % LANES) {
    distances[distancesOffset + pair] = finishMetric(finalValue);
  } else if (skipped && lane == 0u && pair < PAIR_COUNT) {
    distances[distancesOffset + pair] = notANumber;
  }`})),u.dynamicTimeWarping&&h.push(l(e,{id:`${r}-dynamic-time-warping`,operation:O,variant:`dynamic-time-warping`,bindings:N(u.dynamicTimeWarping),workgroupSize:A,invocationCount:m*A,guardIndex:!1,declarations:`${D}
${_(A,this.maximumFrechetVertexCount,i.dynamicTimeWarpingBandRadius??4294967295)}
${z}`,body:v(F(`(STATUS_EMPTY | STATUS_INVALID | STATUS_TOO_LONG)`))}));for(let[t,n,a]of[[`edr`,u.editDistanceOnRealSequence,`edit-distance-on-real-sequence`],[`erp`,u.editDistanceWithRealPenalty,`edit-distance-with-real-penalty`]])n&&h.push(l(e,{id:`${r}-${a}`,operation:O,variant:a,bindings:N(n),workgroupSize:A,invocationCount:m*A,guardIndex:!1,declarations:`${D}
${C(t,A,this.maximumFrechetVertexCount,i.editDistanceTolerance??1,i.editDistanceGapPoint??[0,0])}
${z}`,body:w(F(`(STATUS_EMPTY | STATUS_INVALID | STATUS_TOO_LONG)`))}));if(u.symmetricSegmentPathDistance&&h.push(l(e,{id:`${r}-symmetric-segment-path`,operation:O,variant:`symmetric-segment-path`,bindings:N(u.symmetricSegmentPathDistance),workgroupSize:k,invocationCount:m*k,guardIndex:!1,declarations:`${D}
${T(k)}
${z}`,body:E(F(`(STATUS_EMPTY | STATUS_INVALID)`),k)})),u.longestCommonSubsequence&&h.push(l(e,{id:`${r}-longest-common-subsequence`,operation:O,variant:`longest-common-subsequence`,bindings:N(u.longestCommonSubsequence),workgroupSize:A,invocationCount:m*A,guardIndex:!1,declarations:`${D}
${_(A,this.maximumFrechetVertexCount,i.lcssBandRadius??4294967295)}
${z}`,body:x(F(`(STATUS_EMPTY | STATUS_INVALID | STATUS_TOO_LONG)`),i.lcssTolerance??1)})),u.frechetWithin&&i.frechetEpsilon){let t=s(e,`${r}-pair-ranges`,`uint32`,m*4);h.push(l(e,{id:`${r}-pair-ranges`,operation:O,variant:`pair-ranges`,bindings:[{name:`pairA`,view:i.pairA,type:`u32`,access:`read`},{name:`pairB`,view:i.pairB,type:`u32`,access:`read`},{name:`offsetsA`,view:a.lineOffsets,type:`u32`,access:`read`},{name:`offsetsB`,view:p,type:`u32`,access:`read`},{name:`status`,view:j,type:`u32`,access:`read`},{name:`pairRanges`,view:t,type:`u32`,access:`read_write`}],invocationCount:m,declarations:D,body:`var range = vec4<u32>(0u);
  if (status[statusOffset + index] == 0u) {
    let a = pairA[pairAOffset + index];
    let b = pairB[pairBOffset + index];
    let startA = offsetsA[offsetsAOffset + a];
    let startB = offsetsB[offsetsBOffset + b];
    range = vec4<u32>(startA, offsetsA[offsetsAOffset + a + 1u] - startA,
      startB, offsetsB[offsetsBOffset + b + 1u] - startB);
  }
  pairRanges[pairRangesOffset + index * 4u] = range.x;
  pairRanges[pairRangesOffset + index * 4u + 1u] = range.y;
  pairRanges[pairRangesOffset + index * 4u + 2u] = range.z;
  pairRanges[pairRangesOffset + index * 4u + 3u] = range.w;`}),l(e,{id:`${r}-frechet-decision`,operation:O,variant:`frechet-decision`,bindings:[{name:`pairRanges`,view:t,type:`u32`,access:`read`},{name:`positionsA`,view:a.positions,type:`f32`,access:`read`},{name:`positionsB`,view:d,type:`f32`,access:`read`},{name:`status`,view:j,type:`u32`,access:`read`},{name:`frechetEpsilon`,view:i.frechetEpsilon,type:`f32`,access:`read`},{name:`frechetWithin`,view:u.frechetWithin,type:`u32`,access:`read_write`}],workgroupSize:A,invocationCount:m*A,guardIndex:!1,declarations:`${D}
${y(A,this.maximumFrechetVertexCount,this.sphereRadius!==void 0)}
${z}`,body:b(`let pair = index / WORKGROUP_SIZE;
  let lane = localInvocationIndex;
  let flags = status[statusOffset + min(pair, PAIR_COUNT - 1u)];
  let skipped = pair >= PAIR_COUNT ||
    (flags & (STATUS_EMPTY | STATUS_INVALID | STATUS_TOO_LONG)) != 0u;
  let rangeBase = pairRangesOffset + min(pair, PAIR_COUNT - 1u) * 4u;
  let startA = pairRanges[rangeBase];
  let countA = select(pairRanges[rangeBase + 1u], 0u, skipped);
  let startB = pairRanges[rangeBase + 2u];
  let countB = select(pairRanges[rangeBase + 3u], 0u, skipped);`)}))}return h}};export{z as t};