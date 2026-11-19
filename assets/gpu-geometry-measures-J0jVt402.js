import{r as e}from"./kernel-C3tfQOn4.js";import{t}from"./gpu-sort-xRm8l5yn.js";import{E as n,T as r,f as i,w as a,y as o}from"./gpu-contributor-utils-BIwLQON5.js";import{c as s,o as c,t as l}from"./wgsl-kernel-nodes-DagJklH2.js";import{i as u}from"./ports-xmdKgL07.js";import{c as d,d as f,f as p,g as m,h,p as g,u as _,v}from"./spatial-context-DwZEd4EZ.js";import{a as y}from"./sorted-segment-sums-Dvmm6OdR.js";import{t as b}from"./sorted-segment-offsets-ChmOtA9F.js";import{t as x}from"./common-CshkX-d_.js";import{t as S}from"./lon-lat-ring-wgsl-DpZBIfuE.js";import{t as C}from"./addition-rounding-error-wgsl-DVwPAqKC.js";import{t as w}from"./geodesic-wgsl-I0CYDKcz.js";import{a as T,i as E}from"./geodesic-kernels-DpMzdskM.js";import{a as D}from"./graph-chunk-dispatch-zBt8k2-o.js";import{a as O,t as k}from"./geometry-storage-validation-T89ZhmCj.js";function A(e,t){let n=`return vec2<u32>(feature, feature + 1u);`;e.hasPolygonOffsets?n=`let polygonBegin = min(featureOffsets[featureOffsetsOffset + feature], POLYGON_COUNT);
  let polygonEnd = min(max(featureOffsets[featureOffsetsOffset + feature + 1u], polygonBegin), POLYGON_COUNT);
  let ringBegin = min(polygonOffsets[polygonOffsetsOffset + polygonBegin], RING_COUNT);
  let ringEnd = min(max(polygonOffsets[polygonOffsetsOffset + polygonEnd], ringBegin), RING_COUNT);
  return vec2<u32>(ringBegin, ringEnd);`:e.hasFeatureOffsets&&(n=`let ringBegin = min(featureOffsets[featureOffsetsOffset + feature], RING_COUNT);
  let ringEnd = min(max(featureOffsets[featureOffsetsOffset + feature + 1u], ringBegin), RING_COUNT);
  return vec2<u32>(ringBegin, ringEnd);`);let r=e.hasRingOffsets?`let rowBegin = min(ringOffsets[ringOffsetsOffset + ring], ROW_COUNT);
  let rowEnd = min(max(ringOffsets[ringOffsetsOffset + ring + 1u], rowBegin), ROW_COUNT);
  return vec2<u32>(rowBegin, rowEnd - rowBegin);`:`return vec2<u32>(min(ring, ROW_COUNT), select(0u, 1u, ring < ROW_COUNT));`;return`
const COOPERATIVE_RING_ROWS: u32 = ${t}u;

// Ring range (begin, end) of a feature.
fn getFeatureRingRange(feature: u32) -> vec2<u32> {
  ${n}
}

// First row and row count of a ring.
fn getRingRows(ring: u32) -> vec2<u32> {
  ${r}
}

// A feature with a ring above the threshold is measured by the cooperative kernel.
fn isCooperativeFeature(feature: u32) -> bool {
  let range = getFeatureRingRange(feature);
  for (var ring = range.x; ring < range.y; ring++) {
    if (getRingRows(ring).y > COOPERATIVE_RING_ROWS) {
      return true;
    }
  }
  return false;
}
`}var j=`
  let lane = localInvocationIndex;
  if (lane == 0u) {
    sharedGroup = index / COOPERATIVE_LANES;
  }
  let group = workgroupUniformLoad(&sharedGroup);
  // Workgroups stride over batches of 64 features; each batch flags its cooperative features, and
  // the whole workgroup measures them one after another.
  for (var base = group * COOPERATIVE_LANES; base < FEATURE_COUNT; base += COOPERATIVE_GROUPS * COOPERATIVE_LANES) {
    let candidate = base + lane;
    sharedFlags[lane] = select(0u, 1u, candidate < FEATURE_COUNT && isCooperativeFeature(candidate));
    workgroupBarrier();
    for (var slot = 0u; slot < COOPERATIVE_LANES; slot++) {
      if (workgroupUniformLoad(&sharedFlags[slot]) == 1u) {
        measureFeatureCooperatively(base + slot, lane);
      }
    }
    workgroupBarrier();
  }`;function M(e){return`
const COOPERATIVE_LANES: u32 = 64u;
const COOPERATIVE_GROUPS: u32 = ${e.groupCount}u;
const FEATURE_COUNT: u32 = ${e.featureCount}u;

var<workgroup> sharedGroup: u32;
var<workgroup> sharedFlags: array<u32, 64>;
var<workgroup> sharedRing: u32;
var<workgroup> sharedRingEnd: u32;
var<workgroup> sharedOrigin: vec2<f32>;
var<workgroup> laneScan: array<f32, 64>;
var<workgroup> laneLengthSum: array<vec2<f32>, 64>;
var<workgroup> laneLineMomentX: array<vec2<f32>, 64>;
var<workgroup> laneLineMomentY: array<vec2<f32>, 64>;
var<workgroup> laneArea: array<vec2<f32>, 64>;
var<workgroup> laneStraightArea: array<vec2<f32>, 64>;
var<workgroup> laneMomentX: array<vec2<f32>, 64>;
var<workgroup> laneMomentY: array<vec2<f32>, 64>;
var<workgroup> laneVertexSum: array<vec2<f32>, 64>;
var<workgroup> laneBoundsMin: array<vec2<f32>, 64>;
var<workgroup> laneBoundsMax: array<vec2<f32>, 64>;
var<workgroup> laneExtremeRows: array<vec4<u32>, 64>;
var<workgroup> laneWinding: array<f32, 64>;
var<workgroup> laneAmbiguousEdges: array<u32, 64>;

// Sums of one ring (or one lane's chunk of it): each is a Neumaier (sum, compensation) pair.
struct RingAccumulators {
  lengthSum: vec2<f32>,
  lineMomentX: vec2<f32>,
  lineMomentY: vec2<f32>,
  area: vec2<f32>,
  straightArea: vec2<f32>,
  momentX: vec2<f32>,
  momentY: vec2<f32>,
  vertexSum: vec2<f32>,
  boundsMin: vec2<f32>,
  boundsMax: vec2<f32>,
  extremeRows: vec4<u32>,
  winding: f32,
  ambiguousEdges: u32
}

struct FeatureAccumulators {
  lengthSum: vec2<f32>,
  areaSum: vec2<f32>,
  momentX: vec2<f32>,
  momentY: vec2<f32>,
  lineMomentX: vec2<f32>,
  lineMomentY: vec2<f32>,
  straightSum: vec2<f32>,
  vertexSum: vec2<f32>,
  boundsMin: vec2<f32>,
  boundsMax: vec2<f32>,
  extremeRows: vec4<u32>,
  vertexCount: u32,
  isFirstRing: bool,
  invalidRings: u32
}

fn initRing() -> RingAccumulators {
  return RingAccumulators(
    vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0),
    vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(3.4e38), vec2<f32>(-3.4e38),
    vec4<u32>(0xffffffffu), 0.0, 0u
  );
}

fn initFeature() -> FeatureAccumulators {
  return FeatureAccumulators(
    vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0),
    vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(3.4e38), vec2<f32>(-3.4e38),
    vec4<u32>(0xffffffffu), 0u, true, 0u
  );
}

// The serial kernel's per-vertex loop over vertices [vertexStart, vertexEnd) of a ring: bounds and extremes,
// vertex sum, and per edge the length, line moments, ring area and area moments. startLocalX is
// the unwrapped local longitude of vertex vertexStart (geographic coordinates, not point sets).
fn accumulateRange(
  rowBegin: u32, rowCount: u32, vertexStart: u32, vertexEnd: u32, startLocalX: f32, origin: vec2<f32>,
  acc: ptr<function, RingAccumulators>
) {
  if (vertexStart >= vertexEnd) {
    return;
  }
  let firstPosition = getPosition(rowBegin);
  var firstLocal = firstPosition - origin;
  if (IS_GEOGRAPHIC) {
    firstLocal.x = geodesicWrapLongitudeDelta(firstPosition.x - origin.x);
  }
  var position = getPosition(rowBegin + vertexStart);
  var local = position - origin;
  if (IS_GEOGRAPHIC) {
    local.x = select(startLocalX, geodesicWrapLongitudeDelta(position.x - origin.x), IS_POINT);
  }
  var areaX = local.x;
  if (IS_GEOGRAPHIC) {
    areaX = local.x * GEODESIC_DEGREES_TO_RADIANS;
  }
  var areaY = getAreaY(position, origin);
  let edgeCount = select(rowCount - 1u, rowCount, IS_POLYGON);
  for (var vertex = vertexStart; vertex < vertexEnd; vertex++) {
    // Strict comparisons keep the lowest vertex row on ties.
    if (local.x < (*acc).boundsMin.x) {
      (*acc).extremeRows.x = rowBegin + vertex;
    }
    if (local.y < (*acc).boundsMin.y) {
      (*acc).extremeRows.y = rowBegin + vertex;
    }
    if (local.x > (*acc).boundsMax.x) {
      (*acc).extremeRows.z = rowBegin + vertex;
    }
    if (local.y > (*acc).boundsMax.y) {
      (*acc).extremeRows.w = rowBegin + vertex;
    }
    (*acc).boundsMin = min((*acc).boundsMin, local);
    (*acc).boundsMax = max((*acc).boundsMax, local);
    (*acc).vertexSum += local;
    if (IS_POINT) {
      if (vertex + 1u < vertexEnd) {
        let nextPoint = getPosition(rowBegin + vertex + 1u);
        local = nextPoint - origin;
        if (IS_GEOGRAPHIC) {
          local.x = geodesicWrapLongitudeDelta(nextPoint.x - origin.x);
        }
      }
      continue;
    }
    if (vertex >= edgeCount) {
      break;
    }
    let nextRow = select(rowBegin + vertex + 1u, rowBegin, vertex + 1u == rowCount);
    let nextPosition = getPosition(nextRow);
    var nextLocal = nextPosition - origin;
    if (IS_GEOGRAPHIC) {
      nextLocal = vec2<f32>(
        local.x + lonLatRingStep(position, nextPosition),
        nextPosition.y - origin.y
      );
    }
    if (vertex + 1u == rowCount) {
      nextLocal = firstLocal;
    }
    var nextAreaX = nextLocal.x;
    if (IS_GEOGRAPHIC) {
      nextAreaX = nextLocal.x * GEODESIC_DEGREES_TO_RADIANS;
    }
    let nextAreaY = getAreaY(nextPosition, origin);
    if (IS_POLYGON && IS_GEOGRAPHIC) {
      (*acc).winding += lonLatRingStep(position, nextPosition);
    }
    if (IS_GEOGRAPHIC) { (*acc).ambiguousEdges += select(0u, 1u, isAmbiguousEdge(position, nextPosition)); }
    let edgeLength = getEdgeLength(position, nextPosition);
    addCompensated(&(*acc).lengthSum, edgeLength);
    addCompensated(&(*acc).lineMomentX, edgeLength * 0.5 * (local.x + nextLocal.x));
    addCompensated(&(*acc).lineMomentY, edgeLength * 0.5 * (local.y + nextLocal.y));
    let cross = areaX * nextAreaY - nextAreaX * areaY;
    ${e.edgeCrossSource}
    if (IS_GEOGRAPHIC && (bitcast<u32>(edgeCross) & 0x7f800000u) == 0x7f800000u) { (*acc).ambiguousEdges += 1u; }
    addCompensated(&(*acc).area, edgeCross);
    addCompensated(&(*acc).straightArea, cross);
    addCompensated(&(*acc).momentX, (areaX + nextAreaX) * cross);
    addCompensated(&(*acc).momentY, (areaY + nextAreaY) * cross);
    position = nextPosition;
    local = nextLocal;
    areaX = nextAreaX;
    areaY = nextAreaY;
  }
}

// Folds a finished ring into the feature (the serial kernel's end-of-ring step).
fn mergeRing(feature: ptr<function, FeatureAccumulators>, ring: ptr<function, RingAccumulators>, ringIndex: u32, rowCount: u32) {
  (*feature).vertexCount += rowCount;
  if (IS_GEOGRAPHIC && ((*ring).ambiguousEdges > 0u || (IS_POLYGON && (lonLatIsPoleWinding((*ring).winding) || (*ring).boundsMax.x - (*ring).boundsMin.x > 360.0)))) {
    (*feature).invalidRings += 1u;
  }
  addCompensated(&(*feature).lengthSum, (*ring).lengthSum.x);
  addCompensated(&(*feature).lengthSum, (*ring).lengthSum.y);
  addCompensated(&(*feature).lineMomentX, (*ring).lineMomentX.x);
  addCompensated(&(*feature).lineMomentX, (*ring).lineMomentX.y);
  addCompensated(&(*feature).lineMomentY, (*ring).lineMomentY.x);
  addCompensated(&(*feature).lineMomentY, (*ring).lineMomentY.y);
  (*feature).vertexSum += (*ring).vertexSum;
  if ((*ring).boundsMin.x < (*feature).boundsMin.x) {
    (*feature).extremeRows.x = (*ring).extremeRows.x;
  }
  if ((*ring).boundsMin.y < (*feature).boundsMin.y) {
    (*feature).extremeRows.y = (*ring).extremeRows.y;
  }
  if ((*ring).boundsMax.x > (*feature).boundsMax.x) {
    (*feature).extremeRows.z = (*ring).extremeRows.z;
  }
  if ((*ring).boundsMax.y > (*feature).boundsMax.y) {
    (*feature).extremeRows.w = (*ring).extremeRows.w;
  }
  (*feature).boundsMin = min((*feature).boundsMin, (*ring).boundsMin);
  (*feature).boundsMax = max((*feature).boundsMax, (*ring).boundsMax);
  var ringFactor = 1.0;
  if (FIRST_RING_EXTERIOR) {
    let ringSign = select(-1.0, 1.0, getCompensated((*ring).area) >= 0.0);
    ringFactor = select(-1.0, 1.0, isShellRing(ringIndex, (*feature).isFirstRing)) * ringSign;
  }
  addCompensated(&(*feature).areaSum, ringFactor * getCompensated((*ring).area));
  addCompensated(&(*feature).straightSum, ringFactor * getCompensated((*ring).straightArea));
  addCompensated(&(*feature).momentX, ringFactor * getCompensated((*ring).momentX));
  addCompensated(&(*feature).momentY, ringFactor * getCompensated((*ring).momentY));
  (*feature).isFirstRing = false;
}

// Adds lane laneIndex's published partial to a ring accumulator.
fn addLanePartial(ring: ptr<function, RingAccumulators>, laneIndex: u32) {
  addCompensated(&(*ring).lengthSum, laneLengthSum[laneIndex].x);
  addCompensated(&(*ring).lengthSum, laneLengthSum[laneIndex].y);
  addCompensated(&(*ring).lineMomentX, laneLineMomentX[laneIndex].x);
  addCompensated(&(*ring).lineMomentX, laneLineMomentX[laneIndex].y);
  addCompensated(&(*ring).lineMomentY, laneLineMomentY[laneIndex].x);
  addCompensated(&(*ring).lineMomentY, laneLineMomentY[laneIndex].y);
  addCompensated(&(*ring).area, laneArea[laneIndex].x);
  addCompensated(&(*ring).area, laneArea[laneIndex].y);
  addCompensated(&(*ring).straightArea, laneStraightArea[laneIndex].x);
  addCompensated(&(*ring).straightArea, laneStraightArea[laneIndex].y);
  addCompensated(&(*ring).momentX, laneMomentX[laneIndex].x);
  addCompensated(&(*ring).momentX, laneMomentX[laneIndex].y);
  addCompensated(&(*ring).momentY, laneMomentY[laneIndex].x);
  addCompensated(&(*ring).momentY, laneMomentY[laneIndex].y);
  (*ring).vertexSum += laneVertexSum[laneIndex];
  (*ring).winding += laneWinding[laneIndex];
  (*ring).ambiguousEdges += laneAmbiguousEdges[laneIndex];
  // Lanes own ascending chunks, so a strict comparison keeps the lowest row on ties.
  if (laneBoundsMin[laneIndex].x < (*ring).boundsMin.x) {
    (*ring).extremeRows.x = laneExtremeRows[laneIndex].x;
  }
  if (laneBoundsMin[laneIndex].y < (*ring).boundsMin.y) {
    (*ring).extremeRows.y = laneExtremeRows[laneIndex].y;
  }
  if (laneBoundsMax[laneIndex].x > (*ring).boundsMax.x) {
    (*ring).extremeRows.z = laneExtremeRows[laneIndex].z;
  }
  if (laneBoundsMax[laneIndex].y > (*ring).boundsMax.y) {
    (*ring).extremeRows.w = laneExtremeRows[laneIndex].w;
  }
  (*ring).boundsMin = min((*ring).boundsMin, laneBoundsMin[laneIndex]);
  (*ring).boundsMax = max((*ring).boundsMax, laneBoundsMax[laneIndex]);
}

fn writeFeatureStats(index: u32, origin: vec2<f32>, acc: FeatureAccumulators) {
${e.tail}
}

// Wrapped longitude step into vertex k of a ring (vertex 0 steps from the feature origin).
fn getLongitudeStep(rowBegin: u32, vertex: u32, origin: vec2<f32>) -> f32 {
  let current = getPosition(rowBegin + vertex);
  if (vertex == 0u) {
    return geodesicWrapLongitudeDelta(current.x - origin.x);
  }
  return lonLatRingStep(getPosition(rowBegin + vertex - 1u), current);
}

// Measures one cooperative feature with the whole workgroup. Called from uniform control flow.
fn measureFeatureCooperatively(feature: u32, lane: u32) {
  let range = getFeatureRingRange(feature);
  if (lane == 0u) {
    for (var ring = range.x; ring < range.y; ring++) {
      let rows = getRingRows(ring);
      if (rows.y > 0u) {
        sharedOrigin = getPosition(rows.x);
        break;
      }
    }
    sharedRingEnd = range.y;
  }
  let ringEnd = workgroupUniformLoad(&sharedRingEnd);
  let origin = workgroupUniformLoad(&sharedOrigin);
  var accumulated = initFeature();
  var ring = range.x;
  loop {
    if (lane == 0u) {
      // Small rings are lane 0's: no barriers until the next large ring.
      while (ring < ringEnd) {
        let rows = getRingRows(ring);
        if (rows.y > COOPERATIVE_RING_ROWS) {
          break;
        }
        if (rows.y > 0u) {
          var smallRing = initRing();
          let firstPosition = getPosition(rows.x);
          accumulateRange(
            rows.x, rows.y, 0u, rows.y,
            geodesicWrapLongitudeDelta(firstPosition.x - origin.x), origin, &smallRing
          );
          mergeRing(&accumulated, &smallRing, ring, rows.y);
        }
        ring++;
      }
      sharedRing = ring;
    }
    let current = workgroupUniformLoad(&sharedRing);
    if (current >= ringEnd) {
      break;
    }
    let rows = getRingRows(current);
    let chunk = (rows.y + COOPERATIVE_LANES - 1u) / COOPERATIVE_LANES;
    let vertexStart = min(lane * chunk, rows.y);
    let vertexEnd = min(vertexStart + chunk, rows.y);
    var startLocalX = 0.0;
    if (IS_GEOGRAPHIC && !IS_POINT) {
      // Unwrapped longitude of the chunk's first vertex: exclusive scan of the chunks' sums of
      // wrapped longitude steps (Hillis-Steele in workgroup memory).
      var stepSum = vec2<f32>(0.0);
      for (var vertex = vertexStart; vertex < vertexEnd; vertex++) {
        addCompensated(&stepSum, getLongitudeStep(rows.x, vertex, origin));
      }
      laneScan[lane] = stepSum.x + stepSum.y;
      workgroupBarrier();
      for (var offset = 1u; offset < COOPERATIVE_LANES; offset = offset * 2u) {
        var value = laneScan[lane];
        if (lane >= offset) {
          value = value + laneScan[lane - offset];
        }
        workgroupBarrier();
        laneScan[lane] = value;
        workgroupBarrier();
      }
      var prefix = 0.0;
      if (lane > 0u) {
        prefix = laneScan[lane - 1u];
      }
      if (vertexStart < vertexEnd) {
        startLocalX = prefix + getLongitudeStep(rows.x, vertexStart, origin);
      }
    }
    var partial = initRing();
    accumulateRange(rows.x, rows.y, vertexStart, vertexEnd, startLocalX, origin, &partial);
    laneLengthSum[lane] = partial.lengthSum;
    laneLineMomentX[lane] = partial.lineMomentX;
    laneLineMomentY[lane] = partial.lineMomentY;
    laneArea[lane] = partial.area;
    laneStraightArea[lane] = partial.straightArea;
    laneMomentX[lane] = partial.momentX;
    laneMomentY[lane] = partial.momentY;
    laneVertexSum[lane] = partial.vertexSum;
    laneBoundsMin[lane] = partial.boundsMin;
    laneBoundsMax[lane] = partial.boundsMax;
    laneExtremeRows[lane] = partial.extremeRows;
    laneWinding[lane] = partial.winding;
    laneAmbiguousEdges[lane] = partial.ambiguousEdges;
    workgroupBarrier();
    if (lane == 0u) {
      var combined = initRing();
      for (var laneIndex = 0u; laneIndex < COOPERATIVE_LANES; laneIndex++) {
        addLanePartial(&combined, laneIndex);
      }
      mergeRing(&accumulated, &combined, current, rows.y);
      ring = current + 1u;
    }
  }
  if (lane == 0u) {
    writeFeatureStats(feature, origin, accumulated);
  }
}
`}function N(e=d){let{flattening:t}=e,n=t*(2-t),r=Math.sqrt(n),i=r===0?2:1+(1-n)/(2*r)*Math.log((1+r)/(1-r));return{eccentricitySquared:n,authalicRadius:e.semiMajorAxis*Math.sqrt(i/2),sinAuthalicScale:2*(1-n)/i,polarQ:i}}function P(e,t,n,r,i){let a=N(r),o=Math.sqrt(a.eccentricitySquared),c=e===`spherical`?n:a.authalicRadius;return`
const LENGTH_SCALE: f32 = ${s(i)};
const RADIUS: f32 = ${s(t)};
const AREA_RADIUS: f32 = ${s(c)};
const WGS84_A: f32 = ${s(r.semiMajorAxis)};
const WGS84_F: f32 = ${s(r.flattening)};
const WGS84_E2: f32 = ${s(a.eccentricitySquared)};
const WGS84_E: f32 = ${s(o)};
const SIN_AUTHALIC_SCALE: f32 = ${s(a.sinAuthalicScale)};
const POLAR_Q: f32 = ${s(a.polarQ)};
`}var F=`
fn getWgs84Length(a: vec2<f32>, b: vec2<f32>) -> f32 {
  let phi1 = a.y * GEODESIC_DEGREES_TO_RADIANS;
  let phi2 = b.y * GEODESIC_DEGREES_TO_RADIANS;
  let deltaPhi = (b.y - a.y) * GEODESIC_DEGREES_TO_RADIANS;
  let deltaLambda = geodesicWrapLongitudeDelta(b.x - a.x) * GEODESIC_DEGREES_TO_RADIANS;
  let beta1 = atan2((1.0 - WGS84_F) * sin(phi1), cos(phi1));
  let beta2 = atan2((1.0 - WGS84_F) * sin(phi2), cos(phi2));
  // beta2 - beta1 without subtracting two rounded reduced latitudes: the midpoint derivative
  // dβ/dφ for short spans (error below 1e-10 rad), else deltaPhi minus the difference of the small
  // closed-form gaps phi - beta.
  var halfDifference = 0.5 * (deltaPhi - (getReducedLatitudeGap(phi2) - getReducedLatitudeGap(phi1)));
  if (abs(deltaPhi) < 0.01) {
    let sinMid = sin(0.5 * (phi1 + phi2));
    let cosMid = cos(0.5 * (phi1 + phi2));
    halfDifference = 0.5 * deltaPhi * (1.0 - WGS84_F) /
      (cosMid * cosMid + (1.0 - WGS84_F) * (1.0 - WGS84_F) * sinMid * sinMid);
  }
  let halfSum = 0.5 * (beta1 + beta2);
  let sinHalfDifference = sin(halfDifference);
  let sinHalfLambda = sin(0.5 * deltaLambda);
  let haversine = clamp(
    sinHalfDifference * sinHalfDifference + cos(beta1) * cos(beta2) * sinHalfLambda * sinHalfLambda,
    0.0,
    1.0
  );
  if (haversine <= 0.0) {
    return 0.0;
  }
  let sigma = 2.0 * atan2(sqrt(haversine), sqrt(1.0 - haversine));
  let sinSigma = sin(sigma);
  let sinP = sin(halfSum);
  let cosP = cos(halfSum);
  let cosQ = cos(halfDifference);
  let x = (sigma - sinSigma) * sinP * sinP * cosQ * cosQ / max(1.0 - haversine, 1e-12);
  let y = (sigma + sinSigma) * cosP * cosP * sinHalfDifference * sinHalfDifference / haversine;
  return WGS84_A * (sigma - 0.5 * WGS84_F * (x + y));
}

fn getAuthalicIntegrand(s: f32) -> f32 {
  let denominator = 1.0 - WGS84_E2 * s * s;
  return 1.0 / (denominator * denominator);
}

fn getAuthalicSinDelta(sinOrigin: f32, sinDelta: f32) -> f32 {
  let step = sinDelta / 8.0;
  var sum = getAuthalicIntegrand(sinOrigin) + getAuthalicIntegrand(sinOrigin + sinDelta);
  for (var interval = 1u; interval < 8u; interval++) {
    let weight = select(2.0, 4.0, (interval & 1u) == 1u);
    sum = sum + weight * getAuthalicIntegrand(sinOrigin + f32(interval) * step);
  }
  return SIN_AUTHALIC_SCALE * sum * step / 3.0;
}

fn getSinAuthalic(s: f32) -> f32 {
  if (WGS84_E == 0.0) {
    return s;
  }
  let es = WGS84_E * s;
  let es2 = es * es;
  // atanh(es) / e. For a small eccentricity the f32 quotient (1 + es) / (1 - es) rounds next to 1
  // and its logarithm keeps six digits (1.6 m of latitude on WGS84); the series keeps them all.
  var inverseHyperbolic = log((1.0 + es) / (1.0 - es)) / (2.0 * WGS84_E);
  if (es2 < 0.01) {
    inverseHyperbolic = s * (1.0 + es2 * (1.0 / 3.0 + es2 * (0.2 + es2 * (1.0 / 7.0 + es2 / 9.0))));
  }
  let q = (1.0 - WGS84_E2) * (s / (1.0 - es2) + inverseHyperbolic);
  return q / POLAR_Q;
}

// Snyder (1987) equation 3-18, the inverse series. (17 e^4 / 360 belongs to the forward series
// 3-14; using it here misplaces the latitude by up to 4.8 m on WGS84.)
fn getGeodeticFromAuthalic(xi: f32) -> f32 {
  let e4 = WGS84_E2 * WGS84_E2;
  let e6 = e4 * WGS84_E2;
  return xi +
    (WGS84_E2 / 3.0 + 31.0 * e4 / 180.0 + 517.0 * e6 / 5040.0) * sin(2.0 * xi) +
    (23.0 * e4 / 360.0 + 251.0 * e6 / 3780.0) * sin(4.0 * xi) +
    (761.0 * e6 / 45360.0) * sin(6.0 * xi);
}
`,I=`
const GEODESIC_AREA_MIN_LENGTH: f32 = 20000.0;
const GEODESIC_AREA_SEGMENT_LENGTH: f32 = 250000.0;
const GEODESIC_AREA_MAX_SEGMENTS: f32 = 16.0;

fn getGeodesicAreaPoint(
  start: vec2<f32>,
  startX: f32,
  bearing: f32,
  distance: f32,
  origin: vec2<f32>
) -> vec2<f32> {
  let direct = vincentyDirect(start, bearing, distance);
  if (direct.hasLocalDelta) {
    // Keep the small displacement out of global f32 coordinates. Rounding a
    // 51-degree midpoint first quantizes its latitude by about 0.4 m, which
    // alone can exceed a city polygon's area tolerance. Add sine differences
    // rather than latitude differences, so fast math cannot reassociate this
    // into the same lossy global-coordinate subtraction.
    let sineChange = geodesicSinLatitudeDelta(start.y, origin.y) + direct.sinLatitudeDelta;
    return vec2<f32>(
      startX + direct.longitudeDelta * GEODESIC_DEGREES_TO_RADIANS,
      getAuthalicSinDelta(sin(origin.y * GEODESIC_DEGREES_TO_RADIANS), sineChange)
    );
  }
  let point = direct.destination;
  return vec2<f32>(
    startX + (point.x - start.x) * GEODESIC_DEGREES_TO_RADIANS,
    getAreaY(point, origin)
  );
}

fn getGeodesicEdgeCross(
  position: vec2<f32>,
  areaX: f32,
  areaY: f32,
  nextPosition: vec2<f32>,
  nextAreaX: f32,
  nextAreaY: f32,
  origin: vec2<f32>
) -> f32 {
  let inverse = vincentyInverse(position, nextPosition);
  if (!inverse.converged) { return bitcast<f32>(0x7fc00000u | (bitcast<u32>(position.x) & 0u)); }
  if (inverse.distance < GEODESIC_AREA_MIN_LENGTH) {
    return areaX * nextAreaY - nextAreaX * areaY;
  }
  let segments = clamp(ceil(inverse.distance / GEODESIC_AREA_SEGMENT_LENGTH), 1.0, GEODESIC_AREA_MAX_SEGMENTS);
  var total = 0.0;
  var startX = areaX;
  var startY = areaY;
  for (var segment = 0u; segment < u32(segments); segment++) {
    var endX = nextAreaX;
    var endY = nextAreaY;
    if (f32(segment + 1u) < segments) {
      let end = getGeodesicAreaPoint(
        position, areaX, inverse.initialBearing, inverse.distance * f32(segment + 1u) / segments, origin
      );
      endX = end.x;
      endY = end.y;
    }
    let middle = getGeodesicAreaPoint(
      position, areaX, inverse.initialBearing, inverse.distance * (f32(segment) + 0.5) / segments, origin
    );
    let bulge = (middle.x - startX) * (endY - startY) - (endX - startX) * (middle.y - startY);
    total += startX * endY - endX * startY + (4.0 / 3.0) * bulge;
    startX = endX;
    startY = endY;
  }
  return total;
}
`,L=`
fn getGreatCircleEdgeBulge(a: vec2<f32>, b: vec2<f32>, stepDegrees: f32) -> f32 {
  let deltaLambda = stepDegrees * GEODESIC_DEGREES_TO_RADIANS;
  let halfDeltaPhiDegrees = 0.5 * (b.y - a.y);
  let middleDegrees = 0.5 * (a.y + b.y);
  let cosHalfDeltaPhi = max(cos(halfDeltaPhiDegrees * GEODESIC_DEGREES_TO_RADIANS), 1e-20);
  let sinHalfDeltaPhi = sin(halfDeltaPhiDegrees * GEODESIC_DEGREES_TO_RADIANS);
  let sinMiddle = sin(middleDegrees * GEODESIC_DEGREES_TO_RADIANS);
  // k is at most 1 in magnitude; the bulge is odd in k, so work with |k| and restore the sign.
  let ratio = abs(sinMiddle) / cosHalfDeltaPhi;
  // 1 - k = (sin(90 - |half difference|) - sin(|middle|)) / cos(half difference), as a product:
  // 90 - |half difference| - |middle| is the colatitude of the vertex nearer a pole, exact in f32.
  let colatitudeDegrees = 90.0 - max(abs(a.y), abs(b.y));
  let complement = 2.0 *
    cos(0.5 * (90.0 - abs(halfDeltaPhiDegrees) + abs(middleDegrees)) * GEODESIC_DEGREES_TO_RADIANS) *
    sin(0.5 * colatitudeDegrees * GEODESIC_DEGREES_TO_RADIANS) / cosHalfDeltaPhi;
  let sinHalfLambda = sin(0.5 * deltaLambda);
  let cosHalfLambda = abs(cos(0.5 * deltaLambda));
  var excess = 0.0;
  if (abs(sinHalfLambda) < 0.25 * cosHalfLambda) {
    let tangent = sinHalfLambda / cosHalfLambda;
    let tangentSquared = tangent * tangent;
    let ratioSquared = ratio * ratio;
    var partialSum = 0.0;
    var ratioPower = 1.0;
    var tangentPower = tangent;
    var termSign = 1.0;
    var series = 0.0;
    for (var order = 1u; order <= 6u; order++) {
      partialSum += ratioPower;
      ratioPower *= ratioSquared;
      tangentPower *= tangentSquared;
      series += termSign * partialSum * tangentPower / f32(2u * order + 1u);
      termSign = -termSign;
    }
    excess = 2.0 * ratio * complement * (1.0 + ratio) * series;
  } else {
    // The second argument is positive (a floor keeps atan2(0, 0) out of a 180 degree step,
    // which the ring check already reports as ambiguous).
    excess = complement * deltaLambda - 2.0 * atan2(
      complement * sinHalfLambda * cosHalfLambda,
      max(cosHalfLambda * cosHalfLambda + ratio * sinHalfLambda * sinHalfLambda, 1e-30)
    );
  }
  let bulge = excess + ratio * deltaLambda * sinHalfDeltaPhi * sinHalfDeltaPhi;
  return select(bulge, -bulge, sinMiddle < 0.0);
}
`;function R(e,t,n){return I.replace(`fn getGeodesicEdgeCross(`,`fn integrateGeodesicEdgeCross(`).replace(`  origin: vec2<f32>
) -> f32 {`,`  origin: vec2<f32>,
  segments: f32
) -> f32 {`).replace(/  if \(inverse.distance < GEODESIC_AREA_MIN_LENGTH\) \{[\s\S]*?\n  let segments =[^;]+;/,``)+`
fn getGeodesicEdgeCross(position: vec2f, areaX: f32, areaY: f32, nextPosition: vec2f, nextAreaX: f32, nextAreaY: f32, origin: vec2f) -> f32 {
  var segments = 1u;
  var previous = integrateGeodesicEdgeCross(position, areaX, areaY, nextPosition, nextAreaX, nextAreaY, origin, 1.0);
  for (var refinement = 0u; refinement < 8u; refinement++) {
    segments *= 2u;
    let current = integrateGeodesicEdgeCross(position, areaX, areaY, nextPosition, nextAreaX, nextAreaY, origin, f32(segments));
    if ((bitcast<u32>(current) & 0x7f800000u) == 0x7f800000u) { return current; }
    if (abs(current - previous) / 15.0 <= ${s(2*e/(n*n))}) { return current; }
    previous = current;
    if (segments >= ${t}u) { break; }
  }
  return bitcast<f32>(0x7fc00000u | (bitcast<u32>(position.x) & 0u));
}
`}function z(e,t){let{coordinateSystem:n}=t,r=n!==`planar`,i=n===`wgs84`||n===`geodesic`,a=n===`geodesic`,o=[{name:`positions`,view:t.positions,type:`f32`,access:`read`}];t.ringOffsets&&o.push({name:`ringOffsets`,view:t.ringOffsets,type:`u32`,access:`read`}),t.featureOffsets&&o.push({name:`featureOffsets`,view:t.featureOffsets,type:`u32`,access:`read`}),t.polygonOffsets&&o.push({name:`polygonOffsets`,view:t.polygonOffsets,type:`u32`,access:`read`}),o.push({name:`featureStats`,view:t.featureStats,type:`u32`,access:`read_write`});let s={planar:`return length(b - a);`,spherical:`return geodesicCentralAngle(a, b) * RADIUS;`,geodesic:`let inverse = vincentyInverse(a, b);
  if (inverse.converged) {
    return inverse.distance;
  }
  return bitcast<f32>(0x7fc00000u | (bitcast<u32>(a.x) & 0u));`,wgs84:`let inverse = vincentyInverse(a, b);
  if (inverse.converged) {
    return inverse.distance;
  }
  return bitcast<f32>(0x7fc00000u | (bitcast<u32>(a.x) & 0u));`}[n],l={planar:`return position.y - origin.y;`,spherical:`return geodesicSinLatitudeDelta(position.y, origin.y);`,wgs84:`return getAuthalicSinDelta(sin(origin.y * GEODESIC_DEGREES_TO_RADIANS), geodesicSinLatitudeDelta(position.y, origin.y));`,geodesic:`return getAuthalicSinDelta(sin(origin.y * GEODESIC_DEGREES_TO_RADIANS), geodesicSinLatitudeDelta(position.y, origin.y));`}[n],u={planar:`origin.y + centroid.y`,spherical:`asin(clamp(sin(origin.y * GEODESIC_DEGREES_TO_RADIANS) + centroid.y, -1.0, 1.0)) * GEODESIC_RADIANS_TO_DEGREES`,wgs84:`getGeodeticFromAuthalic(asin(clamp(getSinAuthalic(sin(origin.y * GEODESIC_DEGREES_TO_RADIANS)) + centroid.y, -1.0, 1.0))) * GEODESIC_RADIANS_TO_DEGREES`,geodesic:`getGeodeticFromAuthalic(asin(clamp(getSinAuthalic(sin(origin.y * GEODESIC_DEGREES_TO_RADIANS)) + centroid.y, -1.0, 1.0))) * GEODESIC_RADIANS_TO_DEGREES`}[n],d=t.greatCircleEdges?`var edgeCross = cross;
      if (IS_POLYGON) {
        // Lambert chord from local differences, less twice the great-circle bulge over it.
        edgeCross = cross - 2.0 * getGreatCircleEdgeBulge(
          position, nextPosition, lonLatRingStep(position, nextPosition));
      }`:a?`var edgeCross = cross;
      if (IS_POLYGON) {
        edgeCross = getGeodesicEdgeCross(position, areaX, areaY, nextPosition, nextAreaX, nextAreaY, origin);
      }`:`let edgeCross = cross;`,f=`
const ROW_COUNT: u32 = ${t.positions.length}u;
const RING_COUNT: u32 = ${t.ringOffsets?t.ringOffsets.length-1:t.positions.length}u;
const IS_POLYGON: bool = ${t.isPolygon};
const IS_POINT: bool = ${t.isPoint??!1};
const FIRST_RING_EXTERIOR: bool = ${!!t.polygonOffsets};
${t.polygonOffsets?`const POLYGON_COUNT: u32 = ${t.polygonOffsets.length-1}u;

// GeoArrow ring role by position: ring 'ring' is a shell when it starts a polygon (binary search
// of polygonOffsets). A ring outside the polygonOffsets range keeps the first-ring rule.
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
}`:`fn isShellRing(ring: u32, isFirstRing: bool) -> bool {
  return isFirstRing;
}`}
const IS_GEOGRAPHIC: bool = ${r};
const STATS_STRIDE: u32 = 16u;
${w}
${P(n,t.radius,t.areaRadius,t.ellipsoid,t.lengthScale)}
${i?`${(t.ellipsoidalInverse===`safeguarded`?E:T)(16,t.ellipsoid)}
${F}`:``}

${S}
${t.greatCircleEdges?L:``}
fn isAmbiguousEdge(first: vec2<f32>, second: vec2<f32>) -> bool {
  if (!IS_POLYGON) { return ${i?`!vincentyInverse(first, second).converged`:`false`}; }
  return ${i?`!vincentyInverse(first, second).converged || `:``}${t.greatCircleEdges?`lonLatIsAmbiguousGreatCircleEdge`:`lonLatIsAmbiguousLinearEdge`}(first, second);
}

fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

fn getEdgeLength(a: vec2<f32>, b: vec2<f32>) -> f32 {
  ${s}
}

fn getAreaY(position: vec2<f32>, origin: vec2<f32>) -> f32 {
  ${l}
}
${a?t.geodesicAreaTolerance===void 0?I:R(t.geodesicAreaTolerance,t.geodesicAreaMaximumSegments??256,t.areaRadius):``}

${C}
// Compensated addition: sum.x holds the running sum, sum.y the accumulated rounding errors of its
// additions. The error comes from getAdditionRoundingError, which reads it off the integer
// significands. The float Neumaier term ((sum - total) + value) is algebraically zero and the
// test GPU's shader compiler folds it away: a 300,000-vertex ring was 1.1e-3 off, the plain f32 sum.
fn addCompensated(sum: ptr<function, vec2<f32>>, value: f32) {
  let previous = (*sum).x;
  let total = previous + value;
  (*sum).y = (*sum).y + getAdditionRoundingError(previous, value, total);
  (*sum).x = total;
}

fn getCompensated(sum: vec2<f32>) -> f32 {
  return sum.x + sum.y;
}

fn writeStat(feature: u32, slot: u32, value: f32) {
  featureStats[featureStatsOffset + feature * STATS_STRIDE + slot] = bitcast<u32>(value);
}
`,p=e=>`  let crossSum = getCompensated(areaSum);
  let totalLength = getCompensated(lengthSum);
  let areaScale = select(1.0, AREA_RADIUS * AREA_RADIUS, IS_GEOGRAPHIC);
  let invalidNan = bitcast<f32>(0x7fc00000u | (index & 0u));
  let isInvalid = invalidRings > 0u;
  let signedArea = select(select(0.0, 0.5 * crossSum * areaScale, IS_POLYGON), invalidNan, isInvalid);
  var centroid = vertexSum / f32(vertexCount);
  var centroidX = origin.x + centroid.x;
  var centroidY = origin.y + centroid.y;
  // Centroids use the straight (Lambert) edges even when the area follows geodesics.
  let straightCrossSum = getCompensated(straightSum);
  if (IS_POLYGON && straightCrossSum != 0.0) {
    centroid = vec2<f32>(getCompensated(momentX), getCompensated(momentY)) / (3.0 * straightCrossSum);
    if (IS_GEOGRAPHIC) {
      centroidX = origin.x + centroid.x * GEODESIC_RADIANS_TO_DEGREES;
    } else {
      centroidX = origin.x + centroid.x;
    }
    centroidY = ${u};
  } else if (!IS_POLYGON && totalLength > 0.0) {
    centroid = vec2<f32>(getCompensated(lineMomentX), getCompensated(lineMomentY)) / totalLength;
    centroidX = origin.x + centroid.x;
    centroidY = origin.y + centroid.y;
  }
  // Convert once before both feature output and group reduction, shared with the cooperative path.
  writeStat(index, 0u, totalLength * LENGTH_SCALE);
  writeStat(index, 1u, signedArea * LENGTH_SCALE * LENGTH_SCALE);
  writeStat(index, 2u, abs(signedArea) * LENGTH_SCALE * LENGTH_SCALE);
  writeStat(index, 3u, select(centroidX, invalidNan, isInvalid));
  writeStat(index, 4u, select(centroidY, invalidNan, isInvalid));
  writeStat(index, 5u, origin.x + boundsMin.x);
  writeStat(index, 6u, origin.y + boundsMin.y);
  writeStat(index, 7u, origin.x + boundsMax.x);
  writeStat(index, 8u, origin.y + boundsMax.y);
  featureStats[featureStatsOffset + index * STATS_STRIDE + 9u] = vertexCount;
  featureStats[featureStatsOffset + index * STATS_STRIDE + 10u] = 1u;
  for (var slot = 0u; slot < 4u; slot++) {
    featureStats[featureStatsOffset + index * STATS_STRIDE + 11u + slot] = extremeRows[slot];
  }
  featureStats[featureStatsOffset + index * STATS_STRIDE + 15u] = invalidRings;`.replace(/\b(invalidRings|areaSum|lengthSum|vertexSum|vertexCount|straightSum|momentX|momentY|lineMomentX|lineMomentY|boundsMin|boundsMax|extremeRows)\b/g,t=>`${e}${t}`),m=t.cooperativeRingRows??0,h=A({hasRingOffsets:!!t.ringOffsets,hasFeatureOffsets:!!t.featureOffsets,hasPolygonOffsets:!!t.polygonOffsets},m);if(t.cooperative){let n=x(Math.floor(t.positions.length/(m+1)),1,t.featureCount);return c(e,{id:t.id,operation:t.operation,variant:`feature-measures-cooperative`,bindings:o,workgroupSize:64,invocationCount:n*64,guardIndex:!1,declarations:`${f}
${h}
${M({groupCount:n,featureCount:t.featureCount,edgeCrossSource:d,tail:p(`acc.`)})}`,body:j})}return c(e,{id:t.id,operation:t.operation,variant:`feature-measures`,bindings:o,invocationCount:t.featureCount,declarations:`${f}\n${h}`,body:`
  ${m>0?`if (isCooperativeFeature(index)) {
    return;
  }`:``}
  let ringRange = getFeatureRingRange(index);
  let ringBegin = ringRange.x;
  let ringEnd = ringRange.y;
  // Local origin: the first vertex of the first non-empty ring.
  var origin = vec2<f32>(0.0);
  var hasVertex = false;
  for (var ring = ringBegin; ring < ringEnd; ring++) {
    let ringRows = getRingRows(ring);
    let rowBegin = ringRows.x;
    let rowEnd = rowBegin + ringRows.y;
    if (rowEnd > rowBegin) {
      origin = getPosition(rowBegin);
      hasVertex = true;
      break;
    }
  }
  // A runtime operand keeps the NaN out of const-expression evaluation, where it is an error.
  let nan = bitcast<f32>(0x7fc00000u | (index & 0u));
  if (!hasVertex) {
    writeStat(index, 0u, 0.0);
    writeStat(index, 1u, 0.0);
    writeStat(index, 2u, 0.0);
    for (var slot = 3u; slot < 9u; slot++) {
      writeStat(index, slot, nan);
    }
    featureStats[featureStatsOffset + index * STATS_STRIDE + 9u] = 0u;
    featureStats[featureStatsOffset + index * STATS_STRIDE + 10u] = 1u;
    for (var slot = 11u; slot < 15u; slot++) {
      featureStats[featureStatsOffset + index * STATS_STRIDE + slot] = 0xffffffffu;
    }
    featureStats[featureStatsOffset + index * STATS_STRIDE + 15u] = 0u;
    return;
  }
  var lengthSum = vec2<f32>(0.0);
  var areaSum = vec2<f32>(0.0);
  var momentX = vec2<f32>(0.0);
  var momentY = vec2<f32>(0.0);
  var lineMomentX = vec2<f32>(0.0);
  var lineMomentY = vec2<f32>(0.0);
  var vertexSum = vec2<f32>(0.0);
  var boundsMin = vec2<f32>(3.4e38);
  var boundsMax = vec2<f32>(-3.4e38);
  var vertexCount = 0u;
  var extremeRows = vec4<u32>(0xffffffffu);
  var straightSum = vec2<f32>(0.0);
  var invalidRings = 0u;
  var isFirstRing = true;
  for (var ring = ringBegin; ring < ringEnd; ring++) {
    let ringRows = getRingRows(ring);
    let rowBegin = ringRows.x;
    let rowEnd = rowBegin + ringRows.y;
    let rowCount = rowEnd - rowBegin;
    if (rowCount == 0u) {
      continue;
    }
    vertexCount += rowCount;
    var ringArea = vec2<f32>(0.0);
    var ringStraightArea = vec2<f32>(0.0);
    var ringMomentX = vec2<f32>(0.0);
    var ringMomentY = vec2<f32>(0.0);
    var ringWinding = 0.0;
    var ringMinimumLongitude = 0.0;
    var ringMaximumLongitude = 0.0;
    var ringAmbiguousEdges = 0u;
    let firstPosition = getPosition(rowBegin);
    // Local display coordinates (degrees for geographic, longitudes unwrapped along the ring).
    var local = firstPosition - origin;
    if (IS_GEOGRAPHIC) {
      local.x = geodesicWrapLongitudeDelta(firstPosition.x - origin.x);
    }
    let firstLocal = local;
    var areaX = local.x;
    if (IS_GEOGRAPHIC) {
      areaX = local.x * GEODESIC_DEGREES_TO_RADIANS;
    }
    var areaY = getAreaY(firstPosition, origin);
    var position = firstPosition;
    let edgeCount = select(rowCount - 1u, rowCount, IS_POLYGON);
    for (var vertex = 0u; vertex < rowCount; vertex++) {
      // Strict comparisons keep the lowest vertex row on ties.
      if (local.x < boundsMin.x) {
        extremeRows.x = rowBegin + vertex;
      }
      if (local.y < boundsMin.y) {
        extremeRows.y = rowBegin + vertex;
      }
      if (local.x > boundsMax.x) {
        extremeRows.z = rowBegin + vertex;
      }
      if (local.y > boundsMax.y) {
        extremeRows.w = rowBegin + vertex;
      }
      boundsMin = min(boundsMin, local);
      boundsMax = max(boundsMax, local);
      vertexSum += local;
      if (IS_POINT) {
        // Point sets have no edges; longitudes unwrap against the shared origin.
        if (vertex + 1u < rowCount) {
          let nextPoint = getPosition(rowBegin + vertex + 1u);
          local = nextPoint - origin;
          if (IS_GEOGRAPHIC) {
            local.x = geodesicWrapLongitudeDelta(nextPoint.x - origin.x);
          }
        }
        continue;
      }
      if (vertex >= edgeCount) {
        break;
      }
      let nextRow = select(rowBegin + vertex + 1u, rowBegin, vertex + 1u == rowCount);
      let nextPosition = getPosition(nextRow);
      var nextLocal = nextPosition - origin;
      if (IS_GEOGRAPHIC) {
        nextLocal = vec2<f32>(
          local.x + lonLatRingStep(position, nextPosition),
          nextPosition.y - origin.y
        );
      }
      if (vertex + 1u == rowCount) {
        // Closing edge: return to the ring's first local coordinates.
        nextLocal = firstLocal;
      }
      var nextAreaX = nextLocal.x;
      if (IS_GEOGRAPHIC) {
        nextAreaX = nextLocal.x * GEODESIC_DEGREES_TO_RADIANS;
      }
      let nextAreaY = getAreaY(nextPosition, origin);
      if (IS_POLYGON && IS_GEOGRAPHIC) {
        ringWinding += lonLatRingStep(position, nextPosition);
        ringMinimumLongitude = min(ringMinimumLongitude, ringWinding);
        ringMaximumLongitude = max(ringMaximumLongitude, ringWinding);
      }
      if (IS_GEOGRAPHIC) { ringAmbiguousEdges += select(0u, 1u, isAmbiguousEdge(position, nextPosition)); }
      let edgeLength = getEdgeLength(position, nextPosition);
      addCompensated(&lengthSum, edgeLength);
      addCompensated(&lineMomentX, edgeLength * 0.5 * (local.x + nextLocal.x));
      addCompensated(&lineMomentY, edgeLength * 0.5 * (local.y + nextLocal.y));
      let cross = areaX * nextAreaY - nextAreaX * areaY;
      ${d}
      if (IS_GEOGRAPHIC && (bitcast<u32>(edgeCross) & 0x7f800000u) == 0x7f800000u) { ringAmbiguousEdges += 1u; }
      addCompensated(&ringArea, edgeCross);
      addCompensated(&ringStraightArea, cross);
      addCompensated(&ringMomentX, (areaX + nextAreaX) * cross);
      addCompensated(&ringMomentY, (areaY + nextAreaY) * cross);
      position = nextPosition;
      local = nextLocal;
      areaX = nextAreaX;
      areaY = nextAreaY;
    }
    if (IS_GEOGRAPHIC && (ringAmbiguousEdges > 0u || (IS_POLYGON && (lonLatIsPoleWinding(ringWinding) || ringMaximumLongitude - ringMinimumLongitude > 360.0)))) {
      invalidRings += 1u;
    }
    var ringFactor = 1.0;
    if (FIRST_RING_EXTERIOR) {
      let ringSign = select(-1.0, 1.0, getCompensated(ringArea) >= 0.0);
      ringFactor = select(-1.0, 1.0, isShellRing(ring, isFirstRing)) * ringSign;
    }
    addCompensated(&areaSum, ringFactor * getCompensated(ringArea));
    addCompensated(&straightSum, ringFactor * getCompensated(ringStraightArea));
    addCompensated(&momentX, ringFactor * getCompensated(ringMomentX));
    addCompensated(&momentY, ringFactor * getCompensated(ringMomentY));
    isFirstRing = false;
  }
${p(``)}`})}function B(e,t){let{columns:n}=t,r=[{name:`stats`,view:t.stats,type:`u32`,access:`read`}],i=[],a=(e,t,a)=>{let o=n[e];o&&(r.push({name:e,view:o,type:t,access:`read_write`}),i.push(a))};return a(`lengths`,`f32`,`lengths[lengthsOffset + index] = getStat(0u);`),a(`areas`,`f32`,`areas[areasOffset + index] = getStat(2u);`),a(`centroids`,`f32`,`centroids[centroidsOffset + 2u * index] = getStat(3u);
  centroids[centroidsOffset + 2u * index + 1u] = getStat(4u);`),a(`bounds`,`f32`,`for (var corner = 0u; corner < 4u; corner++) {
    bounds[boundsOffset + 4u * index + corner] = getStat(5u + corner);
  }`),a(`vertexCounts`,`u32`,`vertexCounts[vertexCountsOffset + index] = getStatBits(9u);`),a(`extremeVertices`,`u32`,`for (var corner = 0u; corner < 4u; corner++) {
    extremeVertices[extremeVerticesOffset + 4u * index + corner] = getStatBits(11u + corner);
  }`),a(`featureCounts`,`u32`,`featureCounts[featureCountsOffset + index] = getStatBits(10u);`),c(e,{id:t.id,operation:t.operation,variant:`scatter`,bindings:r,invocationCount:t.rowCount,declarations:`const STATS_STRIDE: u32 = 16u;
var<private> row: u32;

fn getStatBits(slot: u32) -> u32 {
  return stats[statsOffset + row * STATS_STRIDE + slot];
}

fn getStat(slot: u32) -> f32 {
  return bitcast<f32>(getStatBits(slot));
}`,body:`row = index;
  ${i.join(`
  `)}`})}function V(e,t){return c(e,{id:t.id,operation:t.operation,variant:`group-sort-prepare`,bindings:[{name:`groupIds`,view:t.groupIds,type:`u32`,access:`read`},{name:`sortKeys`,view:t.sortKeys,type:`u32`,access:`read_write`},{name:`sortValues`,view:t.sortValues,type:`u32`,access:`read_write`}],invocationCount:t.groupIds.length,declarations:`const GROUP_COUNT: u32 = ${t.groupCount}u;`,body:`sortKeys[sortKeysOffset + index] = min(groupIds[groupIdsOffset + index], GROUP_COUNT);
  sortValues[sortValuesOffset + index] = index;`})}var H=128,U=11;function W(e,t){return c(e,{id:t.id,operation:t.operation,variant:`group-reduce`,bindings:[{name:`sortedIndices`,view:t.sortedIndices,type:`u32`,access:`read`},{name:`groupOffsets`,view:t.groupOffsets,type:`u32`,access:`read`},{name:`featureStats`,view:t.featureStats,type:`u32`,access:`read`},{name:`groupStats`,view:t.groupStats,type:`u32`,access:`read_write`}],workgroupSize:H,invocationCount:t.groupCount*H,guardIndex:!1,declarations:`const STATS_STRIDE: u32 = 16u;
const SUM_SLOTS: u32 = ${U}u;
const IS_POLYGON: bool = ${t.isPolygon};
const IS_POINT: bool = ${t.isPoint??!1};
const IS_GEOGRAPHIC: bool = ${t.isGeographic??!1};
const ROW_COUNT: u32 = ${t.sortedIndices.length}u;
const DEGREES_TO_RADIANS: f32 = 0.017453292519943295;
const RADIANS_TO_DEGREES: f32 = 57.2957795130823;
// Slots: length, signedArea, area, weight, weightedX, weightedY, meanX, meanY, centroidCount,
// weightedZ, meanZ. Geographic groups sum unit vectors (x, y, z) in the X, Y, Z slots, so the
// centroid is the spherical mean and longitudes are never averaged naively across the antimeridian.
var<workgroup> partialSums: array<array<f32, ${U}>, ${H}>;
var<workgroup> partialMinimum: array<vec2<f32>, ${H}>;
var<workgroup> partialMaximum: array<vec2<f32>, ${H}>;
var<workgroup> partialVertices: array<u32, ${H}>;

fn getFeatureStat(word: u32) -> f32 {
  return bitcast<f32>(featureStats[word]);
}`,body:`let group = index / ${H}u;
  let isInRange = index < INVOCATION_COUNT;
  var sums: array<f32, ${U}>;
  for (var slot = 0u; slot < SUM_SLOTS; slot++) {
    sums[slot] = 0.0;
  }
  var minimum = vec2<f32>(3.4e38);
  var maximum = vec2<f32>(-3.4e38);
  var vertices = 0u;
  var begin = 0u;
  var end = 0u;
  if (isInRange) {
    begin = groupOffsets[groupOffsetsOffset + group];
    end = min(groupOffsets[groupOffsetsOffset + group + 1u], ROW_COUNT);
    for (var row = begin + localInvocationIndex; row < end; row += ${H}u) {
      let base = featureStatsOffset + sortedIndices[sortedIndicesOffset + row] * STATS_STRIDE;
      let vertexCount = featureStats[base + 9u];
      let featureLength = getFeatureStat(base);
      let featureArea = getFeatureStat(base + 2u);
      sums[0] += featureLength;
      sums[1] += getFeatureStat(base + 1u);
      sums[2] += featureArea;
      if (vertexCount > 0u) {
        let centroid = vec2<f32>(getFeatureStat(base + 3u), getFeatureStat(base + 4u));
        var weight = select(featureLength, featureArea, IS_POLYGON);
        if (IS_POINT) {
          weight = f32(vertexCount);
        }
        var point = vec3<f32>(centroid.x, centroid.y, 0.0);
        if (IS_GEOGRAPHIC) {
          let longitude = centroid.x * DEGREES_TO_RADIANS;
          let latitude = centroid.y * DEGREES_TO_RADIANS;
          let cosLatitude = sin((90.0 - abs(centroid.y)) * DEGREES_TO_RADIANS);
          point = vec3<f32>(cosLatitude * cos(longitude), cosLatitude * sin(longitude), sin(latitude));
        }
        sums[3] += weight;
        sums[4] += weight * point.x;
        sums[5] += weight * point.y;
        sums[9] += weight * point.z;
        sums[6] += point.x;
        sums[7] += point.y;
        sums[10] += point.z;
        sums[8] += 1.0;
        minimum = min(minimum, vec2<f32>(getFeatureStat(base + 5u), getFeatureStat(base + 6u)));
        maximum = max(maximum, vec2<f32>(getFeatureStat(base + 7u), getFeatureStat(base + 8u)));
        vertices += vertexCount;
      }
    }
  }
  for (var slot = 0u; slot < SUM_SLOTS; slot++) {
    partialSums[localInvocationIndex][slot] = sums[slot];
  }
  partialMinimum[localInvocationIndex] = minimum;
  partialMaximum[localInvocationIndex] = maximum;
  partialVertices[localInvocationIndex] = vertices;
  workgroupBarrier();
  for (var stride = ${H/2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      let other = localInvocationIndex + stride;
      for (var slot = 0u; slot < SUM_SLOTS; slot++) {
        partialSums[localInvocationIndex][slot] += partialSums[other][slot];
      }
      partialMinimum[localInvocationIndex] = min(partialMinimum[localInvocationIndex], partialMinimum[other]);
      partialMaximum[localInvocationIndex] = max(partialMaximum[localInvocationIndex], partialMaximum[other]);
      partialVertices[localInvocationIndex] += partialVertices[other];
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    let base = groupStatsOffset + group * STATS_STRIDE;
    let nan = bitcast<f32>(0x7fc00000u | (group & 0u));
    let weight = partialSums[0][3];
    let centroidCount = partialSums[0][8];
    var centroid = vec2<f32>(nan);
    var sumVector = vec3<f32>(nan);
    if (weight > 0.0) {
      sumVector = vec3<f32>(partialSums[0][4], partialSums[0][5], partialSums[0][9]) / weight;
    } else if (centroidCount > 0.0) {
      sumVector = vec3<f32>(partialSums[0][6], partialSums[0][7], partialSums[0][10]) / centroidCount;
    }
    if (IS_GEOGRAPHIC) {
      // Antipodal members cancel to a zero vector, which has no direction: NaN.
      let horizontal = length(sumVector.xy);
      if (horizontal + abs(sumVector.z) > 1e-6) {
        centroid = vec2<f32>(
          atan2(sumVector.y, sumVector.x) * RADIANS_TO_DEGREES,
          atan2(sumVector.z, horizontal) * RADIANS_TO_DEGREES
        );
      }
    } else {
      centroid = sumVector.xy;
    }
    groupStats[base] = bitcast<u32>(partialSums[0][0]);
    groupStats[base + 1u] = bitcast<u32>(partialSums[0][1]);
    groupStats[base + 2u] = bitcast<u32>(partialSums[0][2]);
    groupStats[base + 3u] = bitcast<u32>(centroid.x);
    groupStats[base + 4u] = bitcast<u32>(centroid.y);
    let hasBounds = centroidCount > 0.0;
    groupStats[base + 5u] = bitcast<u32>(select(nan, partialMinimum[0].x, hasBounds));
    groupStats[base + 6u] = bitcast<u32>(select(nan, partialMinimum[0].y, hasBounds));
    groupStats[base + 7u] = bitcast<u32>(select(nan, partialMaximum[0].x, hasBounds));
    groupStats[base + 8u] = bitcast<u32>(select(nan, partialMaximum[0].y, hasBounds));
    groupStats[base + 9u] = partialVertices[0];
    groupStats[base + 10u] = end - begin;
  }`})}var G=`GPUGeometryMeasures`,K=Object.freeze({edgeModels:Object.freeze([`linear`,`great-circle`]),coordinateSpaces:Object.freeze([`planar`,`longitude-latitude`]),metricsByCoordinateSpace:Object.freeze({planar:Object.freeze([`none`,`native`]),"longitude-latitude":Object.freeze([`great-circle`,`ellipsoidal`])})}),q=class{id;props;spatialContext;featureCount;measures;coordinateSystem;radius;areaRadius;cooperativeRingRows;constructor(e){this.id=e.id??`geometry-measures`,this.props=e,this.spatialContext=v(this.id,e.spatialContext,K),this.coordinateSystem=Z(this.id,this.spatialContext,e.ellipsoidalEdgeModel),this.measures=Y(this.coordinateSystem,this.spatialContext),this.radius=m(this.spatialContext),this.areaRadius=p(this.spatialContext)===`great-circle`?this.radius:f(this.spatialContext),this.cooperativeRingRows=e.cooperativeRingRows??512;let{id:t}=this,{status:s,...c}=e.output??{},l=e.groupOutput??{};if(![`lines`,`polygons`,`points`].includes(e.geometryType))throw Error(`${t} geometryType must be 'lines', 'polygons' or 'points'`);i(t,`cooperativeRingRows`,this.cooperativeRingRows,0),r(t,e.geodesicAreaTolerance===void 0||Number.isFinite(e.geodesicAreaTolerance)&&e.geodesicAreaTolerance>0,`geodesicAreaTolerance must be a positive finite number`);let d=e.geodesicAreaMaximumSegments??256;if(a(t,`geodesicAreaMaximumSegments`,d,2,256),r(t,Number.isFinite(this.radius)&&this.radius>0,`spatialContext sphere radius must be a positive finite number`),n(e.positions,[`float32x2`],`${t} positions`),e.positions.length<1)throw Error(`${t} needs at least one position`);let h=J(e);e.geometryType===`points`&&e.featureOffsets?u(t,{kind:`points`,positions:e.positions,featureOffsets:e.featureOffsets}):e.geometryType===`lines`&&e.featureOffsets&&u(t,{kind:`lines`,positions:e.positions,lineOffsets:e.lineOffsets,featureOffsets:e.featureOffsets});for(let e of[`ringOffsets`,`featureOffsets`,`polygonOffsets`]){let r=h[e];if(r&&(n(r,[`uint32`],`${t} ${e}`),r.length<2))throw Error(`${t} ${e} must contain at least two rows`)}this.featureCount=h.featureOffsets?h.featureOffsets.length-1:h.ringOffsets?h.ringOffsets.length-1:e.positions.length;let g=e.geometryType===`polygons`;Q(t,`output`,c,this.featureCount,g);let _=Object.values(l).some(Boolean);if(e.groupIds){if(n(e.groupIds,[`uint32`],`${t} groupIds`),e.groupIds.length!==this.featureCount)throw Error(`${t} groupIds length must equal the feature count`);if(i(t,`groupCount`,e.groupCount??0),!_)throw Error(`${t} groupIds require at least one groupOutput column`);Q(t,`groupOutput`,l,e.groupCount??0,g)}else if(_)throw Error(`${t} groupOutput requires groupIds`);let y=s?.invalidCount;if(y&&(n(y,[`uint32`],`${t} output.status.invalidCount`),y.length<1))throw Error(`${t} output.status.invalidCount must contain one uint32 row`);if(!Object.values(c).some(Boolean)&&!_)throw Error(`${t} requires at least one output column`);o(t,[...Object.values(c),...Object.values(l),y],[e.positions,h.ringOffsets,h.featureOffsets,h.polygonOffsets,e.groupIds])}getCommandNodes(t){let{props:n,id:r,featureCount:i}=this,{status:a,...o}=n.output??{},s=n.groupOutput??{},c=J(n);D(t,r,Object.fromEntries([n.positions,c.ringOffsets,c.featureOffsets,c.polygonOffsets,n.groupIds,a?.invalidCount,...Object.values(o),...Object.values(s)].map((e,t)=>[String(t),e]))),O(t,r,{featureStatistics:i*16*4,...n.groupIds&&n.groupCount?{groupStatistics:n.groupCount*16*4,groupOffsets:(n.groupCount+1)*4,...k(i)}:{}});let u=n.geometryType===`polygons`,d=e(t,`${r}-feature-stats`,`uint32`,i*16),f={id:`${r}-features`,operation:G,positions:n.positions,ringOffsets:c.ringOffsets,featureOffsets:c.featureOffsets,polygonOffsets:c.polygonOffsets,featureCount:i,isPolygon:u,isPoint:n.geometryType===`points`,coordinateSystem:this.coordinateSystem,greatCircleEdges:p(this.spatialContext)===`great-circle`,radius:this.radius,areaRadius:this.areaRadius,ellipsoid:g(this.spatialContext),ellipsoidalInverse:n.ellipsoidalInverse,geodesicAreaTolerance:n.geodesicAreaTolerance,geodesicAreaMaximumSegments:n.geodesicAreaMaximumSegments,lengthScale:this.spatialContext.metric===`ellipsoidal`&&this.spatialContext.units===`kilometers`?.001:1,featureStats:d},m=a?.invalidCount,h=this.cooperativeRingRows>0&&n.positions.length>this.cooperativeRingRows,_=[z(t,{...f,cooperativeRingRows:h?this.cooperativeRingRows:0})];return h&&_.push(z(t,{...f,id:`${r}-features-cooperative`,cooperative:!0,cooperativeRingRows:this.cooperativeRingRows})),m&&_.push(l(t,{id:`${r}-clear-invalid-count`,operation:G,view:m,type:`u32`,value:`0u`}),X(t,r,i,d,m)),Object.values(o).some(Boolean)&&_.push(B(t,{id:`${r}-feature-output`,operation:G,rowCount:i,stats:d,columns:o})),n.groupIds&&n.groupCount&&_.push(...this._getGroupNodes(t,n.groupIds,n.groupCount,d)),_}_getGroupNodes(n,r,i,a){let{id:o,featureCount:s}=this,c=e(n,`${o}-group-keys`,`uint32`,s),l=e(n,`${o}-group-values`,`uint32`,s),u=e(n,`${o}-group-sorted-keys`,`uint32`,s),d=e(n,`${o}-group-sorted-indices`,`uint32`,s),f=e(n,`${o}-group-offsets`,`uint32`,i+1),p=e(n,`${o}-group-stats`,`uint32`,i*16);return[V(n,{id:`${o}-group-sort-prepare`,operation:G,groupIds:r,groupCount:i,sortKeys:c,sortValues:l}),...new t({id:`${o}-group-sort`,keys:c,values:l,outputKeys:u,outputValues:d,keyBits:y(i)}).getCommandNodes(n),b(n,{id:`${o}-group-offsets`,operation:G,variant:`group-offsets`,segmentCount:i,sortedKeys:u,segmentOffsets:f}),W(n,{id:`${o}-group-reduce`,operation:G,isPolygon:this.props.geometryType===`polygons`,isPoint:this.props.geometryType===`points`,isGeographic:this.coordinateSystem!==`planar`,groupCount:i,sortedIndices:d,groupOffsets:f,featureStats:a,groupStats:p}),B(n,{id:`${o}-group-output`,operation:G,rowCount:i,stats:p,columns:this.props.groupOutput})]}};function J(e){switch(e.geometryType){case`polygons`:return{ringOffsets:e.ringOffsets,featureOffsets:e.featureOffsets,polygonOffsets:e.polygonOffsets};case`lines`:return{ringOffsets:e.lineOffsets,featureOffsets:e.featureOffsets};default:return{ringOffsets:e.featureOffsets}}}function Y(e,t){let n={planar:`planar`,spherical:`sphere-equal-area`,wgs84:`wgs84-authalic`,geodesic:`geodesic`},r={planar:`planar`,spherical:`great-circle`,wgs84:`ellipsoidal-geodesic`,geodesic:`ellipsoidal-geodesic`},i=h(t);return{lengthModel:r[e],areaModel:e===`spherical`&&p(t)===`great-circle`?`sphere-polygon`:n[e],lengthUnits:i,areaUnits:_(i)}}function X(e,t,n,r,i){let a=[{name:`featureStats`,view:r,type:`u32`,access:`read`},{name:`invalidCount`,view:i,type:`atomic<u32>`,access:`read_write`}];return c(e,{id:`${t}-invalid-count`,operation:G,variant:`invalid-count`,bindings:a,invocationCount:n,declarations:`const STATS_STRIDE: u32 = 16u;`,body:`if (featureStats[featureStatsOffset + index * STATS_STRIDE + 15u] > 0u) {
    atomicAdd(&invalidCount[invalidCountOffset], 1u);
  }`})}function Z(e,t,n){if(n&&t.metric!==`ellipsoidal`)throw Error(`${e} ellipsoidalEdgeModel requires an ellipsoidal spatial context`);return t.coordinateSpace===`planar`?`planar`:t.metric===`great-circle`?`spherical`:n===`coordinate-linear`?`wgs84`:`geodesic`}function Q(e,t,r,i,a){let o={lengths:`float32`,areas:`float32`,centroids:`float32x2`,bounds:`float32x4`,vertexCounts:`uint32`,extremeVertices:`uint32x4`,featureCounts:`uint32`};for(let[s,c]of Object.entries(r))if(c){if(!(s in o)||t===`groupOutput`&&s===`extremeVertices`)throw Error(`${e} ${t}.${s} is not a measure column`);if(n(c,[o[s]],`${e} ${t}.${s}`),c.length!==i)throw Error(`${e} ${t}.${s} must hold ${i} rows`);if(!a&&s===`areas`)throw Error(`${e} ${t}.${s} requires geometryType 'polygons'`)}}export{q as t};