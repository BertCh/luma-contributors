import{i as e}from"./gpu-raster-extrema-pyramid-BYXR99nY.js";import{t}from"./raster-band-utils-BcTxc8_Y.js";import{s as n}from"./raster-grid-VB0AM_GY.js";function r(e){return`
fn getRayGroundCell(cellSize: vec2<f32>, northEdge: f32, southEdge: f32, rowCoordinate: f32) -> vec2<f32> {
  ${e===`uniform`?`return cellSize;`:`let edge = mix(northEdge, southEdge, rowCoordinate / f32(HEIGHT));
  ${n(e)}
  return cellSize * groundScale;`}
}`}var i=762939453125e-17;function a(n){let{width:r,height:a,traversal:o,layout:s,clearance:c}=n;if(o===`pyramid`&&!s)throw Error(`terrain sight line traversal pyramid requires a pyramid layout`);let l=o===`pyramid`?e(s):``;return`const WIDTH: u32 = ${r}u;
const HEIGHT: u32 = ${a}u;
const HIDDEN: u32 = 0u;
const VISIBLE: u32 = 1u;
const OUT_OF_RANGE: u32 = 2u;
const NO_DATA: u32 = 3u;
const MARGINAL: u32 = 4u;
const MAXIMUM_FLOAT: f32 = 3.4028234663852886e38;
const SKIP_EPSILON: f32 = ${i};
${t}
${l}
fn isValidPixel(column: u32, row: u32) -> bool {
  return elevationValidity[elevationValidityOffset + row * WIDTH + column] != 0u;
}
fn getElevation(column: u32, row: u32) -> f32 {
  return elevationValues[elevationValuesOffset + row * WIDTH + column];
}
// Bilinear elevation at a clamped pixel-center position; .y is 0 when any corner is invalid.
fn sampleElevation(position: vec2<f32>) -> vec2<f32> {
  let maximum = vec2<f32>(f32(WIDTH - 1u), f32(HEIGHT - 1u));
  let clamped = clamp(position, vec2<f32>(0.0), maximum);
  let base = vec2<u32>(floor(clamped));
  let next = min(base + vec2<u32>(1u), vec2<u32>(WIDTH - 1u, HEIGHT - 1u));
  let fraction = clamped - floor(clamped);
  // Canonical elevation is NaN exactly where a pixel is invalid, so one bit test per corner
  // replaces four validity loads; the interpolation below is unchanged.
  let corner00 = getElevation(base.x, base.y);
  let corner10 = getElevation(next.x, base.y);
  let corner01 = getElevation(base.x, next.y);
  let corner11 = getElevation(next.x, next.y);
  if (!isFiniteValue(corner00) || !isFiniteValue(corner10) ||
      !isFiniteValue(corner01) || !isFiniteValue(corner11)) {
    return vec2<f32>(0.0, 0.0);
  }
  let top = mix(corner00, corner10, fraction.x);
  let bottom = mix(corner01, corner11, fraction.x);
  return vec2<f32>(mix(top, bottom, fraction.y), 1.0);
}
fn rayPosition(observer: vec2<f32>, delta: vec2<f32>, sampleIndex: u32, count: u32) -> vec2<f32> {
  let fraction = f32(sampleIndex) / f32(count);
  return observer + delta * fraction;
}
fn rayDistance(targetDistance: f32, sampleIndex: u32, count: u32) -> f32 {
  let fraction = f32(sampleIndex) / f32(count);
  return targetDistance * fraction;
}
struct SightLineResult {
  code: u32,
  clearance: f32
}
${o===`pyramid`?`
// Tries to skip samples \`first\` .. last at one pyramid level; returns last + 1, or 0 when the block
// does not contain the ray long enough or its bound exceeds the threshold.
fn trySkipLevel(
  level: u32,
  observer: vec2<f32>,
  delta: vec2<f32>,
  targetDistance: f32,
  first: u32,
  count: u32,
  baseFirst: vec2<u32>,
  distanceFirst: f32,
  eye: f32,
  curvature: f32,
  threshold: f32
) -> u32 {
  let gridMaximum = vec2<f32>(f32(WIDTH - 1u), f32(HEIGHT - 1u));
  let blockSize = pyramidLevelBlockSize(level);
  let cell = baseFirst / vec2<u32>(blockSize);
  let origin = vec2<f32>(cell * vec2<u32>(blockSize));
  let size = f32(blockSize);
  var exitFraction = 2.0;
  if (delta.x > 0.0) {
    exitFraction = min(exitFraction, (origin.x + size - observer.x) / delta.x);
  } else if (delta.x < 0.0) {
    exitFraction = min(exitFraction, (origin.x - observer.x) / delta.x);
  }
  if (delta.y > 0.0) {
    exitFraction = min(exitFraction, (origin.y + size - observer.y) / delta.y);
  } else if (delta.y < 0.0) {
    exitFraction = min(exitFraction, (origin.y - observer.y) / delta.y);
  }
  var last = u32(clamp(floor(exitFraction * f32(count)), f32(first), f32(count - 1u)));
  var inside = false;
  for (var attempt = 0u; attempt < 4u; attempt++) {
    let baseLast = vec2<u32>(floor(clamp(rayPosition(observer, delta, last, count), vec2<f32>(0.0), gridMaximum)));
    if (all(baseLast / vec2<u32>(blockSize) == cell)) {
      inside = true;
      break;
    }
    if (last <= first) {
      break;
    }
    last = last - 1u;
  }
  if (!inside) {
    return 0u;
  }
  let cellIndex = pyramidLevelIndex(level, baseFirst.x, baseFirst.y);
  let heightMaximum = pyramid[pyramidOffset + cellIndex];
  if (heightMaximum <= PYRAMID_EMPTY_MAXIMUM) {
    return last + 1u;
  }
  let heightMinimum = pyramid[pyramidOffset + PYRAMID_MINIMUM_OFFSET + cellIndex];
  let distanceLast = rayDistance(targetDistance, last, count);
  let magnitude = max(abs(heightMaximum), abs(heightMinimum));
  let dropFirst = curvature * distanceFirst * distanceFirst;
  let dropLast = curvature * distanceLast * distanceLast;
  let dropLow = min(dropFirst, dropLast);
  let dropAbsolute = max(abs(dropFirst), abs(dropLast));
  let numerator = (heightMaximum - eye) - dropLow +
    SKIP_EPSILON * (magnitude + abs(eye) + dropAbsolute);
  let upper = select(
    (numerator / distanceLast) * (1.0 - SKIP_EPSILON),
    (numerator / distanceFirst) * (1.0 + SKIP_EPSILON),
    numerator >= 0.0
  );
  return select(0u, last + 1u, upper <= threshold);
}
// Returns (index of the first sample that must still be evaluated, coarsest level that skipped).
// Every level is an independent valid skip, so the search starts at the level that worked last,
// descends until one works, then climbs while coarser levels also work.
fn skipSamples(
  observer: vec2<f32>,
  delta: vec2<f32>,
  targetDistance: f32,
  first: u32,
  count: u32,
  eye: f32,
  curvature: f32,
  threshold: f32,
  startLevel: u32
) -> vec2<u32> {
  let gridMaximum = vec2<f32>(f32(WIDTH - 1u), f32(HEIGHT - 1u));
  let baseFirst = vec2<u32>(floor(clamp(rayPosition(observer, delta, first, count), vec2<f32>(0.0), gridMaximum)));
  let distanceFirst = rayDistance(targetDistance, first, count);
  var level = min(startLevel, PYRAMID_LEVEL_COUNT - 1u);
  var next = trySkipLevel(level, observer, delta, targetDistance, first, count, baseFirst, distanceFirst, eye, curvature, threshold);
  while (next == 0u && level > 0u) {
    level = level - 1u;
    next = trySkipLevel(level, observer, delta, targetDistance, first, count, baseFirst, distanceFirst, eye, curvature, threshold);
  }
  if (next == 0u) {
    return vec2<u32>(first, 0u);
  }
  while (level + 1u < PYRAMID_LEVEL_COUNT) {
    let coarser = trySkipLevel(level + 1u, observer, delta, targetDistance, first, count, baseFirst, distanceFirst, eye, curvature, threshold);
    if (coarser == 0u) {
      break;
    }
    next = max(next, coarser);
    level = level + 1u;
  }
  return vec2<u32>(next, level);
}`:``}
// Observer and target are pixel-center positions; eye is the observer elevation (ground + height);
// targetTop is the target elevation (ground + height) before the curvature drop;
// tolerance is [toleranceMeters, tolerancePerKilometer, targetIgnoreDistance, targetIgnoreFraction].
fn traceSightLine(
  observer: vec2<f32>,
  targetPosition: vec2<f32>,
  eye: f32,
  targetTop: f32,
  cellSize: vec2<f32>,
  curvature: f32,
  maxDistance: f32,
  tolerance: vec4<f32>
) -> SightLineResult {
  let delta = targetPosition - observer;
  let targetDistance = length(delta * cellSize);
  if (maxDistance > 0.0 && targetDistance > maxDistance) {
    return SightLineResult(OUT_OF_RANGE, 0.0);
  }
  if (targetDistance == 0.0) {
    return SightLineResult(VISIBLE, MAXIMUM_FLOAT);
  }
  let targetElevation = targetTop - curvature * targetDistance * targetDistance;
  let targetSlope = (targetElevation - eye) / targetDistance;
  let band = (tolerance.x + tolerance.y * targetDistance / 1000.0) / targetDistance;
  let highLimit = targetSlope + band;
  let lowLimit = targetSlope - band;
  let stopDistance = targetDistance - max(tolerance.z, tolerance.w * targetDistance);
  let stepCount = u32(ceil(max(abs(delta.x), abs(delta.y))));
  var maxSlope = -MAXIMUM_FLOAT;
  var tested = false;
  var sampleIndex = 1u;
  var skipLevel = 0u;
  loop {
    if (sampleIndex >= stepCount) { break; }
    if (rayDistance(targetDistance, sampleIndex, stepCount) > stopDistance) { break; }${o===`pyramid`?`
    let threshold = ${c?`maxSlope`:`select(lowLimit, highLimit, maxSlope > lowLimit)`};
    let skipped = skipSamples(observer, delta, targetDistance, sampleIndex, stepCount, eye, curvature, threshold, skipLevel);
    if (skipped.x > sampleIndex) {
      sampleIndex = skipped.x;
      skipLevel = skipped.y;
      continue;
    }`:``}
    let current = sampleIndex;
    sampleIndex = sampleIndex + 1u;
    let elevationSample = sampleElevation(rayPosition(observer, delta, current, stepCount));
    // Invalid samples never block.
    if (elevationSample.y == 0.0) { continue; }
    let sampleDistance = rayDistance(targetDistance, current, stepCount);
    let sampleSlope =
      (elevationSample.x - curvature * sampleDistance * sampleDistance - eye) / sampleDistance;
    tested = true;
    if (sampleSlope > maxSlope) {
      maxSlope = sampleSlope;${c?``:`
      if (maxSlope > highLimit) {
        return SightLineResult(HIDDEN, 0.0);
      }`}
    }
  }
  var code = MARGINAL;
  if (maxSlope > highLimit) {
    code = HIDDEN;
  } else if (maxSlope <= lowLimit) {
    code = VISIBLE;
  }
  return SightLineResult(code, select(MAXIMUM_FLOAT, (targetSlope - maxSlope) * targetDistance, tested));
}`}export{a as n,r as t};