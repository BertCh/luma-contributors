// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {SPATIAL_PROJECTIVE_WGSL} from './spatial-projective-wgsl';

/** Shared exact integer arithmetic, without the polygon projective-query helpers. @internal */
export const SPATIAL_DISTANCE_INTEGER_WGSL = SPATIAL_PROJECTIVE_WGSL.slice(
  0,
  SPATIAL_PROJECTIVE_WGSL.indexOf('fn spatialEndpointQuery')
);

/**
 * Inclusive planar threshold comparison. The fast error envelope includes subtraction of
 * original coordinates; an inconclusive result uses exact degree-four integer predicates.
 * Requires SCALED_PLANAR_DISTANCE_WGSL and the shared SpatialInteger arithmetic. @internal
 */
export const SPATIAL_DISTANCE_COMPARE_WGSL = /* wgsl */ `
fn spatialSquaredLength(x: SpatialInteger, y: SpatialInteger) -> SpatialInteger {
  return spatialAdd(spatialMultiply(x, x), spatialMultiply(y, y));
}
fn exactPointWithinSegmentDistance(point: vec2f, start: vec2f, end: vec2f, radius: f32) -> bool {
  let directionX = spatialSubtract(spatialInteger(end.x), spatialInteger(start.x));
  let directionY = spatialSubtract(spatialInteger(end.y), spatialInteger(start.y));
  let offsetX = spatialSubtract(spatialInteger(point.x), spatialInteger(start.x));
  let offsetY = spatialSubtract(spatialInteger(point.y), spatialInteger(start.y));
  let lengthSquared = spatialSquaredLength(directionX, directionY);
  let projection = spatialAdd(spatialMultiply(offsetX, directionX), spatialMultiply(offsetY, directionY));
  let radiusInteger = spatialInteger(radius);
  let radiusSquared = spatialMultiply(radiusInteger, radiusInteger);
  if (spatialSign(lengthSquared) == 0i || spatialSign(projection) <= 0i) {
    return spatialCompareMagnitude(spatialSquaredLength(offsetX, offsetY), radiusSquared) <= 0i;
  }
  let endOffsetX = spatialSubtract(spatialInteger(point.x), spatialInteger(end.x));
  let endOffsetY = spatialSubtract(spatialInteger(point.y), spatialInteger(end.y));
  let endProjection = spatialAdd(spatialMultiply(endOffsetX, directionX), spatialMultiply(endOffsetY, directionY));
  if (spatialSign(endProjection) >= 0i) {
    return spatialCompareMagnitude(spatialSquaredLength(endOffsetX, endOffsetY), radiusSquared) <= 0i;
  }
  let determinant = spatialCross(offsetX, offsetY, directionX, directionY);
  return spatialCompareMagnitude(spatialMultiply(determinant, determinant),
    spatialMultiply(radiusSquared, lengthSquared)) <= 0i;
}
fn pointWithinSegmentDistance(point: vec2f, start: vec2f, end: vec2f, radius: f32) -> bool {
  let approximate = getPlanarPointSegmentDistance(point, start, end);
  let scale = max(max(max(abs(point.x), abs(point.y)), max(abs(start.x), abs(start.y))),
    max(max(abs(end.x), abs(end.y)), radius));
  let error = scale * 0.000003814697265625;
  if (isFiniteValue(approximate) && isFiniteValue(error) && abs(approximate - radius) > error) {
    return approximate < radius;
  }
  return exactPointWithinSegmentDistance(point, start, end, radius);
}
fn segmentWithinDistance(a: vec2f, b: vec2f, c: vec2f, d: vec2f, radius: f32) -> bool {
  if (segmentsIntersect(a, b, c, d)) { return true; }
  return pointWithinSegmentDistance(a, c, d, radius) || pointWithinSegmentDistance(b, c, d, radius) ||
    pointWithinSegmentDistance(c, a, b, radius) || pointWithinSegmentDistance(d, a, b, radius);
}
`;
