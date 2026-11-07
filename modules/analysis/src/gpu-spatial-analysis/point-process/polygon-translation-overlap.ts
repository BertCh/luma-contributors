// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {POLYGON_OVERLAP_METRIC_WGSL} from './polygon-overlap-metric';

/**
 * Signed fan composition: the indicator of an oriented polygon (including holes) is the sum
 * of its signed origin/edge triangles. Multiplying two indicators reduces intersection area
 * to signed triangle intersections. Convex clipping needs at most six vertices; eight slots
 * also hold intermediate polygons. Exact orientation decides clipping sides; coordinates
 * and areas are f32. Negative return values distinguish work exhaustion from numeric failure.
 * Requires the prepared study-window and exact-orientation declarations.
 */
export function getPolygonTranslationOverlapWGSL(vertices: number, edgePairBudget: number): string {
  return /* wgsl */ `
${POLYGON_OVERLAP_METRIC_WGSL}
fn overlapTriangle(first: vec2f, second: vec2f, shift: vec2f, third: vec2f, fourth: vec2f) -> f32 {
  var polygon: array<vec2f, 8>;
  var clipped: array<vec2f, 8>;
  polygon[0] = vec2f(0.0); polygon[1] = first; polygon[2] = second;
  var count = 3u;
  let clip = array<vec2f, 3>(shift, shift + third, shift + fourth);
  for (var edge = 0u; edge < 3u; edge++) {
    let start = clip[edge]; let finish = clip[(edge + 1u) % 3u];
    var nextCount = 0u;
    for (var vertex = 0u; vertex < count; vertex++) {
      let previous = polygon[(vertex + count - 1u) % count];
      let current = polygon[vertex];
      let previousSign = orientSign(start, finish, previous);
      let currentSign = orientSign(start, finish, current);
      if (previousSign == 2i || currentSign == 2i) { return -1.0; }
      if ((previousSign < 0i) != (currentSign < 0i)) {
        if (nextCount >= 8u) { return -1.0; }
        var intersection = previous;
        if (currentSign == 0i) { intersection = current; }
        else if (previousSign != 0i) {
          let previousDistance = overlapDeterminant(start, finish, previous);
          let currentDistance = overlapDeterminant(start, finish, current);
          let denominator = previousDistance - currentDistance;
          if (denominator == 0.0 || !isFiniteFloat(denominator)) { return -1.0; }
          let fraction = clamp(previousDistance / denominator, 0.0, 1.0);
          intersection = previous + fraction * (current - previous);
        }
        clipped[nextCount] = intersection; nextCount++;
      }
      if (currentSign >= 0i) {
        if (nextCount >= 8u) { return -1.0; }
        clipped[nextCount] = current; nextCount++;
      }
    }
    count = nextCount;
    if (count < 3u) { return 0.0; }
    for (var vertex = 0u; vertex < count; vertex++) { polygon[vertex] = clipped[vertex]; }
  }
  var area = 0.0;
  for (var vertex = 1u; vertex + 1u < count; vertex++) {
    area += studyWindowCross(polygon[vertex] - polygon[0], polygon[vertex + 1u] - polygon[0]);
  }
  return abs(area) * 0.5;
}
fn polygonTranslationOverlap(shift: vec2f) -> f32 {
  let origin = vec2f(studyWindowSummaryValue(0u), studyWindowSummaryValue(1u));
  let extent = vec2f(studyWindowSummaryValue(2u), studyWindowSummaryValue(3u)) - origin;
  if (abs(shift.x) >= extent.x || abs(shift.y) >= extent.y) { return 0.0; }
  if (all(shift == vec2f(0.0))) { return studyWindowSummaryValue(4u); }
  // Streaming binary reduction: each occupied bin contains 2^level intersections.
  // Even at the maximum 2^20 work budget, summation depth stays logarithmic.
  var areaBins: array<f32, 21>;
  var magnitudeBins: array<f32, 21>;
  var visited = 0u;
  for (var firstEdge = 0u; firstEdge < ${vertices}u; firstEdge++) {
    let first = studyWindowEdge(firstEdge);
    if (!studyWindowFinitePoint(first.xy) || !studyWindowFinitePoint(first.zw)) { continue; }
    var firstStart = first.xy - origin; var firstEnd = first.zw - origin;
    let firstSign = orientSign(vec2f(0.0), firstStart, firstEnd);
    if (firstSign == 2i) { return -2.0; }
    if (firstSign == 0i) { continue; }
    if (firstSign < 0i) { let saved = firstStart; firstStart = firstEnd; firstEnd = saved; }
    for (var secondEdge = 0u; secondEdge < ${vertices}u; secondEdge++) {
      let second = studyWindowEdge(secondEdge);
      if (!studyWindowFinitePoint(second.xy) || !studyWindowFinitePoint(second.zw)) { continue; }
      var secondStart = second.xy - origin; var secondEnd = second.zw - origin;
      let secondSign = orientSign(vec2f(0.0), secondStart, secondEnd);
      if (secondSign == 2i) { return -2.0; }
      if (secondSign == 0i) { continue; }
      if (visited >= ${edgePairBudget}u) { return -1.0; } visited++;
      if (secondSign < 0i) { let saved = secondStart; secondStart = secondEnd; secondEnd = saved; }
      let intersection = overlapTriangle(firstStart, firstEnd, shift, secondStart, secondEnd);
      if (intersection < 0.0 || !isFiniteFloat(intersection)) { return -2.0; }
      var contribution = f32(firstSign * secondSign) * intersection;
      var absoluteContribution = intersection;
      var occupied = visited - 1u;
      var level = 0u;
      while ((occupied & 1u) != 0u) {
        contribution += areaBins[level]; absoluteContribution += magnitudeBins[level];
        occupied >>= 1u; level++;
      }
      areaBins[level] = contribution; magnitudeBins[level] = absoluteContribution;
    }
  }
  var area = 0.0; var magnitude = 0.0;
  for (var level = 0u; level < 21u; level++) {
    if ((visited & (1u << level)) != 0u) { area += areaBins[level]; magnitude += magnitudeBins[level]; }
  }
  // Refuse an unresolved cancellation/sliver instead of inventing a large reciprocal weight.
  if (!isFiniteFloat(area) || area <= max(magnitude, studyWindowSummaryValue(4u)) * 0.00000762939453125) { return -2.0; }
  return area;
}
`;
}
