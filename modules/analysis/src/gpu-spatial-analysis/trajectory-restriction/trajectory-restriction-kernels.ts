// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createRestrictionCompactionNodes} from './trajectory-restriction-compaction';
import {EXACT_GEOMETRY_WGSL} from '../../analysis-core/kernels/exact-geometry-wgsl';
import {EXACT_ORIENTATION_WGSL} from '../../analysis-core/kernels/exact-orientation-wgsl';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {TIME_WORDS_WGSL} from '../../dataframe-analysis/time-window-filter/time-words';
import {getEffectiveMaximumGapWGSL} from '../trajectory-analysis/trajectory-common';
import type {GPUTrajectoryRestrictionResolvedProps} from './gpu-trajectory-restriction';

/** Word-mode time table: `[relative time, elapsed to the next row]` f32 pairs per row. @internal */
export type RestrictionTimeTable = {
  /** Two f32 per row, written by the node of {@link createRestrictionWordTimeNodes}. */
  view: GraphDataView<'float32'>;
};

/**
 * Time helpers of every restriction kernel. `rowTime(row)` is the time relative to the time origin
 * and `rowElapsed(second, first)` the per-step difference. In word mode both read the precomputed
 * table, so the restriction kernels stay inside the core binding limit; steps were subtracted
 * exactly before the f32 conversion. @internal
 */
export function getRestrictionTimeSource(
  props: GPUTrajectoryRestrictionResolvedProps,
  timeTable?: RestrictionTimeTable
): {
  bindings: WGSLKernelBinding[];
  source: string;
  suffix: string;
} {
  if (!timeTable) {
    return {
      bindings: [{name: 'times', view: props.times, type: 'f32', access: 'read'}],
      source: `fn rowTime(row: u32) -> f32 { return times[timesOffset + row]; }
fn rowElapsed(second: u32, first: u32) -> f32 { return times[timesOffset + second] - times[timesOffset + first]; }`,
      suffix: ''
    };
  }
  return {
    bindings: [{name: 'timeTable', view: timeTable.view, type: 'f32', access: 'read'}],
    // `rowElapsed` is only used for a row and its successor within a track (0 at the track's last row).
    source: `fn rowTime(row: u32) -> f32 { return timeTable[timeTableOffset + 2u * row]; }
fn rowElapsed(second: u32, first: u32) -> f32 { _ = second; return timeTable[timeTableOffset + 2u * first + 1u]; }`,
    suffix: '-words'
  };
}

/**
 * Builds the word-time table of restriction word mode: per row the f32 time relative to
 * `timeOrigin` (exact words subtracted, then rounded) and the exact step to the next row of its
 * track (0 at the last row). @internal
 */
export function createRestrictionWordTimeNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUTrajectoryRestrictionResolvedProps
): {timeTable: RestrictionTimeTable; nodes: GPUCommandNode<Parameters>[]} {
  const rowCount = props.positions.length;
  const view = createTransientView(graph, `${id}-word-times`, 'float32', Math.max(2 * rowCount, 1));
  const nodes = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-word-times`,
      operation: 'GPUTrajectoryRestriction',
      variant: 'word-times',
      invocationCount: props.offsets.length - 1,
      workgroupSize: 32,
      bindings: [
        {name: 'times', view: props.times, type: 'u32', access: 'read'},
        {name: 'timeOrigin', view: props.timeOrigin as GraphDataView, type: 'u32', access: 'read'},
        {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
        {name: 'timeTable', view, type: 'f32', access: 'read_write'}
      ],
      declarations: `const OBSERVATION_COUNT: u32 = ${rowCount}u;
${TIME_WORDS_WGSL}
fn getTimeWords(row: u32) -> vec2<u32> { return vec2<u32>(times[timesOffset + 2u * row], times[timesOffset + 2u * row + 1u]); }`,
      body: `let start = min(offsets[offsetsOffset + index], OBSERVATION_COUNT); let end = min(offsets[offsetsOffset + index + 1u], OBSERVATION_COUNT);
let origin = vec2<u32>(timeOrigin[timeOriginOffset], timeOrigin[timeOriginOffset + 1u]);
for (var row = start; row < end; row++) {
  timeTable[timeTableOffset + 2u * row] = timeWordsDifferenceToF32(getTimeWords(row), origin);
  timeTable[timeTableOffset + 2u * row + 1u] = select(0.0, timeWordsDifferenceToF32(getTimeWords(row + 1u), getTimeWords(row)), row + 1u < end);
}`
    })
  ];
  return {timeTable: {view}, nodes};
}

/** General planar interval restriction. Static ring topology is distinct from runtime coordinates. */
export function createTrajectoryRestrictionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUTrajectoryRestrictionResolvedProps,
  timeTable?: RestrictionTimeTable
): GPUCommandNode<Parameters>[] {
  const rings = props.polygonRingVertexCounts ?? [];
  const vertexCount = rings.reduce((total, count) => total + count, 0);
  const slots = props.intervalsPerSegment ?? 1;
  // The time window's two ends plus up to two boundary parameters per polygon edge (four box sides).
  const boundaryCapacity = 2 + Math.max(vertexCount * 2, 4);
  const closedGeometry = props.geometryBoundary !== 'open';
  const closedStart = props.timeBoundary !== 'open' && props.timeBoundary !== 'right-closed';
  const closedEnd = props.timeBoundary !== 'open' && props.timeBoundary !== 'left-closed';
  let ringStart = 0;
  const nextVertices = rings.flatMap(count => {
    const start = ringStart;
    ringStart += count;
    return Array.from({length: count}, (_, vertex) => start + ((vertex + 1) % count));
  });
  const time = getRestrictionTimeSource(props, timeTable);
  // Sorts the boundary parameters, drops repeats, and sets the number of classification queries.
  const sortBoundariesSource = `for (var positionIndex = 1u; positionIndex < boundaryCount; positionIndex++) {
    let value = boundaries[positionIndex]; var cursor = positionIndex;
    while (cursor > 0u && boundaries[cursor - 1u] > value) { boundaries[cursor] = boundaries[cursor - 1u]; cursor--; } boundaries[cursor] = value;
  }
  for (var cursor = 1u; cursor < boundaryCount; cursor++) { if (boundaries[cursor] != boundaries[uniqueCount - 1u]) { boundaries[uniqueCount] = boundaries[cursor]; uniqueCount++; } }
  queryCount = select(2u * uniqueCount - 1u, 1u, isGap || duration <= 0.0);`;
  // Sets queryPoint and queryOnBoundary for 'query': boundary 'query / 2', or the midpoint of the
  // piece after it when the query is odd.
  const querySetupSource = `{
      let slot = query / 2u; var fraction = boundaries[slot];
      var onBoundary = ${vertexCount ? 'fraction > lower && fraction < upper' : 'false'};
      if ((query & 1u) == 1u) {
        let finish = boundaries[slot + 1u]; let betweenCrossings = onBoundary && finish < upper;
        let beginPoint = first + direction * fraction; let finishPoint = first + direction * finish;
        fraction = (fraction + finish) * 0.5; let middlePoint = first + direction * fraction;
        onBoundary = betweenCrossings && all((middlePoint == beginPoint) | (middlePoint == finishPoint));
        ${vertexCount ? 'for (var range = 0u; range < collinearCount; range++) { if (fraction >= collinearRanges[range].x && fraction <= collinearRanges[range].y) { onBoundary = true; } }' : ''}
      }
      queryPoint = first + direction * fraction;
      if (fraction <= 0.0) { queryPoint = first; }
      if (fraction >= 1.0) { queryPoint = second; }
      queryOnBoundary = onBoundary;
    }`;
  const nodes: GPUCommandNode<Parameters>[] = [
    // Geometry only visits current track ranges; clear all presence/loss bits each submission.
    createFillNode<Parameters>(graph, {
      id: `${id}-interval-status-clear`,
      operation: 'GPUTrajectoryRestriction',
      view: props.output.status,
      type: 'u32',
      value: '0u'
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-intervals`,
      operation: 'GPUTrajectoryRestriction',
      variant: `${vertexCount ? 'polygon-intervals' : 'rectangle-intervals'}${time.suffix}`,
      invocationCount: props.offsets.length - 1,
      workgroupSize: 32,
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        ...time.bindings,
        {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
        {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
        {name: 'intervals', view: props.output.intervals, type: 'f32', access: 'read_write'},
        {name: 'clipped', view: props.output.positions, type: 'f32', access: 'read_write'},
        {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'},
        {name: 'relationships', view: props.output.relationships, type: 'u32', access: 'read_write'}
      ],
      declarations: `const OBSERVATION_COUNT = ${props.positions.length}u;
const SLOT_COUNT = ${slots}u;
${time.source}
fn finite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
fn cross(first: vec2<f32>, second: vec2<f32>) -> f32 { return first.x * second.y - first.y * second.x; }
fn position(row: u32) -> vec2<f32> { return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]); }
fn vertex(row: u32) -> vec2<f32> { return vec2<f32>(parameters[parametersOffset + 7u + 2u * row], parameters[parametersOffset + 8u + 2u * row]); }
${vertexCount ? `const NEXT = array<u32, ${vertexCount}>(${nextVertices.map(value => `${value}u`).join(',')});\n${EXACT_ORIENTATION_WGSL}\n${EXACT_GEOMETRY_WGSL}` : ''}
const CLOSED_GEOMETRY: bool = ${closedGeometry};
${
  vertexCount
    ? ''
    : `fn inside(point: vec2<f32>) -> bool {
let minimum = vec2<f32>(parameters[parametersOffset + 2u], parameters[parametersOffset + 3u]); let maximum = vec2<f32>(parameters[parametersOffset + 4u], parameters[parametersOffset + 5u]);
return all(point ${closedGeometry ? '>=' : '>'} minimum) && all(point ${closedGeometry ? '<=' : '<'} maximum);
}`
}
fn timeIncluded(time: f32) -> bool { return time ${closedStart ? '>=' : '>'} parameters[parametersOffset] && time ${closedEnd ? '<=' : '<'} parameters[parametersOffset + 1u]; }
fn writeInterval(row: u32, slot: u32, firstTime: f32, duration: f32, first: vec2<f32>, direction: vec2<f32>, lower: f32, upper: f32, includeStart: bool, includeEnd: bool) {
  if (slot >= SLOT_COUNT) { status[statusOffset + row * SLOT_COUNT] |= 256u; return; }
  let outputRow = row * SLOT_COUNT + slot;
  intervals[intervalsOffset + 2u * outputRow] = firstTime + duration * lower; intervals[intervalsOffset + 2u * outputRow + 1u] = firstTime + duration * upper;
  let begin = first + direction * lower; let end = first + direction * upper;
  clipped[clippedOffset + 4u * outputRow] = begin.x; clipped[clippedOffset + 4u * outputRow + 1u] = begin.y; clipped[clippedOffset + 4u * outputRow + 2u] = end.x; clipped[clippedOffset + 4u * outputRow + 3u] = end.y;
  status[statusOffset + outputRow] |= select(1u, 2u, duration == 0.0 || lower == upper) | select(64u, 0u, includeStart) | select(128u, 0u, includeEnd);
}`,
      body: `let start = min(offsets[offsetsOffset + index], OBSERVATION_COUNT); let end = min(offsets[offsetsOffset + index + 1u], OBSERVATION_COUNT);
let minimumTime = parameters[parametersOffset]; let maximumTime = parameters[parametersOffset + 1u]; let maximumGap = ${getEffectiveMaximumGapWGSL('parameters[parametersOffset + 6u]')};
var validParameters = finite(minimumTime) && finite(maximumTime) && minimumTime <= maximumTime && maximumGap > 0.0;
for (var parameter = 2u; parameter < ${7 + vertexCount * 2}u; parameter++) { if (parameter != 6u) { validParameters = validParameters && finite(parameters[parametersOffset + parameter]); } }
${vertexCount ? '' : 'validParameters = validParameters && parameters[parametersOffset + 2u] <= parameters[parametersOffset + 4u] && parameters[parametersOffset + 3u] <= parameters[parametersOffset + 5u];'}
var nanBits = 0x7fc00000u; let missing = bitcast<f32>(nanBits);
var anyInside = false; var allInside = true; var supportCount = 0u; var unknownCount = 0u;
for (var row = start; row < end; row++) {
  for (var slot = 0u; slot < SLOT_COUNT; slot++) {
    let outputRow = row * SLOT_COUNT + slot; status[statusOffset + outputRow] = 0u;
    for (var component = 0u; component < 2u; component++) { intervals[intervalsOffset + 2u * outputRow + component] = missing; }
    for (var component = 0u; component < 4u; component++) { clipped[clippedOffset + 4u * outputRow + component] = missing; }
  }
  let next = min(row + 1u, end - 1u); let first = position(row); let second = position(next); let direction = second - first;
  let firstTime = rowTime(row); let secondTime = rowTime(next); let duration = rowElapsed(next, row);
  if (!validParameters || !finite(firstTime) || !finite(secondTime) || duration < 0.0 || !finite(first.x) || !finite(first.y) || !finite(second.x) || !finite(second.y) || (duration == 0.0 && any(direction != vec2<f32>(0.0)))) { status[statusOffset + row * SLOT_COUNT] = 4u; unknownCount++; continue; }
  if (secondTime < minimumTime || firstTime > maximumTime) { continue; }
  let isGap = duration > maximumGap;
  var lower = 0.0; var upper = select(1.0, 0.0, duration == 0.0);
  var boundaries: array<f32, ${boundaryCapacity}>; var uniqueCount = 1u; var boundaryCount = 1u;
  ${vertexCount ? `var collinearRanges: array<vec2<f32>, ${vertexCount}>; var collinearCount = 0u;` : ''}
  if (isGap) {
    status[statusOffset + row * SLOT_COUNT] = 8u;
    if (max(minimumTime, firstTime) < min(maximumTime, secondTime)) { unknownCount++; }
    if (!timeIncluded(firstTime)) { continue; }
    supportCount++; status[statusOffset + row * SLOT_COUNT] |= 32u;
    // Only the fix itself is classified.
    upper = 0.0; boundaries[0] = 0.0;
  } else {
    if (duration > 0.0) { lower = max(0.0, (minimumTime - firstTime) / duration); upper = min(1.0, (maximumTime - firstTime) / duration); }
    if (lower > upper || ((lower == upper || duration == 0.0) && !timeIncluded(firstTime + duration * lower))) { continue; }
    supportCount++; status[statusOffset + row * SLOT_COUNT] = 32u;
    boundaries[0] = lower; boundaries[1] = upper; boundaryCount = 2u;
    ${
      vertexCount
        ? ''
        : `for (var axis = 0u; axis < 2u; axis++) { if (direction[axis] != 0.0) { for (var side = 0u; side < 2u; side++) { let fraction = (parameters[parametersOffset + 2u + axis + 2u * side] - first[axis]) / direction[axis]; if (fraction > lower && fraction < upper) { boundaries[boundaryCount] = fraction; boundaryCount++; } } } }`
    }
  }
  // Classify every boundary point (even query) and the midpoint of every piece between two of
  // them (odd query) in one place. A fix is tested as it was read. A polygon crossing is on the
  // boundary because it was constructed there, not because its rounded coordinates say so; so is
  // a piece that runs along an edge, and a piece between two crossings that has no float32 point
  // of its own. Everything else is the exact test of the computed point.
  var containment: array<u32, ${2 * boundaryCapacity}>; var queryCount = 1u;
  var queryPoint = first; var queryOnBoundary = false;
  ${
    vertexCount
      ? `// One loop holds the only call of exactOrientSigns (the orientation code is large and is
  // inlined at every call). Jobs 0 .. V - 1 decide, edge by edge, whether the segment meets the
  // edge, from the exact sides of the two fixes against the edge and of the edge's two ends
  // against the segment; float32 only places a crossing that exists. The following jobs are the
  // exact even-odd test of each query point against each edge, with the rule of
  // exactProbeCrossing (the shared predicate GPUZoneEvents uses): on the boundary only when the
  // exact orientation is zero inside the edge's box.
  var job = select(0u, ${vertexCount}u, isGap || all(direction == vec2<f32>(0.0)));
  var jobCount = ${vertexCount + 1}u; var winding = 0; var boundaryHit = false;
  loop {
    if (job >= jobCount) { break; }
    if (job == ${vertexCount}u) {
      ${sortBoundariesSource}
      jobCount = ${vertexCount}u + queryCount * ${vertexCount}u;
    }
    let edge = job % ${vertexCount}u; let isProbe = job >= ${vertexCount}u;
    let edgeStart = vertex(edge); let edgeEnd = vertex(NEXT[edge]);
    var signCount = 4u; var straddles = false; var inBox = false;
    var starts = array<vec2<f32>, 4>(edgeStart, edgeStart, first, first);
    var ends = array<vec2<f32>, 4>(edgeEnd, edgeEnd, second, second);
    var points = array<vec2<f32>, 4>(first, second, edgeStart, edgeEnd);
    if (isProbe) {
      let query = job / ${vertexCount}u - 1u;
      if (edge == 0u) {
        ${querySetupSource}
        winding = 0; boundaryHit = false;
      }
      signCount = 0u;
      if (!queryOnBoundary) {
        straddles = (edgeStart.y >= queryPoint.y) != (edgeEnd.y >= queryPoint.y);
        inBox = all(queryPoint >= min(edgeStart, edgeEnd)) && all(queryPoint <= max(edgeStart, edgeEnd));
        if (straddles || inBox) {
          signCount = 1u;
          starts = array<vec2<f32>, 4>(edgeStart, edgeStart, edgeStart, edgeStart);
          ends = array<vec2<f32>, 4>(edgeEnd, edgeEnd, edgeEnd, edgeEnd);
          points = array<vec2<f32>, 4>(queryPoint, queryPoint, queryPoint, queryPoint);
        }
      }
    } else if (all(edgeStart == edgeEnd)) {
      // A repeated vertex is not an edge; its neighbours carry the crossing.
      signCount = 0u;
    }
    let exact = exactOrientSigns(signCount, starts, ends, points);
    if (!isProbe) {
      if (signCount == 4u) {
        // x, y: the two fixes against the edge. z, w: the edge's two ends against the segment.
        let sides = exact.signs; let lengthSquared = dot(direction, direction);
        if (sides.x == 0 && sides.y == 0) {
          // The segment runs along this edge's line: between the edge's two ends it is on the boundary.
          if (lengthSquared > 0.0) {
            let startFraction = dot(edgeStart - first, direction) / lengthSquared; let endFraction = dot(edgeEnd - first, direction) / lengthSquared;
            if (startFraction > lower && startFraction < upper) { boundaries[boundaryCount] = startFraction; boundaryCount++; }
            if (endFraction > lower && endFraction < upper) { boundaries[boundaryCount] = endFraction; boundaryCount++; }
            collinearRanges[collinearCount] = vec2<f32>(min(startFraction, endFraction), max(startFraction, endFraction)); collinearCount++;
          }
        } else if (sides.x * sides.y <= 0 && sides.z * sides.w <= 0) {
          // The segment and the edge share exactly one point. A fix on the edge is the query at
          // 0 or 1; a vertex on the segment is placed by its own projection, so the two edges
          // that share it give the same parameter; otherwise the float32 quotient places it.
          var fraction = 0.5;
          if (sides.x == 0) { fraction = 0.0; }
          else if (sides.y == 0) { fraction = 1.0; }
          else if (sides.z == 0 && lengthSquared > 0.0) { fraction = dot(edgeStart - first, direction) / lengthSquared; }
          else if (sides.w == 0 && lengthSquared > 0.0) { fraction = dot(edgeEnd - first, direction) / lengthSquared; }
          else {
            let edgeDirection = edgeEnd - edgeStart; let denominator = cross(direction, edgeDirection);
            if (denominator != 0.0) { fraction = cross(edgeStart - first, edgeDirection) / denominator; }
          }
          fraction = clamp(fraction, 0.0, 1.0);
          if (fraction > lower && fraction < upper) { boundaries[boundaryCount] = fraction; boundaryCount++; }
        }
      }
    } else {
      if (signCount == 1u) {
        if (inBox && exact.signs.x == 0) { boundaryHit = true; }
        let side = sosPointSide(exact.signs.x, edgeStart, edgeEnd);
        if (straddles) {
          let upward = edgeEnd.y > edgeStart.y;
          if (upward && side > 0) { winding += 1; }
          if (!upward && side < 0) { winding -= 1; }
        }
      }
      if (edge == ${vertexCount - 1}u) {
        var contained = CLOSED_GEOMETRY;
        if (!queryOnBoundary && !boundaryHit) { contained = (winding & 1) != 0; }
        containment[job / ${vertexCount}u - 1u] = select(0u, 1u, contained);
      }
    }
    job++;
  }`
      : `${sortBoundariesSource}
  for (var query = 0u; query < queryCount; query++) {
    ${querySetupSource}
    containment[query] = select(0u, 1u, inside(queryPoint));
  }`
  }
  if (isGap) {
    let contained = containment[0] != 0u; anyInside = anyInside || contained; allInside = allInside && contained;
    if (contained) { writeInterval(row, 0u, firstTime, 0.0, first, vec2<f32>(0.0), 0.0, 0.0, true, true); status[statusOffset + row * SLOT_COUNT] |= 16u; }
    continue;
  }
  var written = 0u; var fullyInside = true; var previousInterior = false;
  for (var cursor = 0u; cursor < uniqueCount; cursor++) {
    let begin = boundaries[cursor]; let beginTime = firstTime + duration * begin;
    let pointSupported = timeIncluded(beginTime); let pointInside = containment[2u * cursor] != 0u;
    let hasNext = cursor + 1u < uniqueCount && duration > 0.0;
    var nextInterior = false; var finish = begin;
    if (hasNext) { finish = boundaries[cursor + 1u]; nextInterior = containment[2u * cursor + 1u] != 0u; fullyInside = fullyInside && nextInterior; }
    if (pointSupported) { fullyInside = fullyInside && pointInside; }
    if (nextInterior) {
      let endIncluded = timeIncluded(firstTime + duration * finish) && containment[2u * cursor + 2u] != 0u;
      writeInterval(row, written, firstTime, duration, first, direction, begin, finish, pointSupported && pointInside, endIncluded); written++; anyInside = true;
    } else if (pointSupported && pointInside && !previousInterior) {
      writeInterval(row, written, firstTime, duration, first, direction, begin, begin, true, true); written++; anyInside = true;
    }
    previousInterior = nextInterior;
  }
  allInside = allInside && fullyInside;
  if (fullyInside) { status[statusOffset + row * SLOT_COUNT] |= 16u; }
}
var ever = select(0u, 1u, anyInside); var always = select(0u, 1u, allInside);
if (supportCount == 0u || unknownCount > 0u) { if (!anyInside) { ever = 0xffffffffu; } if (allInside) { always = 0xffffffffu; } }
relationships[relationshipsOffset + 4u * index] = ever; relationships[relationshipsOffset + 4u * index + 1u] = always; relationships[relationshipsOffset + 4u * index + 2u] = supportCount; relationships[relationshipsOffset + 4u * index + 3u] = unknownCount;`
    })
  ];
  nodes.push(...createRestrictionCompactionNodes(graph, id, props));
  return nodes;
}
