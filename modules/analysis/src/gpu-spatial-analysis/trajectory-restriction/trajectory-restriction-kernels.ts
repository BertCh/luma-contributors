// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
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
  const nodes: GPUCommandNode<Parameters>[] = [
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
${vertexCount ? `const NEXT = array<u32, ${vertexCount}>(${nextVertices.map(value => `${value}u`).join(',')});` : ''}
fn inside(point: vec2<f32>) -> bool {
${
  vertexCount
    ? `var parity = false;
for (var edge = 0u; edge < ${vertexCount}u; edge++) {
  let first = vertex(edge); let second = vertex(NEXT[edge]); let direction = second - first;
  if (abs(cross(point - first, direction)) <= 0.000001 * max(1.0, dot(direction, direction)) && all(point >= min(first, second) - vec2<f32>(0.000001)) && all(point <= max(first, second) + vec2<f32>(0.000001))) { return ${closedGeometry}; }
  if ((first.y > point.y) != (second.y > point.y)) {
    let crossing = first.x + (point.y - first.y) * direction.x / direction.y;
    if (point.x < crossing) { parity = !parity; }
  }
}
return parity;`
    : `let minimum = vec2<f32>(parameters[parametersOffset + 2u], parameters[parametersOffset + 3u]); let maximum = vec2<f32>(parameters[parametersOffset + 4u], parameters[parametersOffset + 5u]);
return all(point ${closedGeometry ? '>=' : '>'} minimum) && all(point ${closedGeometry ? '<=' : '<'} maximum);`
}
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
  if (duration > maximumGap) {
    status[statusOffset + row * SLOT_COUNT] = 8u;
    if (max(minimumTime, firstTime) < min(maximumTime, secondTime)) { unknownCount++; }
    if (timeIncluded(firstTime)) {
      supportCount++; status[statusOffset + row * SLOT_COUNT] |= 32u;
      let contained = inside(first); anyInside = anyInside || contained; allInside = allInside && contained;
      if (contained) { writeInterval(row, 0u, firstTime, 0.0, first, vec2<f32>(0.0), 0.0, 0.0, true, true); status[statusOffset + row * SLOT_COUNT] |= 16u; }
    }
    continue;
  }
  var lower = 0.0; var upper = select(1.0, 0.0, duration == 0.0);
  if (duration > 0.0) { lower = max(0.0, (minimumTime - firstTime) / duration); upper = min(1.0, (maximumTime - firstTime) / duration); }
  if (lower > upper || ((lower == upper || duration == 0.0) && !timeIncluded(firstTime + duration * lower))) { continue; }
  supportCount++; status[statusOffset + row * SLOT_COUNT] = 32u;
  var boundaries: array<f32, ${2 + Math.max(vertexCount * 2, 4)}>;
  boundaries[0] = lower; boundaries[1] = upper; var boundaryCount = 2u;
  ${
    vertexCount
      ? `for (var edge = 0u; edge < ${vertexCount}u; edge++) {
    let edgeStart = vertex(edge); let edgeDirection = vertex(NEXT[edge]) - edgeStart; let denominator = cross(direction, edgeDirection);
    if (denominator != 0.0) {
      let fraction = cross(edgeStart - first, edgeDirection) / denominator; let edgeFraction = cross(edgeStart - first, direction) / denominator;
      if (fraction > lower && fraction < upper && edgeFraction >= 0.0 && edgeFraction <= 1.0) { boundaries[boundaryCount] = fraction; boundaryCount++; }
    } else if (cross(edgeStart - first, direction) == 0.0 && dot(direction, direction) > 0.0) {
      for (var endpoint = 0u; endpoint < 2u; endpoint++) { let fraction = dot(edgeStart + edgeDirection * f32(endpoint) - first, direction) / dot(direction, direction); if (fraction > lower && fraction < upper) { boundaries[boundaryCount] = fraction; boundaryCount++; } }
    }
  }`
      : `for (var axis = 0u; axis < 2u; axis++) { if (direction[axis] != 0.0) { for (var side = 0u; side < 2u; side++) { let fraction = (parameters[parametersOffset + 2u + axis + 2u * side] - first[axis]) / direction[axis]; if (fraction > lower && fraction < upper) { boundaries[boundaryCount] = fraction; boundaryCount++; } } } }`
  }
  for (var positionIndex = 1u; positionIndex < boundaryCount; positionIndex++) {
    let value = boundaries[positionIndex]; var cursor = positionIndex;
    while (cursor > 0u && boundaries[cursor - 1u] > value) { boundaries[cursor] = boundaries[cursor - 1u]; cursor--; } boundaries[cursor] = value;
  }
  var uniqueCount = 1u;
  for (var cursor = 1u; cursor < boundaryCount; cursor++) { if (boundaries[cursor] != boundaries[uniqueCount - 1u]) { boundaries[uniqueCount] = boundaries[cursor]; uniqueCount++; } }
  var written = 0u; var fullyInside = true; var previousInterior = false;
  for (var cursor = 0u; cursor < uniqueCount; cursor++) {
    let begin = boundaries[cursor]; let beginTime = firstTime + duration * begin;
    let pointSupported = timeIncluded(beginTime); let pointInside = inside(first + direction * begin);
    let hasNext = cursor + 1u < uniqueCount && duration > 0.0;
    var nextInterior = false; var finish = begin;
    if (hasNext) { finish = boundaries[cursor + 1u]; nextInterior = inside(first + direction * ((begin + finish) * 0.5)); fullyInside = fullyInside && nextInterior; }
    if (pointSupported) { fullyInside = fullyInside && pointInside; }
    if (nextInterior) {
      let endIncluded = timeIncluded(firstTime + duration * finish) && inside(first + direction * finish);
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
  const output = props.output;
  if (
    output.compactIntervals &&
    output.compactPositions &&
    output.compactSources &&
    output.compactStatus &&
    output.compactSummary
  ) {
    // Stable streaming interval union: the count remains exact even when output capacity is exhausted.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-compact`,
        operation: 'GPUTrajectoryRestriction',
        variant: props.mergeIntervals ? 'merge-intervals' : 'compact-intervals',
        invocationCount: 1,
        workgroupSize: 2,
        bindings: [
          {name: 'intervals', view: output.intervals, type: 'f32', access: 'read'},
          {name: 'status', view: output.status, type: 'u32', access: 'read'},
          {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
          {name: 'result', view: output.compactIntervals, type: 'f32', access: 'read_write'},
          {name: 'sources', view: output.compactSources, type: 'u32', access: 'read_write'},
          {name: 'resultStatus', view: output.compactStatus, type: 'u32', access: 'read_write'},
          {name: 'summary', view: output.compactSummary, type: 'u32', access: 'read_write'}
        ],
        body: `var nanBits = 0x7fc00000u; let missing = bitcast<f32>(nanBits);
for (var row = 0u; row < ${output.compactIntervals.length}u; row++) {
  result[resultOffset + 2u * row] = missing; result[resultOffset + 2u * row + 1u] = missing;
  sources[sourcesOffset + 2u * row] = 0xffffffffu; sources[sourcesOffset + 2u * row + 1u] = 0xffffffffu; resultStatus[resultStatusOffset + row] = 0u;
}
var required = 0u; var upstreamOverflow = 0u; var track = 0u; var previousTrack = 0xffffffffu; var previousBegin = 0.0; var previousEnd = 0.0; var previousFlags = 0u;
for (var row = 0u; row < ${output.intervals.length}u; row++) {
  let flags = status[statusOffset + row]; if ((flags & 256u) != 0u) { upstreamOverflow++; }
  if ((flags & 3u) == 0u) { continue; }
  let sourceRow = row / ${slots}u;
  while (track + 1u < ${props.offsets.length - 1}u && sourceRow >= offsets[offsetsOffset + track + 1u]) { track++; }
  let begin = intervals[intervalsOffset + 2u * row]; let finish = intervals[intervalsOffset + 2u * row + 1u];
  let merge = ${props.mergeIntervals ? 'required > 0u && track == previousTrack && (begin < previousEnd || (begin == previousEnd && ((previousFlags & 128u) == 0u || (flags & 64u) == 0u)))' : 'false'};
  if (merge) {
    let extendsInterval = finish >= previousEnd;
    if (extendsInterval) { previousEnd = finish; previousFlags = (previousFlags & ~128u) | (flags & 128u); }
    previousFlags = (previousFlags & ~3u) | select(1u,2u,previousBegin == previousEnd);
    if (required <= ${output.compactIntervals.length}u) {
      let previous = required - 1u; result[resultOffset + 2u * previous + 1u] = previousEnd; resultStatus[resultStatusOffset + previous] = previousFlags;
      if (extendsInterval) { sources[sourcesOffset + 2u * previous + 1u] = row; }
    }
    continue;
  }
  previousTrack = track; previousBegin = begin; previousEnd = finish; previousFlags = flags;
  if (required < ${output.compactIntervals.length}u) {
    result[resultOffset + 2u * required] = begin; result[resultOffset + 2u * required + 1u] = finish;
    sources[sourcesOffset + 2u * required] = row; sources[sourcesOffset + 2u * required + 1u] = row; resultStatus[resultStatusOffset + required] = flags;
  }
  required++;
}
summary[summaryOffset] = required; summary[summaryOffset + 1u] = min(required, ${output.compactIntervals.length}u); summary[summaryOffset + 2u] = required - min(required, ${output.compactIntervals.length}u) + upstreamOverflow;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-compact-positions`,
        operation: 'GPUTrajectoryRestriction',
        variant: 'compact-positions-and-provenance',
        invocationCount: output.compactIntervals.length,
        bindings: [
          {name: 'positions', view: output.positions, type: 'f32', access: 'read'},
          {name: 'sources', view: output.compactSources, type: 'u32', access: 'read_write'},
          {name: 'result', view: output.compactPositions, type: 'f32', access: 'read_write'}
        ],
        body: `let first = sources[sourcesOffset + 2u * index]; let last = sources[sourcesOffset + 2u * index + 1u];
if (first == 0xffffffffu) { var bits = 0x7fc00000u; for (var component = 0u; component < 4u; component++) { result[resultOffset + 4u * index + component] = bitcast<f32>(bits); } return; }
result[resultOffset + 4u * index] = positions[positionsOffset + 4u * first]; result[resultOffset + 4u * index + 1u] = positions[positionsOffset + 4u * first + 1u];
result[resultOffset + 4u * index + 2u] = positions[positionsOffset + 4u * last + 2u]; result[resultOffset + 4u * index + 3u] = positions[positionsOffset + 4u * last + 3u];
sources[sourcesOffset + 2u * index] = first / ${slots}u; sources[sourcesOffset + 2u * index + 1u] = last / ${slots}u;`
      })
    );
  }
  return nodes;
}
