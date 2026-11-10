// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getEffectiveMaximumGapWGSL} from '../trajectory-analysis/trajectory-common';
import type {GPUTrajectoryRestrictionResolvedProps} from './gpu-trajectory-restriction';
import {
  getRestrictionTimeSource,
  type RestrictionTimeTable
} from './trajectory-restriction-kernels';

/** Pairs of track indices of the pairwise mode, from a GPU view or compiled in. @internal */
export type RestrictionPairSource = {
  /** Compile-time pairs (the `pairwiseTracks` convenience); a data view is preferred. */
  staticPairs?: readonly (readonly [number, number])[];
  /** Data-driven pairs, e.g. `GPUTrajectoryEncounters` `output.pairs.ids` and `.partners`. */
  first?: GraphDataView<'uint32'>;
  second?: GraphDataView<'uint32'>;
  /** One row: pairs at and after this index are inactive. */
  count?: GraphDataView<'uint32'>;
  /** Number of pair rows (the capacity). */
  capacity: number;
};

/**
 * Pairwise within-distance intervals by a two-pointer time merge of the segments of both tracks.
 * One thread per pair walks the two tracks once: `O(nA + nB)` plus the number of overlapping
 * segment pairs, instead of the `O(nA * nB)` nested loop. Tracks must be sorted by time.
 *
 * Segments touch at their end rows, so a segment can overlap several segments of the other track;
 * the second pointer only moves forward past segments that end before the current one starts, and
 * the pairs come out in the order of the nested loop (`rowA`, then `rowB`).
 *
 * Invalid segments (non-finite, negative duration, movement without duration) never overlap
 * anything; each one adds one to the unknown count and sets status bit 4 of the pair's first slot.
 * Invalid parameters set that bit once. Slots are cleared by a separate dispatch over all output
 * rows, so no thread loops over the interval capacity.
 * @internal
 */
export function createTrajectoryPairwiseMergeNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUTrajectoryRestrictionResolvedProps,
  source: RestrictionPairSource,
  timeTable?: RestrictionTimeTable
): GPUCommandNode<Parameters>[] {
  const capacity = props.intervalsPerPair!;
  if (!source.capacity) return [];
  const closedStart = props.timeBoundary !== 'open' && props.timeBoundary !== 'right-closed';
  const closedEnd = props.timeBoundary !== 'open' && props.timeBoundary !== 'left-closed';
  const closedGeometry = props.geometryBoundary !== 'open';
  const time = getRestrictionTimeSource(props, timeTable);
  const trackCount = props.offsets.length - 1;
  // The pair list (compiled in or from GPU views) is resolved into a table of track ranges
  // `[startA, endA, startB, endB]` first, so the merge kernel stays inside the storage binding
  // limit. 0xffffffff in the first word marks an inactive row, 0xfffffffe a track out of range.
  const pairTable = createTransientView(graph, `${id}-pair-table`, 'uint32', 4 * source.capacity);
  const resolveNodes = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-pairwise-resolve`,
      operation: 'GPUTrajectoryRestriction',
      variant: `pairwise-resolve-pairs${source.staticPairs ? '-static' : ''}${source.count ? '-counted' : ''}`,
      invocationCount: source.capacity,
      bindings: [
        {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
        ...(source.staticPairs
          ? []
          : [
              {
                name: 'pairFirst',
                view: source.first!,
                type: 'u32' as const,
                access: 'read' as const
              },
              {
                name: 'pairSecond',
                view: source.second!,
                type: 'u32' as const,
                access: 'read' as const
              },
              ...(source.count
                ? [
                    {
                      name: 'pairCount',
                      view: source.count,
                      type: 'u32' as const,
                      access: 'read' as const
                    }
                  ]
                : [])
            ]),
        {name: 'pairTable', view: pairTable, type: 'u32', access: 'read_write'}
      ],
      declarations: source.staticPairs
        ? `const PAIRS = array<vec2<u32>, ${source.staticPairs.length}>(${source.staticPairs.map(pair => `vec2<u32>(${pair[0]}u,${pair[1]}u)`).join(',')});`
        : '',
      body: `${
        source.staticPairs
          ? 'let isLive = true; let tracks = PAIRS[index];'
          : `let isLive = ${source.count ? 'index < pairCount[pairCountOffset]' : 'true'}; let tracks = vec2<u32>(pairFirst[pairFirstOffset + index], pairSecond[pairSecondOffset + index]);`
      }
var row = vec4<u32>(0xffffffffu);
if (isLive) {
  row = vec4<u32>(0xfffffffeu);
  if (tracks.x < ${trackCount}u && tracks.y < ${trackCount}u) {
    row = vec4<u32>(min(offsets[offsetsOffset + tracks.x], ${props.positions.length}u), min(offsets[offsetsOffset + tracks.x + 1u], ${props.positions.length}u), min(offsets[offsetsOffset + tracks.y], ${props.positions.length}u), min(offsets[offsetsOffset + tracks.y + 1u], ${props.positions.length}u));
  }
}
for (var component = 0u; component < 4u; component++) { pairTable[pairTableOffset + 4u * index + component] = row[component]; }`
    })
  ];
  const pairDeclarations = `fn pairActive(pair: u32) -> bool { return pairTable[pairTableOffset + 4u * pair] != 0xffffffffu; }
fn pairInRange(pair: u32) -> bool { return pairTable[pairTableOffset + 4u * pair] != 0xfffffffeu; }`;
  return [
    ...resolveNodes,
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-pairwise-clear`,
      operation: 'GPUTrajectoryRestriction',
      variant: 'pairwise-clear-slots',
      invocationCount: source.capacity * capacity,
      workgroupSize: 64,
      bindings: [
        {name: 'intervals', view: props.output.intervals, type: 'f32', access: 'read_write'},
        {name: 'clipped', view: props.output.positions, type: 'f32', access: 'read_write'},
        {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
      ],
      body: `var nanBits = 0x7fc00000u; let missing = bitcast<f32>(nanBits);
status[statusOffset + index] = 0u;
for (var component = 0u; component < 2u; component++) { intervals[intervalsOffset + 2u * index + component] = missing; }
for (var component = 0u; component < 4u; component++) { clipped[clippedOffset + 4u * index + component] = missing; }`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-pairwise-merge`,
      operation: 'GPUTrajectoryRestriction',
      variant: `pairwise-time-merge${time.suffix}`,
      invocationCount: source.capacity,
      workgroupSize: 32,
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        ...time.bindings,
        {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
        {name: 'pairTable', view: pairTable, type: 'u32', access: 'read'},
        {name: 'intervals', view: props.output.intervals, type: 'f32', access: 'read_write'},
        {name: 'clipped', view: props.output.positions, type: 'f32', access: 'read_write'},
        {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'},
        {name: 'relationships', view: props.output.relationships, type: 'u32', access: 'read_write'}
      ],
      declarations: `${pairDeclarations}
${time.source}
fn finite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
fn position(row: u32) -> vec2<f32> { return vec2<f32>(positions[positionsOffset + 2u * row],positions[positionsOffset + 2u * row + 1u]); }
fn timeIncluded(time: f32) -> bool { return time ${closedStart ? '>=' : '>'} parameters[parametersOffset] && time ${closedEnd ? '<=' : '<'} parameters[parametersOffset + 1u]; }
// One segment of a track: from \`row\` to the next row, or a singleton at the last row and before a gap.
struct Segment {
  time: f32, finish: f32, rawFinish: f32, duration: f32, position: vec2<f32>, movement: vec2<f32>, gap: bool, invalid: bool
}
fn loadSegment(row: u32, end: u32, maximumGap: f32) -> Segment {
  let next = min(row + 1u, end - 1u);
  let time = rowTime(row); let rawFinish = rowTime(next); let rawDuration = rowElapsed(next, row);
  let gap = rawDuration > maximumGap;
  let finish = select(rawFinish, time, gap); let duration = select(rawDuration, 0.0, gap);
  let start = position(row); let movement = select(position(next) - start, vec2<f32>(0.0), gap);
  let invalid = !finite(time) || !finite(finish) || duration < 0.0 || !finite(start.x) || !finite(start.y) || !finite(movement.x) || !finite(movement.y) || (duration == 0.0 && any(movement != vec2<f32>(0.0)));
  return Segment(time, finish, rawFinish, duration, start, movement, gap, invalid);
}`,
      body: `let base = index * ${capacity}u;
var anyInside = false; var allInside = true; var support = 0u; var unknown = 0u; var written = 0u;
let minimumTime = parameters[parametersOffset]; let maximumTime = parameters[parametersOffset + 1u]; let distanceLimit = parameters[parametersOffset + 2u]; let maximumGap = ${getEffectiveMaximumGapWGSL('parameters[parametersOffset + 6u]')};
let validParameters = finite(minimumTime) && finite(maximumTime) && minimumTime <= maximumTime && finite(distanceLimit) && distanceLimit >= 0.0 && maximumGap > 0.0;
if (!pairActive(index)) {
  relationships[relationshipsOffset + 4u * index] = 0xffffffffu; relationships[relationshipsOffset + 4u * index + 1u] = 0xffffffffu; relationships[relationshipsOffset + 4u * index + 2u] = 0u; relationships[relationshipsOffset + 4u * index + 3u] = 0u;
  return;
}
if (!pairInRange(index) || !validParameters) {
  status[statusOffset + base] |= 4u;
  relationships[relationshipsOffset + 4u * index] = 0xffffffffu; relationships[relationshipsOffset + 4u * index + 1u] = 0xffffffffu; relationships[relationshipsOffset + 4u * index + 2u] = 0u; relationships[relationshipsOffset + 4u * index + 3u] = 1u;
  return;
}
let startA = pairTable[pairTableOffset + 4u * index]; let endA = pairTable[pairTableOffset + 4u * index + 1u];
let startB = pairTable[pairTableOffset + 4u * index + 2u]; let endB = pairTable[pairTableOffset + 4u * index + 3u];
// Invalid segments never overlap anything; count each once.
for (var rowA = startA; rowA < endA; rowA++) { if (loadSegment(rowA, endA, maximumGap).invalid) { unknown++; } }
for (var rowB = startB; rowB < endB; rowB++) { if (loadSegment(rowB, endB, maximumGap).invalid) { unknown++; } }
if (unknown > 0u) { status[statusOffset + base] |= 4u; }
// Two-pointer merge: \`firstB\` is the first segment of B that does not end before the current segment of A starts.
var firstB = startB;
for (var rowA = startA; rowA < endA; rowA++) {
  let segmentA = loadSegment(rowA, endA, maximumGap);
  if (segmentA.invalid) { continue; }
  if (segmentA.time > maximumTime) { break; }
  if (segmentA.rawFinish < minimumTime) { continue; }
  for (; firstB < endB; firstB++) {
    let segmentB = loadSegment(firstB, endB, maximumGap);
    if (!segmentB.invalid && segmentB.rawFinish >= segmentA.time) { break; }
  }
  for (var rowB = firstB; rowB < endB; rowB++) {
    let segmentB = loadSegment(rowB, endB, maximumGap);
    if (segmentB.invalid) { continue; }
    if (segmentB.time > segmentA.rawFinish || segmentB.time > maximumTime) { break; }
    let timeA = segmentA.time; let timeB = segmentB.time;
    let finishA = segmentA.finish; let finishB = segmentB.finish;
    let durationA = segmentA.duration; let durationB = segmentB.duration;
    let positionA = segmentA.position; let movementA = segmentA.movement; let movementB = segmentB.movement;
    let lower = max(minimumTime, max(timeA, timeB)); let upper = min(maximumTime, min(finishA, finishB));
    if ((segmentA.gap || segmentB.gap) && lower < min(maximumTime, min(segmentA.rawFinish, segmentB.rawFinish))) { unknown++; status[statusOffset + base] |= 8u; }
    if (lower > upper || (lower == upper && !timeIncluded(lower))) { continue; }
    support++;
    var velocityA = vec2<f32>(0.0); var velocityB = vec2<f32>(0.0); if (durationA > 0.0) { velocityA = movementA / durationA; } if (durationB > 0.0) { velocityB = movementB / durationB; }
    let firstA = positionA + velocityA * (lower - timeA); let firstPointB = segmentB.position + velocityB * (lower - timeB); let relative = firstA - firstPointB; let velocity = velocityA - velocityB;
    let quadratic = dot(velocity, velocity); let projection = dot(relative, velocity); let limitSquared = distanceLimit * distanceLimit;
    if (!finite(quadratic) || !finite(projection) || !finite(limitSquared)) { unknown++; status[statusOffset + base] |= 4u; continue; }
    var begin = 0.0; var finish = upper - lower; var matches = true;
    if (quadratic == 0.0) {
      let distanceSquared = dot(relative, relative);
      if (!finite(distanceSquared)) { unknown++; status[statusOffset + base] |= 4u; continue; }
      matches = distanceSquared ${closedGeometry ? '<=' : '<'} limitSquared;
    } else {
      // Evaluate distance at closest approach before subtracting the radius. The conventional
      // b*b-4*a*c discriminant loses narrow encounters when absolute separation is large.
      let center = -projection / quadratic; let closest = relative + velocity * center;
      let remainingSquared = limitSquared - dot(closest, closest);
      if (!finite(center) || !finite(remainingSquared)) { unknown++; status[statusOffset + base] |= 4u; continue; }
      if (remainingSquared < 0.0) { matches = false; }
      else {
        let radius = sqrt(remainingSquared / quadratic);
        if (!finite(radius)) { unknown++; status[statusOffset + base] |= 4u; continue; }
        begin = max(begin, center - radius); finish = min(finish, center + radius); matches = begin <= finish;
      }
    }
    let beginTime = lower + begin; let finishTime = lower + finish;
    let beginInside = dot(relative + velocity * begin, relative + velocity * begin) ${closedGeometry ? '<=' : '<'} distanceLimit * distanceLimit;
    let finishInside = dot(relative + velocity * finish, relative + velocity * finish) ${closedGeometry ? '<=' : '<'} distanceLimit * distanceLimit;
    let includeStart = timeIncluded(beginTime) && beginInside; let includeEnd = timeIncluded(finishTime) && finishInside;
    if (begin == finish && !(includeStart && includeEnd)) { matches = false; }
    let fullyInside = matches && begin == 0.0 && finish == upper - lower && (!timeIncluded(lower) || includeStart) && (!timeIncluded(upper) || includeEnd);
    allInside = allInside && fullyInside;
    if (!matches) { continue; } anyInside = true;
    if (written >= ${capacity}u) { status[statusOffset + base] |= 256u; continue; }
    let outputRow = base + written; intervals[intervalsOffset + 2u * outputRow] = beginTime; intervals[intervalsOffset + 2u * outputRow + 1u] = finishTime;
    let beginPosition = firstA + velocityA * begin; let finishPosition = firstA + velocityA * finish;
    clipped[clippedOffset + 4u * outputRow] = beginPosition.x; clipped[clippedOffset + 4u * outputRow + 1u] = beginPosition.y; clipped[clippedOffset + 4u * outputRow + 2u] = finishPosition.x; clipped[clippedOffset + 4u * outputRow + 3u] = finishPosition.y;
    status[statusOffset + outputRow] |= 32u | select(1u, 2u, begin == finish) | select(0u,16u,fullyInside) | select(64u,0u,includeStart) | select(128u,0u,includeEnd); written++;
  }
}
var ever = select(0u,1u,anyInside); var always = select(0u,1u,allInside);
if (support == 0u || unknown > 0u) { if (!anyInside) { ever = 0xffffffffu; } if (allInside) { always = 0xffffffffu; } }
relationships[relationshipsOffset + 4u * index] = ever; relationships[relationshipsOffset + 4u * index + 1u] = always; relationships[relationshipsOffset + 4u * index + 2u] = support; relationships[relationshipsOffset + 4u * index + 3u] = unknown;`
    })
  ];
}
