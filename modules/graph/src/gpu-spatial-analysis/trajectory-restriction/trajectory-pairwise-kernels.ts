// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode} from '@luma-contributors/graph/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import type {GPUTrajectoryRestrictionProps} from './gpu-trajectory-restriction';

/** Analytic continuous within-distance intervals on overlapping linearly interpolated support. */
export function createTrajectoryPairwiseNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUTrajectoryRestrictionProps
): GPUCommandNode<Parameters>[] {
  const pairs = props.pairwiseTracks!;
  const capacity = props.intervalsPerPair!;
  if (!pairs.length) return [];
  const closedStart = props.timeBoundary !== 'open' && props.timeBoundary !== 'right-closed';
  const closedEnd = props.timeBoundary !== 'open' && props.timeBoundary !== 'left-closed';
  const closedGeometry = props.geometryBoundary !== 'open';
  return [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-pairwise`,
      operation: 'GPUTrajectoryRestriction',
      variant: 'pairwise-within-distance',
      invocationCount: pairs.length,
      workgroupSize: 32,
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'times', view: props.times, type: 'f32', access: 'read'},
        {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
        {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
        {name: 'intervals', view: props.output.intervals, type: 'f32', access: 'read_write'},
        {name: 'clipped', view: props.output.positions, type: 'f32', access: 'read_write'},
        {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'},
        {name: 'relationships', view: props.output.relationships, type: 'u32', access: 'read_write'}
      ],
      declarations: `const PAIRS = array<vec2<u32>, ${pairs.length}>(${pairs.map(pair => `vec2<u32>(${pair[0]}u,${pair[1]}u)`).join(',')});
fn finite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
fn position(row: u32) -> vec2<f32> { return vec2<f32>(positions[positionsOffset + 2u * row],positions[positionsOffset + 2u * row + 1u]); }
fn timeIncluded(time: f32) -> bool { return time ${closedStart ? '>=' : '>'} parameters[parametersOffset] && time ${closedEnd ? '<=' : '<'} parameters[parametersOffset + 1u]; }`,
      body: `var nanBits = 0x7fc00000u; let missing = bitcast<f32>(nanBits); let base = index * ${capacity}u;
for (var slot = 0u; slot < ${capacity}u; slot++) { status[statusOffset + base + slot] = 0u; for (var component = 0u; component < 2u; component++) { intervals[intervalsOffset + 2u * (base + slot) + component] = missing; } for (var component = 0u; component < 4u; component++) { clipped[clippedOffset + 4u * (base + slot) + component] = missing; } }
var anyInside = false; var allInside = true; var support = 0u; var unknown = 0u; var written = 0u;
let minimumTime = parameters[parametersOffset]; let maximumTime = parameters[parametersOffset + 1u]; let distanceLimit = parameters[parametersOffset + 2u]; let maximumGap = parameters[parametersOffset + 6u];
let validParameters = finite(minimumTime) && finite(maximumTime) && minimumTime <= maximumTime && finite(distanceLimit) && distanceLimit >= 0.0 && maximumGap > 0.0;
let tracks = PAIRS[index]; let startA = min(offsets[offsetsOffset + tracks.x], ${props.positions.length}u); let endA = min(offsets[offsetsOffset + tracks.x + 1u], ${props.positions.length}u); let startB = min(offsets[offsetsOffset + tracks.y], ${props.positions.length}u); let endB = min(offsets[offsetsOffset + tracks.y + 1u], ${props.positions.length}u);
for (var rowA = startA; rowA < endA; rowA++) {
  let nextA = min(rowA + 1u, endA - 1u); let timeA = times[timesOffset + rowA]; let rawFinishA = times[timesOffset + nextA]; let gapA = rawFinishA - timeA > maximumGap; let finishA = select(rawFinishA, timeA, gapA); let durationA = finishA - timeA; let positionA = position(rowA); let movementA = select(position(nextA) - positionA, vec2<f32>(0.0), gapA);
  for (var rowB = startB; rowB < endB; rowB++) {
    let nextB = min(rowB + 1u, endB - 1u); let timeB = times[timesOffset + rowB]; let rawFinishB = times[timesOffset + nextB]; let gapB = rawFinishB - timeB > maximumGap; let finishB = select(rawFinishB, timeB, gapB); let durationB = finishB - timeB; let positionB = position(rowB); let movementB = select(position(nextB) - positionB, vec2<f32>(0.0), gapB);
    let lower = max(minimumTime, max(timeA, timeB)); let upper = min(maximumTime, min(finishA, finishB));
    if (!validParameters || !finite(timeA) || !finite(timeB) || !finite(finishA) || !finite(finishB) || durationA < 0.0 || durationB < 0.0 || !all(vec4<bool>(finite(positionA.x),finite(positionA.y),finite(positionB.x),finite(positionB.y))) || !all(vec4<bool>(finite(movementA.x),finite(movementA.y),finite(movementB.x),finite(movementB.y))) || (durationA == 0.0 && any(movementA != vec2<f32>(0.0))) || (durationB == 0.0 && any(movementB != vec2<f32>(0.0)))) { unknown++; status[statusOffset + base] |= 4u; continue; }
    if ((gapA || gapB) && max(minimumTime, max(timeA, timeB)) < min(maximumTime, min(rawFinishA, rawFinishB))) { unknown++; status[statusOffset + base] |= 8u; }
    if (lower > upper || (lower == upper && !timeIncluded(lower))) { continue; }
    if (durationA > maximumGap || durationB > maximumGap) { if (lower < upper) { unknown++; status[statusOffset + base] |= 8u; } continue; }
    support++;
    var velocityA = vec2<f32>(0.0); var velocityB = vec2<f32>(0.0); if (durationA > 0.0) { velocityA = movementA / durationA; } if (durationB > 0.0) { velocityB = movementB / durationB; }
    let firstA = positionA + velocityA * (lower - timeA); let firstB = positionB + velocityB * (lower - timeB); let relative = firstA - firstB; let velocity = velocityA - velocityB;
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
