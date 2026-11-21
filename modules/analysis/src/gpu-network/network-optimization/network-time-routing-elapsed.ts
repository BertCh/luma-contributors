// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import type {GPUNetworkTimeRoutingProps} from './gpu-network-time-routing';

/** Cooperative elapsed-label solve. Each transition commits before the next, preserving sweeps. */
export function createNetworkTimeRoutingElapsedNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUNetworkTimeRoutingProps,
  elapsed: GraphDataView<'float32'>,
  summary: GraphDataView<'uint32'>,
  countsIterations: boolean
) {
  const stateCount = props.stateVertices.length;
  const transitionCount = props.transitions.length;
  const sampleCount = props.samplesPerTransition ?? 1;
  const workgroupSize = 64;
  return createWGSLKernelNode<Parameters>(graph, {
    id: `${id}-elapsed`,
    operation: 'GPUNetworkTimeRouting',
    variant: 'cooperative-elapsed',
    invocationCount: workgroupSize,
    workgroupSize,
    // Every lane participates in barriers; a local-index early return is nonuniform.
    guardIndex: false,
    bindings: [
      {name: 'vertices', view: props.stateVertices, type: 'u32', access: 'read'},
      {name: 'transitions', view: props.transitions, type: 'u32', access: 'read'},
      {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
      {name: 'endpoints', view: props.endpoints, type: 'u32', access: 'read'},
      {name: 'departure', view: props.departure, type: 'f32', access: 'read'},
      {name: 'elapsed', view: elapsed, type: 'f32', access: 'read_write'},
      {
        name: 'predecessors',
        view: props.output.predecessorTransitions,
        type: 'u32',
        access: 'read_write'
      },
      {name: 'summary', view: summary, type: 'u32', access: 'read_write'}
    ],
    declarations: `${IS_FINITE_VALUE_WGSL}
var<workgroup> labels: array<f32, ${stateCount}>;
var<workgroup> parents: array<u32, ${stateCount}>;
var<workgroup> invalid: atomic<u32>;
var<workgroup> changed: atomic<u32>;
var<workgroup> running: u32;
var<workgroup> valid: u32;
var<workgroup> converged: u32;
var<workgroup> iterations: u32;
fn elapsedArrival(edge: u32, time: f32, origin: f32) -> f32 {
${
  props.costModel === 'piecewise-arrival'
    ? `
  let base = costsOffset + 2u * edge * ${sampleCount}u;
  let firstTime = costs[base] - origin;
  if (time <= firstTime) { return costs[base + 1u] - origin; }
  for (var sample = 1u; sample < ${sampleCount}u; sample++) {
    let right = base + 2u * sample;
    if (time <= costs[right] - origin) {
      let fraction = (time - (costs[right - 2u] - origin)) / (costs[right] - costs[right - 2u]);
      let leftDuration = max(0.0, costs[right - 1u] - costs[right - 2u]);
      let rightDuration = max(0.0, costs[right + 1u] - costs[right]);
      return time + max(0.0, mix(leftDuration, rightDuration, fraction));
    }
  }
  let last = base + 2u * (${sampleCount}u - 1u);
  return time + max(0.0, costs[last + 1u] - costs[last]);
`
    : `
  let base = costs[costsOffset + 2u * edge];
  let slope = costs[costsOffset + 2u * edge + 1u];
  let durationAtOrigin = base + slope * origin;
  return time + durationAtOrigin + slope * time;
`
}
}`,
    body: `var infinityBits = 0x7f800000u; let infinity = bitcast<f32>(infinityBits);
let origin = departure[departureOffset];
let initial = endpoints[endpointsOffset];
if (index == 0u) {
  atomicStore(&invalid, select(1u, 0u, initial < ${stateCount}u && isFiniteValue(origin) && origin >= 0.0));
  valid = 0u; converged = 0u; iterations = 0u; running = 0u;
}
for (var state = index; state < ${stateCount}u; state += ${workgroupSize}u) { labels[state] = select(infinity, 0.0, state == initial); parents[state] = 0xffffffffu; }
workgroupBarrier();
for (var edge = index; edge < ${transitionCount}u; edge += ${workgroupSize}u) {
  if (transitions[transitionsOffset + 2u * edge] >= ${stateCount}u || transitions[transitionsOffset + 2u * edge + 1u] >= ${stateCount}u) { atomicStore(&invalid, 1u); }
  ${
    props.costModel === 'piecewise-arrival'
      ? `
  var previousDeparture = -1.0; var previousArrival = -1.0;
  for (var sample = 0u; sample < ${sampleCount}u; sample++) {
    let offset = costsOffset + 2u * (edge * ${sampleCount}u + sample);
    let sampleDeparture = costs[offset]; let sampleArrival = costs[offset + 1u];
    if (!isFiniteValue(sampleDeparture) || !isFiniteValue(sampleArrival) || sampleDeparture < 0.0 || sampleDeparture <= previousDeparture || sampleArrival < sampleDeparture || sampleArrival < previousArrival) { atomicStore(&invalid, 1u); }
    previousDeparture = sampleDeparture; previousArrival = sampleArrival;
  }`
      : `
  let base = costs[costsOffset + 2u * edge]; let slope = costs[costsOffset + 2u * edge + 1u];
  if (!isFiniteValue(base) || !isFiniteValue(slope) || base < 0.0 || slope < 0.0) { atomicStore(&invalid, 1u); }`
  }
}
workgroupBarrier();
if (index == 0u) { valid = 1u - atomicLoad(&invalid); running = valid; }
for (var round = 0u; round < ${props.maximumRelaxations ?? stateCount}u; round++) {
  if (workgroupUniformLoad(&running) == 0u) { break; }
  if (index == 0u) { atomicStore(&changed, 0u); iterations++; }
  workgroupBarrier();
  for (var edge = 0u; edge < ${transitionCount}u; edge++) {
    let source = transitions[transitionsOffset + 2u * edge];
    let targetState = transitions[transitionsOffset + 2u * edge + 1u];
    if (index == targetState % ${workgroupSize}u && isFiniteValue(labels[source]) && atomicLoad(&invalid) == 0u) {
      let candidate = elapsedArrival(edge, labels[source], origin);
      if (!isFiniteValue(candidate)) { atomicStore(&invalid, 1u); }
      else if (candidate < labels[targetState]) { labels[targetState] = candidate; parents[targetState] = edge; atomicStore(&changed, 1u); }
    }
    workgroupBarrier();
  }
  if (index == 0u) {
    if (atomicLoad(&invalid) != 0u) { valid = 0u; running = 0u; }
    else if (atomicLoad(&changed) == 0u) { converged = 1u; running = 0u; }
  }
}
workgroupBarrier();
for (var state = index; state < ${stateCount}u; state += ${workgroupSize}u) {
  elapsed[elapsedOffset + state] = select(infinity, labels[state], valid != 0u);
  predecessors[predecessorsOffset + state] = select(0xffffffffu, parents[state], valid != 0u);
}
if (index == 0u) {
  var best = infinity; var bestState = 0xffffffffu;
  for (var state = 0u; state < ${stateCount}u; state++) {
    if (valid != 0u && vertices[verticesOffset + state] == endpoints[endpointsOffset + 1u] && labels[state] < best) { best = labels[state]; bestState = state; }
  }
  summary[summaryOffset] = valid; summary[summaryOffset + 1u] = converged;
  summary[summaryOffset + 2u] = bestState; summary[summaryOffset + 3u] = bitcast<u32>(select(infinity, origin + best, bestState != 0xffffffffu));
  ${countsIterations ? 'summary[summaryOffset + 4u] = iterations;' : ''}
}`
  });
}
