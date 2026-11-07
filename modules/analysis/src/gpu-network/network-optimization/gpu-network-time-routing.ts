// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validatePackedView,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';

export type GPUNetworkTimeRoutingProps = {
  id?: string;
  /** Physical vertex of each explicit routing/restriction state. */
  stateVertices: GraphDataView<'uint32'>;
  /** Directed transitions between expanded states. */
  transitions: GraphDataView<'uint32x2'>;
  /** Affine duration, or fixed-width [departure,arrival] samples for each transition. */
  costModel?: 'affine-duration' | 'piecewise-arrival';
  /** At least two samples per transition in piecewise mode. Times/arrivals must be nondecreasing. */
  samplesPerTransition?: number;
  /** Piecewise mode waits until the first sample; after the last, its duration is held constant. */
  waitingPolicy?: 'wait-until-first-sample';
  /** [base duration, slope] per transition, or transition-major piecewise samples. */
  costs: GraphDataView<'float32x2'>;
  /** [initial state, physical destination vertex]. */
  endpoints: GraphDataView<'uint32'>;
  /** Nonnegative relative departure time; arbitrary epoch subtraction is caller-owned. */
  departure: GraphDataView<'float32'>;
  maximumRelaxations?: number;
  output: {
    arrivals: GraphDataView<'float32'>;
    /** Transition index per state; 0xffffffff for initial/unreached. */
    predecessorTransitions: GraphDataView<'uint32'>;
    /** [valid, converged, targetState, arrivalTimeBits]. */
    summary: GraphDataView<'uint32'>;
  };
};

/**
 * FIFO earliest-arrival routing on caller-prepared restriction states. No waiting is necessary
 * for these everywhere-nondecreasing affine travel durations. State count <= 256, transitions <=
 * 4096. Unreached +Infinity is conclusive only when converged=1; exhausted labels are incumbents.
 */
export class GPUNetworkTimeRouting implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUNetworkTimeRoutingProps;
  constructor(props: GPUNetworkTimeRoutingProps) {
    this.id = props.id ?? 'network-time-routing';
    this.props = props;
    assert(
      props.stateVertices.length > 0 &&
        props.stateVertices.length <= 256 &&
        props.transitions.length <= 4096
    );
    if (props.costModel === 'piecewise-arrival') {
      assert(
        Number.isInteger(props.samplesPerTransition) &&
          props.samplesPerTransition! >= 2 &&
          props.samplesPerTransition! <= 256
      );
    } else {
      assert(props.samplesPerTransition === undefined && props.waitingPolicy === undefined);
    }
    const rounds = props.maximumRelaxations ?? props.stateVertices.length;
    assert(Number.isInteger(rounds) && rounds >= 0 && rounds <= 65536);
    for (const view of [
      props.stateVertices,
      props.endpoints,
      props.output.predecessorTransitions,
      props.output.summary
    ])
      validatePackedView(view, ['uint32'], this.id);
    validatePackedView(props.transitions, ['uint32x2'], this.id);
    validatePackedView(props.costs, ['float32x2'], this.id);
    for (const view of [props.departure, props.output.arrivals])
      validatePackedView(view, ['float32'], this.id);
    assert(
      props.costs.length ===
        props.transitions.length *
          (props.costModel === 'piecewise-arrival' ? props.samplesPerTransition! : 1) &&
        props.endpoints.length === 2 &&
        props.departure.length === 1 &&
        props.output.arrivals.length === props.stateVertices.length &&
        props.output.predecessorTransitions.length === props.stateVertices.length &&
        props.output.summary.length === 4
    );
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.stateVertices,
      props.transitions,
      props.costs,
      props.endpoints,
      props.departure
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.stateVertices,
      props.transitions,
      props.costs,
      props.endpoints,
      props.departure,
      ...Object.values(props.output)
    ]);
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id,
        operation: 'GPUNetworkTimeRouting',
        variant: props.costModel === 'piecewise-arrival' ? 'piecewise-fifo-wait' : 'affine-fifo',
        invocationCount: 1,
        workgroupSize: 2,
        bindings: [
          {name: 'vertices', view: props.stateVertices, type: 'u32', access: 'read'},
          {name: 'transitions', view: props.transitions, type: 'u32', access: 'read'},
          {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
          {name: 'endpoints', view: props.endpoints, type: 'u32', access: 'read'},
          {name: 'departure', view: props.departure, type: 'f32', access: 'read'},
          {name: 'arrivals', view: props.output.arrivals, type: 'f32', access: 'read_write'},
          {
            name: 'predecessors',
            view: props.output.predecessorTransitions,
            type: 'u32',
            access: 'read_write'
          },
          {name: 'summary', view: props.output.summary, type: 'u32', access: 'read_write'}
        ],
        declarations: `const STATE_COUNT: u32 = ${props.stateVertices.length}u;
const TRANSITION_COUNT: u32 = ${props.transitions.length}u;
fn finite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
fn arrival(edge: u32, time: f32) -> f32 {
${
  props.costModel === 'piecewise-arrival'
    ? `
  let base = costsOffset + 2u * edge * ${props.samplesPerTransition}u;
  let firstTime = costs[base]; let firstArrival = costs[base + 1u];
  if (time <= firstTime) { return firstArrival; }
  for (var sample = 1u; sample < ${props.samplesPerTransition}u; sample++) {
    let rightTime = costs[base + 2u * sample]; let rightArrival = costs[base + 2u * sample + 1u];
    if (time <= rightTime) {
      let leftTime = costs[base + 2u * (sample - 1u)]; let leftArrival = costs[base + 2u * (sample - 1u) + 1u];
      return mix(leftArrival, rightArrival, (time - leftTime) / (rightTime - leftTime));
    }
  }
  // Preserve the nonnegative held duration before addition; max also prevents fast-math
  // reassociation into time + absoluteArrival - departure, whose partial sum can overflow.
  let duration = max(0.0, costs[base + 2u * (${props.samplesPerTransition}u - 1u) + 1u] - costs[base + 2u * (${props.samplesPerTransition}u - 1u)]);
  return time + duration;
`
    : 'return time + costs[costsOffset + 2u * edge] + costs[costsOffset + 2u * edge + 1u] * time;'
}
}`,

        body: `var infinityBits = 0x7f800000u; let infinity = bitcast<f32>(infinityBits);
summary[summaryOffset] = 0u; summary[summaryOffset + 1u] = 0u; summary[summaryOffset + 2u] = 0xffffffffu; summary[summaryOffset + 3u] = bitcast<u32>(infinity);
for (var state = 0u; state < STATE_COUNT; state++) { arrivals[arrivalsOffset + state] = infinity; predecessors[predecessorsOffset + state] = 0xffffffffu; }
let initial = endpoints[endpointsOffset];
let destination = endpoints[endpointsOffset + 1u];
let startTime = departure[departureOffset];
if (initial >= STATE_COUNT || !finite(startTime) || startTime < 0.0) { return; }
for (var edge = 0u; edge < TRANSITION_COUNT; edge++) {
  let source = transitions[transitionsOffset + 2u * edge]; let destination = transitions[transitionsOffset + 2u * edge + 1u];
  if (source >= STATE_COUNT || destination >= STATE_COUNT) { return; }
  ${
    props.costModel === 'piecewise-arrival'
      ? `
  var previousDeparture = -1.0; var previousArrival = -1.0;
  for (var sample = 0u; sample < ${props.samplesPerTransition}u; sample++) {
    let offset = costsOffset + 2u * (edge * ${props.samplesPerTransition}u + sample);
    let sampleDeparture = costs[offset]; let sampleArrival = costs[offset + 1u];
    if (!finite(sampleDeparture) || !finite(sampleArrival) || sampleDeparture < 0.0 || sampleDeparture <= previousDeparture || sampleArrival < sampleDeparture || sampleArrival < previousArrival) { return; }
    previousDeparture = sampleDeparture; previousArrival = sampleArrival;
  }`
      : `let base = costs[costsOffset + 2u * edge]; let slope = costs[costsOffset + 2u * edge + 1u];
  if (!finite(base) || !finite(slope) || base < 0.0 || slope < 0.0) { return; }`
  }
}
summary[summaryOffset] = 1u;
arrivals[arrivalsOffset + initial] = startTime;
for (var round = 0u; round < ${props.maximumRelaxations ?? props.stateVertices.length}u; round++) {
  var changed = false;
  for (var edge = 0u; edge < TRANSITION_COUNT; edge++) {
    let source = transitions[transitionsOffset + 2u * edge]; let destination = transitions[transitionsOffset + 2u * edge + 1u];
    let time = arrivals[arrivalsOffset + source];
    if (!finite(time)) { continue; }
    let candidate = arrival(edge, time);
    if (!finite(candidate)) { summary[summaryOffset] = 0u; return; }
    if (candidate < arrivals[arrivalsOffset + destination]) {
      arrivals[arrivalsOffset + destination] = candidate; predecessors[predecessorsOffset + destination] = edge; changed = true;
    }
  }
  if (!changed) { summary[summaryOffset + 1u] = 1u; break; }
}
var best = infinity;
var bestState = 0xffffffffu;
for (var state = 0u; state < STATE_COUNT; state++) {
  if (vertices[verticesOffset + state] == destination && arrivals[arrivalsOffset + state] < best) { best = arrivals[arrivalsOffset + state]; bestState = state; }
}
summary[summaryOffset + 2u] = bestState;
summary[summaryOffset + 3u] = bitcast<u32>(best);`
      })
    ];
  }
}

/** Static directed network edge; IDs are indices in the input array. */
export type GPUNetworkRestrictionEdge = {source: number; target: number};

/**
 * Explicit CPU topology preparation, never a route solver. Builds the product of vertices with
 * the proper-prefix automaton of forbidden edge sequences. Upload the returned arrays and map
 * each transition's cost using originalEdges. Parallel edges retain distinct identities.
 */
export function prepareGPUNetworkRestrictionStates(
  nodeCount: number,
  edges: readonly GPUNetworkRestrictionEdge[],
  restrictions: readonly (readonly number[])[]
) {
  assert(Number.isInteger(nodeCount) && nodeCount > 0 && nodeCount <= 256);
  for (const edge of edges)
    assert(
      Number.isInteger(edge.source) &&
        Number.isInteger(edge.target) &&
        edge.source >= 0 &&
        edge.source < nodeCount &&
        edge.target >= 0 &&
        edge.target < nodeCount
    );
  const prefixes: number[][] = [[]];
  const keys = new Set(['']);
  for (const restriction of restrictions) {
    assert(
      restriction.length > 0 &&
        restriction.every(edge => Number.isInteger(edge) && edge >= 0 && edge < edges.length)
    );
    for (let length = 1; length < restriction.length; length++) {
      const prefix = restriction.slice(0, length);
      const key = prefix.join(',');
      if (!keys.has(key)) {
        keys.add(key);
        prefixes.push(prefix);
      }
    }
  }
  // Keep only reachable automaton/vertex pairs, while retaining one empty-history source per vertex.
  const stateVertices: number[] = Array.from({length: nodeCount}, (_, vertex) => vertex);
  const statePrefixes: number[] = Array.from({length: nodeCount}, () => 0);
  const states = new Map(stateVertices.map(vertex => [`0:${vertex}`, vertex]));
  const transitions: number[] = [];
  const originalEdges: number[] = [];
  const endsWith = (sequence: readonly number[], suffix: readonly number[]) =>
    sequence.length >= suffix.length &&
    suffix.every((value, offset) => sequence[sequence.length - suffix.length + offset] === value);
  for (let state = 0; state < stateVertices.length; state++) {
    const prefix = prefixes[statePrefixes[state]];
    edges.forEach((edge, originalEdge) => {
      if (edge.source !== stateVertices[state]) return;
      const sequence = [...prefix, originalEdge];
      if (restrictions.some(restriction => endsWith(sequence, restriction))) return;
      let nextPrefix = 0;
      prefixes.forEach((candidate, candidateIndex) => {
        if (candidate.length > prefixes[nextPrefix].length && endsWith(sequence, candidate))
          nextPrefix = candidateIndex;
      });
      const key = `${nextPrefix}:${edge.target}`;
      let destination = states.get(key);
      if (destination === undefined) {
        destination = stateVertices.length;
        assert(destination < 256);
        states.set(key, destination);
        stateVertices.push(edge.target);
        statePrefixes.push(nextPrefix);
      }
      transitions.push(state, destination);
      originalEdges.push(originalEdge);
      assert(originalEdges.length <= 4096);
    });
  }
  return {
    stateVertices: Uint32Array.from(stateVertices),
    transitions: Uint32Array.from(transitions),
    originalEdges: Uint32Array.from(originalEdges),
    initialStates: Uint32Array.from({length: nodeCount}, (_, vertex) => vertex)
  };
}
