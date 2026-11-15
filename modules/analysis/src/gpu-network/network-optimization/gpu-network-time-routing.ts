// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView} from '@luma.gl/graph/gpu-core';
import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {
  GPUInvalidStatusPort,
  GPUIterationStatusPort
} from '../../analysis-core/contracts/status';
import {
  createNetworkOptimizationStatusNodes,
  getNetworkOptimizationStatusViews,
  validateNetworkOptimizationStatus
} from './network-optimization-status';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateMaximumCount,
  validateSinglePackedView,
  validateRequirement,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';

/** Largest state count: the single invocation relaxes every state per round (O(states * transitions)). */
const MAXIMUM_STATE_COUNT = 256;
/** Largest transition count: bounds the work of each relaxation round in the single invocation. */
const MAXIMUM_TRANSITION_COUNT = 4096;
/** Largest `samplesPerTransition` of the piecewise arrival model. */
const MAXIMUM_SAMPLES_PER_TRANSITION = 256;
/** Largest `maximumRelaxations`: bounds the relaxation loop of the single invocation. */
const MAXIMUM_RELAXATIONS = 65536;
/** Rows of the `endpoints` view: initial state and physical destination vertex. */
const ENDPOINT_COUNT = 2;
/** Words of the `summary` view: valid, converged, target state, arrival time bits. */
const SUMMARY_WORD_COUNT = 4;

export type GPUNetworkTimeRoutingOutput = {
  arrivals: GraphDataView<'float32'>;
  /** Transition index per state; 0xffffffff for initial/unreached. */
  predecessorTransitions: GraphDataView<'uint32'>;
  /** [valid, converged, targetState, arrivalTimeBits]. */
  summary: GraphDataView<'uint32'>;
  /** invalidCount is 1 for rejected inputs or nonfinite arrival arithmetic.
   * iterationCount counts executed transition sweeps, including the final no-change sweep.
   * Invalid problems set neither converged nor iterationLimitReached.
   */
  status?: GPUInvalidStatusPort & GPUIterationStatusPort;
};

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
  output: GPUNetworkTimeRoutingOutput;
};

/**
 * FIFO earliest-arrival routing on caller-prepared restriction states. No waiting is necessary
 * for these everywhere-nondecreasing affine travel durations. State count <= 256, transitions <=
 * 4096. Unreached +Infinity is conclusive only when converged=1; exhausted labels are incumbents.
 *
 * Reference tier (contributor contract rule 2): the label-correcting search is one GPU invocation
 * (`invocationCount: 1`) that relaxes every transition in each round, O(rounds * transitions),
 * with at most 256 states, 4,096 transitions and 65,536 relaxation rounds (default: one per
 * state), enforced at construction. A frontier relaxation with a time-dependent arrival function
 * on the reachability engine would replace it for larger networks.
 * Compile-time: view lengths, costModel, samplesPerTransition, waitingPolicy, maximumRelaxations
 * and optional status views. Per-frame: state vertices, transitions, costs, endpoints and departure.
 * Every encoding rewrites the outputs.
 */
export class GPUNetworkTimeRouting implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUNetworkTimeRoutingProps;
  constructor(props: GPUNetworkTimeRoutingProps) {
    this.id = props.id ?? 'network-time-routing';
    this.props = props;
    const {id} = this;
    validateCapacity(id, 'stateVertices rows', props.stateVertices.length, 1, MAXIMUM_STATE_COUNT);
    validateCapacity(id, 'transitions rows', props.transitions.length, 0, MAXIMUM_TRANSITION_COUNT);
    if (props.costModel === 'piecewise-arrival') {
      validateCapacity(
        id,
        'samplesPerTransition',
        props.samplesPerTransition as number,
        2,
        MAXIMUM_SAMPLES_PER_TRANSITION
      );
    } else {
      validateRequirement(
        id,
        props.samplesPerTransition === undefined && props.waitingPolicy === undefined,
        'samplesPerTransition and waitingPolicy require costModel piecewise-arrival'
      );
    }
    validateCapacity(
      id,
      'maximumRelaxations',
      props.maximumRelaxations ?? props.stateVertices.length,
      0,
      MAXIMUM_RELAXATIONS
    );
    for (const view of [
      props.stateVertices,
      props.endpoints,
      props.output.predecessorTransitions,
      props.output.summary
    ])
      validateSinglePackedView(view, ['uint32'], this.id);
    validateSinglePackedView(props.transitions, ['uint32x2'], this.id);
    validateSinglePackedView(props.costs, ['float32x2'], this.id);
    for (const view of [props.departure, props.output.arrivals])
      validateSinglePackedView(view, ['float32'], this.id);
    validateRequirement(
      id,
      props.costs.length ===
        props.transitions.length *
          (props.costModel === 'piecewise-arrival' ? props.samplesPerTransition! : 1),
      'costs must hold samplesPerTransition rows per transition (one for the affine model)'
    );
    validateRequirement(
      id,
      props.endpoints.length === ENDPOINT_COUNT,
      `endpoints must hold ${ENDPOINT_COUNT} rows: initial state and destination vertex`
    );
    validateRequirement(id, props.departure.length === 1, 'departure must hold one row');
    validateRequirement(
      id,
      props.output.arrivals.length === props.stateVertices.length,
      'output.arrivals must hold one row per state'
    );
    validateRequirement(
      id,
      props.output.predecessorTransitions.length === props.stateVertices.length,
      'output.predecessorTransitions must hold one row per state'
    );
    validateRequirement(
      id,
      props.output.summary.length === SUMMARY_WORD_COUNT,
      `output.summary must hold ${SUMMARY_WORD_COUNT} rows`
    );
    validateNetworkOptimizationStatus(
      id,
      [props.output.arrivals, props.output.predecessorTransitions, props.output.summary],
      props.output.status
    );
    validateGraphOutputsDisjointFromInputs(this.id, getTimeRoutingOutputViews(props.output), [
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
      ...getTimeRoutingOutputViews(props.output)
    ]);
    // A fifth private summary word tracks sweeps only when requested. The solve already uses
    // eight bindings; publish its four public words separately rather than adding a ninth binding.
    const countsIterations = Boolean(props.output.status?.iterationCount);
    const summary = countsIterations
      ? createTransientView(
          graph,
          `${id}-summary-with-iterations`,
          'uint32',
          SUMMARY_WORD_COUNT + 1
        )
      : props.output.summary;
    const nodes = [
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
          {name: 'summary', view: summary, type: 'u32', access: 'read_write'}
        ],
        declarations: `const STATE_COUNT: u32 = ${props.stateVertices.length}u;
const TRANSITION_COUNT: u32 = ${props.transitions.length}u;
${IS_FINITE_VALUE_WGSL}
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

        body: `${countsIterations ? 'summary[summaryOffset + 4u] = 0u;' : ''}
var infinityBits = 0x7f800000u; let infinity = bitcast<f32>(infinityBits);
summary[summaryOffset] = 0u; summary[summaryOffset + 1u] = 0u; summary[summaryOffset + 2u] = 0xffffffffu; summary[summaryOffset + 3u] = bitcast<u32>(infinity);
for (var state = 0u; state < STATE_COUNT; state++) { arrivals[arrivalsOffset + state] = infinity; predecessors[predecessorsOffset + state] = 0xffffffffu; }
let initial = endpoints[endpointsOffset];
let destination = endpoints[endpointsOffset + 1u];
let startTime = departure[departureOffset];
if (initial >= STATE_COUNT || !isFiniteValue(startTime) || startTime < 0.0) { return; }
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
    if (!isFiniteValue(sampleDeparture) || !isFiniteValue(sampleArrival) || sampleDeparture < 0.0 || sampleDeparture <= previousDeparture || sampleArrival < sampleDeparture || sampleArrival < previousArrival) { return; }
    previousDeparture = sampleDeparture; previousArrival = sampleArrival;
  }`
      : `let base = costs[costsOffset + 2u * edge]; let slope = costs[costsOffset + 2u * edge + 1u];
  if (!isFiniteValue(base) || !isFiniteValue(slope) || base < 0.0 || slope < 0.0) { return; }`
  }
}
summary[summaryOffset] = 1u;
arrivals[arrivalsOffset + initial] = startTime;
for (var round = 0u; round < ${props.maximumRelaxations ?? props.stateVertices.length}u; round++) {
  ${countsIterations ? 'summary[summaryOffset + 4u]++;' : ''}
  var changed = false;
  for (var edge = 0u; edge < TRANSITION_COUNT; edge++) {
    let source = transitions[transitionsOffset + 2u * edge]; let destination = transitions[transitionsOffset + 2u * edge + 1u];
    let time = arrivals[arrivalsOffset + source];
    if (!isFiniteValue(time)) { continue; }
    let candidate = arrival(edge, time);
    if (!isFiniteValue(candidate)) { summary[summaryOffset] = 0u; return; }
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
    if (countsIterations) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-publish-summary`,
          operation: 'GPUNetworkTimeRouting',
          variant: 'publish-summary',
          invocationCount: SUMMARY_WORD_COUNT,
          bindings: [
            {name: 'source', view: summary, type: 'u32', access: 'read'},
            {name: 'destination', view: props.output.summary, type: 'u32', access: 'read_write'}
          ],
          body: 'destination[destinationOffset + index] = source[sourceOffset + index];'
        })
      );
    }
    nodes.push(
      ...createNetworkOptimizationStatusNodes(
        graph,
        id,
        'GPUNetworkTimeRouting',
        summary,
        props.output.status,
        {
          invalidCount: '1u - summary[summaryOffset]',
          iterationCount: 'summary[summaryOffset + 4u]',
          converged: 'summary[summaryOffset] * summary[summaryOffset + 1u]',
          iterationLimitReached: 'summary[summaryOffset] * (1u - summary[summaryOffset + 1u])'
        }
      )
    );
    return nodes;
  }
}

function getTimeRoutingOutputViews(output: GPUNetworkTimeRoutingProps['output']): GraphDataView[] {
  return [
    output.arrivals,
    output.predecessorTransitions,
    output.summary,
    ...getNetworkOptimizationStatusViews(output.status)
  ];
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
  const id = 'prepareGPUNetworkRestrictionStates';
  validateCapacity(id, 'nodeCount', nodeCount, 1, MAXIMUM_STATE_COUNT);
  for (const edge of edges)
    validateRequirement(
      id,
      Number.isInteger(edge.source) &&
        Number.isInteger(edge.target) &&
        edge.source >= 0 &&
        edge.source < nodeCount &&
        edge.target >= 0 &&
        edge.target < nodeCount,
      'edges must have integer source and target below nodeCount'
    );
  const prefixes: number[][] = [[]];
  const keys = new Set(['']);
  for (const restriction of restrictions) {
    validateRequirement(
      id,
      restriction.length > 0 &&
        restriction.every(edge => Number.isInteger(edge) && edge >= 0 && edge < edges.length),
      'each restriction must list at least one edge index below the edge count'
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
        validateMaximumCount(id, 'restriction states', destination + 1, MAXIMUM_STATE_COUNT);
        states.set(key, destination);
        stateVertices.push(edge.target);
        statePrefixes.push(nextPrefix);
      }
      transitions.push(state, destination);
      originalEdges.push(originalEdge);
      validateMaximumCount(
        id,
        'restriction transitions',
        originalEdges.length,
        MAXIMUM_TRANSITION_COUNT
      );
    });
  }
  return {
    stateVertices: Uint32Array.from(stateVertices),
    transitions: Uint32Array.from(transitions),
    originalEdges: Uint32Array.from(originalEdges),
    initialStates: Uint32Array.from({length: nodeCount}, (_, vertex) => vertex)
  };
}
