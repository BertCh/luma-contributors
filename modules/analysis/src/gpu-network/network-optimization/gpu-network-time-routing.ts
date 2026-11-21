// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createNetworkTimeRoutingElapsedNode} from './network-time-routing-elapsed';
import {
  validateNetworkOptimizationBindings,
  validateNetworkOptimizationInputOutputBuffers
} from './network-optimization-bindings';

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
  validateGPUStorageBindingByteLength,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import type {GPUNetworkEdgeEndpoints} from './prepare-flow-demands';

import {
  MAXIMUM_STATE_COUNT,
  MAXIMUM_TRANSITION_COUNT,
  MAXIMUM_SAMPLES_PER_TRANSITION,
  MAXIMUM_RELAXATIONS,
  validateNetworkTimeRoutingOptions
} from './network-time-routing-parameters';

/** Rows of the `endpoints` view: initial state and physical destination vertex. */
const ENDPOINT_COUNT = 2;
/** Words of the `summary` view: valid, converged, target state, arrival time bits. */
const SUMMARY_WORD_COUNT = 4;

export type GPUNetworkTimeRoutingOutput = {
  arrivals: GraphDataView<'float32'>;
  /** Optional one elapsed duration per state, before rounding onto the public clock. */
  elapsedTimes?: GraphDataView<'float32'>;
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
  /**
   * Nonnegative clock time in the duration unit. Rebase t′ = t - origin together with costs:
   * affine base′ = base + slope * origin (base in duration units, slope duration/time).
   * Departure 100 and [10, .5] become 0 and [60, .5], preserving duration 60.
   * For piecewise schedules subtract origin from both time columns; all relative samples
   * must remain nonnegative and retain the ordered FIFO constraints.
   */
  departure: GraphDataView<'float32'>;
  maximumRelaxations?: number;
  output: GPUNetworkTimeRoutingOutput;
};

/** Allocation-free validation shared by direct routing and departure-window wrappers. */
export function validateNetworkTimeRoutingInputs(
  id: string,
  props: Pick<
    GPUNetworkTimeRoutingProps,
    | 'stateVertices'
    | 'transitions'
    | 'costs'
    | 'costModel'
    | 'samplesPerTransition'
    | 'waitingPolicy'
    | 'maximumRelaxations'
  >
): void {
  validateNetworkTimeRoutingOptions(id, props);
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
  }
  validateCapacity(
    id,
    'maximumRelaxations',
    props.maximumRelaxations ?? props.stateVertices.length,
    0,
    MAXIMUM_RELAXATIONS
  );
  validateSinglePackedView(props.stateVertices, ['uint32'], id);
  validateSinglePackedView(props.transitions, ['uint32x2'], id);
  validateSinglePackedView(props.costs, ['float32x2'], id);
  validateRequirement(
    id,
    props.costs.length ===
      props.transitions.length *
        (props.costModel === 'piecewise-arrival' ? props.samplesPerTransition! : 1),
    'costs must hold samplesPerTransition rows per transition (one for the affine model)'
  );
}

/**
 * FIFO earliest-arrival routing on caller-prepared restriction states. No waiting is necessary
 * for these everywhere-nondecreasing affine travel durations. State count <= 256, transitions <=
 * 4096. Unreached +Infinity is conclusive only when converged=1; exhausted labels are incumbents.
 *
 * Reference tier (contributor contract rule 2): one cooperative workgroup commits transitions
 * in scan order, with a barrier after each transition, O(rounds * transitions) barriers,
 * with at most 256 states, 4,096 transitions and 65,536 relaxation rounds (default: one per
 * state), enforced at construction. Labels store elapsed durations; absolute arrivals round only
 * at publication. Affine and piecewise models retain their original clock. A frontier relaxation with a time-dependent arrival function
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
    validateNetworkTimeRoutingInputs(id, props);
    for (const view of [
      props.stateVertices,
      props.endpoints,
      props.output.predecessorTransitions,
      props.output.summary
    ])
      validateSinglePackedView(view, ['uint32'], this.id);
    for (const view of [props.departure, props.output.arrivals])
      validateSinglePackedView(view, ['float32'], this.id);
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
    if (props.output.elapsedTimes) {
      validateSinglePackedView(props.output.elapsedTimes, ['float32'], `${id} output.elapsedTimes`);
      validateRequirement(
        id,
        props.output.elapsedTimes.length === props.stateVertices.length,
        'output.elapsedTimes must have one row per state'
      );
    }
    validateNetworkOptimizationStatus(
      id,
      [
        props.output.arrivals,
        ...(props.output.elapsedTimes ? [props.output.elapsedTimes] : []),
        props.output.predecessorTransitions,
        props.output.summary
      ],
      props.output.status
    );
    validateNetworkOptimizationInputOutputBuffers(
      this.id,
      getTimeRoutingOutputViews(props.output),
      [props.stateVertices, props.transitions, props.costs, props.endpoints, props.departure]
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateNetworkOptimizationBindings(
      graph,
      id,
      Object.fromEntries(
        [
          props.stateVertices,
          props.transitions,
          props.costs,
          props.endpoints,
          props.departure,
          ...getTimeRoutingOutputViews(props.output)
        ].map((view, index) => [`view ${index}`, view])
      )
    );

    // A fifth private summary word tracks sweeps only when requested. The solve already uses
    // eight bindings; publish its four public words separately rather than adding a ninth binding.
    const countsIterations = Boolean(props.output.status?.iterationCount);
    if (countsIterations) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'summary with iteration count',
        (SUMMARY_WORD_COUNT + 1) * 4,
        'omit output.status.iterationCount or request a larger binding limit'
      );
    }
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
    // Keep the fingerprinted legacy reference text above intact; execute the separate
    // cooperative implementation, whose barriers preserve each transition's commit order.
    const elapsed =
      props.output.elapsedTimes ??
      createTransientView(graph, `${id}-elapsed-times`, 'float32', props.stateVertices.length);
    nodes[0] = createNetworkTimeRoutingElapsedNode(
      graph,
      id,
      props,
      elapsed,
      summary,
      countsIterations
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish-arrivals`,
        operation: 'GPUNetworkTimeRouting',
        variant: 'absolute-arrivals',
        invocationCount: props.stateVertices.length,
        bindings: [
          {name: 'elapsed', view: elapsed, type: 'f32', access: 'read'},
          {name: 'departure', view: props.departure, type: 'f32', access: 'read'},
          {name: 'arrivals', view: props.output.arrivals, type: 'f32', access: 'read_write'}
        ],
        body: `let duration = elapsed[elapsedOffset + index];
var missingBits = 0x7f800000u;
arrivals[arrivalsOffset + index] = select(bitcast<f32>(missingBits), departure[departureOffset] + duration, (bitcast<u32>(duration) & 0x7f800000u) != 0x7f800000u);`
      })
    );
    return nodes;
  }
}

function getTimeRoutingOutputViews(output: GPUNetworkTimeRoutingProps['output']): GraphDataView[] {
  return [
    output.arrivals,
    ...(output.elapsedTimes ? [output.elapsedTimes] : []),
    output.predecessorTransitions,
    output.summary,
    ...getNetworkOptimizationStatusViews(output.status)
  ];
}

/**
 * Prepares reachable routing states on the CPU using the proper-prefix automaton of forbidden
 * edge sequences. Parallel edges retain distinct identities. The expanded graph can contain up
 * to 256 states and 4,096 transitions.
 *
 * @returns Packed Uint32Array columns with their GPU `formats` and row counts:
 * - `stateVertices`: `uint32`, `stateCount` rows, one physical vertex per expanded state.
 * - `transitions`: `uint32x2`, `transitionCount` rows, one interleaved pair of expanded-state
 *   endpoints per allowed transition (two array elements per row).
 * - `originalEdges`: `uint32`, `transitionCount` rows, one original edge index per transition
 *   for mapping costs and predecessors.
 * - `initialStates`: `uint32`, `nodeCount` rows, one empty-history state index per physical vertex.
 */
export function prepareGPUNetworkRestrictionStates(props: {
  nodeCount: number;
  /** Directed [source, target] vertex pairs, indexed by original edge ID. */
  edges: readonly GPUNetworkEdgeEndpoints[];
  /** Forbidden sequences of original edge indices. */
  restrictions: readonly (readonly number[])[];
}) {
  const id = 'prepareGPUNetworkRestrictionStates';
  validateRequirement(
    id,
    typeof props === 'object' && props !== null && !Array.isArray(props),
    `options must be passed as {nodeCount, edges, restrictions}; received ${typeof props === 'number' ? props : typeof props}`
  );
  const {nodeCount, edges, restrictions} = props;
  validateCapacity(id, 'nodeCount', nodeCount, 1, MAXIMUM_STATE_COUNT);
  for (const [edgeIndex, edge] of edges.entries()) {
    validateRequirement(
      id,
      Array.isArray(edge) && edge.length === 2,
      `edges[${edgeIndex}] must be the tuple [source, target]; received ${Array.isArray(edge) ? `length ${edge.length}` : typeof edge}`
    );
    const [source, target] = edge;
    validateRequirement(
      id,
      Number.isInteger(source) &&
        Number.isInteger(target) &&
        source >= 0 &&
        source < nodeCount &&
        target >= 0 &&
        target < nodeCount,
      `edges[${edgeIndex}] source and target must be integers in [0, ${nodeCount - 1}]; received [${source}, ${target}]`
    );
  }
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
    edges.forEach(([source, target], originalEdge) => {
      if (source !== stateVertices[state]) return;
      const sequence = [...prefix, originalEdge];
      if (restrictions.some(restriction => endsWith(sequence, restriction))) return;
      let nextPrefix = 0;
      prefixes.forEach((candidate, candidateIndex) => {
        if (candidate.length > prefixes[nextPrefix].length && endsWith(sequence, candidate))
          nextPrefix = candidateIndex;
      });
      const key = `${nextPrefix}:${target}`;
      let destination = states.get(key);
      if (destination === undefined) {
        destination = stateVertices.length;
        validateMaximumCount(id, 'restriction states', destination + 1, MAXIMUM_STATE_COUNT);
        states.set(key, destination);
        stateVertices.push(target);
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
    nodeCount,
    stateCount: stateVertices.length,
    transitionCount: originalEdges.length,
    formats: {
      stateVertices: 'uint32',
      transitions: 'uint32x2',
      originalEdges: 'uint32',
      initialStates: 'uint32'
    } as const,
    stateVertices: Uint32Array.from(stateVertices),
    transitions: Uint32Array.from(transitions),
    originalEdges: Uint32Array.from(originalEdges),
    initialStates: Uint32Array.from({length: nodeCount}, (_, vertex) => vertex)
  };
}
