// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateNetworkViewBindings,
  validateNetworkWritableViews,
  alignElementCount,
  NETWORK_ALIGNMENT_ELEMENTS
} from '../network-view-utils';

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateGraphOutputsDisjointFromInputs,
  validateSinglePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPUNetworkTimeRouting,
  type GPUNetworkTimeRoutingProps
} from '../network-optimization/gpu-network-time-routing';

const OPERATION = 'GPUNetworkTravelTimePercentiles';

/** Most departure times of the window: the per-vertex sort is a private array of this size. */
export const GPU_NETWORK_TRAVEL_TIME_PERCENTILES_MAXIMUM_DEPARTURE_COUNT = 60;
/** Most percentiles: one output column each, and the percentile list is compiled into the shader. */
export const GPU_NETWORK_TRAVEL_TIME_PERCENTILES_MAXIMUM_PERCENTILE_COUNT = 16;
/** Rows of `window`: first departure and step between departures. */
const WINDOW_LENGTH = 2;
/** Words per routing summary (valid, converged, target state, arrival bits). */
const ROUTING_SUMMARY_LENGTH = 4;

/**
 * Properties for {@link GPUNetworkTravelTimePercentiles}.
 *
 * Compile-time: the restriction automaton shape, `departureCount`, `percentiles`, `vertexCount`.
 * Per-frame: `window`, `costs` and `initialState`.
 */
export type GPUNetworkTravelTimePercentilesProps = Pick<
  GPUNetworkTimeRoutingProps,
  'stateVertices' | 'transitions' | 'costModel' | 'samplesPerTransition' | 'waitingPolicy' | 'costs'
> & {
  /** Prefix for generated node and transient IDs. Defaults to `'network-travel-time-percentiles'`. */
  id?: string;
  /** Number of physical vertices; every `stateVertices` entry is below it. */
  vertexCount: number;
  /** One row: the initial state (the origin's empty-history state). */
  initialState: GraphDataView<'uint32'>;
  /** Departures in the window, 1 to 60: `window[0] + k * window[1]` for `k < departureCount`. */
  departureCount: number;
  /**
   * Percentiles in `(0, 100]`, ascending, at most 16. The `p`-th percentile of `n` departures is
   * the value of 1-based rank `ceil(p / 100 * n)` of the ascending travel times (nearest rank),
   * departures that cannot reach a vertex counting as `+Infinity`.
   */
  percentiles: readonly number[];
  /** Two rows: the first departure time and the step between departures (nonnegative, relative). */
  window: GraphDataView<'float32'>;
  /** Forwarded to {@link GPUNetworkTimeRouting.props.maximumRelaxations}. */
  maximumRelaxations?: number;
  output: {
    /**
     * `vertexCount * percentiles.length` rows, vertex-major: travel time (arrival minus departure,
     * minimum over the vertex's states) at each percentile; `+Infinity` when too few departures
     * reach the vertex.
     */
    travelTimes: GraphDataView<'float32'>;
    /** Optional `vertexCount` rows: how many departures reach each vertex. */
    reachedCounts?: GraphDataView<'uint32'>;
    /**
     * Optional `departureCount * 2` rows: per departure `[valid, converged]` from its routing run.
     * A run that is not valid or not converged makes its row of the percentiles unreliable.
     */
    status?: GraphDataView<'uint32'>;
  };
};

/**
 * Travel-time percentiles over a departure window on a FIFO time-dependent network: the
 * departure-time analogue of r5r's `time_window` (one result per departure, then percentiles of
 * the travel time at each destination).
 *
 * The window is an unrolled batch: one {@link GPUNetworkTimeRouting} per departure, in the same
 * graph, each over the same restriction automaton. Because the arrival functions are FIFO, the
 * routing is exact for every departure; a vertex's travel time is the earliest arrival over its
 * states minus the departure. The percentile kernel then sorts the `departureCount` values of
 * each vertex (one thread per vertex, at most 60 values).
 *
 * Reference tier (contributor contract rule 2): each routing is the single-invocation search of
 * {@link GPUNetworkTimeRouting} (at most 256 states and 4096 transitions), so a window costs
 * `departureCount` such dispatches. The window is regular; a list of arbitrary departures is not
 * supported.
 */
export class GPUNetworkTravelTimePercentiles implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUNetworkTravelTimePercentilesProps;
  /** One departure-window node, one solve per departure and one percentile publication. */
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPUNetworkTravelTimePercentilesProps) {
    this.id = props.id ?? 'network-travel-time-percentiles';
    this.props = props;
    validateNetworkWritableViews(this.id, props.output);
    const {id} = this;
    this.estimatedCommandNodeCount = 2 + props.departureCount;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateCapacity(
      id,
      'departureCount',
      props.departureCount,
      1,
      GPU_NETWORK_TRAVEL_TIME_PERCENTILES_MAXIMUM_DEPARTURE_COUNT
    );
    validateCapacity(
      id,
      'percentiles rows',
      props.percentiles.length,
      1,
      GPU_NETWORK_TRAVEL_TIME_PERCENTILES_MAXIMUM_PERCENTILE_COUNT
    );
    validateCapacity(id, 'vertexCount', props.vertexCount);
    validateRequirement(
      id,
      props.percentiles.every(
        (percentile, index) =>
          Number.isFinite(percentile) &&
          percentile > 0 &&
          percentile <= 100 &&
          (index === 0 || percentile >= props.percentiles[index - 1])
      ),
      'percentiles must be ascending numbers in (0, 100]'
    );
    validateSinglePackedView(props.window, ['float32'], `${id} window`);
    validateSinglePackedView(props.initialState, ['uint32'], `${id} initialState`);
    validateSinglePackedView(props.output.travelTimes, ['float32'], `${id} output.travelTimes`);
    validateViewLength(id, 'window', props.window, WINDOW_LENGTH);
    validateViewLength(id, 'initialState', props.initialState, 1);
    validateViewLength(
      id,
      'output.travelTimes',
      props.output.travelTimes,
      props.vertexCount * props.percentiles.length
    );
    if (props.output.reachedCounts) {
      validateSinglePackedView(
        props.output.reachedCounts,
        ['uint32'],
        `${id} output.reachedCounts`
      );
      validateViewLength(id, 'output.reachedCounts', props.output.reachedCounts, props.vertexCount);
    }
    if (props.output.status) {
      validateSinglePackedView(props.output.status, ['uint32'], `${id} output.status`);
      validateViewLength(id, 'output.status', props.output.status, 2 * props.departureCount);
    }
    validateGraphOutputsDisjointFromInputs(id, Object.values(props.output), [
      props.stateVertices,
      props.transitions,
      props.costs,
      props.initialState,
      props.window
    ]);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    validateNetworkViewBindings(id, graph, [
      props.stateVertices,
      props.transitions,
      props.costs,
      props.initialState,
      props.window,
      ...Object.values(output)
    ]);
    const stateCount = props.stateVertices.length;
    const departureCount = props.departureCount;
    // Each solve binds the same view lengths and alignment residue. Padding is scratch only;
    // departure indices never reach WGSL text or pipeline constants.
    const stateStride = alignElementCount(stateCount);
    const parameterStride = NETWORK_ALIGNMENT_ELEMENTS;
    validateNetworkViewBindings(id, graph, [], {
      departures: departureCount * parameterStride * 4,
      arrivals: departureCount * stateStride * 4,
      summaries: departureCount * parameterStride * 4,
      endpoints: 8
    });
    const departures = createTransientView(
      graph,
      `${id}-departures`,
      'float32',
      departureCount * parameterStride
    );
    const arrivals = createTransientView(
      graph,
      `${id}-arrivals`,
      'float32',
      departureCount * stateStride
    );
    const predecessors = createTransientView(
      graph,
      `${id}-predecessors`,
      'uint32',
      departureCount * stateStride
    );
    const summaries = createTransientView(
      graph,
      `${id}-summaries`,
      'uint32',
      departureCount * parameterStride
    );
    const endpoints = createTransientView(graph, `${id}-endpoints`, 'uint32', 2);
    const slice = <Format extends 'float32' | 'uint32'>(
      view: GraphDataView<Format>,
      start: number,
      length: number
    ): GraphDataView<Format> =>
      graph.createDataView(view.buffer, {
        format: view.format,
        length,
        byteOffset: view.byteOffset + start * view.byteStride,
        byteStride: view.byteStride,
        rowByteLength: view.rowByteLength
      }) as GraphDataView<Format>;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-window`,
        operation: OPERATION,
        variant: 'departure-window',
        invocationCount: Math.max(departureCount, 2),
        bindings: [
          {name: 'window', view: props.window, type: 'f32', access: 'read'},
          {name: 'initialState', view: props.initialState, type: 'u32', access: 'read'},
          {name: 'departures', view: departures, type: 'f32', access: 'read_write'},
          {name: 'endpoints', view: endpoints, type: 'u32', access: 'read_write'}
        ],
        // The routing's destination is unused here: every state's arrival is read afterwards.
        body: `if (index < ${departureCount}u) { departures[departuresOffset + index * ${parameterStride}u] = window[windowOffset] + f32(index) * window[windowOffset + 1u]; }
if (index == 0u) { endpoints[endpointsOffset] = initialState[initialStateOffset]; endpoints[endpointsOffset + 1u] = 0u; }`
      })
    ];
    for (let departure = 0; departure < departureCount; departure++) {
      nodes.push(
        ...new GPUNetworkTimeRouting({
          id: `${id}-routing-${departure}`,
          stateVertices: props.stateVertices,
          transitions: props.transitions,
          costModel: props.costModel,
          samplesPerTransition: props.samplesPerTransition,
          waitingPolicy: props.waitingPolicy,
          costs: props.costs,
          endpoints,
          departure: slice(departures, departure * parameterStride, 1),
          maximumRelaxations: props.maximumRelaxations,
          output: {
            arrivals: slice(arrivals, departure * stateStride, stateCount),
            predecessorTransitions: slice(predecessors, departure * stateStride, stateCount),
            summary: slice(summaries, departure * parameterStride, ROUTING_SUMMARY_LENGTH)
          }
        }).getCommandNodes(graph)
      );
    }
    const percentileCount = props.percentiles.length;
    const ranks = props.percentiles.map(percentile =>
      Math.max(1, Math.ceil((percentile / 100) * departureCount - 1e-9))
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-percentiles`,
        operation: OPERATION,
        variant: 'nearest-rank',
        invocationCount: props.vertexCount,
        bindings: [
          {name: 'stateVertices', view: props.stateVertices, type: 'u32', access: 'read'},
          {name: 'arrivals', view: arrivals, type: 'f32', access: 'read'},
          {name: 'departures', view: departures, type: 'f32', access: 'read'},
          {name: 'summaries', view: summaries, type: 'u32', access: 'read'},
          {name: 'travelTimes', view: output.travelTimes, type: 'f32', access: 'read_write'},
          ...(output.reachedCounts
            ? [
                {
                  name: 'reachedCounts',
                  view: output.reachedCounts,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : []),
          ...(output.status
            ? [
                {
                  name: 'status',
                  view: output.status,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        declarations: `const RANKS = array<u32, ${percentileCount}>(${ranks.map(rank => `${rank}u`).join(',')});`,
        // Insertion sort of at most 60 values. Infinity (unreached or invalid run) sorts last.
        body: `var infinityBits = 0x7f800000u; let infinity = bitcast<f32>(infinityBits);
var values: array<f32, ${departureCount}>;
var reached = 0u;
for (var departure = 0u; departure < ${departureCount}u; departure++) {
  var best = infinity;
  if ((summaries[summariesOffset + departure * ${parameterStride}u] & 1u) != 0u) {
    for (var state = 0u; state < ${stateCount}u; state++) {
      if (stateVertices[stateVerticesOffset + state] == index) {
        let arrival = arrivals[arrivalsOffset + departure * ${stateStride}u + state];
        best = min(best, arrival - departures[departuresOffset + departure * ${parameterStride}u]);
      }
    }
  }
  if (best < infinity) { reached++; }
  var slot = departure;
  while (slot > 0u && values[slot - 1u] > best) { values[slot] = values[slot - 1u]; slot--; }
  values[slot] = best;
}
for (var column = 0u; column < ${percentileCount}u; column++) { travelTimes[travelTimesOffset + index * ${percentileCount}u + column] = values[RANKS[column] - 1u]; }
${output.reachedCounts ? 'reachedCounts[reachedCountsOffset + index] = reached;' : ''}
${
  output.status
    ? `if (index == 0u) { for (var departure = 0u; departure < ${departureCount}u; departure++) { status[statusOffset + 2u * departure] = summaries[summariesOffset + departure * ${parameterStride}u]; status[statusOffset + 2u * departure + 1u] = summaries[summariesOffset + departure * ${parameterStride}u + 1u]; } }`
    : ''
}`
      })
    );
    return nodes;
  }
}
