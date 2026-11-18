// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateNetworkViewBindings, validateNetworkWritableViews} from '../network-view-utils';

import {clamp} from '@math.gl/core';
import type {Device} from '@luma.gl/core';
import {
  resolveGPUSpatialContext,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT
} from '../../analysis-core/contracts/spatial-context';
import {
  createTransientView,
  GPUReduction,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getStorageBindingElementLimit,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateCapacity,
  validateCommandNodeBudget,
  validateMaximumCount,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUNetworkTopologyViews,
  validateGPUNetworkTopology,
  type GPUNetworkTopology
} from '../../analysis-core/contracts/network-topology';
import {IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {PERMUTATION_RANDOM_WGSL} from '../../analysis-core/kernels/permutation-random';
import type {
  GPUSpatialContext,
  GPUSpatialContextSupport
} from '../../analysis-core/contracts/spatial-context';
import {GPUNetworkSnapping} from '../network-accessibility/index';
import {recommendLaneCount} from '../network-accessibility/network-accessibility-lanes';
import {ACCESSIBILITY_NONE} from '../network-accessibility/network-accessibility-passes';
import {GPUNetworkReachability} from '../network-reachability/index';
import {GPU_NETWORK_K_FUNCTION_PARAMETER_LENGTH} from './network-k-function-parameters';
import {PHILOX_PURPOSES} from '../../analysis-core/kernels/philox-random';
import {alignElementCount, getSubView} from '../network-view-utils';

const K_FUNCTION_SPATIAL_CONTEXT_SUPPORT: GPUSpatialContextSupport = Object.freeze({
  ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance'
});

const OPERATION = 'GPUNetworkKFunction';

/** Philox counter tag of the simulated event streams, keeping them apart from other users of the RNG. @internal */
export const NETWORK_K_FUNCTION_STREAM = PHILOX_PURPOSES.networkKFunction.id;

/** Largest quantization range of one chunk total in the random edge draw. @internal */
export const NETWORK_K_FUNCTION_LENGTH_RANGE = 2 ** 29;

/**
 * CSR slots per chunk of the two-level random edge draw. With edge lengths quantized to at most
 * {@link NETWORK_K_FUNCTION_EDGE_LENGTH_RANGE} units, a chunk total is at most 2^31. @internal
 */
export const NETWORK_K_FUNCTION_CHUNK_EDGE_COUNT = 128;

/**
 * Units of the longest live edge in the random edge draw: 2^24, the float32 mantissa, so an edge
 * gets at least one unit down to 2^-25 of the longest edge. @internal
 */
export const NETWORK_K_FUNCTION_EDGE_LENGTH_RANGE = 2 ** 24;

/**
 * Quantization range of the coarse level of the random edge draw for `count` rows (chunks): the
 * largest row maps to at most this many units, so the integer prefix sum is at most
 * `count * range <= 2^32 - 1` and cannot wrap (review B-24). A power of two, at most
 * {@link NETWORK_K_FUNCTION_LENGTH_RANGE}. Before review network-paths-04 this range was applied
 * to single edges, which left short edges with no units on large networks. @internal
 */
export function getNetworkKFunctionLengthRange(count: number): number {
  const exponent = Math.floor(Math.log2((2 ** 32 - 1) / Math.max(count, 1)));
  return 2 ** clamp(exponent, 0, 29);
}

/** Number of chunks of the two-level random edge draw for `edgeCount` CSR slots. @internal */
export function getNetworkKFunctionChunkCount(edgeCount: number): number {
  return Math.max(Math.ceil(edgeCount / NETWORK_K_FUNCTION_CHUNK_EDGE_COUNT), 1);
}

/** Largest accepted `bandCount`. */
export const GPU_NETWORK_K_FUNCTION_MAXIMUM_BAND_COUNT = 1024;

/**
 * Largest expanded seed table of {@link GPUNetworkKFunction}, in uint32 words: one 256-byte-aligned
 * slice of `2 * laneCount` words per search batch (`ceil(rows / rowsPerBlock) *
 * ceil(rowsPerBlock / laneCount)` batches). It must fit one storage binding on `device`, so
 * 33554432 on a default device (128 MiB bindings).
 */
export function getGPUNetworkKFunctionMaximumSeedWordCount(device: Device): number {
  return getStorageBindingElementLimit(device, 4);
}

const NONE = 0xffffffff;
/**
 * Elements per 256-byte storage binding alignment. Per-batch and per-block sub-views start at
 * multiples of it, so every batch and block compiles to the same pipelines (library rule 12).
 */
const ALIGNMENT_ELEMENTS = 64;
const DEFAULT_BAND_COUNT = 32;
const DEFAULT_ROWS_PER_BLOCK = 32;
const DEFAULT_LOCAL_ITERATIONS = 16;
/** Matches the `GPUNetworkReachability` default, which the estimate of command nodes must mirror. */
const DEFAULT_MAXIMUM_ITERATIONS = 64;
/**
 * Command nodes added to {@link GPUNetworkKFunction.estimatedCommandNodeCount} for the snapping,
 * reduction and scan primitives, whose node counts depend on data sizes inside published luma.gl
 * classes (a hierarchical reduction or scan adds a pass per factor of the workgroup size, and the
 * snapping BVH adds sort and build passes). Measured at most 131 for snapping with a Morton sort on
 * 600,000 edges and 5 per reduction or scan up to 3,000,000 rows, so 192 covers all of them.
 */
const SHARED_PRIMITIVE_NODE_ALLOWANCE = 192;

/** GPU-written views of {@link GPUNetworkKFunction}. */
export type GPUNetworkKFunctionOutput = {
  /**
   * `(1 + simulationCount) * bandCount` float32 values: `K(d_k)` of the observed pattern in row 0
   * (the first `bandCount` values) and of simulation `s` in row `s`, where
   * `d_k = maximumDistance * k / (bandCount - 1)`, in the context's length unit.
   */
  kValues: GraphDataView<'float32'>;
  /**
   * Optional `(1 + simulationCount) * bandCount` uint32 cumulative counts of unordered event pairs
   * with network distance strictly below `d_k`, laid out like `kValues`.
   */
  pairCounts?: GraphDataView<'uint32'>;
  /**
   * Optional `3 * bandCount` float32 simulation envelope: the minimum, the mean and the maximum
   * of the active simulations per band, in the context's length unit, in three consecutive rows.
   * All zero without active simulations.
   */
  envelope?: GraphDataView<'float32'>;
  /** Optional one-row number of events that snapped onto the network (the `n` of K). */
  snappedEventCount?: GraphDataView<'uint32'>;
  /** GPU-written status words. */
  status?: {
    /**
     * One-row flag: 1 when the snapping candidate search overflowed. Required with
     * `candidateCapacity` (library rule 9), not allowed without it.
     */
    overflow?: GraphDataView<'uint32'>;
    /** Optional one-row flag: 1 when every shortest-path batch reached a fixpoint. */
    converged?: GraphDataView<'uint32'>;
    /** Optional canonical complement: 1 when any shortest-path batch exhausted its round budget. */
    iterationLimitReached?: GraphDataView<'uint32'>;
  };
};

/**
 * Properties for {@link GPUNetworkKFunction}.
 *
 * Compile-time: node, edge and event counts, `bandCount`, `simulationCount`, `rowsPerBlock`,
 * `laneCount`, `maximumIterations`, `localIterations`, `candidateCapacity` and which optional views
 * exist. Per-frame: event and node positions, the topology contents, `maximumDistance`, `networkLength`,
 * `maximumSnapDistance` and `parameters` (seed and active simulation count).
 */
export type GPUNetworkKFunctionProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-k-function'`. */
  id?: string;
  /**
   * Coordinates of `positions` and `topology.nodePositions`, passed to `GPUNetworkSnapping`. Omitted means
   * native planar coordinates. `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}`
   * snaps longitude/latitude events with great-circle distances in the context's length unit
   * (meters by default, or kilometers), as are `topology.weights`, `maximumSnapDistance`,
   * `maximumDistance`, `networkLength`, `kValues` and `envelope`. Local planar frames must be
   * azimuthal-equidistant, not Lambert equal-area.
   */
  spatialContext?: GPUSpatialContext;
  /** Events (for example points of interest) to snap onto the network. */
  positions: GraphDataView<'float32x2'>;
  /**
   * The network, with node positions in the same coordinates as `positions`. It must list both
   * directions of every road. Edge slots past `offsets[nodeCount]` (a topology shorter than its
   * edge capacity) are not part of the network: they are never snapped to or drawn by the
   * simulation. `weights` are the non-negative lengths (costs) per edge, equal for the two
   * directions of a road, in the context's length unit.
   */
  topology: GPUNetworkTopology & {nodePositions: GraphDataView<'float32x2'>};
  /**
   * Optional one-row maximum snap distance in the context's length unit. Events farther from every edge are excluded.
   * On longitude/latitude this distance arrives as a GPU view, so no short-distance warning is possible;
   * below about 17 m use createGPULocalFrame (see Precision of longitude/latitude positions).
   */
  maximumSnapDistance?: GraphDataView<'float32'>;
  /**
   * Optional compile-time candidate capacity of `GPUNetworkSnapping`. Requires `maximumSnapDistance`.
   * Without it the events are snapped by an exact scan of every event against every edge with no
   * index, and the snapping's unindexed-work limit is raised for it, so snapping becomes slow once
   * events times edges reaches about 2^27 (a frame).
   */
  candidateCapacity?: number;
  /** Compile-time. Morton-sort edges before the BVH build of the snapping candidate search. */
  spatialSort?: boolean;
  /**
   * One-row largest distance `d_max` in the context's length unit: the distance bands span `[0, d_max]`.
   * On longitude/latitude this distance arrives as a GPU view, so no short-distance warning is possible;
   * below about 17 m use createGPULocalFrame (see Precision of longitude/latitude positions).
   */
  maximumDistance: GraphDataView<'float32'>;
  /**
   * One-row total length `L` in the context's length unit with every road counted once, that is, half of the sum
   * of `topology.weights` over the live rows `0..offsets[nodeCount]` of a topology that lists both
   * directions.
   * It scales K. The random edge draw does not read it: edge lengths are quantized against the
   * longest live edge (a GPU maximum) and chunk totals against the largest chunk total, so a wrong
   * `networkLength` changes K but never the draw.
   */
  networkLength: GraphDataView<'float32'>;
  /**
   * `GPU_NETWORK_K_FUNCTION_PARAMETER_LENGTH` uint32 elements written with
   * `getGPUNetworkKFunctionParameterValues`. Required.
   */
  parameters: GraphDataView<'uint32'>;
  /** Compile-time number of distance bands (thresholds) `B`, 2 to 1024. Defaults to 32. */
  bandCount?: number;
  /** Compile-time number of simulated patterns `S` of the envelope. Defaults to 0. */
  simulationCount?: number;
  /**
   * Compile-time shortest-path searches (rows) run together per block. Scratch memory grows with
   * `rowsPerBlock * nodeCount`. `rowsPerBlock * events` is at most 4294967295 (2^32 - 1), the size
   * of one 1-D dispatch, and the seed table at most
   * {@link getGPUNetworkKFunctionMaximumSeedWordCount} words (33554432 on a default device). Defaults to `recommendLaneCount` for the whole row set, capped at 32.
   */
  rowsPerBlock?: number;
  /** Compile-time lanes (searches sharing one CSR) of each block's searches, shared by all blocks. Defaults to `rowsPerBlock`. */
  laneCount?: number;
  /** Rounds of every search, as `GPUNetworkReachability.maximumIterations`. Defaults to 64. */
  maximumIterations?: number;
  /** Hops chained per round, as `GPUNetworkReachability.localIterations`. Defaults to 16. */
  localIterations?: number;
  /** The GPU-written views. */
  output: GPUNetworkKFunctionOutput;
};

/**
 * Network-constrained Ripley K function (Okabe and Yamada; spaghetti `GlobalAutoK`) of events on
 * a road network, with a Monte Carlo envelope of random points on the same network.
 *
 * Events are snapped onto their nearest edge with `GPUNetworkSnapping`. For every event a
 * multi-source `GPUNetworkReachability` search (as `GPUNetworkCostMatrix`, seeded at the two edge
 * endpoints with the snapped offsets and limited to `maximumDistance`) gives the network distance to
 * every node, and the distance to another event is the cheaper of its two edge endpoints, or the
 * direct offset difference when both lie on the same road. Pairs are counted per band with integer
 * atomics, so the result is deterministic. `K(d) = 2 * pairs(d) * L / n^2` where `pairs(d)`
 * counts unordered pairs closer than `d` (strictly, as spaghetti does), `n` the snapped events and
 * `L` the network length, over `bandCount` thresholds `d_k = maximumDistance * k / (bandCount - 1)`.
 * The `n^2` denominator is spaghetti's `GlobalAutoK` normalization; the ratio-unbiased form of
 * Okabe and Yamada divides by `n (n - 1)`, so multiply by `n / (n - 1)` to compare with it.
 *
 * The envelope draws `simulationCount` patterns of `n` events from a Philox stream keyed by the
 * seed: an edge with probability proportional to its length and a uniform position on it. The edge
 * draw has two integer levels, so that short edges keep their share on large networks. Every live
 * edge length is quantized to 2^24 units of the longest live edge (rounded to nearest). The slots
 * are grouped in chunks of 128 with an integer prefix sum inside each chunk; the chunk totals,
 * shifted right until the largest fits the range `2^floor(log2((2^32 - 1) / chunkCount))` (rounded
 * to nearest), get a second integer prefix sum. A draw picks a chunk by the coarse sum and then an
 * edge by the chunk's own sum, both with unbiased bounded integers. An edge shorter than 2^-25 of
 * the longest is never drawn, and a chunk's share is exact to half a coarse unit (under
 * `2 * chunkCount / 2^32` of the largest chunk total). Everything is an integer or an exact maximum, so the
 * draw does not depend on thread order. Searches of all patterns share one set of row blocks
 * (`rowsPerBlock` rows each, searched by one `GPUNetworkReachability` per block over one lane-expanded CSR shared by all blocks), so peak scratch
 * memory is set by `rowsPerBlock`, not by the event or simulation count. Unlike spaghetti, whose
 * envelope scales its extremes by the significance threshold, the envelope here is the plain
 * minimum, mean and maximum of the simulated K values.
 *
 * Network distances come from `topology.weights` only; positions are used to snap events. With a
 * great-circle `spatialContext` events are snapped on longitude/latitude positions. Every distance
 * input and output, including `topology.weights`, is in the context's length unit.
 *
 * Roads that share both end nodes (parallel edges) count as one road when two events fall on them.
 * Each block adds about `maximumIterations + 6` command nodes and the total is
 * `ceil((1 + simulationCount) * events / rowsPerBlock)` blocks.
 */
export class GPUNetworkKFunction implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNetworkKFunctionProps;
  /** Number of events (positions). */
  readonly eventCount: number;
  /** Number of network nodes. */
  readonly nodeCount: number;
  /** Number of CSR edges. */
  readonly edgeCount: number;
  /** Resolved number of distance bands. */
  readonly bandCount: number;
  /** Resolved number of simulated patterns. */
  readonly simulationCount: number;
  /** Total searched rows, `(1 + simulationCount) * eventCount`. */
  readonly rowCount: number;
  /** Resolved rows per block. */
  readonly rowsPerBlock: number;
  /** Number of row blocks. */
  readonly blockCount: number;
  /**
   * Upper bound of the command nodes `getCommandNodes` returns, checked against the shared command
   * node budget at construction. Every node this class creates is counted exactly (the cost-matrix
   * searches, `count` nodes and the fixed kernels). The snapping, reduction and scan primitives are
   * counted as a flat allowance of 192, so the value exceeds the true count by at most 192.
   */
  readonly estimatedCommandNodeCount: number;
  /** Resolved spatial context of the event snapping. */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUNetworkKFunctionProps) {
    this.id = props.id ?? 'network-k-function';
    this.props = props;
    validateNetworkWritableViews(this.id, props.output);
    const {id} = this;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      K_FUNCTION_SPATIAL_CONTEXT_SUPPORT
    );
    const {output} = props;
    const {nodeCount, edgeCount} = validateGPUNetworkTopology(id, props.topology, {
      weights: true,
      nodePositions: true
    });
    for (const [name, view] of [
      ['parameters', props.parameters],
      ['output.pairCounts', output.pairCounts],
      ['output.snappedEventCount', output.snappedEventCount],
      ['output.status.overflow', output.status?.overflow],
      ['output.status.converged', output.status?.converged],
      ['output.status.iterationLimitReached', output.status?.iterationLimitReached]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['uint32'], `${id} ${name}`);
      }
    }
    for (const [name, view] of [
      ['maximumSnapDistance', props.maximumSnapDistance],
      ['maximumDistance', props.maximumDistance],
      ['networkLength', props.networkLength],
      ['output.kValues', output.kValues],
      ['output.envelope', output.envelope]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['float32'], `${id} ${name}`);
      }
    }
    validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);
    this.eventCount = props.positions.length;
    this.nodeCount = nodeCount;
    this.edgeCount = edgeCount;
    this.bandCount = props.bandCount ?? DEFAULT_BAND_COUNT;
    this.simulationCount = props.simulationCount ?? 0;
    if (this.eventCount < 2) {
      throw new Error(`${id} requires at least two events`);
    }
    if (this.edgeCount < 1) {
      throw new Error(`${id} requires at least one edge`);
    }
    validateCapacity(id, 'bandCount', this.bandCount, 2, GPU_NETWORK_K_FUNCTION_MAXIMUM_BAND_COUNT);
    validateCapacity(id, 'simulationCount', this.simulationCount, 0);
    if (props.parameters.length < GPU_NETWORK_K_FUNCTION_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_NETWORK_K_FUNCTION_PARAMETER_LENGTH} elements`
      );
    }
    for (const [name, view] of [
      ['maximumSnapDistance', props.maximumSnapDistance],
      ['maximumDistance', props.maximumDistance],
      ['networkLength', props.networkLength],
      ['output.snappedEventCount', output.snappedEventCount],
      ['output.status.overflow', output.status?.overflow],
      ['output.status.converged', output.status?.converged],
      ['output.status.iterationLimitReached', output.status?.iterationLimitReached]
    ] as const) {
      if (view && view.length !== 1) {
        throw new Error(`${id} ${name} must contain exactly one row`);
      }
    }
    const patternCount = 1 + this.simulationCount;
    for (const [name, view, length] of [
      ['output.kValues', output.kValues, patternCount * this.bandCount],
      ['output.pairCounts', output.pairCounts, patternCount * this.bandCount],
      ['output.envelope', output.envelope, 3 * this.bandCount]
    ] as const) {
      if (view && view.length !== length) {
        throw new Error(`${id} ${name} length must be ${length}`);
      }
    }
    if (output.status?.overflow && props.candidateCapacity === undefined) {
      throw new Error(`${id} output.status.overflow requires candidateCapacity`);
    }
    if (props.candidateCapacity !== undefined && !output.status?.overflow) {
      throw new Error(`${id} candidateCapacity requires output.status.overflow`);
    }
    this.rowCount = patternCount * this.eventCount;
    this.rowsPerBlock =
      props.rowsPerBlock ??
      clamp(
        Math.min(
          DEFAULT_ROWS_PER_BLOCK,
          recommendLaneCount({
            rowCount: this.rowCount,
            nodeCount: this.nodeCount,
            edgeCount: this.edgeCount
          })
        ),
        1,
        this.rowCount
      );
    validateCapacity(id, 'rowsPerBlock', this.rowsPerBlock, 1, this.rowCount);
    // The pair-count kernel launches `rowsPerBlock * events` invocations in one 1-D dispatch.
    validateMaximumCount(
      id,
      'rowsPerBlock x events',
      this.rowsPerBlock * this.eventCount,
      0xffffffff
    );
    this.blockCount = Math.ceil(this.rowCount / this.rowsPerBlock);
    const laneCount = Math.min(props.laneCount ?? this.rowsPerBlock, this.rowsPerBlock);
    const batchCount = this.blockCount * Math.ceil(this.rowsPerBlock / laneCount);
    // Reachability per batch: initialize, seed, one node per round, finalize when `converged`.
    const reachabilityNodeCount =
      2 +
      (props.maximumIterations ?? DEFAULT_MAXIMUM_ITERATIONS) +
      Number(Boolean(output.status?.converged || output.status?.iterationLimitReached));
    this.estimatedCommandNodeCount =
      SHARED_PRIMITIVE_NODE_ALLOWANCE +
      // event-flags, clear, expand-seeds, block-info, finalize
      5 +
      // live-weights, quantize, chunk-prefix, chunk-quantize, simulate
      (this.simulationCount > 0 ? 5 : 0) +
      batchCount * reachabilityNodeCount +
      this.blockCount +
      Number(Boolean(output.envelope)) +
      2 * Number(Boolean(output.status?.converged || output.status?.iterationLimitReached));
    // The rounds of every block are unrolled; lower simulationCount or maximumIterations, or raise
    // rowsPerBlock.
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    const outputs = getOutputs(props);
    validateGraphOutputsDisjointFromInputs(id, outputs, getInputs(props));
    const outputBuffers = outputs.filter(view => view !== undefined).map(view => view.buffer);
    if (new Set(outputBuffers).size !== outputBuffers.length) {
      throw new Error(`${id} outputs must use separate buffers`);
    }
  }

  /**
   * Returns the snapping nodes, the event count, the random-edge preparation (`quantize`,
   * `chunk-prefix`, `chunk-quantize`, a `GPUScan` prefix sum of the chunks, `simulate`; only with
   * simulations), one `clear`, then per block the cost-matrix nodes and a `count`
   * node, and finally `finalize`, optional `envelope` and `converged` nodes.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, eventCount, nodeCount, edgeCount, bandCount, simulationCount} = this;
    const {rowCount, rowsPerBlock, blockCount} = this;
    const {output} = props;
    const {offsets, neighbors, weights} = props.topology;
    validateNetworkViewBindings(id, graph, [...getInputs(props), ...getOutputs(props)]);
    const seedLaneCount = Math.min(props.laneCount ?? rowsPerBlock, rowsPerBlock);
    const plannedBatchesPerBlock = Math.ceil(rowsPerBlock / seedLaneCount);
    validateNetworkViewBindings(id, graph, [], {
      seedTable: rowCount * 2 * 4,
      events: eventCount * 4,
      histogram: (1 + simulationCount) * bandCount * 4,
      lengthPrefix: (alignElementCount(edgeCount) + getNetworkKFunctionChunkCount(edgeCount)) * 4,
      blockCosts: plannedBatchesPerBlock * alignElementCount(seedLaneCount * nodeCount) * 4,
      expandedSeeds: blockCount * plannedBatchesPerBlock * alignElementCount(2 * seedLaneCount) * 4,
      frontier: seedLaneCount * nodeCount * 8,
      batchConverged: blockCount * plannedBatchesPerBlock * ALIGNMENT_ELEMENTS * 4,
      blockInfo: blockCount * ALIGNMENT_ELEMENTS * 4
    });
    validateMaximumCount(
      id,
      'expanded seed words (rows, rowsPerBlock, laneCount)',
      blockCount * Math.ceil(rowsPerBlock / seedLaneCount) * alignElementCount(2 * seedLaneCount),
      getGPUNetworkKFunctionMaximumSeedWordCount(graph.device)
    );
    const nodes: GPUCommandNode<Parameters>[] = [];
    const patternCount = 1 + simulationCount;
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);

    // Seed table: row r = pattern * eventCount + event; entries 2r and 2r + 1 are the edge
    // endpoint nodes and the costs from the event to them (NONE and -1 for an inactive event).
    const seedNodes = u32('seed-nodes', 2 * rowCount);
    const seedCosts = createTransientView(graph, `${id}-seed-costs`, 'float32', 2 * rowCount);
    const snappedEdges = u32('snapped-edges', eventCount);
    nodes.push(
      ...new GPUNetworkSnapping({
        id: `${id}-snap`,
        spatialContext: this.spatialContext,
        positions: props.positions,
        topology: props.topology,
        maximumSnapDistance: props.maximumSnapDistance,
        candidateCapacity: props.candidateCapacity,
        // Without a radius there is no indexed search to fall back to, and every event count times
        // edge count built before snapping had a limit, so the K function keeps accepting them.
        maximumUnindexedWork: Number.MAX_SAFE_INTEGER,
        spatialSort: props.spatialSort,
        startDirection: 'both',
        output: {
          snappedEdges,
          startNodes: getSubView(graph, seedNodes, 0, 2 * eventCount),
          startCosts: getSubView(graph, seedCosts, 0, 2 * eventCount),
          status: {overflow: output.status?.overflow}
        }
      }).getCommandNodes(graph)
    );
    // Snapped events: one flag per event, summed by a hierarchical reduction (rule 14).
    const validCount = output.snappedEventCount ?? u32('event-count', 1);
    const validFlags = u32('event-flags', eventCount);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-event-flags`,
        operation: OPERATION,
        variant: 'event-flags',
        bindings: [
          {name: 'seedNodes', view: seedNodes, type: 'u32', access: 'read'},
          {name: 'validFlags', view: validFlags, type: 'u32', access: 'read_write'}
        ],
        invocationCount: eventCount,
        declarations: `const NODE_COUNT: u32 = ${nodeCount}u;`,
        body: `validFlags[validFlagsOffset + index] =
    select(0u, 1u, seedNodes[seedNodesOffset + 2u * index] < NODE_COUNT);`
      }),
      ...new GPUReduction({
        id: `${id}-event-count`,
        input: validFlags,
        output: validCount,
        operation: 'sum'
      }).getCommandNodes(graph)
    );

    if (simulationCount > 0) {
      const quantized = u32('quantized-lengths', edgeCount);
      // Two-level draw (review network-paths-04). One buffer holds both prefix sums so that the
      // simulate kernel stays within 8 storage bindings: the prefix inside each chunk at slot
      // `edge`, and the prefix over the quantized chunk totals at CHUNK_PREFIX_BASE + chunk.
      const chunkCount = getNetworkKFunctionChunkCount(edgeCount);
      const chunkPrefixBase = alignElementCount(edgeCount);
      const prefix = u32('length-prefix', chunkPrefixBase + chunkCount);
      const chunkPrefix = getSubView(graph, prefix, chunkPrefixBase, chunkCount);
      const chunkTotals = u32('chunk-totals', chunkCount);
      const chunkMaximum = u32('chunk-maximum', 1);
      const quantizedChunks = u32('quantized-chunks', chunkCount);
      const chunkRangeBits = Math.log2(getNetworkKFunctionLengthRange(chunkCount));
      const liveWeights = createTransientView(graph, `${id}-live-weights`, 'float32', edgeCount);
      const liveMaximum = createTransientView(graph, `${id}-live-maximum`, 'float32', 1);
      // Rows past offsets[NODE_COUNT] (a CSR shorter than its capacity) are not edges: drawing
      // one would emit a node past NODE_COUNT. Non-finite or negative weights are not drawable.
      const liveEdgeSource = `${IS_FINITE_BITS_WGSL}
fn getLiveWeight(edge: u32) -> f32 {
  let weight = weights[weightsOffset + edge];
  let isLive = edge < offsets[offsetsOffset + NODE_COUNT] &&
    neighbors[neighborsOffset + edge] < NODE_COUNT && isFiniteBits(weight) && weight >= 0.0;
  return select(0.0, weight, isLive);
}`;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-live-weights`,
          operation: OPERATION,
          variant: 'live-weights',
          bindings: [
            {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: neighbors, type: 'u32', access: 'read'},
            {name: 'weights', view: weights, type: 'f32', access: 'read'},
            {name: 'liveWeights', view: liveWeights, type: 'f32', access: 'read_write'}
          ],
          invocationCount: edgeCount,
          declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
${liveEdgeSource}`,
          body: 'liveWeights[liveWeightsOffset + index] = getLiveWeight(index);'
        }),
        ...new GPUReduction({
          id: `${id}-live-maximum`,
          input: liveWeights,
          output: liveMaximum,
          operation: 'max'
        }).getCommandNodes(graph),
        // The longest live edge maps to 2^24 units, rounded to nearest (half up, on the fraction, so
        // no float tie is involved). The ratio is at most 1 and the scale a power of two, so a
        // value is at most 2^24 whatever `networkLength` says (review B-24). A maximum is exact
        // and order independent, so the draw is reproducible on the CPU.
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-quantize`,
          operation: OPERATION,
          variant: 'quantize',
          bindings: [
            {name: 'liveWeights', view: liveWeights, type: 'f32', access: 'read'},
            {name: 'liveMaximum', view: liveMaximum, type: 'f32', access: 'read'},
            {name: 'quantized', view: quantized, type: 'u32', access: 'read_write'}
          ],
          invocationCount: edgeCount,
          body: `let maximum = liveMaximum[liveMaximumOffset];
  var value = 0u;
  if (maximum > 0.0) {
    let scaled = min(liveWeights[liveWeightsOffset + index] / maximum, 1.0) * ${NETWORK_K_FUNCTION_EDGE_LENGTH_RANGE}.0;
    let whole = floor(scaled);
    value = u32(whole) + select(0u, 1u, scaled - whole >= 0.5);
  }
  quantized[quantizedOffset + index] = value;`
        }),
        // One thread per chunk: the inclusive integer prefix sum of its slots, and its total (at
        // most 128 * 2^24 = 2^31).
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-chunk-prefix`,
          operation: OPERATION,
          variant: 'chunk-prefix',
          bindings: [
            {name: 'quantized', view: quantized, type: 'u32', access: 'read'},
            {name: 'prefix', view: prefix, type: 'u32', access: 'read_write'},
            {name: 'chunkTotals', view: chunkTotals, type: 'u32', access: 'read_write'}
          ],
          invocationCount: chunkCount,
          declarations: `const EDGE_COUNT: u32 = ${edgeCount}u;
const CHUNK_EDGE_COUNT: u32 = ${NETWORK_K_FUNCTION_CHUNK_EDGE_COUNT}u;`,
          body: `let firstEdge = index * CHUNK_EDGE_COUNT;
  let endEdge = min(firstEdge + CHUNK_EDGE_COUNT, EDGE_COUNT);
  var running = 0u;
  for (var edge = firstEdge; edge < endEdge; edge++) {
    running += quantized[quantizedOffset + edge];
    prefix[prefixOffset + edge] = running;
  }
  chunkTotals[chunkTotalsOffset + index] = running;`
        }),
        ...new GPUReduction({
          id: `${id}-chunk-maximum`,
          input: chunkTotals,
          output: chunkMaximum,
          operation: 'max'
        }).getCommandNodes(graph),
        // Chunk totals shifted right until the largest is below the coarse range, rounded to
        // nearest, so their prefix sum is at most chunkCount * range <= 2^32 - 1.
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-chunk-quantize`,
          operation: OPERATION,
          variant: 'chunk-quantize',
          bindings: [
            {name: 'chunkTotals', view: chunkTotals, type: 'u32', access: 'read'},
            {name: 'chunkMaximum', view: chunkMaximum, type: 'u32', access: 'read'},
            {name: 'quantizedChunks', view: quantizedChunks, type: 'u32', access: 'read_write'}
          ],
          invocationCount: chunkCount,
          declarations: `const CHUNK_RANGE_BITS: u32 = ${chunkRangeBits}u;`,
          body: `let maximumBits = 32u - countLeadingZeros(chunkMaximum[chunkMaximumOffset]);
  let total = chunkTotals[chunkTotalsOffset + index];
  var value = total;
  if (maximumBits > CHUNK_RANGE_BITS) {
    let shift = min(maximumBits - CHUNK_RANGE_BITS, 31u);
    value = (total >> shift) + ((total >> (shift - 1u)) & 1u);
  }
  quantizedChunks[quantizedChunksOffset + index] = value;`
        }),
        ...new GPUScan({
          id: `${id}-length-scan`,
          input: quantizedChunks,
          output: chunkPrefix,
          mode: 'inclusive'
        }).getCommandNodes(graph),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-simulate`,
          operation: OPERATION,
          variant: 'simulate',
          bindings: [
            {name: 'prefix', view: prefix, type: 'u32', access: 'read'},
            {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: neighbors, type: 'u32', access: 'read'},
            {name: 'weights', view: weights, type: 'f32', access: 'read'},
            {name: 'validCount', view: validCount, type: 'u32', access: 'read'},
            {name: 'parameters', view: props.parameters, type: 'u32', access: 'read'},
            {name: 'outNodes', view: seedNodes, type: 'u32', access: 'read_write'},
            {name: 'outCosts', view: seedCosts, type: 'f32', access: 'read_write'}
          ],
          invocationCount: simulationCount * eventCount,
          declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
const EDGE_COUNT: u32 = ${edgeCount}u;
const EVENT_COUNT: u32 = ${eventCount}u;
const SIMULATION_COUNT: u32 = ${simulationCount}u;
const CHUNK_EDGE_COUNT: u32 = ${NETWORK_K_FUNCTION_CHUNK_EDGE_COUNT}u;
const CHUNK_COUNT: u32 = ${chunkCount}u;
const CHUNK_PREFIX_BASE: u32 = ${chunkPrefixBase}u;
const NONE: u32 = ${NONE}u;
const STREAM_TAG: u32 = ${NETWORK_K_FUNCTION_STREAM}u;
${PERMUTATION_RANDOM_WGSL}`,
          body: `let pattern = 1u + index / EVENT_COUNT;
  let eventIndex = index - (pattern - 1u) * EVENT_COUNT;
  let slot = 2u * (pattern * EVENT_COUNT + eventIndex);
  let activeCount = min(parameters[parametersOffset + 2u], SIMULATION_COUNT);
  let total = prefix[prefixOffset + CHUNK_PREFIX_BASE + CHUNK_COUNT - 1u];
  var firstNode = NONE;
  var secondNode = NONE;
  var firstCost = -1.0;
  var secondCost = -1.0;
  if (pattern <= activeCount && eventIndex < validCount[validCountOffset] && total > 0u) {
    var stream = createPhiloxStream(
      vec2<u32>(parameters[parametersOffset], parameters[parametersOffset + 1u]),
      eventIndex,
      pattern,
      STREAM_TAG
    );
    // Level 1: a chunk, by the prefix sum of the quantized chunk totals. The first chunk whose
    // prefix exceeds the draw has a positive quantized total, so its own total is positive too.
    let chunkDraw = nextPhiloxBelow(&stream, total);
    var chunkLow = 0u;
    var chunkHigh = CHUNK_COUNT;
    while (chunkLow < chunkHigh) {
      let middle = (chunkLow + chunkHigh) / 2u;
      if (prefix[prefixOffset + CHUNK_PREFIX_BASE + middle] > chunkDraw) {
        chunkHigh = middle;
      } else {
        chunkLow = middle + 1u;
      }
    }
    // Level 2: an edge of the chunk, by the chunk's own prefix sum of 2^24-unit edge lengths.
    let chunkFirstEdge = chunkLow * CHUNK_EDGE_COUNT;
    let chunkEndEdge = min(chunkFirstEdge + CHUNK_EDGE_COUNT, EDGE_COUNT);
    let draw = nextPhiloxBelow(&stream, prefix[prefixOffset + chunkEndEdge - 1u]);
    let fractionBits = nextPhiloxUint32(&stream);
    var low = chunkFirstEdge;
    var high = chunkEndEdge;
    while (low < high) {
      let middle = (low + high) / 2u;
      if (prefix[prefixOffset + middle] > draw) {
        high = middle;
      } else {
        low = middle + 1u;
      }
    }
    let edge = low;
    var nodeLow = 0u;
    var nodeHigh = NODE_COUNT;
    while (nodeLow < nodeHigh) {
      let middle = (nodeLow + nodeHigh) / 2u;
      if (offsets[offsetsOffset + middle + 1u] > edge) {
        nodeHigh = middle;
      } else {
        nodeLow = middle + 1u;
      }
    }
    let fraction = f32(fractionBits >> 8u) / 16777216.0;
    let weight = weights[weightsOffset + edge];
    firstNode = nodeLow;
    secondNode = neighbors[neighborsOffset + edge];
    firstCost = fraction * weight;
    secondCost = (1.0 - fraction) * weight;
  }
  outNodes[outNodesOffset + slot] = firstNode;
  outNodes[outNodesOffset + slot + 1u] = secondNode;
  outCosts[outCostsOffset + slot] = firstCost;
  outCosts[outCostsOffset + slot + 1u] = secondCost;`
        })
      );
    }

    const histogram = u32('histogram', patternCount * bandCount);
    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-clear`,
        operation: OPERATION,
        view: histogram,
        type: 'u32',
        value: '0u'
      })
    );

    // Batches of `laneCount` lanes that all read the one CSR (no per-lane copy). Every block runs
    // `batchesPerBlock` full-lane batches, so a short block or batch only masks lanes, and every
    // per-batch or per-block sub-view starts at a multiple of ALIGNMENT_ELEMENTS. All blocks and
    // batches therefore compile to the same pipelines (review E-02, library rule 12).
    const laneCount = Math.min(props.laneCount ?? rowsPerBlock, rowsPerBlock);
    if (laneCount * nodeCount >= ACCESSIBILITY_NONE) {
      throw new Error(`${id} laneCount * nodeCount must fit in uint32`);
    }
    const batchesPerBlock = Math.ceil(rowsPerBlock / laneCount);
    const batchCount = blockCount * batchesPerBlock;
    const batchStride = alignElementCount(laneCount * nodeCount);
    const seedStride = alignElementCount(2 * laneCount);
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'block costs',
      batchesPerBlock * batchStride * 4,
      'lower rowsPerBlock'
    );
    const costs = createTransientView(
      graph,
      `${id}-costs`,
      'float32',
      batchesPerBlock * batchStride
    );
    const expandedSeeds = u32('expanded-seeds', batchCount * seedStride);
    const expandedSeedCosts = createTransientView(
      graph,
      `${id}-expanded-seed-costs`,
      'float32',
      batchCount * seedStride
    );
    nodes.push(
      // Seed slot j of batch b is lane j / 2 of that batch; rows past the block or the row count
      // get no seed, so their lanes stay unreached.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-expand-seeds`,
        operation: OPERATION,
        variant: 'expand-seeds',
        bindings: [
          {name: 'seedNodes', view: seedNodes, type: 'u32', access: 'read'},
          {name: 'seedCosts', view: seedCosts, type: 'f32', access: 'read'},
          {name: 'expandedSeeds', view: expandedSeeds, type: 'u32', access: 'read_write'},
          {name: 'expandedSeedCosts', view: expandedSeedCosts, type: 'f32', access: 'read_write'}
        ],
        invocationCount: batchCount * 2 * laneCount,
        declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
const ROW_COUNT: u32 = ${rowCount}u;
const ROWS_PER_BLOCK: u32 = ${rowsPerBlock}u;
const LANE_COUNT: u32 = ${laneCount}u;
const BATCHES_PER_BLOCK: u32 = ${batchesPerBlock}u;
const SEED_STRIDE: u32 = ${seedStride}u;`,
        body: `let batch = index / (2u * LANE_COUNT);
  let slot = index - batch * 2u * LANE_COUNT;
  let lane = slot / 2u;
  let blockIndex = batch / BATCHES_PER_BLOCK;
  let rowInBlock = (batch - blockIndex * BATCHES_PER_BLOCK) * LANE_COUNT + lane;
  let row = blockIndex * ROWS_PER_BLOCK + rowInBlock;
  var node = ${ACCESSIBILITY_NONE}u;
  var cost = -1.0;
  if (rowInBlock < ROWS_PER_BLOCK && row < ROW_COUNT) {
    let seedNode = seedNodes[seedNodesOffset + 2u * row + (slot & 1u)];
    if (seedNode < NODE_COUNT) {
      node = lane * NODE_COUNT + seedNode;
      cost = seedCosts[seedCostsOffset + 2u * row + (slot & 1u)];
    }
  }
  expandedSeeds[expandedSeedsOffset + batch * SEED_STRIDE + slot] = node;
  expandedSeedCosts[expandedSeedCostsOffset + batch * SEED_STRIDE + slot] = cost;`
      })
    );
    const blockInfo = u32('block-info', blockCount * ALIGNMENT_ELEMENTS);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-block-info`,
        operation: OPERATION,
        variant: 'block-info',
        bindings: [{name: 'blockInfo', view: blockInfo, type: 'u32', access: 'read_write'}],
        invocationCount: blockCount,
        declarations: `const ROW_COUNT: u32 = ${rowCount}u;
const ROWS_PER_BLOCK: u32 = ${rowsPerBlock}u;
const SLOT_WORDS: u32 = ${ALIGNMENT_ELEMENTS}u;`,
        // Word 0: first row of the block. Word 1: rows in the block.
        body: `let firstRow = index * ROWS_PER_BLOCK;
  blockInfo[blockInfoOffset + index * SLOT_WORDS] = firstRow;
  blockInfo[blockInfoOffset + index * SLOT_WORDS + 1u] = min(ROWS_PER_BLOCK, ROW_COUNT - firstRow);`
      })
    );

    const batchConverged =
      output.status?.converged || output.status?.iterationLimitReached
        ? u32('batch-converged', batchCount * ALIGNMENT_ELEMENTS)
        : undefined;
    for (let block = 0; block < blockCount; block++) {
      for (let batchInBlock = 0; batchInBlock < batchesPerBlock; batchInBlock++) {
        const batch = block * batchesPerBlock + batchInBlock;
        nodes.push(
          ...new GPUNetworkReachability({
            id: `${id}-block-${block}-batch-${batchInBlock}`,
            topology: {offsets, neighbors, weights},
            laneCount,
            startNodes: getSubView(graph, expandedSeeds, batch * seedStride, 2 * laneCount),
            startCosts: getSubView(graph, expandedSeedCosts, batch * seedStride, 2 * laneCount),
            costLimit: props.maximumDistance,
            maximumIterations: props.maximumIterations,
            localIterations: props.localIterations ?? DEFAULT_LOCAL_ITERATIONS,
            output: {
              costs: getSubView(graph, costs, batchInBlock * batchStride, laneCount * nodeCount),
              status: {
                converged: batchConverged
                  ? getSubView(graph, batchConverged, batch * ALIGNMENT_ELEMENTS, 1)
                  : undefined
              }
            }
          }).getCommandNodes(graph)
        );
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-block-${block}-count`,
          operation: OPERATION,
          variant: 'count',
          bindings: [
            {name: 'costs', view: costs, type: 'f32', access: 'read'},
            {name: 'seedNodes', view: seedNodes, type: 'u32', access: 'read'},
            {name: 'seedCosts', view: seedCosts, type: 'f32', access: 'read'},
            {name: 'maximumDistanceIn', view: props.maximumDistance, type: 'f32', access: 'read'},
            {name: 'histogram', view: histogram, type: 'atomic<u32>', access: 'read_write'},
            {
              name: 'blockInfo',
              view: getSubView(graph, blockInfo, block * ALIGNMENT_ELEMENTS, 2),
              type: 'u32',
              access: 'read'
            }
          ],
          invocationCount: rowsPerBlock * eventCount,
          // Needs workgroup barriers, so out-of-range invocations must not return early.
          guardIndex: false,
          declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
const EVENT_COUNT: u32 = ${eventCount}u;
const BAND_COUNT: u32 = ${bandCount}u;
const LANE_COUNT: u32 = ${laneCount}u;
const BATCH_STRIDE: u32 = ${batchStride}u;
const WORKGROUP_SIZE: u32 = 64u;
// Per-workgroup histogram of the pattern that owns the workgroup's first invocation. A workgroup
// spans 64 consecutive (row, column) pairs, so nearly all of them share one pattern and one
// global atomic per pair collapses into one local atomic and at most BAND_COUNT global adds.
var<workgroup> localHistogram: array<atomic<u32>, ${bandCount}>;
fn isFiniteCost(cost: f32) -> bool {
  return bitcast<u32>(cost) < 0x7f800000u;
}
fn getThreshold(maximumDistance: f32, band: u32) -> f32 {
  return maximumDistance * f32(band) / f32(BAND_COUNT - 1u);
}
// The block's first row, read from its info slot (rule 12: the block index is data).
var<private> blockFirstRow: u32;
fn getPattern(index: u32) -> u32 {
  return (blockFirstRow + index / EVENT_COUNT) / EVENT_COUNT;
}
// Start of a block row's costs: batch (localRow / LANE_COUNT) at BATCH_STRIDE, then its lane.
fn getRowCostBase(localRow: u32) -> u32 {
  let batch = localRow / LANE_COUNT;
  return costsOffset + batch * BATCH_STRIDE + (localRow - batch * LANE_COUNT) * NODE_COUNT;
}
// Distance band of the unordered pair of row and column events, or BAND_COUNT when the pair is
// not counted.
fn classifyPair(index: u32) -> u32 {
  let localRow = index / EVENT_COUNT;
  let columnEvent = index - localRow * EVENT_COUNT;
  let row = blockFirstRow + localRow;
  let pattern = row / EVENT_COUNT;
  let eventIndex = row - pattern * EVENT_COUNT;
  let maximumDistance = maximumDistanceIn[maximumDistanceInOffset];
  if (columnEvent <= eventIndex || !(maximumDistance > 0.0)) {
    return BAND_COUNT;
  }
  let rowSlot = 2u * row;
  let columnSlot = 2u * (pattern * EVENT_COUNT + columnEvent);
  let rowFirst = seedNodes[seedNodesOffset + rowSlot];
  let rowSecond = seedNodes[seedNodesOffset + rowSlot + 1u];
  let columnFirst = seedNodes[seedNodesOffset + columnSlot];
  let columnSecond = seedNodes[seedNodesOffset + columnSlot + 1u];
  if (rowFirst >= NODE_COUNT || rowSecond >= NODE_COUNT ||
      columnFirst >= NODE_COUNT || columnSecond >= NODE_COUNT) {
    return BAND_COUNT;
  }
  let rowFirstCost = seedCosts[seedCostsOffset + rowSlot];
  let columnFirstCost = seedCosts[seedCostsOffset + columnSlot];
  let columnSecondCost = seedCosts[seedCostsOffset + columnSlot + 1u];
  var best = 0.0;
  var found = false;
  let viaFirst = costs[getRowCostBase(localRow) + columnFirst];
  if (isFiniteCost(viaFirst)) {
    best = viaFirst + columnFirstCost;
    found = true;
  }
  let viaSecond = costs[getRowCostBase(localRow) + columnSecond];
  if (isFiniteCost(viaSecond)) {
    let candidate = viaSecond + columnSecondCost;
    if (!found || candidate < best) {
      best = candidate;
      found = true;
    }
  }
  var sameEdge = -1.0;
  if (rowFirst != rowSecond) {
    if (rowFirst == columnFirst && rowSecond == columnSecond) {
      sameEdge = abs(rowFirstCost - columnFirstCost);
    } else if (rowFirst == columnSecond && rowSecond == columnFirst) {
      sameEdge = abs(rowFirstCost - columnSecondCost);
    }
  }
  if (sameEdge >= 0.0 && (!found || sameEdge < best)) {
    best = sameEdge;
    found = true;
  }
  if (!found || !(best < maximumDistance)) {
    return BAND_COUNT;
  }
  var band = u32(max(floor(best / (maximumDistance / f32(BAND_COUNT - 1u))), 0.0)) + 1u;
  band = min(band, BAND_COUNT);
  while (band > 0u && best < getThreshold(maximumDistance, band - 1u)) {
    band--;
  }
  while (band < BAND_COUNT && !(best < getThreshold(maximumDistance, band))) {
    band++;
  }
  return band;
}`,
          workgroupSize: 64,
          body: `blockFirstRow = blockInfo[blockInfoOffset];
  let liveCount = blockInfo[blockInfoOffset + 1u] * EVENT_COUNT;
  let firstIndex = index - localInvocationIndex;
  let ownerPattern = getPattern(min(firstIndex, liveCount - 1u));
  for (var bin = localInvocationIndex; bin < BAND_COUNT; bin += WORKGROUP_SIZE) {
    atomicStore(&localHistogram[bin], 0u);
  }
  workgroupBarrier();
  if (index < liveCount) {
    let band = classifyPair(index);
    if (band < BAND_COUNT) {
      let pattern = getPattern(index);
      if (pattern == ownerPattern) {
        atomicAdd(&localHistogram[band], 1u);
      } else {
        atomicAdd(&histogram[histogramOffset + pattern * BAND_COUNT + band], 1u);
      }
    }
  }
  workgroupBarrier();
  for (var bin = localInvocationIndex; bin < BAND_COUNT; bin += WORKGROUP_SIZE) {
    let binCount = atomicLoad(&localHistogram[bin]);
    if (binCount > 0u) {
      atomicAdd(&histogram[histogramOffset + ownerPattern * BAND_COUNT + bin], binCount);
    }
  }`
        })
      );
    }

    const finalizeBindings: WGSLKernelBinding[] = [
      {name: 'histogram', view: histogram, type: 'u32', access: 'read'},
      {name: 'networkLength', view: props.networkLength, type: 'f32', access: 'read'},
      {name: 'validCount', view: validCount, type: 'u32', access: 'read'},
      {name: 'parameters', view: props.parameters, type: 'u32', access: 'read'},
      {name: 'kValues', view: output.kValues, type: 'f32', access: 'read_write'}
    ];
    if (output.pairCounts) {
      finalizeBindings.push({
        name: 'pairCounts',
        view: output.pairCounts,
        type: 'u32',
        access: 'read_write'
      });
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finalize`,
        operation: OPERATION,
        variant: 'finalize',
        bindings: finalizeBindings,
        invocationCount: patternCount * bandCount,
        declarations: `const BAND_COUNT: u32 = ${bandCount}u;
const SIMULATION_COUNT: u32 = ${simulationCount}u;`,
        body: `let pattern = index / BAND_COUNT;
  let band = index - pattern * BAND_COUNT;
  let activeCount = min(parameters[parametersOffset + 2u], SIMULATION_COUNT);
  var cumulative = 0u;
  if (pattern <= activeCount) {
    for (var bin = 0u; bin <= band; bin++) {
      cumulative += histogram[histogramOffset + pattern * BAND_COUNT + bin];
    }
  }
  let count = f32(validCount[validCountOffset]);
  var k = 0.0;
  if (count >= 2.0) {
    k = 2.0 * f32(cumulative) * networkLength[networkLengthOffset] / (count * count);
  }
  kValues[kValuesOffset + index] = k;
  ${output.pairCounts ? 'pairCounts[pairCountsOffset + index] = cumulative;' : ''}`
      })
    );

    if (output.envelope) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-envelope`,
          operation: OPERATION,
          variant: 'envelope',
          bindings: [
            {name: 'kValues', view: output.kValues, type: 'f32', access: 'read'},
            {name: 'parameters', view: props.parameters, type: 'u32', access: 'read'},
            {name: 'envelope', view: output.envelope, type: 'f32', access: 'read_write'}
          ],
          invocationCount: bandCount,
          declarations: `const BAND_COUNT: u32 = ${bandCount}u;
const SIMULATION_COUNT: u32 = ${simulationCount}u;`,
          body: `let activeCount = min(parameters[parametersOffset + 2u], SIMULATION_COUNT);
  var lowest = 0.0;
  var highest = 0.0;
  var sum = 0.0;
  for (var pattern = 1u; pattern <= activeCount; pattern++) {
    let value = kValues[kValuesOffset + pattern * BAND_COUNT + index];
    lowest = select(min(lowest, value), value, pattern == 1u);
    highest = select(max(highest, value), value, pattern == 1u);
    sum += value;
  }
  envelope[envelopeOffset + index] = lowest;
  envelope[envelopeOffset + BAND_COUNT + index] = select(0.0, sum / f32(activeCount), activeCount > 0u);
  envelope[envelopeOffset + 2u * BAND_COUNT + index] = highest;`
        })
      );
    }

    if (batchConverged) {
      const ports = [
        ...(output.status?.converged ? [{name: 'converged', view: output.status.converged}] : []),
        ...(output.status?.iterationLimitReached
          ? [{name: 'iterationLimitReached', view: output.status.iterationLimitReached}]
          : [])
      ];
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-converged-initialize`,
          operation: OPERATION,
          variant: 'converged-initialize',
          invocationCount: 1,
          bindings: ports.map(port => ({...port, type: 'u32', access: 'read_write'})),
          body: `${output.status?.converged ? 'converged[convergedOffset] = 1u;' : ''}
          ${output.status?.iterationLimitReached ? 'iterationLimitReached[iterationLimitReachedOffset] = 0u;' : ''}`
        }),
        createWGSLKernelNode(graph, {
          id: `${id}-converged`,
          operation: OPERATION,
          variant: 'converged',
          invocationCount: batchCount,
          bindings: [
            {name: 'batchConverged', view: batchConverged, type: 'u32', access: 'read'},
            ...ports.map(port => ({
              ...port,
              type: 'atomic<u32>' as const,
              access: 'read_write' as const
            }))
          ],
          body: `let done = batchConverged[batchConvergedOffset + index * ${ALIGNMENT_ELEMENTS}u];
          ${output.status?.converged ? 'atomicMin(&converged[convergedOffset], done);' : ''}
          ${output.status?.iterationLimitReached ? 'atomicMax(&iterationLimitReached[iterationLimitReachedOffset], 1u - done);' : ''}`
        })
      );
    }
    return nodes;
  }
}

/** Returns every read-only view. */
function getInputs(props: GPUNetworkKFunctionProps): (GraphDataView | undefined)[] {
  return [
    props.positions,
    ...getGPUNetworkTopologyViews(props.topology),
    props.maximumSnapDistance,
    props.maximumDistance,
    props.networkLength,
    props.parameters
  ];
}

/** Returns every writable view. */
function getOutputs(props: GPUNetworkKFunctionProps): (GraphDataView | undefined)[] {
  const {output} = props;
  return [
    output.kValues,
    output.pairCounts,
    output.envelope,
    output.snappedEventCount,
    output.status?.overflow,
    output.status?.converged,
    output.status?.iterationLimitReached
  ];
}
