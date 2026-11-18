// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {clamp} from '@math.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  getCoreBindingElementLimit,
  validateCapacity,
  validateFiniteNumber,
  validateRequirement,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateGPUNetworkTopology,
  type GPUNetworkTopology
} from '../../analysis-core/contracts/network-topology';
import {
  GPUNetworkReachability,
  GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS,
  GPU_NETWORK_REACHABILITY_MAXIMUM_LOCAL_ITERATIONS
} from '../network-reachability/index';
import {recommendLaneCount} from '../network-accessibility/network-accessibility-lanes';
import {
  createFrontierState,
  getFrontierPhaseLayout,
  getFrontierPhaseMaximumStamp,
  type FrontierPhase
} from '../network-reachability/network-frontier';
import {
  CLOSENESS_WORKGROUP_SIZE,
  CENTRALITY_ALIGNMENT_ELEMENTS,
  CENTRALITY_BATCH_STATUS_WORDS,
  createCentralityAccumulateNode,
  createCentralityEdgeAccumulateNode,
  createCentralityBatchInfoNode,
  createCentralityBatchStatusNode,
  createCentralityClosenessNode,
  createCentralityConvergedNode,
  createCentralityDeltaInitializeNode,
  createCentralityDeltaRoundNode,
  createCentralityExpandSeedsNode,
  createCentralityPackNode,
  createCentralityResetNode,
  createCentralitySigmaInitializeNode,
  createCentralitySigmaRoundNode,
  createCentralityZeroNode,
  getCentralityAdjacencyLayout
} from './network-centrality-passes';
import {alignElementCount, getSubView} from '../network-view-utils';
import {
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateCentralityLogicalRanges} from './network-centrality-bindings';
import {createFillNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {FRONTIER_DISPATCH_SLOT_WORDS} from '../network-reachability/network-frontier';

/** A closeness lane dispatches 256 invocations; the published dispatch index is uint32. */
const CENTRALITY_MAXIMUM_CLOSENESS_LANE_COUNT = Math.floor(0xffffffff / CLOSENESS_WORKGROUP_SIZE);

const DEFAULT_MAXIMUM_ITERATIONS = 64;
const DEFAULT_LOCAL_ITERATIONS = 16;
/**
 * Hops per workgroup in the sigma and delta phases. A node becomes ready when the last of its
 * tight predecessors decrements its counter, but the other predecessors' sums may have been
 * written by other workgroups of the same dispatch, and WGSL gives no cross-workgroup ordering
 * inside one dispatch. With one hop per round a ready node is visited in the next dispatch, after
 * every predecessor's write.
 */
const DEPENDENCY_LOCAL_ITERATIONS = 1;

/** Views written by {@link GPUNetworkCentrality}. At least one centrality value is required. */
export type GPUNetworkCentralityOutput = {
  /**
   * Optional per-node betweenness: the sum over the pivots of the Brandes dependency of the pivot
   * on the node (the pivot itself excluded, endpoints not counted), times `betweennessScale`. Its
   * length is the node count. Requesting it adds the sigma and delta phases to every batch.
   */
  betweenness?: GraphDataView<'float32'>;
  /** Per-CSR-slot directed edge Brandes dependencies; same scale and pivot conventions as nodes. */
  edgeBetweenness?: GraphDataView<'float32'>;
  /**
   * Optional per-pivot closeness `(r - 1) / S`, where `r` counts the nodes reached from the pivot
   * (the pivot included) and `S` is the sum of their costs; 0 when `S` is 0. Its length is the
   * pivot count; out-of-range pivots get 0.
   */
  closeness?: GraphDataView<'float32'>;
  /**
   * Optional per-pivot harmonic closeness: the sum of `1 / cost` over the reached nodes with a
   * cost above 0. Its length is the pivot count.
   */
  harmonicCloseness?: GraphDataView<'float32'>;
  /** Optional per-pivot reached node count (the pivot included). Its length is the pivot count. */
  reachedCounts?: GraphDataView<'uint32'>;
  /** Optional GPU-written status words. */
  status?: {
    /** One-row flag: 1 when sampling is 'sampled'; independent of convergence. */
    approximation?: GraphDataView<'uint32'>;
    /**
     * Optional one-row flag: 1 when every batch's reachability and, with `betweenness`, both
     * accumulation phases finished within `maximumIterations` and ordered every reached node; 0
     * when a round limit stopped one, or when a cycle of zero-weight tight edges left nodes
     * unordered (see `zeroWeightCycle`). With 0 the betweenness values are incomplete.
     */
    converged?: GraphDataView<'uint32'>;
    /**
     * Optional one-row flag: 1 when any reachability, path-count or dependency phase exhausted
     * its round budget, otherwise 0. May be requested alone. A zero-weight cycle without a
     * round-budget exit clears converged but does not set iterationLimitReached.
     * Incumbent centralities are still published.
     */
    iterationLimitReached?: GraphDataView<'uint32'>;
    /**
     * Optional one-row flag, with a betweenness output: 1 when, for some pivot, shortest paths
     * cross a cycle of edges of weight exactly zero between nodes of equal cost (for example an
     * undirected zero-weight edge, which lists both directions). A positive weight never sets it,
     * however small. The number of shortest paths is then not
     * defined, the nodes on and after the cycle get no path count, the betweenness values are
     * incomplete and `converged` is 0. Raising `maximumIterations` does not help: give such edges
     * a positive weight or merge their end nodes. 0 otherwise, also when a round limit stopped the
     * path-count phase first (`converged` reports that).
     */
    zeroWeightCycle?: GraphDataView<'uint32'>;
  };
};

/**
 * Properties for {@link GPUNetworkCentrality}.
 *
 * Compile-time: node and edge counts, pivot capacity, `laneCount`, `maximumIterations`,
 * `localIterations`, `betweennessScale`, `sampling`, and which outputs exist. Per-frame: the contents of both
 * topologies, the pivot nodes and `costLimit`.
 */
export type GPUNetworkCentralityProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-centrality'`. */
  id?: string;
  /**
   * Directed network with weights; distances are measured outward from the pivots along its edges.
   * An undirected network lists both directions of every edge. Targets `>= nodeCount` are ignored
   * and negative or NaN weights are impassable.
   */
  topology: GPUNetworkTopology;
  /**
   * The transpose of `topology`: the same edges reversed, with bitwise-identical weights (for
   * example from `GPUCOOToCSR` on the reversed edge list). An undirected network that lists both
   * directions of every edge with equal weights may pass the same topology. The sigma and delta
   * phases walk both directions to find each node's tight predecessors and successors, so a
   * reverse topology that is not the exact transpose gives wrong centrality values.
   */
  reverseTopology: GPUNetworkTopology;
  /**
   * Pivot (source) nodes, one centrality row each. Use every node for exact centrality, or a
   * sample for estimates (set `betweennessScale`). Out-of-range nodes are ignored and their
   * per-pivot output rows are 0.
   */
  pivotNodes: GraphDataView<'uint32'>;
  /**
   * Optional one-row per-frame radius: paths longer than it are not counted, and nodes whose cost
   * from a pivot exceeds it are not part of that pivot's shortest-path graph. Inclusive.
   */
  costLimit?: GraphDataView<'float32'>;
  /**
   * Compile-time number of pivots searched together in one lane-expanded batch, 1 to the pivot
   * count. Defaults to `recommendLaneCount()` (about 1M expanded nodes per batch), at most the pivot
   * count, and low enough that the three scratch arrays of `laneCount * nodeCount` words fit one
   * storage binding. Every batch runs at the full lane count (a short last batch masks its unused
   * lanes), so all batches compile to the same pipelines. Closeness outputs are bit-identical for
   * every lane count; betweenness changes only in the last bits, because the lane count sets the
   * grouping of the sums. With any closeness output, laneCount is at most 16,777,215:
   * each lane dispatches 256 invocations within the published uint32 dispatch range.
   */
  laneCount?: number;
  /**
   * Compile-time number of unrolled rounds of the reachability search and of each of the sigma and
   * delta phases of every batch, 1 to 1024. Defaults to 64. The search needs about
   * `ceil(h / (0.7 * localIterations)) + 4` rounds for `h` hops along the longest shortest path;
   * the sigma and delta phases run one hop per round, so each needs `h + 1` rounds for the longest
   * tight chain (`h` is small for radius-bounded centrality). A shortfall clears `status.converged`.
   * Scheduling caps this budget at the node count: nonnegative shortest paths need at most
   * `nodeCount - 1` edges, and their strictly increasing tight-edge graph has at most `nodeCount`
   * levels. The final round drains the frontier without changing the configured budget.
   *
   * Compile cost: every round of every batch is an unrolled command node, so the node count grows
   * with `batchCount * min(maximumIterations, nodeCount)`, and with betweenness each batch emits
   * about three such rounds (search, sigma, delta). The constructor rejects configurations whose
   * `estimatedCommandNodeCount` exceeds 32,768. Lower `maximumIterations` to the longest path you
   * need, raise `laneCount` to cut the batch count, or reduce the pivot count.
   */
  maximumIterations?: number;
  /**
   * Compile-time number of hops one workgroup chains inside a round, 1 to 64. Defaults to 16, which
   * suits lane-expanded batches. Applies to the reachability search only: the sigma and delta
   * phases always run one hop per round, so every hand-off between workgroups crosses a dispatch
   * boundary.
   */
  localIterations?: number;
  /**
   * Compile-time factor applied to the sum of each batch before it is added to `betweenness`.
   * Defaults to 1. For estimates from sampled pivots use `nodeCount / pivotCount`.
   */
  betweennessScale?: number;
  /**
   * Labels deliberate pivot sampling in status.approximation. Defaults to 'exact' (the supplied
   * pivots define the requested subset). For a uniform sample estimating all-node betweenness,
   * select 'sampled' and set betweennessScale to nodeCount / totalSampleCount, including when
   * the sample is split across host-scheduled submissions. No pivots are selected on the CPU.
   */
  sampling?: 'exact' | 'sampled';
  /** The GPU-written views. */
  output: GPUNetworkCentralityOutput;
};

/**
 * Computes radius-bounded betweenness and closeness centrality of a directed weighted CSR network
 * from a set of pivot nodes, exactly or from a sample.
 *
 * Pivots are processed `laneCount` at a time. Each batch runs one lane-expanded
 * `GPUNetworkReachability` (lane `l`, node `u` is expanded node `l * nodeCount + u`; all lanes read
 * the one CSR), whose costs give closeness directly: one workgroup per lane reduces the reached
 * count, the cost sum and the harmonic sum in a fixed order. For betweenness the batch then runs
 * Brandes' (2001) accumulation on the tight edges, those between two different nodes with
 * `cost[u] + weight == cost[w]` in f32 exactly as the relaxation computes it, and either
 * `cost[u] < cost[w]` or a weight of exactly zero:
 *
 * 1. The sigma phase (shortest-path counts) is a push-ready, pull-sum frontier schedule. A node
 *    becomes ready when its last tight predecessor releases it (`atomicSub` on a per-node counter
 *    that starts at its tight in-degree); its visit then sums its predecessors' sigma in reverse-CSR
 *    order and releases its tight successors.
 * 2. The delta phase mirrors it from the nodes without tight out-edges: a node sums
 *    `sigma(v) / sigma(w) * (1 + delta(w))` over its tight successors in forward-CSR order, then
 *    releases its tight predecessors.
 * 3. The batch adds, per node, the lane-ordered sum of the lanes' delta (skipping each lane's own
 *    pivot) times `betweennessScale` to the output. Betweenness is zeroed once, and batches add in
 *    batch order.
 *
 * Every sum has a fixed order, so results are deterministic for a given `laneCount`. Pivots may repeat; each
 * counts once per occurrence.
 *
 * Conventions (networkx parity; the oracle is `test/oracles/network-centrality.py`):
 * - Betweenness equals `networkx.betweenness_centrality(G, weight='weight', normalized=False)` on a
 *   `DiGraph` when all nodes are pivots, and `betweenness_centrality_subset(G, sources=pivots,
 *   targets=list(G), ...)` for a sample (times `betweennessScale`). Endpoints are not counted. No
 *   1/2 factor is applied; halve the result for an undirected graph that lists both directions.
 * - Closeness is outward: the cost from the pivot to the nodes it reaches. networkx's
 *   `closeness_centrality` on a `DiGraph` uses incoming distances, so compare against
 *   `closeness_centrality(G.reverse(), u=pivot, wf_improved=False)` (and `harmonic_centrality(G.reverse(), ...)`),
 *   or pass the reverse topologies (swap `topology` and `reverseTopology`) to get networkx's incoming
 *   convention. The Wasserman-Faust factor `(r - 1) / (nodeCount - 1)` is not applied.
 * - Edges of weight exactly zero are tight when the two end nodes have equal cost, and are counted
 *   like any other edge on a shortest path, as networkx counts them: on the path
 *   `s -1-> a -1-> m -0-> t -1-> u` the pivot `s` gives `a`, `m`, `t` the dependencies 3, 2, 1. A
 *   positive weight below half an ulp of the cost, which float32 absorbs (`cost + weight == cost`:
 *   1 at a cost of 2^24, or 3 mm at 100 km in metres), is never tight: the edge is left out of
 *   the path counts, a node reached only across it is a root with one path, and the run still
 *   converges. Counting zero-weight edges needs the zero-weight tight edges to form no cycle. With a cycle
 *   (an undirected zero-weight edge listed in both directions is one) the number of shortest paths
 *   is not defined (networkx requires positive weights for the same reason): the nodes on and
 *   after the cycle are left out, `status.zeroWeightCycle` is 1 and `status.converged` is 0. Give
 *   such edges a positive weight or merge their end nodes. Self-loops are never on a shortest
 *   path. Closeness is not affected; harmonic closeness skips nodes of cost 0. Parallel edges
 *   count as separate paths.
 * - Path counts are a float32 mantissa with a separate base-2 exponent: exact below 2^24 paths per
 *   pair, rounded to 24 bits beyond, and never out of range (a corner-to-corner count on a unit
 *   lattice of 67 x 67 nodes or more exceeds the float32 range). Dependencies use only ratios of
 *   neighbouring counts.
 *
 * Accuracy: sigma sums are exact below 2^24. Delta rounds once per division (within 2.5 ULP in
 * WGSL), addition and multiplication of a term and once per added term, with only positive terms, so a
 * node's relative error is at most `h (7 + K) 2^-24` for `h` hops along the longest tight chain and
 * a largest tight out-degree `K`; adding `P` pivots adds `(P + 2) 2^-24`.
 *
 * Scratch memory is the frontier state plus `4 * laneCount * nodeCount` words (path-count
 * mantissa, counter, dependency, path-count exponent) and a packed copy of both CSRs, copied once
 * per encoding.
 *
 * Compile cost: the rounds of every batch are unrolled, so the command node count is about
 * `batchCount * phases * min(maximumIterations, nodeCount)` (see
 * {@link GPUNetworkCentrality.estimatedCommandNodeCount}), where `phases` is 1 for the reachability
 * search alone and 3 with betweenness (search, sigma and delta), plus a few nodes per batch. The
 * constructor throws above 32,768 nodes. Lower `maximumIterations`, raise `laneCount` or reduce the
 * pivot count to stay below it. For larger source sets, reuse GPUNetworkCentralityScheduleStep
 * with planNetworkCentralitySchedule: each submitted graph contains only one lane batch.
 */
export class GPUNetworkCentrality implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNetworkCentralityProps;
  /** Number of network nodes. */
  readonly nodeCount: number;
  /** Number of pivot rows. */
  readonly pivotCount: number;
  /** Resolved pivots per batch. */
  readonly laneCount: number;
  /** Number of lane-expanded batches. */
  readonly batchCount: number;
  /** Resolved compile-time round count of the reachability search and of each phase. */
  readonly maximumIterations: number;
  /** Resolved compile-time hops per workgroup per round. */
  readonly localIterations: number;
  /** Resolved compile-time betweenness factor. */
  readonly betweennessScale: number;
  /**
   * Exact number of command nodes `getCommandNodes` returns. With `R = min(maximumIterations,
   * nodeCount)` and `B = batchCount`, it is the fixed nodes (seed expansion, plus one each for batch
   * info with a closeness output, adjacency packing with betweenness, one zeroing per betweenness
   * output, and the final status reduction) plus `B` times the per-batch nodes: `R + 2` for the
   * reachability search (plus a finalize node with a status flag), one closeness reduction
   * with a closeness output, and with betweenness `2R + 3` (reset, two initializers, `R` sigma and
   * `R` delta rounds) plus one accumulate node per betweenness output and one status node with
   * a status flag (`status.converged`, `status.zeroWeightCycle` or `status.iterationLimitReached`). Rejected by the constructor
   * above 32,768. Requesting status.approximation adds one fixed node.
   */
  readonly estimatedCommandNodeCount: number;

  /**
   * @throws If a view has the wrong format or length, no output is requested, an option is out of
   * range, `estimatedCommandNodeCount` exceeds the 32,768 command node cap, or conflicting output
   * or input byte ranges overlap. Fields used in separate dispatches may share aligned prefixes.
   */
  constructor(props: GPUNetworkCentralityProps);
  constructor(
    props: GPUNetworkCentralityProps,
    /** @internal Metadata validation for schedulers that publish each scalar status separately. */
    separateStatusDispatches = false
  ) {
    this.id = props.id ?? 'network-centrality';
    this.props = props;
    const {id} = this;
    const {output} = props;
    const forward = validateGPUNetworkTopology(id, props.topology, {weights: true});
    const reverse = validateGPUNetworkTopology(id, props.reverseTopology, {weights: true});
    validateRequirement(
      id,
      forward.nodeCount === reverse.nodeCount,
      'reverseTopology must have the node count of topology'
    );
    this.nodeCount = forward.nodeCount;
    validateSinglePackedView(props.pivotNodes, ['uint32'], `${id} pivotNodes`);
    this.pivotCount = props.pivotNodes.length;
    validateRequirement(id, this.pivotCount >= 1, 'pivotNodes must contain at least one pivot');
    if (props.costLimit) {
      validateSinglePackedView(props.costLimit, ['float32'], `${id} costLimit`);
      validateRequirement(
        id,
        props.costLimit.length === 1,
        'costLimit must contain exactly one row'
      );
    }
    validateRequirement(
      id,
      Boolean(
        output.betweenness ||
          output.edgeBetweenness ||
          output.closeness ||
          output.harmonicCloseness ||
          output.reachedCounts
      ),
      'output must contain betweenness, edgeBetweenness, closeness, harmonicCloseness or reachedCounts'
    );
    for (const [name, view, length] of [
      ['output.betweenness', output.betweenness, this.nodeCount],
      ['output.edgeBetweenness', output.edgeBetweenness, forward.edgeCount],
      ['output.closeness', output.closeness, this.pivotCount],
      ['output.harmonicCloseness', output.harmonicCloseness, this.pivotCount]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['float32'], `${id} ${name}`);
        validateRequirement(
          id,
          view.length === length,
          `${name} must contain one row per ${name === 'output.betweenness' ? 'node' : name === 'output.edgeBetweenness' ? 'edge slot' : 'pivot'}`
        );
      }
    }
    if (output.reachedCounts) {
      validateSinglePackedView(output.reachedCounts, ['uint32'], `${id} output.reachedCounts`);
      validateRequirement(
        id,
        output.reachedCounts.length === this.pivotCount,
        'output.reachedCounts must contain one row per pivot'
      );
    }
    validateRequirement(
      id,
      props.sampling === undefined || props.sampling === 'exact' || props.sampling === 'sampled',
      "sampling must be 'exact' or 'sampled'"
    );
    const approximation = output.status?.approximation;
    if (approximation) {
      validateSinglePackedView(approximation, ['uint32'], `${id} output.status.approximation`);
      validateCapacity(id, 'output.status.approximation.length', approximation.length, 1, 1);
    }
    const converged = output.status?.converged;
    if (converged) {
      validateSinglePackedView(converged, ['uint32'], `${id} output.status.converged`);
      validateRequirement(
        id,
        converged.length === 1,
        'output.status.converged must contain exactly one row'
      );
    }
    const iterationLimitReached = output.status?.iterationLimitReached;
    if (iterationLimitReached) {
      validateSinglePackedView(
        iterationLimitReached,
        ['uint32'],
        `${id} output.status.iterationLimitReached`
      );
      validateRequirement(
        id,
        iterationLimitReached.length === 1,
        'output.status.iterationLimitReached must contain exactly one row'
      );
    }
    const zeroWeightCycle = output.status?.zeroWeightCycle;
    if (zeroWeightCycle) {
      validateSinglePackedView(zeroWeightCycle, ['uint32'], `${id} output.status.zeroWeightCycle`);
      validateRequirement(
        id,
        zeroWeightCycle.length === 1,
        'output.status.zeroWeightCycle must contain exactly one row'
      );
      validateRequirement(
        id,
        Boolean(output.betweenness || output.edgeBetweenness),
        'output.status.zeroWeightCycle needs output.betweenness or output.edgeBetweenness'
      );
    }
    // Any stopping or cycle status flag gives every batch a status slot, its search a finalize
    // node and the betweenness phases a status node.
    const hasStatus = Boolean(converged || zeroWeightCycle || iterationLimitReached);
    this.maximumIterations = props.maximumIterations ?? DEFAULT_MAXIMUM_ITERATIONS;
    this.localIterations = props.localIterations ?? DEFAULT_LOCAL_ITERATIONS;
    validateCapacity(
      id,
      'maximumIterations',
      this.maximumIterations,
      1,
      GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS
    );
    validateCapacity(
      id,
      'localIterations',
      this.localIterations,
      1,
      GPU_NETWORK_REACHABILITY_MAXIMUM_LOCAL_ITERATIONS
    );
    this.betweennessScale = props.betweennessScale ?? 1;
    validateFiniteNumber(id, 'betweennessScale', Math.fround(this.betweennessScale));
    // Reachability bounds the expanded size; betweenness also keeps three scratch arrays
    // (sigma, counters, delta) of one word per expanded node in one storage binding.
    // The lane count is fixed here, before the device is known, so it sizes the lanes for the
    // default 128 MiB binding (the reachability frontier queues hold 8 bytes per expanded node).
    const maximumExpandedNodeCount =
      output.betweenness || output.edgeBetweenness
        ? Math.min(getCoreBindingElementLimit(8), getCoreBindingElementLimit(12))
        : getCoreBindingElementLimit(8);
    const maximumLaneCount = clamp(
      Math.floor(maximumExpandedNodeCount / this.nodeCount),
      1,
      this.pivotCount
    );
    this.laneCount =
      props.laneCount ??
      Math.min(
        maximumLaneCount,
        recommendLaneCount({
          rowCount: this.pivotCount,
          nodeCount: this.nodeCount,
          edgeCount: forward.edgeCount
        })
      );
    validateCapacity(id, 'laneCount', this.laneCount, 1, this.pivotCount);
    validateMaximumCount(
      id,
      'laneCount * nodeCount',
      this.laneCount * this.nodeCount,
      maximumExpandedNodeCount
    );
    this.batchCount = Math.ceil(this.pivotCount / this.laneCount);
    const hasBetweenness = Boolean(output.betweenness || output.edgeBetweenness);
    const hasCloseness = Boolean(
      output.closeness || output.harmonicCloseness || output.reachedCounts
    );
    if (hasCloseness)
      validateMaximumCount(
        id,
        'laneCount for closeness outputs',
        this.laneCount,
        CENTRALITY_MAXIMUM_CLOSENESS_LANE_COUNT
      );
    const roundCount = Math.min(this.maximumIterations, this.nodeCount);
    const betweennessOutputCount =
      Number(Boolean(output.betweenness)) + Number(Boolean(output.edgeBetweenness));
    const nodeCountPerBatch =
      roundCount +
      2 +
      Number(hasStatus) +
      Number(hasCloseness) +
      (hasBetweenness ? 2 * roundCount + 3 + betweennessOutputCount + Number(hasStatus) : 0);
    this.estimatedCommandNodeCount =
      1 +
      Number(hasCloseness) +
      (hasBetweenness ? 1 + betweennessOutputCount : 0) +
      Number(hasStatus) +
      Number(Boolean(approximation)) +
      this.batchCount * nodeCountPerBatch;
    // The rounds of every batch are unrolled into command nodes, so this count is the compile cost.
    // Past the cap, lower maximumIterations or the pivot count, or raise laneCount.
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateCentralityLogicalRanges(
      id,
      {
        betweenness: output.betweenness,
        edgeBetweenness: output.edgeBetweenness,
        closeness: output.closeness,
        harmonicCloseness: output.harmonicCloseness,
        reachedCounts: output.reachedCounts,
        ...output.status
      },
      {
        offsets: props.topology.offsets,
        neighbors: props.topology.neighbors,
        weights: props.topology.weights,
        reverseOffsets: props.reverseTopology.offsets,
        reverseNeighbors: props.reverseTopology.neighbors,
        reverseWeights: props.reverseTopology.weights,
        pivotNodes: props.pivotNodes,
        costLimit: props.costLimit
      }
    );
    validateDisjointWritableViews(
      id,
      {
        closeness: output.closeness,
        harmonicCloseness: output.harmonicCloseness,
        reachedCounts: output.reachedCounts
      },
      {strictAlignedBindings: true}
    );
    if (!separateStatusDispatches) validateCentralityStatusBindings(id, output);
  }

  /**
   * Returns, in order: the adjacency copy, the seed expansion, the betweenness zeroing and batch
   * info nodes, then per batch `b` the nodes of one `GPUNetworkReachability` (IDs
   * `${id}-batch-${b}-reachability-*`), the closeness reduction (`-closeness`), and with betweenness
   * the phase reset, `-sigma-initialize`, `-sigma-${round}`, `-delta-initialize`, `-delta-${round}`,
   * `-accumulate` and `-status` nodes; finally an optional `-converged` reduction.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, nodeCount, pivotCount, laneCount, batchCount} = this;
    const {output} = props;
    validateCentralityStatusBindings(id, output);
    this.validateGraphBindings(graph);
    const hasBetweenness = Boolean(output.betweenness || output.edgeBetweenness);
    const hasCloseness = Boolean(
      output.closeness || output.harmonicCloseness || output.reachedCounts
    );
    const expandedNodeCount = laneCount * nodeCount;
    // Every improvement is visited locally or in the next global frontier. A simple shortest
    // path has at most nodeCount - 1 edges (nonnegative f32 addition is monotone); one final
    // round drains its last frontier. Sigma/delta traverse a strict-cost DAG with at most
    // nodeCount levels. Additional rounds can only be empty, even across independent lanes.
    const maximumRounds = Math.min(this.maximumIterations, nodeCount);
    const nodes: GPUCommandNode<Parameters>[] = [];

    // Per-batch sub-views start at multiples of the 256-byte alignment (rule 12): the seeds of
    // batch b at b * seedStride, its info and converged slots at b * 64.
    const seedStride = alignElementCount(laneCount);
    const expandedSeeds = createTransientView(
      graph,
      `${id}-expanded-seeds`,
      'uint32',
      batchCount * seedStride
    );
    nodes.push(
      createCentralityExpandSeedsNode<Parameters>(graph, {
        id: `${id}-expand-seeds`,
        nodeCount,
        laneCount,
        batchCount,
        seedStride,
        pivotNodes: props.pivotNodes,
        expandedSeeds
      })
    );
    const costs = createTransientView(graph, `${id}-costs`, 'float32', expandedNodeCount);
    // One aligned slot per batch: convergence, zero-weight cycle, and round-limit exit.
    const batchConverged =
      output.status?.converged ||
      output.status?.zeroWeightCycle ||
      output.status?.iterationLimitReached
        ? createTransientView(
            graph,
            `${id}-batch-converged`,
            'uint32',
            batchCount * CENTRALITY_ALIGNMENT_ELEMENTS
          )
        : undefined;
    const batchInfo = hasCloseness
      ? createTransientView(
          graph,
          `${id}-batch-info`,
          'uint32',
          batchCount * CENTRALITY_ALIGNMENT_ELEMENTS
        )
      : undefined;
    if (batchInfo) {
      nodes.push(
        createCentralityBatchInfoNode<Parameters>(graph, {
          id: `${id}-batch-info`,
          laneCount,
          batchCount,
          batchInfo
        })
      );
    }

    // Betweenness-only state, shared by every batch (batches run in order and reset it).
    let betweennessState:
      | {
          adjacency: GraphDataView<'uint32'>;
          scratch: GraphDataView<'uint32'>;
          exponents: GraphDataView<'uint32'>;
          layout: ReturnType<typeof getCentralityAdjacencyLayout>;
          state: ReturnType<typeof createFrontierState>;
          sigmaPhase: FrontierPhase;
          deltaPhase: FrontierPhase;
        }
      | undefined;
    if (hasBetweenness) {
      const layout = getCentralityAdjacencyLayout(
        nodeCount,
        props.topology.neighbors.length,
        props.reverseTopology.neighbors.length
      );
      const adjacency = createTransientView(graph, `${id}-adjacency`, 'uint32', layout.wordCount);
      const scratch = createTransientView(graph, `${id}-scratch`, 'uint32', 3 * expandedNodeCount);
      // Base-2 exponents of the path counts, in their own binding so the scratch limit is unchanged.
      const exponents = createTransientView(
        graph,
        `${id}-sigma-exponents`,
        'uint32',
        expandedNodeCount
      );
      const phaseLayout = getFrontierPhaseLayout(maximumRounds);
      const sigmaPhase: FrontierPhase = {
        maxRounds: maximumRounds,
        stampBase: 0,
        controlWordOffset: 0,
        dispatchSlotOffset: 0
      };
      const deltaPhase: FrontierPhase = {
        maxRounds: maximumRounds,
        stampBase: getFrontierPhaseMaximumStamp(sigmaPhase),
        controlWordOffset: phaseLayout.controlWordCount,
        dispatchSlotOffset: phaseLayout.dispatchSlotCount
      };
      const state = createFrontierState(graph, `${id}-frontier`, {
        nodeCount: expandedNodeCount,
        controlWordCount: 2 * phaseLayout.controlWordCount,
        dispatchSlotCount: 2 * phaseLayout.dispatchSlotCount
      });
      betweennessState = {adjacency, scratch, exponents, layout, state, sigmaPhase, deltaPhase};
      nodes.push(
        createCentralityPackNode<Parameters>(graph, {
          id: `${id}-pack-adjacency`,
          layout,
          forward: props.topology,
          reverse: props.reverseTopology,
          adjacency
        })
      );
      for (const [name, view] of [
        ['betweenness', output.betweenness],
        ['edge-betweenness', output.edgeBetweenness]
      ] as const) {
        if (view)
          nodes.push(
            createCentralityZeroNode(graph, {id: `${id}-zero-${name}`, betweenness: view})
          );
      }
    }

    for (let batch = 0; batch < batchCount; batch++) {
      const batchId = `${id}-batch-${batch}`;
      const seeds = getSubView(graph, expandedSeeds, batch * seedStride, laneCount);
      const convergedSlot = batchConverged
        ? getSubView(graph, batchConverged, batch * CENTRALITY_ALIGNMENT_ELEMENTS, 1)
        : undefined;
      nodes.push(
        ...new GPUNetworkReachability({
          id: `${batchId}-reachability`,
          // Lanes share the one CSR; reachability maps expanded node `l * nodeCount + u` onto it.
          topology: {
            offsets: props.topology.offsets,
            neighbors: props.topology.neighbors,
            weights: props.topology.weights
          },
          laneCount,
          startNodes: seeds,
          costLimit: props.costLimit,
          maximumIterations: maximumRounds,
          localIterations: this.localIterations,
          output: {costs, status: {converged: convergedSlot}}
        }).getCommandNodes(graph)
      );
      if (batchInfo) {
        nodes.push(
          createCentralityClosenessNode<Parameters>(graph, {
            id: `${batchId}-closeness`,
            nodeCount,
            laneCount,
            rowCount: pivotCount,
            costs,
            batchInfo: getSubView(graph, batchInfo, batch * CENTRALITY_ALIGNMENT_ELEMENTS, 1),
            closeness: output.closeness,
            harmonicCloseness: output.harmonicCloseness,
            reachedCounts: output.reachedCounts
          })
        );
      }
      if (betweennessState) {
        const {adjacency, scratch, exponents, layout, state, sigmaPhase, deltaPhase} =
          betweennessState;
        const phaseProps = {layout, expandedNodeCount, state, adjacency, costs, scratch, exponents};
        nodes.push(
          createCentralityResetNode<Parameters>(graph, {
            id: `${batchId}-reset`,
            state,
            sigmaPhase,
            deltaPhase
          }),
          createCentralitySigmaInitializeNode<Parameters>(graph, {
            ...phaseProps,
            id: `${batchId}-sigma-initialize`,
            phase: sigmaPhase
          })
        );
        for (let round = 0; round < maximumRounds; round++) {
          nodes.push(
            createCentralitySigmaRoundNode<Parameters>(graph, {
              ...phaseProps,
              id: `${batchId}-sigma-${round}`,
              phase: sigmaPhase,
              round,
              // One hop per round: every sigma and delta hand-off crosses a dispatch boundary.
              localIterations: DEPENDENCY_LOCAL_ITERATIONS
            })
          );
        }
        nodes.push(
          createCentralityDeltaInitializeNode<Parameters>(graph, {
            ...phaseProps,
            id: `${batchId}-delta-initialize`,
            phase: deltaPhase
          })
        );
        for (let round = 0; round < maximumRounds; round++) {
          nodes.push(
            createCentralityDeltaRoundNode<Parameters>(graph, {
              ...phaseProps,
              id: `${batchId}-delta-${round}`,
              phase: deltaPhase,
              round,
              // One hop per round: every sigma and delta hand-off crosses a dispatch boundary.
              localIterations: DEPENDENCY_LOCAL_ITERATIONS
            })
          );
        }
        if (output.betweenness)
          nodes.push(
            createCentralityAccumulateNode<Parameters>(graph, {
              id: `${batchId}-accumulate`,
              nodeCount,
              laneCount,
              expandedNodeCount,
              betweennessScale: this.betweennessScale,
              scratch,
              seeds,
              betweenness: output.betweenness
            })
          );
        if (output.edgeBetweenness)
          nodes.push(
            createCentralityEdgeAccumulateNode(graph, {
              id: `${batchId}-accumulate-edges`,
              layout,
              expandedNodeCount,
              laneCount,
              betweennessScale: this.betweennessScale,
              adjacency,
              scratch,
              exponents,
              costs,
              edgeBetweenness: output.edgeBetweenness
            })
          );
        if (batchConverged) {
          nodes.push(
            createCentralityBatchStatusNode<Parameters>(graph, {
              id: `${batchId}-status`,
              state,
              sigmaPhase,
              deltaPhase,
              batchStatus: getSubView(
                graph,
                batchConverged,
                batch * CENTRALITY_ALIGNMENT_ELEMENTS,
                CENTRALITY_BATCH_STATUS_WORDS
              )
            })
          );
        }
      }
    }
    if (batchConverged) {
      nodes.push(
        createCentralityConvergedNode<Parameters>(graph, {
          id: `${id}-converged`,
          batchCount,
          batchStatus: batchConverged,
          converged: output.status?.converged,
          iterationLimitReached: output.status?.iterationLimitReached,
          hasBetweenness,
          zeroWeightCycle: output.status?.zeroWeightCycle
        })
      );
    }
    if (output.status?.approximation) {
      nodes.push(
        createFillNode(graph, {
          id: `${id}-approximation`,
          operation: 'GPUNetworkCentrality',
          view: output.status.approximation,
          type: 'u32',
          value: props.sampling === 'sampled' ? '1u' : '0u'
        })
      );
    }
    return nodes;
  }

  /** Allocation-free device binding preflight, also used by host-scheduled centrality. @internal */
  private validateGraphBindings<Parameters>(graph: GPUCommandGraph<Parameters>): void {
    const {id, props, nodeCount, laneCount, batchCount} = this;
    const {output} = props;
    const views = {
      'topology.offsets': props.topology.offsets,
      'topology.neighbors': props.topology.neighbors,
      'topology.weights': props.topology.weights,
      'reverseTopology.offsets': props.reverseTopology.offsets,
      'reverseTopology.neighbors': props.reverseTopology.neighbors,
      'reverseTopology.weights': props.reverseTopology.weights,
      pivotNodes: props.pivotNodes,
      costLimit: props.costLimit,
      'output.betweenness': output.betweenness,
      'output.edgeBetweenness': output.edgeBetweenness,
      'output.closeness': output.closeness,
      'output.harmonicCloseness': output.harmonicCloseness,
      'output.reachedCounts': output.reachedCounts,
      ...Object.fromEntries(
        Object.entries(output.status ?? {}).map(([name, view]) => [`output.status.${name}`, view])
      )
    };
    // Positions retain ownership/shape validation but are never bound by centrality kernels.
    validateGraphViewsBelongToGraph(id, graph, [
      ...Object.values(views),
      props.topology.nodePositions,
      props.reverseTopology.nodePositions
    ]);
    validateGraphChunkBindingRanges(graph, id, views);
    const expandedNodeCount = laneCount * nodeCount;
    const roundCount = Math.min(this.maximumIterations, nodeCount);
    const phaseLayout = getFrontierPhaseLayout(roundCount);
    const hasBetweenness = Boolean(output.betweenness || output.edgeBetweenness);
    const hasStatus = Boolean(
      output.status?.converged ||
        output.status?.zeroWeightCycle ||
        output.status?.iterationLimitReached
    );
    const hasCloseness = Boolean(
      output.closeness || output.harmonicCloseness || output.reachedCounts
    );
    const bindings: [string, number][] = [
      ['expanded seeds', batchCount * alignElementCount(laneCount)],
      ['costs', expandedNodeCount],
      ['reachability queues', 2 * expandedNodeCount],
      ['reachability queued rounds', expandedNodeCount],
      ['reachability control', phaseLayout.controlWordCount],
      ['reachability dispatch slots', phaseLayout.dispatchSlotCount * FRONTIER_DISPATCH_SLOT_WORDS]
    ];
    if (hasCloseness) bindings.push(['batch info', batchCount * CENTRALITY_ALIGNMENT_ELEMENTS]);
    if (hasStatus) bindings.push(['batch status', batchCount * CENTRALITY_ALIGNMENT_ELEMENTS]);
    if (hasBetweenness)
      bindings.push(
        [
          'adjacency',
          getCentralityAdjacencyLayout(
            nodeCount,
            props.topology.neighbors.length,
            props.reverseTopology.neighbors.length
          ).wordCount
        ],
        ['scratch', 3 * expandedNodeCount],
        ['sigma exponents', expandedNodeCount],
        ['dependency queues', 2 * expandedNodeCount],
        ['dependency queued rounds', expandedNodeCount],
        ['dependency control', 2 * phaseLayout.controlWordCount],
        [
          'dependency dispatch slots',
          2 * phaseLayout.dispatchSlotCount * FRONTIER_DISPATCH_SLOT_WORDS
        ]
      );
    for (const [name, wordCount] of bindings) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        4 * wordCount,
        'reduce laneCount, pivot count, edge count or maximumIterations'
      );
    }
  }
}

/** Runs the private allocation-free preflight for scheduler composition. @internal */
export function validateNetworkCentralityBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  contributor: GPUNetworkCentrality
): void {
  Reflect.apply(Reflect.get(contributor, 'validateGraphBindings'), contributor, [graph]);
}

/** Validates a scheduler whose status scalars are published in separate dispatches. @internal */
export function createNetworkCentralityForComposition(
  props: GPUNetworkCentralityProps
): GPUNetworkCentrality {
  return Reflect.construct(GPUNetworkCentrality, [props, true]) as GPUNetworkCentrality;
}

/** Scalar statuses of the direct contributor are bound together; schedulers merge them separately. */
function validateCentralityStatusBindings(id: string, output: GPUNetworkCentralityOutput): void {
  validateDisjointWritableViews(
    id,
    {
      converged: output.status?.converged,
      iterationLimitReached: output.status?.iterationLimitReached,
      zeroWeightCycle: output.status?.zeroWeightCycle
    },
    {strictAlignedBindings: true}
  );
}
