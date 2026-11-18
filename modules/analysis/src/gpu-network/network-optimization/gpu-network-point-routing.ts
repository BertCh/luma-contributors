// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateNetworkOptimizationBindings} from './network-optimization-bindings';

import {
  getViewBindingRange,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength,
  validateSinglePackedView,
  validateRequirement,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';
import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  getFrontierPhaseLayout,
  FRONTIER_DISPATCH_SLOT_WORDS
} from '../network-reachability/network-frontier';
import {getNetworkPointRoutingReferenceNodes} from './network-point-routing-reference';
import {getNetworkPointRoutingScalableNodes} from './network-point-routing-scalable';

/** Reference queries hold one private distance per edge. */
const MAXIMUM_REFERENCE_EDGE_COUNT = 128;
/** Reference queries scan every turn in each serial relaxation round. */
const MAXIMUM_REFERENCE_TURN_COUNT = 4096;
/** Bounds the reference kernel's serial relaxation loop. */
const MAXIMUM_REFERENCE_RELAXATIONS = 65536;
/** CSR offsets need edgeCount + 1 rows addressable by uint32. */
const MAXIMUM_EDGE_COUNT = 0xfffffffe;
/** CSR turn slots and their exclusive sum must fit uint32. */
const MAXIMUM_TURN_COUNT = 0xffffffff;
/** The composed reachability contributor supports at most 1024 frontier rounds per phase. */
const MAXIMUM_RELAXATIONS = 1024;
/** 64 rounds of up to 64 local hops cover ordinary road-network diameters; exhaustion is reported. */
const DEFAULT_RELAXATIONS = 64;
/** A uint32-length scan has at most four 256-row levels and seven scan/offset nodes. */
const SCAN_COMMAND_NODE_ALLOWANCE = 7;

export type GPUNetworkPointRoutingProps = {
  id?: string;
  /** Allowed directed [incomingEdge, outgoingEdge] turns, normally from an explicit line graph. */
  turns: GraphDataView<'uint32x2'>;
  /** Optional nonnegative finite penalty per allowed turn; omitted means zero. Penalties are
   * charged in full even when the destination edge is only partially traversed. */
  turnPenalties?: GraphDataView<'float32'>;
  /** Base-edge travel costs. Negative or non-finite values close that edge; queries touching a
   * closed endpoint get status 1, and turns through closed edges are skipped. */
  costs: GraphDataView<'float32'>;
  /** [sourceEdge, targetEdge], preserving parallel-edge identity. Snap coordinates with GPUNetworkSnapping first. */
  queries: GraphDataView<'uint32x2'>;
  /** [sourceFraction, targetFraction], measured in the selected edge direction. */
  fractions: GraphDataView<'float32x2'>;
  /** CSR global-memory search by default; reference retains the original bounded turn-order kernel. */
  mode?: 'scalable' | 'reference';
  /**
   * Compile-time search budget. Scalable: 0..1024 frontier rounds of up to 64 local hops,
   * for both distances and zero-cost predecessor ties (default 64). Reference: 0..65536
   * serial turn scans (default edge count). A zero budget publishes direct same-edge costs only.
   */
  maximumRelaxations?: number;
  output: {
    costs: GraphDataView<'float32'>;
    /** Per query: 0 complete (including unreachable), 1 invalid data, 2 iteration exhaustion. */
    status: GraphDataView<'uint32'>;
    /**
     * Row-major query by edge predecessors for the full explored forest, including edges outside
     * the selected terminal path. A returning same-edge route records its final incoming edge at source.
     */
    predecessorEdges: GraphDataView<'uint32'>;
  };
};

/**
 * Batched routing between snapped directed edge positions with explicit incoming turn state.
 * Source pays its remaining fraction and destination its leading fraction; forward same-edge travel
 * competes with legal cycles. Reverse travel needs a separately identified reverse edge.
 *
 * The default builds a CSR line graph from turns with a histogram, GPUScan and parallel scatter,
 * then composes GPUNetworkReachability per query. Scratch is O(edges + turns), reused between
 * queries; the caller-owned predecessor matrix remains O(queries * edges). Input views are packed.
 * Partial destinations select cost-attaining parents with source-rooted chains that avoid the
 * full destination, including when a frontier or tie round limit is reached. Finite partial costs
 * always have reconstructable predecessors; unrooted labels are excluded from terminal candidates.
 * Equal-cost predecessor ties follow reachability's smallest-parent, acyclic plateau rule;
 * reference mode preserves turn-order ties. For finite converged costs, comparison tolerance is
 * max(1e-6, 1e-5 * abs(referenceCost)); all routing arithmetic is float32.
 * A same-edge return intentionally closes the source's predecessor chain; stop at the source when
 * reconstructing that route. All other predecessor chains terminate at the source.
 *
 * Compile-time: shapes, mode and maximumRelaxations. Per-frame: every input buffer's contents.
 * Reference mode is capped at 128 edges and 4096 turns. The scalable route is bounded by device
 * storage bindings (two frontier words per edge, plus the predecessor output) and the command budget.
 */
export class GPUNetworkPointRouting implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUNetworkPointRoutingProps;
  readonly mode: 'scalable' | 'reference';
  readonly maximumRelaxations: number;
  /** Pointer doubling certifies every source-rooted terminal path in logarithmically many passes. */
  readonly pathRoundCount: number;
  /** Upper bound including the published GPUScan's hierarchy. */
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPUNetworkPointRoutingProps) {
    this.id = props.id ?? 'network-point-routing';
    this.props = props;
    this.mode = props.mode ?? 'scalable';
    const {id, mode} = this;
    validateRequirement(
      id,
      mode === 'scalable' || mode === 'reference',
      'mode must be scalable or reference'
    );
    this.pathRoundCount = Math.ceil(Math.log2(Math.max(props.costs.length, 2)));
    const reference = mode === 'reference';
    this.maximumRelaxations =
      props.maximumRelaxations ?? (reference ? props.costs.length : DEFAULT_RELAXATIONS);
    validateCapacity(
      id,
      'costs rows',
      props.costs.length,
      1,
      reference ? MAXIMUM_REFERENCE_EDGE_COUNT : MAXIMUM_EDGE_COUNT
    );
    validateCapacity(
      id,
      'turns rows',
      props.turns.length,
      0,
      reference ? MAXIMUM_REFERENCE_TURN_COUNT : MAXIMUM_TURN_COUNT
    );
    validateCapacity(
      id,
      'maximumRelaxations',
      this.maximumRelaxations,
      0,
      reference ? MAXIMUM_REFERENCE_RELAXATIONS : MAXIMUM_RELAXATIONS
    );
    validateCapacity(id, 'queries rows', props.queries.length, 0);
    validateCapacity(id, 'queries rows * costs rows', props.queries.length * props.costs.length, 0);
    validateSinglePackedView(props.turns, ['uint32x2'], `${id} turns`);
    if (props.turnPenalties) {
      validateSinglePackedView(props.turnPenalties, ['float32'], `${id} turnPenalties`);
      validateRequirement(
        id,
        props.turnPenalties.length === props.turns.length,
        'turnPenalties must hold one row per turn'
      );
    }
    validateSinglePackedView(props.queries, ['uint32x2'], `${id} queries`);
    validateSinglePackedView(props.fractions, ['float32x2'], `${id} fractions`);
    for (const [name, view] of [
      ['costs', props.costs],
      ['output.costs', props.output.costs]
    ] as const)
      validateSinglePackedView(view, ['float32'], `${id} ${name}`);
    for (const [name, view] of [
      ['output.status', props.output.status],
      ['output.predecessorEdges', props.output.predecessorEdges]
    ] as const)
      validateSinglePackedView(view, ['uint32'], `${id} ${name}`);
    validateRequirement(
      id,
      props.fractions.length === props.queries.length,
      'fractions must hold one row per query'
    );
    validateRequirement(
      id,
      props.output.costs.length === props.queries.length,
      'output.costs must hold one row per query'
    );
    validateRequirement(
      id,
      props.output.status.length === props.queries.length,
      'output.status must hold one row per query'
    );
    validateRequirement(
      id,
      props.output.predecessorEdges.length === props.queries.length * props.costs.length,
      'output.predecessorEdges must hold costs rows per query'
    );
    validateGraphOutputsDisjointFromInputs(id, Object.values(props.output), [
      props.turns,
      props.turnPenalties,
      props.costs,
      props.queries,
      props.fractions
    ]);
    // Packed outputs may share a backing allocation when their aligned binding ranges are
    // disjoint. The shared validator also resolves distinct imported handles to core buffers.
    validateDisjointWritableViews(id, props.output);
    const getAllocation = (view: GraphDataView) =>
      view.buffer.defaultBuffer ? getCoreDefaultBuffer(view.buffer.defaultBuffer) : view.buffer;
    const outputAllocations = Object.values(props.output).map(getAllocation);
    const inputAllocations = new Set(
      [props.turns, props.turnPenalties, props.costs, props.queries, props.fractions]
        .filter(view => view !== undefined)
        .map(getAllocation)
    );
    validateRequirement(
      id,
      outputAllocations.every(allocation => !inputAllocations.has(allocation)),
      'output views must use separate physical buffers from inputs'
    );
    // CSR: reset, validate/count, scan, reset cursors, scatter. Each query adds seed setup,
    // reachability (2 * max(rounds, 1) + 8 nodes), partial-target reduction, parent reduction,
    // source-rooted path certification, predecessor copy and publication. Zero queries do not declare work.
    this.estimatedCommandNodeCount =
      props.queries.length === 0
        ? 0
        : reference
          ? 1
          : 4 +
            SCAN_COMMAND_NODE_ALLOWANCE +
            props.queries.length *
              (2 * Math.max(this.maximumRelaxations, 1) + 14 + this.pathRoundCount);
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.turns,
      props.turnPenalties,
      props.costs,
      props.queries,
      props.fractions,
      ...Object.values(props.output)
    ]);
    // Check every row-proportional binding before any transient resource or command is created.
    for (const [name, view] of Object.entries({
      turns: props.turns,
      ...(props.turnPenalties ? {turnPenalties: props.turnPenalties} : {}),
      edgeCosts: props.costs,
      queries: props.queries,
      fractions: props.fractions,
      ...props.output
    })) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        getViewBindingRange(view).size,
        'reduce costs rows, turns rows or queries rows'
      );
    }
    validateNetworkOptimizationBindings(graph, id, {
      turns: props.turns,
      turnPenalties: props.turnPenalties,
      edgeCosts: props.costs,
      queries: props.queries,
      fractions: props.fractions,
      ...props.output
    });
    if (this.mode === 'scalable') {
      const phaseLayout = getFrontierPhaseLayout(Math.max(this.maximumRelaxations, 1));
      // Scratch starts at byte offset zero. Both frontier phases share control/dispatch buffers.
      // All other per-edge scalar arrays fit the checked queues/path range; every unsegmented
      // scan hierarchy range is no larger than the checked counts/offsets range. Dispatch slot
      // subviews start on 256-byte boundaries; the active-round subview fits query state.
      for (const [name, byteLength] of [
        ['frontier queues', props.costs.length * 8],
        ['path certification', props.costs.length * 8],
        ['CSR offsets', (props.costs.length + 1) * 4],
        ['CSR turn slots', Math.max(props.turns.length, 1) * 4],
        ['tie bits', Math.max(Math.ceil(Math.max(props.turns.length, 1) / 32), 1) * 4],
        ['query state', 5 * 4],
        ['frontier control', 2 * phaseLayout.controlWordCount * 4],
        ['frontier dispatch', 2 * phaseLayout.dispatchSlotCount * FRONTIER_DISPATCH_SLOT_WORDS * 4]
      ] as const) {
        validateGPUStorageBindingByteLength(
          graph.device,
          id,
          name,
          byteLength,
          'reduce costs rows or turns rows'
        );
      }
    }
    if (props.queries.length === 0) return [];
    return this.mode === 'reference'
      ? getNetworkPointRoutingReferenceNodes(graph, id, props)
      : getNetworkPointRoutingScalableNodes(
          graph,
          id,
          props,
          this.maximumRelaxations,
          this.pathRoundCount
        );
  }
}
