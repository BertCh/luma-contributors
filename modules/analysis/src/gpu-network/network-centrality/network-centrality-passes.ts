// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  createFrontierRoundNode,
  getFrontierBindings,
  getFrontierPushSource,
  getFrontierResetInvocationCount,
  getFrontierResetSource,
  FRONTIER_CONTROL_PARAMETER_0,
  FRONTIER_CONTROL_TRUNCATED,
  type FrontierPhase,
  type FrontierState
} from '../network-reachability/network-frontier';

const OPERATION = 'GPUNetworkCentrality';

/**
 * Elements per 256-byte storage binding alignment. Per-batch sub-views start at multiples of it,
 * so the element offset baked into each kernel is the same for every batch and all batches share
 * pipelines (library rule 12).
 *
 * @internal
 */
export const CENTRALITY_ALIGNMENT_ELEMENTS = 64;

/** Workgroup size of the per-lane closeness reduction. @internal */
export const CLOSENESS_WORKGROUP_SIZE = 256;

/**
 * Word offsets of the six packed arrays of the adjacency buffer: the forward and the reverse CSR
 * (offsets, neighbors, weight bits) copied once per encoding so that the round kernels read both
 * directions through one storage binding. @internal
 */
export type CentralityAdjacencyLayout = {
  nodeCount: number;
  forwardEdgeCount: number;
  reverseEdgeCount: number;
  forwardOffsets: number;
  forwardNeighbors: number;
  forwardWeights: number;
  reverseOffsets: number;
  reverseNeighbors: number;
  reverseWeights: number;
  /** Total words. */
  wordCount: number;
};

/** Returns the {@link CentralityAdjacencyLayout} of a network. @internal */
export function getCentralityAdjacencyLayout(
  nodeCount: number,
  forwardEdgeCount: number,
  reverseEdgeCount: number
): CentralityAdjacencyLayout {
  const forwardOffsets = 0;
  const forwardNeighbors = forwardOffsets + nodeCount + 1;
  const forwardWeights = forwardNeighbors + forwardEdgeCount;
  const reverseOffsets = forwardWeights + forwardEdgeCount;
  const reverseNeighbors = reverseOffsets + nodeCount + 1;
  const reverseWeights = reverseNeighbors + reverseEdgeCount;
  return {
    nodeCount,
    forwardEdgeCount,
    reverseEdgeCount,
    forwardOffsets,
    forwardNeighbors,
    forwardWeights,
    reverseOffsets,
    reverseNeighbors,
    reverseWeights,
    wordCount: reverseWeights + reverseEdgeCount
  };
}

/** WGSL constants and the tight-edge test shared by the sigma and delta kernels. @internal */
function getCentralitySharedSource(layout: CentralityAdjacencyLayout, expandedNodeCount: number) {
  return /* wgsl */ `
const INFINITY_BITS: u32 = 0x7f800000u;
const NONE: u32 = 0xffffffffu;
const NODE_COUNT: u32 = ${layout.nodeCount}u;
const FORWARD_EDGE_COUNT: u32 = ${layout.forwardEdgeCount}u;
const REVERSE_EDGE_COUNT: u32 = ${layout.reverseEdgeCount}u;
const FORWARD_OFFSETS: u32 = ${layout.forwardOffsets}u;
const FORWARD_NEIGHBORS: u32 = ${layout.forwardNeighbors}u;
const FORWARD_WEIGHTS: u32 = ${layout.forwardWeights}u;
const REVERSE_OFFSETS: u32 = ${layout.reverseOffsets}u;
const REVERSE_NEIGHBORS: u32 = ${layout.reverseNeighbors}u;
const REVERSE_WEIGHTS: u32 = ${layout.reverseWeights}u;
// Scratch words: sigma bits, remaining-edge counter, delta bits, one word per expanded node each.
const SIGMA_BASE: u32 = 0u;
const REMAINING_BASE: u32 = ${expandedNodeCount}u;
const DELTA_BASE: u32 = ${2 * expandedNodeCount}u;

// A tight edge lies on a shortest path: the f32 sum the relaxation computes equals the target
// cost, and either the cost rises or the weight is exactly zero (review network-paths-02). A
// self-loop never is tight. A positive weight that float32 absorbs (cost + weight == cost, for
// example 1 at a cost of 2^24) is not tight: it would be tight in both directions of an
// undirected road and form a cycle that no positive weight could remove (fix review
// fr-fields-02). So only a cycle of exactly-zero weights can make the tight graph cyclic; the
// delta initialize kernel reports it.
fn isTightEdge(sourceNode: u32, targetNode: u32, sourceCost: f32, weight: f32, targetCost: f32) -> bool {
  return sourceNode != targetNode && weight >= 0.0 && bitcast<u32>(targetCost) < INFINITY_BITS &&
    sourceCost + weight == targetCost && (sourceCost < targetCost || weight == 0.0);
}

// Path counts are held as a float32 mantissa in [1, 2) (scratch word SIGMA_BASE + node, 0 when
// the node has no count) and a base-2 exponent (exponents view), so a count is not bounded by the
// float32 range (review network-paths-03). 2^shift for a shift in [-126, 0], exactly; 0 below
// -126, where the scaled term is under the resolution of anything it is added to.
fn getSigmaScale(shift: i32) -> f32 {
  if (shift < -126) { return 0.0; }
  return bitcast<f32>(u32(min(shift, 0) + 127) << 23u);
}`;
}

/** Copies both CSR directions into the packed adjacency buffer. @internal */
export function createCentralityPackNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    layout: CentralityAdjacencyLayout;
    forward: {
      offsets: GraphDataView<'uint32'>;
      neighbors: GraphDataView<'uint32'>;
      weights: GraphDataView<'float32'>;
    };
    reverse: {
      offsets: GraphDataView<'uint32'>;
      neighbors: GraphDataView<'uint32'>;
      weights: GraphDataView<'float32'>;
    };
    adjacency: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const {layout} = props;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'pack-adjacency',
    bindings: [
      {name: 'forwardOffsets', view: props.forward.offsets, type: 'u32', access: 'read'},
      {name: 'forwardNeighbors', view: props.forward.neighbors, type: 'u32', access: 'read'},
      {name: 'forwardWeights', view: props.forward.weights, type: 'u32', access: 'read'},
      {name: 'reverseOffsets', view: props.reverse.offsets, type: 'u32', access: 'read'},
      {name: 'reverseNeighbors', view: props.reverse.neighbors, type: 'u32', access: 'read'},
      {name: 'reverseWeights', view: props.reverse.weights, type: 'u32', access: 'read'},
      {name: 'adjacency', view: props.adjacency, type: 'u32', access: 'read_write'}
    ],
    invocationCount: layout.wordCount,
    declarations: `const FORWARD_NEIGHBORS: u32 = ${layout.forwardNeighbors}u;
const FORWARD_WEIGHTS: u32 = ${layout.forwardWeights}u;
const REVERSE_OFFSETS: u32 = ${layout.reverseOffsets}u;
const REVERSE_NEIGHBORS: u32 = ${layout.reverseNeighbors}u;
const REVERSE_WEIGHTS: u32 = ${layout.reverseWeights}u;`,
    body: `var value = 0u;
  if (index < FORWARD_NEIGHBORS) {
    value = forwardOffsets[forwardOffsetsOffset + index];
  } else if (index < FORWARD_WEIGHTS) {
    value = forwardNeighbors[forwardNeighborsOffset + index - FORWARD_NEIGHBORS];
  } else if (index < REVERSE_OFFSETS) {
    value = forwardWeights[forwardWeightsOffset + index - FORWARD_WEIGHTS];
  } else if (index < REVERSE_NEIGHBORS) {
    value = reverseOffsets[reverseOffsetsOffset + index - REVERSE_OFFSETS];
  } else if (index < REVERSE_WEIGHTS) {
    value = reverseNeighbors[reverseNeighborsOffset + index - REVERSE_NEIGHBORS];
  } else {
    value = reverseWeights[reverseWeightsOffset + index - REVERSE_WEIGHTS];
  }
  adjacency[adjacencyOffset + index] = value;`
  });
}

/**
 * Expands the pivots into per-batch lane seeds: pivot row `batch * laneCount + lane` becomes
 * expanded node `lane * nodeCount + pivot`, or `NONE` for unused lanes and out-of-range pivots.
 * Batch `b` writes at `b * seedStride`. @internal
 */
export function createCentralityExpandSeedsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    laneCount: number;
    batchCount: number;
    seedStride: number;
    pivotNodes: GraphDataView<'uint32'>;
    expandedSeeds: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'expand-seeds',
    bindings: [
      {name: 'pivotNodes', view: props.pivotNodes, type: 'u32', access: 'read'},
      {name: 'expandedSeeds', view: props.expandedSeeds, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.batchCount * props.laneCount,
    declarations: `const NONE: u32 = 0xffffffffu;
const NODE_COUNT: u32 = ${props.nodeCount}u;
const LANE_COUNT: u32 = ${props.laneCount}u;
const PIVOT_COUNT: u32 = ${props.pivotNodes.length}u;
const SEED_STRIDE: u32 = ${props.seedStride}u;`,
    body: `let batch = index / LANE_COUNT;
  let lane = index - batch * LANE_COUNT;
  let row = batch * LANE_COUNT + lane;
  var seed = NONE;
  if (row < PIVOT_COUNT) {
    let pivot = pivotNodes[pivotNodesOffset + row];
    if (pivot < NODE_COUNT) { seed = lane * NODE_COUNT + pivot; }
  }
  expandedSeeds[expandedSeedsOffset + batch * SEED_STRIDE + lane] = seed;`
  });
}

/** Writes each batch's first pivot row into its 256-byte info slot (word 0). @internal */
export function createCentralityBatchInfoNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {id: string; laneCount: number; batchCount: number; batchInfo: GraphDataView<'uint32'>}
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'batch-info',
    bindings: [{name: 'batchInfo', view: props.batchInfo, type: 'u32', access: 'read_write'}],
    invocationCount: props.batchCount,
    declarations: `const LANE_COUNT: u32 = ${props.laneCount}u;
const SLOT_WORDS: u32 = ${CENTRALITY_ALIGNMENT_ELEMENTS}u;`,
    body: `batchInfo[batchInfoOffset + index * SLOT_WORDS] = index * LANE_COUNT;`
  });
}

/** Zeroes the betweenness output once, before the first batch adds to it. @internal */
export function createCentralityZeroNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {id: string; betweenness: GraphDataView<'float32'>}
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'zero',
    bindings: [{name: 'betweenness', view: props.betweenness, type: 'f32', access: 'read_write'}],
    invocationCount: props.betweenness.length,
    body: 'betweenness[betweennessOffset + index] = 0.0;'
  });
}

/**
 * Per-lane closeness: one workgroup reduces the costs of one lane in a fixed order (strided
 * per-thread partial sums, then a binary tree) to the reached count, the sum of costs and the sum of
 * `1 / cost` over costs above zero, and writes the pivot row `firstRow + lane`. @internal
 */
export function createCentralityClosenessNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    laneCount: number;
    rowCount: number;
    costs: GraphDataView<'float32'>;
    /** Slot of this batch; word 0 is the first pivot row. */
    batchInfo: GraphDataView<'uint32'>;
    closeness?: GraphDataView<'float32'>;
    harmonicCloseness?: GraphDataView<'float32'>;
    reachedCounts?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
    {name: 'batchInfo', view: props.batchInfo, type: 'u32', access: 'read'}
  ];
  if (props.closeness) {
    bindings.push({name: 'closeness', view: props.closeness, type: 'f32', access: 'read_write'});
  }
  if (props.harmonicCloseness) {
    bindings.push({
      name: 'harmonicCloseness',
      view: props.harmonicCloseness,
      type: 'f32',
      access: 'read_write'
    });
  }
  if (props.reachedCounts) {
    bindings.push({
      name: 'reachedCounts',
      view: props.reachedCounts,
      type: 'u32',
      access: 'read_write'
    });
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'closeness',
    // One workgroup per pivot lane, so the invocations are not one datum each.
    bindings,
    invocationCount: props.laneCount * CLOSENESS_WORKGROUP_SIZE,
    workgroupSize: CLOSENESS_WORKGROUP_SIZE,
    guardIndex: false,
    declarations: `const INFINITY_BITS: u32 = 0x7f800000u;
const NODE_COUNT: u32 = ${props.nodeCount}u;
const LANE_COUNT: u32 = ${props.laneCount}u;
const ROW_COUNT: u32 = ${props.rowCount}u;
var<workgroup> reachedPartial: array<u32, ${CLOSENESS_WORKGROUP_SIZE}>;
var<workgroup> costPartial: array<f32, ${CLOSENESS_WORKGROUP_SIZE}>;
var<workgroup> harmonicPartial: array<f32, ${CLOSENESS_WORKGROUP_SIZE}>;`,
    body: `let lane = index / ${CLOSENESS_WORKGROUP_SIZE}u;
  var reached = 0u;
  var costSum = 0.0;
  var harmonicSum = 0.0;
  if (lane < LANE_COUNT) {
    for (var node = localInvocationIndex; node < NODE_COUNT; node += ${CLOSENESS_WORKGROUP_SIZE}u) {
      let cost = costs[costsOffset + lane * NODE_COUNT + node];
      if (bitcast<u32>(cost) < INFINITY_BITS) {
        reached++;
        costSum += cost;
        if (cost > 0.0) { harmonicSum += 1.0 / cost; }
      }
    }
  }
  reachedPartial[localInvocationIndex] = reached;
  costPartial[localInvocationIndex] = costSum;
  harmonicPartial[localInvocationIndex] = harmonicSum;
  workgroupBarrier();
  for (var stride = ${CLOSENESS_WORKGROUP_SIZE / 2}u; stride > 0u; stride >>= 1u) {
    if (localInvocationIndex < stride) {
      reachedPartial[localInvocationIndex] += reachedPartial[localInvocationIndex + stride];
      costPartial[localInvocationIndex] += costPartial[localInvocationIndex + stride];
      harmonicPartial[localInvocationIndex] += harmonicPartial[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  let row = batchInfo[batchInfoOffset] + lane;
  if (localInvocationIndex == 0u && lane < LANE_COUNT && row < ROW_COUNT) {
    let totalCost = costPartial[0];
    ${
      props.closeness
        ? `closeness[closenessOffset + row] = select(0.0, (f32(reachedPartial[0]) - 1.0) / totalCost, totalCost > 0.0);`
        : ''
    }
    ${props.harmonicCloseness ? 'harmonicCloseness[harmonicClosenessOffset + row] = harmonicPartial[0];' : ''}
    ${props.reachedCounts ? 'reachedCounts[reachedCountsOffset + row] = reachedPartial[0];' : ''}
  }`
  });
}

/**
 * Clears the queued-round stamps and resets the control words and dispatch slots of the sigma and
 * delta phases of one batch. @internal
 */
export function createCentralityResetNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    state: FrontierState;
    sigmaPhase: FrontierPhase;
    deltaPhase: FrontierPhase;
  }
): GPUCommandNode<Parameters> {
  const {state, sigmaPhase, deltaPhase} = props;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'reset',
    bindings: [
      {name: 'queuedRound', view: state.queuedRound, type: 'u32', access: 'read_write'},
      {name: 'control', view: state.control, type: 'u32', access: 'read_write'},
      {name: 'dispatchEven', view: state.dispatchEven, type: 'u32', access: 'read_write'},
      {name: 'dispatchOdd', view: state.dispatchOdd, type: 'u32', access: 'read_write'}
    ],
    invocationCount: Math.max(
      state.nodeCount,
      getFrontierResetInvocationCount(sigmaPhase),
      getFrontierResetInvocationCount(deltaPhase)
    ),
    body: `${getFrontierResetSource(sigmaPhase, {limit: `${sigmaPhase.maxRounds}u`})}
  ${getFrontierResetSource(deltaPhase, {limit: `${deltaPhase.maxRounds}u`})}
  if (index < ${state.nodeCount}u) {
    queuedRound[queuedRoundOffset + index] = 0u;
  }`
  });
}

/** Properties shared by the sigma and delta initialize and round kernels. @internal */
type CentralityPhaseProps = {
  id: string;
  layout: CentralityAdjacencyLayout;
  expandedNodeCount: number;
  state: FrontierState;
  phase: FrontierPhase;
  adjacency: GraphDataView<'uint32'>;
  costs: GraphDataView<'float32'>;
  scratch: GraphDataView<'uint32'>;
  /** Base-2 exponent (i32 bits) of every expanded node's path count. */
  exponents: GraphDataView<'uint32'>;
};

/**
 * Counts the tight in-edges of every expanded node into `remaining`, gives nodes without one (the
 * pivots) sigma 1 and pushes them into round 0 of the sigma phase. @internal
 */
export function createCentralitySigmaInitializeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: CentralityPhaseProps
): GPUCommandNode<Parameters> {
  const {layout, expandedNodeCount, state, phase} = props;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'sigma-initialize',
    bindings: [
      {name: 'adjacency', view: props.adjacency, type: 'u32', access: 'read'},
      {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
      {name: 'scratch', view: props.scratch, type: 'u32', access: 'read_write'},
      {name: 'exponents', view: props.exponents, type: 'u32', access: 'read_write'},
      ...getFrontierBindings(graph, state, phase, 0)
    ],
    invocationCount: expandedNodeCount,
    declarations: `${getCentralitySharedSource(layout, expandedNodeCount)}
${getFrontierPushSource(phase, expandedNodeCount, graph.device.limits.maxComputeWorkgroupsPerDimension)}`,
    body: `let laneBase = (index / NODE_COUNT) * NODE_COUNT;
  let csrNode = index - laneBase;
  let cost = costs[costsOffset + index];
  var tightInCount = 0u;
  if (bitcast<u32>(cost) < INFINITY_BITS) {
    let first = min(adjacency[adjacencyOffset + REVERSE_OFFSETS + csrNode], REVERSE_EDGE_COUNT);
    let last = min(adjacency[adjacencyOffset + REVERSE_OFFSETS + csrNode + 1u], REVERSE_EDGE_COUNT);
    for (var edge = first; edge < last; edge++) {
      let sourceCsrNode = adjacency[adjacencyOffset + REVERSE_NEIGHBORS + edge];
      if (sourceCsrNode >= NODE_COUNT) { continue; }
      let weight = bitcast<f32>(adjacency[adjacencyOffset + REVERSE_WEIGHTS + edge]);
      if (isTightEdge(sourceCsrNode, csrNode, costs[costsOffset + laneBase + sourceCsrNode], weight, cost)) {
        tightInCount++;
      }
    }
  }
  scratch[scratchOffset + REMAINING_BASE + index] = tightInCount;
  scratch[scratchOffset + SIGMA_BASE + index] = 0u;
  exponents[exponentsOffset + index] = 0u;
  // Reached nodes without a tight in-edge are the roots: the pivots, and nodes reached only
  // across a positive weight that float32 absorbs. A root has one path: mantissa 1, exponent 0.
  if (tightInCount == 0u && bitcast<u32>(cost) < INFINITY_BITS) {
    scratch[scratchOffset + SIGMA_BASE + index] = 0x3f800000u;
    frontierPush(index);
  }`
  });
}

/**
 * One round of the sigma phase. A queued node has all its tight predecessors final: it sums their
 * sigma in reverse-CSR order, then releases its tight successors, pushing each whose last
 * predecessor it was. The sum is formed at the largest exponent seen so far (scaling by a power of
 * two is exact, so it rounds exactly like the plain float32 sum while that stays in range) and is
 * stored normalised to a mantissa in [1, 2) and an exponent. @internal
 */
export function createCentralitySigmaRoundNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: CentralityPhaseProps & {round: number; localIterations: number}
): GPUCommandNode<Parameters> {
  const {layout, expandedNodeCount} = props;
  return createFrontierRoundNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'sigma',
    state: props.state,
    phase: props.phase,
    round: props.round,
    localIterations: props.localIterations,
    bindings: [
      {name: 'adjacency', view: props.adjacency, type: 'u32', access: 'read'},
      {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
      {name: 'scratch', view: props.scratch, type: 'atomic<u32>', access: 'read_write'},
      // Written once by the node's own visit and read in later dispatches only (one hop per round).
      {name: 'exponents', view: props.exponents, type: 'u32', access: 'read_write'}
    ],
    declarations: `${getCentralitySharedSource(layout, expandedNodeCount)}
fn loadScratchFloat(word: u32) -> f32 {
  return bitcast<f32>(atomicLoad(&scratch[scratchOffset + word]));
}`,
    visitSource: `fn visit(node: u32) {
  let laneBase = (node / NODE_COUNT) * NODE_COUNT;
  let csrNode = node - laneBase;
  let cost = costs[costsOffset + node];
  var tightInCount = 0u;
  // total * 2^totalExponent is the running path count.
  var total = 0.0;
  var totalExponent = 0;
  let firstReverse = min(adjacency[adjacencyOffset + REVERSE_OFFSETS + csrNode], REVERSE_EDGE_COUNT);
  let lastReverse = min(adjacency[adjacencyOffset + REVERSE_OFFSETS + csrNode + 1u], REVERSE_EDGE_COUNT);
  for (var edge = firstReverse; edge < lastReverse; edge++) {
    let sourceCsrNode = adjacency[adjacencyOffset + REVERSE_NEIGHBORS + edge];
    if (sourceCsrNode >= NODE_COUNT) { continue; }
    let weight = bitcast<f32>(adjacency[adjacencyOffset + REVERSE_WEIGHTS + edge]);
    if (!isTightEdge(sourceCsrNode, csrNode, costs[costsOffset + laneBase + sourceCsrNode], weight, cost)) { continue; }
    tightInCount++;
    let mantissa = loadScratchFloat(SIGMA_BASE + laneBase + sourceCsrNode);
    if (!(mantissa > 0.0)) { continue; }
    let exponent = bitcast<i32>(exponents[exponentsOffset + laneBase + sourceCsrNode]);
    if (total == 0.0) {
      total = mantissa;
      totalExponent = exponent;
    } else if (exponent > totalExponent) {
      total = total * getSigmaScale(totalExponent - exponent) + mantissa;
      totalExponent = exponent;
    } else {
      total += mantissa * getSigmaScale(exponent - totalExponent);
    }
  }
  // A root keeps the sigma 1 written by the initialize kernel.
  if (tightInCount > 0u) {
    var mantissaBits = 0u;
    if (total > 0.0) {
      // Normalise: the sum of at most 2^24 mantissas below 2 is a normal float32 of at least 1.
      let totalBits = bitcast<u32>(total);
      totalExponent += i32((totalBits >> 23u) & 0xffu) - 127;
      mantissaBits = (totalBits & 0x007fffffu) | 0x3f800000u;
    }
    exponents[exponentsOffset + node] = bitcast<u32>(totalExponent);
    atomicStore(&scratch[scratchOffset + SIGMA_BASE + node], mantissaBits);
  }
  let firstForward = min(adjacency[adjacencyOffset + FORWARD_OFFSETS + csrNode], FORWARD_EDGE_COUNT);
  let lastForward = min(adjacency[adjacencyOffset + FORWARD_OFFSETS + csrNode + 1u], FORWARD_EDGE_COUNT);
  for (var edge = firstForward; edge < lastForward; edge++) {
    let targetCsrNode = adjacency[adjacencyOffset + FORWARD_NEIGHBORS + edge];
    if (targetCsrNode >= NODE_COUNT) { continue; }
    let weight = bitcast<f32>(adjacency[adjacencyOffset + FORWARD_WEIGHTS + edge]);
    if (!isTightEdge(csrNode, targetCsrNode, cost, weight, costs[costsOffset + laneBase + targetCsrNode])) { continue; }
    if (atomicSub(&scratch[scratchOffset + REMAINING_BASE + laneBase + targetCsrNode], 1u) == 1u) {
      push(laneBase + targetCsrNode);
    }
  }
}`
  });
}

/**
 * Counts the tight out-edges of every expanded node into `remaining` (reusing the sigma counters),
 * clears delta and pushes the reached nodes without a tight out-edge into round 0 of the delta
 * phase. Before it overwrites a counter it sets the delta phase's first parameter word when the
 * sigma phase left a reached node unordered. @internal
 */
export function createCentralityDeltaInitializeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: CentralityPhaseProps
): GPUCommandNode<Parameters> {
  const {layout, expandedNodeCount, state, phase} = props;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'delta-initialize',
    bindings: [
      {name: 'adjacency', view: props.adjacency, type: 'u32', access: 'read'},
      {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
      {name: 'scratch', view: props.scratch, type: 'u32', access: 'read_write'},
      ...getFrontierBindings(graph, state, phase, 0)
    ],
    invocationCount: expandedNodeCount,
    declarations: `${getCentralitySharedSource(layout, expandedNodeCount)}
${getFrontierPushSource(phase, expandedNodeCount, graph.device.limits.maxComputeWorkgroupsPerDimension)}`,
    body: `let laneBase = (index / NODE_COUNT) * NODE_COUNT;
  let csrNode = index - laneBase;
  let cost = costs[costsOffset + index];
  var tightOutCount = 0u;
  if (bitcast<u32>(cost) < INFINITY_BITS) {
    let first = min(adjacency[adjacencyOffset + FORWARD_OFFSETS + csrNode], FORWARD_EDGE_COUNT);
    let last = min(adjacency[adjacencyOffset + FORWARD_OFFSETS + csrNode + 1u], FORWARD_EDGE_COUNT);
    for (var edge = first; edge < last; edge++) {
      let targetCsrNode = adjacency[adjacencyOffset + FORWARD_NEIGHBORS + edge];
      if (targetCsrNode >= NODE_COUNT) { continue; }
      let weight = bitcast<f32>(adjacency[adjacencyOffset + FORWARD_WEIGHTS + edge]);
      if (isTightEdge(csrNode, targetCsrNode, cost, weight, costs[costsOffset + laneBase + targetCsrNode])) {
        tightOutCount++;
      }
    }
    // The sigma phase left this counter at the number of tight predecessors that never released
    // the node. Above zero, the node was never ordered: the sigma phase hit its round limit, or
    // the node lies on or after a cycle of zero-weight tight edges. The batch status node tells
    // the two apart.
    if (scratch[scratchOffset + REMAINING_BASE + index] != 0u) {
      atomicStore(&control[controlOffset + FRONTIER_PARAMETER_0_WORD], 1u);
    }
  }
  scratch[scratchOffset + REMAINING_BASE + index] = tightOutCount;
  scratch[scratchOffset + DELTA_BASE + index] = 0u;
  if (tightOutCount == 0u && bitcast<u32>(cost) < INFINITY_BITS) {
    frontierPush(index);
  }`
  });
}

/**
 * One round of the delta phase, the mirror image of the sigma round. A queued node has all its tight
 * successors final: it sums `sigma(v) / sigma(w) * (1 + delta(w))` over its tight out-edges in
 * forward-CSR order, then releases its tight predecessors. @internal
 */
export function createCentralityDeltaRoundNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: CentralityPhaseProps & {round: number; localIterations: number}
): GPUCommandNode<Parameters> {
  const {layout, expandedNodeCount} = props;
  return createFrontierRoundNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'delta',
    state: props.state,
    phase: props.phase,
    round: props.round,
    localIterations: props.localIterations,
    bindings: [
      {name: 'adjacency', view: props.adjacency, type: 'u32', access: 'read'},
      {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
      {name: 'scratch', view: props.scratch, type: 'atomic<u32>', access: 'read_write'},
      {name: 'exponents', view: props.exponents, type: 'u32', access: 'read'}
    ],
    declarations: `${getCentralitySharedSource(layout, expandedNodeCount)}
fn loadScratchFloat(word: u32) -> f32 {
  return bitcast<f32>(atomicLoad(&scratch[scratchOffset + word]));
}`,
    visitSource: `fn visit(node: u32) {
  let laneBase = (node / NODE_COUNT) * NODE_COUNT;
  let csrNode = node - laneBase;
  let cost = costs[costsOffset + node];
  let nodeSigma = loadScratchFloat(SIGMA_BASE + node);
  let nodeExponent = bitcast<i32>(exponents[exponentsOffset + node]);
  var total = 0.0;
  let firstForward = min(adjacency[adjacencyOffset + FORWARD_OFFSETS + csrNode], FORWARD_EDGE_COUNT);
  let lastForward = min(adjacency[adjacencyOffset + FORWARD_OFFSETS + csrNode + 1u], FORWARD_EDGE_COUNT);
  for (var edge = firstForward; edge < lastForward; edge++) {
    let targetCsrNode = adjacency[adjacencyOffset + FORWARD_NEIGHBORS + edge];
    if (targetCsrNode >= NODE_COUNT) { continue; }
    let weight = bitcast<f32>(adjacency[adjacencyOffset + FORWARD_WEIGHTS + edge]);
    if (!isTightEdge(csrNode, targetCsrNode, cost, weight, costs[costsOffset + laneBase + targetCsrNode])) { continue; }
    let targetSigma = loadScratchFloat(SIGMA_BASE + laneBase + targetCsrNode);
    // A zero sigma only occurs when the sigma phase left the target unordered (round limit or a
    // zero-weight cycle).
    if (!(targetSigma > 0.0)) { continue; }
    // sigma(v) / sigma(w): the ratio of the mantissas times an exact power of two. A target has at
    // least the paths of each predecessor, so the exponent difference is never positive.
    let targetExponent = bitcast<i32>(exponents[exponentsOffset + laneBase + targetCsrNode]);
    let ratio = nodeSigma / targetSigma * getSigmaScale(nodeExponent - targetExponent);
    total += ratio * (1.0 + loadScratchFloat(DELTA_BASE + laneBase + targetCsrNode));
  }
  atomicStore(&scratch[scratchOffset + DELTA_BASE + node], bitcast<u32>(total));
  let firstReverse = min(adjacency[adjacencyOffset + REVERSE_OFFSETS + csrNode], REVERSE_EDGE_COUNT);
  let lastReverse = min(adjacency[adjacencyOffset + REVERSE_OFFSETS + csrNode + 1u], REVERSE_EDGE_COUNT);
  for (var edge = firstReverse; edge < lastReverse; edge++) {
    let sourceCsrNode = adjacency[adjacencyOffset + REVERSE_NEIGHBORS + edge];
    if (sourceCsrNode >= NODE_COUNT) { continue; }
    let weight = bitcast<f32>(adjacency[adjacencyOffset + REVERSE_WEIGHTS + edge]);
    if (!isTightEdge(sourceCsrNode, csrNode, costs[costsOffset + laneBase + sourceCsrNode], weight, cost)) { continue; }
    if (atomicSub(&scratch[scratchOffset + REMAINING_BASE + laneBase + sourceCsrNode], 1u) == 1u) {
      push(laneBase + sourceCsrNode);
    }
  }
}`
  });
}

/**
 * Adds one batch to the betweenness output: for every node, the lane-ordered sum of the lanes'
 * delta (skipping the lane's own pivot), scaled and added to the running value. @internal
 */
export function createCentralityAccumulateNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    laneCount: number;
    expandedNodeCount: number;
    betweennessScale: number;
    scratch: GraphDataView<'uint32'>;
    /** The batch's lane seeds (expanded pivot node per lane, or `NONE`). */
    seeds: GraphDataView<'uint32'>;
    betweenness: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'accumulate',
    bindings: [
      {name: 'scratch', view: props.scratch, type: 'u32', access: 'read'},
      {name: 'seeds', view: props.seeds, type: 'u32', access: 'read'},
      {name: 'betweenness', view: props.betweenness, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.nodeCount,
    declarations: `const NODE_COUNT: u32 = ${props.nodeCount}u;
const LANE_COUNT: u32 = ${props.laneCount}u;
const DELTA_BASE: u32 = ${2 * props.expandedNodeCount}u;
const BETWEENNESS_SCALE: f32 = ${getWGSLFloatLiteral(props.betweennessScale)};`,
    body: `var total = 0.0;
  for (var lane = 0u; lane < LANE_COUNT; lane++) {
    let expanded = lane * NODE_COUNT + index;
    if (seeds[seedsOffset + lane] == expanded) { continue; }
    total += bitcast<f32>(scratch[scratchOffset + DELTA_BASE + expanded]);
  }
  betweenness[betweennessOffset + index] = betweenness[betweennessOffset + index] + BETWEENNESS_SCALE * total;`
  });
}

/** Word of a batch status slot that holds the batch's zero-weight cycle flag. @internal */
export const CENTRALITY_BATCH_ZERO_WEIGHT_CYCLE_WORD = 1;
/** Word of a batch status slot that holds a round-limit exit. @internal */
const CENTRALITY_BATCH_ITERATION_LIMIT_REACHED_WORD = 2;
/** Convergence, zero-weight cycle, and round-limit exit per betweenness batch. @internal */
export const CENTRALITY_BATCH_STATUS_WORDS = 3;

/**
 * Finishes a batch's status slot (converged, zero-weight cycle, round-limit exit). Word 0, written by
 * the batch's reachability search, is cleared when either phase hit its round limit or the sigma
 * phase left a reached node unordered. Word 1 is set when nodes were left unordered although the
 * sigma phase did not hit its round limit, which only a cycle of zero-weight tight edges causes.
 * Word 2 records any search, sigma or delta round-limit exit independently of cycle detection.
 * @internal
 */
export function createCentralityBatchStatusNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    state: FrontierState;
    sigmaPhase: FrontierPhase;
    deltaPhase: FrontierPhase;
    /** The batch's three status words. */
    batchStatus: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'batch-status',
    bindings: [
      {name: 'control', view: props.state.control, type: 'u32', access: 'read'},
      {name: 'batchStatus', view: props.batchStatus, type: 'u32', access: 'read_write'}
    ],
    invocationCount: 1,
    body: `let searchTruncated = batchStatus[batchStatusOffset] == 0u;
  let sigmaTruncated = control[controlOffset + ${props.sigmaPhase.controlWordOffset + FRONTIER_CONTROL_TRUNCATED}u] != 0u;
  let deltaTruncated = control[controlOffset + ${props.deltaPhase.controlWordOffset + FRONTIER_CONTROL_TRUNCATED}u] != 0u;
  let unordered = control[controlOffset + ${props.deltaPhase.controlWordOffset + FRONTIER_CONTROL_PARAMETER_0}u] != 0u;
  if (sigmaTruncated || deltaTruncated || unordered) {
    batchStatus[batchStatusOffset] = 0u;
  }
  batchStatus[batchStatusOffset + ${CENTRALITY_BATCH_ZERO_WEIGHT_CYCLE_WORD}u] = select(0u, 1u, unordered && !sigmaTruncated);
  batchStatus[batchStatusOffset + ${CENTRALITY_BATCH_ITERATION_LIMIT_REACHED_WORD}u] = select(0u, 1u, searchTruncated || sigmaTruncated || deltaTruncated);`
  });
}

/**
 * Combines the batch status slots into the one-row output flags: `converged` is the minimum of
 * the batches' word 0, `zeroWeightCycle` the maximum of their word 1, and iterationLimitReached
 * the maximum of their independent round-limit words. Closeness-only batches use the search's
 * convergence flag directly, since they have no cycle failure. @internal
 */
export function createCentralityConvergedNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    batchCount: number;
    batchStatus: GraphDataView<'uint32'>;
    converged?: GraphDataView<'uint32'>;
    zeroWeightCycle?: GraphDataView<'uint32'>;
    iterationLimitReached?: GraphDataView<'uint32'>;
    hasBetweenness: boolean;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'batchStatus', view: props.batchStatus, type: 'u32', access: 'read'}
  ];
  if (props.converged) {
    bindings.push({name: 'converged', view: props.converged, type: 'u32', access: 'read_write'});
  }
  if (props.zeroWeightCycle) {
    bindings.push({
      name: 'zeroWeightCycle',
      view: props.zeroWeightCycle,
      type: 'u32',
      access: 'read_write'
    });
  }
  if (props.iterationLimitReached) {
    bindings.push({
      name: 'iterationLimitReached',
      view: props.iterationLimitReached,
      type: 'u32',
      access: 'read_write'
    });
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: `converged${props.converged ? '' : '-no-flag'}${props.zeroWeightCycle ? '-cycle' : ''}`,
    bindings,
    invocationCount: 1,
    declarations: `const BATCH_COUNT: u32 = ${props.batchCount}u;
const SLOT_WORDS: u32 = ${CENTRALITY_ALIGNMENT_ELEMENTS}u;`,
    body: `var allConverged = 1u;
  var anyCycle = 0u;
  var anyLimitReached = 0u;
  for (var batch = 0u; batch < BATCH_COUNT; batch++) {
    allConverged = min(allConverged, batchStatus[batchStatusOffset + batch * SLOT_WORDS]);
    anyCycle = max(anyCycle, batchStatus[batchStatusOffset + batch * SLOT_WORDS + ${CENTRALITY_BATCH_ZERO_WEIGHT_CYCLE_WORD}u]);
    anyLimitReached = max(anyLimitReached, ${
      props.hasBetweenness
        ? `batchStatus[batchStatusOffset + batch * SLOT_WORDS + ${CENTRALITY_BATCH_ITERATION_LIMIT_REACHED_WORD}u]`
        : '1u - batchStatus[batchStatusOffset + batch * SLOT_WORDS]'
    });
  }
  ${props.converged ? 'converged[convergedOffset] = allConverged;' : ''}
  ${props.zeroWeightCycle ? 'zeroWeightCycle[zeroWeightCycleOffset] = anyCycle;' : ''}
  ${props.iterationLimitReached ? 'iterationLimitReached[iterationLimitReachedOffset] = anyLimitReached;' : ''}`
  });
}

/** Adds Brandes dependencies to stable forward CSR edge slots. @internal */
export function createCentralityEdgeAccumulateNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    layout: CentralityAdjacencyLayout;
    expandedNodeCount: number;
    laneCount: number;
    betweennessScale: number;
    adjacency: GraphDataView<'uint32'>;
    scratch: GraphDataView<'uint32'>;
    exponents: GraphDataView<'uint32'>;
    costs: GraphDataView<'float32'>;
    edgeBetweenness: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'edge-accumulate',
    invocationCount: props.layout.nodeCount,
    bindings: [
      {name: 'adjacency', view: props.adjacency, type: 'u32', access: 'read'},
      {name: 'scratch', view: props.scratch, type: 'u32', access: 'read'},
      {name: 'exponents', view: props.exponents, type: 'u32', access: 'read'},
      {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
      {name: 'edges', view: props.edgeBetweenness, type: 'f32', access: 'read_write'}
    ],
    declarations: getCentralitySharedSource(props.layout, props.expandedNodeCount),
    body: `let first = min(adjacency[adjacencyOffset + FORWARD_OFFSETS + index], FORWARD_EDGE_COUNT);
  let last = min(adjacency[adjacencyOffset + FORWARD_OFFSETS + index + 1u], FORWARD_EDGE_COUNT);
  for (var edge = first; edge < last; edge++) {
    let destination = adjacency[adjacencyOffset + FORWARD_NEIGHBORS + edge];
    let weight = bitcast<f32>(adjacency[adjacencyOffset + FORWARD_WEIGHTS + edge]);
    var total = 0.0;
    if (destination < NODE_COUNT) {
      for (var lane = 0u; lane < ${props.laneCount}u; lane++) {
        let sourceRow = lane * NODE_COUNT + index;
        let targetRow = lane * NODE_COUNT + destination;
        if (isTightEdge(index, destination, costs[costsOffset + sourceRow], weight, costs[costsOffset + targetRow])) {
          let denominator = bitcast<f32>(scratch[scratchOffset + targetRow]);
          if (denominator > 0.0) {
            let ratio = bitcast<f32>(scratch[scratchOffset + sourceRow]) / denominator * getSigmaScale(
              bitcast<i32>(exponents[exponentsOffset + sourceRow]) -
              bitcast<i32>(exponents[exponentsOffset + targetRow])
            );
            total += ratio * (1.0 + bitcast<f32>(scratch[scratchOffset + DELTA_BASE + targetRow]));
          }
        }
      }
    }
    edges[edgesOffset + edge] += ${getWGSLFloatLiteral(props.betweennessScale)} * total;
  }`
  });
}
