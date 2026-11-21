// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPU_MAXIMUM_COMMAND_NODE_COUNT,
  getCoreBindingElementLimit,
  validateCapacity,
  validateMaximumCount,
  validateRequirement,
  validateSinglePackedView,
  validateViewLength,
  validateCommandNodeBudget
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {
  GPUAnalysisExecutionPlanner,
  GPUAnalysisRoute
} from '../../gpu-spatial-analysis/ontology/spatial-analysis-ontology';
import type {GPUSpatialContext} from '../../analysis-core/contracts/index';
import {alignElementCount} from '../network-view-utils';
import {
  getFrontierPhaseLayout,
  FRONTIER_DISPATCH_SLOT_WORDS
} from '../network-reachability/network-frontier';
import {getUrbanPointGridShape} from '../../gpu-spatial-analysis/urban-morphology/visibility-graph-point-grid';
import {
  addVisibilityGraphIntegration,
  addVisibilityGraphMetricNodes
} from './visibility-graph-integration';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {createVisibilityGraphPivotsNode} from './visibility-graph-pivots';
import {GPUNetworkCentrality} from '../network-centrality/gpu-network-centrality';
import {
  GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS,
  GPU_NETWORK_REACHABILITY_MAXIMUM_LOCAL_ITERATIONS
} from '../network-reachability/gpu-network-reachability';
import {
  addVisibilityGraphRecipe as addSpatialVisibilityGraphRecipe,
  type GPUVisibilityGraphProps,
  type GPUVisibilityGraphReferenceRecipeResult,
  VISIBILITY_GRAPH_MAXIMUM_POINT_COUNT
} from '../../gpu-spatial-analysis/urban-morphology/visibility-graph';
import {
  prepareVisibilityGraphCSRRecipe,
  VISIBILITY_GRAPH_CSR_MAXIMUM_PIVOT_COUNT,
  VISIBILITY_GRAPH_CSR_DEFAULT_PIVOT_COUNT,
  type GPUVisibilityGraphAdjacencyCSRProps as GPUVisibilityGraphAdjacencyProps,
  type GPUVisibilityGraphAdjacencyCSROutput as GPUVisibilityGraphAdjacencyOutput,
  type GPUVisibilityGraphAdjacencyCSRStatus as GPUVisibilityGraphAdjacencyStatus
} from '../../gpu-spatial-analysis/urban-morphology/visibility-graph-csr';
import {validateUrbanStorageByteLength} from '../../gpu-spatial-analysis/urban-morphology/building-tessellation-validation';
import {validateVisibilityGraphCSRBindings} from '../../gpu-spatial-analysis/urban-morphology/visibility-graph-bindings';
import {getRecipeView} from '../../gpu-spatial-analysis/recipes/expansion-recipe-utils';
import {createVisibilityGraphKernelNode} from '../../gpu-spatial-analysis/urban-morphology/visibility-graph-kernel';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

/** More simultaneous sources amortize all-source commands; the binding limit further lowers this. */
const VISIBILITY_GRAPH_NETWORK_DEFAULT_LANE_COUNT = 32;
/** One 256-byte aligned control slot per batch/round in the published centrality contributor. */
const VISIBILITY_GRAPH_NETWORK_CONTROL_BYTE_ALLOWANCE = 256;
/** Bounds initialization, reductions, pivot checks and safety masks outside centrality rounds. */
const VISIBILITY_GRAPH_NETWORK_COMMAND_NODE_ALLOWANCE = 32;
/** Centrality has three frontier phases; integration adds one. A fifth node per round is conservative. */
const VISIBILITY_GRAPH_NETWORK_COMMANDS_PER_ROUND = 5;
/** Bounds initialization, finalization, accumulation and stopping-status nodes of one pivot batch. */
const VISIBILITY_GRAPH_NETWORK_BATCH_COMMAND_ALLOWANCE = 64;

export type GPUNetworkVisibilityGraphCSRStatus = GPUVisibilityGraphAdjacencyStatus & {
  /** One when fewer than all viewpoints are pivots; closeness follows the listed pivot order. */
  approximation: GraphDataView<'uint32'>;
  converged: GraphDataView<'uint32'>;
  iterationLimitReached: GraphDataView<'uint32'>;
  zeroWeightCycle: GraphDataView<'uint32'>;
};
export type GPUNetworkVisibilityGraphCSROutput = Omit<
  GPUVisibilityGraphAdjacencyOutput,
  'status'
> & {
  /** Unnormalised undirected betweenness, scaled by nodeCount / pivotCount for sampled pivots. */
  betweenness: GraphDataView<'float32'>;
  /** (reachableCount - 1) / totalHopDepth, one row per pivot rather than per viewpoint. */
  closeness: GraphDataView<'float32'>;
  reachedCounts: GraphDataView<'uint32'>;
  /** Per viewpoint: degree, reachable other nodes, total hop depth, reachable / total depth. Recipe-owned metric fields support COPY_SRC snapshots. */
  metrics: GraphDataView<'float32x4'>;
  /** Per viewpoint: mean depth, relative asymmetry, real relative asymmetry, Hillier integration. */
  integration: GraphDataView<'float32x4'>;
  /** Per viewpoint: pyramid, Teklenburg, Penn integration and maximum observed hop depth. */
  extendedIntegration: GraphDataView<'float32x4'>;
  /** 2 * undirected betweenness / ((nodeCount - 1) * (nodeCount - 2)); NaN below three nodes. */
  normalizedChoice: GraphDataView<'float32'>;
  status: GPUNetworkVisibilityGraphCSRStatus;
};
type NetworkVisibilityDestinations = Partial<Omit<GPUNetworkVisibilityGraphCSROutput, 'status'>> & {
  status?: Partial<GPUNetworkVisibilityGraphCSRStatus>;
};
export type GPUNetworkVisibilityGraphCSRProps = Omit<GPUVisibilityGraphAdjacencyProps, 'output'> & {
  /** Caller-selected pivots; rows must be distinct valid node IDs. Contents may change each run. */
  pivotNodes?: GraphDataView<'uint32'>;
  /**
   * Defaults to all viewpoints when lanes, binding bytes and commands fit; otherwise the largest
   * feasible sample at most 32 (or laneCount when larger).
   * Explicit counts and pivotNodes never silently fall back. Used only without pivotNodes.
   */
  pivotCount?: number;
  /** 'exact' requires every viewpoint and rejects runs exceeding the command budget. Default 'auto'. */
  metricsMode?: 'auto' | 'exact';
  /** Must divide pivotCount, avoiding a padded final batch. */
  laneCount?: number;
  maximumIterations?: number;
  localIterations?: number;
  output?: NetworkVisibilityDestinations;
};

/**
 * Visibility with sampled network metrics in mode 'csr'; the dense reference delegates to the
 * spatial recipe. Import from @luma-contributors/analysis/gpu-network for CSR centrality.
 * The gpu-spatial-analysis entry builds adjacency and degree only in mode 'csr'.
 *
 * Converged all-source metrics agree with the dense reference within 1e-4 relative error for
 * finite values on at most 256 viewpoints (hop counts and sums there are exact). CSR has no quadratic
 * tables: frontier scratch is laneCount * nodeCount and reused between batches.
 *
 * CSR recipes are limited to 32,768 estimated commands in total; every storage binding must fit
 * the device's binding and allocation limits. When automatic all-source lanes, bytes or commands
 * cannot fit, systematically spaced pivots
 * estimate every viewpoint's integration: sampled reachable counts and hop sums, excluding self,
 * are multiplied by (nodeCount - 1) / (pivotCount - sampledSelf). Betweenness is scaled by
 * nodeCount / pivotCount. These deterministic estimates have no distribution-free error bound;
 * disconnected or spatially ordered inputs can be biased. Maximum depth is an observed lower
 * bound, so extended integration is also approximate. status.approximation is 1. Caller pivots
 * follow the same estimator. With no sampled sources other than self, reachable and depth are
 * estimated as zero and integration is NaN. Closeness and reachedCounts retain one row per pivot. Convergence
 * is independent of sampling: an exhausted frontier budget publishes incumbents and sets status.
 * Larger hop sums accumulate in float32; ill-conditioned integration ratios have no relative
 * tolerance guarantee. The default round budget covers all 256 reference viewpoints; explicit
 * smaller maximumIterations can publish partial results. Input shapes, options and output layouts
 * are compile-time; view contents and the CSR radius parameter are per-frame.
 */
export function addVisibilityGraphRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUNetworkVisibilityGraphCSRProps
): ReturnType<typeof addVisibilityGraphNetworkCSRRecipe>;
export function addVisibilityGraphRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUVisibilityGraphProps
): GPUVisibilityGraphReferenceRecipeResult;
export function addVisibilityGraphRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUVisibilityGraphProps | GPUNetworkVisibilityGraphCSRProps
): ReturnType<typeof addVisibilityGraphNetworkCSRRecipe> | GPUVisibilityGraphReferenceRecipeResult;
export function addVisibilityGraphRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUVisibilityGraphProps | GPUNetworkVisibilityGraphCSRProps
) {
  return props.mode === 'csr'
    ? addVisibilityGraphNetworkCSRRecipe(graph, props)
    : addSpatialVisibilityGraphRecipe(graph, props);
}

/** Preflight available through the existing public recipe export, without declaring graph work. */
addVisibilityGraphRecipe.getExecutionPlan = getNetworkVisibilityGraphExecutionPlan;

/** Catalog adapter: keeps the existing descriptive string on the ordinary plan unchanged. */
addVisibilityGraphRecipe.executionPlanner = {
  getExecutionPlan<Parameters>(
    props: GPUNetworkVisibilityGraphCSRProps,
    graph: GPUCommandGraph<Parameters>
  ) {
    const plan = getNetworkVisibilityGraphExecutionPlan(graph, props);
    return {...plan, accuracyImplications: [plan.accuracyImplications]};
  }
} satisfies GPUAnalysisExecutionPlanner<GPUNetworkVisibilityGraphCSRProps, GPUCommandGraph>;

/** Catalog-compatible top-level identity for the existing CSR recipe planner. */
export const GPU_NETWORK_VISIBILITY_GRAPH_PLANNER = addVisibilityGraphRecipe.executionPlanner;

/** Side-effect-free CSR metric preflight, shared with recipe construction. */
export function getNetworkVisibilityGraphExecutionPlan<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUNetworkVisibilityGraphCSRProps
): GPUNetworkVisibilityGraphExecutionPlan {
  validateRequirement(
    props.id ?? 'visibility-graph',
    props.mode === 'csr',
    `mode must be 'csr' for network visibility preflight; received ${props.mode}`
  );
  return prepareNetworkVisibilityGraphRecipe(graph, props).plan;
}

/** Composed adjacency and network metric binding requirements, with the total recipe budget. */
export type GPUNetworkVisibilityGraphExecutionPlan = Readonly<{
  route: 'exact' | 'sampled';
  feasible: boolean;
  pivotCount: number;
  laneCount: number;
  maximumIterations: number;
  localIterations: number;
  /** Systematic pivot row i is floor(i * positions.length / pivotCount); caller views remain resident. */
  pivotSelection: 'systematic' | 'caller';
  accuracyImplications: string;
  bindingBytes: Readonly<Record<string, number>>;
  /** Conservative command budget for the entire adjacency + centrality + integration recipe. */
  estimatedCommandNodeCount: number;
  constraints: readonly (GPUAnalysisRoute & {received: number; satisfied: boolean})[];
}>;

function prepareNetworkVisibilityGraphRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUNetworkVisibilityGraphCSRProps
) {
  const hasRemovedOutputs = 'outputs' in props;
  props = snapshotConstructionFields(props, [
    'id',
    'mode',
    'positions',
    'segments',
    'spatialContext',
    'maximumDistance',
    'tolerance',
    'parameters',
    'edgeCapacity',
    'maximumGridVisits',
    'queryChunkSize',
    'region',
    'output',
    'pivotNodes',
    'pivotCount',
    'metricsMode',
    'laneCount',
    'maximumIterations',
    'localIterations'
  ]);
  const id = props.id ?? 'visibility-graph';
  validateRequirement(id, !hasRemovedOutputs, 'outputs was removed; use output');
  const destinationFields = [
    'offsets',
    'neighbors',
    'weights',
    'degree',
    'gridVisits',
    'viewpointClassifications',
    'betweenness',
    'closeness',
    'reachedCounts',
    'metrics',
    'integration',
    'extendedIntegration',
    'normalizedChoice',
    'status'
  ] as const;
  const statusFields = [
    'count',
    'requiredCount',
    'overflow',
    'candidateOverflow',
    'invalidCount',
    'approximation',
    'converged',
    'iterationLimitReached',
    'zeroWeightCycle'
  ] as const;
  const captured = props.output && snapshotConstructionFields(props.output, destinationFields);
  const output = captured && {
    ...captured,
    status: captured.status && snapshotConstructionFields(captured.status, statusFields)
  };
  // This plan validates spatial props, caller ranges and all adjacency scratch without mutation.
  const adjacencyPlan = prepareVisibilityGraphCSRRecipe(graph, {...props, output});
  const count = props.positions.length;
  const maximumIterations =
    props.maximumIterations ?? Math.max(64, Math.min(count, VISIBILITY_GRAPH_MAXIMUM_POINT_COUNT));
  const localIterations = props.localIterations ?? 16;
  const roundCount = Math.min(maximumIterations, count);
  const phaseLayout = getFrontierPhaseLayout(roundCount);
  const queryChunkSize = alignElementCount(props.queryChunkSize ?? count);
  const queryChunkCount = Math.ceil(count / queryChunkSize);
  const queryChunkStride = alignElementCount(queryChunkSize + 4);
  const leafCapacity = 2 ** Math.max(1, Math.ceil(Math.log2(props.segments.length)));
  // Match the adjacency preparer's independently validated packed binding layouts. Shared
  // layout helpers supply grid padding; BVH uses a complete binary tree with padded leaves.
  const adjacencyBindingBytes = {
    'adjacency point grid': getUrbanPointGridShape(count).byteLength,
    'adjacency CSR offsets': (count + 1) * 4,
    'adjacency CSR neighbors and weights': props.edgeCapacity * 4,
    'adjacency count scan uint64 scratch': count * 8,
    'adjacency query records': count * 12,
    'adjacency query parameters and offsets': queryChunkCount * queryChunkStride * 4,
    'adjacency prepared positions': count * 8,
    'adjacency viewpoint classification scratch': count * 4,
    'adjacency segment bounds': props.segments.length * 8,
    'adjacency BVH node bounds': (2 * leafCapacity - 1) * 8,
    'adjacency BVH leaf rows': leafCapacity * 4
  };
  const bindingLimit = Math.min(
    graph.device.limits.maxStorageBufferBindingSize,
    graph.device.limits.maxBufferSize
  );
  const getLaneCount = (sources: number) => {
    let lanes = Math.min(
      sources,
      count <= VISIBILITY_GRAPH_MAXIMUM_POINT_COUNT
        ? sources
        : VISIBILITY_GRAPH_NETWORK_DEFAULT_LANE_COUNT,
      Math.floor(Math.min(getCoreBindingElementLimit(12), Math.floor(bindingLimit / 12)) / count)
    );
    while (lanes > 1 && sources % lanes !== 0) lanes--;
    return props.laneCount ?? lanes;
  };
  const estimateCommands = (sources: number) =>
    adjacencyPlan.estimatedCommandNodeCount +
    VISIBILITY_GRAPH_NETWORK_COMMAND_NODE_ALLOWANCE +
    Math.ceil(sources / getLaneCount(sources)) *
      (VISIBILITY_GRAPH_NETWORK_COMMANDS_PER_ROUND * roundCount +
        VISIBILITY_GRAPH_NETWORK_BATCH_COMMAND_ALLOWANCE);
  validateRequirement(
    id,
    props.metricsMode === undefined ||
      props.metricsMode === 'auto' ||
      props.metricsMode === 'exact',
    "metricsMode must be 'auto' or 'exact'"
  );
  const getBindingBytes = (sources: number, lanes: number) => ({
    ...adjacencyBindingBytes,
    'centrality adjacency': (2 * (count + 1) + 4 * props.edgeCapacity) * 4,
    'centrality lane scratch': lanes * count * 12,
    'centrality frontier queues': lanes * count * 8,
    'centrality batch parameters':
      (sources / lanes) *
      Math.max(VISIBILITY_GRAPH_NETWORK_CONTROL_BYTE_ALLOWANCE, alignElementCount(lanes) * 4),
    'integration totals and outputs': count * 16,
    'pivot validation counts': count * 4,
    'pivot rows': sources * 4,
    'centrality round state':
      2 *
      phaseLayout.dispatchSlotCount *
      FRONTIER_DISPATCH_SLOT_WORDS *
      Uint32Array.BYTES_PER_ELEMENT
  });
  const getFailure = (sources: number): string | undefined => {
    if (!Number.isSafeInteger(sources) || sources < 1 || sources > count)
      return `pivotCount ${sources} must be between 1 and positions.length ${count}`;
    const lanes = getLaneCount(sources);
    if (!Number.isSafeInteger(lanes) || lanes < 1 || lanes > sources || sources % lanes !== 0)
      return `laneCount ${lanes} must be a positive divisor of pivotCount ${sources}`;
    if (lanes * count > getCoreBindingElementLimit(12))
      return `laneCount * nodeCount ${lanes * count} exceeds ${getCoreBindingElementLimit(12)}`;
    for (const [name, bytes] of Object.entries(getBindingBytes(sources, lanes))) {
      if (bytes > bindingLimit)
        return `${name} needs ${bytes} binding bytes; limit ${bindingLimit}`;
    }
    const commands = estimateCommands(sources);
    if (commands > GPU_MAXIMUM_COMMAND_NODE_COUNT)
      return `estimated command node count ${commands} exceeds ${GPU_MAXIMUM_COMMAND_NODE_COUNT}`;
    return undefined;
  };
  let defaultPivotCount = count;
  if (
    props.metricsMode !== 'exact' &&
    !props.pivotNodes &&
    props.pivotCount === undefined &&
    getFailure(count)
  ) {
    const minimumCandidate =
      props.laneCount !== undefined && Number.isSafeInteger(props.laneCount) && props.laneCount >= 1
        ? props.laneCount
        : 1;
    const preferredCount = Math.min(
      count,
      Math.max(VISIBILITY_GRAPH_CSR_DEFAULT_PIVOT_COUNT, minimumCandidate)
    );
    // Preserve the preferred sample when legal, otherwise take the largest feasible smaller
    // sample. Fixed lanes make every candidate a multiple; with inferred lanes, one always fits
    // when the fixed adjacency/round allocations and one batch fit.
    defaultPivotCount = 0;
    for (let candidate = preferredCount; candidate >= minimumCandidate; candidate--) {
      if (!getFailure(candidate)) {
        defaultPivotCount = candidate;
        break;
      }
    }
    validateRequirement(
      id,
      defaultPivotCount > 0,
      `automatic visibility smallest candidate pivotCount ${minimumCandidate} failed: ${getFailure(minimumCandidate)}; reduce laneCount, maximumIterations or input size`
    );
  }
  const pivotCount = props.pivotNodes?.length ?? props.pivotCount ?? defaultPivotCount;
  validateCapacity(
    id,
    'pivotCount',
    pivotCount,
    1,
    Math.min(count, VISIBILITY_GRAPH_CSR_MAXIMUM_PIVOT_COUNT)
  );
  validateRequirement(
    id,
    props.metricsMode !== 'exact' || pivotCount === count,
    "metricsMode 'exact' requires pivotCount or pivotNodes.length to equal positions.length"
  );
  const laneCount = getLaneCount(pivotCount);
  validateCapacity(id, 'laneCount', laneCount, 1, pivotCount);
  validateRequirement(
    id,
    pivotCount % laneCount === 0,
    'laneCount must divide pivotCount so every centrality batch has its own unpadded lane count'
  );
  validateMaximumCount(
    id,
    'laneCount * nodeCount',
    laneCount * count,
    getCoreBindingElementLimit(12)
  );
  validateCapacity(
    id,
    'maximumIterations',
    maximumIterations,
    1,
    GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS
  );
  validateCapacity(
    id,
    'localIterations',
    localIterations,
    1,
    GPU_NETWORK_REACHABILITY_MAXIMUM_LOCAL_ITERATIONS
  );
  if (props.pivotNodes) validateSinglePackedView(props.pivotNodes, ['uint32'], `${id} pivotNodes`);
  validateVisibilityGraphCSRBindings(
    graph,
    id,
    {
      positions: props.positions,
      segments: props.segments,
      parameters: props.parameters,
      pivotNodes: props.pivotNodes
    },
    output,
    true
  );
  for (const [name, format, length] of [
    ['betweenness', 'float32', count],
    ['closeness', 'float32', pivotCount],
    ['reachedCounts', 'uint32', pivotCount],
    ['metrics', 'float32x4', count],
    ['integration', 'float32x4', count],
    ['extendedIntegration', 'float32x4', count],
    ['normalizedChoice', 'float32', count]
  ] as const) {
    const view = output?.[name];
    if (view) {
      validateSinglePackedView(view, [format], `${id} output.${name}`);
      validateViewLength(id, `output.${name}`, view, length);
    }
  }
  for (const name of [
    'approximation',
    'converged',
    'iterationLimitReached',
    'zeroWeightCycle'
  ] as const) {
    const view = output?.status?.[name];
    if (view) {
      validateSinglePackedView(
        view,
        ['uint32'],
        {id, contributor: 'addVisibilityGraphRecipe'},
        `output.status.${name}`
      );
      validateViewLength(id, `output.status.${name}`, view, 1);
    }
  }
  const bindingBytes = getBindingBytes(pivotCount, laneCount);
  for (const [name, bytes] of Object.entries(bindingBytes))
    validateUrbanStorageByteLength(
      graph.device,
      id,
      name,
      bytes,
      'reduce the input length, capacity or laneCount'
    );
  const estimatedCommandNodeCount = estimateCommands(pivotCount);
  validateCommandNodeBudget(id, estimatedCommandNodeCount);
  const accuracyImplications =
    pivotCount === count
      ? 'All-source metrics; convergence and adjacency completeness are separate runtime statuses.'
      : 'Deterministic sampled estimates without a distribution-free error bound; maximum depth is a lower bound.';
  const source = 'modules/analysis/src/gpu-network/visibility-graph/visibility-graph.ts';
  const constraints: GPUNetworkVisibilityGraphExecutionPlan['constraints'] = [
    ...Object.entries(bindingBytes).map(([name, received]) => ({
      name,
      kind: 'allocation' as const,
      quantity: name,
      unit: 'bytes' as const,
      appliesWhen: "mode === 'csr'",
      maximum: bindingLimit,
      scope: 'binding' as const,
      alternative: 'reduce laneCount, pivotCount, maximumIterations or input size',
      accuracyImplications,
      source,
      received,
      satisfied: received <= bindingLimit
    })),
    {
      name: 'recipe commands',
      kind: 'hard-limit',
      quantity: 'total recipe command nodes',
      unit: 'commands',
      appliesWhen: "mode === 'csr'",
      maximum: GPU_MAXIMUM_COMMAND_NODE_COUNT,
      scope: 'domain',
      alternative: 'sampled pivots or fewer maximumIterations',
      accuracyImplications,
      source,
      received: estimatedCommandNodeCount,
      satisfied: true
    }
  ];
  const plan: GPUNetworkVisibilityGraphExecutionPlan = {
    route: pivotCount === count ? 'exact' : 'sampled',
    feasible: true,
    pivotCount,
    laneCount,
    maximumIterations,
    localIterations,
    pivotSelection: props.pivotNodes ? 'caller' : 'systematic',
    accuracyImplications,
    bindingBytes,
    estimatedCommandNodeCount,
    constraints
  };
  return {adjacencyPlan, plan, output, props};
}

function addVisibilityGraphNetworkCSRRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUNetworkVisibilityGraphCSRProps
) {
  const prepared = prepareNetworkVisibilityGraphRecipe(graph, props);
  props = prepared.props;
  const id = props.id ?? 'visibility-graph';
  const {adjacencyPlan, plan, output} = prepared;
  const {pivotCount, laneCount, maximumIterations, localIterations, estimatedCommandNodeCount} =
    plan;
  const count = props.positions.length;
  // All caller validation and derived scratch checks have completed before declarations begin.
  // Core kernels honor these validated aligned ranges, including disjoint views of one buffer.
  const getView = <Format extends 'uint32' | 'float32' | 'float32x4'>(
    name: string,
    format: Format,
    length: number,
    supplied?: GraphDataView<Format>
  ) => getRecipeView(graph, `${id}-csr-${name}`, format, length, supplied);
  const pivots = props.pivotNodes ?? getView('pivots', 'uint32', pivotCount);
  if (!props.pivotNodes)
    graph.add(
      createVisibilityGraphPivotsNode(graph, {
        id: `${id}-csr-default-pivots`,
        operation: id,
        pointCount: count,
        pivots
      })
    );
  const invalidPivotRows = getView('invalid-pivot-rows', 'uint32', pivotCount);
  const invalidPivotCount = getView('invalid-pivot-count', 'uint32', 1);
  const pivotOccurrences = getView('pivot-occurrences', 'uint32', count);
  graph.add(
    createVisibilityGraphKernelNode(graph, {
      id: `${id}-csr-clear-pivot-occurrences`,
      operation: id,
      invocationCount: count,
      bindings: [{name: 'occurrences', view: pivotOccurrences, type: 'u32', access: 'read_write'}],
      body: 'occurrences[occurrencesOffset+index]=0u;'
    })
  );
  graph.add(
    createVisibilityGraphKernelNode(graph, {
      id: `${id}-csr-count-pivots`,
      operation: id,
      invocationCount: pivotCount,
      bindings: [
        {name: 'pivots', view: pivots, type: 'u32', access: 'read'},
        {name: 'occurrences', view: pivotOccurrences, type: 'atomic<u32>', access: 'read_write'}
      ],
      body: `let pivot=pivots[pivotsOffset+index];if(pivot<${count}u){atomicAdd(&occurrences[occurrencesOffset+pivot],1u);}`
    })
  );
  graph.add(
    createVisibilityGraphKernelNode(graph, {
      id: `${id}-csr-validate-pivots`,
      operation: id,
      invocationCount: pivotCount,
      bindings: [
        {name: 'pivots', view: pivots, type: 'u32', access: 'read'},
        {name: 'occurrences', view: pivotOccurrences, type: 'u32', access: 'read'},
        {name: 'flags', view: invalidPivotRows, type: 'u32', access: 'read_write'}
      ],
      body: `let pivot=pivots[pivotsOffset+index];var invalid=true;
if(pivot<${count}u){invalid=occurrences[occurrencesOffset+pivot]!=1u;}
flags[flagsOffset+index]=u32(invalid);`
    })
  );
  graph.add(
    new GPUReduction({
      id: `${id}-csr-invalid-pivots`,
      input: invalidPivotRows,
      output: invalidPivotCount,
      operation: 'sum'
    }).getCommandNodes(graph)
  );
  const adjacency = adjacencyPlan.addNodes(invalidPivotCount);
  const topology = adjacency.topology;
  const betweenness = getView('betweenness', 'float32', count, output?.betweenness);
  const closeness = getView('closeness', 'float32', pivotCount, output?.closeness);
  const reachedCounts = getView('reached-counts', 'uint32', pivotCount, output?.reachedCounts);
  // Generic hosts observe every returned output. Give the newly exposed terminal metric fields
  // copy capability at allocation, so snapshot taps do not require host-specific retention lists.
  // Caller-owned views keep their declared usage; STORAGE-only outputs remain valid for compute.
  const getMetricView = <Format extends 'float32' | 'float32x4'>(
    name: string,
    format: Format,
    supplied?: GraphDataView<Format>
  ) =>
    supplied
      ? getView(name, format, count, supplied)
      : createTransientView(
          graph,
          `${id}-csr-${name}`,
          format,
          count,
          Buffer.STORAGE | Buffer.COPY_SRC
        );
  const metrics = getMetricView('metrics', 'float32x4', output?.metrics);
  const integration = getMetricView('integration', 'float32x4', output?.integration);
  const extendedIntegration = getMetricView(
    'extended-integration',
    'float32x4',
    output?.extendedIntegration
  );
  const normalizedChoice = getMetricView('normalized-choice', 'float32', output?.normalizedChoice);
  const scalar = (
    name: 'approximation' | 'converged' | 'iterationLimitReached' | 'zeroWeightCycle'
  ) => getView(name, 'uint32', 1, output?.status?.[name]);
  const status: GPUNetworkVisibilityGraphCSRStatus = {
    ...adjacency.status,
    approximation: scalar('approximation'),
    converged: scalar('converged'),
    iterationLimitReached: scalar('iterationLimitReached'),
    zeroWeightCycle: scalar('zeroWeightCycle')
  };
  const centrality = new GPUNetworkCentrality({
    id: `${id}-csr-centrality`,
    topology,
    reverseTopology: topology,
    pivotNodes: pivots,
    laneCount,
    maximumIterations,
    localIterations,
    betweennessScale: (0.5 * count) / pivotCount,
    output: {
      betweenness,
      closeness,
      reachedCounts,
      status: {
        converged: status.converged,
        iterationLimitReached: status.iterationLimitReached,
        zeroWeightCycle: status.zeroWeightCycle
      }
    }
  });
  const centralityEnd = addVisibilityGraphMetricNodes(graph, centrality.getCommandNodes(graph));
  addVisibilityGraphIntegration(graph, {
    id: `${id}-csr-integration`,
    after: centralityEnd,
    topology,
    pivots,
    laneCount,
    maximumIterations,
    localIterations,
    degree: adjacency.outputs.degree,
    metrics,
    integration,
    extendedIntegration,
    converged: status.converged,
    iterationLimitReached: status.iterationLimitReached
  });
  graph.add(
    createVisibilityGraphKernelNode(graph, {
      id: `${id}-csr-normalized-choice`,
      operation: id,
      invocationCount: count,
      declarations: GET_NAN_WGSL,
      bindings: [
        {name: 'betweenness', view: betweenness, type: 'f32', access: 'read'},
        {name: 'normalized', view: normalizedChoice, type: 'f32', access: 'read_write'}
      ],
      body: `normalized[normalizedOffset+index]=select(getNaN(),2.0*betweenness[betweennessOffset+index]/(f32(${Math.max(1, count - 1)}u)*f32(${Math.max(1, count - 2)}u)),${count}>2);`
    })
  );
  // Never present metrics on the empty safety graph as measurements of a failed visibility graph.
  graph.add(
    createVisibilityGraphKernelNode(graph, {
      id: `${id}-csr-mask-metrics`,
      operation: id,
      declarations: GET_NAN_WGSL,
      invocationCount: Math.max(count, pivotCount),
      bindings: [
        {name: 'overflow', view: status.overflow, type: 'u32', access: 'read'},
        {name: 'candidateOverflow', view: status.candidateOverflow, type: 'u32', access: 'read'},
        {name: 'invalidCount', view: status.invalidCount, type: 'u32', access: 'read'},
        {name: 'betweenness', view: betweenness, type: 'f32', access: 'read_write'},
        {name: 'closeness', view: closeness, type: 'f32', access: 'read_write'},
        {name: 'reachedCounts', view: reachedCounts, type: 'u32', access: 'read_write'},
        {name: 'converged', view: status.converged, type: 'u32', access: 'read_write'}
      ],
      body: `if(overflow[overflowOffset]==0u&&candidateOverflow[candidateOverflowOffset]==0u&&invalidCount[invalidCountOffset]==0u){return;}
if(index<${count}u){betweenness[betweennessOffset+index]=getNaN();}
if(index<${pivotCount}u){closeness[closenessOffset+index]=getNaN();reachedCounts[reachedCountsOffset+index]=0u;}
if(index==0u){converged[convergedOffset]=0u;}`
    })
  );
  graph.add(
    createVisibilityGraphKernelNode(graph, {
      id: `${id}-csr-mask-integration`,
      operation: id,
      invocationCount: count,
      declarations: GET_NAN_WGSL,
      bindings: [
        {name: 'overflow', view: status.overflow, type: 'u32', access: 'read'},
        {name: 'candidateOverflow', view: status.candidateOverflow, type: 'u32', access: 'read'},
        {name: 'invalidCount', view: status.invalidCount, type: 'u32', access: 'read'},
        {name: 'metrics', view: metrics, type: 'f32', access: 'read_write'},
        {name: 'integration', view: integration, type: 'f32', access: 'read_write'},
        {name: 'extended', view: extendedIntegration, type: 'f32', access: 'read_write'},
        {name: 'normalized', view: normalizedChoice, type: 'f32', access: 'read_write'}
      ],
      body: `if(overflow[overflowOffset]==0u&&candidateOverflow[candidateOverflowOffset]==0u&&invalidCount[invalidCountOffset]==0u){return;}
normalized[normalizedOffset+index]=getNaN();for(var component=0u;component<4u;component++){
metrics[metricsOffset+4u*index+component]=getNaN();integration[integrationOffset+4u*index+component]=getNaN();extended[extendedOffset+4u*index+component]=getNaN();}`
    })
  );
  graph.add(
    createVisibilityGraphKernelNode(graph, {
      id: `${id}-csr-approximation`,
      operation: id,
      invocationCount: 1,
      bindings: [
        {name: 'approximation', view: status.approximation, type: 'u32', access: 'read_write'}
      ],
      body: `approximation[approximationOffset]=${Number(pivotCount < count)}u;`
    })
  );
  // Named so the emitted declaration references the imported type, not an extensionless path.
  const spatialContext: GPUSpatialContext = adjacency.provenance.spatialContext;
  return {
    ...adjacency,
    outputs: {
      ...adjacency.outputs,
      betweenness,
      closeness,
      reachedCounts,
      metrics,
      integration,
      extendedIntegration,
      normalizedChoice,
      status
    },
    status,
    pivotNodes: pivots,
    estimatedCommandNodeCount,
    provenance: {
      ...adjacency.provenance,
      spatialContext,
      pivotCount,
      laneCount,
      executionPlan: plan
    }
  };
}
