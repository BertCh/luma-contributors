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
import {alignElementCount} from '../network-view-utils';
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
  VISIBILITY_GRAPH_MAXIMUM_POINT_COUNT
} from '../../gpu-spatial-analysis/urban-morphology/visibility-graph';
import {
  prepareVisibilityGraphCSRRecipe,
  VISIBILITY_GRAPH_CSR_MAXIMUM_PIVOT_COUNT,
  VISIBILITY_GRAPH_CSR_DEFAULT_PIVOT_COUNT,
  type GPUVisibilityGraphCSRProps as GPUVisibilityGraphAdjacencyProps,
  type GPUVisibilityGraphCSROutput as GPUVisibilityGraphAdjacencyOutput,
  type GPUVisibilityGraphCSRStatus as GPUVisibilityGraphAdjacencyStatus
} from '../../gpu-spatial-analysis/urban-morphology/visibility-graph-csr';
import {validateUrbanStorageByteLength} from '../../gpu-spatial-analysis/urban-morphology/building-tessellation-validation';
import {validateVisibilityGraphCSRBindings} from '../../gpu-spatial-analysis/urban-morphology/visibility-graph-bindings';
import {getRecipeView} from '../../gpu-spatial-analysis/recipes/expansion-recipe-utils';
import {createVisibilityGraphKernelNode} from '../../gpu-spatial-analysis/urban-morphology/visibility-graph-kernel';

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

export type GPUVisibilityGraphCSRStatus = GPUVisibilityGraphAdjacencyStatus & {
  /** One when fewer than all viewpoints are pivots; closeness follows the listed pivot order. */
  approximation: GraphDataView<'uint32'>;
  converged: GraphDataView<'uint32'>;
  iterationLimitReached: GraphDataView<'uint32'>;
  zeroWeightCycle: GraphDataView<'uint32'>;
};
export type GPUVisibilityGraphCSROutput = Omit<GPUVisibilityGraphAdjacencyOutput, 'status'> & {
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
  status: GPUVisibilityGraphCSRStatus;
};
export type GPUVisibilityGraphCSRProps = Omit<GPUVisibilityGraphAdjacencyProps, 'output'> & {
  /** Caller-selected pivots; rows must be distinct valid node IDs. Contents may change each run. */
  pivotNodes?: GraphDataView<'uint32'>;
  /**
   * Defaults to all viewpoints when the composed command budget fits, otherwise min(32, count).
   * Explicit counts and pivotNodes never silently fall back. Used only without pivotNodes.
   */
  pivotCount?: number;
  /** 'exact' requires every viewpoint and rejects runs exceeding the command budget. Default 'auto'. */
  metricsMode?: 'auto' | 'exact';
  /** Must divide pivotCount, avoiding a padded final batch. */
  laneCount?: number;
  maximumIterations?: number;
  localIterations?: number;
  output?: Partial<Omit<GPUVisibilityGraphCSROutput, 'status'>> & {
    status?: Partial<GPUVisibilityGraphCSRStatus>;
  };
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
 * When an automatic all-source run exceeds the command budget, systematically spaced pivots
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
  props: GPUVisibilityGraphCSRProps
): ReturnType<typeof addVisibilityGraphNetworkCSRRecipe>;
export function addVisibilityGraphRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUVisibilityGraphProps
): ReturnType<typeof addSpatialVisibilityGraphRecipe>;
export function addVisibilityGraphRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUVisibilityGraphProps | GPUVisibilityGraphCSRProps
) {
  return props.mode === 'csr'
    ? addVisibilityGraphNetworkCSRRecipe(graph, props)
    : addSpatialVisibilityGraphRecipe(graph, props);
}

function addVisibilityGraphNetworkCSRRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUVisibilityGraphCSRProps
) {
  const id = props.id ?? 'visibility-graph';
  // This plan validates spatial props, caller ranges and all adjacency scratch without mutation.
  const adjacencyPlan = prepareVisibilityGraphCSRRecipe(graph, props);
  const count = props.positions.length;
  const maximumIterations =
    props.maximumIterations ?? Math.max(64, Math.min(count, VISIBILITY_GRAPH_MAXIMUM_POINT_COUNT));
  const localIterations = props.localIterations ?? 16;
  const roundCount = Math.min(maximumIterations, count);
  const getLaneCount = (sources: number) => {
    let lanes = Math.min(
      sources,
      count <= VISIBILITY_GRAPH_MAXIMUM_POINT_COUNT
        ? sources
        : VISIBILITY_GRAPH_NETWORK_DEFAULT_LANE_COUNT,
      Math.floor(getCoreBindingElementLimit(12) / count)
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
  const defaultPivotCount =
    props.metricsMode === 'exact' || estimateCommands(count) <= GPU_MAXIMUM_COMMAND_NODE_COUNT
      ? count
      : Math.min(count, VISIBILITY_GRAPH_CSR_DEFAULT_PIVOT_COUNT);
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
    props.output,
    true
  );
  const output = props.output;
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
  const batchCount = pivotCount / laneCount;
  for (const [name, bytes] of [
    ['centrality adjacency', (2 * (count + 1) + 4 * props.edgeCapacity) * 4],
    ['centrality lane scratch', laneCount * count * 12],
    ['centrality frontier queues', laneCount * count * 8],
    [
      'centrality batch parameters',
      batchCount *
        Math.max(VISIBILITY_GRAPH_NETWORK_CONTROL_BYTE_ALLOWANCE, alignElementCount(laneCount) * 4)
    ],
    ['integration totals and outputs', count * 16],
    ['pivot validation counts', count * 4],
    ['pivot rows', pivotCount * 4],
    ['centrality round state', (roundCount + 2) * VISIBILITY_GRAPH_NETWORK_CONTROL_BYTE_ALLOWANCE]
  ] as const)
    validateUrbanStorageByteLength(
      graph.device,
      id,
      name,
      bytes,
      'reduce the input length, capacity or laneCount'
    );
  const estimatedCommandNodeCount =
    adjacencyPlan.estimatedCommandNodeCount +
    VISIBILITY_GRAPH_NETWORK_COMMAND_NODE_ALLOWANCE +
    batchCount *
      (VISIBILITY_GRAPH_NETWORK_COMMANDS_PER_ROUND * roundCount +
        VISIBILITY_GRAPH_NETWORK_BATCH_COMMAND_ALLOWANCE);
  validateCommandNodeBudget(id, estimatedCommandNodeCount);
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
  const status: GPUVisibilityGraphCSRStatus = {
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
    provenance: {...adjacency.provenance, pivotCount, laneCount}
  };
}
