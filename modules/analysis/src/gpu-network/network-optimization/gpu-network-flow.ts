// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateNetworkOptimizationBindings,
  validateNetworkOptimizationInputOutputBuffers
} from './network-optimization-bindings';

import {getViewBindingRange} from '@luma.gl/graph/gpu-core';
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
  getGPUNetworkTopologyViews,
  type GPUNetworkAdjacency
} from '../../analysis-core/network-topology';
import {
  estimateParallelNetworkFlowNodes,
  getParallelNetworkFlowNodes,
  validateParallelNetworkFlow
} from './network-flow-parallel';
import {
  createNetworkOptimizationStatusNodes,
  getNetworkOptimizationStatusViews,
  validateNetworkOptimizationStatus
} from './network-optimization-status';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateSinglePackedView,
  validateRequirement,
  validateGPUStorageBindingByteLength
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Largest vertex count: the kernel keeps per-vertex distances in private arrays. */
const MAXIMUM_NODE_COUNT = 128;
/** Largest edge count: each augmentation relaxes every arc in the single invocation (O(V * E)). */
const MAXIMUM_EDGE_COUNT = 1024;
/** Largest `maximumAugmentations`: bounds the single invocation's augmenting-path loop. */
const MAXIMUM_AUGMENTATIONS = 65536;
/** Default `maximumAugmentations`. */
const DEFAULT_MAXIMUM_AUGMENTATIONS = 1024;
/** Rows of the `parameters` view: source, sink, required flow. */
const PARAMETER_COUNT = 3;
/** Words of the `summary` view, indexed by `GPU_NETWORK_FLOW_WORD`. */
const SUMMARY_WORD_COUNT = 8;

/** Summary words returned by the bounded integral residual-flow solver. */
export const GPU_NETWORK_FLOW_WORD = {
  valid: 0,
  complete: 1,
  value: 2,
  unserved: 3,
  augmentations: 4,
  cost: 5,
  cutCapacity: 6,
  cutCertified: 7
} as const;

/**
 * Directed integral flow, parallel/antiparallel edges preserved in input order.
 * All routes preserve feasibility after every committed batch. Path tie-breaking and batch
 * counts can differ; maximum value and integral minimum cost agree on convergence.
 */
export type GPUNetworkFlowOutput = {
  flow: GraphDataView<'uint32'>;
  /** Residual source-reachable vertices, one word per node. An unfinished forward cut search
   * publishes its reached subset and leaves cutCertified zero. */
  cut: GraphDataView<'uint32'>;
  /** Eight words, indexed by GPU_NETWORK_FLOW_WORD; cost is f32 bits. */
  summary: GraphDataView<'uint32'>;
  /** Optional exact unsigned integer objective [low32,high32]. Summary cost remains f32. */
  exactCost?: GraphDataView<'uint32x2'>;
  /** invalidCount is 1 for a rejected problem; iterationCount counts feasible batches (one path for successive-shortest-path/reference routes).
   * Convergence includes a fully solved infeasible finite request; inspect summary.unserved.
   * Invalid problems set neither converged nor iterationLimitReached.
   */
  status?: GPUInvalidStatusPort & GPUIterationStatusPort;
};

export type GPUNetworkFlowProps = {
  id?: string;
  nodeCount: number;
  edges: GraphDataView<'uint32x2'>;
  capacities: GraphDataView<'uint32'>;
  /** Nonnegative integer original-edge costs. Omit for maximum flow. */
  costs?: GraphDataView<'uint32'>;
  /** Optional shared CSR in original-edge order. The parallel route validates its connectivity
   * against edges on the GPU, then builds a residual CSR containing both arc directions.
   * Omit to build the residual CSR directly from the existing edge list. Weights are ignored.
   */
  topology?: GPUNetworkAdjacency;
  /** auto (default) selects parallel above the reference limits or with topology, mode or maximumCostPhases.
   * reference preserves the original exact small-graph solver; parallel explicitly opts in.
   */
  solver?: 'auto' | 'reference' | 'parallel';
  /** Costed parallel route. Defaults to successive-shortest-path, preserving the existing
   * maximumAugmentations budget. Explicit primal-dual batches shortest admissible paths.
   * Setting maximumCostPhases with mode omitted also opts into primal-dual.
   * Requires costs and solver parallel (auto selects parallel when mode is supplied). */
  mode?: 'primal-dual' | 'successive-shortest-path';
  /** Primal-dual route: maximum feasible shortest-path preflow batches (default 3, maximum 4096,
   * additionally bounded by the command budget). One batch may send many paths. When mode is
   * omitted, supplying this prop opts into primal-dual; incompatible with successive-shortest-path. */
  maximumCostPhases?: number;
  /** Source, sink, required flow; 0xffffffff required flow means maximize. */
  parameters: GraphDataView<'uint32'>;
  /** Default: 1024 for reference, 8 for successive-shortest-path mode. Primal-dual and uncosted
   * parallel ignore this prop and use maximumCostPhases and maximumRelabels respectively. */
  maximumAugmentations?: number;
  /** Parallel route only: BFS or Bellman-Ford sweeps per search, including the final unchanged
   * sweep; also bounds forward/return discharge rounds in each preflow phase.
   * Defaults to min(nodeCount, 1024). An unfinished solve publishes the
   * already feasible flow with complete=0, converged=0 and iterationLimitReached=1, unless the
   * finite request was already optimally served (a cut search cannot revoke that convergence).
   */
  maximumRelaxationRounds?: number;
  /** Uncosted parallel route: maximum global BFS relabel phases (default 3, maximum 4096,
   * additionally bounded by the shared command budget). Each phase fixes global BFS heights,
   * runs synchronous vertex discharges, then returns stranded excess along the phase DAG
   * before committing a feasible batch. There is a final relabel and source-side cut search.
   * maximumAugmentations is used only by reference and successive-shortest-path solvers. */
  maximumRelabels?: number;
  output: GPUNetworkFlowOutput;
};

/**
 * Integral flow with explicit reverse arcs. Computation stays on the graph; no submission/readback.
 * Each committed batch preserves feasibility, including on budget
 * exhaustion. The parallel max-flow route uses global BFS relabelling and parallel preflow
 * pushes with bounded excess returns before committing feasible batches. Heights stay fixed
 * within each phase; all relabelling is global between phases. The costed primal-dual route uses
 * reverse Bellman-Ford labels (integer cost, then hop count), and pushes only along tight arcs
 * decreasing that immutable hop count. Every committed batch is a sum of shortest residual
 * paths, preserving integral minimum cost at its value; zero-cost cycles cannot enter the DAG.
 * Excess returns undo only that batch's pushes. Recomputing labels updates the dual between
 * batches. Equal-cost/equal-hop paths are sent together. Graphs with many distinct shortest-path
 * costs or hop lengths can still exhaust maximumCostPhases; this is a bounded solver.
 * Explicit mode successive-shortest-path retains parallel pointer doubling for one-path updates.
 * Total capacities must be <= 0x00ffffff for reference or 0xffffffff for parallel; costs <= 32767.
 * Invalid input has valid=0.
 * Budget exhaustion has complete=0 unless a finite request was already optimally served.
 *
 * Reference route (contributor contract rule 2): the solver is one GPU invocation
 * (`invocationCount: 1`) and every augmentation is a Bellman-Ford pass over all arcs, O(V * E).
 * The maximum size is enforced at construction: 128 nodes, 1,024 edges and 65,536 augmentations
 * (default 1,024), about 8.6e9 serial operations at the limits. Auto retains reference behaviour
 * for existing small inputs; larger inputs use parallel (3 uncosted relabel phases or 8 costed
 * augmentations). Primal-dual requires an explicit mode or maximumCostPhases; its default is 3 phases.
 * The 100k-vertex fan with 99,998 unit-capacity paths completes in one preflow phase; larger
 * diameters/cut structures require explicit budgets. The full source-side cut uses a final BFS. The parallel route accepts up
 * to 30 million vertices/edges subject to device binding sizes and the command-node budget.
 * Its signed distance bound requires costs <= min(32767, floor(2147483646 / nodeCount)); invalid
 * contents set valid=0. Costs and total capacities remain integral; objective arithmetic is exact
 * in exactCost; summary.cost uses the shared uint64-to-f32 conversion (within two f32 spacings).
 * Per-edge flows can differ among equal-cost optima between routes.
 * Compile-time: node/edge counts, topology, solver, optional cost/exact-cost/status views,
 * mode, maximumCostPhases, maximumAugmentations, maximumRelabels and maximumRelaxationRounds.
 * Per-frame: edge, capacity, cost and parameter contents. Every encoding rewrites the outputs.
 */
export class GPUNetworkFlow implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUNetworkFlowProps;
  readonly solver: 'reference' | 'parallel';
  readonly estimatedCommandNodeCount: number;
  constructor(props: GPUNetworkFlowProps) {
    this.props = props;
    this.id = props.id ?? 'network-flow';
    const {id} = this;
    validateRequirement(
      id,
      props.solver === undefined || ['auto', 'reference', 'parallel'].includes(props.solver),
      'solver must be auto, reference or parallel'
    );
    validateRequirement(
      id,
      props.mode === undefined || ['primal-dual', 'successive-shortest-path'].includes(props.mode),
      'mode must be primal-dual or successive-shortest-path'
    );
    validateRequirement(
      id,
      props.mode === undefined || Boolean(props.costs),
      'mode requires costs'
    );
    this.solver =
      props.solver === 'parallel' ||
      ((props.solver ?? 'auto') === 'auto' &&
        (props.mode !== undefined ||
          props.maximumCostPhases !== undefined ||
          props.topology !== undefined ||
          props.nodeCount > MAXIMUM_NODE_COUNT ||
          props.edges.length > MAXIMUM_EDGE_COUNT))
        ? 'parallel'
        : 'reference';
    validateCapacity(
      id,
      'maximumAugmentations',
      props.maximumAugmentations ?? DEFAULT_MAXIMUM_AUGMENTATIONS,
      0,
      MAXIMUM_AUGMENTATIONS
    );
    this.estimatedCommandNodeCount =
      this.solver === 'parallel'
        ? estimateParallelNetworkFlowNodes(props)
        : 1 + Number(getNetworkOptimizationStatusViews(props.output.status).length > 0);
    if (this.solver === 'parallel') {
      validateParallelNetworkFlow(id, props, this.estimatedCommandNodeCount);
    } else {
      validateCapacity(id, 'nodeCount', props.nodeCount, 2, MAXIMUM_NODE_COUNT);
      validateCapacity(id, 'edges rows', props.edges.length, 0, MAXIMUM_EDGE_COUNT);
      validateRequirement(
        id,
        props.maximumRelaxationRounds === undefined &&
          props.maximumRelabels === undefined &&
          props.mode === undefined &&
          props.maximumCostPhases === undefined,
        'maximumRelaxationRounds, maximumRelabels, mode and maximumCostPhases require solver parallel'
      );
    }
    validateSinglePackedView(props.edges, ['uint32x2'], this.id);
    for (const view of [
      props.capacities,
      props.costs,
      props.parameters,
      props.output.flow,
      props.output.cut,
      props.output.summary
    ]) {
      if (view) validateSinglePackedView(view, ['uint32'], this.id);
    }
    validateRequirement(
      id,
      props.capacities.length === props.edges.length,
      'capacities must hold one row per edge'
    );
    validateRequirement(
      id,
      !props.costs || props.costs.length === props.edges.length,
      'costs must hold one row per edge'
    );
    validateRequirement(
      id,
      props.parameters.length === PARAMETER_COUNT,
      `parameters must hold ${PARAMETER_COUNT} rows: source, sink, required flow`
    );
    validateRequirement(
      id,
      props.output.flow.length === props.edges.length,
      'output.flow must hold one row per edge'
    );
    validateRequirement(
      id,
      props.output.cut.length === props.nodeCount,
      'output.cut must hold nodeCount rows'
    );
    validateRequirement(
      id,
      props.output.summary.length === SUMMARY_WORD_COUNT,
      `output.summary must hold ${SUMMARY_WORD_COUNT} rows`
    );
    if (props.output.exactCost) {
      validateSinglePackedView(props.output.exactCost, ['uint32x2'], this.id);
      validateRequirement(
        id,
        props.output.exactCost.length === 1,
        'output.exactCost must hold one row'
      );
    }
    validateNetworkOptimizationStatus(
      id,
      [
        props.output.flow,
        props.output.cut,
        props.output.summary,
        ...(props.output.exactCost ? [props.output.exactCost] : [])
      ],
      props.output.status
    );
    validateNetworkOptimizationInputOutputBuffers(this.id, getFlowOutputViews(props.output), [
      props.edges,
      props.capacities,
      props.costs,
      props.parameters,
      ...(props.topology ? getGPUNetworkTopologyViews(props.topology) : [])
    ]);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    for (const [index, view] of [
      props.edges,
      props.capacities,
      props.costs,
      props.parameters,
      ...(props.topology ? getGPUNetworkTopologyViews(props.topology) : []),
      ...getFlowOutputViews(props.output)
    ].entries()) {
      if (view)
        validateGPUStorageBindingByteLength(
          graph.device,
          id,
          `view ${index} aligned binding range`,
          getViewBindingRange(view).size,
          'reduce the view length or its alignment prefix'
        );
    }
    validateNetworkOptimizationBindings(
      graph,
      id,
      Object.fromEntries(
        [
          props.edges,
          props.capacities,
          props.costs,
          props.parameters,
          ...(props.topology ? getGPUNetworkTopologyViews(props.topology) : []),
          ...getFlowOutputViews(props.output)
        ].map((view, index) => [`view ${index}`, view])
      )
    );
    const nodes =
      this.solver === 'parallel'
        ? getParallelNetworkFlowNodes(graph, id, props)
        : [
            createWGSLKernelNode<Parameters>(graph, {
              id,
              operation: 'GPUNetworkFlow',
              variant: props.costs ? 'minimum-cost' : 'maximum-flow',
              invocationCount: 1,
              workgroupSize: 2,
              bindings: [
                {name: 'edges', view: props.edges, type: 'u32', access: 'read'},
                {name: 'capacities', view: props.capacities, type: 'u32', access: 'read'},
                ...(props.costs
                  ? [
                      {
                        name: 'costs',
                        view: props.costs,
                        type: 'u32' as const,
                        access: 'read' as const
                      }
                    ]
                  : []),
                {name: 'parameters', view: props.parameters, type: 'u32', access: 'read'},
                {name: 'flow', view: props.output.flow, type: 'u32', access: 'read_write'},
                {name: 'cut', view: props.output.cut, type: 'u32', access: 'read_write'},
                {name: 'summary', view: props.output.summary, type: 'u32', access: 'read_write'},
                ...(props.output.exactCost
                  ? [
                      {
                        name: 'exactCost',
                        view: props.output.exactCost,
                        type: 'u32' as const,
                        access: 'read_write' as const
                      }
                    ]
                  : [])
              ],
              declarations: `const NODE_COUNT: u32 = ${props.nodeCount}u;
const EDGE_COUNT: u32 = ${props.edges.length}u;
const MAXIMUM_AUGMENTATIONS: u32 = ${props.maximumAugmentations ?? 1024}u;
fn edgeCost(edge: u32) -> i32 { return ${props.costs ? 'i32(costs[costsOffset + edge])' : '0'}; }`,
              body: `${props.output.exactCost ? 'exactCost[exactCostOffset] = 0u; exactCost[exactCostOffset + 1u] = 0u;' : ''}
for (var word = 0u; word < 8u; word++) { summary[summaryOffset + word] = 0u; }
for (var edge = 0u; edge < EDGE_COUNT; edge++) { flow[flowOffset + edge] = 0u; }
for (var vertex = 0u; vertex < NODE_COUNT; vertex++) { cut[cutOffset + vertex] = 0u; }
let source = parameters[parametersOffset];
let destination = parameters[parametersOffset + 1u];
let required = parameters[parametersOffset + 2u];
if (source >= NODE_COUNT || destination >= NODE_COUNT || source == destination) { return; }
var totalCapacity = 0u;
for (var edge = 0u; edge < EDGE_COUNT; edge++) {
  let endpoints = vec2<u32>(edges[edgesOffset + 2u * edge], edges[edgesOffset + 2u * edge + 1u]);
  let capacity = capacities[capacitiesOffset + edge];
  if (any(endpoints >= vec2<u32>(NODE_COUNT)) || capacity > 0x00ffffffu - totalCapacity || edgeCost(edge) < 0 || edgeCost(edge) > 32767) { return; }
  totalCapacity += capacity;
}
summary[summaryOffset] = 1u;
var value = 0u;
var augmentations = 0u;
var complete = false;
var exhausted = false;
var distances: array<i32, ${props.nodeCount}>;
var predecessor: array<u32, ${props.nodeCount}>;
var reversed: array<bool, ${props.nodeCount}>;
loop {
  for (var vertex = 0u; vertex < NODE_COUNT; vertex++) {
    distances[vertex] = 0x3fffffff;
    predecessor[vertex] = 0xffffffffu;
  }
  distances[source] = 0;
  for (var round = 1u; round < NODE_COUNT; round++) {
    var changed = false;
    for (var edge = 0u; edge < EDGE_COUNT; edge++) {
      let endpoints = vec2<u32>(edges[edgesOffset + 2u * edge], edges[edgesOffset + 2u * edge + 1u]);
      let edgeFlow = flow[flowOffset + edge];
      let cost = edgeCost(edge);
      if (edgeFlow < capacities[capacitiesOffset + edge] && distances[endpoints.x] != 0x3fffffff && distances[endpoints.x] + cost < distances[endpoints.y]) {
        distances[endpoints.y] = distances[endpoints.x] + cost;
        predecessor[endpoints.y] = edge;
        reversed[endpoints.y] = false;
        changed = true;
      }
      if (edgeFlow > 0u && distances[endpoints.y] != 0x3fffffff && distances[endpoints.y] - cost < distances[endpoints.x]) {
        distances[endpoints.x] = distances[endpoints.y] - cost;
        predecessor[endpoints.x] = edge;
        reversed[endpoints.x] = true;
        changed = true;
      }
    }
    if (!changed) { break; }
  }
  exhausted = distances[destination] == 0x3fffffff;
  if (exhausted || value == required) { complete = true; break; }
  if (augmentations == MAXIMUM_AUGMENTATIONS) { break; }
  var increment = min(0x00ffffffu, required - value);
  var cursor = destination;
  var hops = 0u;
  loop {
    if (cursor == source || hops == NODE_COUNT) { break; }
    let edge = predecessor[cursor];
    if (edge >= EDGE_COUNT) { summary[summaryOffset] = 0u; return; }
    let endpoints = vec2<u32>(edges[edgesOffset + 2u * edge], edges[edgesOffset + 2u * edge + 1u]);
    increment = min(increment, select(capacities[capacitiesOffset + edge] - flow[flowOffset + edge], flow[flowOffset + edge], reversed[cursor]));
    cursor = select(endpoints.x, endpoints.y, reversed[cursor]);
    hops++;
  }
  if (cursor != source || increment == 0u) { summary[summaryOffset] = 0u; return; }
  cursor = destination;
  loop {
    if (cursor == source) { break; }
    let edge = predecessor[cursor];
    let endpoints = vec2<u32>(edges[edgesOffset + 2u * edge], edges[edgesOffset + 2u * edge + 1u]);
    if (reversed[cursor]) { flow[flowOffset + edge] -= increment; }
    else { flow[flowOffset + edge] += increment; }
    cursor = select(endpoints.x, endpoints.y, reversed[cursor]);
  }
  value += increment;
  augmentations++;
}
var cost = 0.0; var exactLow = 0u; var exactHigh = 0u;
var cutCapacity = 0u;
for (var vertex = 0u; vertex < NODE_COUNT; vertex++) { cut[cutOffset + vertex] = select(0u, 1u, distances[vertex] != 0x3fffffff); }
for (var edge = 0u; edge < EDGE_COUNT; edge++) {
  let endpoints = vec2<u32>(edges[edgesOffset + 2u * edge], edges[edgesOffset + 2u * edge + 1u]);
  let amount = flow[flowOffset + edge]; let unitCost = u32(edgeCost(edge));
  cost += f32(amount) * f32(unitCost);
  let lowProduct = (amount & 0xffffu) * unitCost; let highProduct = (amount >> 16u) * unitCost;
  let productLow = lowProduct + (highProduct << 16u); let productHigh = (highProduct >> 16u) + select(0u, 1u, productLow < lowProduct);
  let previousLow = exactLow; exactLow += productLow; exactHigh += productHigh + select(0u, 1u, exactLow < previousLow);
  if (distances[endpoints.x] != 0x3fffffff && distances[endpoints.y] == 0x3fffffff) { cutCapacity += capacities[capacitiesOffset + edge]; }
}
${props.output.exactCost ? 'exactCost[exactCostOffset] = exactLow; exactCost[exactCostOffset + 1u] = exactHigh;' : ''}
summary[summaryOffset + 1u] = select(0u, 1u, complete);
summary[summaryOffset + 2u] = value;
summary[summaryOffset + 3u] = select(required - value, 0u, required == 0xffffffffu);
summary[summaryOffset + 4u] = augmentations;
summary[summaryOffset + 5u] = bitcast<u32>(cost);
summary[summaryOffset + 6u] = cutCapacity;
summary[summaryOffset + 7u] = select(0u, 1u, exhausted && cutCapacity == value);`
            })
          ];
    nodes.push(
      ...createNetworkOptimizationStatusNodes(
        graph,
        id,
        'GPUNetworkFlow',
        props.output.summary,
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

function getFlowOutputViews(output: GPUNetworkFlowProps['output']): GraphDataView[] {
  return [
    output.flow,
    output.cut,
    output.summary,
    ...(output.exactCost ? [output.exactCost] : []),
    ...getNetworkOptimizationStatusViews(output.status)
  ];
}
