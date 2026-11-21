// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  getViewBindingRange,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  getGPUNetworkTopologyViews,
  validateGPUNetworkTopology,
  type GPUNetworkTopology
} from '../../analysis-core/contracts/network-topology';
import {
  validateCapacity,
  validateRequirement,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateViewLength,
  getGraphOutputViews
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createFillNode,
  createWGSLDispatchGate,
  createWGSLKernelNode
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {
  GPUCandidateStatusPort,
  GPUIterationStatusPort
} from '../../analysis-core/contracts/status';
import {getBinarySearchWGSL} from '../../analysis-core/kernels/binary-search-wgsl';
import {
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {GPUNetworkReachability} from '../network-reachability/gpu-network-reachability';
import {getKShortestPathsWGSL} from './k-shortest-paths-wgsl';
import {gateKShortestPathSearchNodes} from './k-shortest-paths-search-gate';
import {getGPUKShortestPathsOutputLayout} from './k-shortest-paths-output';

// Eight setup/publication nodes, and fourteen per Yen round plus two spare nodes for the
// reachability composition. Node specs measure the emitted graph against this allowance.
const FIXED_COMMAND_NODE_COUNT = 8;
const ROUND_COMMAND_NODE_ALLOWANCE = 16;

/** Yen's accepted-path dimension; bounds prefix comparisons and the command graph. */
export const GPU_K_SHORTEST_PATHS_MAXIMUM_ROUTE_COUNT = 32;
/** Bounds reconstruction and root comparisons, independently of the number of CSR vertices. */
export const GPU_K_SHORTEST_PATHS_MAXIMUM_PATH_LENGTH = 1024;
/** Bounds the serial merge of Yen's persistent candidate set, not the CSR size. */
export const GPU_K_SHORTEST_PATHS_MAXIMUM_CANDIDATE_CAPACITY = 4096;
/** Matches the frontier and zero-cost tie phase limits of GPUNetworkReachability. */
export const GPU_K_SHORTEST_PATHS_MAXIMUM_ITERATIONS = 1024;

export type GPUKShortestPathsOutput = {
  /** query × routeCount × maximumPathLength, padded with 0xffffffff. */
  vertices: GraphDataView<'uint32'>;
  /** CSR edge IDs, same shape as vertices; the last vertex has no edge. */
  edges: GraphDataView<'uint32'>;
  /** query × routeCount; zero for an empty slot. */
  lengths: GraphDataView<'uint32'>;
  /** query × routeCount; infinity for an empty slot. */
  costs: GraphDataView<'float32'>;
  /** Accepted-route counts, one uint32 row per query (queryCount rows). */
  counts: GraphDataView<'uint32'>;
  /**
   * Required batch scalars: any interrupted query prevents convergence of the batch.
   * iterationCount is the maximum number of Yen rounds attempted by any query.
   */
  status: Required<GPUIterationStatusPort & GPUCandidateStatusPort>;
};

/** Compile-time: all lengths, capacities and budgets. Per-frame: all input view contents. */
export type GPUKShortestPathsProps = {
  id?: string;
  /** Directed CSR with nondecreasing offsets in [0, neighbors.length]. Invalid neighbors and nonfinite/negative costs are impassable. */
  topology: GPUNetworkTopology;
  sources: GraphDataView<'uint32'>;
  targets: GraphDataView<'uint32'>;
  routeCount: number;
  /** Maximum vertices in one returned route; default 64. An overlong shortest spur stops the query. */
  maximumPathLength?: number;
  /** Persistent Yen candidates per query; default 256. Exhaustion sets candidateOverflow. */
  candidateCapacity?: number;
  /** Rounds per shortest-path and zero-cost tie search; default 64, with 64 local hops per round. */
  maximumIterations?: number;
  output: GPUKShortestPathsOutput;
};

/**
 * Yen's k-shortest simple edge paths, driven entirely by GPU CSR shortest paths.
 * Each Yen round searches all deviations of the last accepted path concurrently with
 * GPUNetworkReachability. Root vertices and accepted next edges are removed in lane-local CSR
 * copies. The candidate pool persists between rounds; parallel edges remain distinct routes.
 *
 * There is no 256-vertex bound or graph-sized private array. With Q queries, P spur slots and
 * C candidates, scratch is O(Q * P * (V + E) + Q * (C + k + P) * maximumPathLength).
 * The device binding limits still apply, so reduce query batches or path length for large CSR.
 *
 * A path/shortest-search budget stops the affected query, retaining only its previously certified
 * routes, and sets iterationLimitReached. Candidate exhaustion sets candidateOverflow instead.
 * Converged means k paths were accepted or every remaining simple path was exhausted. Invalid
 * endpoints are empty converged queries; source == target yields the zero-edge path.
 * If any shortest-path lane fails to converge, all active queries stop conservatively.
 *
 * Equal-cost membership follows reachability's acyclic predecessor rule, then lexicographic CSR
 * edge IDs in the candidate pool; no global lexicographic order is promised for ties. Costs use
 * float32 relaxation, including its rounding and overflow semantics. No CPU readback is used.
 * Reference: Yen, Finding the K Shortest Loopless Paths in a Network, Management Science (1971).
 */
export class GPUKShortestPaths implements GPUCommandNodeProducer {
  readonly id: string;
  readonly nodeCount: number;
  readonly maximumPathLength: number;
  readonly candidateCapacity: number;
  readonly maximumIterations: number;
  readonly spurCount: number;
  /** Upper bound including one batch update and a predicate node per frontier round. */
  readonly estimatedCommandNodeCount: number;

  constructor(readonly props: GPUKShortestPathsProps) {
    this.id = props.id ?? 'k-shortest-paths';
    const {id} = this;
    this.nodeCount = validateGPUNetworkTopology(id, props.topology, {weights: true}).nodeCount;
    this.maximumPathLength = props.maximumPathLength ?? 64;
    this.candidateCapacity = props.candidateCapacity ?? 256;
    this.maximumIterations = props.maximumIterations ?? 64;
    validateCapacity(
      id,
      'routeCount',
      props.routeCount,
      1,
      GPU_K_SHORTEST_PATHS_MAXIMUM_ROUTE_COUNT
    );
    validateCapacity(
      id,
      'maximumPathLength',
      this.maximumPathLength,
      1,
      GPU_K_SHORTEST_PATHS_MAXIMUM_PATH_LENGTH
    );
    validateCapacity(
      id,
      'candidateCapacity',
      this.candidateCapacity,
      1,
      GPU_K_SHORTEST_PATHS_MAXIMUM_CANDIDATE_CAPACITY
    );
    validateCapacity(
      id,
      'maximumIterations',
      this.maximumIterations,
      1,
      GPU_K_SHORTEST_PATHS_MAXIMUM_ITERATIONS
    );
    this.spurCount =
      props.routeCount === 1
        ? 1
        : Math.max(1, Math.min(this.maximumPathLength - 1, this.nodeCount - 1));
    this.estimatedCommandNodeCount =
      props.sources.length === 0
        ? 4
        : FIXED_COMMAND_NODE_COUNT +
          props.routeCount * (ROUND_COMMAND_NODE_ALLOWANCE + 4 * this.maximumIterations);
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateSinglePackedView(props.sources, ['uint32'], `${id} sources`);
    validateSinglePackedView(props.targets, ['uint32'], `${id} targets`);
    validateCapacity(id, 'queryCount', props.sources.length, 0);
    validateViewLength(id, 'targets', props.targets, props.sources.length);
    const {status: statusLayout, ...columnLayout} = getGPUKShortestPathsOutputLayout({
      id,
      queryCount: props.sources.length,
      routeCount: props.routeCount,
      maximumPathLength: this.maximumPathLength
    });
    validateRequirement(
      id,
      props.output.status !== null && typeof props.output.status === 'object',
      'output.status must contain the required status views'
    );
    const outputByName = {...props.output, ...props.output.status};
    for (const [name, descriptor] of Object.entries({...columnLayout, ...statusLayout})) {
      const scalar = outputByName[name as keyof typeof outputByName] as GraphDataView;
      const path = `output.${Object.hasOwn(statusLayout, name) ? 'status.' : ''}${name}`;
      validateRequirement(
        id,
        scalar !== null && scalar !== undefined,
        `${path} must be a packed ${descriptor.format} view`
      );
      validateSinglePackedView(scalar, [descriptor.format], `${id} ${path}`);
      validateViewLength(id, path, scalar, descriptor.length);
    }
    const outputViews = getGraphOutputViews(props.output);
    validateDisjointWritableViews(
      id,
      Object.fromEntries(outputViews.map((view, index) => [`output views ${index}`, view]))
    );
    validateGraphOutputsDisjointFromInputs(id, outputViews, [
      ...getGPUNetworkTopologyViews(props.topology),
      props.sources,
      props.targets
    ]);
    const laneCount = props.sources.length * this.spurCount;
    for (const [name, length] of Object.entries({
      expandedNodes: laneCount * this.nodeCount,
      expandedEdges: laneCount * props.topology.neighbors.length,
      pathWords:
        props.sources.length *
        (5 +
          (props.routeCount + this.candidateCapacity + this.spurCount) *
            (2 + 2 * this.maximumPathLength))
    }))
      validateCapacity(id, name, length, 0);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, nodeCount, maximumPathLength, candidateCapacity, spurCount} = this;
    const queryCount = props.sources.length;
    if (queryCount === 0) {
      return (
        ['converged', 'iterationLimitReached', 'candidateOverflow', 'iterationCount'] as const
      ).map(name =>
        createWGSLKernelNode(graph, {
          id: `${id}-empty-${name}`,
          operation: 'GPUKShortestPaths',
          variant: `empty-${name}`,
          invocationCount: 1,
          workgroupSize: 2,
          bindings: [
            {name: 'result', view: props.output.status[name], type: 'u32', access: 'read_write'}
          ],
          body: `result[resultOffset] = ${name === 'converged' ? 1 : 0}u;`
        })
      );
    }
    const laneCount = queryCount * spurCount;
    const expandedNodes = laneCount * nodeCount;
    const edgeCount = props.topology.neighbors.length;
    const expandedEdges = laneCount * edgeCount;
    const recordWords = 2 + 2 * maximumPathLength;
    const queryWords = 5 + (props.routeCount + candidateCapacity + spurCount) * recordWords;
    validateGraphViewsBelongToGraph(id, graph, [
      ...getGPUNetworkTopologyViews(props.topology),
      props.sources,
      props.targets,
      ...getGraphOutputViews(props.output)
    ]);
    // Validate all row-proportional bindings before constructing any command node. Reachability's
    // two-slot frontier is larger than its costs/predecessors and bounds their binding sizes too.
    validateGraphChunkBindingRanges(graph, id, {
      sources: props.sources,
      targets: props.targets,
      ...Object.fromEntries(
        [...getGPUNetworkTopologyViews(props.topology), ...getGraphOutputViews(props.output)].map(
          (view, index) => [`view ${index}`, view]
        )
      )
    });
    for (const [name, bytes] of Object.entries({
      frontier: expandedNodes * 8,
      offsets: (expandedNodes + 1) * 4,
      neighbors: expandedEdges * 4,
      state: queryCount * queryWords * 4,
      starts: laneCount * 4,
      sources: getViewBindingRange(props.sources).size,
      targets: getViewBindingRange(props.targets).size,
      ...Object.fromEntries(
        getGraphOutputViews(props.output).map((view, index) => [
          `output ${index}`,
          getViewBindingRange(view).size
        ])
      ),
      ...Object.fromEntries(
        getGPUNetworkTopologyViews(props.topology)
          .filter(view => view !== undefined)
          .map((view, index) => [`topology ${index}`, getViewBindingRange(view).size])
      )
    }))
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        bytes,
        'reduce queryCount, maximumPathLength or candidateCapacity'
      );
    const unsigned = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const state = unsigned('state', queryCount * queryWords);
    const offsets = unsigned('offsets', expandedNodes + 1);
    const neighbors = unsigned('neighbors', expandedEdges);
    const weights = createTransientView(graph, `${id}-weights`, 'float32', expandedEdges);
    const starts = unsigned('starts', laneCount);
    const startCosts = createTransientView(graph, `${id}-start-costs`, 'float32', laneCount);
    const costs = createTransientView(graph, `${id}-costs`, 'float32', expandedNodes);
    const predecessors = unsigned('predecessors', expandedNodes);
    const converged = unsigned('search-converged', 1);
    const activeQueries = unsigned('active-queries', 1);
    const gate = createWGSLDispatchGate(graph, {
      id: `${id}-dispatch`,
      operation: 'GPUKShortestPaths',
      shapes: [[expandedEdges], [laneCount]]
    });
    const declarations = getKShortestPathsWGSL({
      nodeCount,
      edgeCount,
      maximumPathLength,
      routeCount: props.routeCount,
      candidateCapacity,
      spurCount,
      queryWords,
      recordWords
    });
    const stateBinding = {
      name: 'state',
      view: state,
      type: 'u32' as const,
      access: 'read_write' as const
    };
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode(graph, {
        id: `${id}-active-queries-initialize`,
        operation: 'GPUKShortestPaths',
        variant: 'active-queries-initialize',
        invocationCount: 1,
        bindings: [{name: 'activeQueries', view: activeQueries, type: 'u32', access: 'read_write'}],
        body: 'activeQueries[activeQueriesOffset] = 0u;'
      })
    ];
    nodes.push(
      createFillNode(graph, {
        id: `${id}-clear`,
        operation: 'GPUKShortestPaths',
        view: state,
        type: 'u32',
        value: '0u'
      })
    );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-initialize`,
        operation: 'GPUKShortestPaths',
        variant: 'initialize',
        invocationCount: queryCount,
        declarations,
        bindings: [
          stateBinding,
          {name: 'activeQueries', view: activeQueries, type: 'atomic<u32>', access: 'read_write'},
          {name: 'sources', view: props.sources, type: 'u32', access: 'read'},
          {name: 'targets', view: props.targets, type: 'u32', access: 'read'}
        ],
        body: `let base = stateOffset + index * QUERY_WORDS;
        state[base + 1u] = select(0u, 1u, sources[sourcesOffset + index] < NODE_COUNT && targets[targetsOffset + index] < NODE_COUNT);
        if (state[base + 1u] != 0u) { atomicAdd(&activeQueries[activeQueriesOffset], 1u); }`
      })
    );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-offsets`,
        operation: 'GPUKShortestPaths',
        variant: 'offsets',
        invocationCount: expandedNodes + 1,
        declarations,
        bindings: [
          stateBinding,
          {name: 'offsets', view: props.topology.offsets, type: 'u32', access: 'read'},
          {name: 'expandedOffsets', view: offsets, type: 'u32', access: 'read_write'}
        ],
        body: `if (index == ${expandedNodes}u) { expandedOffsets[expandedOffsetsOffset + index] = ${expandedEdges}u; }
        else { expandedOffsets[expandedOffsetsOffset + index] = (index / NODE_COUNT) * EDGE_COUNT + min(offsets[offsetsOffset + index % NODE_COUNT], EDGE_COUNT); }`
      })
    );
    for (let route = 0; route < props.routeCount; route++) {
      // Every active query has accepted exactly one path per completed round. Read that count
      // from GPU state, keeping every round on the same shader and pipeline (also at 8 bindings).
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-${route}-starts`,
          operation: 'GPUKShortestPaths',
          variant: 'starts',
          invocationCount: laneCount,
          declarations,
          bindings: [
            stateBinding,
            {name: 'sources', view: props.sources, type: 'u32', access: 'read'},
            {name: 'weights', view: props.topology.weights, type: 'f32', access: 'read'},
            {name: 'starts', view: starts, type: 'u32', access: 'read_write'},
            {name: 'startCosts', view: startCosts, type: 'f32', access: 'read_write'}
          ],
          body: `let query = index / SPUR_COUNT; let spur = index % SPUR_COUNT;
          let base = stateOffset + query * QUERY_WORDS;
          let round = state[base];
          starts[startsOffset + index] = NONE; startCosts[startCostsOffset + index] = 0.0;
          if (state[base + 1u] == 0u) { return; }
          var source = sources[sourcesOffset + query]; var cost = 0.0;
          if (round == 0u) { if (spur > 0u) { return; } }
          else {
            let previous = accepted(base, max(round, 1u) - 1u);
            if (spur + 1u >= state[previous + 1u]) { return; }
            source = state[previous + 2u + spur];
            for (var slot = 0u; slot < spur; slot++) { cost += weights[weightsOffset + state[previous + 2u + PATH_LENGTH + slot]]; }
          }
          starts[startsOffset + index] = index * NODE_COUNT + source;
          startCosts[startCostsOffset + index] = cost;`
        })
      );
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-${route}-gate`,
          operation: 'GPUKShortestPaths',
          variant: 'gate',
          invocationCount: 1,
          bindings: [
            {name: 'activeQueries', view: activeQueries, type: 'u32', access: 'read'},
            gate.binding
          ],
          body: `${gate.disarmWGSL} if (activeQueries[activeQueriesOffset] != 0u) { ${gate.armWGSL} }`
        })
      );
      nodes.push(
        createWGSLKernelNode(graph, {
          ...gate.getGate(0),
          id: `${id}-${route}-edges`,
          operation: 'GPUKShortestPaths',
          variant: 'edges',
          invocationCount: expandedEdges,
          declarations: declarations + getBinarySearchWGSL('offsets'),
          bindings: [
            stateBinding,
            {name: 'offsets', view: props.topology.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: props.topology.neighbors, type: 'u32', access: 'read'},
            {name: 'weights', view: props.topology.weights, type: 'f32', access: 'read'},
            {name: 'expandedNeighbors', view: neighbors, type: 'u32', access: 'read_write'},
            {name: 'expandedWeights', view: weights, type: 'f32', access: 'read_write'}
          ],
          body: `let lane = index / max(EDGE_COUNT, 1u); let edge = index % max(EDGE_COUNT, 1u);
          let query = lane / SPUR_COUNT; let spur = lane % SPUR_COUNT;
          let base = stateOffset + query * QUERY_WORDS;
          let round = state[base];
          expandedNeighbors[expandedNeighborsOffset + index] = NONE;
          expandedWeights[expandedWeightsOffset + index] = -1.0;
          if (state[base + 1u] == 0u || edge < offsets[offsetsOffset] || edge >= offsets[offsetsOffset + NODE_COUNT]) { return; }
          let vertex = upperBoundOffsets(0u, NODE_COUNT + 1u, edge) - 1u;
          let neighbor = neighbors[neighborsOffset + edge]; let weight = weights[weightsOffset + edge];
          if (neighbor >= NODE_COUNT || !(weight >= 0.0) || !isFiniteValue(weight)) { return; }
          if (round == 0u) { if (spur > 0u) { return; } }
          else {
            let previous = accepted(base, max(round, 1u) - 1u);
            if (spur + 1u >= state[previous + 1u]) { return; }
            for (var slot = 0u; slot < spur; slot++) {
              let rootVertex = state[previous + 2u + slot];
              if (vertex == rootVertex || neighbor == rootVertex) { return; }
            }
            for (var route = 0u; route < round; route++) {
              let other = accepted(base, route);
              if (state[other + 1u] <= spur + 1u) { continue; }
              var same = true;
              for (var slot = 0u; slot < spur; slot++) {
                if (state[other + 2u + PATH_LENGTH + slot] != state[previous + 2u + PATH_LENGTH + slot]) { same = false; break; }
              }
              if (same && edge == state[other + 2u + PATH_LENGTH + spur]) { return; }
            }
          }
          expandedNeighbors[expandedNeighborsOffset + index] = lane * NODE_COUNT + neighbor;
          expandedWeights[expandedWeightsOffset + index] = weight;`
        })
      );
      nodes.push(
        ...gateKShortestPathSearchNodes(
          graph,
          `${id}-${route}-shortest`,
          new GPUNetworkReachability({
            id: `${id}-${route}-shortest`,
            topology: {offsets, neighbors, weights},
            startNodes: starts,
            startCosts,
            maximumIterations: this.maximumIterations,
            maximumTieIterations: this.maximumIterations,
            localIterations: 64,
            output: {costs, predecessors, status: {converged}}
          }).getCommandNodes(graph),
          activeQueries
        )
      );
      nodes.push(
        createWGSLKernelNode(graph, {
          ...gate.getGate(1),
          id: `${id}-${route}-proposals`,
          operation: 'GPUKShortestPaths',
          variant: 'proposals',
          invocationCount: laneCount,
          declarations,
          bindings: [
            stateBinding,
            {name: 'targets', view: props.targets, type: 'u32', access: 'read'},
            {name: 'starts', view: starts, type: 'u32', access: 'read'},
            {name: 'costs', view: costs, type: 'f32', access: 'read'},
            {name: 'predecessors', view: predecessors, type: 'u32', access: 'read'},
            {name: 'offsets', view: props.topology.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: props.topology.neighbors, type: 'u32', access: 'read'},
            {name: 'weights', view: weights, type: 'f32', access: 'read'}
          ],
          body: `let query = index / SPUR_COUNT; let spur = index % SPUR_COUNT;
          let base = stateOffset + query * QUERY_WORDS; let proposal = proposed(base, spur);
          let round = state[base];
          state[proposal + 1u] = 0u;
          let source = starts[startsOffset + index];
          if (source == NONE || state[base + 1u] == 0u) { return; }
          let destination = index * NODE_COUNT + targets[targetsOffset + query];
          let cost = costs[costsOffset + destination];
          if (!isFiniteValue(cost)) { return; }
          state[proposal] = bitcast<u32>(cost);
          var vertex = destination; var length = 0u;
          // Store the suffix backwards, then reverse it in storage. No graph-sized private array.
          loop {
            if (length + spur >= PATH_LENGTH) { state[proposal + 1u] = NONE; return; }
            state[proposal + 2u + spur + length] = vertex % NODE_COUNT;
            length++;
            if (vertex == source) { break; }
            // The reverse edge uses slot spur + length, and its parent needs another vertex slot.
            if (length + spur >= PATH_LENGTH) { state[proposal + 1u] = NONE; return; }
            let parent = predecessors[predecessorsOffset + vertex];
            if (parent == NONE || parent / NODE_COUNT != index) { state[proposal + 1u] = NONE; return; }
            var chosen = NONE;
            for (var edge = min(offsets[offsetsOffset + parent % NODE_COUNT], EDGE_COUNT); edge < min(offsets[offsetsOffset + parent % NODE_COUNT + 1u], EDGE_COUNT); edge++) {
              let weight = weights[weightsOffset + index * EDGE_COUNT + edge];
              if (neighbors[neighborsOffset + edge] == vertex % NODE_COUNT && weight >= 0.0 && costs[costsOffset + parent] + weight == costs[costsOffset + vertex]) { chosen = edge; break; }
            }
            if (chosen == NONE) { state[proposal + 1u] = NONE; return; }
            state[proposal + 2u + PATH_LENGTH + spur + length] = chosen;
            vertex = parent;
          }
          for (var slot = 0u; slot < length / 2u; slot++) {
            let first = proposal + 2u + spur + slot; let last = proposal + 2u + spur + length - 1u - slot;
            let saved = state[first]; state[first] = state[last]; state[last] = saved;
          }
          for (var slot = 0u; slot < (length - 1u) / 2u; slot++) {
            let first = proposal + 2u + PATH_LENGTH + spur + slot; let last = proposal + 2u + PATH_LENGTH + spur + length - 2u - slot;
            // Reverse edges were stored at slots 1..length-1. Shift them down below.
            let saved = state[first + 1u]; state[first + 1u] = state[last + 1u]; state[last + 1u] = saved;
          }
          for (var slot = 0u; slot + 1u < length; slot++) { state[proposal + 2u + PATH_LENGTH + spur + slot] = state[proposal + 2u + PATH_LENGTH + spur + slot + 1u]; }
          if (round > 0u) {
            let previous = accepted(base, max(round, 1u) - 1u);
            for (var slot = 0u; slot < spur; slot++) {
              state[proposal + 2u + slot] = state[previous + 2u + slot];
              state[proposal + 2u + PATH_LENGTH + slot] = state[previous + 2u + PATH_LENGTH + slot];
            }
          }
          state[proposal + 1u] = spur + length;`
        })
      );
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-${route}-merge`,
          operation: 'GPUKShortestPaths',
          variant: 'merge',
          invocationCount: queryCount,
          declarations:
            declarations +
            '\nfn stopQuery(base: u32) { state[base + 1u] = 0u; atomicSub(&activeQueries[activeQueriesOffset], 1u); }',
          bindings: [
            stateBinding,
            {name: 'activeQueries', view: activeQueries, type: 'atomic<u32>', access: 'read_write'},
            {name: 'converged', view: converged, type: 'u32', access: 'read'}
          ],
          body: `let base = stateOffset + index * QUERY_WORDS;
          let round = state[base];
          if (state[base + 1u] == 0u) { return; }
          state[base + 4u]++;
          if (converged[convergedOffset] == 0u) { state[base + 2u] = 1u; stopQuery(base); return; }
          for (var spur = 0u; spur < SPUR_COUNT; spur++) {
            let proposal = proposed(base, spur); let length = state[proposal + 1u];
            if (length == NONE) { state[base + 2u] = 1u; stopQuery(base); return; }
            if (length == 0u) { continue; }
            var duplicate = false; var empty = NONE;
            for (var slot = 0u; slot < CANDIDATE_CAPACITY; slot++) {
              let candidate = pending(base, slot);
              if (state[candidate + 1u] == 0u) { if (empty == NONE) { empty = candidate; } }
              else if (samePath(proposal, candidate)) { duplicate = true; break; }
            }
            for (var slot = 0u; slot < round && !duplicate; slot++) { duplicate = samePath(proposal, accepted(base, slot)); }
            if (duplicate) { continue; }
            if (empty == NONE) { state[base + 3u] = 1u; stopQuery(base); return; }
            copyPath(proposal, empty);
          }
          var best = NONE;
          for (var slot = 0u; slot < CANDIDATE_CAPACITY; slot++) {
            let candidate = pending(base, slot);
            if (state[candidate + 1u] != 0u && (best == NONE || precedes(candidate, best))) { best = candidate; }
          }
          if (best == NONE) { stopQuery(base); return; }
          copyPath(best, accepted(base, round)); state[best + 1u] = 0u;
          state[base]++;
          if (state[base] == ROUTE_COUNT) { stopQuery(base); }`
        })
      );
    }
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-paths`,
        operation: 'GPUKShortestPaths',
        variant: 'paths',
        invocationCount: queryCount * props.routeCount * maximumPathLength,
        declarations,
        bindings: [
          stateBinding,
          {name: 'vertices', view: props.output.vertices, type: 'u32', access: 'read_write'},
          {name: 'edges', view: props.output.edges, type: 'u32', access: 'read_write'}
        ],
        body: `let query = index / (ROUTE_COUNT * PATH_LENGTH); let route = (index / PATH_LENGTH) % ROUTE_COUNT; let slot = index % PATH_LENGTH;
        let record = accepted(stateOffset + query * QUERY_WORDS, route); let length = state[record + 1u];
        vertices[verticesOffset + index] = select(NONE, state[record + 2u + slot], slot < length);
        edges[edgesOffset + index] = select(NONE, state[record + 2u + PATH_LENGTH + slot], slot + 1u < length);`
      })
    );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-metadata`,
        operation: 'GPUKShortestPaths',
        variant: 'metadata',
        invocationCount: queryCount,
        declarations,
        bindings: [
          stateBinding,
          {name: 'lengths', view: props.output.lengths, type: 'u32', access: 'read_write'},
          {name: 'costs', view: props.output.costs, type: 'f32', access: 'read_write'},
          {name: 'counts', view: props.output.counts, type: 'u32', access: 'read_write'}
        ],
        body: `let base = stateOffset + index * QUERY_WORDS; counts[countsOffset + index] = state[base];
        var infinityBits = 0x7f800000u;
        for (var route = 0u; route < ROUTE_COUNT; route++) {
          let record = accepted(base, route); let length = state[record + 1u];
          lengths[lengthsOffset + index * ROUTE_COUNT + route] = length;
          costs[costsOffset + index * ROUTE_COUNT + route] = select(bitcast<f32>(infinityBits), bitcast<f32>(state[record]), length > 0u);
        }`
      })
    );
    // Batch status uses atomics so no invocation scans the query-sized range.
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-status-initialize`,
        operation: 'GPUKShortestPaths',
        variant: 'status-initialize',
        invocationCount: 1,
        bindings: Object.entries(props.output.status).map(([name, view]) => ({
          name,
          view,
          type: 'u32',
          access: 'read_write'
        })),
        body: 'converged[convergedOffset] = 1u; iterationLimitReached[iterationLimitReachedOffset] = 0u; candidateOverflow[candidateOverflowOffset] = 0u; iterationCount[iterationCountOffset] = 0u;'
      })
    );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-status`,
        operation: 'GPUKShortestPaths',
        variant: 'status',
        invocationCount: queryCount,
        declarations,
        bindings: [
          stateBinding,
          ...Object.entries(props.output.status).map(([name, view]) => ({
            name,
            view,
            type: 'atomic<u32>' as const,
            access: 'read_write' as const
          }))
        ],
        body: `let base = stateOffset + index * QUERY_WORDS;
        atomicMin(&converged[convergedOffset], select(1u, 0u, state[base + 2u] != 0u || state[base + 3u] != 0u));
        atomicMax(&iterationLimitReached[iterationLimitReachedOffset], state[base + 2u]);
        atomicMax(&candidateOverflow[candidateOverflowOffset], state[base + 3u]);
        atomicMax(&iterationCount[iterationCountOffset], state[base + 4u]);`
      })
    );
    return nodes;
  }
}
