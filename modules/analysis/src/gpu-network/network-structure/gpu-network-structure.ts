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
  type GPUNetworkAdjacency,
  validateGPUNetworkTopology,
  getGPUNetworkTopologyViews
} from '../../analysis-core/contracts/network-topology';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateSinglePackedView,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs,
  validateCapacity,
  validateMaximumCount,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';
import type {GPUIterationStatusPort} from '../../analysis-core/contracts/status';
import {
  GPUNetworkStronglyConnectedComponents,
  GPU_NETWORK_STRONGLY_CONNECTED_COMPONENTS_DEFAULT_ITERATIONS,
  GPU_NETWORK_STRONGLY_CONNECTED_COMPONENTS_MAXIMUM_ITERATIONS
} from './gpu-network-strongly-connected-components';
import {
  createNetworkStructureResilienceNodes,
  getNetworkStructureResilienceShape,
  GPU_NETWORK_STRUCTURE_MAXIMUM_AUXILIARY_COUNT
} from './gpu-network-structure-resilience';

/** Four SCC phase gates are the largest fixed scratch binding (four dispatch triples). */
const MAXIMUM_SCC_DISPATCH_WORD_COUNT = 12;
/** Resilience dispatches vertices, edges and segments through three dispatch triples. */
const MAXIMUM_RESILIENCE_DISPATCH_WORD_COUNT = 9;

/** Reserve NONE and keep doubled Euler-tour scratch offsets within uint32. Device limits also apply. */
export const GPU_NETWORK_STRUCTURE_MAXIMUM_NODE_COUNT = 0x7ffffffe;
/** Keep the two-word endpoint and Euler-tour scratch tables addressable with uint32 offsets. */
export const GPU_NETWORK_STRUCTURE_MAXIMUM_EDGE_COUNT = 0x7ffffffe;
/** The exact reference SCC uses a quadratic reachability matrix and a private per-vertex queue. */
export const GPU_NETWORK_STRUCTURE_EXACT_MAXIMUM_NODE_COUNT = 1024;
/** The exact reference resilience traversal visits every slot in a single invocation. */
export const GPU_NETWORK_STRUCTURE_EXACT_MAXIMUM_EDGE_COUNT = 65536;

export type GPUNetworkStructureProps = {
  id?: string;
  /** Default auto preserves bounded exact SCCs and uses parallel resilience. Explicit parallel opts into budgeted SCCs at every size; exact selects both reference kernels. */
  mode?: 'auto' | 'parallel' | 'exact';
  /** SCC state-machine budget only; exhaustion publishes 0xffffffff labels. Default 256. */
  strongComponentsMaximumIterations?: number;
  /** Optional caller-owned SCC status scalars, published even when the labels are invalidated. */
  strongComponentsStatus?: Required<GPUIterationStatusPort>;
  topology: GPUNetworkAdjacency;
  /** Exact reverse slot for every CSR edge. Required for undirected resilience outputs.
   * Parallel edges must pair separately. Self-loops pair with themselves and form singleton blocks.
   */
  reverseEdgeSlots?: GraphDataView<'uint32'>;
  output: {
    /** Lowest vertex ID in each directed strongly connected component. */
    strongComponents?: GraphDataView<'uint32'>;
    /** One flag per undirected CSR slot; both directions carry the same result. */
    bridges?: GraphDataView<'uint32'>;
    /** One flag per vertex. */
    articulationPoints?: GraphDataView<'uint32'>;
    /** Biconnected edge block ID (lowest CSR slot in the block); invalid slots are 0xffffffff. */
    biconnectedComponents?: GraphDataView<'uint32'>;
  };
};

/**
 * Exact graph structure on single packed GPU-resident CSR views. Weights are ignored.
 *
 * Default mode: 'auto' preserves the original exact SCC route through 1,024 vertices and
 * 65,536 slots, so previously supported SCC results never depend on an iteration budget.
 * Larger graphs, or explicit mode: 'parallel', compose GPUNetworkStronglyConnectedComponents
 * (O(V) scratch, budgeted trim/colour/backward-mark phases). Auto and parallel use a spanning forest, Euler-tour ranking,
 * binary tree lifting and fundamental-cycle auxiliary connectivity for resilience (O(E + V log V)
 * scratch). It has no depth-sized serial traversal. Resilience always completes in the compiled
 * logarithmic round bound; SCC may exhaust its independently configurable state-machine budget.
 * Read strongComponentsStatus to distinguish exhaustion from convergence; without that optional
 * port, an all-0xffffffff label vector still signals exhaustion. Other outputs remain valid.
 *
 * mode: 'exact' retains the bounded reference: mutual reachability with O(V²) scratch for SCC,
 * and single-invocation iterative Tarjan for resilience, limited to 1,024 vertices/65,536 slots.
 *
 * All props are compile-time. Output layouts, minimum vertex/CSR-slot labels and singleton-loop
 * blocks are identical in both routes. CSR offsets must be monotonic and within capacity;
 * reverseEdgeSlots must be an involution connecting reversed endpoints. Parallel edges pair
 * separately. Unowned CSR tail slots and out-of-range destinations/reverse slots are ignored.
 * Device storage-binding limits further bound accepted graph and auxiliary scratch sizes.
 */
export class GPUNetworkStructure implements GPUCommandNodeProducer {
  readonly id: string;
  readonly nodeCount: number;
  readonly edgeCount: number;
  readonly mode: 'auto' | 'parallel' | 'exact';
  readonly strongComponentsMode: 'parallel' | 'exact';
  readonly strongComponentsMaximumIterations: number;
  readonly estimatedCommandNodeCount: number;
  constructor(readonly props: GPUNetworkStructureProps) {
    this.id = props.id ?? 'network-structure';
    this.mode = props.mode ?? 'auto';
    validateRequirement(
      this.id,
      this.mode === 'auto' || this.mode === 'parallel' || this.mode === 'exact',
      "mode must be 'auto', 'parallel' or 'exact'"
    );
    this.strongComponentsMaximumIterations =
      props.strongComponentsMaximumIterations ??
      GPU_NETWORK_STRONGLY_CONNECTED_COMPONENTS_DEFAULT_ITERATIONS;
    validateCapacity(
      this.id,
      'strongComponentsMaximumIterations',
      this.strongComponentsMaximumIterations,
      1,
      GPU_NETWORK_STRONGLY_CONNECTED_COMPONENTS_MAXIMUM_ITERATIONS
    );
    const shape = validateGPUNetworkTopology(this.id, props.topology);
    this.nodeCount = shape.nodeCount;
    this.edgeCount = shape.edgeCount;
    this.strongComponentsMode =
      this.mode === 'auto'
        ? this.nodeCount <= GPU_NETWORK_STRUCTURE_EXACT_MAXIMUM_NODE_COUNT &&
          this.edgeCount <= GPU_NETWORK_STRUCTURE_EXACT_MAXIMUM_EDGE_COUNT
          ? 'exact'
          : 'parallel'
        : this.mode;
    const {output} = props;
    const hasResilienceOutput = Boolean(
      output.bridges || output.articulationPoints || output.biconnectedComponents
    );
    validateCapacity(
      this.id,
      this.mode === 'exact'
        ? "nodeCount for exact SCC (use mode: 'auto' for automatic selection)"
        : 'nodeCount',
      this.nodeCount,
      1,
      this.mode === 'exact'
        ? GPU_NETWORK_STRUCTURE_EXACT_MAXIMUM_NODE_COUNT
        : GPU_NETWORK_STRUCTURE_MAXIMUM_NODE_COUNT
    );
    validateCapacity(
      this.id,
      this.mode === 'exact'
        ? "edgeCount for exact SCC (use mode: 'auto' for automatic selection)"
        : 'edgeCount',
      this.edgeCount,
      0,
      this.mode === 'exact'
        ? GPU_NETWORK_STRUCTURE_EXACT_MAXIMUM_EDGE_COUNT
        : GPU_NETWORK_STRUCTURE_MAXIMUM_EDGE_COUNT
    );
    validateRequirement(this.id, Object.values(output).some(Boolean), 'requires an output');
    for (const [name, view] of Object.entries(output)) {
      if (!view) continue;
      validateSinglePackedView(view, ['uint32'], `${this.id} ${name}`);
      const length =
        name === 'strongComponents' || name === 'articulationPoints'
          ? this.nodeCount
          : this.edgeCount;
      validateRequirement(
        this.id,
        view.length === length,
        `output.${name} must hold ${length} rows (one per ${
          length === this.nodeCount ? 'vertex' : 'CSR slot'
        })`
      );
    }
    if (hasResilienceOutput) {
      validateRequirement(
        this.id,
        props.reverseEdgeSlots !== undefined,
        'reverseEdgeSlots is required for bridges, articulationPoints and biconnectedComponents'
      );
      validateSinglePackedView(props.reverseEdgeSlots, ['uint32'], `${this.id} reverseEdgeSlots`);
      validateRequirement(
        this.id,
        props.reverseEdgeSlots.length === this.edgeCount,
        'reverseEdgeSlots must hold one row per CSR slot'
      );
    }
    const statusViews = props.strongComponentsStatus
      ? Object.values(props.strongComponentsStatus)
      : [];
    if (props.strongComponentsStatus) {
      validateRequirement(
        this.id,
        Boolean(output.strongComponents),
        'strongComponentsStatus requires output.strongComponents'
      );
      for (const name of ['converged', 'iterationCount', 'iterationLimitReached'] as const) {
        const view: GraphDataView<'uint32'> = props.strongComponentsStatus[name];
        validateSinglePackedView(view, ['uint32'], `${this.id} strongComponentsStatus.${name}`);
        validateRequirement(
          this.id,
          view.length === 1,
          `strongComponentsStatus.${name} must hold exactly one uint32 row`
        );
      }
    }
    const resilienceShape = getNetworkStructureResilienceShape(this.nodeCount, this.edgeCount);
    if (hasResilienceOutput && this.mode !== 'exact') {
      validateMaximumCount(
        this.id,
        'edgeCount + nodeCount * treeLevels (auxiliaryCount)',
        resilienceShape.auxiliaryCount,
        GPU_NETWORK_STRUCTURE_MAXIMUM_AUXILIARY_COUNT
      );
    }
    this.estimatedCommandNodeCount =
      (output.strongComponents
        ? this.strongComponentsMode === 'exact'
          ? 2 + (props.strongComponentsStatus ? 1 : 0)
          : 2 + 6 * this.strongComponentsMaximumIterations
        : 0) +
      (hasResilienceOutput
        ? this.mode === 'exact'
          ? 1
          : resilienceShape.estimatedCommandNodeCount
        : 0);
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    // Include physical aliases imported through different handles and the aligned prefixes
    // of co-bound packed fields. Keep the existing distinct-handle requirement below as well.
    validateDisjointWritableViews(this.id, {...output, ...props.strongComponentsStatus});
    const outputBuffers = [...Object.values(props.output), ...statusViews]
      .filter(view => view !== undefined)
      .map(view => view.buffer);
    validateRequirement(
      this.id,
      new Set(outputBuffers).size === outputBuffers.length,
      'outputs must use distinct buffers'
    );
    validateGraphOutputsDisjointFromInputs(
      this.id,
      [...Object.values(output), ...statusViews],
      [...getGPUNetworkTopologyViews(props.topology), props.reverseEdgeSlots]
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, nodeCount, edgeCount} = this;
    const {output, topology} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      ...getGPUNetworkTopologyViews(topology),
      props.reverseEdgeSlots,
      ...Object.values(output),
      ...Object.values(props.strongComponentsStatus ?? {})
    ]);
    // Validate every row-proportional allocation before creating any command nodes.
    for (const [name, view] of Object.entries({
      offsets: topology.offsets,
      neighbors: topology.neighbors,
      weights: topology.weights,
      nodePositions: topology.nodePositions,
      reverseEdgeSlots: props.reverseEdgeSlots,
      ...output,
      ...props.strongComponentsStatus
    })) {
      if (view) {
        validateMaximumCount(
          id,
          `${name} buffer byteLength (maxBufferSize)`,
          view.buffer.byteLength,
          graph.device.limits.maxBufferSize
        );
        validateGPUStorageBindingByteLength(
          graph.device,
          id,
          name,
          getViewBindingRange(view).size,
          'reduce nodeCount or edgeCount to fit the device storage binding limit'
        );
      }
    }
    const hasResilienceOutput = Boolean(
      output.bridges || output.articulationPoints || output.biconnectedComponents
    );
    // Every scratch allocation is a complete packed view at byteOffset 0, so its
    // aligned binding range is exactly the sizes below (including one word for empty tables).
    const scratchSizes: Record<string, number> = {
      controlAndDispatch:
        (output.strongComponents && this.strongComponentsMode === 'parallel'
          ? MAXIMUM_SCC_DISPATCH_WORD_COUNT
          : hasResilienceOutput && this.mode !== 'exact'
            ? MAXIMUM_RESILIENCE_DISPATCH_WORD_COUNT
            : 1) * 4,
      vertices: nodeCount * 4,
      ...(output.strongComponents && this.strongComponentsMode === 'exact'
        ? {reachable: nodeCount * nodeCount * 4}
        : {}),
      ...(hasResilienceOutput && this.mode === 'exact'
        ? {traversal: nodeCount * 6 * 4, stack: Math.max(edgeCount, 1) * 4}
        : {}),
      ...(hasResilienceOutput && this.mode !== 'exact'
        ? {
            endpointsAndTour: Math.max(edgeCount * 2, 1) * 4,
            auxiliary: getNetworkStructureResilienceShape(nodeCount, edgeCount).auxiliaryCount * 4
          }
        : {})
    };
    for (const [name, byteLength] of Object.entries(scratchSizes)) {
      validateMaximumCount(
        id,
        `${name} scratch byteLength (maxBufferSize)`,
        byteLength,
        graph.device.limits.maxBufferSize
      );
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        `${name} scratch`,
        byteLength,
        'reduce nodeCount or edgeCount to fit the device storage binding limit'
      );
    }
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (this.mode !== 'exact') {
      if (output.strongComponents && this.strongComponentsMode === 'parallel') {
        const status = props.strongComponentsStatus ?? {
          converged: createTransientView(graph, `${id}-strong-converged`, 'uint32', 1),
          iterationCount: createTransientView(graph, `${id}-strong-iteration-count`, 'uint32', 1),
          iterationLimitReached: createTransientView(
            graph,
            `${id}-strong-iteration-limit`,
            'uint32',
            1
          )
        };
        nodes.push(
          ...new GPUNetworkStronglyConnectedComponents({
            id: `${id}-strong`,
            topology,
            maximumIterations: this.strongComponentsMaximumIterations,
            output: {strongComponents: output.strongComponents, status}
          }).getCommandNodes(graph)
        );
      }
      if (hasResilienceOutput) {
        nodes.push(
          ...createNetworkStructureResilienceNodes(
            graph,
            `${id}-resilience`,
            props,
            nodeCount,
            edgeCount
          )
        );
      }
    }
    const adjacency = [
      {name: 'offsets', view: topology.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: topology.neighbors, type: 'u32', access: 'read'}
    ] as const;
    const constants = `const NODE_COUNT: u32 = ${nodeCount}u; const EDGE_COUNT: u32 = ${edgeCount}u; const NONE: u32 = 0xffffffffu;`;
    if (output.strongComponents && this.strongComponentsMode === 'exact') {
      const reachable = createTransientView(
        graph,
        `${id}-reachable`,
        'uint32',
        nodeCount * nodeCount
      );
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-reach`,
          operation: 'GPUNetworkStructure',
          variant: 'reach',
          invocationCount: nodeCount,
          bindings: [
            ...adjacency,
            {name: 'reachable', view: reachable, type: 'u32', access: 'read_write'}
          ],
          declarations: constants,
          body: `var queue: array<u32, ${nodeCount}>;
  let base = reachableOffset + index * NODE_COUNT;
  for (var vertex = 0u; vertex < NODE_COUNT; vertex++) { reachable[base + vertex] = 0u; }
  reachable[base + index] = 1u; queue[0] = index;
  var head = 0u; var tail = 1u;
  while (head < tail) {
    let vertex = queue[head]; head++;
    let first = min(offsets[offsetsOffset + vertex], EDGE_COUNT);
    let last = min(offsets[offsetsOffset + vertex + 1u], EDGE_COUNT);
    for (var edge = first; edge < last; edge++) {
      let destination = neighbors[neighborsOffset + edge];
      if (destination < NODE_COUNT && reachable[base + destination] == 0u) {
        reachable[base + destination] = 1u; queue[tail] = destination; tail++;
      }
    }
  }`
        }),
        createWGSLKernelNode(graph, {
          id: `${id}-strong-components`,
          operation: 'GPUNetworkStructure',
          variant: 'strong-components',
          invocationCount: nodeCount,
          bindings: [
            {name: 'reachable', view: reachable, type: 'u32', access: 'read'},
            {name: 'labels', view: output.strongComponents, type: 'u32', access: 'read_write'}
          ],
          declarations: constants,
          body: `var label = index;
  for (var vertex = 0u; vertex < index; vertex++) {
    if (reachable[reachableOffset + index * NODE_COUNT + vertex] != 0u && reachable[reachableOffset + vertex * NODE_COUNT + index] != 0u) { label = vertex; break; }
  }
  labels[labelsOffset + index] = label;`
        })
      );
    }
    if (
      this.mode === 'exact' &&
      props.reverseEdgeSlots &&
      (output.bridges || output.articulationPoints || output.biconnectedComponents)
    ) {
      const bridges =
        output.bridges ??
        createTransientView(graph, `${id}-bridges`, 'uint32', Math.max(edgeCount, 1));
      const articulation =
        output.articulationPoints ??
        createTransientView(graph, `${id}-articulation`, 'uint32', nodeCount);
      const blocks =
        output.biconnectedComponents ??
        createTransientView(graph, `${id}-blocks`, 'uint32', Math.max(edgeCount, 1));
      const stack = createTransientView(
        graph,
        `${id}-edge-stack`,
        'uint32',
        Math.max(edgeCount, 1)
      );
      const traversal = createTransientView(graph, `${id}-traversal`, 'uint32', nodeCount * 6);
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-resilience`,
          operation: 'GPUNetworkStructure',
          variant: 'resilience',
          invocationCount: 1,
          workgroupSize: 2,
          bindings: [
            ...adjacency,
            {name: 'reverse', view: props.reverseEdgeSlots, type: 'u32', access: 'read'},
            {name: 'bridges', view: bridges, type: 'u32', access: 'read_write'},
            {name: 'articulation', view: articulation, type: 'u32', access: 'read_write'},
            {name: 'blocks', view: blocks, type: 'u32', access: 'read_write'},
            {name: 'stack', view: stack, type: 'u32', access: 'read_write'},
            {name: 'traversal', view: traversal, type: 'u32', access: 'read_write'}
          ],
          declarations: constants,
          body: `for (var vertex = 0u; vertex < NODE_COUNT; vertex++) {
    traversal[traversalOffset + 5u * NODE_COUNT + vertex] = 0u;
    traversal[traversalOffset + 0u * NODE_COUNT + vertex] = NONE; traversal[traversalOffset + 2u * NODE_COUNT + vertex] = NONE; traversal[traversalOffset + 3u * NODE_COUNT + vertex] = NONE;
    traversal[traversalOffset + 4u * NODE_COUNT + vertex] = min(offsets[offsetsOffset + vertex], EDGE_COUNT);
    articulation[articulationOffset + vertex] = 0u;
  }
  for (var edge = 0u; edge < EDGE_COUNT; edge++) { bridges[bridgesOffset + edge] = 0u; blocks[blocksOffset + edge] = NONE; }
  var clock = 0u; var stackCount = 0u;
  for (var root = 0u; root < NODE_COUNT; root++) {
    if (traversal[traversalOffset + 0u * NODE_COUNT + root] != NONE) { continue; }
    var vertex = root; traversal[traversalOffset + 0u * NODE_COUNT + vertex] = clock; traversal[traversalOffset + 1u * NODE_COUNT + vertex] = clock; clock++;
    loop {
      let last = min(offsets[offsetsOffset + vertex + 1u], EDGE_COUNT);
      if (traversal[traversalOffset + 4u * NODE_COUNT + vertex] < last) {
        let edge = traversal[traversalOffset + 4u * NODE_COUNT + vertex]; traversal[traversalOffset + 4u * NODE_COUNT + vertex]++;
        let destination = neighbors[neighborsOffset + edge];
        let twin = reverse[reverseOffset + edge];
        if (destination >= NODE_COUNT || twin >= EDGE_COUNT) { continue; }
        if (destination == vertex) { blocks[blocksOffset + edge] = min(edge, twin); continue; }
        if (traversal[traversalOffset + 3u * NODE_COUNT + vertex] != NONE && edge == reverse[reverseOffset + traversal[traversalOffset + 3u * NODE_COUNT + vertex]]) { continue; }
        if (traversal[traversalOffset + 0u * NODE_COUNT + destination] == NONE) {
          stack[stackOffset + stackCount] = edge; stackCount++;
          traversal[traversalOffset + 5u * NODE_COUNT + vertex]++; traversal[traversalOffset + 2u * NODE_COUNT + destination] = vertex; traversal[traversalOffset + 3u * NODE_COUNT + destination] = edge;
          vertex = destination; traversal[traversalOffset + 0u * NODE_COUNT + vertex] = clock; traversal[traversalOffset + 1u * NODE_COUNT + vertex] = clock; clock++;
        } else if (traversal[traversalOffset + 0u * NODE_COUNT + destination] < traversal[traversalOffset + 0u * NODE_COUNT + vertex]) {
          stack[stackOffset + stackCount] = edge; stackCount++;
          traversal[traversalOffset + 1u * NODE_COUNT + vertex] = min(traversal[traversalOffset + 1u * NODE_COUNT + vertex], traversal[traversalOffset + 0u * NODE_COUNT + destination]);
        }
      } else {
        if (traversal[traversalOffset + 2u * NODE_COUNT + vertex] == NONE) {
          articulation[articulationOffset + vertex] = select(0u, 1u, traversal[traversalOffset + 5u * NODE_COUNT + vertex] > 1u); break;
        }
        let ancestor = traversal[traversalOffset + 2u * NODE_COUNT + vertex]; let treeEdge = traversal[traversalOffset + 3u * NODE_COUNT + vertex];
        traversal[traversalOffset + 1u * NODE_COUNT + ancestor] = min(traversal[traversalOffset + 1u * NODE_COUNT + ancestor], traversal[traversalOffset + 1u * NODE_COUNT + vertex]);
        if (traversal[traversalOffset + 1u * NODE_COUNT + vertex] > traversal[traversalOffset + 0u * NODE_COUNT + ancestor]) {
          bridges[bridgesOffset + treeEdge] = 1u;
          bridges[bridgesOffset + reverse[reverseOffset + treeEdge]] = 1u;
        }
        if (traversal[traversalOffset + 1u * NODE_COUNT + vertex] >= traversal[traversalOffset + 0u * NODE_COUNT + ancestor]) {
          if (traversal[traversalOffset + 2u * NODE_COUNT + ancestor] != NONE) { articulation[articulationOffset + ancestor] = 1u; }
          let end = stackCount; var label = NONE;
          loop {
            if (stackCount == 0u) { break; }
            stackCount--; let popped = stack[stackOffset + stackCount];
            label = min(label, min(popped, reverse[reverseOffset + popped]));
            if (popped == treeEdge) { break; }
          }
          for (var position = stackCount; position < end; position++) {
            let popped = stack[stackOffset + position]; blocks[blocksOffset + popped] = label;
            blocks[blocksOffset + reverse[reverseOffset + popped]] = label;
          }
        }
        vertex = ancestor;
      }
    }
  }`
        })
      );
    }
    if (props.strongComponentsStatus && this.strongComponentsMode === 'exact') {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-exact-status`,
          operation: 'GPUNetworkStructure',
          variant: 'exact-status',
          invocationCount: 1,
          bindings: Object.entries(props.strongComponentsStatus).map(([name, view]) => ({
            name,
            view,
            type: 'u32',
            access: 'read_write'
          })),
          body: 'converged[convergedOffset] = 1u; iterationCount[iterationCountOffset] = 1u; iterationLimitReached[iterationLimitReachedOffset] = 0u;'
        })
      );
    }
    return nodes;
  }
}
