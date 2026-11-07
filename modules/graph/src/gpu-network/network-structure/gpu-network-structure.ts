// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {
  GPUGraphAlgorithmsStronglyConnectedComponents,
  GPUGraphAlgorithmsTopologyView
} from '@luma-contributors/graph/gpu-graph-algorithms';
import {
  type GPUNetworkAdjacency,
  validateGPUNetworkTopology,
  getGPUNetworkTopologyViews
} from '../../analysis-core/network-topology';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  captureGraphCommandNodes,
  validatePackedView,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs,
  validateCapacity
} from '../../utils/gpu-contributor-utils';

export type GPUNetworkStructureProps = {
  id?: string;
  topology: GPUNetworkAdjacency;
  /** Exact reverse slot for every CSR edge. Required for undirected resilience outputs.
   * Parallel edges must pair separately. Self-loops pair with themselves and form singleton blocks.
   */
  reverseEdgeSlots?: GraphDataView<'uint32'>;
  output: {
    /** Lowest vertex ID in each directed strongly connected component. */
    strongComponents?: GraphDataView<'uint32'>;
    /**
     * One row, set to one only when every strong component was resolved within the iteration
     * budget. Only meaningful with the default `'iterative'` strong-component method.
     */
    strongComponentsConverged?: GraphDataView<'uint32'>;
    /** One flag per undirected CSR slot; both directions carry the same result. */
    bridges?: GraphDataView<'uint32'>;
    /** One flag per vertex. */
    articulationPoints?: GraphDataView<'uint32'>;
    /** Biconnected edge block ID (lowest CSR slot in the block); invalid slots are 0xffffffff. */
    biconnectedComponents?: GraphDataView<'uint32'>;
  };
  /**
   * How `strongComponents` is computed. `'iterative'` runs the O(V) scratch GPU state machine of
   * `GPUGraphAlgorithmsStronglyConnectedComponents` and has no size limit. `'reachability'` keeps
   * the original exact O(V(V+E)) work, O(V^2) scratch path for at most 1024 vertices and 65536 CSR
   * slots, whose result never depends on an iteration budget. `'auto'` (default) uses
   * `'reachability'` within those caps unless `strongComponentsConverged` or
   * `strongComponentsIterations` is given, and `'iterative'` otherwise.
   */
  strongComponentsMethod?: 'auto' | 'iterative' | 'reachability';
  /**
   * State-machine step budget of the `'iterative'` method (default 256, at most 4096). When it is
   * exhausted before every component is resolved, `strongComponents` publishes `0xffffffff` and
   * `strongComponentsConverged` stays zero.
   */
  strongComponentsIterations?: number;
};

/** Exact bounded graph structure on GPU-resident CSR. Connectivity ignores weights.
 * SCC uses the exact mutual-reachability path (O(V(V+E)) work, O(V²) scratch) within 1024
 * vertices and the bounded GPU state machine of `GPUGraphAlgorithmsStronglyConnectedComponents`
 * (O(V) scratch, no vertex limit) above that; see `strongComponentsMethod`.
 * Undirected resilience uses an iterative Tarjan traversal in one GPU invocation; this bounded
 * fallback is intended for subgraphs, not throughput-oriented giant graphs.
 * Resilience outputs and the reachability SCC path allow at most 1024 vertices and 65536 CSR
 * slots. CSR offsets must be monotonic and within capacity;
 * reverseEdgeSlots must be an involution connecting reversed endpoints.
 */
export class GPUNetworkStructure implements GPUCommandNodeProducer {
  readonly id: string;
  readonly nodeCount: number;
  readonly edgeCount: number;
  private readonly strongComponents?: GPUGraphAlgorithmsStronglyConnectedComponents<
    GPUGraphAlgorithmsTopologyView,
    GraphDataView<'uint32'>
  >;
  constructor(readonly props: GPUNetworkStructureProps) {
    this.id = props.id ?? 'network-structure';
    const shape = validateGPUNetworkTopology(this.id, props.topology);
    this.nodeCount = shape.nodeCount;
    this.edgeCount = shape.edgeCount;
    const {output} = props;
    const strongComponentsMethod = props.strongComponentsMethod ?? 'auto';
    const fitsReachability = this.nodeCount <= 1024 && this.edgeCount <= 65536;
    const isIterative =
      Boolean(output.strongComponents) &&
      (strongComponentsMethod === 'iterative' ||
        (strongComponentsMethod === 'auto' &&
          (!fitsReachability ||
            Boolean(output.strongComponentsConverged) ||
            props.strongComponentsIterations !== undefined)));
    const hasResilienceOutput = Boolean(
      output.bridges || output.articulationPoints || output.biconnectedComponents
    );
    // Only the iterative strong-component path scales beyond the bounded single-invocation kernels.
    if (!isIterative || hasResilienceOutput) {
      validateCapacity(this.id, 'nodeCount', this.nodeCount, 1, 1024);
      validateCapacity(this.id, 'edgeCount', this.edgeCount, 0, 65536);
    }
    if (!Object.values(output).some(Boolean))
      throw new Error('Network structure requires an output');
    for (const [name, view] of Object.entries(output)) {
      if (!view) continue;
      validatePackedView(view, ['uint32'], `${this.id} ${name}`);
      const length =
        name === 'strongComponentsConverged'
          ? 1
          : name === 'strongComponents' || name === 'articulationPoints'
            ? this.nodeCount
            : this.edgeCount;
      if (view.length !== length) throw new Error('Network structure output length mismatch');
    }
    if (output.strongComponentsConverged && !isIterative) {
      throw new Error('strongComponentsConverged requires the iterative strong-component method');
    }
    if (isIterative) {
      this.strongComponents = new GPUGraphAlgorithmsStronglyConnectedComponents({
        id: `${this.id}-strong-components`,
        topology: new GPUGraphAlgorithmsTopologyView({
          id: `${this.id}-topology`,
          vertexCount: this.nodeCount,
          directed: true,
          forward: {offsets: props.topology.offsets, neighbors: props.topology.neighbors}
        }),
        output: output.strongComponents!,
        iterations: props.strongComponentsIterations,
        converged: output.strongComponentsConverged
      });
    }
    if (hasResilienceOutput) {
      if (!props.reverseEdgeSlots)
        throw new Error('Undirected structure requires reverseEdgeSlots');
      validatePackedView(props.reverseEdgeSlots, ['uint32'], `${this.id} reverseEdgeSlots`);
      if (props.reverseEdgeSlots.length !== this.edgeCount)
        throw new Error('Reverse edge length mismatch');
    }
    const outputBuffers = Object.values(props.output)
      .filter(view => view !== undefined)
      .map(view => view.buffer);
    if (new Set(outputBuffers).size !== outputBuffers.length)
      throw new Error('Outputs must use distinct buffers');
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(output), [
      ...getGPUNetworkTopologyViews(props.topology),
      props.reverseEdgeSlots
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, nodeCount, edgeCount} = this;
    const {output, topology} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      ...getGPUNetworkTopologyViews(topology),
      props.reverseEdgeSlots,
      ...Object.values(output)
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const adjacency = [
      {name: 'offsets', view: topology.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: topology.neighbors, type: 'u32', access: 'read'}
    ] as const;
    const constants = `const NODE_COUNT: u32 = ${nodeCount}u; const EDGE_COUNT: u32 = ${edgeCount}u; const NONE: u32 = 0xffffffffu;`;
    if (this.strongComponents) {
      const {strongComponents} = this;
      nodes.push(...captureGraphCommandNodes(graph, () => strongComponents.addToGraph(graph)));
    } else if (output.strongComponents) {
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
    return nodes;
  }
}
