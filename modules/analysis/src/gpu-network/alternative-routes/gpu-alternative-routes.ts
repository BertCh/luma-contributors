// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  type GPUNetworkTopology,
  validateGPUNetworkTopology,
  getGPUNetworkTopologyViews
} from '../../analysis-core/network-topology';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validatePackedView,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs,
  validateCapacity
} from '../../analysis-core/kernels/gpu-contributor-utils';

export type GPUAlternativeRoutesProps = {
  id?: string;
  topology: GPUNetworkTopology;
  sources: GraphDataView<'uint32'>;
  targets: GraphDataView<'uint32'>;
  routeCount: number;
  /** Edge examinations per query. Default 100000; incomplete results are explicitly flagged. */
  maximumExpansions?: number;
  /** Pending simple paths per query. Default 1024; overflow stops with complete = 0. */
  maximumFrontierSize?: number;
  output: {
    /** query × routeCount × nodeCount, padded with 0xffffffff. */
    vertices: GraphDataView<'uint32'>;
    /** query × routeCount × nodeCount, CSR edge IDs padded with 0xffffffff. Last slot is padding. */
    edges: GraphDataView<'uint32'>;
    /** Number of vertices per route, zero for missing routes. */
    lengths: GraphDataView<'uint32'>;
    /** Path cost, infinity for missing routes. */
    costs: GraphDataView<'float32'>;
    /** Number of discovered routes retained, per query. */
    counts: GraphDataView<'uint32'>;
    /** 1 iff the requested top-k is certified or all routes exhausted; 0 means search stopped early. */
    complete: GraphDataView<'uint32'>;
  };
};

/** Bounded uniform-cost search over simple edge paths, independent queries on the GPU.
 * A storage-buffer min heap orders prefixes by float32 cost then lexicographic CSR edge IDs.
 * Nonnegative finite weights make each popped destination the next shortest simple route.
 * Search stops after k destinations, avoiding enumeration of more expensive alternatives.
 * The worst case remains exponential; maximumExpansions and maximumFrontierSize bound it.
 * Parallel edges are distinct routes; repeated vertices are prohibited, including zero-cost cycles.
 * Invalid endpoints yield an empty certified result; invalid weights/neighbors are impassable.
 * Source == destination returns the zero-edge route. Overflowed float32 path sums are impassable.
 * No CPU result readback or fallback is used. Scratch is 4 * queries * frontier * (2 + 2 * nodes).
 */
export class GPUAlternativeRoutes implements GPUCommandNodeProducer {
  readonly id: string;
  readonly nodeCount: number;
  readonly maximumExpansions: number;
  readonly maximumFrontierSize: number;
  constructor(readonly props: GPUAlternativeRoutesProps) {
    this.id = props.id ?? 'alternative-routes';
    this.nodeCount = validateGPUNetworkTopology(this.id, props.topology, {weights: true}).nodeCount;
    validateCapacity(this.id, 'nodeCount', this.nodeCount, 1, 256);
    validateCapacity(this.id, 'routeCount', props.routeCount, 1, 32);
    this.maximumFrontierSize = props.maximumFrontierSize ?? 1024;
    validateCapacity(this.id, 'maximumFrontierSize', this.maximumFrontierSize, 1, 65536);
    this.maximumExpansions = props.maximumExpansions ?? 100000;
    validateCapacity(this.id, 'maximumExpansions', this.maximumExpansions, 1, 10000000);
    for (const view of [props.sources, props.targets])
      validatePackedView(view, ['uint32'], this.id);
    if (props.sources.length !== props.targets.length || props.sources.length < 1)
      throw new Error('Route queries must have equal positive lengths');
    const routes = props.sources.length * props.routeCount;
    for (const [name, view] of Object.entries(props.output)) {
      validatePackedView(view, [name === 'costs' ? 'float32' : 'uint32'], this.id);
      const expected =
        name === 'vertices' || name === 'edges'
          ? routes * this.nodeCount
          : name === 'counts' || name === 'complete'
            ? props.sources.length
            : routes;
      if (view.length !== expected) throw new Error('Route output length mismatch');
    }
    const outputBuffers = Object.values(props.output)
      .filter(view => view !== undefined)
      .map(view => view.buffer);
    if (new Set(outputBuffers).size !== outputBuffers.length)
      throw new Error('Outputs must use distinct buffers');
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      ...getGPUNetworkTopologyViews(props.topology),
      props.sources,
      props.targets
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, nodeCount} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      ...getGPUNetworkTopologyViews(props.topology),
      props.sources,
      props.targets,
      ...Object.values(props.output)
    ]);
    // Pack query endpoints and output metadata to keep the traversal within eight storage bindings.
    // Metadata outputs are separated by a second pass, while path arrays are caller-owned.
    const metadata = createTransientView(
      graph,
      `${id}-metadata`,
      'uint32',
      props.sources.length * (4 + props.routeCount * 2)
    );
    const frontierWords = props.sources.length * this.maximumFrontierSize * (2 + 2 * nodeCount);
    validateCapacity(
      id,
      'frontierWords',
      frontierWords,
      1,
      Math.min(0xffffffff, Math.floor(graph.device.limits.maxStorageBufferBindingSize / 4))
    );
    const frontier = createTransientView(graph, `${id}-frontier`, 'uint32', frontierWords);
    const stride = 4 + props.routeCount * 2;
    const constants = `const NODE_COUNT: u32 = ${nodeCount}u; const EDGE_COUNT: u32 = ${props.topology.neighbors.length}u; const ROUTE_COUNT: u32 = ${props.routeCount}u; const STRIDE: u32 = ${stride}u; const NONE: u32 = 0xffffffffu;`;
    return [
      createWGSLKernelNode(graph, {
        id: `${id}-queries`,
        operation: 'GPUAlternativeRoutes',
        variant: 'queries',
        invocationCount: props.sources.length,
        bindings: [
          {name: 'sources', view: props.sources, type: 'u32', access: 'read'},
          {name: 'targets', view: props.targets, type: 'u32', access: 'read'},
          {name: 'metadata', view: metadata, type: 'u32', access: 'read_write'}
        ],
        declarations: constants,
        body: `metadata[metadataOffset + index * STRIDE] = sources[sourcesOffset + index]; metadata[metadataOffset + index * STRIDE + 1u] = targets[targetsOffset + index];`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-search`,
        operation: 'GPUAlternativeRoutes',
        variant: 'search',
        invocationCount: props.sources.length,
        bindings: [
          {name: 'offsets', view: props.topology.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: props.topology.neighbors, type: 'u32', access: 'read'},
          {name: 'weights', view: props.topology.weights, type: 'f32', access: 'read'},
          {name: 'metadata', view: metadata, type: 'u32', access: 'read_write'},
          {name: 'vertices', view: props.output.vertices, type: 'u32', access: 'read_write'},
          {name: 'edges', view: props.output.edges, type: 'u32', access: 'read_write'},
          {name: 'frontier', view: frontier, type: 'u32', access: 'read_write'}
        ],
        declarations:
          constants +
          `
const RECORD_SIZE: u32 = ${2 + 2 * nodeCount}u;
const FRONTIER_SIZE: u32 = ${this.maximumFrontierSize}u;
fn precedes(first: u32, second: u32) -> bool {
  let firstCost = bitcast<f32>(frontier[first]);
  let secondCost = bitcast<f32>(frontier[second]);
  if (firstCost != secondCost) { return firstCost < secondCost; }
  let firstLength = frontier[first + 1u]; let secondLength = frontier[second + 1u];
  for (var slot = 0u; slot + 1u < min(firstLength, secondLength); slot++) {
    let firstEdge = frontier[first + 2u + NODE_COUNT + slot];
    let secondEdge = frontier[second + 2u + NODE_COUNT + slot];
    if (firstEdge != secondEdge) { return firstEdge < secondEdge; }
  }
  return firstLength < secondLength;
}
fn swapRecords(first: u32, second: u32) {
  for (var slot = 0u; slot < RECORD_SIZE; slot++) {
    let temporary = frontier[first + slot];
    frontier[first + slot] = frontier[second + slot]; frontier[second + slot] = temporary;
  }
}`,
        body: `let base = metadataOffset + index * STRIDE;
  let source = metadata[base]; let destination = metadata[base + 1u];
  let pathBase = index * ROUTE_COUNT * NODE_COUNT;
  for (var slot = 0u; slot < ROUTE_COUNT * NODE_COUNT; slot++) { vertices[verticesOffset + pathBase + slot] = NONE; edges[edgesOffset + pathBase + slot] = NONE; }
  for (var route = 0u; route < ROUTE_COUNT; route++) { metadata[base + 4u + route * 2u] = 0u; metadata[base + 5u + route * 2u] = 0x7f800000u; }
  metadata[base + 2u] = 0u; metadata[base + 3u] = 1u;
  if (source >= NODE_COUNT || destination >= NODE_COUNT) { return; }
  var path: array<u32, ${nodeCount}>; var pathEdges: array<u32, ${nodeCount}>;
  let frontierBase = frontierOffset + index * FRONTIER_SIZE * RECORD_SIZE;
  frontier[frontierBase] = 0u; frontier[frontierBase + 1u] = 1u;
  frontier[frontierBase + 2u] = source;
  var pending = 1u; var count = 0u; var expansions = 0u; var stopped = false;
  loop {
    if (pending == 0u || count == ROUTE_COUNT || stopped) { break; }
    let cost = bitcast<f32>(frontier[frontierBase]);
    let length = frontier[frontierBase + 1u];
    for (var slot = 0u; slot < length; slot++) {
      path[slot] = frontier[frontierBase + 2u + slot];
      if (slot + 1u < length) { pathEdges[slot] = frontier[frontierBase + 2u + NODE_COUNT + slot]; }
    }
    pending--;
    if (pending > 0u) {
      swapRecords(frontierBase, frontierBase + pending * RECORD_SIZE);
      var parent = 0u;
      loop {
        var child = parent * 2u + 1u;
        if (child >= pending) { break; }
        if (child + 1u < pending && precedes(frontierBase + (child + 1u) * RECORD_SIZE, frontierBase + child * RECORD_SIZE)) { child++; }
        if (!precedes(frontierBase + child * RECORD_SIZE, frontierBase + parent * RECORD_SIZE)) { break; }
        swapRecords(frontierBase + child * RECORD_SIZE, frontierBase + parent * RECORD_SIZE); parent = child;
      }
    }
    let vertex = path[length - 1u];
    if (vertex == destination) {
      metadata[base + 4u + count * 2u] = length;
      metadata[base + 5u + count * 2u] = bitcast<u32>(cost);
      for (var slot = 0u; slot < length; slot++) {
        vertices[verticesOffset + pathBase + count * NODE_COUNT + slot] = path[slot];
        if (slot + 1u < length) { edges[edgesOffset + pathBase + count * NODE_COUNT + slot] = pathEdges[slot]; }
      }
      count++; continue;
    }
    for (var edge = min(offsets[offsetsOffset + vertex], EDGE_COUNT); edge < min(offsets[offsetsOffset + vertex + 1u], EDGE_COUNT); edge++) {
      if (expansions >= ${this.maximumExpansions}u) { stopped = true; break; }
      expansions++;
      let neighbor = neighbors[neighborsOffset + edge]; let weight = weights[weightsOffset + edge];
      if (neighbor >= NODE_COUNT || !(weight >= 0.0) || (bitcast<u32>(weight) & 0x7f800000u) == 0x7f800000u) { continue; }
      var repeated = false;
      for (var slot = 0u; slot < length; slot++) { if (path[slot] == neighbor) { repeated = true; break; } }
      if (repeated || length == NODE_COUNT) { continue; }
      let nextCost = cost + weight;
      if ((bitcast<u32>(nextCost) & 0x7f800000u) == 0x7f800000u) { continue; }
      if (pending == FRONTIER_SIZE) { stopped = true; break; }
      let record = frontierBase + pending * RECORD_SIZE;
      frontier[record] = bitcast<u32>(nextCost); frontier[record + 1u] = length + 1u;
      for (var slot = 0u; slot < length; slot++) {
        frontier[record + 2u + slot] = path[slot];
        if (slot + 1u < length) { frontier[record + 2u + NODE_COUNT + slot] = pathEdges[slot]; }
      }
      frontier[record + 2u + length] = neighbor;
      frontier[record + 2u + NODE_COUNT + length - 1u] = edge;
      var child = pending; pending++;
      loop {
        if (child == 0u) { break; }
        let parent = (child - 1u) / 2u;
        if (!precedes(frontierBase + child * RECORD_SIZE, frontierBase + parent * RECORD_SIZE)) { break; }
        swapRecords(frontierBase + child * RECORD_SIZE, frontierBase + parent * RECORD_SIZE); child = parent;
      }
    }
  }
  if (stopped) { metadata[base + 3u] = 0u; }
  metadata[base + 2u] = count;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-publish`,
        operation: 'GPUAlternativeRoutes',
        variant: 'publish',
        invocationCount: props.sources.length,
        bindings: [
          {name: 'metadata', view: metadata, type: 'u32', access: 'read'},
          {name: 'lengths', view: props.output.lengths, type: 'u32', access: 'read_write'},
          {name: 'costs', view: props.output.costs, type: 'f32', access: 'read_write'},
          {name: 'counts', view: props.output.counts, type: 'u32', access: 'read_write'},
          {name: 'complete', view: props.output.complete, type: 'u32', access: 'read_write'}
        ],
        declarations: constants,
        body: `let base = metadataOffset + index * STRIDE;
  counts[countsOffset + index] = metadata[base + 2u]; complete[completeOffset + index] = metadata[base + 3u];
  for (var route = 0u; route < ROUTE_COUNT; route++) {
    lengths[lengthsOffset + index * ROUTE_COUNT + route] = metadata[base + 4u + route * 2u];
    costs[costsOffset + index * ROUTE_COUNT + route] = bitcast<f32>(metadata[base + 5u + route * 2u]);
  }`
      })
    ];
  }
}
