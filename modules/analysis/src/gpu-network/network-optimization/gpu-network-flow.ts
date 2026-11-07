// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validatePackedView,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';

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
 * At most 128 nodes and 1024 edges; intended for small exact scenarios.
 */
export type GPUNetworkFlowProps = {
  id?: string;
  nodeCount: number;
  edges: GraphDataView<'uint32x2'>;
  capacities: GraphDataView<'uint32'>;
  /** Nonnegative integer original-edge costs. Omit for maximum flow. */
  costs?: GraphDataView<'uint32'>;
  /** Source, sink, required flow; 0xffffffff required flow means maximize. */
  parameters: GraphDataView<'uint32'>;
  maximumAugmentations?: number;
  output: {
    flow: GraphDataView<'uint32'>;
    /** Residual source-reachable vertices, one word per node. */
    cut: GraphDataView<'uint32'>;
    /** Eight words, indexed by GPU_NETWORK_FLOW_WORD; cost is f32 bits. */
    summary: GraphDataView<'uint32'>;
    /** Optional exact unsigned integer objective [low32,high32]. Summary cost remains f32. */
    exactCost?: GraphDataView<'uint32x2'>;
  };
};

/**
 * Successive shortest augmenting paths with explicit reverse arcs. Computation stays on the
 * graph; no submission/readback. The sequential bounded solver is a correctness baseline,
 * not a scalable push/relabel implementation. Each augmentation preserves feasibility.
 * Total capacities must be <= 0x00ffffff; costs <= 32767. Invalid input has valid=0.
 * Exhaustion has complete=0, never a claim of infeasibility or optimality.
 */
export class GPUNetworkFlow implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUNetworkFlowProps;
  constructor(props: GPUNetworkFlowProps) {
    this.props = props;
    this.id = props.id ?? 'network-flow';
    assert(Number.isInteger(props.nodeCount) && props.nodeCount >= 2 && props.nodeCount <= 128);
    assert(props.edges.length <= 1024);
    assert(
      Number.isInteger(props.maximumAugmentations ?? 1024) &&
        (props.maximumAugmentations ?? 1024) >= 0 &&
        (props.maximumAugmentations ?? 1024) <= 65536
    );
    validatePackedView(props.edges, ['uint32x2'], this.id);
    for (const view of [
      props.capacities,
      props.costs,
      props.parameters,
      props.output.flow,
      props.output.cut,
      props.output.summary
    ]) {
      if (view) validatePackedView(view, ['uint32'], this.id);
    }
    assert(
      props.capacities.length === props.edges.length &&
        (!props.costs || props.costs.length === props.edges.length)
    );
    assert(
      props.parameters.length === 3 &&
        props.output.flow.length === props.edges.length &&
        props.output.cut.length === props.nodeCount &&
        props.output.summary.length === 8
    );
    if (props.output.exactCost) {
      validatePackedView(props.output.exactCost, ['uint32x2'], this.id);
      assert(props.output.exactCost.length === 1);
    }
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.edges,
      props.capacities,
      props.costs,
      props.parameters
    ]);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.edges,
      props.capacities,
      props.costs,
      props.parameters,
      ...Object.values(props.output)
    ]);
    return [
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
            ? [{name: 'costs', view: props.costs, type: 'u32' as const, access: 'read' as const}]
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
  }
}
