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
  createFillNode,
  createWGSLKernelNode,
  getWGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateGPUNetworkTopology,
  type GPUNetworkTopology
} from '../../analysis-core/network-topology';

const OPERATION = 'GPUNetworkPotentialPathArea';
const FINITE_WGSL = `fn finite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }`;

/** Words of `output.summary`: nodes in the prism, edges in the prism. */
export const GPU_NETWORK_POTENTIAL_PATH_AREA_SUMMARY_LENGTH = 2;

/**
 * Properties for {@link GPUNetworkPotentialPathArea}.
 *
 * Compile-time: the topology shape and which outputs exist. Per-frame: every input and the budget.
 */
export type GPUNetworkPotentialPathAreaProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-potential-path-area'`. */
  id?: string;
  /** Directed network whose `weights` are the travel costs the two trees were computed with. */
  topology: GPUNetworkTopology;
  /**
   * Cost from the origin anchor `a` to every node, one row per node, `+Infinity` when unreached:
   * the `costs` of a `GPUNetworkReachability` run from `a` on `topology`.
   */
  forwardCosts: GraphDataView<'float32'>;
  /**
   * Cost from every node to the destination anchor `b`, `+Infinity` when none: the `costs` of a
   * `GPUNetworkReachability` run from `b` on the reversed network (every edge turned around, same
   * weights).
   */
  reverseCosts: GraphDataView<'float32'>;
  /** One row: the time budget `t_b - t_a`, in the cost unit of the weights. */
  budget: GraphDataView<'float32'>;
  output: {
    /** One row per edge slot: 1 when some path `a -> u -> v -> b` through the edge fits the budget. */
    edgeInPrism: GraphDataView<'uint32'>;
    /**
     * One row per edge slot: `budget - (forward[u] + weight + reverse[v])`, the time to spare when
     * the edge is on a path from `a` to `b`. Negative or `-Infinity`/NaN rows are outside the
     * prism (rows of an impassable or unreachable edge are `-Infinity`).
     */
    edgeSlack: GraphDataView<'float32'>;
    /** One row per node: 1 when `forward[v] + reverse[v] <= budget`. */
    nodeInPrism: GraphDataView<'uint32'>;
    /** Optional {@link GPU_NETWORK_POTENTIAL_PATH_AREA_SUMMARY_LENGTH} rows: node and edge counts. */
    summary?: GraphDataView<'uint32'>;
  };
};

/**
 * Network potential path area of a space-time prism: the nodes and directed edges a traveller who
 * leaves `a` and must reach `b` within a budget could have used.
 *
 * An edge `(u, v)` is in the prism when `forward[u] + weight + reverse[v] <= budget`, a node when
 * `forward[v] + reverse[v] <= budget` (Miller 1991 and Downs and Horner's network prisms; the
 * reference is a pair of NetworkX `single_source_dijkstra_path_length` calls). The edge is whole:
 * a directed edge is entered at `u` and left at `v`, so every point on it shares the slack. An
 * undirected street is two directed edges, each tested.
 *
 * One thread per node visits that node's out-edges (a bounded neighbourhood), then one thread per edge counts the members; the trees come from
 * existing reachability runs, so no search is repeated here. Costs add in float32: an edge within
 * a few float32 roundings of the budget may fall on either side.
 */
export class GPUNetworkPotentialPathArea implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUNetworkPotentialPathAreaProps;
  /** Number of nodes. */
  readonly nodeCount: number;
  /** Number of edge slots. */
  readonly edgeCount: number;

  constructor(props: GPUNetworkPotentialPathAreaProps) {
    this.id = props.id ?? 'network-potential-path-area';
    this.props = props;
    const {id} = this;
    const topology = validateGPUNetworkTopology(id, props.topology, {weights: true});
    this.nodeCount = topology.nodeCount;
    this.edgeCount = topology.edgeCount;
    for (const [name, view] of [
      ['forwardCosts', props.forwardCosts],
      ['reverseCosts', props.reverseCosts],
      ['budget', props.budget],
      ['output.edgeSlack', props.output.edgeSlack]
    ] as const) {
      validatePackedView(view, ['float32'], `${id} ${name}`);
    }
    for (const [name, view] of [
      ['output.edgeInPrism', props.output.edgeInPrism],
      ['output.nodeInPrism', props.output.nodeInPrism],
      ['output.summary', props.output.summary]
    ] as const) {
      if (view) validatePackedView(view, ['uint32'], `${id} ${name}`);
    }
    validateViewLength(id, 'forwardCosts', props.forwardCosts, this.nodeCount);
    validateViewLength(id, 'reverseCosts', props.reverseCosts, this.nodeCount);
    validateViewLength(id, 'budget', props.budget, 1);
    validateViewLength(id, 'output.edgeInPrism', props.output.edgeInPrism, this.edgeCount);
    validateViewLength(id, 'output.edgeSlack', props.output.edgeSlack, this.edgeCount);
    validateViewLength(id, 'output.nodeInPrism', props.output.nodeInPrism, this.nodeCount);
    if (props.output.summary) {
      validateViewLength(
        id,
        'output.summary',
        props.output.summary,
        GPU_NETWORK_POTENTIAL_PATH_AREA_SUMMARY_LENGTH
      );
    }
    validateGraphOutputsDisjointFromInputs(id, Object.values(props.output), [
      props.topology.offsets,
      props.topology.neighbors,
      props.topology.weights,
      props.forwardCosts,
      props.reverseCosts,
      props.budget
    ]);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, nodeCount} = this;
    const {topology, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      topology.offsets,
      topology.neighbors,
      topology.weights,
      props.forwardCosts,
      props.reverseCosts,
      props.budget,
      ...Object.values(output)
    ]);
    const summary =
      output.summary ??
      createTransientView(
        graph,
        `${id}-summary`,
        'uint32',
        GPU_NETWORK_POTENTIAL_PATH_AREA_SUMMARY_LENGTH
      );
    return [
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-summary`,
        operation: OPERATION,
        view: summary,
        type: 'u32',
        value: '0u'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-nodes`,
        operation: OPERATION,
        variant: 'node-prism',
        invocationCount: nodeCount,
        bindings: [
          {name: 'forward', view: props.forwardCosts, type: 'f32', access: 'read'},
          {name: 'reverse', view: props.reverseCosts, type: 'f32', access: 'read'},
          {name: 'budget', view: props.budget, type: 'f32', access: 'read'},
          {name: 'nodeInPrism', view: output.nodeInPrism, type: 'u32', access: 'read_write'},
          getWGSLKernelBinding('summary', summary, 'atomic')
        ],
        declarations: FINITE_WGSL,
        body: `let limit = budget[budgetOffset];
let fromOrigin = forward[forwardOffset + index];
let toDestination = reverse[reverseOffset + index];
let nodeFits = finite(limit) && finite(fromOrigin) && finite(toDestination) && fromOrigin + toDestination <= limit;
nodeInPrism[nodeInPrismOffset + index] = select(0u, 1u, nodeFits);
if (nodeFits) { atomicAdd(&summary[summaryOffset], 1u); }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-edges`,
        operation: OPERATION,
        variant: 'edge-prism',
        invocationCount: nodeCount,
        bindings: [
          {name: 'offsets', view: topology.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: topology.neighbors, type: 'u32', access: 'read'},
          {name: 'weights', view: topology.weights, type: 'f32', access: 'read'},
          {name: 'forward', view: props.forwardCosts, type: 'f32', access: 'read'},
          {name: 'reverse', view: props.reverseCosts, type: 'f32', access: 'read'},
          {name: 'budget', view: props.budget, type: 'f32', access: 'read'},
          {name: 'edgeInPrism', view: output.edgeInPrism, type: 'u32', access: 'read_write'},
          {name: 'edgeSlack', view: output.edgeSlack, type: 'f32', access: 'read_write'}
        ],
        declarations: FINITE_WGSL,
        // One thread per node visits that node's out-edges (a bounded neighbourhood).
        body: `var infinityBits = 0xff800000u; let negativeInfinity = bitcast<f32>(infinityBits);
let limit = budget[budgetOffset];
let fromOrigin = forward[forwardOffset + index];
for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
  let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot];
  var slack = negativeInfinity;
  if (neighbor < ${nodeCount}u && finite(weight) && weight >= 0.0 && finite(fromOrigin) && finite(limit)) {
    let arrival = reverse[reverseOffset + neighbor];
    if (finite(arrival)) { slack = limit - (fromOrigin + weight + arrival); }
  }
  edgeSlack[edgeSlackOffset + slot] = slack;
  edgeInPrism[edgeInPrismOffset + slot] = select(0u, 1u, finite(slack) && slack >= 0.0);
}`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-edge-count`,
        operation: OPERATION,
        variant: 'edge-count',
        invocationCount: this.edgeCount,
        bindings: [
          {name: 'edgeInPrism', view: output.edgeInPrism, type: 'u32', access: 'read'},
          getWGSLKernelBinding('summary', summary, 'atomic')
        ],
        body: `if (edgeInPrism[edgeInPrismOffset + index] != 0u) { atomicAdd(&summary[summaryOffset + 1u], 1u); }`
      })
    ];
  }
}
