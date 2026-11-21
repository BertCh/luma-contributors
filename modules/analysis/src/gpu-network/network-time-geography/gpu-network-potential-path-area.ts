// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {validateNetworkViewBindings, validateNetworkWritableViews} from '../network-view-utils';

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
  validateSinglePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateGPUNetworkTopology,
  type GPUNetworkTopology
} from '../../analysis-core/contracts/network-topology';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';

const OPERATION = 'GPUNetworkPotentialPathArea';

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
  /** Optional one-word certificate for the forward tree: nonzero means complete. */
  forwardComplete?: GraphDataView<'uint32'>;
  /** Optional one-word certificate for the reverse tree: nonzero means complete. */
  reverseComplete?: GraphDataView<'uint32'>;
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
    /** Optional one-word evidence. Incomplete trees suppress all dependent results. */
    status?: {complete: GraphDataView<'uint32'>; invalidInput: GraphDataView<'uint32'>};
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
 * Optional forwardComplete/reverseComplete evidence suppresses membership, slack and counts
 * when either supplied tree is incomplete. Omitted evidence preserves standalone tree inputs.
 * Malformed CSR suppresses results and reports status.invalidInput.
 * An edge within a few float32 roundings of the budget may fall on either side.
 */
export class GPUNetworkPotentialPathArea implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUNetworkPotentialPathAreaProps;
  /** Number of nodes. */
  readonly nodeCount: number;
  /** Number of edge slots. */
  readonly edgeCount: number;

  constructor(originalProps: GPUNetworkPotentialPathAreaProps) {
    const settings = snapshotConstructionFields(originalProps, [
      'id',
      'topology',
      'forwardCosts',
      'reverseCosts',
      'forwardComplete',
      'reverseComplete',
      'budget',
      'output'
    ]);
    const outputSettings = snapshotConstructionFields(settings.output, [
      'edgeSlack',
      'edgeInPrism',
      'nodeInPrism',
      'summary',
      'status'
    ]);
    const props = {
      ...settings,
      topology: snapshotConstructionFields(settings.topology, [
        'offsets',
        'neighbors',
        'weights',
        'nodePositions'
      ]),
      output: {
        ...outputSettings,
        ...(outputSettings.status
          ? {
              status: snapshotConstructionFields(outputSettings.status, [
                'complete',
                'invalidInput'
              ])
            }
          : {})
      }
    };
    this.id = props.id ?? 'network-potential-path-area';
    this.props = props;
    validateNetworkWritableViews(this.id, {
      ...props.output,
      status: undefined,
      ...props.output.status
    });
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
      validateSinglePackedView(view, ['float32'], `${id} ${name}`);
    }
    for (const [name, view] of [
      ['output.edgeInPrism', props.output.edgeInPrism],
      ['output.nodeInPrism', props.output.nodeInPrism],
      ['forwardComplete', props.forwardComplete],
      ['reverseComplete', props.reverseComplete],
      ['output.complete', props.output.status?.complete],
      ['output.invalidInput', props.output.status?.invalidInput],
      ['output.summary', props.output.summary]
    ] as const) {
      if (view) validateSinglePackedView(view, ['uint32'], `${id} ${name}`);
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
    for (const [name, view] of Object.entries({
      forwardComplete: props.forwardComplete,
      reverseComplete: props.reverseComplete,
      ...props.output.status
    })) {
      if (view) validateViewLength(id, name, view, 1);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        props.output.edgeInPrism,
        props.output.edgeSlack,
        props.output.nodeInPrism,
        props.output.summary,
        ...Object.values(props.output.status ?? {})
      ],
      [
        props.topology.offsets,
        props.topology.neighbors,
        props.topology.weights,
        props.forwardCosts,
        props.reverseCosts,
        props.forwardComplete,
        props.reverseComplete,
        props.budget
      ]
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, nodeCount} = this;
    const {topology, output} = props;
    validateNetworkViewBindings(id, graph, [
      topology.offsets,
      topology.neighbors,
      topology.weights,
      props.forwardCosts,
      props.reverseCosts,
      props.forwardComplete,
      props.reverseComplete,
      props.budget,
      output.edgeInPrism,
      output.edgeSlack,
      output.nodeInPrism,
      output.summary,
      ...Object.values(output.status ?? {})
    ]);
    validateNetworkViewBindings(id, graph, [], {
      summary: GPU_NETWORK_POTENTIAL_PATH_AREA_SUMMARY_LENGTH * 4
    });
    const summary =
      output.summary ??
      createTransientView(
        graph,
        `${id}-summary`,
        'uint32',
        GPU_NETWORK_POTENTIAL_PATH_AREA_SUMMARY_LENGTH
      );
    const certifiedForward = createTransientView(
      graph,
      `${id}-certified-forward`,
      'float32',
      nodeCount
    );
    const malformed = createTransientView(graph, `${id}-malformed`, 'uint32', 1);
    const evidenceBindings = [
      {name: 'malformed', view: malformed, type: 'u32' as const, access: 'read' as const},
      ...(props.forwardComplete
        ? [
            {
              name: 'forwardComplete',
              view: props.forwardComplete,
              type: 'u32' as const,
              access: 'read' as const
            }
          ]
        : []),
      ...(props.reverseComplete
        ? [
            {
              name: 'reverseComplete',
              view: props.reverseComplete,
              type: 'u32' as const,
              access: 'read' as const
            }
          ]
        : [])
    ];
    const complete = `malformed[malformedOffset] == 0u${props.forwardComplete ? ' && forwardComplete[forwardCompleteOffset] != 0u' : ''}${props.reverseComplete ? ' && reverseComplete[reverseCompleteOffset] != 0u' : ''}`;
    return [
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-malformed`,
        operation: OPERATION,
        view: malformed,
        type: 'u32',
        value: '0u'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-validate-topology`,
        operation: OPERATION,
        variant: 'validate-topology',
        invocationCount: nodeCount,
        bindings: [
          {name: 'offsets', view: topology.offsets, type: 'u32', access: 'read'},
          getWGSLKernelBinding('malformed', malformed, 'atomic')
        ],
        body: `let start = offsets[offsetsOffset + index]; let end = offsets[offsetsOffset + index + 1u];
if (start > end || end > ${this.edgeCount}u) { atomicOr(&malformed[malformedOffset], 1u); }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-certify-forward`,
        operation: OPERATION,
        variant: 'certify-forward',
        invocationCount: nodeCount,
        bindings: [
          ...evidenceBindings,
          {name: 'source', view: props.forwardCosts, type: 'f32', access: 'read'},
          {name: 'destination', view: certifiedForward, type: 'f32', access: 'read_write'}
        ],
        body: `var missingBits = 0xff800000u;
destination[destinationOffset + index] = select(bitcast<f32>(missingBits), source[sourceOffset + index], ${complete});`
      }),
      ...(output.status
        ? [
            createWGSLKernelNode<Parameters>(graph, {
              id: `${id}-publish-status`,
              operation: OPERATION,
              variant: 'publish-status',
              invocationCount: nodeCount,
              bindings: [
                ...evidenceBindings,
                {
                  name: 'complete',
                  view: output.status.complete,
                  type: 'u32' as const,
                  access: 'read_write' as const
                },
                {
                  name: 'invalidInput',
                  view: output.status.invalidInput,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ],
              body: `if (index == 0u) { complete[completeOffset] = select(0u, 1u, ${complete}); invalidInput[invalidInputOffset] = malformed[malformedOffset]; }`
            })
          ]
        : []),
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
          {name: 'forward', view: certifiedForward, type: 'f32', access: 'read'},
          {name: 'reverse', view: props.reverseCosts, type: 'f32', access: 'read'},
          {name: 'budget', view: props.budget, type: 'f32', access: 'read'},
          {name: 'nodeInPrism', view: output.nodeInPrism, type: 'u32', access: 'read_write'},
          getWGSLKernelBinding('summary', summary, 'atomic')
        ],
        declarations: IS_FINITE_VALUE_WGSL,
        body: `let limit = budget[budgetOffset];
let fromOrigin = forward[forwardOffset + index];
let toDestination = reverse[reverseOffset + index];
let nodeFits = isFiniteValue(limit) && isFiniteValue(fromOrigin) && isFiniteValue(toDestination) && fromOrigin + toDestination <= limit;
nodeInPrism[nodeInPrismOffset + index] = select(0u, 1u, nodeFits);
if (nodeFits) { atomicAdd(&summary[summaryOffset], 1u); }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-clear-edge-outputs`,
        operation: OPERATION,
        variant: 'clear-edge-outputs',
        invocationCount: this.edgeCount,
        bindings: [
          {name: 'edgeInPrism', view: output.edgeInPrism, type: 'u32', access: 'read_write'},
          {name: 'edgeSlack', view: output.edgeSlack, type: 'f32', access: 'read_write'}
        ],
        body: `var infinityBits = 0xff800000u;
edgeSlack[edgeSlackOffset + index] = bitcast<f32>(infinityBits);
edgeInPrism[edgeInPrismOffset + index] = 0u;`
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
          {name: 'forward', view: certifiedForward, type: 'f32', access: 'read'},
          {name: 'reverse', view: props.reverseCosts, type: 'f32', access: 'read'},
          {name: 'budget', view: props.budget, type: 'f32', access: 'read'},
          {name: 'edgeInPrism', view: output.edgeInPrism, type: 'u32', access: 'read_write'},
          {name: 'edgeSlack', view: output.edgeSlack, type: 'f32', access: 'read_write'}
        ],
        declarations: IS_FINITE_VALUE_WGSL,
        // One thread per node visits that node's out-edges (a bounded neighbourhood).
        body: `var infinityBits = 0xff800000u; let negativeInfinity = bitcast<f32>(infinityBits);
let limit = budget[budgetOffset];
let fromOrigin = forward[forwardOffset + index];
if (!isFiniteValue(fromOrigin)) { return; }
for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
  let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot];
  var slack = negativeInfinity;
  if (neighbor < ${nodeCount}u && isFiniteValue(weight) && weight >= 0.0 && isFiniteValue(fromOrigin) && isFiniteValue(limit)) {
    let arrival = reverse[reverseOffset + neighbor];
    if (isFiniteValue(arrival)) { slack = limit - (fromOrigin + weight + arrival); }
  }
  edgeSlack[edgeSlackOffset + slot] = slack;
  edgeInPrism[edgeInPrismOffset + slot] = select(0u, 1u, isFiniteValue(slack) && slack >= 0.0);
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
