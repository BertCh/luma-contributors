// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateSinglePackedView,
  validateRequirement,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';

/** Largest edge count: each query keeps one private distance array with an entry per edge. */
const MAXIMUM_EDGE_COUNT = 128;
/** Largest turn count: bounds the work of each relaxation round of a query. */
const MAXIMUM_TURN_COUNT = 4096;
/** Largest `maximumRelaxations`: bounds the relaxation loop of a query. */
const MAXIMUM_RELAXATIONS = 65536;

export type GPUNetworkPointRoutingProps = {
  id?: string;
  /** Allowed directed [incomingEdge, outgoingEdge] turns, normally from an explicit line graph. */
  turns: GraphDataView<'uint32x2'>;
  costs: GraphDataView<'float32'>;
  /** [sourceEdge, targetEdge], preserving parallel-edge identity. */
  queries: GraphDataView<'uint32x2'>;
  /** [sourceFraction, targetFraction], measured in the selected edge direction. */
  fractions: GraphDataView<'float32x2'>;
  maximumRelaxations?: number;
  output: {
    costs: GraphDataView<'float32'>;
    /** Per query: 0 complete, 1 invalid data, 2 iteration exhaustion. */
    status: GraphDataView<'uint32'>;
    /** Row-major query by edge predecessors; a returning same-edge route records its final incoming edge at source. */
    predecessorEdges: GraphDataView<'uint32'>;
  };
};

/**
 * Batched static routing between directed edge positions, with explicit incoming turn state.
 * Source pays its remaining fraction and destination its leading fraction; forward same-edge travel
 * competes with legal cycles. Reverse travel needs a separately identified reverse edge.
 * Endpoint fractions retain the chosen directed-edge state. At most 128 edges and 4096 turns.
 */
export class GPUNetworkPointRouting implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUNetworkPointRoutingProps;
  constructor(props: GPUNetworkPointRoutingProps) {
    this.id = props.id ?? 'network-point-routing';
    this.props = props;
    const {id} = this;
    validateCapacity(id, 'costs rows', props.costs.length, 1, MAXIMUM_EDGE_COUNT);
    validateCapacity(id, 'turns rows', props.turns.length, 0, MAXIMUM_TURN_COUNT);
    validateCapacity(
      id,
      'maximumRelaxations',
      props.maximumRelaxations ?? props.costs.length,
      0,
      MAXIMUM_RELAXATIONS
    );
    validateSinglePackedView(props.turns, ['uint32x2'], this.id);
    validateSinglePackedView(props.queries, ['uint32x2'], this.id);
    validateSinglePackedView(props.fractions, ['float32x2'], this.id);
    for (const view of [props.costs, props.output.costs])
      validateSinglePackedView(view, ['float32'], this.id);
    for (const view of [props.output.status, props.output.predecessorEdges])
      validateSinglePackedView(view, ['uint32'], this.id);
    validateRequirement(
      id,
      props.fractions.length === props.queries.length,
      'fractions must hold one row per query'
    );
    validateRequirement(
      id,
      props.output.costs.length === props.queries.length,
      'output.costs must hold one row per query'
    );
    validateRequirement(
      id,
      props.output.status.length === props.queries.length,
      'output.status must hold one row per query'
    );
    validateRequirement(
      id,
      props.output.predecessorEdges.length === props.queries.length * props.costs.length,
      'output.predecessorEdges must hold costs rows per query'
    );
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.turns,
      props.costs,
      props.queries,
      props.fractions
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.turns,
      props.costs,
      props.queries,
      props.fractions,
      ...Object.values(props.output)
    ]);
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id,
        operation: 'GPUNetworkPointRouting',
        variant: 'directed-edge-positions',
        invocationCount: props.queries.length,
        workgroupSize: 32,
        bindings: [
          {name: 'turns', view: props.turns, type: 'u32', access: 'read'},
          {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
          {name: 'queries', view: props.queries, type: 'u32', access: 'read'},
          {name: 'fractions', view: props.fractions, type: 'f32', access: 'read'},
          {name: 'outputCosts', view: props.output.costs, type: 'f32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'},
          {
            name: 'predecessors',
            view: props.output.predecessorEdges,
            type: 'u32',
            access: 'read_write'
          }
        ],
        declarations: `const EDGE_COUNT: u32 = ${props.costs.length}u;
${IS_FINITE_VALUE_WGSL}`,
        body: `var infinityBits = 0x7f800000u; let infinity = bitcast<f32>(infinityBits);
status[statusOffset + index] = 1u; outputCosts[outputCostsOffset + index] = infinity;
var distances: array<f32, ${props.costs.length}>;
for (var edge = 0u; edge < EDGE_COUNT; edge++) { distances[edge] = infinity; predecessors[predecessorsOffset + index * EDGE_COUNT + edge] = 0xffffffffu; }
let source = queries[queriesOffset + 2u * index]; let destination = queries[queriesOffset + 2u * index + 1u];
let sourceFraction = fractions[fractionsOffset + 2u * index]; let targetFraction = fractions[fractionsOffset + 2u * index + 1u];
if (source >= EDGE_COUNT || destination >= EDGE_COUNT || !isFiniteValue(sourceFraction) || !isFiniteValue(targetFraction) || sourceFraction < 0.0 || sourceFraction > 1.0 || targetFraction < 0.0 || targetFraction > 1.0) { return; }
for (var edge = 0u; edge < EDGE_COUNT; edge++) { if (!isFiniteValue(costs[costsOffset + edge]) || costs[costsOffset + edge] < 0.0) { return; } }
for (var turn = 0u; turn < ${props.turns.length}u; turn++) { if (turns[turnsOffset + 2u * turn] >= EDGE_COUNT || turns[turnsOffset + 2u * turn + 1u] >= EDGE_COUNT) { return; } }
var best = infinity; var targetPredecessor = 0xffffffffu;
if (source == destination && targetFraction >= sourceFraction) { best = (targetFraction - sourceFraction) * costs[costsOffset + source]; }
distances[source] = (1.0 - sourceFraction) * costs[costsOffset + source];
var converged = false;
for (var round = 0u; round < ${props.maximumRelaxations ?? props.costs.length}u; round++) {
  var changed = false;
  for (var turn = 0u; turn < ${props.turns.length}u; turn++) {
    let incoming = turns[turnsOffset + 2u * turn]; let outgoing = turns[turnsOffset + 2u * turn + 1u];
    if (!isFiniteValue(distances[incoming])) { continue; }
    if (outgoing == destination) {
      let targetCost = distances[incoming] + targetFraction * costs[costsOffset + destination];
      if (targetCost < best) { best = targetCost; targetPredecessor = incoming; }
    }
    let candidate = distances[incoming] + costs[costsOffset + outgoing];
    if (!isFiniteValue(candidate)) { return; }
    if (candidate < distances[outgoing]) { distances[outgoing] = candidate; predecessors[predecessorsOffset + index * EDGE_COUNT + outgoing] = incoming; changed = true; }
  }
  if (!changed) { converged = true; break; }
}
status[statusOffset + index] = select(2u, 0u, converged);
outputCosts[outputCostsOffset + index] = best;
// The source label cannot improve through a nonnegative cycle, but its partial target can be reached.
if (source == destination) { predecessors[predecessorsOffset + index * EDGE_COUNT + source] = targetPredecessor; }`
      })
    ];
  }
}
