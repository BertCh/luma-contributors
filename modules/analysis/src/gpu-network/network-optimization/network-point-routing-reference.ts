// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import type {GPUNetworkPointRoutingProps} from './gpu-network-point-routing';

/** The original bounded, turn-order reference route. @internal */
export function getNetworkPointRoutingReferenceNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUNetworkPointRoutingProps
): readonly GPUCommandNode<Parameters>[] {
  return [
    createWGSLKernelNode<Parameters>(graph, {
      id,
      operation: 'GPUNetworkPointRouting',
      variant: 'directed-edge-positions',
      invocationCount: props.queries.length,
      workgroupSize: 32,
      bindings: [
        {name: 'turns', view: props.turns, type: 'u32', access: 'read'},
        ...(props.turnPenalties
          ? [
              {
                name: 'turnPenalties',
                view: props.turnPenalties,
                type: 'f32',
                access: 'read'
              } as const
            ]
          : []),
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
if (!isFiniteValue(costs[costsOffset + source]) || costs[costsOffset + source] < 0.0 || !isFiniteValue(costs[costsOffset + destination]) || costs[costsOffset + destination] < 0.0) { return; }
for (var turn = 0u; turn < ${props.turns.length}u; turn++) { if (turns[turnsOffset + 2u * turn] >= EDGE_COUNT || turns[turnsOffset + 2u * turn + 1u] >= EDGE_COUNT) { return; }
${props.turnPenalties ? 'let penalty = turnPenalties[turnPenaltiesOffset + turn]; if (!isFiniteValue(penalty) || penalty < 0.0) { return; }' : ''} }
var best = infinity; var targetPredecessor = 0xffffffffu;
if (source == destination && targetFraction >= sourceFraction) { best = (targetFraction - sourceFraction) * costs[costsOffset + source]; }
distances[source] = (1.0 - sourceFraction) * costs[costsOffset + source];
var converged = false;
for (var round = 0u; round < ${props.maximumRelaxations ?? props.costs.length}u; round++) {
  var changed = false;
  for (var turn = 0u; turn < ${props.turns.length}u; turn++) {
    let incoming = turns[turnsOffset + 2u * turn]; let outgoing = turns[turnsOffset + 2u * turn + 1u];
    let penalty = ${props.turnPenalties ? 'turnPenalties[turnPenaltiesOffset + turn]' : '0.0'};
    if (!isFiniteValue(distances[incoming]) || !isFiniteValue(costs[costsOffset + outgoing]) || costs[costsOffset + outgoing] < 0.0) { continue; }
    let candidate = distances[incoming] + (penalty + costs[costsOffset + outgoing]);
    if (!isFiniteValue(candidate)) { return; }
    if (candidate < distances[outgoing]) { distances[outgoing] = candidate; predecessors[predecessorsOffset + index * EDGE_COUNT + outgoing] = incoming; changed = true; }
  }
  if (!changed) { converged = true; break; }
}
status[statusOffset + index] = select(2u, 0u, converged);
// Reduce the partial destination separately: its full-edge labels may have absorbed a penalty.
if (${props.maximumRelaxations ?? props.costs.length}u > 0u) {
  for (var turn = 0u; turn < ${props.turns.length}u; turn++) {
    let incoming = turns[turnsOffset + 2u * turn]; let outgoing = turns[turnsOffset + 2u * turn + 1u];
    if (outgoing != destination || !isFiniteValue(distances[incoming])) { continue; }
    var cursor = incoming;
    for (var hop = 0u; hop < EDGE_COUNT && cursor != source; hop++) {
      if (cursor >= EDGE_COUNT || cursor == destination) { break; }
      cursor = predecessors[predecessorsOffset + index * EDGE_COUNT + cursor];
    }
    if (cursor != source) { continue; }
    let penalty = ${props.turnPenalties ? 'turnPenalties[turnPenaltiesOffset + turn]' : '0.0'};
    let targetCost = distances[incoming] + (penalty + targetFraction * costs[costsOffset + destination]);
    if (targetCost < best) { best = targetCost; targetPredecessor = incoming; }
  }
}
outputCosts[outputCostsOffset + index] = best;
predecessors[predecessorsOffset + index * EDGE_COUNT + destination] = targetPredecessor;`
    })
  ];
}
