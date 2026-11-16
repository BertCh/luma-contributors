// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {GPUNetworkReachability} from '../network-reachability/gpu-network-reachability';
import type {GPUNetworkPointRoutingProps} from './gpu-network-point-routing';

/** Builds a line-graph CSR once, searches it per query, then reduces partial destination costs. @internal */
export function getNetworkPointRoutingScalableNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUNetworkPointRoutingProps,
  maximumRelaxations: number,
  pathRoundCount: number
): readonly GPUCommandNode<Parameters>[] {
  const edgeCount = props.costs.length;
  const turnCount = props.turns.length;
  const unsigned = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'uint32', length);
  const floating = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'float32', length);
  const counts = unsigned('counts', edgeCount + 1);
  const offsets = unsigned('offsets', edgeCount + 1);
  const neighbors = unsigned('neighbors', Math.max(turnCount, 1));
  const weights = floating('weights', Math.max(turnCount, 1));
  const invalidNetwork = unsigned('invalid-network', 1);
  const startNodes = unsigned('start-nodes', 1);
  const startCosts = floating('start-costs', 1);
  const distances = floating('distances', edgeCount);
  const predecessors = unsigned('predecessors', edgeCount);
  const converged = unsigned('converged', 1);
  // Shared per-query reduction: [invalid, best-cost bits, target predecessor, round limit, query cursor].
  const state = unsigned('query-state', 5);
  // Each record holds a jumping predecessor and flags: bit 0 reaches the source,
  // bit 1 crosses the full destination (excluded from a replacement terminal path).
  let paths = createTransientView(graph, `${id}-paths-first`, 'uint32x2', edgeCount);
  let nextPaths = createTransientView(graph, `${id}-paths-second`, 'uint32x2', edgeCount);
  const activeIterations = graph.createDataView(state.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 3 * 4
  });
  const read = (
    name: string,
    view: GraphDataView,
    type: WGSLKernelBinding['type'] = 'u32'
  ): WGSLKernelBinding => ({
    name,
    view,
    type,
    access: 'read'
  });
  const write = (
    name: string,
    view: GraphDataView,
    type: WGSLKernelBinding['type'] = 'u32'
  ): WGSLKernelBinding => ({
    name,
    view,
    type,
    access: 'read_write'
  });
  const declarations = `const EDGE_COUNT: u32 = ${edgeCount}u;
const TURN_COUNT: u32 = ${turnCount}u;
const NONE: u32 = 0xffffffffu;
const INFINITY_BITS: u32 = 0x7f800000u;
${IS_FINITE_VALUE_WGSL}`;
  const kernel = (
    suffix: string,
    invocationCount: number,
    bindings: WGSLKernelBinding[],
    body: string
  ) =>
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-${suffix}`,
      operation: 'GPUNetworkPointRouting',
      variant: suffix.replace(/query-\d+-/, ''),
      invocationCount,
      bindings,
      declarations,
      body
    });
  const nodes: GPUCommandNode<Parameters>[] = [
    kernel(
      'reset',
      edgeCount + 1,
      [
        write('counts', counts),
        write('invalidNetwork', invalidNetwork),
        write('neighbors', neighbors),
        write('weights', weights, 'f32'),
        write('state', state)
      ],
      `counts[countsOffset + index] = 0u;
if (index == 0u) {
  invalidNetwork[invalidNetworkOffset] = 0u; state[stateOffset + 4u] = 0u;
  // The padded slot is outside offsets when there are no turns.
  neighbors[neighborsOffset] = NONE; weights[weightsOffset] = -1.0;
}`
    ),
    kernel(
      'validate-count',
      Math.max(edgeCount, turnCount),
      [
        read('turns', props.turns),
        read('costs', props.costs, 'f32'),
        write('counts', counts, 'atomic<u32>'),
        write('invalidNetwork', invalidNetwork, 'atomic<u32>')
      ],
      `if (index < EDGE_COUNT) {
  let cost = costs[costsOffset + index];
  if (!isFiniteValue(cost) || cost < 0.0) { atomicStore(&invalidNetwork[invalidNetworkOffset], 1u); }
}
if (index < TURN_COUNT) {
  let incoming = turns[turnsOffset + 2u * index]; let outgoing = turns[turnsOffset + 2u * index + 1u];
  if (incoming >= EDGE_COUNT || outgoing >= EDGE_COUNT) { atomicStore(&invalidNetwork[invalidNetworkOffset], 1u); }
  else { atomicAdd(&counts[countsOffset + incoming], 1u); }
}`
    ),
    ...new GPUScan({
      id: `${id}-scan`,
      input: counts,
      output: offsets,
      mode: 'exclusive'
    }).getCommandNodes(graph),
    kernel(
      'reset-cursors',
      edgeCount + 1,
      [write('counts', counts)],
      'counts[countsOffset + index] = 0u;'
    ),
    kernel(
      'scatter',
      turnCount,
      [
        read('turns', props.turns),
        read('costs', props.costs, 'f32'),
        read('offsets', offsets),
        write('counts', counts, 'atomic<u32>'),
        write('neighbors', neighbors),
        write('weights', weights, 'f32')
      ],
      `let incoming = turns[turnsOffset + 2u * index]; let outgoing = turns[turnsOffset + 2u * index + 1u];
if (incoming >= EDGE_COUNT || outgoing >= EDGE_COUNT) { return; }
let slot = offsets[offsetsOffset + incoming] + atomicAdd(&counts[countsOffset + incoming], 1u);
neighbors[neighborsOffset + slot] = outgoing;
weights[weightsOffset + slot] = costs[costsOffset + outgoing];`
    )
  ];
  for (let query = 0; query < props.queries.length; query++) {
    const prefix = `query-${query}`;
    nodes.push(
      kernel(
        `${prefix}-seed`,
        1,
        [
          read('queries', props.queries),
          read('fractions', props.fractions, 'f32'),
          read('costs', props.costs, 'f32'),
          read('invalidNetwork', invalidNetwork),
          write('state', state),
          write('startNodes', startNodes),
          write('startCosts', startCosts, 'f32')
        ],
        `let query = state[stateOffset + 4u];
let source = queries[queriesOffset + 2u * query]; let destination = queries[queriesOffset + 2u * query + 1u];
let sourceFraction = fractions[fractionsOffset + 2u * query]; let targetFraction = fractions[fractionsOffset + 2u * query + 1u];
state[stateOffset] = 1u; state[stateOffset + 1u] = INFINITY_BITS; state[stateOffset + 2u] = NONE;
state[stateOffset + 3u] = ${maximumRelaxations}u;
startNodes[startNodesOffset] = NONE; startCosts[startCostsOffset] = 0.0;
if (invalidNetwork[invalidNetworkOffset] != 0u || source >= EDGE_COUNT || destination >= EDGE_COUNT || !isFiniteValue(sourceFraction) || !isFiniteValue(targetFraction) || sourceFraction < 0.0 || sourceFraction > 1.0 || targetFraction < 0.0 || targetFraction > 1.0) { return; }
state[stateOffset] = 0u;
startNodes[startNodesOffset] = source;
startCosts[startCostsOffset] = (1.0 - sourceFraction) * costs[costsOffset + source];
if (source == destination && targetFraction >= sourceFraction) {
  state[stateOffset + 1u] = bitcast<u32>(max(0.0, (targetFraction - sourceFraction) * costs[costsOffset + source]));
}`
      )
    );
    nodes.push(
      ...new GPUNetworkReachability({
        id: `${id}-${prefix}-search`,
        topology: {offsets, neighbors, weights},
        startNodes,
        startCosts,
        maximumIterations: Math.max(maximumRelaxations, 1),
        maximumTieIterations: Math.max(maximumRelaxations, 1),
        localIterations: 64,
        activeIterations,
        output: {costs: distances, predecessors, status: {converged}}
      }).getCommandNodes(graph)
    );
    nodes.push(
      kernel(
        `${prefix}-paths-initialize`,
        edgeCount,
        [
          read('queries', props.queries),
          read('state', state),
          read('predecessors', predecessors),
          write('paths', paths)
        ],
        `let query = state[stateOffset + 4u];
let source = queries[queriesOffset + 2u * query]; let destination = queries[queriesOffset + 2u * query + 1u];
var ancestor = predecessors[predecessorsOffset + index];
var flags = select(0u, 2u, index == destination && source != destination);
if (index == source && state[stateOffset] == 0u) { ancestor = source; flags = 1u; }
paths[pathsOffset + 2u * index] = ancestor;
paths[pathsOffset + 2u * index + 1u] = flags;`
      )
    );
    for (let round = 0; round < pathRoundCount; round++) {
      nodes.push(
        kernel(
          `${prefix}-paths-jump-${round}`,
          edgeCount,
          [read('paths', paths), write('nextPaths', nextPaths)],
          `let ancestor = paths[pathsOffset + 2u * index];
var nextAncestor = ancestor; var flags = paths[pathsOffset + 2u * index + 1u];
if (ancestor < EDGE_COUNT) {
  nextAncestor = paths[pathsOffset + 2u * ancestor];
  flags |= paths[pathsOffset + 2u * ancestor + 1u];
}
nextPaths[nextPathsOffset + 2u * index] = nextAncestor;
nextPaths[nextPathsOffset + 2u * index + 1u] = flags;`
        )
      );
      [paths, nextPaths] = [nextPaths, paths];
    }
    nodes.push(
      kernel(
        `${prefix}-partial-costs`,
        turnCount,
        [
          read('queries', props.queries),
          read('fractions', props.fractions, 'f32'),
          read('turns', props.turns),
          read('costs', props.costs, 'f32'),
          read('distances', distances, 'f32'),
          write('state', state, 'atomic<u32>'),
          read('paths', paths)
        ],
        `let query = atomicLoad(&state[stateOffset + 4u]);
if (${maximumRelaxations}u == 0u || atomicLoad(&state[stateOffset]) != 0u) { return; }
let incoming = turns[turnsOffset + 2u * index]; let outgoing = turns[turnsOffset + 2u * index + 1u];
if (incoming >= EDGE_COUNT || outgoing >= EDGE_COUNT) { return; }
let distance = distances[distancesOffset + incoming];
if (!isFiniteValue(distance)) { return; }
// Retain the reference route's invalid-data status for a reachable full-edge f32 overflow.
if (!isFiniteValue(distance + costs[costsOffset + outgoing])) { atomicStore(&state[stateOffset], 1u); return; }
let flags = paths[pathsOffset + 2u * incoming + 1u];
if (flags != 1u) { return; }
let destination = queries[queriesOffset + 2u * query + 1u];
if (outgoing == destination) {
  let targetFraction = fractions[fractionsOffset + 2u * query + 1u];
  let candidate = distance + targetFraction * costs[costsOffset + destination];
  atomicMin(&state[stateOffset + 1u], bitcast<u32>(max(0.0, candidate)));
}`
      )
    );
    nodes.push(
      kernel(
        `${prefix}-target-parent`,
        turnCount,
        [
          read('queries', props.queries),
          read('fractions', props.fractions, 'f32'),
          read('turns', props.turns),
          read('costs', props.costs, 'f32'),
          read('distances', distances, 'f32'),
          write('state', state, 'atomic<u32>'),
          read('paths', paths)
        ],
        `let query = atomicLoad(&state[stateOffset + 4u]);
if (${maximumRelaxations}u == 0u || atomicLoad(&state[stateOffset]) != 0u) { return; }
let source = queries[queriesOffset + 2u * query]; let destination = queries[queriesOffset + 2u * query + 1u];
let incoming = turns[turnsOffset + 2u * index]; let outgoing = turns[turnsOffset + 2u * index + 1u];
if (incoming >= EDGE_COUNT || outgoing != destination) { return; }
if (paths[pathsOffset + 2u * incoming + 1u] != 1u) { return; }
let best = bitcast<f32>(atomicLoad(&state[stateOffset + 1u]));
if (!isFiniteValue(best)) { return; }
let sourceFraction = fractions[fractionsOffset + 2u * query]; let targetFraction = fractions[fractionsOffset + 2u * query + 1u];
// Direct forward travel wins ties, just as in the reference kernel.
if (source == destination && targetFraction >= sourceFraction && (targetFraction - sourceFraction) * costs[costsOffset + source] <= best) { return; }
let candidate = distances[distancesOffset + incoming] + targetFraction * costs[costsOffset + destination];
if (candidate == best) { atomicMin(&state[stateOffset + 2u], incoming); }`
      )
    );
    nodes.push(
      kernel(
        `${prefix}-copy-parents`,
        edgeCount,
        [
          read('queries', props.queries),
          read('state', state),
          read('predecessors', predecessors),
          read('paths', paths),
          write('outputPredecessors', props.output.predecessorEdges)
        ],
        `let query = state[stateOffset + 4u];
let destination = queries[queriesOffset + 2u * query + 1u];
var parent = predecessors[predecessorsOffset + index];
if ((paths[pathsOffset + 2u * index + 1u] & 1u) == 0u) { parent = NONE; }
// The terminal edge is partial: its minimizing parent can differ from the full-edge parent
// after f32 absorption, or the full edge can still be unreached when rounds are exhausted.
if (index == destination) { parent = state[stateOffset + 2u]; }
if (state[stateOffset] != 0u || ${maximumRelaxations}u == 0u) { parent = NONE; }
outputPredecessors[outputPredecessorsOffset + query * EDGE_COUNT + index] = parent;`
      )
    );
    nodes.push(
      kernel(
        `${prefix}-publish`,
        1,
        [
          write('state', state),
          read('converged', converged),
          write('outputCosts', props.output.costs, 'f32'),
          write('status', props.output.status)
        ],
        `let query = state[stateOffset + 4u];
let invalid = state[stateOffset] != 0u;
let complete = converged[convergedOffset] != 0u && ${maximumRelaxations}u > 0u;
status[statusOffset + query] = select(select(2u, 0u, complete), 1u, invalid);
outputCosts[outputCostsOffset + query] = bitcast<f32>(select(state[stateOffset + 1u], INFINITY_BITS, invalid));
state[stateOffset + 4u] = query + 1u;`
      )
    );
  }
  return nodes;
}
