// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../utils/wgsl-kernel-nodes';
import {assertRasterStorageBindingFits} from './raster-utils';
import type {GPURasterBufferBand} from './types';
import type {GPURasterWatershedSegmentationProps} from './gpu-raster-watershed-segmentation';

/**
 * Rooted Boruvka: only unseeded components select their cheapest outgoing edge. Seeded roots
 * never move, so a selected functional graph cannot join two seeds. A total edge order excludes
 * cycles longer than two; reciprocal selections are rooted at the smaller component ID.
 * This is Boruvka on the graph with all markers joined by virtual edges lighter than real edges,
 * omitting the virtual component's outgoing selection. It yields the rooted Kruskal forest under
 * the same edge order. Unseeded active components at least halve each contraction round.
 * @internal
 */
export function getRasterWatershedForestNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPURasterWatershedSegmentationProps,
  band: GPURasterBufferBand
): GPUCommandNode<Parameters>[] {
  const count = props.grid.width * props.grid.height;
  const rounds = props.maximumIterations ?? Math.ceil(Math.log2(count)) + 1;
  const component = createTransientView(graph, `${id}-forest-components`, 'uint32', count);
  const bestCost = createTransientView(graph, `${id}-forest-costs`, 'uint32', count);
  const bestEdge = createTransientView(graph, `${id}-forest-edges`, 'uint32', count);
  const parentFirst = createTransientView(graph, `${id}-forest-parent-first`, 'uint32', count);
  const parentSecond = createTransientView(graph, `${id}-forest-parent-second`, 'uint32', count);
  const active = createTransientView(graph, `${id}-forest-active`, 'uint32', 1);
  const operation = 'GPURasterWatershedSegmentation';
  const gate = createWGSLDispatchGate<Parameters>(graph, {
    id: `${id}-forest-dispatch`,
    operation,
    shapes: [[count]]
  });
  const binding = (
    name: string,
    view: GraphDataView,
    access: 'read' | 'read_write' = 'read',
    type: 'u32' | 'f32' | 'atomic<u32>' = 'u32'
  ): WGSLKernelBinding => ({name, view, access, type});
  for (const view of [
    component,
    bestCost,
    bestEdge,
    parentFirst,
    parentSecond,
    active,
    gate.view,
    band.storage.values,
    band.validity!,
    props.markers,
    props.output.labels,
    props.output.status.converged
  ]) {
    assertRasterStorageBindingFits(graph.device, view, id);
  }
  const declarations = `const WIDTH: u32 = ${props.grid.width}u;
const HEIGHT: u32 = ${props.grid.height}u;
const NONE: u32 = 0xffffffffu;
// Four forward directions: right, down-left, down, down-right.
fn edgeTarget(source: u32, direction: u32) -> u32 {
  let column = source % WIDTH; let row = source / WIDTH;
  if (direction == 0u) { return select(NONE, source + 1u, column + 1u < WIDTH); }
  if (row + 1u >= HEIGHT) { return NONE; }
  if (direction == 1u) { return select(NONE, source + WIDTH - 1u, column > 0u); }
  if (direction == 2u) { return source + WIDTH; }
  return select(NONE, source + WIDTH + 1u, column + 1u < WIDTH);
}
fn costKey(value: f32) -> u32 {
  // Signed zero is one elevation; negative finite floats must sort before nonnegative floats.
  let bits = bitcast<u32>(select(value, 0.0, value == 0.0));
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}`;
  const rasterBindings = [
    binding('elevation', band.storage.values, 'read', 'f32'),
    binding('validity', band.validity!),
    binding('markers', props.markers),
    binding('component', component)
  ];
  const edgeLoop = (body: string) => `
if (validity[validityOffset + index] == 0u || markers[markersOffset + index] >= 0xfffffffeu) { return; }
let first = component[componentOffset + index];
for (var direction = 0u; direction < 4u; direction++) {
  ${props.connectivity === 8 ? '' : 'if (direction == 1u || direction == 3u) { continue; }'}
  let neighbor = edgeTarget(index, direction);
  if (neighbor == NONE) { continue; }
  if (validity[validityOffset + neighbor] == 0u || markers[markersOffset + neighbor] >= 0xfffffffeu) { continue; }
  let second = component[componentOffset + neighbor];
  if (first == second) { continue; }
  let edge = index * 4u + direction;
  let cost = costKey(max(elevation[elevationOffset + index], elevation[elevationOffset + neighbor]));
  ${body}
}`;
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-forest-initialize`,
      operation,
      variant: 'forest-initialize',
      invocationCount: count,
      bindings: [
        binding('component', component, 'read_write'),
        binding('bestCost', bestCost, 'read_write'),
        binding('bestEdge', bestEdge, 'read_write'),
        binding('roundState', active, 'read_write'),
        binding('converged', props.output.status.converged, 'read_write')
      ],
      body: `component[componentOffset + index] = index;
bestCost[bestCostOffset + index] = 0xffffffffu;
bestEdge[bestEdgeOffset + index] = 0xffffffffu;
if (index == 0u) { roundState[roundStateOffset] = 1u; converged[convergedOffset] = 1u; }`
    })
  ];
  for (let round = 0; round < rounds; round++) {
    const roundId = `${id}-forest-round-${round}`;
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${roundId}-gate`,
        operation,
        variant: 'forest-gate',
        invocationCount: 1,
        bindings: [binding('roundState', active, 'read_write'), gate.binding],
        body: `if (roundState[roundStateOffset] != 0u) { ${gate.armWGSL} } else { ${gate.disarmWGSL} }
roundState[roundStateOffset] = 0u;`
      }),
      createWGSLKernelNode(graph, {
        id: `${roundId}-cost`,
        operation,
        variant: 'forest-cost',
        invocationCount: count,
        ...gate.getGate(0),
        declarations,
        bindings: [
          ...rasterBindings,
          binding('bestCost', bestCost, 'read_write', 'atomic<u32>'),
          binding('roundState', active, 'read_write', 'atomic<u32>')
        ],
        body: edgeLoop(`if (markers[markersOffset + first] == 0u) {
  atomicMin(&bestCost[bestCostOffset + first], cost); atomicStore(&roundState[roundStateOffset], 1u);
}
if (markers[markersOffset + second] == 0u) {
  atomicMin(&bestCost[bestCostOffset + second], cost); atomicStore(&roundState[roundStateOffset], 1u);
}`)
      }),
      createWGSLKernelNode(graph, {
        id: `${roundId}-edge`,
        operation,
        variant: 'forest-edge',
        invocationCount: count,
        ...gate.getGate(0),
        declarations,
        bindings: [
          ...rasterBindings,
          binding('bestCost', bestCost),
          binding('bestEdge', bestEdge, 'read_write', 'atomic<u32>')
        ],
        body: edgeLoop(`if (markers[markersOffset + first] == 0u && bestCost[bestCostOffset + first] == cost) {
  atomicMin(&bestEdge[bestEdgeOffset + first], edge);
}
if (markers[markersOffset + second] == 0u && bestCost[bestCostOffset + second] == cost) {
  atomicMin(&bestEdge[bestEdgeOffset + second], edge);
}`)
      }),
      createWGSLKernelNode(graph, {
        id: `${roundId}-hook`,
        operation,
        variant: 'forest-hook',
        invocationCount: count,
        ...gate.getGate(0),
        declarations,
        bindings: [
          binding('component', component),
          binding('bestEdge', bestEdge),
          binding('parent', parentFirst, 'read_write')
        ],
        body: `var parentIndex = index;
let edge = bestEdge[bestEdgeOffset + index];
if (component[componentOffset + index] == index && edge != NONE) {
  let source = edge / 4u;
  let first = component[componentOffset + source];
  let second = component[componentOffset + edgeTarget(source, edge % 4u)];
  parentIndex = select(first, second, first == index);
  if (bestEdge[bestEdgeOffset + parentIndex] == edge && index < parentIndex) { parentIndex = index; }
}
parent[parentOffset + index] = parentIndex;`
      })
    );
    let source = parentFirst;
    let destination = parentSecond;
    // A selected path contains at most N roots. Ping-pong storage avoids inter-invocation races.
    for (let jump = 0; jump < Math.ceil(Math.log2(count)); jump++) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${roundId}-jump-${jump}`,
          operation,
          variant: 'forest-jump',
          invocationCount: count,
          ...gate.getGate(0),
          bindings: [binding('source', source), binding('destination', destination, 'read_write')],
          body: 'destination[destinationOffset + index] = source[sourceOffset + source[sourceOffset + index]];'
        })
      );
      [source, destination] = [destination, source];
    }
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${roundId}-relabel`,
        operation,
        variant: 'forest-relabel',
        invocationCount: count,
        ...gate.getGate(0),
        bindings: [
          binding('parent', source),
          binding('component', component, 'read_write'),
          binding('bestCost', bestCost, 'read_write'),
          binding('bestEdge', bestEdge, 'read_write')
        ],
        body: `component[componentOffset + index] = parent[parentOffset + component[componentOffset + index]];
bestCost[bestCostOffset + index] = 0xffffffffu;
bestEdge[bestEdgeOffset + index] = 0xffffffffu;`
      })
    );
  }
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-forest-validate`,
      operation,
      variant: 'forest-validate',
      invocationCount: count,
      declarations,
      bindings: [
        ...rasterBindings,
        binding('converged', props.output.status.converged, 'read_write', 'atomic<u32>')
      ],
      body: edgeLoop(`if (markers[markersOffset + first] == 0u || markers[markersOffset + second] == 0u) {
  atomicStore(&converged[convergedOffset], 0u);
}`)
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-forest-publish`,
      operation,
      variant: 'forest-publish',
      invocationCount: count,
      bindings: [
        binding('component', component),
        binding('validity', band.validity!),
        binding('markers', props.markers),
        binding('converged', props.output.status.converged),
        binding('labels', props.output.labels, 'read_write')
      ],
      body: `let marker = markers[markersOffset + index];
var label = 0u;
if (validity[validityOffset + index] != 0u && marker < 0xfffffffeu) {
  label = markers[markersOffset + component[componentOffset + index]];
  if (converged[convergedOffset] == 0u && marker == 0u) { label = 0xfffffffeu; }
}
labels[labelsOffset + index] = label;`
    })
  );
  return nodes;
}
