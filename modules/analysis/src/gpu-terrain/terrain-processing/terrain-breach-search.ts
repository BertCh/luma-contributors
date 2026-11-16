// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding,
  type WGSLKernelProps
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getTerrainFlowTracingWGSL} from '../hydrology/terrain-flow-tracing';
import type {GPUTerrainBreachProps} from './gpu-terrain-breach';
import {terrainBinding, TERRAIN_FINITE_WGSL} from './terrain-processing-utils';

/** Step-layered Bellman recurrence, one lane per pit/window cell. @internal */
export function createTerrainBreachSearchNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  {
    id,
    props,
    elevation,
    searchWidth,
    searchHeight
  }: {
    id: string;
    props: GPUTerrainBreachProps;
    elevation: GraphDataView<'float32'>;
    searchWidth: number;
    searchHeight: number;
  }
): GPUCommandNode<Parameters>[] {
  const pitCount = props.pitCells.length;
  if (pitCount === 0) return [];
  const searchCount = searchWidth * searchHeight;
  const layerCount = props.maximumSteps + 1;
  const entriesPerPit = layerCount * searchCount;
  const entryCount = pitCount * entriesPerPit;
  const scratch = createTransientView(graph, `${id}-search`, 'uint32', entryCount * 2);
  // [minimum nonnegative float cost bits, earliest (step, local cell)] per pit.
  const winners = createTransientView(graph, `${id}-winners`, 'uint32', pitCount * 2);
  // Pending frontier and current step. Each run resets both, including after a gated early stop.
  const state = createTransientView(graph, `${id}-search-state`, 'uint32', 2);
  const dispatch = createWGSLDispatchGate<Parameters>(graph, {
    id: `${id}-search-dispatch`,
    operation: 'GPUTerrainBreach',
    shapes: [[pitCount * searchCount]]
  });
  const decrement = getWGSLFloatLiteral(props.elevationDecrement);
  const declarations = `${getTerrainFlowTracingWGSL(props.grid.width, props.grid.height)}
${TERRAIN_FINITE_WGSL}
fn searchOrigin(pit: u32) -> vec2<u32> {
 let column = pit % GRID_WIDTH; let row = pit / GRID_WIDTH;
 return min(vec2<u32>(select(0u, column - ${props.maximumSteps}u, column >= ${props.maximumSteps}u), select(0u, row - ${props.maximumSteps}u, row >= ${props.maximumSteps}u)), vec2<u32>(${props.grid.width - searchWidth}u, ${props.grid.height - searchHeight}u));
}
fn searchCell(localCell: u32, origin: vec2<u32>) -> u32 {
 return (origin.y + localCell / ${searchWidth}u) * GRID_WIDTH + origin.x + localCell % ${searchWidth}u;
}
fn searchIndex(cell: u32, origin: vec2<u32>) -> u32 {
 let column = cell % GRID_WIDTH; let row = cell / GRID_WIDTH;
 if (column < origin.x || row < origin.y || column >= origin.x + ${searchWidth}u || row >= origin.y + ${searchHeight}u) { return NO_CELL; }
 return (row - origin.y) * ${searchWidth}u + column - origin.x;
}`;
  const nodes: GPUCommandNode<Parameters>[] = [];
  const append = (
    name: string,
    invocationCount: number,
    bindings: WGSLKernelBinding[],
    body: string,
    gate?: Pick<WGSLKernelProps<Parameters>, 'condition' | 'extraResources'>
  ) =>
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-${name}`,
        operation: 'GPUTerrainBreach',
        variant: name.replace(/-\d+$/, ''),
        invocationCount,
        bindings,
        declarations,
        body,
        ...gate
      })
    );
  append(
    'search-initialize',
    entryCount,
    [
      terrainBinding('elevation', elevation),
      terrainBinding('pits', props.pitCells),
      terrainBinding('scratch', scratch, 'read_write')
    ],
    `let pitIndex = index / ${entriesPerPit}u;
let slot = index % ${entriesPerPit}u;
let pit = pits[pitsOffset + pitIndex];
var cost = 0x7f800000u;
if (pit < ${elevation.length}u && slot < ${searchCount}u) {
 let base = elevation[elevationOffset + pit];
 if (isFiniteValue(base) && base - ${decrement} < base && searchCell(slot, searchOrigin(pit)) == pit) { cost = 0u; }
}
scratch[scratchOffset + index * 2u] = cost;
scratch[scratchOffset + index * 2u + 1u] = NO_CELL;`
  );
  append(
    'paths-initialize',
    pitCount * layerCount,
    [terrainBinding('paths', props.output.pathCells, 'read_write')],
    'paths[pathsOffset + index] = NO_CELL;'
  );
  append(
    'search-seed',
    pitCount,
    [
      terrainBinding('winners', winners, 'read_write'),
      terrainBinding('state', state, 'read_write'),
      terrainBinding('lengths', props.output.pathLengths, 'read_write'),
      terrainBinding('status', props.output.pitStatus, 'read_write')
    ],
    `winners[winnersOffset + index * 2u] = 0x7f800000u;
winners[winnersOffset + index * 2u + 1u] = NO_CELL;
lengths[lengthsOffset + index] = 0u; status[statusOffset + index] = 0u;
if (index == 0u) { state[stateOffset] = 1u; state[stateOffset + 1u] = 0u; }`
  );
  for (let step = 1; step <= props.maximumSteps; step++) {
    append(
      `search-gate-${step}`,
      1,
      [terrainBinding('state', state, 'read_write'), dispatch.binding],
      `if (state[stateOffset] != 0u) { ${dispatch.armWGSL} } else { ${dispatch.disarmWGSL} }
state[stateOffset] = 0u; state[stateOffset + 1u]++;`
    );
    append(
      `search-relax-${step}`,
      pitCount * searchCount,
      [
        terrainBinding('elevation', elevation),
        terrainBinding('pits', props.pitCells),
        terrainBinding('scratch', scratch, 'read_write'),
        {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'}
      ],
      `let pitIndex = index / ${searchCount}u; let localCell = index % ${searchCount}u;
let pit = pits[pitsOffset + pitIndex];
if (pit >= ${elevation.length}u) { return; }
let base = elevation[elevationOffset + pit];
if (!isFiniteValue(base) || !(base - ${decrement} < base)) { return; }
let step = atomicLoad(&state[stateOffset + 1u]);
let origin = searchOrigin(pit); let cell = searchCell(localCell, origin);
let original = elevation[elevationOffset + cell];
let cut = max(original - (base - f32(step) * ${decrement}), 0.0);
if (!isFiniteValue(original) || cut > ${getWGSLFloatLiteral(props.maximumCutDepth)}) { return; }
let workOffset = scratchOffset + pitIndex * ${entriesPerPit * 2}u;
var predecessor = NO_CELL; var cost = infinity();
for (var direction = 0u; direction < 8u; direction++) {
 let neighbor = getFlowReceiver(cell, 1u << direction); if (neighbor == NO_CELL) { continue; }
 let localNeighbor = searchIndex(neighbor, origin); if (localNeighbor == NO_CELL) { continue; }
 let candidate = bitcast<f32>(scratch[workOffset + ((step - 1u) * ${searchCount}u + localNeighbor) * 2u]) + cut;
 if (candidate < cost || (candidate == cost && neighbor < predecessor)) { cost = candidate; predecessor = neighbor; }
}
// Costs are nonnegative; clear the sign bit to canonicalize a possible negative zero.
scratch[workOffset + (step * ${searchCount}u + localCell) * 2u] = bitcast<u32>(cost) & 0x7fffffffu;
scratch[workOffset + (step * ${searchCount}u + localCell) * 2u + 1u] = predecessor;
if (isFiniteValue(cost)) { atomicOr(&state[stateOffset], 1u); }`,
      dispatch.getGate(0)
    );
  }
  // Two exact integer atomic reductions implement the serial (cost, step, cell) ordering.
  // Nonnegative finite f32 costs have the same order as their uint32 bit patterns.
  const candidateBody = `let pitIndex = index / ${entriesPerPit}u;
let slot = index % ${entriesPerPit}u; let step = slot / ${searchCount}u;
if (step == 0u) { return; }
let pit = pits[pitsOffset + pitIndex]; if (pit >= ${elevation.length}u) { return; }
let base = elevation[elevationOffset + pit];
if (!isFiniteValue(base) || !(base - ${decrement} < base)) { return; }
let original = elevation[elevationOffset + searchCell(slot % ${searchCount}u, searchOrigin(pit))];
let cost = scratch[scratchOffset + index * 2u];
if (!isFiniteValue(original) || original > base - f32(step) * ${decrement} || cost >= 0x7f800000u) { return; }`;
  const candidateBindings: WGSLKernelBinding[] = [
    terrainBinding('elevation', elevation),
    terrainBinding('pits', props.pitCells),
    terrainBinding('scratch', scratch),
    {name: 'winners', view: winners, type: 'atomic<u32>', access: 'read_write'}
  ];
  append(
    'search-minimum-cost',
    entryCount,
    candidateBindings,
    `${candidateBody}\natomicMin(&winners[winnersOffset + pitIndex * 2u], cost);`
  );
  append(
    'search-minimum-slot',
    entryCount,
    candidateBindings,
    `${candidateBody}\nif (cost == atomicLoad(&winners[winnersOffset + pitIndex * 2u])) { atomicMin(&winners[winnersOffset + pitIndex * 2u + 1u], slot); }`
  );
  append(
    'search-trace',
    pitCount,
    [
      terrainBinding('elevation', elevation),
      terrainBinding('pits', props.pitCells),
      terrainBinding('scratch', scratch),
      terrainBinding('winners', winners),
      terrainBinding('paths', props.output.pathCells, 'read_write'),
      terrainBinding('lengths', props.output.pathLengths, 'read_write'),
      terrainBinding('status', props.output.pitStatus, 'read_write')
    ],
    `let pit = pits[pitsOffset + index]; if (pit >= ${elevation.length}u) { return; }
let base = elevation[elevationOffset + pit];
if (!isFiniteValue(base) || !(base - ${decrement} < base)) { return; }
let slot = winners[winnersOffset + index * 2u + 1u];
if (slot == NO_CELL) { status[statusOffset + index] = 2u; return; }
let origin = searchOrigin(pit); let bestStep = slot / ${searchCount}u;
var current = searchCell(slot % ${searchCount}u, origin);
let pathOffset = pathsOffset + index * ${layerCount}u;
let workOffset = scratchOffset + index * ${entriesPerPit * 2}u;
// The only serial walk follows the selected path, bounded by maximumSteps and the node budget.
for (var step = bestStep; step > 0u; step--) {
 paths[pathOffset + step] = current;
 current = scratch[workOffset + (step * ${searchCount}u + searchIndex(current, origin)) * 2u + 1u];
}
paths[pathOffset] = pit; lengths[lengthsOffset + index] = bestStep + 1u; status[statusOffset + index] = 1u;`
  );
  return nodes;
}
