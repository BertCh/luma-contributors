// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import type {GPURasterMetadata} from '../../gpu-raster/types';
import {createRasterCountNodes} from '../../gpu-raster/remote-sensing/raster-kernel-utils';
import {validateRasterGrid} from '../../gpu-raster/raster-band-utils';
import {createWGSLKernelNode, getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';
import {getTerrainFlowTracingWGSL} from '../hydrology/terrain-flow-tracing';
import {
  checkTerrainViews,
  checkTerrainGraph,
  terrainBinding,
  TERRAIN_FINITE_WGSL
} from './terrain-processing-utils';

export type GPUTerrainBreachProps = {
  id?: string;
  grid: GPURasterMetadata;
  elevation: GraphDataView<'float32'>;
  /** Explicit candidate pits; no implicit filling or burning. Empty lists are supported. */
  pitCells: GraphDataView<'uint32'>;
  maximumSteps: number;
  maximumCutDepth: number;
  /** Positive representable decrease per path step in elevation units. */
  elevationDecrement: number;
  /** Optional explicit burn depth per cell, applied after breach cuts. */
  burnDepth?: GraphDataView<'float32'>;
  output: {
    elevation: GraphDataView<'float32'>;
    cutDepth: GraphDataView<'float32'>;
    /** Bit 1 breach, bit 2 burn, bit 4 invalid burn depth. */
    modificationMask: GraphDataView<'uint32'>;
    /** Pit-major paths, pit first, exactly pitCount * (maximumSteps + 1) capacity. */
    pathCells: GraphDataView<'uint32'>;
    pathLengths: GraphDataView<'uint32'>;
    /** Per pit: 0 invalid, 1 breached, 2 no feasible outlet within bounds. */
    pitStatus: GraphDataView<'uint32'>;
    unresolvedCount: GraphDataView<'uint32'>;
  };
};

/** Bounded least total excavation-depth path on a step-layered D8 graph, with deterministic simultaneous minimum-elevation edits. */
export class GPUTerrainBreach implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUTerrainBreachProps) {
    this.id = props.id ?? 'terrain-breach';
    const count = validateRasterGrid(this.id, props.grid);
    assert(
      props.elevation.length === count && (!props.burnDepth || props.burnDepth.length === count)
    );
    assert(Number.isInteger(props.maximumSteps) && props.maximumSteps > 0);
    assert(Number.isFinite(props.maximumCutDepth) && props.maximumCutDepth >= 0);
    assert(
      Number.isFinite(props.elevationDecrement) &&
        props.elevationDecrement > 0 &&
        Math.fround(props.elevationDecrement) > 0
    );
    for (const view of [
      props.output.elevation,
      props.output.cutDepth,
      props.output.modificationMask
    ])
      assert(view.length === count);
    assert(props.output.pathCells.length === props.pitCells.length * (props.maximumSteps + 1));
    assert(
      props.output.pathLengths.length === props.pitCells.length &&
        props.output.pitStatus.length === props.pitCells.length &&
        props.output.unresolvedCount.length === 1
    );
    const searchCount =
      Math.min(props.grid.width, props.maximumSteps * 2 + 1) *
      Math.min(props.grid.height, props.maximumSteps * 2 + 1);
    assert(
      Number.isSafeInteger(props.pitCells.length * (props.maximumSteps + 1) * searchCount * 2) &&
        props.pitCells.length * (props.maximumSteps + 1) * searchCount * 2 <= 0xffffffff
    );
    checkTerrainViews(
      this.id,
      [props.elevation, props.pitCells, ...(props.burnDepth ? [props.burnDepth] : [])],
      Object.values(props.output)
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const count = props.elevation.length;
    const pitCount = props.pitCells.length;
    // A path with at most S eight-neighbor steps cannot leave this exact Chebyshev window.
    const searchWidth = Math.min(props.grid.width, props.maximumSteps * 2 + 1);
    const searchHeight = Math.min(props.grid.height, props.maximumSteps * 2 + 1);
    const searchCount = searchWidth * searchHeight;
    checkTerrainGraph(id, graph, [
      props.elevation,
      props.pitCells,
      ...(props.burnDepth ? [props.burnDepth] : []),
      ...Object.values(props.output)
    ]);
    const scratch = createTransientView(
      graph,
      `${id}-search`,
      'uint32',
      Math.max(1, pitCount * (props.maximumSteps + 1) * searchCount * 2)
    );
    const edits = createTransientView(graph, `${id}-edits`, 'uint32', count);
    const decrement = getWGSLFloatLiteral(props.elevationDecrement);
    const declarations = `${getTerrainFlowTracingWGSL(props.grid.width, props.grid.height)}\n${TERRAIN_FINITE_WGSL}
fn searchIndex(cell: u32, origin: vec2<u32>) -> u32 {
 let column = cell % GRID_WIDTH; let row = cell / GRID_WIDTH;
 if (column < origin.x || row < origin.y || column >= origin.x + ${searchWidth}u || row >= origin.y + ${searchHeight}u) { return NO_CELL; }
 return (row - origin.y) * ${searchWidth}u + column - origin.x;
}`;
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (pitCount > 0)
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-search`,
          operation: 'GPUTerrainBreach',
          invocationCount: pitCount,
          declarations,
          bindings: [
            terrainBinding('elevation', props.elevation),
            terrainBinding('pits', props.pitCells),
            terrainBinding('scratch', scratch, 'read_write'),
            terrainBinding('paths', props.output.pathCells, 'read_write'),
            terrainBinding('lengths', props.output.pathLengths, 'read_write'),
            terrainBinding('status', props.output.pitStatus, 'read_write')
          ],
          body: `let pit = pits[pitsOffset + index];
let pathOffset = pathsOffset + index * ${props.maximumSteps + 1}u;
lengths[lengthsOffset + index] = 0u; status[statusOffset + index] = 0u;
for (var step = 0u; step <= ${props.maximumSteps}u; step++) { paths[pathOffset + step] = NO_CELL; }
if (pit >= ${count}u) { return; }
let base = elevation[elevationOffset + pit];
if (!finite(base) || !(base - ${decrement} < base)) { return; }
let origin = vec2<u32>(u32(clamp(i32(pit % GRID_WIDTH) - ${props.maximumSteps}, 0, ${props.grid.width - searchWidth})), u32(clamp(i32(pit / GRID_WIDTH) - ${props.maximumSteps}, 0, ${props.grid.height - searchHeight})));
let workOffset = scratchOffset + index * ${(props.maximumSteps + 1) * searchCount * 2}u;
for (var entry = 0u; entry < ${(props.maximumSteps + 1) * searchCount}u; entry++) { scratch[workOffset + entry * 2u] = 0x7f800000u; scratch[workOffset + entry * 2u + 1u] = NO_CELL; }
scratch[workOffset + searchIndex(pit, origin) * 2u] = bitcast<u32>(0.0);
var bestCost = infinity(); var bestCell = NO_CELL; var bestStep = 0u;
for (var step = 1u; step <= ${props.maximumSteps}u; step++) {
 let level = base - f32(step) * ${decrement};
 for (var localCell = 0u; localCell < ${searchCount}u; localCell++) {
  let cell = (origin.y + localCell / ${searchWidth}u) * GRID_WIDTH + origin.x + localCell % ${searchWidth}u;
  let original = elevation[elevationOffset + cell];
  let cut = max(original - level, 0.0);
  if (!finite(original) || cut > ${getWGSLFloatLiteral(props.maximumCutDepth)}) { continue; }
  var predecessor = NO_CELL; var cost = infinity();
  for (var direction = 0u; direction < 8u; direction++) {
   let neighbor = getFlowReceiver(cell, 1u << direction); if (neighbor == NO_CELL) { continue; }
   let localNeighbor = searchIndex(neighbor, origin); if (localNeighbor == NO_CELL) { continue; }
   let candidate = bitcast<f32>(scratch[workOffset + ((step - 1u) * ${searchCount}u + localNeighbor) * 2u]) + cut;
   if (candidate < cost || (candidate == cost && neighbor < predecessor)) { cost = candidate; predecessor = neighbor; }
  }
  scratch[workOffset + (step * ${searchCount}u + localCell) * 2u] = bitcast<u32>(cost);
  scratch[workOffset + (step * ${searchCount}u + localCell) * 2u + 1u] = predecessor;
  if (original <= level && cost < bestCost) { bestCost = cost; bestCell = cell; bestStep = step; }
 }
}
if (bestCell == NO_CELL) { status[statusOffset + index] = 2u; return; }
var current = bestCell;
for (var step = bestStep; step > 0u; step--) {
 paths[pathOffset + step] = current;
 current = scratch[workOffset + (step * ${searchCount}u + searchIndex(current, origin)) * 2u + 1u];
}
paths[pathOffset] = pit; lengths[lengthsOffset + index] = bestStep + 1u; status[statusOffset + index] = 1u;`
        })
      );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-edit-initialize`,
        operation: 'GPUTerrainBreach',
        invocationCount: count,
        bindings: [
          terrainBinding('source', props.elevation),
          terrainBinding('edits', edits, 'read_write')
        ],
        body: `let bits = bitcast<u32>(source[sourceOffset + index]); edits[editsOffset + index] = select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-edit-proposals`,
        operation: 'GPUTerrainBreach',
        invocationCount: pitCount * (props.maximumSteps + 1),
        bindings: [
          terrainBinding('source', props.elevation),
          terrainBinding('paths', props.output.pathCells),
          terrainBinding('lengths', props.output.pathLengths),
          {name: 'edits', view: edits, type: 'atomic<u32>', access: 'read_write'}
        ],
        body: `let pitIndex = index / ${props.maximumSteps + 1}u; let step = index % ${props.maximumSteps + 1}u;
if (step == 0u || step >= lengths[lengthsOffset + pitIndex]) { return; }
let start = paths[pathsOffset + pitIndex * ${props.maximumSteps + 1}u];
let cell = paths[pathsOffset + index];
let value = source[sourceOffset + start] - f32(step) * ${decrement};
let bits = bitcast<u32>(value);
atomicMin(&edits[editsOffset + cell], select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u));`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-apply`,
        operation: 'GPUTerrainBreach',
        invocationCount: count,
        declarations,
        bindings: [
          terrainBinding('source', props.elevation),
          terrainBinding('edits', edits),
          terrainBinding('elevation', props.output.elevation, 'read_write'),
          terrainBinding('depth', props.output.cutDepth, 'read_write'),
          terrainBinding('mask', props.output.modificationMask, 'read_write'),
          ...(props.burnDepth ? [terrainBinding('burn', props.burnDepth)] : [])
        ],
        body: `let original = source[sourceOffset + index]; var edited = original; var flags = 0u;
if (finite(original)) {
 let key = edits[editsOffset + index];
 edited = bitcast<f32>(select(~key, key ^ 0x80000000u, (key & 0x80000000u) != 0u));
 if (edited < original) { flags |= 1u; }
 ${props.burnDepth ? 'let burnDepth = burn[burnOffset + index]; if (!finite(burnDepth) || burnDepth < 0.0) { edited = missing(); flags |= 4u; } else if (burnDepth > 0.0) { edited -= burnDepth; flags |= 2u; }' : ''}
}
elevation[elevationOffset + index] = edited; depth[depthOffset + index] = original - edited; mask[maskOffset + index] = flags;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-verify-paths`,
        operation: 'GPUTerrainBreach',
        invocationCount: pitCount,
        declarations: TERRAIN_FINITE_WGSL,
        bindings: [
          terrainBinding('elevation', props.output.elevation),
          terrainBinding('mask', props.output.modificationMask),
          terrainBinding('paths', props.output.pathCells),
          terrainBinding('lengths', props.output.pathLengths),
          terrainBinding('status', props.output.pitStatus, 'read_write')
        ],
        body: `if (status[statusOffset + index] != 1u) { return; }
let pathLength = lengths[lengthsOffset + index];
for (var step = 1u; step < pathLength; step++) {
 let previous = paths[pathsOffset + index * ${props.maximumSteps + 1}u + step - 1u];
 let current = paths[pathsOffset + index * ${props.maximumSteps + 1}u + step];
 let previousElevation = elevation[elevationOffset + previous];
 let currentElevation = elevation[elevationOffset + current];
 if (((mask[maskOffset + previous] | mask[maskOffset + current]) & 4u) != 0u || !finite(previousElevation) || !finite(currentElevation) || !(currentElevation < previousElevation)) { status[statusOffset + index] = 2u; }
}`
      }),
      ...createRasterCountNodes(graph, {
        id: `${id}-status`,
        operation: 'GPUTerrainBreach',
        values: props.output.pitStatus,
        count: pitCount,
        condition: 'value != 1u',
        output: props.output.unresolvedCount
      })
    );
    return nodes;
  }
}
