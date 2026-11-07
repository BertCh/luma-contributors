// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addBuildingAdjacencyRecipe turns the shared walls found by GPUSharedPaths into building
// adjacency: wall length per pair, neighbors and connected terraces. Buildings are colored by the
// terrace (connected component) the recipe found; thick orange lines are the shared walls. A gap
// between two buildings removes their shared wall and splits the terrace.

import {
  addBuildingAdjacencyRecipe,
  GPUSharedPaths
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addKernelPass, getViewBuffer} from './shared/kernel';
import {addPackedSegments, HIGHLIGHT_COLOR} from './shared/lines';
import {parameters} from './building-adjacency-recipe.parameters';

const ROWS = 3;
const PER_ROW = 8;
const BUILDING_COUNT = ROWS * PER_ROW;
const RING_VERTICES = 5;

export default defineOperationRun({
  parameters,
  caption: 'Terraces found from shared walls: one color per connected group of buildings.',
  createInputs: ({random, world}) => {
    const widths = Float32Array.from({length: BUILDING_COUNT}, () => 0.7 + 0.6 * random());
    const getRings = (splitEvery: number, gap: number) => {
      const positions = new Float32Array(BUILDING_COUNT * RING_VERTICES * 2);
      for (let row = 0; row < ROWS; row++) {
        const rowWidth = Array.from({length: PER_ROW}, (_, slot) => widths[row * PER_ROW + slot]);
        const breaks = Math.floor((PER_ROW - 1) / splitEvery);
        const total = rowWidth.reduce((sum, width) => sum + width, 0);
        const unit = (world.width - 0.1 - gap * breaks) / total;
        const y0 = 0.1 + row * 0.3;
        const y1 = y0 + 0.2;
        let x = 0.05;
        for (let slot = 0; slot < PER_ROW; slot++) {
          if (slot > 0 && slot % splitEvery === 0) x += gap;
          const x1 = x + rowWidth[slot] * unit;
          const building = row * PER_ROW + slot;
          positions.set([x, y0, x1, y0, x1, y1, x, y1, x, y0], building * RING_VERTICES * 2);
          x = x1;
        }
      }
      return positions;
    };
    return {getRings};
  },
  build(builder, inputs, values) {
    const initial = inputs.getRings(values.splitEvery, values.gap);
    const positions = builder.upload('rings', 'float32x2', initial);
    const lineOffsets = builder.upload(
      'ring-offsets',
      'uint32',
      Uint32Array.from({length: BUILDING_COUNT + 1}, (_, building) => building * RING_VERTICES)
    );
    const runCapacity = 256;
    const vertexCapacity = 2048;
    const leftLineIds = builder.allocate('run-left', 'uint32', runCapacity);
    const rightLineIds = builder.allocate('run-right', 'uint32', runCapacity);
    const forward = builder.allocate('run-forward', 'uint32', runCapacity);
    const offsets = builder.allocate('run-offsets', 'uint32', runCapacity + 1);
    const runPositions = builder.allocate('run-positions', 'float32x2', vertexCapacity);
    const count = builder.allocate('run-count', 'uint32', 1);
    const vertexCount = builder.allocate('run-vertex-count', 'uint32', 1);
    const overflow = builder.allocate('run-overflow', 'uint32', 1);
    const requiredCount = builder.allocate('run-required', 'uint32', 1);
    const sharedOutput = {
      runs: {
        leftLineIds: leftLineIds.view,
        rightLineIds: rightLineIds.view,
        forward: forward.view,
        offsets: offsets.view,
        positions: runPositions.view
      },
      status: {
        count: count.view,
        vertexCount: vertexCount.view,
        overflow: overflow.view,
        requiredCount: requiredCount.view
      }
    };
    builder.graph.add(
      new GPUSharedPaths({
        left: {kind: 'lines', positions: positions.view, lineOffsets: lineOffsets.view},
        right: {kind: 'lines', positions: positions.view, lineOffsets: lineOffsets.view},
        intersectionCapacity: 8192,
        output: sharedOutput
      })
    );
    const owners = builder.upload(
      'ring-owners',
      'uint32',
      Uint32Array.from({length: BUILDING_COUNT}, (_, index) => index)
    );
    const orientations = builder.upload(
      'orientations',
      'float32',
      new Float32Array(BUILDING_COUNT)
    );
    const recipe = addBuildingAdjacencyRecipe(builder.graph, {
      sharedPaths: sharedOutput,
      lineBuildingIds: owners.view,
      orientations: orientations.view
    });
    const nodeComponents = builder.allocate(
      'ring-components',
      'uint32',
      BUILDING_COUNT * RING_VERTICES
    );
    addKernelPass(builder, {
      id: 'adjacency-publish-pass',
      invocationCount: BUILDING_COUNT * RING_VERTICES,
      bindings: [
        {name: 'components', buffer: getViewBuffer(recipe.outputs.components), access: 'read'},
        {name: 'nodeComponents', buffer: nodeComponents, access: 'read_write'}
      ],
      body: /* wgsl */ `
  // Component ids are building indices; spread them over the eight palette colors.
  let component = components[componentsOffset + index / ${RING_VERTICES}u];
  nodeComponents[nodeComponentsOffset + index] = select(component * 3u + component / 8u, component, component == 0xffffffffu);`
    });
    const edges = builder.upload(
      'ring-edges',
      'uint32x2',
      Uint32Array.from({length: BUILDING_COUNT * (RING_VERTICES - 1) * 2}, (_, slot) => {
        const edge = Math.floor(slot / 2);
        const building = Math.floor(edge / (RING_VERTICES - 1));
        const step = edge % (RING_VERTICES - 1);
        return building * RING_VERTICES + step + (slot % 2);
      })
    );
    // Every building shares its whole ring with itself; only runs between two buildings are walls.
    const isWall = builder.allocate('run-is-wall', 'uint32', runCapacity);
    addKernelPass(builder, {
      id: 'adjacency-wall-pass',
      invocationCount: runCapacity,
      bindings: [
        {name: 'left', buffer: leftLineIds, access: 'read'},
        {name: 'right', buffer: rightLineIds, access: 'read'},
        {name: 'isWall', buffer: isWall, access: 'read_write'}
      ],
      body: /* wgsl */ `
  isWall[isWallOffset + index] = u32(left[leftOffset + index] != right[rightOffset + index]);`
    });
    const shared = addPackedSegments(builder, 'shared-walls', runPositions, offsets, {
      lineCount: count,
      lineFlags: isWall
    });
    return {
      outputs: {positions, edges, nodeComponents, shared},
      update: current => positions.write(inputs.getRings(current.splitEvery, current.gap))
    };
  },
  draw: ({outputs}) => [
    {
      layer: 'segments',
      nodes: outputs.positions,
      edges: outputs.edges,
      values: outputs.nodeComponents,
      radius: 3
    },
    {layer: 'lines', segments: outputs.shared.segments, color: HIGHLIGHT_COLOR, radius: 5}
  ]
});
