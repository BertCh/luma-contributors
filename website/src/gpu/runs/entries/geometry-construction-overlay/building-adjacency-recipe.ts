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
import type {RunGlyph} from '../../interaction';
import {addKernelPass, getViewBuffer} from './shared/kernel';
import {addPackedSegments, HIGHLIGHT_COLOR} from './shared/lines';
import {createCityLayout, findBuildingAt} from './shared/urban';
import {parameters} from './building-adjacency-recipe.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Terraces found from shared walls: one color per connected group of buildings.',
  createInputs: ({world}) => {
    const layout = createCityLayout(world, 24);
    const offsets = [0];
    const vertexOwners: number[] = [];
    const edges: number[] = [];
    layout.rings.forEach((ring, building) => {
      const offset = vertexOwners.length;
      for (let corner = 0; corner <= ring.length; corner++) {
        vertexOwners.push(building);
        if (corner < ring.length) edges.push(offset + corner, offset + corner + 1);
      }
      offsets.push(vertexOwners.length);
    });
    const getRings = (splitEvery: number, gap: number, hovered: number) => {
      const positions: number[] = [];
      layout.rings.forEach((ring, building) => {
        const [centerX, centerY] = layout.centers[building];
        // Recede every nth footprint to simulate interruptions in the terraces.
        const isReceded = (building + 1) % splitEvery === 0 || building === hovered;
        const factor = isReceded ? 1 - Math.max(gap, 0.02) * 8 : 1;
        for (const [x, y] of [...ring, ring[0]]) {
          positions.push(centerX + (x - centerX) * factor, centerY + (y - centerY) * factor);
        }
      });
      return Float32Array.from(positions);
    };
    return {
      layout,
      world,
      hovered: {building: -1},
      count: layout.rings.length,
      getRings,
      offsets: Uint32Array.from(offsets),
      edges: Uint32Array.from(edges),
      vertexOwners: Uint32Array.from(vertexOwners)
    };
  },
  build(builder, inputs, values) {
    const initial = inputs.getRings(values.splitEvery, values.gap, inputs.hovered.building);
    const positions = builder.upload('rings', 'float32x2', initial);
    const lineOffsets = builder.upload('ring-offsets', 'uint32', inputs.offsets);
    const runCapacity = 512;
    const vertexCapacity = 4096;
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
      Uint32Array.from({length: inputs.count}, (_, index) => index)
    );
    const orientations = builder.upload('orientations', 'float32', new Float32Array(inputs.count));
    const recipe = addBuildingAdjacencyRecipe(builder.graph, {
      sharedPaths: sharedOutput,
      lineBuildingIds: owners.view,
      orientations: orientations.view
    });
    const nodeComponents = builder.allocate(
      'ring-components',
      'uint32',
      inputs.vertexOwners.length
    );
    const vertexOwners = builder.upload('vertex-owners', 'uint32', inputs.vertexOwners);
    addKernelPass(builder, {
      id: 'adjacency-publish-pass',
      invocationCount: inputs.vertexOwners.length,
      bindings: [
        {name: 'components', buffer: getViewBuffer(recipe.outputs.components), access: 'read'},
        {name: 'vertexOwners', buffer: vertexOwners, access: 'read'},
        {name: 'nodeComponents', buffer: nodeComponents, access: 'read_write'}
      ],
      body: /* wgsl */ `
  // Component ids are building indices; spread them over the eight palette colors.
  let component = components[componentsOffset + vertexOwners[vertexOwnersOffset + index]];
  nodeComponents[nodeComponentsOffset + index] = select(component * 3u + component / 8u, component, component == 0xffffffffu);`
    });
    const edges = builder.upload('ring-edges', 'uint32x2', inputs.edges);
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
      update: current =>
        positions.write(inputs.getRings(current.splitEvery, current.gap, inputs.hovered.building))
    };
  },
  // The pointer pushes the building under it back from its neighbors (a pointer-owned input of the
  // recipe): its shared walls disappear and its terrace splits in two.
  interaction: {
    kind: 'probe',
    hint: 'hover a building',
    rest: ({inputs, toUnit}) => toUnit([inputs.world.width / 2, inputs.layout.streetY]),
    resolve(pointer, {inputs, outputs, peek}) {
      const building = findBuildingAt(inputs.layout, pointer.x, pointer.y);
      const changed = building !== inputs.hovered.building;
      inputs.hovered.building = building;
      if (building < 0) return {rerun: changed, readout: 'no building hovered'};
      const components = peek(outputs.nodeComponents);
      const glyphs: RunGlyph[] = [
        {shape: 'polyline', points: inputs.layout.rings[building].flat(), closed: true}
      ];
      const terraces = new Set<number>();
      if (components) {
        for (let member = 0; member < inputs.count; member++) {
          terraces.add(components[inputs.offsets[member]]);
        }
      }
      return {
        rerun: changed,
        glyphs,
        readout: `building ${building} set back${components ? ` · ${terraces.size} terraces` : ''}`
      };
    }
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
