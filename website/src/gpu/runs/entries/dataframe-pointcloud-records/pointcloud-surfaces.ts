// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointCloudSurfaces rasterizes a classified cloud into terrain (lowest ground), surface (highest
// point) and height above terrain. The height grid shows above-ground structures
// over the bare ground returns. Class 2 is the survey ground class.

import {GPUPointCloudSurfaces} from '@luma-contributors/analysis/pointcloud-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {getRasterCell} from '../../pick';
import {addFloatCopy} from './_lib/convert';
import {createCloudGrid, createTerrainCloud, getJitteredPositions} from './_lib/helpers';
import {parameters} from './pointcloud-surfaces.parameters';

const POINT_COUNT = 9000;

function createInputs({
  random,
  world
}: {
  random: () => number;
  world: {width: number; height: number};
}) {
  const cloud = createTerrainCloud(random, world.width, POINT_COUNT);
  // One color range for both surfaces: the survey's ground relief, with tree tops clipped to the
  // top of the ramp, so the terrain is not flattened by the canopy.
  const ground = Array.from({length: POINT_COUNT}, (_, row) => cloud.positions[row * 3 + 2])
    .filter((_, row) => cloud.classes[row] === 2)
    .sort((first, second) => first - second);
  const low = ground[Math.floor(ground.length * 0.02)];
  const high = ground[Math.floor(ground.length * 0.98)];
  return {cloud, world, elevationRange: [low, low + (high - low) * 1.6] as const};
}
type Inputs = ReturnType<typeof createInputs>;
type Outputs = Record<'terrain' | 'surface' | 'canopy', RunBuffer>;

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'Surface of the highest return (left of the curtain) and bare terrain (right) in each sampled cell; the difference is the canopy height.',
  createInputs,
  build(builder, inputs, values) {
    const cloud = inputs.cloud;
    const grid = createCloudGrid(cloud.width, values.cellSize);
    const cellCount = grid.width * grid.height;
    const positions = builder.allocate('positions', 'float32x3', POINT_COUNT);
    const terrain = builder.allocate('terrain', 'float32', cellCount);
    const surface = builder.allocate('surface', 'float32', cellCount);
    const canopy = builder.allocate('canopy', 'float32', cellCount);
    builder.graph.add(
      new GPUPointCloudSurfaces({
        positions: positions.view,
        classifications: builder.upload('classes', 'uint32', cloud.classes).view,
        groundClass: values.groundClass,
        grid,
        terrain: terrain.view,
        surface: surface.view,
        canopy: canopy.view,
        coverage: builder.allocate('coverage', 'uint32', cellCount).view
      })
    );
    const flip = {flipColumns: grid.width};
    return {
      outputs: {
        terrain: addFloatCopy(builder, 'terrain-grid', terrain, cellCount, flip),
        surface: addFloatCopy(builder, 'surface-grid', surface, cellCount, flip),
        canopy: addFloatCopy(builder, 'canopy-grid', canopy, cellCount, flip)
      },
      update: current => positions.write(getJitteredPositions(cloud, current.noise))
    };
  },
  rebuildOn: ['groundClass', 'cellSize'],
  // The pointer is a curtain between the surface (tree tops) and the terrain (bare earth); the
  // canopy height of the cell under it is read out.
  interaction: {
    kind: 'swipe',
    hint: 'swipe surface and terrain',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const grid = createCloudGrid(inputs.cloud.width, values.cellSize);
      const cell = getRasterCell(
        grid.width,
        grid.height,
        [0, 0, inputs.world.width, 1],
        pointer.x,
        pointer.y
      );
      const canopy = peek(outputs.canopy)?.[cell.index];
      const metres = canopy === undefined ? undefined : canopy / inputs.cloud.scale;
      return {
        curtain: pointer.x,
        glyphs: [{shape: 'curtain', x: pointer.x}],
        readout: `cell ${values.cellSize} · canopy ${
          canopy === undefined || !Number.isFinite(canopy) ? '…' : canopy.toFixed(2)
        }${metres === undefined || !Number.isFinite(metres) ? '' : ` (${metres.toFixed(0)} m)`} here`
      };
    }
  },
  draw: ({inputs, outputs, values, interaction, world}) => {
    const grid = createCloudGrid(inputs.cloud.width, values.cellSize);
    const curtain = interaction.curtain ?? world.width / 2;
    const look = {
      layer: 'raster' as const,
      columns: grid.width,
      rows: grid.height,
      range: inputs.elevationRange,
      hillshade: 0.8
    };
    return [
      {...look, values: outputs.surface, clip: {side: 'left', x: curtain}},
      {...look, values: outputs.terrain, clip: {side: 'right', x: curtain}}
    ];
  }
});
