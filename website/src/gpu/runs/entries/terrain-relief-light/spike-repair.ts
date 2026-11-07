// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainSpikeRepair on a DEM damaged the way Terrarium tiles are: blocks of pixels whose red
// byte is off by one or two steps (256 or 512 m), which read as sudden cliffs. The kernel finds
// connected regions separated by jumps that are multiples of the step and shifts them back. The
// context is the damaged input; the output is the repaired surface.

import {GPUTerrainSpikeRepair} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {createTerrainInput, drawElevationContext, drawGrid, uploadTerrain} from './support/terrain';
import {parameters} from './spike-repair.parameters';

export default defineOperationRun({
  parameters,
  rebuildOn: ['jump', 'tolerance'],
  caption: 'A damaged elevation tile in; the repaired tile out.',
  createInputs: ({random, world}) => {
    const terrain = createTerrainInput(random, world, 48);
    const blocks = Array.from({length: 12}, () => ({
      column: Math.floor(random() * (terrain.columns - 10)),
      row: Math.floor(random() * (terrain.rows - 10)),
      width: 3 + Math.floor(random() * 6),
      height: 3 + Math.floor(random() * 6),
      offset: (random() < 0.5 ? -1 : 1) * 256 * (random() < 0.7 ? 1 : 2)
    }));
    return {...terrain, blocks};
  },
  build(builder, terrain, values) {
    const uploaded = uploadTerrain(builder, terrain);
    const {grid, band, elevation, cellCount} = uploaded;
    const repaired = builder.allocate('repaired', 'float32', cellCount);
    const outputValidity = builder.allocate('output-validity', 'uint32', cellCount);
    const statistics = builder.allocate('statistics', 'uint32', 5);
    builder.graph.add(
      new GPUTerrainSpikeRepair({
        grid,
        elevation: band,
        step: 256,
        jump: values.jump,
        tolerance: values.tolerance,
        maximumComponentIterations: 32,
        output: {
          values: repaired.view,
          outputValidity: outputValidity.view,
          statistics: statistics.view
        }
      })
    );
    return {
      outputs: {elevation, repaired},
      update: current => {
        const damaged = Float32Array.from(terrain.elevation);
        for (const block of terrain.blocks.slice(0, current.spikeCount)) {
          for (let row = block.row; row < block.row + block.height; row++) {
            for (let column = block.column; column < block.column + block.width; column++) {
              damaged[row * terrain.columns + column] += block.offset;
            }
          }
        }
        elevation.write(damaged);
      }
    };
  },
  draw: ({inputs, outputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(inputs, outputs.repaired, {hillshade: 0.5, elevation: 60})
  ]
});
