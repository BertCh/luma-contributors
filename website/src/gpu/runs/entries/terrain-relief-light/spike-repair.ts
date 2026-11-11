// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainSpikeRepair on a DEM damaged the way Terrarium tiles are: blocks of pixels whose red
// byte is off by one or two steps (256 or 512 m), which read as sudden cliffs. The kernel finds
// connected regions separated by jumps that are multiples of the step and shifts them back. The
// context is the damaged input; the output is the repaired surface. Scenario: each block is offset
// by one step plus up to 30 m, so a seam must fit `tolerance` to count as damage.

import {GPUTerrainSpikeRepair} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {createTerrainInput, drawGrid, uploadTerrain} from './support/terrain';
import {parameters} from './spike-repair.parameters';

const SEAM_DEVIATION = 30;
// Fixed props: on this real tile `jump` and `tolerance` change nothing, so they have no slider.
const JUMP = 200;
const TOLERANCE = 40;

export default defineOperationRun({
  parameters,
  caption:
    'A damaged elevation tile in; the repaired tile out. Scenario: each damaged block is offset by one step plus up to 30 m, drawn at random; jump (200 m) and tolerance (40 m) are fixed.',
  createInputs: ({random, world}) => {
    const terrain = createTerrainInput(world, 48);
    const blocks = Array.from({length: 12}, () => ({
      column: Math.floor(random() * (terrain.columns - 10)),
      row: Math.floor(random() * (terrain.rows - 10)),
      width: 3 + Math.floor(random() * 6),
      height: 3 + Math.floor(random() * 6),
      // One whole step plus a random deviation, so the seam is not an exact multiple of the step:
      // `tolerance` decides which blocks still count as damaged.
      offset: (random() < 0.5 ? -1 : 1) * (256 + (random() * 2 - 1) * SEAM_DEVIATION)
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
        jump: JUMP,
        tolerance: TOLERANCE,
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
  // The pointer is a curtain: the damaged tile on the left, the repaired tile on the right.
  interaction: {
    kind: 'swipe',
    hint: 'swipe damaged and repaired',
    rest: [0.5, 0.5],
    resolve: (pointer, {values}) => ({
      curtain: pointer.x,
      glyphs: [{shape: 'curtain', x: pointer.x}],
      readout: `damaged | repaired · ${values.spikeCount} blocks shifted`
    })
  },
  draw: ({inputs, outputs, interaction, world}) => {
    const curtain = interaction.curtain ?? world.width / 2;
    const look = {range: [650, 2300] as const, hillshade: 0.6, elevation: 60};
    return [
      drawGrid(inputs, outputs.elevation, {...look, clip: {side: 'left', x: curtain}}),
      drawGrid(inputs, outputs.repaired, {...look, clip: {side: 'right', x: curtain}})
    ];
  }
});
