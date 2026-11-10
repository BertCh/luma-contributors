// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointCloudSMRF (simple morphological filter). The raster is the provisional ground surface the
// GPU inpainted under the cloud; the dots are the points it classified: ground orange, removed
// (trees and other objects) blue.

import {GPUPointCloudSMRF} from '@luma-contributors/analysis/pointcloud-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {clamp} from '../../interaction';
import {addFloatCopy} from './_lib/convert';
import {createCloudGrid, createTerrainCloud} from './_lib/helpers';
import {parameters} from './pointcloud-smrf.parameters';

const POINT_COUNT = 9000;
const CELL_SIZE = 0.03;
// The thresholds the pointer's height picks among (applied once it rests).
const THRESHOLDS = [0.01, 0.04, 0.1, 0.2, 0.3] as const;

function createInputs({
  random,
  world
}: {
  random: () => number;
  world: {width: number; height: number};
}) {
  const cloud = createTerrainCloud(random, world.width, POINT_COUNT);
  // The surface color range spans the surveyed ground returns (class 2), so the relief fills the ramp.
  const ground = Array.from({length: POINT_COUNT}, (_, row) => cloud.positions[row * 3 + 2])
    .filter((_, row) => cloud.classes[row] === 2)
    .sort((first, second) => first - second);
  return {
    cloud,
    width: world.width,
    surfaceRange: [
      ground[Math.floor(ground.length * 0.02)],
      ground[Math.floor(ground.length * 0.98)]
    ] as const
  };
}
type Inputs = ReturnType<typeof createInputs>;
type Outputs = Record<'footprint' | 'groundMask' | 'surface', RunBuffer>;

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'Inpainted ground surface (left of the curtain, hillshaded) and the points classified as ground (orange) or removed (blue) (right). The tile has little intermediate relief, so window barely changes the result.',
  buildCache: 5,
  createInputs,
  build(builder, inputs, values) {
    const grid = createCloudGrid(inputs.cloud.width, CELL_SIZE);
    const cellCount = grid.width * grid.height;
    const groundSurface = builder.allocate('ground-surface', 'float32', cellCount);
    const groundMask = builder.allocate('ground-mask', 'uint32', POINT_COUNT);
    builder.graph.add(
      new GPUPointCloudSMRF({
        positions: builder.upload('positions', 'float32x3', inputs.cloud.positions).view,
        classifications: builder.upload('classes', 'uint32', new Uint32Array(POINT_COUNT).fill(1))
          .view,
        grid,
        slope: values.slope,
        window: values.window,
        threshold: values.threshold,
        cut: values.cut,
        groundSurface: groundSurface.view,
        groundMask: groundMask.view,
        outputClassifications: builder.allocate('output-classes', 'uint32', POINT_COUNT).view,
        status: builder.allocate('status', 'uint32', POINT_COUNT).view
      })
    );
    return {
      outputs: {
        footprint: builder.upload('footprint', 'float32x2', inputs.cloud.footprint),
        groundMask,
        surface: addFloatCopy(builder, 'surface-grid', groundSurface, cellCount, {
          flipColumns: grid.width
        })
      }
    };
  },
  rebuildOn: ['slope', 'window', 'threshold', 'cut'],
  // The pointer is a curtain between the ground surface and the classified points; its height
  // picks the threshold among five values (applied once it rests).
  interaction: {
    kind: 'swipe',
    hint: 'swipe surface and points, height sets threshold',
    rest: [0.5, 0.95 - 0.9 * 0.3],
    resolve(pointer, {outputs, peek}) {
      const fraction = clamp((0.95 - pointer.unitY) / 0.9, 0, 0.9999);
      const level = Math.floor(fraction * THRESHOLDS.length);
      const mask = peek(outputs.groundMask);
      let ground = 0;
      for (let row = 0; mask && row < mask.length; row++) ground += mask[row];
      return {
        curtain: pointer.x,
        settle: {threshold: THRESHOLDS[level]},
        glyphs: [{shape: 'curtain', x: pointer.x}],
        readout: `threshold ${THRESHOLDS[level]} · ground ${mask ? ground : '…'} of ${POINT_COUNT}`
      };
    }
  },
  draw: ({inputs, outputs, interaction, world}) => {
    const grid = createCloudGrid(inputs.width, CELL_SIZE);
    const curtain = interaction.curtain ?? world.width / 2;
    return [
      {
        layer: 'raster',
        values: outputs.surface,
        columns: grid.width,
        rows: grid.height,
        range: inputs.surfaceRange,
        hillshade: 0.8,
        smoothing: true,
        clip: {side: 'left', x: curtain}
      },
      {
        layer: 'dots',
        positions: outputs.footprint,
        values: outputs.groundMask,
        radius: 2.4,
        clip: {side: 'right', x: curtain}
      }
    ];
  }
});
