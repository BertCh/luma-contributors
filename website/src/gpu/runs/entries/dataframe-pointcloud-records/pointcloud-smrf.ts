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
import {summarizePointCloudGroundClasses} from './_lib/pointcloud-display';
import {
  addGroundClasses,
  addPointCloudDisplayCopy,
  groundClassLook
} from './_lib/pointcloud-display-run';

// Fixed prop: on this tile the window changes few classifications, so it has no slider. It is kept
// small because the filter marks points within about twice the window (and four times the cut) of the
// tile edge as provisional; a larger window left no fully supported point on this 1-unit-tall tile.
const SMOOTHING_WINDOW = 0.12;
const POINT_COUNT = 9000;
const CELL_SIZE = 0.03;
// The thresholds the pointer's height picks among (applied once it rests).
const THRESHOLDS = [0.01, 0.04, 0.1, 0.2, 0.3] as const;
// Status bits 1, 2 and 4 (INVALID, NO_TERRAIN, EXCLUDED): the point was not classified. Bit 8
// (INPAINTED: inpainted terrain or within the filter margin of the edge) marks a provisional
// classification; on this small tile the margin covers most points, so they stay visible.
const UNCLASSIFIED_BITS = 1 | 2 | 4;

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
type Outputs = Record<
  'footprint' | 'groundMask' | 'groundClasses' | 'surface' | 'status',
  RunBuffer
>;

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'Inpainted ground surface (left of the curtain, hillshaded) and the points classified as ground (orange) or removed (blue) (right), faded where the terrain was inpainted or the point is within the filter margin of the tile edge (provisional). Raising cut widens that margin. The tile has little intermediate relief, so the window (fixed at 0.12) barely changes the result.',
  buildCache: 5,
  createInputs,
  build(builder, inputs, values) {
    const grid = createCloudGrid(inputs.cloud.width, CELL_SIZE);
    const cellCount = grid.width * grid.height;
    const groundSurface = builder.allocate('ground-surface', 'float32', cellCount);
    const groundMask = builder.allocate('ground-mask', 'uint32', POINT_COUNT);
    const status = builder.allocate('status', 'uint32', POINT_COUNT);
    builder.graph.add(
      new GPUPointCloudSMRF({
        positions: builder.upload('positions', 'float32x3', inputs.cloud.positions).view,
        classifications: builder.upload('classes', 'uint32', new Uint32Array(POINT_COUNT).fill(1))
          .view,
        grid,
        slope: values.slope,
        window: SMOOTHING_WINDOW,
        threshold: values.threshold,
        cut: values.cut,
        output: {
          groundSurface: groundSurface.view,
          groundMask: groundMask.view,
          outputClassifications: builder.allocate('output-classes', 'uint32', POINT_COUNT).view,
          status: status.view
        }
      })
    );
    const footprint = builder.upload('footprint', 'float32x2', inputs.cloud.footprint);
    return {
      outputs: {
        footprint: addPointCloudDisplayCopy(
          builder,
          'supported-footprint',
          footprint,
          status,
          POINT_COUNT,
          {components: 2, invalidBits: UNCLASSIFIED_BITS}
        ),
        groundMask,
        groundClasses: addGroundClasses(
          builder,
          groundMask,
          status,
          POINT_COUNT,
          UNCLASSIFIED_BITS
        ),
        status,
        surface: addFloatCopy(builder, 'surface-grid', groundSurface, cellCount)
      }
    };
  },
  rebuildOn: ['slope', 'threshold', 'cut'],
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
      const status = peek(outputs.status);
      const summary = summarizePointCloudGroundClasses(mask, status, UNCLASSIFIED_BITS);
      return {
        curtain: pointer.x,
        settle: {threshold: THRESHOLDS[level]},
        glyphs: [{shape: 'curtain', x: pointer.x}],
        readout: `threshold ${THRESHOLDS[level]} · ground ${status ? summary.ground : '…'} · removed ${summary.removed} · provisional ${summary.provisionalGround + summary.provisionalRemoved} (inpainted or near the edge, faded) · ${summary.unclassified} unclassified (hidden)`
      };
    }
  },
  draw: ({inputs, outputs, interaction, world, theme}) => {
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
        values: outputs.groundClasses,
        ...groundClassLook(theme),
        radius: 2.4,
        clip: {side: 'right', x: curtain}
      }
    ];
  }
});
