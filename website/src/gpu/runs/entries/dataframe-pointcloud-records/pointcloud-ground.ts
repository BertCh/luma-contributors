// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointCloudGround (progressive morphological filter) on the lowest-point grid from
// GPUPointCloudSurfaces. Points kept as ground are orange; points the filter removes (trees) are
// blue. Larger windows and thresholds remove more.

import {
  GPUPointCloudGround,
  GPUPointCloudSurfaces
} from '@luma-contributors/analysis/pointcloud-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {clamp} from '../../interaction';
import {
  createCloudGrid,
  createCloudLayout,
  createMapPositions,
  createProfilePositions,
  createTerrainCloud,
  getHeightMaximum
} from './_lib/helpers';
import {parameters} from './pointcloud-ground.parameters';
import {summarizePointCloudGroundClasses} from './_lib/pointcloud-display';
import {
  addGroundClasses,
  addPointCloudDisplayCopy,
  groundClassLook
} from './_lib/pointcloud-display-run';

const POINT_COUNT = 9000;
const CELL_SIZE = 0.03;
// The initial thresholds the pointer's height picks among (applied once it rests).
const THRESHOLDS = [0.005, 0.02, 0.08, 0.2, 0.6] as const;
// Fixed props: on this tile `slope` and `maximumThreshold` change no classification, so they have no
// slider. The maximum is at least the largest initialThreshold the slider and gesture reach (0.8).
const SLOPE = 0.1;
const MAXIMUM_THRESHOLD = 0.8;
const RADII = [1, 2, 4, 8] as const;
// Status bit 1 (GPU_POINT_CLOUD_GROUND_STATUS_UNSUPPORTED): the point was not classified. Bit 2
// (INCOMPLETE_WINDOW) only marks a classification as provisional, so those points stay visible.
const UNSUPPORTED_BIT = 1;

function createInputs({
  random,
  world
}: {
  random: () => number;
  world: {width: number; height: number};
}) {
  const cloud = createTerrainCloud(random, world.width, POINT_COUNT);
  const layout = createCloudLayout(world.width, getHeightMaximum(cloud.positions, POINT_COUNT));
  return {
    cloud,
    layout,
    mapPositions: createMapPositions(layout, cloud.positions, POINT_COUNT),
    profilePositions: createProfilePositions(layout, cloud.positions, POINT_COUNT),
    heights: Float32Array.from({length: POINT_COUNT}, (_, row) => cloud.positions[row * 3 + 2])
  };
}
type Inputs = ReturnType<typeof createInputs>;
type Outputs = Record<
  | 'map'
  | 'profile'
  | 'displayMap'
  | 'displayProfile'
  | 'heights'
  | 'groundMask'
  | 'groundClasses'
  | 'status',
  RunBuffer
>;

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'Points colored by height (left of the curtain) and by the class the morphological filter wrote (right: ground orange, removed blue, faded where the window reached past cells with no points, so the classification is provisional); the squares are its window sizes. In this lidar tile ground and trees are far apart in height, so only initialThreshold above about 0.2 moves the classification; the window count changes few points. slope (0.1) and maximumThreshold (0.8) are fixed.',

  buildCache: 5,
  createInputs,
  build(builder, inputs, values) {
    const grid = createCloudGrid(inputs.cloud.width, CELL_SIZE);
    const cellCount = grid.width * grid.height;
    const positions = builder.upload('positions', 'float32x3', inputs.cloud.positions);
    const classes = builder.upload('classes', 'uint32', new Uint32Array(POINT_COUNT).fill(1));
    const minimum = builder.allocate('minimum', 'float32', cellCount);
    const coverage = builder.allocate('coverage', 'uint32', cellCount);
    builder.graph.add(
      new GPUPointCloudSurfaces({
        positions: positions.view,
        classifications: classes.view,
        groundClass: 1,
        grid,
        output: {
          terrain: minimum.view,
          surface: builder.allocate('surface', 'float32', cellCount).view,
          canopy: builder.allocate('canopy', 'float32', cellCount).view,
          coverage: coverage.view
        }
      })
    );
    const groundMask = builder.allocate('ground-mask', 'uint32', POINT_COUNT);
    const status = builder.allocate('status', 'uint32', POINT_COUNT);
    builder.graph.add(
      new GPUPointCloudGround({
        positions: positions.view,
        classifications: classes.view,
        grid,
        radii: [1, 2, 4, 8].slice(0, values.radiusCount),
        initialThreshold: values.initialThreshold,
        slope: SLOPE,
        maximumThreshold: MAXIMUM_THRESHOLD,
        minimumSurface: minimum.view,
        coverage: coverage.view,
        output: {
          groundMask: groundMask.view,
          outputClassifications: builder.allocate('output-classes', 'uint32', POINT_COUNT).view,
          status: status.view
        }
      })
    );
    const map = builder.upload('map', 'float32x2', inputs.mapPositions);
    const profile = builder.upload('profile', 'float32x2', inputs.profilePositions);
    return {
      outputs: {
        map,
        profile,
        displayMap: addPointCloudDisplayCopy(builder, 'supported-map', map, status, POINT_COUNT, {
          components: 2,
          invalidBits: UNSUPPORTED_BIT
        }),
        displayProfile: addPointCloudDisplayCopy(
          builder,
          'supported-profile',
          profile,
          status,
          POINT_COUNT,
          {components: 2, invalidBits: UNSUPPORTED_BIT}
        ),
        heights: builder.upload('heights', 'float32', inputs.heights),
        groundMask,
        groundClasses: addGroundClasses(builder, groundMask, status, POINT_COUNT, UNSUPPORTED_BIT),
        status
      }
    };
  },
  rebuildOn: ['radiusCount', 'initialThreshold'],
  // The pointer is a curtain between the input (height) and the output (ground mask); its height
  // picks initialThreshold among five values, and the squares around it are the window radii.
  interaction: {
    kind: 'swipe',
    hint: 'swipe height and ground, height sets threshold',
    rest: [0.5, 0.95 - 0.9 * 0.3],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const fraction = clamp((0.95 - pointer.unitY) / 0.9, 0, 0.9999);
      const level = Math.floor(fraction * THRESHOLDS.length);
      const mask = peek(outputs.groundMask);
      const status = peek(outputs.status);
      const summary = summarizePointCloudGroundClasses(mask, status, UNSUPPORTED_BIT);
      const glyphs: RunGlyph[] = [{shape: 'curtain', x: pointer.x}];
      for (const radius of RADII.slice(0, values.radiusCount)) {
        const half = radius * CELL_SIZE * inputs.layout.scale;
        glyphs.push({
          shape: 'box',
          minimum: [pointer.x - half, pointer.y - half],
          maximum: [pointer.x + half, pointer.y + half],
          dashed: true
        });
      }
      return {
        curtain: pointer.x,
        settle: {initialThreshold: THRESHOLDS[level]},

        glyphs,
        readout: `initialThreshold ${THRESHOLDS[level]} · ground ${status ? summary.ground : '…'} · removed ${summary.removed} · provisional ${summary.provisionalGround + summary.provisionalRemoved} (incomplete window, faded) · ${summary.unclassified} unclassified (hidden)`
      };
    }
  },
  draw: ({inputs, outputs, interaction, world, theme}) => {
    const curtain = interaction.curtain ?? world.width / 2;
    // The ramp starts below zero so ground returns land on a visible purple, not the near-black
    // bottom of the ramp that disappears on the dark theme.
    const heightMaximum = inputs.layout.heightMaximum;
    const height = {values: outputs.heights, range: [-0.5 * heightMaximum, heightMaximum]} as const;
    return [
      {
        layer: 'dots',
        positions: outputs.map,
        ...height,
        radius: 2.4,
        clip: {side: 'left', x: curtain}
      },
      {
        layer: 'dots',
        positions: outputs.profile,
        ...height,
        radius: 2,
        clip: {side: 'left', x: curtain}
      },
      {
        layer: 'dots',
        positions: outputs.displayMap,
        values: outputs.groundClasses,
        ...groundClassLook(theme),
        radius: 2.4,
        clip: {side: 'right', x: curtain}
      },
      {
        layer: 'dots',
        positions: outputs.displayProfile,
        values: outputs.groundClasses,
        ...groundClassLook(theme),
        radius: 2,
        clip: {side: 'right', x: curtain}
      }
    ];
  }
});
