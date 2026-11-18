// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointCloudHeight after GPUPointCloudSurfaces: each point's height above the terrain grid under
// it. Ground points are near zero (muted base dots); structures rise to the top of the ramp.

import {
  GPUPointCloudHeight,
  GPUPointCloudSurfaces
} from '@luma-contributors/analysis/pointcloud-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {chooseSticky} from '../../pick';
import {createLegend, mergeText, type RunLabel} from '../../text';
import {
  createCloudGrid,
  createCloudLayout,
  createCloudProbe,
  createMapPositions,
  createProfilePositions,
  createTerrainCloud,
  fromMapPoint,
  getHeightMaximum,
  getJitteredPositions,
  getProfilePoint,
  toMapPoint
} from './_lib/helpers';
import {parameters} from './pointcloud-height.parameters';
import {getPointCloudHeightSupportText} from './_lib/pointcloud-display';
import {addPointCloudDisplayCopy} from './_lib/pointcloud-display-run';

const POINT_COUNT = 9000;

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
    probe: createCloudProbe(cloud),
    mapPositions: createMapPositions(layout, cloud.positions, POINT_COUNT),
    profilePositions: createProfilePositions(layout, cloud.positions, POINT_COUNT),
    focus: {row: -1}
  };
}
type Inputs = ReturnType<typeof createInputs>;
type Outputs = Record<
  'map' | 'profile' | 'displayMap' | 'displayProfile' | 'heights' | 'status',
  RunBuffer
>;

type Extra = {maximumHeight: number};

export default defineOperationRun<Inputs, Outputs, Extra>({
  parameters,
  caption: 'Height above ground per point; the strip below is the side profile (x against height).',
  createInputs,
  build(builder, inputs, values) {
    const grid = createCloudGrid(inputs.cloud.width, values.cellSize);
    const cellCount = grid.width * grid.height;
    const positions = builder.allocate('positions', 'float32x3', POINT_COUNT);
    const terrain = builder.allocate('terrain', 'float32', cellCount);
    const coverage = builder.allocate('coverage', 'uint32', cellCount);
    builder.graph.add(
      new GPUPointCloudSurfaces({
        positions: positions.view,
        classifications: builder.upload('classes', 'uint32', inputs.cloud.classes).view,
        groundClass: 2,
        grid,
        output: {
          terrain: terrain.view,
          surface: builder.allocate('surface', 'float32', cellCount).view,
          canopy: builder.allocate('canopy', 'float32', cellCount).view,
          coverage: coverage.view
        }
      })
    );
    const heights = builder.allocate('heights', 'float32', POINT_COUNT);
    const status = builder.allocate('status', 'uint32', POINT_COUNT);
    builder.graph.add(
      new GPUPointCloudHeight({
        positions: positions.view,
        grid,
        terrain: terrain.view,
        coverage: coverage.view,
        coverageBit: 2,
        interpolation:
          values.interpolation === 0
            ? undefined // the prop's default: sample the containing cell
            : values.interpolation === 1
              ? {method: 'bilinear'}
              : {method: 'idw', radius: values.cellSize * 3},
        output: {
          heights: heights.view,
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
          components: 2
        }),
        displayProfile: addPointCloudDisplayCopy(
          builder,
          'supported-profile',
          profile,
          status,
          POINT_COUNT,
          {components: 2}
        ),
        heights: addPointCloudDisplayCopy(
          builder,
          'supported-heights',
          heights,
          status,
          POINT_COUNT
        ),
        status
      },
      update: current => positions.write(getJitteredPositions(inputs.cloud, current.noise))
    };
  },
  // The tallest height the GPU wrote sets the ramp, so the structures use all of it.
  async afterRun({outputs, inputs}) {
    const bytes = await outputs.heights.buffer.readAsync();
    const heights = new Float32Array(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
    );
    let maximum = 0;
    for (const height of heights) if (Number.isFinite(height) && height > maximum) maximum = height;
    return {maximumHeight: Math.max(maximum, 0.01) / inputs.cloud.scale};
  },
  rebuildOn: ['interpolation', 'cellSize'],
  // The pointer is a point: the terrain cell under it is outlined and a tick on the profile joins
  // the point to the ground height the GPU subtracted.
  interaction: {
    kind: 'probe',
    hint: 'hover a point',
    rest: ({inputs}) => {
      const row = inputs.probe.restRow;
      const [x, y] = toMapPoint(
        inputs.layout,
        inputs.cloud.footprint[row * 2],
        inputs.cloud.footprint[row * 2 + 1]
      );
      return [x / inputs.layout.width, 1 - y];
    },
    resolve(pointer, {inputs, outputs, values, peek}) {
      const {cloud, layout, probe} = inputs;
      const [x, y] = fromMapPoint(layout, pointer.x, Math.max(pointer.y, layout.offsetY));
      // The probe takes the tallest return within a tenth of the frame of the pointer, so it lands on a tree
      // or a roof when there is one under it (the nearest point is nearly always ground).
      const nearby = probe.index.within(x, y, cloud.width * 0.1);
      let row = chooseSticky(probe.index, x, y, inputs.focus.row);
      for (const candidate of nearby) {
        if (row < 0 || cloud.positions[candidate * 3 + 2] > cloud.positions[row * 3 + 2]) {
          row = candidate;
        }
      }
      inputs.focus.row = row;
      if (row < 0) return {};
      const status = peek(outputs.status)?.[row];
      const height = status === 0 ? peek(outputs.heights)?.[row] : undefined;
      const pointX = cloud.positions[row * 3];
      const pointY = cloud.positions[row * 3 + 1];
      const pointZ = cloud.positions[row * 3 + 2];
      const cellSize = values.cellSize;
      const cellMinimum = toMapPoint(
        layout,
        Math.floor(pointX / cellSize) * cellSize,
        Math.floor(pointY / cellSize) * cellSize
      );
      const top = getProfilePoint(layout, pointX, pointZ);
      const ground =
        height === undefined ? top : getProfilePoint(layout, pointX, Math.max(pointZ - height, 0));
      const metres = height === undefined ? undefined : height / cloud.scale;
      return {
        glyphs: [
          {
            shape: 'box',
            minimum: cellMinimum,
            maximum: [
              cellMinimum[0] + cellSize * layout.scale,
              cellMinimum[1] + cellSize * layout.scale
            ]
          },
          {shape: 'line', from: ground, to: top},
          {shape: 'dot', at: top}
        ],
        readout: `${getPointCloudHeightSupportText(status)} · height ${height === undefined ? 'unavailable' : height.toFixed(2)}${
          metres === undefined ? '' : ` (${metres.toFixed(0)} m)`
        } · ${['cell', 'bilinear', 'idw'][values.interpolation]}`
      };
    }
  },
  draw: ({inputs, outputs, extra, theme}) => {
    const isDark = theme.background[0] < 0.5;
    const maximum = extra ? extra.maximumHeight * inputs.cloud.scale : 0.2;
    const {layout} = inputs;
    const look = {values: outputs.heights, range: [0, maximum], fade: true} as const;
    const labels: RunLabel[] = [
      {
        text: 'top view',
        at: [layout.offsetX, 0.995],
        anchor: 'top-left',
        offset: [2, 0],
        role: 'title'
      },
      {
        text: 'side profile',
        at: [layout.offsetX, layout.profileBottom + layout.profileHeight + 0.004],
        anchor: 'bottom-left',
        offset: [2, 0],
        role: 'title'
      }
    ];
    return [
      {layer: 'dots', context: true, positions: outputs.map, radius: 2.4},
      {layer: 'dots', context: true, positions: outputs.profile, radius: 2},
      {layer: 'dots', positions: outputs.displayMap, ...look, radius: 2.8},
      {layer: 'dots', positions: outputs.displayProfile, ...look, radius: 2.2},
      {
        layer: 'text',
        ...mergeText(
          {labels},
          createLegend({
            corner: 'top-right',
            title: 'height above ground',
            entries: [
              {label: 'ground (0 m)', swatch: [0.45, 0.48, 0.58]},
              {
                label: extra
                  ? `top of the ramp: ${Math.round(extra.maximumHeight)} m`
                  : 'top of the ramp: higher',
                swatch: isDark ? ([1, 0.93, 0.55] as const) : ([0.35, 0.05, 0.35] as const)
              }
            ]
          })
        )
      }
    ];
  }
});
