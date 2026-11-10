// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointCloudFeatures after GPUPointCloudNeighborhoods: the eigenvalues of each point's local
// covariance become shape descriptors. Ground is planar (bright with `planarity`); tree crowns
// scatter in all directions (bright with `scattering`).

import {
  GPUPointCloudFeatures,
  GPUPointCloudNeighborhoods
} from '@luma-contributors/analysis/pointcloud-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {chooseSticky} from '../../pick';
import {addFloatCopy} from './_lib/convert';
import {
  createCloudLayout,
  createCloudProbe,
  createMapPositions,
  createProfilePositions,
  createTerrainCloud,
  fromMapPoint,
  getHeightMaximum,
  getJitteredPositions,
  getProfilePoint,
  toMapPoint,
  writeRowPositions
} from './_lib/helpers';
import {parameters} from './pointcloud-features.parameters';

const POINT_COUNT = 5000;
const RADIUS = 0.05;
const NEIGHBOR_LIMIT = 160;
// Linear, planar and scattered points of the 'dominant' view.
const DESCRIPTOR_COLORS = [
  [0.95, 0.6, 0.2],
  [0.3, 0.7, 1],
  [0.4, 0.8, 0.4]
] as const;

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
  | 'map'
  | 'profile'
  | 'linearity'
  | 'planarity'
  | 'scattering'
  | 'neighborMap'
  | 'neighborProfile'
  | 'dominant',
  RunBuffer
>;

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'Local shape descriptor per point from covariance eigenvalues; the strip below is the side profile (x against height).',
  createInputs,
  build(builder, inputs, values) {
    const cloud = inputs.cloud;
    const capacity = values.capacity;
    const positions = builder.allocate('positions', 'float32x3', POINT_COUNT);
    const indices = builder.allocate('indices', 'uint32', POINT_COUNT * capacity);
    const counts = builder.allocate('counts', 'uint32', POINT_COUNT);
    builder.graph.add(
      new GPUPointCloudNeighborhoods({
        positions: positions.view,
        radius: RADIUS,
        capacity,
        indices: indices.view,
        squaredDistances: builder.allocate('squared-distances', 'float32', POINT_COUNT * capacity)
          .view,
        counts: counts.view
      })
    );
    const shapes = builder.allocate('shapes', 'float32x3', POINT_COUNT);
    builder.graph.add(
      new GPUPointCloudFeatures({
        positions: positions.view,
        indices: indices.view,
        counts: counts.view,
        capacity,
        normals: builder.allocate('normals', 'float32x3', POINT_COUNT).view,
        eigenvalues: builder.allocate('eigenvalues', 'float32x3', POINT_COUNT).view,
        shapes: shapes.view,
        status: builder.allocate('status', 'uint32', POINT_COUNT).view
      })
    );
    return {
      outputs: {
        map: builder.upload('map', 'float32x2', inputs.mapPositions),
        profile: builder.upload('profile', 'float32x2', inputs.profilePositions),
        neighborMap: builder.allocate('neighbor-map', 'float32x2', NEIGHBOR_LIMIT),
        neighborProfile: builder.allocate('neighbor-profile', 'float32x2', NEIGHBOR_LIMIT),
        dominant: builder.allocate('dominant', 'float32', POINT_COUNT),
        linearity: addFloatCopy(builder, 'linearity', shapes, POINT_COUNT, {componentStart: 0}),
        planarity: addFloatCopy(builder, 'planarity', shapes, POINT_COUNT, {componentStart: 1}),
        scattering: addFloatCopy(builder, 'scattering', shapes, POINT_COUNT, {componentStart: 2})
      },
      update: current => positions.write(getJitteredPositions(cloud, current.noise))
    };
  },
  rebuildOn: ['capacity'],
  // The pointer is a point: its neighborhood (the radius the GPU searched) is outlined on the map
  // and the profile, its neighbors are bright, and the three descriptors are read out.
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
      // The strongest of the three descriptors per point, as a category for the 'dominant' view.
      const linearity = peek(outputs.linearity);
      const planarity = peek(outputs.planarity);
      const scattering = peek(outputs.scattering);
      if (linearity && planarity && scattering) {
        outputs.dominant.write(
          Float32Array.from(linearity, (line, index) =>
            planarity[index] >= line && planarity[index] >= scattering[index]
              ? 1
              : scattering[index] > line
                ? 2
                : 0
          )
        );
      }
      const [x, y] = fromMapPoint(layout, pointer.x, Math.max(pointer.y, layout.offsetY));
      const row = chooseSticky(probe.index, x, y, inputs.focus.row);
      inputs.focus.row = row;
      if (row < 0) return {};
      const neighbors = probe.findNeighbors(row, RADIUS).slice(0, NEIGHBOR_LIMIT);
      const count = writeRowPositions(outputs.neighborMap, inputs.mapPositions, neighbors);
      writeRowPositions(outputs.neighborProfile, inputs.profilePositions, neighbors);
      const center = toMapPoint(layout, cloud.footprint[row * 2], cloud.footprint[row * 2 + 1]);
      const profileCenter = getProfilePoint(
        layout,
        cloud.positions[row * 3],
        cloud.positions[row * 3 + 2]
      );
      const halfWidth = RADIUS * layout.scale;
      const halfHeight = (RADIUS / layout.heightMaximum) * layout.profileHeight;
      const text = (buffer: RunBuffer) => {
        const value = peek(buffer)?.[row];
        return value === undefined ? '…' : value.toFixed(2);
      };
      return {
        focus: count,
        glyphs: [
          {shape: 'ring', at: center, radius: halfWidth},
          {
            shape: 'box',
            minimum: [profileCenter[0] - halfWidth, profileCenter[1] - halfHeight],
            maximum: [profileCenter[0] + halfWidth, profileCenter[1] + halfHeight]
          }
        ],
        readout: `lin ${text(outputs.linearity)} · planar ${text(outputs.planarity)} · scatter ${text(outputs.scattering)} · ${count} neighbors`
      };
    }
  },
  draw: ({outputs, values, interaction}) => {
    const isDominant = values.shape === 3;
    const shape = [outputs.linearity, outputs.planarity, outputs.scattering, outputs.dominant][
      values.shape
    ];
    const look = isDominant
      ? ({ramp: 'categorical', categories: DESCRIPTOR_COLORS} as const)
      : ({range: [0, 1]} as const);
    const highlight = typeof interaction.focus === 'number' ? Math.max(1, interaction.focus) : 1;
    return [
      {layer: 'dots', positions: outputs.map, values: shape, ...look, radius: 2.6},
      {layer: 'dots', positions: outputs.profile, values: shape, ...look, radius: 2.2},
      {
        layer: 'dots',
        positions: outputs.neighborMap,
        count: highlight,
        color: [1, 1, 1, 1],
        radius: 3.6
      },
      {
        layer: 'dots',
        positions: outputs.neighborProfile,
        count: highlight,
        color: [1, 1, 1, 1],
        radius: 3
      }
    ];
  }
});
