// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointCloudNeighborhoods on a lidar-like cloud (top view). Each point is colored by how many
// neighbors the GPU found within `radius` (counts); the table keeps at most `capacity` of them.

import {GPUPointCloudNeighborhoods} from '@luma-contributors/analysis/pointcloud-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {chooseSticky} from '../../pick';
import {addIntegerToFloat} from './_lib/convert';
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
import {parameters} from './pointcloud-neighborhoods.parameters';

const POINT_COUNT = 3000;
const HIGHLIGHT_LIMIT = 160;

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
  'map' | 'profile' | 'counts' | 'keptMap' | 'keptProfile' | 'truncatedMap' | 'truncatedProfile',
  RunBuffer
>;

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'Neighbors within the radius, per point; the strip below is the side profile (x against height).',
  createInputs,
  build(builder, inputs, values) {
    const positions = builder.allocate('positions', 'float32x3', POINT_COUNT);
    const counts = builder.allocate('counts', 'uint32', POINT_COUNT);
    builder.graph.add(
      new GPUPointCloudNeighborhoods({
        positions: positions.view,
        radius: values.radius,
        capacity: values.capacity,
        indices: builder.allocate('indices', 'uint32', POINT_COUNT * values.capacity).view,
        squaredDistances: builder.allocate(
          'squared-distances',
          'float32',
          POINT_COUNT * values.capacity
        ).view,
        counts: counts.view
      })
    );
    return {
      outputs: {
        map: builder.upload('map', 'float32x2', inputs.mapPositions),
        profile: builder.upload('profile', 'float32x2', inputs.profilePositions),
        keptMap: builder.allocate('kept-map', 'float32x2', HIGHLIGHT_LIMIT),
        keptProfile: builder.allocate('kept-profile', 'float32x2', HIGHLIGHT_LIMIT),
        truncatedMap: builder.allocate('truncated-map', 'float32x2', HIGHLIGHT_LIMIT),
        truncatedProfile: builder.allocate('truncated-profile', 'float32x2', HIGHLIGHT_LIMIT),
        counts: addIntegerToFloat(builder, 'neighbor-counts', counts)
      },
      update: current => positions.write(getJitteredPositions(inputs.cloud, current.noise))
    };
  },
  rebuildOn: ['radius', 'capacity'],
  // The pointer is a point: the neighbors within the radius are bright, and those beyond the
  // table's capacity (found but not kept) are red.
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
    resolve(pointer, {inputs, outputs, values}) {
      const {cloud, layout, probe} = inputs;
      const [x, y] = fromMapPoint(layout, pointer.x, Math.max(pointer.y, layout.offsetY));
      const row = chooseSticky(probe.index, x, y, inputs.focus.row);
      inputs.focus.row = row;
      if (row < 0) return {};
      const found = probe.findNeighbors(row, values.radius);
      const kept = found.slice(0, values.capacity);
      const truncated = found.slice(values.capacity, values.capacity + HIGHLIGHT_LIMIT);
      writeRowPositions(outputs.keptMap, inputs.mapPositions, kept);
      writeRowPositions(outputs.keptProfile, inputs.profilePositions, kept);
      writeRowPositions(outputs.truncatedMap, inputs.mapPositions, truncated);
      writeRowPositions(outputs.truncatedProfile, inputs.profilePositions, truncated);
      const center = toMapPoint(layout, cloud.footprint[row * 2], cloud.footprint[row * 2 + 1]);
      const profileCenter = getProfilePoint(
        layout,
        cloud.positions[row * 3],
        cloud.positions[row * 3 + 2]
      );
      const halfWidth = values.radius * layout.scale;
      const halfHeight = (values.radius / layout.heightMaximum) * layout.profileHeight;
      return {
        focus: [kept.length, truncated.length],
        glyphs: [
          {shape: 'ring', at: center, radius: halfWidth},
          {
            shape: 'box',
            minimum: [profileCenter[0] - halfWidth, profileCenter[1] - halfHeight],
            maximum: [profileCenter[0] + halfWidth, profileCenter[1] + halfHeight]
          }
        ],
        readout: `radius ${values.radius} · ${found.length} found, ${kept.length} kept (capacity ${values.capacity})`
      };
    }
  },
  draw: ({outputs, values, interaction}) => {
    const focus = Array.isArray(interaction.focus) ? interaction.focus : [1, 1];
    const kept = Math.max(1, focus[0]);
    const truncated = Math.max(1, focus[1]);
    const look = {values: outputs.counts, range: [0, values.capacity * 0.6]} as const;
    return [
      {layer: 'dots', positions: outputs.map, ...look, radius: 3.2},
      {layer: 'dots', positions: outputs.profile, ...look, radius: 2.4},
      {
        layer: 'dots',
        positions: outputs.truncatedMap,
        count: truncated,
        color: [1, 0.3, 0.3, 1],
        radius: 3.8
      },
      {
        layer: 'dots',
        positions: outputs.truncatedProfile,
        count: truncated,
        color: [1, 0.3, 0.3, 1],
        radius: 3
      },
      {
        layer: 'dots',
        positions: outputs.keptMap,
        count: kept,
        color: [1, 1, 1, 1],
        radius: 3.8
      },
      {
        layer: 'dots',
        positions: outputs.keptProfile,
        count: kept,
        color: [1, 1, 1, 1],
        radius: 3
      }
    ];
  }
});
