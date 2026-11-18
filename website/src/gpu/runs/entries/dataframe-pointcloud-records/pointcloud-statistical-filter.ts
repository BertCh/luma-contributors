// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointCloudStatisticalFilter after GPUPointCloudNeighborhoods: points whose mean neighbor
// distance is more than `standardDeviations` above the cloud's mean are rejected (large red
// dots); the rest are kept (light blue dots). Isolated strays are rejected first. The pointer's
// height picks the threshold level; the operation reruns at the picked level once it settles.

import {
  GPUPointCloudNeighborhoods,
  GPUPointCloudStatisticalFilter
} from '@luma-contributors/analysis/pointcloud-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {clamp} from '../../interaction';
import {createLegend, mergeText, type RunLabel} from '../../text';
import {createCloudLayout, createTerrainCloud, getProfilePoint, toMapPoint} from './_lib/helpers';
import {parameters} from './pointcloud-statistical-filter.parameters';

const TERRAIN_COUNT = 2400;
const STRAY_CAPACITY = 300;
const POINT_COUNT = TERRAIN_COUNT + STRAY_CAPACITY;
// The rejection thresholds the pointer picks among, in standard deviations above the mean distance.
const RADIUS = 0.06;
const LEVELS = [0.75, 1.5, 2.25, 3, 4] as const;
const REJECTED_COLOR = [1, 0.3, 0.3] as const;
const KEPT_COLOR = [0.55, 0.68, 0.85] as const;

function createInputs({
  random,
  world
}: {
  random: () => number;
  world: {width: number; height: number};
}) {
  const cloud = createTerrainCloud(random, world.width, TERRAIN_COUNT);
  const strays = Float32Array.from({length: STRAY_CAPACITY * 3}, (_, index) =>
    index % 3 === 0 ? random() * world.width : index % 3 === 1 ? random() : random() * 0.5
  );
  // The side profile reaches the highest stray.
  const layout = createCloudLayout(world.width, 0.5 * 1.05);
  return {
    cloud,
    strays,
    width: world.width,
    layout
  };
}

type Inputs = ReturnType<typeof createInputs>;
type Outputs = {
  map: RunBuffer;
  profile: RunBuffer;
  mask: RunBuffer;
  meanDistances: RunBuffer;
  counts: RunBuffer;
};

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'Kept points (blue) and rejected outliers (red); the strip below is the side profile, where floating strays stand out above the terrain.',
  buildCache: 5,
  createInputs,
  build(builder, inputs, values) {
    const capacity = values.capacity;
    const positions = builder.allocate('positions', 'float32x3', POINT_COUNT);
    const map = builder.allocate('map', 'float32x2', POINT_COUNT);
    const profile = builder.allocate('profile', 'float32x2', POINT_COUNT);
    const squaredDistances = builder.allocate(
      'squared-distances',
      'float32',
      POINT_COUNT * capacity
    );
    const counts = builder.allocate('counts', 'uint32', POINT_COUNT);
    builder.graph.add(
      new GPUPointCloudNeighborhoods({
        positions: positions.view,
        radius: RADIUS,
        capacity,
        output: {
          indices: builder.allocate('indices', 'uint32', POINT_COUNT * capacity).view,
          squaredDistances: squaredDistances.view,
          counts: counts.view
        }
      })
    );
    const mask = builder.allocate('mask', 'uint32', POINT_COUNT);
    const meanDistances = builder.allocate('mean-distances', 'float32', POINT_COUNT);
    builder.graph.add(
      new GPUPointCloudStatisticalFilter({
        squaredDistances: squaredDistances.view,
        counts: counts.view,
        capacity,
        standardDeviations: values.standardDeviations,
        output: {meanDistances: meanDistances.view, validity: mask.view}
      })
    );
    return {
      outputs: {
        map,
        profile,
        mask,
        meanDistances,
        counts
      },
      update: current => {
        // Stray points beyond `strays` are moved onto the terrain so they stop being isolated.
        const all = new Float32Array(POINT_COUNT * 3);
        all.set(inputs.cloud.positions, 0);
        for (let stray = 0; stray < STRAY_CAPACITY; stray++) {
          const base = TERRAIN_COUNT * 3 + stray * 3;
          const source = stray < current.strays ? inputs.strays : inputs.cloud.positions;
          all.set(source.subarray(stray * 3, stray * 3 + 3), base);
        }
        positions.write(all);
        map.write(
          Float32Array.from({length: POINT_COUNT * 2}, (_, index) => {
            const row = index >> 1;
            return toMapPoint(inputs.layout, all[row * 3], all[row * 3 + 1])[index & 1];
          })
        );
        profile.write(
          Float32Array.from({length: POINT_COUNT * 2}, (_, index) => {
            const row = index >> 1;
            return getProfilePoint(inputs.layout, all[row * 3], all[row * 3 + 2])[index & 1];
          })
        );
      }
    };
  },
  rebuildOn: ['standardDeviations', 'capacity'],
  // The pointer's height picks the rejection threshold (five levels, applied once it rests): up is
  // more tolerant. Rejected points show on the map and, above the terrain, on the side profile.
  interaction: {
    kind: 'level',
    hint: 'raise or lower the threshold',
    rest: [0.5, 0.95 - 0.9 * 0.3],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const fraction = clamp((0.95 - pointer.unitY) / 0.9, 0, 0.9999);
      const level = Math.floor(fraction * LEVELS.length);
      const mask = peek(outputs.mask);
      let rejected = 0;
      for (let row = 0; mask && row < mask.length; row++) if (!mask[row]) rejected++;
      const glyphs = LEVELS.map((_, index) => {
        const y = 1 - (0.95 - 0.9 * ((index + 0.5) / LEVELS.length));
        return {
          shape: 'line' as const,
          from: [0.01, y] as const,
          to: [index === level ? 0.1 : 0.04, y] as const
        };
      });
      return {
        settle: {standardDeviations: LEVELS[level]},
        glyphs,
        readout: `standardDeviations ${LEVELS[level]} · ${
          values.standardDeviations === LEVELS[level] ? `rejected ${rejected}` : 'release to apply'
        } of ${POINT_COUNT}`
      };
    }
  },
  draw: ({inputs, outputs, theme}) => {
    const {layout} = inputs;
    // Rejected points first at full size (kept rows draw in the background color, so they hide
    // nothing), then every point small: rejected ones stay red, kept ones are grey.
    const rejectedLook = {
      ramp: 'categorical',
      categories: [REJECTED_COLOR, theme.background]
    } as const;
    const allLook = {ramp: 'categorical', categories: [REJECTED_COLOR, KEPT_COLOR]} as const;
    const labels: RunLabel[] = [
      {
        text: 'top view',
        at: [layout.offsetX, 0.995],
        anchor: 'top-left',
        offset: [2, 0],
        role: 'title'
      },
      {
        text: 'side profile (height)',
        at: [layout.offsetX, layout.profileBottom + layout.profileHeight + 0.004],
        anchor: 'bottom-left',
        offset: [2, 0],
        role: 'title'
      }
    ];
    return [
      {layer: 'dots', positions: outputs.map, values: outputs.mask, radius: 5, ...rejectedLook},
      {
        layer: 'dots',
        positions: outputs.profile,
        values: outputs.mask,
        radius: 4.4,
        ...rejectedLook
      },
      {layer: 'dots', positions: outputs.map, values: outputs.mask, radius: 2, ...allLook},
      {layer: 'dots', positions: outputs.profile, values: outputs.mask, radius: 1.8, ...allLook},
      {
        layer: 'text',
        ...mergeText(
          {labels},
          createLegend({
            corner: 'top-right',
            entries: [
              {label: 'kept', swatch: KEPT_COLOR},
              {label: 'rejected by the filter', swatch: REJECTED_COLOR}
            ]
          })
        )
      }
    ];
  }
});
