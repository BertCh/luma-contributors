// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointCloudStatisticalFilter after GPUPointCloudNeighborhoods: points whose mean neighbor
// distance is more than `standardDeviations` above the cloud's mean are rejected (blue); the rest
// are kept (orange). Isolated strays are rejected first.

import {
  GPUPointCloudNeighborhoods,
  GPUPointCloudStatisticalFilter
} from '@luma-contributors/analysis/pointcloud-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {clamp} from '../../interaction';
import {createPointIndex, type PointIndex} from '../../pick';
import {
  createCloudLayout,
  createTerrainCloud,
  getProfilePoint,
  toMapPoint,
  writeRowPositions
} from './_lib/helpers';
import {parameters} from './pointcloud-statistical-filter.parameters';

const TERRAIN_COUNT = 2400;
const STRAY_CAPACITY = 300;
const POINT_COUNT = TERRAIN_COUNT + STRAY_CAPACITY;
// The rejection thresholds the pointer picks among, in standard deviations above the mean distance.
const RADIUS = 0.06;
const LEVELS = [0.75, 1.5, 2.25, 3, 4] as const;

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
    layout,
    preview: {
      key: '',
      positions: new Float32Array(0) as Float32Array,
      meanDistances: new Float32Array(0) as Float32Array,
      index: null as PointIndex | null
    }
  };
}

/** Mean distance to the nearest neighbors within the radius: a CPU estimate of what the GPU filter sees. */
function getPreview(inputs: ReturnType<typeof createInputs>, strayCount: number, capacity: number) {
  const key = `${strayCount}:${capacity}`;
  if (inputs.preview.key === key) return inputs.preview;
  const positions = new Float32Array(POINT_COUNT * 3);
  positions.set(inputs.cloud.positions, 0);
  for (let stray = 0; stray < STRAY_CAPACITY; stray++) {
    const source = stray < strayCount ? inputs.strays : inputs.cloud.positions;
    positions.set(source.subarray(stray * 3, stray * 3 + 3), (TERRAIN_COUNT + stray) * 3);
  }
  const footprint = Float32Array.from(
    {length: POINT_COUNT * 2},
    (_, index) => positions[(index >> 1) * 3 + (index & 1)]
  );
  const index = createPointIndex(footprint);
  const meanDistances = new Float32Array(POINT_COUNT);
  for (let row = 0; row < POINT_COUNT; row++) {
    const distances: number[] = [];
    for (const other of index.within(footprint[row * 2], footprint[row * 2 + 1], RADIUS)) {
      if (other === row) continue;
      const distance = Math.hypot(
        positions[other * 3] - positions[row * 3],
        positions[other * 3 + 1] - positions[row * 3 + 1],
        positions[other * 3 + 2] - positions[row * 3 + 2]
      );
      if (distance <= RADIUS) distances.push(distance);
    }
    distances.sort((first, second) => first - second);
    const kept = distances.slice(0, capacity);
    meanDistances[row] = kept.length
      ? kept.reduce((sum, value) => sum + value, 0) / kept.length
      : Number.NaN;
  }
  inputs.preview = {key, positions, meanDistances, index};
  return inputs.preview;
}
type Inputs = ReturnType<typeof createInputs>;
type Outputs = {
  map: RunBuffer;
  profile: RunBuffer;
  mask: RunBuffer;
  candidateMap: RunBuffer;
  candidateProfile: RunBuffer;
};

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'Kept points (orange) and rejected outliers (blue); the strip below is the side profile, where floating strays stand out above the terrain.',
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
        radius: 0.06,
        capacity,
        indices: builder.allocate('indices', 'uint32', POINT_COUNT * capacity).view,
        squaredDistances: squaredDistances.view,
        counts: counts.view
      })
    );
    const mask = builder.allocate('mask', 'uint32', POINT_COUNT);
    builder.graph.add(
      new GPUPointCloudStatisticalFilter({
        squaredDistances: squaredDistances.view,
        counts: counts.view,
        capacity,
        standardDeviations: values.standardDeviations,
        meanDistances: builder.allocate('mean-distances', 'float32', POINT_COUNT).view,
        mask: mask.view
      })
    );
    return {
      outputs: {
        map,
        profile,
        mask,
        candidateMap: builder.allocate('candidate-map', 'float32x2', POINT_COUNT),
        candidateProfile: builder.allocate('candidate-profile', 'float32x2', POINT_COUNT)
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
      // Immediate feedback while the graph waits to rebuild: the points the candidate level would
      // reject (a CPU estimate from mean neighbor distances), in red.
      const preview = getPreview(inputs, values.strays, values.capacity);
      let sum = 0;
      let sumOfSquares = 0;
      let valid = 0;
      for (const distance of preview.meanDistances) {
        if (!Number.isFinite(distance)) continue;
        sum += distance;
        sumOfSquares += distance * distance;
        valid++;
      }
      const mean = sum / Math.max(valid, 1);
      const deviation = Math.sqrt(Math.max(sumOfSquares / Math.max(valid, 1) - mean * mean, 0));
      const threshold = mean + LEVELS[level] * deviation;
      const candidates: number[] = [];
      preview.meanDistances.forEach((distance, row) => {
        if (!Number.isFinite(distance) || distance > threshold) candidates.push(row);
      });
      const mapped = new Float32Array(POINT_COUNT * 2);
      const profiled = new Float32Array(POINT_COUNT * 2);
      candidates.forEach((row, index) => {
        mapped.set(
          toMapPoint(inputs.layout, preview.positions[row * 3], preview.positions[row * 3 + 1]),
          index * 2
        );
        profiled.set(
          getProfilePoint(
            inputs.layout,
            preview.positions[row * 3],
            preview.positions[row * 3 + 2]
          ),
          index * 2
        );
      });
      const order = candidates.map((_, index) => index);
      const candidateCount = writeRowPositions(outputs.candidateMap, mapped, order);
      writeRowPositions(outputs.candidateProfile, profiled, order);
      return {
        focus: candidateCount,
        settle: {standardDeviations: LEVELS[level]},
        glyphs,
        readout: `standardDeviations ${LEVELS[level]} · ${
          values.standardDeviations === LEVELS[level]
            ? `rejected ${rejected}`
            : `about ${candidateCount} rejected`
        } of ${POINT_COUNT}`
      };
    }
  },
  draw: ({outputs, interaction}) => [
    {layer: 'dots', positions: outputs.map, values: outputs.mask, radius: 2.8},
    {layer: 'dots', positions: outputs.profile, values: outputs.mask, radius: 2.4},
    {
      layer: 'dots',
      positions: outputs.candidateMap,
      count: Math.max(1, typeof interaction.focus === 'number' ? interaction.focus : 0),
      color: [1, 0.3, 0.3, 0.9],
      radius: 3.6
    },
    {
      layer: 'dots',
      positions: outputs.candidateProfile,
      count: Math.max(1, typeof interaction.focus === 'number' ? interaction.focus : 0),
      color: [1, 0.3, 0.3, 0.9],
      radius: 3
    }
  ]
});
