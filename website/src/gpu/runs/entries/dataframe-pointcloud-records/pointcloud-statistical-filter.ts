// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointCloudStatisticalFilter after GPUPointCloudNeighborhoods: points whose mean neighbor
// distance is more than `standardDeviations` above the cloud's mean are rejected (blue); the rest
// are kept (orange). Isolated strays are rejected first.

import {
  GPUPointCloudNeighborhoods,
  GPUPointCloudStatisticalFilter
} from '@luma-contributors/analysis/pointcloud-analysis';
import {defineOperationRun} from '../../define-run';
import {createTerrainCloud} from './_lib/helpers';
import {parameters} from './pointcloud-statistical-filter.parameters';

const TERRAIN_COUNT = 2400;
const STRAY_CAPACITY = 300;
const POINT_COUNT = TERRAIN_COUNT + STRAY_CAPACITY;

export default defineOperationRun({
  parameters,
  caption: 'Kept points (orange) and rejected outliers (blue).',
  createInputs: ({random, world}) => {
    const cloud = createTerrainCloud(random, world.width, TERRAIN_COUNT);
    const strays = Float32Array.from({length: STRAY_CAPACITY * 3}, (_, index) =>
      index % 3 === 0 ? random() * world.width : index % 3 === 1 ? random() : random() * 0.5
    );
    return {cloud, strays, width: world.width};
  },
  build(builder, inputs, values) {
    const capacity = values.capacity;
    const positions = builder.allocate('positions', 'float32x3', POINT_COUNT);
    const footprint = builder.allocate('footprint', 'float32x2', POINT_COUNT);
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
      outputs: {footprint, mask},
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
        footprint.write(
          Float32Array.from(
            {length: POINT_COUNT * 2},
            (_, index) => all[Math.floor(index / 2) * 3 + (index % 2)]
          )
        );
      }
    };
  },
  rebuildOn: ['standardDeviations', 'capacity'],
  draw: ({outputs}) => [
    {layer: 'dots', positions: outputs.footprint, values: outputs.mask, radius: 3}
  ]
});
