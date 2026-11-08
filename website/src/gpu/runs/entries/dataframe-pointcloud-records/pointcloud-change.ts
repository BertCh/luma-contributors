// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointCloudChange (M3C2) between two epochs of a terrain, along the vertical. Core points are
// colored by the signed change the GPU measured: blue is lowering (a slump in the center), red is
// raising, near zero is stable.

import {GPUPointCloudChange} from '@luma-contributors/analysis/pointcloud-analysis';
import {defineOperationRun} from '../../define-run';
import {addFloatCopy} from './_lib/convert';
import {createTerrainCloud} from './_lib/helpers';
import {parameters} from './pointcloud-change.parameters';

const POINT_COUNT = 8000;
const CORE_STEP = 8;

export default defineOperationRun({
  parameters,
  caption: 'Signed vertical change per core point between two epochs.',
  createInputs: ({random, world}) => createTerrainCloud(random, world.width, POINT_COUNT),
  build(builder, cloud, values) {
    // Epoch A owns even rows, epoch B odd rows: two independent samples of one surface.
    const reference = new Float32Array((POINT_COUNT / 2) * 3);
    const baseline = new Float32Array((POINT_COUNT / 2) * 3);
    const cores = new Float32Array((POINT_COUNT / CORE_STEP) * 3);
    const normals = new Float32Array((POINT_COUNT / CORE_STEP) * 3);
    for (let index = 0; index < POINT_COUNT / 2; index++) {
      reference.set(cloud.positions.subarray(index * 6, index * 6 + 3), index * 3);
      baseline.set(cloud.positions.subarray(index * 6 + 3, index * 6 + 6), index * 3);
    }
    for (let index = 0; index < POINT_COUNT / CORE_STEP; index++) {
      cores.set(reference.subarray(index * 4 * 3, index * 4 * 3 + 3), index * 3);
      normals.set([0, 0, 1], index * 3);
    }
    const comparison = builder.allocate('comparison', 'float32x3', POINT_COUNT / 2);
    const changes = builder.allocate('changes', 'float32x3', POINT_COUNT / CORE_STEP);
    builder.graph.add(
      new GPUPointCloudChange({
        reference: builder.upload('reference', 'float32x3', reference).view,
        comparison: comparison.view,
        cores: builder.upload('cores', 'float32x3', cores).view,
        normals: builder.upload('normals', 'float32x3', normals).view,
        alignment: 'already-aligned',
        radius: values.radius,
        halfDepth: values.halfDepth,
        changes: changes.view,
        roughness: builder.allocate('roughness', 'float32x3', POINT_COUNT / CORE_STEP).view,
        counts: builder.allocate('counts', 'uint32', (POINT_COUNT / CORE_STEP) * 2).view,
        status: builder.allocate('status', 'uint32', POINT_COUNT / CORE_STEP).view
      })
    );
    const coreFootprint = new Float32Array((POINT_COUNT / CORE_STEP) * 2);
    for (let index = 0; index < POINT_COUNT / CORE_STEP; index++) {
      coreFootprint.set(cores.subarray(index * 3, index * 3 + 2), index * 2);
    }
    return {
      outputs: {
        cores: builder.upload('core-footprint', 'float32x2', coreFootprint),
        change: addFloatCopy(builder, 'change', changes, POINT_COUNT / CORE_STEP, {
          componentStart: 0
        })
      },
      update: current => {
        // A slump in the middle of the scene lowers epoch B by `change`.
        const lowered = Float32Array.from(baseline, (value, index) => {
          if (index % 3 !== 2) return value;
          const x = baseline[index - 2] - cloud.width / 2;
          const y = baseline[index - 1] - 0.5;
          return value - current.change * Math.exp(-(x * x + y * y) / 0.05);
        });
        comparison.write(lowered);
      }
    };
  },
  rebuildOn: ['radius', 'halfDepth'],
  draw: ({outputs, theme}) => [
    // Every core point as a muted-but-visible dot; the changed ones stand out in saturated colors.
    {
      layer: 'dots',
      context: true,
      positions: outputs.cores,
      radius: 5.5,
      color: [...theme.muted, 0.7]
    },
    {
      layer: 'dots',
      positions: outputs.cores,
      values: outputs.change,
      ramp: 'diverging',
      range: [-0.05, 0.05],
      radius: 8
    }
  ]
});
