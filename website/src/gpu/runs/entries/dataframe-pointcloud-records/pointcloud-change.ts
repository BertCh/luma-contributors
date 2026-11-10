// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointCloudChange (M3C2) between two epochs of a terrain, along the vertical. Core points are
// colored by the signed change the GPU measured: blue is lowering (a slump in the center), red is
// raising, near zero is stable.

import {GPUPointCloudChange} from '@luma-contributors/analysis/pointcloud-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {chooseSticky, createPointIndex} from '../../pick';
import {addFloatCopy} from './_lib/convert';
import {createTerrainCloud} from './_lib/helpers';
import {parameters} from './pointcloud-change.parameters';

const POINT_COUNT = 8000;
const CORE_STEP = 8;

type Inputs = ReturnType<typeof createInputs>;
type Outputs = {cores: RunBuffer; change: RunBuffer};

function createInputs({
  random,
  world
}: {
  random: () => number;
  world: {width: number; height: number};
}) {
  const cloud = createTerrainCloud(random, world.width, POINT_COUNT);
  const coreFootprint = new Float32Array((POINT_COUNT / CORE_STEP) * 2);
  for (let index = 0; index < POINT_COUNT / CORE_STEP; index++) {
    coreFootprint.set(cloud.positions.subarray(index * 4 * 6, index * 4 * 6 + 2), index * 2);
  }
  return {
    ...cloud,
    // The slump's center is the pointer's: the middle of the scene until it moves.
    center: [world.width / 2, 0.5] as [number, number],
    coreFootprint,
    coreIndex: createPointIndex(coreFootprint),
    focus: {row: -1}
  };
}

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption: 'Vertical change after a simulated slump in the King County lidar sample.',
  createInputs,
  build(builder, inputs, values) {
    const cloud = inputs;
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
          const x = baseline[index - 2] - inputs.center[0];
          const y = baseline[index - 1] - inputs.center[1];
          return value - current.change * Math.exp(-(x * x + y * y) / 0.05);
        });
        comparison.write(lowered);
      }
    };
  },
  rebuildOn: ['radius', 'halfDepth'],
  // The pointer is the center of the simulated slump in the second epoch (an input of the
  // comparison); the M3C2 cylinder radius is drawn around the core point nearest to it.
  interaction: {
    kind: 'place',
    hint: 'move the slump',
    rest: ({inputs}) => [inputs.center[0] / inputs.width, 1 - inputs.center[1]],
    resolve(pointer, {inputs, outputs, values, peek}) {
      inputs.center = [pointer.x, pointer.y];
      const row = chooseSticky(inputs.coreIndex, pointer.x, pointer.y, inputs.focus.row);
      inputs.focus.row = row;
      const changes = peek(outputs.change);
      let strongest = 0;
      let changed = 0;
      for (let index = 0; changes && index < changes.length; index++) {
        if (Math.abs(changes[index]) > 0.01) changed++;
        if (Math.abs(changes[index]) > Math.abs(strongest)) strongest = changes[index];
      }
      return {
        rerun: true,
        glyphs: [
          {
            shape: 'ring',
            at: [pointer.x, pointer.y],
            radius: Math.sqrt(0.05 * Math.LN2),
            dashed: true
          },
          ...(row >= 0
            ? [
                {
                  shape: 'ring' as const,
                  at: [inputs.coreFootprint[row * 2], inputs.coreFootprint[row * 2 + 1]] as const,
                  radius: values.radius
                }
              ]
            : [])
        ],
        readout: `slump at (${pointer.x.toFixed(1)}, ${pointer.y.toFixed(1)}) · max ${strongest.toFixed(2)} · ${changed} cores changed`
      };
    }
  },
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
