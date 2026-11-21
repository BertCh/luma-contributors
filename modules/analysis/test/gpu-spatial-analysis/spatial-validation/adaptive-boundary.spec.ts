// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUSpatialCrossValidation} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation';
import {GraphRig} from '../areal-interpolation/areal-interpolation-harness';
import reference from '../spatial-regression/spatial-validation-advanced-reference.json';

it.each([2 ** -13, 2 ** -27])('B4 resolves rounded defining-neighbor ties (%s)', async vertical => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new GraphRig(device);
  try {
    const folds = rig.output('uint32', 3),
      training = rig.output('uint32', 9),
      testing = rig.output('uint32', 9),
      counts = rig.output('uint32', 9),
      status = rig.output('uint32', 3);
    rig.run(
      new GPUSpatialCrossValidation({
        positions: rig.input(Float32Array.of(0, 0, 1, vertical, 1, 0), 'float32x2', 3),
        blocks: rig.input(new Uint32Array(3), 'uint32'),
        foldCount: 3,
        bufferDistance: 0,
        design: {
          mode: 'nndm',
          predictionPositions: rig.input(Float32Array.of(0, 0), 'float32x2', 1),
          maximumDistance: 0
        },
        output: {
          folds: folds.view,
          training: training.view,
          testing: testing.view,
          counts: counts.view,
          status: status.view
        }
      })
    );
    // d²(0,2)=1 < d²(0,1)=1+vertical². Other folds exclude their sole nearest row.
    expect(await folds.readUint32()).toEqual([0, 1, 2]);
    expect(await training.readUint32()).toEqual([0, 1, 0, 1, 0, 0, 1, 0, 0]);
    expect(await testing.readUint32()).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    expect(await counts.readUint32()).toEqual([1, 1, 1, 1, 1, 1, 1, 1, 1]);
    expect(await status.readUint32()).toEqual([0, 0, 0]);
  } finally {
    rig.destroy();
  }
});

it.each([
  {positions: [0, 0, 1, 1]},
  {positions: [0, 0, 3, 8]},
  {positions: reference.positions}
])('B1 preserves exact adaptive boundaries and the blockCV oracle %#', async ({positions}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new GraphRig(device),
    rows = positions.length / 2;
  try {
    const folds = rig.output('uint32', rows),
      training = rig.output('uint32', rows * rows),
      testing = rig.output('uint32', rows * rows),
      counts = rig.output('uint32', rows * 3),
      status = rig.output('uint32', rows);
    rig.run(
      new GPUSpatialCrossValidation({
        positions: rig.input(Float32Array.from(positions), 'float32x2', rows),
        blocks: rig.input(new Uint32Array(rows), 'uint32'),
        foldCount: rows,
        design: {
          mode: 'nndm',
          predictionPositions: rig.input(
            Float32Array.from(rows === 2 ? [0, 0] : reference.predictionPositions),
            'float32x2',
            rows === 2 ? 1 : reference.predictionPositions.length / 2
          ),
          maximumDistance: rows === 2 ? 0 : 4,
          minimumTrainingFraction: 0.2
        },
        output: {
          folds: folds.view,
          training: training.view,
          testing: testing.view,
          counts: counts.view,
          status: status.view
        }
      })
    );
    expect(await training.readUint32()).toEqual(
      rows === 2 ? [0, 0, 0, 0] : reference.nndm.training
    );
  } finally {
    rig.destroy();
  }
});
