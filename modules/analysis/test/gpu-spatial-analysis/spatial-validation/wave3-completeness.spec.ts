// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUSpatialCrossValidation} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation';
import {GPUGeoConformalPredictionIntervals} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-geo-conformal-prediction-intervals';
import {getGPUConformalPredictionParameterValues} from '../../../src/gpu-spatial-analysis/spatial-validation/conformal-prediction-nodes';
import {GraphRig} from '../areal-interpolation/areal-interpolation-harness';

it.each([
  [1, 2 ** -13, 1],
  [2 ** -63, 2 ** -64, (17 / 16) * 2 ** -63]
])('TWN-41/B4 uses inclusive Euclidean membership (%s,%s) radius %s', async (horizontal, vertical, radius) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new GraphRig(device);
  try {
    const folds = rig.output('uint32', 2),
      training = rig.output('uint32', 4),
      testing = rig.output('uint32', 4),
      counts = rig.output('uint32', 6),
      status = rig.output('uint32', 2);
    rig.run(
      new GPUSpatialCrossValidation({
        positions: rig.input(Float32Array.of(0, 0, horizontal, vertical), 'float32x2', 2),
        blocks: rig.input(Uint32Array.of(1, 0), 'uint32'),
        foldCount: 2,
        bufferDistance: radius,
        output: {
          folds: folds.view,
          training: training.view,
          testing: testing.view,
          counts: counts.view,
          status: status.view
        }
      })
    );
    // Exact d² = 1+2^-26, strictly outside the unit disk, despite f32 dot(d,d)=1.
    expect(await training.readUint32()).toEqual([0, 1, 1, 0]);
    expect(await status.readUint32()).toEqual([0, 0]);
  } finally {
    rig.destroy();
  }
});

it('B2 clears unbounded bits on suppression and restores unbounded endpoints', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new GraphRig(device);
  try {
    const candidateOverflow = rig.output('uint32', 1),
      intervals = rig.output('float32', 2),
      status = rig.output('uint32', 1);
    candidateOverflow.buffer.write(Uint32Array.of(0));
    rig.run(
      new GPUGeoConformalPredictionIntervals({
        calibration: {
          observations: rig.input(Float32Array.of(2, 5), 'float32'),
          predictions: rig.input(Float32Array.of(0, 0), 'float32')
        },
        query: {predictions: rig.input(Float32Array.of(0), 'float32')},
        neighbors: {
          offsets: rig.input(Uint32Array.of(0, 2), 'uint32'),
          indices: rig.input(Uint32Array.of(0, 1), 'uint32'),
          weights: rig.input(Float32Array.of(0.5, 0.25), 'float32'),
          status: {candidateOverflow: candidateOverflow.view}
        },
        parameters: rig.input(
          getGPUConformalPredictionParameterValues({miscoverage: 0.1, selfWeight: 1}),
          'float32'
        ),
        output: {intervals: intervals.view, status: status.view}
      })
    );
    for (const lost of [0, 1, 0]) {
      candidateOverflow.buffer.write(Uint32Array.of(lost));
      rig.resubmit();
      expect(await intervals.readFloat32()).toEqual(lost ? [NaN, NaN] : [-Infinity, Infinity]);
      expect(await status.readUint32()).toEqual([lost ? 32 : 3]);
    }
  } finally {
    rig.destroy();
  }
});

it('CPL-04 suppresses neighbor loss and a zero completeness certificate, then recovers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new GraphRig(device);
  try {
    const overflow = rig.output('uint32', 1),
      candidateOverflow = rig.output('uint32', 1),
      complete = rig.output('uint32', 1);
    overflow.buffer.write(Uint32Array.of(0));
    candidateOverflow.buffer.write(Uint32Array.of(0));
    complete.buffer.write(Uint32Array.of(1));
    const intervals = rig.output('float32', 2),
      status = rig.output('uint32', 1);
    rig.run(
      new GPUGeoConformalPredictionIntervals({
        calibration: {
          observations: rig.input(Float32Array.of(1, 9), 'float32'),
          predictions: rig.input(Float32Array.of(0, 0), 'float32')
        },
        query: {predictions: rig.input(Float32Array.of(0), 'float32')},
        neighbors: {
          offsets: rig.input(Uint32Array.of(0, 1), 'uint32'),
          indices: rig.input(Uint32Array.of(0), 'uint32'),
          weights: rig.input(Float32Array.of(1), 'float32'),
          status: {
            overflow: overflow.view,
            candidateOverflow: candidateOverflow.view,
            complete: complete.view
          }
        },
        parameters: rig.input(
          getGPUConformalPredictionParameterValues({miscoverage: 0.5, selfWeight: 1}),
          'float32'
        ),
        output: {intervals: intervals.view, status: status.view}
      })
    );
    for (const [lost, candidates, certified] of [
      [0, 0, 1],
      [1, 0, 1],
      [0, 1, 1],
      [0, 0, 0],
      [0, 0, 1]
    ]) {
      overflow.buffer.write(Uint32Array.of(lost));
      candidateOverflow.buffer.write(Uint32Array.of(candidates));
      complete.buffer.write(Uint32Array.of(certified));
      rig.resubmit();
      const incomplete = lost || candidates || !certified;
      expect(await status.readUint32()).toEqual([incomplete ? 32 : 0]);
      const result = await intervals.readFloat32();
      if (incomplete) expect(result).toEqual([NaN, NaN]);
      else expect(result).toEqual([-1, 1]);
    }
  } finally {
    rig.destroy();
  }
});
