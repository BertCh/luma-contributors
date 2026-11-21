// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  GPUSegregation,
  getGPUSegregationLayout
} from '../../../src/gpu-spatial-analysis/segregation';
import {GraphRig} from '../areal-interpolation/areal-interpolation-harness';

it.each([
  'zero',
  'exclude-unit'
] as const)('UST-10 distinguishes missing and observed-zero counts under %s', async missingCountPolicy => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new GraphRig(device);
  const layout = getGPUSegregationLayout(2);
  try {
    const counts = rig.output('float32', 4),
      indices = rig.output('float32', layout.stride),
      invalidUnitCount = rig.output('uint32', 1);
    counts.buffer.write(Float32Array.of(8, NaN, 2, 8));
    rig.run(
      new GPUSegregation({
        groupCounts: counts.view,
        groupCount: 2,
        missingCountPolicy,
        output: {indices: indices.view, invalidUnitCount: invalidUnitCount.view}
      })
    );
    for (const [missing, invalid, expected] of [
      [NaN, 1, missingCountPolicy === 'zero' ? 0.8 : 0],
      [0, 0, 0.8],
      [2, 0, 0.6]
    ]) {
      counts.buffer.write(Float32Array.of(8, missing, 2, 8));
      rig.resubmit();
      const result = await indices.readFloat32();
      expect(result).toHaveLength(layout.stride);
      // Independent Python evaluation of H = sum(t/T * KL(p_i||P))/E, the textbook
      // binary D, xPx=sum(x_i²/t_i)/X and A=1-(sum sqrt(x_i*y_i))²/(X*Y).
      const allIndices =
        missing === 2
          ? [
              0.27807190511263763, 0.6, 0.6931471805599453, 0.6, 0.6, 0.68, 0.68, 0.36, 0.36, 0.68,
              0.32, 0.32, 0.68
            ]
          : invalid && missingCountPolicy === 'exclude-unit'
            ? [0, 0, 0.5004024235381879, 0, 0, 0.2, 0.8, 0, 0, 0.2, 0.8, 0.2, 0.8]
            : [
                0.5953174735228888, 0.8, 0.6869615765973234, 0.8, 0.8, 0.84, 0.8, 0.8, 0.8, 0.84,
                0.16, 0.2, 0.8
              ];
      for (let column = 0; column < allIndices.length; column++) {
        expect(Number.isFinite(result[column])).toBe(true);
        expect(Math.abs(result[column] - allIndices[column])).toBeLessThan(2e-6);
      }
      // Hand calculation: D = 1/2 sum |x/X - y/Y| over retained units.
      expect(Number.isFinite(result[layout.dissimilarity])).toBe(true);
      expect(Math.abs(result[layout.dissimilarity] - expected)).toBeLessThan(1e-6);
      expect(await invalidUnitCount.readUint32()).toEqual([invalid]);
    }
  } finally {
    rig.destroy();
  }
});

it('S4 excludes a weighted target from smoothed populations and all local outputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new GraphRig(device),
    layout = getGPUSegregationLayout(2);
  try {
    const indices = rig.output('float32', layout.stride),
      localEnvironment = rig.output('float32', 4),
      localEntropy = rig.output('float32', 2),
      localDissimilarity = rig.output('float32', 4),
      localTheil = rig.output('float32', 2),
      invalidUnitCount = rig.output('uint32', 1);
    rig.run(
      new GPUSegregation({
        groupCount: 2,
        groupCounts: rig.input(Float32Array.of(8, NaN, 2, 8), 'float32'),
        missingCountPolicy: 'exclude-unit',
        spatialForm: 'smoothed-population',
        scales: [
          {
            offsets: rig.input(Uint32Array.of(0, 1, 2), 'uint32'),
            neighbors: rig.input(Uint32Array.of(1, 0), 'uint32'),
            weights: rig.input(Float32Array.of(1, 1), 'float32')
          }
        ],
        output: {
          indices: indices.view,
          localEnvironment: localEnvironment.view,
          localEntropy: localEntropy.view,
          localDissimilarity: localDissimilarity.view,
          localTheil: localTheil.view,
          invalidUnitCount: invalidUnitCount.view
        }
      })
    );
    const expected = [0, 0, 0.5004024235381879, 0, 0, 0.2, 0.8, 0, 0, 0.2, 0.8, 0.2, 0.8],
      actual = await indices.readFloat32();
    expect(actual).toHaveLength(expected.length);
    for (let column = 0; column < expected.length; column++) {
      expect(Number.isFinite(actual[column])).toBe(true);
      expect(Math.abs(actual[column] - expected[column])).toBeLessThan(2e-6);
    }
    const environment = await localEnvironment.readFloat32(),
      entropy = await localEntropy.readFloat32();
    expect(environment.slice(0, 2)).toEqual([NaN, NaN]);
    expect(entropy[0]).toBeNaN();
    expect(environment).toHaveLength(4);
    expect(entropy).toHaveLength(2);
    expect(Number.isFinite(environment[2])).toBe(true);
    expect(Math.abs(environment[2] - 0.2)).toBeLessThan(1e-6);
    expect(Number.isFinite(environment[3])).toBe(true);
    expect(Math.abs(environment[3] - 0.8)).toBeLessThan(1e-6);
    expect(Number.isFinite(entropy[1])).toBe(true);
    expect(Math.abs(entropy[1] - 0.5004024235381879)).toBeLessThan(1e-6);
    expect(await localDissimilarity.readFloat32()).toEqual([NaN, NaN, 0, 0]);
    const theil = await localTheil.readFloat32();
    expect(theil).toHaveLength(2);
    expect(theil[0]).toBeNaN();
    expect(Number.isFinite(theil[1])).toBe(true);
    expect(Math.abs(theil[1])).toBeLessThan(1e-6);
    expect(await invalidUnitCount.readUint32()).toEqual([1]);
  } finally {
    rig.destroy();
  }
});
