// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUPointCloudICP} from '../../src/pointcloud-analysis/index';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {makeSpatialFixture, readFloats, readIntegers} from '../spatial-test-utils';
import {expectExtremeICPReadback} from './pointcloud-icp-extreme-assertions';

it('CORE-106: point-to-point ICP reaches the extreme Horn rotation through finite moments', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    // Closed form: rotate this noncollinear triangle about z by 0.05 radians, no translation.
    // /usr/bin/python3 + NumPy float32 trace: raw covariance max 2.936326e38, normalized
    // Horn diagonals +/-1.937576e38, coupling 9.695962e36; diagonal subtraction overflows.
    // Nearest self-distance squared <=4.899e35 < (1e18)^2; all weights are exactly one.
    const sourceValues = new Float32Array([1.4e19, 0, 0, -0.7e19, 1.2e19, 0, -0.7e19, -1.2e19, 0]);
    const targetValues = new Float32Array([
      1.398250364552953e19, 6.997083697349591e17, 0, -7.591001854012906e18, 1.1635148939872117e19,
      0, -6.391501791516624e18, -1.2334857309607076e19, 0
    ]);
    const transform = fixture.view('float32', new Float32Array(16));
    const status = fixture.view('uint32', new Uint32Array([99]));
    const history = fixture.view('float32x3', new Float32Array(3));
    const mask = fixture.view('uint32', new Uint32Array(3));
    fixture.graph.add(
      new GPUPointCloudICP({
        source: fixture.view('float32x3', sourceValues).view,
        target: fixture.view('float32x3', targetValues).view,
        maximumDistance: 1e18,
        // The padded cell walk visits only each point's matching target; all six far
        // targets live outside its window, so their squared distances are never evaluated.
        targetGrid: {origin: [-2e19, -2e19, -2e18], cellSize: 1e18, dimensions: [40, 40, 4]},
        maximumIterations: 1,
        translationTolerance: 1e14,
        rotationTolerance: 1e-5,
        output: {
          transform: transform.view,
          status: status.view,
          history: history.view,
          inlierMask: mask.view
        }
      })
    );
    await fixture.run();
    expect(Array.from(await readIntegers(status.buffer))).toEqual([1]); // iteration limit, valid last iterate
    expect(Array.from(await readIntegers(mask.buffer))).toEqual([1, 1, 1]);
    const actual = await readFloats(transform.buffer);
    const residual = await readFloats(history.buffer);
    expectExtremeICPReadback(actual, residual);
    // Literal unit probe transformed by R_z(0.05): [cos(0.05), sin(0.05), 0].
    expect(Math.abs(actual[0] - 0.9987502603949663)).toBeLessThan(2e-5);
    expect(Math.abs(actual[1] - 0.04997916927067833)).toBeLessThan(2e-5);
    expect(Math.abs(actual[2])).toBeLessThan(2e-5);
    for (let point = 0; point < 3; point++) {
      for (let axis = 0; axis < 3; axis++) {
        const moved =
          actual[axis] * sourceValues[point * 3] +
          actual[4 + axis] * sourceValues[point * 3 + 1] +
          actual[12 + axis];
        expect(Number.isFinite(moved)).toBe(true);
        expect(Math.abs(moved - targetValues[point * 3 + axis]) / 1.4e19).toBeLessThan(3e-5);
      }
    }
    expect(residual[0]).toBe(3);
  } finally {
    fixture.destroy();
  }
});

it('S4: indexed ICP compiles a large integer-valued float32 origin and preserves the rigid transform', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    // Three noncollinear points in a yz plane. Identical source/target implies identity exactly.
    const values = new Float32Array([1e19, 0, 0, 1e19, 1, 0, 1e19, 0, 1]);
    const transform = fixture.view('float32', new Float32Array(16));
    const history = fixture.view('float32x3', new Float32Array(3));
    const status = fixture.view('uint32', new Uint32Array([99]));
    const mask = fixture.view('uint32', new Uint32Array(3));
    fixture.graph.add(
      new GPUPointCloudICP({
        source: fixture.view('float32x3', values).view,
        target: fixture.view('float32x3', values).view,
        maximumDistance: 0.25,
        maximumIterations: 1,
        targetGrid: {origin: [1e19, 0, 0], cellSize: 1, dimensions: [1, 2, 2]},
        output: {
          transform: transform.view,
          history: history.view,
          status: status.view,
          inlierMask: mask.view
        }
      })
    );
    await fixture.run();
    const actual = await readFloats(transform.buffer);
    expect(actual).toHaveLength(16);
    expect(actual.every(Number.isFinite)).toBe(true);
    const expected = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    for (const [index, value] of expected.entries())
      expect(Math.abs(actual[index] - value)).toBeLessThan(1e-5);
    expect(await readIntegers(status.buffer)).toEqual([0]);
    expect(await readIntegers(mask.buffer)).toEqual([1, 1, 1]);
    expect(await readFloats(history.buffer)).toEqual([3, 0, 0]);
  } finally {
    fixture.destroy();
  }
});
