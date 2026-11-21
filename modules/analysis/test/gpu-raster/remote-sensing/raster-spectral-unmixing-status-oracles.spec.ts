// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRemoteSensingHarness} from './remote-sensing-test-utils';
import {GPURasterSpectralUnmixing} from '../../../src/raster-analysis/remote-sensing/gpu-raster-spectral-unmixing';

for (const [constraint, maximumIterations] of [
  ['unconstrained', 1],
  ['nonnegative', 1],
  ['nonnegative', 2]
] as const) {
  it(`derives unmixing status and exact iteration words with an unresolved coordinate: ${constraint}, budget ${maximumIterations}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRemoteSensingHarness(device);
    const status = harness.output('status', 'uint32', 2);
    const iterationCounts = harness.output('iterations', 'uint32', 2);
    for (const view of [status, iterationCounts]) {
      (view.buffer.defaultBuffer as Buffer).write(new Uint32Array(2).fill(0xdeadbeef));
    }
    const result = await harness.run(
      new GPURasterSpectralUnmixing({
        cellCount: 2,
        bandCount: 2,
        endmemberCount: 2,
        values: harness.input([2, 4, 0, 0]),
        endmembers: harness.input([1, 0, 0, 1]),
        constraint,
        maximumIterations,
        tolerance: 1e-6,
        output: {
          abundances: harness.output('abundances', 'float32', 4),
          residual: harness.output('residual', 'float32', 2),
          status,
          iterationCounts
        }
      })
    );
    // Identity design: the NNLS optimum is exactly [2,4], and trace(E'E)=2.
    // From zero, one projected-gradient step yields [1,2], leaving gradient [-1,-2].
    // The next coordinate update [1/2,1] is nonzero: positive-cap exhaustion is forced.
    // A second step gives [1.5,3], still short of [2,4], and must report TWO iterations.
    // The zero spectrum starts at its optimum, so its first mapping is zero and solves.
    // QR is a direct solve, so it performs zero projected-gradient iterations.
    expect(result['status']).toEqual(constraint === 'nonnegative' ? [4, 0] : [0, 0]);
    expect(result['iterations']).toEqual(
      constraint === 'nonnegative' ? (maximumIterations === 2 ? [2, 1] : [1, 1]) : [0, 0]
    );
    const expected =
      constraint === 'nonnegative'
        ? maximumIterations === 2
          ? [1.5, 3, 0, 0]
          : [1, 2, 0, 0]
        : [2, 4, 0, 0];
    result['abundances'].forEach((value, index) => {
      // Identity QR can round; 1e-6 absolute bounds float32 arithmetic here.
      expect(Number.isFinite(value)).toBe(true);
      expect(Math.abs(value - expected[index])).toBeLessThanOrEqual(1e-6);
    });
    // Independent squared distance from the identity target: 1²+2²=5 after one step,
    // (1/2)²+1²=1.25 after two; QR and the zero-target cell have zero residual.
    const expectedResidual =
      constraint === 'nonnegative' ? (maximumIterations === 2 ? [1.25, 0] : [5, 0]) : [0, 0];
    result['residual'].forEach((value, index) => {
      expect(Number.isFinite(value)).toBe(true);
      expect(Math.abs(value - expectedResidual[index])).toBeLessThanOrEqual(1e-6);
    });
  });
}
