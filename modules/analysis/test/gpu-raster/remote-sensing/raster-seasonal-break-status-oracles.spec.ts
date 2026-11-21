// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRemoteSensingHarness} from './remote-sensing-test-utils';
import {GPURasterSeasonalBreak} from '../../../src/raster-analysis/remote-sensing/gpu-raster-seasonal-break';

for (const scenario of ['joint', 'limited', 'missing'] as const) {
  it(`derives the entire seasonal status from rank, support and unresolved updates: ${scenario}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRemoteSensingHarness(device);
    const status = harness.output('status', 'uint32', 1);
    (status.buffer.defaultBuffer as Buffer).write(Uint32Array.of(0xdeadbeef));
    const result = await harness.run(
      new GPURasterSeasonalBreak({
        cellCount: 1,
        sliceCount: 12,
        period: 4,
        minimumSegmentLength: 3,
        minimumBICGain: 1e6,
        fitMethod: scenario === 'limited' ? 'iterative' : 'joint',
        maximumIterations: 1,
        tolerance: 1e-6,
        values: harness.input(Array.from({length: 12}, (_, slice) => 1 + slice / 12)),
        mask: harness.mask(new Array(12).fill(scenario === 'missing' ? 0 : 1)),
        output: {
          coefficients: harness.output('coefficients', 'float32', 5),
          components: harness.output('components', 'float32', 36),
          breakIndex: harness.output('breakIndex', 'uint32', 1),
          improvement: harness.output('improvement', 'float32', 1),
          status,
          ...(scenario === 'limited'
            ? {diagnostics: harness.output('diagnostics', 'float32', 6)}
            : {})
        }
      })
    );
    // Joint design [1,j/12,cos(pi*j/2),sin(pi*j/2)] has rank four and exactly fits 1+j/12.
    // No observed samples gives code 1. A single cyclic sweep gives coefficients
    // [35/24,13/46,-11/184,-11/184]; residual mean = -143/1104.
    // Thus the NEXT intercept update is -0.1295289855, far above tolerance:
    // the limited case has an independently named unresolved update, not just a positive cap.
    expect(result['status']).toEqual([scenario === 'missing' ? 1 : scenario === 'limited' ? 8 : 0]);
    if (scenario === 'limited') {
      const expected = [
        1.4583333333333333, 0.2826086956521739, -0.059782608695652176, -0.059782608695652176, 0
      ];
      result['coefficients'].forEach((value, index) => {
        // Float32 inputs contain thirds; WGSL trig also rounds. Absolute 3e-6 is ample.
        expect(Number.isFinite(value)).toBe(true);
        expect(Math.abs(value - expected[index])).toBeLessThanOrEqual(3e-6);
      });
      // One bounded sweep, not converged, with all twelve observations retained.
      expect(result['diagnostics'].slice(0, 3)).toEqual([1, 0, 12]);
      // The rational residual above has SSE=34001/50784; four fitted columns leave
      // eight residual degrees of freedom, hence scale=sqrt((34001/50784)/8).
      // Float32 input/trig rounding is covered by the same absolute tolerance.
      expect(Number.isFinite(result['diagnostics'][3])).toBe(true);
      expect(Math.abs(result['diagnostics'][3] - 0.2892926495480756)).toBeLessThanOrEqual(3e-6);
      // Monitoring is absent, so no alarm and no monitoring score are possible.
      expect(result['diagnostics'].slice(4)).toEqual([-1, 0]);
    }
  });
}
