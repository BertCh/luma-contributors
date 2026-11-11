// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUGeodesicPairsOperation} from '../../../src/gpu-spatial-analysis/geometry-measures/cpu-geodesic-pairs-operation';
import {
  GPUGeodesicPairsOperation,
  type GeodesicPairsOperationProps
} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-pairs-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const metric of ['great-circle', 'ellipsoidal', 'rhumb'] as const) {
  for (const coincident of [false, true]) {
    it(`compares every ${metric} output (${coincident ? (metric === 'rhumb' ? 'zero latitude span' : 'coincident and seam identities') : 'long, short, polar and seam paths'})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const origins = new Float32Array([-73.75, 40.5, 179.998, 12.5, -4, -30, 20, 80, 15, 0]);
      const targets = coincident
        ? origins.slice()
        : new Float32Array([2.25, 48.75, -179.996, 12.501, 100, 60, 40, 81, 15.001, 0]);
      if (coincident && metric === 'rhumb') {
        // rhumbBearingDegrees currently evaluates atan2(0, 0) for coincident pairs;
        // that result is unspecified in WGSL, so use the east-west limiting case here.
        for (let row = 0; row < 5; row++) targets[row * 2] += 0.5;
      } else if (coincident) {
        origins[2] = -180;
        targets[2] = 180;
        origins[3] = 0;
        targets[3] = 0;
      }
      const props: GeodesicPairsOperationProps = {
        origins: 'origins',
        targets: 'targets',
        pairCount: 5,
        maximumIterations: 100,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric},
        output: {
          distances: 'distances',
          initialBearings: 'initial',
          finalBearings: 'final',
          midpoints: 'midpoints',
          ...(metric === 'ellipsoidal' ? {status: {converged: 'converged'}} : {})
        }
      };
      const outputs = [
        'distances',
        'initial',
        'final',
        'midpoints',
        ...(metric === 'ellipsoidal' ? ['converged'] : [])
      ];
      const results = [];
      for (const Operation of [GPUGeodesicPairsOperation, CPUGeodesicPairsOperation])
        results.push(
          await runOperationGraph(device, {
            inputs: {
              origins: {format: 'float32x2', values: origins},
              targets: {format: 'float32x2', values: targets}
            },
            operations: [new Operation(props)],
            outputs
          })
        );
      for (const port of outputs) {
        const expected = results[1][port];
        const actual = results[0][port];
        expect(actual.length).toBe(expected.length);
        for (let index = 0; index < expected.length; index++) {
          if (port === 'converged' || (port === 'distances' && expected[index] === 0))
            expect(actual[index] === expected[index]).toBe(true);
          // Positive distances: 3e-5 relative (no absolute floor that could hide a short path).
          else if (port === 'distances')
            expect(Math.abs(actual[index] - expected[index])).toBeLessThanOrEqual(
              Math.abs(expected[index]) * 3e-5
            );
          // Angles compare modulo a turn, never signed zeros. Slerp midpoints allow 5e-5 degrees.
          else if (port === 'initial' || port === 'final') {
            const difference = actual[index] - expected[index];
            expect(Math.abs(difference - 360 * Math.round(difference / 360))).toBeLessThanOrEqual(
              0.002
            );
          } else expect(Math.abs(actual[index] - expected[index])).toBeLessThanOrEqual(5e-5);
        }
      }
    });
  }
}
