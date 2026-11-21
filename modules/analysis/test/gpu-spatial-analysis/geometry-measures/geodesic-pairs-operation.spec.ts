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
import {requireTestDevice} from '../../../src/testing/require-test-device';

for (const metric of ['great-circle', 'ellipsoidal', 'rhumb'] as const) {
  it(`equivalent extreme longitude has a valid nonzero ${metric} inverse in CPU and GPU operations`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: GeodesicPairsOperationProps = {
      origins: 'origins',
      targets: 'targets',
      pairCount: 5,
      spatialContext: {coordinateSpace: 'longitude-latitude', metric, units: 'meters'},
      output: {
        distances: 'distances',
        status: {
          approximation: 'approximation',
          iterationLimitReached: 'iterationLimit',
          ...(metric === 'ellipsoidal' ? {converged: 'converged'} : {})
        }
      }
    };
    for (const Operation of [CPUGeodesicPairsOperation, GPUGeodesicPairsOperation]) {
      const actual = await runOperationGraph(device, {
        inputs: {
          origins: {
            format: 'float32x2',
            values: Float32Array.of(2 ** 62, 0, -176, 0, 179.999, 0, 0, 0, 360, 0)
          },
          targets: {
            format: 'float32x2',
            values: Float32Array.of(0, 0, 0, 0, -179.999, 0, 1e-6, 0, 1e-6, 0)
          }
        },
        operations: [new Operation(props)],
        outputs: [
          'distances',
          'approximation',
          'iterationLimit',
          ...(metric === 'ellipsoidal' ? ['converged'] : [])
        ]
      });
      expect(actual['distances']).toHaveLength(5);
      // 2^62 mod 360=184=-176. Equatorial distance is R*176*pi/180.
      const expected = metric === 'ellipsoidal' ? 19592230.37961615 : 19570334.12110179;
      for (let row = 0; row < 2; row++) {
        expect(Number.isFinite(actual['distances'][row])).toBe(true);
        expect(Math.abs(actual['distances'][row] - expected)).toBeLessThan(100);
      }
      expect(actual['distances'][0]).toBe(actual['distances'][1]);
      // Preserve the short-seam control using its uploaded float32 longitudes.
      const origin = Math.fround(179.999),
        target = Math.fround(-179.999);
      const seam =
        (((360 - origin + target) * Math.PI) / 180) *
        (metric === 'ellipsoidal' ? 6378137 : 6371008.8);
      expect(Number.isFinite(actual['distances'][2])).toBe(true);
      expect(Math.abs(actual['distances'][2] - seam)).toBeLessThan(0.01);
      const tiny =
        (((metric === 'ellipsoidal' ? 6378137 : 6371008.8) * Math.PI) / 180) * Math.fround(1e-6);
      for (const row of [3, 4]) {
        expect(Number.isFinite(actual['distances'][row])).toBe(true);
        expect(Math.abs(actual['distances'][row] - tiny)).toBeLessThan(1e-6);
      }
      expect(Array.from(actual['approximation'])).toEqual([0, 0, 0, 0, 0]);
      expect(Array.from(actual['iterationLimit'])).toEqual([0, 0, 0, 0, 0]);
      if (metric === 'ellipsoidal')
        expect(Array.from(actual['converged'])).toEqual([1, 1, 1, 1, 1]);
    }
  });
}

for (const metric of ['great-circle', 'ellipsoidal', 'rhumb'] as const) {
  for (const coincident of [false, true]) {
    it(`compares every ${metric} output (${coincident ? 'coincident and seam identities' : 'long, short, polar and seam paths'})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const origins = new Float32Array([-73.75, 40.5, 179.998, 12.5, -4, -30, 20, 80, 15, 0]);
      const targets = coincident
        ? origins.slice()
        : new Float32Array([2.25, 48.75, -179.996, 12.501, 100, 60, 40, 81, 15.001, 0]);
      if (coincident) {
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

it('rhumb cardinal and coincident bearings match the hand-derived oracle for signed zeros', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // On the equator/prime meridian, one degree has central angle pi / 180. Bearings are
  // clockwise from north; a coincident pair has the documented zero bearing.
  const targets = new Float32Array([1, 0, 1, -0, -1, 0, -1, -0, -0, 1, -0, -1, 0, 0, -0, -0]);
  const pairCount = targets.length / 2;
  const bearings = [90, 90, -90, -90, 0, 180, 0, 0];
  for (const Operation of [GPUGeodesicPairsOperation, CPUGeodesicPairsOperation]) {
    const result = await runOperationGraph(device, {
      inputs: {
        origins: {format: 'float32x2', values: new Float32Array(targets.length)},
        targets: {format: 'float32x2', values: targets}
      },
      operations: [
        new Operation({
          origins: 'origins',
          targets: 'targets',
          pairCount,
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'rhumb', sphereRadius: 1},
          output: {distances: 'distances', initialBearings: 'initial', finalBearings: 'final'}
        })
      ],
      outputs: ['distances', 'initial', 'final']
    });
    for (let row = 0; row < pairCount; row++) {
      for (const port of ['initial', 'final']) {
        const difference = result[port][row] - bearings[row];
        expect(Math.abs(difference - 360 * Math.round(difference / 360))).toBeLessThan(1e-4);
      }
      expect(Math.abs(result['distances'][row] - (row < 6 ? Math.PI / 180 : 0))).toBeLessThan(1e-8);
    }
  }
});
