// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPURoseStatistic} from '../../../src/gpu-spatial-analysis/geographic-distribution/gpu-rose-statistic';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {expectOracleRows} from '../spatial-interaction/spatial-interaction-audit-fixtures';

// Independent values from rose-audit.oracle.py: exact cardinal complex moments,
// NumPy weighted histogram, unweighted Rayleigh z. Empty group has no moments.
const EXPECTED = {
  'directional/cartesian': {
    summary: [
      4,
      2.4668517113662403,
      0.42687494916218993,
      0.5731250508378101,
      0.5,
      1,
      1.5707963267948966,
      1,
      0,
      1,
      0,
      NaN,
      NaN,
      NaN,
      NaN
    ],
    bins: [3, 4, 8, 0, 0, 3, 0, 0, 0, 0, 0, 0]
  },
  'directional/compass': {
    summary: [
      4,
      5.387129922608242,
      0.42687494916218993,
      0.5731250508378101,
      0.5,
      1,
      0,
      1,
      0,
      1,
      0,
      NaN,
      NaN,
      NaN,
      NaN
    ],
    bins: [4, 3, 0, 8, 3, 0, 0, 0, 0, 0, 0, 0]
  },
  'axial/cartesian': {
    summary: [
      4,
      0,
      0.4666666666666667,
      0.5333333333333333,
      1,
      1,
      1.5707963267948966,
      1,
      0,
      1,
      0,
      NaN,
      NaN,
      NaN,
      NaN
    ],
    bins: [11, 0, 4, 0, 0, 0, 3, 0, 0, 0, 0, 0]
  },
  'axial/compass': {
    summary: [
      4,
      1.5707963267948966,
      0.4666666666666667,
      0.5333333333333333,
      1,
      1,
      0,
      1,
      0,
      1,
      0,
      NaN,
      NaN,
      NaN,
      NaN
    ],
    bins: [4, 0, 11, 0, 3, 0, 0, 0, 0, 0, 0, 0]
  }
};

for (const mode of ['directional', 'axial'] as const) {
  it(`bounds Rose ${mode} resultant and variance for identical non-cardinal lines`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    try {
      const summary = rig.output('float32', 5),
        bins = rig.output('float32', 4);
      rig.run(
        new GPURoseStatistic({
          starts: rig.input(new Float32Array(16), 'float32x2', 8),
          ends: rig.input(
            Float32Array.from({length: 16}, (_, index) => (index % 2 ? 29 : 24)),
            'float32x2',
            8
          ),
          circular: {
            mode,
            angleUnit: 'radians',
            origin: 'positive-x-counter-clockwise',
            binCount: 4
          },
          output: {summary, bins}
        })
      );
      // Identical unit vectors: resultant=1, variance=0, Rayleigh z=n=8;
      // mean direction atan2(29,24), unchanged by axial folding in this quadrant.
      const actual = await rig.readFloat(summary);
      // Rayleigh's eight unweighted contributions allow five f32 ulps; bounds below are exact.
      expectOracleRows(actual, [8, Math.atan2(29, 24), 1, 0, 8], 5e-6);
      expect(actual[2]).toBeGreaterThanOrEqual(0);
      expect(actual[2]).toBeLessThanOrEqual(1);
      expect(actual[3]).toBeGreaterThanOrEqual(0);
      expect(actual[3]).toBeLessThanOrEqual(1);
      expectOracleRows(
        await rig.readFloat(bins),
        mode === 'directional' ? [8, 0, 0, 0] : [0, 8, 0, 0],
        0
      );
    } finally {
      rig.destroy();
    }
  });
  for (const convention of ['cartesian', 'compass'] as const) {
    it(`checks all Rose ${mode}/${convention} summary slots and bins against the independent oracle`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      for (const transformed of [false, true]) {
        const rig = new AnalysisRig(device);
        try {
          const scale = transformed ? 3 : 1;
          const shift = transformed ? 32 : 0;
          const weightScale = transformed ? 1000 : 1;
          const summary = rig.output('float32', 15);
          const bins = rig.output('float32', 12);
          const circular = {
            mode,
            angleUnit: 'radians' as const,
            origin:
              convention === 'cartesian'
                ? ('positive-x-counter-clockwise' as const)
                : ('positive-y-clockwise' as const),
            binCount: 4
          };
          const contributor = new GPURoseStatistic({
            starts: rig.input(new Float32Array(16).fill(shift), 'float32x2', 8),
            ends: rig.input(
              Float32Array.from(
                [1, 0, 1, 0, 0, 1, -1, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                value => value * scale + shift
              ),
              'float32x2',
              8
            ),
            weights: rig.input(
              Float32Array.from([1, 2, 4, 8, 3, 1, 1, 1], value => value * weightScale),
              'float32'
            ),
            groupIds: rig.input(Uint32Array.of(0, 0, 0, 0, 1, 0, 0, 9), 'uint32'),
            mask: rig.input(Uint32Array.of(1, 1, 1, 1, 1, 0, 1, 1), 'uint32'),
            groupCount: 3,
            circular,
            output: {summary, bins}
          });
          circular.binCount = 8;
          rig.run(contributor);
          const expected = EXPECTED[`${mode}/${convention}`];
          // Moments have <=15 mass and small angles; 2e-6 allows a few f32 ulps.
          expectOracleRows(await rig.readFloat(summary), expected.summary, 2e-6);
          expectOracleRows(
            await rig.readFloat(bins),
            expected.bins.map(value => value * weightScale),
            0
          );
        } finally {
          rig.destroy();
        }
      }
    });
  }
}
