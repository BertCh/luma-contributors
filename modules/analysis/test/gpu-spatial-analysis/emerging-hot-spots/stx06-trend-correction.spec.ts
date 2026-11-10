// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {
  GPUEmergingHotSpots,
  getGPUEmergingHotSpotsParameterValues,
  GPU_EMERGING_HOT_SPOTS_MAXIMUM_CORRECTED_TREND_SLICE_COUNT
} from '../../../src/gpu-spatial-analysis/emerging-hot-spots/index';
import fixture from '../../oracles/fixtures/stx06-change-points.json';
import {requireTestDevice} from '../../utils/require-test-device';
import {AnalysisRig} from '../catchment-accessibility/rig';

type Correction = 'none' | 'hamed-rao' | 'yue-wang';

async function runHotSpots(options: {
  series: number[][];
  correction?: Correction;
  maximumLag?: number;
}) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  const cellCount = options.series.length;
  const sliceCount = options.series[0].length;
  const values = new Float32Array(cellCount * sliceCount);
  options.series.forEach((row, cell) => values.set(row, cell * sliceCount));
  const trendZ = rig.output('float32', cellCount);
  const trendP = rig.output('float32', cellCount);
  const trendS = rig.output('sint32', cellCount);
  const output = {
    giZScores: rig.output('float32', cellCount * sliceCount),
    trendZ,
    trendP,
    trendS,
    category: rig.output('uint32', cellCount),
    hotSliceCount: rig.output('uint32', cellCount),
    coldSliceCount: rig.output('uint32', cellCount)
  };
  rig.run(
    new GPUEmergingHotSpots({
      values: rig.input(values, 'float32'),
      gridWidth: cellCount,
      gridHeight: 1,
      sliceCount,
      maximumRadius: 1,
      multipleTesting: 'none',
      ...(options.correction ? {trendAutocorrelationCorrection: options.correction} : {}),
      ...(options.maximumLag ? {trendCorrectionMaximumLag: options.maximumLag} : {}),
      parameters: rig.input(
        getGPUEmergingHotSpotsParameterValues({
          radius: 0,
          temporalWindow: 0,
          confidenceLevel: 0.95,
          multipleTesting: 'none'
        }),
        'float32'
      ),
      output
    })
  );
  const result = {
    z: await rig.readFloat(trendZ),
    p: await rig.readFloat(trendP),
    s: await rig.readFloat(trendS as never)
  };
  rig.destroy();
  return result;
}

const fixtureCases: [
  string,
  Correction,
  number | undefined,
  keyof (typeof fixture.modifiedMannKendall)[number]
][] = [
  ['hamed-rao, all lags', 'hamed-rao', undefined, 'hamedRao'],
  ['hamed-rao, first 3 lags', 'hamed-rao', 3, 'hamedRaoLag3'],
  ['yue-wang, all lags', 'yue-wang', undefined, 'yueWang'],
  ['yue-wang, lag 1', 'yue-wang', 1, 'yueWangLag1']
];

describe('GPUEmergingHotSpots trend autocorrelation correction', () => {
  const series = fixture.modifiedMannKendall.map(entry => entry.series);

  it('is the uncorrected Mann-Kendall by default and matches pymannkendall.original_test', async () => {
    const result = await runHotSpots({series});
    if (!result.z) return;
    fixture.modifiedMannKendall.forEach((entry, cell) => {
      expect(Math.abs(result.z[cell] - entry.original.z)).toBeLessThan(1e-3);
    });
  });

  for (const [name, correction, maximumLag, key] of fixtureCases) {
    it(`${name} matches pymannkendall`, async () => {
      const result = await runHotSpots({series, correction, maximumLag});
      if (!result.z) return;
      fixture.modifiedMannKendall.forEach((entry, cell) => {
        const expected = entry[key] as {z: number; p: number};
        expect(Math.abs(result.z[cell] - expected.z)).toBeLessThan(
          2e-3 * Math.max(1, Math.abs(expected.z))
        );
        expect(Math.abs(result.p[cell] - expected.p)).toBeLessThan(
          2e-3 * Math.max(expected.p, 1e-3)
        );
      });
    });
  }

  it('is less significant on strong autocorrelation than the uncorrected test', async () => {
    const uncorrected = await runHotSpots({series});
    const corrected = await runHotSpots({series, correction: 'hamed-rao'});
    if (!uncorrected.z || !corrected.z) return;
    // Series 0 is trend-free AR(1): the uncorrected z overstates it.
    expect(Math.abs(corrected.z[0])).toBeLessThanOrEqual(Math.abs(uncorrected.z[0]) + 1e-6);
  });

  it('rejects a series longer than the corrected path by name', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    const sliceCount = GPU_EMERGING_HOT_SPOTS_MAXIMUM_CORRECTED_TREND_SLICE_COUNT + 1;
    expect(
      () =>
        new GPUEmergingHotSpots({
          values: rig.output('float32', sliceCount),
          gridWidth: 1,
          gridHeight: 1,
          sliceCount,
          trendAutocorrelationCorrection: 'yue-wang',
          parameters: rig.output('float32', 8),
          output: {
            giZScores: rig.output('float32', sliceCount),
            trendZ: rig.output('float32', 1),
            trendP: rig.output('float32', 1),
            trendS: rig.output('sint32', 1),
            category: rig.output('uint32', 1),
            hotSliceCount: rig.output('uint32', 1),
            coldSliceCount: rig.output('uint32', 1)
          }
        })
    ).toThrow(/sliceCount/);
    rig.destroy();
  });
});
