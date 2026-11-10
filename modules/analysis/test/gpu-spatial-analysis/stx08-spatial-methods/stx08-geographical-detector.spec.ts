// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {
  getGeographicalDetectorPValue,
  getNonCentralFUpperTail,
  GPU_GEOGRAPHICAL_DETECTOR_LAYOUT,
  GPUGeographicalDetector
} from '../../../src/gpu-spatial-analysis/geographical-detector/index';
import {GPURegionPartitionEvaluation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-region-partition-evaluation';
import fixture from '../../oracles/fixtures/stx08-spatial-methods.json';
import {requireTestDevice} from '../../utils/require-test-device';
import {AnalysisRig} from '../catchment-accessibility/rig';

const NONE = 0xffffffff;
const layout = GPU_GEOGRAPHICAL_DETECTOR_LAYOUT;

async function runDetector(values: number[], labels: number[], labelCapacity?: number) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  const summary = rig.output('float32', layout.length);
  rig.run(
    new GPUGeographicalDetector({
      values: rig.input(new Float32Array(values), 'float32'),
      labels: rig.input(new Uint32Array(labels), 'uint32'),
      labelCapacity,
      output: {summary}
    })
  );
  const words = await rig.readFloat(summary);
  rig.destroy();
  return words;
}

describe('GPUGeographicalDetector', () => {
  it('matches a hand computation: q = 1 - SSW / SST', async () => {
    // Strata {1,2,3} and {10,11,12}: SSW = 2 + 2, SST = 125.5.
    const words = await runDetector([1, 2, 3, 10, 11, 12], [0, 0, 0, 1, 1, 1]);
    expect(words[layout.q]).toBeCloseTo(1 - 4 / 125.5, 6);
    expect(words[layout.totalSumOfSquares]).toBeCloseTo(125.5, 3);
    expect(words[layout.strataCount]).toBe(2);
    expect(words[layout.observationCount]).toBe(6);
    expect(words[layout.numeratorDegreesOfFreedom]).toBe(1);
    expect(words[layout.denominatorDegreesOfFreedom]).toBe(4);
    const q = 1 - 4 / 125.5;
    expect(words[layout.fStatistic] / ((4 * q) / (1 * (1 - q)))).toBeCloseTo(1, 4);
  });

  it('leaves unlabelled and out-of-capacity rows out of every statistic', async () => {
    const withExtras = await runDetector(
      [1, 2, 3, 10, 11, 12, 1000, -500, 77],
      [0, 0, 0, 1, 1, 1, NONE, 9, NONE],
      3
    );
    const without = await runDetector([1, 2, 3, 10, 11, 12], [0, 0, 0, 1, 1, 1], 3);
    for (let word = 0; word < layout.length; word++) {
      expect(withExtras[word]).toBe(without[word]);
    }
    expect(withExtras[layout.observationCount]).toBe(6);
  });

  it('reports undefined statistics for a constant response and for a single stratum', async () => {
    const constant = await runDetector([4, 4, 4, 4], [0, 0, 1, 1]);
    expect(constant[layout.q]).toBeNaN();
    expect(constant[layout.fStatistic]).toBeNaN();
    const single = await runDetector([1, 2, 3, 4], [0, 0, 0, 0]);
    expect(single[layout.q]).toBeCloseTo(0, 6);
    expect(single[layout.fStatistic]).toBeNaN();
    const perfect = await runDetector([1, 1, 5, 5], [0, 0, 1, 1]);
    expect(perfect[layout.q]).toBe(1);
    expect(perfect[layout.fStatistic]).toBe(Infinity);
    expect(getGeographicalDetectorPValue(perfect)).toBe(0);
  });

  for (const reference of fixture.geographicalDetector) {
    it(`agrees with geodetector ${fixture.versions.geodetector} on the ${reference.name} case`, async () => {
      const words = await runDetector(reference.values, reference.labels);
      expect(Math.abs(words[layout.q] - reference.q)).toBeLessThan(2e-5);
      expect(words[layout.fStatistic] / reference.fStatistic).toBeCloseTo(1, 3);
      // lambda subtracts nearly equal sums when the means dwarf the spread (the offset case).
      expect(words[layout.nonCentrality] / reference.nonCentrality).toBeCloseTo(
        1,
        reference.name === 'offset' ? 1 : 3
      );
      expect(words[layout.totalSumOfSquares] / reference.totalSumOfSquares).toBeCloseTo(1, 5);
      if (reference.name !== 'offset') {
        expect(Math.abs(getGeographicalDetectorPValue(words) - reference.pValue)).toBeLessThan(
          2e-3
        );
      }
    });

    it(`non-central F tail equals scipy ncf.sf on the ${reference.name} reference F and lambda`, () => {
      const p = getNonCentralFUpperTail(
        reference.fStatistic,
        reference.strataCount - 1,
        reference.values.length - reference.strataCount,
        reference.nonCentrality
      );
      expect(Math.abs(p - reference.pValue)).toBeLessThan(1e-9);
    });
  }
});

describe('GPURegionPartitionEvaluation regionMeans', () => {
  it('writes the mean per label and NaN for an empty label', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    const summary = rig.output('float32', 9);
    const means = rig.output('float32', 3 * 2);
    rig.run(
      new GPURegionPartitionEvaluation({
        values: rig.input(new Float32Array([1, 10, 3, 20, 5, 30]), 'float32'),
        columnCount: 2,
        labels: rig.input(new Uint32Array([0, 0, 2]), 'uint32'),
        labelCapacity: 3,
        output: {summary, regionMeans: means}
      })
    );
    const read = await rig.readFloat(means);
    rig.destroy();
    expect(read.slice(0, 2)).toEqual([2, 15]);
    expect(read[2]).toBeNaN();
    expect(read[3]).toBeNaN();
    expect(read.slice(4, 6)).toEqual([5, 30]);
  });
});
