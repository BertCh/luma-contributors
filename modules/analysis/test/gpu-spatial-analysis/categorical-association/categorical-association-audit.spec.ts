// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUColocationQuotient} from '../../../src/gpu-spatial-analysis/categorical-association/gpu-colocation-quotient';
import {GPULocalCategoryDependence} from '../../../src/gpu-spatial-analysis/categorical-association/gpu-local-category-dependence';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  CATEGORICAL_AUDIT_CATEGORIES,
  CATEGORICAL_AUDIT_OFFSETS,
  CATEGORICAL_AUDIT_NEIGHBORS,
  CATEGORICAL_AUDIT_WEIGHTS,
  CATEGORICAL_AUDIT_ROW_COUNTS,
  CATEGORICAL_AUDIT_GLOBAL,
  CATEGORICAL_AUDIT_LOCAL
} from './categorical-association-audit.test-utils';

function expectOracle(actual: number[], expected: number[], tolerance: number, label = '') {
  expect(actual).toHaveLength(expected.length);
  expected.forEach((value, index) => {
    const message = `${label} word ${index}: actual ${actual[index]}, expected ${value}`;
    if (Number.isNaN(value)) expect(actual[index], message).toBeNaN();
    else {
      expect(Number.isFinite(actual[index]), message).toBe(true);
      expect(Math.abs(actual[index] - value), message).toBeLessThanOrEqual(tolerance);
    }
  });
}

it('R7 local category tail preserves the 0.05 decision with 2190 neighbors', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    const statistics = rig.output('float32', 2270 * 6),
      categoryRowCounts = rig.output('uint32', 2);
    rig.run(
      new GPULocalCategoryDependence({
        weights: rig.weights({
          offsets: Array.from({length: 2271}, (_, index) => (index === 0 ? 0 : 2190)),
          neighbors: Array.from({length: 2190}, (_, index) => index + 1),
          weights: new Array(2190).fill(1)
        }),
        categories: rig.input(
          Uint32Array.from({length: 2270}, (_, row) => Number(row >= 1135)),
          'uint32'
        ),
        categoryCount: 2,
        output: {statistics, categoryRowCounts}
      })
    );
    const actual = await rig.readFloat(statistics);
    // SciPy binom.sf(1133,2190,.5); 2e-7 absolute covers f32 mode-centered
    // summation while distinguishing the old 0.0500103 result across the 0.05 boundary.
    expectOracle(
      actual.slice(0, 6),
      [1134, 2190, 0.5178082191780822, 0.5, 0.049933762130512475, 1.666757988365684],
      2e-7,
      'focal row'
    );
    expect(actual[4]).toBeLessThan(0.05);
    for (let row = 1; row < 2270; row++)
      expectOracle(
        actual.slice(row * 6, row * 6 + 6),
        [0, 0, NaN, NaN, NaN, NaN],
        0,
        `island ${row}`
      );
    expect(await rig.readUint(categoryRowCounts)).toEqual([1135, 1135]);
  } finally {
    rig.destroy();
  }
});

it('audit-validation co-location matches every Fraction oracle word, including overflow-safe weight scaling', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scale of [1, 2 ** 64, 2 ** 126, 1.25 * 2 ** 126]) {
    const rig = new AnalysisRig(device);
    try {
      const globalQuotients = rig.output('float32', 16),
        localQuotients = rig.output('float32', 32),
        categoryRowCounts = rig.output('uint32', 4);
      const props = {
        weights: rig.weights({
          offsets: CATEGORICAL_AUDIT_OFFSETS,
          neighbors: CATEGORICAL_AUDIT_NEIGHBORS,
          weights: CATEGORICAL_AUDIT_WEIGHTS.map(weight => weight * scale)
        }),
        categories: rig.input(Uint32Array.from(CATEGORICAL_AUDIT_CATEGORIES), 'uint32'),
        categoryCount: 4,
        output: {globalQuotients, localQuotients, categoryRowCounts}
      };
      const contributor = new GPUColocationQuotient(props);
      props.categoryCount = 1;
      props.categories = rig.input(Uint32Array.of(0), 'uint32');
      props.weights.offsets = rig.input(Uint32Array.of(0, 0), 'uint32');
      props.weights.weights = rig.input(Float32Array.of(0), 'float32');
      props.output.globalQuotients = rig.output('float32', 1);
      props.output.localQuotients = rig.output('float32', 1);
      props.output.categoryRowCounts = rig.output('uint32', 1);
      rig.run(contributor);
      // categorical-association-audit.oracle.py: all shares are dyadic. Allow 2e-6
      // for float32 divisions and fixed-point accumulation/decode.
      expectOracle(
        await rig.readFloat(globalQuotients),
        CATEGORICAL_AUDIT_GLOBAL,
        2e-6,
        `global scale ${scale}`
      );
      expectOracle(
        await rig.readFloat(localQuotients),
        CATEGORICAL_AUDIT_LOCAL,
        2e-6,
        `local scale ${scale}`
      );
      expect(await rig.readUint(categoryRowCounts)).toEqual(CATEGORICAL_AUDIT_ROW_COUNTS);
    } finally {
      rig.destroy();
    }
  }
});

it('audit-validation local category composition matches every SciPy binomial oracle word', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    const statistics = rig.output('float32', 48),
      categoryRowCounts = rig.output('uint32', 4);
    const props = {
      weights: rig.weights({
        offsets: CATEGORICAL_AUDIT_OFFSETS,
        neighbors: CATEGORICAL_AUDIT_NEIGHBORS,
        weights: CATEGORICAL_AUDIT_WEIGHTS
      }),
      categories: rig.input(Uint32Array.from(CATEGORICAL_AUDIT_CATEGORIES), 'uint32'),
      categoryCount: 4,
      output: {statistics, categoryRowCounts}
    };
    const contributor = new GPULocalCategoryDependence(props);
    props.categoryCount = 1;
    props.categories = rig.input(Uint32Array.of(0), 'uint32');
    props.weights.neighbors = rig.input(Uint32Array.of(0), 'uint32');
    props.output.statistics = rig.output('float32', 1);
    props.output.categoryRowCounts = rig.output('uint32', 1);
    rig.run(contributor);
    // scipy.stats.binom.sf(like-1,n,p), with the focal row excluded from trials.
    // The 3e-6 absolute tolerance covers short log/exp recurrence and z-score arithmetic.
    expectOracle(
      await rig.readFloat(statistics),
      [
        1,
        2,
        0.5,
        0.42857142857142855,
        0.673469387755102,
        0.2041241452319316,
        1,
        2,
        0.5,
        0.42857142857142855,
        0.673469387755102,
        0.2041241452319316,
        1,
        3,
        0.3333333333333333,
        0.2857142857142857,
        0.6355685131195334,
        0.18257418583505544,
        1,
        1,
        1,
        0.2857142857142857,
        0.2857142857142857,
        1.5811388300841895,
        1,
        2,
        0.5,
        0.2857142857142857,
        0.4897959183673469,
        0.6708203932499369,
        0,
        0,
        NaN,
        NaN,
        NaN,
        NaN,
        0,
        0,
        NaN,
        NaN,
        NaN,
        NaN,
        0,
        0,
        NaN,
        NaN,
        NaN,
        NaN
      ],
      3e-6
    );
    expect(await rig.readUint(categoryRowCounts)).toEqual(CATEGORICAL_AUDIT_ROW_COUNTS);
  } finally {
    rig.destroy();
  }
});
