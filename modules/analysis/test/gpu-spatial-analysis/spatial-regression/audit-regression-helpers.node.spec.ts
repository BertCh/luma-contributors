// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  factorCPURegressionCorrelation,
  solveCPURegressionFactor,
  getCPURegressionSingularValueRange,
  fitCPURegressionWithQR
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-regression-linear-algebra';
import {AUDIT_OLS_FIXTURES} from './audit-ols-fixtures';
import {
  getLogGammaOfHalfInteger,
  getChiSquareSurvival,
  getStudentTTwoSidedPValue,
  getStudentTSurvival
} from '../../../src/gpu-spatial-analysis/spatial-regression/ordinary-least-squares-statistics';
import {AUDIT_STATISTICS} from './audit-regression-helpers-fixtures';
import {getOrdinaryLeastSquaresAllocationLengths} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-ordinary-least-squares';
import {getRegimeMomentStride} from '../../../src/gpu-spatial-analysis/spatial-regression/regime-moments';
import {
  getRegressionTraceAllocationLengths,
  getRegressionErrorGMAllocationLengths,
  getRegressionTwoStageAllocationLengths
} from '../../../src/gpu-spatial-analysis/spatial-regression/regression-binding-plans';

function expectClose(actual: number, expected: number) {
  expect(Number.isFinite(actual)).toBe(true);
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(1e-10 * Math.max(1, Math.abs(expected)));
}

it('every exported distribution helper matches independent SciPy tail and gamma values', () => {
  for (const row of AUDIT_STATISTICS.gamma)
    expectClose(getLogGammaOfHalfInteger(row.freedom), row.value);
  const expectTail = (actual: number, expected: number) => {
    expect(Number.isFinite(actual)).toBe(true);
    expect(Math.abs(actual / expected - 1)).toBeLessThan(1e-10);
  };
  for (const row of AUDIT_STATISTICS.chiSquare)
    expectTail(getChiSquareSurvival(row.value, row.freedom), row.survival);
  for (const row of AUDIT_STATISTICS.student) {
    expectTail(getStudentTSurvival(row.statistic, row.freedom), row.survival);
    expectTail(getStudentTTwoSidedPValue(row.statistic, row.freedom), row.twoSided);
  }
});

for (const fixture of AUDIT_OLS_FIXTURES) {
  it(`CPU factor, solve, singular values and QR covariance match SciPy: ${fixture.name}`, () => {
    const rows = fixture.response.flatMap((response, row) => {
      const values = fixture.predictors.slice(row * 2, row * 2 + 2);
      return fixture.mask[row] && Number.isFinite(response) && values.every(Number.isFinite)
        ? [{design: [1, ...values], response}]
        : [];
    });
    const design = rows.map(row => row.design);
    const response = rows.map(row => row.response);
    const gram = Array.from({length: 3}, (_, first) =>
      Array.from({length: 3}, (_, second) =>
        design.reduce((sum, row) => sum + row[first] * row[second], 0)
      )
    );
    const right = Array.from({length: 3}, (_, column) =>
      rows.reduce((sum, row) => sum + row.design[column] * row.response, 0)
    );
    const factor = factorCPURegressionCorrelation(gram);
    expect(factor).toBeDefined();
    for (let first = 0; first < 3; first++)
      for (let second = 0; second < 3; second++)
        expectClose(factor![first][second], fixture.cholesky[first][second]);
    solveCPURegressionFactor(factor!, right).forEach((value, column) =>
      expectClose(value, fixture.expected.coefficients[column])
    );
    getCPURegressionSingularValueRange(design).forEach((value, index) =>
      expectClose(value, fixture.singularRange[index])
    );
    const fit = fitCPURegressionWithQR(design, response);
    expect(fit).toBeDefined();
    fit!.estimate.forEach((value, column) =>
      expectClose(value, fixture.expected.coefficients[column])
    );
    for (let first = 0; first < 3; first++)
      for (let second = 0; second < 3; second++) {
        const covariance = fit!.inverseRoot[first].reduce(
          (sum, value, column) => sum + value * fit!.inverseRoot[second][column],
          0
        );
        expectClose(covariance, fixture.inverseNormal[first][second]);
      }
    expectClose(
      rows.reduce(
        (sum, row) =>
          sum +
          (row.response -
            row.design.reduce(
              (total, value, column) => total + value * fit!.estimate[column],
              0
            )) **
            2,
        0
      ),
      fixture.expected.summary[11]
    );
  });
}

it('CPU numerical helpers report zero singular values and refuse rank-deficient systems', () => {
  expect(
    getCPURegressionSingularValueRange([
      [0, 0],
      [0, 0]
    ])
  ).toEqual([0, 0]);
  expect(
    factorCPURegressionCorrelation([
      [1, 1],
      [1, 1]
    ])
  ).toBeUndefined();
  expect(
    factorCPURegressionCorrelation([
      [1, 2],
      [2, 1]
    ])
  ).toBeUndefined();
  expect(
    fitCPURegressionWithQR(
      [
        [1, 1],
        [1, 1],
        [1, 1]
      ],
      [1, 2, 3]
    )
  ).toBeUndefined();
});

it('allocation helpers count the independently enumerated small array layouts', () => {
  // OLS p=2: D=3, six packed products, three means and three deviations; workspace
  // has 32 words including four residual-normalization words. Nine rows in three tiles
  // need eight mean and nine moment words per tile.
  const ordinary = getOrdinaryLeastSquaresAllocationLengths(9, 2, 4);
  expect(ordinary).toEqual({
    // One normalized-response/magnitude word per each of the nine rows.
    'OLS normalized response/magnitude scratch': 9,
    'OLS residuals/squared residuals': 9,
    'OLS workspace': 32,
    'OLS mean partials': 24,
    'OLS moment partials': 27,
    'OLS residual partials': 9,
    'OLS mean metadata groups': 0,
    'OLS merge groups': 0
  });
  expect([getRegimeMomentStride(1), getRegimeMomentStride(2), getRegimeMomentStride(11)]).toEqual([
    7, 12, 102
  ]);
  // Five probes, batches [2,2,1], one 64-word slot per batch, four Gram channels.
  expect(
    getRegressionTraceAllocationLengths({
      rowCount: 8,
      edgeCount: 12,
      maximumPower: 2,
      tileCount: 3,
      mode: 'stochastic',
      probeCount: 5,
      probeBatchSize: 2,
      includeGram: true
    })
  ).toEqual({
    'probe vectors': 16,
    'probe partials': 6,
    'probe merge groups': 0,
    'probe batch information': 192,
    'probe dots': 768,
    'trace norm partials': 3,
    'trace merge groups': 0,
    'trace scaled weights': 12,
    'trace sums partials': 128,
    'trace ones vectors': 8,
    'trace infinity norm': 1,
    'Gram partials': 6,
    'transpose offsets': 9,
    'transpose slots and sort scratch': 12,
    'transpose tolerance': 1
  });
  // Three regression columns and four instruments: 16 instrument Gram entries,
  // 12 cross entries, four instrument-response entries and three design-response entries,
  // plus six correction terms = 41 words per moment tile.
  expect(getRegressionTwoStageAllocationLengths(8, 2, 12, 1, 4)).toEqual({
    '2SLS mean partials': 12,
    '2SLS merge groups': 0,
    '2SLS moment partials': 82,
    '2SLS moments': 41,
    '2SLS workspace': 18,
    '2SLS residual partials': 8,
    '2SLS residual totals': 4,
    '2SLS refinement partials': 10,
    '2SLS refinement totals': 5,
    '2SLS norm partials': 6,
    '2SLS AK partials': 16,
    '2SLS AK totals': 8,
    '2SLS row scratch': 8,
    '2SLS transpose offsets': 9,
    '2SLS transpose slots and sort scratch': 12,
    '2SLS transpose tolerance': 1
  });
  expect(
    getRegressionErrorGMAllocationLengths({
      rowCount: 9,
      predictorCount: 2,
      edgeCount: 12,
      tileRowCount: 4
    })
  ).toEqual({
    ...ordinary,
    'GM OLS outputs': 22,
    'GM row scratch': 9,
    'GM workspace': 25,
    'GM mean partials': 9,
    'GM merge groups': 0,
    'GM moment partials': 21,
    'GM moments': 7,
    'GM filtered partials': 39,
    'GM filtered moments': 13,
    'GM residual partials': 15,
    'GM residual totals': 5,
    'GM trace outputs': 4
  });
});
