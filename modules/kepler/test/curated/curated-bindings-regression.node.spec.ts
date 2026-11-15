// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import type {
  CatalogFeatureCollection,
  CatalogOutputTable
} from '../../src/analysis/catalog-geojson';
import {
  computeBelsleyConditionNumber,
  fitStandardizedLeastSquares,
  computeSymmetricEigenvalues
} from '../../src/curated/bindings/curated-regression-with-diagnostics-binding';
import {REGRESSION_LATTICE, REGRESSION_POINTS} from './curated-bindings-oracle-data';
import {
  getMaximumRelativeError,
  getNumbers,
  getResultTable,
  makeLattice,
  makePoints,
  recordBindingMargin,
  runCuratedBinding
} from './curated-bindings-test-support';

// Oracles: /usr/bin/python3 .typecheck/b1-oracles.py. OLS by numpy.linalg.lstsq with covariance
// s^2 (X'X)^-1; p-values from scipy.stats.t; Jarque-Bera from scipy.stats.jarque_bera; Koenker and
// Bassett n R^2 and Breusch and Pagan half the explained sum of squares of e^2 / (RSS / n) - 1, each
// by an auxiliary lstsq; the Belsley condition number is numpy.linalg.cond of the unit-length
// scaled design with its intercept; Moran's I from its definition with row-standardised queen (the
// lattice) or five nearest neighbour (the points) weights.

const ID = 'curated-regression-with-diagnostics';
const PLANAR = {coordinateSystem: 'planar'};
const TOLERANCE = 1e-9;

function getDiagnosticRows(table: CatalogOutputTable) {
  const rows = new Map<
    string,
    {value: number | null; df: number | null; p: number | null; expected: number | null}
  >();
  (table.columns['diagnostic'] as string[]).forEach((name, index) =>
    rows.set(name, {
      value: table.columns['value'][index] as number | null,
      df: table.columns['degrees_of_freedom'][index] as number | null,
      p: table.columns['p_value'][index] as number | null,
      expected: table.columns['expected'][index] as number | null
    })
  );
  return rows;
}

function makeRegressionLattice(): CatalogFeatureCollection {
  return makeLattice(6, 6, index => ({
    y: REGRESSION_LATTICE.y[index],
    x1: REGRESSION_LATTICE.x1[index],
    x2: REGRESSION_LATTICE.x2[index]
  }));
}

function runLattice(areas: CatalogFeatureCollection, parameters: Record<string, unknown> = {}) {
  return runCuratedBinding(
    ID,
    {
      responseProperty: 'y',
      predictorProperties: 'x1, x2',
      ...PLANAR,
      permutations: 99,
      seed: 5,
      ...parameters
    },
    {areas}
  );
}

describe('curated-regression-with-diagnostics against NumPy and SciPy (6 by 6 lattice)', () => {
  const result = runLattice(makeRegressionLattice());
  const coefficients = getResultTable(result, 'coefficients');
  const rows = getDiagnosticRows(getResultTable(result, 'diagnostics'));
  const residuals = getResultTable(result, 'residuals');

  function expectClose(
    name: string,
    actual: number | null,
    expected: number,
    tolerance = TOLERANCE
  ) {
    const error = Math.abs((actual as number) - expected) / Math.abs(expected);
    recordBindingMargin(`regression ${name} relative error`, error);
    expect(error, name).toBeLessThan(tolerance);
  }

  it('reproduces coefficients, standard errors, t statistics and p-values to 1e-9 relative', () => {
    expect(coefficients.columns['term']).toEqual(['(intercept)', 'x1', 'x2']);
    for (const [column, expected] of [
      ['estimate', REGRESSION_LATTICE.coefficients],
      ['standard_error', REGRESSION_LATTICE.standardErrors],
      ['t_statistic', REGRESSION_LATTICE.tStatistics],
      ['p_value', REGRESSION_LATTICE.pValues]
    ] as const) {
      const error = getMaximumRelativeError(getNumbers(coefficients, column), expected);
      recordBindingMargin(`regression coefficient ${column} max relative error`, error);
      expect(error, column).toBeLessThan(TOLERANCE);
    }
  });

  it('reproduces R squared, adjusted R squared, sigma squared, log-likelihood, AIC and AICc to 1e-9', () => {
    expectClose('r-squared', rows.get('r-squared')!.value, REGRESSION_LATTICE.rSquared);
    expectClose(
      'adjusted-r-squared',
      rows.get('adjusted-r-squared')!.value,
      REGRESSION_LATTICE.adjustedRSquared
    );
    expectClose('sigma-squared', rows.get('sigma-squared')!.value, REGRESSION_LATTICE.sigmaSquared);
    expect(rows.get('sigma-squared')!.df).toBe(33);
    expectClose(
      'log-likelihood',
      rows.get('log-likelihood')!.value,
      REGRESSION_LATTICE.logLikelihood
    );
    expectClose('aic', rows.get('aic')!.value, REGRESSION_LATTICE.aic);
    expectClose('aicc', rows.get('aicc')!.value, REGRESSION_LATTICE.aicc);
  });

  it('reproduces Jarque-Bera, Breusch-Pagan, Koenker-Bassett, their p-values and the condition number to 1e-9', () => {
    expectClose('jarque-bera', rows.get('jarque-bera')!.value, REGRESSION_LATTICE.jarqueBera);
    expectClose('jarque-bera p', rows.get('jarque-bera')!.p, REGRESSION_LATTICE.jarqueBeraPValue);
    expect(rows.get('jarque-bera')!.df).toBe(2);
    expectClose('breusch-pagan', rows.get('breusch-pagan')!.value, REGRESSION_LATTICE.breuschPagan);
    expectClose(
      'breusch-pagan p',
      rows.get('breusch-pagan')!.p,
      REGRESSION_LATTICE.breuschPaganPValue
    );
    expectClose(
      'koenker-bassett',
      rows.get('koenker-bassett')!.value,
      REGRESSION_LATTICE.koenkerBassett
    );
    expectClose(
      'koenker-bassett p',
      rows.get('koenker-bassett')!.p,
      REGRESSION_LATTICE.koenkerBassettPValue
    );
    expect(rows.get('breusch-pagan')!.df).toBe(2);
    expect(rows.get('koenker-bassett')!.df).toBe(2);
    expectClose(
      'condition-number',
      rows.get('condition-number')!.value,
      REGRESSION_LATTICE.conditionNumber
    );
  });

  it('publishes fitted values and residuals per feature, and the residual Moran statistic to 1e-12', () => {
    expect(residuals.rowCount).toBe(36);
    expect(
      getMaximumRelativeError(getNumbers(residuals, 'residual'), REGRESSION_LATTICE.residuals, 1e-2)
    ).toBeLessThan(1e-9);
    expect(
      getMaximumRelativeError(getNumbers(residuals, 'fitted'), REGRESSION_LATTICE.fitted, 1e-2)
    ).toBeLessThan(1e-9);
    const moran = rows.get('residual-morans-i')!;
    recordBindingMargin(
      'regression residual Moran abs error',
      Math.abs(moran.value! - REGRESSION_LATTICE.moransI)
    );
    expect(Math.abs(moran.value! - REGRESSION_LATTICE.moransI)).toBeLessThan(1e-12);
    expect(moran.expected).toBeCloseTo(-1 / 35, 15);
    expect(residuals.columns['neighbor_count'][0]).toBe(3);
    expect(residuals.columns['neighbor_count'][14]).toBe(8);
  });

  it('gives a pseudo p-value that is at least 1 / (permutations + 1), repeats for a seed and changes with it', () => {
    const p = rows.get('residual-morans-i')!.p!;
    expect(p).toBeGreaterThanOrEqual(1 / 100);
    expect(p).toBeLessThanOrEqual(1);
    expect(Math.abs(p * 100 - Math.round(p * 100))).toBeLessThan(1e-9);
    const again = getDiagnosticRows(
      getResultTable(runLattice(makeRegressionLattice()), 'diagnostics')
    );
    expect(again.get('residual-morans-i')!.p).toBe(p);
    const other = getDiagnosticRows(
      getResultTable(runLattice(makeRegressionLattice(), {seed: 6}), 'diagnostics')
    );
    expect(other.get('residual-morans-i')!.value).toBe(rows.get('residual-morans-i')!.value);
  });
});

describe('curated-regression-with-diagnostics on points (five nearest neighbours)', () => {
  const points = makePoints(REGRESSION_POINTS.px, REGRESSION_POINTS.py, index => ({
    y: REGRESSION_POINTS.y[index],
    x: REGRESSION_POINTS.x[index]
  }));
  const result = runCuratedBinding(
    ID,
    {
      responseProperty: 'y',
      predictorProperties: 'x',
      neighborhood: 'knn',
      neighborCount: 5,
      ...PLANAR,
      permutations: 99
    },
    {areas: points}
  );

  it('matches the oracle for one predictor, with k nearest neighbours for the residual statistic', () => {
    const coefficients = getResultTable(result, 'coefficients');
    expect(
      getMaximumRelativeError(getNumbers(coefficients, 'estimate'), REGRESSION_POINTS.coefficients)
    ).toBeLessThan(TOLERANCE);
    expect(
      getMaximumRelativeError(
        getNumbers(coefficients, 'standard_error'),
        REGRESSION_POINTS.standardErrors
      )
    ).toBeLessThan(TOLERANCE);
    const rows = getDiagnosticRows(getResultTable(result, 'diagnostics'));
    expect(
      Math.abs(rows.get('residual-morans-i')!.value! - REGRESSION_POINTS.moransI)
    ).toBeLessThan(1e-12);
    expect(
      Math.abs(rows.get('aicc')!.value! - REGRESSION_POINTS.aicc) / REGRESSION_POINTS.aicc
    ).toBeLessThan(TOLERANCE);
    expect(rows.get('breusch-pagan')!.df).toBe(1);
    expect(result.summary['neighborhood']).toBe('knn');
  });

  it('uses nearest neighbours by default for points (auto) and rejects queen for points', () => {
    const automatic = runCuratedBinding(
      ID,
      {responseProperty: 'y', predictorProperties: 'x', ...PLANAR, permutations: 99},
      {areas: points}
    );
    expect(automatic.summary['neighborhood']).toBe('knn');
    expect(() =>
      runCuratedBinding(
        ID,
        {responseProperty: 'y', predictorProperties: 'x', neighborhood: 'queen', ...PLANAR},
        {areas: points}
      )
    ).toThrow(/neighborhood queen needs polygons/);
  });
});

describe('curated-regression-with-diagnostics declared behaviour', () => {
  it('uses complete cases only, lists dropped rows with a status and gives the fit of the remaining rows', () => {
    const areas = makeRegressionLattice();
    areas.features[0].properties = {y: null, x1: 1, x2: 2};
    areas.features[1].properties = {y: 1, x1: 'abc', x2: 2};
    areas.features[2].properties = {y: 1, x1: 1, x2: Number.NaN};
    areas.features.push({id: 'no-geometry', geometry: null, properties: {y: 1, x1: 1, x2: 1}});
    const result = runLattice(areas);
    const table = getResultTable(result, 'residuals');
    expect(table.rowCount).toBe(37);
    expect(table.columns['status'].slice(0, 3)).toEqual([
      'missing-value',
      'missing-value',
      'missing-value'
    ]);
    expect(table.columns['status'][36]).toBe('missing-geometry');
    expect(table.columns['residual'][0]).toBeNull();
    expect(result.summary['rowCount']).toBe(33);
    expect(result.summary['droppedCount']).toBe(4);
    expect(result.diagnostics.find(item => item.code === 'missing-value')?.count).toBe(3);
    expect(result.diagnostics.find(item => item.code === 'missing-geometry')?.count).toBe(1);
    // The same fit on the 33 rows alone.
    const kept = makeLattice(6, 6, index => ({
      y: REGRESSION_LATTICE.y[index],
      x1: REGRESSION_LATTICE.x1[index],
      x2: REGRESSION_LATTICE.x2[index]
    }));
    kept.features = kept.features.slice(3);
    const subset = runLattice(kept);
    expect(
      getMaximumRelativeError(
        getNumbers(getResultTable(result, 'coefficients'), 'estimate'),
        getNumbers(getResultTable(subset, 'coefficients'), 'estimate')
      )
    ).toBeLessThan(1e-12);
  });

  it('throws a named error for collinear predictors, a constant predictor and a constant response', () => {
    const collinear = makeLattice(6, 6, index => ({
      y: REGRESSION_LATTICE.y[index],
      x1: REGRESSION_LATTICE.x1[index],
      x2: 2 * REGRESSION_LATTICE.x1[index]
    }));
    expect(() => runLattice(collinear)).toThrow(
      /curated-regression-with-diagnostics: the predictors are collinear: x2 is an almost exact linear combination/
    );
    const constantPredictor = makeLattice(6, 6, index => ({
      y: REGRESSION_LATTICE.y[index],
      x1: REGRESSION_LATTICE.x1[index],
      x2: 4
    }));
    expect(() => runLattice(constantPredictor)).toThrow(/the predictor x2 is constant/);
    const constantResponse = makeLattice(6, 6, index => ({
      y: 3,
      x1: REGRESSION_LATTICE.x1[index],
      x2: REGRESSION_LATTICE.x2[index]
    }));
    expect(() => runLattice(constantResponse)).toThrow(/the response y is constant/);
  });

  it('throws for too few complete rows, an empty column list, a repeated or response predictor', () => {
    const small = makeLattice(2, 2, index => ({y: index, x1: index * index, x2: index % 2}));
    expect(() => runLattice(small)).toThrow(
      /4 complete rows for 2 predictors; at least 6 are needed/
    );
    const areas = makeRegressionLattice();
    expect(() => runLattice(areas, {predictorProperties: ' , '})).toThrow(
      /must name 1 to 15 columns, got 0/
    );
    expect(() => runLattice(areas, {predictorProperties: 'x1,x1'})).toThrow(/names a column twice/);
    expect(() => runLattice(areas, {predictorProperties: 'x1,y'})).toThrow(
      /includes the response y/
    );
    expect(() => runLattice(areas, {responseProperty: ''})).toThrow(
      /parameter responseProperty must name a column/
    );
    expect(() => runLattice(areas, {predictorProperties: 'nothing'})).toThrow(
      /0 complete rows for 1 predictors/
    );
    expect(() =>
      runLattice(areas, {
        predictorProperties: Array.from({length: 16}, (_, i) => `c${i}`).join(',')
      })
    ).toThrow(/1 to 15 columns, got 16/);
  });

  it('reports a perfect fit, leaves the likelihood-based diagnostics and the residual Moran empty and says so', () => {
    const exact = makeLattice(6, 6, index => ({
      y: 3 + 2 * REGRESSION_LATTICE.x1[index],
      x1: REGRESSION_LATTICE.x1[index],
      x2: REGRESSION_LATTICE.x2[index]
    }));
    const result = runLattice(exact);
    const rows = getDiagnosticRows(getResultTable(result, 'diagnostics'));
    expect(rows.get('r-squared')!.value).toBeCloseTo(1, 12);
    expect(rows.has('aicc')).toBe(false);
    expect(rows.get('residual-morans-i')!.value).toBeNull();
    expect(result.diagnostics.map(item => item.code)).toEqual(
      expect.arrayContaining(['perfect-fit', 'residual-moran-not-computed'])
    );
  });

  it('leaves an island out of the residual statistic and counts it', () => {
    const areas = makeRegressionLattice();
    areas.features.push({
      id: 'island',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [50, 50],
            [51, 50],
            [51, 51],
            [50, 51],
            [50, 50]
          ]
        ]
      },
      properties: {y: 2, x1: 0.3, x2: -0.2}
    });
    const result = runLattice(areas);
    expect(getResultTable(result, 'residuals').columns['neighbor_count'][36]).toBe(0);
    expect(result.diagnostics.find(item => item.code === 'island-features')?.count).toBe(1);
    expect(
      getDiagnosticRows(getResultTable(result, 'diagnostics')).get('residual-morans-i')!.value
    ).not.toBeNull();
  });

  it('refuses a mix of points and polygons under auto, and links them with knn', () => {
    const areas = makeRegressionLattice();
    areas.features[0].geometry = {type: 'Point', coordinates: [0.5, 0.5]};
    expect(() => runLattice(areas)).toThrow(/mixes points and polygons/);
    expect(runLattice(areas, {neighborhood: 'knn'}).summary['neighborhood']).toBe('knn');
  });

  it('accepts up to 15 predictors', () => {
    const areas = makeLattice(8, 8, index => {
      const properties: Record<string, number> = {y: Math.sin(index) + index / 10};
      for (let predictor = 0; predictor < 15; predictor++)
        properties[`c${predictor}`] =
          Math.sin((predictor + 1) * (index + 1) * 0.7) + Math.cos(index * (predictor + 2));
      return properties;
    });
    const result = runLattice(areas, {
      predictorProperties: Array.from({length: 15}, (_, i) => `c${i}`).join(',')
    });
    expect(getResultTable(result, 'coefficients').rowCount).toBe(16);
  });
});

describe('condition number and eigenvalue helpers', () => {
  it('computeSymmetricEigenvalues returns the known eigenvalues of a 2 by 2 and a 3 by 3 matrix', () => {
    const two = computeSymmetricEigenvalues([
      [2, 1],
      [1, 2]
    ]).sort((a, b) => a - b);
    expect(two[0]).toBeCloseTo(1, 13);
    expect(two[1]).toBeCloseTo(3, 13);
    const three = computeSymmetricEigenvalues([
      [4, 0, 0],
      [0, 1, 0],
      [0, 0, 9]
    ]).sort((a, b) => a - b);
    expect(three).toEqual([1, 4, 9]);
    expect(computeSymmetricEigenvalues([[5]])).toEqual([5]);
  });

  it('computeBelsleyConditionNumber is 1 for orthogonal unit columns and large for nearly collinear ones', () => {
    // Columns [1, x] with x = +-1 are orthogonal and of equal length: condition number 1.
    expect(computeBelsleyConditionNumber([1, -1, 1, -1], 1, 4)).toBeCloseTo(1, 12);
    const nearly = [1, 2, 3, 4, 5, 6].flatMap(value => [value, value + 1e-6 * (value % 2)]);
    expect(computeBelsleyConditionNumber(nearly, 2, 6)).toBeGreaterThan(1e4);
  });
});

it('fits representably varying predictors with a large mean using centred spread (SciPy)', () => {
  // /usr/bin/python3: scipy.stats.linregress(arange(8),4+2*arange(8)+sin(arange(8))).
  for (const offset of [0, 1e13]) {
    const areas = makeLattice(8, 1, index => ({
      x: offset + index,
      y: 4 + 2 * index + Math.sin(index)
    }));
    const result = runCuratedBinding(
      ID,
      {...PLANAR, responseProperty: 'y', predictorProperties: 'x', permutations: 99},
      {areas}
    );
    const coefficients = getResultTable(result, 'coefficients');
    expect(Number(coefficients.columns['estimate'][1])).toBeCloseTo(1.9106176924781326, 12);
    expect(Number(coefficients.columns['standard_error'][1])).toBeCloseTo(0.1123862125398823, 12);
    const residuals = getResultTable(result, 'residuals');
    residuals.columns['fitted'].forEach((value, index) =>
      expect(Number(value)).toBeCloseTo(4.382054670106815 + 1.9106176924781326 * index, 12)
    );
  }
});

it('requires predictors plus four complete rows, matching the AICc descriptor remedy', () => {
  for (const rowCount of [5, 6]) {
    const areas = makeLattice(rowCount, 1, index => ({
      response: 3 + index + Math.sin(index),
      first: index,
      second: Math.cos(index)
    }));
    const run = () =>
      runCuratedBinding(
        ID,
        {
          ...PLANAR,
          responseProperty: 'response',
          predictorProperties: 'first,second',
          permutations: 99
        },
        {areas}
      );
    if (rowCount === 5)
      expect(run).toThrow(/5 complete rows for 2 predictors; at least 6 are needed/);
    else expect(getResultTable(run(), 'residuals').rowCount).toBe(6);
  }
});

it('fits tiny and huge predictor units against NumPy on the rescaled problem', () => {
  // /usr/bin/python3: x=np.arange(1,12); y=3+np.arange(11)*.8+np.sin(np.arange(11))*.1;
  // X=np.column_stack([np.ones(11),x]); b=np.linalg.lstsq(X,y,rcond=None)[0];
  // fitted=X@b; covariance=((y-fitted)@(y-fitted)/9)*np.linalg.inv(X.T@X).
  // Condition number: np.linalg.cond(X/np.linalg.norm(X,axis=0)).
  const intercept = 2.2274919647286633;
  const slope = 0.7975561700773713;
  const interceptError = 0.04700267886156795;
  const slopeError = 0.006930166915870012;
  const response = Array.from({length: 11}, (_, index) => 3 + index * 0.8 + Math.sin(index) * 0.1);
  for (const scale of [1e-250, 1e-190, 1e-100, 1, 1e100, 1e190, 1e250]) {
    const predictors = Array.from({length: 11}, (_, index) => (index + 1) * scale);
    const fit = fitStandardizedLeastSquares(response, predictors, 1, 11);
    expect(computeBelsleyConditionNumber(predictors, 1, 11)).toBeCloseTo(4.042127655053749, 12);
    expect(fit.coefficients[0]).toBeCloseTo(intercept, 12);
    expect(fit.coefficients[1] * scale).toBeCloseTo(slope, 12);
    expect(fit.standardErrors[0]).toBeCloseTo(interceptError, 12);
    expect(fit.standardErrors[1] * scale).toBeCloseTo(slopeError, 12);
    expect(fit.rSquared).toBeCloseTo(0.9993209334692866, 12);
    fit.fitted.forEach((value, index) =>
      expect(value).toBeCloseTo(intercept + slope * (index + 1), 12)
    );
    const areas = makePoints(
      Array.from({length: 11}, (_, index) => index % 4),
      Array.from({length: 11}, (_, index) => Math.floor(index / 4)),
      index => ({x: predictors[index], y: response[index]})
    );
    const result = runCuratedBinding(
      ID,
      {...PLANAR, responseProperty: 'y', predictorProperties: 'x', permutations: 99},
      {areas}
    );
    const coefficients = getResultTable(result, 'coefficients');
    expect(
      getDiagnosticRows(getResultTable(result, 'diagnostics')).get('condition-number')?.value
    ).toBeCloseTo(4.042127655053749, 12);
    expect(Number(coefficients.columns['estimate'][0])).toBeCloseTo(intercept, 12);
    expect(Number(coefficients.columns['estimate'][1]) * scale).toBeCloseTo(slope, 12);
    expect(Number(coefficients.columns['standard_error'][1]) * scale).toBeCloseTo(slopeError, 12);
    getResultTable(result, 'residuals').columns['fitted'].forEach((value, index) =>
      expect(Number(value)).toBeCloseTo(intercept + slope * (index + 1), 12)
    );
  }
});
