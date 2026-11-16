// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it, vi} from 'vitest';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '@luma-contributors/analysis/testing';
import {CatalogGpuSession} from '../../src/analysis/catalog-gpu-session';
import {getKeplerCuratedAnalysis} from '../../src/curated/curated-analyses';
import {createKeplerCuratedAnalysisSample} from '../../src/curated/curated-analysis-samples';
import {
  createKeplerCuratedAnalysisMapping,
  updateKeplerCuratedAnalysisMapping,
  buildKeplerCuratedAnalysisRequest
} from '../../src/curated/curated-analysis-mapping';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import type {
  CatalogFeatureCollection,
  CatalogOutputTable
} from '../../src/analysis/catalog-geojson';
import {
  CURATED_REGRESSION_WITH_DIAGNOSTICS_BINDING,
  prepareCuratedRegression,
  publishCuratedRegression,
  computeBelsleyConditionNumber,
  fitStandardizedLeastSquares,
  computeSymmetricEigenvalues
} from '../../src/curated/bindings/curated-regression-with-diagnostics-binding';
import {
  normalizeCatalogParameters,
  runCatalogOperationAsync
} from '../../src/analysis/catalog-bindings';
import {
  prepareCuratedRegressionGpu,
  publishCuratedRegressionGpu,
  type CuratedRegressionGpuReadback
} from '../../src/curated/bindings/curated-regression-with-diagnostics-gpu';
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

// Injected readbacks qualify preparation/publication separately from external WebGPU dispatch.
describe('curated regression GPU adapter', () => {
  function prepareReadback(
    areas = makeRegressionLattice(),
    responseProperty = 'y',
    predictorProperties = 'x1,x2'
  ) {
    const prepared = prepareCuratedRegression(
      {areas},
      normalizeCatalogParameters(CURATED_REGRESSION_WITH_DIAGNOSTICS_BINDING, {
        ...PLANAR,
        responseProperty,
        predictorProperties,
        permutations: 99
      })
    );
    const scaled = prepareCuratedRegressionGpu(prepared);
    const fit = prepared.referenceFit;
    const slopes = fit.coefficients
      .slice(1)
      .map((value, column) => (value * scaled.columns[column].scale) / scaled.response.scale);
    const intercept =
      (fit.coefficients[0] +
        scaled.columns.reduce(
          (sum, column, index) => sum + column.origin * fit.coefficients[index + 1],
          0
        )) /
      scaled.response.scale;
    const readback: CuratedRegressionGpuReadback = {
      coefficients: Float32Array.from([intercept, ...slopes]),
      fitted: Float32Array.from(fit.fitted, value => value / scaled.response.scale),
      residuals: Float32Array.from(fit.residuals, value => value / scaled.response.scale),
      olsStatus: 0,
      diagnosticsStatus: 0,
      decision: 0,
      tests: Float32Array.from({length: 18}, (_, index) => (index % 3 === 2 ? 0.5 : 1)),
      diagnosticsSummary: new Float32Array(13),
      localI: Float32Array.from({length: prepared.rowCount}, (_, index) => index / 100),
      localPValues: new Float32Array(prepared.rowCount).fill(0.5)
    };
    return {prepared, scaled, readback};
  }

  function makeLargeResponseOffset(noiseDeviation: number) {
    const areas = makePoints(
      Array.from({length: 36}, (_, row) => row % 6),
      Array.from({length: 36}, (_, row) => Math.floor(row / 6)),
      row => ({x: row, y: 100000000 + 2 * row + noiseDeviation * Math.sin(row)})
    );
    const parameters = normalizeCatalogParameters(CURATED_REGRESSION_WITH_DIAGNOSTICS_BINDING, {
      ...PLANAR,
      responseProperty: 'y',
      predictorProperties: 'x',
      permutations: 99
    });
    return {areas, parameters, prepared: prepareCuratedRegression({areas}, parameters)};
  }

  it.each([
    0, 0.1
  ])('falls back before allocation for a large response offset (noise %s)', async noiseDeviation => {
    const {areas, parameters, prepared} = makeLargeResponseOffset(noiseDeviation);
    const device = createNullWebGPUDevice();
    const allocated = vi.spyOn(device, 'createBuffer');
    try {
      if (noiseDeviation === 0) expect(prepared.referenceFit.residualSumOfSquares).toBe(0);
      else expect(prepared.referenceFit.residualSumOfSquares).toBeGreaterThan(0);
      const result = await runCatalogOperationAsync(
        CURATED_REGRESSION_WITH_DIAGNOSTICS_BINDING,
        parameters,
        {areas},
        {device}
      );
      expect(result.engine).toBe('cpu-reference');
      expect(result.dispatchedContributors).toEqual([]);
      expect(result.fallbackReason).toMatch(
        noiseDeviation === 0 ? /CPU perfect-fit cutoff/ : /upload rounding error/
      );
      expect(allocated).not.toHaveBeenCalled();
      const diagnostics = getDiagnosticRows(getResultTable(result, 'diagnostics'));
      expect(diagnostics.has('lm-lag')).toBe(false);
      if (noiseDeviation === 0) {
        expect(diagnostics.has('aic')).toBe(false);
        expect(diagnostics.has('jarque-bera')).toBe(false);
        expect(result.summary['residualMoransI']).toBe('not computed');
        expect(result.diagnostics.map(diagnostic => diagnostic.code)).toContain('perfect-fit');
      } else {
        expect(diagnostics.has('aic')).toBe(true);
      }
    } finally {
      allocated.mockRestore();
      device.destroy();
    }
  });

  it('rejects uploaded-fit readbacks that turn a reference-perfect large-offset fit into apparent residual noise', () => {
    const {prepared} = makeLargeResponseOffset(0);
    const responseScale = Math.max(...prepared.responseValues);
    const origin = prepared.predictorArray[0];
    const predictorScale = Math.max(
      ...Array.from(prepared.predictorArray, value => Math.abs(value - origin))
    );
    const predictors = Float32Array.from(
      prepared.predictorArray,
      value => (value - origin) / predictorScale
    );
    const response = Float32Array.from(prepared.responseValues, value => value / responseScale);
    const uploadedFit = fitStandardizedLeastSquares(response, predictors, 1, 36);
    // Reproduces the reviewer's failure entirely on Node: response quantization creates ~94.6 RSS.
    expect(uploadedFit.residualSumOfSquares * responseScale ** 2).toBeGreaterThan(90);
    expect(prepared.referenceFit.residualSumOfSquares).toBe(0);
    const {scaled: ordinaryScaled, readback: ordinaryReadback} = prepareReadback();
    const scaled = {
      ...ordinaryScaled,
      predictors,
      response: {origin: 0, scale: responseScale, values: response},
      columns: [{origin, scale: predictorScale, values: predictors}]
    };
    const readback = {
      ...ordinaryReadback,
      coefficients: Float32Array.from(uploadedFit.coefficients),
      fitted: Float32Array.from(uploadedFit.fitted),
      residuals: Float32Array.from(uploadedFit.residuals)
    };
    expect(() => publishCuratedRegressionGpu(prepared, scaled, readback)).toThrow(
      /CPU perfect-fit cutoff/
    );
  });

  it('rejects response quantization that overwhelms a nonzero reference residual', () => {
    const {prepared} = makeLargeResponseOffset(0.1);
    const responseScale = Math.max(...prepared.responseValues);
    const predictors = Float32Array.from(prepared.predictorArray, value => value / 35);
    const response = Float32Array.from(prepared.responseValues, value => value / responseScale);
    const uploadedFit = fitStandardizedLeastSquares(response, predictors, 1, 36);
    expect(uploadedFit.residualSumOfSquares * responseScale ** 2).toBeGreaterThan(
      100 * prepared.referenceFit.residualSumOfSquares
    );
    expect(() => prepareCuratedRegressionGpu(prepared)).toThrow(/upload rounding error/);
  });

  it('rejects inference that becomes nonfinite while decoding finite GPU values', () => {
    const {prepared, scaled, readback} = prepareReadback();
    prepared.referenceFit.standardErrors[0] = Number.MAX_VALUE;
    readback.residuals = Float32Array.from(readback.residuals, value => value * 2);
    expect(() => publishCuratedRegressionGpu(prepared, scaled, readback)).toThrow(
      /decoded fit inference must have finite/
    );
    const ordinary = prepareReadback();
    expect(() =>
      publishCuratedRegressionGpu(
        ordinary.prepared,
        {
          ...ordinary.scaled,
          response: {...ordinary.scaled.response, scale: Number.MAX_VALUE}
        },
        ordinary.readback
      )
    ).toThrow(/decoded fit inference must have finite/);
  });

  it('rejects out-of-range diagnostic p-values before publishing inference', () => {
    for (const target of ['tests', 'localPValues', 'diagnosticsSummary'] as const) {
      const {prepared, scaled, readback} = prepareReadback();
      const index = target === 'tests' ? 2 : target === 'diagnosticsSummary' ? 9 : 0;
      readback[target][index] = 1.1;
      expect(() => publishCuratedRegressionGpu(prepared, scaled, readback)).toThrow(
        /p-values must be in/
      );
    }
  });

  it('publishes missing source IDs as null for the 144-row sample on CPU and GPU readbacks', () => {
    const areas = createKeplerCuratedAnalysisSample('district-regression')['areas'];
    expect(areas.features).toHaveLength(144);
    expect(areas.features.every(feature => feature.id === undefined)).toBe(true);
    const {prepared, scaled, readback} = prepareReadback(areas, 'outcome', 'income,density');
    const expectedSourceIds = Array<null>(144).fill(null);
    const expectedSourceIndices = Array.from({length: 144}, (_, index) => index);
    for (const result of [
      publishCuratedRegression(prepared, prepared.referenceFit),
      publishCuratedRegressionGpu(prepared, scaled, readback)
    ]) {
      const residuals = result.tables.find(table => table.id === 'residuals')!;
      expect(residuals.columns['source_id']).toEqual(expectedSourceIds);
      expect(residuals.columns['source_index']).toEqual(expectedSourceIndices);
    }
  });

  it('preserves string IDs and stringifies numeric IDs consistently for CPU and GPU readbacks', () => {
    const areas = makeRegressionLattice();
    delete areas.features[0].id;
    areas.features[1].id = null;
    areas.features[2].id = 42;
    areas.features[3].id = 0;
    areas.features[4].id = '';
    areas.features[5].id = 'district-5';
    const {prepared, scaled, readback} = prepareReadback(areas);
    for (const result of [
      publishCuratedRegression(prepared, prepared.referenceFit),
      publishCuratedRegressionGpu(prepared, scaled, readback)
    ]) {
      const residuals = result.tables.find(table => table.id === 'residuals')!;
      expect(residuals.columns['source_id'].slice(0, 6)).toEqual([
        null,
        null,
        '42',
        '0',
        '',
        'district-5'
      ]);
      expect(residuals.columns['source_index']).toEqual(
        Array.from({length: 36}, (_, index) => index)
      );
    }
  });

  it('keeps the CPU oracle and registers an actual optional GPU path', () => {
    expect(CURATED_REGRESSION_WITH_DIAGNOSTICS_BINDING.engine).toBe('gpu-or-cpu');
    expect(CURATED_REGRESSION_WITH_DIAGNOSTICS_BINDING.runGpu).toBeTypeOf('function');
    expect(CURATED_REGRESSION_WITH_DIAGNOSTICS_BINDING.gpuContributorNames).toEqual([
      'GPUOrdinaryLeastSquares',
      'GPUSpatialRegressionDiagnostics',
      'GPULocalMoran'
    ]);
  });

  it('converts canonical complete-case weights to CSR without losing source identities', () => {
    const {prepared, scaled} = prepareReadback();
    expect(scaled.weights.offsets.length).toBe(37);
    expect(scaled.weights.neighbors.length).toBe(scaled.weights.weights.length);
    prepared.set.sourceIndex.forEach((row, position) => {
      const start = scaled.weights.offsets[row];
      const end = scaled.weights.offsets[row + 1];
      expect(Array.from(scaled.weights.neighbors.slice(start, end))).toEqual(
        prepared.set.weights.neighbors[position].map(other => prepared.set.sourceIndex[other])
      );
      expect(
        Array.from(scaled.weights.weights.slice(start, end)).reduce((sum, value) => sum + value, 0)
      ).toBeCloseTo(1, 6);
    });
  });

  it('publishes GPU-supplied fitted values, coefficients and local values in input order', () => {
    const {prepared, scaled, readback} = prepareReadback();
    const result = publishCuratedRegressionGpu(prepared, scaled, readback);
    const coefficients = result.tables.find(table => table.id === 'coefficients')!;
    REGRESSION_LATTICE.coefficients.forEach((value, index) =>
      expect(Number(coefficients.columns['estimate'][index])).toBeCloseTo(value, 5)
    );
    const residuals = result.tables.find(table => table.id === 'residuals')!;
    prepared.rows.forEach((row, index) => {
      expect(residuals.columns['local_morans_i'][row.sourceIndex]).toBe(readback.localI[index]);
      expect(residuals.columns['fitted'][row.sourceIndex]).toBe(
        readback.fitted[index] * scaled.response.scale
      );
    });
    // A changed readback must change the published fit: a hidden CPU-only publication fails this.
    readback.coefficients[1] += 0.1;
    const changed = publishCuratedRegressionGpu(prepared, scaled, readback);
    expect(changed.tables[0].columns['estimate'][1]).not.toBe(coefficients.columns['estimate'][1]);
    expect(result.tables[1].columns['diagnostic']).toEqual(
      expect.arrayContaining([
        'lm-lag',
        'lm-error',
        'robust-lm-lag',
        'robust-lm-error',
        'lm-sarma',
        'residual-morans-i-analytic',
        'specification-decision'
      ])
    );
    expect(result.summary['localMoranInference']).toContain('uncorrected');
  });

  it('refuses failed, nonfinite, truncated and rounding-floor GPU results before publication', () => {
    const {prepared, scaled, readback} = prepareReadback();
    for (const olsStatus of [1, 2])
      expect(() => publishCuratedRegressionGpu(prepared, scaled, {...readback, olsStatus})).toThrow(
        /ols status code must be 0/
      );
    for (const diagnosticsStatus of [1, 2, 3, 4])
      expect(() =>
        publishCuratedRegressionGpu(prepared, scaled, {...readback, diagnosticsStatus})
      ).toThrow(/spatial diagnostics status code must be 0/);
    expect(() => publishCuratedRegressionGpu(prepared, scaled, {...readback, decision: 4})).toThrow(
      /specification decision must be/
    );
    expect(() =>
      publishCuratedRegressionGpu(prepared, scaled, {...readback, fitted: new Float32Array(1)})
    ).toThrow(/readback lengths/);
    expect(() =>
      publishCuratedRegressionGpu(prepared, scaled, {
        ...readback,
        residuals: new Float32Array(prepared.rowCount)
      })
    ).toThrow(/float32 rounding floor/);
    readback.coefficients[0] = Number.NaN;
    expect(() => publishCuratedRegressionGpu(prepared, scaled, readback)).toThrow(/must be finite/);
  });

  it('preserves large-offset/tiny-unit predictor variation before upload and never translates the response', () => {
    for (const scale of [1e-250, 1, 1e250]) {
      const areas = makeLattice(8, 1, index => ({
        x: (index + 1) * scale,
        y: 4 + index * 2 + Math.sin(index)
      }));
      const prepared = prepareCuratedRegression(
        {areas},
        normalizeCatalogParameters(CURATED_REGRESSION_WITH_DIAGNOSTICS_BINDING, {
          ...PLANAR,
          responseProperty: 'y',
          predictorProperties: 'x'
        })
      );
      const scaled = prepareCuratedRegressionGpu(prepared);
      expect(new Set(scaled.predictors).size).toBe(8);
      expect(Array.from(scaled.predictors).every(Number.isFinite)).toBe(true);
      expect(scaled.response.origin).toBe(0);
    }
    const areas = makeLattice(8, 1, index => ({
      x: 1e13 + index,
      y: 4 + index * 2 + Math.sin(index)
    }));
    const prepared = prepareCuratedRegression(
      {areas},
      normalizeCatalogParameters(CURATED_REGRESSION_WITH_DIAGNOSTICS_BINDING, {
        ...PLANAR,
        responseProperty: 'y',
        predictorProperties: 'x'
      })
    );
    expect(new Set(prepareCuratedRegressionGpu(prepared).predictors).size).toBe(8);
  });
});

it('constructs real recipe kernels and releases the session and parameter buffer on submission failure', async () => {
  const device = createNullWebGPUDevice();
  const sessionDestroyed = vi.spyOn(CatalogGpuSession.prototype, 'dispose');
  const parametersDestroyed = vi.spyOn(GPUParameterBuffer.prototype, 'destroy');
  const submitted = vi.spyOn(CatalogGpuSession.prototype, 'run').mockImplementation(function (
    this: CatalogGpuSession
  ) {
    const kernels = getGPUCommandNodeKernelSources(
      (this.graph as unknown as {nodes: unknown[]}).nodes
    );
    for (const prefix of [
      'spatial-regression-ols',
      'spatial-regression-diagnostics',
      'spatial-regression-residual-moran',
      'spatial-regression-specification'
    ])
      expect(
        kernels.some(kernel => kernel.id.startsWith(prefix)),
        prefix
      ).toBe(true);
    throw new Error('regression construction reached submission');
  });
  try {
    await expect(
      CURATED_REGRESSION_WITH_DIAGNOSTICS_BINDING.runGpu!(
        {device},
        {areas: makeRegressionLattice()},
        normalizeCatalogParameters(CURATED_REGRESSION_WITH_DIAGNOSTICS_BINDING, {
          ...PLANAR,
          responseProperty: 'y',
          predictorProperties: 'x1,x2',
          permutations: 99
        })
      )
    ).rejects.toThrow('regression construction reached submission');
    expect(submitted).toHaveBeenCalledOnce();
    expect(sessionDestroyed).toHaveBeenCalledOnce();
    expect(parametersDestroyed).toHaveBeenCalledOnce();
    expect(sessionDestroyed.mock.invocationCallOrder[0]).toBeLessThan(
      parametersDestroyed.mock.invocationCallOrder[0]
    );
  } finally {
    submitted.mockRestore();
    sessionDestroyed.mockRestore();
    parametersDestroyed.mockRestore();
    device.destroy();
  }
});

it('maps the sample and a second dataset with independently named columns through the existing AX mapping model', () => {
  const analysis = getKeplerCuratedAnalysis('regression-with-diagnostics')!;
  for (const [datasetId, areas, response, predictors] of [
    [
      'sample',
      createKeplerCuratedAnalysisSample('district-regression')['areas'],
      'outcome',
      ['income', 'density']
    ],
    [
      'second',
      makeLattice(6, 6, row => ({
        measured: REGRESSION_LATTICE.y[row],
        firstDriver: REGRESSION_LATTICE.x1[row],
        secondDriver: REGRESSION_LATTICE.x2[row]
      })),
      'measured',
      ['firstDriver', 'secondDriver']
    ]
  ] as const) {
    const profiles = [
      profileKeplerFeatureCollection(areas, {
        datasetId,
        label: datasetId,
        coordinateSystem: 'planar',
        declarations: {coordinateModel: 'planar', rowSemantics: 'objects'}
      })
    ];
    let mapping = createKeplerCuratedAnalysisMapping(analysis, profiles, {
      parameters: {...PLANAR, permutations: 99}
    });
    mapping = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: 'areas', datasetId},
      analysis,
      profiles
    );
    mapping = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: 'areas', role: 'response', columns: [response]},
      analysis,
      profiles
    );
    mapping = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: 'areas', role: 'predictors', columns: [...predictors]},
      analysis,
      profiles
    );
    expect(mapping.ready, JSON.stringify(mapping.problems)).toBe(true);
    const request = buildKeplerCuratedAnalysisRequest(analysis, mapping);
    expect(request.inputDatasetIds).toEqual({areas: datasetId});
    expect(request.parameters['responseProperty']).toBe(response);
    expect(request.parameters['predictorProperties']).toBe(predictors.join(','));
    const prepared = prepareCuratedRegression(
      {areas},
      normalizeCatalogParameters(CURATED_REGRESSION_WITH_DIAGNOSTICS_BINDING, request.parameters)
    );
    expect(prepareCuratedRegressionGpu(prepared).predictors.length).toBe(areas.features.length * 2);
  }
});
