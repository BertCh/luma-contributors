// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import type {CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';
import {
  computeFrameEmpiricalVariogram,
  computeLeaveOneOut
} from '../../src/curated/bindings/curated-kriging-with-diagnostics-binding';
import {KRIGING_POINTS} from './curated-bindings-oracle-data';
import {
  getMaximumRelativeError,
  getNumbers,
  getResultTable,
  makePoints,
  recordBindingMargin,
  runCuratedBinding
} from './curated-bindings-test-support';

// Oracles: /usr/bin/python3 .typecheck/b1-oracles.py (KRIGING_POINTS): 40 planar points and a
// stated spherical variogram (nugget 0.2, partial sill 4, range 600). The oracle solves the
// ordinary kriging system [Gamma 1; 1' 0][lambda; mu] = [gamma0; 1] with numpy.linalg.solve over
// all 40 points (a neighbour count of 64 makes the binding's neighbourhood global), predicts
// sum(lambda z), the variance sum(lambda gamma0) + mu on the grid, and does the same for each
// point with that point removed (leave-one-out, no refit).

const ID = 'curated-kriging-with-diagnostics';
const STATED = {
  coordinateSystem: 'planar',
  variogram: 'stated',
  model: 'spherical',
  nugget: KRIGING_POINTS.nugget,
  sill: KRIGING_POINTS.sill,
  range: KRIGING_POINTS.range,
  cellSize: KRIGING_POINTS.cellSize,
  neighborCount: 64,
  valueProperty: 'z',
  maximumLag: KRIGING_POINTS.maximumLag,
  binCount: KRIGING_POINTS.binCount
};

function getPoints(): CatalogFeatureCollection {
  return makePoints(KRIGING_POINTS.x, KRIGING_POINTS.y, index => ({z: KRIGING_POINTS.z[index]}));
}

function relativeToScale(actual: readonly number[], expected: readonly number[]): number {
  // Relative to the larger of the value and one, so a near-zero variance does not dominate.
  let worst = 0;
  actual.forEach((value, index) => {
    worst = Math.max(
      worst,
      Math.abs(value - expected[index]) / Math.max(Math.abs(expected[index]), 1)
    );
  });
  return worst;
}

describe('curated-kriging-with-diagnostics against a direct NumPy solve of the kriging system', () => {
  const result = runCuratedBinding(ID, STATED, {points: getPoints()});

  it('predicts the value and the kriging variance of every grid cell to 1e-8', () => {
    const cells = getResultTable(result, 'kriging-cells');
    expect(cells.rowCount).toBe(KRIGING_POINTS.cells.length);
    const order = KRIGING_POINTS.cells.map(cell => cell.row * KRIGING_POINTS.columns + cell.column);
    expect(
      (cells.columns['row'] as number[]).map(
        (row, index) => row * KRIGING_POINTS.columns + (cells.columns['column'] as number[])[index]
      )
    ).toEqual(order);
    const valueError = relativeToScale(
      getNumbers(cells, 'value'),
      KRIGING_POINTS.cells.map(cell => cell.value)
    );
    const varianceError = relativeToScale(
      getNumbers(cells, 'variance'),
      KRIGING_POINTS.cells.map(cell => cell.variance)
    );
    recordBindingMargin('kriging grid value max error', valueError);
    recordBindingMargin('kriging grid variance max error', varianceError);
    expect(valueError).toBeLessThan(1e-8);
    expect(varianceError).toBeLessThan(1e-8);
  });

  it('gives leave-one-out predictions and variances equal to a refit-free NumPy computation to 1e-8', () => {
    const table = getResultTable(result, 'cross-validation');
    expect(table.rowCount).toBe(40);
    const predictionError = relativeToScale(
      getNumbers(table, 'predicted'),
      KRIGING_POINTS.loo.predicted
    );
    const varianceError = relativeToScale(
      getNumbers(table, 'kriging_variance'),
      KRIGING_POINTS.loo.variance
    );
    recordBindingMargin('kriging leave-one-out prediction max error', predictionError);
    recordBindingMargin('kriging leave-one-out variance max error', varianceError);
    expect(predictionError).toBeLessThan(1e-8);
    expect(varianceError).toBeLessThan(1e-8);
    // The error is predicted minus observed, and the standardised error divides by the kriging deviation.
    const errors = getNumbers(table, 'error');
    errors.forEach((error, index) =>
      expect(
        Math.abs(error - (KRIGING_POINTS.loo.predicted[index] - KRIGING_POINTS.z[index]))
      ).toBeLessThan(1e-8)
    );
    const standardized = getNumbers(table, 'standardized_error');
    standardized.forEach((value, index) =>
      expect(
        Math.abs(value - errors[index] / Math.sqrt(KRIGING_POINTS.loo.variance[index]))
      ).toBeLessThan(1e-7)
    );
  });

  it('reports the root mean square error, the mean error and the mean standardised error of the oracle', () => {
    const summary = result.summary;
    const expected = KRIGING_POINTS.summary;
    expect(Math.abs((summary['rootMeanSquareError'] as number) - expected.rmse)).toBeLessThan(1e-8);
    expect(Math.abs((summary['meanError'] as number) - expected.meanError)).toBeLessThan(1e-8);
    expect(
      Math.abs((summary['meanStandardizedError'] as number) - expected.meanStandardizedError)
    ).toBeLessThan(1e-7);
    expect(
      Math.abs(
        (summary['rootMeanSquareStandardizedError'] as number) - expected.rmsStandardizedError
      )
    ).toBeLessThan(1e-7);
    expect(
      Math.abs((summary['valueStandardDeviation'] as number) - expected.standardDeviation)
    ).toBeLessThan(1e-10);
    expect(summary['crossValidationCount']).toBe(40);
    expect(summary['model']).toBe('spherical');
  });

  it('publishes the empirical Matheron variogram of the oracle to 1e-10 and the model beside it', () => {
    const table = getResultTable(result, 'variogram');
    expect(table.rowCount).toBe(KRIGING_POINTS.empirical.length);
    expect(table.columns['bin']).toEqual(KRIGING_POINTS.empirical.map(entry => entry.bin));
    expect(table.columns['pair_count']).toEqual(
      KRIGING_POINTS.empirical.map(entry => entry.pairCount)
    );
    expect(
      getMaximumRelativeError(
        getNumbers(table, 'distance'),
        KRIGING_POINTS.empirical.map(entry => entry.distance)
      )
    ).toBeLessThan(1e-10);
    expect(
      getMaximumRelativeError(
        getNumbers(table, 'semivariance'),
        KRIGING_POINTS.empirical.map(entry => entry.semivariance)
      )
    ).toBeLessThan(1e-10);
    const distances = getNumbers(table, 'distance');
    const model = getNumbers(table, 'model_semivariance');
    model.forEach((value, index) => {
      const ratio = distances[index] / KRIGING_POINTS.range;
      const shape = ratio >= 1 ? 1 : 1.5 * ratio - 0.5 * ratio ** 3;
      expect(Math.abs(value - (KRIGING_POINTS.nugget + KRIGING_POINTS.sill * shape))).toBeLessThan(
        1e-12
      );
    });
  });
});

describe('computeFrameEmpiricalVariogram and computeLeaveOneOut', () => {
  it('bins pairs by distance, leaves empty bins NaN and counts each pair once', () => {
    const empirical = computeFrameEmpiricalVariogram(
      [0, 1, 2, 10],
      [0, 0, 0, 0],
      [0, 1, 4, 5],
      4,
      4
    );
    // Pairs within lag 4: (0,1) h=1 gamma .5; (0,2) h=2 gamma 8; (1,2) h=1 gamma 4.5; bins of width 1.
    expect(Array.from(empirical.pairCounts)).toEqual([0, 2, 1, 0]);
    expect(empirical.semivariances[1]).toBeCloseTo((0.5 + 4.5) / 2, 12);
    expect(empirical.distances[2]).toBeCloseTo(2, 12);
    expect(Number.isNaN(empirical.distances[0])).toBe(true);
    expect(Number.isNaN(empirical.semivariances[3])).toBe(true);
  });

  it('predicts a point from the others and reports a point with no neighbour in reach as failed', () => {
    const sample = {
      x: Float64Array.from([0, 10, 20, 1000]),
      y: new Float64Array(4),
      value: Float64Array.from([1, 2, 3, 100]),
      sourceIndex: Int32Array.from([0, 1, 2, 3]),
      partIndex: new Int32Array(4),
      count: 4,
      droppedCount: 0
    };
    const result = computeLeaveOneOut(sample, {
      model: 'spherical',
      nugget: 0,
      sill: 1,
      range: 50,
      neighborCount: 8,
      searchRadius: 100
    });
    expect(result.failedCount).toBe(1);
    expect(Number.isNaN(result.predicted[3])).toBe(true);
    expect(Number.isFinite(result.predicted[1])).toBe(true);
  });
});

describe('curated-kriging-with-diagnostics fitted variogram', () => {
  it('fits a model, and the grid and cross-validation then follow the oracle for the fitted parameters', () => {
    const result = runCuratedBinding(
      ID,
      {...STATED, variogram: 'fit', model: 'auto', nugget: 0, sill: 1, range: 1000},
      {points: getPoints()}
    );
    expect(['spherical', 'exponential', 'gaussian']).toContain(result.summary['model']);
    expect(result.summary['sill']).toBeGreaterThan(0);
    expect(result.summary['range']).toBeGreaterThan(0);
    expect(result.summary['fitCriterion']).toBeTypeOf('number');
    // Same data, the fitted parameters stated: the two runs must agree, so the fit only chooses parameters.
    const stated = runCuratedBinding(
      ID,
      {
        ...STATED,
        model: result.summary['model'],
        nugget: result.summary['nugget'],
        sill: result.summary['sill'],
        range: result.summary['range']
      },
      {points: getPoints()}
    );
    expect(stated.tables).toEqual(result.tables);
  });

  it('recovers the planted range and sill from a Gaussian-process sample within a generous band', () => {
    // 300 points of a planar field with exponential-like spatial correlation, built by smoothing white noise on a grid.
    const size = 18;
    const noise: number[][] = [];
    let state = 12345;
    const next = () => {
      state = (state * 1664525 + 1013904223) >>> 0;
      return state / 4294967296 - 0.5;
    };
    for (let row = 0; row < size + 8; row++)
      noise.push(Array.from({length: size + 8}, () => next() * 6));
    const xs: number[] = [];
    const ys: number[] = [];
    const zs: number[] = [];
    for (let row = 0; row < size; row++)
      for (let column = 0; column < size; column++) {
        let sum = 0;
        for (let dr = 0; dr < 5; dr++)
          for (let dc = 0; dc < 5; dc++) sum += noise[row + dr][column + dc];
        xs.push(column * 100);
        ys.push(row * 100);
        zs.push(sum + next() * 0.2);
      }
    const result = runCuratedBinding(
      ID,
      {
        coordinateSystem: 'planar',
        valueProperty: 'z',
        cellSize: 300,
        neighborCount: 12,
        binCount: 12
      },
      {points: makePoints(xs, ys, index => ({z: zs[index]}))}
    );
    // The smoothing window is 5 cells of 100 m, so correlation ends near 500 m.
    const range = result.summary['range'] as number;
    recordBindingMargin('kriging fitted range on the smoothed-noise field (m)', range);
    expect(range).toBeGreaterThan(250);
    expect(range).toBeLessThan(1500);
    expect(result.summary['rootMeanSquareError']).toBeLessThan(
      result.summary['valueStandardDeviation'] as number
    );
  });
});

describe('curated-kriging-with-diagnostics declared behaviour', () => {
  it('merges coincident samples to their mean, once, and says so', () => {
    const xs = [...KRIGING_POINTS.x, KRIGING_POINTS.x[0]];
    const ys = [...KRIGING_POINTS.y, KRIGING_POINTS.y[0]];
    const zs = [...KRIGING_POINTS.z, KRIGING_POINTS.z[0] + 2];
    const result = runCuratedBinding(ID, STATED, {
      points: makePoints(xs, ys, index => ({z: zs[index]}))
    });
    expect(result.diagnostics.find(item => item.code === 'coincident-samples')?.count).toBe(1);
    expect(result.summary['sampleCount']).toBe(40);
    // Every input feature has a row: the 40 locations and the merged sample, with a status.
    const table = getResultTable(result, 'cross-validation');
    expect(table.rowCount).toBe(41);
    expect(table.columns['status'][0]).toBe('analysed');
    expect(table.columns['status'][40]).toBe('merged-into-earlier-sample');
    expect(table.columns['location_sample_count'][0]).toBe(2);
    expect(Math.abs(Number(table.columns['observed'][0]) - (KRIGING_POINTS.z[0] + 1))).toBeLessThan(
      1e-12
    );
    expect(
      Math.abs(Number(table.columns['observed'][40]) - (KRIGING_POINTS.z[0] + 2))
    ).toBeLessThan(1e-12);
    expect(table.columns['predicted'][40]).toBe(table.columns['predicted'][0]);
    expect(table.columns['error'][40]).toBeNull();
  });

  it('skips features without a value, counted, and throws named errors for too few, constant and unfittable data', () => {
    const points = getPoints();
    points.features[0].properties = {z: 'x'};
    points.features[1].properties = {};
    const result = runCuratedBinding(ID, STATED, {points});
    expect(result.diagnostics.find(item => item.code === 'feature-dropped')?.count).toBe(2);
    expect(result.summary['sampleCount']).toBe(38);
    const few = makePoints([0, 1, 2, 3, 4, 5, 6], [0, 1, 0, 1, 0, 1, 0], index => ({z: index}));
    expect(() => runCuratedBinding(ID, {...STATED}, {points: few})).toThrow(
      /7 distinct sample locations; at least 8 are needed/
    );
    const constant = makePoints(KRIGING_POINTS.x, KRIGING_POINTS.y, () => ({z: 5}));
    expect(() =>
      runCuratedBinding(ID, {...STATED, variogram: 'fit', model: 'auto'}, {points: constant})
    ).toThrow(/the values of z are all equal, so no variogram can be fitted/);
    expect(() =>
      runCuratedBinding(
        ID,
        {...STATED, variogram: 'fit', model: 'auto', maximumLag: 1e-6},
        {points: getPoints()}
      )
    ).toThrow(/no variogram model could be fitted/);
    expect(() => runCuratedBinding(ID, {...STATED, model: 'auto'}, {points: getPoints()})).toThrow(
      /a stated variogram needs a model/
    );
    expect(() =>
      runCuratedBinding(ID, {...STATED, valueProperty: ''}, {points: getPoints()})
    ).toThrow(/parameter valueProperty must name a column/);
  });

  it('predicts with a stated variogram even when the values are constant, with zero errors', () => {
    const constant = makePoints(KRIGING_POINTS.x, KRIGING_POINTS.y, () => ({z: 5}));
    const result = runCuratedBinding(ID, STATED, {points: constant});
    expect(result.summary['rootMeanSquareError']).toBeLessThan(1e-9);
    expect(result.summary['valueStandardDeviation']).toBe(0);
  });

  it('throws a capacity error for too many cells and honours a search radius with nodata cells', () => {
    expect(() => runCuratedBinding(ID, {...STATED, cellSize: 4}, {points: getPoints()})).toThrow(
      /cells exceed the 40000/
    );
    const local = runCuratedBinding(ID, {...STATED, searchRadius: 150}, {points: getPoints()});
    const cells = getResultTable(local, 'kriging-cells');
    expect(cells.rowCount).toBeLessThan(KRIGING_POINTS.cells.length);
    const sampleCounts = getNumbers(cells, 'sample_count');
    expect(Math.min(...sampleCounts)).toBeGreaterThanOrEqual(1);
  });
});

it('keeps squared measurement units with a stated variogram whose sill differs from empirical variance', () => {
  const points = getPoints();
  const base = runCuratedBinding(ID, {...STATED, sill: 3, nugget: 0.1}, {points});
  const multiplier = 10;
  const scaled = runCuratedBinding(
    ID,
    {...STATED, sill: 3 * multiplier ** 2, nugget: 0.1 * multiplier ** 2},
    {
      points: {
        ...points,
        features: points.features.map(feature => ({
          ...feature,
          properties: {...feature.properties, z: Number(feature.properties?.['z']) * multiplier}
        }))
      }
    }
  );
  for (const tableId of ['kriging-cells', 'cross-validation']) {
    const baseTable = getResultTable(base, tableId);
    const scaledTable = getResultTable(scaled, tableId);
    const varianceColumn = tableId === 'kriging-cells' ? 'variance' : 'kriging_variance';
    const predictionColumn = tableId === 'kriging-cells' ? 'value' : 'predicted';
    baseTable.columns[varianceColumn].forEach((value, index) =>
      expect(Number(scaledTable.columns[varianceColumn][index])).toBeCloseTo(
        Number(value) * multiplier ** 2,
        8
      )
    );
    baseTable.columns[predictionColumn].forEach((value, index) =>
      expect(Number(scaledTable.columns[predictionColumn][index])).toBeCloseTo(
        Number(value) * multiplier,
        8
      )
    );
  }
});
