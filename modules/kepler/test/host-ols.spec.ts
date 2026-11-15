// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession, type KeplerOLSSettings} from '../src';
import {GPUKeplerTableInspector} from '../src/exploration/gpu-table-inspector';
import {AnalysisRuntime} from '../../../examples/deck/kepler-compute/src/analysis-runtime';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';

const settings: KeplerOLSSettings = {
  responseColumn: 'response',
  predictorColumns: ['predictor'],
  intercept: 'included',
  ridgeLambda: 0,
  variance: 'degrees-of-freedom',
  missing: 'complete-cases'
};

async function createFixture(table: arrow.Table) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const dataset = session.importArrowDataset({id: 'ols-source', table});
  const explorer = session.openExplorer(dataset.id, {
    workingMemoryBudgetBytes: 8 * 1024 * 1024,
    dimensions: [{id: 'row', column: 'row', kind: 'range'}]
  });
  const analysis = new AnalysisRuntime({session, explorer, requestUpdate() {}});
  const inspector = new GPUKeplerTableInspector({
    session,
    explorer,
    sortMemoryBudgetBytes: 8 * 1024 * 1024,
    getDerivedColumns: () => analysis.getTableColumns()
  });
  return {
    session,
    dataset,
    explorer,
    analysis,
    inspector,
    async render() {
      const encoder = device.createCommandEncoder();
      explorer.encode(encoder);
      analysis.encode(encoder);
      device.submit(encoder.finish());
      // A newly encoded fit must not publish before its solver status has been read.
      expect(analysis.active).toBe(false);
      expect(analysis.getTableColumns()).toBeUndefined();
      await analysis.readSummary();
    },
    async destroy() {
      const fence = device.createFence();
      try {
        await inspector.destroy();
        await analysis.destroy(fence.signaled);
        await session.destroy(fence.signaled);
      } finally {
        fence.destroy();
      }
    }
  };
}

test('OLS fits complete source rows, gates failed models, refits selections and reports ridge without inference', async () => {
  const predictors = [0, 1, 2, 3, 4, 5, null, Infinity];
  const response = [1, 3, 5, 7, 9, 11, 999, 999];
  const batch = (start: number, end: number) =>
    new arrow.Table({
      response: arrow.makeVector(new Float32Array(response.slice(start, end))),
      predictor: arrow.vectorFromArray(predictors.slice(start, end), new arrow.Float32()),
      duplicate: arrow.makeVector(
        Float32Array.from(predictors.slice(start, end), value => 2 * (value ?? 0))
      ),
      constant: arrow.makeVector(new Float32Array(end - start).fill(1)),
      row: arrow.makeVector(Float32Array.from({length: end - start}, (_, index) => start + index))
    });
  const context = await createFixture(batch(0, 3).concat(batch(3, 8)));
  const {analysis, inspector, explorer, dataset} = context;
  const sourceChunks = [...dataset.value.columns['predictor'].data];
  try {
    analysis.configureOLS(settings);
    expect(analysis.active).toBe(false);
    const savedRecipe = analysis.serializeRecipe();
    await context.render();
    expect(analysis.error).toBe('');
    expect(analysis.active).toBe(true);
    expect(analysis.summary?.ols).toMatchObject({
      rowCount: 6,
      degreesOfFreedom: 4,
      ridgeLambda: 0,
      inference: 'classical-homoskedastic'
    });
    expect(analysis.summary!.ols!.coefficients.map(coefficient => coefficient.name)).toEqual([
      'Intercept',
      'predictor'
    ]);
    expect(analysis.summary!.ols!.coefficients[0].estimate).toBeCloseTo(1, 5);
    expect(analysis.summary!.ols!.coefficients[1].estimate).toBeCloseTo(2, 5);
    expect(analysis.summary!.ols!.rSquared).toBeCloseTo(1, 6);
    for (let row = 0; row < 6; row++)
      expect((await analysis.readRow(row))?.value).toBeCloseTo(0, 5);
    for (const row of [6, 7]) expect((await analysis.readRow(row))?.value).toBeNaN();
    const snapshot = analysis.getTableColumns()!;
    expect(Object.keys(snapshot.columns)).toEqual([analysis.resultColumn, 'analysis_ols_fitted']);
    for (const vector of Object.values(snapshot.columns))
      expect(vector.data.map(chunk => chunk.length)).toEqual([3, 5]);
    const page = await inspector.readPage({
      offset: 0,
      limit: 8,
      population: 'all',
      columns: [analysis.resultColumn, 'analysis_ols_fitted']
    });
    expect(Array.from(page!.sourceRowIndices)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(Array.from(page!.validity[analysis.resultColumn])).toEqual([1, 1, 1, 1, 1, 1, 0, 0]);
    for (let row = 0; row < 6; row++)
      expect(page!.columns['analysis_ols_fitted'][row]).toBeCloseTo(response[row], 5);
    for (const row of [6, 7]) expect(page!.columns['analysis_ols_fitted'][row]).toBeNaN();

    analysis.setPopulation('selected');
    explorer.setRange('row', 2, 5);
    await context.render();
    expect(analysis.error).toBe('');
    expect(analysis.summary?.ols?.rowCount).toBe(4);
    expect(analysis.summary!.ols!.coefficients[1].estimate).toBeCloseTo(2, 5);
    expect((await analysis.readRow(0))?.value).toBeNaN();
    expect(snapshot.isCurrent()).toBe(false);

    explorer.setRange('row', 0, 1);
    await context.render();
    expect(analysis.error).toContain('more complete observations');
    expect(analysis.summary).toBeNull();
    expect(analysis.active).toBe(false);
    expect(analysis.getTableColumns()).toBeUndefined();
    expect(await analysis.readRow(0)).toBeNull();
    // Change only the filter: the same analysis settings must recover from a failed fit.
    explorer.setRange('row', 0, 5);
    await context.render();
    expect(analysis.error).toBe('');
    expect(analysis.active).toBe(true);
    expect(analysis.summary?.ols?.rowCount).toBe(6);

    analysis.configureOLS({...settings, predictorColumns: ['predictor', 'duplicate']});
    await context.render();
    expect(analysis.error).toContain('singular');
    expect(analysis.getTableColumns()).toBeUndefined();
    // Ridge can rescue nonconstant collinearity, but the core still rejects zero-variance columns.
    analysis.configureOLS({...settings, predictorColumns: ['constant'], ridgeLambda: 6});
    await context.render();
    expect(analysis.error).toContain('singular');
    expect(analysis.active).toBe(false);
    expect(analysis.summary).toBeNull();

    analysis.configureOLS({...settings, ridgeLambda: 6});
    await context.render();
    expect(analysis.error).toBe('');
    const ridge = analysis.summary!.ols!;
    const expectedSlope = 35 / 23.5; // centered X'X=17.5, X'y=35, plus lambda6.
    const expectedIntercept = 6 - 2.5 * expectedSlope;
    expect(ridge.coefficients[0].estimate).toBeCloseTo(expectedIntercept, 5);
    expect(ridge.coefficients[1].estimate).toBeCloseTo(expectedSlope, 5);
    expect(ridge).toMatchObject({
      ridgeLambda: 6,
      inference: 'penalized-descriptive',
      adjustedRSquared: null,
      sigmaSquared: null,
      logLikelihood: null,
      aic: null,
      bic: null,
      tests: null
    });
    for (const coefficient of ridge.coefficients)
      expect(coefficient).toMatchObject({standardError: null, tStatistic: null, pValue: null});
    for (let row = 0; row < 6; row++)
      expect((await analysis.readRow(row))?.value).toBeCloseTo(
        response[row] - expectedIntercept - expectedSlope * row,
        5
      );
    analysis.setMode('off');
    analysis.restoreRecipe(savedRecipe);
    await context.render();
    expect(analysis.error).toBe('');
    expect(analysis.settings.ols).toEqual(settings);
    expect(analysis.settings.population).toBe('all');
    expect(analysis.summary!.ols!.coefficients[1].estimate).toBeCloseTo(2, 5);
    expect(dataset.value.columns['predictor'].data).toEqual(sourceChunks);
    expect(sourceChunks.map(chunk => chunk.length)).toEqual([3, 5]);
  } finally {
    await context.destroy();
  }
  expect(context.session.registry.memoryUsage.totalBytes).toBe(0);
});

test('OLS packs all 15 predictors and applies the last predictor null mask across source batches', async () => {
  const predictorNames = Array.from({length: 15}, (_, index) => `predictor${index + 1}`);
  // Source names must not collide with internal validity resources for another predictor.
  predictorNames[0] = 'validity-predictor15';
  const hadamard = (row: number, column: number): number => {
    let parity = 0;
    for (let bits = row & column; bits; bits >>>= 1) parity ^= bits & 1;
    return parity ? -1 : 1;
  };
  const responses = Array.from(
    {length: 32},
    (_, row) =>
      1 + predictorNames.reduce((sum, _, index) => sum + (index + 1) * hadamard(row, index + 1), 0)
  );
  const batch = (start: number, end: number) => {
    const columns: Record<string, arrow.Vector<arrow.Float32>> = {
      response: arrow.makeVector(new Float32Array(responses.slice(start, end))),
      row: arrow.makeVector(Float32Array.from({length: end - start}, (_, index) => start + index))
    };
    for (const [index, name] of predictorNames.entries()) {
      columns[name] = arrow.vectorFromArray(
        Array.from({length: end - start}, (_, offset) => {
          const row = start + offset;
          return row === 31 && index === 14 ? null : hadamard(row, index + 1);
        }),
        new arrow.Float32()
      );
    }
    return new arrow.Table(columns);
  };
  const context = await createFixture(batch(0, 16).concat(batch(16, 32)));
  try {
    const {analysis, inspector, dataset} = context;
    analysis.configureOLS({...settings, predictorColumns: predictorNames});
    await context.render();
    expect(analysis.error).toBe('');
    expect(analysis.active).toBe(true);
    const model = analysis.summary!.ols!;
    expect(model.rowCount).toBe(31);
    expect(model.degreesOfFreedom).toBe(15);
    expect(model.coefficients).toHaveLength(16);
    expect(model.coefficients[0].estimate).toBeCloseTo(1, 3);
    for (let index = 1; index <= 15; index++)
      expect(model.coefficients[index].estimate).toBeCloseTo(index, 3);
    expect(model.rSquared).toBeCloseTo(1, 6);
    const page = await inspector.readPage({
      offset: 0,
      limit: 32,
      population: 'all',
      columns: [analysis.resultColumn, 'analysis_ols_fitted']
    });
    for (let row = 0; row < 31; row++) {
      expect(page!.validity[analysis.resultColumn][row]).toBe(1);
      expect(page!.columns[analysis.resultColumn][row]).toBeCloseTo(0, 3);
      expect(page!.columns['analysis_ols_fitted'][row]).toBeCloseTo(responses[row], 3);
    }
    expect(page!.validity[analysis.resultColumn][31]).toBe(0);
    expect(page!.columns[analysis.resultColumn][31]).toBeNaN();
    expect(page!.columns['analysis_ols_fitted'][31]).toBeNaN();
    expect(dataset.value.columns['predictor15'].data.map(chunk => chunk.length)).toEqual([16, 16]);
    for (const vector of Object.values(analysis.getTableColumns()!.columns))
      expect(vector.data.map(chunk => chunk.length)).toEqual([16, 16]);
  } finally {
    await context.destroy();
  }
  expect(context.session.registry.memoryUsage.totalBytes).toBe(0);
});
