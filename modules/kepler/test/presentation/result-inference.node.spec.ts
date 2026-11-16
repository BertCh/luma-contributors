// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {createElement} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {AnalysisResultCard} from '../../../../examples/deck/kepler-compute/src/analysis-result-card';
import {createAnalysisResultCardModel} from '../../../../examples/deck/kepler-compute/src/analysis-result-card-model';
import {createReviewCard} from '../ax-result-card-fixtures';
import {
  applyKeplerResultInference,
  getKeplerResultInferenceSettings,
  hasKeplerResultInference
} from '../../src/presentation/index';
import {getCatalogBinding, runCatalogOperationAsync} from '../../src/analysis/catalog-bindings';
import {createKeplerCuratedAnalysisSample} from '../../src/curated/curated-analysis-samples';
import {getKeplerOperationDescriptor} from '../../src/descriptors/index';
import type {CatalogRunResult} from '../../src/analysis/catalog-binding-types';
import type {CatalogOutputTable} from '../../src/analysis/catalog-geojson';

function makeResult(hot = false): CatalogRunResult {
  const operationId = hot ? 'recipe-hot-spot-lattice' : 'local-moran-statistics';
  const table: CatalogOutputTable = {
    id: hot ? 'recipe-hot-spots' : 'local-moran',
    label: 'Retained inference',
    rowCount: 6,
    columns: {
      value: [-3, -2, -1, 1, 2, 3],
      spatial_lag: [-2, 1, -1, 1, -1, 1],
      z_score: [-2, 2, -1, 1, -1, 1],
      p_value: [0.001, 0.01, 0.04, 0.2, null, 0],
      adjusted_p_value: [0.004, 0.02, (0.04 * 4) / 3, 0.2, null, null],
      neighbor_count: [2, 2, 2, 2, 2, 0],
      cluster: ['LL', 'LH', 'not-significant', 'not-significant', 'not-tested', 'not-tested'],
      bin: [-3, 2, 0, 0, null, null],
      cell_index: [0, 1, 2, 3, 4, 5]
    },
    columnTypes: {
      value: 'float64',
      spatial_lag: 'float64',
      z_score: 'float64',
      p_value: 'float64',
      adjusted_p_value: 'float64',
      neighbor_count: 'int32',
      cluster: 'string',
      bin: 'int32',
      cell_index: 'int32'
    }
  };
  return {
    operationId,
    parameters: {significance: 0.05, multipleTesting: 'benjamini-hochberg', permutations: 999},
    tables: [table],
    summary: {featureCount: 6, columns: 3, rows: 2, testedCellCount: 6},
    diagnostics: [],
    engine: 'cpu-reference',
    dispatchedContributors: []
  };
}

function apply(
  result: CatalogRunResult,
  significance = 0.05,
  multipleTesting: 'none' | 'benjamini-hochberg' | 'bonferroni' = 'benjamini-hochberg',
  scope: 'full' | 'sample' | 'viewport' = 'full'
) {
  return applyKeplerResultInference(
    result,
    getCatalogBinding(result.operationId)!,
    getKeplerOperationDescriptor(result.operationId)!,
    {significance, multipleTesting},
    {},
    scope
  );
}

describe('retained result inference', () => {
  test('a truncated 2 by 2 lattice cannot shrink the BH family from four cells to one', () => {
    const result = makeResult(true);
    const table = result.tables[0];
    table.rowCount = 4;
    table.columns = Object.fromEntries(
      Object.entries(table.columns).map(([name, values]) => [name, values.slice(0, 4)])
    );
    table.columns['p_value'] = [0.01, 0.2, 0.3, 0.4];
    result.summary = {columns: 2, rows: 2, testedCellCount: 4};
    const complete = apply(result, 0.01);
    if (complete.status !== 'display') throw new Error(complete.reason);
    expect(complete.displayResult.tables[0].columns['adjusted_p_value'][0]).toBe(0.04);
    expect(complete.displayResult.tables[0].columns['bin'][0]).toBe(0);
    table.rowCount = 1;
    table.columns = Object.fromEntries(
      Object.entries(table.columns).map(([name, values]) => [name, values.slice(0, 1)])
    );
    expect(apply(result, 0.01)).toMatchObject({
      status: 'rerun-required',
      reason: expect.stringContaining('complete original lattice population')
    });
  });

  test('lattice dimensions, cell identities and original tested counts must all be retained', () => {
    for (const mutation of [
      (result: CatalogRunResult) => {
        delete result.summary['columns'];
      },
      (result: CatalogRunResult) => {
        result.summary['testedCellCount'] = 5;
      },
      (result: CatalogRunResult) => {
        result.tables[0].columns['cell_index'] = [0, 1, 2, 3, 4, 4];
      },
      (result: CatalogRunResult) => {
        result.tables[0].columns['cell_index'] = [0, 1, 2, 3, 4, 6];
      }
    ]) {
      const result = makeResult(true);
      mutation(result);
      expect(apply(result)).toMatchObject({
        status: 'rerun-required',
        reason: expect.stringContaining('complete original lattice population')
      });
    }
    const reordered = makeResult(true);
    reordered.tables[0].columns = Object.fromEntries(
      Object.entries(reordered.tables[0].columns).map(([name, values]) => [
        name,
        values.slice().reverse()
      ])
    );
    expect(apply(reordered).status).toBe('display');
  });

  test('sample and viewport global tests suppress inference, disable Keep and offer full-population rerun', async () => {
    const binding = getCatalogBinding('global-spatial-statistics')!;
    const areas = createKeplerCuratedAnalysisSample('city-district-rates')['areas'];
    const result = await runCatalogOperationAsync(
      binding,
      {valueProperty: 'rate', neighborhood: 'queen', coordinateSystem: 'EPSG:4326'},
      {features: areas},
      {engine: 'cpu'}
    );
    const model = createAnalysisResultCardModel(result);
    expect(model.report.outputs.some(output => output.usedTemplate === 'global-statistic')).toBe(
      true
    );
    expect(model.presentations.every(presentation => !presentation.presentation?.classScheme)).toBe(
      true
    );
    const callbacks = {
      onNext: () => {},
      onAddToStack: () => {},
      onSaveWorkflow: () => {},
      onInferenceRerun: () => {}
    };
    const full = renderToStaticMarkup(createElement(AnalysisResultCard, {model, ...callbacks}));
    expect(full).toContain('p-value');
    expect(full).toMatch(/<button type="button">Add to stack<\/button>/);
    for (const scope of ['sample', 'viewport'] as const) {
      const markup = renderToStaticMarkup(
        createElement(AnalysisResultCard, {model, ...callbacks, inferenceScope: scope})
      );
      expect(markup).toContain(`Significance is hidden for this ${scope} preview.`);
      expect(markup).not.toContain('would be rare under a random pattern');
      expect(markup).toMatch(/<button type="button" disabled="">Add to stack<\/button>/);
      expect(markup).toMatch(/<button type="button" disabled="">Save as workflow<\/button>/);
      expect(markup).toContain('Rerun required: Recompute on the full population');
      expect(markup).toMatch(/<button type="button">Rerun analysis<\/button>/);
    }
  });

  test('a non-inferential density preview keeps its report and Keep actions', async () => {
    const {model} = await createReviewCard('density-surface');
    const markup = renderToStaticMarkup(
      createElement(AnalysisResultCard, {
        model,
        onNext: () => {},
        onAddToStack: () => {},
        onSaveWorkflow: () => {},
        inferenceScope: 'sample'
      })
    );
    expect(markup).not.toContain('Significance is hidden');
    expect(markup).not.toContain('Inference display');
    expect(markup).toContain('Density ranges');
    expect(markup).toMatch(/<button type="button">Add to stack<\/button>/);
  });
  test.each([
    'cluster-and-outlier-map',
    'hot-spot-map'
  ] as const)('default display matches the actual reference binding: %s', async identifier => {
    const {model} = await createReviewCard(identifier);
    const settings = getKeplerResultInferenceSettings(model.result);
    const update = apply(model.result, settings.significance, settings.multipleTesting);
    if (update.status !== 'display') throw new Error(update.reason);
    const excluded = new Set(update.excludedRows);
    const column = identifier === 'hot-spot-map' ? 'bin' : 'cluster';
    const before = model.result.tables[0].columns[column];
    const after = update.displayResult.tables[0].columns[column];
    // The binding writes signed zero for neutral cold rows; both zeros encode one neutral class.
    const normalize = (values: readonly unknown[]) =>
      values.map(value => (value === 0 ? 0 : value));
    expect(normalize(after.filter((_, row) => !excluded.has(row)))).toEqual(
      normalize(before.filter((_, row) => !excluded.has(row)))
    );
    expect(update.report.sentence).toBe(model.report.sentence);
    expect(update.displayResult.dispatchedContributors).toBe(model.result.dispatchedContributors);
  });

  test('BH uses raw values, excludes untested/islands and keeps the full family in source order', () => {
    const original = makeResult();
    const snapshot = JSON.stringify(original);
    const update = apply(original);
    expect(update.status).toBe('display');
    if (update.status !== 'display') throw new Error(update.reason);
    const columns = update.displayResult.tables[0].columns;
    expect(columns['adjusted_p_value']).toEqual([0.004, 0.02, (0.04 * 4) / 3, 0.2, null, null]);
    expect(columns['cluster']).toEqual([
      'LL',
      'LH',
      'not-significant',
      'not-significant',
      'not-tested',
      'not-tested'
    ]);
    expect(update.excludedRows).toEqual([4, 5]);
    expect(update.testedCount).toBe(4);
    expect(update.report.sentence).toBe(
      '2 of 4 features are significant at a pseudo p-value of 0.05 or less after false discovery rate correction: 1 low-low core and 1 low-high outlier.'
    );
    expect(
      update.presentations[0].legend.reduce((total, entry) => total + (entry.count ?? 0), 0)
    ).toBe(6);
    expect(update.originalResult).toBe(original);
    expect(JSON.stringify(original)).toBe(snapshot);
  });

  test('cut-off and correction changes use inclusive thresholds and recover nonsignificant quadrants', () => {
    const original = makeResult();
    for (const [significance, method, expected] of [
      [
        0.01,
        'none',
        ['LL', 'LH', 'not-significant', 'not-significant', 'not-tested', 'not-tested']
      ],
      [0.05, 'none', ['LL', 'LH', 'LL', 'not-significant', 'not-tested', 'not-tested']],
      [
        0.01,
        'benjamini-hochberg',
        ['LL', 'not-significant', 'not-significant', 'not-significant', 'not-tested', 'not-tested']
      ],
      [
        0.01,
        'bonferroni',
        ['LL', 'not-significant', 'not-significant', 'not-significant', 'not-tested', 'not-tested']
      ]
    ] as const) {
      const update = apply(original, significance, method);
      if (update.status !== 'display') throw new Error(update.reason);
      expect(update.displayResult.tables[0].columns['cluster']).toEqual(expected);
      expect(update.displayResult.summary['multipleTesting']).toBe(method);
      expect(update.displayResult.tables[0].columns['p_value']).toBe(
        original.tables[0].columns['p_value']
      );
    }
  });

  test('adjusted values allow same-correction cut-offs, never recovery of raw p-values', () => {
    const original = makeResult();
    delete original.tables[0].columns['p_value'];
    const update = apply(original, 0.01);
    expect(update.status).toBe('display');
    expect(apply(original, 0.01, 'none')).toMatchObject({
      status: 'rerun-required',
      reason: expect.stringContaining('Raw p-values')
    });
  });

  test('raw values, rather than previously adjusted values, define each new correction', () => {
    const original = makeResult();
    original.tables[0].columns['adjusted_p_value'] = [1, 1, 1, 1, 1, 1];
    const update = apply(original, 0.05, 'bonferroni');
    if (update.status !== 'display') throw new Error(update.reason);
    expect(update.displayResult.tables[0].columns['adjusted_p_value']).toEqual([
      0.004,
      0.04,
      0.16,
      0.8,
      null,
      null
    ]);
  });

  test('row permutation preserves family adjustments, classes and excluded source rows', () => {
    const result = makeResult();
    const order = [5, 1, 3, 0, 4, 2];
    result.tables[0].columns = Object.fromEntries(
      Object.entries(result.tables[0].columns).map(([name, values]) => [
        name,
        order.map(row => values[row])
      ])
    );
    const update = apply(result);
    if (update.status !== 'display') throw new Error(update.reason);
    expect(update.displayResult.tables[0].columns['cluster']).toEqual([
      'not-tested',
      'LH',
      'not-significant',
      'LL',
      'not-tested',
      'not-significant'
    ]);
    expect(update.displayResult.tables[0].columns['adjusted_p_value']).toEqual([
      null,
      0.02,
      0.2,
      0.004,
      null,
      (0.04 * 4) / 3
    ]);
    expect(update.excludedRows).toEqual([0, 4]);
  });

  test('the card requires a display consumer and labels the full-population rerun path', () => {
    const model = createAnalysisResultCardModel(makeResult());
    const markup = renderToStaticMarkup(
      createElement(AnalysisResultCard, {model, onNext: () => {}})
    );
    expect(markup).toContain('Inference display');
    expect(markup).toContain('Display changes are unavailable in this view.');
    expect(markup).toContain('disabled="">Apply to display');
    const sampled = renderToStaticMarkup(
      createElement(AnalysisResultCard, {
        model,
        onNext: () => {},
        inferenceScope: 'sample',
        onInferenceRerun: () => {}
      })
    );
    expect(sampled).toContain('Rerun required: Recompute on the full population');
    expect(sampled).toContain('Rerun analysis');
    expect(sampled).toContain('Significance is hidden for this sample preview.');
    expect(sampled).not.toContain('aria-label="local-moran legend"');
  });

  test('hot-spot bins, legend and sentence describe the same cut-off and exclusions', () => {
    const update = apply(makeResult(true), 0.05);
    if (update.status !== 'display') throw new Error(update.reason);
    expect(update.displayResult.tables[0].columns['bin']).toEqual([-3, 2, 0, 0, null, null]);
    expect(update.report.sentence).toBe(
      '1 of 4 cells is a significant hot spot and 1 is a significant cold spot at 95% confidence after false discovery rate correction.'
    );
    expect(update.presentations[0].presentation?.excludedRows).toEqual([4, 5]);
    expect(
      update.presentations[0].legend.reduce((total, entry) => total + (entry.count ?? 0), 0)
    ).toBe(6);
    expect(update.presentations[0].legend.find(entry => entry.label === 'Untested')?.count).toBe(2);
  });

  test('missing probabilities, class direction, unsupported bindings and partial populations require rerun', () => {
    const missing = makeResult();
    delete missing.tables[0].columns['p_value'];
    delete missing.tables[0].columns['adjusted_p_value'];
    expect(apply(missing)).toMatchObject({status: 'rerun-required'});
    const noDirection = makeResult();
    delete noDirection.tables[0].columns['spatial_lag'];
    expect(apply(noDirection)).toMatchObject({
      status: 'rerun-required',
      reason: expect.stringContaining('direction')
    });
    const noInference = makeResult();
    noInference.tables[0].columns['p_value'] = [null, null, null, null, null, null];
    expect(apply(noInference)).toMatchObject({
      status: 'rerun-required',
      reason: expect.stringContaining('No retained p-values')
    });
    for (const scope of ['sample', 'viewport'] as const)
      expect(apply(makeResult(), 0.05, 'none', scope)).toMatchObject({
        status: 'rerun-required',
        reason: expect.stringContaining('full population')
      });
    const unsupported = {...makeResult(), operationId: 'hotspot-gi-star'};
    expect(apply(unsupported)).toMatchObject({
      status: 'rerun-required',
      reason: expect.stringContaining('no verified')
    });
  });

  test('invalid raw probabilities never enter the correction family', () => {
    const result = makeResult();
    result.tables[0].columns['p_value'] = [0.01, Number.NaN, -1, 2, Number.POSITIVE_INFINITY, null];
    const update = apply(result);
    if (update.status !== 'display') throw new Error(update.reason);
    expect(update.testedCount).toBe(1);
    expect(update.displayResult.tables[0].columns['adjusted_p_value']).toEqual([
      0.01,
      null,
      null,
      null,
      null,
      null
    ]);
  });

  test('run defaults and named-limit validation are explicit', () => {
    expect(getKeplerResultInferenceSettings(makeResult(true)).significance).toBe(0.1);
    for (const value of [0, -0.1, 1.1, Number.NaN])
      expect(() => apply(makeResult(), value)).toThrow(/applyKeplerResultInference.*significance/);
  });
});

test('emerging hot spot categories explicitly guard sampled and viewport cards despite a row-count summary', () => {
  const result: CatalogRunResult = {
    operationId: 'curated-emerging-hot-spots',
    engine: 'gpu',
    dispatchedContributors: [],
    parameters: {},
    diagnostics: [],
    summary: {},
    tables: [
      {
        id: 'cells',
        label: 'Emerging cells',
        rowCount: 1,
        columns: {category_name: ['NEW_HOT_SPOT']},
        columnTypes: {category_name: 'string'}
      }
    ]
  };
  const model = createAnalysisResultCardModel(result, {}, {outputProfiles: []});
  expect(model.report.outputs[0]).toMatchObject({inference: true, usedTemplate: 'row-count'});
  expect(model.presentations[0].presentation?.classScheme).toBeUndefined();
  const callbacks = {onNext: () => {}, onAddToStack: () => {}, onSaveWorkflow: () => {}};
  const full = renderToStaticMarkup(createElement(AnalysisResultCard, {model, ...callbacks}));
  expect(full).toContain('cells legend');
  expect(full).toMatch(/<button type="button">Add to stack<\/button>/);
  for (const inferenceScope of ['sample', 'viewport'] as const) {
    const markup = renderToStaticMarkup(
      createElement(AnalysisResultCard, {model, ...callbacks, inferenceScope})
    );
    expect(markup).toContain(`Significance is hidden for this ${inferenceScope} preview.`);
    expect(markup).not.toContain('cells legend');
    expect(markup).not.toContain('NEW_HOT_SPOT');
    expect(markup).toMatch(/<button type="button" disabled="">Add to stack<\/button>/);
    expect(markup).toMatch(/<button type="button" disabled="">Save as workflow<\/button>/);
  }
});

test('explicit inference survives missing optional summary and palette metadata', () => {
  const result = makeResult();
  const descriptor = getKeplerOperationDescriptor(result.operationId)!;
  const model = createAnalysisResultCardModel(
    result,
    {},
    {
      outputProfiles: [],
      descriptor: {
        ...descriptor,
        outputs: [{tableId: result.tables[0].id, kind: 'attributes', inference: true}]
      }
    }
  );
  expect(model.report.outputs[0]).toMatchObject({
    inference: true,
    usedTemplate: 'row-count'
  });
  expect(hasKeplerResultInference(model.report, [])).toBe(true);
});
