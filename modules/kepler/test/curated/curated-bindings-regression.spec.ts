// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../analysis/test/utils/require-test-device';
import {
  computeSpatialRegressionDiagnosticsOnCPU,
  createDiagnosticsScene
} from '../../../analysis/test/gpu-spatial-analysis/spatial-regression/spatial-regression-diagnostics-oracle';
import {computeLocalMoranOracle} from '../../../analysis/test/gpu-spatial-analysis/spatial-autocorrelation/spatial-autocorrelation-oracle';
import {getSpatialRegressionSpecification} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  runCatalogOperation,
  runCatalogOperationAsync,
  normalizeCatalogParameters
} from '../../src/analysis/catalog-bindings';
import type {
  CatalogFeatureCollection,
  CatalogOutputTable
} from '../../src/analysis/catalog-geojson';
import {
  CURATED_REGRESSION_WITH_DIAGNOSTICS_BINDING,
  prepareCuratedRegression
} from '../../src/curated/bindings/curated-regression-with-diagnostics-binding';
import {prepareCuratedRegressionGpu} from '../../src/curated/bindings/curated-regression-with-diagnostics-gpu';
import {createKeplerCuratedAnalysisSample} from '../../src/curated/curated-analysis-samples';
import {getKeplerCuratedAnalysis} from '../../src/curated/curated-analyses';
import {
  createKeplerCuratedAnalysisMapping,
  updateKeplerCuratedAnalysisMapping,
  buildKeplerCuratedAnalysisRequest
} from '../../src/curated/curated-analysis-mapping';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import {makeLattice, makePoints} from './curated-bindings-fixtures';
import {REGRESSION_POINTS} from './curated-bindings-oracle-data';

const BINDING = CURATED_REGRESSION_WITH_DIAGNOSTICS_BINDING;

/** Include actual values in thrown messages: the headless runner suppresses console output. */
function compareValues(
  label: string,
  actual: readonly unknown[],
  expected: ArrayLike<number>,
  absolute = 5e-3,
  relative = 5e-3
) {
  const errors: string[] = [];
  if (actual.length !== expected.length)
    errors.push(`length ${actual.length} != ${expected.length}`);
  Array.from(expected).forEach((value, row) => {
    const observed = actual[row];
    if (!Number.isFinite(value)) {
      if (observed !== null) errors.push(`row ${row}: ${String(observed)} != null`);
    } else if (
      typeof observed !== 'number' ||
      !Number.isFinite(observed) ||
      Math.abs(observed - value) > absolute + relative * Math.abs(value)
    ) {
      errors.push(
        `row ${row}: ${String(observed)} != ${value}; tolerance ${absolute} + ${relative} * abs(expected)`
      );
    }
  });
  if (errors.length) throw new Error(`curated GPU regression ${label}:\n${errors.join('\n')}`);
}

function getTable(tables: CatalogOutputTable[], identifier: string) {
  const table = tables.find(candidate => candidate.id === identifier);
  if (!table) throw new Error(`curated GPU regression missing table ${identifier}`);
  return table;
}

function mapDataset(
  areas: CatalogFeatureCollection,
  datasetId: string,
  response: string,
  predictors: string[],
  coordinateSystem: 'planar' | 'EPSG:4326',
  neighborhood = 'auto'
) {
  const analysis = getKeplerCuratedAnalysis('regression-with-diagnostics')!;
  const profiles = [
    profileKeplerFeatureCollection(areas, {
      datasetId,
      label: datasetId,
      coordinateSystem,
      declarations: {
        coordinateModel: coordinateSystem === 'planar' ? 'planar' : 'geographic',
        rowSemantics: 'objects'
      }
    })
  ];
  let mapping = createKeplerCuratedAnalysisMapping(analysis, profiles, {
    parameters: {coordinateSystem, neighborhood, permutations: 99, seed: 5}
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
    {slot: 'areas', role: 'predictors', columns: predictors},
    analysis,
    profiles
  );
  expect(mapping.ready, JSON.stringify(mapping.problems)).toBe(true);
  const request = buildKeplerCuratedAnalysisRequest(analysis, mapping);
  expect(request.inputDatasetIds).toEqual({areas: datasetId});
  expect(request.parameters['responseProperty']).toBe(response);
  expect(request.parameters['predictorProperties']).toBe(predictors.join(','));
  return request.parameters;
}

it('dispatches the recipe on the mapped sample and an independently mapped, renamed existing recipe fixture', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createDiagnosticsScene(8, 11, 0.6, true); // Existing spatial-regression-recipe.spec.ts fixture.
  const independentAreas = makeLattice(8, 8, row => ({
    measured: scene.response[row],
    firstDriver: scene.predictors[row * 2],
    secondDriver: scene.predictors[row * 2 + 1]
  }));
  const sample = createKeplerCuratedAnalysisSample('district-regression')['areas'];
  const recipeOracle = computeSpatialRegressionDiagnosticsOnCPU(
    scene.weights,
    scene.predictors,
    scene.response,
    2
  );
  for (const fixture of [
    {
      id: 'sample',
      areas: sample,
      response: 'outcome',
      predictors: ['income', 'density'],
      coordinateSystem: 'EPSG:4326',
      neighborhood: 'queen'
    },
    {
      id: 'independent-recipe-fixture',
      areas: independentAreas,
      response: 'measured',
      predictors: ['firstDriver', 'secondDriver'],
      coordinateSystem: 'planar',
      neighborhood: 'rook'
    }
  ] as const) {
    const parameters = mapDataset(
      fixture.areas,
      fixture.id,
      fixture.response,
      [...fixture.predictors],
      fixture.coordinateSystem,
      fixture.neighborhood
    );
    const result = await runCatalogOperationAsync(
      BINDING,
      parameters,
      {areas: fixture.areas},
      {device}
    );
    expect(
      result.engine,
      JSON.stringify({fallback: result.fallbackReason, diagnostics: result.diagnostics})
    ).toBe('gpu');
    expect(result.dispatchedContributors).toEqual([
      'GPUOrdinaryLeastSquares',
      'GPUSpatialRegressionDiagnostics',
      'GPULocalMoran'
    ]);
    const reference = runCatalogOperation(BINDING, parameters, {areas: fixture.areas});
    for (const [tableId, columns] of [
      ['coefficients', ['estimate', 'standard_error', 't_statistic', 'p_value']],
      ['residuals', ['fitted', 'residual', 'neighbor_count']]
    ] as const) {
      const actual = getTable(result.tables, tableId);
      const expected = getTable(reference.tables, tableId);
      for (const column of columns)
        compareValues(
          `${fixture.id}/${tableId}/${column}`,
          actual.columns[column],
          expected.columns[column] as number[]
        );
    }
    const residuals = getTable(result.tables, 'residuals');
    // Catalog identity columns encode absent IDs as null and numeric IDs as strings.
    const expectedSourceIds = fixture.areas.features.map(feature =>
      feature.id === undefined || feature.id === null ? null : String(feature.id)
    );
    expect(residuals.columns['source_id'], `${fixture.id}/GPU source IDs`).toEqual(
      expectedSourceIds
    );
    expect(
      getTable(reference.tables, 'residuals').columns['source_id'],
      `${fixture.id}/CPU source IDs`
    ).toEqual(expectedSourceIds);
    expect(residuals.columns['source_index']).toEqual(
      fixture.areas.features.map((_, index) => index)
    );
    expect(residuals.columns['geometry']).toEqual(
      fixture.areas.features.map(feature => JSON.stringify(feature.geometry))
    );
    for (const table of result.tables)
      expect(
        Object.values(table.columns).every(column => column.length === table.rowCount),
        `${fixture.id}/${table.id} rectangular table`
      ).toBe(true);
    const prepared = prepareCuratedRegression(
      {areas: fixture.areas},
      normalizeCatalogParameters(BINDING, parameters)
    );
    const scaled = prepareCuratedRegressionGpu(prepared);
    const oracle = computeSpatialRegressionDiagnosticsOnCPU(
      scaled.weights,
      scaled.predictors,
      scaled.response.values,
      2
    );
    const diagnostics = getTable(result.tables, 'diagnostics');
    const tests = new Float32Array(18);
    for (const [row, name, expected] of [
      [0, 'lm-lag', oracle.lmLag],
      [1, 'lm-error', oracle.lmError],
      [2, 'robust-lm-lag', oracle.robustLmLag],
      [3, 'robust-lm-error', oracle.robustLmError],
      [4, 'lm-sarma', oracle.lmSarma]
    ] as const) {
      const position = diagnostics.columns['diagnostic'].indexOf(name);
      compareValues(
        `${fixture.id}/${name}`,
        [
          diagnostics.columns['value'][position],
          diagnostics.columns['degrees_of_freedom'][position],
          diagnostics.columns['p_value'][position]
        ],
        [expected.statistic, expected.degreesOfFreedom, expected.pValue]
      );
      tests.set([expected.statistic, expected.degreesOfFreedom, expected.pValue], row * 3);
      if (fixture.id === 'independent-recipe-fixture') {
        const original = [
          recipeOracle.lmLag,
          recipeOracle.lmError,
          recipeOracle.robustLmLag,
          recipeOracle.robustLmError,
          recipeOracle.lmSarma
        ][row];
        compareValues(
          `original recipe fixture/${name}`,
          [diagnostics.columns['value'][position]],
          [original.statistic]
        );
      }
    }
    expect(result.summary['specificationDecision']).toBe(
      getSpatialRegressionSpecification(tests, 0)
    );
    const analyticPosition = diagnostics.columns['diagnostic'].indexOf(
      'residual-morans-i-analytic'
    );
    compareValues(
      `${fixture.id}/Cliff-Ord`,
      [
        diagnostics.columns['value'][analyticPosition],
        diagnostics.columns['expected'][analyticPosition],
        diagnostics.columns['p_value'][analyticPosition]
      ],
      [oracle.moranI, oracle.moranExpectation, oracle.moranPValue]
    );
    const localOracle = computeLocalMoranOracle({
      weights: scaled.weights,
      values: Float32Array.from(oracle.residuals)
    });
    const canonicalBySource = new Map(prepared.rows.map((row, index) => [row.sourceIndex, index]));
    compareValues(
      `${fixture.id}/local-I`,
      residuals.columns['local_morans_i'],
      fixture.areas.features.map((_, row) => localOracle.localI[canonicalBySource.get(row)!])
    );
    compareValues(
      `${fixture.id}/local-p`,
      residuals.columns['local_moran_p_value'],
      fixture.areas.features.map((_, row) => localOracle.pValues[canonicalBySource.get(row)!])
    );
    // Supplementary fit diagnostics are computed from GPU residuals and retain their existing conventions.
    const expectedDiagnostics = getTable(reference.tables, 'diagnostics');
    for (const name of [
      'r-squared',
      'adjusted-r-squared',
      'sigma-squared',
      'aic',
      'aicc',
      'jarque-bera',
      'breusch-pagan',
      'koenker-bassett',
      'condition-number',
      'residual-morans-i'
    ]) {
      const position = diagnostics.columns['diagnostic'].indexOf(name);
      const expectedPosition = expectedDiagnostics.columns['diagnostic'].indexOf(name);
      compareValues(
        `${fixture.id}/${name}`,
        [diagnostics.columns['value'][position]],
        [Number(expectedDiagnostics.columns['value'][expectedPosition])]
      );
    }
  }
});

it('dispatches asymmetric point kNN and preserves complete-case, island and shuffle identities', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const areas = makePoints(REGRESSION_POINTS.px, REGRESSION_POINTS.py, row => ({
    y: REGRESSION_POINTS.y[row],
    x: REGRESSION_POINTS.x[row]
  }));
  const parameters = {
    coordinateSystem: 'planar',
    responseProperty: 'y',
    predictorProperties: 'x',
    neighborhood: 'knn',
    neighborCount: 5,
    permutations: 99,
    seed: 5
  };
  const result = await runCatalogOperationAsync(BINDING, parameters, {areas}, {device});
  expect(result.engine, result.fallbackReason).toBe('gpu');
  compareValues(
    'point coefficients/NumPy',
    result.tables[0].columns['estimate'],
    REGRESSION_POINTS.coefficients
  );
  compareValues(
    'point residuals/NumPy',
    result.tables[2].columns['residual'],
    REGRESSION_POINTS.residuals
  );
  const shuffled = {...areas, features: [...areas.features].reverse()};
  const reversed = await runCatalogOperationAsync(BINDING, parameters, {areas: shuffled}, {device});
  expect(reversed.engine, reversed.fallbackReason).toBe('gpu');
  compareValues(
    'shuffled coefficients',
    reversed.tables[0].columns['estimate'],
    result.tables[0].columns['estimate'] as number[],
    1e-6,
    1e-6
  );
  compareValues(
    'shuffled residuals',
    reversed.tables[2].columns['residual'],
    [...result.tables[2].columns['residual']].reverse() as number[],
    1e-6,
    1e-6
  );
  const polygons = makeLattice(6, 6, row => ({y: 2 + row / 10 + Math.sin(row), x: Math.cos(row)}));
  polygons.features[0].properties!['y'] = null;
  polygons.features.push({id: 'missing-geometry', geometry: null, properties: {y: 4, x: 2}});
  polygons.features.push({
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
    properties: {y: 3, x: 1}
  });
  const incomplete = await runCatalogOperationAsync(
    BINDING,
    {...parameters, neighborhood: 'queen'},
    {areas: polygons},
    {device}
  );
  expect(incomplete.engine, incomplete.fallbackReason).toBe('gpu');
  const residuals = incomplete.tables[2];
  expect(residuals.columns['status'][0]).toBe('missing-value');
  expect(residuals.columns['status'][36]).toBe('missing-geometry');
  expect(residuals.columns['neighbor_count'][37]).toBe(0);
  expect(residuals.columns['local_morans_i'][37]).toBeNull();
  expect(residuals.columns['local_moran_p_value'][37]).toBeNull();
  // Response translation would incorrectly change LM-lag when W has an island row.
  const oraclePrepared = prepareCuratedRegression(
    {areas: polygons},
    normalizeCatalogParameters(BINDING, {...parameters, neighborhood: 'queen'})
  );
  const oracleScaled = prepareCuratedRegressionGpu(oraclePrepared);
  const oracle = computeSpatialRegressionDiagnosticsOnCPU(
    oracleScaled.weights,
    Float32Array.from(oraclePrepared.predictorArray),
    Float32Array.from(oraclePrepared.responseValues),
    1
  );
  const position = incomplete.tables[1].columns['diagnostic'].indexOf('lm-lag');
  compareValues(
    'island raw-response LM-lag',
    [incomplete.tables[1].columns['value'][position]],
    [oracle.lmLag.statistic]
  );
});

it.each([
  3, 100000000
])('falls back truthfully for perfect fits with response offset %s', async responseOffset => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const areas = makePoints(
    Array.from({length: 36}, (_, row) => row % 6),
    Array.from({length: 36}, (_, row) => Math.floor(row / 6)),
    row => ({x: row, y: responseOffset + 2 * row})
  );
  const result = await runCatalogOperationAsync(
    BINDING,
    {coordinateSystem: 'planar', responseProperty: 'y', predictorProperties: 'x', permutations: 99},
    {areas},
    {device}
  );
  expect(result.engine).toBe('cpu-reference');
  expect(result.dispatchedContributors).toEqual([]);
  expect(result.fallbackReason).toMatch(/CPU perfect-fit cutoff/);
  expect(result.diagnostics.map(diagnostic => diagnostic.code)).toEqual(
    expect.arrayContaining(['gpu-fallback', 'perfect-fit'])
  );
  expect(result.summary['residualMoransI']).toBe('not computed');
});
