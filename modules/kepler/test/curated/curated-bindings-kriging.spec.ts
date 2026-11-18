// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUKriging} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {CatalogGpuSession} from '../../src/analysis/catalog-gpu-session';
import {it, vi} from 'vitest';
import {requireTestDevice} from '../../../analysis/test/utils/require-test-device';
import {
  getCatalogBinding,
  runCatalogOperation,
  runCatalogOperationAsync
} from '../../src/analysis/catalog-bindings';
import type {CatalogRunResult} from '../../src/analysis/catalog-binding-types';
import type {
  CatalogFeatureCollection,
  CatalogOutputTable
} from '../../src/analysis/catalog-geojson';
import {getKeplerCuratedAnalysis} from '../../src/curated/curated-analyses';
import {createKeplerCuratedAnalysisSample} from '../../src/curated/curated-analysis-samples';
import {
  buildKeplerCuratedAnalysisRequest,
  createKeplerCuratedAnalysisMapping,
  updateKeplerCuratedAnalysisMapping
} from '../../src/curated/curated-analysis-mapping';
import {getKeplerOperationDescriptor} from '../../src/descriptors/operation-descriptors';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import {planKeplerResultPresentation} from '../../src/presentation/result-presentation';
import {KRIGING_POINTS} from './curated-bindings-oracle-data';

const BINDING_ID = 'curated-kriging-with-diagnostics';
const binding = getCatalogBinding(BINDING_ID)!;

/** Numerical errors are relative to max(1, |expected|), allowing f32 solves and reductions. */
const NUMERICAL_TOLERANCE = 3e-4;

function getTable(result: CatalogRunResult, tableId: string): CatalogOutputTable {
  const table = result.tables.find(candidate => candidate.id === tableId);
  if (!table) throw new Error(`${BINDING_ID}: missing ${tableId}; ${JSON.stringify(result)}`);
  return table;
}

function makeIndependentMeasurements(): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: KRIGING_POINTS.x.map((value, index) => ({
      id: `sensor-${index}`,
      geometry: {type: 'Point', coordinates: [value + 500000, KRIGING_POINTS.y[index] + 4000000]},
      properties: {temperature: KRIGING_POINTS.z[index]}
    }))
  };
}

function mapMeasurements(points: CatalogFeatureCollection, independent: boolean) {
  const analysis = getKeplerCuratedAnalysis('kriging-with-diagnostics')!;
  const datasetId = independent ? 'survey' : 'gauges';
  const column = independent ? 'temperature' : 'rainfall';
  const coordinateSystem = independent ? 'planar' : 'EPSG:4326';
  const profiles = [
    profileKeplerFeatureCollection(points, {
      datasetId,
      label: datasetId,
      coordinateSystem,
      declarations: {rowSemantics: 'surface-samples', measurement: {[column]: 'amount'}}
    })
  ];
  let mapping = createKeplerCuratedAnalysisMapping(analysis, profiles);
  mapping = updateKeplerCuratedAnalysisMapping(
    mapping,
    {slot: 'points', datasetId},
    analysis,
    profiles
  );
  mapping = updateKeplerCuratedAnalysisMapping(
    mapping,
    {slot: 'points', role: 'value', columns: [column]},
    analysis,
    profiles
  );
  if (!mapping.ready) throw new Error(`Kriging mapping incomplete: ${JSON.stringify(mapping)}`);
  return buildKeplerCuratedAnalysisRequest(analysis, mapping, {
    parameterOverrides: {
      coordinateSystem,
      cellSize: independent ? 200 : 250,
      neighborCount: 16,
      ...(independent
        ? {variogram: 'stated', model: 'spherical', nugget: 0.2, sill: 4, range: 600}
        : {})
    }
  });
}

function makeReport(label: string) {
  const mismatches: string[] = [];
  const check = (
    name: string,
    actual: unknown,
    expected: unknown,
    tolerance = NUMERICAL_TOLERANCE
  ) => {
    const matches =
      typeof actual === 'number' && typeof expected === 'number'
        ? Number.isNaN(expected)
          ? Number.isNaN(actual)
          : Number.isFinite(actual) &&
            Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected))
        : JSON.stringify(actual) === JSON.stringify(expected);
    if (!matches)
      mismatches.push(
        `${name}: actual=${JSON.stringify(actual)}, expected=${JSON.stringify(expected)}, tolerance=${tolerance}`
      );
  };
  return {
    check,
    finish: (result: CatalogRunResult) => {
      if (mismatches.length)
        throw new Error(
          `${label}: ${mismatches.length} mismatches\n${mismatches.join('\n')}\nsummary=${JSON.stringify(result.summary)}\nfallback=${result.fallbackReason ?? 'none'}`
        );
    }
  };
}

// Independent NumPy oracle: same committed KRIGING_POINTS fixture, now selecting the nearest 16
// points for each query (and excluding the held-out row). Solve the full augmented semivariogram
// system with numpy.linalg.solve; spherical nugget .2, partial sill 4, range 600. Cell centers
// are min(x/y) + (column/row + .5) * 200. This is a fresh local oracle, not the binding's solver.
const LOCAL_ORACLE = {
  cells: [
    [12.384809982652644, 1.3259431828349366],
    [13.430571900340095, 2.1369341526913463],
    [14.49353739795459, 1.4129818809956978],
    [14.05895658455927, 0.4873338949949566],
    [12.365004777253285, 1.5766340366104705],
    [11.476047715750756, 1.6701366679210052],
    [12.860577025136191, 0.8788821802968643],
    [13.550325089030146, 0.48466463751420386],
    [12.433653466147252, 0.48525235299066094],
    [10.465801265465, 0.4056535841937722],
    [9.864188367182452, 0.792577534607216],
    [11.791521681296416, 0.9410123276798559],
    [11.766811009170672, 1.0141235207045538],
    [11.066425651888773, 1.3189528699507194],
    [10.054239804073278, 1.1452973463232003],
    [9.158049862588028, 1.3074501272812316],
    [10.61639548908676, 0.8646947643201692],
    [10.633825245441, 1.9348723572597144],
    [9.892714832255237, 1.5067188638704296],
    [8.505386135715788, 0.9230681967875647]
  ],
  validation: [
    [11.072616026804125, 1.299387091712623],
    [9.952697962381304, 1.0633038799733565],
    [14.417902995125637, 0.7231109683988155],
    [12.942775308038495, 1.697916723091101],
    [9.040317714285294, 0.934825407911771],
    [12.445255746041736, 0.6394386781055017],
    [12.868372730991005, 0.6630802594355193],
    [11.93735580514069, 1.0666375821067582],
    [12.35541915125629, 2.826739444614505],
    [7.686653042014831, 0.902539909288832],
    [14.274426974132368, 0.9195002722965869],
    [10.219456983126074, 1.210774814200375],
    [13.582203258092001, 1.0709887061319967],
    [7.78938008509019, 0.8108853866912277],
    [10.653511780193387, 0.6709318423102875],
    [8.895770794249238, 0.9885887326578031],
    [12.385780943756952, 0.8101926351109878],
    [14.40899928203605, 0.71880776959712],
    [10.618752834963898, 1.598614219559447],
    [9.368551878888239, 0.7280236508712927],
    [11.41302381011909, 1.500190190372448],
    [12.315555890863624, 1.322503795868413],
    [11.862803388302645, 2.004180221330888],
    [9.622014587311506, 0.8124267151179558],
    [14.211536012122032, 0.6679091013756402],
    [9.64182602042283, 1.4085163225961495],
    [11.06698803070558, 0.7726386535209964],
    [10.678241429847281, 0.6534628125203942],
    [9.025917341970104, 1.0104444500033836],
    [11.905993216905648, 1.0381019176565287],
    [12.35787131801748, 1.577342031683818],
    [8.675470976749533, 1.2796318356752059],
    [13.426744161807346, 1.163121875690252],
    [11.563091940253567, 1.3719953364661759],
    [11.17483136744484, 0.8025298216878248],
    [10.858329693913912, 2.161269347238118],
    [8.914500926413258, 1.3881282050937225],
    [11.310811992817493, 1.2626229273957128],
    [9.743484816833881, 1.1621880380229501],
    [13.871836253499934, 0.9553472660834267]
  ]
};

for (const independent of [false, true]) {
  it(`dispatches the diagnostics recipe on mapped ${independent ? 'independent projected measurements' : 'rain-gauge sample'} and publishes CPU/recipe parity`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const points = independent
      ? makeIndependentMeasurements()
      : (createKeplerCuratedAnalysisSample('rain-gauges')['points'] as CatalogFeatureCollection);
    const request = mapMeasurements(points, independent);
    const datasets = {[independent ? 'survey' : 'gauges']: points};
    const inputs = Object.fromEntries(
      Object.entries(request.inputDatasetIds).map(([slot, datasetId]) => [
        slot,
        datasets[datasetId]
      ])
    );
    const reference = runCatalogOperation(binding, request.parameters, inputs);
    const contributor = vi.spyOn(GPUKriging.prototype, 'getCommandNodes');
    const submission = vi.spyOn(CatalogGpuSession.prototype, 'run');
    let result: CatalogRunResult;
    try {
      result = await runCatalogOperationAsync(binding, request.parameters, inputs, {device});
      const report = makeReport(
        independent ? 'projected local NumPy oracle' : 'fitted sample parity'
      );
      const {check} = report;
      check('engine', result.engine, 'gpu');
      check('fallback', result.fallbackReason, undefined);
      check('dispatched contributor names', result.dispatchedContributors, ['GPUKriging']);
      const callerGraphs = new Set(
        submission.mock.contexts.map(session => (session as CatalogGpuSession).graph)
      );
      check('GPU session submissions', submission.mock.calls.length, 1, 0);
      check(
        'actual GPUKriging caller-graph construction',
        contributor.mock.calls.filter(([graph]) => callerGraphs.has(graph)).length,
        1,
        0
      );
      check(
        'GPUKriging preflight construction',
        contributor.mock.calls.filter(([graph]) => !callerGraphs.has(graph)).length,
        1,
        0
      );
      check(
        'mapped column',
        result.parameters['valueProperty'],
        independent ? 'temperature' : 'rainfall'
      );
      const cells = getTable(result, 'kriging-cells');
      const validation = getTable(result, 'cross-validation');
      const descriptor = getKeplerOperationDescriptor(BINDING_ID)!;
      for (const [tableId, numericColumns, identityColumns] of [
        ['kriging-cells', ['value', 'variance'], ['cell_id', 'row', 'column', 'geometry']],
        [
          'cross-validation',
          ['observed', 'predicted', 'error', 'kriging_variance', 'standardized_error'],
          [
            'sample_index',
            'source_index',
            'source_id',
            'status',
            'location_sample_count',
            'geometry'
          ]
        ],
        ['variogram', ['distance', 'semivariance', 'model_semivariance'], ['bin', 'pair_count']]
      ] as const) {
        const actual = getTable(result, tableId);
        const expected = getTable(reference, tableId);
        check(`${tableId} row count`, actual.rowCount, expected.rowCount);
        check(`${tableId} schema`, actual.columnTypes, expected.columnTypes);
        check(`${tableId} geometry column`, actual.geometryColumn, expected.geometryColumn);
        for (const column of identityColumns)
          check(`${tableId}.${column}`, actual.columns[column], expected.columns[column]);
        for (const column of numericColumns)
          expected.columns[column].forEach((value, row) =>
            check(`${tableId}.${column}[${row}]`, actual.columns[column][row], value)
          );
      }
      check(
        'source identities in input order',
        validation.columns['source_id'],
        points.features.map(feature => feature.id)
      );
      check(
        'GPU neighbour counts honestly unavailable',
        cells.columns['sample_count'].every(value => value === null),
        true
      );
      for (const key of [
        'sampleCount',
        'cellCount',
        'model',
        'nugget',
        'sill',
        'range',
        'fitCriterion',
        'crossValidationCount',
        'rootMeanSquareError',
        'meanError',
        'meanAbsoluteError',
        'meanStandardizedError',
        'rootMeanSquareStandardizedError',
        'coverage95'
      ])
        check(`summary.${key}`, result.summary[key], reference.summary[key]);
      check('invalid grid cells', result.summary['gpuInvalidCellCount'], 0);
      check('uncertain grid cells', result.summary['gpuUncertainCellCount'], 0);
      check('invalid validation rows', result.summary['gpuInvalidValidationCount'], 0);
      check(
        'recipe standardized count',
        result.summary['gpuStandardizedCount'],
        points.features.length
      );
      const cellPresentation = planKeplerResultPresentation(
        cells,
        descriptor.outputs.find(output => output.tableId === 'kriging-cells')!
      );
      const validationPresentation = planKeplerResultPresentation(
        validation,
        descriptor.outputs.find(output => output.tableId === 'cross-validation')!
      );
      check('prediction publication column', cellPresentation?.column, 'value');
      check('prediction publication scale', cellPresentation?.scale, 'sequential');
      check('validation publication column', validationPresentation?.column, 'standardized_error');
      check('validation publication scale', validationPresentation?.scale, 'diverging');
      if (independent) {
        LOCAL_ORACLE.cells.forEach(([value, variance], row) => {
          check(`NumPy cell value[${row}]`, cells.columns['value'][row], value);
          check(`NumPy cell variance[${row}]`, cells.columns['variance'][row], variance);
        });
        LOCAL_ORACLE.validation.forEach(([predicted, variance], row) => {
          check(
            `NumPy validation prediction[${row}]`,
            validation.columns['predicted'][row],
            predicted
          );
          check(
            `NumPy validation variance[${row}]`,
            validation.columns['kriging_variance'][row],
            variance
          );
        });
      }
      report.finish(result);
    } finally {
      submission.mockRestore();
      contributor.mockRestore();
    }
  });
}

it('matches the constant-field diagnostics recipe fixture, merges duplicates and preserves missing feature statuses', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // expansion-recipes.spec.ts: constant values yield zero bias and RMSE. Extend its two
  // duplicate locations to eight distinct locations to satisfy the curated fit/validation gate.
  const points: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      [0, 0],
      [0.5, 0.5],
      [1.5, 0.5],
      [2.5, 0.5],
      [0, 2],
      [1, 3],
      [3, 2],
      [2, 3],
      [0.5, 0.5],
      [8, 8]
    ].map((coordinates, index) => ({
      id: `constant-${index}`,
      geometry: {type: 'Point', coordinates},
      properties: index === 9 ? {} : {value: 2}
    }))
  };
  points.features.push({id: 'no-geometry', geometry: null, properties: {value: 2}});
  const parameters = {
    coordinateSystem: 'planar',
    valueProperty: 'value',
    cellSize: 1,
    variogram: 'stated',
    model: 'spherical',
    nugget: 0,
    sill: 1,
    range: 10,
    neighborCount: 8
  };
  const result = await runCatalogOperationAsync(binding, parameters, {points}, {device});
  const {check, finish} = makeReport('constant diagnostics fixture');
  check('engine', result.engine, 'gpu');
  const cells = getTable(result, 'kriging-cells');
  const validation = getTable(result, 'cross-validation');
  for (const value of cells.columns['value']) check('constant prediction', value, 2);
  check('sample locations after merge', result.summary['sampleCount'], 8);
  check('recipe bias', result.summary['meanError'], 0);
  check('recipe RMSE', result.summary['rootMeanSquareError'], 0);
  check('recipe valid count', result.summary['crossValidationCount'], 8);
  check('merged status', validation.columns['status'][8], 'merged-into-earlier-sample');
  check('merged error excluded', validation.columns['error'][8], null);
  check('missing value status', validation.columns['status'][9], 'missing-value');
  check('missing geometry status', validation.columns['status'][10], 'missing-geometry');
  check(
    'cross-validation identity',
    validation.columns['source_id'],
    points.features.map(feature => feature.id)
  );
  check('exact-hit variance', cells.columns['variance'][0], 0);
  const reference = runCatalogOperation(binding, parameters, {points});
  getTable(reference, 'kriging-cells').columns['variance'].forEach((value, row) =>
    check(`variance[${row}]`, cells.columns['variance'][row], value)
  );
  finish(result);
});

it('reports radius-limited nodata and failed leave-one-out rows without hiding them in summaries', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const points = makeIndependentMeasurements();
  const request = mapMeasurements(points, true);
  const result = await runCatalogOperationAsync(
    binding,
    {...request.parameters, searchRadius: 1e-6},
    {points},
    {device}
  );
  const {check, finish} = makeReport('empty radius neighbourhoods');
  check('engine', result.engine, 'gpu');
  check('nodata grid excluded from publication', getTable(result, 'kriging-cells').rowCount, 0);
  check('invalid grid count', result.summary['gpuInvalidCellCount'], result.summary['cellCount']);
  check(
    'validation failure count',
    result.summary['gpuInvalidValidationCount'],
    points.features.length
  );
  check('valid validation count', result.summary['crossValidationCount'], 0);
  check('RMSE unavailable', Number.isNaN(result.summary['rootMeanSquareError']), true);
  const validation = getTable(result, 'cross-validation');
  check('validation identity retained', validation.rowCount, points.features.length);
  check(
    'failed predictions are null',
    validation.columns['predicted'].every(value => value === null),
    true
  );
  check(
    'failed errors are null',
    validation.columns['error'].every(value => value === null),
    true
  );
  check(
    'failure diagnostic',
    result.diagnostics.find(diagnostic => diagnostic.code === 'leave-one-out-failed')?.count,
    points.features.length
  );
  finish(result);
});

it('preserves squared units and standardized errors under a measurement scale and large baseline change', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const points = makeIndependentMeasurements();
  const request = mapMeasurements(points, true);
  const base = await runCatalogOperationAsync(binding, request.parameters, {points}, {device});
  const multiplier = 10;
  const baseline = 1e9;
  const scaledPoints: CatalogFeatureCollection = {
    ...points,
    features: points.features.map(feature => ({
      ...feature,
      properties: {temperature: Number(feature.properties?.['temperature']) * multiplier + baseline}
    }))
  };
  const result = await runCatalogOperationAsync(
    binding,
    {
      ...request.parameters,
      nugget: 0.2 * multiplier ** 2,
      sill: 4 * multiplier ** 2
    },
    {points: scaledPoints},
    {device}
  );
  const {check, finish} = makeReport('measurement scale and baseline');
  check('base engine', base.engine, 'gpu');
  check('scaled engine', result.engine, 'gpu');
  for (const tableId of ['kriging-cells', 'cross-validation']) {
    const actual = getTable(result, tableId);
    const expected = getTable(base, tableId);
    const predictionColumn = tableId === 'kriging-cells' ? 'value' : 'predicted';
    const varianceColumn = tableId === 'kriging-cells' ? 'variance' : 'kriging_variance';
    expected.columns[predictionColumn].forEach((value, row) =>
      // Compare offsets, so a billion-unit baseline cannot weaken the relative tolerance.
      check(
        `${tableId} prediction[${row}]`,
        Number(actual.columns[predictionColumn][row]) - baseline,
        Number(value) * multiplier
      )
    );
    expected.columns[varianceColumn].forEach((value, row) =>
      check(
        `${tableId} variance[${row}]`,
        actual.columns[varianceColumn][row],
        Number(value) * multiplier ** 2
      )
    );
  }
  getTable(base, 'cross-validation').columns['standardized_error'].forEach((value, row) =>
    check(
      `standardized error[${row}]`,
      getTable(result, 'cross-validation').columns['standardized_error'][row],
      value
    )
  );
  check(
    'RMSE in measurement units',
    result.summary['rootMeanSquareError'],
    Number(base.summary['rootMeanSquareError']) * multiplier
  );
  finish(result);
});

it('retains the recipe uncertainty status when float32 rejects an ill-conditioned Gaussian system', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const points = makeIndependentMeasurements();
  const request = mapMeasurements(points, true);
  const result = await runCatalogOperationAsync(
    binding,
    {...request.parameters, model: 'gaussian', nugget: 0, range: 1e9},
    {points},
    {device}
  );
  const {check, finish} = makeReport('ill-conditioned variogram');
  check('engine', result.engine, 'gpu');
  check('uncertain cells', result.summary['gpuUncertainCellCount'], result.summary['cellCount']);
  check('unreliable cells excluded', getTable(result, 'kriging-cells').rowCount, 0);
  check(
    'invalid validation predictions',
    result.summary['gpuInvalidValidationCount'],
    points.features.length
  );
  check(
    'uncertainty diagnostic',
    result.diagnostics.find(diagnostic => diagnostic.code === 'singular-kriging-system')?.count,
    result.summary['cellCount']
  );
  finish(result);
});

it('review finding 1: preserves exact-hit measurements of 1, 2 and 3 beside a 1e8 outlier', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const locations = [
    [0, 0],
    [0.5, 0.5],
    [1.5, 0.5],
    [2.5, 0.5],
    [0, 2],
    [1, 3],
    [3, 2],
    [2, 3]
  ];
  const measurements = [1e8, 1, 2, 3, 4, 5, 6, 7];
  const points: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: locations.map((coordinates, index) => ({
      id: `mixed-scale-${index}`,
      geometry: {type: 'Point', coordinates},
      properties: {value: measurements[index]}
    }))
  };
  const parameters = {
    coordinateSystem: 'planar',
    valueProperty: 'value',
    cellSize: 1,
    variogram: 'stated',
    model: 'spherical',
    nugget: 0,
    sill: 1,
    range: 10,
    neighborCount: 8
  };
  const result = await runCatalogOperationAsync(binding, parameters, {points}, {device});
  const {check, finish} = makeReport('review finding 1: mixed measurement magnitudes');
  check('engine', result.engine, 'gpu');
  check('fallback absent', result.fallbackReason, undefined);
  const cells = getTable(result, 'kriging-cells');
  for (let column = 0; column < 3; column++) {
    const row = cells.columns['row'].findIndex(
      (value, index) => value === 0 && cells.columns['column'][index] === column
    );
    check(`exact-hit prediction at column ${column}`, cells.columns['value'][row], column + 1);
    check(`exact-hit variance at column ${column}`, cells.columns['variance'][row], 0);
  }
  finish(result);
});

it('review finding 2: visibly falls back for distinct locations that coincide after float32 conversion', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const locations = [
    [0, 0],
    [0.5, 0.5],
    [0.50000001, 0.5],
    [1.5, 0.5],
    [2.5, 0.5],
    [0, 2],
    [1, 3],
    [3, 2]
  ];
  const measurements = [0, 1, 3, 2, 3, 4, 5, 6];
  const points: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: locations.map((coordinates, index) => ({
      id: `nearby-${index}`,
      geometry: {type: 'Point', coordinates},
      properties: {value: measurements[index]}
    }))
  };
  const parameters = {
    coordinateSystem: 'planar',
    valueProperty: 'value',
    cellSize: 1,
    variogram: 'stated',
    model: 'spherical',
    nugget: 1,
    sill: 1,
    range: 10,
    neighborCount: 8
  };
  const result = await runCatalogOperationAsync(binding, parameters, {points}, {device});
  const {check, finish} = makeReport('review finding 2: float32 coordinate collision');
  check('engine', result.engine, 'cpu-reference');
  check('no GPU dispatch', result.dispatchedContributors, []);
  check(
    'collision fallback reason',
    /points.*distinct.*collide in float32/.test(result.fallbackReason ?? ''),
    true
  );
  check(
    'visible fallback diagnostic',
    result.diagnostics.some(diagnostic => diagnostic.code === 'gpu-fallback'),
    true
  );
  check('exact-hit prediction', getTable(result, 'kriging-cells').columns['value'][0], 1);
  check('distinct sample count', result.summary['sampleCount'], 8);
  check(
    'no original coincidence diagnostic',
    result.diagnostics.some(diagnostic => diagnostic.code === 'coincident-samples'),
    false
  );
  check(
    'positive leave-one-out variance',
    Number(getTable(result, 'cross-validation').columns['kriging_variance'][1]) > 0,
    true
  );
  finish(result);
});
