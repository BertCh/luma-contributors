// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import reference from '../../../analysis/test/oracles/fixtures/dasymetric-redistribution.json';
import {requireTestDevice} from '../../../analysis/test/utils/require-test-device';
import {runCatalogOperationAsync} from '../../src/analysis/catalog-bindings';
import type {CatalogRunResult} from '../../src/analysis/catalog-binding-types';
import type {
  CatalogFeatureCollection,
  CatalogOutputTable
} from '../../src/analysis/catalog-geojson';
import type {CatalogRasterInput} from '../../src/analysis/catalog-raster-input';
import {CURATED_DASYMETRIC_REDISTRIBUTION_BINDING} from '../../src/curated/bindings/curated-dasymetric-redistribution-binding';
import {
  CURATED_DASYMETRIC_REDISTRIBUTION_ANALYSIS,
  createDasymetricRedistributionSample
} from '../../src/curated/samples/index';
import {
  createKeplerCuratedAnalysisSample,
  getKeplerCuratedAnalysisSampleParameters
} from '../../src/curated/curated-analysis-samples';
import {
  buildKeplerCuratedAnalysisRequest,
  createKeplerCuratedAnalysisMapping,
  updateKeplerCuratedAnalysisMapping
} from '../../src/curated/curated-analysis-mapping';
import {CURATED_DASYMETRIC_REDISTRIBUTION_DESCRIPTOR} from '../../src/descriptors/descriptors-curated-dasymetric-redistribution';
import {planKeplerResultPresentation} from '../../src/presentation/result-presentation';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import type {KeplerDatasetProfile} from '../../src/profile/dataset-profile-types';
import {getWorkflowFeatures, makeWorkflowData} from '../../src/workflows/workflow-data';
import {DASYMETRIC_GEOMETRY_CASES} from './curated-dasymetric-redistribution-fixtures';

const binding = CURATED_DASYMETRIC_REDISTRIBUTION_BINDING;

function readTable(result: CatalogRunResult, name: string): CatalogOutputTable {
  const table = result.tables.find(candidate => candidate.id === name);
  if (!table) throw new Error(`Missing ${name}: ${JSON.stringify(result)}`);
  return table;
}

function expectNumbers(
  table: CatalogOutputTable,
  column: string,
  expected: readonly (number | null)[]
): void {
  const actual = table.columns[column];
  expect(actual?.length, `${table.id}.${column}`).toBe(expected.length);
  expected.forEach((value, index) => {
    const observed = actual[index];
    const report = `${table.id}.${column}[${index}]: expected ${value}, actual ${observed}; table=${JSON.stringify(table)}`;
    if (value === null) expect(observed, report).toBeNull();
    else {
      expect(typeof observed, report).toBe('number');
      expect(Math.abs(Number(observed) - value), report).toBeLessThanOrEqual(
        1e-4 + 1e-5 * Math.abs(value)
      );
    }
  });
}

test.each([
  false,
  true
])('GPU recipe dispatch and publication after mapping (second=%s)', async second => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const sample = createDasymetricRedistributionSample(second);
  if (!second) {
    sample.inputs = {...createKeplerCuratedAnalysisSample('dasymetric-fractional-support')};
    sample.parameters = {...getKeplerCuratedAnalysisSampleParameters('dasymetric-redistribution')};
  }
  const analysis = CURATED_DASYMETRIC_REDISTRIBUTION_ANALYSIS;
  const totalColumn = sample.parameters['totalProperty'] as string;
  const profiles: KeplerDatasetProfile[] = ['sources', 'targets'].map(name =>
    profileKeplerFeatureCollection(sample.inputs[name] as CatalogFeatureCollection, {
      datasetId: `loaded-${name}`,
      label: name,
      coordinateSystem: 'planar',
      declarations: {measurement: {[totalColumn]: 'count'}}
    })
  );
  profiles.push({
    datasetId: 'loaded-ancillary',
    label: 'ancillary',
    kind: 'surface',
    rowCount: 6,
    columns: [],
    geometryColumns: []
  });
  let mapping = createKeplerCuratedAnalysisMapping(analysis, profiles, {
    parameters: sample.parameters
  });
  for (const slot of analysis.slots)
    mapping = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: slot.name, datasetId: `loaded-${slot.name}`},
      analysis,
      profiles
    );
  mapping = updateKeplerCuratedAnalysisMapping(
    mapping,
    {slot: 'sources', role: 'total', columns: [totalColumn]},
    analysis,
    profiles
  );
  const request = buildKeplerCuratedAnalysisRequest(analysis, mapping);
  const loaded = Object.fromEntries(
    Object.entries(sample.inputs).map(([name, input]) => [`loaded-${name}`, input])
  );
  const inputs = Object.fromEntries(
    Object.entries(request.inputDatasetIds).map(([name, datasetId]) => [name, loaded[datasetId]])
  );
  for (const denominator of ['zone', 'overlap'] as const) {
    const result = await runCatalogOperationAsync(
      binding,
      {...request.parameters, denominator},
      inputs,
      {device}
    );
    expect(result.engine, JSON.stringify(result)).toBe('gpu');
    expect(result.dispatchedContributors, JSON.stringify(result)).toEqual(
      expect.arrayContaining([...binding.gpuContributorNames])
    );
    expect(result.summary['complete'], JSON.stringify(result)).toBe(1);
    const targets = readTable(result, 'dasymetric-targets');
    expect(targets.coordinateSystem, JSON.stringify(targets)).toBe('planar');
    const ledger = readTable(result, 'dasymetric-ledger');
    expectNumbers(targets, 'total', sample.expected[denominator].targets);
    expectNumbers(ledger, 'allocated_total', sample.expected[denominator].allocated);
    expectNumbers(ledger, 'unallocated_total', sample.expected[denominator].residual);
    expectNumbers(ledger, 'eligible_weight', sample.expected.zone.support);
    expectNumbers(ledger, 'validity', second ? [1, 1, 1] : [1, 1, 0]);
    expectNumbers(
      ledger,
      'transferred_fraction',
      second
        ? denominator === 'zone'
          ? [1, 0.25, 1]
          : [1, 1, 1]
        : denominator === 'zone'
          ? [1, 0.6, null]
          : [1, 1, null]
    );
    expect(targets.columns['target_id']).toEqual(
      (inputs['targets'] as CatalogFeatureCollection).features.map(feature => String(feature.id))
    );
    expect(ledger.columns['source_id']).toEqual(
      (inputs['sources'] as CatalogFeatureCollection).features.map(feature => String(feature.id))
    );
    const sourceTotal = sample.expected.sourceTotals.reduce((sum, value) => sum + value, 0);
    expect(Number(result.summary['massBalanceError']), JSON.stringify(result)).toBeLessThanOrEqual(
      1e-4 + 1e-5 * sourceTotal
    );
    expect(
      planKeplerResultPresentation(
        targets,
        CURATED_DASYMETRIC_REDISTRIBUTION_DESCRIPTOR.outputs[0]
      ),
      JSON.stringify(result)
    ).toMatchObject({column: 'total'});
    const published = getWorkflowFeatures(makeWorkflowData(targets, {identityColumn: 'target_id'}));
    expect(published.features.map(feature => feature.id)).toEqual(targets.columns['target_id']);
    expect(published.features.map(feature => feature.geometry)).toEqual(
      (inputs['targets'] as CatalogFeatureCollection).features.map(feature => feature.geometry)
    );
    expect(published.features.map(feature => feature.properties?.['total'])).toEqual(
      targets.columns['total']
    );
  }
});

test.each(reference.cases)('GPU binding equals committed recipe fixture: $name', async fixture => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const sample = createDasymetricRedistributionSample();
  const raster = sample.inputs['ancillary'] as CatalogRasterInput;
  sample.inputs['ancillary'] = {
    ...raster,
    band: {...raster.band, values: Float32Array.from(fixture.ancillary)}
  };
  if (fixture.coverage.fractions.every(value => value === 1)) {
    const sources = sample.inputs['sources'] as CatalogFeatureCollection;
    sources.features[0].geometry = {
      type: 'Polygon',
      coordinates: [
        [
          [0, 0],
          [3, 0],
          [3, 1],
          [0, 1],
          [0, 0]
        ]
      ]
    };
    sources.features[1].geometry = {
      type: 'Polygon',
      coordinates: [
        [
          [3, 0],
          [5, 0],
          [5, 1],
          [3, 1],
          [3, 0]
        ]
      ]
    };
  }
  const result = await runCatalogOperationAsync(
    binding,
    {...sample.parameters, denominator: fixture.denominator},
    sample.inputs,
    {device}
  );
  expect(result.engine, JSON.stringify(result)).toBe('gpu');
  expect(result.summary['complete'], JSON.stringify(result)).toBe(1);
  expectNumbers(readTable(result, 'dasymetric-targets'), 'total', fixture.targetTotals);
  const ledger = readTable(result, 'dasymetric-ledger');
  expectNumbers(ledger, 'eligible_weight', fixture.sourceEligibleWeights);
  expectNumbers(ledger, 'transferred_fraction', fixture.sourceTransferredFractions);
  expectNumbers(ledger, 'allocated_total', fixture.sourceAllocatedTotals);
  expectNumbers(ledger, 'unallocated_total', fixture.sourceUnallocatedTotals);
});

test.each([
  'negative-total',
  'missing-total',
  'pair-overflow',
  'float32-overflow',
  'invalid-geometry'
])('invalid %s suppresses derived numbers and retains stage evidence', async failure => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const sample = createDasymetricRedistributionSample();
  const sources = sample.inputs['sources'] as CatalogFeatureCollection;
  if (failure === 'negative-total') sources.features[0].properties = {residents: -1};
  if (failure === 'missing-total') sources.features[0].properties = {};
  if (failure === 'pair-overflow') sample.parameters['pairCapacity'] = 1;
  if (failure === 'float32-overflow') {
    const raster = sample.inputs['ancillary'] as CatalogRasterInput;
    sample.inputs['ancillary'] = {
      ...raster,
      band: {...raster.band, values: new Float32Array(6).fill(3e38)}
    };
  }
  if (failure === 'invalid-geometry')
    sources.features[0].geometry = DASYMETRIC_GEOMETRY_CASES.unbridgeableHole;
  const result = await runCatalogOperationAsync(binding, sample.parameters, sample.inputs, {
    device
  });
  expect(result.engine, JSON.stringify(result)).toBe('gpu');
  expect(result.summary['complete'], JSON.stringify(result)).toBe(0);
  expectNumbers(readTable(result, 'dasymetric-targets'), 'total', [null, null, null, null]);
  expectNumbers(readTable(result, 'dasymetric-targets'), 'support_weight', [
    null,
    null,
    null,
    null
  ]);
  for (const column of [
    'eligible_weight',
    'transferred_fraction',
    'allocated_total',
    'unallocated_total'
  ])
    expectNumbers(readTable(result, 'dasymetric-ledger'), column, [null, null, null]);
  expect(result.diagnostics, JSON.stringify(result)).toContainEqual(
    expect.objectContaining({code: 'incomplete-dasymetric', severity: 'error'})
  );
  const evidence = readTable(result, 'dasymetric-status');
  expect(evidence.rowCount).toBeGreaterThan(5);
  if (failure === 'invalid-geometry') {
    const coverageRow = evidence.columns['stage'].findIndex(
      (stage, index) =>
        stage === 'coverage-validation' && evidence.columns['metric'][index] === 'converged'
    );
    expect(coverageRow, JSON.stringify(result)).toBeGreaterThanOrEqual(0);
    expect(evidence.columns['value'][coverageRow], JSON.stringify(result)).toBe(0);
  }
  expect(result.summary['massBalanceError']).toBeUndefined();
});

test('a collapsed source shell completes with zero support and retains its mass', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const sample = createDasymetricRedistributionSample();
  const sources = sample.inputs['sources'] as CatalogFeatureCollection;
  sources.features[0].geometry = DASYMETRIC_GEOMETRY_CASES.collapsedShell;
  const result = await runCatalogOperationAsync(binding, sample.parameters, sample.inputs, {
    device
  });
  expect(result.engine, JSON.stringify(result)).toBe('gpu');
  expect(result.summary['complete'], JSON.stringify(result)).toBe(1);
  expect(result.summary['zeroEligibleCount'], JSON.stringify(result)).toBe(2);
  expectNumbers(readTable(result, 'dasymetric-targets'), 'total', [0, 48, 0, 0]);
  const ledger = readTable(result, 'dasymetric-ledger');
  expect(ledger.columns['source_id'], JSON.stringify(result)).toEqual([
    'source-9',
    'source-2',
    'source-7'
  ]);
  expect(ledger.columns['status'], JSON.stringify(result)).toEqual([
    'zero-support',
    'complete',
    'zero-support'
  ]);
  expectNumbers(ledger, 'eligible_weight', [0, 2.5, 0]);
  expectNumbers(ledger, 'transferred_fraction', [null, 0.6, null]);
  expectNumbers(ledger, 'allocated_total', [0, 48, 0]);
  expectNumbers(ledger, 'unallocated_total', [120, 32, 50]);
  expectNumbers(ledger, 'validity', [0, 1, 0]);
  expect(
    Math.abs(Number(result.summary['massBalanceError'])),
    JSON.stringify(result)
  ).toBeLessThanOrEqual(1e-4);
});

test('large planar origins preserve fractional allocations and original target geometry', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const sample = createDasymetricRedistributionSample();
  const origin = 2 ** 24;
  const rectangle = (minimumX: number, maximumX: number) => ({
    type: 'Polygon',
    coordinates: [
      [
        [origin + minimumX, origin],
        [origin + maximumX, origin],
        [origin + maximumX, origin + 1],
        [origin + minimumX, origin + 1],
        [origin + minimumX, origin]
      ]
    ]
  });
  const sources: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [{id: 'large-origin', geometry: rectangle(0.5, 2.5), properties: {residents: 100}}]
  };
  const targets: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [0, 1, 2].map(index => ({id: `cell-${index}`, geometry: rectangle(index, index + 1)}))
  };
  const raster = sample.inputs['ancillary'] as CatalogRasterInput;
  const result = await runCatalogOperationAsync(
    binding,
    {...sample.parameters, targetCellOwners: [0, 1, 2]},
    {
      sources,
      targets,
      ancillary: {
        ...raster,
        grid: {...raster.grid, width: 3, affine: [1, 0, origin, 0, 1, origin]},
        band: {
          ...raster.band,
          values: new Float32Array([1, 1, 1]),
          validity: new Uint8Array([1, 1, 1])
        }
      }
    },
    {device}
  );
  expect(result.summary['complete'], JSON.stringify(result)).toBe(1);
  expectNumbers(readTable(result, 'dasymetric-targets'), 'total', [25, 50, 25]);
  expect(readTable(result, 'dasymetric-targets').columns['geometry']).toEqual(
    targets.features.map(feature => JSON.stringify(feature.geometry))
  );
});

test('valid float64 density overflow rejects before publishing a completed allocation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const sample = createDasymetricRedistributionSample();
  const raster = sample.inputs['ancillary'] as CatalogRasterInput;
  const ancillary = {
    ...raster,
    band: {...raster.band, values: new Float64Array([1e40, 0, 1, 3, 1, 0])}
  };
  await expect(
    runCatalogOperationAsync(binding, sample.parameters, {...sample.inputs, ancillary}, {device})
  ).rejects.toThrow(/finite valid density samples must remain finite in float32/);
});

test('masked support retains every source total with a complete ledger', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const sample = createDasymetricRedistributionSample();
  const raster = sample.inputs['ancillary'] as CatalogRasterInput;
  sample.inputs['ancillary'] = {...raster, band: {...raster.band, validity: new Uint8Array(6)}};
  const result = await runCatalogOperationAsync(binding, sample.parameters, sample.inputs, {
    device
  });
  expect(result.summary['complete'], JSON.stringify(result)).toBe(1);
  expectNumbers(readTable(result, 'dasymetric-targets'), 'total', [0, 0, 0, 0]);
  expectNumbers(readTable(result, 'dasymetric-ledger'), 'unallocated_total', [120, 80, 50]);
  expectNumbers(readTable(result, 'dasymetric-ledger'), 'validity', [0, 0, 0]);
});

test('fractional source coverage retains holes and multipart row identities', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const sample = createDasymetricRedistributionSample();
  const sources = sample.inputs['sources'] as CatalogFeatureCollection;
  sources.features[0].geometry = {
    type: 'Polygon',
    coordinates: [
      [
        [0, 0],
        [3, 0],
        [3, 1],
        [0, 1],
        [0, 0]
      ],
      [
        [0.25, 0.25],
        [0.25, 0.75],
        [0.75, 0.75],
        [0.75, 0.25],
        [0.25, 0.25]
      ]
    ]
  };
  sources.features[2].geometry = {
    type: 'MultiPolygon',
    coordinates: [
      [
        [
          [5, 0],
          [5.5, 0],
          [5.5, 1],
          [5, 1],
          [5, 0]
        ]
      ],
      [
        [
          [5.5, 0],
          [6, 0],
          [6, 1],
          [5.5, 1],
          [5.5, 0]
        ]
      ]
    ]
  };
  const result = await runCatalogOperationAsync(binding, sample.parameters, sample.inputs, {
    device
  });
  expect(result.summary['complete'], JSON.stringify(result)).toBe(1);
  // Source 0 support is .75*2 + 1 = 2.5; its target allocations are 72 and 48.
  expectNumbers(readTable(result, 'dasymetric-targets'), 'total', [72, 96, 0, 0]);
  expectNumbers(readTable(result, 'dasymetric-ledger'), 'eligible_weight', [2.5, 2.5, 0]);
  expectNumbers(readTable(result, 'dasymetric-ledger'), 'unallocated_total', [0, 32, 50]);
  expect(readTable(result, 'dasymetric-ledger').columns['source_id']).toEqual(
    sources.features.map(feature => feature.id)
  );
});
