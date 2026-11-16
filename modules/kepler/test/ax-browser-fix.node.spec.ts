// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, test} from 'vitest';
import {getCatalogBinding, parseCatalogFeatureCollection, runCatalogOperation} from '../src/index';
import {decodeCsvToArrowTable} from '../../../examples/deck/kepler-compute/src/remote-source-loader';
import {readCatalogFeaturesFromResidentDataset} from '../../../examples/deck/kepler-compute/src/analysis-catalog-resident-input';
import {withOriginalSourceTable} from '../../../examples/deck/kepler-compute/src/resident-table-provider';
import {createAnalysisSheetModel} from '../../../examples/deck/kepler-compute/src/analysis-sheet-model';
import {SPACETIME_BINDINGS} from '../src/analysis/catalog-spacetime-bindings';
import {makeCatalogPointColumns} from '../../../examples/deck/kepler-compute/src/analysis-catalog-input';
import {getCatalogSampleInputs} from '../../../examples/deck/kepler-compute/src/analysis-catalog-input';
import {
  createAnalysisCatalogFormModel,
  profileAnalysisCatalogInput,
  resolveAnalysisCatalogInputProfiles
} from '../../../examples/deck/kepler-compute/src/analysis-catalog-form-model';
import {NETWORK_CATEGORICAL_OPERATIONS} from '../../../examples/deck/kepler-compute/scripts/network-categorical-catalog-utils.mjs';
import {
  VALIDATION_OPERATIONS,
  VALIDATION_SAMPLE_FIELDS
} from '../../../examples/deck/kepler-compute/scripts/analysis-catalog-validation.mjs';

for (const [operationId] of [...NETWORK_CATEGORICAL_OPERATIONS, ...VALIDATION_OPERATIONS]) {
  test(`${operationId}: authored catalog sample passes the actual default preflight`, () => {
    const binding = getCatalogBinding(operationId)!;
    const selected = resolveAnalysisCatalogInputProfiles(binding, {}, () => undefined);
    const model = createAnalysisCatalogFormModel(binding, selected.profiles, {
      executionEngine: 'gpu',
      overrides: Object.fromEntries(
        Object.entries(VALIDATION_SAMPLE_FIELDS).filter(
          ([name]) =>
            VALIDATION_OPERATIONS.some(([identifier]) => identifier === operationId) &&
            binding.parameters.some(parameter => parameter.name === name)
        )
      ),
      inputProblems: selected.problems
    });
    expect(model.problems).toEqual([]);
    expect(model.missing).toEqual([]);
    expect(model.runDisabled).toBe(false);
  });
}

for (const operationId of [
  'hotspot-gi-star',
  'local-moran-statistics',
  'global-spatial-statistics',
  'recipe-points-in-polygons-choropleth'
]) {
  test(`${operationId}: publication sample runs with the exact preflight parameters`, () => {
    const binding = getCatalogBinding(operationId)!;
    const selected = resolveAnalysisCatalogInputProfiles(
      binding,
      {},
      () => undefined,
      undefined,
      'planar'
    );
    const model = createAnalysisCatalogFormModel(binding, selected.profiles, {
      executionEngine: 'cpu',
      overrides: {coordinateSystem: 'planar'},
      inputProblems: selected.problems
    });
    expect(model.problems).toEqual([]);
    expect(model.missing).toEqual([]);
    expect(model.runDisabled).toBe(false);
    const samples = getCatalogSampleInputs(operationId);
    const inputs = Object.fromEntries(
      binding.inputs.map(input => [input.name, parseCatalogFeatureCollection(samples[input.name])])
    );
    const result = runCatalogOperation(binding, model.resolved.parameters, inputs);
    expect(result.tables.length).toBeGreaterThan(0);
    expect(result.tables.every(table => table.rowCount > 0)).toBe(true);
    if (operationId !== 'recipe-points-in-polygons-choropleth')
      expect(model.resolved.parameters['valueProperty']).toBe('rate');
  });
}

for (const [operationId, coordinateSystem] of [
  ['geometry-overlay-intersection', 'EPSG:4326'],
  ['network-crossing-noding', 'EPSG:4326'],
  ['geometry-convex-clip', 'planar']
] as const) {
  test(`${operationId}: the catalog journey uses the known fixture coordinate model`, () => {
    const binding = getCatalogBinding(operationId)!;
    const hasCoordinateParameter = binding.parameters.some(
      parameter => parameter.name === 'coordinateSystem'
    );
    const selected = resolveAnalysisCatalogInputProfiles(
      binding,
      {},
      () => undefined,
      undefined,
      hasCoordinateParameter ? coordinateSystem : undefined
    );
    const model = createAnalysisCatalogFormModel(binding, selected.profiles, {
      executionEngine: 'gpu',
      overrides: hasCoordinateParameter ? {coordinateSystem} : {},
      inputProblems: selected.problems
    });
    expect(model.problems).toEqual([]);
    expect(model.missing).toEqual([]);
    expect(model.runDisabled).toBe(false);
  });
}

test('network input checking refines held GeoJSON without certifying changed topology or engine', () => {
  const binding = getCatalogBinding('network-service-area')!;
  const overrides = {
    coordinateSystem: 'planar',
    onewayProperty: 'oneway',
    budgets: [15],
    maximumSnapDistance: 1
  };
  const selected = resolveAnalysisCatalogInputProfiles(
    binding,
    {},
    () => undefined,
    undefined,
    'planar'
  );
  const before = createAnalysisCatalogFormModel(binding, selected.profiles, {
    overrides,
    executionEngine: 'gpu'
  });
  expect(before.runDisabled).toBe(true);
  expect(before.missing.map(item => item.what).join(' ')).toContain('physical edge count');
  const samples = getCatalogSampleInputs(binding.id);
  const checked = Object.fromEntries(
    binding.inputs.map(input => [
      input.name,
      profileAnalysisCatalogInput(
        binding,
        input.name,
        parseCatalogFeatureCollection(samples[input.name]),
        {parameters: before.resolved.parameters, executionEngine: 'gpu'}
      )
    ])
  );
  const refined = resolveAnalysisCatalogInputProfiles(
    binding,
    {},
    () => undefined,
    checked,
    'planar'
  );
  const after = createAnalysisCatalogFormModel(binding, refined.profiles, {
    overrides,
    executionEngine: 'gpu'
  });
  expect(after.missing).toEqual([]);
  expect(after.runDisabled).toBe(false);
  expect(refined.profiles['lines']?.network).toMatchObject({
    nodeCount: 9,
    edgeCount: 8,
    arcCount: 12
  });
  const result = runCatalogOperation(
    binding,
    after.resolved.parameters,
    Object.fromEntries(
      binding.inputs.map(input => [input.name, parseCatalogFeatureCollection(samples[input.name])])
    )
  );
  expect(result.summary['reachedEdgeCount']).toBe(3);
  for (const changed of [
    {splitAtSharedVertices: false},
    {snapTolerance: 1},
    {crossings: 'noded'}
  ]) {
    const stale = createAnalysisCatalogFormModel(binding, refined.profiles, {
      overrides: {...overrides, ...changed},
      executionEngine: 'gpu'
    });
    expect(stale.runDisabled).toBe(true);
    expect(stale.missing.map(item => item.what).join(' ')).toContain('physical edge count');
  }
});

test('pasted OD guesses warn without inheriting authored sample declarations', () => {
  const binding = getCatalogBinding('kernel-flow-smoothing')!;
  const samples = getCatalogSampleInputs(binding.id);
  const drafts = Object.fromEntries(
    binding.inputs.map(input => [
      `${binding.id}:${input.name}`,
      {
        mode: 'text' as const,
        text: samples[input.name],
        datasetId: '',
        field: ''
      }
    ])
  );
  const selected = resolveAnalysisCatalogInputProfiles(binding, drafts, () => undefined);
  const unknown = createAnalysisCatalogFormModel(binding, selected.profiles);
  expect(unknown.runDisabled).toBe(false);
  expect(unknown.applicability.warnings.length).toBeGreaterThan(0);
  expect(
    selected.profiles['flows']?.columns.find(column => column.name === 'origin')?.measurementSource
  ).not.toBe('declared');
  const confirmed = resolveAnalysisCatalogInputProfiles(
    binding,
    drafts,
    () => undefined,
    undefined,
    'EPSG:4326'
  );
  const mapped = createAnalysisCatalogFormModel(binding, confirmed.profiles, {
    overrides: {
      coordinateSystem: 'EPSG:4326',
      originProperty: 'origin',
      destinationProperty: 'destination'
    }
  });
  expect(mapped.missing).toEqual([]);
  expect(mapped.runDisabled).toBe(false);
});

test('kernel density exposes a retained wave-one graph with live bandwidth and bounded admission', () => {
  const binding = getCatalogBinding('kernel-density-grid')!;
  const inputs = {
    points: {
      type: 'FeatureCollection' as const,
      features: [
        {geometry: {type: 'Point' as const, coordinates: [0, 0]}},
        {geometry: {type: 'Point' as const, coordinates: [10, 10]}}
      ]
    }
  };
  const parameters = {
    coordinateSystem: 'planar',
    cellSize: 2,
    bandwidth: 3,
    padding: 0,
    kernel: 'gaussian',
    weightProperty: '',
    includeEmptyCells: true
  };
  const first = binding.planPreview!(inputs, parameters);
  const live = binding.planPreview!(inputs, {
    ...parameters,
    bandwidth: 4,
    includeEmptyCells: false
  });
  expect(live.key).toBe(first.key);
  expect(live.gpuByteLength).toBe(first.gpuByteLength);
  expect(first.gpuByteLength).toBeGreaterThan(25 * 4 * 2);
  expect(first.cpuByteLength).toBeGreaterThan(25 * 4 * 2);
  const rebuilt = binding.planPreview!(inputs, {...parameters, cellSize: 1});
  expect(rebuilt.key).not.toBe(first.key);
  expect(rebuilt.gpuByteLength).toBeGreaterThan(first.gpuByteLength);
});

test('the Prepare data CSV path offers the fixture numeric epochs to every spacetime sheet', async () => {
  const fixture = readFileSync(
    new URL(
      '../../../examples/deck/kepler-compute/data/spacetime-observations.csv',
      import.meta.url
    ),
    'utf8'
  ).replace(/^#.*\r?\n/gm, '');
  const source = await decodeCsvToArrowTable(new TextEncoder().encode(fixture));
  expect(source.numRows).toBe(48);
  expect(source.getChild('event_seconds')!.type.toString()).toBe('Int32');
  const fields = source.schema.fields.map(field => ({name: field.name, sortable: true}));
  const provider = withOriginalSourceTable(
    {
      datasetId: 'csv-fixture',
      sourceRevision: 1,
      fields,
      maximumPageSize: 512,
      getRevision: () => '1',
      subscribe: () => () => {},
      readPage: async () => {
        throw new Error('Numeric epochs must use the original source');
      }
    },
    source,
    () => true
  );
  const collection = await readCatalogFeaturesFromResidentDataset(
    {
      id: 'csv-fixture',
      label: 'spacetime-observations.csv',
      rowCount: source.numRows,
      provider,
      isCurrent: () => true
    },
    makeCatalogPointColumns('longitude', 'latitude'),
    512
  );
  expect(collection.features.map(feature => feature.properties?.['event_seconds'])).toEqual(
    Array.from(source.getChild('event_seconds')!)
  );
  for (const binding of SPACETIME_BINDINGS) {
    const profile = profileAnalysisCatalogInput(binding, 'points', collection);
    const form = createAnalysisCatalogFormModel(
      binding,
      {points: profile},
      {
        overrides: {timeUnit: 'seconds', timeProperty: 'event_seconds'}
      }
    );
    expect(form.runDisabled, `${binding.id}: ${JSON.stringify(form.missing)}`).toBe(false);
    const sheet = createAnalysisSheetModel(binding, form, {points: profile});
    expect(
      sheet.rows.find(row => row.specification.name === 'timeProperty')?.columns,
      binding.id
    ).toContain('event_seconds');
    expect(form.resolved.parameters['timeProperty'], binding.id).toBe('event_seconds');
    expect(
      form.applicability.missing.filter(item => item.what.includes('event_seconds')),
      binding.id
    ).toEqual([]);
  }
});
