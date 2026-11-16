// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterEach, expect, test, vi} from 'vitest';
import {createRequire} from 'node:module';
import type {ReactElement, ReactNode} from 'react';
import {AnalysisProfileDeclarations} from '../../../examples/deck/kepler-compute/src/analysis-profile-declarations';
import * as arrow from 'apache-arrow';
import {
  analysisDatasetContext,
  ANALYSIS_PROFILE_LIMITS,
  createAnalysisDatasetContext,
  getAnalysisDatasetRoles,
  parseAnalysisProfileDeclarations
} from '../../../examples/deck/kepler-compute/src/analysis-dataset-context';
import {makeCatalogPointColumns} from '../../../examples/deck/kepler-compute/src/analysis-catalog-input';
import {
  readCatalogFeaturesFromResidentDataset,
  makeCatalogResidentDatasets,
  type CatalogResidentDataset
} from '../../../examples/deck/kepler-compute/src/analysis-catalog-resident-input';
import {profileNativeDataset} from '../../../examples/deck/kepler-compute/src/workflow/workbench-host-data';
import {
  applySandboxExtensions,
  collectSandboxExtensions,
  SANDBOX_EXTENSION_ANALYSIS_PROFILES,
  validateSandboxExtensions
} from '../../../examples/deck/kepler-compute/src/sandbox-host-extensions';

function makeDataset() {
  let revision = 'source-1';
  let current = true;
  const values: Record<string, unknown[]> = {
    longitude: [0, 1, 2],
    latitude: [3, 4, 5],
    east: [-87.65, -87.64, -87.63],
    north: [41.88, 41.89, 41.9],
    recorded: [1700000000001, 1700000000002, 1700000000003],
    amount: [1, 4, 9]
  };
  const dataset: CatalogResidentDataset = {
    id: 'observations',
    label: 'Observations',
    revision: 1,
    owner: {},
    rowCount: 3,
    fieldTypes: Object.fromEntries(Object.keys(values).map(name => [name, 'real'])),
    coordinates: {longitude: 'east', latitude: 'north'},
    timeColumns: ['recorded'],
    valueColumn: 'amount',
    provider: {
      fields: Object.keys(values).map(name => ({name, sortable: true})),
      maximumPageSize: 100,
      getRevision: () => revision,
      readPage: vi.fn(async () => {
        throw new Error('Unexpected page read');
      }),
      readColumnValues: vi.fn(async ({column}) => ({values: values[column], fieldType: 'real'}))
    },
    isCurrent: () => current
  };
  return {
    dataset,
    values,
    changeRevision: () => {
      revision = 'source-2';
    },
    remove: () => {
      current = false;
    }
  };
}

afterEach(() => {
  analysisDatasetContext.synchronize([]);
  analysisDatasetContext.restoreDeclarations(undefined);
});

test('metadata is immediate, uses publication roles before names, preserves time types and performs no readback', () => {
  const {dataset} = makeDataset();
  const context = createAnalysisDatasetContext();
  context.synchronize([dataset]);
  const entry = context.get(dataset.id)!;
  expect(entry.stage).toBe('metadata');
  expect(entry.profile.pointColumns).toEqual({longitude: 'east', latitude: 'north'});
  expect(entry.profile.columns.find(column => column.name === 'recorded')).toEqual({
    name: 'recorded',
    type: 'time',
    rowCount: 3,
    nullCount: 0,
    sampleSize: 0
  });
  expect(entry.profile.columns.every(column => column.minimum === undefined)).toBe(true);
  context.synchronize([{...dataset}]);
  expect(context.get(dataset.id)).toBe(entry);
  expect(dataset.provider.readColumnValues).not.toHaveBeenCalled();
  expect(dataset.provider.readPage).not.toHaveBeenCalled();
});

test('bounded refinement preserves original time and user declarations, and shares the adapter with catalog runs', async () => {
  const {dataset} = makeDataset();
  const context = createAnalysisDatasetContext();
  context.synchronize([dataset]);
  const reading = context.readStatistics(dataset.id);
  context.setDeclarations(dataset.id, {
    rowSemantics: 'surface-samples',
    coordinateModel: 'geographic',
    measurement: {amount: 'amount'}
  });
  const profile = await reading;
  expect(profile.pointColumns).toEqual(dataset.coordinates);
  expect(profile.columns.find(column => column.name === 'recorded')?.minimum).toBe(1700000000001);
  expect(profile.columns.find(column => column.name === 'amount')?.measurementSource).toBe(
    'declared'
  );
  expect(profile.rowSemantics?.value).toBe('surface-samples');
  expect(context.get(dataset.id)?.stage).toBe('statistics');
  const features = await readCatalogFeaturesFromResidentDataset(
    dataset,
    makeCatalogPointColumns('', '')
  );
  expect(features.features[0].geometry?.coordinates).toEqual([-87.65, 41.88]);
});

test('same-id same-revision replacement, revision changes and removal invalidate profiles while declarations survive', async () => {
  const fixture = makeDataset();
  const context = createAnalysisDatasetContext();
  context.synchronize([fixture.dataset]);
  context.setDeclarations(fixture.dataset.id, {rowSemantics: 'events'});
  await context.readStatistics(fixture.dataset.id);
  const previous = context.get(fixture.dataset.id);
  context.synchronize([{...fixture.dataset, owner: {}}]);
  expect(context.get(fixture.dataset.id)).not.toBe(previous);
  expect(context.get(fixture.dataset.id)?.stage).toBe('metadata');
  fixture.changeRevision();
  context.synchronize([fixture.dataset]);
  expect(context.get(fixture.dataset.id)?.profile.rowSemantics?.source).toBe('declared');
  context.synchronize([]);
  expect(context.get(fixture.dataset.id)).toBeUndefined();
  context.synchronize([fixture.dataset]);
  expect(context.get(fixture.dataset.id)?.profile.rowSemantics?.value).toBe('events');
  fixture.remove();
  expect(context.get(fixture.dataset.id)).toBeUndefined();
});

test.each([
  'revision',
  'replacement',
  'removal'
])('in-flight %s cannot install statistics into a newer context', async change => {
  const fixture = makeDataset();
  const context = createAnalysisDatasetContext();
  context.synchronize([fixture.dataset]);
  const reading = context.readStatistics(fixture.dataset.id);
  if (change === 'revision') fixture.changeRevision();
  if (change === 'removal') fixture.remove();
  context.synchronize(change === 'removal' ? [] : [{...fixture.dataset, owner: {}}]);
  await expect(reading).rejects.toThrow(/changed|stale/);
  expect(context.get(fixture.dataset.id)?.stage).not.toBe('statistics');
});

test('workbench resident metadata comes from the same service without reading native cells', () => {
  const {dataset} = makeDataset();
  analysisDatasetContext.synchronize([dataset]);
  const valueAt = vi.fn(() => {
    throw new Error('Resident cell read forbidden');
  });
  const profile = profileNativeDataset(
    {
      id: dataset.id,
      label: dataset.label,
      fields: Object.entries(dataset.fieldTypes!).map(([name, type]) => ({name, type})),
      dataContainer: {numRows: () => 3, valueAt}
    },
    true
  );
  expect(profile.pointColumns).toEqual(dataset.coordinates);
  expect(valueAt).not.toHaveBeenCalled();
});

test('unsupported original readback and excessive rows remain explicit metadata-only limitations', async () => {
  const context = createAnalysisDatasetContext();
  const {dataset} = makeDataset();
  context.synchronize([{...dataset, provider: {...dataset.provider, readColumnValues: undefined}}]);
  await expect(context.readStatistics(dataset.id)).rejects.toThrow('no original column readback');
  expect(context.get(dataset.id)?.stage).toBe('unavailable');
  context.synchronize([{...dataset, rowCount: ANALYSIS_PROFILE_LIMITS.maximumRows + 1}]);
  await expect(context.readStatistics(dataset.id)).rejects.toThrow('capacity');
  expect(dataset.provider.readColumnValues).not.toHaveBeenCalled();
});

test('large populations retain sampled evidence; geometry and readback caps are enforced', async () => {
  const {dataset, values} = makeDataset();
  for (const name of Object.keys(values))
    values[name] = Array.from({length: 600}, (_, index) =>
      name === 'east' ? -87 + index / 10000 : name === 'north' ? 41 + index / 10000 : index
    );
  const context = createAnalysisDatasetContext();
  context.synchronize([{...dataset, rowCount: 600}]);
  const profile = await context.readStatistics(dataset.id);
  expect(profile.columns[0].rowCount).toBe(600);
  expect(profile.columns[0].sampleSize).toBe(512);
  expect(profile.spacing?.sampled).toBe(true);

  const geometry = JSON.stringify({
    type: 'LineString',
    coordinates: Array.from({length: 200001}, () => [1, 2])
  });
  const geometryDataset = {
    ...dataset,
    coordinates: undefined,
    geometryColumn: 'shape',
    provider: {
      ...dataset.provider,
      fields: [{name: 'shape', sortable: true}],
      readColumnValues: vi.fn(async () => ({values: [geometry, null, null], fieldType: 'string'}))
    }
  };
  context.synchronize([geometryDataset]);
  await expect(context.readStatistics(dataset.id)).rejects.toThrow('200000 vertices');
  context.synchronize([
    {
      ...geometryDataset,
      owner: {},
      provider: {
        ...geometryDataset.provider,
        readColumnValues: vi.fn(async () => ({
          values: ['x'.repeat(34 * 1024 * 1024), null, null],
          fieldType: 'string'
        }))
      }
    }
  ]);
  await expect(context.readStatistics(dataset.id)).rejects.toThrow('67108864 bytes');
});

test('sandbox extension validates atomically, saves and replays declarations, and clears missing members', () => {
  const {dataset} = makeDataset();
  analysisDatasetContext.synchronize([dataset]);
  analysisDatasetContext.setDeclarations(dataset.id, {
    rowSemantics: 'objects',
    coordinateModel: 'planar',
    measurement: {amount: 'rate'}
  });
  const extensions = collectSandboxExtensions([])!;
  expect(
    parseAnalysisProfileDeclarations(extensions[SANDBOX_EXTENSION_ANALYSIS_PROFILES]).get(
      dataset.id
    )
  ).toEqual({
    rowSemantics: 'objects',
    coordinateModel: 'planar',
    measurement: {amount: 'rate'}
  });
  analysisDatasetContext.setDeclarations(dataset.id, {rowSemantics: 'events'});
  applySandboxExtensions(extensions, vi.fn());
  expect(analysisDatasetContext.get(dataset.id)?.profile.rowSemantics?.value).toBe('objects');
  expect(analysisDatasetContext.get(dataset.id)?.profile.coordinates?.model).toBe('planar');
  expect(() =>
    validateSandboxExtensions({
      [SANDBOX_EXTENSION_ANALYSIS_PROFILES]: {
        schemaVersion: 1,
        datasets: {observations: '{"version":1,"measurement":{"amount":"bogus"}}'}
      }
    })
  ).toThrow(/measurement.amount/);
  expect(analysisDatasetContext.getDeclarations(dataset.id)?.rowSemantics).toBe('objects');
  applySandboxExtensions(undefined, vi.fn());
  expect(analysisDatasetContext.getDeclarations(dataset.id)).toBeUndefined();
});

test('saved declarations rebind by dataset id before metadata exists, including prototype-like names', () => {
  const context = createAnalysisDatasetContext();
  const document = {
    schemaVersion: 1,
    datasets: Object.fromEntries([
      ['__proto__', '{"version":1,"measurement":{"amount":"count"},"rowSemantics":"events"}']
    ])
  };
  context.restoreDeclarations(document);
  const {dataset} = makeDataset();
  context.synchronize([{...dataset, id: '__proto__'}]);
  expect(context.get('__proto__')?.profile.rowSemantics?.value).toBe('events');
  expect(context.collectDeclarations()).toEqual(document);
  expect(() =>
    context.restoreDeclarations({schemaVersion: 1, datasets: {good: '{"version":1}', bad: '{}'}})
  ).toThrow();
  expect(context.collectDeclarations()).toEqual(document);
});

test('publication value/time/geometry roles outrank AX guesses, and removed declarations return to guesses', async () => {
  const {dataset, values} = makeDataset();
  values['value'] = [100, 200, 300];
  dataset.provider.fields = [...dataset.provider.fields, {name: 'value', sortable: true}];
  dataset.fieldTypes = {...dataset.fieldTypes, value: 'real'};
  const context = createAnalysisDatasetContext();
  context.synchronize([dataset]);
  expect(getAnalysisDatasetRoles(context.get(dataset.id)!)).toMatchObject({
    valueColumn: 'amount',
    valueSource: 'declared',
    timeColumn: 'recorded',
    timeSource: 'declared'
  });
  const reading = context.readStatistics(dataset.id);
  expect(context.readStatistics(dataset.id)).toBe(reading);
  await reading;
  context.setDeclarations(dataset.id, {measurement: {amount: 'rate'}, coordinateModel: 'planar'});
  expect(context.get(dataset.id)?.profile.coordinates?.source).toBe('declared');
  context.setDeclarations(dataset.id, undefined);
  expect(context.get(dataset.id)?.profile.coordinates?.source).toBe('guess');
  context.synchronize([
    {
      ...dataset,
      valueColumn: undefined,
      geometryColumn: 'value',
      geometryKind: 'polygon',
      coordinates: undefined
    }
  ]);
  expect(context.get(dataset.id)?.profile.kind).toBe('polygons');
  expect(context.get(dataset.id)?.profile.geometryColumns).toEqual(['value']);
  expect(context.get(dataset.id)?.profile.pointColumns).toBeUndefined();
});

test('host carries the selected analysis value ahead of publication and name defaults', () => {
  const {dataset} = makeDataset();
  const snapshot = {
    id: dataset.id,
    revision: 1,
    value: {source: {numRows: 3}, prepared: {timeColumns: {recorded: {}}}}
  };
  const publication = {
    dataset: snapshot,
    label: dataset.label,
    coordinates: {...dataset.coordinates, flow: {count: 'count'}},
    analysis: {settings: {column: 'amount'}},
    consumer: {table: dataset.provider, explorer: {isCurrent: true}}
  };
  const runtime = {
    explorer: {dataset: {...snapshot, id: 'primary'}, isCurrent: true},
    analysis: {settings: {column: 'chosen'}},
    nativeTable: dataset.provider,
    publications: new Map([[dataset.id, publication]])
  };
  const inputs = makeCatalogResidentDatasets(runtime as never, () => runtime as never);
  expect(inputs[0].valueColumn).toBe('chosen');
  expect(inputs[1].valueColumn).toBe('amount');
  expect(inputs[1].timeColumns).toEqual(['recorded']);
  expect(inputs[1].coordinates).toMatchObject(dataset.coordinates!);
});

function makeProfileColumnsDataset(
  columns: Record<string, unknown[]>,
  fieldTypes: Record<string, string> = {}
): CatalogResidentDataset {
  return {
    id: 'profile-review',
    label: 'Profile review',
    rowCount: Object.values(columns)[0].length,
    revision: 1,
    owner: {},
    fieldTypes,
    provider: {
      fields: Object.keys(columns).map(name => ({name, sortable: true})),
      maximumPageSize: 100,
      getRevision: () => 'review-1',
      readPage: vi.fn(async () => {
        throw new Error('Unexpected page read');
      }),
      readColumnValues: vi.fn(async ({column}) => ({
        values: columns[column],
        fieldType: fieldTypes[column] ?? 'string'
      }))
    },
    isCurrent: () => true
  };
}

const REVIEW_POLYGON = {
  type: 'Polygon',
  coordinates: [
    [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 0]
    ]
  ]
};
const REVIEW_FEATURE = {type: 'Feature', properties: {}, geometry: REVIEW_POLYGON};

test.each([
  JSON.stringify(REVIEW_FEATURE),
  REVIEW_FEATURE
])('review 1: Feature cells retain catalog geometry and extent during profiling (%j)', async value => {
  const dataset = {
    ...makeProfileColumnsDataset({shape: [value, null]}, {shape: 'string'}),
    geometryColumn: 'shape'
  };
  const context = createAnalysisDatasetContext();
  context.synchronize([dataset]);
  const profile = await context.readStatistics(dataset.id);
  const catalog = await readCatalogFeaturesFromResidentDataset(dataset, 'shape');
  expect(catalog.features[0].geometry).toEqual(REVIEW_POLYGON);
  expect(profile).toMatchObject({
    kind: 'polygons',
    catalogGeometry: 'polygon',
    vertexCount: 4,
    extent: [0, 0, 1, 1]
  });
});

test('review 2: storage strings retain AX category and ISO time inference', async () => {
  const dataset = makeProfileColumnsDataset(
    {
      zone: ['a', 'a', 'b', 'b'],
      recorded: [
        '2026-11-14T12:00:00Z',
        '2026-11-14T12:01:00Z',
        '2026-11-14T12:02:00Z',
        '2026-11-14T12:03:00Z'
      ]
    },
    {zone: 'string', recorded: 'string'}
  );
  const context = createAnalysisDatasetContext();
  context.synchronize([dataset]);
  const profile = await context.readStatistics(dataset.id);
  expect(profile.columns[0]).toMatchObject({
    name: 'zone',
    type: 'category',
    distinctCount: 2,
    nullCount: 0
  });
  expect(profile.columns[1]).toMatchObject({
    name: 'recorded',
    type: 'time',
    nullCount: 0,
    minimum: Date.parse('2026-11-14T12:00:00Z'),
    maximum: Date.parse('2026-11-14T12:03:00Z')
  });
  expect(getAnalysisDatasetRoles(context.get(dataset.id)!).timeColumn).toBe('recorded');
});

test('review 3: derived original-view fields absent from the source schema infer numeric statistics', async () => {
  const dataset = makeProfileColumnsDataset({source: ['a', 'b']}, {source: 'string'});
  dataset.provider.originalFields = [
    ...dataset.provider.fields,
    {name: 'analysis_score', sortable: true, derived: true}
  ];
  dataset.provider.readPage = vi.fn(async () => ({
    datasetId: dataset.id,
    datasetRevision: 1,
    revision: 1,
    totalCount: 2,
    sourceRowIndices: new Uint32Array([0, 1]),
    columns: {analysis_score: new Float32Array([1.5, 2.5])},
    validity: {}
  }));
  const context = createAnalysisDatasetContext();
  context.synchronize([dataset]);
  const profile = await context.readStatistics(dataset.id);
  expect(dataset.provider.readPage).toHaveBeenCalledWith({
    offset: 0,
    limit: 2,
    population: 'all',
    columns: ['analysis_score'],
    view: 'original'
  });
  expect(profile.columns.find(column => column.name === 'analysis_score')).toMatchObject({
    type: 'number',
    nullCount: 0,
    minimum: 1.5,
    maximum: 2.5,
    mean: 2
  });
});

test('review 4: dictionary numeric values retain their original Arrow storage type in the host', async () => {
  const source = new arrow.Table({
    measure: arrow.vectorFromArray(
      [5, 6],
      new arrow.Dictionary(new arrow.Float64(), new arrow.Int32())
    )
  });
  const dataset = makeProfileColumnsDataset({measure: [5, 6]});
  const snapshot = {id: dataset.id, revision: 1, value: {source, prepared: {timeColumns: {}}}};
  const publication = {
    dataset: snapshot,
    label: dataset.label,
    consumer: {table: dataset.provider, explorer: {isCurrent: true}}
  };
  const runtime = {
    explorer: {dataset: {...snapshot, id: 'primary'}, isCurrent: true},
    nativeTable: dataset.provider,
    publications: new Map([[dataset.id, publication]])
  };
  const resident = makeCatalogResidentDatasets(runtime as never, () => runtime as never)[1];
  expect(resident.fieldTypes?.['measure']).toBe('real');
  const context = createAnalysisDatasetContext();
  context.synchronize([resident]);
  const profile = await context.readStatistics(resident.id);
  expect(profile.columns[0]).toMatchObject({type: 'number', nullCount: 0, minimum: 5, maximum: 6});
});

test('review 5: metadata coordinate guesses are validated against source values; only publication roles are explicit', async () => {
  const dataset = makeProfileColumnsDataset(
    {longitude: [500000, 500001], latitude: [4500000, 4500001]},
    {longitude: 'real', latitude: 'real'}
  );
  const context = createAnalysisDatasetContext();
  context.synchronize([dataset]);
  expect(context.get(dataset.id)?.profile.pointColumns).toEqual({
    longitude: 'longitude',
    latitude: 'latitude'
  });
  const profile = await context.readStatistics(dataset.id);
  expect(profile.kind).toBe('attributes');
  expect(profile.pointColumns).toBeUndefined();
  expect(profile).toMatchObject({
    notes: expect.arrayContaining([expect.stringMatching(/rejected as coordinates.*outside/)])
  });
  context.synchronize([{...dataset, coordinates: {longitude: 'longitude', latitude: 'latitude'}}]);
  context.setDeclarations(dataset.id, {coordinateModel: 'planar'});
  const declared = await context.readStatistics(dataset.id);
  expect(declared.kind).toBe('points');
  expect(declared.coordinates).toMatchObject({model: 'planar', source: 'declared'});
});

test('review round 2 finding 1: a confirmed planar pair survives statistics and saved declaration replay', async () => {
  const dataset = makeProfileColumnsDataset(
    {longitude: [500000, 500001], latitude: [4500000, 4500001]},
    {longitude: 'real', latitude: 'real'}
  );
  const context = createAnalysisDatasetContext();
  context.synchronize([dataset]);
  context.setDeclarations(dataset.id, {coordinateModel: 'planar'});
  const expectedColumns = {longitude: 'longitude', latitude: 'latitude'};
  expect(context.get(dataset.id)?.profile.pointColumns).toEqual(expectedColumns);
  expect(context.get(dataset.id)?.profile.coordinates).toMatchObject({
    model: 'planar',
    source: 'declared'
  });
  const saved = context.collectDeclarations();
  for (const current of [context, createAnalysisDatasetContext()]) {
    if (current !== context) {
      current.restoreDeclarations(saved);
      current.synchronize([dataset]);
    }
    const profile = await current.readStatistics(dataset.id);
    expect(profile.kind).toBe('points');
    expect(profile.pointColumns).toEqual(expectedColumns);
    expect(profile.coordinates).toMatchObject({model: 'planar', source: 'declared'});
    expect(profile.extent).toEqual([500000, 4500000, 500001, 4500001]);
    expect(profile.spacing).toMatchObject({
      coordinateModel: 'planar',
      unit: 'coordinate-unit',
      pointCount: 2
    });
  }
});

test('review round 2 finding 3: synchronization publishes actual lifecycle changes exactly once per batch', () => {
  const fixture = makeDataset();
  const context = createAnalysisDatasetContext();
  const observed: number[] = [];
  const unsubscribe = context.subscribe(() => {
    observed.push(context.getVersion());
  });
  context.synchronize([fixture.dataset]);
  expect(observed).toEqual([1]);
  context.synchronize([{...fixture.dataset}]);
  expect(observed).toEqual([1]);
  const replacement = {...fixture.dataset, owner: {}};
  context.synchronize([replacement]);
  expect(observed).toEqual([1, 2]);
  fixture.changeRevision();
  context.synchronize([replacement]);
  expect(observed).toEqual([1, 2, 3]);
  fixture.remove();
  context.synchronize([replacement]);
  expect(observed).toEqual([1, 2, 3, 4]);
  context.synchronize([]);
  expect(observed).toEqual([1, 2, 3, 4]);
  const second = makeDataset().dataset;
  context.synchronize([second, {...second, id: 'second'}]);
  expect(observed).toEqual([1, 2, 3, 4, 5]);
  context.synchronize([]);
  expect(observed).toEqual([1, 2, 3, 4, 5, 6]);
  unsubscribe();
  context.synchronize([second]);
  expect(context.getVersion()).toBe(7);
  expect(observed).toHaveLength(6);
  expect(fixture.dataset.provider.readColumnValues).not.toHaveBeenCalled();
});

const profileHostRequire = createRequire(
  new URL('../../../examples/deck/kepler-compute/package.json', import.meta.url)
);
const profileReact = profileHostRequire('react') as typeof import('react') & {
  __CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE: {H: unknown};
};

function findProfileElements(node: ReactNode): ReactElement<Record<string, unknown>>[] {
  if (Array.isArray(node)) return node.flatMap(findProfileElements);
  if (!profileReact.isValidElement<Record<string, unknown>>(node)) return [];
  return [node, ...findProfileElements(node.props['children'] as ReactNode)];
}

function renderProfileControls(dataset: CatalogResidentDataset) {
  // Exercise the actual select handlers using the same small dispatcher probe as workbench UI specs.
  const internals = profileReact.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;
  const previous = internals.H;
  internals.H = {
    useSyncExternalStore: (_subscribe: unknown, getSnapshot: () => unknown) => getSnapshot()
  };
  try {
    return findProfileElements(AnalysisProfileDeclarations({datasets: [dataset]}));
  } finally {
    internals.H = previous;
  }
}

test.each([
  '__proto__',
  'constructor',
  'toString'
])('review round 2 finding 4: measurement UI saves, replays and removes own column %s', columnName => {
  const dataset = makeProfileColumnsDataset(
    Object.fromEntries([[columnName, [1, 2]]]),
    Object.fromEntries([[columnName, 'real']])
  );
  analysisDatasetContext.synchronize([dataset]);
  const getControl = () => {
    const element = renderProfileControls(dataset).find(
      control => control.props['aria-label'] === `Measurement of ${columnName} for ${dataset.label}`
    );
    expect(element).toBeDefined();
    return element!.props as {value: unknown; onChange(event: {target: {value: string}}): void};
  };
  expect(getControl().value).toBe('');
  getControl().onChange({target: {value: 'count'}});
  const measurement = analysisDatasetContext.getDeclarations(dataset.id)!.measurement!;
  expect(Object.hasOwn(measurement, columnName)).toBe(true);
  expect(measurement[columnName]).toBe('count');
  const saved = analysisDatasetContext.collectDeclarations();
  analysisDatasetContext.restoreDeclarations(undefined);
  analysisDatasetContext.restoreDeclarations(saved);
  expect(getControl().value).toBe('count');
  const profileColumn = analysisDatasetContext.get(dataset.id)!.profile.columns[0];
  expect(profileColumn).toMatchObject({
    name: columnName,
    measurement: 'count',
    measurementSource: 'declared'
  });
  getControl().onChange({target: {value: ''}});
  expect(
    Object.hasOwn(analysisDatasetContext.getDeclarations(dataset.id)!.measurement!, columnName)
  ).toBe(false);
  expect(getControl().value).toBe('');
});

test('review round 2 finding 1: changing planar confirmation invalidates previously refined name guesses', async () => {
  const dataset = makeProfileColumnsDataset(
    {longitude: [500000, 500001], latitude: [4500000, 4500001]},
    {longitude: 'real', latitude: 'real'}
  );
  const context = createAnalysisDatasetContext();
  context.synchronize([dataset]);
  expect((await context.readStatistics(dataset.id)).kind).toBe('attributes');
  context.setDeclarations(dataset.id, {coordinateModel: 'planar'});
  expect(context.get(dataset.id)?.stage).toBe('metadata');
  expect((await context.readStatistics(dataset.id)).kind).toBe('points');
  const saved = context.collectDeclarations();
  context.setDeclarations(dataset.id, undefined);
  expect(context.get(dataset.id)?.stage).toBe('metadata');
  expect((await context.readStatistics(dataset.id)).kind).toBe('attributes');
  context.restoreDeclarations(saved);
  expect(context.get(dataset.id)?.stage).toBe('metadata');
  expect((await context.readStatistics(dataset.id)).kind).toBe('points');
});
