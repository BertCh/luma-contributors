// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {afterEach, expect, test, vi} from 'vitest';
import * as arrow from 'apache-arrow';
import {NullDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession, type KeplerSessionExplorer} from '../src/session/compute-session';
import {
  analysisDatasetContext,
  createAnalysisDatasetContext
} from '../../../examples/deck/kepler-compute/src/analysis-dataset-context';
import {
  editAnalysisFormDraft,
  getAnalysisFormInitialization,
  getAnalysisLatticeShape,
  reconcileAnalysisFormDraft,
  useAnalysisFormDraft,
  type AnalysisFormDraftState
} from '../../../examples/deck/kepler-compute/src/analysis-form-initialization';
import type {CatalogResidentDataset} from '../../../examples/deck/kepler-compute/src/analysis-catalog-resident-input';
import {AnalysisRuntime} from '../../../examples/deck/kepler-compute/src/analysis-runtime';
import {getAnalysisRecipeColumns} from '../../../examples/deck/kepler-compute/src/analysis-recipe-settings';
import {usePointCoordinates} from '../../../examples/deck/kepler-compute/src/analysis-point-coordinate-controls';
import {InterpolationPanel} from '../../../examples/deck/kepler-compute/src/interpolation-panel';
import {KrigingPanel} from '../../../examples/deck/kepler-compute/src/analysis-kriging-panel';
import {DBSCANPanel} from '../../../examples/deck/kepler-compute/src/analysis-dbscan-panel';
import {AnalysisPanel} from '../../../examples/deck/kepler-compute/src/analysis-panel';
import {PointWeightsPanel} from '../../../examples/deck/kepler-compute/src/analysis-point-weights-panel';
import {PointHotspotPanel} from '../../../examples/deck/kepler-compute/src/analysis-point-hotspot-panel';
import {LocalMoranPanel} from '../../../examples/deck/kepler-compute/src/analysis-local-moran-panel';
import {QuantileLisaPanel} from '../../../examples/deck/kepler-compute/src/analysis-quantile-lisa-panel';
import {PolygonPanel} from '../../../examples/deck/kepler-compute/src/analysis-polygon-panel';
import {CartogramPanel} from '../../../examples/deck/kepler-compute/src/analysis-cartogram-panel';
import {OLSPanel} from '../../../examples/deck/kepler-compute/src/analysis-ols-panel';
import {HotspotPanel} from '../../../examples/deck/kepler-compute/src/analysis-hotspot-panel';

// Use the host's React renderer, matching its React version and hook dispatcher.
const hostRequire = createRequire(
  new URL('../../../examples/deck/kepler-compute/package.json', import.meta.url)
);
const {createElement, useState} = hostRequire('react') as typeof import('react');
const {renderToStaticMarkup} = hostRequire('react-dom/server') as typeof import('react-dom/server');
const COLUMNS = ['east', 'north', 'temperature', 'population'];
const EXTENT = [-122.5, 37.7, -122.3, 37.9] as const;

function makeDataset(overrides: Partial<CatalogResidentDataset> = {}): CatalogResidentDataset {
  return {
    id: 'form-fixture',
    label: 'Non-New-York observations',
    rowCount: 6,
    revision: 1,
    coordinates: {longitude: 'east', latitude: 'north'},
    valueColumn: 'temperature',
    timeColumns: ['recorded'],
    bounds: EXTENT,
    fieldTypes: {
      east: 'real',
      north: 'real',
      temperature: 'real',
      population: 'real',
      recorded: 'timestamp'
    },
    provider: {
      fields: [...COLUMNS, 'recorded'].map(name => ({name, sortable: true})),
      getRevision: () => '1',
      readColumnValues: vi.fn(),
      readPage: vi.fn(),
      maximumPageSize: 16
    },
    isCurrent: () => true,
    ...overrides
  };
}

function makeInitialization(planar = false) {
  const context = createAnalysisDatasetContext();
  const dataset = makeDataset();
  context.synchronize([dataset]);
  if (planar) context.setDeclarations(dataset.id, {coordinateModel: 'planar'});
  return getAnalysisFormInitialization(context.get(dataset.id), COLUMNS, {
    grid: {width: 3, height: 2, order: 'row-major'}
  });
}

function makePanelRuntime(planar = false): AnalysisRuntime {
  const initialization = makeInitialization(planar);
  return {
    datasetId: 'form-fixture',
    rowCount: 6,
    availableColumns: COLUMNS,
    availablePolygonColumns: ['shape'],
    formInitialization: {...initialization, geometryColumn: 'shape'},
    settings: {column: 'temperature', denominatorColumn: 'population'},
    summary: null
  } as unknown as AnalysisRuntime;
}

function expectSelected(markup: string, identifier: string, value: string) {
  const select = markup.match(
    new RegExp(`<select[^>]*id="${identifier}"[^>]*>([\\s\\S]*?)</select>`)
  );
  expect(select, identifier).not.toBeNull();
  expect(select![1], identifier).toMatch(
    new RegExp(`<option[^>]*value="${value}"[^>]*selected=""`)
  );
}

function expectInput(markup: string, identifier: string, value: string) {
  const input = markup.match(new RegExp(`<input[^>]*id="${identifier}"[^>]*>`));
  expect(input, identifier).not.toBeNull();
  expect(input![0], identifier).toContain(`value="${value}"`);
}

afterEach(() => {
  analysisDatasetContext.synchronize([]);
  analysisDatasetContext.restoreDeclarations(undefined);
});

test('publication roles, selected value, time and geographic extent seed without reading rows', () => {
  const context = createAnalysisDatasetContext();
  const dataset = makeDataset();
  context.synchronize([dataset]);
  const initialization = getAnalysisFormInitialization(context.get(dataset.id), COLUMNS);
  expect(initialization.column).toBe('temperature');
  expect(initialization.denominatorColumn).toBe('population');
  expect(initialization.timeColumn).toBe('recorded');
  expect(initialization.coordinates).toMatchObject({
    coordinateSpace: 'longitude-latitude',
    longitudeColumn: 'east',
    latitudeColumn: 'north',
    xColumn: '',
    yColumn: '',
    crs: ''
  });
  // Hand-computed fixture: each bound is exactly the corresponding source extent edge.
  expect(initialization.interpolation).toEqual({
    longitudeColumn: 'east',
    latitudeColumn: 'north',
    west: '-122.5',
    south: '37.7',
    east: '-122.3',
    north: '37.9'
  });
  expect(initialization.coordinateSource).toBe('Publication coordinate roles');
  expect(dataset.provider.readColumnValues).not.toHaveBeenCalled();
  expect(dataset.provider.readPage).not.toHaveBeenCalled();
});

test('implicit values exclude coordinates and unavailable fields; no arbitrary coordinate fallback', () => {
  const context = createAnalysisDatasetContext();
  const dataset = makeDataset({valueColumn: 'east'});
  context.synchronize([dataset]);
  expect(getAnalysisFormInitialization(context.get(dataset.id), COLUMNS).column).toBe(
    'temperature'
  );
  expect(getAnalysisFormInitialization(context.get(dataset.id), ['east', 'north']).column).toBe('');
  expect(getAnalysisFormInitialization(undefined, ['longitude', 'latitude', 'income']).column).toBe(
    'income'
  );
  expect(
    getAnalysisFormInitialization(context.get(dataset.id), ['temperature']).coordinates
      .longitudeColumn
  ).toBe('');
});

test('projected profiles seed X/Y and native bounds without inventing a metre CRS or geographic grid', () => {
  const context = createAnalysisDatasetContext();
  const dataset = makeDataset({bounds: [500000, 4200000, 500400, 4200200]});
  context.synchronize([dataset]);
  context.setDeclarations(dataset.id, {coordinateModel: 'planar'});
  const initialization = getAnalysisFormInitialization(context.get(dataset.id), COLUMNS);
  expect(initialization.coordinates).toEqual({
    coordinateSpace: 'planar',
    xColumn: 'east',
    yColumn: 'north',
    longitudeColumn: '',
    latitudeColumn: '',
    crs: '',
    minX: '500000',
    minY: '4200000',
    maxX: '500400',
    maxY: '4200200'
  });
  expect(initialization.interpolation).toEqual({
    longitudeColumn: '',
    latitudeColumn: '',
    west: '',
    south: '',
    east: '',
    north: ''
  });
});

test.each(
  ([undefined, [0, 0, 0, 1], [1, 0, 0, 1], [0, 0, NaN, 1], [170, -1, -170, 1]] as const).map(
    bounds => ({bounds})
  )
)('missing, degenerate or wrapped geographic extent stays explicit: %j', ({bounds}) => {
  const context = createAnalysisDatasetContext();
  const dataset = makeDataset({bounds: undefined});
  context.synchronize([dataset]);
  const entry = context.get(dataset.id)!;
  entry.profile = {...entry.profile, extent: bounds};
  const initialization = getAnalysisFormInitialization(entry, COLUMNS);
  expect(initialization.interpolation.west).toBe('');
  expect(initialization.interpolation.east).toBe('');
});

test('lattice shape requires matching complete row-major metadata, never a row-count guess', () => {
  expect(getAnalysisLatticeShape({grid: {width: 3, height: 2, order: 'row-major'}}, 6)).toEqual({
    width: '3',
    height: '2'
  });
  for (const grid of [
    {width: 3, height: 2},
    {width: 3, height: 2, order: 'column-major'},
    {width: 3, height: 3, order: 'row-major'},
    {width: -3, height: -2, order: 'row-major'},
    {width: 1.5, height: 4, order: 'row-major'}
  ])
    expect(getAnalysisLatticeShape({grid}, 6)).toBeUndefined();
  expect(getAnalysisLatticeShape(undefined, 256)).toBeUndefined();
});

test('late profile refinement fills untouched fields and preserves cleared and restored user edits', () => {
  const defaults = {longitudeColumn: '', latitudeColumn: '', west: '', east: ''};
  const state: AnalysisFormDraftState<typeof defaults> = {
    defaults,
    draft: defaults,
    edited: new Set()
  };
  const seeded = reconcileAnalysisFormDraft(state, {
    longitudeColumn: 'east',
    latitudeColumn: 'north',
    west: '-122.5',
    east: '-122.3'
  });
  const edited = editAnalysisFormDraft(seeded, {
    ...seeded.draft,
    longitudeColumn: '',
    west: '-123'
  });
  const refined = reconcileAnalysisFormDraft(edited, {
    longitudeColumn: 'newEast',
    latitudeColumn: 'newNorth',
    west: '-124',
    east: '-122'
  });
  expect(refined.draft).toEqual({
    longitudeColumn: '',
    latitudeColumn: 'newNorth',
    west: '-123',
    east: '-122'
  });
  // Changing a field back to its original value still records an explicit override.
  const restored = editAnalysisFormDraft(edited, {...edited.draft, west: '-122.5'});
  expect(reconcileAnalysisFormDraft(restored, {...seeded.defaults, west: '-125'}).draft.west).toBe(
    '-122.5'
  );
});

test('the React draft hook preserves edits on refinement and resets on explicit recipe or owner replacement', () => {
  const observations: unknown[] = [];
  function DraftHarness() {
    const [stage, setStage] = useState(0);
    const defaults =
      stage === 0
        ? {west: '-122.5', north: '37.9'}
        : stage === 1
          ? {west: '-124', north: '38'}
          : {west: '10', north: '21'};
    const [draft, setDraft] = useAnalysisFormDraft(
      defaults,
      stage < 2 ? undefined : 'restored-recipe',
      stage < 3 ? 'owner-one' : 'owner-two'
    );
    if (stage === 0) {
      setDraft({...draft, west: ''});
      setStage(1);
    } else if (stage === 1) {
      observations.push({stage, ...draft});
      setStage(2);
    } else if (stage === 2) {
      observations.push({stage, ...draft});
      setDraft({...draft, west: '11'});
      setStage(3);
    } else {
      observations.push({stage, ...draft});
    }
    return createElement('span', null, draft.west);
  }
  renderToStaticMarkup(createElement(DraftHarness));
  expect(observations).toContainEqual({stage: 1, west: '', north: '38'});
  expect(observations).toContainEqual({stage: 2, west: '10', north: '21'});
  expect(observations).toContainEqual({stage: 3, west: '10', north: '21'});
  expect(observations.filter(value => (value as {stage: number}).stage === 3)).toEqual(
    expect.arrayContaining([{stage: 3, west: '10', north: '21'}])
  );
});

test('all eleven actual panels receive their applicable initial roles and metadata', () => {
  const analysis = makePanelRuntime();
  const panels = [
    [InterpolationPanel, 'interpolation-longitudeColumn', 'interpolation-latitudeColumn'],
    [KrigingPanel, 'kriging-longitudeColumn', 'kriging-latitudeColumn'],
    [DBSCANPanel, 'analysis-dbscan-longitude', 'analysis-dbscan-latitude'],
    [PointWeightsPanel, 'analysis-point-longitude', 'analysis-point-latitude'],
    [PointHotspotPanel, 'analysis-point-hotspot-longitude', 'analysis-point-hotspot-latitude']
  ] as const;
  for (const [Panel, longitudeIdentifier, latitudeIdentifier] of panels) {
    const markup = renderToStaticMarkup(createElement(Panel, {analysis}));
    expectSelected(markup, longitudeIdentifier, 'east');
    expectSelected(markup, latitudeIdentifier, 'north');
  }
  const pointProps = {
    columns: COLUMNS,
    rowCount: 6,
    coordinateInitialization: analysis.formInitialization.coordinates,
    onApply: () => {}
  };
  for (const [element, longitudeIdentifier, latitudeIdentifier] of [
    [
      createElement(LocalMoranPanel, pointProps),
      'analysis-moran-longitude',
      'analysis-moran-latitude'
    ],
    [
      createElement(QuantileLisaPanel, pointProps),
      'analysis-quantile-lisa-longitudeColumn',
      'analysis-quantile-lisa-latitudeColumn'
    ]
  ] as const) {
    const markup = renderToStaticMarkup(element);
    expectSelected(markup, longitudeIdentifier, 'east');
    expectSelected(markup, latitudeIdentifier, 'north');
  }
  const interpolation = renderToStaticMarkup(createElement(InterpolationPanel, {analysis}));
  const kriging = renderToStaticMarkup(createElement(KrigingPanel, {analysis}));
  for (const [markup, prefix] of [
    [interpolation, 'interpolation'],
    [kriging, 'kriging']
  ] as const) {
    expectInput(markup, `${prefix}-west`, '-122.5');
    expectInput(markup, `${prefix}-south`, '37.7');
    expectInput(markup, `${prefix}-east`, '-122.3');
    expectInput(markup, `${prefix}-north`, '37.9');
  }
  expectSelected(
    renderToStaticMarkup(createElement(OLSPanel, {analysis})),
    'analysis-ols-response',
    'temperature'
  );
  expectSelected(
    renderToStaticMarkup(createElement(PolygonPanel, {analysis})),
    'analysis-polygon-geometry',
    'shape'
  );
  const lattice = renderToStaticMarkup(createElement(HotspotPanel, {analysis}));
  expectInput(lattice, 'analysis-grid-width', '3');
  expectInput(lattice, 'analysis-grid-height', '2');
  const cartogram = renderToStaticMarkup(
    createElement(CartogramPanel, {analysis: makePanelRuntime(true)})
  );
  expectSelected(cartogram, 'analysis-cartogram-xColumn', 'east');
  expectSelected(cartogram, 'analysis-cartogram-yColumn', 'north');
  expectSelected(cartogram, 'analysis-cartogram-valueColumn', 'temperature');
});

test('saved coordinate roles and grid bounds take priority over shared seeds in rendered forms', () => {
  const analysis = makePanelRuntime();
  Object.assign(analysis, {
    settings: {
      ...analysis.settings,
      pointWeights: {
        longitudeColumn: 'population',
        latitudeColumn: 'temperature',
        k: 2,
        scale: 1,
        islandPolicy: 'undefined'
      },
      idw: {
        longitudeColumn: 'population',
        latitudeColumn: 'temperature',
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle',
        units: 'meters',
        grid: {bounds: [10, 20, 11, 21], width: 4, height: 5},
        searchRadius: 2000,
        power: 3,
        neighborCount: 4,
        minimumNeighborCount: 1,
        taper: 'none'
      }
    }
  });
  const weights = renderToStaticMarkup(createElement(PointWeightsPanel, {analysis}));
  expectSelected(weights, 'analysis-point-longitude', 'population');
  const interpolation = renderToStaticMarkup(createElement(InterpolationPanel, {analysis}));
  expectSelected(interpolation, 'interpolation-latitudeColumn', 'temperature');
  expectInput(interpolation, 'interpolation-west', '10');
  expectInput(interpolation, 'interpolation-width', '4');
});

test('runtime starts from shared value roles, ignores same-id replacement profiles and preserves recipes', async () => {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const table = arrow.tableFromArrays(
    Object.fromEntries(COLUMNS.map(name => [name, new Float32Array([1, 2, 3, 4, 5, 6])]))
  );
  table.schema.metadata.set(
    'luma:sandbox-sample',
    JSON.stringify({grid: {width: 3, height: 2, order: 'row-major'}})
  );
  const dataset = session.importArrowDataset({id: 'form-fixture', table});
  const explorer = {
    dataset,
    isCurrent: true,
    currentRevision: 0
  } as unknown as KeplerSessionExplorer;
  const analysis = new AnalysisRuntime({
    session,
    explorer,
    pointColumns: {longitude: 'east', latitude: 'north'},
    requestUpdate: () => {}
  });
  const resident = makeDataset({owner: dataset, revision: dataset.revision});
  try {
    expect(analysis.settings.column).toBe('temperature');
    expect(analysis.formInitialization.lattice).toEqual({width: '3', height: '2'});
    analysisDatasetContext.synchronize([resident]);
    expect(analysis.formInitialization.interpolation.west).toBe('-122.5');
    analysisDatasetContext.synchronize([{...resident, valueColumn: 'population'}]);
    expect(analysis.settings.column).toBe('population');
    analysis.setColumn('temperature');
    analysis.setDenominatorColumn('population');
    analysis.setMode('z-score');
    const saved = analysis.serializeRecipe();
    analysis.setMode('off');
    analysisDatasetContext.synchronize([{...resident, valueColumn: 'population'}]);
    expect(analysis.settings.column).toBe('temperature');
    analysis.restoreRecipe(saved);
    expect(analysis.settings.column).toBe('temperature');
    analysisDatasetContext.synchronize([{...resident, owner: {}, valueColumn: 'population'}]);
    expect(analysis.formInitialization.interpolation.west).toBe('');
    expect(analysis.settings.column).toBe('temperature');
    expect(resident.provider.readColumnValues).not.toHaveBeenCalled();
  } finally {
    await analysis.destroy(Promise.resolve());
    await session.destroy(Promise.resolve());
    device.destroy();
  }
});

test('coordinate-only batches open analysis and configure DBSCAN without selecting a numeric value', async () => {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const first = arrow.tableFromArrays({
    longitude: new Float32Array([0, 0.001]),
    latitude: new Float32Array([0, 0])
  });
  const second = arrow.tableFromArrays({
    longitude: new Float32Array([1, 1.001]),
    latitude: new Float32Array([0, 0])
  });
  const dataset = session.importArrowDataset({id: 'coordinate-only', table: first.concat(second)});
  const explorer = {
    dataset,
    isCurrent: true,
    currentRevision: 0
  } as unknown as KeplerSessionExplorer;
  const analysis = new AnalysisRuntime({session, explorer, requestUpdate: () => {}});
  try {
    expect(analysis.settings.column).toBe('');
    expect(analysis.settings.denominatorColumn).toBe('');
    expect(analysis.rowCount).toBe(4);
    expect(analysis.colors.data.map(chunk => chunk.length)).toEqual([2, 2]);
    expect(analysis.supportsExactNaturalBreaks).toBe(true);
    analysis.configureDBSCAN({
      longitudeColumn: 'longitude',
      latitudeColumn: 'latitude',
      coordinateSpace: 'longitude-latitude',
      metric: 'great-circle',
      units: 'meters',
      epsilon: 200,
      minimumSamples: 2
    });
    expect(analysis.settings.column).toBe('');
    const saved = analysis.serializeRecipe();
    expect(JSON.parse(saved).operation).toMatchObject({
      kind: 'dbscan-points',
      longitudeColumn: 'longitude',
      latitudeColumn: 'latitude'
    });
    expect(JSON.parse(saved).operation).not.toHaveProperty('column');
    expect(getAnalysisRecipeColumns(JSON.parse(saved))).toEqual(['longitude', 'latitude']);
    // Derived DBSCAN columns become available only after its GPU outputs are encoded.
    expect(analysis.getTableColumns()).toBeUndefined();
    analysis.setMode('off');
    analysis.restoreRecipe(saved);
    expect(analysis.settings.column).toBe('');
    expect(analysis.settings.denominatorColumn).toBe('');
    const restoredKey = analysis.coordinateRestorationKey;
    analysis.restoreRecipe(saved);
    expect(analysis.coordinateRestorationKey).not.toBe(restoredKey);
    const previous = analysis.settings;
    expect(() => analysis.setMode('z-score')).toThrow('Float32 analysis column');
    expect(() => analysis.setMode('excess-risk')).toThrow('Float32 analysis column');
    expect(analysis.settings).toEqual(previous);
  } finally {
    await analysis.destroy(Promise.resolve());
    await session.destroy(Promise.resolve());
    device.destroy();
  }
});

test('constructor value override chooses its denominator relative to the effective column, including later metadata', async () => {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const dataset = session.importArrowDataset({
    id: 'rate-override',
    table: arrow.tableFromArrays({
      population: new Float32Array([100, 200]),
      cases: new Float32Array([10, 40])
    })
  });
  const explorer = {
    dataset,
    isCurrent: true,
    currentRevision: 0
  } as unknown as KeplerSessionExplorer;
  const analysis = new AnalysisRuntime({
    session,
    explorer,
    column: 'cases',
    requestUpdate: () => {}
  });
  try {
    expect(analysis.settings.column).toBe('cases');
    expect(analysis.settings.denominatorColumn).toBe('population');
    analysisDatasetContext.synchronize([
      makeDataset({
        id: dataset.id,
        owner: dataset,
        revision: dataset.revision,
        rowCount: 2,
        coordinates: undefined,
        timeColumns: undefined,
        bounds: undefined,
        valueColumn: 'population',
        fieldTypes: {population: 'real', cases: 'real'},
        provider: {
          fields: ['population', 'cases'].map(name => ({name, sortable: true})),
          getRevision: () => '1',
          readColumnValues: vi.fn(),
          readPage: vi.fn(),
          maximumPageSize: 16
        }
      })
    ]);
    expect(analysis.settings.column).toBe('cases');
    expect(analysis.settings.denominatorColumn).toBe('population');
    analysis.setMode('excess-risk');
    expect(JSON.parse(analysis.serializeRecipe()).operation).toEqual({
      kind: 'excess-risk',
      events: 'cases',
      populations: 'population'
    });
    analysis.restoreRecipe(analysis.serializeRecipe());
    expect(analysis.settings.denominatorColumn).toBe('population');
  } finally {
    await analysis.destroy(Promise.resolve());
    await session.destroy(Promise.resolve());
    device.destroy();
  }
});

test('inherited coordinate drafts ignore neighbor edits and reset for changed coordinates or explicit restoration', () => {
  const observations: {stage: number; longitude: string; latitude: string}[] = [];
  function InheritedHotspotCoordinates() {
    const [stage, setStage] = useState(0);
    // PointHotspotPanel inherits these rate weights until it owns a saved hotspot recipe.
    const rateWeights = {
      longitudeColumn: stage < 2 ? 'longitude' : 'restoredLongitude',
      latitudeColumn: 'latitude',
      k: stage === 0 ? 8 : 12
    };
    const coordinates = usePointCoordinates(rateWeights, stage < 3 ? 'dataset:0' : 'dataset:1');
    observations.push({
      stage,
      longitude: coordinates.draft.longitudeColumn,
      latitude: coordinates.draft.latitudeColumn
    });
    if (stage === 0 || stage === 2) {
      coordinates.setDraft({...coordinates.draft, longitudeColumn: 'alternateLongitude'});
      setStage(stage + 1);
    } else if (stage === 1) setStage(2);
    return createElement('span', null, coordinates.draft.longitudeColumn);
  }
  renderToStaticMarkup(createElement(InheritedHotspotCoordinates));
  expect(observations).toContainEqual({
    stage: 1,
    longitude: 'alternateLongitude',
    latitude: 'latitude'
  });
  expect(observations).toContainEqual({
    stage: 2,
    longitude: 'restoredLongitude',
    latitude: 'latitude'
  });
  const restored = observations.filter(observation => observation.stage === 3);
  expect(restored.length).toBeGreaterThan(0);
  for (const observation of restored)
    expect(observation).toEqual({stage: 3, longitude: 'restoredLongitude', latitude: 'latitude'});
});

test('the shared numeric field remains selectable after DBSCAN for the next point analysis', async () => {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const dataset = session.importArrowDataset({
    id: 'dbscan-next-field',
    table: arrow.tableFromArrays({
      east: new Float32Array([0, 0.001, 0.1]),
      north: new Float32Array([0, 0, 0]),
      reading: new Float32Array([1, 2, 3])
    })
  });
  const explorer = {
    dataset,
    isCurrent: true,
    currentRevision: 0
  } as unknown as KeplerSessionExplorer;
  const analysis = new AnalysisRuntime({session, explorer, requestUpdate: () => {}});
  try {
    analysis.configureDBSCAN({
      longitudeColumn: 'east',
      latitudeColumn: 'north',
      coordinateSpace: 'longitude-latitude',
      metric: 'great-circle',
      units: 'meters',
      epsilon: 30,
      minimumSamples: 3
    });
    const markup = renderToStaticMarkup(createElement(AnalysisPanel, {analysis}));
    const select = markup.match(/<select[^>]*id="analysis-column"[^>]*>/)?.[0];
    expect(select).toBeDefined();
    expect(select).not.toContain('disabled');
    analysis.setColumn('reading');
    expect(analysis.settings.column).toBe('reading');
    expect(analysis.settings.mode).toBe('dbscan-points');
  } finally {
    await analysis.destroy(Promise.resolve());
    await session.destroy(Promise.resolve());
    device.destroy();
  }
});
