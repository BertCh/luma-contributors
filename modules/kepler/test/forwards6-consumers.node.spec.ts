// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {OptionsStore} from '../../../examples/deck/spatial-analysis-showcase/src/scenes/options-state';
import {parseOptionText} from '../../../examples/deck/spatial-analysis-showcase/src/shell/story-readouts';
import type {SelectOption} from '../../../examples/deck/spatial-analysis-showcase/src/scenes/scene';
import {
  createDemProbe,
  type TerrainDem
} from '../../../examples/deck/spatial-analysis-showcase/src/scenes/terrain/cpu-dem';
import {
  getCatalogBinding,
  normalizeCatalogParameters,
  runCatalogOperation
} from '../src/analysis/catalog-bindings';
import {
  getIDWSheetSettings,
  makeInterpolationSheetDefaults
} from '../../../examples/deck/kepler-compute/src/interpolation-panel';
import {getAnalysisFormInitialization} from '../../../examples/deck/kepler-compute/src/analysis-form-initialization';
import {
  parseKeplerIDWSettings,
  restoreKeplerAnalysisRecipe,
  serializeKeplerAnalysisRecipe,
  type KeplerAnalysisRecipe
} from '../src/recipes/analysis-recipe';

// Legacy state must be canonical before listeners, snapshots and controls read it.
test('saved select aliases normalize in the store and action parser', () => {
  const specs: SelectOption<never>[] = [
    {
      kind: 'select',
      id: 'ties',
      label: 'Ties',
      apply: 'param',
      default: 'all',
      aliases: {'lowest-id': 'lowest-row'},
      options: [
        {value: 'all', label: 'All'},
        {value: 'lowest-row', label: 'Lowest row'}
      ]
    },
    {
      kind: 'select',
      id: 'standardization',
      label: 'Standardization',
      apply: 'param',
      default: 'rank',
      options: [
        {value: 'rank', label: 'Rank'},
        {value: 'z-score', label: 'Z-score'}
      ]
    }
  ];
  const store = new OptionsStore(specs);
  const observed: unknown[] = [];
  store.subscribe((id, value) => observed.push([id, value]));
  store.deserialize('ties:lowest-id~standardization:zscore');
  expect(store.snapshot()).toEqual({ties: 'lowest-row', standardization: 'z-score'});
  expect(observed).toEqual([
    ['ties', 'lowest-row'],
    ['standardization', 'z-score']
  ]);
  expect(store.serialize()).toBe('ties:lowest-row~standardization:z-score');
  store.set('ties', 'all');
  const action = parseOptionText(specs[0], 'lowest-id');
  expect(action).toBe('lowest-row');
  if (action !== undefined) store.set('ties', action);
  expect(store.get('ties')).toBe('lowest-row');
  const legacy = new OptionsStore([
    {
      kind: 'select',
      id: 'standardization',
      label: 'Legacy',
      apply: 'param',
      default: 'rank',
      options: [
        {value: 'rank', label: 'Rank'},
        {value: 'zscore', label: 'Z'}
      ]
    }
  ]);
  legacy.deserialize('standardization:zscore');
  expect(legacy.get('standardization')).toBe('zscore');
});

test('top-row sight line is blocked by the finite ridge beside zero-weight no-data', () => {
  const values = new Float32Array(15);
  values[2] = 8;
  values[7] = Number.NaN;
  const dem: TerrainDem = {
    width: 5,
    height: 3,
    values,
    lngLatBounds: [0, 0, 5, 3],
    layerBounds: [0, 0, 5, 3],
    origin: [0, 0],
    groundCellSize: 1,
    getGroundCellSize: () => 1,
    getPixelCoordinates: (longitude, latitude) => [longitude, latitude],
    getLongitudeLatitude: (column, row) => [column + 0.5, row + 0.5]
  };
  const probe = createDemProbe(dem);
  expect(probe.sampleAtPixel(2.5, 0.5)).toBe(8);
  // Ridge slope 8/2 exceeds the target slope 1/4; unused bottom corners contribute nothing.
  expect(
    probe.sampleRay([0.5, 0.5], [4.5, 0.5], 1, {eyeHeight: 0, targetHeight: 1, refraction: 1})
      .blocked
  ).toBe(true);
});

test('restored wrapped IDW retains its longitude choice through Apply and serialization', () => {
  const settings = parseKeplerIDWSettings({
    longitudeColumn: 'longitude',
    latitudeColumn: 'latitude',
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle',
    units: 'meters',
    longitudeSearch: 'wrapped',
    grid: {bounds: [-179.99, 10, -179.98, 11], width: 2, height: 2},
    searchRadius: 30000,
    power: 2,
    neighborCount: 4,
    minimumNeighborCount: 3,
    taper: 'none'
  });
  const recipe: KeplerAnalysisRecipe = {
    schemaVersion: 1,
    algorithmVersion: 1,
    id: 'wrapped-idw',
    source: {datasetId: 'stations', revision: 1},
    population: 'all',
    backendPolicy: 'require-gpu',
    operation: {kind: 'idw-points', column: 'value', ...settings}
  };
  const restored = restoreKeplerAnalysisRecipe(serializeKeplerAnalysisRecipe(recipe), {
    resolveDatasetRevision: () => 1
  });
  if (restored.operation.kind !== 'idw-points') throw new Error('Expected IDW');
  const applied = getIDWSheetSettings({
    ...makeInterpolationSheetDefaults(
      restored.operation,
      getAnalysisFormInitialization(undefined, ['longitude', 'latitude', 'value'])
    ),
    searchRadius: settings.searchRadius,
    power: settings.power,
    neighborCount: settings.neighborCount,
    minimumNeighborCount: settings.minimumNeighborCount,
    taper: settings.taper
  });
  expect(applied).toEqual(settings);
  expect(
    serializeKeplerAnalysisRecipe({
      ...recipe,
      operation: {kind: 'idw-points', column: 'value', ...applied}
    })
  ).toBe(serializeKeplerAnalysisRecipe(recipe));
});

test('catalog KDE admits bisquare with the same quartic result', () => {
  const binding = getCatalogBinding('kernel-density-grid')!;
  expect(normalizeCatalogParameters(binding, {kernel: 'bisquare'})['kernel']).toBe('bisquare');
  const inputs = {
    points: {
      type: 'FeatureCollection' as const,
      features: [{geometry: {type: 'Point' as const, coordinates: [0, 0]}, properties: {}}]
    }
  };
  const parameters = {
    coordinateSystem: 'planar',
    cellSize: 1,
    bandwidth: 2,
    includeEmptyCells: true
  };
  expect(runCatalogOperation(binding, {...parameters, kernel: 'bisquare'}, inputs).tables).toEqual(
    runCatalogOperation(binding, {...parameters, kernel: 'quartic'}, inputs).tables
  );
});

test('inscribed-circle binding surfaces an exhausted search budget', async () => {
  const sessions = await import('../src/analysis/catalog-gpu-session');
  const {runGpuLabelPoints} = await import('../src/analysis/catalog-platform4-geometry-bounds');
  const read = vi.spyOn(sessions, 'withCatalogGpuSession').mockResolvedValue({
    centers: Float32Array.of(1, 1, 0.5, 0),
    stride: 4,
    status: Uint32Array.of(1)
  } as never);
  try {
    const result = await runGpuLabelPoints(
      {device: {} as never},
      {
        type: 'FeatureCollection',
        features: [
          {
            geometry: {
              type: 'Polygon',
              coordinates: [
                [
                  [0, 0],
                  [2, 0],
                  [2, 2],
                  [0, 2],
                  [0, 0]
                ]
              ]
            },
            properties: {}
          }
        ]
      },
      true,
      0.01,
      'planar'
    );
    expect(
      result.diagnostics.some(diagnostic => diagnostic.code === 'search-budget-exhausted')
    ).toBe(true);
  } finally {
    read.mockRestore();
  }
});

test('global envelope decisions use the attained two-of-forty fraction', async () => {
  const {getGlobalEnvelopeProbability} = await import(
    '../../../examples/deck/spatial-analysis-showcase/src/scenes/points/point-patterns-probability'
  );
  // 39 simulated curves plus the observed curve: conservative numerator 2.
  const probability = getGlobalEnvelopeProbability(Math.fround(2 / 40), 39);
  expect(probability).toBe(0.05);
  expect(probability <= 0.05).toBe(true);
  expect(probability <= 0.04).toBe(false);
  expect(getGlobalEnvelopeProbability(Math.fround(1 / 40), 39)).toBe(0.025);
});

test('bisquare bandwidth policy uses the quartic canonical factor', async () => {
  const {INTERPOLATION_DENSITY_OPERATION_DESCRIPTORS} = await import(
    '../src/descriptors/descriptors-interpolation-density'
  );
  const descriptor = INTERPOLATION_DENSITY_OPERATION_DESCRIPTORS.find(
    entry => entry.bindingId === 'kernel-density-grid'
  )!;
  const policy = descriptor.parameters.find(
    parameter => parameter.name === 'bandwidth'
  )!.defaultPolicy;
  if (policy?.from !== 'data') throw new Error('Expected data policy');
  const factors = policy.factorByParameter!.factors;
  // Equal-smoothing 2D quartic support radius: (2304/5)^(1/6), Silverman 1986.
  expect(factors['bisquare']).toBe((2304 / 5) ** (1 / 6));
  expect(factors['bisquare']).toBe(factors['quartic']);
});

test('actual cell snippets keep producer and consumer status buffers distinct', async () => {
  const {default: scene} = await import(
    '../../../examples/deck/spatial-analysis-showcase/src/scenes/cells/h3-aggregation.scene'
  );
  const {getDefaultState} = await import(
    '../../../examples/deck/spatial-analysis-showcase/src/scenes/options-state'
  );
  if (typeof scene.snippet !== 'function') throw new Error('Expected snippet function');
  const state = getDefaultState(scene.options as readonly OptionSpec<never>[]);
  const snippet = scene.snippet({
    ...state,
    family: 'h3',
    outline: 'rings',
    showCover: true,
    showCompacted: true
  } as unknown as Parameters<typeof scene.snippet>[0]);
  expect(snippet).toContain('cells: tableCells, counts, status: {count, overflow}');
  expect(snippet).toContain('source: {cells: tableCells, status: {count, overflow}}');
  expect(snippet).toContain('status: {count: outlineCount, overflow: outlineOverflow}');
  expect(snippet).toContain(
    'count: coverCount, overflow: coverOverflow, candidateOverflow: coverCandidateOverflow'
  );
  expect(snippet).toContain(
    'count: compactCount, overflow: compactOverflow, candidateOverflow: compactCandidateOverflow'
  );
});

test('actual nearest-facility scene restores its tie setting before rendering its snippet', async () => {
  const {default: scene} = await import(
    '../../../examples/deck/spatial-analysis-showcase/src/scenes/joins/nearest-facility.scene'
  );
  if (typeof scene.snippet !== 'function') throw new Error('Expected snippet function');
  const store = new OptionsStore(scene.options as readonly OptionSpec<never>[]);
  store.set('ties', 'all');
  expect(store.get('ties')).toBe('all');
  store.deserialize('ties:lowest-row');
  expect(store.get('ties')).toBe('lowest-row');
  const spec = store.specs.find(option => option.id === 'ties');
  if (spec?.kind !== 'select') throw new Error('Expected ties select');
  expect(spec.options.filter(option => option.value === store.get('ties'))).toHaveLength(1);
  expect(store.serialize()).not.toContain('lowest-id');
  store.set('mode', 'neighbors');
  const snippet = scene.snippet(store.snapshot() as unknown as Parameters<typeof scene.snippet>[0]);
  expect(snippet).toContain("ties: 'lowest-row'");
});

test('palette follows an inherited table representation before a result exists', async () => {
  const {makeFixtureRegistry, makeFixtureDocument} = await import('./workflows/workflow-fixtures');
  const {getPaletteConnection, getPalettePortLine} = await import(
    '../../../examples/deck/kepler-compute/src/workflow/workbench-palette'
  );
  const original = makeFixtureRegistry();
  const registry = {
    list: original.list,
    get(id: string) {
      const operation = original.get(id);
      if (!operation) return undefined;
      if (id === 'fixture-source')
        return {
          ...operation,
          outputs: operation.outputs.map(output => ({...output, kind: 'table' as const}))
        };
      if (id === 'fixture-scale')
        return {
          ...operation,
          outputs: operation.outputs.map(output => ({...output, kindFrom: 'points'}))
        };
      return operation;
    }
  };
  const state = {
    document: makeFixtureDocument(),
    palette: {open: true, connectFrom: {stepId: 'scale', output: 'scaled'}},
    runner: null
  };
  expect(
    getPaletteConnection(state as Parameters<typeof getPaletteConnection>[0], registry)?.kind
  ).toBe('table');
  expect(getPalettePortLine(registry.get('fixture-scale')!).outputs).toEqual(['features']);
});

test('result-card Next serializes geometry-free result rows as a scalar table', async () => {
  // Synthetic complete readback isolates handoff serialization from GPU execution.
  const {createAnalysisResultCardModel, createAnalysisResultNextDraft} = await import(
    '../../../examples/deck/kepler-compute/src/analysis-result-card-model'
  );
  const result: import('../src/analysis/catalog-binding-types').CatalogRunResult = {
    operationId: 'validation-fdr',
    engine: 'gpu',
    dispatchedContributors: ['GPURasterFalseDiscoveryRateMask'],
    parameters: {pValueProperty: 'pvalue', alpha: 0.05, method: 'benjamini-hochberg'},
    diagnostics: [],
    summary: {rowCount: 2, testedCount: 2, significantCount: 1},
    tables: [
      {
        id: 'validation-fdr',
        label: 'Tests',
        rowCount: 2,
        columns: {pvalue: [0.01, 0.2], adjusted_p_value: [0.02, 0.2], significant: [1, 0]},
        columnTypes: {pvalue: 'float64', adjusted_p_value: 'float64', significant: 'int32'}
      }
    ]
  };
  const model = createAnalysisResultCardModel(result, {}, {executionEngine: 'gpu'});
  const action = model.next.find(candidate => candidate.applicability.status !== 'blocked');
  expect(action).toBeDefined();
  const next = createAnalysisResultNextDraft(model, action!);
  const draft = Object.values(next.drafts)[0];
  const snapshot = JSON.parse(draft.text);
  expect(snapshot.type).toBe('table');
  expect(snapshot.table.rowCount).toBe(
    result.tables.find(table => table.id === action!.tableId)!.rowCount
  );
  expect(snapshot.table.geometryColumn).toBeUndefined();
});

test('spatial weights snippet preserves search and transform handles for downstream consumers', async () => {
  const {default: scene} = await import(
    '../../../examples/deck/spatial-analysis-showcase/src/scenes/weights/spatial-weights.scene'
  );
  if (typeof scene.snippet !== 'function') throw new Error('Expected snippet function');
  const store = new OptionsStore(scene.options as readonly OptionSpec<never>[]);
  store.deserialize('weightKind:inverseDistance');
  expect(store.get('weightKind')).toBe('inverse-distance');
  store.deserialize('weightKind:kernel');
  expect(store.get('weightKind')).toBe('kernel');
  const snippet = scene.snippet({
    ...store.snapshot(),
    lattice: false,
    source: 'knn',
    transform: 'kernel',
    bandwidth: 'adaptive',
    combine: 'none'
  } as unknown as Parameters<typeof scene.snippet>[0]);
  expect(snippet).toContain("weightType: 'kernel'");
  expect(snippet).toContain('weights: search.outputWeights');
  expect(snippet).toContain('const transform = new GPUSpatialWeightsTransform');
  expect(snippet).toContain('const resultWeights = transform.outputWeights;');
  expect(snippet).toContain('GPUSpatialWeightsSummary({weights: resultWeights');
  expect(snippet).toContain('status: {overflow: summaryOverflow}');
});
