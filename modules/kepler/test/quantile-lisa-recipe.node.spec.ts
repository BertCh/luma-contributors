// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  MAXIMUM_KEPLER_QUANTILE_LISA_PERMUTATIONS,
  parseKeplerAnalysisRecipe,
  restoreKeplerAnalysisRecipe,
  serializeKeplerAnalysisRecipe,
  type KeplerAnalysisRecipe
} from '../src/recipes/analysis-recipe';
import {selectKeplerExecutionRoute} from '../src/execution/execution-router';
import {getQuantileLisaSheetSettings} from '../../../examples/deck/kepler-compute/src/analysis-quantile-lisa-panel';
import {
  makePolygonSheetDefaults,
  getPolygonSheetOperation
} from '../../../examples/deck/kepler-compute/src/analysis-polygon-panel';
import {makePointSheetDefaults} from '../../../examples/deck/kepler-compute/src/analysis-point-weights-panel';
import {normalizeQuantileLisaSettings} from '../../../examples/deck/kepler-compute/src/analysis-quantile-lisa-inference';
import {AnalysisRuntime} from '../../../examples/deck/kepler-compute/src/analysis-runtime';
import {getAnalysisRecipeSettings} from '../../../examples/deck/kepler-compute/src/analysis-recipe-settings';
import {makeAnalysisTooltipProvider} from '../../../examples/deck/kepler-compute/src/analysis-tooltip-provider';

const recipe: KeplerAnalysisRecipe = {
  schemaVersion: 1,
  algorithmVersion: 1,
  id: 'quantile-lisa',
  source: {datasetId: 'observations', revision: 2},
  population: 'selected',
  selectionRevision: 3,
  backendPolicy: 'require-gpu',
  operation: {
    kind: 'quantile-lisa-points',
    column: 'value',
    weights: {
      kind: 'knn',
      longitudeColumn: 'longitude',
      latitudeColumn: 'latitude',
      k: 8,
      coordinateSpace: 'longitude-latitude',
      metric: 'great-circle',
      units: 'meters'
    },
    quantileCount: 4,
    selectedQuantile: 4,
    seed: 2 ** 40 + 5,
    permutationCount: 999,
    significanceLevel: 0.05,
    alternative: 'geoda-legacy',
    multipleTesting: 'none',
    islands: 'exclude',
    quantileMethod: 'type-5'
  }
};

test('Quantile LISA recipes preserve exact seed, membership and limited inference semantics', () => {
  const restored = restoreKeplerAnalysisRecipe(serializeKeplerAnalysisRecipe(recipe), {
    resolveDatasetRevision: () => 2,
    resolveSelectionRevision: () => 3
  });
  expect(restored).toEqual({
    ...recipe,
    operation: {...recipe.operation, alternative: 'geoda-legacy'}
  });
  expect(restored.operation).not.toBe(recipe.operation);
  expect(Object.isFrozen(restored.operation)).toBe(true);
  expect(() =>
    restoreKeplerAnalysisRecipe(serializeKeplerAnalysisRecipe(recipe), {
      resolveDatasetRevision: () => 2,
      resolveSelectionRevision: () => 4
    })
  ).toThrow('selection');
});

test('Quantile LISA point and polygon recipes reject the retired inference option and resolve explicit alternatives', () => {
  for (const kind of ['quantile-lisa-points', 'quantile-lisa-polygons']) {
    if (recipe.operation.kind !== 'quantile-lisa-points') throw new Error('point fixture');
    const weights =
      kind === 'quantile-lisa-points'
        ? recipe.operation['weights']
        : {
            kind: 'contiguity',
            geometryColumn: 'geometry',
            criterion: 'queen',
            equality: 'exact-source-xy',
            neighborCapacity: 32,
            pairCapacity: 128
          };
    const saved = {...recipe, operation: {...recipe.operation, kind, weights}};
    expect(() =>
      parseKeplerAnalysisRecipe({
        ...saved,
        operation: {...saved.operation, inference: 'legacy-directed-pseudo'}
      })
    ).toThrow(/inference was removed.*alternative/);
    for (const alternative of ['greater', 'two-sided', 'geoda-legacy']) {
      const explicit = parseKeplerAnalysisRecipe({
        ...saved,
        operation: {...saved.operation, alternative}
      });
      expect(explicit.operation).toMatchObject({alternative});
      expect(
        parseKeplerAnalysisRecipe(JSON.parse(serializeKeplerAnalysisRecipe(explicit)))
      ).toEqual(explicit);
    }
    expect(() =>
      parseKeplerAnalysisRecipe({...saved, operation: {...saved.operation, alternative: 'upper'}})
    ).toThrow('greater');
  }
});

test('point and polygon UI settings preserve explicit p-value alternatives', () => {
  const defaults = makePointSheetDefaults(undefined, undefined, 12, {
    quantileCount: 2,
    selectedQuantile: 2,
    seed: 7,
    permutationCount: 9,
    significanceLevel: 0.05
  });
  for (const alternative of ['greater', 'two-sided', 'geoda-legacy'] as const) {
    const point = getQuantileLisaSheetSettings(
      {...defaults, longitudeColumn: 'longitude', latitudeColumn: 'latitude', alternative},
      12
    );
    expect(point.alternative).toBe(alternative);
    expect(normalizeQuantileLisaSettings(point, 12).alternative).toBe(alternative);
    const polygonDefaults = makePolygonSheetDefaults(undefined, 'geometry', '', 12);
    const polygon = getPolygonSheetOperation(
      {...polygonDefaults, kind: 'quantile-lisa-polygons', quantileAlternative: alternative},
      'value',
      12
    );
    expect(polygon).toMatchObject({alternative});
    if (polygon.kind !== 'quantile-lisa-polygons') throw new Error('polygon fixture');
    expect(makePolygonSheetDefaults(polygon, 'geometry', '', 12)['quantileAlternative']).toBe(
      alternative
    );
  }
});

test('Quantile LISA host rejects the retired inference key before applying its default', () => {
  const settings = {
    kind: 'knn' as const,
    longitudeColumn: 'longitude',
    latitudeColumn: 'latitude',
    k: 3,
    quantileCount: 2,
    selectedQuantile: 2,
    seed: 7,
    permutationCount: 9,
    significanceLevel: 0.05
  };
  expect(normalizeQuantileLisaSettings(settings, 12).alternative).toBe('geoda-legacy');
  const retired = {...settings, inference: 'legacy-directed-pseudo'};
  for (const input of [
    retired,
    {...retired, alternative: 'greater' as const},
    Object.assign(Object.create({inference: 'legacy-directed-pseudo'}), settings)
  ]) {
    expect(() => normalizeQuantileLisaSettings(input, 12)).toThrow(
      "inference was removed; use alternative: 'geoda-legacy'"
    );
  }
});

test('Quantile LISA rejects unsupported inference, invalid classes and unrepresentable controls', () => {
  for (const changes of [
    {quantileCount: 1},
    {selectedQuantile: 0},
    {selectedQuantile: 5},
    {seed: -1},
    {seed: Number.MAX_SAFE_INTEGER + 1},
    {permutationCount: 0},
    {permutationCount: MAXIMUM_KEPLER_QUANTILE_LISA_PERMUTATIONS + 1},
    {significanceLevel: Number.MIN_VALUE},
    {significanceLevel: 1 - Number.EPSILON},
    {inference: 'two-sided'},
    {alternative: undefined},
    {multipleTesting: 'benjamini-hochberg'},
    {islands: 'keep'},
    {quantileMethod: 'type-7'},
    {minimumNeighbors: 4}
  ]) {
    expect(() =>
      parseKeplerAnalysisRecipe({...recipe, operation: {...recipe.operation, ...changes}})
    ).toThrow();
  }
});

test('Quantile LISA routing checks source quantile capacity and geographic precision', () => {
  const capabilities = {webgpu: true, availableBytes: 1024, duckdb: false, analysisBackend: false};
  const request = {
    kind: 'analysis' as const,
    recipe,
    input: {
      rowCount: 10,
      chunkCount: 1,
      formats: {value: 'float32', longitude: 'float32', latitude: 'float32'}
    },
    estimatedBytes: 128
  };
  expect(selectKeplerExecutionRoute(request, capabilities)).toEqual({
    backend: 'luma',
    reason: 'supported'
  });
  for (const rowCount of [0, 2, 4, 2 ** 24]) {
    expect(
      selectKeplerExecutionRoute({...request, input: {...request.input, rowCount}}, capabilities)
    ).toEqual({backend: 'unavailable', reason: 'capacity'});
  }
  expect(
    selectKeplerExecutionRoute(
      {
        ...request,
        input: {
          ...request.input,
          formats: {...request.input.formats, longitude: 'float64'}
        }
      },
      capabilities
    )
  ).toEqual({backend: 'unavailable', reason: 'precision'});
});

test('point normalization rejects an explicit null alternative', () => {
  const defaults = makePointSheetDefaults(undefined, undefined, 12, {
    quantileCount: 2,
    selectedQuantile: 2,
    seed: 7,
    permutationCount: 9,
    significanceLevel: 0.05
  });
  const settings = getQuantileLisaSheetSettings(
    {...defaults, longitudeColumn: 'longitude', latitudeColumn: 'latitude', alternative: 'greater'},
    12
  );
  expect(() =>
    normalizeQuantileLisaSettings(
      {...settings, alternative: null} as unknown as typeof settings,
      12
    )
  ).toThrow('alternative');
});

test.each([
  'greater',
  'two-sided'
] as const)('save restore and named topology reselection preserve %s', async alternative => {
  if (recipe.operation.kind !== 'quantile-lisa-points') throw new Error('point fixture');
  const operation = {...recipe.operation, alternative};
  const saved = serializeKeplerAnalysisRecipe({...recipe, operation});
  const restored = restoreKeplerAnalysisRecipe(saved, {
    resolveDatasetRevision: () => 2,
    resolveSelectionRevision: () => 3
  });
  expect(getAnalysisRecipeSettings(restored).quantileLisa?.alternative).toBe(alternative);
  // Exercise the actual runtime methods, substituting only their session/UI dependencies.
  const runtime = Object.create(AnalysisRuntime.prototype);
  Object.defineProperty(runtime, 'datasetId', {value: 'observations'});
  Object.defineProperty(runtime, 'rowCount', {value: 12});
  Object.defineProperty(runtime, 'weightsStore', {
    value: {
      get: () => ({source: {datasetId: 'observations', revision: 2}, settings: operation.weights}),
      bind() {}
    }
  });
  Object.assign(runtime, {
    requestedSettings: {},
    lease: {dataset: {id: 'observations', revision: 2}},
    session: {registry: {getDataset: () => ({revision: 2})}},
    explorer: {currentRevision: 2},
    update(settings: unknown) {
      runtime.requestedSettings = settings;
    },
    notify() {},
    requestUpdate() {},
    revision: 0
  });
  runtime.restoreRecipe(saved);
  expect(runtime.requestedSettings.quantileLisa.alternative).toBe(alternative);
  runtime.selectNamedWeights('knn');
  expect(runtime.requestedSettings.quantileLisa.alternative).toBe(alternative);
  const provider = makeAnalysisTooltipProvider({
    datasetId: 'observations',
    settings: {...runtime.requestedSettings, mode: 'quantile-lisa-points', column: 'value'},
    readRow: async () => ({
      kind: 'quantile-lisa-points',
      value: 1,
      sourceRowIndex: 0,
      sourceRevision: 2,
      analysisRevision: 1,
      alternative: runtime.requestedSettings.quantileLisa.alternative
    })
  });
  expect(await provider('observations', 'layer', 0)).toEqual({
    [`Quantile LISA p-value (${alternative})`]: 1
  });
});

test.each([
  'quantile-lisa-points',
  'quantile-lisa-polygons'
] as const)('%s tooltips label the row p-value with its actual alternative', async kind => {
  for (const alternative of ['greater', 'two-sided', 'geoda-legacy'] as const) {
    const provider = makeAnalysisTooltipProvider({
      datasetId: 'observations',
      settings: {
        mode: kind,
        column: 'value',
        population: 'all',
        classCount: 4,
        variance: 'sample',
        naturalBreaksMode: 'approximate',
        denominatorColumn: ''
      },
      readRow: async () => ({
        kind,
        value: 0.025,
        alternative,
        sourceRowIndex: 0,
        sourceRevision: 2,
        analysisRevision: 1
      })
    });
    expect(await provider('observations', 'layer', 0)).toEqual({
      [alternative === 'geoda-legacy'
        ? 'Legacy pseudo p-value (geoda-legacy)'
        : `Quantile LISA p-value (${alternative})`]: 0.025
    });
  }
});
