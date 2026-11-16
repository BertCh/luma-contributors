// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {createElement} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {AnalysisWeightsPanel} from '../../../examples/deck/kepler-compute/src/analysis-weights-panel';
import {prepareKeplerGeometrySource} from '../../../examples/deck/kepler-compute/src/geometry-preparation/index';
import * as arrow from 'apache-arrow';
import {NullDevice} from '@luma.gl/core/test-utils';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {KeplerComputeSession, type KeplerSessionExplorer} from '../src/session/compute-session';
import type {KeplerPointWeightsSettings} from '../src/recipes/analysis-recipe';
import type {PointTopologySettings} from '../../../examples/deck/kepler-compute/src/analysis-point-weights-settings';
import {AnalysisRuntime} from '../../../examples/deck/kepler-compute/src/analysis-runtime';
import {
  AnalysisWeightsStore,
  getAnalysisWeightsStore,
  parseAnalysisWeightsDefinitions,
  parseAnalysisWeightsWorkspace,
  SANDBOX_EXTENSION_NAMED_WEIGHTS
} from '../../../examples/deck/kepler-compute/src/analysis-weights-store';
import {saveSandbox, replaySandbox} from '../../../examples/deck/kepler-compute/src/sandbox-state';
import type {ComputeRuntime} from '../../../examples/deck/kepler-compute/src/compute-runtime';

const settings: KeplerPointWeightsSettings = {
  kind: 'knn',
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  units: 'meters',
  longitudeColumn: 'longitude',
  latitudeColumn: 'latitude',
  k: 1
};

function fixture() {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const table = arrow.tableFromArrays({
    events: new Float32Array([10, 20, 90]),
    population: new Float32Array([100, 100, 100]),
    longitude: new Float32Array([0, 1, 3]),
    latitude: new Float32Array([0, 0, 0]),
    x: new Float32Array([0, 1000, 1]),
    y: new Float32Array([0, 0, 0]),
    geometry: ['{}', '{}', '{}']
  });
  const dataset = session.importArrowDataset({id: 'observations', table});
  const explorer = {
    dataset,
    isCurrent: true,
    currentRevision: 0
  } as unknown as KeplerSessionExplorer;
  const analyses: AnalysisRuntime[] = [];
  const createAnalysis = () => {
    const analysis = new AnalysisRuntime({session, explorer, requestUpdate() {}});
    analysis.configurePointWeights({
      longitudeColumn: 'longitude',
      latitudeColumn: 'latitude',
      k: 1,
      scale: 100,
      islandPolicy: 'undefined'
    });
    analyses.push(analysis);
    return analysis;
  };
  return {
    device,
    session,
    table,
    dataset,
    explorer,
    createAnalysis,
    cleanup: async () => {
      for (const analysis of analyses) await analysis.destroy(Promise.resolve());
      await session.destroy(Promise.resolve());
      device.destroy();
    }
  };
}

test('two analyses reference one named object; rename preserves identity; editing invalidates both until accepted', async () => {
  const context = fixture();
  try {
    const first = context.createAnalysis();
    const second = context.createAnalysis();
    const store = first.weightsStore;
    const reference = first.weightsReference!;
    expect(second.weightsReference?.weightsId).toBe(reference.weightsId);
    expect(store.getConsumers(reference.weightsId)).toHaveLength(2);
    expect(store.list()).toHaveLength(1);
    store.rename(reference.weightsId, 'Shared source neighbors');
    expect(first.weightsReference).toEqual(reference);
    expect(first.weightsStaleReason).toBeUndefined();
    store.edit(reference.weightsId, {...settings, k: 2});
    expect(store.get(reference.weightsId)).toMatchObject({
      name: 'Shared source neighbors',
      version: 2
    });
    for (const analysis of [first, second]) {
      expect(analysis.weightsStaleReason).toContain('version 2');
      expect(analysis.active).toBe(false);
      expect(analysis.summary).toBeNull();
      expect(analysis.getTableColumns()).toBeUndefined();
    }
    first.configurePointWeights(first.settings.pointWeights!);
    expect(first.weightsStaleReason).toContain('version 2');
    expect(first.weightsReference?.version).toBe(1);
    first.setColumn('population');
    expect(first.weightsStaleReason).toContain('version 2');
    expect(store.list()).toHaveLength(1);
    first.selectNamedWeights(reference.weightsId);
    expect(first.settings.pointWeights).toMatchObject({k: 2});
    expect(first.weightsReference).toMatchObject({weightsId: reference.weightsId, version: 2});
    expect(first.weightsStaleReason).toBeUndefined();
    expect(second.weightsStaleReason).toContain('version 2');
    second.selectNamedWeights(reference.weightsId);
    expect(second.weightsReference?.version).toBe(2);
  } finally {
    await context.cleanup();
  }
});

test('two consumers import the same cached topology; submission and read leases release exactly once', async () => {
  const context = fixture();
  try {
    const store = getAnalysisWeightsStore(context.session);
    const releaseFirstOwner = store.retainSource(context.dataset);
    const releaseSecondOwner = store.retainSource(context.dataset);
    const cache = store.pointCache(context.dataset);
    expect(store.pointCache(context.dataset)).toBe(cache);
    const encode = vi.fn();
    const destroy = vi.fn();
    const entry = {
      key: JSON.stringify([context.dataset.id, context.dataset.revision, settings]),
      references: 0,
      ready: false,
      byteLength: 128,
      diagnostics: [],
      compiled: {encode},
      destroy,
      importBinding: vi.fn(() => ({
        settings,
        source: {datasetId: context.dataset.id, revision: context.dataset.revision}
      }))
    };
    // Replace only GPU compilation: exercise the real cache's lease, submission and eviction paths.
    const createEntry = vi
      .spyOn(cache as unknown as {createEntry(...arguments_: unknown[]): unknown}, 'createEntry')
      .mockReturnValue(entry);
    const first = cache.acquire({} as GPUCommandGraph, settings);
    const second = cache.acquire({} as GPUCommandGraph, settings);
    expect(first.topology).toBeDefined();
    expect(first.topology).toBe(entry);
    expect(second.topology).toBe(first.topology);
    expect(createEntry).toHaveBeenCalledTimes(1);
    expect(entry.references).toBe(2);
    const encoder = context.device.createCommandEncoder();
    first.encode(encoder);
    second.encode(encoder);
    second.markConsumerEncoded(encoder);
    cache.markSubmitted(encoder);
    expect(encode).toHaveBeenCalledTimes(1);
    expect(cache.stats).toMatchObject({buildCount: 1, encodeCount: 1});
    const releaseRead = first.retain();
    first.release();
    first.release();
    second.release();
    expect(entry.references).toBe(1);
    releaseRead();
    releaseRead();
    expect(entry.references).toBe(0);
    releaseFirstOwner();
    expect(destroy).not.toHaveBeenCalled();
    releaseSecondOwner();
    releaseSecondOwner();
    expect(destroy).toHaveBeenCalledTimes(1);
    createEntry.mockRestore();
  } finally {
    await context.cleanup();
  }
});

test('source replacement invalidates all consumers and rejects old definitions; replay can explicitly rebind', async () => {
  const context = fixture();
  try {
    const first = context.createAnalysis();
    const second = context.createAnalysis();
    const store = first.weightsStore;
    const saved = store.serialize();
    const weightsId = first.weightsReference!.weightsId;
    context.session.importArrowDataset({id: context.dataset.id, table: context.table});
    expect(first.weightsStaleReason).toContain('revision changed');
    expect(second.weightsStaleReason).toContain('revision changed');
    expect(() => store.bind('third-analysis', weightsId)).toThrow('source.revision');
    const restored = new AnalysisWeightsStore(context.session);
    expect(() => restored.restore(saved)).toThrow('source.revision');
    restored.restore(saved, true);
    expect(restored.get(weightsId).source.revision).toBe(context.dataset.revision + 1);
    restored.bind('replayed-analysis', weightsId);
    expect(restored.getConsumers(weightsId)).toHaveLength(1);
  } finally {
    await context.cleanup();
  }
});

test('definitions are detached and strict enough to reject invalid settings, duplicate ids and future references atomically', async () => {
  const context = fixture();
  try {
    const store = getAnalysisWeightsStore(context.session);
    const definition = store.create(
      {datasetId: context.dataset.id, revision: context.dataset.revision},
      settings,
      'Neighbors'
    );
    definition.name = 'External mutation';
    definition.source = {...definition.source, revision: 99};
    expect(store.get(definition.id).name).toBe('Neighbors');
    const before = store.serialize();
    expect(() => store.rename(definition.id, ' ')).toThrow('id/name');
    expect(() => store.edit(definition.id, {...settings, k: 0})).toThrow();
    expect(() =>
      store.restore({
        schemaVersion: 1,
        definitions: [store.get(definition.id), store.get(definition.id)]
      })
    ).toThrow('unique');
    expect(store.serialize()).toEqual(before);
    expect(() => parseAnalysisWeightsDefinitions({schemaVersion: 2, definitions: []})).toThrow(
      'version 1'
    );
    expect(() =>
      parseAnalysisWeightsWorkspace({
        ...(before as object),
        bindings: [{datasetId: context.dataset.id, weightsId: definition.id, version: 2}]
      })
    ).toThrow('saved weights version');
    expect(() => store.bind('future', definition.id, 2)).toThrow('consumer.version');
  } finally {
    await context.cleanup();
  }
});

test('sandbox save and replay restore named definitions and the accepted version, including stale references', async () => {
  const context = fixture();
  try {
    const analysis = context.createAnalysis();
    const weightsId = analysis.weightsReference!.weightsId;
    analysis.weightsStore.rename(weightsId, 'Reusable neighbors');
    analysis.weightsStore.edit(weightsId, {...settings, k: 2});
    const runtime = {
      session: context.session,
      explorer: context.explorer,
      analysis,
      publications: new Map(),
      requestUpdate() {}
    } as unknown as ComputeRuntime;
    const callbacks = {
      runtime,
      getNativeConfig: () => ({version: 'v1', config: {visState: {filters: []}}}),
      applyNativeConfig() {}
    };
    const saved = saveSandbox(callbacks);
    const extension = JSON.parse(saved).settings.extensions[SANDBOX_EXTENSION_NAMED_WEIGHTS];
    expect(extension.definitions).toHaveLength(1);
    expect(extension.bindings).toEqual([{datasetId: context.dataset.id, weightsId, version: 1}]);
    await replaySandbox(saved, callbacks);
    expect(analysis.weightsStore.get(weightsId)).toMatchObject({
      name: 'Reusable neighbors',
      version: 2
    });
    expect(analysis.weightsReference).toMatchObject({weightsId, version: 1});
    expect(analysis.weightsStore.list()).toHaveLength(1);
    expect(analysis.weightsStaleReason).toContain('version 2');
    analysis.selectNamedWeights(weightsId);
    const currentSaved = saveSandbox(callbacks);
    await replaySandbox(currentSaved, callbacks);
    expect(analysis.weightsReference).toMatchObject({weightsId, version: 2});
    expect(analysis.weightsStaleReason).toBeUndefined();
    const invalid = JSON.parse(currentSaved);
    invalid.settings.extensions[SANDBOX_EXTENSION_NAMED_WEIGHTS].bindings[0].version = 999;
    await expect(replaySandbox(JSON.stringify(invalid), callbacks)).rejects.toThrow(
      'saved weights version'
    );
    expect(analysis.weightsReference?.version).toBe(2);
  } finally {
    await context.cleanup();
  }
});

test('polygon consumers retain one topology and release an unsubmitted failed build without leaking', async () => {
  const context = fixture();
  try {
    const store = getAnalysisWeightsStore(context.session);
    const source = prepareKeplerGeometrySource([{id: 'areas', rows: [null, null, null]}]);
    const releaseOwner = store.retainSource(context.dataset, {column: 'geometry', source});
    const cache = store.polygonCache(context.dataset);
    expect(store.polygonCache(context.dataset)).toBe(cache);
    const polygonSettings = {
      kind: 'contiguity' as const,
      geometryColumn: 'geometry',
      criterion: 'queen' as const,
      equality: 'exact-source-xy' as const,
      neighborCapacity: 8,
      pairCapacity: 32
    };
    const destroy = vi.fn();
    const entry = {
      key: '',
      references: 0,
      ready: false,
      byteLength: 128,
      diagnostics: [],
      compiled: {encode: vi.fn()},
      destroy,
      importBinding: vi.fn(() => ({settings: polygonSettings}))
    };
    const createEntry = vi
      .spyOn(
        cache as unknown as {createEntry(key: string, settings: unknown): unknown},
        'createEntry'
      )
      .mockImplementation(key => {
        entry.key = key;
        return entry;
      });
    const graph = {device: context.device} as unknown as GPUCommandGraph;
    const first = cache.acquire(graph, polygonSettings);
    const second = cache.acquire(graph, polygonSettings);
    expect(first.topology).toBeDefined();
    expect(first.topology).toBe(entry);
    expect(second.topology).toBe(first.topology);
    expect(createEntry).toHaveBeenCalledTimes(1);
    first.release();
    expect(destroy).not.toHaveBeenCalled();
    second.release();
    second.release();
    expect(destroy).toHaveBeenCalledTimes(1);
    expect(cache.stats.entryCount).toBe(0);
    releaseOwner();
    releaseOwner();
    expect(destroy).toHaveBeenCalledTimes(1);
    createEntry.mockRestore();
  } finally {
    await context.cleanup();
  }
});

test('failed inspector setup releases its request and source lease; the panel renders identity and stale consumer status', async () => {
  const context = fixture();
  try {
    const analysis = context.createAnalysis();
    const store = analysis.weightsStore;
    const weightsId = analysis.weightsReference!.weightsId;
    const beforeBytes = context.session.registry.memoryUsage.totalBytes;
    await expect(store.inspect(weightsId)).rejects.toThrow('WebGPU');
    expect(context.session.registry.getActiveRequests()).toHaveLength(0);
    expect(context.session.registry.memoryUsage.totalBytes).toBe(beforeBytes);
    store.rename(weightsId, 'Shared neighbors');
    store.edit(weightsId, {...settings, k: 2});
    const markup = renderToStaticMarkup(
      createElement(AnalysisWeightsPanel, {analysis, family: 'point', getSettings: () => settings})
    );
    expect(markup).toContain('Shared neighbors');
    expect(markup).toContain('version 2');
    expect(markup).toContain('Weights consumers');
    expect(markup).toContain('stale:');
    expect(markup).toContain('Inspect neighbors');
    expect(markup).toContain('Use current version');
  } finally {
    await context.cleanup();
  }
});

test.each([
  'spatial-rate',
  'hot-spots',
  'bivariate-moran',
  'multivariate-geary',
  'quantile-lisa'
] as const)('%s replaces named point coordinates and topology in both space transitions while preserving analysis settings', async technique => {
  const context = fixture();
  const geographic = {longitudeColumn: 'longitude', latitudeColumn: 'latitude'};
  const planar = {
    coordinateSpace: 'planar' as const,
    xColumn: 'x',
    yColumn: 'y',
    crs: 'EPSG:3857',
    bounds: [0, -1, 1001, 1] as const
  };
  const geographicNearest = {host: {...geographic, k: 1}, definition: settings};
  const planarNearest = {
    host: {...planar, k: 1},
    definition: {
      ...planar,
      kind: 'knn' as const,
      k: 1,
      metric: 'native' as const,
      units: 'meters' as const
    }
  };
  const band = {
    kind: 'distance-band' as const,
    radiusMeters: 2,
    neighborCapacity: 8,
    permutationDegree: 128
  };
  const geographicBand = {
    host: {...geographic, ...band},
    definition: {
      ...geographic,
      coordinateSpace: 'longitude-latitude' as const,
      metric: 'great-circle' as const,
      units: 'meters' as const,
      ...band
    }
  };
  const planarBand = {
    host: {...planar, ...band},
    definition: {...planar, ...band, metric: 'native' as const, units: 'meters' as const}
  };
  const transitions = [
    {from: planarNearest, to: geographicNearest},
    {from: geographicNearest, to: planarNearest},
    {from: planarBand, to: geographicNearest},
    {from: geographicBand, to: planarNearest},
    {from: planarNearest, to: geographicBand},
    {from: geographicNearest, to: planarBand}
  ];
  try {
    const analysis = context.createAnalysis();
    const configure = (topology: PointTopologySettings) => {
      if (technique === 'spatial-rate') {
        const analysisSettings = {scale: 1000, islandPolicy: 'own-rate' as const};
        analysis.configurePointWeights({...topology, ...analysisSettings});
        return analysisSettings;
      }
      if (technique === 'hot-spots') {
        const analysisSettings = {
          minimumNeighbors: 1,
          selfWeight: 2,
          multipleTesting: 'holm' as const,
          islands: 'keep' as const
        };
        analysis.configurePointHotspots({...topology, ...analysisSettings});
        return analysisSettings;
      }
      if (technique === 'quantile-lisa') {
        const analysisSettings = {
          quantileCount: 2,
          selectedQuantile: 2,
          seed: 7,
          permutationCount: 19,
          significanceLevel: 0.125
        };
        analysis.configureQuantileLisa({...topology, ...analysisSettings});
        return analysisSettings;
      }
      const analysisSettings = {
        minimumNeighbors: 1,
        islands: 'keep' as const,
        multipleTesting: 'bonferroni' as const,
        significanceLevel: 0.125,
        inference: 'permutation-conditional-two-sided' as const,
        permutationCount: 19,
        seed: 7,
        ...(technique === 'bivariate-moran'
          ? {secondColumn: 'population'}
          : {variableColumns: ['events', 'population']})
      };
      analysis.configureLocalMoran({...topology, ...analysisSettings});
      return analysisSettings;
    };
    for (const {from, to} of transitions) {
      const analysisSettings = configure(from.host);
      const beforeMode = analysis.settings.mode;
      const definition = analysis.weightsStore.create(
        analysis.namedWeightsSource,
        to.definition,
        'Selected coordinate system'
      );
      analysis.selectNamedWeights(definition.id);
      expect(analysis.settings.mode).toBe(beforeMode);
      expect(analysis.weightsReference).toMatchObject({
        weightsId: definition.id,
        version: definition.version
      });
      expect(JSON.parse(analysis.serializeRecipe()).operation.weights).toEqual(definition.settings);
      const hostSettings =
        technique === 'spatial-rate'
          ? analysis.settings.pointWeights!
          : technique === 'hot-spots'
            ? analysis.settings.pointHotspot!
            : technique === 'quantile-lisa'
              ? analysis.settings.quantileLisa!
              : analysis.settings.localMoran!;
      expect(hostSettings).toMatchObject(analysisSettings);
      const unusedCoordinates =
        to.definition.coordinateSpace === 'planar'
          ? ['longitudeColumn', 'latitudeColumn']
          : ['xColumn', 'yColumn', 'crs', 'bounds'];
      for (const property of unusedCoordinates) expect(hostSettings).not.toHaveProperty(property);
      const unusedTopology =
        definition.settings.kind === 'knn'
          ? ['radiusMeters', 'neighborCapacity', 'permutationDegree']
          : ['k'];
      for (const property of unusedTopology) expect(hostSettings).not.toHaveProperty(property);
    }
  } finally {
    await context.cleanup();
  }
});

test.each([
  'no-named-weights-extension',
  'no-settings'
] as const)('legacy sandbox replay with %s repairs dependency tracking when the recipe is unchanged', async format => {
  const context = fixture();
  try {
    const analysis = context.createAnalysis();
    const runtime = {
      session: context.session,
      explorer: context.explorer,
      analysis,
      publications: new Map(),
      requestUpdate() {}
    } as unknown as ComputeRuntime;
    const callbacks = {
      runtime,
      getNativeConfig: () => ({version: 'v1', config: {visState: {filters: []}}}),
      applyNativeConfig() {}
    };
    const recipe = analysis.serializeRecipe();
    const legacy = JSON.parse(saveSandbox(callbacks));
    if (format === 'no-settings') delete legacy.settings;
    else delete legacy.settings.extensions[SANDBOX_EXTENSION_NAMED_WEIGHTS];
    await replaySandbox(JSON.stringify(legacy), callbacks);
    expect(analysis.serializeRecipe()).toBe(recipe);
    expect(analysis.settings.mode).toBe('spatial-rate-points');
    expect(analysis.weightsStore.list()).toHaveLength(1);
    const reference = analysis.weightsReference;
    expect(reference).toBeDefined();
    expect(reference?.version).toBe(1);
    const saved = JSON.parse(saveSandbox(callbacks));
    const extension = saved.settings.extensions[SANDBOX_EXTENSION_NAMED_WEIGHTS];
    expect(extension.definitions).toHaveLength(1);
    expect(extension.bindings).toEqual([
      {datasetId: context.dataset.id, weightsId: reference!.weightsId, version: 1}
    ]);
    await replaySandbox(JSON.stringify(saved), callbacks);
    expect(analysis.weightsReference).toEqual(reference);
    analysis.weightsStore.edit(reference!.weightsId, {...settings, k: 2});
    expect(analysis.weightsStaleReason).toContain('version 2');
    expect(analysis.active).toBe(false);
  } finally {
    await context.cleanup();
  }
});
