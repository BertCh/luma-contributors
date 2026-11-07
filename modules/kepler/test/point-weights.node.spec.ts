// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {NullDevice} from '@luma.gl/core/test-utils';
import {createTransientView, GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  parseKeplerAnalysisRecipe,
  parseKeplerPointWeightsSettings,
  restoreKeplerAnalysisRecipe,
  serializeKeplerAnalysisRecipe,
  type KeplerAnalysisRecipe,
  type KeplerPointWeightsSettings
} from '../src/recipes/analysis-recipe';
import {selectKeplerExecutionRoute} from '../src/execution/execution-router';
import {
  addKeplerPointWeights,
  getKeplerPointWeightsCapacity,
  getKeplerPointWeightsMaximumNeighbors,
  type KeplerPointWeightsBinding
} from '../src/analysis/point-weights';
import {addKeplerAnalysisRecipe} from '../src/analysis/recipe-bindings';

const weights: KeplerPointWeightsSettings = {
  kind: 'knn',
  longitudeColumn: 'longitude',
  latitudeColumn: 'latitude',
  k: 2,
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  units: 'meters'
};
const recipe: KeplerAnalysisRecipe = {
  schemaVersion: 1,
  algorithmVersion: 1,
  id: 'point-rates',
  source: {datasetId: 'points', revision: 2},
  population: 'all',
  backendPolicy: 'require-gpu',
  operation: {
    kind: 'spatial-rate-points',
    events: 'events',
    populations: 'population',
    weights,
    scale: 1000,
    islandPolicy: 'undefined'
  }
};

test('point recipes detach explicit geographic choices and restore without external weight handles', () => {
  const parsed = parseKeplerAnalysisRecipe(recipe);
  expect(parsed).toEqual(recipe);
  if (parsed.operation.kind !== 'spatial-rate-points') throw new Error('Expected point recipe');
  expect(parsed.operation.weights).not.toBe(weights);
  expect(Object.isFrozen(parsed.operation.weights)).toBe(true);
  expect(
    restoreKeplerAnalysisRecipe(serializeKeplerAnalysisRecipe(recipe), {
      resolveDatasetRevision: () => 2
    })
  ).toEqual(recipe);
  for (const override of [
    {k: 0},
    {k: 33},
    {k: 1.5},
    {metric: 'native'},
    {units: 'kilometers'},
    {coordinateSpace: 'planar'},
    {longitudeColumn: ''},
    {hiddenMask: true}
  ]) {
    expect(() =>
      parseKeplerAnalysisRecipe({
        ...recipe,
        operation: {...recipe.operation, weights: {...weights, ...override}}
      })
    ).toThrow();
  }
});

test('point routing checks coordinate precision and zero-row capacity before GPU execution', () => {
  const capabilities = {webgpu: true, availableBytes: 1024};
  const request = {
    kind: 'analysis' as const,
    recipe,
    input: {
      rowCount: 3,
      chunkCount: 1,
      formats: {events: 'float32', population: 'float32', longitude: 'float32', latitude: 'float32'}
    },
    estimatedBytes: 128
  };
  expect(selectKeplerExecutionRoute(request, capabilities).backend).toBe('luma');
  expect(
    selectKeplerExecutionRoute(
      {
        ...request,
        input: {...request.input, formats: {...request.input.formats, longitude: 'float64'}}
      },
      capabilities
    )
  ).toEqual({backend: 'unavailable', reason: 'precision'});
  expect(
    selectKeplerExecutionRoute(
      {
        ...request,
        input: {...request.input, rowCount: 0}
      },
      capabilities
    )
  ).toEqual({backend: 'unavailable', reason: 'capacity'});
});

test('point weights reserve complete directed kNN capacity and preserve source metadata', () => {
  const device = new NullDevice({});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.assign(device.limits, {
    maxStorageBuffersPerShaderStage: 8,
    maxStorageBufferBindingSize: 128 * 1024 * 1024,
    maxBufferSize: 256 * 1024 * 1024,
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupSizeX: 256,
    maxComputeWorkgroupStorageSize: 16_384,
    maxComputeWorkgroupsPerDimension: 65_535
  });
  const graph = new GPUCommandGraph(device);
  const longitude = createTransientView(graph, 'longitude', 'float32', 3);
  const latitude = createTransientView(graph, 'latitude', 'float32', 3);
  const result = addKeplerPointWeights(graph, {
    source: recipe.source,
    settings: {...weights, k: 32},
    longitude,
    latitude
  });
  try {
    expect(result.weights.neighbors.length).toBe(6);
    expect(result.weights.offsets.length).toBe(4);
    expect(result.coordinateMask.length).toBe(3);
    expect(result.source).toEqual(recipe.source);
    expect(result.weights.information?.transform).toBe('binary');
    expect(result.weights.spatialContext?.metric).toBe('great-circle');
    expect(result.parameterByteLength).toBe(48);
  } finally {
    result.destroyParameters();
    device.destroy();
  }
});

test('point hotspots persist inference and island choices and validate geographic precision', () => {
  const hotspot: KeplerAnalysisRecipe = {
    ...recipe,
    operation: {
      kind: 'hot-spots-points',
      column: 'events',
      weights,
      minimumNeighbors: 1,
      selfWeight: 1,
      multipleTesting: 'benjamini-hochberg',
      inference: 'analytic-two-sided',
      islands: 'keep'
    }
  };
  expect(
    restoreKeplerAnalysisRecipe(serializeKeplerAnalysisRecipe(hotspot), {
      resolveDatasetRevision: () => 2
    })
  ).toEqual(hotspot);
  for (const changes of [
    {minimumNeighbors: 0},
    {selfWeight: -1},
    {islands: 'own-rate'},
    {inference: 'permutation'}
  ]) {
    expect(() =>
      parseKeplerAnalysisRecipe({...hotspot, operation: {...hotspot.operation, ...changes}})
    ).toThrow();
  }
  expect(
    selectKeplerExecutionRoute(
      {
        kind: 'analysis',
        recipe: hotspot,
        input: {
          rowCount: 3,
          chunkCount: 1,
          formats: {events: 'float32', latitude: 'float32', longitude: 'float64'}
        },
        estimatedBytes: 128
      },
      {webgpu: true, availableBytes: 1024}
    )
  ).toEqual({backend: 'unavailable', reason: 'precision'});
});

test('borrowed point weights validate source, settings, topology and graph before recording dependent work', () => {
  const device = new NullDevice({});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.assign(device.limits, {
    maxStorageBuffersPerShaderStage: 8,
    maxStorageBufferBindingSize: 128 * 1024 * 1024,
    maxBufferSize: 256 * 1024 * 1024,
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupSizeX: 256,
    maxComputeWorkgroupStorageSize: 16_384,
    maxComputeWorkgroupsPerDimension: 65_535
  });
  const graph = new GPUCommandGraph(device);
  const status = {
    overflow: createTransientView(graph, 'overflow', 'uint32', 1),
    requiredCount: createTransientView(graph, 'required', 'uint32', 1),
    invalidCount: createTransientView(graph, 'invalid', 'uint32', 1)
  };
  const pointWeights: KeplerPointWeightsBinding = {
    source: recipe.source,
    settings: weights,
    coordinateMask: createTransientView(graph, 'coordinate-mask', 'uint32', 3),
    weights: {
      offsets: createTransientView(graph, 'offsets', 'uint32', 4),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 6),
      weights: createTransientView(graph, 'weights', 'float32', 6),
      information: {
        rowCount: 3,
        transform: 'binary',
        symmetric: 'unknown',
        hasSelfWeights: false,
        provenance: {source: 'GPUNeighborSearch', criterion: 'knn', k: weights.k}
      },
      spatialContext: {
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle',
        units: 'meters'
      },
      status
    },
    status
  };
  const columns = {
    events: createTransientView(graph, 'events', 'float32', 3),
    population: createTransientView(graph, 'population', 'float32', 3)
  };
  const otherGraph = new GPUCommandGraph(device);
  const wrongMask = createTransientView(otherGraph, 'wrong-mask', 'uint32', 3);
  const wrongRows = createTransientView(graph, 'wrong-rows', 'uint32', 2);
  const addCompute = vi.spyOn(graph, 'addComputePass');
  try {
    for (const invalid of [
      {...pointWeights, source: {...recipe.source, revision: 3}},
      {...pointWeights, source: {...recipe.source, datasetId: 'other'}},
      {...pointWeights, settings: {...weights, k: 1}},
      {...pointWeights, settings: {...weights, longitudeColumn: 'other'}},
      {
        ...pointWeights,
        weights: {
          ...pointWeights.weights,
          information: {...pointWeights.weights.information!, transform: 'row' as const}
        }
      },
      {
        ...pointWeights,
        weights: {
          ...pointWeights.weights,
          information: {...pointWeights.weights.information!, hasSelfWeights: true}
        }
      },
      {
        ...pointWeights,
        weights: {
          ...pointWeights.weights,
          spatialContext: {coordinateSpace: 'planar' as const, metric: 'native' as const}
        }
      },
      {...pointWeights, coordinateMask: wrongRows},
      {...pointWeights, coordinateMask: wrongMask}
    ]) {
      expect(() =>
        addKeplerAnalysisRecipe(graph, recipe, {
          source: recipe.source,
          columns,
          pointWeights: invalid
        })
      ).toThrow();
      expect(addCompute).not.toHaveBeenCalled();
    }
    const result = addKeplerAnalysisRecipe(graph, recipe, {
      source: recipe.source,
      columns,
      pointWeights
    });
    if (!('pointWeights' in result)) throw new Error('Expected point weights result');
    expect(result.pointWeights.parameterByteLength).toBe(0);
    expect(result.pointWeights.coordinateMask).toBe(pointWeights.coordinateMask);
    expect(result.pointWeights.weights.offsets).toBe(pointWeights.weights.offsets);
    result.destroyParameters();
  } finally {
    addCompute.mockRestore();
    device.destroy();
  }
});

const distanceBand: KeplerPointWeightsSettings = {
  kind: 'distance-band',
  longitudeColumn: 'longitude',
  latitudeColumn: 'latitude',
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  units: 'meters',
  radiusMeters: 1234.5,
  neighborCapacity: 6
};

test('distance-band recipes retain radius and total capacity without kNN fields', () => {
  const bandRecipe = parseKeplerAnalysisRecipe({
    ...recipe,
    operation: {...recipe.operation, weights: distanceBand}
  });
  expect(
    restoreKeplerAnalysisRecipe(serializeKeplerAnalysisRecipe(bandRecipe), {
      resolveDatasetRevision: () => 2
    })
  ).toEqual(bandRecipe);
  const normalized = parseKeplerPointWeightsSettings({...distanceBand, radiusMeters: 0.1});
  expect(normalized).toEqual({...distanceBand, radiusMeters: Math.fround(0.1)});
  expect(Object.isFrozen(normalized)).toBe(true);
  for (const radiusMeters of [0, -1, Infinity, NaN, Number.MIN_VALUE, Number.MAX_VALUE])
    expect(() => parseKeplerPointWeightsSettings({...distanceBand, radiusMeters})).toThrow();
  for (const neighborCapacity of [0, -1, 1.5, 2 ** 25 + 1, Infinity])
    expect(() => parseKeplerPointWeightsSettings({...distanceBand, neighborCapacity})).toThrow();
  expect(() => parseKeplerPointWeightsSettings({...distanceBand, k: 2})).toThrow();
  expect(() => parseKeplerPointWeightsSettings({...weights, radiusMeters: 10})).toThrow();
  expect(getKeplerPointWeightsCapacity(weights, 1)).toBe(1);
  expect(getKeplerPointWeightsCapacity(weights, 3)).toBe(6);
  expect(getKeplerPointWeightsCapacity(distanceBand, 100)).toBe(6);
  expect(getKeplerPointWeightsCapacity({...distanceBand, neighborCapacity: 1}, 1)).toBe(1);
  expect(getKeplerPointWeightsMaximumNeighbors(weights)).toBe(2);
  expect(getKeplerPointWeightsMaximumNeighbors(distanceBand)).toBe(64);
});

test('distance-band GPU graph uses explicit capacity and rejects stale reusable radius topology', () => {
  const device = new NullDevice({});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.assign(device.limits, {
    maxStorageBuffersPerShaderStage: 8,
    maxStorageBufferBindingSize: 128 * 1024 * 1024,
    maxBufferSize: 256 * 1024 * 1024,
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupSizeX: 256,
    maxComputeWorkgroupStorageSize: 16384,
    maxComputeWorkgroupsPerDimension: 65535
  });
  const graph = new GPUCommandGraph(device);
  const producer = addKeplerPointWeights(graph, {
    source: recipe.source,
    settings: distanceBand,
    longitude: createTransientView(graph, 'band-longitude', 'float32', 3),
    latitude: createTransientView(graph, 'band-latitude', 'float32', 3)
  });
  const bandRecipe = parseKeplerAnalysisRecipe({
    ...recipe,
    operation: {...recipe.operation, weights: distanceBand}
  });
  const columns = {
    events: createTransientView(graph, 'band-events', 'float32', 3),
    population: createTransientView(graph, 'band-population', 'float32', 3)
  };
  const addCompute = vi.spyOn(graph, 'addComputePass');
  try {
    expect(producer.weights.neighbors.length).toBe(6);
    expect(producer.weights.information?.provenance?.criterion).toBe('distance-band');
    expect(producer.weights.information?.provenance?.k).toBeUndefined();
    expect(producer.weights.information?.transform).toBe('binary');
    expect(producer.weights.information?.hasSelfWeights).toBe(false);
    expect(producer.parameterByteLength).toBe(48);
    for (const settings of [
      {...distanceBand, radiusMeters: 2345},
      {...distanceBand, neighborCapacity: 5},
      weights
    ]) {
      expect(() =>
        addKeplerAnalysisRecipe(graph, bandRecipe, {
          source: recipe.source,
          columns,
          pointWeights: {...producer, settings}
        })
      ).toThrow('stale');
      expect(addCompute).not.toHaveBeenCalled();
    }
    expect(() =>
      addKeplerAnalysisRecipe(graph, bandRecipe, {
        source: recipe.source,
        columns,
        pointWeights: {
          ...producer,
          weights: {
            ...producer.weights,
            information: {
              ...producer.weights.information!,
              provenance: {source: 'GPUNeighborSearch', criterion: 'knn', k: 2}
            }
          }
        }
      })
    ).toThrow('metadata');
    expect(addCompute).not.toHaveBeenCalled();
    const result = addKeplerAnalysisRecipe(graph, bandRecipe, {
      source: recipe.source,
      columns,
      pointWeights: producer
    });
    if (!('pointWeights' in result)) throw new Error('Expected point topology');
    expect(result.pointWeights.weights.neighbors).toBe(producer.weights.neighbors);
    expect(result.pointWeights.parameterByteLength).toBe(0);
    result.destroyParameters();
  } finally {
    addCompute.mockRestore();
    producer.destroyParameters();
    device.destroy();
  }
});

test('distance-band source admission protects uint32 required-slot totals without changing kNN', () => {
  expect(getKeplerPointWeightsCapacity(distanceBand, 65536)).toBe(6);
  // Above 65536 rows the build is admitted and a 64-bit guard detects wrapped totals.
  expect(getKeplerPointWeightsCapacity(distanceBand, 65537)).toBe(6);
  expect(() => getKeplerPointWeightsCapacity(distanceBand, 2 ** 24 + 1)).toThrow();
  expect(getKeplerPointWeightsCapacity(weights, 65537)).toBe(65537 * 2);
  for (const rowCount of [65536, 2 ** 24 + 1]) {
    const bandRecipe = parseKeplerAnalysisRecipe({
      ...recipe,
      operation: {...recipe.operation, weights: distanceBand}
    });
    expect(
      selectKeplerExecutionRoute(
        {
          kind: 'analysis',
          recipe: bandRecipe,
          input: {
            rowCount,
            chunkCount: 1,
            formats: {
              events: 'float32',
              population: 'float32',
              longitude: 'float32',
              latitude: 'float32'
            }
          },
          estimatedBytes: 128
        },
        {webgpu: true, availableBytes: 1024}
      ).reason
    ).toBe(rowCount === 65536 ? 'supported' : 'capacity');
  }
});
