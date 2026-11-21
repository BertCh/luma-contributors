// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {NullDevice} from '@luma.gl/core/test-utils';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUContiguityWeights} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {addKeplerPolygonWeights} from '../src/analysis/polygon-weights';
import {addKeplerAnalysisRecipe} from '../src/analysis/recipe-bindings';
import {
  parseKeplerAnalysisRecipe,
  parseKeplerPolygonWeightsSettings,
  restoreKeplerAnalysisRecipe,
  serializeKeplerAnalysisRecipe,
  type KeplerAnalysisRecipe,
  type KeplerPolygonAnalysisOperation,
  type KeplerPolygonWeightsSettings
} from '../src/recipes/analysis-recipe';
import {selectKeplerExecutionRoute} from '../src/execution/execution-router';

const weights: KeplerPolygonWeightsSettings = {
  kind: 'contiguity',
  geometryColumn: 'geometry',
  criterion: 'queen',
  equality: 'exact-source-xy',
  neighborCapacity: 8,
  pairCapacity: 32
};
const source = {datasetId: 'areas', revision: 2};
const operations: KeplerPolygonAnalysisOperation[] = [
  {
    kind: 'spatial-rate-polygons',
    events: 'events',
    populations: 'population',
    weights,
    scale: 100,
    islandPolicy: 'undefined'
  },
  {
    kind: 'hot-spots-polygons',
    column: 'events',
    weights,
    minimumNeighbors: 1,
    selfWeight: 1,
    multipleTesting: 'none',
    islands: 'exclude',
    inference: 'analytic-two-sided'
  },
  {
    kind: 'local-moran-polygons',
    column: 'events',
    weights,
    minimumNeighbors: 1,
    multipleTesting: 'benjamini-hochberg',
    islands: 'keep',
    significanceLevel: 0.05,
    inference: 'permutation-conditional-two-sided',
    seed: 17,
    permutationCount: 99
  },
  {
    kind: 'quantile-lisa-polygons',
    column: 'events',
    weights,
    quantileCount: 2,
    selectedQuantile: 2,
    seed: 17,
    permutationCount: 99,
    significanceLevel: 0.05,
    alternative: 'geoda-legacy',
    multipleTesting: 'none',
    islands: 'exclude',
    quantileMethod: 'type-5'
  }
];
const makeRecipe = (operation: KeplerPolygonAnalysisOperation): KeplerAnalysisRecipe => ({
  schemaVersion: 1,
  algorithmVersion: 1,
  id: operation.kind,
  source,
  population: 'all',
  backendPolicy: 'require-gpu',
  operation
});

test('polygon recipes preserve exact-source topology and explicit statistical choices', () => {
  for (const operation of operations) {
    const recipe = makeRecipe(operation);
    expect(
      restoreKeplerAnalysisRecipe(serializeKeplerAnalysisRecipe(recipe), {
        resolveDatasetRevision: () => 2
      })
    ).toEqual({
      ...recipe,
      operation:
        operation.kind === 'quantile-lisa-polygons'
          ? {...operation, alternative: 'geoda-legacy'}
          : operation
    });
    for (const rowCount of [3, 4097])
      expect(
        selectKeplerExecutionRoute(
          {
            kind: 'analysis',
            recipe,
            estimatedBytes: 128,
            input: {rowCount, chunkCount: 1, formats: {events: 'float32', population: 'float32'}}
          },
          {webgpu: true, availableBytes: 1024, duckdb: false, analysisBackend: false}
        ).reason
      ).toBe(rowCount === 3 ? 'supported' : 'capacity');
  }
  for (const changes of [
    {criterion: 'distance'},
    {equality: 'float32'},
    {neighborCapacity: 0},
    {neighborCapacity: 262145},
    {pairCapacity: 1048577},
    {pairCapacity: 1.5},
    {snapTolerance: 0.1},
    {geometryColumn: ''}
  ])
    expect(() => parseKeplerPolygonWeightsSettings({...weights, ...changes})).toThrow();
  expect(() =>
    parseKeplerAnalysisRecipe(
      makeRecipe({...operations[2], minimumNeighbors: 2} as KeplerPolygonAnalysisOperation)
    )
  ).toThrow('minimumNeighbors');
});

function makeDevice() {
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
  return device;
}

test('polygon bindings retain original rows and reject stale topology before statistical work', () => {
  const device = makeDevice();
  const graph = new GPUCommandGraph(device);
  const producer = addKeplerPolygonWeights(graph, {
    source,
    settings: weights,
    polygons: {
      kind: 'polygons',
      positions: createTransientView(graph, 'tokens', 'float32x2', 6),
      featureOffsets: createTransientView(graph, 'features', 'uint32', 4),
      polygonOffsets: createTransientView(graph, 'parts', 'uint32', 3),
      ringOffsets: createTransientView(graph, 'rings', 'uint32', 3)
    },
    geometryMask: createTransientView(graph, 'geometry-validity', 'uint32', 3)
  });
  const columns = {
    events: createTransientView(graph, 'events', 'float32', 3),
    population: createTransientView(graph, 'population', 'float32', 3)
  };
  const addCompute = vi.spyOn(graph, 'addComputePass');
  try {
    expect(producer.weights.spatialContext).toBeUndefined();
    expect(producer.weights.status).toBe(producer.status);
    expect(new Set(Object.values(producer.status)).size).toBe(4);
    expect(producer.weights.information).toMatchObject({
      rowCount: 3,
      transform: 'binary',
      symmetric: true,
      hasSelfWeights: false,
      provenance: {source: 'GPUContiguityWeights', criterion: 'queen'}
    });
    for (const invalid of [
      {...producer, source: {...source, revision: 3}},
      {...producer, settings: {...weights, criterion: 'rook' as const}},
      {...producer, settings: {...weights, pairCapacity: 64}},
      {
        ...producer,
        weights: {
          ...producer.weights,
          information: {...producer.weights.information!, transform: 'row' as const}
        }
      }
    ]) {
      expect(() =>
        addKeplerAnalysisRecipe(graph, makeRecipe(operations[0]), {
          source,
          columns,
          polygonWeights: invalid
        })
      ).toThrow();
      expect(addCompute).not.toHaveBeenCalled();
    }
    for (const operation of operations) {
      const result = addKeplerAnalysisRecipe(graph, makeRecipe(operation), {
        source,
        columns,
        polygonWeights: producer
      });
      if (!('polygonWeights' in result)) throw new Error('Expected polygon result');
      expect(result.polygonWeights.weights.offsets).toBe(producer.weights.offsets);
      expect(result.polygonWeights.parameterByteLength).toBe(0);
      expect(result.effectiveMask.length).toBe(3);
      expect('pointWeights' in result).toBe(false);
      if (operation.kind === 'quantile-lisa-polygons') {
        if (!('pseudoPValues' in result)) throw new Error('Expected Quantile LISA result');
        expect(result.parameterByteLength).toBe(24);
        expect(result.pseudoPValues.length).toBe(3);
        expect(result.inference).toMatchObject({
          method: 'legacy-directed-pseudo',
          multipleTesting: 'none',
          twoSidedErrorControl: false
        });
      }
      result.destroyParameters();
    }
  } finally {
    addCompute.mockRestore();
    producer.destroyParameters();
    device.destroy();
  }
});

test('all-empty polygon topology is legal and records an empty CSR without a fabricated polygon', () => {
  const device = makeDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const result = addKeplerPolygonWeights(graph, {
      source,
      settings: weights,
      polygons: {
        kind: 'polygons',
        positions: createTransientView(graph, 'empty-tokens', 'float32x2', 0),
        featureOffsets: createTransientView(graph, 'empty-features', 'uint32', 4),
        polygonOffsets: createTransientView(graph, 'empty-parts', 'uint32', 1),
        ringOffsets: createTransientView(graph, 'empty-rings', 'uint32', 1)
      },
      geometryMask: createTransientView(graph, 'empty-mask', 'uint32', 3)
    });
    expect(result.weights.offsets.length).toBe(4);
    expect(result.geometryMask.length).toBe(3);
    expect(result.parameterByteLength).toBe(0);
    expect(Object.keys(result.status)).toEqual([
      'overflow',
      'requiredCount',
      'candidateOverflow',
      'invalidCount'
    ]);
    result.destroyParameters();
  } finally {
    device.destroy();
  }
});

test('polygon Quantile LISA rejects changed inference and admits only meaningful source quantile counts', () => {
  const operation = operations[3];
  for (const changes of [
    {multipleTesting: 'holm'},
    {islands: 'keep'},
    {inference: 'two-sided'},
    {selectedQuantile: 3}
  ])
    expect(() =>
      parseKeplerAnalysisRecipe({...makeRecipe(operation), operation: {...operation, ...changes}})
    ).toThrow();
  expect(
    selectKeplerExecutionRoute(
      {
        kind: 'analysis',
        recipe: makeRecipe(operation),
        input: {rowCount: 2, chunkCount: 1, formats: {events: 'float32'}},
        estimatedBytes: 128
      },
      {webgpu: true, availableBytes: 1024, duckdb: false, analysisBackend: false}
    ).reason
  ).toBe('capacity');
});

test.each([false, true])('polygon count has a writer when topology is empty=%s', empty => {
  const device = makeDevice();
  const graph = new GPUCommandGraph(device);
  const count = createTransientView(graph, 'retained-count', 'uint32', 1);
  const add = vi.spyOn(graph, 'add');
  const compute = vi.spyOn(graph, 'addComputePass');
  try {
    const result = addKeplerPolygonWeights(graph, {
      source,
      settings: weights,
      polygons: {
        kind: 'polygons',
        positions: createTransientView(graph, 'tokens', 'float32x2', empty ? 0 : 6),
        featureOffsets: createTransientView(graph, 'features', 'uint32', 4),
        polygonOffsets: createTransientView(graph, 'polygons', 'uint32', empty ? 1 : 3),
        ringOffsets: createTransientView(graph, 'rings', 'uint32', empty ? 1 : 3)
      },
      geometryMask: createTransientView(graph, 'mask', 'uint32', 3),
      output: {
        offsets: createTransientView(graph, 'offsets', 'uint32', 4),
        neighbors: createTransientView(graph, 'neighbors', 'uint32', 8),
        weights: createTransientView(graph, 'values', 'float32', 8),
        status: {
          count,
          overflow: createTransientView(graph, 'overflow', 'uint32', 1),
          requiredCount: createTransientView(graph, 'required', 'uint32', 1),
          candidateOverflow: createTransientView(graph, 'candidate', 'uint32', 1),
          invalidCount: createTransientView(graph, 'invalid', 'uint32', 1)
        }
      }
    });
    expect(result.weights.status?.count).toBe(count);
    if (empty) {
      expect(
        compute.mock.calls.some(([node]) =>
          node.resources?.some(
            resource =>
              'buffer' in resource &&
              resource.buffer === count &&
              resource.usage === 'storage-read-write'
          )
        )
      ).toBe(true);
    } else {
      const contributor = add.mock.calls
        .map(([node]) => node)
        .find(node => node instanceof GPUContiguityWeights);
      expect(contributor?.props.output.status.count?.buffer.id).toBe(count.buffer.id);
    }
  } finally {
    add.mockRestore();
    compute.mockRestore();
    device.destroy();
  }
});

test('polygon status rejects destinations without a writer before declaring work', () => {
  const device = makeDevice();
  const graph = new GPUCommandGraph(device);
  const add = vi.spyOn(graph, 'addComputePass');
  try {
    for (const name of ['complete', 'invalidInput']) {
      const status = {
        overflow: createTransientView(graph, `overflow-${name}`, 'uint32', 1),
        requiredCount: createTransientView(graph, `required-${name}`, 'uint32', 1),
        candidateOverflow: createTransientView(graph, `candidate-${name}`, 'uint32', 1),
        invalidCount: createTransientView(graph, `invalid-${name}`, 'uint32', 1),
        [name]: createTransientView(graph, `unsupported-${name}`, 'uint32', 1)
      };
      expect(() =>
        addKeplerPolygonWeights(graph, {
          source,
          settings: weights,
          status,
          polygons: {
            kind: 'polygons',
            positions: createTransientView(graph, `tokens-${name}`, 'float32x2', 0),
            featureOffsets: createTransientView(graph, `features-${name}`, 'uint32', 4),
            polygonOffsets: createTransientView(graph, `polygons-${name}`, 'uint32', 1),
            ringOffsets: createTransientView(graph, `rings-${name}`, 'uint32', 1)
          },
          geometryMask: createTransientView(graph, `mask-${name}`, 'uint32', 3)
        })
      ).toThrow(new RegExp(`kepler-polygon-weights.*${name}.*unsupported`));
    }
    expect(add).not.toHaveBeenCalled();
  } finally {
    add.mockRestore();
    device.destroy();
  }
});
