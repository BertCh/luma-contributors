// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {GPUInverseDistanceWeighting} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-inverse-distance-weighting';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';
import {GPUNearestFeatureWeights} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-weights';
import {getStorageTopKNodeRecord} from '../../../src/analysis-core/kernels/storage-backed-top-k';
import {getNearestNeighborNodes} from '../../../src/gpu-spatial-analysis/spatial-join/nearest-nodes';
import {createNullWebGPUDevice, createVectorView} from '../../utils/gpu-contributor-test-utils';
import {getReservedWGSLWords} from './wgsl-reserved-words';

const viewSerials = new WeakMap<GPUCommandGraph, number>();

it('uses the full WGSL reserved-word set and accepts sample identifiers', () => {
  expect(getReservedWGSLWords('let sample = 1u; // target\n/* active */')).toEqual([]);
  expect(getReservedWGSLWords('target active filter module meta')).toEqual([
    'target',
    'active',
    'filter',
    'module',
    'meta'
  ]);
});

function createViews(graph: GPUCommandGraph) {
  return <Format extends 'float32' | 'float32x2' | 'uint32'>(format: Format, length: number) => {
    const serial = viewSerials.get(graph) ?? 0;
    viewSerials.set(graph, serial + 1);
    return createTransientView(graph, `top-k-view-${serial}`, format, length);
  };
}

function recordPipelines(device: Device, nodes: readonly unknown[]) {
  const pipelines: {id: string; source: string; key: string}[] = [];
  Object.defineProperty(device, 'props', {
    value: {...device.props, _cacheShaders: false, _cachePipelines: false},
    configurable: true
  });
  Object.defineProperty(device, 'createComputePipeline', {
    configurable: true,
    value: (props: {id?: string; shader: {source: string}; constants?: object}) => {
      const source = props.shader.source;
      expect(getReservedWGSLWords(source), props.id).toEqual([]);
      expect(() => new WgslReflect(source), props.id).not.toThrow();
      pipelines.push({id: props.id ?? '', source, key: source + JSON.stringify(props.constants)});
      return {id: props.id ?? '', hash: '', destroy() {}};
    }
  });
  for (const node of nodes as {compile?: (context: {device: Device}) => unknown}[]) {
    node.compile?.({device});
  }
  return pipelines;
}

function createInterpolation(
  graph: GPUCommandGraph,
  capacity: number,
  cells = 5,
  tileSize?: number
) {
  const view = createViews(graph);
  return new GPUInverseDistanceWeighting({
    positions: view('float32x2', 1100),
    values: view('float32', 1100),
    mask: view('uint32', 1100),
    parameters: view('float32', 9),
    grid: {width: cells, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
    indexGridSize: [4, 4],
    indexBounds: [-1, -1, 2, 2],
    maximumNeighborCount: capacity,
    invocationTileSize: tileSize,
    output: {values: view('float32', cells), counts: view('uint32', cells)}
  });
}

it('rejects non-f32 shader constants at construction', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = createViews(graph);
  const interpolation = createInterpolation(graph, 65);
  for (const value of [NaN, Infinity, -Infinity, 1e100, -1e100]) {
    expect(
      () =>
        new GPUInverseDistanceWeighting({
          ...interpolation.props,
          indexBounds: [value, -1, 2, 2]
        })
    ).toThrow(/inverse-distance-weighting indexBounds\[0\]/);
  }
  expect(
    () =>
      new GPUInverseDistanceWeighting({
        ...interpolation.props,
        indexGridSize: [0x100000000, 1]
      })
  ).toThrow(/inverse-distance-weighting indexGridSize\[0\]/);
  const weights = new GPUNearestFeatureWeights({
    slotCapacity: 65,
    neighborIds: view('uint32', 65),
    neighborCounts: view('uint32', 1),
    output: {
      offsets: view('uint32', 2),
      neighbors: view('uint32', 65),
      weights: view('float32', 65),
      status: {overflow: view('uint32', 1)}
    }
  });
  for (const value of [NaN, Infinity, 1e100, 1e-100, 0, -1]) {
    for (const name of ['power', 'distanceFloor'] as const) {
      expect(() => new GPUNearestFeatureWeights({...weights.props, [name]: value})).toThrow(
        new RegExp(`nearest-feature-weights ${name}`)
      );
    }
  }
  for (const sphereRadius of [1e100, 1e-100]) {
    const spatialContext = {
      coordinateSpace: 'longitude-latitude',
      metric: 'great-circle',
      sphereRadius
    } as const;
    expect(() => new GPUInverseDistanceWeighting({...interpolation.props, spatialContext})).toThrow(
      /inverse-distance-weighting spatialContext.sphereRadius/
    );
    expect(
      () =>
        new GPUNearestFeatureJoin({
          spatialContext,
          queries: {kind: 'points', positions: view('float32x2', 1)},
          features: {kind: 'points', positions: view('float32x2', 1)},
          output: {
            neighborIds: view('uint32', 1),
            neighborCounts: view('uint32', 1),
            status: {overflow: view('uint32', 1)}
          }
        })
    ).toThrow(/nearest-feature-join spatialContext.sphereRadius/);
  }
  device.destroy();
});

it.each([
  64, 65, 256, 1024
])('IDW compiles bounded private or storage selection for capacity %i', capacity => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const contributor = createInterpolation(graph, capacity);
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
  const gather = recordPipelines(device, nodes).find(pipeline => pipeline.id.includes('-gather-'))!;
  expect(gather.source.includes('var nearRows: array')).toBe(capacity <= 64);
  expect(gather.source.includes('var nearDistances: array')).toBe(capacity <= 64);
  expect(/\blet sample =/.test(gather.source)).toBe(true);
  device.destroy();
});

it('IDW shader and pipeline counts stay constant as gather tile count grows', () => {
  const counts = [5, 9, 13].map(cells => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const contributor = createInterpolation(graph, 256, cells, 4);
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    const pipelines = recordPipelines(device, nodes);
    const gather = pipelines.filter(pipeline => pipeline.id.includes('-gather-'));
    expect(new Set(gather.map(pipeline => pipeline.key)).size).toBe(1);
    const result = [
      new Set(pipelines.map(pipeline => pipeline.source)).size,
      new Set(pipelines.map(pipeline => pipeline.key)).size
    ];
    device.destroy();
    return result;
  });
  expect(counts[1]).toEqual(counts[0]);
  expect(counts[2]).toEqual(counts[0]);
});

it.each([
  false,
  true
])('nearest storage selection shares pipelines over unequal chunks (querySort=%s)', querySort => {
  const counts = [2, 3, 4].map(chunkCount => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const view = createViews(graph);
    const chunks = Array.from({length: chunkCount}, (_, index) =>
      graph.createDataView(view('float32x2', 100).buffer, {
        format: 'float32x2',
        byteOffset: (index % 3) * 4,
        length: index === chunkCount - 1 ? 5 : 65
      })
    );
    const queryCount = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const nodes = getNearestNeighborNodes(graph, {
      id: 'storage-nearest',
      operation: 'GPUNearestFeatureJoin',
      queries: {kind: 'points', points: createVectorView('queries', 'float32x2', chunks)},
      queryCount,
      features: {kind: 'points', positions: view('float32x2', 1100)},
      featureCount: 1100,
      k: 256,
      capacity: 256,
      leafCapacity: 2048,
      spatialSort: false,
      querySort,
      scratchChunkQueryCount: 64,
      neighborIds: view('uint32', queryCount * 256),
      neighborCounts: view('uint32', queryCount),
      neighborDistances: view('float32', queryCount * 256),
      neighborFootPoints: view('float32x2', queryCount * 256),
      overflow: view('uint32', 1)
    }).nodes;
    const pipelines = recordPipelines(device, nodes);
    const traversals = pipelines.filter(pipeline => pipeline.id.endsWith('-traverse'));
    expect(new Set(traversals.map(pipeline => pipeline.key)).size).toBe(1);
    expect(traversals[0].source).not.toContain('var listRow: array');
    for (const suffix of ['-columns', '-foot-points', '-foot-columns']) {
      const matching = pipelines.filter(
        pipeline =>
          pipeline.id.endsWith(suffix) &&
          (suffix !== '-columns' || !pipeline.id.endsWith('-foot-columns'))
      );
      expect(new Set(matching.map(pipeline => pipeline.key)).size).toBe(1);
    }
    const result = [
      new Set(pipelines.map(pipeline => pipeline.source)).size,
      new Set(pipelines.map(pipeline => pipeline.key)).size
    ];
    device.destroy();
    return result;
  });
  expect(counts[1]).toEqual(counts[0]);
  expect(counts[2]).toEqual(counts[0]);
});

it.each([64, 65, 256, 1024])('weights compiles storage sorting above 64 slots (%i)', capacity => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: capacity * 8 + 4});
  const view = createViews(graph);
  const contributor = new GPUNearestFeatureWeights({
    slotCapacity: capacity,
    neighborIds: view('uint32', capacity * 2),
    neighborCounts: view('uint32', 2),
    neighborDistances: view('float32', capacity * 2),
    output: {
      offsets: view('uint32', 3),
      neighbors: view('uint32', capacity * 2),
      weights: view('float32', capacity * 2),
      distances: view('float32', capacity * 2),
      status: {overflow: view('uint32', 1)}
    }
  });
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
  const pipelines = recordPipelines(device, nodes);
  expect(
    new Set(
      pipelines.filter(pipeline => pipeline.id.includes('-emit')).map(pipeline => pipeline.key)
    ).size
  ).toBe(1);
  const emit = pipelines.find(pipeline => pipeline.id.includes('-emit'))!;
  expect(emit.source.includes('var keptIds: array')).toBe(capacity <= 64);
  device.destroy();
});

it('rejects IDW storage scratch before declaring graph resources', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const contributor = createInterpolation(graph, 32768);
  const declare = vi.spyOn(graph, 'createTransientBuffer');
  Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 65536});
  expect(() => contributor.getCommandNodes(graph)).toThrow(/inverse-distance-weighting .*binding/);
  expect(declare).not.toHaveBeenCalled();
  device.destroy();
});

it('rejects aligned nearest output ranges before declaring graph resources', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = createViews(graph);
  const queryCount = 252;
  const capacity = 65;
  const backing = view('uint32', queryCount * capacity + 63);
  const neighborIds = graph.createDataView(backing.buffer, {
    format: 'uint32',
    byteOffset: 252,
    length: queryCount * capacity
  });
  const contributor = new GPUNearestFeatureJoin({
    queries: {kind: 'points', positions: view('float32x2', queryCount)},
    features: {kind: 'points', positions: view('float32x2', 8)},
    k: capacity,
    output: {
      neighborIds,
      neighborCounts: view('uint32', queryCount),
      status: {overflow: view('uint32', 1)}
    }
  });
  const declare = vi.spyOn(graph, 'createTransientBuffer');
  Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 65536});
  expect(() => contributor.getCommandNodes(graph)).toThrow(/nearest-feature-join .*binding/);
  expect(declare).not.toHaveBeenCalled();
  device.destroy();
});

it('rejects overlapping writable weights fields and bounds the count scan total at construction', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = createViews(graph);
  const backing = view('uint32', 300);
  const output = {
    offsets: graph.createDataView(backing.buffer, {format: 'uint32', length: 2}),
    neighbors: graph.createDataView(backing.buffer, {format: 'uint32', byteOffset: 4, length: 256}),
    weights: view('float32', 256),
    status: {overflow: view('uint32', 1)}
  };
  expect(
    () =>
      new GPUNearestFeatureWeights({
        slotCapacity: 256,
        neighborIds: view('uint32', 256),
        neighborCounts: view('uint32', 1),
        output
      })
  ).toThrow(/overlap/);
  expect(
    () =>
      new GPUNearestFeatureWeights({
        slotCapacity: 65536,
        neighborIds: view('uint32', 1),
        neighborCounts: view('uint32', 65536),
        output
      })
  ).toThrow(/rows \* slotCapacity/);
  device.destroy();
});

it('fits 63 large-k scratch rows when fewer than 64 rows fit a binding', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = createViews(graph);
  const contributor = new GPUNearestFeatureJoin({
    id: 'large-selection',
    queries: {kind: 'points', positions: view('float32x2', 64)},
    features: {kind: 'points', positions: view('float32x2', 3)},
    k: 262144,
    querySort: false,
    spatialSort: false,
    output: {
      neighborIds: view('uint32', 64 * 262144),
      neighborCounts: view('uint32', 64),
      status: {overflow: view('uint32', 1)}
    }
  });
  const nodes = contributor.getCommandNodes(graph);
  const traversals = nodes.filter(node => node.id.endsWith('-traverse'));
  const records = traversals.map(node => getStorageTopKNodeRecord(node)!);
  expect(records.map(record => record.settings[0])).toEqual([63, 1]);
  expect(new Set(records.map(record => record.source)).size).toBe(1);
  const scratch = records[0].bindings.find(binding => binding.name === 'topk')!.view;
  expect(scratch.buffer.byteLength).toBe(132121344);
  expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
  device.destroy();
});

it.each([
  false,
  true
])('profiles 2000 short chunks at their own workgroup counts (querySort=%s)', querySort => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = createViews(graph);
  const chunks = Array.from({length: 2000}, () => view('float32x2', 64));
  const queryCount = 128000;
  const nodes = getNearestNeighborNodes(graph, {
    id: 'short-chunks',
    operation: 'GPUNearestFeatureJoin',
    queries: {kind: 'points', points: createVectorView('many-queries', 'float32x2', chunks)},
    queryCount,
    features: {kind: 'points', positions: view('float32x2', 3)},
    featureCount: 3,
    k: 65,
    capacity: 65,
    leafCapacity: 4,
    spatialSort: false,
    querySort,
    neighborIds: view('uint32', queryCount * 65),
    neighborCounts: view('uint32', queryCount),
    overflow: view('uint32', 1)
  }).nodes;
  const records = nodes
    .filter(node => node.id.endsWith('-traverse'))
    .map(node => getStorageTopKNodeRecord(node)!);
  expect(records).toHaveLength(2000);
  if (querySort) {
    const keyRecords = nodes
      .filter(node => /-query-keys-\d+$/.test(node.id))
      .map(node => getStorageTopKNodeRecord(node)!);
    expect(keyRecords.reduce((total, record) => total + record.settings[0], 0)).toBe(queryCount);
    expect(new Set(keyRecords.map(record => record.source)).size).toBe(1);
  }
  expect(records.reduce((total, record) => total + record.settings[0], 0)).toBe(queryCount);
  expect(
    records.reduce(
      (total, record) => total + record.layout.x * record.layout.y * record.layout.z,
      0
    )
  ).toBe(2000);
  expect(new Set(records.map(record => record.source)).size).toBe(1);
  device.destroy();
});

it('accepts empty point and ID chunk vectors in nearest-feature mode', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = createViews(graph);
  const contributor = new GPUNearestFeatureJoin({
    queries: {kind: 'points', positions: createVectorView('empty-points', 'float32x2', [])},
    features: {kind: 'points', positions: view('float32x2', 1)},
    radius: view('float32', 1),
    candidateCapacity: 8,
    output: {
      nearestFeatureIds: createVectorView('empty-ids', 'uint32', []),
      status: {overflow: view('uint32', 1)}
    }
  });
  expect(contributor.queryCount).toBe(0);
  expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  device.destroy();
});

it('rejects aligned aliases for IDW and weights before any node declaration', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = createViews(graph);
  const backing = view('uint32', 512);
  const source = createInterpolation(graph, 65, 1).props;
  expect(
    () =>
      new GPUInverseDistanceWeighting({
        ...source,
        output: {
          values: graph.createDataView(backing.buffer, {
            format: 'float32',
            byteOffset: 0,
            length: 1
          }),
          counts: graph.createDataView(backing.buffer, {format: 'uint32', byteOffset: 4, length: 1})
        }
      })
  ).toThrow(/aligned binding byte ranges must not overlap/);
  expect(
    () =>
      new GPUNearestFeatureWeights({
        slotCapacity: 65,
        neighborIds: view('uint32', 65),
        neighborCounts: view('uint32', 1),
        output: {
          offsets: graph.createDataView(backing.buffer, {
            format: 'uint32',
            byteOffset: 0,
            length: 2
          }),
          neighbors: graph.createDataView(backing.buffer, {
            format: 'uint32',
            byteOffset: 8,
            length: 65
          }),
          weights: view('float32', 65),
          status: {overflow: view('uint32', 1)}
        }
      })
  ).toThrow(/aligned binding byte ranges must not overlap/);
  device.destroy();
});

it('nearest rejects aligned writable aliases across fields and chunks', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = createViews(graph);
  const backing = view('uint32', 128);
  expect(
    () =>
      new GPUNearestFeatureJoin({
        queries: {kind: 'points', positions: view('float32x2', 1)},
        features: {kind: 'points', positions: view('float32x2', 1)},
        k: 65,
        output: {
          neighborIds: graph.createDataView(backing.buffer, {format: 'uint32', length: 65}),
          neighborCounts: graph.createDataView(backing.buffer, {
            format: 'uint32',
            byteOffset: 260,
            length: 1
          }),
          status: {overflow: view('uint32', 1)}
        }
      })
  ).toThrow(/byte ranges must not overlap/);
  const ids = [0, 0].map(byteOffset =>
    graph.createDataView(backing.buffer, {format: 'uint32', byteOffset, length: 1})
  );
  expect(
    () =>
      new GPUNearestFeatureJoin({
        queries: {
          kind: 'points',
          positions: createVectorView('alias-queries', 'float32x2', [
            view('float32x2', 1),
            view('float32x2', 1)
          ])
        },
        features: {kind: 'points', positions: view('float32x2', 1)},
        radius: view('float32', 1),
        candidateCapacity: 8,
        output: {
          nearestFeatureIds: createVectorView('aliased-ids', 'uint32', ids),
          status: {overflow: view('uint32', 1)}
        }
      })
  ).toThrow(/byte ranges must not overlap/);
  device.destroy();
});
