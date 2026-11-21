// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {createTransientView, GPUCommandGraph, GraphVectorView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUBufferSelection} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-buffer-selection';
import {GPUPointInPolygonJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join';
import {computePointInPolygonCountsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-point-in-polygon-join-operation';
import {
  computeBufferSelectionOnCPU,
  createCPUBufferSelectionOperation
} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-buffer-selection-operation';
import type {BufferSelectionOperationProps} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-buffer-selection-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const common = createRequire(import.meta.url)(
  '@luma.gl/graph/gpu-core'
) as typeof import('@luma.gl/graph/gpu-core');
function fixture() {
  const device = createNullWebGPUDevice(),
    graph = new GPUCommandGraph(device);
  let serial = 0;
  const integers = (length: number) =>
    createTransientView(graph, `integers-${serial++}`, 'uint32', length);
  const positions = (length: number) =>
    createTransientView(graph, `positions-${serial++}`, 'float32x2', length);
  function vector<Format extends 'uint32' | 'float32x2'>(format: Format, foreign = false) {
    const data = [2, 1].map(length =>
      createTransientView(graph, `chunk-${serial++}`, format, length)
    );
    const stride = format === 'float32x2' ? 2 : 1;
    const Constructor = foreign ? common.GraphVectorView : GraphVectorView;
    return new Constructor({
      id: `vector-${serial++}`,
      name: 'chunks',
      format,
      length: 3,
      valueLength: 3 * stride,
      stride,
      byteStride: stride * 4,
      rowByteLength: stride * 4,
      data
    });
  }
  return {device, graph, integers, positions, vector};
}
for (const callerMask of [false, true])
  it(`B1 foreign visibility vectors preserve [2,1] chunks, callerMask=${callerMask}`, () => {
    const {device, graph, integers, positions, vector} = fixture();
    try {
      const selection = new GPUBufferSelection({
        points: {kind: 'points', positions: vector('float32x2'), sourceIds: vector('uint32', true)},
        features: {kind: 'points', positions: positions(1)},
        radius: createTransientView(graph, 'radius', 'float32', 1),
        candidateCapacity: 4,
        spatialSort: false,
        querySort: false,
        output: callerMask ? {outputMask: vector('uint32', true)} : {}
      });
      expect(selection.getMatches(graph).ids.length).toBe(3);
      expect(() => selection.getCommandNodes(graph)).not.toThrow();
      expect(integers(1).length).toBe(1);
    } finally {
      device.destroy();
    }
  });
for (const route of ['sort', 'great-circle'] as const)
  for (const direct of [false, true]) {
    it(`B2 ${route} complete preflight precedes allocations direct=${direct}`, () => {
      const {device, graph, positions} = fixture();
      const selection = new GPUBufferSelection({
        points: {kind: 'points', positions: positions(1)},
        features: {kind: 'points', positions: positions(2)},
        radius: createTransientView(graph, 'radius', 'float32', 1),
        candidateCapacity: 1,
        spatialSort: route === 'sort',
        querySort: false,
        ...(route === 'great-circle'
          ? {
              spatialContext: {
                coordinateSpace: 'longitude-latitude' as const,
                metric: 'great-circle' as const
              }
            }
          : {}),
        output: {}
      });
      const limits = device.limits;
      Object.defineProperty(device, 'limits', {
        configurable: true,
        value: {...limits, maxStorageBufferBindingSize: route === 'sort' ? 32 : 64}
      });
      const declarations = vi.spyOn(graph, 'createTransientBuffer');
      try {
        expect(() =>
          direct ? selection.getMatches(graph) : selection.getCommandNodes(graph)
        ).toThrow(route === 'sort' ? /histogram/ : /nodeData|node table/);
        expect(declarations).not.toHaveBeenCalled();
        Object.defineProperty(device, 'limits', {value: limits});
        expect(() => selection.getCommandNodes(graph)).not.toThrow();
      } finally {
        declarations.mockRestore();
        device.destroy();
      }
    });
  }
it('S1 spatial bounds index prunes alternating distant polygons but preserves candidate row cutoff', () => {
  const positions: number[] = [],
    offsets = [0];
  for (let row = 0; row < 128; row++) {
    const x = row % 2 ? 10 : -10;
    positions.push(x, -1, x + 1, -1, x, 1);
    offsets.push((row + 1) * 3);
  }
  const polygons = {
    kind: 'polygons' as const,
    positions: Float32Array.from(positions),
    featureOffsets: Uint32Array.from({length: 129}, (_, row) => row),
    polygonOffsets: Uint32Array.from({length: 129}, (_, row) => row),
    ringOffsets: Uint32Array.from(offsets)
  };
  const result = computePointInPolygonCountsOnCPU(
    {points: {kind: 'points', positions: Float32Array.of(0, 0)}, polygons},
    {candidateCapacity: 1, matchCapacity: 0}
  );
  expect(Array.from(result.candidateCount)).toEqual([0]);
  expect(result.candidateBoundsTestCount[0]).toBeLessThan(32);
  const reordered = computePointInPolygonCountsOnCPU(
    {
      points: {kind: 'points', positions: Float32Array.of(0, 0)},
      polygons: {
        kind: 'polygons',
        positions: Float32Array.of(-1, -1, 1, -1, 1, 1, -1, 1, -2, -2, 2, -2, 2, 2, -2, 2),
        featureOffsets: Uint32Array.of(0, 1, 2),
        polygonOffsets: Uint32Array.of(0, 1, 2),
        ringOffsets: Uint32Array.of(0, 4, 8)
      }
    },
    {candidateCapacity: 1, matchCapacity: 1}
  );
  expect(Array.from(reordered.pointFeatureRows)).toEqual([0]);
  expect(Array.from(reordered.polygonMembershipCounts)).toEqual([1, 0]);
  expect(Array.from(reordered.candidateOverflow)).toEqual([1]);
});
for (const multipart of [false, true])
  it(`N2 buffer winner uses the nearest distance rank multipart=${multipart}`, () => {
    const result = computeBufferSelectionOnCPU(
      {
        points: {
          kind: 'points',
          positions: Float32Array.of(0, 0),
          ...(multipart ? {featureOffsets: Uint32Array.of(0, 1)} : {})
        },
        features: {kind: 'points', positions: Float32Array.of(1, 2 ** -14, 1, 0)},
        radius: 2
      },
      {candidateCapacity: 4, matchCapacity: 1}
    );
    // Squared ranks round 1+2^-28 and 1 to the same float32 value; lowest row wins.
    expect(Array.from(result.nearestFeatureRows)).toEqual([0]);
  });
it('S2 row publication rejects read/write binding aliases even for disjoint byte ranges', () => {
  const {device, graph, integers, positions} = fixture();
  const shared = integers(4);
  const sourceIds = graph.createDataView(shared.buffer, {format: 'uint32', length: 2});
  const rows = graph.createDataView(shared.buffer, {format: 'uint32', byteOffset: 8, length: 1});
  try {
    expect(
      () =>
        new GPUPointInPolygonJoin({
          points: {kind: 'points', positions: positions(1)},
          polygons: {
            kind: 'polygons',
            positions: positions(8),
            featureOffsets: integers(3),
            polygonOffsets: integers(3),
            ringOffsets: integers(3),
            sourceIds
          },
          candidateCapacity: 4,
          output: {
            pointFeatureRows: rows,
            pointFeatureIds: integers(1),
            status: {overflow: integers(1)}
          }
        })
    ).toThrow(/buffer|binding/);
  } finally {
    device.destroy();
  }
});
it('S4 assignment count publication needs no second assignment row buffer', () => {
  const {device, graph, integers, positions} = fixture();
  const join = new GPUPointInPolygonJoin({
    points: {kind: 'points', positions: positions(1)},
    polygons: {
      kind: 'polygons',
      positions: positions(4),
      featureOffsets: integers(2),
      polygonOffsets: integers(2),
      ringOffsets: integers(2)
    },
    candidateCapacity: 1,
    spatialSort: false,
    querySort: false,
    edgeSlabs: false,
    output: {
      pointFeatureRows: integers(1),
      polygonAssignmentCounts: integers(1),
      status: {overflow: integers(1)}
    }
  });
  const declaration = vi.spyOn(graph, 'createTransientBuffer');
  try {
    join.getCommandNodes(graph);
    expect(
      declaration.mock.calls.some(([descriptor]) => descriptor.id.includes('assignment-count-rows'))
    ).toBe(false);
  } finally {
    declaration.mockRestore();
    device.destroy();
  }
});
it('S3 point hierarchy is checked once before traversal', () => {
  let reads = 0;
  const offsets = new Proxy(Uint32Array.of(0, 1), {
    get(target, key) {
      if (typeof key === 'string' && /^\d+$/.test(key)) reads++;
      return Reflect.get(target, key, target);
    }
  });
  const result = computePointInPolygonCountsOnCPU(
    {
      points: {kind: 'points', positions: Float32Array.of(0.25, 0.25), featureOffsets: offsets},
      polygons: {
        kind: 'polygons',
        positions: Float32Array.of(0, 0, 1, 0, 0, 1),
        featureOffsets: Uint32Array.of(0, 1),
        polygonOffsets: Uint32Array.of(0, 1),
        ringOffsets: Uint32Array.of(0, 3)
      }
    },
    {candidateCapacity: 1, matchCapacity: 1}
  );
  expect(Array.from(result.pointFeatureRows)).toEqual([0]);
  // One complete monotonic/endpoints validation (4 reads), then three traversal reads.
  expect(reads).toBeLessThanOrEqual(7);
});
it('U1 CPU buffer operation accepts radius-only descriptors and rejects retired distance', () => {
  const output = Object.fromEntries(
    [
      'outputMask',
      'distances',
      'nearestFeatureRows',
      'candidateTruncated',
      'ids',
      'count',
      'requiredCount',
      'matchOverflow',
      'matchCandidateOverflow',
      'overflow',
      'candidateOverflow',
      'drawInstanceCount'
    ].map(name => [name, name])
  ) as BufferSelectionOperationProps['output'];
  const input = {
    points: {kind: 'points' as const, positions: {value: 'points', length: 1}},
    features: {kind: 'points' as const, positions: {value: 'features', length: 1}},
    radius: {value: 'radius', length: 1},
    candidateCapacity: 1,
    matchCapacity: 1,
    output
  };
  const operation = createCPUBufferSelectionOperation(input);
  expect(operation.inputs['distance']).toBe('radius');
  const legacyInput = {...input, distance: {value: 'different', length: 1}};
  expect(() => createCPUBufferSelectionOperation(legacyInput)).toThrow(/distance.*radius/);
});
