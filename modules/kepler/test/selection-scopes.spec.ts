// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {GPUKeplerSelectionScopes} from '../src/selection/gpu-selection-scopes';
import type {
  KeplerPolygonMaskFilter,
  KeplerSelectionLayer
} from '../src/selection/selection-scope-plan';

test('source-only scopes preserve explicit empty chunks without manufacturing scalar columns', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('WebGPU is required for selection scope verification');
  const positions = pointVector(device, [
    new Float32Array(0),
    new Float32Array([0, 0, 1, 1]),
    new Float32Array(0),
    new Float32Array([2, 2])
  ]);
  const scopes = new GPUKeplerSelectionScopes({
    device,
    sourceChunkLengths: [0, 2, 0, 1],
    layers: [{id: 'geometry-only', geometryRevision: 1, geometry: {kind: 'point', positions}}],
    polygonFilters: [
      {
        id: 'shape',
        layerIds: ['geometry-only'],
        selection: {kind: 'rectangle', bounds: [-0.5, -0.5, 1.5, 1.5]}
      }
    ]
  });
  try {
    const encoder = device.createCommandEncoder();
    scopes.encode(encoder);
    device.submit(encoder.finish());
    expect(await scopes.readStatus()).toMatchObject({complete: true});
    expect(scopes.displayMask.data.map(chunk => chunk.length)).toEqual([0, 2, 0, 1]);
    expect(await selected(scopes.nativeTopologyMask)).toEqual([0, 1, 2]);
    expect(await selected(scopes.getLayerTopologyMask('geometry-only'))).toEqual([0, 1]);
    expect(await selected(scopes.nativeFilteredMask)).toEqual([0, 1]);
    expect(
      () => new GPUKeplerSelectionScopes({device, rows: positions, sourceChunkLengths: [3]})
    ).toThrow('source topology');
  } finally {
    const fence = device.createFence();
    try {
      await scopes.destroy(fence.signaled);
    } finally {
      fence.destroy();
      positions.destroy();
    }
  }
});

test('native multi-layer scopes preserve source identity, exact own-filter exclusion, endpoint and MultiPoint predicates', async () => {
  const availableDevice = await getWebGPUTestDevice();
  if (!availableDevice) throw new Error('WebGPU is required for selection scope verification');
  const device = availableDevice;
  const points = pointVector(device, [
    new Float32Array([0, 0, 1, 1, 2, 2]),
    new Float32Array([3, 3, 4, 4, 5, 5])
  ]);
  const reverse = pointVector(device, [
    new Float32Array([5, 5, 4, 4, 3, 3]),
    new Float32Array([2, 2, 1, 1, 0, 0])
  ]);
  const first = maskVector(device, [new Uint32Array([1, 0, 1]), new Uint32Array([1, 1, 1])]);
  const second = maskVector(device, [new Uint32Array([0, 1, 1]), new Uint32Array([1, 1, 1])]);
  const multiPoints = pointVector(device, [
    new Float32Array([10, 10, 1, 1, 2, 2, 4, 4]),
    new Float32Array([3, 3, 1, 1, 9, 9])
  ]);
  const offsets = maskVector(device, [
    new Uint32Array([0, 2, 2, 4]),
    new Uint32Array([0, 1, 2, 3])
  ]);
  const validity = maskVector(device, [new Uint32Array([1, 1, 1]), new Uint32Array([1, 0, 1])]);
  const layers: KeplerSelectionLayer[] = [
    {id: 'points', geometryRevision: 1, geometry: {kind: 'point', positions: points}},
    {
      id: 'hidden',
      geometryRevision: 1,
      visible: false,
      geometry: {kind: 'point', positions: reverse}
    },
    {id: 'untargeted', geometryRevision: 1, geometry: {kind: 'point', positions: points}},
    {
      id: 'arc',
      geometryRevision: 1,
      geometry: {kind: 'endpoint-pair', sourcePositions: points, targetPositions: reverse}
    },
    {
      id: 'multi',
      geometryRevision: 1,
      geometry: {kind: 'multipoint', positions: multiPoints, rowOffsets: offsets, validity}
    },
    {
      id: 'representative',
      geometryRevision: 1,
      geometry: {
        kind: 'representative-point',
        positions: points,
        recipe: 'json-bounds-center',
        nativeRectangleBounds: true
      }
    }
  ];
  const scalar = [
    {
      filterId: 'first-value',
      mask: first,
      dynamicDomain: true,
      nativeTopology: false,
      nativeFieldName: 'value'
    },
    {filterId: 'second-value', mask: second, dynamicDomain: false, nativeFieldName: 'value'}
  ];
  const polygon: KeplerPolygonMaskFilter = {
    id: 'drawn',
    layerIds: ['points'],
    selection: {kind: 'rectangle', bounds: [-0.5, -0.5, 2.5, 2.5]}
  };
  const scopes = new GPUKeplerSelectionScopes({
    device,
    rows: points,
    layers,
    scalarFilters: scalar,
    polygonFilters: [polygon],
    datasetRevision: 7,
    maximumConsumerScopes: 2
  });
  const firstPlot = scopes.getPlotMask('first-value');
  const secondPlot = scopes.getPlotMask('second-value');
  const stableLayer = scopes.getLayerMask('points');
  const stableTopology = scopes.getLayerTopologyMask('points');
  const chart = scopes.getConsumerMask({id: 'chart', excludedGpuFieldNames: ['value']});
  const count = scopes.getConsumerMask({id: 'count', excludedCpuFilterIds: ['second-value']});
  expect(() => scopes.getConsumerMask({id: 'over-capacity'})).toThrow('capacity');
  async function execute(): Promise<void> {
    const encoder = device.createCommandEncoder();
    scopes.encode(encoder);
    device.submit(encoder.finish());
    expect(await scopes.readStatus()).toMatchObject({
      complete: true,
      stale: false,
      datasetRevision: 7
    });
    const fence = device.createFence();
    try {
      await scopes.collectRetired(fence.signaled);
    } finally {
      fence.destroy();
    }
  }
  try {
    await execute();
    expect(await selected(scopes.displayMask)).toEqual([2, 3, 4, 5]);
    expect(await selected(scopes.domainMask)).toEqual([0, 2, 3, 4, 5]);
    expect(await selected(stableLayer)).toEqual([2]);
    expect(await selected(scopes.getLayerMask('untargeted'))).toEqual([2, 3, 4, 5]);
    expect(await selected(scopes.nativeFilteredMask)).toEqual([2]);
    expect(await selected(scopes.nativeTopologyMask)).toEqual([1, 2, 3, 4, 5]);
    expect(await selected(scopes.getLayerTopologyMask('points'))).toEqual([1, 2]);
    expect(await selected(scopes.getLayerTopologyMask('untargeted'))).toEqual([1, 2, 3, 4, 5]);
    expect(await selected(firstPlot)).toEqual([1, 2]);
    expect(await selected(secondPlot)).toEqual([0, 2]);
    expect(await selected(chart)).toEqual([1, 2]);
    expect(await selected(count)).toEqual([0, 2]);
    const previousRevision = scopes.currentRevision;
    expect(scopes.getConsumerMask({id: 'chart', excludedGpuFieldNames: ['value', 'value']})).toBe(
      chart
    );
    expect(scopes.currentRevision).toBe(previousRevision);
    expect(stableLayer.data.map(chunk => chunk.length)).toEqual([3, 3]);
    expect(stableTopology.data.map(chunk => chunk.length)).toEqual([3, 3]);

    scopes.setFilters(scalar, [{...polygon, layerIds: ['points', 'hidden']}]);
    await execute();
    expect(await selected(scopes.getLayerMask('hidden'))).toEqual([3, 4, 5]);
    expect(await selected(scopes.nativeFilteredMask)).toEqual([2, 3, 4, 5]);
    expect(await selected(scopes.getLayerTopologyMask('points'))).toEqual([1, 2]);
    expect(await selected(scopes.getLayerTopologyMask('hidden'))).toEqual([3, 4, 5]);
    expect(await selected(scopes.getLayerTopologyMask('untargeted'))).toEqual([1, 2, 3, 4, 5]);
    expect(await selected(chart)).toEqual([1, 2, 3, 4, 5]);
    expect(
      scopes.getConsumerMask({
        id: 'chart',
        excludedCpuFilterIds: ['second-value'],
        excludedGpuFieldNames: ['value']
      })
    ).toBe(chart);
    await execute();
    expect(await selected(chart)).toEqual([0, 1, 2, 3, 4, 5]);
    scopes.removeConsumerMask('count');
    expect(scopes.getConsumerMask({id: 'replacement'})).not.toBe(count);

    scopes.setFilters(scalar, [
      {
        ...polygon,
        layerIds: ['arc'],
        selection: {kind: 'rectangle', bounds: [-0.5, -0.5, 3.5, 3.5]}
      }
    ]);
    await execute();
    expect(await selected(scopes.getLayerMask('arc'))).toEqual([2, 3]);
    scopes.setFilters(scalar, [
      {
        ...polygon,
        layerIds: ['arc'],
        inverted: true,
        selection: {kind: 'rectangle', bounds: [-0.5, -0.5, 3.5, 3.5]}
      }
    ]);
    await execute();
    expect(await selected(scopes.getLayerMask('arc'))).toEqual([4, 5]);

    scopes.setFilters(scalar, [{...polygon, layerIds: ['multi']}]);
    await execute();
    expect(await selected(scopes.getLayerMask('multi'))).toEqual([2]);
    scopes.setFilters(scalar, [{...polygon, layerIds: ['multi'], inverted: true}]);
    await execute();
    expect(await selected(scopes.getLayerMask('multi'))).toEqual([3, 4, 5]);

    const rectangle: KeplerPolygonMaskFilter = {
      ...polygon,
      layerIds: ['representative'],
      selection: {kind: 'rectangle', bounds: [0, 0, 2, 2]},
      rectangleBounds: [0, 0, 2, 2]
    };
    scopes.setFilters([], [rectangle]);
    await execute();
    expect(await selected(scopes.getLayerMask('representative'))).toEqual([0, 1, 2]);
    scopes.setFilters([], [{...rectangle, rectangleBounds: undefined}]);
    await execute();
    expect(await selected(scopes.getLayerMask('representative'))).toEqual([1]);

    scopes.setFilters(scalar, [{...polygon, enabled: false}]);
    await execute();
    expect(await selected(scopes.nativeFilteredMask)).toEqual([2, 3, 4, 5]);
    scopes.setFilters(scalar, []);
    await execute();
    expect(scopes.getLayerMask('points')).toBe(stableLayer);
    expect(scopes.getLayerTopologyMask('points')).toBe(stableTopology);
    expect(await selected(scopes.nativeFilteredMask)).toEqual([2, 3, 4, 5]);

    const encoder = device.createCommandEncoder();
    scopes.encode(encoder);
    device.submit(encoder.finish());
    const pending = scopes.readStatus();
    scopes.invalidate();
    expect((await pending).stale).toBe(true);
  } finally {
    const fence = device.createFence();
    try {
      await scopes.destroy(fence.signaled);
    } finally {
      fence.destroy();
    }
    for (const vector of [points, reverse, first, second, multiPoints, offsets, validity])
      vector.destroy();
  }
  expect(scopes.workingMemoryByteLength).toBe(0);
});

function pointVector(device: Device, values: Float32Array[]): GPUVector<'float32x2'> {
  return new GPUVector({
    name: 'points',
    type: 'data',
    format: 'float32x2',
    ownsData: true,
    data: values.map(
      value =>
        new GPUData({
          buffer: device.createBuffer({
            data: value.length ? value : undefined,
            byteLength: Math.max(8, value.byteLength),
            usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
          }),
          format: 'float32x2',
          length: value.length / 2,
          ownsBuffer: true
        })
    )
  });
}
function maskVector(device: Device, values: Uint32Array[]): GPUVector<'uint32'> {
  return new GPUVector({
    name: 'mask',
    type: 'data',
    format: 'uint32',
    ownsData: true,
    data: values.map(
      value =>
        new GPUData({
          buffer: device.createBuffer({
            data: value,
            usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
          }),
          format: 'uint32',
          length: value.length,
          ownsBuffer: true
        })
    )
  });
}
async function selected(mask: GPUVector<'uint32'>): Promise<number[]> {
  const result: number[] = [];
  let offset = 0;
  for (const chunk of mask.data) {
    if (!chunk.length) continue;
    const bytes = await chunk.buffer.readAsync(0, chunk.length * 4);
    const values = new Uint32Array(bytes.buffer, bytes.byteOffset, chunk.length);
    for (const [index, value] of values.entries()) if (value) result.push(offset + index);
    offset += chunk.length;
  }
  return result;
}

test('unlabelled-union rows match native no-centroid semantics and antimeridian edges stay linear', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('WebGPU is required for selection scope verification');
  // Native booleanWithin is linear in raw degrees: 190 is a distinct planar coordinate from -170,
  // and the [-170, 170] edge spans the full interior rather than wrapping across the antimeridian.
  const points = pointVector(device, [new Float32Array([0, 0, 190, 0, -170, 0, 175, 0])]);
  const scopes = new GPUKeplerSelectionScopes({
    device,
    rows: points,
    layers: [
      {id: 'points', geometryRevision: 1, geometry: {kind: 'point', positions: points}},
      {id: 'union', geometryRevision: 1, geometry: {kind: 'no-representative'}}
    ],
    polygonFilters: [
      {
        id: 'drawn',
        layerIds: ['points', 'union'],
        selection: {
          kind: 'polygon',
          polygons: [
            [
              [
                [170, -1],
                [200, -1],
                [200, 1],
                [170, 1]
              ]
            ]
          ]
        }
      }
    ]
  });
  const execute = async () => {
    const encoder = device.createCommandEncoder();
    scopes.encode(encoder);
    device.submit(encoder.finish());
    await scopes.readStatus();
  };
  try {
    await execute();
    expect(await selected(scopes.getLayerMask('points'))).toEqual([1, 3]);
    expect(await selected(scopes.getLayerMask('union'))).toEqual([]);
    scopes.setFilters(
      [],
      [
        {
          id: 'drawn',
          layerIds: ['union'],
          inverted: true,
          selection: {kind: 'rectangle', bounds: [0, 0, 1, 1]}
        }
      ]
    );
    await execute();
    expect(await selected(scopes.getLayerMask('union'))).toEqual([0, 1, 2, 3]);
  } finally {
    const fence = device.createFence();
    try {
      await scopes.destroy(fence.signaled);
    } finally {
      fence.destroy();
      points.destroy();
    }
  }
});
