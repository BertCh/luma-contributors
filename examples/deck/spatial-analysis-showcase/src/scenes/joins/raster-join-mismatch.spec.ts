// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../../../../modules/analysis/src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../../../../../modules/analysis/src/testing/gpu-contributor-test-utils';
import {requireTestDevice} from '../../../../../../modules/analysis/src/testing/require-test-device';
import {addRasterJoinMismatchNode} from './raster-join-mismatch';

it.each([
  [0, 0, 0, 8],
  [1, 0, 0, 8],
  [0, 1, 0, 8],
  [0, 0, 1, 8],
  [0, 0, 0, 16],
  [1, 0, 0, 16],
  [0, 1, 0, 16],
  [0, 0, 1, 16]
])('raster join mismatch reads paired position components with loss flags %s/%s/%s and byte stride %s', async (producerOverflow, exactOverflow, exactUncertainCount, byteStride) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: 'raster-join-mismatch-test'});
  // Ignore the sentinel prefix and padding pairs. The four coordinates in cell units are
  // (0.5, 0.5), (1.5, 0.5), (0.5, 2.5), (0.5, 1.5).
  // Only rows 0 and 3 are both inside the 2x2 extent and have unequal zones.
  const positionValues = new Float32Array([
    -99,
    -99,
    -99,
    -99,
    ...[
      [1, 1],
      [3, 1],
      [1, 5],
      [1, 3]
    ].flatMap(point => (byteStride === 16 ? [...point, -10, -10] : point))
  ]);
  const positions = createInputBuffer(device, positionValues);
  const placement = createInputBuffer(device, new Float32Array([0, 0, 2, 2]));
  const rasterZones = createInputBuffer(device, new Uint32Array([2, 1, 2, 2]));
  const exactZones = createInputBuffer(device, new Uint32Array([1, 1, 1, 1]));
  const producerLoss = createInputBuffer(device, new Uint32Array([producerOverflow]));
  const exactLoss = createInputBuffer(device, new Uint32Array([exactOverflow]));
  const exactUncertainty = createInputBuffer(device, new Uint32Array([exactUncertainCount]));
  const output = createOutputBuffer(device, 4);
  const extentMembership = createOutputBuffer(device, 4);
  output.write(new Uint32Array(4).fill(1));
  const allPositions = importGraphBuffer(
    graph,
    'positions',
    positions,
    'float32x2',
    positionValues.length / 2
  );
  addRasterJoinMismatchNode(graph, {
    positions: graph.createDataView(allPositions.buffer, {
      format: 'float32x2',
      byteOffset: 16,
      byteStride,
      length: 4
    }),
    parameters: importGraphBuffer(graph, 'placement', placement, 'float32', 4),
    rasterZones: importGraphBuffer(graph, 'raster-zones', rasterZones, 'uint32', 4),
    exactZones: importGraphBuffer(graph, 'exact-zones', exactZones, 'uint32', 4),
    producerOverflow: importGraphBuffer(graph, 'producer-loss', producerLoss, 'uint32', 1),
    exactOverflow: importGraphBuffer(graph, 'exact-loss', exactLoss, 'uint32', 1),
    exactUncertainCount: importGraphBuffer(
      graph,
      'exact-uncertainty',
      exactUncertainty,
      'uint32',
      1
    ),
    width: 2,
    height: 2,
    output: importGraphBuffer(graph, 'output', output, 'uint32', 4),
    extentMembership: importGraphBuffer(graph, 'extent-membership', extentMembership, 'uint32', 4)
  });
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(extentMembership, 4)).toEqual([1, 1, 0, 1]);
    expect(await readUint32(output, 4)).toEqual(
      producerOverflow || exactOverflow || exactUncertainCount ? [0, 0, 0, 0] : [1, 0, 0, 1]
    );
  } finally {
    compiled.destroy();
    for (const buffer of [
      positions,
      placement,
      rasterZones,
      exactZones,
      producerLoss,
      exactLoss,
      exactUncertainty,
      extentMembership,
      output
    ]) {
      buffer.destroy();
    }
  }
});

it('GPU extent mask and mismatch agree at the rounded upper edge', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: 'raster-rounded-upper-edge'});
  const positions = createInputBuffer(device, Float32Array.of(-0.000001, 0.5));
  const placement = createInputBuffer(device, Float32Array.of(-128, 0, 1, 1));
  const rasterZones = createInputBuffer(device, Uint32Array.of(2));
  const exactZones = createInputBuffer(device, Uint32Array.of(1));
  const clean = createInputBuffer(device, Uint32Array.of(0));
  const output = createOutputBuffer(device, 1);
  const extentMembership = createOutputBuffer(device, 1);
  const cleanView = importGraphBuffer(graph, 'clean', clean, 'uint32', 1);
  addRasterJoinMismatchNode(graph, {
    positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', 1),
    parameters: importGraphBuffer(graph, 'placement', placement, 'float32', 4),
    rasterZones: importGraphBuffer(graph, 'raster-zones', rasterZones, 'uint32', 1),
    exactZones: importGraphBuffer(graph, 'exact-zones', exactZones, 'uint32', 1),
    producerOverflow: cleanView,
    exactOverflow: cleanView,
    exactUncertainCount: cleanView,
    width: 128,
    height: 1,
    output: importGraphBuffer(graph, 'output', output, 'uint32', 1),
    extentMembership: importGraphBuffer(graph, 'membership', extentMembership, 'uint32', 1)
  });
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    // float32(-0.000001 - -128) = 128, outside the half-open raster extent.
    expect(await readUint32(extentMembership, 1)).toEqual([0]);
    expect(await readUint32(output, 1)).toEqual([0]);
  } finally {
    compiled.destroy();
    for (const buffer of [
      positions,
      placement,
      rasterZones,
      exactZones,
      clean,
      output,
      extentMembership
    ])
      buffer.destroy();
  }
});
