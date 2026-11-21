// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUGeometryValidity} from '../../../src/gpu-spatial-analysis/geometry-validity/gpu-geometry-validity';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createWaveHarness} from './w5-test-harness';
it.each([
  4, 8
])('GPU validity aggregates six-vertex candidate loss with %s leaves and replays', async leafCapacity => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createWaveHarness(device);
  const allocate = harness.allocate;
  const overflow = allocate('overflow', 'uint32', 1);
  const validity = allocate('validity', 'uint32', 1);
  const nodes = new GPUGeometryValidity({
    polygons: {
      kind: 'polygons',
      positions: allocate('hexagon', 'float32x2', 6, [0, 0, 2, 0, 3, 1, 2, 2, 0, 2, -1, 1]),
      featureOffsets: allocate('features', 'uint32', 2, [0, 1]),
      polygonOffsets: allocate('polygons', 'uint32', 2, [0, 1]),
      ringOffsets: allocate('rings', 'uint32', 2, [0, 6])
    },
    orientation: 'ignore',
    leafCapacity,
    intersectionCapacity: 32,
    output: {issueCode: allocate('issues', 'uint32', 1), validity, status: {overflow}}
  }).getCommandNodes(harness.graph);
  try {
    for (let replay = 0; replay < 2; replay++) {
      harness.write(overflow, Uint32Array.of(99));
      harness.write(validity, Uint32Array.of(99));
      await harness.run(nodes);
      expect(await harness.read(overflow)).toEqual([leafCapacity === 4 ? 1 : 0]);
      expect(await harness.read(validity)).toEqual([leafCapacity === 4 ? 0 : 1]);
    }
  } finally {
    harness.destroy();
  }
});

// Review B7: line and point validity build no intersection list and clear each status word in
// its own dispatch, so adjacent overflow/intersectionCount words are valid and read [0,0].
it.each([
  'lines',
  'points'
] as const)('GPU %s validity clears adjacent status words', async kind => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createWaveHarness(device);
  const allocate = harness.allocate;
  const packed = allocate('packed', 'uint32', 2);
  const overflow = harness.graph.createDataView(packed.buffer, {format: 'uint32', length: 1});
  const intersectionCount = harness.graph.createDataView(packed.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 4
  });
  const positions = allocate('positions', 'float32x2', 2, [0, 0, 1, 1]);
  const geometry =
    kind === 'lines'
      ? {lines: {kind, positions, lineOffsets: allocate('parts', 'uint32', 2, [0, 2])}}
      : {points: {kind, positions}};
  const nodes = new GPUGeometryValidity({
    ...geometry,
    intersectionCapacity: 16,
    output: {
      issueCode: allocate('issues', 'uint32', kind === 'points' ? 2 : 1),
      status: {overflow, intersectionCount}
    }
  }).getCommandNodes(harness.graph);
  try {
    await harness.run(nodes);
    expect(await harness.read(packed)).toEqual([0, 0]);
  } finally {
    harness.destroy();
  }
});
