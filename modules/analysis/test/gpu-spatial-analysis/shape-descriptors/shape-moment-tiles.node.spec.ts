// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {getShapeMomentTileNodes} from '../../../src/gpu-spatial-analysis/shape-descriptors/shape-moment-tiles';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {createCPUGraphMemory} from '../segment-intersection/wgsl-graph-executor';
import {executeGeometryNodesOnCPU} from '../polygon-triangulation/geometry-cpu-executor';

useCPUExecutorTestHooks();

it('executes fixed-order moment tiles against closed-form rectangle integrals', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const loops = 256;
  const positions = createTransientView(graph, 'positions', 'float32x2', 4 * loops);
  const ringOffsets = createTransientView(graph, 'rings', 'uint32', 2);
  const polygonOffsets = createTransientView(graph, 'polygons', 'uint32', 2);
  const featureOffsets = createTransientView(graph, 'features', 'uint32', 2);
  const centroids = createTransientView(graph, 'centroids', 'float32x2', 1);
  const moments = createTransientView(graph, 'moments', 'float32x4', 1);
  const coordinates = new Float32Array(8 * loops);
  for (let loop = 0; loop < loops; loop++) coordinates.set([0, 0, 4, 0, 4, 2, 0, 2], 8 * loop);
  const memory = createCPUGraphMemory();
  memory.write(positions, coordinates);
  memory.write(ringOffsets, new Uint32Array([0, 4 * loops]));
  memory.write(polygonOffsets, new Uint32Array([0, 1]));
  memory.write(featureOffsets, new Uint32Array([0, 1]));
  memory.write(centroids, new Float32Array([2, 1]));
  const createView = vi.spyOn(graph, 'createDataView');
  const nodes = getShapeMomentTileNodes(graph, {
    id: 'moments',
    operation: 'GPUShapeDescriptors',
    positions,
    ringOffsets,
    polygonOffsets,
    featureOffsets,
    centroids,
    moments,
    featureCount: 1
  });
  // The interpreter cannot synchronize workgroup reductions. Execute the real bounded tile
  // shader and inspect its outputs; the headless summation spec checks the full reduction chain.
  const tileNode = nodes.findIndex(node => node.id.endsWith('-tiles'));
  executeGeometryNodesOnCPU(nodes.slice(0, tileNode + 1), memory);
  for (const [name, expected] of [
    ['tile-counts', [4, 0]],
    ['tile-offsets', [0, 4]]
  ] as const) {
    const view = createView.mock.results.find(
      result => result.type === 'return' && result.value.buffer.id === `moments-${name}`
    )!.value;
    expect(Array.from(memory.readUint32(view))).toEqual(expected);
  }
  const centroidView = createView.mock.results.find(
    result => result.type === 'return' && result.value.buffer.id === 'moments-ring-centroids'
  )!.value;
  expect(Array.from(memory.readFloat32(centroidView))).toEqual([2, 1]);
  for (const [name, expected] of [
    ['area', 64 * 16],
    ['xx', (64 * 8 * 16) / 12],
    ['yy', (64 * 8 * 4) / 12],
    ['xy', 0]
  ] as const) {
    const view = createView.mock.results.find(
      result => result.type === 'return' && result.value.buffer.id === `moments-tile-${name}`
    )!.value;
    const actual = memory.readFloat32(view);
    expect(actual).toHaveLength(5);
    for (const value of actual.slice(0, 4)) {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeCloseTo(expected, 3);
    }
    expect(actual[4]).toBe(0);
  }
  vi.restoreAllMocks();
  device.destroy();
});
