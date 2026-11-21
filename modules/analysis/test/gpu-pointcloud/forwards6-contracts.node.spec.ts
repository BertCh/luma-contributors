// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  decodeGPUPointCloudFeaturesStatus,
  createGroundSeedCoverageNode,
  decodeGPUPointCloudNeighborhoodsStatus,
  GPUPointCloudNormalScale,
  GPUPointCloudStatisticalFilter,
  GPUPointCloudSurfaces,
  GPUPointCloudVoxelMerge,
  getRasterMetadataFromPointCloudGrid,
  type PointCloudRasterGrid
} from '@luma-contributors/analysis/pointcloud-analysis';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../utils/gpu-contributor-test-utils';

it('UNI-01: public ground seed conversion needs only one output and preserves the ground bit', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const coverage = createTransientView(graph, 'seed-coverage', 'uint32', 7);
    const output = createTransientView(graph, 'seeds', 'uint32', 7);
    const node = createGroundSeedCoverageNode(graph, coverage, output);
    const sources = getGPUCommandNodeKernelSources([node]);
    expect(sources).toHaveLength(1);
    expect(sources[0].source).toContain('& 2u) != 0u');
    expect(sources[0].source).toContain('select(0u, 1u,');
    expect(() =>
      createGroundSeedCoverageNode(
        graph,
        coverage,
        createTransientView(graph, 'short', 'uint32', 6)
      )
    ).toThrow(/output.*7/);
  } finally {
    device.destroy();
  }
});

for (const decode of [decodeGPUPointCloudFeaturesStatus, decodeGPUPointCloudNeighborhoodsStatus]) {
  it(`${decode.name}: admits only uint32 or complete raw-byte words`, () => {
    for (const input of [
      Uint8ClampedArray.of(0),
      Int32Array.of(0),
      Float32Array.of(0),
      [0],
      new DataView(new ArrayBuffer(4))
    ]) {
      expect(() => decode(input as unknown as Uint32Array)).toThrow(/Uint32Array/);
    }
    for (const input of [
      new Uint32Array(0),
      new Uint8Array(0),
      new Uint8Array(3),
      new ArrayBuffer(5)
    ]) {
      expect(() => decode(input)).toThrow();
    }
    const bytes = new Uint8Array(6);
    new DataView(bytes.buffer).setUint32(1, 2, true);
    const expected = decode(Uint32Array.of(2));
    expect(decode(bytes.subarray(1, 5))).toEqual(expected);
    expect(decode(bytes.buffer.slice(1, 5))).toEqual(expected);
  });
}

it('grid metadata and surface constructors reject missing, sparse, extra and float32-overflow origin coordinates', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let sequence = 0;
  const view = <Format extends 'float32x3' | 'float32' | 'uint32'>(id: string, format: Format) =>
    createTransientView(graph, `${id}-${sequence++}`, format, 1);
  try {
    for (const origin of [[17], new Array(2), [17, ,], [17, 0, 1], [17, 1e40]]) {
      const grid = {origin, cellSize: 2, width: 1, height: 1} as unknown as PointCloudRasterGrid;
      expect(() => getRasterMetadataFromPointCloudGrid(grid)).toThrow(
        /getRasterMetadataFromPointCloudGrid grid.origin.*received/
      );
      expect(
        () =>
          new GPUPointCloudSurfaces({
            id: 'invalid-grid',
            grid,
            positions: view('positions', 'float32x3'),
            classifications: view('classes', 'uint32'),
            output: {
              terrain: view('terrain', 'float32'),
              surface: view('surface', 'float32'),
              canopy: view('canopy', 'float32'),
              coverage: view('coverage', 'uint32')
            }
          })
      ).toThrow(/invalid-grid grid.origin.*received/);
    }
    expect(
      getRasterMetadataFromPointCloudGrid({origin: [17, 0], cellSize: 2, width: 1, height: 1})
        .affine
    ).toEqual([2, 0, 17, 0, -2, 2]);
  } finally {
    device.destroy();
  }
});

it.each([
  'float32x2',
  'float32x4'
] as const)('voxel merge accepts advertised %s attribute means', format => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let sequence = 0;
  const view = <Format extends 'float32x2' | 'float32x4' | 'float32x3' | 'uint32'>(value: Format) =>
    createTransientView(graph, `merge-${sequence++}`, value, 1);
  const part = () => ({
    counts: view('uint32'),
    centroids: view('float32x3'),
    representatives: view('uint32'),
    sourceRowOffset: 0,
    attribute: {means: view(format), counts: view('uint32')}
  });
  try {
    const contributor = new GPUPointCloudVoxelMerge({
      partitionPolicy: 'disjoint-owned-rows',
      parts: [part()],
      output: {...part(), status: view('uint32')}
    });
    expect(contributor.getCommandNodes(graph).length).toBeGreaterThan(0);
  } finally {
    device.destroy();
  }
});

it('normal-scale support refines rounded boundary keys using the actual query coordinates', () => {
  // Exact binary32 coordinate (1, f32(.0001), 0) has squared norm
  // 1.0000000099999995 > 1; its published float32 key is 1. Supports are [2,3].
  const squaredNorm = 1 + Math.fround(0.0001) ** 2;
  expect(squaredNorm).toBeGreaterThan(1);
  expect(Math.fround(squaredNorm)).toBe(1);
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let sequence = 0;
  const view = <Format extends 'float32x3' | 'float32' | 'uint32'>(format: Format, length = 1) =>
    createTransientView(graph, `scale-${sequence++}`, format, length);
  try {
    const contributor = new GPUPointCloudNormalScale({
      positions: view('float32x3', 4),
      queryRows: view('uint32'),
      parameters: view('float32', 5),
      radii: [1, 2],
      capacity: 3,
      output: {
        normals: view('float32x3'),
        normalScales: view('float32'),
        normalVariances: view('float32'),
        status: view('uint32')
      }
    });
    const support = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).find(
      kernel => kernel.source.includes('var support = 0u')
    )!;
    expect(support).toBeDefined();
    expect(support.source).toContain('pointCloudNormKeysSeparated');
    expect(support.source).toContain('comparePointCloudNorms');
    expect(support.source).toContain('readQueryRows(index)');
    expect(support.source).toContain('readPositions(readIndices(');
  } finally {
    device.destroy();
  }
});

it('statistical filter uses the residual-refined root for both reference and retained distances', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const contributor = new GPUPointCloudStatisticalFilter({
      counts: createTransientView(graph, 'counts', 'uint32', 3),
      squaredDistances: createTransientView(graph, 'distances', 'float32', 3),
      capacity: 1,
      output: {
        meanDistances: createTransientView(graph, 'means', 'float32', 3),
        validity: createTransientView(graph, 'validity', 'uint32', 3)
      }
    });
    const means = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).find(kernel =>
      kernel.source.includes('let distanceReference')
    )!;
    expect(means.source).toContain('let distanceReference = sqrt_fp64(');
    expect(means.source).toContain('distanceSum += sqrt_fp64(');
  } finally {
    device.destroy();
  }
});
