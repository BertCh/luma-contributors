// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUGroupConvexHull} from '../../../src/gpu-spatial-analysis/group-geometry/gpu-group-convex-hull';
import {GPUMinimumBounds} from '../../../src/gpu-spatial-analysis/minimum-bounds/gpu-minimum-bounds';
import {
  GPUGridGenerator,
  getGPUGridGeneratorParameterValues
} from '../../../src/gpu-spatial-analysis/grid-generators/gpu-grid-generator';
import {
  GPUShapeGenerator,
  getGPUShapeGeneratorParameterValues
} from '../../../src/gpu-spatial-analysis/grid-generators/gpu-shape-generator';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('preflights group hull and enclosing minimum-bounds scratch before allocating', () => {
  for (const nested of [false, true]) {
    const device = createNullWebGPUDevice();
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 65536});
    const graph = new GPUCommandGraph(device);
    const positions = createTransientView(graph, 'positions', 'float32x2', 8192);
    const labels = createTransientView(graph, 'labels', 'uint32', 8192);
    const overflow = createTransientView(graph, 'overflow', 'uint32', 1);
    const contributor = nested
      ? new GPUMinimumBounds({
          positions,
          labels,
          groupCount: 1,
          output: {
            circles: createTransientView(graph, 'circles', 'float32x3', 1),
            status: {overflow}
          }
        })
      : new GPUGroupConvexHull({
          positions,
          labels,
          groupCount: 1,
          maximumVerticesPerGroup: 8,
          vertexCapacity: 1,
          output: {
            vertexIndices: createTransientView(graph, 'indices', 'uint32', 1),
            counts: createTransientView(graph, 'counts', 'uint32', 1),
            offsets: createTransientView(graph, 'offsets', 'uint32', 2),
            status: {overflow}
          }
        });
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /hull prefilter scratch needs 65544 bytes/
    );
    expect(allocate).not.toHaveBeenCalled();
    vi.restoreAllMocks();
    device.destroy();
  }
});

it('preflights ellipse arc tables before allocating or declaring nodes', () => {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 4096});
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUShapeGenerator({
    shape: 'ellipse',
    ellipseSpacing: 'arc-length',
    maximumSegments: 3,
    centers: createTransientView(graph, 'centers', 'float32x2', 4),
    radii: createTransientView(graph, 'radii', 'float32x2', 4),
    parameters: createTransientView(graph, 'parameters', 'float32', 2),
    output: {
      positions: createTransientView(graph, 'positions', 'float32x2', 16),
      offsets: createTransientView(graph, 'offsets', 'uint32', 5)
    }
  });
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /ellipse arc-length scratch needs 4112 bytes/
  );
  expect(allocate).not.toHaveBeenCalled();
  vi.restoreAllMocks();
  device.destroy();
});

it('checks STORAGE usage and aligned ranges, while accepting valid packed fields in separate passes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const parameters = createTransientView(graph, 'parameters', 'float32', 4);
  const backing = createTransientView(graph, 'packed-output', 'float32x2', 5);
  const contributor = new GPUGridGenerator({
    gridType: 'square',
    columns: 1,
    rows: 1,
    parameters,
    output: {
      centers: graph.createDataView(backing.buffer, {format: 'float32x2', length: 1}),
      corners: graph.createDataView(backing.buffer, {format: 'float32x2', byteOffset: 8, length: 4})
    }
  });
  expect(() => graph.add(contributor.getCommandNodes(graph))).not.toThrow();
  const nonStorage = graph.createTransientBuffer({
    id: 'non-storage',
    byteLength: 16,
    usage: Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const rejected = new GPUGridGenerator({
    gridType: 'point',
    columns: 1,
    rows: 1,
    parameters: graph.createDataView(nonStorage, {format: 'float32', length: 4}),
    output: {centers: createTransientView(graph, 'centers', 'float32x2', 1)}
  });
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => rejected.getCommandNodes(graph)).toThrow(/buffer usage must include Buffer.STORAGE/);
  expect(allocate).not.toHaveBeenCalled();
  vi.restoreAllMocks();
  device.destroy();
});

it('rejects grid and shape settings that cannot be encoded as finite float32', () => {
  expect(() =>
    getGPUGridGeneratorParameterValues({minX: 1e40, minY: 0, cellWidth: 1, cellHeight: 1})
  ).toThrow(/finite/);
  expect(() =>
    getGPUGridGeneratorParameterValues({minX: 0, minY: 0, cellWidth: 1e-50, cellHeight: 1})
  ).toThrow(/positive/);
  expect(() => getGPUShapeGeneratorParameterValues({segmentCount: 1e40})).toThrow(/finite/);
  expect(() => getGPUShapeGeneratorParameterValues({segmentCount: 8, radiusScale: 1e40})).toThrow(
    /finite/
  );
});
