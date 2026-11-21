// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createGridIndexNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/gwr-grid-index';
import {getGWRGridScratchSizes} from '../../../src/gpu-spatial-analysis/spatial-regression/gwr-scalability-utils';
import {GWR_GRID_EXTENT_COMPONENTS} from '../../../src/gpu-spatial-analysis/spatial-regression/gwr-grid-layout';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('GWR extent publication uses one invocation per ordered layout component', () => {
  expect(
    GWR_GRID_EXTENT_COMPONENTS.map(component => [component.name, component.operation])
  ).toEqual([
    ['minimumX', 'min'],
    ['minimumY', 'min'],
    ['maximumX', 'max'],
    ['maximumY', 'max']
  ]);
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const grid = createGridIndexNodes(graph, {
      id: 'extent',
      operation: 'extent-test',
      positions: createTransientView(graph, 'positions', 'float32x2', 0),
      rowCount: 0,
      gridSize: [1, 1]
    });
    expect(grid.sortedIds.length).toBe(4);
    const bounds = getGPUCommandNodeKernelSources(grid.nodes).find(
      kernel => kernel.id === 'extent-grid-bounds'
    )!;
    expect(bounds.constants?.['INVOCATION_COUNT']).toBe(4);
    expect(bounds.source).not.toContain('for (var component');
    expect(bounds.source).toContain('isEmpty');
    const scratch = Object.fromEntries(getGWRGridScratchSizes(257, [1, 1]));
    expect(scratch['grid bounds']).toBe(16);
    expect(scratch['grid tile bounds']).toBe(1024);
    expect(scratch['grid sorted IDs and extent']).toBe(1044);
  } finally {
    device.destroy();
  }
});

for (const behavior of ['publication', 'normalized-input'] as const) {
  it(`GWR zero-row extent preserves empty ${behavior}`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const grid = createGridIndexNodes(graph, {
        id: 'zero',
        operation: 'extent-test',
        positions: createTransientView(graph, 'positions', 'float32x2', 0),
        rowCount: 0,
        gridSize: [1, 1]
      });
      if (behavior === 'publication') {
        const publication = getGPUCommandNodeKernelSources(grid.nodes).find(
          kernel => kernel.id === 'zero-grid-bounds'
        )!;
        // Empty GPUReduction writes zero, so extent ordering cannot detect the zero-row case.
        expect(publication.source).toContain('ROW_COUNT == 0u');
      } else {
        const normalization = grid.nodes.find(node => node.id === 'zero-unit-positions')!;
        const destination = normalization.resources?.find(
          resource => resource.usage === 'storage-read-write'
        );
        const logicalLength =
          destination && 'buffer' in destination && 'length' in destination.buffer
            ? destination.buffer.length
            : undefined;
        expect(logicalLength).toBe(0);
      }
    } finally {
      device.destroy();
    }
  });
}

for (const direction of [1, -1]) {
  it(`GWR tile extrema do not clip finite float32 coordinates: direction=${direction}`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const grid = createGridIndexNodes(graph, {
        id: 'extreme',
        operation: 'extent-test',
        positions: createTransientView(graph, 'positions', 'float32x2', 2),
        rowCount: 2,
        gridSize: [1, 1]
      });
      const tile = getGPUCommandNodeKernelSources(grid.nodes).find(
        kernel => kernel.id === 'extreme-tile-bounds'
      )!;
      const literal = /const SENTINEL: f32 = ([^;]+);/.exec(tile.source)![1];
      const sentinel = Math.fround(Number(literal));
      // Independent closed form: the minimum/maximum of two same-sign coordinates must be
      // an input coordinate. Transcribe only the generated reduction identity and min/max fold.
      const coordinates = Float32Array.of(direction * 3.1e38, direction * 3.2e38);
      const minimum = Math.min(sentinel, ...coordinates);
      const maximum = Math.max(-sentinel, ...coordinates);
      expect(Number.isFinite(minimum)).toBe(true);
      expect(Number.isFinite(maximum)).toBe(true);
      expect(minimum).toBe(direction > 0 ? coordinates[0] : coordinates[1]);
      expect(maximum).toBe(direction > 0 ? coordinates[1] : coordinates[0]);
    } finally {
      device.destroy();
    }
  });
}
