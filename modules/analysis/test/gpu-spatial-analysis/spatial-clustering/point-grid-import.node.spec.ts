// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {POINT_NEIGHBOR_GRID_UNIT_POSITION_WGSL} from '../../../src/analysis-core/kernels/point-neighbor-grid-index';
import {POINT_NEIGHBOR_GRID_UNIT_POSITION_WGSL as COMPATIBILITY_WGSL} from '../../../src/gpu-spatial-analysis/shared/point-neighbor-grid-index';
import {
  createSpatialClusteringGridPositionsNode,
  getSpatialClusteringSharedWGSL
} from '../../../src/gpu-spatial-analysis/spatial-clustering/spatial-clustering-kernels';
import {
  SPATIAL_CLUSTERING_MINIMUM_PLANAR_EPSILON,
  SPATIAL_CLUSTERING_MAXIMUM_PLANAR_EPSILON
} from '../../../src/gpu-spatial-analysis/spatial-clustering/spatial-clustering-parameters';

it('retains the identical point-grid declaration and planar epsilon guards with the core import', () => {
  expect(POINT_NEIGHBOR_GRID_UNIT_POSITION_WGSL).toBe(COMPATIBILITY_WGSL);
  const source = getSpatialClusteringSharedWGSL(4, 4);
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const node = createSpatialClusteringGridPositionsNode(graph, {
      id: 'grid',
      gridSize: [4, 4],
      positions: createTransientView(graph, 'positions', 'float32x2', 2),
      parameters: createTransientView(graph, 'parameters', 'float32', 8),
      gridPositions: createTransientView(graph, 'grid-positions', 'float32x2', 2),
      rowIds: createTransientView(graph, 'rows', 'uint32', 2)
    });
    expect(getGPUCommandNodeKernelSources([node])[0].source).toContain(
      POINT_NEIGHBOR_GRID_UNIT_POSITION_WGSL
    );
  } finally {
    device.destroy();
  }
  expect(SPATIAL_CLUSTERING_MINIMUM_PLANAR_EPSILON).toBe(2 ** -63);
  expect(SPATIAL_CLUSTERING_MAXIMUM_PLANAR_EPSILON).toBe(2 ** 63);
  expect(source).toContain('epsilon >=');
  expect(source).toContain('epsilon <=');
});
