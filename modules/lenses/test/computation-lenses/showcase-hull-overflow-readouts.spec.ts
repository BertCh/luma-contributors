// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPU_PLANAR_SPATIAL_CONTEXT} from '@luma-contributors/analysis/analysis-core';
import {GPUGroupConvexHull} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {requireTestDevice} from '../../../analysis/src/testing/require-test-device';
import {readUint32, submitGraph} from '../../../analysis/src/testing/gpu-contributor-test-utils';
import {SpatialAnalysisResources} from '../../../../examples/deck/spatial-analysis-showcase/src/engine/resources';
import {importGraphBuffer} from '../../../../examples/deck/spatial-analysis-showcase/src/engine/graph-buffers';
import showcaseSource from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/points/nature-clusters.compute.ts?raw';
import explorerSource from '../../../../examples/deck/spatial-analysis-explorer/modes/clusters-mode.ts?raw';
import {createApplicationContributor} from './showcase-accessibility-fixture';

it.each([
  'showcase',
  'explorer'
] as const)('GEO-118 %s direct hull wiring reads separate group and vertex capacity causes', async route => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // A square has four hull vertices; the separate triangle has three. With max=3
  // only the square fails. With max=4 and capacity=4 only the triangle fails.
  // With max=3 and capacity=2 both limits fail, each for a different group.
  for (const [maximum, capacity, aggregate, group, vertex] of [
    [3, 8, 1, 1, 0],
    [4, 4, 1, 0, 1],
    [3, 2, 1, 1, 1],
    [4, 8, 0, 0, 0]
  ]) {
    const resources = new SpatialAnalysisResources(
      device,
      `hull-causes-${route}-${maximum}-${capacity}`
    );
    try {
      const shapes = new GPUCommandGraph<void>(device);
      const positions = resources.createBuffer(
        'positions',
        Float32Array.of(0, 0, 2, 0, 2, 2, 0, 2, 5, 0, 7, 0, 5, 2)
      );
      const labels = resources.createBuffer('labels', Uint32Array.of(0, 0, 0, 0, 1, 1, 1));
      const hullOverflow = resources.createBuffer('overflow', 4);
      const hullGroupOverflow = resources.createBuffer('group-overflow', 4);
      const hullVertexOverflow = resources.createBuffer('vertex-overflow', 4);
      shapes.add(
        createApplicationContributor(
          route === 'showcase' ? showcaseSource : explorerSource,
          {
            GPUGroupConvexHull,
            importGraphBuffer,
            shapes,
            data: {frame: {spatialContext: GPU_PLANAR_SPATIAL_CONTEXT}},
            projection: {spatialContext: GPU_PLANAR_SPATIAL_CONTEXT},
            options: {prefilterLevels: 0},
            shapePositions: importGraphBuffer(shapes, 'positions', positions, 'float32x2', 7),
            shapeLabels: importGraphBuffer(shapes, 'labels', labels, 'uint32', 7),
            groupCount: 2,
            HULL_MAXIMUM_VERTICES: maximum,
            HULL_CAPACITY: capacity,
            hullIndices: resources.createBuffer('indices', capacity * 4),
            hullPositions: resources.createBuffer('vertices', capacity * 8),
            hullOffsets: resources.createBuffer('offsets', 12),
            hullCounts: resources.createBuffer('counts', 8),
            hullOverflow,
            hullGroupOverflow,
            hullVertexOverflow
          },
          'GPUGroupConvexHull'
        )
      );
      const compiled = resources.track(await shapes.compileAsync());
      submitGraph(device, compiled, undefined);
      expect(await readUint32(hullOverflow, 1)).toEqual([aggregate]);
      expect(await readUint32(hullGroupOverflow, 1)).toEqual([group]);
      expect(await readUint32(hullVertexOverflow, 1)).toEqual([vertex]);
    } finally {
      resources.destroy();
    }
  }
});
