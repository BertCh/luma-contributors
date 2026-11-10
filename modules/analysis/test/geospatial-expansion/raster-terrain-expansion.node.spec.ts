// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPURasterFillNoData} from '../../src/raster-analysis/gpu-raster-fill-nodata';
import {GPURasterRankFilter} from '../../src/raster-analysis/gpu-raster-rank-filter';
import {
  GPURasterFillHoles,
  GPURasterReconstruction
} from '../../src/raster-analysis/gpu-raster-reconstruction';
import {GPURasterWatershedSegmentation} from '../../src/raster-analysis/gpu-raster-watershed-segmentation';
import {GPURasterWalkingCost} from '../../src/raster-analysis/gpu-raster-walking-cost';
import {
  GPURasterSkeleton,
  GPURasterMedialAxis
} from '../../src/raster-analysis/gpu-raster-skeleton';
import {GPURasterReprojection} from '../../src/raster-analysis/gpu-raster-reprojection';
import {GPUTerrainFlowDistance} from '../../src/gpu-terrain/hydrology/gpu-terrain-flow-distance';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount,
  getPrivateMemoryBytes,
  CORE_PRIVATE_MEMORY_LIMIT_BYTES
} from '../utils/gpu-contributor-test-utils';

it('bounded raster expansion variants parse, preserve eight bindings and fit private memory', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let identifier = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length = 9) =>
    createTransientView(graph, `view-${identifier++}`, format, length);
  const grid = {
    width: 3,
    height: 3,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const input = {
    id: 'input',
    format: 'float32' as const,
    storage: {kind: 'buffer' as const, values: view('float32')}
  };
  const contributors = [
    new GPURasterFillNoData({
      grid,
      input,
      maximumDistance: 256,
      method: 'nearest',
      output: {
        values: view('float32'),
        validity: view('uint32'),
        filled: view('uint32'),
        unresolvedCount: view('uint32', 1)
      }
    }),
    new GPURasterRankFilter({
      grid,
      input,
      radius: 8,
      quantile: 1,
      borderMode: 'nodata',
      noDataPolicy: 'ignore',
      output: {values: view('float32'), validity: view('uint32'), changed: view('uint32')}
    }),
    new GPURasterReconstruction({
      grid,
      marker: view('float32'),
      mask: view('float32'),
      method: 'erosion',
      connectivity: 8,
      maximumIterations: 2,
      output: {
        values: view('float32'),
        validity: view('uint32'),
        status: {converged: view('uint32', 1)}
      }
    }),
    new GPURasterWatershedSegmentation({
      grid,
      input,
      markers: view('uint32'),
      connectivity: 8,
      output: {labels: view('uint32'), status: {converged: view('uint32', 1)}}
    }),
    new GPURasterWalkingCost({
      grid,
      elevation: input,
      friction: input,
      sources: view('uint32'),
      maximumIterations: 2,
      output: {
        costs: view('float32'),
        validity: view('uint32'),
        status: {converged: view('uint32', 1)}
      }
    }),
    new GPURasterSkeleton({
      grid,
      input: view('uint32'),
      maximumIterations: 2,
      output: {skeleton: view('uint32'), status: {converged: view('uint32', 1)}}
    }),
    new GPURasterMedialAxis({
      grid,
      input: view('uint32'),
      output: {medialAxis: view('uint32'), radius: view('float32')}
    }),
    new GPURasterReprojection({
      sourceGrid: grid,
      destinationGrid: grid,
      input,
      method: 'bilinear',
      noDataPolicy: 'renormalize',
      output: {values: view('float32'), validity: view('uint32')}
    }),
    new GPUTerrainFlowDistance({
      grid,
      flowDirections: view('uint32'),
      streams: view('uint32'),
      transitionCosts: view('float32'),
      output: {distance: view('float32'), validity: view('uint32'), terminalCells: view('uint32')}
    })
  ];
  for (const contributor of contributors) {
    const sources = getParsedKernelSources(contributor.getCommandNodes(graph));
    expect(sources.size).toBeGreaterThan(0);
    for (const [name, source] of sources) {
      expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
      expect(getPrivateMemoryBytes(source), name).toBeLessThanOrEqual(
        CORE_PRIVATE_MEMORY_LIMIT_BYTES
      );
    }
  }
});

import {GPUConcaveHull} from '../../src/gpu-spatial-analysis/concave-hull/gpu-concave-hull';
import {GPUConstrainedDelaunayTriangulation} from '../../src/gpu-spatial-analysis/polygon-triangulation/gpu-constrained-delaunay-triangulation';

it('geometry expansion rejects mismatched dense capacities and aliased convergence outputs', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let identifier = 0;
  const view = <Format extends 'uint32' | 'float32x2' | 'uint32x2' | 'uint32x3'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `geometry-${identifier++}`, format, length);
  const positions = view('float32x2', 4);
  expect(
    () =>
      new GPUConcaveHull({
        positions,
        maximumCircumradius: 1,
        output: {
          triangles: view('uint32x3', 4),
          triangleValidity: view('uint32', 3),
          boundaryEdges: view('uint32x2', 9),
          edgeValidity: view('uint32', 9),
          status: {converged: view('uint32', 1)}
        }
      })
  ).toThrow();
  const valid = view('uint32', 1);
  expect(
    () =>
      new GPUConstrainedDelaunayTriangulation({
        positions,
        featureOffsets: view('uint32', 2),
        polygonOffsets: view('uint32', 2),
        ringOffsets: view('uint32', 2),
        output: {indices: view('uint32', 6), valid, status: {converged: valid}}
      })
  ).toThrow();
});

it('bounded raster iterations accept the documented largest maximumIterations and reject one more', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let identifier = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length = 9) =>
    createTransientView(graph, `limit-view-${identifier++}`, format, length);
  const grid = {
    width: 3,
    height: 3,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const input = {
    id: 'input',
    format: 'float32' as const,
    storage: {kind: 'buffer' as const, values: view('float32')}
  };
  const status = () => ({converged: view('uint32', 1)});
  const classes = {
    walkingCost: (maximumIterations: number) =>
      new GPURasterWalkingCost({
        grid,
        elevation: input,
        friction: input,
        sources: view('uint32'),
        maximumIterations,
        output: {costs: view('float32'), validity: view('uint32'), status: status()}
      }),
    reconstruction: (maximumIterations: number) =>
      new GPURasterReconstruction({
        grid,
        marker: view('float32'),
        mask: view('float32'),
        method: 'erosion',
        connectivity: 8,
        maximumIterations,
        output: {values: view('float32'), validity: view('uint32'), status: status()}
      }),
    fillHoles: (maximumIterations: number) =>
      new GPURasterFillHoles({
        grid,
        input: view('uint32'),
        maximumIterations,
        output: {values: view('uint32'), status: status()}
      }),
    skeleton: (maximumIterations: number) =>
      new GPURasterSkeleton({
        grid,
        input: view('uint32'),
        maximumIterations,
        output: {skeleton: view('uint32'), status: status()}
      })
  };
  const largest = {walkingCost: 26196, reconstruction: 26209, fillHoles: 26207, skeleton: 10920};
  for (const [name, create] of Object.entries(classes) as [
    keyof typeof classes,
    (maximumIterations: number) => unknown
  ][]) {
    expect(() => create(largest[name])).not.toThrow();
    expect(() => create(largest[name] + 1)).toThrow(/estimated command node count/);
  }
});
