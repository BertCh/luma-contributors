// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  GPUPointCloudNeighborhoods,
  GPUPointCloudFeatures,
  GPUPointCloudStatisticalFilter,
  GPUPointCloudVoxelMerge,
  GPUPointCloudSurfaces,
  GPUPointCloudHeight,
  GPUPointCloudGround,
  GPUPointCloudSMRF,
  GPUPointCloudICP,
  GPUPointCloudChange
} from '@luma-contributors/analysis/pointcloud-analysis';
import {GPUPointCloudVoxelDownsample} from '@luma.gl/graph/gpu-pointcloud';
import {createNullWebGPUDevice, getParsedKernelSources} from '../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../analysis-core/kernel-audit/kernel-audit-wgsl';

it('point-cloud analysis generated shaders pass reserved-word and baseline resource audits', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x3'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `pointcloud-audit-${serial++}`, format, length);
  const positions = view('float32x3', 9);
  const indices = view('uint32', 9 * 64);
  const squaredDistances = view('float32', 9 * 64);
  const counts = view('uint32', 9);
  const normals = view('float32x3', 9);
  const classifications = view('uint32', 9);
  const terrain = view('float32', 9);
  const coverage = view('uint32', 9);
  const grid = {origin: [0, 0] as const, cellSize: 1, width: 3, height: 3};
  const contributors = [
    new GPUPointCloudNeighborhoods({
      positions,
      indices,
      squaredDistances,
      counts,
      capacity: 64,
      radius: 10
    }),
    new GPUPointCloudNeighborhoods({
      id: 'indexed-neighborhood-audit',
      positions,
      indices,
      squaredDistances,
      counts,
      capacity: 64,
      radius: 10,
      grid: {origin: [0, 0, 0], cellSize: 2, dimensions: [3, 3, 3]},
      status: view('uint32', 9)
    }),
    new GPUPointCloudFeatures({
      positions,
      indices,
      counts,
      capacity: 64,
      normals,
      eigenvalues: view('float32x3', 9),
      covariance: view('float32', 54),
      shapes: view('float32x3', 9),
      status: view('uint32', 9),
      viewpoint: [0, 0, 20]
    }),
    new GPUPointCloudStatisticalFilter({
      squaredDistances,
      counts,
      capacity: 64,
      meanDistances: view('float32', 9),
      mask: view('uint32', 9)
    }),
    new GPUPointCloudVoxelDownsample({
      positions,
      origin: [0, 0, 0],
      voxelSize: 1,
      gridSize: [3, 3, 1],
      representatives: view('uint32', 9),
      counts: view('uint32', 9),
      output: view('float32x3', 9)
    }),
    new GPUPointCloudVoxelMerge({
      partitionPolicy: 'disjoint-owned-rows',
      parts: [
        {
          counts: view('uint32', 9),
          centroids: view('float32x3', 9),
          representatives: view('uint32', 9),
          sourceRowOffset: 9,
          attribute: {means: view('float32x3', 9), counts: view('uint32', 9)}
        }
      ],
      counts: view('uint32', 9),
      centroids: view('float32x3', 9),
      representatives: view('uint32', 9),
      status: view('uint32', 9),
      attribute: {means: view('float32x3', 9), counts: view('uint32', 9)}
    }),
    new GPUPointCloudSurfaces({
      positions,
      classifications,
      grid,
      terrain,
      coverage,
      surface: view('float32', 9),
      canopy: view('float32', 9)
    }),
    new GPUPointCloudHeight({
      positions,
      grid,
      terrain,
      coverage,
      heights: view('float32', 9),
      status: view('uint32', 9)
    }),
    new GPUPointCloudGround({
      positions,
      classifications,
      grid,
      minimumSurface: terrain,
      coverage,
      radii: [1, 2],
      initialThreshold: 0.1,
      slope: 0.1,
      maximumThreshold: 1,
      groundMask: view('uint32', 9),
      outputClassifications: view('uint32', 9),
      status: view('uint32', 9)
    }),
    new GPUPointCloudSMRF({
      positions,
      classifications,
      selection: view('uint32', 9),
      grid,
      window: 2,
      cut: 1,
      groundSurface: view('float32', 9),
      groundMask: view('uint32', 9),
      outputClassifications: view('uint32', 9),
      status: view('uint32', 9)
    }),
    new GPUPointCloudICP({
      source: positions,
      target: view('float32x3', 7),
      targetNormals: view('float32x3', 7),
      metric: 'point-to-plane',
      targetGrid: {origin: [0, 0, 0], cellSize: 2, dimensions: [3, 3, 3]},
      robust: {method: 'tukey', scale: 1},
      maximumDistance: 10,
      iterations: 2,
      transform: view('float32', 16),
      history: view('float32x3', 2),
      inlierMask: view('uint32', 9),
      status: view('uint32', 1)
    }),
    new GPUPointCloudChange({
      reference: positions,
      comparison: view('float32x3', 7),
      cores: view('float32x3', 9),
      normals,
      alignment: 'already-aligned',
      registrationCovariance: [1, 0, 0, 1, 0, 1],
      uncertaintyCombination: 'quadrature',
      samplingCovariance: {referenceVariance: 1, comparisonVariance: 1, correlationLength: 1},
      radius: 1,
      halfDepth: 1,
      changes: view('float32x3', 9),
      roughness: view('float32x3', 9),
      counts: view('uint32', 18),
      status: view('uint32', 9)
    })
  ];
  for (const contributor of contributors) {
    const nodes = contributor.getCommandNodes(graph);
    for (const [identifier, source] of getParsedKernelSources(nodes)) {
      expect(auditWGSLModule(source), identifier).toEqual([]);
    }
  }
});

it('rejects indefinite registration covariance independently of coordinate scale and float32 overflow', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x3'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `change-validation-${serial++}`, format, length);
  const props = {
    reference: view('float32x3', 3),
    comparison: view('float32x3', 3),
    cores: view('float32x3', 1),
    normals: view('float32x3', 1),
    alignment: 'already-aligned' as const,
    radius: 1,
    halfDepth: 1,
    changes: view('float32x3', 1),
    roughness: view('float32x3', 1),
    counts: view('uint32', 2),
    status: view('uint32', 1)
  };
  for (const scale of [1e-20, 1e-6, 1, 1e20]) {
    expect(
      () =>
        new GPUPointCloudChange({
          ...props,
          registrationCovariance: [scale, 0.9 * scale, 0.9 * scale, scale, -0.9 * scale, scale]
        })
    ).toThrow();
    expect(
      () =>
        new GPUPointCloudChange({
          ...props,
          registrationCovariance: [scale, scale, scale, scale, scale, scale]
        })
    ).not.toThrow();
  }
  expect(() => new GPUPointCloudChange({...props, registrationError: 1e30})).toThrow();
  expect(
    () =>
      new GPUPointCloudChange({
        ...props,
        samplingCovariance: {
          referenceVariance: 1,
          comparisonVariance: 1,
          correlationLength: 1.5e19
        }
      })
  ).toThrow();
});

it('rejects robust ICP scales whose generated squared float32 scale underflows or overflows', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x3'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `icp-validation-${serial++}`, format, length);
  const props = {
    source: view('float32x3', 3),
    target: view('float32x3', 3),
    maximumDistance: 1,
    iterations: 1,
    transform: view('float32', 16),
    history: view('float32x3', 1),
    inlierMask: view('uint32', 3),
    status: view('uint32', 1)
  };
  for (const scale of [1e-30, 1e30]) {
    expect(() => new GPUPointCloudICP({...props, robust: {method: 'tukey', scale}})).toThrow();
  }
});

it('voxel merge rejects cross-pass output aliases before they overwrite final records', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x3'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `merge-validation-${serial++}`, format, length);
  const shared = view('uint32', 1);
  expect(
    () =>
      new GPUPointCloudVoxelMerge({
        partitionPolicy: 'disjoint-owned-rows',
        parts: [
          {
            counts: view('uint32', 1),
            centroids: view('float32x3', 1),
            representatives: view('uint32', 1),
            sourceRowOffset: 0,
            attribute: {means: view('float32', 1), counts: view('uint32', 1)}
          }
        ],
        counts: view('uint32', 1),
        centroids: view('float32x3', 1),
        representatives: view('uint32', 1),
        status: shared,
        attribute: {means: view('float32', 1), counts: shared}
      })
  ).toThrow();
});

it('height interpolation rejects radius and support constants that cannot be represented in WGSL', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x3'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `height-validation-${serial++}`, format, length);
  const props = {
    positions: view('float32x3', 1),
    terrain: view('float32', 1),
    coverage: view('uint32', 1),
    heights: view('float32', 1),
    status: view('uint32', 1),
    grid: {origin: [0, 0] as const, cellSize: 1, width: 1, height: 1}
  };
  for (const radius of [1e-30, 1e30]) {
    expect(
      () =>
        new GPUPointCloudHeight({
          ...props,
          grid: {...props.grid, cellSize: radius},
          interpolation: {method: 'idw', radius}
        })
    ).toThrow();
  }
  expect(
    () =>
      new GPUPointCloudHeight({
        ...props,
        interpolation: {method: 'idw', radius: 1, minimumSupport: 0x100000000}
      })
  ).toThrow();
});
