// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  addVisibilityGraphRecipe as addNetworkVisibilityGraphRecipe,
  type GPUNetworkVisibilityGraphCSRProps as GPUVisibilityGraphNetworkProps
} from '@luma-contributors/analysis/gpu-network';
import {
  addBuildingTessellationReferenceRecipe,
  addLocalBuildingTessellationRecipe,
  getGPUBuildingTessellationLocalParameterValues,
  getGPUVisibilityGraphCSRParameterValues,
  GPU_VISIBILITY_GRAPH_CSR_PARAMETER_LENGTH,
  VISIBILITY_GRAPH_MAXIMUM_RAY_TOLERANCE,
  type GPUVisibilityGraphAdjacencyCSRProps,
  GPU_BUILDING_TESSELLATION_LOCAL_PARAMETER_LENGTH,
  type GPUBuildingTessellationLocalProps,
  type GPUBuildingTessellationLocalOutput,
  type GPUBuildingTessellationLocalStatus
} from '@luma-contributors/analysis/gpu-spatial-analysis';

it('the package entry exposes the local parameter API and types', () => {
  const values = getGPUBuildingTessellationLocalParameterValues({
    sampleSpacing: 2,
    candidateRadius: 9,
    vertexTolerance: 0.125
  });
  expect(Array.from(values)).toEqual([2, 9, 0.125]);
  expect(values.length).toBe(GPU_BUILDING_TESSELLATION_LOCAL_PARAMETER_LENGTH);
  const publicTypes: [
    GPUBuildingTessellationLocalProps?,
    GPUBuildingTessellationLocalOutput?,
    GPUBuildingTessellationLocalStatus?
  ] = [];
  expect(publicTypes).toHaveLength(0);
});

it('the package entry exposes the sparse visibility parameter API', () => {
  expect(
    Array.from(getGPUVisibilityGraphCSRParameterValues({maximumDistance: 3, tolerance: 0.125}))
  ).toEqual([3, 0.125]);
  expect(GPU_VISIBILITY_GRAPH_CSR_PARAMETER_LENGTH).toBe(2);
  expect(Math.fround(4 * VISIBILITY_GRAPH_MAXIMUM_RAY_TOLERANCE)).toBe(3.4028234663852886e38);
  const publicProps: GPUVisibilityGraphAdjacencyCSRProps | undefined = undefined;
  expect(publicProps).toBeUndefined();
});

it('the gpu-network entry exposes the original visibility recipe name with pivot options', () => {
  expect(typeof addNetworkVisibilityGraphRecipe).toBe('function');
  const networkOnlyProps: Pick<
    GPUVisibilityGraphNetworkProps,
    'pivotCount' | 'laneCount' | 'maximumIterations'
  > = {pivotCount: 4, laneCount: 2, maximumIterations: 8};
  expect(networkOnlyProps).toEqual({pivotCount: 4, laneCount: 2, maximumIterations: 8});
});

it('RCP-107 the package entry exposes both typed tessellation routes', () => {
  expect(typeof addBuildingTessellationReferenceRecipe).toBe('function');
  expect(typeof addLocalBuildingTessellationRecipe).toBe('function');
});
