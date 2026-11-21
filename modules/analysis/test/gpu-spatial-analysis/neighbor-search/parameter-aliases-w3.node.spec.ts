// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {packGPUSpatialParameterValues} from '../../../src/analysis-core/contracts/parameter-schema';
import {
  getGPUNeighborSearchParameterValues,
  getGPUNeighborSearchWeightTransform,
  GPU_NEIGHBOR_SEARCH_PARAMETER_SCHEMA
} from '../../../src/gpu-spatial-analysis/neighbor-search/index';
import {GPUNeighborSearch} from '../../../src/gpu-spatial-analysis/neighbor-search/gpu-neighbor-search';
import {getGPUSpatialWeightsKernelValue} from '../../../src/gpu-spatial-analysis/spatial-weights/index';

const bounds = [0, 0, 4, 4] as const;

it('SCL-16 plans actual auto and forced routes with numeric binding admission', () => {
  const shape = {
    mode: 'knn' as const,
    targetRowCount: 2,
    queryRowCount: 4,
    neighborCount: 4,
    gridSize: [2, 2] as const
  };
  const limits = {maxBufferSize: 1024, maxStorageBufferBindingSize: 64};
  const oldShape = {...shape, k: 4};
  expect(() => GPUNeighborSearch.getExecutionPlan(oldShape, limits)).toThrow(
    'GPUNeighborSearch.getExecutionPlan: k was renamed; use neighborCount'
  );
  const plan = GPUNeighborSearch.getExecutionPlan(shape, limits);
  expect(plan.route).toBe('grid');
  expect(plan.gridCellCount).toBe(4);
  expect(plan.nearestListLength).toBe(16);
  expect(plan.feasible).toBe(true); // grid=20, query triples=48, each kNN list=4*4*4=64
  expect(GPUNeighborSearch.getExecutionPlan({...shape, queryRowCount: 5}, limits).feasible).toBe(
    false
  );
  expect(GPUNeighborSearch.getExecutionPlan({...shape, searchStructure: 'bvh'}, limits).route).toBe(
    'bvh'
  );
  expect(() =>
    GPUNeighborSearch.getExecutionPlan({...shape, searchStructure: 'bvh', hasTimes: true}, limits)
  ).toThrow(/times/);
  expect(() =>
    GPUNeighborSearch.getExecutionPlan(
      {...shape, searchStructure: 'bvh', coordinateSpace: 'longitude-latitude'},
      limits
    )
  ).toThrow(/planar/);
  const temporal = GPUNeighborSearch.getExecutionPlan(
    {...shape, hasTimes: true, timeBucketCount: 4},
    limits
  );
  expect(temporal.gridCellCount).toBe(16);
  expect(temporal.recordWords).toBe(4);
  expect(temporal.feasible).toBe(false); // (16+1)*4=68 > 64
});

it('NAM-02 shares the inverse-distance spelling with nearest-feature weights', () => {
  const weighting = {weightType: 'inverse-distance'} as const;
  const settings = {bounds, ...weighting};
  expect(getGPUNeighborSearchParameterValues(settings)[5]).toBe(1);
  expect(getGPUNeighborSearchWeightTransform(settings)).toBe('raw');
  expect(getGPUNeighborSearchWeightTransform({...settings, rowStandardize: true})).toBe('row');
  expect(() =>
    getGPUNeighborSearchParameterValues({...settings, weightKind: 'binary'} as never)
  ).toThrow(/weightKind was renamed; use weightType/);
  expect(() =>
    getGPUNeighborSearchWeightTransform({...settings, weightKind: 'binary'} as never)
  ).toThrow(/weightKind was renamed; use weightType/);
  for (const weightType of ['unknown', 'toString', 'inverseDistance']) {
    expect(() => getGPUNeighborSearchParameterValues({bounds, weightType} as never)).toThrow(
      /binary.*inverse-distance.*kernel/
    );
  }
});

it('NAM-12 requires a finite positive float32 searchRadius in radius mode', () => {
  for (const searchRadius of [undefined, 0, -1, Infinity, NaN, 1e100, Number.MIN_VALUE]) {
    expect(() =>
      getGPUNeighborSearchParameterValues({mode: 'radius', bounds, searchRadius} as never)
    ).toThrow(/searchRadius/);
  }
  expect(getGPUNeighborSearchParameterValues({mode: 'radius', bounds, searchRadius: 1.1})[4]).toBe(
    Math.fround(1.1)
  );
  expect(() =>
    getGPUNeighborSearchParameterValues({mode: 'radius', bounds, radius: 1.1} as never)
  ).toThrow('radius was renamed; use searchRadius');
  expect(getGPUNeighborSearchParameterValues({mode: 'knn', bounds})[4]).toBe(Infinity);
  expect(getGPUNeighborSearchParameterValues({bounds})[4]).toBe(Infinity); // legacy overload
  expect(getGPUNeighborSearchParameterValues({mode: 'knn', bounds, maximumDistance: 2})[4]).toBe(2);
  expect(() =>
    getGPUNeighborSearchParameterValues({
      mode: 'knn',
      bounds,
      maximumDistance: 2,
      radius: 3
    } as never)
  ).toThrow('radius was renamed; use maximumDistance');
  expect(() =>
    getGPUNeighborSearchParameterValues({
      mode: 'radius',
      bounds,
      searchRadius: 2,
      radius: 3
    } as never)
  ).toThrow('radius was renamed; use searchRadius');
  expect(() => getGPUNeighborSearchParameterValues({mode: 'other', bounds} as never)).toThrow(
    /radius.*knn/
  );
});

it('NAM-16 packs quartic and bisquare as the same normalized kernel', () => {
  for (const kernel of ['quartic', 'bisquare'] as const) {
    expect(getGPUNeighborSearchParameterValues({bounds, weightType: 'kernel', kernel})[8]).toBe(3);
    // Independent exact value: 15/16 * (1 - (1/2)^2)^2 = 135/256.
    // Neighbor shaders read code 3 for either spelling. The weighting host reference retains
    // its legacy bisquare spelling (its public quartic alias belongs to the weights stream).
    expect(getGPUSpatialWeightsKernelValue('bisquare', 0.5)).toBe(0.52734375);
  }
});

it('the raw parameter schema encodes the zero radius used for an empty neighbor list', () => {
  const packed = packGPUSpatialParameterValues(GPU_NEIGHBOR_SEARCH_PARAMETER_SCHEMA, {
    minimumX: -500,
    minimumY: -500,
    maximumX: 1500,
    maximumY: 1500,
    maximumDistance: 0
  });
  expect(packed).toHaveLength(12);
  expect(Array.from(packed.subarray(0, 6))).toEqual([-500, -500, 1500, 1500, 0, 0]);
});
