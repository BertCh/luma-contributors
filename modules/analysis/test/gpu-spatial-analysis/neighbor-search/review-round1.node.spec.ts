// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUNeighborSearch,
  getGPUNeighborSearchParameterValues
} from '../../../src/gpu-spatial-analysis/neighbor-search';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {computeNeighborSearchOracle} from './neighbor-search-oracle';

it('B2 includes shape-determined grid records and temporal admission', () => {
  const device = createNullWebGPUDevice();
  const shape = {mode: 'radius' as const, targetRowCount: 12000000, gridSize: [8, 8] as const};
  const plan = GPUNeighborSearch.getExecutionPlan(shape, device.limits);
  // Input coordinates need 96 MB, but each sorted target triple needs 144 MB > 128 MiB.
  expect(plan.feasible).toBe(false);
  expect(
    plan.constraints.some(constraint => constraint.received === 144000000 && !constraint.satisfied)
  ).toBe(true);
  expect(() =>
    GPUNeighborSearch.getExecutionPlan({...shape, timeBucketCount: 9}, device.limits)
  ).toThrow(/timeBucketCount.*times/);
  const boundary = Math.floor(134217728 / 12);
  expect(
    GPUNeighborSearch.getExecutionPlan({...shape, targetRowCount: boundary}, device.limits).feasible
  ).toBe(true);
  expect(
    GPUNeighborSearch.getExecutionPlan({...shape, targetRowCount: boundary + 1}, device.limits)
      .feasible
  ).toBe(false);
  // Three rows pad to four BVH leaves: seven nodes * five uint32 words = 140 bytes.
  const bvh = GPUNeighborSearch.getExecutionPlan(
    {...shape, gridSize: [2, 2], targetRowCount: 3, searchStructure: 'bvh'},
    {maxBufferSize: 1024, maxStorageBufferBindingSize: 128}
  );
  expect(bvh.bvhLeafCapacity).toBe(4);
  expect(bvh.feasible).toBe(false);
  expect(bvh.constraints.find(constraint => constraint.name === 'BVH nodeData')?.received).toBe(
    140
  );
  expect(
    GPUNeighborSearch.getExecutionPlan(
      {...shape, gridSize: [2, 2], targetRowCount: 2, searchStructure: 'bvh'},
      {maxBufferSize: 1024, maxStorageBufferBindingSize: 128}
    ).feasible
  ).toBe(true);
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `review-${serial++}`, format, length);
  const contributor = new GPUNeighborSearch({
    mode: 'radius',
    positions: view('float32x2', 12000000),
    gridSize: [8, 8],
    parameters: view('float32', 12),
    output: {
      offsets: view('uint32', 12000001),
      neighbors: view('uint32', 1),
      weights: view('float32', 1),
      status: {overflow: view('uint32', 1)}
    }
  });
  expect(() => contributor.getCommandNodes(graph)).toThrow(/sortedTargets.*144000000/);
  device.destroy();
});

it('B5 resolves new aliases in the independent brute-force CSR oracle', () => {
  for (const mode of ['radius', 'knn'] as const) {
    const result = computeNeighborSearchOracle({
      mode,
      neighborCount: 2,
      positions: Float32Array.of(0, 0, 2, 0, 5, 0),
      parameters: {
        bounds: [-1, -1, 6, 1],
        weightType: 'inverse-distance',
        ...(mode === 'radius' ? {searchRadius: 2.5} : {maximumDistance: 2.5})
      }
    });
    // Distances 2,3,5: only the symmetric 0<->1 edge fits; reciprocal distance = 1/2.
    expect(result).toEqual({
      offsets: [0, 1, 2, 2],
      neighbors: [1, 0],
      distances: [2, 2],
      distancesSquared: [4, 4],
      weights: [0.5, 0.5],
      counts: [1, 1, 0]
    });
  }
});

it('S1 rejects finite planar distance limits whose float32 square overflows', () => {
  const bounds = [0, 0, 1, 1] as const;
  expect(Math.fround(Math.fround(2e19) ** 2)).toBe(Infinity);
  expect(() =>
    getGPUNeighborSearchParameterValues({mode: 'radius', bounds, searchRadius: 2e19})
  ).toThrow(/searchRadius.*square/);
  expect(() =>
    getGPUNeighborSearchParameterValues({mode: 'knn', bounds, maximumDistance: 2e19})
  ).toThrow(/maximumDistance.*square/);
  expect(getGPUNeighborSearchParameterValues({mode: 'radius', bounds, searchRadius: 1e19})[4]).toBe(
    Math.fround(1e19)
  );
  expect(
    getGPUNeighborSearchParameterValues({mode: 'knn', bounds, maximumDistance: Infinity})[4]
  ).toBe(Infinity);
});

for (const field of ['complete', 'unresolvedCount'] as const) {
  it(`rejects inherited unwritten neighbor-search ${field} after reading the destination once`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
    const target = scalar('unsupported');
    let reads = 0;
    const status = Object.setPrototypeOf(
      {overflow: scalar('overflow')},
      {
        get [field]() {
          reads++;
          return target;
        }
      }
    );
    try {
      expect(
        () =>
          new GPUNeighborSearch({
            mode: 'radius',
            positions: createTransientView(graph, 'positions', 'float32x2', 2),
            gridSize: [2, 2],
            parameters: createTransientView(graph, 'parameters', 'float32', 12),
            output: {
              offsets: createTransientView(graph, 'offsets', 'uint32', 3),
              neighbors: createTransientView(graph, 'neighbors', 'uint32', 2),
              weights: createTransientView(graph, 'weights', 'float32', 2),
              status
            }
          })
      ).toThrow(new RegExp(`output.status.${field}.*unsupported`));
      expect(reads).toBe(1);
    } finally {
      device.destroy();
    }
  });
}
