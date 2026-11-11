// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {
  CPUCostDistanceOperation,
  computeCostDistanceOnCPU
} from '../../../src/raster-analysis/cost-distance/cpu-cost-distance-operation';
import {
  GPUCostDistanceOperation,
  type CostDistanceOperationProps
} from '../../../src/raster-analysis/cost-distance/gpu-cost-distance-operation';
import {getGPUCostDistanceParameterValues} from '../../../src/raster-analysis/cost-distance/gpu-cost-distance';
import type {GPUCPUVector} from '@luma.gl/graph/gpu-core';

const grid = makeGPURasterMetadata({width: 4, height: 1});
const parameters = getGPUCostDistanceParameterValues({}, grid);
const friction = {id: 'friction', format: 'float32' as const, values: Float32Array.of(2, 0, 0, 4)};

it('matches hand-integrated line costs, source competition, plateau links and inclusive bands', () => {
  const result = computeCostDistanceOnCPU(
    {
      friction,
      parameters,
      sources: Uint32Array.of(0, 3),
      sourceCosts: Float32Array.of(0, 0.5),
      costBreaks: Float32Array.of(0.5, 1, 2)
    },
    {grid}
  );
  expect(Array.from(result.costs)).toEqual([0, 1, 1, 0.5]);
  expect(Array.from(result.backLinks)).toEqual([0, 16, 16, 0]);
  expect(Array.from(result.bands)).toEqual([0, 1, 1, 0]);
  expect(Array.from(result.bandCounts)).toEqual([2, 2, 0]);
  expect(Array.from(result.converged)).toEqual([1]);
  expect(Array.from(result.iterationLimitReached)).toEqual([0]);
});

it('handles rounded nodata, masks, calibration and ignored source rows', () => {
  const sentinel = 0.1;
  const result = computeCostDistanceOnCPU(
    {
      friction: {...friction, values: Float32Array.of(2, sentinel, 0, 4), noDataValue: sentinel},
      parameters,
      sources: Uint32Array.of(0, 3),
      sourceCount: Uint32Array.of(1)
    },
    {grid}
  );
  expect(Array.from(result.costs)).toEqual([0, Infinity, Infinity, Infinity]);
  expect(Array.from(result.backLinks)).toEqual([0, 0xffffffff, 0xffffffff, 0xffffffff]);
  expect(() =>
    computeCostDistanceOnCPU(
      {
        friction: {...friction, validity: new Uint32Array(3)},
        parameters,
        sources: Uint32Array.of(0)
      },
      {grid}
    )
  ).toThrow(/validity/);
  expect(() =>
    computeCostDistanceOnCPU(
      {friction, parameters, sources: Uint32Array.of(0), sourceCosts: new Float32Array(2)},
      {grid}
    )
  ).toThrow(/sourceCosts/);
});

it('validates CPU props and GPU graph output declarations at construction', () => {
  const props: CostDistanceOperationProps = {
    id: 'cost-test',
    grid,
    friction: {id: 'friction', format: 'float32', storage: {kind: 'buffer', values: 'friction'}},
    parameters: 'parameters',
    sources: 'sources',

    output: {costs: 'costs'}
  };
  expect(() => new CPUCostDistanceOperation(props)).not.toThrow();
  for (const Operation of [GPUCostDistanceOperation, CPUCostDistanceOperation])
    expect(() => new Operation({...props, output: {costs: 'friction'}})).toThrow(/cost-test/);
});

it('matches exhaustive simple-path enumeration on a weighted D8 graph with a barrier', () => {
  const sceneGrid = makeGPURasterMetadata({width: 3, height: 2, cellSize: [2, 3]});
  const values = Float32Array.of(1, 4, 2, 3, -1, 5);
  const seeds = [
    {cell: 0, cost: 0.75},
    {cell: 5, cost: 1.25}
  ];
  // Independent oracle: enumerate every simple path. With nonnegative edges a shortest path
  // always has a simple representative; no heap or relaxation is used here.
  const expected = new Float64Array(6).fill(Infinity);
  const visit = (cell: number, cost: number, visited: Set<number>) => {
    expected[cell] = Math.min(expected[cell], cost);
    for (let target = 0; target < 6; target++) {
      if (visited.has(target) || values[target] < 0) continue;
      const columnDifference = Math.abs((target % 3) - (cell % 3));
      const rowDifference = Math.abs(Math.floor(target / 3) - Math.floor(cell / 3));
      if (columnDifference > 1 || rowDifference > 1) continue;
      const distance = Math.hypot(columnDifference * 2, rowDifference * 3);
      const nextCost = cost + (distance * (values[cell] + values[target])) / 2;
      visit(target, nextCost, new Set([...visited, target]));
    }
  };
  for (const source of seeds) visit(source.cell, source.cost, new Set([source.cell]));
  const result = computeCostDistanceOnCPU(
    {
      friction: {...friction, values},
      parameters: getGPUCostDistanceParameterValues({}, sceneGrid),
      sources: Uint32Array.from(seeds, source => source.cell),
      sourceCosts: Float32Array.from(seeds, source => source.cost)
    },
    {grid: sceneGrid}
  );
  expect(Array.from(result.costs)).toEqual(Array.from(Float32Array.from(expected)));
  // Every back-link follows a tight edge and terminates at one of the competing sources.
  const columnSteps = [1, 1, 0, -1, -1, -1, 0, 1];
  const rowSteps = [0, 1, 1, 1, 0, -1, -1, -1];
  for (let start = 0; start < 6; start++) {
    if (!Number.isFinite(expected[start])) {
      expect(result.backLinks[start]).toBe(0xffffffff);
      continue;
    }
    let cell = start;
    const visited = new Set<number>();
    while (result.backLinks[cell] !== 0) {
      expect(visited.has(cell)).toBe(false);
      visited.add(cell);
      const direction = Math.log2(result.backLinks[cell]);
      const neighbor = cell + columnSteps[direction] + rowSteps[direction] * 3;
      const edge =
        (Math.hypot(columnSteps[direction] * 2, rowSteps[direction] * 3) *
          (values[cell] + values[neighbor])) /
        2;
      expect(expected[neighbor] + edge).toBeCloseTo(expected[cell], 12);
      cell = neighbor;
    }
    expect(seeds.some(source => source.cell === cell && source.cost === expected[cell])).toBe(true);
  }
});

it('applies both calibration stages, source masks and inclusive cost limits', () => {
  const result = computeCostDistanceOnCPU(
    {
      friction: {...friction, values: Float32Array.of(1, 2, 3, 4), scale: 2, offset: 1},
      frictionParameters: Float32Array.of(0.5, -0.5),
      parameters: getGPUCostDistanceParameterValues({costLimit: 4}, grid),
      sourceMask: Uint32Array.of(1, 0, 0, 0)
    },
    {grid}
  );
  expect(Array.from(result.costs)).toEqual([0, 1.5, 4, Infinity]);
  expect(Array.from(result.backLinks)).toEqual([0, 16, 16, 0xffffffff]);
});

it('uses equatorial ground metres on geographic and Web Mercator rasters', () => {
  const geographicGrid = makeGPURasterMetadata({width: 3, height: 1, origin: [0, 0.5]});
  const geographic = computeCostDistanceOnCPU(
    {
      friction: {...friction, values: Float32Array.of(1, 1, 1)},
      parameters: getGPUCostDistanceParameterValues({}, geographicGrid),
      sources: Uint32Array.of(0)
    },
    {grid: geographicGrid, cellSizeMode: 'geographic'}
  );
  // Along the equator the ellipsoid's parallel is a circle of radius a = 6378137 m.
  const degreeLength = (6378137 * Math.PI) / 180;
  expect(Array.from(geographic.costs)).toEqual([
    0,
    Math.fround(degreeLength),
    Math.fround(2 * degreeLength)
  ]);
  const mercator = computeCostDistanceOnCPU(
    {
      friction: {...friction, values: Float32Array.of(1, 1, 1)},
      parameters: getGPUCostDistanceParameterValues({}, geographicGrid),
      sources: Uint32Array.of(0)
    },
    {grid: geographicGrid, cellSizeMode: 'web-mercator'}
  );
  // East-west EPSG:3857 scale is one at y=0.
  expect(Array.from(mercator.costs)).toEqual([0, 1, 2]);
});

it('does not publish reachable metadata when the final float32 cost overflows', () => {
  const sceneGrid = makeGPURasterMetadata({width: 5, height: 1});
  const value = Math.fround(1e38);
  const result = computeCostDistanceOnCPU(
    {
      friction: {...friction, values: new Float32Array(5).fill(value)},
      parameters: getGPUCostDistanceParameterValues({}, sceneGrid),
      sources: Uint32Array.of(0),
      costBreaks: Float32Array.of(Infinity)
    },
    {grid: sceneGrid}
  );
  // Unit-length edges each cost value: the fourth accumulated edge exceeds float32 range.
  expect(Array.from(result.costs)).toEqual([
    0,
    value,
    Math.fround(2 * value),
    Math.fround(3 * value),
    Infinity
  ]);
  expect(Array.from(result.backLinks)).toEqual([0, 16, 16, 16, 0xffffffff]);
  expect(Array.from(result.bands)).toEqual([0, 0, 0, 0, 0xffffffff]);
  expect(Array.from(result.bandCounts)).toEqual([4]);
  expect(Array.from(result.settledAt)).toEqual([0, 1, 2, 3, 0]);
});

it('derives the source count from rows, including callers with a stale sourceCapacity field', async () => {
  const sceneGrid = makeGPURasterMetadata({width: 2, height: 1});
  const rows = (
    format: 'float32' | 'uint32',
    values: Float32Array | Uint32Array
  ): GPUCPUVector => ({
    format,
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  for (const legacy of [{}, {sourceCapacity: 1}]) {
    const props: CostDistanceOperationProps = {
      ...legacy,
      grid: sceneGrid,
      friction: {id: 'friction', format: 'float32', storage: {kind: 'buffer', values: 'friction'}},
      parameters: 'parameters',
      sources: 'sources',
      output: {costs: 'costs'}
    };
    const operation = new CPUCostDistanceOperation(props);
    const result = await operation.execute({
      inputs: {
        values: rows('float32', Float32Array.of(1, 1)),
        parameters: rows('float32', getGPUCostDistanceParameterValues({}, sceneGrid)),
        sources: rows('uint32', Uint32Array.of(0, 1))
      }
    });
    expect(Array.from(result['costs'] as Float32Array)).toEqual([0, 0]);
  }
});

it('reports synchronous hop diagnostics on a two-cell line', () => {
  const sceneGrid = makeGPURasterMetadata({width: 2, height: 1});
  const result = computeCostDistanceOnCPU(
    {
      friction: {...friction, values: Float32Array.of(1, 1)},
      parameters: getGPUCostDistanceParameterValues({}, sceneGrid),
      sources: Uint32Array.of(0)
    },
    {grid: sceneGrid}
  );
  expect(Array.from(result.costs)).toEqual([0, 1]);
  expect(Array.from(result.settledAt)).toEqual([0, 1]);
  expect(Array.from(result.iterationCount)).toEqual([2]);
});
