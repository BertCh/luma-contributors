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

async function executeBudgetedLine(
  values: Float32Array,
  maximumIterations: number,
  maximumTieIterations?: number,
  includeBackLinks = true
) {
  const sceneGrid = makeGPURasterMetadata({width: values.length, height: 1});
  const rows = (format: 'float32' | 'uint32', data: Float32Array | Uint32Array): GPUCPUVector => ({
    format,
    length: data.length,
    chunks: [{values: data, length: data.length}],
    pack: () => data
  });
  const operation = new CPUCostDistanceOperation({
    grid: sceneGrid,
    maximumIterations,
    maximumTieIterations,
    friction: {id: 'friction', format: 'float32', storage: {kind: 'buffer', values: 'friction'}},
    parameters: 'parameters',
    sources: 'sources',
    costBreaks: 'costBreaks',
    costBreakCount: 1,
    output: {
      costs: 'costs',
      settledAt: 'settledAt',
      ...(includeBackLinks ? {backLinks: 'backLinks'} : {}),
      bands: 'bands',
      bandCounts: 'bandCounts',
      status: {
        iterationCount: 'iterationCount',
        converged: 'converged',
        iterationLimitReached: 'limited'
      }
    }
  });
  return operation.execute({
    inputs: {
      values: rows('float32', values),
      parameters: rows('float32', getGPUCostDistanceParameterValues({}, sceneGrid)),
      sources: rows('uint32', Uint32Array.of(0)),
      costBreaks: rows('float32', Float32Array.of(Infinity))
    }
  });
}

it('honors explicit cost budgets before certifying the two-cell fixed point on the public CPU callback', async () => {
  // Hand integration: one unit edge. Round 1 lowers cell 1 to 1; round 2 certifies no change.
  // A budget-1 implementation that computes the whole Dijkstra result must not publish it as final.
  for (const budget of [1, 2]) {
    const result = await executeBudgetedLine(Float32Array.of(1, 1), budget);
    expect(Array.from(result['costs'] as Float32Array)).toEqual(budget === 1 ? [NaN, NaN] : [0, 1]);
    expect(Array.from(result['settledAt'] as Uint32Array)).toEqual([0, 1]);
    expect(Array.from(result['backLinks'] as Uint32Array)).toEqual(
      budget === 1 ? [0xfffffffe, 0xfffffffe] : [0, 16]
    );
    expect(Array.from(result['bands'] as Uint32Array)).toEqual(
      budget === 1 ? [0xfffffffe, 0xfffffffe] : [0, 0]
    );
    expect(Array.from(result['bandCounts'] as Uint32Array)).toEqual(
      budget === 1 ? [0xfffffffe] : [2]
    );
    expect(Array.from(result['iterationCount'] as Uint32Array)).toEqual([budget]);
    expect(Array.from(result['converged'] as Uint32Array)).toEqual([budget === 2 ? 1 : 0]);
    expect(Array.from(result['limited'] as Uint32Array)).toEqual([budget === 1 ? 1 : 0]);
  }
});

it('honors explicit tie budgets while preserving certified zero-cost values and valid partial links', async () => {
  // All three costs are zero. Cost and tie fronts each improve in round 1 and certify in round 2.
  // The cost phase succeeds at budget 2; tie budget 1 cannot certify, even if all links are usable.
  for (const tieBudget of [1, 2]) {
    const result = await executeBudgetedLine(Float32Array.of(0, 0, 0), 2, tieBudget);
    expect(Array.from(result['costs'] as Float32Array)).toEqual([0, 0, 0]);
    expect(Array.from(result['settledAt'] as Uint32Array)).toEqual([0, 1, 1]);
    expect(Array.from(result['backLinks'] as Uint32Array)).toEqual([0, 16, 16]);
    expect(Array.from(result['bands'] as Uint32Array)).toEqual([0, 0, 0]);
    expect(Array.from(result['bandCounts'] as Uint32Array)).toEqual([3]);
    expect(Array.from(result['iterationCount'] as Uint32Array)).toEqual([2]);
    expect(Array.from(result['converged'] as Uint32Array)).toEqual([tieBudget === 2 ? 1 : 0]);
    expect(Array.from(result['limited'] as Uint32Array)).toEqual([tieBudget === 1 ? 1 : 0]);
  }
});

it('cost truncation preserves impassable sentinels while suppressing every passable output', async () => {
  const result = await executeBudgetedLine(Float32Array.of(1, 1, NaN), 1);
  expect(Array.from(result['costs'] as Float32Array)).toEqual([NaN, NaN, Infinity]);
  expect(Array.from(result['settledAt'] as Uint32Array)).toEqual([0, 1, 0]);
  expect(Array.from(result['backLinks'] as Uint32Array)).toEqual([
    0xfffffffe, 0xfffffffe, 0xffffffff
  ]);
  expect(Array.from(result['bands'] as Uint32Array)).toEqual([0xfffffffe, 0xfffffffe, 0xffffffff]);
  expect(Array.from(result['bandCounts'] as Uint32Array)).toEqual([0xfffffffe]);
  expect(Array.from(result['iterationCount'] as Uint32Array)).toEqual([1]);
  expect(Array.from(result['converged'] as Uint32Array)).toEqual([0]);
  expect(Array.from(result['limited'] as Uint32Array)).toEqual([1]);
});

it('marks unreached tie levels unresolved across tile boundaries and skips an unrequested tie phase', async () => {
  // Three tiles. Snapshot visibility needs three improving cost rounds plus certification.
  // A single tie round reaches only source tile 0; later tiles cannot claim source link 0.
  const result = await executeBudgetedLine(new Float32Array(33), 8, 1);
  expect(Array.from(result['costs'] as Float32Array)).toEqual(new Array(33).fill(0));
  expect(Array.from(result['backLinks'] as Uint32Array)).toEqual([
    0,
    ...new Array(15).fill(16),
    ...new Array(17).fill(0xfffffffe)
  ]);
  expect(Array.from(result['settledAt'] as Uint32Array)).toEqual([
    0,
    ...new Array(15).fill(1),
    ...new Array(16).fill(2),
    3
  ]);
  expect(Array.from(result['bands'] as Uint32Array)).toEqual(new Array(33).fill(0));
  expect(Array.from(result['bandCounts'] as Uint32Array)).toEqual([33]);
  expect(Array.from(result['iterationCount'] as Uint32Array)).toEqual([4]);
  expect(Array.from(result['converged'] as Uint32Array)).toEqual([0]);
  expect(Array.from(result['limited'] as Uint32Array)).toEqual([1]);

  const costsOnly = await executeBudgetedLine(Float32Array.of(0, 0, 0), 2, 1, false);
  expect(Array.from(costsOnly['costs'] as Float32Array)).toEqual([0, 0, 0]);
  expect(Array.from(costsOnly['converged'] as Uint32Array)).toEqual([1]);
  expect(Array.from(costsOnly['limited'] as Uint32Array)).toEqual([0]);
});

it('the pure CPU form uses bounded tiled rounds and the cumulative directional schedule', () => {
  const sceneGrid = makeGPURasterMetadata({width: 128, height: 1});
  for (const budget of [3, 4]) {
    const result = computeCostDistanceOnCPU(
      {
        friction: {...friction, values: new Float32Array(128).fill(1)},
        parameters: getGPUCostDistanceParameterValues({}, sceneGrid),
        sources: Uint32Array.of(0)
      },
      {grid: sceneGrid, maximumIterations: budget, sweepAfterIteration: 1}
    );
    expect(Array.from(result.costs)).toEqual(
      budget === 3 ? new Array(128).fill(NaN) : Array.from({length: 128}, (_, cell) => cell)
    );
    expect(Array.from(result.iterationCount)).toEqual([budget]);
    expect(Array.from(result.converged)).toEqual([budget === 4 ? 1 : 0]);
    expect(Array.from(result.iterationLimitReached)).toEqual([budget === 3 ? 1 : 0]);
    expect(result.settledAt[127]).toBe(3);
  }
});

it('bounded CPU tiles match the closed-form octile surface across both raster axes and tail tiles', () => {
  const width = 17;
  const sceneGrid = makeGPURasterMetadata({width, height: width});
  const input = {
    friction: {...friction, values: new Float32Array(width * width).fill(1)},
    parameters: getGPUCostDistanceParameterValues({}, sceneGrid),
    sources: Uint32Array.of(0)
  };
  // Unit D8 geometry: min(column,row) diagonal moves, then axial moves. Two improving
  // snapshot rounds reach all four tiles; the third unchanged round certifies them.
  const limited = computeCostDistanceOnCPU(input, {grid: sceneGrid, maximumIterations: 2});
  expect(Array.from(limited.costs)).toEqual(new Array(width * width).fill(NaN));
  expect(Array.from(limited.iterationCount)).toEqual([2]);
  expect(Array.from(limited.converged)).toEqual([0]);
  const complete = computeCostDistanceOnCPU(input, {grid: sceneGrid, maximumIterations: 3});
  const expected = Array.from({length: width * width}, (_, cell) => {
    const column = cell % width;
    const row = Math.floor(cell / width);
    return Math.fround(Math.max(column, row) + (Math.SQRT2 - 1) * Math.min(column, row));
  });
  expect(Array.from(complete.costs)).toEqual(expected);
  expect(Array.from(complete.iterationCount)).toEqual([3]);
  expect(Array.from(complete.converged)).toEqual([1]);
  expect(Array.from(complete.iterationLimitReached)).toEqual([0]);
  expect([
    complete.settledAt[0],
    complete.settledAt[15],
    complete.settledAt[16],
    complete.settledAt[width * width - 1]
  ]).toEqual([0, 1, 2, 2]);
});

it('rejects explicit cost and tie budgets outside the shared round limit', () => {
  for (const name of ['maximumIterations', 'maximumTieIterations'] as const) {
    for (const value of [0, 1025, NaN]) {
      expect(() =>
        computeCostDistanceOnCPU(
          {friction, parameters, sources: Uint32Array.of(0)},
          {
            grid,
            [name]: value
          }
        )
      ).toThrow(
        new RegExp(`^computeCostDistanceOnCPU ${name} must be an integer in \\[1, 1024\\]`)
      );
    }
  }
});
