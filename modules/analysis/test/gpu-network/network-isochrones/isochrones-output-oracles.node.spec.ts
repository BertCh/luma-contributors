// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUNetworkIsochrones} from '../../../src/gpu-network/network-isochrones/index';
import type {GPUNetworkIsochronesProps} from '../../../src/gpu-network/network-isochrones/index';
import {addDriveTimeCatchmentRecipe} from '../../../src/gpu-network/recipes/drive-time-catchment-recipe';
import type {GPUCellTable} from '../../../src/gpu-spatial-analysis/cell-aggregation/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {splatIsochroneRasterOnCPU} from './network-isochrones-oracle';
import {RASTER_SQUARE} from './isochrones-output-oracle-fixtures';
import {expectOracleSquareRings, RASTER_ORACLE_SELECTIONS} from './isochrones-output-oracle-utils';

for (const mode of ['min', 'max'] as const) {
  for (const [label, cost, weight, expected] of [
    ['negative cost', -1, 0, 9],
    ['infinite weight', 3, Infinity, 9],
    ['negative zero', -0, 0, 0]
  ] as const) {
    it(`R3 CPU ${mode} splat handles ${label}`, () => {
      const actual = splatIsochroneRasterOnCPU(
        {
          offsets: Uint32Array.of(0, 1),
          neighbors: Uint32Array.of(0),
          weights: Float32Array.of(weight)
        },
        Float32Array.of(1, 1),
        Float32Array.of(cost),
        {
          width: 2,
          height: 2,
          extent: [0, 0, 2, 2],
          bufferRadius: 2,
          walkCostPerUnit: 0,
          mode,
          maximumBufferPixels: 6,
          maximumSamplesPerEdge: 64,
          unreachedCost: 9
        }
      );
      expect(Array.from(actual)).toEqual([expected, expected, expected, expected]);
    });
  }
}

it.each([
  'min',
  'max'
] as const)('R10 CPU %s splat rejects costs rounded to the shader cutoff', mode => {
  const unitInLastPlace = 2 ** 104;
  const shaderCutoff = 14791142 * unitInLastPlace;
  const sourceCostBelowCutoff = shaderCutoff - unitInLastPlace;
  // The pixel center (1, 1) is one unit from the self-loop at (0, 1). Adding 1/4 of
  // a step rounds back to the source cost. Adding 3/4 of a step rounds to the cutoff.
  // In the last case, packing the walking cost gives one step below the cutoff.
  // The half-step source cost then lands on a tie and rounds to the even cutoff.
  for (const [sourceCost, walkingCost, expected] of [
    [sourceCostBelowCutoff, unitInLastPlace / 4, sourceCostBelowCutoff],
    [sourceCostBelowCutoff, (3 * unitInLastPlace) / 4, 9],
    [2 ** 103, shaderCutoff - 1.25 * unitInLastPlace, 9]
  ] as const) {
    const actual = splatIsochroneRasterOnCPU(
      {offsets: Uint32Array.of(0, 1), neighbors: Uint32Array.of(0), weights: Float32Array.of(0)},
      Float32Array.of(0, 1),
      Float32Array.of(sourceCost),
      {
        width: 1,
        height: 1,
        extent: [0, 0, 2, 2],
        bufferRadius: 2,
        walkCostPerUnit: walkingCost,
        mode,
        maximumBufferPixels: 6,
        maximumSamplesPerEdge: 2,
        unreachedCost: 9
      }
    );
    expect(Array.from(actual)).toEqual([expected]);
  }
});

it('R5 square oracle rejects a rotated ring with stale segment vertex indices', () => {
  const shifted = [1.5, 0.5, 1.5, 1.5, 0.5, 1.5, 0.5, 0.5, 1.5, 0.5];
  const result = {
    ringOffsets: [0, 5, 5],
    ringPositions: shifted,
    ringCount: [1],
    ringOverflow: [0],
    segmentVertices: [0, 1, 2, 3]
  };
  expect(() =>
    expectOracleSquareRings(result, RASTER_SQUARE.corners, 1, 1, [0, 1, 2, 3], RASTER_SQUARE.edges)
  ).toThrow();
  const correct = {
    ...result,
    ringPositions: [...RASTER_SQUARE.corners, ...RASTER_SQUARE.corners.slice(0, 2)]
  };
  expect(() =>
    expectOracleSquareRings(correct, RASTER_SQUARE.corners, 1, 1, [0, 1, 2, 3], RASTER_SQUARE.edges)
  ).not.toThrow();
});

it('R7 raster selectors have unique names for triangle and ring counters', () => {
  const selectors = RASTER_ORACLE_SELECTIONS;
  expect(new Set(selectors).size).toBe(selectors.length);
  expect(selectors).toContain('triangleRequiredCount');
  expect(selectors).toContain('ringRequiredCount');
});

it('R2 CPU splat independently pins the rectangular raster', () => {
  const values = splatIsochroneRasterOnCPU(
    {
      offsets: Uint32Array.of(0, 1, 2),
      neighbors: Uint32Array.of(0, 1),
      weights: Float32Array.of(0, 0)
    },
    Float32Array.of(2, 0.5, 7.75, 1.75),
    Float32Array.of(3, 1),
    {
      width: 2,
      height: 2,
      extent: [0, 0, 8, 2],
      bufferRadius: 0,
      walkCostPerUnit: 0,
      mode: 'min',
      maximumBufferPixels: 6,
      maximumSamplesPerEdge: 64,
      unreachedCost: 9
    }
  );
  // Half the 4-by-1 pixel diagonal is sqrt(17)/2. Only the top-right
  // sample is within that radius of (7.75,1.75); the left samples belong to (2,.5).
  expect(Array.from(values)).toEqual([3, 9, 3, 1]);
});

it('R6 CPU splat independently pins the raster with an excluded center', () => {
  const values = splatIsochroneRasterOnCPU(
    {
      offsets: Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
      neighbors: Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7, 8),
      weights: new Float32Array(9)
    },
    Float32Array.of(
      0.5,
      0.5,
      1.5,
      0.5,
      2.5,
      0.5,
      0.5,
      1.5,
      1.5,
      1.5,
      2.5,
      1.5,
      0.5,
      2.5,
      1.5,
      2.5,
      2.5,
      2.5
    ),
    Float32Array.of(3, 3, 3, 3, 5, 3, 3, 3, 3),
    {
      width: 3,
      height: 3,
      extent: [0, 0, 3, 3],
      bufferRadius: 0,
      walkCostPerUnit: 0,
      mode: 'min',
      maximumBufferPixels: 6,
      maximumSamplesPerEdge: 64,
      unreachedCost: 9
    }
  );
  // The effective radius sqrt(.5) is below unit sample spacing; each loop owns only its center.
  expect(Array.from(values)).toEqual([3, 3, 3, 3, 5, 3, 3, 3, 3]);
});

it.each(['min', 'max'] as const)('isochrone CPU splat oracle: constant square in %s mode', mode => {
  // A zero-weight self-loop, zero walking cost and radius 2 cover every sample center.
  // This independently pins the CPU mirror used by the existing GPU comparison spec.
  const values = splatIsochroneRasterOnCPU(
    {offsets: Uint32Array.of(0, 1), neighbors: Uint32Array.of(0), weights: Float32Array.of(0)},
    Float32Array.of(1, 1),
    Float32Array.of(3),
    {
      width: 2,
      height: 2,
      extent: [0, 0, 2, 2],
      bufferRadius: 2,
      walkCostPerUnit: 0,
      mode,
      maximumBufferPixels: 6,
      maximumSamplesPerEdge: 64,
      unreachedCost: 100
    }
  );
  expect(Array.from(values)).toEqual(RASTER_SQUARE.values);
});

it('isochrone CPU splat oracle: two independently covered facility patches', () => {
  const values = splatIsochroneRasterOnCPU(
    {
      offsets: Uint32Array.of(0, 1, 2),
      neighbors: Uint32Array.of(0, 1),
      weights: Float32Array.of(0, 0)
    },
    Float32Array.of(1, 1, 3, 1),
    Float32Array.of(3, 3),
    {
      width: 4,
      height: 2,
      extent: [0, 0, 4, 2],
      bufferRadius: 0.75,
      walkCostPerUnit: 0,
      mode: 'min',
      maximumBufferPixels: 6,
      maximumSamplesPerEdge: 64,
      unreachedCost: 100
    }
  );
  expect(Array.from(values)).toEqual([3, 3, 3, 3, 3, 3, 3, 3]);
});

for (const aggregate of ['sums', 'sumValues', 'minimums', 'maximums'] as const) {
  it(`isochrone census correction: table.${aggregate} requires a value column unavailable to this composition`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'uint32' | 'uint32x2' | 'float32' | 'float32x2' | 'float32x4'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `refusal-${serial++}`, format, length);
    const table: GPUCellTable = {
      cells: view('uint32x2', 1),
      counts: view('uint32', 1),
      status: {count: view('uint32', 1), overflow: view('uint32', 1)}
    };
    if (aggregate === 'sums') table.sums = view('uint32x2', 1);
    else table[aggregate] = view('float32', 1);
    try {
      const props: GPUNetworkIsochronesProps = {
        topology: {
          offsets: view('uint32', 2),
          neighbors: view('uint32', 1),
          weights: view('float32', 1),
          nodePositions: view('float32x2', 1)
        },
        costs: view('float32', 1),
        costBreaks: view('float32', 1),
        parameters: view('float32', 12),
        cellOutline: {
          family: 'quadbin',
          resolution: 2,
          table,
          output: {
            rows: view('uint32', 4),
            cells: view('uint32x2', 4),
            edgeIndices: view('uint32', 4),
            endpoints: view('float32x4', 4),
            status: {count: view('uint32', 1), overflow: view('uint32', 1)}
          }
        }
      };
      expect(() => new GPUNetworkIsochrones(props)).toThrow(
        `network-isochrones cellOutline.table.${aggregate} has no values input; use GPUCellAggregation with values for reductions`
      );
      // The recipe forwards this same cell-table option to the existing contributor.
      // A wrapper that silently drops the unsupported column must fail this assertion.
      expect(() =>
        addDriveTimeCatchmentRecipe(graph, {
          topology: props.topology,
          facilities: view('float32x2', 1),
          demand: view('float32x2', 1),
          costBreaks: props.costBreaks,
          isochrones: {
            costBreaks: props.costBreaks,
            parameters: props.parameters,
            cellOutline: props.cellOutline
          }
        })
      ).toThrow(
        `drive-time-catchment-isochrones cellOutline.table.${aggregate} has no values input; use GPUCellAggregation with values for reductions`
      );
    } finally {
      device.destroy();
    }
  });
}
