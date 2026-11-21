// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {addDriveTimeCatchmentRecipe} from '../../../src/gpu-network/recipes/drive-time-catchment-recipe';
import {getGPUNetworkIsochronesParameterValues} from '../../../src/gpu-network/network-isochrones/index';
import {RecipeTestFixture} from '../../gpu-spatial-analysis/recipes/recipe-harness';
import {requireTestDevice} from '../../utils/require-test-device';

// Hand-computed drive times on networks of two and three nodes. They pin two things the two-end
// minimum `min(cost[source] + f w, cost[target] + (1 - f) w)` gets wrong: demand on the same road
// as its facility drives along the road, not out to a node and back; and an edge that exists in
// one direction only is driven in that direction only.

const NONE = 0xffffffff;

type SmallNetwork = {
  /** x, y per node. */
  nodePositions: number[];
  /** CSR row offsets, one more than the node count. */
  offsets: number[];
  /** Target node per directed edge row. */
  neighbors: number[];
  /** Travel cost per directed edge row. */
  weights: number[];
};

type DriveTimeResult = {
  demandTimes: number[];
  demandBands: number[];
  nodeCosts: number[];
  startNodes: number[];
  startCosts: number[];
  converged: number;
  rasterValues?: number[];
};

async function runDriveTimeCatchment(
  device: Device,
  network: SmallNetwork,
  facilities: number[],
  demand: number[],
  costBreaks: number[],
  rasterMode?: 'unlabelled' | 'labelled'
): Promise<DriveTimeResult> {
  const nodeCount = network.offsets.length - 1;
  const facilityCount = facilities.length / 2;
  const demandCount = demand.length / 2;
  const fixture = new RecipeTestFixture(device, 'drive-time-direct-travel-test');
  try {
    const outputs = {
      nodeCosts: fixture.output('node-costs', 'float32', nodeCount),
      demandTimes: fixture.output('demand-times', 'float32', demandCount),
      demandBands: fixture.output('demand-bands', 'uint32', demandCount),
      startNodes: fixture.output('start-nodes', 'uint32', 2 * facilityCount),
      startCosts: fixture.output('start-costs', 'float32', 2 * facilityCount),
      converged: fixture.output('converged', 'uint32', 1)
    };
    const rasterValues = rasterMode
      ? fixture.output('raster-values', 'float32', 48 * 8)
      : undefined;
    addDriveTimeCatchmentRecipe(fixture.graph, {
      topology: {
        offsets: fixture.input(
          'offsets',
          Uint32Array.from(network.offsets),
          'uint32',
          nodeCount + 1
        ),
        neighbors: fixture.input(
          'neighbors',
          Uint32Array.from(network.neighbors),
          'uint32',
          network.neighbors.length
        ),
        weights: fixture.input(
          'weights',
          Float32Array.from(network.weights),
          'float32',
          network.weights.length
        ),
        nodePositions: fixture.input(
          'positions',
          Float32Array.from(network.nodePositions),
          'float32x2',
          nodeCount
        )
      },
      facilities: fixture.input(
        'facilities',
        Float32Array.from(facilities),
        'float32x2',
        facilityCount
      ),
      demand: fixture.input('demand', Float32Array.from(demand), 'float32x2', demandCount),
      maximumIterations: 8,
      costBreaks: fixture.input(
        'band-breaks',
        Float32Array.from(costBreaks),
        'float32',
        costBreaks.length
      ),
      output: {
        nodeCosts: outputs.nodeCosts.view,
        demandTimes: outputs.demandTimes.view,
        demandBands: outputs.demandBands.view,
        status: {serviceAreaConverged: outputs.converged.view}
      },
      scratch: {startNodes: outputs.startNodes.view, startCosts: outputs.startCosts.view},
      ...(rasterValues
        ? {
            isochrones: {
              costBreaks: fixture.input(
                'isochrone-breaks',
                Float32Array.from(costBreaks),
                'float32',
                costBreaks.length
              ),
              parameters: fixture.parameters(
                'isochrone-parameters',
                'float32',
                getGPUNetworkIsochronesParameterValues({
                  breakCount: costBreaks.length,
                  extent: [-1, -1, 11, 1],
                  bufferRadius: 0.4
                })
              ),
              raster: {
                width: 48,
                height: 8,
                output: {
                  values: rasterValues.view,
                  bandClasses: fixture.output('raster-band-classes', 'uint32', 48 * 8).view,
                  ...(rasterMode === 'labelled'
                    ? {pixelFacilities: fixture.output('pixel-facilities', 'uint32', 48 * 8).view}
                    : {})
                }
              }
            }
          }
        : {})
    });
    fixture.run();
    return {
      demandTimes: await fixture.readFloat32(outputs.demandTimes, demandCount),
      demandBands: await fixture.readUint32(outputs.demandBands, demandCount),
      nodeCosts: await fixture.readFloat32(outputs.nodeCosts, nodeCount),
      startNodes: await fixture.readUint32(outputs.startNodes, 2 * facilityCount),
      startCosts: await fixture.readFloat32(outputs.startCosts, 2 * facilityCount),
      converged: (await fixture.readUint32(outputs.converged, 1))[0],
      ...(rasterValues ? {rasterValues: await fixture.readFloat32(rasterValues, 48 * 8)} : {})
    };
  } finally {
    fixture.destroy();
  }
}

function expectValuesClose(actual: ArrayLike<number>, expected: number[], label: string): void {
  expect(actual.length, label).toBe(expected.length);
  for (let row = 0; row < expected.length; row++) {
    expect(Math.abs(actual[row] - expected[row]), `${label} ${row}`).toBeLessThanOrEqual(1e-3);
  }
}

it('drive-time source rasters preserve routing costs and convergence in the same submission', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const rasterMode of [undefined, 'unlabelled', 'labelled'] as const) {
    const result = await runDriveTimeCatchment(
      device,
      {
        nodePositions: [0, 0, 10, 0],
        offsets: [0, 1, 2],
        neighbors: [1, 0],
        weights: [10, 30]
      },
      [5, 0],
      [2, 0, 5, 0, 8, 0],
      [1, 10, 30],
      rasterMode
    );
    // Closed form: the facility drives 5 toward B and 15 toward A; the demand rows cost
    // 0.3 * 30, zero and 0.3 * 10. Enabling raster passes must preserve these outputs.
    if (
      result.converged !== 1 ||
      result.nodeCosts.some(
        (cost, row) => !Number.isFinite(cost) || Math.abs(cost - [15, 5][row]) > 1e-3
      ) ||
      result.demandTimes.some(
        (cost, row) => !Number.isFinite(cost) || Math.abs(cost - [9, 0, 3][row]) > 1e-3
      ) ||
      result.demandBands.some((band, row) => band !== [1, 0, 1][row]) ||
      (result.rasterValues && !result.rasterValues.some(cost => cost >= 0 && cost <= 1))
    ) {
      throw new Error(
        `drive-time routing/raster submission failed (${rasterMode ?? 'no raster'}): ${JSON.stringify(result)}`
      );
    }
  }
});

it('addDriveTimeCatchmentRecipe drives along the road when demand shares it with the facility', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Two-way roads A(0,0) - B(10,0) of cost 100 and B - C(10,5) of cost 30. The facility is on
  // A-B at x = 4, so it is 40 from A and 60 from B; C is 90.
  const result = await runDriveTimeCatchment(
    device,
    {
      nodePositions: [0, 0, 10, 0, 10, 5],
      offsets: [0, 1, 3, 4],
      neighbors: [1, 0, 2, 1],
      weights: [100, 100, 30, 30]
    },
    [4, 0],
    // At the facility; 2 past it; 3 before it; halfway along B-C; beside the facility, off the road.
    [4, 0, 6, 0, 1, 0, 10, 2.5, 4, 1],
    [25, 1000]
  );
  expectValuesClose(result.nodeCosts, [40, 60, 90], 'node cost');
  // 0.2 and 0.3 of the road at cost 100; B-C is reached through B: 60 + 15. The two-end minimum
  // gave 80, 100 and 50 for the first three and 80 for the last.
  expectValuesClose(result.demandTimes, [0, 20, 30, 75, 0], 'demand time');
  expect([...result.demandBands]).toEqual([0, 0, 1, 1, 0]);
});

it('addDriveTimeCatchmentRecipe drives a one-way edge in its direction only', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // One-way loop A(0,0) -> B(10,0) cost 10, B -> C(5,8) cost 40, C -> A cost 60. The facility is
  // on A->B at x = 9: it can only leave toward B (cost 1), so C is 41 and A is 101.
  const result = await runDriveTimeCatchment(
    device,
    {
      nodePositions: [0, 0, 10, 0, 5, 8],
      offsets: [0, 1, 2, 3],
      neighbors: [1, 2, 0],
      weights: [10, 40, 60]
    },
    [9, 0],
    // Behind the facility on its edge (all the way round: 1 + 100 + 1); ahead of it (0.05 of 10);
    // at it; halfway along B->C (1 + 20); halfway along C->A (41 + 30, not backward from A).
    [1, 0, 9.5, 0, 9, 0, 7.5, 4, 2.5, 4],
    [1000]
  );
  expect([...result.startNodes]).toEqual([NONE, 1]);
  expectValuesClose(result.startCosts, [-1, 1], 'start cost');
  expectValuesClose(result.nodeCosts, [101, 1, 41], 'node cost');
  // The two-end minimum gave 10, 1.5, 2, 21 and 39.
  expectValuesClose(result.demandTimes, [102, 0.5, 0, 21, 71], 'demand time');
});

it('addDriveTimeCatchmentRecipe uses the cost of the direction driven on an asymmetric road', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // A(0,0) -> B(10,0) costs 10 and B -> A costs 30. The facility is at the middle: reaching A
  // drives the B -> A direction (0.5 * 30), reaching B the A -> B direction (0.5 * 10).
  const result = await runDriveTimeCatchment(
    device,
    {
      nodePositions: [0, 0, 10, 0],
      offsets: [0, 1, 2],
      neighbors: [1, 0],
      weights: [10, 30]
    },
    [5, 0],
    // 3 behind the facility toward A (0.3 * 30); 3 ahead toward B (0.3 * 10).
    [2, 0, 8, 0],
    [1000]
  );
  expectValuesClose(result.nodeCosts, [15, 5], 'node cost');
  expectValuesClose(result.demandTimes, [9, 3], 'demand time');
});

it('addDriveTimeCatchmentRecipe puts a drive time exactly on a break inside that band', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // A two-way road A(0,0) - B(8,0) of cost 16 with the facility at its middle. Every fraction is
  // a power of two, so the times are exact in float32: 0 at the facility, 4 a quarter of the road
  // to either side, 6 at x = 7. With breaks at 4 and 100 a time of exactly 4 is within the first
  // band, the rule of the reachability bands and the isochrone bands the recipe also draws.
  const result = await runDriveTimeCatchment(
    device,
    {
      nodePositions: [0, 0, 8, 0],
      offsets: [0, 1, 2],
      neighbors: [1, 0],
      weights: [16, 16]
    },
    [4, 0],
    [4, 0, 2, 0, 6, 0, 7, 0],
    [4, 100]
  );
  expect([...result.demandTimes]).toEqual([0, 4, 4, 6]);
  expect([...result.demandBands]).toEqual([0, 0, 0, 1]);
});

it('addDriveTimeCatchmentRecipe finds every facility on the road of a demand point', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Two-way roads A(0,0) - B(8,0) and B - C(8,8), cost 16 each. Three facilities, listed out of
  // road order: one on B-C at y = 4, two on A-B at x = 6 and x = 1. Every fraction is a multiple
  // of 1/8, so the times are exact: a demand point between the two facilities of A-B takes the
  // nearer one (x = 3: 2 of 8 from x = 1; x = 5: 1 of 8 from x = 6), and on B-C the facility of
  // that road competes with the way through B (cost 4 from the facility at x = 6).
  const result = await runDriveTimeCatchment(
    device,
    {
      nodePositions: [0, 0, 8, 0, 8, 8],
      offsets: [0, 1, 3, 4],
      neighbors: [1, 0, 2, 1],
      weights: [16, 16, 16, 16]
    },
    [8, 4, 6, 0, 1, 0],
    [3, 0, 5, 0, 7, 0, 8, 1, 8, 6],
    [100]
  );
  expect(result.nodeCosts).toEqual([2, 4, 8]);
  // x = 3: 4; x = 5: 2; x = 7: 2; y = 1: 4 + 2 through B, or 6 along the road; y = 6: 4.
  expect(result.demandTimes).toEqual([4, 2, 2, 6, 4]);
});

it('drive-time junctions preserve coincident endpoint access on one-way roads', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // The facility coincides with a source that has two outgoing one-way branches. It can
  // start down either branch without first driving an unavailable reverse edge.
  const outward = await runDriveTimeCatchment(
    device,
    {
      nodePositions: [0, 0, 10, 0, 0, 10],
      offsets: [0, 2, 2, 2],
      neighbors: [1, 2],
      weights: [10, 10]
    },
    [0, 0],
    [0, 5, 0, 0],
    [5, 10]
  );
  expectValuesClose(outward.nodeCosts, [0, 10, 10], 'outward junction node cost');
  expectValuesClose(outward.demandTimes, [5, 0], 'outward junction demand time');
  expect(outward.demandBands).toEqual([0, 0]);
  // A demand at the target of an unreachable incoming road is still reached through the other
  // incoming road. It cannot drive backwards from the junction along the unreachable branch.
  const inward = await runDriveTimeCatchment(
    device,
    {
      nodePositions: [10, 0, 0, 0, 0, 10],
      offsets: [0, 1, 1, 2],
      neighbors: [1, 1],
      weights: [10, 10]
    },
    [0, 5],
    [0, 0, 5, 0],
    [5, 10]
  );
  expectValuesClose([inward.demandTimes[0]], [5], 'inward junction demand time');
  if (inward.demandTimes[1] !== Infinity)
    throw new Error(
      `drive-time inward junction must not enable reverse travel: ${JSON.stringify(inward)}`
    );
  expect(inward.demandBands).toEqual([0, NONE]);
});
