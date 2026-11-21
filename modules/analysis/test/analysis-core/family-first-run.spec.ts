// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Declare every literal fixture dependency to the browser runner before dynamic execution.
import {luma} from '@luma.gl/core';
import {webgpuAdapter} from '@luma.gl/core/webgpu';
import '@luma.gl/graph';
import '@luma-contributors/analysis/analysis-core';
import {
  GPUTerrainDerivatives,
  getGPUTerrainDerivativesParameterValues
} from '@luma-contributors/analysis/gpu-terrain';
import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import '@luma-contributors/analysis/dataframe-analysis';
import '@luma-contributors/analysis/pointcloud-analysis';
import {expect, it} from 'vitest';
import {
  loadTerrainExample,
  loadDistanceExample,
  loadTimeExample,
  loadPointCloudExample,
  loadChangedTimeExample
} from './family-first-run.test-utils';
import {requireTestDevice} from '../../src/testing/require-test-device';

function expectFiniteValues(actual: Float32Array | number[], expected: number[]) {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    const value = actual[index];
    expect(Number.isFinite(value), `output ${index}`).toBe(true);
    expect(Math.abs(value - expected[index])).toBeLessThan(1e-5);
  }
}

it('COMMON terrain executes the displayed east-rising ramp with west illumination', async () => {
  requireTestDevice(navigator.gpu);
  const {terrain} = await loadTerrainExample();
  expect(terrain.slope).toHaveLength(9);
  expect(terrain.hillshade).toHaveLength(9);
  for (const value of [...terrain.slope, ...terrain.hillshade])
    expect(Number.isFinite(value)).toBe(true);
  // Rise/run = 30/30 = 1; atan(1) = 45 degrees. The west light matches the normal.
  expect(Math.abs(terrain.slope[4] - 45)).toBeLessThan(1e-5);
  expect(Math.abs(terrain.hillshade[4] - 1)).toBeLessThan(1e-5);
});

it('FST-14 keeps a 45-degree ramp slope and reverses shade under east illumination', async () => {
  requireTestDevice(navigator.gpu);
  const {runTerrainLightingControlExample} = await loadTerrainExample();
  const terrain = await runTerrainLightingControlExample(90);
  expect(terrain.slope).toHaveLength(9);
  expect(terrain.hillshade).toHaveLength(9);
  expectFiniteValues([terrain.slope[4], terrain.hillshade[4]], [45, 0]);
});

it('FST-14 retains a separate flat-gradient terrain control', async () => {
  requireTestDevice(navigator.gpu);
  const {runTerrainLightingControlExample} = await loadTerrainExample();
  const terrain = await runTerrainLightingControlExample(270, true);
  expect(terrain.slope).toHaveLength(9);
  expect(terrain.hillshade).toHaveLength(9);
  // Flat center: sin(45 degrees) = sqrt(1/2).
  expectFiniteValues([terrain.slope[4], terrain.hillshade[4]], [0, 0.7071067811865475]);
});

it('FST-12 borrows a device across successful and failed readbacks and a second computation', async () => {
  requireTestDevice(navigator.gpu);
  const {runFirstComputationOnDevice} = await loadTerrainExample();
  const device = await luma.createDevice({type: 'webgpu', adapters: [webgpuAdapter]});
  try {
    const run = (id: string, azimuthDegrees: number, failReadback = false) =>
      runFirstComputationOnDevice(device, id, ({graph, float32}) => {
        const grid = makeGPURasterMetadata({width: 3, height: 3, cellSize: 30, origin: [0, 90]});
        const heights = float32('heights', Float32Array.of(0, 30, 60, 0, 30, 60, 0, 30, 60));
        const parameters = float32(
          'parameters',
          getGPUTerrainDerivativesParameterValues(
            {
              azimuthDegrees,
              altitudeDegrees: 45
            },
            grid
          )
        );
        const slope = float32('slope', new Float32Array(9).fill(-1));
        const shade = float32('shade', new Float32Array(9).fill(-1));
        graph.add(
          new GPUTerrainDerivatives({
            grid,
            elevation: makeGPURasterBand(heights.view, grid, {id: 'elevation'}),
            parameters: parameters.view,
            output: {slope: slope.view, hillshade: shade.view}
          })
        );
        return async () => {
          const result = [(await slope.read())[4], (await shade.read())[4]];
          if (failReadback) throw new Error('Intentional first-run readback failure');
          return result;
        };
      });
    expectFiniteValues(await run('borrowed-first', 270), [45, 1]);
    await expect(run('borrowed-failure', 270, true)).rejects.toThrow(
      'Intentional first-run readback failure'
    );
    expectFiniteValues(await run('borrowed-second', 90), [45, 0]);
  } finally {
    device.destroy();
  }
});

it('COMMON distance executes the displayed exact route with allocation and radius outputs', async () => {
  requireTestDevice(navigator.gpu);
  const {distanceField} = await loadDistanceExample();
  // Euclidean cell-center distances from the center of the 3 by 3 unit grid.
  expectFiniteValues(
    distanceField.distances,
    [1.4142135623730951, 1, 1.4142135623730951, 1, 0, 1, 1.4142135623730951, 1, 1.4142135623730951]
  );
  expect(Array.from(distanceField.allocation)).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 0]);
  expect(Array.from(distanceField.withinDistance)).toEqual([1, 1, 1, 1, 1, 1, 1, 1, 1]);
});

it('COMMON time executes the displayed inclusive window and reads only its compact result', async () => {
  requireTestDevice(navigator.gpu);
  const {timeWindow} = await loadTimeExample();
  expect(Array.from(timeWindow.rows)).toEqual([1]);
  expect(timeWindow.count).toBe(1);
  expect(timeWindow.requiredCount).toBe(1);
  expect(timeWindow.overflow).toBe(0);
  const {timeWindow: changed} = await loadChangedTimeExample();
  expect(Array.from(changed.rows)).toEqual([]);
  expect(Array.from(changed.rows)).not.toEqual([1]);
  expect(changed.count).toBe(0);
  expect(changed.requiredCount).toBe(0);
  expect(changed.overflow).toBe(0);
});

it('COMMON point-cloud excludes unsupported rows from all three displayed thresholds', async () => {
  requireTestDevice(navigator.gpu);
  const {displayedPointCloudResults} = await loadPointCloudExample();
  // Supported means 1,1,10 have mean 4 and population variance 18. Assert the actual
  // invocation-loop emissions, including labels/order, rather than calling the function again.
  const expectedMultipliers = [2, 0, 1.5];
  const expectedValidity = [
    [1, 1, 1, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 0]
  ];
  expect(displayedPointCloudResults).toHaveLength(3);
  for (const [index, displayed] of displayedPointCloudResults.entries()) {
    expect(displayed).toHaveLength(3);
    const [multiplier, means, validity] = displayed;
    expect(multiplier).toBe(expectedMultipliers[index]);
    expectFiniteValues(means, [1, 1, 10, -1]);
    expect(validity).toEqual(expectedValidity[index]);
  }
});
