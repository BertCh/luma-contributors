// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPURasterRegionGraph,
  type GPURasterRegionGraphProps
} from '../../../src/raster-analysis/remote-sensing/gpu-raster-region-graph';
import {computeRasterRegionGraphOnCPU} from '../../../src/raster-analysis/remote-sensing/cpu-raster-region-graph-operation';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRemoteSensingHarness} from './remote-sensing-test-utils';

async function runRegions(
  labels: number[],
  values: number[],
  regionCount: number,
  mergeThreshold: number,
  searchStructure: GPURasterRegionGraphProps['searchStructure'],
  maximumIterations: number
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRemoteSensingHarness(device);
  const output = {
    adjacency: harness.output('adjacency', 'uint32', regionCount ** 2),
    weights: harness.output('weights', 'float32', regionCount ** 2),
    mapping: harness.output('mapping', 'uint32', regionCount),
    status: {iterationLimitReached: harness.output('pending', 'uint32', 1)}
  };
  for (const view of [
    output.adjacency,
    output.weights,
    output.mapping,
    output.status.iterationLimitReached
  ]) {
    (view.buffer.defaultBuffer as Buffer).write(
      view.format === 'float32'
        ? new Float32Array(view.length).fill(-12345)
        : new Uint32Array(view.length).fill(0xa5a5a5a5)
    );
  }
  return harness.run(
    new GPURasterRegionGraph({
      grid: makeGPURasterMetadata({width: labels.length, height: 1}),
      labels: harness.mask(labels),
      values: harness.input(values),
      regionCount,
      mergeThreshold,
      searchStructure,
      maximumIterations,
      output
    })
  );
}

it('GPU-13 Boruvka and explicit dense routes match the CPU oracle on missing/background and tied edges', async () => {
  const labels = [4, 3, 2, 1, 0, 0xffffffff, 5, 6, 7, 8, 8, 2, 3];
  const values = [10, 10, 12, 12, NaN, 100, 0, 0, 1, NaN, NaN, 12, 10];
  const regionCount = 9;
  const grid = makeGPURasterMetadata({width: labels.length, height: 1});
  const expected = computeRasterRegionGraphOnCPU(
    Uint32Array.from(labels),
    Float32Array.from(values),
    {
      grid,
      regionCount,
      mergeThreshold: 2,
      maximumIterations: 8
    }
  );
  expect(Array.from(expected.mapping)).toEqual([0, 1, 1, 1, 1, 5, 5, 5, 8]);
  for (const searchStructure of ['dense', 'boruvka'] as const) {
    const actual = await runRegions(labels, values, regionCount, 2, searchStructure, 8);
    expect(actual['adjacency']).toEqual(Array.from(expected.adjacency));
    expect(actual['weights']).toEqual(Array.from(expected.weights));
    expect(actual['mapping']).toEqual(Array.from(expected.mapping));
    expect(actual['pending']).toEqual([0]);
  }
});

it('GPU-13 handles 513 regions with closed-form component roots, adjacency and every weight', async () => {
  const regionCount = 513;
  const labels = Array.from({length: regionCount}, (_, region) => region);
  const values = labels.map(region => Math.floor(region / 3));
  const actual = await runRegions(labels, values, regionCount, 0, 'auto', 10);
  expect(actual['mapping']).toEqual(labels.map(region => region - (region % 3)));
  expect(actual['pending']).toEqual([0]);
  expect(actual['adjacency']).toEqual(
    Array.from({length: regionCount ** 2}, (_, entry) =>
      Math.abs(Math.floor(entry / regionCount) - (entry % regionCount)) === 1 ? 1 : 0
    )
  );
  expect(actual['weights']).toEqual(
    Array.from({length: regionCount ** 2}, (_, entry) => {
      const first = Math.floor(entry / regionCount);
      const second = entry % regionCount;
      return Math.abs(first - second) === 1 ? Math.abs(values[first] - values[second]) : NaN;
    })
  );
});

it('GPU-13 reports a pending merge after one phase and closes all components within ceil(log2 regions)', async () => {
  const labels = [0, 1, 2, 3, 4, 5, 6, 7];
  const values = [0, 0, 10, 10, 20, 20, 30, 30];
  const partial = await runRegions(labels, values, 8, 100, 'boruvka', 1);
  expect(partial['mapping']).toEqual([0, 0, 2, 2, 4, 4, 6, 6]);
  expect(partial['pending']).toEqual([1]);
  const complete = await runRegions(labels, values, 8, 100, 'boruvka', 3);
  expect(complete['mapping']).toEqual(new Array(8).fill(0));
  expect(complete['pending']).toEqual([0]);
  expect(complete['weights']).toEqual(partial['weights']);
  expect(complete['adjacency']).toEqual(partial['adjacency']);
});

it('GPU-13 repeatedly closes one connected 1025-region component across workgroups', async () => {
  const regionCount = 1025;
  const labels = Array.from({length: regionCount}, (_, region) => region);
  const values = labels.map(region => (region * (region + 1)) / 2);
  for (let repetition = 0; repetition < 3; repetition++) {
    const actual = await runRegions(
      labels,
      values,
      regionCount,
      1024,
      'boruvka',
      Math.ceil(Math.log2(regionCount))
    );
    expect(actual['mapping'], `repetition ${repetition}`).toEqual(new Array(regionCount).fill(0));
    expect(actual['pending']).toEqual([0]);
    // Adjacent triangular numbers differ by r+1; every path edge is eligible, including 1024.
    for (const region of [0, 255, 256, 511, 512, 1023]) {
      expect(actual['adjacency'][region * regionCount + region + 1]).toBe(1);
      expect(actual['weights'][region * regionCount + region + 1]).toBe(region + 1);
    }
  }
}, 60000);
