// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUPointDensity} from '../../../src/gpu-spatial-analysis/point-density/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// Math correctness review, interpolation-density-01 (luma-official-02). `GPUPointDensity` counts
// with the published `GPUGridBinning` and, in the default `'workgroup'` sum mode, keys the weights
// with its own kernel. A point is one observation: its count and its weight belong to one cell.
// The expected values below do not come from either kernel: the weight of every row is a constant,
// so `sums` must equal that constant times `counts` cell by cell whatever cell a point is put in.

type DensityRun = {counts: number[]; sums: number[]; means: number[]};

async function runWeightedGridDensity(
  positions: number[],
  weights: number[],
  bounds: [number, number, number, number],
  gridSize: [number, number],
  sumAccumulation: 'workgroup' | 'atomic'
): Promise<DensityRun> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const cellCount = gridSize[0] * gridSize[1];
  const rowCount = weights.length;
  const positionsBuffer = createInputBuffer(device, Float32Array.from(positions));
  const weightsBuffer = createInputBuffer(device, Float32Array.from(weights));
  const valuesBuffer = createOutputBuffer(device, cellCount);
  const countsBuffer = createOutputBuffer(device, cellCount);
  const sumsBuffer = createOutputBuffer(device, cellCount);
  const meansBuffer = createOutputBuffer(device, cellCount);
  const graph = new GPUCommandGraph(device, {id: `density-cell-mapping-${sumAccumulation}`});
  graph.add(
    new GPUPointDensity({
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rowCount),
      weights: importGraphBuffer(graph, 'weights', weightsBuffer, 'float32', rowCount),
      bounds,
      gridSize,
      statistic: 'sum',
      sumAccumulation,
      output: {
        values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', cellCount),
        counts: importGraphBuffer(graph, 'counts', countsBuffer, 'uint32', cellCount),
        sums: importGraphBuffer(graph, 'sums', sumsBuffer, 'float32', cellCount),
        means: importGraphBuffer(graph, 'means', meansBuffer, 'float32', cellCount)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    counts: await readUint32(countsBuffer, cellCount),
    sums: await readFloat32(sumsBuffer, cellCount),
    means: await readFloat32(meansBuffer, cellCount)
  };
  compiled.destroy();
  for (const buffer of [positionsBuffer, weightsBuffer, valuesBuffer, countsBuffer, sumsBuffer]) {
    buffer.destroy();
  }
  meansBuffer.destroy();
  return result;
}

for (const sumAccumulation of ['workgroup', 'atomic'] as const) {
  it(`GPUPointDensity (${sumAccumulation}) puts the count and the weight of a point on a cell edge in one cell when the bounds straddle zero`, async () => {
    // Bounds [-3, 7] in ten columns: column k covers [-3 + k, -2 + k). x = -2 is the left edge
    // of column 1, and (x - minimum) / (maximum - minimum) * 10 = (1 / 10) * 10 = 1 in float32.
    const {counts, sums, means} = await runWeightedGridDensity(
      [-2, 0.5],
      [5],
      [-3, 0, 7, 1],
      [10, 1],
      sumAccumulation
    );
    expect(counts).toEqual([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
    expect(sums).toEqual([0, 5, 0, 0, 0, 0, 0, 0, 0, 0]);
    expect(means).toEqual([0, 5, 0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it(`GPUPointDensity (${sumAccumulation}) keeps counts and weights together when every point sits on a cell edge`, async () => {
    // Three lattices with integer coordinates and integer cell sizes, each with bounds that
    // straddle zero. Every weight is 3, so the sum of a cell is three times its count.
    const cases: {bounds: [number, number, number, number]; gridSize: [number, number]}[] = [
      {bounds: [-3, -3, 7, 7], gridSize: [10, 10]},
      {bounds: [-180, -90, 180, 90], gridSize: [360, 180]},
      {bounds: [-500, -700, 1500, 300], gridSize: [100, 50]}
    ];
    for (const {bounds, gridSize} of cases) {
      const [minimumX, minimumY, maximumX, maximumY] = bounds;
      const stepX = (maximumX - minimumX) / gridSize[0];
      const stepY = (maximumY - minimumY) / gridSize[1];
      const positions: number[] = [];
      for (let row = 0; row <= gridSize[1]; row += Math.max(1, Math.floor(gridSize[1] / 20))) {
        for (let column = 0; column <= gridSize[0]; column++) {
          positions.push(minimumX + column * stepX, minimumY + row * stepY);
        }
      }
      const rowCount = positions.length / 2;
      const {counts, sums, means} = await runWeightedGridDensity(
        positions,
        new Array<number>(rowCount).fill(3),
        bounds,
        gridSize,
        sumAccumulation
      );
      expect(counts.reduce((total, count) => total + count, 0)).toBe(rowCount);
      expect(sums).toEqual(counts.map(count => 3 * count));
      expect(means).toEqual(counts.map(count => (count > 0 ? 3 : 0)));
    }
  });
}
