// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {describe, expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUSpaceTimeVariogramParameterValues,
  GPUSpaceTimeVariogram
} from '../../../src/dataframe-analysis/pair-statistics/gpu-space-time-variogram';
import {GPUVariogramModelFit} from '../../../src/dataframe-analysis/pair-statistics/gpu-variogram-model-fit';
import {
  evaluateSpaceTimeVariogramModel,
  fitSpaceTimeVariogramModel,
  getGPUSpaceTimeVariogramBinLags,
  type SpaceTimeVariogramModel
} from '../../../src/dataframe-analysis/pair-statistics/space-time-variogram-model';
import {evaluateVariogramModel} from '../../../src/dataframe-analysis/pair-statistics/variogram-model';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const LAG_COUNT = 6;
const TEMPORAL_LAG_COUNT = 6;
const MAXIMUM_DISTANCE = 9;
const MAXIMUM_TIME_LAG = 7;

/** Normal draws from a seeded generator. */
function createNormal(seed: number): () => number {
  let state = seed | 0;
  const uniform = () => {
    state = (state + 0x6d2b79f5) | 0;
    let mixed = Math.imul(state ^ (state >>> 15), 1 | state);
    mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed;
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
  return () => Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform());
}

/** Separable exponential field on a station-by-time grid, by a binary64 Cholesky factor. */
function simulateScene() {
  const stations: [number, number][] = [];
  for (let y = 0; y < 6; y++) {
    for (let x = 0; x < 6; x++) {
      stations.push([x * 2, y * 2]);
    }
  }
  const steps = 10;
  const rowCount = stations.length * steps;
  const truth: SpaceTimeVariogramModel = {
    type: 'separable',
    sill: 2,
    spatial: {model: 'exponential', nugget: 0, sill: 1, range: 8},
    temporal: {model: 'exponential', nugget: 0, sill: 1, range: 12}
  };
  const rows: [number, number, number][] = [];
  for (const [x, y] of stations) {
    for (let step = 0; step < steps; step++) {
      rows.push([x, y, step]);
    }
  }
  const lower = new Float64Array(rowCount * rowCount);
  for (let row = 0; row < rowCount; row++) {
    for (let column = 0; column <= row; column++) {
      let value =
        2 -
        evaluateSpaceTimeVariogramModel(
          truth,
          Math.hypot(rows[row][0] - rows[column][0], rows[row][1] - rows[column][1]),
          Math.abs(rows[row][2] - rows[column][2])
        );
      if (row === column) {
        value += 1e-9;
      }
      for (let component = 0; component < column; component++) {
        value -= lower[row * rowCount + component] * lower[column * rowCount + component];
      }
      lower[row * rowCount + column] =
        row === column ? Math.sqrt(value) : value / lower[column * rowCount + column];
    }
  }
  const normal = createNormal(77);
  const noise = Array.from({length: rowCount}, normal);
  const values = rows.map((_, row) => {
    let sum = 0;
    for (let column = 0; column <= row; column++) {
      sum += lower[row * rowCount + column] * noise[column];
    }
    return Math.fround(sum);
  });
  return {rows, values};
}

/** Brute-force empirical surface with the documented classes. */
function getReferenceSurface(rows: [number, number, number][], values: number[]) {
  const binCount = (LAG_COUNT + 1) * (TEMPORAL_LAG_COUNT + 1);
  const sums = new Float64Array(binCount);
  const counts = new Float64Array(binCount);
  for (let first = 0; first < rows.length; first++) {
    for (let second = first + 1; second < rows.length; second++) {
      const distance = Math.hypot(
        rows[first][0] - rows[second][0],
        rows[first][1] - rows[second][1]
      );
      const timeLag = Math.abs(rows[first][2] - rows[second][2]);
      if (distance > MAXIMUM_DISTANCE || timeLag > MAXIMUM_TIME_LAG) {
        continue;
      }
      const spatialClass =
        distance === 0
          ? 0
          : 1 + Math.min(Math.floor((distance / MAXIMUM_DISTANCE) * LAG_COUNT), LAG_COUNT - 1);
      const temporalClass =
        timeLag === 0
          ? 0
          : 1 +
            Math.min(
              Math.floor((timeLag / MAXIMUM_TIME_LAG) * TEMPORAL_LAG_COUNT),
              TEMPORAL_LAG_COUNT - 1
            );
      const bin = spatialClass * (TEMPORAL_LAG_COUNT + 1) + temporalClass;
      sums[bin] += (values[first] - values[second]) ** 2;
      counts[bin] += 1;
    }
  }
  return {
    semivariances: Float64Array.from(sums, (sum, bin) =>
      counts[bin] > 0 ? sum / (2 * counts[bin]) : Number.NaN
    ),
    pairCounts: counts
  };
}

describe('space-time variogram model-fit path', () => {
  it('fits the GPU empirical surface, and agrees with the brute-force surface', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {rows, values} = simulateScene();
    const rowCount = rows.length;
    const graph = new GPUCommandGraph(device);
    const binCount = (LAG_COUNT + 1) * (TEMPORAL_LAG_COUNT + 1);
    const positionBuffer = createInputBuffer(
      device,
      Float32Array.from(rows.flatMap(([x, y]) => [x, y]))
    );
    const valueBuffer = createInputBuffer(device, Float32Array.from(values));
    const timeBuffer = createInputBuffer(device, Float32Array.from(rows.map(([, , step]) => step)));
    const parameterBuffer = createInputBuffer(
      device,
      getGPUSpaceTimeVariogramParameterValues({
        bounds: [0, 0, 11, 11],
        maximumDistance: MAXIMUM_DISTANCE,
        maximumTimeLag: MAXIMUM_TIME_LAG
      })
    );
    const semivarianceBuffer = createOutputBuffer(device, binCount);
    const countBuffer = createOutputBuffer(device, binCount);
    const statusBuffer = createOutputBuffer(device, binCount);
    graph.add(
      new GPUSpaceTimeVariogram({
        positions: importGraphBuffer(graph, 'positions', positionBuffer, 'float32x2', rowCount),
        values: importGraphBuffer(graph, 'values', valueBuffer, 'float32', rowCount),
        times: importGraphBuffer(graph, 'times', timeBuffer, 'float32', rowCount),
        parameters: importGraphBuffer(graph, 'parameters', parameterBuffer, 'float32', 8),
        gridSize: [8, 8],
        lagCount: LAG_COUNT,
        temporalLagCount: TEMPORAL_LAG_COUNT,
        output: {
          semivariances: importGraphBuffer(
            graph,
            'semivariances',
            semivarianceBuffer,
            'float32',
            binCount
          ),
          pairCounts: importGraphBuffer(graph, 'counts', countBuffer, 'uint32', binCount),
          status: importGraphBuffer(graph, 'status', statusBuffer, 'uint32', binCount)
        }
      })
    );
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const semivariances = await readFloat32(semivarianceBuffer, binCount);
    const pairCounts = await readUint32(countBuffer, binCount);
    const reference = getReferenceSurface(rows, values);
    pairCounts.forEach((count, bin) => {
      expect(count).toBe(reference.pairCounts[bin]);
      if (count > 0) {
        expect(Math.abs(semivariances[bin] - reference.semivariances[bin])).toBeLessThan(1e-3);
      }
    });
    const lags = getGPUSpaceTimeVariogramBinLags({
      lagCount: LAG_COUNT,
      temporalLagCount: TEMPORAL_LAG_COUNT,
      maximumDistance: MAXIMUM_DISTANCE,
      maximumTimeLag: MAXIMUM_TIME_LAG
    });
    for (const type of ['separable', 'product-sum', 'metric', 'sum-metric', 'gneiting'] as const) {
      const options = {type, weighting: 'pairs'} as const;
      const fromGPU = fitSpaceTimeVariogramModel({...lags, semivariances, pairCounts}, options);
      const fromReference = fitSpaceTimeVariogramModel(
        {...lags, semivariances: reference.semivariances, pairCounts: reference.pairCounts},
        options
      );
      expect(Number.isFinite(fromGPU.residual)).toBe(true);
      // A GPU surface equal to the brute-force surface gives the same fitted surface.
      for (let bin = 1; bin < binCount; bin++) {
        const gpuValue = evaluateSpaceTimeVariogramModel(
          fromGPU.model,
          lags.distances[bin],
          lags.timeLags[bin]
        );
        const referenceValue = evaluateSpaceTimeVariogramModel(
          fromReference.model,
          lags.distances[bin],
          lags.timeLags[bin]
        );
        expect(Math.abs(gpuValue - referenceValue), `${type} bin ${bin}`).toBeLessThan(5e-2);
      }
      if (fromGPU.model.type === 'separable') {
        // The simulated sill is 2 and the ranges 8 and 12; one realization is noisy.
        expect(fromGPU.model.sill).toBeGreaterThan(1);
        expect(fromGPU.model.sill).toBeLessThan(4);
        expect(fromGPU.model.spatial.range).toBeGreaterThan(2);
        expect(fromGPU.model.temporal.range).toBeGreaterThan(3);
      }
    }
    compiled.destroy();
    for (const buffer of [
      positionBuffer,
      valueBuffer,
      timeBuffer,
      parameterBuffer,
      semivarianceBuffer,
      countBuffer,
      statusBuffer
    ]) {
      buffer.destroy();
    }
  }, 120_000);

  it('GPUVariogramModelFit fits Matern models', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const binCount = 24;
    const distances = Array.from({length: binCount}, (_, index) => (index + 1) * 0.5);
    const truth = {model: 'matern-3/2' as const, nugget: 0.05, sill: 1.5, range: 5};
    const lagBuffer = createInputBuffer(device, Float32Array.from(distances));
    const semivarianceBuffer = createInputBuffer(
      device,
      Float32Array.from(distances.map(lag => evaluateVariogramModel(truth, lag)))
    );
    const countBuffer = createInputBuffer(device, new Uint32Array(binCount).fill(100));
    const parameterOutput = createOutputBuffer(device, 8);
    const diagnosticOutput = createOutputBuffer(device, 8);
    const statusOutput = createOutputBuffer(device, 2);
    graph.add(
      new GPUVariogramModelFit({
        distances: importGraphBuffer(graph, 'lags', lagBuffer, 'float32', binCount),
        semivariances: importGraphBuffer(
          graph,
          'semivariances',
          semivarianceBuffer,
          'float32',
          binCount
        ),
        pairCounts: importGraphBuffer(graph, 'counts', countBuffer, 'uint32', binCount),
        models: ['matern-3/2', 'matern-5/2'],
        weighting: 'pairs',
        output: {
          parameters: importGraphBuffer(graph, 'parameters', parameterOutput, 'float32x4', 2),
          diagnostics: importGraphBuffer(graph, 'diagnostics', diagnosticOutput, 'float32x4', 2),
          status: importGraphBuffer(graph, 'status', statusOutput, 'uint32', 2)
        }
      })
    );
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const parameters = await readFloat32(parameterOutput, 8);
    const status = await readUint32(statusOutput, 2);
    // The surface is Matern 3/2: its own row recovers it and has the smaller residual.
    expect(status[0]).toBe(0);
    expect(Math.abs(parameters[0] - 0.05)).toBeLessThan(5e-3);
    expect(Math.abs(parameters[1] - 1.5)).toBeLessThan(5e-3);
    expect(Math.abs(parameters[2] - 5)).toBeLessThan(2e-2);
    expect(parameters[3]).toBeLessThan(parameters[7]);
    compiled.destroy();
    for (const buffer of [
      lagBuffer,
      semivarianceBuffer,
      countBuffer,
      parameterOutput,
      diagnosticOutput,
      statusOutput
    ]) {
      buffer.destroy();
    }
  });
});
