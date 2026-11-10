// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {
  evaluateGPUKernelDensityKernel,
  getGPUAdaptiveBandwidthKernelDensityParameterValues,
  getGPUKernelDensityParameterValues,
  GPUAdaptiveBandwidthKernelDensity,
  GPUKernelDensity,
  type GPUKernelDensityKernel
} from '../../../src/gpu-spatial-analysis/point-density/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {AnalysisRig, createSeededRandom} from '../catchment-accessibility/rig';
import {makeExtentGrid} from '../spatial-interpolation/extent-grid';

const WIDTH = 24;
const HEIGHT = 20;
const EXTENT = [0, 0, 120, 100] as const;
const BANDWIDTH = 8;

/** A dense cluster, a loose cluster and a few isolated points, so the pilot varies a lot. */
function makePoints(seed: number): {positions: Float32Array; weights: Float32Array} {
  const random = createSeededRandom(seed);
  const points: number[] = [];
  const add = (count: number, centerX: number, centerY: number, spread: number) => {
    for (let index = 0; index < count; index++) {
      points.push(
        Math.fround(Math.min(119, Math.max(1, centerX + (random() - 0.5) * spread))),
        Math.fround(Math.min(99, Math.max(1, centerY + (random() - 0.5) * spread)))
      );
    }
  };
  add(120, 30, 30, 10);
  add(60, 85, 60, 40);
  add(6, 100, 15, 4);
  const count = points.length / 2;
  return {
    positions: new Float32Array(points),
    weights: Float32Array.from({length: count}, () => Math.fround(0.5 + random()))
  };
}

type Reference = {values: number[]; bandwidths: number[]; pilots: number[]};

/** Binary64 all-pairs evaluation of the documented estimator (no grid index, no f32). */
function referenceAdaptive(
  positions: Float32Array,
  weights: Float32Array | undefined,
  kernel: GPUKernelDensityKernel,
  sensitivity: number,
  limit: number,
  normalization: 'intensity' | 'probability'
): Reference {
  const count = positions.length / 2;
  const weight = (index: number) => weights?.[index] ?? 1;
  const support = (bandwidth: number) => (kernel === 'gaussian' ? 5 * bandwidth : bandwidth);
  const pilots = Array.from({length: count}, (_, index) => {
    let total = 0;
    for (let other = 0; other < count; other++) {
      const distance = Math.hypot(
        positions[2 * index] - positions[2 * other],
        positions[2 * index + 1] - positions[2 * other + 1]
      );
      total += weight(other) * evaluateGPUKernelDensityKernel(kernel, BANDWIDTH, distance);
    }
    return total;
  });
  let weightTotal = 0;
  let logTotal = 0;
  for (let index = 0; index < count; index++) {
    weightTotal += weight(index);
    logTotal += weight(index) * Math.log(pilots[index]);
  }
  const logMean = logTotal / weightTotal;
  const bandwidths = pilots.map(
    pilot =>
      BANDWIDTH * Math.min(limit, Math.max(1 / limit, (pilot / Math.exp(logMean)) ** -sensitivity))
  );
  const values: number[] = [];
  for (let row = 0; row < HEIGHT; row++) {
    for (let column = 0; column < WIDTH; column++) {
      const x = EXTENT[0] + ((column + 0.5) * (EXTENT[2] - EXTENT[0])) / WIDTH;
      const y = EXTENT[1] + ((row + 0.5) * (EXTENT[3] - EXTENT[1])) / HEIGHT;
      let total = 0;
      for (let index = 0; index < count; index++) {
        const distance = Math.hypot(x - positions[2 * index], y - positions[2 * index + 1]);
        if (distance < support(bandwidths[index])) {
          total +=
            weight(index) * evaluateGPUKernelDensityKernel(kernel, bandwidths[index], distance);
        }
      }
      values.push(normalization === 'probability' ? total / weightTotal : total);
    }
  }
  return {values, bandwidths, pilots};
}

async function runAdaptive(options: {
  positions: Float32Array;
  weights?: Float32Array;
  kernel: GPUKernelDensityKernel;
  sensitivity: number;
  limit: number;
  normalization: 'intensity' | 'probability';
}) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  const grid = makeExtentGrid(WIDTH, HEIGHT, EXTENT);
  const count = options.positions.length / 2;
  const values = rig.output('float32', WIDTH * HEIGHT);
  const pointBandwidths = rig.output('float32', count);
  const pilotDensities = rig.output('float32', count);
  const weightSum = rig.output('float32', 1);
  rig.run(
    new GPUAdaptiveBandwidthKernelDensity({
      positions: rig.input(options.positions, 'float32x2', count),
      weights: options.weights ? rig.input(options.weights, 'float32') : undefined,
      parameters: rig.input(
        getGPUAdaptiveBandwidthKernelDensityParameterValues(
          {bandwidth: BANDWIDTH, sensitivity: options.sensitivity},
          grid
        ),
        'float32'
      ),
      grid,
      indexGridSize: [12, 10],
      indexBounds: EXTENT,
      kernel: options.kernel,
      normalization: options.normalization,
      bandwidthFactorLimit: options.limit,
      output: {values, pointBandwidths, pilotDensities, weightSum}
    })
  );
  const result = {
    values: await rig.readFloat(values),
    bandwidths: await rig.readFloat(pointBandwidths),
    pilots: await rig.readFloat(pilotDensities),
    weightSum: (await rig.readFloat(weightSum))[0]
  };
  rig.destroy();
  return result;
}

async function runFixed(
  positions: Float32Array,
  weights: Float32Array | undefined,
  kernel: GPUKernelDensityKernel
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  const grid = makeExtentGrid(WIDTH, HEIGHT, EXTENT);
  const values = rig.output('float32', WIDTH * HEIGHT);
  rig.run(
    new GPUKernelDensity({
      positions: rig.input(positions, 'float32x2', positions.length / 2),
      weights: weights ? rig.input(weights, 'float32') : undefined,
      parameters: rig.input(
        getGPUKernelDensityParameterValues({bandwidth: BANDWIDTH}, grid),
        'float32'
      ),
      grid,
      indexGridSize: [12, 10],
      indexBounds: EXTENT,
      kernel,
      edgeCorrection: 'none',
      output: {values}
    })
  );
  const result = await rig.readFloat(values);
  rig.destroy();
  return result;
}

describe('GPUAdaptiveBandwidthKernelDensity', () => {
  const {positions, weights} = makePoints(5);

  for (const kernel of ['gaussian', 'quartic', 'epanechnikov'] as const) {
    it(`matches the binary64 Abramson estimator, ${kernel} kernel, weighted`, async () => {
      const expected = referenceAdaptive(positions, weights, kernel, 0.5, 4, 'intensity');
      const actual = await runAdaptive({
        positions,
        weights,
        kernel,
        sensitivity: 0.5,
        limit: 4,
        normalization: 'intensity'
      });
      const peak = Math.max(...expected.values);
      // The adaptation is real: bandwidths span a wide range around h.
      expect(Math.max(...expected.bandwidths) / Math.min(...expected.bandwidths)).toBeGreaterThan(
        2
      );
      for (const [index, bandwidth] of expected.bandwidths.entries()) {
        expect(Math.abs(actual.bandwidths[index] - bandwidth), `bandwidth ${index}`).toBeLessThan(
          1e-3 * bandwidth
        );
        expect(
          Math.abs(actual.pilots[index] - expected.pilots[index]),
          `pilot ${index}`
        ).toBeLessThan(1e-3 * expected.pilots[index]);
      }
      let worst = 0;
      for (const [index, value] of expected.values.entries()) {
        worst = Math.max(worst, Math.abs(actual.values[index] - value) / peak);
      }
      expect(worst).toBeLessThan(2e-3);
    });
  }

  it('equals GPUKernelDensity (no edge correction) when the sensitivity is zero', async () => {
    const adaptive = await runAdaptive({
      positions,
      kernel: 'gaussian',
      sensitivity: 0,
      limit: 4,
      normalization: 'intensity'
    });
    const fixed = await runFixed(positions, undefined, 'gaussian');
    const peak = Math.max(...fixed);
    for (const [index, value] of fixed.entries()) {
      expect(Math.abs(adaptive.values[index] - value) / peak).toBeLessThan(1e-4);
    }
  });

  it('narrows dense and widens sparse points, and probability sums to one over the plane', async () => {
    const adaptive = await runAdaptive({
      positions,
      weights,
      kernel: 'gaussian',
      sensitivity: 0.5,
      limit: 4,
      normalization: 'probability'
    });
    const count = positions.length / 2;
    // Points 0..119 are the dense cluster, the last 6 are isolated.
    const mean = (start: number, end: number) =>
      adaptive.bandwidths.slice(start, end).reduce((sum, value) => sum + value, 0) / (end - start);
    expect(mean(0, 120)).toBeLessThan(BANDWIDTH);
    expect(mean(count - 6, count)).toBeGreaterThan(BANDWIDTH);
    const cellArea = ((EXTENT[2] - EXTENT[0]) / WIDTH) * ((EXTENT[3] - EXTENT[1]) / HEIGHT);
    const mass = adaptive.values.reduce((sum, value) => sum + value, 0) * cellArea;
    // Edge mass outside the window and cell-center sampling make this approximate.
    expect(mass).toBeGreaterThan(0.85);
    expect(mass).toBeLessThan(1.05);
  });

  it('clamps point bandwidths to the factor limit', async () => {
    const adaptive = await runAdaptive({
      positions,
      kernel: 'quartic',
      sensitivity: 1,
      limit: 1.5,
      normalization: 'intensity'
    });
    for (const bandwidth of adaptive.bandwidths) {
      expect(bandwidth).toBeGreaterThanOrEqual((BANDWIDTH / 1.5) * 0.999);
      expect(bandwidth).toBeLessThanOrEqual(BANDWIDTH * 1.5 * 1.001);
    }
  });

  it('rejects an invalid sensitivity and factor limit', () => {
    const grid = makeExtentGrid(WIDTH, HEIGHT, EXTENT);
    expect(() =>
      getGPUAdaptiveBandwidthKernelDensityParameterValues({bandwidth: 1, sensitivity: 2}, grid)
    ).toThrow(/sensitivity/);
  });
});
