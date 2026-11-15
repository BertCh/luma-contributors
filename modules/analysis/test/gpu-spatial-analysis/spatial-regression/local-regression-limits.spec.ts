// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {GPUMultiscaleGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression';
import {GPUGeneralizedGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createGaussianScene,
  createPoissonScene,
  fitLocalPoissonOnCPU,
  fitMultiscaleBackfitOnCPU,
  fitMultiscaleOnCPU,
  getHaversineDistance,
  type LocalGlmScene
} from '../../reference/local-glm-reference';

type Options = {
  bandwidth: number;
  kernel?: 'gaussian' | 'bisquare';
  gaussianCutoff?: number;
  bandwidthMode?: 'fixed' | 'adaptive';
  leaveOneOut?: boolean;
  indexGridSize?: readonly [number, number];
  spatialContext?: GPUSpatialContext;
};

export async function runGeneralized(scene: LocalGlmScene, options: Options) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const buffers: Buffer[] = [];
  const graph = new GPUCommandGraph(device, {id: 'local-limits'});
  const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    name: string,
    format: Format,
    data: Float32Array | Uint32Array
  ) => {
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      name,
      buffer,
      format,
      data.length / (format === 'float32x2' ? 2 : 1)
    );
  };
  const output = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const columns = scene.predictorCount + 1;
  const coefficients = output('coefficients', 'float32', scene.rowCount * columns);
  const fitted = output('fitted', 'float32', scene.rowCount);
  const codes = output('codes', 'uint32', scene.rowCount);
  const iterations = output('iterations', 'uint32', scene.rowCount);
  graph.add(
    new GPUGeneralizedGeographicallyWeightedRegression({
      id: 'generalized',
      family: 'poisson',
      kernel: options.kernel ?? 'bisquare',
      gaussianCutoff: options.gaussianCutoff,
      bandwidthMode: options.bandwidthMode,
      leaveOneOut: options.leaveOneOut,
      maximumNeighborCount: options.bandwidthMode === 'adaptive' ? 64 : undefined,
      positions: input('positions', 'float32x2', scene.positions),
      predictors: input('predictors', 'float32', scene.predictors),
      response: input('response', 'float32', scene.response),
      predictorCount: scene.predictorCount,
      parameters: input('parameters', 'float32', Float32Array.from([options.bandwidth, 1e-6])),
      maximumIterations: 64,
      indexGridSize: options.indexGridSize,
      spatialContext: options.spatialContext,
      output: {
        coefficients: coefficients.view,
        fitted: fitted.view,
        status: {codes: codes.view, iterationCount: iterations.view}
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    coefficients: Array.from(await readFloat32(coefficients.buffer, scene.rowCount * columns)),
    codes: Array.from(await readUint32(codes.buffer, scene.rowCount))
  };
  compiled.destroy();
  for (const buffer of buffers) buffer.destroy();
  return result;
}

async function expectMatchesReference(
  scene: LocalGlmScene,
  options: Options,
  getDistance: (focus: number, row: number) => number,
  sampleCount: number,
  tolerance: number
) {
  const result = await runGeneralized(scene, options);
  const columns = scene.predictorCount + 1;
  const step = Math.floor(scene.rowCount / sampleCount);
  let worst = 0;
  let compared = 0;
  for (let focus = 0; focus < scene.rowCount; focus += step) {
    const expected = fitLocalPoissonOnCPU(scene, focus, options.bandwidth, getDistance);
    if (!expected) continue;
    expect(result.codes[focus], `status of row ${focus}`).toBe(0);
    for (let column = 0; column < columns; column++) {
      const actual = result.coefficients[focus * columns + column];
      worst = Math.max(worst, Math.abs(actual - expected[column]));
      expect(Math.abs(actual - expected[column]), `row ${focus} column ${column}`).toBeLessThan(
        tolerance
      );
    }
    compared++;
  }
  expect(compared).toBeGreaterThan(sampleCount / 2);
  return worst;
}

const planar = (scene: LocalGlmScene) => (focus: number, row: number) =>
  Math.hypot(
    scene.positions[2 * focus] - scene.positions[2 * row],
    scene.positions[2 * focus + 1] - scene.positions[2 * row + 1]
  );

it('generalized GWR fits 14 predictors (above the old limit of 11) like a float64 local Poisson fit', async () => {
  const scene = createPoissonScene(500, 14, 7);
  await expectMatchesReference(scene, {bandwidth: 70}, planar(scene), 20, 0.05);
});

it('generalized GWR fits the largest supported predictor count (24)', async () => {
  const scene = createPoissonScene(900, 24, 11);
  await expectMatchesReference(scene, {bandwidth: 160}, planar(scene), 8, 0.1);
});

it('generalized GWR with a grid index fits 20000 rows (above the old limit of 16384)', async () => {
  const scene = createPoissonScene(20000, 2, 23);
  await expectMatchesReference(
    scene,
    {bandwidth: 6, indexGridSize: [48, 48]},
    planar(scene),
    40,
    0.05
  );
});

it('generalized GWR with great-circle distances matches a float64 haversine reference', async () => {
  const scene = createPoissonScene(400, 2, 31, 100, true);
  const getDistance = (focus: number, row: number) =>
    getHaversineDistance(
      scene.positions[2 * focus],
      scene.positions[2 * focus + 1],
      scene.positions[2 * row],
      scene.positions[2 * row + 1]
    );
  await expectMatchesReference(
    scene,
    {
      bandwidth: 350000,
      spatialContext: {
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle',
        units: 'meters'
      } as GPUSpatialContext
    },
    getDistance,
    20,
    0.05
  );
});

type MultiscaleOptions = {
  bandwidths: number[];
  indexGridSize?: readonly [number, number] | false;
  spatialContext?: GPUSpatialContext;
  candidates?: {values: number[]; count: number};
  maximumIterations?: number;
  tolerance?: number;
  kernel?: 'gaussian' | 'bisquare';
  gaussianCutoff?: number;
  /** Alternative implementation with the same props (the legacy serial convergence kernel). */
  implementation?: typeof GPUMultiscaleGeographicallyWeightedRegression;
};

export async function runMultiscale(scene: LocalGlmScene, options: MultiscaleOptions) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const buffers: Buffer[] = [];
  const graph = new GPUCommandGraph(device, {id: 'multiscale-limits'});
  const input = <Format extends 'float32' | 'float32x2'>(
    name: string,
    format: Format,
    data: Float32Array
  ) => {
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      name,
      buffer,
      format,
      data.length / (format === 'float32x2' ? 2 : 1)
    );
  };
  const output = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const columns = scene.predictorCount + 1;
  const coefficients = output('coefficients', 'float32', scene.rowCount * columns);
  const fitted = output('fitted', 'float32', scene.rowCount);
  const codes = output('codes', 'uint32', scene.rowCount);
  const iterations = output('iterations', 'uint32', 1);
  const selected = output('selected', 'float32', columns);
  graph.add(
    new (options.implementation ?? GPUMultiscaleGeographicallyWeightedRegression)({
      id: 'multiscale',
      kernel: options.kernel ?? 'bisquare',
      gaussianCutoff: options.gaussianCutoff,
      positions: input('positions', 'float32x2', scene.positions),
      predictors: input('predictors', 'float32', scene.predictors),
      response: input('response', 'float32', scene.response),
      predictorCount: scene.predictorCount,
      parameters: input(
        'parameters',
        'float32',
        Float32Array.from([options.tolerance ?? 1e-6, ...options.bandwidths])
      ),
      bandwidthCandidates: options.candidates
        ? input('candidates', 'float32', Float32Array.from(options.candidates.values))
        : undefined,
      bandwidthCandidateCount: options.candidates?.count,
      maximumIterations: options.maximumIterations ?? 40,
      indexGridSize: options.indexGridSize,
      spatialContext: options.spatialContext,
      output: {
        coefficients: coefficients.view,
        fitted: fitted.view,
        selectedBandwidths: selected.view,
        status: {codes: codes.view, iterationCount: iterations.view}
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    coefficients: Array.from(await readFloat32(coefficients.buffer, scene.rowCount * columns)),
    codes: Array.from(await readUint32(codes.buffer, scene.rowCount)),
    fitted: Array.from(await readFloat32(fitted.buffer, scene.rowCount)),
    selected: Array.from(await readFloat32(selected.buffer, columns)),
    iterationCount: (await readUint32(iterations.buffer, 1))[0]
  };
  compiled.destroy();
  for (const buffer of buffers) buffer.destroy();
  return result;
}

async function expectMultiscaleMatchesReference(
  scene: LocalGlmScene,
  options: MultiscaleOptions,
  getDistance: (focus: number, row: number) => number,
  tolerance: number
) {
  const result = await runMultiscale(scene, options);
  const expected = fitMultiscaleOnCPU(scene, options.bandwidths, getDistance, 1e-6, 40)!;
  expect(expected).not.toBeNull();
  expect(result.codes.every(code => code === 0)).toBe(true);
  let worst = 0;
  result.coefficients.forEach((value, index) => {
    worst = Math.max(worst, Math.abs(value - expected[index]));
  });
  expect(worst).toBeLessThan(tolerance);
  return result;
}

it('multiscale GWR fits 14 predictors (above the old limit of 11) like a float64 backfit', async () => {
  const scene = createGaussianScene(350, 14, 41);
  await expectMultiscaleMatchesReference(
    scene,
    {bandwidths: Array.from({length: 15}, (_, index) => 55 + 5 * index)},
    planar(scene),
    5e-3
  );
});

it('multiscale GWR fits the largest supported predictor count (24)', async () => {
  const scene = createGaussianScene(300, 24, 43);
  await expectMultiscaleMatchesReference(
    scene,
    {bandwidths: Array.from({length: 25}, (_, index) => 60 + 3 * index)},
    planar(scene),
    2e-2
  );
});

it('multiscale GWR with great-circle distances matches a float64 haversine backfit', async () => {
  const scene = createGaussianScene(300, 2, 47, 100, true);
  const getDistance = (focus: number, row: number) =>
    getHaversineDistance(
      scene.positions[2 * focus],
      scene.positions[2 * focus + 1],
      scene.positions[2 * row],
      scene.positions[2 * row + 1]
    );
  await expectMultiscaleMatchesReference(
    scene,
    {
      bandwidths: [500000, 400000, 450000],
      spatialContext: {
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle',
        units: 'meters'
      } as GPUSpatialContext
    },
    getDistance,
    5e-3
  );
});

it('multiscale GWR automatically indexes clustered coordinates like the forced dense path', async () => {
  const scene = createGaussianScene(1500, 3, 53);
  // Two distant clusters exercise empty grid cells and short support ranges.
  for (let row = 0; row < scene.rowCount; row++) {
    if (row % 2) {
      scene.positions[2 * row] += 400;
      scene.positions[2 * row + 1] += 200;
    }
  }
  const bandwidths = [30, 20, 25, 18];
  const candidates = {count: 3, values: [20, 30, 45, 15, 20, 30, 18, 25, 35, 14, 18, 28]};
  for (const selection of [undefined, candidates]) {
    const options = {bandwidths, candidates: selection};
    const dense = await runMultiscale(scene, {...options, indexGridSize: false});
    const automatic = await runMultiscale(scene, options);
    expect(automatic.codes.every(code => code === 0)).toBe(true);
    expect(automatic.codes).toEqual(dense.codes);
    expect(automatic.selected).toEqual(dense.selected);
    expectCloseTo(automatic.coefficients, dense.coefficients, 3e-3);
    const residuals = automatic.fitted.map((value, row) => scene.response[row] - value);
    const denseResiduals = dense.fitted.map((value, row) => scene.response[row] - value);
    expectCloseTo(residuals, denseResiduals, 3e-3);
  }
});

it('multiscale GWR with a grid index fits 17000 rows (above the old limit of 16384)', async () => {
  const scene = createGaussianScene(17000, 2, 59);
  const options = {bandwidths: [8, 6, 6], maximumIterations: 20};
  const fine = await runMultiscale(scene, {...options, indexGridSize: [64, 64]});
  const coarse = await runMultiscale(scene, {...options, indexGridSize: [20, 20]});
  expect(fine.codes.every(code => code === 0)).toBe(true);
  expect(fine.coefficients.every(Number.isFinite)).toBe(true);
  fine.coefficients.forEach((value, index) =>
    expect(Math.abs(value - coarse.coefficients[index])).toBeLessThan(2e-3)
  );
  // The surfaces recover the generating intercept 1 + x / 100 to within the noise.
  let squaredError = 0;
  for (let row = 0; row < scene.rowCount; row++) {
    squaredError += (fine.coefficients[row * 3] - (1 + scene.positions[2 * row] / 100)) ** 2;
  }
  expect(Math.sqrt(squaredError / scene.rowCount)).toBeLessThan(0.15);
});

function expectCloseTo(actual: number[], expected: number[], tolerance: number) {
  expect(actual.length).toBe(expected.length);
  let worst = 0;
  actual.forEach((value, index) => {
    if (Number.isNaN(value) || Number.isNaN(expected[index])) {
      expect(Number.isNaN(value) && Number.isNaN(expected[index]), `entry ${index}`).toBe(true);
    } else {
      worst = Math.max(worst, Math.abs(value - expected[index]));
    }
  });
  expect(worst).toBeLessThan(tolerance);
}

it('multiscale GWR parallel convergence reports the sweep count of the float64 backfit', async () => {
  const scene = createGaussianScene(1500, 3, 53);
  const bandwidths = [30, 20, 25, 18];
  // A loose tolerance keeps the decision far from float32 noise, so the counts agree exactly.
  for (const tolerance of [1e-2, 1e-3, 1e-4]) {
    const expected = fitMultiscaleBackfitOnCPU(scene, bandwidths, planar(scene), tolerance, 40)!;
    expect(expected.converged).toBe(true);
    const result = await runMultiscale(scene, {
      bandwidths,
      tolerance,
      indexGridSize: [24, 24],
      maximumIterations: 40
    });
    expect(
      result.codes.every(code => code === 0),
      `tolerance ${tolerance}`
    ).toBe(true);
    expect(result.iterationCount, `tolerance ${tolerance}`).toBe(expected.iterationCount);
  }
});

it('multiscale GWR parallel convergence stops at the sweep that meets the tolerance (120000 rows, grid)', async () => {
  const scene = createGaussianScene(120000, 2, 61);
  const options = {bandwidths: [3, 2.5, 2.5], tolerance: 1e-4, indexGridSize: [128, 128] as const};
  const converged = await runMultiscale(scene, {...options, maximumIterations: 20});
  expect(converged.codes.every(code => code === 0)).toBe(true);
  expect(converged.coefficients.every(Number.isFinite)).toBe(true);
  expect(converged.iterationCount).toBeGreaterThan(1);
  expect(converged.iterationCount).toBeLessThan(20);
  // One sweep fewer is incomplete (status 3, NaN coefficients); the decision is exact, not noisy.
  const limited = await runMultiscale(scene, {
    ...options,
    maximumIterations: converged.iterationCount - 1
  });
  expect(limited.codes.every(code => code === 3)).toBe(true);
  expect(limited.iterationCount).toBe(converged.iterationCount - 1);
  // The same limit as the converged count reproduces the converged surfaces.
  const exact = await runMultiscale(scene, {
    ...options,
    maximumIterations: converged.iterationCount
  });
  expect(exact.iterationCount).toBe(converged.iterationCount);
  expectCloseTo(exact.coefficients, converged.coefficients, 1e-6);
}, 120000);

it('generalized GWR adaptive neighbor selection on the grid matches the dense scan', async () => {
  const scene = createPoissonScene(3000, 2, 71);
  for (const leaveOneOut of [false, true]) {
    const options = {bandwidth: 40, bandwidthMode: 'adaptive' as const, leaveOneOut};
    const dense = await runGeneralized(scene, options);
    const grid = await runGeneralized(scene, {...options, indexGridSize: [24, 24]});
    expect(grid.codes).toEqual(dense.codes);
    expect(grid.codes.filter(code => code === 0).length).toBeGreaterThan(2500);
    expectCloseTo(grid.coefficients, dense.coefficients, 5e-3);
  }
});

it('generalized GWR Gaussian cutoff on the grid stays close to the exact Gaussian', async () => {
  const scene = createPoissonScene(3000, 2, 73);
  const options = {bandwidth: 6, kernel: 'gaussian' as const};
  const exact = await runGeneralized(scene, options);
  const cutoff = await runGeneralized(scene, {
    ...options,
    gaussianCutoff: 5,
    indexGridSize: [24, 24]
  });
  expect(cutoff.codes).toEqual(exact.codes);
  expectCloseTo(cutoff.coefficients, exact.coefficients, 5e-3);
  // Adaptive bandwidth with a cutoff uses rings for the neighbor distance and cells for weights.
  const adaptive = {bandwidth: 40, bandwidthMode: 'adaptive' as const, kernel: 'gaussian' as const};
  const adaptiveExact = await runGeneralized(scene, adaptive);
  const adaptiveGrid = await runGeneralized(scene, {
    ...adaptive,
    gaussianCutoff: 5,
    indexGridSize: [24, 24]
  });
  expect(adaptiveGrid.codes).toEqual(adaptiveExact.codes);
  expectCloseTo(adaptiveGrid.coefficients, adaptiveExact.coefficients, 5e-3);
});

it('multiscale GWR Gaussian cutoff on the grid stays close to the exact Gaussian', async () => {
  const scene = createGaussianScene(2000, 2, 79);
  const options = {bandwidths: [8, 6, 6], kernel: 'gaussian' as const, maximumIterations: 40};
  const exact = await runMultiscale(scene, options);
  const cutoff = await runMultiscale(scene, {
    ...options,
    gaussianCutoff: 5,
    indexGridSize: [24, 24]
  });
  expect(cutoff.codes).toEqual(exact.codes);
  expect(cutoff.iterationCount).toBe(exact.iterationCount);
  expectCloseTo(cutoff.coefficients, exact.coefficients, 5e-3);
});
