// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../analysis/src/testing/require-test-device';
import {
  createKeplerEmbeddingFixture,
  runKeplerRasterEmbedding,
  RASTER_EMBEDDING_OPERATIONS
} from '../src/analysis/raster-embeddings';
import {createKeplerEmbeddingDemoOptions} from '../src/analysis/catalog-raster-embeddings-bindings';
import {GPURasterEmbeddingKMeans} from '@luma-contributors/analysis/raster-analysis';
import {withCatalogGpuSession} from '../src/analysis/catalog-gpu-session';
import {
  runKeplerLagrangianDiagnostics,
  type KeplerLagrangianField
} from '../src/analysis/lagrangian-diagnostics';

async function getDevice() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return device;
}

test('K-means reports the exhausted 32-round budget and Kepler refuses unfinished labels', async () => {
  const device = await getDevice();
  const vectors = Float32Array.from({length: 256}, (_, index) => index);
  for (let index = 0; index < 8; index++) {
    const value = vectors[index];
    vectors[index] = vectors[32 * index];
    vectors[32 * index] = value;
  }
  // Evenly spaced rows now seed centres 0..7. Sequential float32 Lloyd rounds still change
  // ten labels in round 32; this status must prevent the adapter publishing a finished column.
  await withCatalogGpuSession(device, 'kmeans-exhausted-budget', async session => {
    const status = session.output('status', 'uint32', 3);
    session.graph.add(
      new GPURasterEmbeddingKMeans({
        pixelCount: 256,
        dimensions: 1,
        clusterCount: 8,
        maximumIterations: 32,
        vectors: session.input('vectors', vectors, 'float32'),
        output: {
          labels: session.output('labels', 'uint32', 256).view,
          centroids: session.output('centroids', 'float32', 8).view,
          counts: session.output('counts', 'uint32', 8).view,
          status: status.view
        }
      })
    );
    session.run();
    expect(await session.readUint32(status)).toEqual(Uint32Array.of(32, 10, 0));
  });
  await expect(
    runKeplerRasterEmbedding(
      device,
      {width: 256, height: 1, dimensions: 1, vectors},
      {operation: 'kmeans', clusterCount: 8}
    )
  ).rejects.toThrow(/K-means did not converge after 32 iterations \(10 changed labels\)/);
});

// Independent oracles: direct cosine and softmax/nearest-class definitions, unit-length overview,
// and analytic linear strain/translation. No CPU contributor is used as its own oracle.
test('all embedding UI operations dispatch existing contributors and preserve masked pixels', async () => {
  const device = await getDevice();
  const raster = createKeplerEmbeddingFixture();
  const encoded = await runKeplerRasterEmbedding(device, raster, {operation: 'quantize'});
  raster.quantized = encoded.columns['packed_word'] as Uint32Array;
  for (const operation of RASTER_EMBEDDING_OPERATIONS) {
    const result = await runKeplerRasterEmbedding(
      device,
      raster,
      createKeplerEmbeddingDemoOptions(raster, operation)
    );
    expect(result.contributors.length, operation).toBeGreaterThan(0);
    for (const values of Object.values(result.columns))
      expect(values.length, operation).toBe(result.rowCount);
    if (result.layout === 'pixels' && operation !== 'pyramid') {
      if (result.columns['label']) expect(result.columns['label'][63], operation).toBe(0xffffffff);
      if (result.columns['usable']) expect(result.columns['usable'][63], operation).toBe(0);
      if (result.columns['similarity'])
        expect(Number.isNaN(result.columns['similarity'][63]), operation).toBe(true);
    }
    if (operation === 'similarity' || operation === 'nearest-prototype') {
      for (let pixel = 0; pixel < 63; pixel++) {
        expect(result.columns['similarity'][pixel]).toBeCloseTo(
          Math.abs(raster.vectors[pixel * 4]),
          5
        );
        expect(result.columns['prototype'][pixel]).toBe(pixel % 8 < 4 ? 0 : 1);
      }
    }
    if (operation === 'knn' || operation === 'linear-probe')
      for (let pixel = 0; pixel < 63; pixel++) {
        expect(result.columns['label'][pixel]).toBe(pixel % 8 < 4 ? 0 : 1);
        const expected =
          operation === 'knn' ? 1 : 1 / (1 + Math.exp(-2 * Math.abs(raster.vectors[pixel * 4])));
        expect(result.columns['confidence'][pixel]).toBeCloseTo(expected, 5);
      }
    if (operation === 'nearest-pixels') {
      const scores = result.columns['score'];
      const rows = result.columns['pixel_row'];
      for (let query = 0; query < 2; query++)
        for (let rank = 0; rank < 8; rank++) {
          const index = query * 8 + rank;
          expect(rows[index]).not.toBe(63);
          expect(scores[index]).toBeCloseTo(
            raster.vectors[rows[index] * 4] * (query === 0 ? 1 : -1),
            5
          );
          if (rank > 0) expect(scores[index]).toBeLessThanOrEqual(scores[index - 1]);
        }
    }
    if (operation === 'pyramid') {
      expect(result.rowCount).toBe(16);
      expect(result.columns['valid_count'][15]).toBe(3);
      for (let pixel = 0; pixel < 16; pixel++)
        expect(
          Math.hypot(...[1, 2, 3, 4].map(band => result.columns[`band_${band}`][pixel]))
        ).toBeCloseTo(1, 5);
    }
    if (operation === 'zone-mean') {
      expect(result.columns['weight']).toEqual(new Float32Array([32, 31]));
      expect(result.columns['band_1'][0]).toBeGreaterThan(0.9);
      expect(result.columns['band_1'][1]).toBeLessThan(-0.9);
    }
    if (operation === 'covariance') {
      const means = Array.from(
        {length: 4},
        (_, band) =>
          Array.from({length: 63}, (_, pixel) => raster.vectors[pixel * 4 + band]).reduce(
            (sum, value) => sum + value,
            0
          ) / 63
      );
      for (let first = 0; first < 4; first++)
        for (let second = 0; second < 4; second++) {
          let sum = 0;
          for (let pixel = 0; pixel < 63; pixel++)
            sum +=
              (raster.vectors[pixel * 4 + first] - means[first]) *
              (raster.vectors[pixel * 4 + second] - means[second]);
          expect(result.columns['covariance'][first * 4 + second]).toBeCloseTo(sum / 62, 5);
        }
    }
  }
});

function strainField(rate: number): KeplerLagrangianField {
  const velocitiesStart = new Float32Array(8 * 8 * 2);
  for (let row = 0; row < 8; row++)
    for (let column = 0; column < 8; column++)
      velocitiesStart.set([rate * (column - 3.5), -rate * (row - 3.5)], (row * 8 + column) * 2);
  return {width: 8, height: 8, extent: [-4, -4, 1, 1], velocitiesStart};
}

test('flow-map and FTLE workflow matches analytic forward/backward linear strain', async () => {
  const device = await getDevice();
  for (const timeStep of [0.1, -0.1]) {
    const result = await runKeplerLagrangianDiagnostics(device, strainField(0.02), {
      timeStep,
      stepCount: 10
    });
    const multiplier = (1 + 0.02 * timeStep + (0.02 * timeStep) ** 2 / 2) ** 10;
    for (let index = 0; index < 36; index++) {
      expect(result.flowValidity[index]).toBe(1);
      expect(result.validity[index]).toBe(1);
      expect(result.positions[index * 2]).toBeCloseTo(result.seeds[index * 2] * multiplier, 5);
      expect(result.ftle[index]).toBeCloseTo(0.02, 5);
    }
  }
});

test('time-varying slices integrate their ramp, and lost particles invalidate FTLE', async () => {
  const device = await getDevice();
  const field = strainField(0);
  field.velocitiesEnd = strainField(0.04).velocitiesStart;
  for (const timeStep of [0.1, 0.01]) {
    const result = await runKeplerLagrangianDiagnostics(device, field, {timeStep, stepCount: 10});
    for (const value of result.ftle) expect(value).toBeCloseTo(0.02, 4);
  }
  const lost = strainField(0);
  lost.velocitiesStart.fill(100);
  const invalid = await runKeplerLagrangianDiagnostics(device, lost, {timeStep: 1, stepCount: 2});
  expect(Array.from(invalid.flowValidity).every(value => value === 0)).toBe(true);
  expect(Array.from(invalid.validity).every(value => value === 0)).toBe(true);
  expect(Array.from(invalid.ftle).every(Number.isNaN)).toBe(true);
});

test.each([1, 0])('PCA zero components use mid-grey for rank %s covariance', async rank => {
  const device = await getDevice();
  // Unbiased covariance of x=[-1,0,1] is 1. Its two-sigma interval [-2,2]
  // maps those scores to [0.25,0.5,0.75]. Two zero components map to 0.5.
  // The rank-zero control uses [-1,1] for all three zero components.
  const result = await runKeplerRasterEmbedding(
    device,
    {
      width: 3,
      height: 1,
      dimensions: 3,
      vectors: new Float32Array([-rank, 0, 0, 0, 0, 0, rank, 0, 0])
    },
    {operation: 'pca'}
  );
  expect(result.rowCount).toBe(3);
  for (const name of ['component_1', 'component_2', 'component_3', 'rgb_1', 'rgb_2', 'rgb_3'])
    expect(result.columns[name]).toHaveLength(3);
  const expectedRed = rank ? [0.25, 0.5, 0.75] : [0.5, 0.5, 0.5];
  for (let row = 0; row < 3; row++) {
    for (const channel of [0, 1, 2]) {
      const actual = result.columns[`rgb_${channel + 1}`][row];
      expect(Number.isFinite(actual)).toBe(true);
      expect(actual).toBeCloseTo(channel === 0 ? expectedRed[row] : 0.5, 6);
    }
  }
});

test('FLT-10 epoch slices match a zero-origin control through flow and FTLE', async () => {
  const device = await getDevice();
  const field = strainField(0);
  field.velocitiesEnd = strainField(0.04).velocitiesStart;
  const results = [];
  for (const origin of [0, 1700000000]) {
    results.push(
      await runKeplerLagrangianDiagnostics(device, field, {
        timeStep: 0.1,
        stepCount: 10,
        startTime: origin + 4,
        sliceStartTime: origin,
        sliceEndTime: origin + 10
      })
    );
  }
  expect(Array.from(results[1].positions)).toEqual(Array.from(results[0].positions));
  expect(Array.from(results[1].flowValidity)).toEqual(new Array(36).fill(1));
  expect(Array.from(results[1].validity)).toEqual(new Array(36).fill(1));
  // RK2 at midpoint time t+.05 for the linear strain a(t)=.004*t:
  // x' = x*(1+h*a(t+h/2)*(1+h*a(t)/2)); no contributor is used as oracle.
  let multiplier = 1;
  let contractingMultiplier = 1;
  for (let step = 0; step < 10; step++) {
    const time = 4 + step * 0.1;
    multiplier *= 1 + 0.1 * 0.004 * (time + 0.05) * (1 + 0.05 * 0.004 * time);
    contractingMultiplier *= 1 - 0.1 * 0.004 * (time + 0.05) * (1 - 0.05 * 0.004 * time);
  }
  for (const result of results) {
    expect(result.seeds).toHaveLength(72);
    expect(result.positions).toHaveLength(72);
    expect(result.flowValidity).toHaveLength(36);
    expect(result.ftle).toHaveLength(36);
    expect(result.validity).toHaveLength(36);
    for (let index = 0; index < 36; index++) {
      const position = result.positions[index * 2];
      const ftle = result.ftle[index];
      const contractingPosition = result.positions[index * 2 + 1];
      expect(Number.isFinite(contractingPosition)).toBe(true);
      expect(contractingPosition).toBeCloseTo(
        result.seeds[index * 2 + 1] * contractingMultiplier,
        5
      );
      expect(Number.isFinite(position)).toBe(true);
      expect(Number.isFinite(ftle)).toBe(true);
      expect(position).toBeCloseTo(result.seeds[index * 2] * multiplier, 5);
      expect(ftle).toBeCloseTo(Math.log(multiplier), 5);
    }
  }
});

for (const operation of ['covariance', 'pca', 'projection'] as const) {
  test(`sparse-band ${operation} adapter consumes the public covariance and mean`, async () => {
    const device = await getDevice();
    const pixelCount = 262144;
    const vectors = new Float32Array(pixelCount * 3);
    vectors[0] = 0.7;
    const result = await runKeplerRasterEmbedding(
      device,
      {width: 1024, height: 256, dimensions: 3, vectors},
      {operation}
    );
    // One stored a=float32(.7), n-1 zeros. Exact mean=a/n and unbiased covariance=a*a/n.
    // All other bands are zero. The leading eigenvector is [1,0,0], so scores are a-a/n then -a/n.
    if (operation === 'covariance') {
      const covariance = result.columns['covariance'];
      expect(covariance).toHaveLength(9);
      for (let index = 0; index < 9; index++) {
        expect(Number.isFinite(covariance[index])).toBe(true);
        expect(
          Math.abs(covariance[index] - (index === 0 ? 1.8692015964916214e-6 : 0))
        ).toBeLessThan(2e-11);
      }
    } else {
      const scores = result.columns['component_1'];
      for (const name of ['component_1', 'component_2', 'component_3', 'rgb_1', 'rgb_2', 'rgb_3'])
        expect(result.columns[name]).toHaveLength(pixelCount);
      for (const row of [0, 1, pixelCount - 1]) {
        expect(Number.isFinite(scores[row])).toBe(true);
        const expected = row === 0 ? 0.6999973058700562 : -2.670288040462765e-6;
        expect(Math.abs(scores[row] - expected)).toBeLessThan(row === 0 ? 1e-7 : 1e-10);
      }
      for (const name of ['component_2', 'component_3']) {
        for (const row of [0, 1, pixelCount - 1]) expect(result.columns[name][row]).toBe(0);
      }
    }
  });
}

test('B2 omitted absolute flow start retains the established ramp displacement', async () => {
  const field = strainField(0);
  field.extent = [-8, -8, 2, 2];
  field.velocitiesEnd = Float32Array.from({length: 128}, (_, index) => (index % 2 === 0 ? 0.8 : 0));
  const result = await runKeplerLagrangianDiagnostics(await getDevice(), field, {
    timeStep: 1,
    stepCount: 2,
    sliceStartTime: -4,
    sliceEndTime: 4
  });
  expect(result.positions).toHaveLength(72);
  expect(result.seeds).toHaveLength(72);
  expect(result.ftle).toHaveLength(36);
  expect(Array.from(result.flowValidity)).toEqual(new Array(36).fill(1));
  expect(Array.from(result.validity)).toEqual(new Array(36).fill(1));
  // The uniform ramp is v(t)=.1*(t+4). Midpoint steps at .5 and 1.5 move .45+.55=1 metre.
  for (let index = 0; index < 36; index++) {
    const positionX = result.positions[index * 2];
    const positionY = result.positions[index * 2 + 1];
    expect(Number.isFinite(positionX)).toBe(true);
    expect(Number.isFinite(positionY)).toBe(true);
    expect(Number.isFinite(result.ftle[index])).toBe(true);
    expect(positionX).toBeCloseTo(result.seeds[index * 2] + 1, 5);
    expect(positionY).toBeCloseTo(result.seeds[index * 2 + 1], 5);
    expect(result.ftle[index]).toBeCloseTo(0, 5);
  }
});
