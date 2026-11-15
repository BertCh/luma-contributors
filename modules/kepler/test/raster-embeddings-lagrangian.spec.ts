// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {
  createKeplerEmbeddingFixture,
  runKeplerRasterEmbedding,
  RASTER_EMBEDDING_OPERATIONS
} from '../src/analysis/raster-embeddings';
import {createKeplerEmbeddingDemoOptions} from '../src/analysis/catalog-raster-embeddings-bindings';
import {
  runKeplerLagrangianDiagnostics,
  type KeplerLagrangianField
} from '../src/analysis/lagrangian-diagnostics';

async function getDevice() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return device;
}

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
      const ids = result.columns['pixel_id'];
      for (let query = 0; query < 2; query++)
        for (let rank = 0; rank < 8; rank++) {
          const index = query * 8 + rank;
          expect(ids[index]).not.toBe(63);
          expect(scores[index]).toBeCloseTo(
            raster.vectors[ids[index] * 4] * (query === 0 ? 1 : -1),
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
