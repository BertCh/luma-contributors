// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {createGPUCopyCommandNode} from '@luma.gl/graph/gpu-core';
import {GPURasterEmbeddingSimilarity} from '../../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-similarity';
import {GPURasterEmbeddingCovariance} from '../../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-covariance';
import {
  GPURasterEmbeddingProjection,
  getGPURasterEmbeddingProjectionParameterValues
} from '../../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-projection';
import {GPURasterEmbeddingLinearProbe} from '../../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-linear-probe';
import {GPURasterEmbeddingKMeans} from '../../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-kmeans';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createRasterTimeSeriesHarness} from '../raster-time-series/raster-time-series-test-utils';

for (const fixture of [
  {
    name: 'empty zone',
    queries: [-1, NaN],
    metric: 'cosine' as const,
    threshold: 0,
    scores: [-1, NaN],
    maximum: -1,
    winner: 0
  },
  {
    name: 'no eligible zone',
    queries: [NaN, NaN],
    metric: 'cosine' as const,
    threshold: 0,
    scores: [NaN, NaN],
    maximum: NaN,
    winner: 0xffffffff
  },
  {
    name: 'negative dot',
    queries: [-3.3e38, -3.2e38],
    metric: 'dot' as const,
    threshold: -3.1e38,
    scores: Array.from(new Float32Array([-3.3e38, -3.2e38])),
    maximum: Math.fround(-3.2e38),
    winner: 1
  },
  {
    name: 'overflowed dot score',
    queries: [2 ** 64, 1],
    metric: 'dot' as const,
    threshold: 0,
    scores: [Infinity, 2 ** 64],
    maximum: 2 ** 64,
    winner: 1
  },
  {
    name: 'zero vectors',
    queries: [0, 0],
    metric: 'cosine' as const,
    threshold: 2,
    scores: [1, 1],
    maximum: 1,
    winner: 0
  }
]) {
  it(`WGS-01 similarity all four outputs: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRasterTimeSeriesHarness(device);
    const result = await harness.run(
      new GPURasterEmbeddingSimilarity({
        pixelCount: 1,
        dimensions: 1,
        queryCount: 2,
        metric: fixture.metric,
        vectors: harness.floats([
          fixture.name === 'zero vectors'
            ? 0
            : fixture.name === 'overflowed dot score'
              ? 2 ** 64
              : 1
        ]),
        queries: harness.floats(fixture.queries),
        parameters: harness.floats([fixture.threshold]),
        output: {
          similarity: harness.output('scores', 'float32', 2),
          maximum: harness.output('maximum', 'float32', 1),
          bestQuery: harness.output('winner', 'uint32', 1),
          matches: harness.output('matches', 'uint32', 1)
        }
      })
    );
    expect(result['scores']).toEqual(fixture.scores);
    expect(result['maximum']).toEqual([fixture.maximum]);
    expect(result['winner']).toEqual([fixture.winner]);
    expect(result['matches']).toEqual([fixture.name === 'overflowed dot score' ? 1 : 0]);
  });
}

it('FLT-11 covariance retains low contrast before and after host translation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Independent closed form: sample variance of 128 copies of each endpoint is
  // (256/255)*(f32(.1000001)-f32(.1))^2/4 = 2.3545435753619005e-15.
  for (const translated of [false, true]) {
    const harness = createRasterTimeSeriesHarness(device);
    const values = Array.from(
      {length: 256},
      (_, index) => Math.fround(index % 2 ? 0.1000001 : 0.1) - (translated ? Math.fround(0.1) : 0)
    );
    const result = await harness.run(
      new GPURasterEmbeddingCovariance({
        pixelCount: 256,
        dimensions: 1,
        vectors: harness.floats(values),
        output: {
          mean: harness.output('mean', 'float32', 1),
          covariance: harness.output('covariance', 'float32', 1)
        }
      })
    );
    expect(result['mean']).toEqual([translated ? 4.842877388000488e-8 : 0.1000000536441803]);
    expect(Number.isFinite(result['covariance'][0])).toBe(true);
    expect(Math.abs(result['covariance'][0] - 2.3545435753619005e-15)).toBeLessThan(2e-20);
  }
});

it.each([
  1, -1
])('t1 covariance preserves anchor centering across adjacent binades (sign %i)', async sign => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRasterTimeSeriesHarness(device);
  // Two copies of each endpoint: delta = 3*2^-24, sample variance delta^2/3 = 3*2^-48.
  // Exact mean is sign*(1+2^-25), which rounds to sign*1 in float32.
  const lower = sign * (1 - 2 ** -24);
  const upper = sign * (1 + 2 ** -23);
  const result = await harness.run(
    new GPURasterEmbeddingCovariance({
      pixelCount: 4,
      dimensions: 1,
      vectors: harness.floats([lower, upper, lower, upper]),
      output: {
        mean: harness.output('mean', 'float32', 1),
        covariance: harness.output('covariance', 'float32', 1)
      }
    })
  );
  expect(result['mean']).toEqual([sign]);
  expect(result['covariance']).toHaveLength(1);
  expect(Number.isFinite(result['covariance'][0])).toBe(true);
  expect(Math.abs(result['covariance'][0] - 1.0658141036401503e-14)).toBeLessThan(2e-20);
});

for (const symmetric of [true, false]) {
  it(`WGS-06 wide finite projection stretch retains neutral RGB (${symmetric ? 'symmetric' : 'positive'})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRasterTimeSeriesHarness(device);
    const bound = 2 ** 127;
    const result = await harness.run(
      new GPURasterEmbeddingProjection({
        pixelCount: 1,
        dimensions: 3,
        componentCount: 3,
        vectors: harness.floats(symmetric ? [0, 0, 0] : [bound / 2, bound / 2, bound / 2]),
        mean: harness.floats([0, 0, 0]),
        components: harness.floats([1, 0, 0, 0, 1, 0, 0, 0, 1]),
        parameters: harness.floats(
          getGPURasterEmbeddingProjectionParameterValues({
            stretchRanges: [
              [symmetric ? -bound : 0, bound],
              [symmetric ? -bound : 0, bound],
              [symmetric ? -bound : 0, bound]
            ]
          })
        ),
        output: {
          projected: harness.output('scores', 'float32', 3),
          rgb: harness.output('rgb', 'float32', 3)
        }
      })
    );
    expect(result['scores']).toEqual(symmetric ? [0, 0, 0] : [bound / 2, bound / 2, bound / 2]);
    expect(result['rgb']).toEqual([0.5, 0.5, 0.5]);
  });
}

it('STA-11 explicit and graph-owned KMeans stopping views report the same budget-one iterate', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const explicit of [false, true]) {
    const harness = createRasterTimeSeriesHarness(device);
    const statusRead = harness.output('status', 'uint32', 3);
    const contributor = new GPURasterEmbeddingKMeans({
      pixelCount: 5,
      dimensions: 1,
      clusterCount: 2,
      maximumIterations: 1,
      vectors: harness.floats([0, 0, 10, 10, 10]),
      output: {
        labels: harness.output('labels', 'uint32', 5),
        centroids: harness.output('centroids', 'float32', 2),
        counts: harness.output('counts', 'uint32', 2),
        ...(explicit ? {status: statusRead} : {})
      }
    });
    const source = contributor.getResult(harness.graph).status;
    harness.graph.add(contributor);
    if (!explicit)
      harness.graph.add(
        createGPUCopyCommandNode({
          id: 'read-stopping',
          resources: [
            {buffer: source, usage: 'copy-source'},
            {buffer: statusRead, usage: 'copy-destination'}
          ],
          compile: () => ({
            encode: ({commandEncoder, getBuffer}) =>
              commandEncoder.copyBufferToBuffer({
                sourceBuffer: getBuffer(source),
                sourceOffset: source.byteOffset,
                destinationBuffer: getBuffer(statusRead),
                destinationOffset: statusRead.byteOffset,
                size: 12
              })
          })
        })
      );
    const result = await harness.run([]);
    expect(result['labels']).toEqual([0, 0, 1, 1, 1]);
    expect(result['counts']).toEqual([2, 3]);
    expect(result['centroids']).toEqual([0, 10]);
    expect(result['status']).toEqual([1, 5, 0]);
  }
});

it('WGS-15 reports both near-boundary float32 permutations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRasterTimeSeriesHarness(device);
  const vectors = harness.floats([0.5, 2 ** -26, -0.5, 0.5, -0.5, 2 ** -26]);
  const weights = harness.floats([0, 0, 0, 0.5, 0.5, 0.5]);
  const result = await harness.run([
    ...[-1, 1].map(
      direction =>
        new GPURasterEmbeddingLinearProbe({
          id: `margin-control-${direction}`,
          pixelCount: 2,
          dimensions: 3,
          classCount: 2,
          vectors,
          weights,
          bias: harness.floats([direction * 2 ** -20, 0]),
          output: {
            labels: harness.output(direction < 0 ? 'below-labels' : 'above-labels', 'uint32', 2)
          }
        })
    ),
    new GPURasterEmbeddingSimilarity({
      pixelCount: 2,
      dimensions: 3,
      queryCount: 1,
      metric: 'dot',
      vectors,
      queries: harness.floats([0.5, 0.5, 0.5]),
      parameters: harness.floats([2 ** -28]),
      output: {
        similarity: harness.output('scores', 'float32', 2),
        matches: harness.output('matches', 'uint32', 2)
      }
    }),
    new GPURasterEmbeddingLinearProbe({
      pixelCount: 2,
      dimensions: 3,
      classCount: 2,
      vectors,
      weights: harness.floats([0, 0, 0, 0.5, 0.5, 0.5]),
      bias: harness.floats([2 ** -28, 0]),
      output: {labels: harness.output('labels', 'uint32', 2)}
    })
  ]);
  for (const name of ['scores', 'matches', 'labels', 'below-labels', 'above-labels'])
    expect(result[name]).toHaveLength(2);
  expect(result['below-labels']).toEqual([1, 1]);
  expect(result['above-labels']).toEqual([0, 0]);
  // Exact arithmetic gives score 2^-27, match 1, class 1 for both rows.
  // Float32 source-order accumulation gives 0 then 2^-27; backend reassociation
  // may recover the tiny term. Both documented outcomes are within the decision margin.
  for (let row = 0; row < 2; row++) {
    expect([0, 2 ** -27]).toContain(result['scores'][row]);
    expect(result['matches'][row]).toBe(Number(result['scores'][row] >= 2 ** -28));
    expect([0, 1]).toContain(result['labels'][row]);
  }
});

for (const fixture of [
  {pixelCount: 262144, mean: 2.670288040462765e-6, variance: 1.8692015964916214e-6},
  {pixelCount: 1048576, mean: 6.675720101156912e-7, variance: 4.6730039912290535e-7}
]) {
  it(`B1 sparse covariance retains public mean (${fixture.pixelCount} pixels)`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRasterTimeSeriesHarness(device);
    const values = new Float32Array(fixture.pixelCount);
    values[0] = 0.7;
    // Closed form for one stored value a and n-1 zeros: mean=a/n, sample variance=a*a/n.
    const result = await harness.run(
      new GPURasterEmbeddingCovariance({
        pixelCount: fixture.pixelCount,
        dimensions: 1,
        vectors: harness.floats(values),
        output: {
          mean: harness.output('mean', 'float32', 1),
          covariance: harness.output('covariance', 'float32', 1)
        }
      })
    );
    expect(result['mean']).toEqual([fixture.mean]);
    expect(result['covariance']).toHaveLength(1);
    expect(Number.isFinite(result['covariance'][0])).toBe(true);
    expect(Math.abs(result['covariance'][0] - fixture.variance)).toBeLessThan(
      fixture.variance * 1e-5
    );
  });
}

it('S2 all-masked covariance leaves no usable anchor or statistics', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRasterTimeSeriesHarness(device);
  const result = await harness.run(
    new GPURasterEmbeddingCovariance({
      pixelCount: 4096,
      dimensions: 3,
      vectors: harness.floats(new Float32Array(4096 * 3)),
      mask: harness.integers(new Uint32Array(4096)),
      output: {
        mean: harness.output('mean', 'float32', 3),
        covariance: harness.output('covariance', 'float32', 9),
        usableCount: harness.output('count', 'uint32', 1)
      }
    })
  );
  expect(result['mean']).toEqual([NaN, NaN, NaN]);
  expect(result['covariance']).toEqual(new Array(9).fill(NaN));
  expect(result['count']).toEqual([0]);
});
