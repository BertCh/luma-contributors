// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPURasterEmbeddingDequantize,
  GPURasterEmbeddingQuantize
} from '@luma-contributors/analysis/raster-analysis';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRasterTimeSeriesHarness} from '../raster-time-series/raster-time-series-test-utils';
import reference from '../../oracles/fixtures/emb01-embedding-codecs-numpy.json';

async function getHarness() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return createRasterTimeSeriesHarness(device);
}

function restoreNonFinite(values: (number | null)[], kinds: string[]): number[] {
  return values.map((value, index) => {
    if (value !== null) return value;
    if (kinds[index] === 'positiveInfinity') return Number.POSITIVE_INFINITY;
    if (kinds[index] === 'negativeInfinity') return Number.NEGATIVE_INFINITY;
    return Number.NaN;
  });
}

function expectVectors(actual: number[], expected: (number | null)[]) {
  expect(actual).toHaveLength(expected.length);
  expected.forEach((value, index) => {
    if (value === null) expect(actual[index], `component ${index}`).toBeNaN();
    else
      expect(Math.abs(actual[index] - value), `component ${index}`).toBeLessThanOrEqual(
        Math.abs(value) * 3e-6 + 1e-10
      );
  });
}

for (const format of ['tessera-int8', 'float16'] as const) {
  for (const normalize of [false, true]) {
    it(`EMB-01 ${format}, normalize=${normalize}, matches NumPy over masked pixels and padded words`, async () => {
      const harness = await getHarness();
      const fixture = format === 'float16' ? reference.half : reference.tessera;
      const result = await harness.run(
        new GPURasterEmbeddingDequantize({
          pixelCount: reference.pixelCount,
          dimensions: reference.dimensions,
          format,
          normalize,
          quantized: harness.integers(fixture.words),
          ...(format === 'tessera-int8'
            ? {
                scales: harness.floats(
                  restoreNonFinite(reference.tessera.scales, reference.tessera.scaleKinds)
                )
              }
            : {}),
          mask: harness.integers(reference.mask),
          output: {
            vectors: harness.output(
              'vectors',
              'float32',
              reference.pixelCount * reference.dimensions
            ),
            validity: harness.output('validity', 'uint32', reference.pixelCount)
          }
        })
      );
      expect(result['validity']).toEqual(fixture.validity);
      expectVectors(result['vectors'], normalize ? fixture.normalized : fixture.vectors);
    });
  }
}

for (const withValidity of [false, true]) {
  it(`EMB-01 AlphaEarth encoder matches NumPy, including ties and clipping (validity=${withValidity})`, async () => {
    const harness = await getHarness();
    const result = await harness.run(
      new GPURasterEmbeddingQuantize({
        pixelCount: reference.pixelCount,
        dimensions: reference.dimensions,
        vectors: harness.floats(
          restoreNonFinite(reference.encode.vectors, reference.encode.valueKinds)
        ),
        mask: harness.integers(reference.mask),
        output: {
          quantized: harness.output('quantized', 'uint32', reference.encode.words.length),
          ...(withValidity
            ? {validity: harness.output('validity', 'uint32', reference.pixelCount)}
            : {})
        }
      })
    );
    expect(result['quantized']).toEqual(reference.encode.words);
    if (withValidity) expect(result['validity']).toEqual(reference.encode.validity);
    // 201 components: the three unused bytes of the last word are deterministically zero.
    expect(result['quantized'].at(-1)! >>> 8).toBe(0);
  });
}

it('EMB-01 every valid AlphaEarth byte round-trips through decode and encode', async () => {
  const harness = await getHarness();
  const words = Uint32Array.from({length: 64}, (_, word) => {
    let packed = 0;
    for (let lane = 0; lane < 4; lane++) packed |= ((word * 4 + lane - 128) & 255) << (lane * 8);
    return packed >>> 0;
  });
  const vectors = harness.output('vectors', 'float32', 256);
  const validity = harness.output('validity', 'uint32', 256);
  const result = await harness.run([
    new GPURasterEmbeddingDequantize({
      pixelCount: 256,
      dimensions: 1,
      quantized: harness.integers(words),
      output: {vectors, validity}
    }),
    new GPURasterEmbeddingQuantize({
      pixelCount: 256,
      dimensions: 1,
      vectors,
      mask: validity,
      output: {quantized: harness.output('quantized', 'uint32', 64)}
    })
  ]);
  expect(result['quantized']).toEqual(Array.from(words));
});

it('EMB-01 binary16 decode ignores an unused NaN half and preserves finite signed values', async () => {
  const harness = await getHarness();
  const result = await harness.run(
    new GPURasterEmbeddingDequantize({
      pixelCount: 1,
      dimensions: 3,
      format: 'float16',
      quantized: harness.integers([0x80010001, 0x7e008000]),
      output: {vectors: harness.output('vectors', 'float32', 3)}
    })
  );
  expect(result['vectors'][0]).toBe(2 ** -24);
  expect(result['vectors'][1]).toBe(-(2 ** -24));
  expect(result['vectors'][2] === 0).toBe(true);
});

it('EMB-01 all-masked vectors encode to NoData and decode to NaN', async () => {
  const harness = await getHarness();
  const quantized = harness.output('quantized', 'uint32', 1);
  const result = await harness.run([
    new GPURasterEmbeddingQuantize({
      pixelCount: 1,
      dimensions: 3,
      vectors: harness.floats([0, 0, 0]),
      mask: harness.integers([0]),
      output: {quantized}
    }),
    new GPURasterEmbeddingDequantize({
      pixelCount: 1,
      dimensions: 3,
      quantized,
      output: {
        vectors: harness.output('vectors', 'float32', 3),
        validity: harness.output('validity', 'uint32', 1)
      }
    })
  ]);
  expect(result['quantized']).toEqual([0x00808080]);
  expect(result['validity']).toEqual([0]);
  expect(result['vectors'].every(Number.isNaN)).toBe(true);
});
