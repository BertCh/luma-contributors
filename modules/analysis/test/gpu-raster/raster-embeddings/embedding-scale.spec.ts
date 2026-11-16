// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {Buffer} from '@luma.gl/core';
import {
  GPURasterEmbeddingChange,
  GPURasterEmbeddingCovariance,
  GPURasterEmbeddingDequantize,
  GPURasterEmbeddingKNearestClassifier,
  GPURasterEmbeddingLinearProbe,
  GPURasterEmbeddingProjection,
  GPURasterEmbeddingQuantize,
  GPURasterEmbeddingSimilarity,
  GPURasterEmbeddingZoneMean
} from '@luma-contributors/analysis/raster-analysis';
import {
  createVectorView,
  readUint32,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createEmbeddingScaleHarness} from './embedding-scale-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRasterTimeSeriesHarness} from '../raster-time-series/raster-time-series-test-utils';

const NO_LABEL = 0xffffffff;

it('runs packed writable layouts and separate output stages in one physical buffer imported once', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createEmbeddingScaleHarness(device);
  const sentinel = 0xdeadbeef;
  const physical = device.createBuffer({
    data: new Uint32Array(768).fill(sentinel),
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const handle = harness.graph.importBuffer(
    {
      id: 'packed-writable-embedding-results',
      byteLength: physical.byteLength,
      usage: physical.usage
    },
    physical
  );
  const expected = new Map<number, {format: 'float32' | 'uint32'; value: number}>();
  function output<Format extends 'float32' | 'uint32'>(
    format: Format,
    offset: number,
    values: number[]
  ) {
    values.forEach((value, index) => expected.set(offset / 4 + index, {format, value}));
    return harness.graph.createDataView(handle, {
      format,
      length: values.length,
      byteOffset: offset
    });
  }
  const before = harness.packed('packed-before', 'float32', [1, 0]).view;
  const vectors = harness.packed('packed-vectors', 'float32', [1, 0, 3, 0]).view;
  harness.graph.add(
    new GPURasterEmbeddingQuantize({
      pixelCount: 1,
      dimensions: 2,
      vectors: before,
      output: {quantized: output('uint32', 0, [127]), validity: output('uint32', 256, [1])}
    })
  );
  harness.graph.add(
    new GPURasterEmbeddingChange({
      pixelCount: 1,
      dimensions: 2,
      before,
      after: harness.packed('packed-after', 'float32', [0, 1]).view,
      parameters: harness.packed('packed-threshold', 'float32', [0.5]).view,
      output: {
        similarity: output('float32', 512, [0]),
        angle: output('float32', 768, [Math.PI / 2]),
        magnitude: output('float32', 1024, [Math.SQRT2]),
        changed: output('uint32', 772, [1])
      }
    })
  );
  harness.graph.add(
    new GPURasterEmbeddingCovariance({
      pixelCount: 2,
      dimensions: 2,
      rangeCount: 1,
      vectors,
      output: {
        mean: output('float32', 1280, [2, 0]),
        covariance: output('float32', 1288, [2, 0, 0, 0]),
        usableCount: output('uint32', 1536, [2])
      }
    })
  );
  harness.graph.add(
    new GPURasterEmbeddingZoneMean({
      pixelCount: 2,
      dimensions: 2,
      zoneCount: 1,
      vectors,
      zoneRaster: harness.packed('packed-zones', 'uint32', [0, 0]).view,
      maximumComponentMagnitude: 3,
      output: {
        means: output('float32', 1792, [1, 0]),
        weights: output('float32', 2048, [2]),
        meanLengths: output('float32', 2052, [2])
      }
    })
  );
  const compiled = harness.graph.compile();
  try {
    submitGraph(device, compiled);
    const words = await readUint32(physical);
    const floats = await readFloat32(physical);
    for (let word = 0; word < words.length; word++) {
      const entry = expected.get(word);
      if (!entry) expect(words[word], `guard word ${word}`).toBe(sentinel);
      else if (entry.format === 'uint32') expect(words[word]).toBe(entry.value);
      else expectClose([floats[word]], [entry.value]);
    }
  } finally {
    compiled.destroy();
    harness.destroy();
    physical.destroy();
  }
});
async function getHarness() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return createRasterTimeSeriesHarness(device);
}
function expectClose(actual: number[], expected: number[], tolerance = 3e-5) {
  expect(actual).toHaveLength(expected.length);
  expected.forEach((value, index) => {
    if (Number.isNaN(value)) expect(actual[index], `element ${index}`).toBeNaN();
    else
      expect(Math.abs(actual[index] - value), `element ${index}`).toBeLessThanOrEqual(
        tolerance * Math.max(1, Math.abs(value))
      );
  });
}
function cosine(left: number[], right: number[]) {
  const leftLength = Math.hypot(...left);
  const rightLength = Math.hypot(...right);
  if (leftLength === 0 || rightLength === 0) return leftLength === rightLength ? 1 : 0;
  return left.reduce(
    (sum, value, band) => sum + (value / leftLength) * (right[band] / rightLength),
    0
  );
}
function nearestOracle(
  pixels: number[][],
  training: number[][],
  labels: number[],
  neighborCount: number,
  metric: 'euclidean' | 'cosine',
  mask?: number[]
) {
  return pixels.map((pixel, pixelIndex) => {
    const neighbors = training
      .flatMap((sample, sampleIndex) => {
        if (!sample.every(Number.isFinite)) return [];
        const distance =
          metric === 'cosine'
            ? 1 - cosine(pixel, sample)
            : pixel.reduce((sum, value, band) => sum + (value - sample[band]) ** 2, 0);
        return [{distance, sampleIndex}];
      })
      .sort(
        (first, second) =>
          first.distance - second.distance || first.sampleIndex - second.sampleIndex
      )
      .slice(0, neighborCount);
    if (!pixel.every(Number.isFinite) || mask?.[pixelIndex] === 0 || neighbors.length === 0)
      return {
        label: NO_LABEL,
        confidence: Number.NaN,
        probabilities: [Number.NaN, Number.NaN, Number.NaN]
      };
    const votes = [0, 0, 0];
    neighbors.forEach(neighbor => votes[labels[neighbor.sampleIndex]]++);
    const label = votes.indexOf(Math.max(...votes));
    return {
      label,
      confidence: votes[label] / neighbors.length,
      probabilities: votes.map(vote => vote / neighbors.length)
    };
  });
}

for (const lastBandOnly of [false, true]) {
  for (const dimensions of [192, 256, 384, 1024]) {
    it(`streams ${dimensions}-band chunked pixels through all seven contributors against CPU formulas (lastBandOnly=${lastBandOnly})`, async () => {
      const harness = await getHarness();
      const lengths = [4, 3];
      const pixelCount = 7;
      const pixels = Array.from({length: pixelCount}, (_, pixel) =>
        Array.from({length: dimensions}, (_, band) =>
          pixel === 5 && band === dimensions - 1
            ? Number.NaN
            : pixel === 2
              ? 0
              : lastBandOnly
                ? band === dimensions - 1
                  ? (pixel - 3) / 8
                  : 0
                : (((pixel + band) % 7) - 3) / 8
        )
      );
      const mask = [1, 1, 1, 0, 1, 1, 1];
      const after = pixels.map(pixel => pixel.map(value => -value));
      const queries = [
        Array.from({length: dimensions}, (_, band) =>
          band === (lastBandOnly ? dimensions - 1 : 0) ? 1 : 0
        ),
        new Array<number>(dimensions).fill(0)
      ];
      const weights = Array.from({length: 3}, (_, classIndex) =>
        Array.from({length: dimensions}, (_, band) =>
          lastBandOnly
            ? band === dimensions - 1
              ? classIndex - 1
              : 0
            : ((classIndex + 1) * ((band % 3) - 1)) / dimensions
        )
      );
      const bias = [0, -0.25, 0.25];
      const components = Array.from({length: 3}, (_, component) =>
        Array.from({length: dimensions}, (_, band) =>
          lastBandOnly
            ? Number(band === dimensions - 1) * (component === 1 ? -1 : 1)
            : Number(band === component)
        )
      );
      const input = (name: string, values: number[][]) =>
        createVectorView(name, 'float32', [
          harness.floats(values.slice(0, 4).flat()),
          harness.floats(values.slice(4).flat())
        ]);
      const integers = (name: string, values: number[]) =>
        createVectorView(name, 'uint32', [
          harness.integers(values.slice(0, 4)),
          harness.integers(values.slice(4))
        ]);
      const output = <Format extends 'float32' | 'uint32'>(
        name: string,
        format: Format,
        width = 1,
        lanes = 1
      ) =>
        createVectorView(
          name,
          format,
          lengths.map((length, chunk) =>
            harness.output(`${name}-${chunk}`, format, Math.ceil((length * width) / lanes))
          )
        );
      const vectors = input('pixels', pixels);
      const maskView = integers('mask', mask);
      const quantized = output('quantized', 'uint32', dimensions, 4);
      const decoded = output('decoded', 'float32', dimensions);
      const decodedValidity = output('decoded-validity', 'uint32');
      const base = {pixelCount, dimensions, vectors, mask: maskView};
      const training = [new Array<number>(dimensions).fill(0), ...pixels.slice(0, 2)];
      const trainingLabels = [0, 1, 2];
      const result = await harness.run([
        new GPURasterEmbeddingQuantize({...base, output: {quantized}}),
        new GPURasterEmbeddingDequantize({
          pixelCount,
          dimensions,
          quantized,
          output: {vectors: decoded, validity: decodedValidity}
        }),
        new GPURasterEmbeddingChange({
          pixelCount,
          dimensions,
          before: vectors,
          after: input('after', after),
          beforeMask: maskView,
          parameters: harness.floats([0]),
          output: {
            similarity: output('change-similarity', 'float32'),
            angle: output('angle', 'float32'),
            magnitude: output('magnitude', 'float32'),
            changed: output('changed', 'uint32')
          }
        }),
        new GPURasterEmbeddingSimilarity({
          ...base,
          queryCount: 2,
          queries: harness.floats(queries.flat()),
          parameters: harness.floats([0.5]),
          output: {
            similarity: output('similarity', 'float32', 2),
            maximum: output('maximum', 'float32'),
            bestQuery: output('best', 'uint32'),
            matches: output('matches', 'uint32')
          }
        }),
        new GPURasterEmbeddingLinearProbe({
          ...base,
          classCount: 3,
          weights: harness.floats(weights.flat()),
          bias: harness.floats(bias),
          output: {
            labels: output('probe-labels', 'uint32'),
            confidence: output('probe-confidence', 'float32'),
            probabilities: output('probe-probabilities', 'float32', 3)
          }
        }),
        new GPURasterEmbeddingProjection({
          ...base,
          componentCount: 3,
          mean: harness.floats(new Float32Array(dimensions)),
          components: harness.floats(components.flat()),
          parameters: harness.floats([-1, 1, -1, 1, -1, 1]),
          output: {projected: output('projected', 'float32', 3), rgb: output('rgb', 'float32', 3)}
        }),
        new GPURasterEmbeddingKNearestClassifier({
          ...base,
          trainingCount: 3,
          trainingVectors: harness.floats(training.flat()),
          trainingLabels: harness.integers(trainingLabels),
          classCount: 3,
          neighborCount: 2,
          output: {
            labels: output('knn-labels', 'uint32'),
            confidence: output('knn-confidence', 'float32'),
            probabilities: output('knn-probabilities', 'float32', 3)
          }
        })
      ]);
      const read = (name: string) => [...result[`${name}-0`], ...result[`${name}-1`]];
      const isValid = pixels.map(
        (pixel, index) => mask[index] !== 0 && pixel.every(Number.isFinite)
      );
      const expectedWords: number[] = [];
      const expectedDecoded: number[] = [];
      for (let pixel = 0; pixel < pixelCount; pixel++) {
        const codes = pixels[pixel].map(value =>
          isValid[pixel]
            ? Math.max(
                -127,
                Math.min(127, Math.round(Math.sign(value) * Math.sqrt(Math.abs(value)) * 127.5))
              )
            : -128
        );
        for (let band = 0; band < dimensions; band += 4)
          expectedWords.push(
            codes
              .slice(band, band + 4)
              .reduce((word, code, lane) => word | ((code & 255) << (lane * 8)), 0) >>> 0
          );
        expectedDecoded.push(
          ...codes.map(code => (code === -128 ? Number.NaN : Math.sign(code) * (code / 127.5) ** 2))
        );
      }
      expect(read('quantized')).toEqual(expectedWords);
      expect(read('decoded-validity')).toEqual(isValid.map(Number));
      expectClose(read('decoded'), expectedDecoded);
      expectClose(
        read('change-similarity'),
        pixels.map((pixel, index) => (isValid[index] ? cosine(pixel, after[index]) : Number.NaN))
      );
      expectClose(
        read('angle'),
        pixels.map((pixel, index) =>
          !isValid[index] ? Number.NaN : pixel.every(value => value === 0) ? 0 : Math.PI
        )
      );
      expectClose(
        read('magnitude'),
        pixels.map((pixel, index) =>
          isValid[index] ? Math.hypot(...pixel.map(value => value * 2)) : Number.NaN
        )
      );
      expect(read('changed')).toEqual(
        pixels.map((pixel, index) => Number(isValid[index] && cosine(pixel, after[index]) < 0))
      );
      const similarities = pixels.map((pixel, index) =>
        queries.map(query => (isValid[index] ? cosine(pixel, query) : Number.NaN))
      );
      expectClose(read('similarity'), similarities.flat());
      expectClose(
        read('maximum'),
        similarities.map(values => Math.max(...values))
      );
      expect(read('best')).toEqual(
        similarities.map((values, index) =>
          isValid[index] ? values.indexOf(Math.max(...values)) : NO_LABEL
        )
      );
      expect(read('matches')).toEqual(
        similarities.map((values, index) => Number(isValid[index] && Math.max(...values) >= 0.5))
      );
      const probabilities = pixels.map((pixel, index) => {
        if (!isValid[index]) return [Number.NaN, Number.NaN, Number.NaN];
        const logits = weights.map((weight, classIndex) =>
          weight.reduce((sum, value, band) => sum + value * pixel[band], bias[classIndex])
        );
        const exponentials = logits.map(logit => Math.exp(logit - Math.max(...logits)));
        const total = exponentials.reduce((sum, value) => sum + value, 0);
        return exponentials.map(value => value / total);
      });
      expectClose(read('probe-probabilities'), probabilities.flat());
      expect(read('probe-labels')).toEqual(
        probabilities.map((values, index) =>
          isValid[index] ? values.indexOf(Math.max(...values)) : NO_LABEL
        )
      );
      expectClose(
        read('probe-confidence'),
        probabilities.map(values => Math.max(...values))
      );
      const projected = pixels.map((pixel, index) =>
        isValid[index]
          ? lastBandOnly
            ? [pixel[dimensions - 1], -pixel[dimensions - 1], pixel[dimensions - 1]]
            : pixel.slice(0, 3)
          : [Number.NaN, Number.NaN, Number.NaN]
      );
      expectClose(read('projected'), projected.flat());
      expectClose(
        read('rgb'),
        projected.flat().map(value => (value + 1) / 2)
      );
      const nearest = nearestOracle(pixels, training, trainingLabels, 2, 'euclidean', mask);
      expect(read('knn-labels')).toEqual(nearest.map(pixel => pixel.label));
      expectClose(
        read('knn-confidence'),
        nearest.map(pixel => pixel.confidence)
      );
      expectClose(
        read('knn-probabilities'),
        nearest.flatMap(pixel => pixel.probabilities)
      );
    });
  }
}

for (const metric of ['euclidean', 'cosine'] as const) {
  for (const trainingCount of [257, 513, 4097]) {
    it(`tiled ${metric} classifier crosses training tiles and the former cap (${trainingCount}) against CPU top-k`, async () => {
      const harness = await getHarness();
      const pixels = [
        [1, 0],
        [0, 0],
        [-1, 0],
        [Number.NaN, 1],
        [1, 0],
        [0, 1]
      ];
      const mask = [1, 1, 1, 1, 0, 1];
      // Every earlier finite row is class 0 and orthogonal to pixel 0. The last row alone
      // has distance zero and class 2, so skipping a partial tile or truncating at 4096 flips it.
      const training = Array.from({length: trainingCount}, (_, index) =>
        index % 7 === 0 ? [Number.NaN, 0] : [0, 1]
      );
      training[trainingCount - 1] = [1, 0];
      const labels = new Array<number>(trainingCount).fill(0);
      labels[trainingCount - 1] = 2;
      const base = {
        pixelCount: 6,
        dimensions: 2,
        vectors: createVectorView('pixels', 'float32', [
          harness.floats(pixels.slice(0, 1).flat()),
          harness.floats(pixels.slice(1).flat())
        ]),
        mask: createVectorView('mask', 'uint32', [
          harness.integers(mask.slice(0, 1)),
          harness.integers(mask.slice(1))
        ]),
        trainingCount,
        trainingVectors: harness.floats(training.flat()),
        trainingLabels: harness.integers(labels),
        neighborCount: 1,
        classCount: 3,
        metric
      };
      const output = (prefix: string) => ({
        labels: createVectorView(prefix, 'uint32', [
          harness.output(`${prefix}-0`, 'uint32', 1),
          harness.output(`${prefix}-1`, 'uint32', 5)
        ]),
        confidence: createVectorView(prefix, 'float32', [
          harness.output(`${prefix}-confidence-0`, 'float32', 1),
          harness.output(`${prefix}-confidence-1`, 'float32', 5)
        ]),
        probabilities: createVectorView(prefix, 'float32', [
          harness.output(`${prefix}-probabilities-0`, 'float32', 3),
          harness.output(`${prefix}-probabilities-1`, 'float32', 15)
        ])
      });
      const contributors = [
        new GPURasterEmbeddingKNearestClassifier({
          ...base,
          id: 'tiled-classifier',
          output: output('tiled')
        })
      ];
      if (trainingCount <= 4096)
        contributors.push(
          new GPURasterEmbeddingKNearestClassifier({
            ...base,
            id: 'reference-classifier',
            searchStructure: 'brute-force',
            output: output('reference')
          })
        );
      const expected = nearestOracle(pixels, training, labels, 1, metric, mask);
      expect(expected.map(pixel => pixel.label)).toEqual([2, 0, 0, NO_LABEL, NO_LABEL, 0]);
      expect(expected[0]).toEqual({label: 2, confidence: 1, probabilities: [0, 0, 1]});
      expect(
        nearestOracle([pixels[0]], training.slice(0, -1), labels.slice(0, -1), 1, metric)[0].label
      ).toBe(0);
      const result = await harness.run(contributors);
      for (const prefix of trainingCount <= 4096 ? ['tiled', 'reference'] : ['tiled']) {
        expect([...result[`${prefix}-0`], ...result[`${prefix}-1`]]).toEqual(
          expected.map(pixel => pixel.label)
        );
        expectClose(
          [...result[`${prefix}-confidence-0`], ...result[`${prefix}-confidence-1`]],
          expected.map(pixel => pixel.confidence)
        );
        expectClose(
          [...result[`${prefix}-probabilities-0`], ...result[`${prefix}-probabilities-1`]],
          expected.flatMap(pixel => pixel.probabilities)
        );
      }
    });
  }
}

it('cosines, angles and representable magnitudes stay finite for large and tiny finite vectors', async () => {
  const harness = await getHarness();
  const pixels = [
    [1e20, 0],
    [1e-20, 0],
    [2e38, 0],
    [2e-38, 0],
    [0, 0]
  ];
  const before = harness.floats(pixels.flat());
  const after = harness.floats(
    [
      [0, 1e20],
      [0, 1e-20],
      [0, 2e38],
      [0, 2e-38],
      [0, 0]
    ].flat()
  );
  const result = await harness.run([
    new GPURasterEmbeddingChange({
      pixelCount: pixels.length,
      dimensions: 2,
      before,
      after,
      output: {
        similarity: harness.output('change', 'float32', pixels.length),
        angle: harness.output('angle', 'float32', pixels.length),
        magnitude: harness.output('magnitude', 'float32', pixels.length)
      }
    }),
    new GPURasterEmbeddingSimilarity({
      pixelCount: pixels.length,
      dimensions: 2,
      vectors: before,
      queries: harness.floats([2e38, 0]),
      queryCount: 1,
      output: {maximum: harness.output('similarity', 'float32', pixels.length)}
    }),
    new GPURasterEmbeddingDequantize({
      pixelCount: 2,
      dimensions: 2,
      format: 'tessera-int8',
      normalize: true,
      quantized: harness.integers([0x01000001]),
      scales: harness.floats([2e38, 2e-38]),
      output: {vectors: harness.output('decoded', 'float32', 4)}
    }),
    ...(['tiled', 'brute-force'] as const).map(
      searchStructure =>
        new GPURasterEmbeddingKNearestClassifier({
          id: `stable-${searchStructure}`,
          pixelCount: pixels.length,
          dimensions: 2,
          vectors: before,
          trainingCount: 2,
          trainingVectors: harness.floats([0, 2e38, 2e38, 0]),
          trainingLabels: harness.integers([0, 1]),
          metric: 'cosine',
          classCount: 2,
          neighborCount: 1,
          searchStructure,
          output: {labels: harness.output(searchStructure, 'uint32', pixels.length)}
        })
    )
  ]);
  expectClose(result['change'], [0, 0, 0, 0, 1]);
  expectClose(result['angle'], [Math.PI / 2, Math.PI / 2, Math.PI / 2, Math.PI / 2, 0]);
  for (let pixel = 0; pixel < pixels.length - 1; pixel++) {
    expect(result['magnitude'][pixel] / pixels[pixel][0]).toBeCloseTo(Math.SQRT2, 5);
  }
  expect(result['magnitude'][4]).toBe(0);
  expectClose(result['similarity'], [1, 1, 1, 1, 0]);
  expectClose(result['decoded'], [1, 0, 0, 1]);
  expect(result['tiled']).toEqual([1, 1, 1, 1, 0]);
  expect(result['brute-force']).toEqual([1, 1, 1, 1, 0]);
});

it('dispatches two million chunked pixels without a global packed binding', async () => {
  const harness = await getHarness();
  const chunkLength = 1_000_000;
  const input = harness.floats(new Float32Array(chunkLength).fill(0.25));
  const vectors = createVectorView('million-pixels', 'float32', [input, input]);
  const maximum = createVectorView('maximum', 'float32', [
    harness.output('first', 'float32', chunkLength),
    harness.output('second', 'float32', chunkLength)
  ]);
  const result = await harness.run(
    new GPURasterEmbeddingSimilarity({
      pixelCount: chunkLength * 2,
      dimensions: 1,
      vectors,
      queryCount: 1,
      queries: harness.floats([1]),
      output: {maximum}
    })
  );
  for (const name of ['first', 'second'])
    expect(result[name].every(value => Math.abs(value - 1) < 1e-6)).toBe(true);
});

for (const format of ['float16', 'tessera-int8'] as const) {
  for (const normalize of [false, true]) {
    it(`decodes chunked odd-band ${format} records and padded words (normalize=${normalize})`, async () => {
      const harness = await getHarness();
      const lengths = format === 'float16' ? [2, 3] : [4, 1];
      const lanes = format === 'float16' ? 2 : 4;
      const maskValues = [1, 0, 1, 1, 1];
      const scales = [0.5, 1, -2, 0, Number.NaN];
      let pixelBase = 0;
      const quantized = createVectorView(
        'packed',
        'uint32',
        lengths.map(length => {
          const words = new Uint32Array(Math.ceil((length * 3) / lanes));
          for (let element = 0; element < length * 3; element++) {
            const code =
              format === 'float16'
                ? [0x3c00, 0x4000, 0x4000][element % 3]
                : [-2, 0, 2][element % 3] & 255;
            words[Math.floor(element / lanes)] |= code << ((element % lanes) * (32 / lanes));
          }
          // Padding is not a component: a NaN half in the last unused lane is ignored.
          if (format === 'float16' && (length * 3) % 2) words[words.length - 1] |= 0x7e00 << 16;
          return harness.integers(words);
        })
      );
      const mask = createVectorView(
        'mask',
        'uint32',
        lengths.map(length => {
          const values = maskValues.slice(pixelBase, pixelBase + length);
          pixelBase += length;
          return harness.integers(values);
        })
      );
      pixelBase = 0;
      const scaleViews = createVectorView(
        'scales',
        'float32',
        lengths.map(length => {
          const values = scales.slice(pixelBase, pixelBase + length);
          pixelBase += length;
          return harness.floats(values);
        })
      );
      const result = await harness.run(
        new GPURasterEmbeddingDequantize({
          pixelCount: 5,
          dimensions: 3,
          format,
          normalize,
          quantized,
          mask,
          ...(format === 'tessera-int8' ? {scales: scaleViews} : {}),
          output: {
            vectors: createVectorView(
              'decoded',
              'float32',
              lengths.map((length, chunk) =>
                harness.output(`decoded-${chunk}`, 'float32', length * 3)
              )
            ),
            validity: createVectorView(
              'validity',
              'uint32',
              lengths.map((length, chunk) => harness.output(`validity-${chunk}`, 'uint32', length))
            )
          }
        })
      );
      const valid = maskValues.map(
        (value, pixel) => value !== 0 && (format === 'float16' || Number.isFinite(scales[pixel]))
      );
      const expected = valid.flatMap((value, pixel) => {
        if (!value) return [Number.NaN, Number.NaN, Number.NaN];
        const vector =
          format === 'float16' ? [1, 2, 2] : [-2 * scales[pixel], 0, 2 * scales[pixel]];
        const norm = Math.hypot(...vector);
        return normalize && norm > 0 ? vector.map(component => component / norm) : vector;
      });
      expectClose([...result['decoded-0'], ...result['decoded-1']], expected);
      expect([...result['validity-0'], ...result['validity-1']]).toEqual(valid.map(Number));
    });
  }
}

it('checks all seven contributors at uneven workgroup boundaries with 4/252/256-byte offsets and guarded outputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createEmbeddingScaleHarness(device);
  const lengths = [63, 64, 65, 127, 128, 129];
  const pixelCount = lengths.reduce((total, length) => total + length, 0);
  const pixels = Array.from({length: pixelCount}, (_, pixel) => {
    const code = pixel + 1;
    return [Math.fround((pixel % 2 ? -1 : 1) * (code / 1024) ** 2), -0.25, 0.25, 0.5];
  });
  const maskValues = pixels.map((_, pixel) => Number(pixel % 11 !== 0));
  const vectors = harness.rows('offset-vectors', 'float32', lengths, 4);
  vectors.write(pixels.flat());
  const after = harness.rows('offset-after', 'float32', lengths, 4);
  after.write(pixels.flatMap(pixel => pixel.map(value => value * 2)));
  const mask = harness.rows('offset-mask', 'uint32', lengths);
  mask.write(maskValues);
  const output = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    components = 1
  ) => harness.rows(name, format, lengths, components);
  const quantized = output('offset-quantized', 'uint32');
  const encodedValidity = output('offset-encoded-validity', 'uint32');
  const decoded = output('offset-decoded', 'float32', 4);
  const decodedValidity = output('offset-decoded-validity', 'uint32');
  const similarity = output('offset-similarity', 'float32');
  const maximum = output('offset-maximum', 'float32');
  const bestQuery = output('offset-best-query', 'uint32');
  const matches = output('offset-matches', 'uint32');
  const changeSimilarity = output('offset-change-similarity', 'float32');
  const angle = output('offset-angle', 'float32');
  const magnitude = output('offset-magnitude', 'float32');
  const changed = output('offset-changed', 'uint32');
  const projected = output('offset-projected', 'float32');
  const probeLabels = output('offset-probe-labels', 'uint32');
  const probeConfidence = output('offset-probe-confidence', 'float32');
  const probeProbabilities = output('offset-probe-probabilities', 'float32', 2);
  const classifierLabels = output('offset-classifier-labels', 'uint32');
  const classifierConfidence = output('offset-classifier-confidence', 'float32');
  const classifierProbabilities = output('offset-classifier-probabilities', 'float32', 2);
  const base = {pixelCount, dimensions: 4, vectors: vectors.view, mask: mask.view};
  harness.graph.add(
    new GPURasterEmbeddingQuantize({
      ...base,
      output: {quantized: quantized.view, validity: encodedValidity.view}
    })
  );
  harness.graph.add(
    new GPURasterEmbeddingDequantize({
      pixelCount,
      dimensions: 4,
      quantized: quantized.view,
      output: {vectors: decoded.view, validity: decodedValidity.view}
    })
  );
  harness.graph.add(
    new GPURasterEmbeddingSimilarity({
      ...base,
      queryCount: 1,
      queries: harness.packed('offset-query', 'float32', [1, 0, 0, 0]).view,
      parameters: harness.packed('offset-match-threshold', 'float32', [0]).view,
      output: {
        similarity: similarity.view,
        maximum: maximum.view,
        bestQuery: bestQuery.view,
        matches: matches.view
      }
    })
  );
  harness.graph.add(
    new GPURasterEmbeddingChange({
      pixelCount,
      dimensions: 4,
      before: vectors.view,
      after: after.view,
      beforeMask: mask.view,
      parameters: harness.packed('offset-change-threshold', 'float32', [0.5]).view,
      output: {
        similarity: changeSimilarity.view,
        angle: angle.view,
        magnitude: magnitude.view,
        changed: changed.view
      }
    })
  );
  harness.graph.add(
    new GPURasterEmbeddingProjection({
      ...base,
      componentCount: 1,
      mean: harness.packed('offset-mean', 'float32', [0, 0, 0, 0]).view,
      components: harness.packed('offset-component', 'float32', [1, 0, 0, 0]).view,
      output: {projected: projected.view}
    })
  );
  harness.graph.add(
    new GPURasterEmbeddingLinearProbe({
      ...base,
      classCount: 2,
      weights: harness.packed('offset-weights', 'float32', [-1, 0, 0, 0, 1, 0, 0, 0]).view,
      bias: harness.packed('offset-bias', 'float32', [0, 0]).view,
      output: {
        labels: probeLabels.view,
        confidence: probeConfidence.view,
        probabilities: probeProbabilities.view
      }
    })
  );
  harness.graph.add(
    new GPURasterEmbeddingKNearestClassifier({
      ...base,
      trainingCount: 2,
      neighborCount: 1,
      classCount: 2,
      trainingVectors: harness.packed(
        'offset-training-vectors',
        'float32',
        [-1, 0, 0, 0, 1, 0, 0, 0]
      ).view,
      trainingLabels: harness.packed('offset-training-labels', 'uint32', [0, 1]).view,
      output: {
        labels: classifierLabels.view,
        confidence: classifierConfidence.view,
        probabilities: classifierProbabilities.view
      }
    })
  );
  const compiled = harness.graph.compile();
  try {
    submitGraph(device, compiled);
    // Closed-form expected labels and probabilities depend on each distinct row's first value.
    const labels = pixels.map((pixel, index) =>
      maskValues[index] ? Number(pixel[0] > 0) : NO_LABEL
    );
    const expectedSimilarity = pixels.map((pixel, index) =>
      maskValues[index] ? pixel[0] / Math.hypot(...pixel) : Number.NaN
    );
    const probabilities = pixels.map((pixel, index) =>
      maskValues[index] ? 1 / (1 + Math.exp(-2 * pixel[0])) : Number.NaN
    );
    const codes = pixels.map((pixel, index) =>
      pixel.map(value =>
        maskValues[index]
          ? Math.max(
              -127,
              Math.min(127, Math.round(Math.sign(value) * Math.sqrt(Math.abs(value)) * 127.5))
            )
          : -128
      )
    );
    const expectedWords = codes.map(
      pixel => pixel.reduce((word, code, lane) => word | ((code & 255) << (lane * 8)), 0) >>> 0
    );
    expect(await quantized.read()).toEqual(expectedWords);
    expect(await encodedValidity.read()).toEqual(maskValues);
    expect(await decodedValidity.read()).toEqual(maskValues);
    expectClose(
      await decoded.read(),
      codes.flatMap(pixel =>
        pixel.map(code => (code === -128 ? Number.NaN : Math.sign(code) * (code / 127.5) ** 2))
      )
    );
    expectClose(await similarity.read(), expectedSimilarity);
    expectClose(await maximum.read(), expectedSimilarity);
    expect(await bestQuery.read()).toEqual(maskValues.map(value => (value ? 0 : NO_LABEL)));
    expect(await matches.read()).toEqual(labels.map(label => Number(label === 1)));
    expectClose(
      await changeSimilarity.read(),
      maskValues.map(value => (value ? 1 : Number.NaN))
    );
    expectClose(
      await angle.read(),
      maskValues.map(value => (value ? 0 : Number.NaN))
    );
    expectClose(
      await magnitude.read(),
      pixels.map((pixel, index) => (maskValues[index] ? Math.hypot(...pixel) : Number.NaN))
    );
    expect(await changed.read()).toEqual(new Array<number>(pixelCount).fill(0));
    expectClose(
      await projected.read(),
      pixels.map((pixel, index) => (maskValues[index] ? pixel[0] : Number.NaN))
    );
    expect(await probeLabels.read()).toEqual(labels);
    expectClose(
      await probeConfidence.read(),
      probabilities.map(value => Math.max(value, 1 - value))
    );
    expectClose(
      await probeProbabilities.read(),
      probabilities.flatMap(value => [1 - value, value])
    );
    expect(await classifierLabels.read()).toEqual(labels);
    expectClose(
      await classifierConfidence.read(),
      maskValues.map(value => (value ? 1 : Number.NaN))
    );
    expectClose(
      await classifierProbabilities.read(),
      labels.flatMap(label =>
        label === NO_LABEL ? [Number.NaN, Number.NaN] : [Number(label === 0), Number(label === 1)]
      )
    );
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

for (const metric of ['euclidean', 'cosine'] as const) {
  it(`re-encodes one compiled ${metric} classifier graph with changed masks/training and poisoned dispatch metadata`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createEmbeddingScaleHarness(device);
    const lengths = [1, 2];
    const vectors = harness.rows('replay-vectors', 'float32', lengths, 2);
    vectors.write([1, 0, 1, 0, 1, 0]);
    const mask = harness.rows('replay-mask', 'uint32', lengths);
    const training = harness.packed(
      'replay-training',
      'float32',
      new Array<number>(513 * 2).fill(0)
    );
    const labels = harness.packed(
      'replay-training-labels',
      'uint32',
      Array.from({length: 513}, (_, index) => Number(index === 512))
    );
    const outputs = (['tiled', 'brute-force'] as const).map(searchStructure => {
      const prefix = `replay-${searchStructure}`;
      const output = {
        labels: harness.rows(`${prefix}-labels`, 'uint32', lengths),
        confidence: harness.rows(`${prefix}-confidence`, 'float32', lengths),
        probabilities: harness.rows(`${prefix}-probabilities`, 'float32', lengths, 2)
      };
      harness.addWithMetadataPoison(
        new GPURasterEmbeddingKNearestClassifier({
          id: prefix,
          pixelCount: 3,
          dimensions: 2,
          vectors: vectors.view,
          mask: mask.view,
          trainingCount: 513,
          trainingVectors: training.view,
          trainingLabels: labels.view,
          neighborCount: 1,
          classCount: 2,
          metric,
          searchStructure,
          output: {
            labels: output.labels.view,
            confidence: output.confidence.view,
            probabilities: output.probabilities.view
          }
        })
      );
      return output;
    });
    const compiled = harness.graph.compile();
    try {
      for (const frame of [0, 1, 2]) {
        const trainingRows = Array.from({length: 513}, () =>
          frame === 0 ? [-1, 0] : [Number.NaN, 0]
        );
        if (frame === 0) trainingRows[512] = [1, 0];
        if (frame === 2) {
          trainingRows[0] = [1, 0];
          trainingRows[512] = [-1, 0];
        }
        training.write(trainingRows.flat());
        mask.write(
          [
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0]
          ][frame]
        );
        for (const output of outputs) {
          output.labels.reset();
          output.confidence.reset();
          output.probabilities.reset();
        }
        submitGraph(device, compiled);
        const expectedLabels = [
          [1, NO_LABEL, 1],
          [NO_LABEL, NO_LABEL, NO_LABEL],
          [0, 0, NO_LABEL]
        ][frame];
        for (const output of outputs) {
          expect(await output.labels.read()).toEqual(expectedLabels);
          expectClose(
            await output.confidence.read(),
            expectedLabels.map(label => (label === NO_LABEL ? Number.NaN : 1))
          );
          expectClose(
            await output.probabilities.read(),
            expectedLabels.flatMap(label =>
              label === NO_LABEL
                ? [Number.NaN, Number.NaN]
                : [Number(label === 0), Number(label === 1)]
            )
          );
        }
      }
    } finally {
      compiled.destroy();
      harness.destroy();
    }
  });
}

for (const [laterComponent, expectedLabel] of [
  [0.0101, 1],
  [0.0099, 2],
  [0.01, 1]
] as const) {
  it(`resolves nearly tied cosine candidates at training rows 255/256 (later=${laterComponent}) against explicit labels`, async () => {
    const harness = await getHarness();
    const training = Array.from({length: 257}, () => [-1, 0]);
    training[255] = [1, 0.01];
    training[256] = [1, laterComponent];
    const trainingLabels = new Array<number>(257).fill(0);
    trainingLabels[255] = 1;
    trainingLabels[256] = 2;
    const base = {
      pixelCount: 1,
      dimensions: 2,
      vectors: harness.floats([1, 0]),
      trainingCount: 257,
      trainingVectors: harness.floats(training.flat()),
      trainingLabels: harness.integers(trainingLabels),
      neighborCount: 1,
      classCount: 3,
      metric: 'cosine' as const
    };
    const result = await harness.run(
      (['tiled', 'brute-force'] as const).map(
        searchStructure =>
          new GPURasterEmbeddingKNearestClassifier({
            ...base,
            id: `near-tie-${searchStructure}`,
            searchStructure,
            output: {
              labels: harness.output(`${searchStructure}-label`, 'uint32', 1),
              confidence: harness.output(`${searchStructure}-confidence`, 'float32', 1),
              probabilities: harness.output(`${searchStructure}-probabilities`, 'float32', 3)
            }
          })
      )
    );
    // Cosine( [1,0], [1,y] ) decreases with positive y. Equal y retains source row 255.
    expect(nearestOracle([[1, 0]], training, trainingLabels, 1, 'cosine')[0].label).toBe(
      expectedLabel
    );
    for (const route of ['tiled', 'brute-force']) {
      expect(result[`${route}-label`]).toEqual([expectedLabel]);
      expect(result[`${route}-confidence`]).toEqual([1]);
      expect(result[`${route}-probabilities`]).toEqual([
        0,
        Number(expectedLabel === 1),
        Number(expectedLabel === 2)
      ]);
    }
  });
}
