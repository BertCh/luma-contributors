// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  getGPURasterEmbeddingChangeParameterValues,
  getGPURasterEmbeddingProjectionParameterValues,
  getGPURasterEmbeddingSimilarityParameterValues,
  getRasterEmbeddingPrincipalComponents,
  getRasterEmbeddingStretchRanges,
  GPURasterEmbeddingChange,
  GPURasterEmbeddingCovariance,
  GPURasterEmbeddingDequantize,
  GPURasterEmbeddingKMeans,
  GPURasterEmbeddingKNearestClassifier,
  GPURasterEmbeddingLinearProbe,
  GPURasterEmbeddingNearestPixels,
  GPURasterEmbeddingProjection,
  GPURasterEmbeddingPyramidLevel,
  GPURasterEmbeddingSimilarity,
  GPURasterEmbeddingZoneMean
} from '@luma-contributors/analysis/raster-analysis';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createRasterTimeSeriesHarness,
  getMaximumAbsoluteDifference
} from '../raster-time-series/raster-time-series-test-utils';
import reference from '../../oracles/fixtures/rts03-embedding-rasters-numpy.json';

const NO_LABEL = 0xffffffff;
const pixelCount = reference.width * reference.height;
const dimensions = reference.dimensions;

const toFloat = (values: (number | null)[]) => values.map(value => value ?? Number.NaN);

/** Packs signed bytes four to a little-endian uint32 word, as an uploaded `Int8Array` reads. */
function packInt8(values: ArrayLike<number>): number[] {
  const bytes = Int8Array.from(values);
  const padded = new Int8Array(Math.ceil(bytes.length / 4) * 4);
  padded.set(bytes);
  return Array.from(new Uint32Array(padded.buffer));
}

async function getHarness() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return createRasterTimeSeriesHarness(device);
}

const decoded = toFloat(reference.decoded);
const valid = reference.valid;

// ---------------------------------------------------------------------------------------------
// Dequantization

it('RTS-03 dequantize decodes sign(q) (q / 127.5)^2 and flags NoData, by hand', async () => {
  const harness = await getHarness();
  // One pixel of four bands: 127, -64, 0, 1 -> hand values; a second pixel has -128 in band 2.
  const result = await harness.run(
    new GPURasterEmbeddingDequantize({
      pixelCount: 2,
      dimensions: 4,
      quantized: harness.integers(packInt8([127, -64, 0, 1, 5, 6, -128, 7])),
      output: {
        vectors: harness.output('vectors', 'float32', 8),
        validity: harness.output('validity', 'uint32', 2)
      }
    })
  );
  expect(result['validity']).toEqual([1, 0]);
  expect(result['vectors'][0]).toBeCloseTo((127 / 127.5) ** 2, 6);
  expect(result['vectors'][1]).toBeCloseTo(-((64 / 127.5) ** 2), 6);
  expect(result['vectors'][2]).toBe(0);
  expect(result['vectors'][3]).toBeCloseTo((1 / 127.5) ** 2, 9);
  expect(result['vectors'].slice(4).every(Number.isNaN)).toBe(true);
});

it('RTS-03 dequantize matches NumPy, including raw and normalized output and NoData in one band', async () => {
  const harness = await getHarness();
  const normalizedHarness = await getHarness();
  const raw = await harness.run(
    new GPURasterEmbeddingDequantize({
      pixelCount,
      dimensions,
      quantized: harness.integers(packInt8(reference.quantized)),
      output: {
        vectors: harness.output('vectors', 'float32', pixelCount * dimensions),
        validity: harness.output('validity', 'uint32', pixelCount)
      }
    })
  );
  expect(getMaximumAbsoluteDifference(raw['vectors'], decoded)).toBeLessThan(2e-6);
  expect(raw['validity']).toEqual(valid);
  expect(valid[63]).toBe(0);
  const normalized = await normalizedHarness.run(
    new GPURasterEmbeddingDequantize({
      pixelCount,
      dimensions,
      normalize: true,
      quantized: normalizedHarness.integers(packInt8(reference.quantized)),
      output: {vectors: normalizedHarness.output('vectors', 'float32', pixelCount * dimensions)}
    })
  );
  const expected = decoded.map((value, index) => {
    const row = Math.floor(index / dimensions);
    const norm = Math.hypot(...decoded.slice(row * dimensions, (row + 1) * dimensions));
    return value / norm;
  });
  expect(getMaximumAbsoluteDifference(normalized['vectors'], expected)).toBeLessThan(2e-6);
});

// ---------------------------------------------------------------------------------------------
// Similarity, nearest pixels, change

it('RTS-03 similarity matches NumPy cosine and dot against four queries', async () => {
  for (const metric of ['cosine', 'dot'] as const) {
    const harness = await getHarness();
    const queryCount = reference.queryCount;
    const result = await harness.run(
      new GPURasterEmbeddingSimilarity({
        pixelCount,
        dimensions,
        queryCount,
        metric,
        vectors: harness.floats(decoded),
        mask: harness.integers(valid),
        queries: harness.floats(reference.queries),
        parameters: harness.floats(
          getGPURasterEmbeddingSimilarityParameterValues({matchThreshold: 0.9})
        ),
        output: {
          similarity: harness.output('similarity', 'float32', pixelCount * queryCount),
          maximum: harness.output('maximum', 'float32', pixelCount),
          bestQuery: harness.output('bestQuery', 'uint32', pixelCount),
          matches: harness.output('matches', 'uint32', pixelCount)
        }
      })
    );
    const expected = toFloat(reference.similarity[metric]);
    expect(getMaximumAbsoluteDifference(result['similarity'], expected), metric).toBeLessThan(3e-6);
    const expectedMaximum = Array.from({length: pixelCount}, (_, pixel) =>
      valid[pixel]
        ? Math.max(...expected.slice(pixel * queryCount, (pixel + 1) * queryCount))
        : Number.NaN
    );
    expect(getMaximumAbsoluteDifference(result['maximum'], expectedMaximum), metric).toBeLessThan(
      3e-6
    );
    expect(result['matches']).toEqual(expectedMaximum.map(value => (value >= 0.9 ? 1 : 0)));
    if (metric === 'cosine') {
      expect(result['bestQuery']).toEqual(reference.similarity.bestQuery);
    }
  }
  // The long fourth query separates dot from cosine.
  expect(
    reference.similarity.dot.find(
      (_, index) => index % 4 === 3 && reference.similarity.dot[index] !== null
    )
  ).not.toBeCloseTo(
    reference.similarity.cosine.find(
      (_, index) => index % 4 === 3 && reference.similarity.cosine[index] !== null
    ) as number,
    2
  );
});

it('RTS-03 nearest pixels returns the NumPy top-k by cosine for each query', async () => {
  const harness = await getHarness();
  const {k, ids, scores} = reference.nearest;
  const search = new GPURasterEmbeddingNearestPixels({
    pixelCount,
    dimensions,
    vectors: harness.floats(decoded),
    mask: harness.integers(valid),
    queryCount: reference.queryCount,
    queries: harness.floats(reference.queries),
    resultCount: k,
    output: {
      pixelIds: harness.output('ids', 'uint32', reference.queryCount * k),
      scores: harness.output('scores', 'float32', reference.queryCount * k),
      resultCounts: harness.output('counts', 'uint32', reference.queryCount)
    }
  });
  search.addToGraph(harness.graph);
  const output = await harness.run([]);
  expect(output['ids']).toEqual(ids);
  expect(getMaximumAbsoluteDifference(output['scores'], scores)).toBeLessThan(3e-6);
  expect(output['counts']).toEqual(new Array(reference.queryCount).fill(k));
});

it('RTS-03 change reports cosine, angle, magnitude and a threshold mask between years', async () => {
  const harness = await getHarness();
  const threshold = 0.95;
  const result = await harness.run(
    new GPURasterEmbeddingChange({
      pixelCount,
      dimensions,
      before: harness.floats(decoded),
      after: harness.floats(toFloat(reference.decodedYearTwo)),
      parameters: harness.floats(
        getGPURasterEmbeddingChangeParameterValues({similarityThreshold: threshold})
      ),
      output: {
        similarity: harness.output('similarity', 'float32', pixelCount),
        angle: harness.output('angle', 'float32', pixelCount),
        magnitude: harness.output('magnitude', 'float32', pixelCount),
        changed: harness.output('changed', 'uint32', pixelCount)
      }
    })
  );
  const cosine = toFloat(reference.change.cosine);
  expect(getMaximumAbsoluteDifference(result['similarity'], cosine)).toBeLessThan(3e-6);
  // The angle comes from the chord of the unit vectors, so it is as accurate as the cosine
  // (it needed 5e-4 when it was acos of the float32 cosine).
  expect(
    getMaximumAbsoluteDifference(result['angle'], toFloat(reference.change.angle))
  ).toBeLessThan(3e-6);
  expect(
    getMaximumAbsoluteDifference(result['magnitude'], toFloat(reference.change.magnitude))
  ).toBeLessThan(3e-6);
  const changed = cosine.map(value => (value < threshold ? 1 : 0));
  expect(changed.some(flag => flag === 1) && changed.some(flag => flag === 0)).toBe(true);
  // Skip pixels whose cosine is within float32 rounding of the threshold.
  cosine.forEach((value, pixel) => {
    if (Number.isNaN(value) || Math.abs(value - threshold) > 1e-5) {
      expect(result['changed'][pixel], `pixel ${pixel}`).toBe(changed[pixel]);
    }
  });
});

// ---------------------------------------------------------------------------------------------
// Clustering and supervised inference

it('RTS-03 k-means recovers the blobs and their means', async () => {
  const harness = await getHarness();
  const clusterCount = 3;
  const kmeans = new GPURasterEmbeddingKMeans({
    pixelCount,
    dimensions,
    vectors: harness.floats(decoded),
    mask: harness.integers(valid),
    clusterCount,
    maximumIterations: 20,
    output: {
      labels: harness.output('labels', 'uint32', pixelCount),
      centroids: harness.output('centroids', 'float32', clusterCount * dimensions),
      counts: harness.output('counts', 'uint32', clusterCount),
      status: harness.output('status', 'uint32', 3)
    }
  });
  kmeans.addToGraph(harness.graph);
  const result = await harness.run([]);
  const blob = reference.blob;
  // Map each found cluster to the blob that holds most of its pixels, and require a bijection.
  const mapping = new Map<number, number>();
  for (let cluster = 0; cluster < clusterCount; cluster++) {
    const votes = [0, 0, 0];
    result['labels'].forEach((label, pixel) => {
      if (label === cluster) votes[blob[pixel]]++;
    });
    mapping.set(cluster, votes.indexOf(Math.max(...votes)));
  }
  expect(new Set(mapping.values()).size).toBe(clusterCount);
  result['labels'].forEach((label, pixel) => {
    if (valid[pixel]) expect(mapping.get(label), `pixel ${pixel}`).toBe(blob[pixel]);
    else expect(label).toBe(NO_LABEL);
  });
  // A converged k-means centre is the mean of its members: compare with NumPy's blob means.
  for (let cluster = 0; cluster < clusterCount; cluster++) {
    const expected = reference.blobMeans.slice(
      mapping.get(cluster)! * dimensions,
      (mapping.get(cluster)! + 1) * dimensions
    );
    expect(
      getMaximumAbsoluteDifference(
        result['centroids'].slice(cluster * dimensions, (cluster + 1) * dimensions),
        expected
      )
    ).toBeLessThan(3e-6);
  }
  expect(result['counts'].reduce((sum, count) => sum + count, 0)).toBe(reference.pca.validCount);
  expect(result['status'][2]).not.toBe(0);
});

for (const metric of ['euclidean', 'cosine'] as const) {
  it(`RTS-03 kNN classifier (${metric}) matches the brute-force NumPy vote`, async () => {
    const harness = await getHarness();
    const {knn} = reference;
    const classCount = knn.classCount;
    const result = await harness.run(
      new GPURasterEmbeddingKNearestClassifier({
        pixelCount,
        dimensions,
        metric,
        vectors: harness.floats(decoded),
        mask: harness.integers(valid),
        trainingVectors: harness.floats(knn.training),
        trainingLabels: harness.integers(knn.trainingLabels),
        trainingCount: knn.trainingLabels.length,
        classCount,
        neighborCount: knn.k,
        output: {
          labels: harness.output('labels', 'uint32', pixelCount),
          confidence: harness.output('confidence', 'float32', pixelCount),
          probabilities: harness.output('probabilities', 'float32', pixelCount * classCount)
        }
      })
    );
    expect(result['labels']).toEqual(knn[metric].labels);
    expect(
      getMaximumAbsoluteDifference(result['confidence'], toFloat(knn[metric].confidence))
    ).toBeLessThan(1e-6);
    expect(
      getMaximumAbsoluteDifference(result['probabilities'], toFloat(knn[metric].probabilities))
    ).toBeLessThan(1e-6);
  });
}

it('RTS-03 linear probe matches the softmax of W x + b', async () => {
  const harness = await getHarness();
  const {linearProbe} = reference;
  const classCount = linearProbe.bias.length;
  const result = await harness.run(
    new GPURasterEmbeddingLinearProbe({
      pixelCount,
      dimensions,
      classCount,
      vectors: harness.floats(decoded),
      mask: harness.integers(valid),
      weights: harness.floats(linearProbe.weights),
      bias: harness.floats(linearProbe.bias),
      output: {
        labels: harness.output('labels', 'uint32', pixelCount),
        confidence: harness.output('confidence', 'float32', pixelCount),
        probabilities: harness.output('probabilities', 'float32', pixelCount * classCount)
      }
    })
  );
  expect(result['labels']).toEqual(linearProbe.labels);
  const probabilities = toFloat(linearProbe.probabilities);
  expect(getMaximumAbsoluteDifference(result['probabilities'], probabilities)).toBeLessThan(2e-6);
  const confidence = Array.from({length: pixelCount}, (_, pixel) =>
    valid[pixel]
      ? Math.max(...probabilities.slice(pixel * classCount, (pixel + 1) * classCount))
      : Number.NaN
  );
  expect(getMaximumAbsoluteDifference(result['confidence'], confidence)).toBeLessThan(2e-6);
});

it('RTS-03 tie rules: the lowest query wins a similarity tie, and the lowest sample and class win kNN ties', async () => {
  const similarityHarness = await getHarness();
  const similarity = await similarityHarness.run(
    new GPURasterEmbeddingSimilarity({
      pixelCount: 1,
      dimensions: 2,
      queryCount: 3,
      vectors: similarityHarness.floats([1, 0]),
      // Queries 1 and 2 are the best and equal; query 0 is worse.
      queries: similarityHarness.floats([0, 1, 1, 0, 2, 0]),
      output: {bestQuery: similarityHarness.output('bestQuery', 'uint32', 1)}
    })
  );
  expect(similarity['bestQuery']).toEqual([1]);

  const knnHarness = await getHarness();
  // Three identical labelled vectors with labels 0, 1, 2 and k = 2: the neighbours are samples 0 and 1
  // (lower index first), the votes tie 1-1, and the smaller class wins.
  const knn = await knnHarness.run(
    new GPURasterEmbeddingKNearestClassifier({
      pixelCount: 1,
      dimensions: 2,
      vectors: knnHarness.floats([0.5, 0.5]),
      trainingVectors: knnHarness.floats([1, 0, 1, 0, 1, 0]),
      trainingLabels: knnHarness.integers([0, 1, 2]),
      trainingCount: 3,
      classCount: 3,
      neighborCount: 2,
      output: {
        labels: knnHarness.output('labels', 'uint32', 1),
        probabilities: knnHarness.output('probabilities', 'float32', 3)
      }
    })
  );
  expect(knn['labels']).toEqual([0]);
  expect(knn['probabilities']).toEqual([0.5, 0.5, 0]);

  const tieHarness = await getHarness();
  // Two neighbours with different labels equally near: votes tie, the smaller class wins.
  const vote = await tieHarness.run(
    new GPURasterEmbeddingKNearestClassifier({
      pixelCount: 1,
      dimensions: 1,
      vectors: tieHarness.floats([0]),
      trainingVectors: tieHarness.floats([1, -1]),
      trainingLabels: tieHarness.integers([1, 0]),
      trainingCount: 2,
      classCount: 2,
      neighborCount: 2,
      output: {labels: tieHarness.output('labels', 'uint32', 1)}
    })
  );
  expect(vote['labels']).toEqual([0]);
});

it('RTS-03 linear probe stays finite when the logits would overflow a plain exponential', async () => {
  const harness = await getHarness();
  const result = await harness.run(
    new GPURasterEmbeddingLinearProbe({
      pixelCount: 1,
      dimensions: 2,
      classCount: 2,
      vectors: harness.floats([1, 1]),
      weights: harness.floats([100, 100, 90, 90]),
      bias: harness.floats([0, 0]),
      output: {
        labels: harness.output('labels', 'uint32', 1),
        probabilities: harness.output('probabilities', 'float32', 2)
      }
    })
  );
  // Logits 200 and 180: softmax is 1 / (1 + e^-20) and e^-20 / (1 + e^-20).
  expect(result['labels']).toEqual([0]);
  expect(result['probabilities'][0]).toBeCloseTo(1 / (1 + Math.exp(-20)), 6);
  expect(result['probabilities'][1]).toBeCloseTo(Math.exp(-20) / (1 + Math.exp(-20)), 12);
});

// ---------------------------------------------------------------------------------------------
// PCA to RGB

for (const rangeCount of [1, 7, undefined]) {
  it(`RTS-03 covariance matches NumPy (rangeCount ${rangeCount ?? 'default'})`, async () => {
    const harness = await getHarness();
    const result = await harness.run(
      new GPURasterEmbeddingCovariance({
        pixelCount,
        dimensions,
        rangeCount,
        vectors: harness.floats(decoded),
        mask: harness.integers(valid),
        output: {
          mean: harness.output('mean', 'float32', dimensions),
          covariance: harness.output('covariance', 'float32', dimensions * dimensions),
          usableCount: harness.output('usableCount', 'uint32', 1)
        }
      })
    );
    expect(result['usableCount'][0]).toBe(reference.pca.validCount);
    expect(
      getMaximumAbsoluteDifference(result['mean'], reference.pca.mean as number[])
    ).toBeLessThan(2e-6);
    expect(
      getMaximumAbsoluteDifference(result['covariance'], reference.pca.covariance as number[])
    ).toBeLessThan(2e-6);
  });
}

it('RTS-03 principal components reproduce NumPy eigh with the scikit-learn sign convention', () => {
  const {eigenvalues, components, explainedVarianceRatio} = getRasterEmbeddingPrincipalComponents(
    reference.pca.covariance as number[],
    dimensions,
    3
  );
  expect(
    getMaximumAbsoluteDifference(eigenvalues, reference.pca.eigenvalues as number[])
  ).toBeLessThan(1e-12);
  expect(
    getMaximumAbsoluteDifference(components, reference.pca.components as number[])
  ).toBeLessThan(1e-9);
  const trace = Array.from(
    {length: dimensions},
    (_, index) => reference.pca.covariance[index * dimensions + index] as number
  ).reduce((sum, value) => sum + value, 0);
  expect(explainedVarianceRatio[0]).toBeCloseTo(
    (reference.pca.eigenvalues[0] as number) / trace,
    10
  );
  const ranges = getRasterEmbeddingStretchRanges(eigenvalues, 2);
  expect(ranges[0][1]).toBeCloseTo(2 * Math.sqrt(eigenvalues[0]), 12);
  expect(ranges[0][0]).toBeCloseTo(-ranges[0][1], 12);
});

it('RTS-03 projection and RGB stretch match NumPy', async () => {
  const harness = await getHarness();
  const componentCount = 3;
  const stretch = [0, 1, 2].map(
    component =>
      [
        reference.pca.stretch[component * 2] as number,
        reference.pca.stretch[component * 2 + 1] as number
      ] as const
  );
  const result = await harness.run(
    new GPURasterEmbeddingProjection({
      pixelCount,
      dimensions,
      componentCount,
      vectors: harness.floats(decoded),
      mask: harness.integers(valid),
      mean: harness.floats(reference.pca.mean as number[]),
      components: harness.floats(reference.pca.components as number[]),
      parameters: harness.floats(
        getGPURasterEmbeddingProjectionParameterValues({stretchRanges: stretch})
      ),
      output: {
        projected: harness.output('projected', 'float32', pixelCount * componentCount),
        rgb: harness.output('rgb', 'float32', pixelCount * 3)
      }
    })
  );
  expect(
    getMaximumAbsoluteDifference(result['projected'], toFloat(reference.pca.projected))
  ).toBeLessThan(3e-6);
  expect(getMaximumAbsoluteDifference(result['rgb'], toFloat(reference.pca.rgb))).toBeLessThan(
    3e-5
  );
});

it('RTS-03 the GPU covariance feeds the CPU eigenproblem and the projection end to end', async () => {
  const harness = await getHarness();
  const covariance = await harness.run(
    new GPURasterEmbeddingCovariance({
      pixelCount,
      dimensions,
      vectors: harness.floats(decoded),
      mask: harness.integers(valid),
      output: {
        mean: harness.output('mean', 'float32', dimensions),
        covariance: harness.output('covariance', 'float32', dimensions * dimensions)
      }
    })
  );
  const {components} = getRasterEmbeddingPrincipalComponents(
    covariance['covariance'],
    dimensions,
    3
  );
  // Signs follow the convention, so the maps agree with NumPy's up to the float32 covariance error.
  expect(
    getMaximumAbsoluteDifference(components, reference.pca.components as number[])
  ).toBeLessThan(5e-4);
  const projectionHarness = await getHarness();
  const projection = await projectionHarness.run(
    new GPURasterEmbeddingProjection({
      pixelCount,
      dimensions,
      componentCount: 3,
      vectors: projectionHarness.floats(decoded),
      mask: projectionHarness.integers(valid),
      mean: projectionHarness.floats(covariance['mean']),
      components: projectionHarness.floats(Array.from(components)),
      output: {projected: projectionHarness.output('projected', 'float32', pixelCount * 3)}
    })
  );
  expect(
    getMaximumAbsoluteDifference(projection['projected'], toFloat(reference.pca.projected))
  ).toBeLessThan(5e-3);
});

// ---------------------------------------------------------------------------------------------
// Zone means and the unit-vector pyramid

it('RTS-03 zone mean over fractional-coverage tuples matches the weighted NumPy mean', async () => {
  const harness = await getHarness();
  const {zoneTuples} = reference;
  const zoneCount = 4;
  const result = await harness.run(
    new GPURasterEmbeddingZoneMean({
      pixelCount,
      dimensions,
      zoneCount,
      vectors: harness.floats(decoded),
      mask: harness.integers(valid),
      tuples: {
        zoneIds: harness.integers(zoneTuples.zoneIds),
        cellIds: harness.integers(zoneTuples.cellIds),
        fractions: harness.floats(zoneTuples.fractions),
        status: {count: harness.integers([zoneTuples.zoneIds.length])}
      },
      output: {
        means: harness.output('means', 'float32', zoneCount * dimensions),
        weights: harness.output('weights', 'float32', zoneCount),
        meanLengths: harness.output('meanLengths', 'float32', zoneCount)
      }
    })
  );
  expect(getMaximumAbsoluteDifference(result['means'], toFloat(zoneTuples.means))).toBeLessThan(
    3e-6
  );
  expect(getMaximumAbsoluteDifference(result['weights'], toFloat(zoneTuples.weights))).toBeLessThan(
    2e-5
  );
  expect(
    getMaximumAbsoluteDifference(result['meanLengths'], toFloat(zoneTuples.meanLengths))
  ).toBeLessThan(3e-6);
});

it('RTS-03 zone mean over a zone raster matches NumPy and ignores unzoned pixels', async () => {
  const harness = await getHarness();
  const {zoneRaster} = reference;
  const zoneCount = 4;
  const result = await harness.run(
    new GPURasterEmbeddingZoneMean({
      pixelCount,
      dimensions,
      zoneCount,
      vectors: harness.floats(decoded),
      zoneRaster: harness.integers(zoneRaster.zones),
      output: {
        means: harness.output('means', 'float32', zoneCount * dimensions),
        weights: harness.output('weights', 'float32', zoneCount)
      }
    })
  );
  expect(getMaximumAbsoluteDifference(result['means'], toFloat(zoneRaster.means))).toBeLessThan(
    3e-6
  );
  expect(result['weights']).toEqual(toFloat(zoneRaster.weights));
});

for (const scaleName of ['2x2', '3x2'] as const) {
  for (const source of ['quantized', 'vectors'] as const) {
    it(`RTS-03 unit-vector pyramid ${scaleName} from ${source} averages decoded vectors and renormalizes`, async () => {
      const harness = await getHarness();
      const {oddRaster} = reference;
      const expected = oddRaster.overviews[scaleName];
      const [scaleX, scaleY] = scaleName.split('x').map(Number);
      const count = expected.outWidth * expected.outHeight;
      const pyramid = new GPURasterEmbeddingPyramidLevel({
        sourceWidth: oddRaster.width,
        sourceHeight: oddRaster.height,
        dimensions: oddRaster.dimensions,
        scale: [scaleX, scaleY],
        ...(source === 'quantized'
          ? {quantized: harness.integers(packInt8(oddRaster.quantized))}
          : {
              vectors: harness.floats(toFloat(oddRaster.decoded)),
              mask: harness.integers(
                Array.from({length: oddRaster.width * oddRaster.height}, (_, pixel) =>
                  oddRaster.decoded[pixel * oddRaster.dimensions] === null ? 0 : 1
                )
              )
            }),
        output: {
          vectors: harness.output('vectors', 'float32', count * oddRaster.dimensions),
          validity: harness.output('validity', 'uint32', count),
          validCount: harness.output('validCount', 'uint32', count),
          meanLength: harness.output('meanLength', 'float32', count)
        }
      });
      expect([pyramid.width, pyramid.height]).toEqual([expected.outWidth, expected.outHeight]);
      const result = await harness.run(pyramid);
      expect(
        getMaximumAbsoluteDifference(result['vectors'], toFloat(expected.vectors))
      ).toBeLessThan(3e-6);
      expect(result['validCount']).toEqual(expected.validCount);
      expect(
        getMaximumAbsoluteDifference(result['meanLength'], toFloat(expected.meanLength))
      ).toBeLessThan(3e-6);
      // Every valid overview pixel is a unit vector, while the plain mean of decoded vectors is shorter.
      result['validity'].forEach((flag, pixel) => {
        if (!flag) return;
        const vector = result['vectors'].slice(
          pixel * oddRaster.dimensions,
          (pixel + 1) * oddRaster.dimensions
        );
        expect(Math.hypot(...vector)).toBeCloseTo(1, 5);
      });
      expect(Math.min(...toFloat(expected.meanLength).filter(Number.isFinite))).toBeLessThan(0.999);
    });
  }
}

// ---------------------------------------------------------------------------------------------
// Declaration limits

it('RTS-03 refuses sizes above the named limits at declaration', async () => {
  const harness = await getHarness();
  const vectors = (length: number) => harness.floats(new Array(length).fill(0));
  expect(
    () =>
      new GPURasterEmbeddingSimilarity({
        pixelCount: 1,
        dimensions: 129,
        queryCount: 1,
        vectors: vectors(129),
        queries: vectors(129),
        output: {maximum: harness.output('accepted-maximum', 'float32', 1)}
      })
  ).not.toThrow();
  expect(
    () =>
      new GPURasterEmbeddingSimilarity({
        pixelCount: 1,
        dimensions: 1025,
        queryCount: 1,
        vectors: vectors(1025),
        queries: vectors(1025),
        output: {maximum: harness.output('maximum', 'float32', 1)}
      })
  ).toThrow(/raster-embedding-similarity dimensions must be an integer in \[1, 1024\]/);
  expect(
    () =>
      new GPURasterEmbeddingSimilarity({
        pixelCount: 1,
        dimensions: 2,
        queryCount: 65,
        vectors: vectors(2),
        queries: vectors(130),
        output: {maximum: harness.output('maximum2', 'float32', 1)}
      })
  ).toThrow(/raster-embedding-similarity queryCount must be an integer in \[1, 64\]/);
  expect(
    () =>
      new GPURasterEmbeddingZoneMean({
        pixelCount: 1,
        dimensions: 2,
        zoneCount: 1,
        vectors: vectors(2),
        output: {means: harness.output('means', 'float32', 2)}
      })
  ).toThrow(/needs exactly one of zoneRaster and tuples/);
  expect(
    () =>
      new GPURasterEmbeddingPyramidLevel({
        sourceWidth: 4,
        sourceHeight: 4,
        dimensions: 2,
        scale: 17,
        vectors: vectors(32),
        output: {vectors: harness.output('overview', 'float32', 2)}
      })
  ).toThrow(/horizontal scale must be an integer in \[1, 16\]/);
});

// ---------------------------------------------------------------------------------------------
// Math correctness review 2026-11-10 (raster-imagery-04 and raster-imagery-08)

it('embedding change resolves angles far below the float32 cosine step', async () => {
  const harness = await getHarness();
  const bandCount = 64;
  // Pixels 0 to 2: a unit vector with one component moved by 1e-3, 1e-4 and 1e-5.
  // Pixels 3 to 7: closed forms in the first two bands.
  const base = Array.from({length: bandCount}, () => 1 / Math.sqrt(bandCount));
  const before: number[] = [];
  const after: number[] = [];
  for (const step of [1e-3, 1e-4, 1e-5]) {
    before.push(...base);
    after.push(...base.map((value, band) => (band === 5 ? value + step : value)));
  }
  const planar = (x: number, y: number) => [x, y, ...new Array(bandCount - 2).fill(0)];
  const pairs: [number[], number[], number][] = [
    [planar(1, 0), planar(1, 1e-4), Math.atan(1e-4)],
    [planar(2, 0), planar(0, 3), Math.PI / 2],
    [planar(1, 0), planar(-2, 0), Math.PI],
    [planar(0, 0), planar(0, 0), 0],
    [planar(0, 0), planar(0, 1), Math.PI / 2]
  ];
  for (const [left, right] of pairs) {
    before.push(...left);
    after.push(...right);
  }
  const pixels = 3 + pairs.length;
  const result = await harness.run(
    new GPURasterEmbeddingChange({
      pixelCount: pixels,
      dimensions: bandCount,
      before: harness.floats(before),
      after: harness.floats(after),
      output: {angle: harness.output('angle', 'float32', pixels)}
    })
  );
  // Binary64 angle of the float32 inputs from the chord of the unit vectors, 2 asin(|u - v| / 2).
  const expected: number[] = [];
  for (let pixel = 0; pixel < 3; pixel++) {
    const left = Array.from(
      Float32Array.from(before.slice(pixel * bandCount, (pixel + 1) * bandCount))
    );
    const right = Array.from(
      Float32Array.from(after.slice(pixel * bandCount, (pixel + 1) * bandCount))
    );
    const leftLength = Math.hypot(...left);
    const rightLength = Math.hypot(...right);
    const chord = Math.hypot(
      ...left.map((value, band) => value / leftLength - right[band] / rightLength)
    );
    expected.push(2 * Math.asin(chord / 2));
  }
  expected.push(...pairs.map(pair => pair[2]));
  // The first three are about 9.9e-4, 9.9e-5 and 9.9e-6 rad; acos of the float32 cosine gave 0
  // for the last two and was 2e-4 off on the first.
  expect(expected[0]).toBeGreaterThan(9e-4);
  expect(expected[2]).toBeLessThan(1.1e-5);
  for (let pixel = 0; pixel < pixels; pixel++) {
    expect(
      Math.abs(result['angle'][pixel] - expected[pixel]),
      `pixel ${pixel}: ${result['angle'][pixel]} against ${expected[pixel]}`
    ).toBeLessThanOrEqual(4e-7 + 2e-6 * expected[pixel]);
  }
});

it('k-nearest cosine distance between two zero vectors is zero, as in GPUSimilaritySearch', async () => {
  const harness = await getHarness();
  // Pixel 0 is the zero vector. Training sample 1 is the zero vector with class 1: similarity 1,
  // distance 0, so it is the nearest neighbour. Samples 0 and 2 have similarity 0 with pixel 0.
  // Pixel 1 is (0, 2): sample 2 is at distance 0, the zero sample at distance 1.
  const result = await harness.run(
    new GPURasterEmbeddingKNearestClassifier({
      pixelCount: 2,
      dimensions: 2,
      metric: 'cosine',
      vectors: harness.floats([0, 0, 0, 2]),
      trainingVectors: harness.floats([1, 0, 0, 0, 0, 1]),
      trainingLabels: harness.integers([0, 1, 2]),
      trainingCount: 3,
      classCount: 3,
      neighborCount: 1,
      output: {labels: harness.output('labels', 'uint32', 2)}
    })
  );
  expect(result['labels']).toEqual([1, 2]);
});
