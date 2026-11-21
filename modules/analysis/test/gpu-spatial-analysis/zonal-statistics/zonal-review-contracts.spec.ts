// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUZonalStatistics} from '../../../src/gpu-spatial-analysis/zonal-statistics/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('feature-row status copies candidate-only loss and numeric uncertainty and restores on reuse (GEO-116)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const rowsBuffer = createInputBuffer(device, Uint32Array.of(0, 0));
  const sourceBuffers = [0, 1, 0, 3, 2].map(value =>
    createInputBuffer(device, Uint32Array.of(value))
  );
  const targetBuffers = sourceBuffers.map(() => createOutputBuffer(device, 1));
  const countsBuffer = createOutputBuffer(device, 1);
  const names = [
    'overflow',
    'candidateOverflow',
    'uncertainCount',
    'antimeridianUncertainCount',
    'poleUncertainCount'
  ] as const;
  const sourceStatus = Object.fromEntries(
    names.map((name, index) => [
      name,
      importGraphBuffer(graph, `source-${name}`, sourceBuffers[index], 'uint32', 1)
    ])
  );
  const targetStatus = Object.fromEntries(
    names.map((name, index) => [
      name,
      importGraphBuffer(graph, `target-${name}`, targetBuffers[index], 'uint32', 1)
    ])
  );
  graph.add(
    new GPUZonalStatistics({
      features: {
        kind: 'feature-rows',
        featureCount: 1,
        pointFeatureRows: importGraphBuffer(graph, 'rows', rowsBuffer, 'uint32', 2),
        status: sourceStatus
      },
      output: {
        counts: importGraphBuffer(graph, 'counts', countsBuffer, 'uint32', 1),
        status: targetStatus
      }
    })
  );
  const compiled = graph.compile();
  try {
    // These are upstream scalars, not predicates: in particular seven uncertain pairs stay seven.
    for (const expected of [
      [0, 1, 0, 3, 2],
      [1, 0, 7, 4, 9],
      [0, 0, 0, 0, 0]
    ]) {
      sourceBuffers.forEach((buffer, index) => buffer.write(Uint32Array.of(expected[index])));
      submitGraph(device, compiled, undefined);
      expect(
        await Promise.all(targetBuffers.map(async buffer => (await readUint32(buffer, 1))[0]))
      ).toEqual(expected);
      expect(await readUint32(countsBuffer, 1)).toEqual([2]);
    }
  } finally {
    compiled.destroy();
    for (const buffer of [rowsBuffer, ...sourceBuffers, ...targetBuffers, countsBuffer])
      buffer.destroy();
  }
});

for (const sumOrder of ['sorted', 'atomic'] as const) {
  it(`weighted product overflow retains finite-input counts and weights (${sumOrder}, GEO-128)`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const rowsBuffer = createInputBuffer(device, Uint32Array.of(0, 0));
    const valuesBuffer = createInputBuffer(device, Float32Array.of(2 ** 127, 2));
    const weightsBuffer = createInputBuffer(device, Float32Array.of(4, 1));
    const countsBuffer = createOutputBuffer(device, 1);
    const valueCountsBuffer = createOutputBuffer(device, 1);
    const sumsBuffer = createOutputBuffer(device, 1);
    const weightsOutputBuffer = createOutputBuffer(device, 1);
    const meansBuffer = createOutputBuffer(device, 1);
    const variancesBuffer = createOutputBuffer(device, 1);
    const deviationsBuffer = createOutputBuffer(device, 1);
    graph.add(
      new GPUZonalStatistics({
        features: {
          kind: 'feature-rows',
          featureCount: 1,
          pointFeatureRows: importGraphBuffer(graph, 'rows', rowsBuffer, 'uint32', 2)
        },
        values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', 2),
        weights: importGraphBuffer(graph, 'weights', weightsBuffer, 'float32', 2),
        sumOrder,
        output: {
          counts: importGraphBuffer(graph, 'counts', countsBuffer, 'uint32', 1),
          valueCounts: importGraphBuffer(graph, 'value-counts', valueCountsBuffer, 'uint32', 1),
          sums: importGraphBuffer(graph, 'sums', sumsBuffer, 'float32', 1),
          weightSums: importGraphBuffer(graph, 'weight-sums', weightsOutputBuffer, 'float32', 1),
          means: importGraphBuffer(graph, 'means', meansBuffer, 'float32', 1),
          variances: importGraphBuffer(graph, 'variances', variancesBuffer, 'float32', 1),
          standardDeviations: importGraphBuffer(graph, 'deviations', deviationsBuffer, 'float32', 1)
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      // Hand calculation: f32(4 * 2^127) overflows, so numerator = 0 + 2;
      // both finite-input rows remain valid: n = 2, denominator = 4 + 1 = 5, mean = 2/5.
      expect(await readUint32(countsBuffer, 1)).toEqual([2]);
      expect(await readUint32(valueCountsBuffer, 1)).toEqual([2]);
      expect(await readFloat32(sumsBuffer, 1)).toEqual([2]);
      expect(await readFloat32(weightsOutputBuffer, 1)).toEqual([5]);
      const [mean] = await readFloat32(meansBuffer, 1);
      expect(Number.isFinite(mean)).toBe(true);
      expect(mean).toBeCloseTo(0.4, 6);
      // The overflowing deviation terms are also omitted. Retained terms give refined m=.72,
      // SSD=1.6384, correction=.32768 and denominator=1.6: variance .8192, std .90509668.
      // These finite outputs are misleading; they do not certify that all terms were retained.
      const [variance] = await readFloat32(variancesBuffer, 1);
      const [deviation] = await readFloat32(deviationsBuffer, 1);
      expect(Number.isFinite(variance)).toBe(true);
      expect(Number.isFinite(deviation)).toBe(true);
      expect(variance).toBeCloseTo(0.8192, 5);
      expect(deviation).toBeCloseTo(0.90509668, 5);
    } finally {
      compiled.destroy();
      for (const buffer of [
        rowsBuffer,
        valuesBuffer,
        weightsBuffer,
        countsBuffer,
        valueCountsBuffer,
        sumsBuffer,
        weightsOutputBuffer,
        meansBuffer,
        variancesBuffer,
        deviationsBuffer
      ])
        buffer.destroy();
    }
  });
}
