// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {type Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUMultivariateLocalGearyPermutationTest} from '../../../src/gpu-spatial-analysis/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {getMultipleTestingAdjustedPValues} from '../../../src/gpu-spatial-analysis/multiple-testing/gpu-multiple-testing-adjustment';
import type {GPUMultipleTestingMethod} from '../../../src/analysis-core/contracts/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../utils/require-test-device';

type Strategy = {
  strategy: 'exact' | 'tiled';
  rowTileSize?: number;
  permutationTileSize?: number;
};

/** Deterministic pseudo-random rows with a ring-plus-skip neighbor structure and some masked rows. */
function makeData(rows: number) {
  let state = 12345;
  const next = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
  const values = Array.from({length: rows * 3}, () => next() * 10);
  const mask = Array.from({length: rows}, (_, row) => (row % 17 === 5 ? 0 : 1));
  const offsets = [0];
  const neighbors: number[] = [];
  const weights: number[] = [];
  for (let row = 0; row < rows; row++) {
    for (const step of [1, rows - 1, 7]) {
      neighbors.push((row + step) % rows);
      weights.push(1);
    }
    offsets.push(neighbors.length);
  }
  return {values, mask, offsets, neighbors, weights};
}

async function runGeary(
  rows: number,
  strategy: Strategy,
  permutations: number,
  inference: {
    correction?: 'none' | 'bonferroni';
    multipleTesting?: GPUMultipleTestingMethod;
  } = {correction: 'bonferroni'}
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  const input = <Format extends 'float32' | 'uint32'>(format: Format, data: number[]) => {
    const buffer = createInputBuffer(
      device,
      format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data)
    );
    buffers.push(buffer);
    return importGraphBuffer(graph, `input-${serial++}`, buffer, format, data.length);
  };
  const output = <Format extends 'float32' | 'uint32'>(format: Format, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, `output-${serial++}`, buffer, format, length)};
  };
  const data = makeData(rows);
  const statistic = output('float32', rows);
  const pValues = output('float32', rows);
  const classes = output('uint32', rows);
  graph.add(
    new GPUMultivariateLocalGearyPermutationTest({
      weights: {
        offsets: input('uint32', data.offsets),
        neighbors: input('uint32', data.neighbors),
        weights: input('float32', data.weights)
      },
      values: input('float32', data.values),
      variableCount: 3,
      mask: input('uint32', data.mask),
      maximumPermutations: permutations,
      seed: [7, 0],
      ...inference,
      ...strategy,
      output: {statistic: statistic.view, pValues: pValues.view, classes: classes.view}
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return {
      statistic: Array.from(await readFloat32(statistic.buffer)),
      pValues: Array.from(await readFloat32(pValues.buffer)),
      classes: Array.from(await readUint32(classes.buffer))
    };
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

function expectEqual(
  actual: Awaited<ReturnType<typeof runGeary>>,
  expected: Awaited<ReturnType<typeof runGeary>>
) {
  expect(actual.classes).toEqual(expected.classes);
  for (const key of ['statistic', 'pValues'] as const)
    actual[key].forEach((value, index) =>
      Number.isNaN(expected[key][index])
        ? expect(value).toBeNaN()
        : expect(value).toBeCloseTo(expected[key][index], 5)
    );
}

// The exact single-dispatch baseline is the oracle for the tiled strategy.
it('multivariate Geary tiled strategy equals the exact baseline', async () => {
  const exact = await runGeary(200, {strategy: 'exact'}, 99);
  const tiled = await runGeary(200, {strategy: 'tiled'}, 99);
  expectEqual(tiled, exact);
  expect(exact.statistic.some(Number.isNaN)).toBe(true); // masked rows stay untested
  expect(exact.classes.includes(0xffffffff)).toBe(true);
});

it('multivariate Geary tiled strategy is tile-size independent', async () => {
  const whole = await runGeary(256, {strategy: 'tiled'}, 150);
  const tiles = await runGeary(
    256,
    {strategy: 'tiled', rowTileSize: 37, permutationTileSize: 40},
    150
  );
  expectEqual(tiles, whole);
  const exact = await runGeary(256, {strategy: 'exact'}, 150);
  expectEqual(whole, exact);
});

it('multivariate Geary tiled strategy runs above 256 rows', async () => {
  const result = await runGeary(3000, {strategy: 'tiled', rowTileSize: 1024}, 99);
  expect(result.statistic.filter(Number.isFinite).length).toBeGreaterThan(2500);
  expect(result.pValues.filter(value => value > 0 && value <= 1).length).toBeGreaterThan(2500);
});

it.each([
  'exact',
  'tiled'
] as const)('multivariate Geary %s multipleTesting adjusts p-values and classes through the family operator', async strategyName => {
  const raw = await runGeary(200, {strategy: strategyName}, 99, {correction: 'none'});
  expect(raw.classes.filter(value => value === 1 || value === 2).length).toBeGreaterThan(0);
  for (const method of [
    'bonferroni',
    'holm',
    'benjamini-hochberg',
    'benjamini-yekutieli'
  ] as const) {
    const adjusted = await runGeary(200, {strategy: strategyName}, 99, {multipleTesting: method});
    const expectedP = getMultipleTestingAdjustedPValues(raw.pValues, method);
    adjusted.pValues.forEach((value, row) => {
      if (Number.isNaN(expectedP[row])) {
        expect(value).toBeNaN();
      } else {
        expect(value, `${method} p ${row}`).toBeCloseTo(expectedP[row], 5);
      }
    });
    adjusted.classes.forEach((value, row) => {
      if (raw.classes[row] === 0xffffffff) {
        expect(value).toBe(0xffffffff);
      } else if (adjusted.pValues[row] <= 0.05) {
        // Significant after correction: same direction as the uncorrected test.
        expect(value).toBe(raw.classes[row]);
        expect([1, 2]).toContain(value);
      } else {
        expect(value).toBe(0);
      }
    });
    // A correction never creates significance.
    const significantAfter = adjusted.classes.filter(value => value === 1 || value === 2).length;
    expect(significantAfter).toBeLessThanOrEqual(
      raw.classes.filter(value => value === 1 || value === 2).length
    );
  }
});
