// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {
  GPULocalCategoryDependence,
  GPU_LOCAL_CATEGORY_DEPENDENCE_COLUMNS
} from '../../../src/gpu-spatial-analysis/categorical-association/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {AnalysisRig, createSeededRandom, type CPUWeights} from '../catchment-accessibility/rig';

const columns = GPU_LOCAL_CATEGORY_DEPENDENCE_COLUMNS;

/** Binomial upper tail from the pmf `C(n, k) p^k q^(n-k)` with a log-factorial table (binary64). */
function binomialUpperTail(like: number, count: number, p: number): number {
  const logFactorial = [0];
  for (let k = 1; k <= count; k++) logFactorial.push(logFactorial[k - 1] + Math.log(k));
  let tail = 0;
  for (let k = like; k <= count; k++) {
    tail += Math.exp(
      logFactorial[count] -
        logFactorial[k] -
        logFactorial[count - k] +
        k * Math.log(p) +
        (count - k) * Math.log(1 - p)
    );
  }
  return Math.min(tail, 1);
}

function referenceDependence(csr: CPUWeights, categories: number[], categoryCount: number) {
  const rows = new Array<number>(categoryCount).fill(0);
  for (const category of categories) if (category < categoryCount) rows[category]++;
  const total = rows.reduce((sum, value) => sum + value, 0);
  return categories.map((own, row) => {
    let like = 0;
    let count = 0;
    if (own < categoryCount) {
      for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++) {
        const neighbor = csr.neighbors[slot];
        if (neighbor === row || categories[neighbor] >= categoryCount) continue;
        count++;
        if (categories[neighbor] === own) like++;
      }
    }
    if (own >= categoryCount || count === 0) {
      return [like, count, NaN, NaN, NaN, NaN];
    }
    const p = rows[own] / total;
    const variance = count * p * (1 - p);
    return [
      like,
      count,
      like / count,
      p,
      p >= 1 ? 1 : binomialUpperTail(like, count, p),
      variance > 0 ? (like - count * p) / Math.sqrt(variance) : NaN
    ];
  });
}

async function runDependence(csr: CPUWeights, categories: number[], categoryCount: number) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  const statistics = rig.output('float32', categories.length * columns.length);
  rig.run(
    new GPULocalCategoryDependence({
      weights: rig.weights(csr),
      categories: rig.input(new Uint32Array(categories), 'uint32'),
      categoryCount,
      output: {statistics}
    })
  );
  const words = await rig.readFloat(statistics);
  rig.destroy();
  return categories.map((_, row) => words.slice(row * columns.length, (row + 1) * columns.length));
}

function expectRows(actual: number[][], expected: number[][]) {
  for (const [row, expectedRow] of expected.entries()) {
    for (let column = 0; column < columns.length; column++) {
      const label = `row ${row} column ${column}`;
      if (Number.isNaN(expectedRow[column])) {
        expect(actual[row][column], label).toBeNaN();
      } else {
        expect(
          Math.abs(actual[row][column] - expectedRow[column]),
          `${label}: ${actual[row][column]} vs ${expectedRow[column]}`
        ).toBeLessThan(1e-4 * Math.max(1, Math.abs(expectedRow[column])));
      }
    }
  }
}

describe('GPULocalCategoryDependence', () => {
  it('matches a hand-computed binomial tail', async () => {
    // Row 0 (A) has neighbours A, A, B; p(A) = 3 / 5. P(X >= 2 | n = 3, p = 0.6) = 3 * 0.36 * 0.4 + 0.216 = 0.648.
    const csr = {
      offsets: [0, 3, 3, 3, 3, 3],
      neighbors: [1, 2, 3, 0, 0, 0],
      weights: [1, 1, 1, 0, 0, 0]
    };
    const rows = await runDependence({...csr, offsets: [0, 3, 3, 3, 3, 3]}, [0, 0, 0, 1, 1], 2);
    expect(rows[0][columns.likeCount]).toBe(2);
    expect(rows[0][columns.neighborCount]).toBe(3);
    expect(rows[0][columns.proportion]).toBeCloseTo(2 / 3, 6);
    expect(rows[0][columns.expectedProportion]).toBeCloseTo(0.6, 6);
    expect(rows[0][columns.upperTailProbability]).toBeCloseTo(0.648, 5);
    expect(rows[0][columns.standardizedScore]).toBeCloseTo((2 - 1.8) / Math.sqrt(3 * 0.6 * 0.4), 5);
  });

  it('gives NaN statistics for rows without a category or neighbour and ignores self links', async () => {
    const csr = {offsets: [0, 2, 2, 3], neighbors: [0, 2, 0], weights: [1, 1, 1]};
    const rows = await runDependence(csr, [0, 0, 0xffffffff], 2);
    // Row 0 lists itself and row 2 (no category): no neighbour.
    expect(rows[0][columns.neighborCount]).toBe(0);
    expect(rows[0][columns.proportion]).toBeNaN();
    expect(rows[2][columns.upperTailProbability]).toBeNaN();
    expect(rows[2][columns.likeCount]).toBe(0);
  });

  it('has an upper tail of one and no score when there is one category', async () => {
    const csr = {offsets: [0, 1, 2], neighbors: [1, 0], weights: [1, 1]};
    const rows = await runDependence(csr, [0, 0], 1);
    expect(rows[0][columns.upperTailProbability]).toBe(1);
    expect(rows[0][columns.standardizedScore]).toBeNaN();
  });

  for (const [name, seed, rowCount, categoryCount, degree] of [
    ['random 2 categories', 21, 400, 2, 8],
    ['random 7 categories, degree 20', 22, 400, 7, 20],
    ['random 32 categories', 23, 500, 32, 5]
  ] as const) {
    it(`matches the binary64 reference on ${name}`, async () => {
      const random = createSeededRandom(seed);
      const csr: CPUWeights = {offsets: [0], neighbors: [], weights: []};
      const categories: number[] = [];
      for (let row = 0; row < rowCount; row++) {
        const base = Math.floor((row / rowCount) * categoryCount);
        categories.push(
          random() < 0.08
            ? 0xffffffff
            : random() < 0.5
              ? base
              : Math.floor(random() * categoryCount)
        );
        const count = Math.floor(random() * (degree + 1));
        for (let link = 0; link < count; link++) {
          csr.neighbors.push(
            Math.min(rowCount - 1, Math.max(0, row + Math.floor(random() * 21) - 10))
          );
          csr.weights.push(1);
        }
        csr.offsets.push(csr.neighbors.length);
      }
      expectRows(
        await runDependence(csr, categories, categoryCount),
        referenceDependence(csr, categories, categoryCount)
      );
    });
  }
});
