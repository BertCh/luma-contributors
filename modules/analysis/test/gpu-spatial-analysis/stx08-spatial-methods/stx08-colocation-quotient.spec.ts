// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {GPUColocationQuotient} from '../../../src/gpu-spatial-analysis/categorical-association/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {AnalysisRig, createSeededRandom, type CPUWeights} from '../catchment-accessibility/rig';

/**
 * Binary64 evaluation of the Leslie and Kronenfeld (2011) definition, written from the formulas
 * of the class documentation and sharing no code with it.
 */
function referenceColocation(csr: CPUWeights, categories: number[], categoryCount: number) {
  const rowCount = categories.length;
  const valid = (value: number) => value < categoryCount;
  const rows = new Array<number>(categoryCount).fill(0);
  for (const category of categories) {
    if (valid(category)) rows[category]++;
  }
  const total = rows.reduce((sum, value) => sum + value, 0);
  const local = Array.from({length: rowCount}, () =>
    new Array<number>(categoryCount).fill(Number.NaN)
  );
  const shareSums = Array.from({length: categoryCount}, () =>
    new Array<number>(categoryCount).fill(0)
  );
  const linked = new Array<number>(categoryCount).fill(0);
  for (let row = 0; row < rowCount; row++) {
    const own = categories[row];
    if (!valid(own)) continue;
    const sums = new Array<number>(categoryCount).fill(0);
    let weightTotal = 0;
    for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++) {
      const neighbor = csr.neighbors[slot];
      const weight = csr.weights[slot];
      if (neighbor === row || !(weight > 0) || !valid(categories[neighbor])) continue;
      sums[categories[neighbor]] += weight;
      weightTotal += weight;
    }
    if (!(weightTotal > 0)) continue;
    linked[own]++;
    for (let other = 0; other < categoryCount; other++) {
      const share = sums[other] / weightTotal;
      shareSums[own][other] += share;
      const chance = (rows[other] - (own === other ? 1 : 0)) / (total - 1);
      local[row][other] = chance > 0 ? share / chance : Number.NaN;
    }
  }
  const global = Array.from({length: categoryCount}, (_, own) =>
    Array.from({length: categoryCount}, (_, other) => {
      const chance = (rows[other] - (own === other ? 1 : 0)) / (total - 1);
      return linked[own] > 0 && chance > 0
        ? shareSums[own][other] / linked[own] / chance
        : Number.NaN;
    })
  );
  return {global, local, rows};
}

async function runColocation(csr: CPUWeights, categories: number[], categoryCount: number) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  const globalQuotients = rig.output('float32', categoryCount ** 2);
  const localQuotients = rig.output('float32', categories.length * categoryCount);
  const categoryRowCounts = rig.output('uint32', categoryCount);
  rig.run(
    new GPUColocationQuotient({
      weights: rig.weights(csr),
      categories: rig.input(new Uint32Array(categories), 'uint32'),
      categoryCount,
      output: {globalQuotients, localQuotients, categoryRowCounts}
    })
  );
  const result = {
    global: await rig.readFloat(globalQuotients),
    local: await rig.readFloat(localQuotients),
    rows: await rig.readUint(categoryRowCounts)
  };
  rig.destroy();
  return result;
}

function expectClose(actual: number, expected: number, label: string) {
  if (Number.isNaN(expected)) {
    expect(actual, label).toBeNaN();
  } else {
    expect(Math.abs(actual - expected), label).toBeLessThan(2e-5 * Math.max(1, Math.abs(expected)));
  }
}

describe('GPUColocationQuotient', () => {
  it('reproduces the hand-computed nearest-neighbour quotients of a four-point line', async () => {
    // Points A A B B, one nearest neighbour each: 0>1, 1>0, 2>3, 3>2. N = 4, N_A = N_B = 2.
    // CLQ(A->A) = (2/2) / ((2 - 1) / 3) = 3, CLQ(A->B) = 0, and the same for B.
    const csr = {offsets: [0, 1, 2, 3, 4], neighbors: [1, 0, 3, 2], weights: [1, 1, 1, 1]};
    const {global, local, rows} = await runColocation(csr, [0, 0, 1, 1], 2);
    expect(rows).toEqual([2, 2]);
    expect(global[0]).toBeCloseTo(3, 6);
    expect(global[1]).toBeCloseTo(0, 6);
    expect(global[2]).toBeCloseTo(0, 6);
    expect(global[3]).toBeCloseTo(3, 6);
    expect(local.slice(0, 2)).toEqual([3, 0].map(value => expect.closeTo(value, 6)));
  });

  it('is one for a mixed pair: A has a single B neighbour among one other B (chance 1)', async () => {
    // A B: 0>1, 1>0. N = 2. CLQ(A->B) = (1/1) / (1/1) = 1; the A->A chance share is 0, so NaN.
    const csr = {offsets: [0, 1, 2], neighbors: [1, 0], weights: [1, 1]};
    const {global} = await runColocation(csr, [0, 1], 2);
    expect(global[1]).toBeCloseTo(1, 6);
    expect(global[0]).toBeNaN();
  });

  it('leaves out rows without a category or neighbour, self links and non-positive weights', async () => {
    // Row 0 (A) lists itself, a missing-category row, a zero-weight link and row 1 (B).
    const csr = {
      offsets: [0, 4, 5, 5],
      neighbors: [0, 2, 1, 1, 0],
      weights: [1, 1, 0, 2, 1]
    };
    const categories = [0, 1, 0xffffffff];
    const {global, local} = await runColocation(csr, categories, 2);
    const reference = referenceColocation(csr, categories, 2);
    expect(reference.global[0][1]).toBeCloseTo(1, 12);
    expectClose(global[1], reference.global[0][1], 'A->B');
    expect(local.slice(4)).toSatisfy((words: number[]) => words.every(Number.isNaN));
  });

  for (const [name, seed, rowCount, categoryCount, degree] of [
    ['random 3 categories', 11, 300, 3, 6],
    ['random 12 categories', 12, 500, 12, 8],
    ['random 32 categories, sparse', 13, 400, 32, 3]
  ] as const) {
    it(`matches the binary64 reference on ${name}`, async () => {
      const random = createSeededRandom(seed);
      const csr: CPUWeights = {offsets: [0], neighbors: [], weights: []};
      const categories: number[] = [];
      for (let row = 0; row < rowCount; row++) {
        // About a tenth of the rows carry no category; clustered labels give non-trivial quotients.
        const base = Math.floor((row / rowCount) * categoryCount);
        categories.push(
          random() < 0.1 ? 0xffffffff : random() < 0.6 ? base : Math.floor(random() * categoryCount)
        );
        const count = Math.floor(random() * (degree + 1));
        for (let link = 0; link < count; link++) {
          csr.neighbors.push(
            Math.min(rowCount - 1, Math.max(0, row + Math.floor(random() * 9) - 4))
          );
          csr.weights.push(Math.fround(0.1 + random()));
        }
        csr.offsets.push(csr.neighbors.length);
      }
      const reference = referenceColocation(csr, categories, categoryCount);
      const {global, local, rows} = await runColocation(csr, categories, categoryCount);
      expect(rows).toEqual(reference.rows);
      for (let own = 0; own < categoryCount; own++) {
        for (let other = 0; other < categoryCount; other++) {
          expectClose(
            global[own * categoryCount + other],
            reference.global[own][other],
            `global ${own}->${other}`
          );
        }
      }
      for (let row = 0; row < rowCount; row++) {
        for (let other = 0; other < categoryCount; other++) {
          expectClose(
            local[row * categoryCount + other],
            reference.local[row][other],
            `local ${row}->${other}`
          );
        }
      }
    });
  }
});
