// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {
  GPUCategoryPairCounts,
  GPUSpatialEntropy,
  GPU_SPATIAL_ENTROPY_LAYOUT
} from '../../../src/gpu-spatial-analysis/spatial-entropy/index';
import fixture from '../../oracles/fixtures/stx08-spatial-methods.json';
import {requireTestDevice} from '../../utils/require-test-device';
import {AnalysisRig} from '../catchment-accessibility/rig';

const layout = GPU_SPATIAL_ENTROPY_LAYOUT;

async function runEntropy(
  csr: {offsets: number[]; neighbors: number[]},
  categories: number[],
  categoryCount: number
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  const summary = rig.output('float32', layout.length);
  const pairCounts = rig.output('uint32', categoryCount ** 2);
  rig.run(
    new GPUSpatialEntropy({
      weights: rig.weights({...csr, weights: csr.neighbors.map(() => 1)}),
      categories: rig.input(new Uint32Array(categories), 'uint32'),
      categoryCount,
      output: {summary, pairCounts}
    })
  );
  const result = {words: await rig.readFloat(summary), pairs: await rig.readUint(pairCounts)};
  rig.destroy();
  return result;
}

describe('GPUSpatialEntropy', () => {
  it('is zero mutual information for one category and maximal for a perfectly sorted line', async () => {
    // A path of four rows: AABB has links AA, AB, BA, BB, BB... counted both ways.
    const csr = {offsets: [0, 1, 3, 5, 6], neighbors: [1, 0, 2, 1, 3, 2]};
    const sorted = await runEntropy(csr, [0, 0, 1, 1], 2);
    // Ordered links: 0>1 (A,A), 1>0 (A,A), 1>2 (A,B), 2>1 (B,A), 2>3 (B,B), 3>2 (B,B).
    expect(sorted.pairs).toEqual([2, 1, 1, 2]);
    const p = [2 / 6, 1 / 6, 1 / 6, 2 / 6];
    const joint = -p.reduce((sum, value) => sum + value * Math.log(value), 0);
    expect(sorted.words[layout.cooccurrence]).toBeCloseTo(joint, 6);
    expect(sorted.words[layout.originEntropy]).toBeCloseTo(Math.log(2), 6);
    expect(sorted.words[layout.mutualInformation]).toBeCloseTo(2 * Math.log(2) - joint, 6);
    expect(sorted.words[layout.shannon]).toBeCloseTo(Math.log(2), 6);
    const single = await runEntropy(csr, [0, 0, 0, 0], 2);
    expect(single.words[layout.shannon]).toBeCloseTo(0, 6);
    expect(single.words[layout.mutualInformation]).toBeCloseTo(0, 6);
    expect(single.words[layout.relativeMutualInformation]).toBeNaN();
  });

  it('separates the origin and destination entropies of directed links', async () => {
    // Links 0>1, 0>2, 1>2 with categories A, B, B: origins {A, A, B}, destinations {B, B, B}.
    const csr = {offsets: [0, 2, 3, 3], neighbors: [1, 2, 2]};
    const {words, pairs} = await runEntropy(csr, [0, 1, 1], 2);
    expect(pairs).toEqual([0, 2, 0, 1]);
    const origin = -((2 / 3) * Math.log(2 / 3) + (1 / 3) * Math.log(1 / 3));
    expect(words[layout.originEntropy]).toBeCloseTo(origin, 6);
    expect(words[layout.destinationEntropy]).toBeCloseTo(0, 6);
    expect(words[layout.cooccurrence]).toBeCloseTo(origin, 6);
    expect(words[layout.mutualInformation]).toBeCloseTo(0, 6);
  });

  it('reports NaN without links and ignores rows with no category', async () => {
    const empty = await runEntropy({offsets: [0, 0, 0], neighbors: []}, [0, 1], 2);
    expect(empty.words[layout.linkCount]).toBe(0);
    expect(empty.words[layout.cooccurrence]).toBeNaN();
    const csr = {offsets: [0, 1, 3, 4], neighbors: [1, 0, 2, 1]};
    const withNone = await runEntropy(csr, [0, 0xffffffff, 1], 2);
    expect(withNone.words[layout.linkCount]).toBe(0);
    expect(withNone.words[layout.rowCount]).toBe(2);
  });

  for (const reference of fixture.entropy) {
    it(`matches the binary64 reference on ${reference.name}`, async () => {
      const {words, pairs} = await runEntropy(
        reference,
        reference.categories,
        reference.categoryCount
      );
      expect(pairs).toEqual(reference.pairCounts);
      expect(words[layout.linkCount]).toBe(reference.linkCount);
      expect(words[layout.rowCount]).toBe(reference.rowCount);
      for (const name of [
        'shannon',
        'originEntropy',
        'destinationEntropy',
        'cooccurrence',
        'mutualInformation'
      ] as const) {
        expect(Math.abs(words[layout[name]] - reference[name]), name).toBeLessThan(2e-6);
      }
    });
  }

  it('GPUCategoryPairCounts counts rows per category', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    const reference = fixture.entropy[0];
    const pairCounts = rig.output('uint32', reference.categoryCount ** 2);
    const rowCounts = rig.output('uint32', reference.categoryCount);
    rig.run(
      new GPUCategoryPairCounts({
        weights: rig.weights({
          offsets: reference.offsets,
          neighbors: reference.neighbors,
          weights: reference.neighbors.map(() => 1)
        }),
        categories: rig.input(new Uint32Array(reference.categories), 'uint32'),
        categoryCount: reference.categoryCount,
        output: {pairCounts, categoryRowCounts: rowCounts}
      })
    );
    expect(await rig.readUint(rowCounts)).toEqual(reference.categoryRowCounts);
    rig.destroy();
  });
});
