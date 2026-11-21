// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {addSegregationDecompositionRecipe} from '../../../src/gpu-spatial-analysis/landscape-ecology/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {RecipeTestFixture} from '../recipes/recipe-harness';

const UNIT_COUNT = 64;
const UNCHANGED_UNIT_COUNT = 32;

/**
 * 64 units with counts in [1, 997]; units 0-31 have equal baseline and final rows. The LCG is
 * `state = (state * 1103515245 + 12345) mod 2^31`, count `1 + state mod 997`.
 */
function createTables(seed: number): {baseline: Float32Array; final: Float32Array} {
  let state = seed;
  const next = () => {
    state = Number((BigInt(state) * 1103515245n + 12345n) % 2147483648n);
    return 1 + (state % 997);
  };
  const baseline = new Float32Array(2 * UNIT_COUNT);
  for (let index = 0; index < baseline.length; index++) baseline[index] = next();
  const final = baseline.slice();
  for (let index = 2 * UNCHANGED_UNIT_COUNT; index < final.length; index++) final[index] = next();
  return {baseline, final};
}

// URB-101. Every replicate below leaves the two tables unchanged in exact arithmetic up to the
// label exchange, so each ties the observed change: no swap, every swap, swaps of the unchanged
// units only (a no-op), and swaps of the changed units only (the full exchange). The two-sided
// extreme count is therefore 4 of 4 and the plus-one p value (4 + 1) / (4 + 1) = 1, by
// definition of the test. Before the fix the observed change came from tree reductions and the
// replicates from serial sums: a float32 model (/usr/bin/python3, NumPy, pairwise tree) of these
// seeds puts the serial magnitude below the tree magnitude for seeds 1, 3, 4 and 8, which drops
// all four ties from the extreme count (0 of 4, p 0.2). The GPU reduction order may differ from
// the model, so the seeds that fail before the fix may differ too.
it.each([
  1, 2, 3, 4, 5, 6, 7, 8
])('URB-101 exchanges that leave the tables unchanged tie the observed change (seed %i)', async seed => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, `segregation-ties-${seed}`);
  try {
    const {baseline, final} = createTables(seed);
    const swaps = new Uint32Array(4 * UNIT_COUNT);
    for (let unit = 0; unit < UNIT_COUNT; unit++) {
      swaps[UNIT_COUNT + unit] = 1;
      swaps[2 * UNIT_COUNT + unit] = unit < UNCHANGED_UNIT_COUNT ? 1 : 0;
      swaps[3 * UNIT_COUNT + unit] = unit < UNCHANGED_UNIT_COUNT ? 0 : 1;
    }
    const baselineView = fixture.input('baseline', baseline, 'float32x2', UNIT_COUNT);
    const finalView = fixture.input('final', final, 'float32x2', UNIT_COUNT);
    const inference = fixture.output('inference', 'float32x4', 1);
    addSegregationDecompositionRecipe(fixture.graph, {
      populations: [baselineView, baselineView, finalView, finalView],
      supportId: 'ties',
      counterfactualMethod: 'supplied',
      randomization: {
        seed: 1,
        replicates: 4,
        swaps: fixture.input('swaps', swaps, 'uint32', swaps.length)
      },
      output: {inference: inference.view}
    });
    fixture.run();
    const [observed, pValue, valid, extreme] = await fixture.readFloat32(inference, 4);
    expect(Number.isFinite(observed)).toBe(true);
    expect(observed).not.toBe(0);
    expect(valid).toBe(4);
    expect(extreme).toBe(4);
    expect(pValue).toBe(1);
  } finally {
    fixture.destroy();
  }
});
