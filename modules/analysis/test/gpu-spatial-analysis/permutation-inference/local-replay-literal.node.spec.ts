// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeLocalPermutationOracle} from './permutation-oracle';
import {createLocalReplayLiteralFixture} from './local-replay-literal-fixture';

for (const alternative of ['greater', 'lesser'] as const) {
  // Every possible donor gives -6; five draws exceed/equal -6, independently of seed.
  for (const seed of [0, 11, 137]) {
    for (const stopAfterExceedances of [undefined, 3]) {
      it(`host oracle matches literal Moran ties (${alternative}, seed ${seed}, stop ${stopAfterExceedances ?? 'none'}, Tests §5)`, () => {
        const result = computeLocalPermutationOracle({
          ...createLocalReplayLiteralFixture(),
          statistic: 'localMoran',
          islands: 'keep',
          alternative,
          seed,
          permutationCount: 5,
          maximumPermutations: 9,
          significanceLevel: 0.05,
          maximumNeighbors: 1,
          stopAfterExceedances
        });
        expect(Number.isFinite(result.observed[1])).toBe(true);
        expect(result.observed[1]).toBe(-1.5);
        expect(result.observed[0]).toBe(0.25);
        expect(result.exceedances[1]).toBe(stopAfterExceedances === undefined ? 5 : 3);
        expect(result.permutationCounts[1]).toBe(stopAfterExceedances === undefined ? 5 : 3);
        expect(result.pseudoPValues[1]).toBe(1);
        expect(result.exceedances[3]).toBe(0xffffffff);
        expect(result.permutationCounts[3]).toBe(0xffffffff);
        expect(result.pseudoPValues[3]).toBeNaN();
        expect(result.islandCount).toBe(1);
        expect(result.overflow).toBe(0);
      });
    }
  }
}
