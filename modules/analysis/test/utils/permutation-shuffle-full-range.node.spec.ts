// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {hashUint32} from '../../src/analysis-core/kernels/hash-uint32';
import {PHILOX_PURPOSES} from '../../src/analysis-core/kernels/philox-random';
import {
  getSwapOrNotPermutationIndex,
  getSwapOrNotRoundCount,
  getSwapOrNotRounds,
  PERMUTATION_SHUFFLE_FULL_RANGE_ROUNDS,
  type SwapOrNotRounds
} from '../../src/analysis-core/kernels/permutation-shuffle';
import {getSwapOrNotReferenceIndex} from './permutation-shuffle-reference';

// The swap-or-not shuffle on domains above 2^31 (math review finding core-numerics-02). The
// partner of position X in round r is (K_r - X) mod n by definition (Hoang, Morris and Rogaway
// 2012). The expected values are that definition in exact integers.

/** A salt whose round-function bit is set for `canonical`, so the round swaps. */
function findSwappingSalt(canonical: number): number {
  for (let salt = 0; salt < 64; salt++) {
    if (hashUint32((canonical ^ salt) >>> 0) >>> 31) {
      return salt;
    }
  }
  throw new Error('no swapping salt in 64 tries');
}

it('one round pairs a position with (key - position) mod count above 2^31', () => {
  const count = 3_000_000_000;
  // Key at or above the position: (2,000,000,000 - 5) mod 3,000,000,000 = 1,999,999,995.
  const first: SwapOrNotRounds = {
    keys: Uint32Array.of(2_000_000_000),
    salts: Uint32Array.of(findSwappingSalt(1_999_999_995))
  };
  expect(getSwapOrNotPermutationIndex(5, count, first)).toBe(1_999_999_995);
  expect(getSwapOrNotPermutationIndex(1_999_999_995, count, first)).toBe(5);
  // Key below the position: (5 - 2,000,000,000) mod 3,000,000,000 = 1,000,000,005.
  const second: SwapOrNotRounds = {
    keys: Uint32Array.of(5),
    salts: Uint32Array.of(findSwappingSalt(2_000_000_000))
  };
  expect(getSwapOrNotPermutationIndex(2_000_000_000, count, second)).toBe(1_000_000_005);
  expect(getSwapOrNotPermutationIndex(1_000_000_005, count, second)).toBe(2_000_000_000);
  // The largest domain: (4,294,967,293 - 4,294,967,294) mod 4,294,967,295 = 4,294,967,294, a
  // fixed point, and (4,294,967,293 - 4,294,967,293) mod 4,294,967,295 = 0.
  const third: SwapOrNotRounds = {
    keys: Uint32Array.of(0xfffffffd),
    salts: Uint32Array.of(findSwappingSalt(0xfffffffd))
  };
  expect(getSwapOrNotPermutationIndex(0xfffffffe, 0xffffffff, third)).toBe(0xfffffffe);
  expect(getSwapOrNotPermutationIndex(0xfffffffd, 0xffffffff, third)).toBe(0);
  expect(getSwapOrNotPermutationIndex(0, 0xffffffff, third)).toBe(0xfffffffd);
});

it('the TypeScript shuffle equals the exact-integer definition and inverts on domains up to 2^32 - 1', () => {
  for (const count of [0x7fffffff, 0x80000000, 0x80000001, 3_000_000_000, 0xffffffff]) {
    expect(getSwapOrNotRoundCount(count)).toBeLessThanOrEqual(
      PERMUTATION_SHUFFLE_FULL_RANGE_ROUNDS
    );
    const rounds = getSwapOrNotRounds([99, 7], 3, count, PHILOX_PURPOSES.permutationShuffle.id);
    const reversed: SwapOrNotRounds = {
      keys: Uint32Array.from(rounds.keys).reverse(),
      salts: Uint32Array.from(rounds.salts).reverse()
    };
    const images = new Set<number>();
    for (let sample = 0; sample < 512; sample++) {
      const position =
        sample < 8
          ? sample
          : sample < 16
            ? count - 1 - (sample - 8)
            : (sample * 2654435761) % count;
      const image = getSwapOrNotPermutationIndex(position, count, rounds);
      expect(image, `count ${count} position ${position}`).toBe(
        getSwapOrNotReferenceIndex(position, count, rounds)
      );
      expect(image).toBeLessThan(count);
      // Every round is an involution, so the reversed rounds undo the shuffle.
      expect(getSwapOrNotReferenceIndex(image, count, reversed)).toBe(position);
      images.add(image);
    }
    expect(images.size).toBe(512);
  }
});
