// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {hashUint32} from '../../src/analysis-core/kernels/hash-uint32';
import type {SwapOrNotRounds} from '../../src/analysis-core/kernels/permutation-shuffle';

/**
 * The swap-or-not shuffle by its definition (Hoang, Morris and Rogaway 2012), in exact integers:
 * round `r` pairs position `X` with `(K_r - X) mod n` and moves to it when the round function of
 * the larger of the two is set. No 32-bit arithmetic is involved, so this is the reference for
 * domains above 2^31, where a `u32` sum of key and count wraps.
 */
export function getSwapOrNotReferenceIndex(
  index: number,
  count: number,
  rounds: SwapOrNotRounds
): number {
  const size = BigInt(count);
  let position = BigInt(index);
  for (let round = 0; round < rounds.keys.length; round++) {
    const partner = (((BigInt(rounds.keys[round]) - position) % size) + size) % size;
    const canonical = position > partner ? position : partner;
    if (hashUint32((Number(canonical) ^ rounds.salts[round]) >>> 0) >>> 31) {
      position = partner;
    }
  }
  return Number(position);
}
