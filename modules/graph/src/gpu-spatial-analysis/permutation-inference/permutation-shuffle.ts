// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Keyed pseudorandom bijection of [0, n) for the global permutation test: the swap-or-not shuffle
// of Hoang, Morris and Rogaway ("An Enciphering Scheme Based on a Card Shuffle", CRYPTO 2012).
// Round r holds a key K_r uniform in [0, n) and a round function bit F_r. A position X is paired
// with X' = (K_r - X) mod n and moves to X' when F_r(max(X, X')) is set. Each round is an
// involution, so the composition is a bijection. The paper's analysis shows that O(log n) rounds
// make the permutation indistinguishable from uniform for any number of queries up to n; this
// module uses 6 ceil(log2 n) + 6 rounds.
// An earlier Feistel network was measurably non-uniform on small domains (its halves were as small
// as two bits), which biased global Moran and Geary reference distributions; it was removed.
//
// The WGSL and TypeScript implementations are bit-identical. Round keys and salts are drawn from
// Philox 4x32-10 streams with counter (0, permutation, round, tag), unbiased by rejection.

import {HASH_UINT32_WGSL, hashUint32} from '../shared/hash-uint32';
import {PhiloxStream} from './permutation-random';

// Every caller passes its own Philox purpose ID (counter word `c` of the round material, from
// `PHILOX_PURPOSES` in utils/philox-random.ts), so two analyses with the same seed never share a
// permutation stream.

/** Largest number of rounds the WGSL twin supports (domains below 2^24: 6 * 24 + 6). @internal */
export const PERMUTATION_SHUFFLE_MAXIMUM_ROUNDS = 150;

/** Round-array size that covers every uint32 domain: `6 * 32 + 6`. @internal */
export const PERMUTATION_SHUFFLE_FULL_RANGE_ROUNDS = 198;

/** Number of rounds for a domain of `count` values: `6 ceil(log2 count) + 6`, or 0 for `count <= 1`. @internal */
export function getSwapOrNotRoundCount(count: number): number {
  return count <= 1 ? 0 : 6 * Math.ceil(Math.log2(count)) + 6;
}

/** Round keys `K_r` and round-function salts of one permutation. @internal */
export type SwapOrNotRounds = {
  keys: Uint32Array;
  salts: Uint32Array;
};

/**
 * Draws the round material of permutation number `permutation` under `key` for a domain of
 * `count` values, on the Philox purpose `stream` of the caller.
 *
 * @internal
 */
export function getSwapOrNotRounds(
  key: readonly [number, number],
  permutation: number,
  count: number,
  stream: number
): SwapOrNotRounds {
  const roundCount = getSwapOrNotRoundCount(count);
  const keys = new Uint32Array(roundCount);
  const salts = new Uint32Array(roundCount);
  for (let round = 0; round < roundCount; round++) {
    const philox = new PhiloxStream(key, permutation, round, stream);
    keys[round] = philox.nextBelow(count);
    salts[round] = philox.nextUint32();
  }
  return {keys, salts};
}

/**
 * Maps `index` in `[0, count)` through the swap-or-not shuffle defined by `rounds`.
 *
 * @internal
 */
export function getSwapOrNotPermutationIndex(
  index: number,
  count: number,
  rounds: SwapOrNotRounds
): number {
  let position = index;
  for (let round = 0; round < rounds.keys.length; round++) {
    let partner = rounds.keys[round] + count - position;
    if (partner >= count) {
      partner -= count;
    }
    const canonical = Math.max(position, partner);
    if (hashUint32((canonical ^ rounds.salts[round]) >>> 0) >>> 31) {
      position = partner;
    }
  }
  return position;
}

/** Writes the whole permutation `permutation` of `[0, count)` into `output`. @internal */
export function fillShufflePermutation(
  output: Uint32Array,
  count: number,
  key: readonly [number, number],
  permutation: number,
  stream: number
): void {
  const rounds = getSwapOrNotRounds(key, permutation, count, stream);
  for (let index = 0; index < count; index++) {
    output[index] = getSwapOrNotPermutationIndex(index, count, rounds);
  }
}

/**
 * WGSL twin of this module. Requires {@link PERMUTATION_RANDOM_WGSL} (Philox streams) in the same
 * module. Declares workgroup storage for the round material of one permutation:
 * - `loadSwapOrNotRounds(key, permutation, count, local, size)`: every invocation of the
 *   workgroup calls it with its `local` index and the workgroup `size`, followed by a
 *   `workgroupBarrier()` in uniform control flow, before any `getSwapOrNotPermutationIndex`.
 * - `getSwapOrNotPermutationIndex(index, rounds)`: bit-identical to the TypeScript function;
 *   `rounds` is `getSwapOrNotRoundCount(count)`, also available as `getSwapOrNotRoundCount`.
 *
 * @param stream Philox purpose ID of the caller (`PHILOX_PURPOSES.<name>.id`).
 * @param maximumRounds Size of the workgroup round arrays. The default fits domains below 2^24;
 * pass {@link PERMUTATION_SHUFFLE_FULL_RANGE_ROUNDS} for any uint32 domain.
 * @internal
 */
export function getPermutationShuffleWGSL(
  stream: number,
  maximumRounds: number = PERMUTATION_SHUFFLE_MAXIMUM_ROUNDS
): string {
  return /* wgsl */ `
${HASH_UINT32_WGSL}
const PERMUTATION_SHUFFLE_STREAM: u32 = ${stream >>> 0}u;
const PERMUTATION_SHUFFLE_MAXIMUM_ROUNDS: u32 = ${maximumRounds}u;
var<workgroup> swapOrNotKeys: array<u32, ${maximumRounds}>;
var<workgroup> swapOrNotSalts: array<u32, ${maximumRounds}>;

fn getSwapOrNotRoundCount(count: u32) -> u32 {
  return select(6u * (32u - countLeadingZeros(count - 1u)) + 6u, 0u, count <= 1u);
}

fn loadSwapOrNotRounds(key: vec2<u32>, permutation: u32, count: u32, local: u32, size: u32) {
  let roundCount = getSwapOrNotRoundCount(count);
  for (var round = local; round < roundCount; round += size) {
    var stream = createPhiloxStream(key, permutation, round, PERMUTATION_SHUFFLE_STREAM);
    swapOrNotKeys[round] = nextPhiloxBelow(&stream, count);
    swapOrNotSalts[round] = nextPhiloxUint32(&stream);
  }
}

fn getSwapOrNotPermutationIndex(index: u32, count: u32) -> u32 {
  var position = index;
  let roundCount = getSwapOrNotRoundCount(count);
  for (var round = 0u; round < roundCount; round++) {
    var partner = swapOrNotKeys[round] + count - position;
    if (partner >= count) {
      partner = partner - count;
    }
    let canonical = max(position, partner);
    if ((hashUint32(canonical ^ swapOrNotSalts[round]) >> 31u) != 0u) {
      position = partner;
    }
  }
  return position;
}
`;
}
