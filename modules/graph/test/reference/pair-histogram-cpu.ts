// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// CPU mirrors of the pair-histogram fixed-point quantization. Test-only; not part of the package API.

import {PAIR_HISTOGRAM_LARGE_TERM_SHIFT} from '../../src/gpu-dataframe/pair-statistics/pair-histogram';

/**
 * CPU mirror of `quantizePairAmount` for oracles: rounds `scaledValue + 0.5` down in f32 terms,
 * returning 0 for non-positive or NaN input.
 */
export function quantizePairAmountOnCPU(scaledValue: number): number {
  if (!(scaledValue > 0)) {
    return 0;
  }
  return Math.floor(Math.min(Math.fround(Math.fround(scaledValue) + 0.5), 4294967040));
}

/**
 * CPU mirror of `accumulateScaled`'s routing for oracles: returns which channel a scaled term
 * lands in and its integer amount in that channel's unit (`2^PAIR_HISTOGRAM_LARGE_TERM_SHIFT`
 * units for `'large'`), with f32 arithmetic where the kernel uses it.
 */
export function quantizeScaledTermOnCPU(scaledValue: number): {
  channel: 'none' | 'small' | 'large';
  amount: bigint;
  overflow: boolean;
} {
  if (!(scaledValue > 0)) {
    return {channel: 'none', amount: 0n, overflow: false};
  }
  const scaled = Math.fround(scaledValue);
  if (scaled <= 4294967040) {
    return {
      channel: 'small',
      amount: BigInt(Math.floor(Math.fround(scaled + 0.5))),
      overflow: false
    };
  }
  let coarse = Math.fround(scaled * 2 ** -PAIR_HISTOGRAM_LARGE_TERM_SHIFT);
  let overflow = false;
  if (!(coarse <= 2 ** 63)) {
    overflow = true;
    coarse = 2 ** 63;
  }
  const high = Math.floor(Math.fround(coarse * 2 ** -32));
  const low = Math.fround(coarse - high * 2 ** 32);
  const amount =
    BigInt(high) * 4294967296n + BigInt(Math.floor(Math.min(Math.fround(low + 0.5), 4294967040)));
  return {channel: 'large', amount, overflow};
}
