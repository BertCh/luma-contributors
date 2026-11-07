// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {TIME_WORDS_WGSL} from '../../gpu-dataframe/time-window-filter/time-words';

/** Number of radix key bits needed to sort keys up to `maximumKey`. @internal */
export function getKeyBitCount(maximumKey: number): number {
  return Math.max(1, 32 - Math.clz32(Math.max(maximumKey, 1)));
}

/**
 * WGSL `rowTimeDifference(a, b)` over a `timestamps` binding: f32 seconds, or exact two-word
 * timestamps (low, high) when `isWordMode`. @internal
 */
export function getTimestampSourceWGSL(isWordMode: boolean): string {
  return isWordMode
    ? /* wgsl */ `${TIME_WORDS_WGSL}
fn rowTimeDifference(a: u32, b: u32) -> f32 {
  let wordsA = vec2<u32>(timestamps[timestampsOffset + 2u * a], timestamps[timestampsOffset + 2u * a + 1u]);
  let wordsB = vec2<u32>(timestamps[timestampsOffset + 2u * b], timestamps[timestampsOffset + 2u * b + 1u]);
  return timeWordsDifferenceToF32(wordsA, wordsB);
}`
    : /* wgsl */ `
fn rowTimeDifference(a: u32, b: u32) -> f32 {
  return timestamps[timestampsOffset + a] - timestamps[timestampsOffset + b];
}`;
}

/** Throws unless a parameter target holds the required element count. @internal */
export function checkParameterTarget(
  target: Float32Array | Uint32Array,
  length: number,
  name: string
): void {
  if (target.length < length) {
    throw new Error(`${name} target must hold ${length} elements`);
  }
}
