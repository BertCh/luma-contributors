// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUVector} from '@luma-contributors/graph/gpu-data';
import {aggregateInteger64OnGpu} from './wide-numeric-gpu-aggregate';
import {splitBigIntToWords, type WideInteger64Summary} from './wide-numeric-gpu-reference';

export type Integer64Bound = {value: bigint; inclusive: boolean};
export type Integer64Range = {lower?: Integer64Bound; upper?: Integer64Bound};

const SIGNED_MINIMUM = -(1n << 63n);
const SIGNED_MAXIMUM = (1n << 63n) - 1n;
const UNSIGNED_MAXIMUM = (1n << 64n) - 1n;

/** Inclusive integer interval of a range clamped to the 64-bit domain, or null when it is empty. */
export function resolveInteger64Range(
  range: Integer64Range,
  signed: boolean
): {minimum: bigint; maximum: bigint} | null {
  const domainMinimum = signed ? SIGNED_MINIMUM : 0n;
  const domainMaximum = signed ? SIGNED_MAXIMUM : UNSIGNED_MAXIMUM;
  let minimum = domainMinimum;
  let maximum = domainMaximum;
  if (range.lower) {
    const bound = range.lower.inclusive ? range.lower.value : range.lower.value + 1n;
    if (bound > minimum) minimum = bound;
  }
  if (range.upper) {
    const bound = range.upper.inclusive ? range.upper.value : range.upper.value - 1n;
    if (bound < maximum) maximum = bound;
  }
  return minimum > maximum ? null : {minimum, maximum};
}

/** WGSL boolean over `pair` (low, high) for an exact Int64 or Uint64 range; no value is rounded. */
export function makeInteger64RangeSelectionExpression(
  range: Integer64Range,
  signed: boolean
): string {
  const resolved = resolveInteger64Range(range, signed);
  if (!resolved) return 'false';
  const less = signed ? 'word64LessSigned' : 'word64LessUnsigned';
  const literal = (value: bigint) => {
    const [low, high] = splitBigIntToWords(value);
    return `vec2<u32>(${low}u, ${high}u)`;
  };
  const domainMinimum = signed ? SIGNED_MINIMUM : 0n;
  const domainMaximum = signed ? SIGNED_MAXIMUM : UNSIGNED_MAXIMUM;
  const parts: string[] = [];
  if (resolved.minimum > domainMinimum) parts.push(`!${less}(pair, ${literal(resolved.minimum)})`);
  if (resolved.maximum < domainMaximum) parts.push(`!${less}(${literal(resolved.maximum)}, pair)`);
  return parts.length ? parts.join(' && ') : 'true';
}

/** Exact aggregate over the rows whose Int64 or Uint64 value lies in the range. */
export function aggregateInteger64Range(props: {
  device: Device;
  words: GPUVector<'uint32x2'>;
  signed: boolean;
  validity?: GPUVector<'uint32'>;
  range: Integer64Range;
}): Promise<WideInteger64Summary> {
  return aggregateInteger64OnGpu({
    device: props.device,
    words: props.words,
    signed: props.signed,
    validity: props.validity,
    selectionExpression: makeInteger64RangeSelectionExpression(props.range, props.signed)
  });
}

/** Independent BigInt reference for the range filter. */
export function isInteger64InRange(value: bigint, range: Integer64Range): boolean {
  if (
    range.lower &&
    (range.lower.inclusive ? value < range.lower.value : value <= range.lower.value)
  )
    return false;
  return !(
    range.upper && (range.upper.inclusive ? value > range.upper.value : value >= range.upper.value)
  );
}
