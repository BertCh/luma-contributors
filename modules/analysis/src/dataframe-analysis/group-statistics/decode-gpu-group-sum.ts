// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {joinWordsToBigInt} from '../column-profile/wide-numeric-gpu-reference';
import {getWideNumericWordPair} from '../column-profile/wide-numeric-word-readback';

/** Exact signed fixed-point integer and its binary64 approximation in the measure's units. */
export type DecodedGPUGroupSum = {scaledInteger: bigint; value: number};

/**
 * Decode one row of GPUGroupStatistics columns[].sums using that producer's sumScale.
 * Words are signed int64 little-endian (low word then high word), also used by key-join and
 * cell-table scaled sums. Byte readbacks honor their logical offset and length, including
 * unaligned byte views. Partial words or pairs and missing rows are rejected. Each call reads
 * only the requested pair, so traversing every row does not copy the entire column repeatedly.
 * `scaledInteger` retains every bit; `value` can round beyond binary64 precision. This decodes
 * the quantized `sums` statistic, not the independently computed floating-point `sumValues`.
 * Check the producer's sumOverflow separately before interpreting a wrapped/clamped total.
 */
export function decodeGPUGroupSum(
  words: Uint32Array | Uint8Array | ArrayBuffer,
  row: number,
  sumScale: number
): DecodedGPUGroupSum {
  const operation = 'decodeGPUGroupSum';
  validateRequirement(
    operation,
    Number.isSafeInteger(row) && row >= 0 && Number.isSafeInteger((row + 1) * 2),
    `row must be a non-negative addressable integer; received ${row}`
  );
  validateRequirement(
    operation,
    Number.isFinite(sumScale) && sumScale > 0 && Math.fround(sumScale) === sumScale,
    `sumScale must be a positive finite float32 value; received ${sumScale}`
  );
  const pair = getWideNumericWordPair(words, row, `${operation} sums`, false);
  const scaledInteger = joinWordsToBigInt(pair, true);
  return {scaledInteger, value: divideScaledInteger(scaledInteger, sumScale)};
}

/** Round the exact integer / float32 scale rational once, ties to even. */
function divideScaledInteger(integer: bigint, sumScale: number): number {
  if (integer === 0n) return 0;
  const scaleBits = new Uint32Array(Float32Array.of(sumScale).buffer)[0];
  const exponentBits = (scaleBits >>> 23) & 0xff;
  const significand = BigInt((scaleBits & 0x7fffff) | (exponentBits ? 0x800000 : 0));
  const scaleExponent = (exponentBits || 1) - 127 - 23;
  const magnitude = integer < 0n ? -integer : integer;
  const numerator = scaleExponent < 0 ? magnitude << BigInt(-scaleExponent) : magnitude;
  const denominator = scaleExponent > 0 ? significand << BigInt(scaleExponent) : significand;
  let exponent = numerator.toString(2).length - denominator.toString(2).length;
  const belowPower =
    exponent >= 0
      ? numerator < denominator << BigInt(exponent)
      : numerator << BigInt(-exponent) < denominator;
  if (belowPower) exponent--;
  // Nonzero int64 / positive finite float32 always lies in binary64's normal range.
  const shift = exponent - 52;
  const scaledNumerator = shift < 0 ? numerator << BigInt(-shift) : numerator;
  const scaledDenominator = shift > 0 ? denominator << BigInt(shift) : denominator;
  let rounded = scaledNumerator / scaledDenominator;
  const remainder = (scaledNumerator % scaledDenominator) * 2n;
  if (remainder > scaledDenominator || (remainder === scaledDenominator && (rounded & 1n) !== 0n))
    rounded++;
  const value = Number(rounded) * 2 ** shift;
  return integer < 0n ? -value : value;
}
