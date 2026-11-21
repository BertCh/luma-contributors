// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

const floatWord = new DataView(new ArrayBuffer(4));

/** @internal Exact signed integer coefficient of a finite float32 in units of 2^-149. */
export function getPointPatternDyadicInteger(value: number): bigint {
  floatWord.setFloat32(0, value, true);
  const bits = floatWord.getUint32(0, true);
  const exponent = (bits >>> 23) & 0xff;
  const fraction = bits & 0x7fffff;
  const magnitude =
    exponent === 0 ? BigInt(fraction) : BigInt(fraction | 0x800000) << BigInt(exponent - 1);
  return bits >>> 31 === 0 ? magnitude : -magnitude;
}

/** @internal Round a nonnegative integrated coefficient in units of 2^-447 once to float32. */
export function roundPointPatternDyadicStatistic(coefficient: bigint): number {
  if (coefficient === 0n) return 0;
  // Normal outputs retain 24 significant bits. Subnormal outputs have quantum 2^-149,
  // i.e. 2^298 coefficient units. Decide halfway cases before any Number conversion.
  const discardedBitCount = Math.max(coefficient.toString(2).length - 24, 298);
  const shift = BigInt(discardedBitCount);
  let mantissa = coefficient >> shift;
  const remainder = coefficient - (mantissa << shift);
  const halfway = 1n << (shift - 1n);
  if (remainder > halfway || (remainder === halfway && (mantissa & 1n) !== 0n)) mantissa++;
  // At most 25 bits including a rounding carry: conversion and power-of-two scaling are exact.
  return Math.fround(Number(mantissa) * 2 ** (discardedBitCount - 447));
}
