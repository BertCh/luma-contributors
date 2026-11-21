// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

const words = new DataView(new ArrayBuffer(4));
function floatBits(value: number): number {
  words.setFloat32(0, value, true);
  return words.getUint32(0, true);
}
function floatParts(value: number): {significand: bigint; exponent: number} {
  const bits = floatBits(value);
  const exponent = (bits >>> 23) & 255;
  const significand = BigInt((bits & 0x7fffff) | (exponent ? 0x800000 : 0));
  return {
    significand: bits >>> 31 ? -significand : significand,
    exponent: exponent ? exponent - 150 : -149
  };
}

/** Exact single rounding of first + float32(slice) * interval, matching explicit WGSL fma.
 * Integer significands avoid double rounding in JS and keep default endpoints independent of
 * whether the backend contracts an ordinary multiply/add expression.
 */
export function createDensitySliceTime(first: number, interval: number): (slice: number) => number {
  const origin = floatParts(first);
  const step = floatParts(interval);
  const exponent = Math.min(origin.exponent, step.exponent);
  const initial = origin.significand << BigInt(origin.exponent - exponent);
  const increment = step.significand << BigInt(step.exponent - exponent);
  return slice => {
    const exact = initial + BigInt(Math.fround(slice)) * increment;
    if (exact === 0n) return 0;
    const sign = exact < 0n ? -1 : 1;
    const magnitude = exact < 0n ? -exact : exact;
    const bitCount = magnitude.toString(2).length;
    const shift = Math.max(bitCount - 24, -149 - exponent);
    let rounded: bigint;
    if (shift > 0) {
      rounded = magnitude >> BigInt(shift);
      const remainder = magnitude - (rounded << BigInt(shift));
      const half = 1n << BigInt(shift - 1);
      if (remainder > half || (remainder === half && (rounded & 1n) !== 0n)) rounded++;
    } else rounded = magnitude << BigInt(-shift);
    return Math.fround(sign * Number(rounded) * 2 ** (exponent + shift));
  };
}

function floatKey(value: number): number {
  const bits = floatBits(value);
  return bits >>> 31 ? 0x80000000 - (bits & 0x7fffffff) : 0x80000000 + bits;
}
function floatSpacing(value: number): number {
  const exponent = (floatBits(value) >>> 23) & 255;
  return 2 ** (exponent ? exponent - 150 : -149);
}

/** Find the first non-advancing index in bounded exponent-bin searches (at most 255 exponent bins
 * per sign and 24 binary-search levels), rather than walking all slices. Within a binade,
 * steps below one ULP advance by zero or one word, so a prefix word deficit locates the first
 * repeat. An exact one-ULP step checks the first two ties explicitly (ties-to-even can alternate).
 */
export function findDensityRepeatedSlice(
  time: (slice: number) => number,
  interval: number,
  sliceCount: number
): number | undefined {
  const lastIndex = Math.min(sliceCount - 1, 2 ** 24);
  const search = (start: number, end: number): number | undefined => {
    if (end <= start) return undefined;
    const first = time(start);
    const last = time(end);
    if (!Number.isFinite(last)) {
      let low = start + 1;
      let high = end;
      while (low < high) {
        const middle = Math.floor((low + high) / 2);
        if (Number.isFinite(time(middle))) low = middle + 1;
        else high = middle;
      }
      return low;
    }
    const spacing = Math.max(floatSpacing(first), floatSpacing(last));
    if (interval > spacing) return undefined;
    const firstBits = floatBits(first);
    const lastBits = floatBits(last);
    if ((firstBits & 0xff800000) === (lastBits & 0xff800000)) {
      if (interval === spacing) {
        for (let index = start + 1; index <= Math.min(end, start + 2); index++)
          if (!(time(index) > time(index - 1))) return index;
        return undefined;
      }
      const initialKey = floatKey(first);
      if (floatKey(last) - initialKey === end - start) return undefined;
      let low = start + 1;
      let high = end;
      while (low < high) {
        const middle = Math.floor((low + high) / 2);
        if (floatKey(time(middle)) - initialKey < middle - start) high = middle;
        else low = middle + 1;
      }
      return low;
    }
    if (end === start + 1) return last > first ? undefined : end;
    const middle = Math.floor((start + end) / 2);
    return search(start, middle) ?? search(middle, end);
  };
  return search(0, lastIndex) ?? (sliceCount - 1 > 2 ** 24 ? 2 ** 24 + 1 : undefined);
}
