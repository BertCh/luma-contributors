// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUResidentScalarFormat} from '@luma-contributors/kepler';

/** Native d3 bins use double-precision endpoints, independently of uploaded scalar storage. */
export function makeResidentNumericBoundaries(
  minimum: number,
  maximum: number,
  binCount: number
): number[] {
  if (minimum === maximum) return [minimum, maximum];
  const step = (maximum - minimum) / binCount;
  const thresholds = Array.from({length: binCount - 1}, (_, index) => minimum + step * (index + 1));
  // Match d3.bin with an explicit threshold array, including duplicate/upper-bound thresholds.
  return [
    minimum,
    ...thresholds.filter(threshold => threshold > minimum && threshold <= maximum),
    maximum
  ];
}

export function decodeResidentNumericOrder(value: number, format: GPUResidentScalarFormat): number {
  if (format === 'uint32') return value;
  if (format === 'sint32') return (value ^ 0x80000000) | 0;
  const words = new Uint32Array([value & 0x80000000 ? (value ^ 0x80000000) >>> 0 : ~value >>> 0]);
  return new Float32Array(words.buffer)[0];
}

/** Smallest representable source value >= the exact native threshold, encoded in sortable bits. */
export function encodeResidentNumericThreshold(
  threshold: number,
  format: GPUResidentScalarFormat
): number {
  if (format === 'uint32') return Math.ceil(threshold) >>> 0;
  if (format === 'sint32') return (Math.ceil(threshold) ^ 0x80000000) >>> 0;
  const values = new Float32Array([threshold]);
  const words = new Uint32Array(values.buffer);
  if (values[0] < threshold)
    words[0] = values[0] === 0 ? 1 : values[0] > 0 ? words[0] + 1 : words[0] - 1;
  const bits = values[0] === 0 ? 0 : words[0];
  return bits & 0x80000000 ? ~bits >>> 0 : (bits ^ 0x80000000) >>> 0;
}
