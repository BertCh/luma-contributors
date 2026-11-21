// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {AnalysisValidationError} from './analysis-validation-error';

/** Bytes from a GPU read, or numeric words (ordinary arrays retain numeric-word semantics). */
export type GPUPackedReadbackValues = ArrayBuffer | ArrayLike<number>;
/** Scalar formats supported by the packed application reader. */
export type GPUPackedReadbackFormat = 'uint32' | 'float32';

/**
 * Copies little-endian uint32 words from bytes, including unaligned subviews. Numeric arrays are
 * values, not bytes. Rejects partial elements and fewer than `minimumLength` words.
 */
export function getPackedUint32Values(
  values: GPUPackedReadbackValues,
  minimumLength = 0
): Uint32Array {
  return getPackedValues(values, minimumLength, 'uint32') as Uint32Array;
}

/** Like {@link getPackedUint32Values}, with little-endian float32 words. Always owns its storage. */
export function getPackedFloat32Values(
  values: GPUPackedReadbackValues,
  minimumLength = 0
): Float32Array {
  return getPackedValues(values, minimumLength, 'float32') as Float32Array;
}

function getPackedValues(
  values: GPUPackedReadbackValues,
  minimumLength: number,
  format: GPUPackedReadbackFormat
): Uint32Array | Float32Array {
  const name = `getPacked${format === 'uint32' ? 'Uint32' : 'Float32'}Values`;
  if (!Number.isSafeInteger(minimumLength) || minimumLength < 0) {
    throw new AnalysisValidationError(
      name,
      `minimumLength must be a nonnegative safe integer; received ${minimumLength}`,
      {
        code: 'INTEGER_RANGE',
        path: 'minimumLength',
        received: minimumLength,
        expected: {minimum: 0, safeInteger: true}
      }
    );
  }
  // ArrayBuffer.isView and the intrinsic typed-array tag work across module conditions and realms.
  const typedArrayTag = Object.getOwnPropertyDescriptor(
    Object.getPrototypeOf(Uint8Array.prototype),
    Symbol.toStringTag
  )!.get!;
  const bytes =
    ArrayBuffer.isView(values) && typedArrayTag.call(values) === 'Uint8Array'
      ? (values as Uint8Array)
      : undefined;
  const isBuffer = Object.prototype.toString.call(values) === '[object ArrayBuffer]';
  if (bytes || isBuffer) {
    const byteLength = bytes?.byteLength ?? (values as ArrayBuffer).byteLength;
    if (byteLength % 4 !== 0) {
      throw new AnalysisValidationError(
        name,
        `bytes must contain whole 4-byte elements; received ${byteLength} bytes`,
        {
          code: 'VIEW_LENGTH',
          path: 'values',
          received: byteLength,
          expected: {elementBytes: 4, wholeElements: true}
        }
      );
    }
    if (byteLength / 4 < minimumLength) {
      throw new AnalysisValidationError(
        name,
        `requires at least ${minimumLength} words; received ${byteLength / 4}`,
        {
          code: 'VIEW_LENGTH',
          path: 'values',
          received: byteLength / 4,
          expected: {minimumWords: minimumLength}
        }
      );
    }
    const source = new DataView(
      bytes?.buffer ?? (values as ArrayBuffer),
      bytes?.byteOffset ?? 0,
      byteLength
    );
    const result =
      format === 'uint32' ? new Uint32Array(byteLength / 4) : new Float32Array(byteLength / 4);
    for (let index = 0; index < result.length; index++) {
      result[index] =
        format === 'uint32'
          ? source.getUint32(index * 4, true)
          : source.getFloat32(index * 4, true);
    }
    return result;
  }
  const words = values as ArrayLike<number>;
  if (!words || !Number.isSafeInteger(words.length) || words.length < minimumLength) {
    throw new AnalysisValidationError(
      name,
      `requires at least ${minimumLength} numeric words; received ${words?.length}`,
      {
        code: 'VIEW_LENGTH',
        path: 'values',
        received: words?.length,
        expected: {minimumWords: minimumLength}
      }
    );
  }
  const result =
    format === 'uint32' ? new Uint32Array(words.length) : new Float32Array(words.length);
  for (let index = 0; index < words.length; index++) {
    const word = words[index];
    if (!(index in words) || typeof word !== 'number') {
      throw new AnalysisValidationError(name, `numeric word ${index} must exist and be a number`, {
        code: 'MISSING_PROP',
        path: `values[${index}]`,
        received: word,
        expected: {type: 'number'}
      });
    }
    result[index] = word;
  }
  return result;
}

/**
 * Reads exactly `elementCount` packed scalar elements at the logical `byteOffset` in a buffer.
 * Checks allocation and returned-view bounds, skips zero-length reads, and returns owned storage.
 * For strided or chunked data use the published `@luma.gl/graph` vector transfer APIs instead.
 */
export async function readPackedBuffer<Format extends GPUPackedReadbackFormat>(
  buffer: Buffer,
  byteOffset: number,
  format: Format,
  elementCount: number
): Promise<Format extends 'uint32' ? Uint32Array : Float32Array> {
  const name = `readPackedBuffer ${buffer.id ?? 'buffer'}`;
  if (format !== 'uint32' && format !== 'float32') {
    throw new AnalysisValidationError(
      name,
      `format must be uint32 or float32; received ${format}`,
      {
        code: 'INVALID_OPTION',
        path: 'format',
        received: format,
        expected: {values: ['uint32', 'float32']}
      }
    );
  }
  const byteLength = elementCount * 4;
  if (
    !Number.isSafeInteger(byteOffset) ||
    byteOffset < 0 ||
    !Number.isSafeInteger(elementCount) ||
    elementCount < 0 ||
    !Number.isSafeInteger(byteLength) ||
    !Number.isSafeInteger(byteOffset + byteLength) ||
    byteOffset + byteLength > buffer.byteLength
  ) {
    throw new AnalysisValidationError(
      name,
      `requested byteOffset ${byteOffset} and elementCount ${elementCount} must fit ${buffer.byteLength} bytes`,
      {
        code: 'INTEGER_RANGE',
        path: 'range',
        received: {byteOffset, elementCount},
        expected: {minimum: 0, maximumBytes: buffer.byteLength, source: 'buffer.byteLength'}
      }
    );
  }
  const bytes =
    elementCount === 0 ? new Uint8Array(0) : await buffer.readAsync(byteOffset, byteLength);
  if (bytes.byteLength !== byteLength) {
    throw new AnalysisValidationError(
      name,
      `returned ${bytes.byteLength} bytes; expected ${byteLength}`,
      {
        code: 'VIEW_LENGTH',
        path: 'readAsync result',
        received: bytes.byteLength,
        expected: {bytes: byteLength, source: 'elementCount * 4'}
      }
    );
  }
  return (
    format === 'uint32'
      ? getPackedUint32Values(bytes, elementCount)
      : getPackedFloat32Values(bytes, elementCount)
  ) as Format extends 'uint32' ? Uint32Array : Float32Array;
}
