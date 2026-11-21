// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

const getTypedArrayTag = Object.getOwnPropertyDescriptor(
  Object.getPrototypeOf(Uint8Array.prototype),
  Symbol.toStringTag
)!.get!;

/**
 * @internal Validate one point-cloud port once, then borrow only the requested lanes. Typed
 * arrays must have the declared element type; Uint8Array/ArrayBuffer are little-endian bytes.
 * No readback-sized storage is allocated, including for unaligned byte subviews.
 */
export function getPointCloudReadbackLaneReader(
  id: string,
  name: string,
  value: unknown,
  format: 'float32' | 'uint32',
  minimumLength: number
): (lane: number) => number {
  const tag = ArrayBuffer.isView(value) ? getTypedArrayTag.call(value) : undefined;
  const expectedTag = format === 'float32' ? 'Float32Array' : 'Uint32Array';
  const isBuffer = Object.prototype.toString.call(value) === '[object ArrayBuffer]';
  validateRequirement(
    id,
    tag === expectedTag || tag === 'Uint8Array' || isBuffer,
    `${name} must be ${expectedTag}, Uint8Array or ArrayBuffer; received ${tag ?? Object.prototype.toString.call(value)}`
  );
  if (tag === expectedTag) {
    const values = value as Float32Array | Uint32Array;
    validateRequirement(
      id,
      values.length >= minimumLength,
      `${name} requires at least ${minimumLength} elements; received ${values.length}`
    );
    return lane => values[lane];
  }
  const bytes = tag === 'Uint8Array' ? (value as Uint8Array) : undefined;
  const byteLength = bytes?.byteLength ?? (value as ArrayBuffer).byteLength;
  validateRequirement(
    id,
    byteLength % 4 === 0,
    `${name} bytes must contain whole 4-byte elements; received ${byteLength} bytes`
  );
  validateRequirement(
    id,
    byteLength / 4 >= minimumLength,
    `${name} requires at least ${minimumLength} elements; received ${byteLength / 4}`
  );
  const data = new DataView(
    bytes?.buffer ?? (value as ArrayBuffer),
    bytes?.byteOffset ?? 0,
    byteLength
  );
  return format === 'float32'
    ? lane => data.getFloat32(lane * 4, true)
    : lane => data.getUint32(lane * 4, true);
}
