// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getPackedFloat32Values,
  getPackedUint32Values,
  type GPUPackedReadbackValues
} from '../../analysis-core/kernels/packed-readback';

const TYPED_ARRAY_NAME_GETTER = Object.getOwnPropertyDescriptor(
  Object.getPrototypeOf(Uint8Array.prototype),
  Symbol.toStringTag
)!.get!;

/** @internal Retain numeric-array semantics while interpreting byte ranges by port element type. */
export function getVecchiaReadbackValues(
  id: string,
  port: string,
  values: GPUPackedReadbackValues,
  length: number,
  format: 'float32' | 'uint32',
  requireElementType = false
): ArrayLike<number> {
  const typedArrayName = ArrayBuffer.isView(values)
    ? TYPED_ARRAY_NAME_GETTER.call(values)
    : undefined;
  const bytes =
    typedArrayName === 'Uint8Array' ||
    Object.prototype.toString.call(values) === '[object ArrayBuffer]';
  const expectedType = format === 'float32' ? 'Float32Array' : 'Uint32Array';
  validateRequirement(
    id,
    !requireElementType ||
      !ArrayBuffer.isView(values) ||
      typedArrayName === 'Uint8Array' ||
      typedArrayName === expectedType,
    `${port} must use ${expectedType} elements or raw Uint8Array/ArrayBuffer bytes; received ${typedArrayName}`
  );
  const received = bytes
    ? (values as Uint8Array | ArrayBuffer).byteLength / 4
    : (values as ArrayLike<number>).length;
  validateRequirement(
    id,
    Number.isInteger(received) && received >= length,
    `${port} must contain whole 4-byte elements and at least ${length} components; received ${received}`
  );
  if (bytes)
    return format === 'float32'
      ? getPackedFloat32Values(values, length)
      : getPackedUint32Values(values, length);
  const numbers = values as ArrayLike<number>;
  for (let index = 0; index < length; index++)
    validateRequirement(
      id,
      typeof numbers[index] === 'number',
      `${port}[${index}] must exist and be a number; received ${numbers[index]}`
    );
  return numbers;
}

/** @internal Shared named status without changing the three established count words. */
export function decodeVecchiaStatus(
  id: string,
  values: GPUPackedReadbackValues,
  neighborhoodStatus?: GPUPackedReadbackValues
) {
  const words = getVecchiaReadbackValues(id, 'status', values, 3, 'uint32', true);
  const loss =
    neighborhoodStatus &&
    getVecchiaReadbackValues(id, 'neighborhoodStatus', neighborhoodStatus, 3, 'uint32', true);
  return {
    failedSiteCount: words[0],
    excludedSiteCount: words[1],
    usedSiteCount: words[2],
    complete:
      words[0] === 0 &&
      words[2] > 0 &&
      (!loss || (loss[0] !== 0 && loss[1] === 0 && loss[2] === 0)),
    overflow: Boolean(loss && loss[1] !== 0),
    candidateOverflow: Boolean(loss && loss[2] !== 0)
  };
}
