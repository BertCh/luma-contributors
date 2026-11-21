// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCPUVector, GPUGraphVectorFormat} from '@luma.gl/graph/gpu-core';
import {validateRequirement} from '../kernels/gpu-contributor-utils';

/** Inputs the published mixed executor passes to a `GPUCPUOperation` callback. */
export type CPUOperationInputs = Readonly<Record<string, GPUCPUVector | number>>;

type Float32Format = 'float32' | 'float32x2' | 'float32x3' | 'float32x4';
type Uint32Format = 'uint32' | 'uint32x2' | 'uint32x3' | 'uint32x4';
type Int32Format = 'sint32' | 'sint32x2' | 'sint32x3' | 'sint32x4';

// The intrinsic getter reads a typed array's element brand across realms. It also ignores an
// instance's own Symbol.toStringTag, so a differently backed array cannot spoof the expected type.
const TYPED_ARRAY_NAME_GETTER = Object.getOwnPropertyDescriptor(
  Object.getPrototypeOf(Uint8Array.prototype),
  Symbol.toStringTag
)?.get;

function getRows(
  id: string,
  inputs: CPUOperationInputs,
  port: string,
  format: GPUGraphVectorFormat,
  elementClass: typeof Float32Array | typeof Uint32Array | typeof Int32Array
): Float32Array | Uint32Array | Int32Array {
  const input = inputs[port];
  if (input === undefined || typeof input === 'number') {
    throw new Error(`${id} input "${port}" must be a ${format} vector`);
  }
  if (input.format !== format) {
    throw new Error(`${id} input "${port}" must be ${format}, not ${input.format}`);
  }
  // One chunk is already packed; `pack()` would copy it again.
  const rows = input.chunks.length === 1 ? input.chunks[0].values : input.pack();
  const observedClass = TYPED_ARRAY_NAME_GETTER?.call(rows);
  validateRequirement(
    id,
    observedClass === elementClass.name,
    `input "${port}" must contain ${elementClass.name}, not ${observedClass ?? rows?.constructor?.name ?? typeof rows}`
  );
  return rows;
}

/**
 * Returns the packed rows of a float32 input of a CPU operation. The array is a borrowed snapshot:
 * do not write to it.
 *
 * @param id Operation ID, for the message.
 * @param inputs Inputs of the `execute` callback.
 * @param port Port name, a key of the operation's `inputs`.
 * @param format Declared format of the value.
 * @throws If the port is missing, is a scalar, has another format or contains another array class.
 */
export function getCPUOperationFloat32Rows(
  id: string,
  inputs: CPUOperationInputs,
  port: string,
  format: Float32Format = 'float32'
): Float32Array {
  return getRows(id, inputs, port, format, Float32Array) as Float32Array;
}

/**
 * Returns the packed rows of a uint32 input of a CPU operation. The array is a borrowed snapshot:
 * do not write to it.
 *
 * @param id Operation ID, for the message.
 * @param inputs Inputs of the `execute` callback.
 * @param port Port name, a key of the operation's `inputs`.
 * @param format Declared format of the value.
 * @throws If the port is missing, is a scalar, has another format or contains another array class.
 */
export function getCPUOperationUint32Rows(
  id: string,
  inputs: CPUOperationInputs,
  port: string,
  format: Uint32Format = 'uint32'
): Uint32Array {
  return getRows(id, inputs, port, format, Uint32Array) as Uint32Array;
}

/**
 * Returns the packed rows of a sint32 input of a CPU operation. The array is a borrowed snapshot:
 * do not write to it.
 *
 * @param id Operation ID, for the message.
 * @param inputs Inputs of the `execute` callback.
 * @param port Port name, a key of the operation's `inputs`.
 * @param format Declared format of the value.
 * @throws If the port is missing, is a scalar, has another format or contains another array class.
 */
export function getCPUOperationInt32Rows(
  id: string,
  inputs: CPUOperationInputs,
  port: string,
  format: Int32Format = 'sint32'
): Int32Array {
  return getRows(id, inputs, port, format, Int32Array) as Int32Array;
}
