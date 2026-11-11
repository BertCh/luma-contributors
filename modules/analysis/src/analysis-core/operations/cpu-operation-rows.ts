// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCPUVector, GPUGraphVectorFormat} from '@luma.gl/graph/gpu-core';

/** Inputs the published mixed executor passes to a `GPUCPUOperation` callback. */
export type CPUOperationInputs = Readonly<Record<string, GPUCPUVector | number>>;

type Float32Format = 'float32' | 'float32x2' | 'float32x3' | 'float32x4';
type Uint32Format = 'uint32' | 'uint32x2' | 'uint32x3' | 'uint32x4';
type Int32Format = 'sint32' | 'sint32x2' | 'sint32x3' | 'sint32x4';

function getRows(
  id: string,
  inputs: CPUOperationInputs,
  port: string,
  format: GPUGraphVectorFormat
): Float32Array | Uint32Array | Int32Array {
  const input = inputs[port];
  if (input === undefined || typeof input === 'number') {
    throw new Error(`${id} input "${port}" must be a ${format} vector`);
  }
  if (input.format !== format) {
    throw new Error(`${id} input "${port}" must be ${format}, not ${input.format}`);
  }
  // One chunk is already packed; `pack()` would copy it again.
  return input.chunks.length === 1 ? input.chunks[0].values : input.pack();
}

/**
 * Returns the packed rows of a float32 input of a CPU operation. The array is a borrowed snapshot:
 * do not write to it.
 *
 * @param id Operation ID, for the message.
 * @param inputs Inputs of the `execute` callback.
 * @param port Port name, a key of the operation's `inputs`.
 * @param format Declared format of the value.
 * @throws If the port is missing, is a scalar or has another format.
 */
export function getCPUOperationFloat32Rows(
  id: string,
  inputs: CPUOperationInputs,
  port: string,
  format: Float32Format = 'float32'
): Float32Array {
  return getRows(id, inputs, port, format) as Float32Array;
}

/**
 * Returns the packed rows of a uint32 input of a CPU operation. The array is a borrowed snapshot:
 * do not write to it.
 *
 * @param id Operation ID, for the message.
 * @param inputs Inputs of the `execute` callback.
 * @param port Port name, a key of the operation's `inputs`.
 * @param format Declared format of the value.
 * @throws If the port is missing, is a scalar or has another format.
 */
export function getCPUOperationUint32Rows(
  id: string,
  inputs: CPUOperationInputs,
  port: string,
  format: Uint32Format = 'uint32'
): Uint32Array {
  return getRows(id, inputs, port, format) as Uint32Array;
}

/**
 * Returns the packed rows of a sint32 input of a CPU operation. The array is a borrowed snapshot:
 * do not write to it.
 *
 * @param id Operation ID, for the message.
 * @param inputs Inputs of the `execute` callback.
 * @param port Port name, a key of the operation's `inputs`.
 * @param format Declared format of the value.
 * @throws If the port is missing, is a scalar or has another format.
 */
export function getCPUOperationInt32Rows(
  id: string,
  inputs: CPUOperationInputs,
  port: string,
  format: Int32Format = 'sint32'
): Int32Array {
  return getRows(id, inputs, port, format) as Int32Array;
}
