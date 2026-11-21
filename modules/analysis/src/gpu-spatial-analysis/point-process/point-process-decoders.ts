// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

type FloatInput = Float32Array | Uint8Array | ArrayBuffer;
type UintInput = Uint32Array | Uint8Array | ArrayBuffer;
function readWords(id: string, input: FloatInput | UintInput, minimum: number, integer: boolean) {
  const bytes =
    input instanceof ArrayBuffer
      ? new Uint8Array(input)
      : new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  validateRequirement(
    id,
    bytes.byteLength % 4 === 0 && bytes.byteLength / 4 >= minimum,
    `input must contain whole 4-byte words and at least ${minimum} words; received ${bytes.byteLength} bytes`
  );
  const aligned = bytes.byteOffset % 4 === 0 ? bytes : bytes.slice();
  return integer
    ? new Uint32Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 4)
    : new Float32Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 4);
}
/** Decode the four displayed statistics; integrated comparisons retain more precision internally. */
export function decodeGPUPointPatternEnvelopeStatistics(input: FloatInput) {
  const words = readWords('GPUPointPatternEnvelope', input, 4, false);
  return {
    observedStatistic: words[0],
    conservativePValue: words[1],
    pValueResolution: words[2],
    simulationCount: words[3]
  };
}
/** Decode the scalar envelope status without interpreting a missing word as success. */
export function decodeGPUPointPatternEnvelopeStatus(input: UintInput) {
  const code = readWords('GPUPointPatternEnvelope', input, 1, true)[0];
  return {code, complete: code === 0, invalidInput: code !== 0};
}
/** Decode one [selectedCount, code] row in allocation order. Code 2 exhausts random draws. */
export function decodeGPUSpatialBalancedSampleStatus(input: UintInput, allocationRow = 0) {
  validateCapacity('GPUSpatialBalancedSample', 'allocationRow', allocationRow, 0);
  const words = readWords('GPUSpatialBalancedSample', input, 2 * (allocationRow + 1), true);
  const code = words[2 * allocationRow + 1];
  return {
    count: words[2 * allocationRow],
    code,
    complete: code === 0,
    invalidInput: code === 1,
    iterationLimit: code === 2
  };
}
/** Decode one fixed-stride wave status; incomplete waves cannot support weighted estimates. */
export function decodeGPUContinuousSpatialSampleStatus(input: UintInput, waveRow = 0) {
  validateCapacity('GPUContinuousSpatialSample', 'waveRow', waveRow, 0);
  const words = readWords('GPUContinuousSpatialSample', input, 4 * (waveRow + 1), true);
  const offset = 4 * waveRow,
    flags = words[offset + 3];
  return {
    count: words[offset],
    requiredCount: words[offset + 1],
    randomAcceptedCount: words[offset + 2],
    flags,
    complete: flags === 0,
    invalidInput: Boolean(flags & 1),
    overflow: Boolean(flags & 2),
    candidateOverflow: Boolean(flags & 4)
  };
}
/** Decode one output slot's source, replicate, lattice ordinal and role code (0/1/2). */
export function decodeGPUContinuousSpatialSampleProvenance(input: UintInput, slotRow = 0) {
  validateCapacity('GPUContinuousSpatialSample', 'slotRow', slotRow, 0);
  const words = readWords('GPUContinuousSpatialSample', input, 4 * (slotRow + 1), true);
  const offset = 4 * slotRow;
  return {
    sourceRow: words[offset],
    replicate: words[offset + 1],
    candidateOrdinal: words[offset + 2],
    role: words[offset + 3]
  };
}
