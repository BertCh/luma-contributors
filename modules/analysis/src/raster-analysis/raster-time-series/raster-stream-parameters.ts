// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateFixedPointMaximumTermCount} from '../../analysis-core/kernels/fixed-point-accumulator';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement,
  getPackedUint32Values,
  getTypedArrayName,
  type GPUPackedReadbackValues
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateParameterTarget} from '../raster-band-utils';

/** Per-submission controls; scale and model settings belong to the immutable encoding plan. */
export type GPURasterStreamFrameSettings = {reset: boolean; activeSliceCount?: number};

function validateRasterStreamCount(
  id: string,
  name: string,
  value: number,
  minimum: number,
  maximum: number
): void {
  validateCapacity(
    {id, contributor: id, messagePrefix: `${id} received ${value} for ${name};`},
    name,
    value,
    minimum,
    maximum
  );
}

/** @internal */
export function validateRasterStreamFrame(
  id: string,
  chunkSliceCount: number,
  activeSliceCount: number,
  maximumChunkSliceCount: number,
  maximumStreamSliceCount?: number,
  reset?: boolean
): void {
  validateRasterStreamCount(id, 'chunkSliceCount', chunkSliceCount, 1, maximumChunkSliceCount);
  validateRasterStreamCount(id, 'activeSliceCount', activeSliceCount, 0, chunkSliceCount);
  if (maximumStreamSliceCount !== undefined) {
    validateRequirement(
      id,
      activeSliceCount <= maximumStreamSliceCount,
      `activeSliceCount must not exceed maximumStreamSliceCount (${maximumStreamSliceCount}); received ${activeSliceCount}`,
      {
        code: 'INTEGER_RANGE',
        path: 'activeSliceCount',
        received: activeSliceCount,
        expected: {minimum: 0, maximum: maximumStreamSliceCount, source: 'maximumStreamSliceCount'}
      }
    );
    validateRequirement(
      id,
      maximumStreamSliceCount > 0 || (reset === true && activeSliceCount === 0),
      `maximumStreamSliceCount must be positive except for an empty reset; received ${maximumStreamSliceCount}`,
      {
        code: 'INTEGER_RANGE',
        path: 'maximumStreamSliceCount',
        received: maximumStreamSliceCount,
        expected: {minimum: 1, exception: 'reset with activeSliceCount 0'}
      }
    );
  }
}

/** @internal */
export function validateRasterStreamHeadroom(id: string, value: number): void {
  validateRasterStreamCount(id, 'maximumStreamSliceCount', value, 0, 0xffffffff);
  validateFixedPointMaximumTermCount(id, 'maximumStreamSliceCount', value);
}

/** Validate float32 representability, preserving the original value for legacy derived-unit arithmetic. @internal */
export function getRasterStreamPositiveFloat(id: string, name: string, value: number): number {
  validateFiniteNumber(
    {
      id,
      contributor: id,
      messagePrefix: `${id} received ${value} for ${name}; positive finite float32 required;`
    },
    name,
    value,
    Number.MIN_VALUE
  );
  return value;
}

/** @internal */
export function validateRasterStreamParameterTarget(
  id: string,
  target: Float32Array,
  minimumWords: number
): void {
  validateRequirement(
    id,
    target.length >= minimumWords,
    `target must hold at least ${minimumWords} values; received ${target.length}`,
    {code: 'VIEW_LENGTH', path: 'target', received: target.length, expected: {minimumWords}}
  );
  validateParameterTarget(id, target, minimumWords);
}

/** Check the model discriminator and every captured/derived field before a plan can write a packet. @internal */
export function validateRasterStreamEncodingPlan(
  id: string,
  plan: object,
  kind: string,
  createExpected: () => object
): void {
  const encoding = plan as Record<string, unknown>;
  validateRequirement(
    id,
    encoding['kind'] === kind,
    `kind must be ${kind}; received ${String(encoding['kind'])}`,
    {code: 'INVALID_OPTION', path: 'kind', received: encoding['kind'], expected: {values: [kind]}}
  );
  for (const [name, expected] of Object.entries(createExpected())) {
    validateRequirement(
      id,
      Object.is(encoding[name], expected),
      `${name} must match the encoding plan derivation (${String(expected)}); received ${String(encoding[name])}`,
      {code: 'REQUIREMENT', path: name, received: encoding[name], expected}
    );
  }
}

/** Reject accidental encoding edits in an otherwise valid plan-based frame. @internal */
export function validateRasterStreamPlanFrame(
  id: string,
  plan: object,
  frame: GPURasterStreamFrameSettings
): void {
  const fields = frame as Record<string, unknown>;
  const encoding = plan as Record<string, unknown>;
  for (const key of Object.keys(fields)) {
    validateRequirement(
      id,
      key === 'reset' || key === 'activeSliceCount' || Object.is(fields[key], encoding[key]),
      `${key} received ${String(fields[key])} differs from encoding plan; create a new plan and reset state instead`,
      {code: 'REQUIREMENT', path: key, received: fields[key], expected: encoding[key]}
    );
  }
}

/** Validate original numeric certificates before narrowing; GPU byte views retain bounded ingress. @internal */
export function getRasterStreamStatusWord(
  id: string,
  input: GPUPackedReadbackValues,
  maximum: number
): number {
  const isBytes =
    Object.prototype.toString.call(input) === '[object ArrayBuffer]' ||
    (ArrayBuffer.isView(input) && getTypedArrayName(input) === 'Uint8Array');
  if (!isBytes)
    validateRasterStreamCount(id, 'status', (input as ArrayLike<number>)[0], 0, maximum);
  const words = getPackedUint32Values(input, 1);
  validateRasterStreamCount(id, 'status', words[0], 0, maximum);
  return words[0];
}
