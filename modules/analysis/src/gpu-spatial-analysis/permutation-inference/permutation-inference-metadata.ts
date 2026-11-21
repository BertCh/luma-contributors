// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {GPUInferenceMetadata} from '../../analysis-core/contracts/inference';
import {
  getTypedArrayName,
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getPackedFloat32Values,
  getPackedUint32Values
} from '../../analysis-core/kernels/packed-readback';
import {GPU_GLOBAL_PERMUTATION_SUMMARY} from './gpu-global-permutation-test';
import {GPU_LOCAL_PERMUTATION_NOT_TESTED} from './gpu-local-permutation-test';

/** Planned inference; actual counts are attached only after caller readback. */
export type GPUPermutationInferenceMetadata = Omit<GPUInferenceMetadata, 'permutationCount'> & {
  /** Requested draw count. */
  requestedPermutationCount: number;
  /** Actual draws for a global result; present after decoding the summary. */
  actualPermutationCount?: number;
  stoppingRule: {method: 'fixed-count'} | {method: 'besag-clifford'; stopAfterExceedances: number};
};

function getReadbackWords<Value extends Float32Array | Uint32Array>(
  id: string,
  input: Value | Uint8Array | ArrayBuffer,
  kind: 'float32' | 'uint32'
): Value {
  const typedArrayName = getTypedArrayName(input);
  const isBuffer = Object.prototype.toString.call(input) === '[object ArrayBuffer]';
  const expectedType = kind === 'float32' ? 'Float32Array' : 'Uint32Array';
  validateRequirement(
    id,
    isBuffer || typedArrayName === 'Uint8Array' || typedArrayName === expectedType,
    `readback must use ${expectedType} elements or raw Uint8Array/ArrayBuffer bytes; received ${typedArrayName}`
  );
  validateRequirement(
    id,
    input.byteLength % 4 === 0,
    `readback byte length must be a whole number of 4-byte elements; received ${input.byteLength}`
  );
  return (
    kind === 'float32' ? getPackedFloat32Values(input) : getPackedUint32Values(input)
  ) as Value;
}

function validatePermutationInferenceCountName(id: string, metadata: object): void {
  validateRequirement(
    id,
    !('permutationCount' in metadata),
    'permutationCount was removed from planned inference metadata. Use requestedPermutationCount'
  );
}

function getMinimumRecordedPermutationCount(metadata: GPUPermutationInferenceMetadata): number {
  return metadata.stoppingRule.method === 'fixed-count'
    ? metadata.requestedPermutationCount
    : Math.min(metadata.requestedPermutationCount, metadata.stoppingRule.stopAfterExceedances);
}

/** Attach actual global draws recorded in output.summary, without an implicit GPU readback. */
export function decodeGPUGlobalPermutationTestInferenceMetadata(
  metadata: GPUPermutationInferenceMetadata,
  summary: Float32Array | Uint8Array | ArrayBuffer
) {
  const id = 'decodeGPUGlobalPermutationTestInferenceMetadata';
  validatePermutationInferenceCountName(id, metadata);
  const words = getReadbackWords<Float32Array>(id, summary, 'float32');
  validateCapacity(id, 'summary length', words.length, GPU_GLOBAL_PERMUTATION_SUMMARY.length);
  const count = words[GPU_GLOBAL_PERMUTATION_SUMMARY.permutationCount];
  if (!Number.isNaN(count))
    validateCapacity(
      id,
      'recorded permutationCount',
      count,
      getMinimumRecordedPermutationCount(metadata),
      metadata.requestedPermutationCount
    );
  const actualPermutationCount = Number.isNaN(count) ? 0 : count;
  return {
    ...metadata,
    actualPermutationCount,
    available:
      actualPermutationCount > 0 &&
      Number.isFinite(words[GPU_GLOBAL_PERMUTATION_SUMMARY.pseudoPValue]),
    stoppedEarly:
      actualPermutationCount > 0 && actualPermutationCount < metadata.requestedPermutationCount
  };
}

/** Attach actual draws per local row; the existing NOT_TESTED sentinel is retained. */
export function decodeGPULocalPermutationTestInferenceMetadata(
  metadata: GPUPermutationInferenceMetadata,
  counts: Uint32Array | Uint8Array | ArrayBuffer,
  rowCount: number
) {
  const id = 'decodeGPULocalPermutationTestInferenceMetadata';
  validatePermutationInferenceCountName(id, metadata);
  validateCapacity(id, 'rowCount', rowCount, 1);
  const words = getReadbackWords<Uint32Array>(id, counts, 'uint32');
  validateCapacity(id, 'permutationCounts length', words.length, rowCount);
  const actualPermutationCounts = words.slice(0, rowCount);
  for (const count of actualPermutationCounts)
    if (count !== GPU_LOCAL_PERMUTATION_NOT_TESTED)
      validateCapacity(
        id,
        'recorded permutationCount',
        count,
        getMinimumRecordedPermutationCount(metadata),
        metadata.requestedPermutationCount
      );
  return {...metadata, actualPermutationCounts};
}
