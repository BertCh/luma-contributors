// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  GPULocalPermutationTest,
  GPU_LOCAL_GEARY_CLASS,
  GPU_LOCAL_PERMUTATION_EXTENDED_MAXIMUM_NEIGHBORS,
  GPU_LOCAL_PERMUTATION_MAXIMUM_NEIGHBORS,
  GPU_LOCAL_PERMUTATION_NOT_TESTED
} from './gpu-local-permutation-test';
export type {
  GPULocalJoinCountCase,
  GPULocalPermutationStatistic,
  GPULocalPermutationTestOutput,
  GPULocalPermutationTestProps
} from './gpu-local-permutation-test';
export {
  GPUGlobalPermutationTest,
  GPU_GLOBAL_PERMUTATION_SUMMARY,
  GPU_GLOBAL_PERMUTATION_TEST_MAXIMUM_PAIR_BLOCK_COUNT,
  getGPUGlobalPermutationBatchSize
} from './gpu-global-permutation-test';
export type {
  GPUGlobalPermutationStatistic,
  GPUGlobalPermutationTestOutput,
  GPUGlobalPermutationTestProps
} from './gpu-global-permutation-test';
export {
  getGPUPermutationMetadata,
  getGPUPermutationParameterValues,
  GPU_PERMUTATION_PARAMETER_LENGTH,
  MAXIMUM_PERMUTATION_COUNT,
  getPermutationSeedKey
} from './permutation-parameters';
export type {GPUPermutationSettings} from './permutation-parameters';
export {createLocalPermutationReplay} from './gpu-local-permutation-replay';
export type {LocalPermutationReplayProps} from './gpu-local-permutation-replay';
export {
  decodeGPUGlobalPermutationTestInferenceMetadata,
  decodeGPULocalPermutationTestInferenceMetadata
} from './permutation-inference-metadata';
export type {GPUPermutationInferenceMetadata} from './permutation-inference-metadata';
export type {GPUPermutationCauseStatus} from './permutation-cause-status';
export {
  GLOBAL_PERMUTATION_PAIR_BLOCK_ROWS,
  MAXIMUM_GLOBAL_PERMUTATION_PAIR_BLOCK_COUNT
} from './permutation-limits';
