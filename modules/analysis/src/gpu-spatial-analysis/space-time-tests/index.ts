// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {GPUKnoxTest} from './gpu-knox-test';
export type {GPUKnoxTestOutput, GPUKnoxTestProps} from './gpu-knox-test';
export {
  GPU_NEAR_REPEAT_CELL_SUMMARY,
  GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH,
  GPUNearRepeatKnox
} from './gpu-near-repeat-knox';
export type {GPUNearRepeatKnoxOutput, GPUNearRepeatKnoxProps} from './gpu-near-repeat-knox';
export {
  GPU_LOCAL_KNOX_SUMMARY,
  GPU_LOCAL_KNOX_SUMMARY_LENGTH,
  GPULocalKnox
} from './gpu-local-knox';
export type {GPULocalKnoxOutput, GPULocalKnoxProps} from './gpu-local-knox';
export {GPU_JACQUEZ_MAXIMUM_K, GPUJacquezTest} from './gpu-jacquez-test';
export type {GPUJacquezTestOutput, GPUJacquezTestProps} from './gpu-jacquez-test';
export {
  getGPUSpaceTimeGuardedRelativeTimes,
  GPU_NEAR_REPEAT_MAXIMUM_BANDS,
  GPU_NEAR_REPEAT_MAXIMUM_CELLS,
  GPU_SPACE_TIME_BAND_RESOLUTION_FRACTION
} from './space-time-bands';
export type {
  GPUSpaceTimeGuardedRelativeTimes,
  GPUSpaceTimeGuardedRelativeTimesOptions
} from './space-time-bands';
export {GPUMantelTest, GPU_MANTEL_POINTPATS_TRANSFORM} from './gpu-mantel-test';
export type {
  GPUMantelTestOutput,
  GPUMantelTestProps,
  GPUMantelTransform
} from './gpu-mantel-test';
export {
  getGPUSpaceTimeParameterValues,
  getGPUSpaceTimeRelativeTimes,
  getKnoxPoissonPValue,
  GPU_SPACE_TIME_PARAMETER_LENGTH,
  GPU_SPACE_TIME_SUMMARY,
  GPU_SPACE_TIME_SUMMARY_LENGTH
} from './space-time-parameters';
export type {GPUSpaceTimeSettings, GPUSpaceTimeRelativeTimes} from './space-time-parameters';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {
  CPU_MANTEL_TEST_MAXIMUM_PERMUTATIONS,
  computeMantelTestOnCPU,
  CPUMantelTestOperation
} from './cpu-mantel-test-operation';
export type {
  CPUMantelTestInput,
  CPUMantelTestOptions,
  CPUMantelTestResult
} from './cpu-mantel-test-operation';
export {GPUMantelTestOperation} from './gpu-mantel-test-operation';
export type {MantelTestOperationProps} from './gpu-mantel-test-operation';
