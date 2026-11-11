// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  GPU_SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_ITERATIONS,
  GPU_SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_PARENTS,
  GPU_SPACE_TIME_HAWKES_MAXIMUM_ITERATIONS,
  GPU_SPACE_TIME_HAWKES_MAXIMUM_PARENTS,
  GPUSpaceTimeHawkes
} from './gpu-space-time-hawkes';
export type {GPUSpaceTimeHawkesOutput, GPUSpaceTimeHawkesProps} from './gpu-space-time-hawkes';
export {
  getGPUSpaceTimeHawkesParameterValues,
  GPU_SPACE_TIME_HAWKES_DEFAULT_TOLERANCE,
  GPU_SPACE_TIME_HAWKES_ESTIMATE_LENGTH,
  GPU_SPACE_TIME_HAWKES_PARAMETER_LENGTH
} from './space-time-hawkes-parameters';
export type {GPUSpaceTimeHawkesSettings} from './space-time-hawkes-parameters';
export {
  GPU_SPACE_TIME_HAWKES_BRANCHING_RATIO_FLOOR,
  GPU_SPACE_TIME_HAWKES_INTENSITY_FLOOR
} from './space-time-hawkes-wgsl';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {
  CPU_SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_ITERATIONS,
  computeSpaceTimeHawkesOnCPU,
  CPUSpaceTimeHawkesOperation
} from './cpu-space-time-hawkes-operation';
export type {
  CPUSpaceTimeHawkesInput,
  CPUSpaceTimeHawkesOptions,
  CPUSpaceTimeHawkesResult
} from './cpu-space-time-hawkes-operation';
export {GPUSpaceTimeHawkesOperation} from './gpu-space-time-hawkes-operation';
export type {SpaceTimeHawkesOperationProps} from './gpu-space-time-hawkes-operation';
