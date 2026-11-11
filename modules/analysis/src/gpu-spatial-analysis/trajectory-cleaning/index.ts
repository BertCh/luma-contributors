// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  GPUTrajectoryOutlierFlags,
  GPU_TRAJECTORY_OUTLIER_FLAGS,
  GPU_TRAJECTORY_OUTLIER_FLAGS_PARAMETER_LENGTH,
  getGPUTrajectoryOutlierFlagsParameterValues
} from './gpu-trajectory-outlier-flags';
export type {
  GPUTrajectoryOutlierFlagsProps,
  GPUTrajectoryOutlierFlagsSettings
} from './gpu-trajectory-outlier-flags';
export {
  GPUTrajectoryHampel,
  GPU_TRAJECTORY_HAMPEL_MAXIMUM_WINDOW_SIZE,
  GPU_TRAJECTORY_HAMPEL_PARAMETER_LENGTH,
  getGPUTrajectoryHampelParameterValues
} from './gpu-trajectory-hampel';
export type {
  GPUTrajectoryHampelProps,
  GPUTrajectoryHampelSettings
} from './gpu-trajectory-hampel';
export {
  GPUTrajectorySplit,
  GPU_TRAJECTORY_SPLIT_PARAMETER_LENGTH,
  getGPUTrajectorySplitParameterValues
} from './gpu-trajectory-split';
export type {
  GPUTrajectorySplitMode,
  GPUTrajectorySplitOutput,
  GPUTrajectorySplitProps,
  GPUTrajectorySplitSettings
} from './gpu-trajectory-split';
export {
  GPUTrajectoryThinning,
  GPU_TRAJECTORY_THINNING_DEFAULT_MAXIMUM_SEARCH_ROWS,
  GPU_TRAJECTORY_THINNING_MAXIMUM_SEARCH_ROWS,
  GPU_TRAJECTORY_THINNING_PARAMETER_LENGTH,
  getGPUTrajectoryThinningParameterValues
} from './gpu-trajectory-thinning';
export type {
  GPUTrajectoryThinningMode,
  GPUTrajectoryThinningOutput,
  GPUTrajectoryThinningProps,
  GPUTrajectoryThinningSettings
} from './gpu-trajectory-thinning';
export {GPUTrajectoryTrackRelativeTimes} from './gpu-trajectory-track-relative-times';
export type {GPUTrajectoryTrackRelativeTimesProps} from './gpu-trajectory-track-relative-times';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {
  computeTrajectorySplitOnCPU,
  CPUTrajectorySplitOperation
} from './cpu-trajectory-split-operation';
export type {
  CPUTrajectorySplitInput,
  CPUTrajectorySplitOptions,
  CPUTrajectorySplitResult
} from './cpu-trajectory-split-operation';
export {
  computeTrajectoryThinningOnCPU,
  CPUTrajectoryThinningOperation
} from './cpu-trajectory-thinning-operation';
export type {
  CPUTrajectoryThinningInput,
  CPUTrajectoryThinningOptions,
  CPUTrajectoryThinningResult
} from './cpu-trajectory-thinning-operation';
export {GPUTrajectorySplitOperation} from './gpu-trajectory-split-operation';
export type {TrajectorySplitOperationProps} from './gpu-trajectory-split-operation';
export {GPUTrajectoryThinningOperation} from './gpu-trajectory-thinning-operation';
export type {TrajectoryThinningOperationProps} from './gpu-trajectory-thinning-operation';
