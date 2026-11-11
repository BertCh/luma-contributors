// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  GPUTrajectoryRecursions,
  GPU_TRAJECTORY_RECURSIONS_DEFAULT_MAXIMUM_TRACK_SEGMENTS,
  GPU_TRAJECTORY_RECURSIONS_MAXIMUM_RADIUS_COUNT,
  GPU_TRAJECTORY_RECURSIONS_MAXIMUM_TRACK_SEGMENTS
} from './gpu-trajectory-recursions';
export type {
  GPUTrajectoryRecursionsOutput,
  GPUTrajectoryRecursionsProps
} from './gpu-trajectory-recursions';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {
  computeTrajectoryRecursionsOnCPU,
  CPUTrajectoryRecursionsOperation
} from './cpu-trajectory-recursions-operation';
export type {CPUTrajectoryRecursionsResult} from './cpu-trajectory-recursions-operation';
export {GPUTrajectoryRecursionsOperation} from './gpu-trajectory-recursions-operation';
export type {TrajectoryRecursionsOperationProps} from './gpu-trajectory-recursions-operation';
