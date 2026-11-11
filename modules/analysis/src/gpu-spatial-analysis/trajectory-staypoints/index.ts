// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {GPU_MOVEMENT_SENTINEL} from './movement-common';
export {
  GPUTrajectoryStaypoints,
  GPU_TRAJECTORY_STAYPOINTS_DEFAULT_MAXIMUM_SEARCH_ROWS,
  GPU_TRAJECTORY_STAYPOINTS_MAXIMUM_SEARCH_ROWS,
  GPU_TRAJECTORY_STAYPOINTS_PARAMETER_LENGTH,
  getGPUTrajectoryStaypointsParameterValues
} from './gpu-trajectory-staypoints';
export type {
  GPUTrajectoryStaypointsOutput,
  GPUTrajectoryStaypointsProps,
  GPUTrajectoryStaypointsSettings
} from './gpu-trajectory-staypoints';
export {
  GPUTrajectoryLocations,
  GPU_TRAJECTORY_LOCATIONS_DEFAULT_MAXIMUM_ITERATIONS,
  GPU_TRAJECTORY_LOCATIONS_MAXIMUM_ITERATIONS,
  GPU_TRAJECTORY_LOCATIONS_PARAMETER_LENGTH,
  getGPUTrajectoryLocationsParameterValues
} from './gpu-trajectory-locations';
export type {
  GPUTrajectoryLocationsOutput,
  GPUTrajectoryLocationsProps,
  GPUTrajectoryLocationsSettings
} from './gpu-trajectory-locations';
export {
  GPUTrajectorySegments,
  GPU_TRAJECTORY_SEGMENTS_PARAMETER_LENGTH,
  getGPUTrajectorySegmentsParameterValues
} from './gpu-trajectory-segments';
export type {
  GPUTrajectorySegmentsOutput,
  GPUTrajectorySegmentsProps,
  GPUTrajectorySegmentsSettings,
  GPUTrajectoryTriplegOutput,
  GPUTrajectoryTripOutput
} from './gpu-trajectory-segments';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {
  computeTrajectoryStaypointsOnCPU,
  CPUTrajectoryStaypointsOperation
} from './cpu-trajectory-staypoints-operation';
export type {
  CPUTrajectoryStaypointsInput,
  CPUTrajectoryStaypointsResult
} from './cpu-trajectory-staypoints-operation';
export {GPUTrajectoryStaypointsOperation} from './gpu-trajectory-staypoints-operation';
export type {TrajectoryStaypointsOperationProps} from './gpu-trajectory-staypoints-operation';
export {
  computeTrajectoryLocationsOnCPU,
  CPUTrajectoryLocationsOperation
} from './cpu-trajectory-locations-operation';
export type {
  CPUTrajectoryLocationsInput,
  CPUTrajectoryLocationsResult
} from './cpu-trajectory-locations-operation';
export {
  computeTrajectorySegmentsOnCPU,
  CPUTrajectorySegmentsOperation
} from './cpu-trajectory-segments-operation';
export type {
  CPUTrajectorySegmentsInput,
  CPUTrajectorySegmentsOptions,
  CPUTrajectorySegmentsResult
} from './cpu-trajectory-segments-operation';
export {GPUTrajectoryLocationsOperation} from './gpu-trajectory-locations-operation';
export type {TrajectoryLocationsOperationProps} from './gpu-trajectory-locations-operation';
export {GPUTrajectorySegmentsOperation} from './gpu-trajectory-segments-operation';
export type {TrajectorySegmentsOperationProps} from './gpu-trajectory-segments-operation';
