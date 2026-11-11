// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  GPULocationChangePoints,
  GPU_LOCATION_CHANGE_POINTS_MAXIMUM_CHANGE_POINTS,
  GPU_LOCATION_CHANGE_POINTS_MAXIMUM_SLICE_COUNT,
  GPU_LOCATION_CHANGE_POINTS_MAXIMUM_SLOPE_SLICE_COUNT,
  GPU_LOCATION_CHANGE_POINTS_NONE,
  GPU_LOCATION_CHANGE_POINTS_PARAMETER_LENGTH
} from './gpu-location-change-points';
export type {
  GPULocationChangePointCost,
  GPULocationChangePointSettings,
  GPULocationChangePointsOutput,
  GPULocationChangePointsProps
} from './gpu-location-change-points';
export {
  GPUPettittTest,
  GPU_PETTITT_TEST_MAXIMUM_SLICE_COUNT,
  GPU_PETTITT_TEST_NO_CHANGE_SLICE
} from './gpu-pettitt-test';
export type {GPUPettittTestOutput, GPUPettittTestProps} from './gpu-pettitt-test';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {
  CPU_PETTITT_TEST_MAXIMUM_SLICE_COUNT,
  computePettittTestOnCPU,
  CPUPettittTestOperation
} from './cpu-pettitt-test-operation';
export type {CPUPettittTestOptions, CPUPettittTestResult} from './cpu-pettitt-test-operation';
export {GPUPettittTestOperation} from './gpu-pettitt-test-operation';
export type {PettittTestOperationProps} from './gpu-pettitt-test-operation';
export {
  computeLocationChangePointsOnCPU,
  CPULocationChangePointsOperation
} from './cpu-location-change-points-operation';
export type {
  CPULocationChangePointsOptions,
  CPULocationChangePointsResult
} from './cpu-location-change-points-operation';
export {GPULocationChangePointsOperation} from './gpu-location-change-points-operation';
export type {LocationChangePointsOperationProps} from './gpu-location-change-points-operation';
