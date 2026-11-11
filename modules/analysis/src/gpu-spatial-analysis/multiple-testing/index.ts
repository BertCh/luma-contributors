// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  getMultipleTestingAdjustedPValues,
  GPUMultipleTestingAdjustment,
  GPU_MULTIPLE_TESTING_MAXIMUM_ROWS,
  resolveGPUMultipleTestingMethod
} from './gpu-multiple-testing-adjustment';
export type {
  GPUMultipleTestingAdjustmentOutput,
  GPUMultipleTestingAdjustmentProps
} from './gpu-multiple-testing-adjustment';
export {getPermutationRejectionFloor} from './permutation-p-value-floor';
export type {
  PermutationRejectionFloor,
  PermutationRejectionFloorProps
} from './permutation-p-value-floor';
export {
  getGPURasterFalseDiscoveryRateMaskParameterValues,
  GPURasterFalseDiscoveryRateMask,
  GPU_RASTER_FALSE_DISCOVERY_RATE_MASK_MAXIMUM_HISTOGRAM_BINS,
  GPU_RASTER_FALSE_DISCOVERY_RATE_MASK_PARAMETER_LENGTH
} from './gpu-raster-false-discovery-rate-mask';
export type {
  GPURasterFalseDiscoveryRateMaskOutput,
  GPURasterFalseDiscoveryRateMaskProps,
  GPURasterFalseDiscoveryRateMaskSettings,
  GPURasterFalseDiscoveryRateMethod
} from './gpu-raster-false-discovery-rate-mask';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {
  computeMultipleTestingAdjustmentOnCPU,
  CPUMultipleTestingAdjustmentOperation
} from './cpu-multiple-testing-adjustment-operation';
export {GPUMultipleTestingAdjustmentOperation} from './gpu-multiple-testing-adjustment-operation';
export type {MultipleTestingAdjustmentOperationProps} from './gpu-multiple-testing-adjustment-operation';
