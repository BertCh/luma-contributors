// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  GPUBrownianBridgeUtilization,
  getGPUBrownianBridgeUtilizationParameterValues,
  GPU_BROWNIAN_BRIDGE_CUTOFF_SIGMA_COUNT,
  GPU_BROWNIAN_BRIDGE_DEFAULT_SUB_STEP_COUNT,
  GPU_BROWNIAN_BRIDGE_MAXIMUM_FIX_COUNT,
  GPU_BROWNIAN_BRIDGE_MAXIMUM_SUB_STEP_COUNT,
  GPU_BROWNIAN_BRIDGE_UTILIZATION_PARAMETER_LENGTH
} from './gpu-brownian-bridge-utilization';
export type {
  GPUBrownianBridgeUtilizationProps,
  GPUBrownianBridgeUtilizationSettings
} from './gpu-brownian-bridge-utilization';
export {
  GPUUtilizationVolumeIsopleths,
  getGPUUtilizationVolumeIsoplethsParameterValues,
  GPU_UTILIZATION_VOLUME_ISOPLETHS_MAXIMUM_CELLS,
  GPU_UTILIZATION_VOLUME_ISOPLETHS_PARAMETER_LENGTH
} from './gpu-utilization-volume-isopleths';
export type {
  GPUUtilizationVolumeIsoplethsProps,
  GPUUtilizationVolumeIsoplethsSettings
} from './gpu-utilization-volume-isopleths';
export {
  estimateBrownianMotionVariance,
  estimateBrownianMotionVarianceWithStatus,
  estimateWindowedBrownianMotionVariance,
  estimateWindowedBrownianMotionVarianceWithStatus
} from './brownian-bridge-variance-estimation';
export type {
  BrownianBridgeTrack,
  BrownianMotionVarianceEstimate,
  BrownianMotionVarianceOptions,
  WindowedBrownianMotionVarianceEstimate,
  WindowedBrownianMotionVarianceOptions
} from './brownian-bridge-variance-estimation';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {
  computeBrownianBridgeUtilizationOnCPU,
  CPUBrownianBridgeUtilizationOperation
} from './cpu-brownian-bridge-utilization-operation';
export type {CPUBrownianBridgeUtilizationInput} from './cpu-brownian-bridge-utilization-operation';
export {GPUBrownianBridgeUtilizationOperation} from './gpu-brownian-bridge-utilization-operation';
export type {BrownianBridgeUtilizationOperationProps} from './gpu-brownian-bridge-utilization-operation';
