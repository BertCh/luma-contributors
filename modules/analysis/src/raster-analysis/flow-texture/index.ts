// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {GPULineIntegralConvolution} from './gpu-line-integral-convolution';
export type {
  GPULineIntegralConvolutionOutput,
  GPULineIntegralConvolutionProps
} from './gpu-line-integral-convolution';
export {
  getGPULineIntegralConvolutionParameterValues,
  getGPULineIntegralConvolutionUint32ParameterValues,
  GPU_LINE_INTEGRAL_CONVOLUTION_PARAMETER_LENGTH,
  GPU_LINE_INTEGRAL_CONVOLUTION_UINT32_PARAMETER_LENGTH
} from './line-integral-convolution-parameters';
export type {
  GPULineIntegralConvolutionSettings,
  GPULineIntegralConvolutionParameterSettings
} from './line-integral-convolution-parameters';
export {GPUStreamlines} from './gpu-streamlines';
export type {
  GPUStreamlinesOutput,
  GPUStreamlinesProps,
  GPUStreamlinesStatus
} from './gpu-streamlines';
export {
  getGPUStreamlinesParameterValues,
  getGPUStreamlinesUint32ParameterValues,
  GPU_STREAMLINES_CANDIDATE_UNCONVERGED,
  GPU_STREAMLINES_PARAMETER_LENGTH,
  GPU_STREAMLINES_UINT32_PARAMETER_LENGTH
} from './streamlines-parameters';
export type {
  GPUStreamlinesSettings,
  GPUStreamlinesParameterSettings
} from './streamlines-parameters';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {
  CPU_STREAMLINES_MAXIMUM_SEED_COUNT,
  computeStreamlinesOnCPU,
  CPUStreamlinesOperation
} from './cpu-streamlines-operation';
export type {
  CPUStreamlinesInput,
  CPUStreamlinesOptions,
  CPUStreamlinesResult
} from './cpu-streamlines-operation';
export {GPUStreamlinesOperation} from './gpu-streamlines-operation';
export type {StreamlinesOperationProps} from './gpu-streamlines-operation';

export {
  getFlowFieldSpawnRegion,
  getFlowFieldViewportBounds,
  FlowFieldViewportTracker
} from './flow-field-viewport';
export type {FlowFieldBounds} from './flow-field-viewport';
