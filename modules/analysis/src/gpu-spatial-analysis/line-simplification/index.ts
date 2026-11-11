// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  GPULineSimplification,
  GPU_LINE_SIMPLIFICATION_DEFAULT_MAXIMUM_ITERATIONS,
  GPU_LINE_SIMPLIFICATION_MAXIMUM_ITERATIONS,
  GPU_LINE_SIMPLIFICATION_DEFAULT_VISVALINGAM_MAXIMUM_ITERATIONS,
  GPU_LINE_SIMPLIFICATION_DEFAULT_NEIGHBORHOOD_RADIUS,
  GPU_LINE_SIMPLIFICATION_MAXIMUM_NEIGHBORHOOD_RADIUS
} from './gpu-line-simplification';
export type {
  GPULineSimplificationMethod,
  GPULineSimplificationOutput,
  GPULineSimplificationProps,
  GPULineSimplificationSelectionOutput,
  GPULineSimplificationStatus
} from './gpu-line-simplification';
export type {GPULineSimplificationMetric} from './line-simplification-kernels';
export {
  getGPULineSimplificationParameterValues,
  GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH
} from './line-simplification-parameters';
export type {GPULineSimplificationSettings} from './line-simplification-parameters';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {
  computeLineSimplificationOnCPU,
  CPULineSimplificationOperation
} from './cpu-line-simplification-operation';
export type {
  CPULineSimplificationInput,
  CPULineSimplificationOptions,
  CPULineSimplificationResult
} from './cpu-line-simplification-operation';
export {GPULineSimplificationOperation} from './gpu-line-simplification-operation';
export type {LineSimplificationOperationProps} from './gpu-line-simplification-operation';
