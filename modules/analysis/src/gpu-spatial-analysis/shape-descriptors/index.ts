// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  getGPUShapeDescriptorsParameterValues,
  GPU_SHAPE_DESCRIPTORS_DEFAULT_SLIVER_THRESHOLD,
  GPU_SHAPE_DESCRIPTORS_MONOTONE_CHAIN_AVERAGE_VERTICES,
  GPU_SHAPE_DESCRIPTORS_MONOTONE_CHAIN_LARGEST_VERTICES,
  GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH,
  GPUShapeDescriptors
} from './gpu-shape-descriptors';
export type {
  GPUShapeDescriptorsConvexityMethod,
  GPUShapeDescriptorsOutput,
  GPUShapeDescriptorsSettings,
  GPUShapeDescriptorsProps
} from './gpu-shape-descriptors';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {
  computeShapeDescriptorsOnCPU,
  CPUShapeDescriptorsOperation
} from './cpu-shape-descriptors-operation';
export type {CPUShapeDescriptorsResult} from './cpu-shape-descriptors-operation';
export {GPUShapeDescriptorsOperation} from './gpu-shape-descriptors-operation';
export type {ShapeDescriptorsOperationProps} from './gpu-shape-descriptors-operation';
