// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  GPUMapMatching,
  getGPUMapMatchingMaximumEdgeCount,
  getGPUMapMatchingMaximumEntryCapacity,
  getGPUMapMatchingMaximumPointCount,
  GPU_MAP_MATCHING_NONE,
  GPU_MAP_MATCHING_PARAMETER_LENGTH,
  getGPUMapMatchingParameterValues
} from './gpu-map-matching';
export type {
  GPUMapMatchingOutput,
  GPUMapMatchingProps,
  GPUMapMatchingRouteTable,
  GPUMapMatchingSettings,
  GPUMapMatchingTraversedEdges
} from './gpu-map-matching';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {computeMapMatchingOnCPU, CPUMapMatchingOperation} from './cpu-map-matching-operation';
export type {
  CPUMapMatchingInput,
  CPUMapMatchingOptions,
  CPUMapMatchingResult
} from './cpu-map-matching-operation';
export {GPUMapMatchingOperation} from './gpu-map-matching-operation';
export type {MapMatchingOperationProps} from './gpu-map-matching-operation';
