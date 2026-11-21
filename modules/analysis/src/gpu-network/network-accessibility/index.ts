// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  GPUNetworkSnapping,
  getGPUNetworkSnappingMaximumCandidateEdgeCount,
  GPU_NETWORK_SNAPPING_MAXIMUM_UNINDEXED_WORK,
  GPU_NETWORK_SNAPPING_NONE,
  GPU_NETWORK_SNAPPING_NO_VALUE
} from './gpu-network-snapping';
export type {
  GPUNetworkSnappingOutput,
  GPUNetworkSnappingProps,
  GPUNetworkSnappingStartDirection
} from './gpu-network-snapping';
export {GPUNetworkCostMatrix} from './gpu-network-cost-matrix';
export type {
  GPUNetworkCostMatrixOutput,
  GPUNetworkCostMatrixResult,
  GPUNetworkCostMatrixProps
} from './gpu-network-cost-matrix';
export {recommendLaneCount} from './network-accessibility-lanes';
export type {RecommendLaneCountOptions} from './network-accessibility-lanes';
export {
  GPUNetworkAccessibility,
  addNetworkCatchmentAdapter,
  GPU_NETWORK_ACCESSIBILITY_PARAMETER_LENGTH,
  getGPUNetworkAccessibilityParameterValues
} from './gpu-network-accessibility';
export type {
  GPUNetworkAccessibilityCatchment,
  GPUNetworkCatchmentAdapterProps,
  GPUNetworkAccessibilityDecay,
  GPUNetworkAccessibilitySettings,
  GPUNetworkAccessibilityProps
} from './gpu-network-accessibility';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {
  computeNetworkCostMatrixOnCPU,
  CPUNetworkCostMatrixOperation
} from './cpu-network-cost-matrix-operation';
export type {
  CPUNetworkCostMatrixInput,
  CPUNetworkCostMatrixOptions,
  CPUNetworkCostMatrixResult
} from './cpu-network-cost-matrix-operation';
export {GPUNetworkCostMatrixOperation} from './gpu-network-cost-matrix-operation';
export type {
  NetworkCostMatrixOperationProps,
  GPUNetworkCostMatrixOperationProps,
  NetworkCostMatrixOperationStatus
} from './gpu-network-cost-matrix-operation';
