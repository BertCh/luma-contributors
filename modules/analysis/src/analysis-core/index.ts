// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Public helpers shared by the sibling packages (lenses, kepler).
export * from './graph-support/index';
export {createWGSLKernelNode} from './kernels/wgsl-kernel-nodes';
export type {
  WGSLKernelBinding,
  WGSLKernelProps,
  PublishNodeProps,
  WGSLDispatchGate
} from './kernels/wgsl-kernel-nodes';
// Kernel facts for computation records (opt-in per graph; see computation-record).
export {
  getWGSLKernelNodeRecord,
  isRecordingWGSLKernelNodes,
  recordWGSLKernelNodes
} from './kernels/wgsl-kernel-nodes';
export type {
  WGSLKernelNodeRecord,
  WGSLKernelTextureBinding,
  WGSLKernelTextureSampleType
} from './kernels/wgsl-kernel-nodes';
export {getGraphViewChunks} from './kernels/gpu-contributor-utils';

// The contributor contract (dev-docs/contributor-contract.md): contracts, construction-time
// validators and kernel helpers a contributor needs. This is the only entry point that exports
// them; domain entry points export contributors.
export * from './contracts/index';
export type {
  GPUParameterBufferProps,
  GPUParameterFormat,
  GPUParameterValues,
  GPUPackedReadbackValues,
  GPUPackedReadbackFormat,
  GPUGraphPortDescriptor,
  AnalysisValidationCode,
  AnalysisValidationIdentity,
  AnalysisValidationDetails
} from './kernels/gpu-contributor-utils';
export {
  AnalysisValidationError,
  GPUParameterBuffer,
  getPackedUint32Values,
  getPackedFloat32Values,
  readPackedBuffer,
  isGraphDataView,
  isGraphVectorView,
  validateGraphPorts,
  validateOption,
  validateNonZeroFiniteNumber,
  GPU_MAXIMUM_COMMAND_NODE_COUNT,
  getStorageBindingElementLimit,
  validateCapacity,
  validateCommandNodeBudget,
  validateCompactOutput,
  validateExtent,
  validateFiniteNumber,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validatePowerOfTwo,
  validateRequirement,
  validateSinglePackedView,
  validateViewLength
} from './kernels/gpu-contributor-utils';
export {
  createFillNode,
  createPublishNode,
  createWGSLDispatchGate,
  getWGSLFloatLiteral,
  getWGSLKernelBinding
} from './kernels/wgsl-kernel-nodes';

// Contributors as GPUGraph operations and helpers for CPU twins (contract section 14).
export * from './operations/index';

// Shared, bounded model-fitting kernels (LRN-02).
export * from './kernels/gpu-fitting-gradient-sums';
export * from './kernels/gpu-adam-step';
export * from './kernels/adam-step-parameters';
export * from './kernels/gpu-fitting-minibatch-order';
export * from './kernels/gpu-fitting-coordinate-encoder';
export {
  GPU_FITTING_MAXIMUM_BINDING_BYTES,
  GPU_FITTING_MAXIMUM_PARAMETER_COUNT,
  GPU_FITTING_MAXIMUM_ROW_COUNT,
  GPU_FITTING_MAXIMUM_WORK
} from './kernels/fitting-utils';

export {resolveGPURasterContext} from './contracts/raster-grid';
export type {
  RasterGridAffine,
  GPURasterContextOptions,
  GPUResolvedRasterContext,
  GPURasterCellSizeMode
} from './contracts/raster-grid';
export * from './graph-support/device-limits';
export {createGraphChunkGather} from './kernels/graph-chunk-gather';
export * from './kernels/graph-chunk-dispatch';
export * from './kernels/graph-chunk-groups';

export type {
  GPUAnalysisPackage,
  GPUAnalysisOperatorIdentity,
  GPUAnalysisRoute,
  GPUAnalysisExecutionPlanner
} from './contracts/analysis-execution-planning';
