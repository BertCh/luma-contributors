// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Public helpers shared by the sibling packages (lenses, kepler).
export * from './graph-support/index';
export {createWGSLKernelNode} from './kernels/wgsl-kernel-nodes';
export type {WGSLKernelBinding, WGSLKernelProps} from './kernels/wgsl-kernel-nodes';
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
export type {GPUParameterBufferProps, GPUParameterFormat} from './kernels/gpu-contributor-utils';
export {
  GPUParameterBuffer,
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
