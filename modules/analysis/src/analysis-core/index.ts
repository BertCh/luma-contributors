// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Public helpers shared by the sibling packages (lenses, kepler).
export * from './graph-support/index';
export {createWGSLKernelNode} from './kernels/wgsl-kernel-nodes';
export type {WGSLKernelBinding, WGSLKernelProps} from './kernels/wgsl-kernel-nodes';
export {getGraphViewChunks} from './kernels/gpu-contributor-utils';

// The contributor contract (dev-docs/contributor-contract.md): contracts, construction-time
// validators and kernel helpers a contributor needs. Domain entry points re-export parts of this
// for existing consumers; new code imports it from here.
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
