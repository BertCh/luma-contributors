// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  GPULocationAllocation,
  GPU_LOCATION_ALLOCATION_DEFAULT_MAXIMUM_SWAP_ITERATIONS,
  getGPULocationAllocationMaximumCostCount,
  getGPULocationAllocationMaximumDemandCount,
  getGPULocationAllocationMaximumFacilityCount,
  getGPULocationAllocationMaximumSparseEntryCount,
  GPU_LOCATION_ALLOCATION_MAXIMUM_SWAP_ITERATIONS,
  GPU_LOCATION_ALLOCATION_STATUS,
  GPU_LOCATION_ALLOCATION_SWAP_TOLERANCE
} from './gpu-location-allocation';
export type {
  GPULocationAllocationOperation,
  GPULocationAllocationOutput,
  GPULocationAllocationProps,
  GPULocationAllocationSparseCosts
} from './gpu-location-allocation';

export {GPUBoundedFacilityAllocation} from './gpu-bounded-facility-allocation';
export type {GPUBoundedFacilityAllocationProps} from './gpu-bounded-facility-allocation';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {
  computeLocationAllocationOnCPU,
  CPULocationAllocationOperation
} from './cpu-location-allocation-operation';
export type {
  CPULocationAllocationInput,
  CPULocationAllocationResult,
  CPULocationAllocationOptions
} from './cpu-location-allocation-operation';
export {GPULocationAllocationGraphOperation} from './gpu-location-allocation-operation';
export type {LocationAllocationOperationProps} from './gpu-location-allocation-operation';
