// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  GPUSpatialWeightsMinimumSpanningTree,
  GPU_MINIMUM_SPANNING_TREE_MAXIMUM_COLUMNS,
  GPU_MINIMUM_SPANNING_TREE_STATUS
} from './gpu-spatial-weights-minimum-spanning-tree';
export type {
  GPUSpatialWeightsMinimumSpanningTreeOutput,
  GPUSpatialWeightsMinimumSpanningTreeProps
} from './gpu-spatial-weights-minimum-spanning-tree';
export {
  getGPUSkaterRegionsParameterValues,
  GPUSkaterRegions,
  GPU_SKATER_MAXIMUM_COLUMNS,
  getGPUSkaterMaximumStatisticCount,
  GPU_SKATER_NO_CUT,
  GPU_SKATER_REGIONS_PARAMETER_LENGTH
} from './gpu-skater-regions';
export type {
  GPUSkaterRegionsOutput,
  GPUSkaterRegionsProps,
  GPUSkaterRegionsSettings
} from './gpu-skater-regions';
export {
  GPURegionPartitionEvaluation,
  GPU_REGION_PARTITION_EVALUATION_LAYOUT,
  GPU_REGION_PARTITION_EVALUATION_MAXIMUM_COLUMNS,
  getGPURegionPartitionEvaluationMaximumLabelCapacity,
  getGPURegionPartitionEvaluationMaximumRegionValueCount,
  getGPURegionPartitionEvaluationMaximumRowCount
} from './gpu-region-partition-evaluation';
export type {
  GPURegionPartitionEvaluationOutput,
  GPURegionPartitionEvaluationProps
} from './gpu-region-partition-evaluation';
export {
  getGPUAZPRegionsParameterValues,
  GPUAZPRegions,
  GPU_AZP_MAXIMUM_COLUMNS,
  GPU_AZP_LOCAL_NEIGHBORHOOD_CAPACITY,
  GPU_AZP_MAXIMUM_ITERATIONS,
  GPU_AZP_MAXIMUM_SCAN_DEGREE,
  getGPUAZPMaximumTableWordCount,
  GPU_AZP_REGIONS_PARAMETER_LENGTH,
  GPU_AZP_STATUS
} from './gpu-azp-regions';
export type {
  GPUAZPRegionsOutput,
  GPUAZPRegionsProps,
  GPUAZPRegionsSettings
} from './gpu-azp-regions';
export {
  getGPUWardRegionsParameterValues,
  GPUWardRegions,
  GPU_WARD_MAXIMUM_COLUMNS,
  getGPUWardMaximumRegionTableWordCount,
  GPU_WARD_REGIONS_PARAMETER_LENGTH,
  GPU_WARD_ROUND_COUNT_STATUS_WORD,
  GPU_WARD_STATUS
} from './gpu-ward-regions';
export type {
  GPUWardMergeSchedule,
  GPUWardRegionsOutput,
  GPUWardRegionsProps,
  GPUWardRegionsSettings
} from './gpu-ward-regions';
export {
  getGPUMaxPRegionsParameterValues,
  GPUMaxPRegions,
  GPU_MAX_P_DEFAULT_START_COUNT,
  GPU_MAX_P_MAXIMUM_COLUMNS,
  GPU_MAX_P_DEFAULT_MAXIMUM_ROUNDS,
  GPU_MAX_P_MAXIMUM_ROUNDS,
  getGPUMaxPMaximumSlotCount,
  GPU_MAX_P_MAXIMUM_START_COUNT,
  GPU_MAX_P_REGIONS_PARAMETER_LENGTH,
  GPU_MAX_P_STATUS
} from './gpu-max-p-regions';
export type {
  GPUMaxPRegionsOutput,
  GPUMaxPRegionsProps,
  GPUMaxPRegionsSettings
} from './gpu-max-p-regions';
export {getGPURegionalizationMaximumSlotCount} from './regionalization-inputs';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {computeWardRegionsOnCPU, CPUWardRegionsOperation} from './cpu-ward-regions-operation';
export type {CPUWardRegionsOptions, CPUWardRegionsResult} from './cpu-ward-regions-operation';
export {GPUWardRegionsOperation} from './gpu-ward-regions-operation';
export type {WardRegionsOperationProps} from './gpu-ward-regions-operation';
export {computeSkaterRegionsOnCPU, CPUSkaterRegionsOperation} from './cpu-skater-regions-operation';
export type {
  CPUSkaterRegionsInput,
  CPUSkaterRegionsOptions,
  CPUSkaterRegionsResult
} from './cpu-skater-regions-operation';
export {GPUSkaterRegionsOperation} from './gpu-skater-regions-operation';
export type {SkaterRegionsOperationProps} from './gpu-skater-regions-operation';
export {computeAZPRegionsOnCPU, CPUAZPRegionsOperation} from './cpu-azp-regions-operation';
export type {
  CPUAZPRegionsInput,
  CPUAZPRegionsOptions,
  CPUAZPRegionsResult
} from './cpu-azp-regions-operation';
export {computeMaxPRegionsOnCPU, CPUMaxPRegionsOperation} from './cpu-max-p-regions-operation';
export type {
  CPUMaxPRegionsInput,
  CPUMaxPRegionsOptions,
  CPUMaxPRegionsResult
} from './cpu-max-p-regions-operation';
export {GPUAZPRegionsOperation} from './gpu-azp-regions-operation';
export type {AZPRegionsOperationProps} from './gpu-azp-regions-operation';
export {GPUMaxPRegionsOperation} from './gpu-max-p-regions-operation';
export type {MaxPRegionsOperationProps} from './gpu-max-p-regions-operation';
