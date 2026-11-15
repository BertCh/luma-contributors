// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  GPU_TERRAIN_FLOW_CELL_CLASS,
  GPU_TERRAIN_FLOW_NONE,
  GPU_TERRAIN_FLOW_PARAMETER_LENGTH,
  GPU_TERRAIN_FLOW_STREAM_UNRESOLVED,
  GPU_TERRAIN_FLOW_UNRESOLVED,
  GPUTerrainFlow,
  getGPUTerrainFlowParameterValues
} from './gpu-terrain-flow';
export type {
  GPUTerrainFlowAccumulationUnits,
  GPUTerrainFlowFlatContinuation,
  GPUTerrainFlowOutput,
  GPUTerrainFlowProps,
  GPUTerrainFlowRouting,
  GPUTerrainFlowSettings,
  GPUTerrainFlowStatus
} from './gpu-terrain-flow';
export {
  GPU_TERRAIN_DRAINAGE_MULTIPLE,
  GPU_TERRAIN_DRAINAGE_NONE,
  GPU_TERRAIN_DRAINAGE_UNRESOLVED,
  GPUTerrainHeightAboveDrainage
} from './gpu-terrain-height-above-drainage';
export type {
  GPUTerrainHeightAboveDrainageOutput,
  GPUTerrainHeightAboveDrainageProps,
  GPUTerrainHeightAboveDrainageStatus
} from './gpu-terrain-height-above-drainage';
export {
  GPU_TERRAIN_WATERSHED_NONE,
  GPU_TERRAIN_WATERSHED_UNRESOLVED,
  GPUTerrainWatersheds
} from './gpu-terrain-watersheds';
export type {
  GPUTerrainWatershedsOutput,
  GPUTerrainWatershedsProps,
  GPUTerrainWatershedsStatus
} from './gpu-terrain-watersheds';
export {GPUTerrainStreamOrder} from './gpu-terrain-stream-order';
export type {
  GPUTerrainStreamOrderOutput,
  GPUTerrainStreamOrderProps,
  GPUTerrainStreamOrderStatus
} from './gpu-terrain-stream-order';
export {
  GPU_TERRAIN_HYDROLOGIC_INDICES_DEFAULT_MINIMUM_SLOPE,
  GPU_TERRAIN_HYDROLOGIC_INDICES_PARAMETER_LENGTH,
  GPUTerrainHydrologicIndices,
  getGPUTerrainHydrologicIndicesParameterValues
} from './gpu-terrain-hydrologic-indices';
export type {
  GPUTerrainHydrologicIndicesOutput,
  GPUTerrainHydrologicIndicesProps,
  GPUTerrainHydrologicIndicesSettings
} from './gpu-terrain-hydrologic-indices';
export {GPUTerrainFlowDistance} from './gpu-terrain-flow-distance';
export type {GPUTerrainFlowDistanceProps} from './gpu-terrain-flow-distance';
export type {GPUTerrainFlowDirectionEncoding} from './terrain-flow-tracing';
export {GPUTerrainStreamNetwork} from './gpu-terrain-stream-network';
export type {GPUTerrainStreamNetworkProps} from './gpu-terrain-stream-network';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {
  computeTerrainFlowDistanceOnCPU,
  CPUTerrainFlowDistanceOperation
} from './cpu-terrain-flow-distance-operation';
export type {
  CPUTerrainFlowDistanceOptions,
  CPUTerrainFlowDistanceResult
} from './cpu-terrain-flow-distance-operation';
export {
  computeTerrainStreamOrderOnCPU,
  CPUTerrainStreamOrderOperation
} from './cpu-terrain-stream-order-operation';
export type {CPUTerrainStreamOrderResult} from './cpu-terrain-stream-order-operation';
export {
  computeTerrainWatershedsOnCPU,
  CPUTerrainWatershedsOperation
} from './cpu-terrain-watersheds-operation';
export type {CPUTerrainWatershedsResult} from './cpu-terrain-watersheds-operation';
export {GPUTerrainFlowDistanceOperation} from './gpu-terrain-flow-distance-operation';
export type {TerrainFlowDistanceOperationProps} from './gpu-terrain-flow-distance-operation';
export {GPUTerrainStreamOrderOperation} from './gpu-terrain-stream-order-operation';
export type {TerrainStreamOrderOperationProps} from './gpu-terrain-stream-order-operation';
export {GPUTerrainWatershedsOperation} from './gpu-terrain-watersheds-operation';
export type {TerrainWatershedsOperationProps} from './gpu-terrain-watersheds-operation';
export {computeTerrainFlowOnCPU, CPUTerrainFlowOperation} from './cpu-terrain-flow-operation';
export type {
  CPUTerrainFlowInput,
  CPUTerrainFlowOptions,
  CPUTerrainFlowResult
} from './cpu-terrain-flow-operation';
export {GPUTerrainFlowOperation} from './gpu-terrain-flow-operation';
export type {TerrainFlowOperationProps} from './gpu-terrain-flow-operation';
