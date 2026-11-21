// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {GPUNetworkFlow, GPU_NETWORK_FLOW_WORD} from './gpu-network-flow';
export type {GPUNetworkFlowProps, GPUNetworkFlowOutput} from './gpu-network-flow';
export type {GPUNetworkFlowPathOutput, CPUNetworkFlowPaths} from './network-flow-paths';
export {GPUNetworkLouvain, getGPUNetworkLouvainParameterValues} from './gpu-network-louvain';
export type {
  GPUNetworkLouvainProps,
  GPUNetworkLouvainOutput,
  GPUNetworkLouvainSettings
} from './gpu-network-louvain';
export {
  GPUNetworkTimeRouting,
  prepareGPUNetworkRestrictionStates
} from './gpu-network-time-routing';
export type {
  GPUNetworkTimeRoutingProps,
  GPUNetworkTimeRoutingOutput
} from './gpu-network-time-routing';
export {GPUNetworkPointRouting} from './gpu-network-point-routing';
export type {GPUNetworkPointRoutingProps} from './gpu-network-point-routing';
export {GPUNetworkFacilityOptimization} from './gpu-network-facility-optimization';
export type {
  GPUNetworkFacilityOptimizationProps,
  GPUNetworkFacilityOptimizationOutput
} from './gpu-network-facility-optimization';
export {GPUNetworkRationalAccess} from './gpu-network-rational-access';
export type {
  GPUNetworkRationalAccessProps,
  GPUNetworkRationalAccessOutput
} from './gpu-network-rational-access';
export {prepareGPUNetworkFlowDemands} from './prepare-flow-demands';
export type {GPUNetworkEdgeEndpoints} from './prepare-flow-demands';
export {prepareGPUNetworkPointRoutingTurns} from './prepare-point-routing-turns';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {
  computeNetworkTimeRoutingOnCPU,
  CPUNetworkTimeRoutingOperation
} from './cpu-network-time-routing-operation';
export type {
  CPUNetworkTimeRoutingInput,
  CPUNetworkTimeRoutingResult
} from './cpu-network-time-routing-operation';
export {GPUNetworkTimeRoutingOperation} from './gpu-network-time-routing-operation';
export type {NetworkTimeRoutingOperationProps} from './gpu-network-time-routing-operation';
export {
  computeNetworkPointRoutingOnCPU,
  CPUNetworkPointRoutingOperation
} from './cpu-network-point-routing-operation';
export type {
  CPUNetworkPointRoutingInput,
  CPUNetworkPointRoutingResult
} from './cpu-network-point-routing-operation';
export {GPUNetworkPointRoutingOperation} from './gpu-network-point-routing-operation';
export type {NetworkPointRoutingOperationProps} from './gpu-network-point-routing-operation';
export {computeNetworkFlowOnCPU, CPUNetworkFlowOperation} from './cpu-network-flow-operation';
export type {
  CPUNetworkFlowInput,
  CPUNetworkFlowOptions,
  CPUNetworkFlowResult
} from './cpu-network-flow-operation';
export {
  computeNetworkLouvainOnCPU,
  decodeNetworkLouvainStatus,
  CPUNetworkLouvainOperation
} from './cpu-network-louvain-operation';
export type {
  CPUNetworkLouvainInput,
  CPUNetworkLouvainOptions,
  CPUNetworkLouvainResult
} from './cpu-network-louvain-operation';
export {
  computeNetworkRationalAccessOnCPU,
  CPUNetworkRationalAccessOperation
} from './cpu-network-rational-access-operation';
export type {
  CPUNetworkRationalAccessInput,
  CPUNetworkRationalAccessOptions,
  CPUNetworkRationalAccessResult
} from './cpu-network-rational-access-operation';
export {GPUNetworkFlowOperation} from './gpu-network-flow-operation';
export type {NetworkFlowOperationProps} from './gpu-network-flow-operation';
export {GPUNetworkLouvainOperation} from './gpu-network-louvain-operation';
export type {NetworkLouvainOperationProps} from './gpu-network-louvain-operation';
export {GPUNetworkRationalAccessOperation} from './gpu-network-rational-access-operation';
export type {NetworkRationalAccessOperationProps} from './gpu-network-rational-access-operation';

export {
  getNetworkFacilitySearchLimits,
  MAXIMUM_FACILITY_COUNT as GPU_NETWORK_FACILITY_OPTIMIZATION_MAXIMUM_FACILITY_COUNT,
  MAXIMUM_DEMAND_COUNT as GPU_NETWORK_FACILITY_OPTIMIZATION_MAXIMUM_DEMAND_COUNT,
  MAXIMUM_CANDIDATE_COUNT as GPU_NETWORK_FACILITY_OPTIMIZATION_MAXIMUM_CANDIDATE_COUNT
} from './gpu-network-facility-optimization';
