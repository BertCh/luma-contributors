// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {GPUInhomogeneousPointPattern} from './gpu-inhomogeneous-point-pattern';
export type {GPUInhomogeneousPointPatternProps} from './gpu-inhomogeneous-point-pattern';
export {
  GPUPointPatternEnvelope,
  MAXIMUM_CURVE_ROW_COUNT,
  MAXIMUM_SIMULATION_COUNT,
  MAXIMUM_RANK_COMPARISON_COUNT
} from './gpu-point-pattern-envelope';
export type {
  GPUPointPatternEnvelopeProps,
  GPUPointPatternEnvelopeOutput
} from './gpu-point-pattern-envelope';
export {GPUSpatialBalancedSample} from './gpu-spatial-balanced-sample';
export type {
  GPUSpatialBalancedSampleProps,
  GPUSpatialBalancedSampleOutput,
  GPUSpatialSampleAllocation
} from './gpu-spatial-balanced-sample';

export {GPUContinuousSpatialSample} from './gpu-continuous-spatial-sample';
export type {
  GPUContinuousSpatialSampleProps,
  GPUContinuousSpatialSampleOutput,
  GPUContinuousSpatialFrame
} from './gpu-continuous-spatial-sample';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {
  computeContinuousSpatialSampleOnCPU,
  CPUContinuousSpatialSampleOperation
} from './cpu-continuous-spatial-sample-operation';
export type {
  CPUContinuousSpatialSampleInput,
  CPUContinuousSpatialSampleOptions,
  CPUContinuousSpatialSampleResult
} from './cpu-continuous-spatial-sample-operation';
export {
  computeSpatialBalancedSampleOnCPU,
  CPUSpatialBalancedSampleOperation
} from './cpu-spatial-balanced-sample-operation';
export type {
  CPUSpatialBalancedSampleInput,
  CPUSpatialBalancedSampleOptions,
  CPUSpatialBalancedSampleResult
} from './cpu-spatial-balanced-sample-operation';
export {GPUContinuousSpatialSampleOperation} from './gpu-continuous-spatial-sample-operation';
export type {ContinuousSpatialSampleOperationProps} from './gpu-continuous-spatial-sample-operation';
export {GPUSpatialBalancedSampleOperation} from './gpu-spatial-balanced-sample-operation';
export type {SpatialBalancedSampleOperationProps} from './gpu-spatial-balanced-sample-operation';
export {
  CPU_POINT_PATTERN_ENVELOPE_MAXIMUM_VALUE_COUNT,
  computePointPatternEnvelopeOnCPU,
  CPUPointPatternEnvelopeOperation
} from './cpu-point-pattern-envelope-operation';
export type {
  CPUPointPatternEnvelopeInput,
  CPUPointPatternEnvelopeOptions,
  CPUPointPatternEnvelopeResult
} from './cpu-point-pattern-envelope-operation';
export {GPUPointPatternEnvelopeOperation} from './gpu-point-pattern-envelope-operation';
export type {PointPatternEnvelopeOperationProps} from './gpu-point-pattern-envelope-operation';

export {
  decodeGPUPointPatternEnvelopeStatistics,
  decodeGPUPointPatternEnvelopeStatus,
  decodeGPUSpatialBalancedSampleStatus,
  decodeGPUContinuousSpatialSampleStatus,
  decodeGPUContinuousSpatialSampleProvenance
} from './point-process-decoders';
