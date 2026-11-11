// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  GPUIslandAttachment,
  GPU_ISLAND_ATTACHMENT_SCAN_MAXIMUM_ROW_COUNT
} from './gpu-island-attachment';
export type {
  GPUIslandAttachmentNearestSearch,
  GPUIslandAttachmentProps
} from './gpu-island-attachment';

export {
  GPU_DEFAULT_ISLAND_POLICY,
  GPU_SPATIAL_WEIGHTS_CAPACITY_HEADROOM,
  GPU_SPATIAL_WEIGHTS_DISTANCE_BYTE_LENGTH,
  GPU_SPATIAL_WEIGHTS_SLOT_BYTE_LENGTH,
  describeGPUSpatialWeights,
  getGPUSpatialWeightsCapacityPlan,
  isGPUSpatialWeightsRowStandardized,
  resolveGPUIslandPolicy,
  validateGPUSpatialWeights,
  validateGPUSpatialWeightsOutput
} from './spatial-weights';
export type {
  GPUIslandPolicy,
  GPUSpatialWeights,
  GPUSpatialWeightsCapacityPlan,
  GPUSpatialWeightsCapacityPlanOptions,
  GPUSpatialWeightsInformation,
  GPUSpatialWeightsOutput,
  GPUSpatialWeightsProvenance,
  GPUSpatialWeightsTransformState
} from './spatial-weights';
export type {GPUFloat32OrUint32Values} from './float32-values';
export {
  GPUContiguityWeights,
  GPU_CONTIGUITY_WEIGHTS_MAXIMUM_SNAP_CELLS,
  GPU_CONTIGUITY_WEIGHTS_PAIR_CAPACITY_FACTOR
} from './gpu-contiguity-weights';
export type {
  GPUContiguityCriterion,
  GPUContiguityWeightsCapacityPlan,
  GPUContiguityWeightsObservation,
  GPUContiguityWeightsProps
} from './gpu-contiguity-weights';
export {GPULatticeWeights, GPU_LATTICE_WEIGHTS_MAXIMUM_RADIUS} from './gpu-lattice-weights';
export type {
  GPULatticeCriterion,
  GPULatticeWeightsCapacityPlan,
  GPULatticeWeightsProps
} from './gpu-lattice-weights';
export {
  GPUSpatialWeightsTransform,
  GPU_SPATIAL_WEIGHTS_ADAPTIVE_BANDWIDTH_EPSILON,
  getGPUSpatialWeightsKernelValue
} from './gpu-spatial-weights-transform';
export type {
  GPUSpatialWeightsKernel,
  GPUSpatialWeightsTransformOperation,
  GPUSpatialWeightsTransformProps
} from './gpu-spatial-weights-transform';
export {GPUSpatialLag} from './gpu-spatial-lag';
export type {GPUSpatialLagProps} from './gpu-spatial-lag';
export {
  GPUSpatialWeightsAlgebra,
  GPU_SPATIAL_WEIGHTS_MAXIMUM_ORDER
} from './gpu-spatial-weights-algebra';
export type {
  GPUSpatialWeightsAlgebraBaseProps,
  GPUSpatialWeightsAlgebraProps,
  GPUSpatialWeightsBinaryProps,
  GPUSpatialWeightsBlockProps,
  GPUSpatialWeightsCombineRule,
  GPUSpatialWeightsHigherOrderProps,
  GPUSpatialWeightsSelfWeightProps,
  GPUSpatialWeightsSpaceTimeProps,
  GPUSpatialWeightsSubgraphProps
} from './gpu-spatial-weights-algebra';
export type {
  GPUSpaceTimeWeightsLayout,
  GPUSpatialWeightsSpaceTimeLag
} from './spatial-weights-space-time-stages';
export {
  GPUSpatialWeightsSummary,
  GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT
} from './gpu-spatial-weights-summary';
export type {
  GPUSpatialWeightsSummaryOutput,
  GPUSpatialWeightsSummaryProps
} from './gpu-spatial-weights-summary';
export {GPUSpatialWeightsTranspose} from './gpu-spatial-weights-transpose';
export type {
  GPUSpatialWeightsTransposeOutput,
  GPUSpatialWeightsTransposeProps
} from './gpu-spatial-weights-transpose';
export {
  GPUSpatialWeightsValidation,
  GPU_SPATIAL_WEIGHTS_VIOLATION
} from './gpu-spatial-weights-validation';
export type {
  GPUSpatialWeightsValidationOutput,
  GPUSpatialWeightsValidationProps
} from './gpu-spatial-weights-validation';
export {GPUFuzzyContiguity} from './gpu-fuzzy-contiguity';
export type {GPUFuzzyContiguityProps} from './gpu-fuzzy-contiguity';
export {
  getGPUSpatialWeightsChunkDispatches,
  getGPUSpatialWeightsChunks,
  getGPUSpatialWeightsOrChunksViews,
  isGPUChunkedSpatialWeights,
  planGPUSpatialWeightsChunks,
  validateGPUChunkedSpatialWeights,
  validateGPUSpatialWeightsOrChunks,
  wrapGPUSpatialWeightsChunkBody
} from './gpu-chunked-spatial-weights';
export type {
  GPUChunkedSpatialWeights,
  GPUSpatialWeightsChunk,
  GPUSpatialWeightsChunkDispatch
} from './gpu-chunked-spatial-weights';
export {GPUChunkedSpatialWeightsValidation} from './gpu-chunked-spatial-weights-validation';
export type {GPUChunkedSpatialWeightsValidationProps} from './gpu-chunked-spatial-weights-validation';
export {GPUChunkedSpatialWeightsTranspose} from './gpu-chunked-spatial-weights-transpose';
export type {GPUChunkedSpatialWeightsTransposeProps} from './gpu-chunked-spatial-weights-transpose';
export {GPUChunkedSpatialWeightsTransform} from './gpu-chunked-spatial-weights-transform';
export type {GPUChunkedSpatialWeightsTransformProps} from './gpu-chunked-spatial-weights-transform';
export {GPUChunkedSpatialWeightsAlgebra} from './gpu-chunked-spatial-weights-algebra';
export type {
  GPUChunkedSpatialWeightsAlgebraProps,
  GPUChunkedSpatialWeightsOutput
} from './gpu-chunked-spatial-weights-algebra';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {
  computeContiguityWeightsOnCPU,
  CPUContiguityWeightsOperation
} from './cpu-contiguity-weights-operation';
export type {
  CPUContiguityWeightsInput,
  CPUContiguityWeightsResult
} from './cpu-contiguity-weights-operation';
export {
  computeSpatialWeightsValidationOnCPU,
  CPUSpatialWeightsValidationOperation
} from './cpu-spatial-weights-validation-operation';
export {GPUContiguityWeightsOperation} from './gpu-contiguity-weights-operation';
export type {ContiguityWeightsOperationProps} from './gpu-contiguity-weights-operation';
export {GPUSpatialWeightsValidationOperation} from './gpu-spatial-weights-validation-operation';
export type {SpatialWeightsValidationOperationProps} from './gpu-spatial-weights-validation-operation';
export {
  computeSpatialWeightsAlgebraOnCPU,
  CPUSpatialWeightsAlgebraOperation
} from './cpu-spatial-weights-algebra-operation';
export type {
  CPUSpatialWeightsAlgebraInput,
  CPUSpatialWeightsAlgebraOptions,
  CPUSpatialWeightsAlgebraResult
} from './cpu-spatial-weights-algebra-operation';
export {GPUSpatialWeightsAlgebraOperation} from './gpu-spatial-weights-algebra-operation';
export type {SpatialWeightsAlgebraOperationProps} from './gpu-spatial-weights-algebra-operation';
