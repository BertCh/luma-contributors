// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  GPU_SPATIAL_JOIN_NO_DISTANCE,
  GPU_SPATIAL_JOIN_NO_FEATURE
} from './spatial-join-types';
export type {GPUNearestFeatureSegments, GPUNearestFeatureSource} from './spatial-join-types';
export {GPUPointInPolygonJoin} from './gpu-point-in-polygon-join';
export type {
  GPUPointInPolygonBoundaryRule,
  GPUPointInPolygonJoinOutput,
  GPUPointInPolygonJoinProps
} from './gpu-point-in-polygon-join';
export {
  GPUNearestFeatureJoin,
  getGPUNearestFeatureJoinMaximumLeafCapacity,
  getGPUNearestFeatureJoinMaximumNodeTableWordCount
} from './gpu-nearest-feature-join';
export type {
  GPUNearestFeatureJoinOutput,
  GPUNearestFeatureJoinProps
} from './gpu-nearest-feature-join';
export {GPUBufferSelection} from './gpu-buffer-selection';
export type {
  GPUBufferSelectionMatches,
  GPUBufferSelectionOutput,
  GPUBufferSelectionProps
} from './gpu-buffer-selection';
export {
  GPUSpatialPredicateJoin,
  GPU_SPATIAL_PREDICATE_JOIN_PLANNER,
  GPU_SPATIAL_PREDICATE_JOIN_ROUTES,
  getGPUSpatialPredicateJoinExecutionPlan
} from './gpu-spatial-predicate-join';
export type {
  GPUSpatialJoinHow,
  GPUSpatialJoinLeftStatus,
  GPUSpatialPredicate,
  GPUSpatialPredicateJoinObservation,
  GPUSpatialPredicateJoinOutput,
  GPUSpatialPredicateJoinProps,
  GPUSpatialPredicateJoinRouteShape
} from './gpu-spatial-predicate-join';
export {
  GPUSpatialJoinPrepared,
  getGPUSpatialJoinPreparedMemoryPlan,
  GPU_SPATIAL_JOIN_PREPARED_PLANNER,
  GPU_SPATIAL_JOIN_PREPARED_ROUTES
} from './gpu-spatial-join-prepared';
export type {
  GPUSpatialJoinPreparedShape,
  GPUSpatialJoinPreparedDeviceLimits,
  GPUSpatialJoinPreparedProps,
  GPUSpatialJoinPreparedStorage
} from './gpu-spatial-join-prepared';
/** Whole-domain allocation and binding checks returned before preparing a spatial index. */
export type GPUSpatialJoinPreparedMemoryPlan = ReturnType<
  typeof import('./spatial-join-prepared-planning.js').getGPUSpatialJoinPreparedMemoryPlan
>;
export {
  GPUSpatialJoinCandidates,
  getGPUSpatialJoinCandidatesMaximumPairCapacity
} from './gpu-spatial-join-candidates';
export type {GPUSpatialJoinCandidatesProps} from './gpu-spatial-join-candidates';
export {
  formatGPUSpatialRelate,
  GPU_SPATIAL_RELATE_CELLS,
  GPU_SPATIAL_RELATE_PATTERN_WORDS,
  MAXIMUM_SPATIAL_RELATE_PATTERN_SLOT_COUNT,
  GPU_SPATIAL_RELATE_UNCERTAIN_BIT,
  packGPUSpatialRelate,
  packGPUSpatialRelatePattern
} from './spatial-relate-types';
export type {GPUSpatialRelatePattern} from './spatial-relate-types';
export type {GPUSpatialJoinGeometry, GPUSpatialJoinOnAttribute} from './spatial-join-types';
export {GPU_NEAREST_NO_SEGMENT} from './nearest-types';
export type {
  GPUNearestFeatureGeometry,
  GPUNearestQueryGeometry,
  GPUNearestTieMode
} from './nearest-types';
export {GPUNearestFeatureWeights} from './gpu-nearest-feature-weights';
export type {GPUNearestFeatureWeightsProps} from './gpu-nearest-feature-weights';
export type {SpatialSortCurve} from './spatial-join-passes';
export {getGPUSpatialJoinCapacityPlan} from './spatial-join-capacity-plan';
export type {
  GPUSpatialJoinCapacityPlan,
  GPUSpatialJoinCapacityPlanningOptions
} from './spatial-join-capacity-plan';
export {computeSpatialPredicateJoinOnCPU, computeNearestFeatureJoinOnCPU} from './cpu-feature-join';
export type {
  CPUSpatialJoinGeometry,
  CPUSpatialPredicateJoinResult,
  CPUNearestFeatureJoinResult,
  MultipartJoinPredicate,
  CPUFeatureJoinOptions
} from './cpu-feature-join';
export {prepareGeometryInput} from './prepare-geometry-input';
export type {PrepareGeometryInputOptions, PreparedGeometryInput} from './prepare-geometry-input';
export {decodeGPUSpatialJoinGeometry} from './spatial-join-geometry';
export type {GPUSpatialJoinGeometryReadback} from './spatial-join-geometry';
export {computeBufferSelectionOnCPU} from './cpu-buffer-selection-operation';
export type {
  CPUBufferSelectionInput,
  CPUBufferSelectionOptions
} from './cpu-buffer-selection-operation';
export {
  computePointInPolygonJoinOnCPU,
  computePointInPolygonCountsOnCPU
} from './cpu-point-in-polygon-join-operation';
export type {
  CPUPointInPolygonJoinOptions,
  CPUPointInPolygonJoinResult,
  CPUPointInPolygonCountsResult
} from './cpu-point-in-polygon-join-operation';
