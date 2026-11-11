// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  GPUMaximumInscribedCircle,
  GPU_MAXIMUM_INSCRIBED_CIRCLE_STATUS
} from './gpu-maximum-inscribed-circle';
export type {GPUMaximumInscribedCircleProps} from './gpu-maximum-inscribed-circle';
export {GPUNormalizePolygons} from './gpu-normalize-polygons';
export type {GPUNormalizePolygonsProps} from './gpu-normalize-polygons';
export {GPUSegmentSnap} from './gpu-segment-snap';
export type {GPUSegmentSnapProps} from './gpu-segment-snap';
export {GPURatioConcaveHull} from './gpu-ratio-concave-hull';
export type {GPURatioConcaveHullProps} from './gpu-ratio-concave-hull';
export {GPULinePrecisionReduction, GPUPolygonPrecisionReduction} from './gpu-precision-reduction';
export type {
  GPULinePrecisionReductionProps,
  GPUPolygonPrecisionReductionProps
} from './gpu-precision-reduction';
export {GPUSubdividePolygons} from './gpu-subdivide-polygons';
export type {GPUSubdividePolygonsProps} from './gpu-subdivide-polygons';
export {GPUConvexStraightSkeleton} from './gpu-convex-straight-skeleton';
export type {GPUConvexStraightSkeletonProps} from './gpu-convex-straight-skeleton';
export {GPULineLoopRemoval} from './gpu-line-loop-removal';
export type {GPULineLoopRemovalProps} from './gpu-line-loop-removal';
export {GPUCanonicalPolygonEquality} from './gpu-canonical-polygon-equality';
export type {GPUCanonicalPolygonEqualityProps} from './gpu-canonical-polygon-equality';
export {GPUNormalizeGeometry, GPU_CANONICAL_GEOMETRY_TYPE} from './gpu-normalize-geometry';
export type {GPUNormalizeGeometryProps, GPUCanonicalGeometryTree} from './gpu-normalize-geometry';
export {GPUCanonicalGeometryEquality} from './gpu-canonical-geometry-equality';
export type {GPUCanonicalGeometryEqualityProps} from './gpu-canonical-geometry-equality';
export {GPUVectorStraightSkeleton} from './gpu-vector-straight-skeleton';
export type {GPUVectorStraightSkeletonProps} from './gpu-vector-straight-skeleton';
export {GPUOffsetCurveComponents} from './gpu-offset-curve-components';
export type {GPUOffsetCurveComponentsProps} from './gpu-offset-curve-components';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {
  computeMaximumInscribedCircleOnCPU,
  CPUMaximumInscribedCircleOperation
} from './cpu-maximum-inscribed-circle-operation';
export type {CPUMaximumInscribedCircleResult} from './cpu-maximum-inscribed-circle-operation';
export {GPUMaximumInscribedCircleOperation} from './gpu-maximum-inscribed-circle-operation';
export type {MaximumInscribedCircleOperationProps} from './gpu-maximum-inscribed-circle-operation';
export {
  computeNormalizeGeometryOnCPU,
  CPUNormalizeGeometryOperation
} from './cpu-normalize-geometry-operation';
export type {
  CPUNormalizeGeometryInput,
  CPUNormalizeGeometryResult
} from './cpu-normalize-geometry-operation';
export {
  computeNormalizePolygonsOnCPU,
  CPUNormalizePolygonsOperation
} from './cpu-normalize-polygons-operation';
export type {
  CPUNormalizePolygonsInput,
  CPUNormalizePolygonsResult
} from './cpu-normalize-polygons-operation';
export {computeSegmentSnapOnCPU, CPUSegmentSnapOperation} from './cpu-segment-snap-operation';
export type {
  CPUSegmentSnapInput,
  CPUSegmentSnapOptions,
  CPUSegmentSnapResult
} from './cpu-segment-snap-operation';
export {GPUNormalizeGeometryOperation} from './gpu-normalize-geometry-operation';
export type {NormalizeGeometryOperationProps} from './gpu-normalize-geometry-operation';
export {GPUNormalizePolygonsOperation} from './gpu-normalize-polygons-operation';
export type {NormalizePolygonsOperationProps} from './gpu-normalize-polygons-operation';
export {GPUSegmentSnapOperation} from './gpu-segment-snap-operation';
export type {SegmentSnapOperationProps} from './gpu-segment-snap-operation';
export {
  computeConvexStraightSkeletonOnCPU,
  CPUConvexStraightSkeletonOperation
} from './cpu-convex-straight-skeleton-operation';
export type {
  CPUConvexStraightSkeletonInput,
  CPUConvexStraightSkeletonOptions,
  CPUConvexStraightSkeletonResult
} from './cpu-convex-straight-skeleton-operation';
export {
  computeLineLoopRemovalOnCPU,
  CPULineLoopRemovalOperation
} from './cpu-line-loop-removal-operation';
export type {
  CPULineLoopRemovalInput,
  CPULineLoopRemovalOptions,
  CPULineLoopRemovalResult
} from './cpu-line-loop-removal-operation';
export {
  computeOffsetCurveComponentsOnCPU,
  CPUOffsetCurveComponentsOperation
} from './cpu-offset-curve-components-operation';
export type {
  CPUOffsetCurveComponentsInput,
  CPUOffsetCurveComponentsOptions,
  CPUOffsetCurveComponentsResult
} from './cpu-offset-curve-components-operation';
export {GPUConvexStraightSkeletonOperation} from './gpu-convex-straight-skeleton-operation';
export type {ConvexStraightSkeletonOperationProps} from './gpu-convex-straight-skeleton-operation';
export {GPULineLoopRemovalOperation} from './gpu-line-loop-removal-operation';
export type {LineLoopRemovalOperationProps} from './gpu-line-loop-removal-operation';
export {GPUOffsetCurveComponentsOperation} from './gpu-offset-curve-components-operation';
export type {OffsetCurveComponentsOperationProps} from './gpu-offset-curve-components-operation';
export {
  computeRatioConcaveHullOnCPU,
  CPURatioConcaveHullOperation
} from './cpu-ratio-concave-hull-operation';
export type {
  CPURatioConcaveHullOptions,
  CPURatioConcaveHullResult
} from './cpu-ratio-concave-hull-operation';
export {GPURatioConcaveHullOperation} from './gpu-ratio-concave-hull-operation';
export type {RatioConcaveHullOperationProps} from './gpu-ratio-concave-hull-operation';
export {
  computeVectorStraightSkeletonOnCPU,
  CPUVectorStraightSkeletonOperation
} from './cpu-vector-straight-skeleton-operation';
export type {
  CPUVectorStraightSkeletonInput,
  CPUVectorStraightSkeletonOptions,
  CPUVectorStraightSkeletonResult
} from './cpu-vector-straight-skeleton-operation';
export {GPUVectorStraightSkeletonOperation} from './gpu-vector-straight-skeleton-operation';
export type {VectorStraightSkeletonOperationProps} from './gpu-vector-straight-skeleton-operation';
