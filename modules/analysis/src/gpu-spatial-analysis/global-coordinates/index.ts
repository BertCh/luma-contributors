// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {GPUSphericalConvexContains} from './gpu-spherical-convex-contains';
export type {GPUSphericalConvexContainsProps} from './gpu-spherical-convex-contains';
export {GPUSphericalConvexIntersection} from './gpu-spherical-convex-intersection';
export type {GPUSphericalConvexIntersectionProps} from './gpu-spherical-convex-intersection';

export {GPUSphericalPolygonContains} from './gpu-spherical-polygon-contains';
export type {GPUSphericalPolygonContainsProps} from './gpu-spherical-polygon-contains';
export {GPUSphericalPolygonOverlay} from './gpu-spherical-polygon-overlay';
export type {
  GPUSphericalPolygonOverlayProps,
  GPUSphericalPolygonOperand
} from './gpu-spherical-polygon-overlay';
export {GPUSphericalArcOverlay} from './gpu-spherical-arc-overlay';
export type {
  GPUSphericalArcOverlayProps,
  GPUSphericalArcOperand
} from './gpu-spherical-arc-overlay';
export {
  GPU_SPHERICAL_MINIMUM_TOLERANCE,
  GPU_SPHERICAL_MAXIMUM_TOLERANCE,
  GPU_SPHERICAL_MAXIMUM_ARRANGEMENT_TOLERANCE,
  GPU_SPHERICAL_CONVEX_MINIMUM_VERTICES,
  GPU_SPHERICAL_CONVEX_MAXIMUM_VERTICES,
  GPU_SPHERICAL_CONVEX_INTERSECTION_MAXIMUM_CAPACITY,
  GPU_SPHERICAL_ARC_OVERLAY_MAXIMUM_INPUT_VERTICES,
  GPU_SPHERICAL_ARC_OVERLAY_MAXIMUM_OUTPUT_VERTICES,
  GPU_SPHERICAL_OVERLAY_MINIMUM_COSINE,
  GPU_S2_COVERING_MAXIMUM_LEVEL,
  GPU_SPHERICAL_SPATIAL_CONTEXT_SUPPORT,
  GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT
} from './spherical-limits';
export {
  GPU_SPHERICAL_UNIT_VECTOR_LENGTH_TOLERANCE,
  GPU_SPHERICAL_UNIT_VECTOR_SQUARED_LENGTH_TOLERANCE,
  validateUnitVector
} from './assert-unit-vector';
export {
  decodeGPUSphericalConvexContainsStatus,
  decodeGPUSphericalPolygonContainsStatus,
  decodeGPUSphericalConvexIntersectionStatus,
  decodeGPUSphericalArcOverlayStatus,
  decodeGPUSphericalConvexIntersectionCounts,
  decodeGPUSphericalArcOverlayCounts,
  decodeGPUSphericalConvexIntersectionReadback,
  decodeGPUSphericalArcOverlayReadback
} from './spherical-status';
export type {
  SphericalReadback,
  GPUSphericalConvexIntersectionReadbackInput,
  GPUSphericalArcOverlayReadbackInput
} from './spherical-status';
export {
  GPU_SPHERICAL_CONVEX_CONTAINS_STATUS_VALID,
  GPU_SPHERICAL_CONVEX_CONTAINS_STATUS_INVALID,
  GPU_SPHERICAL_CONVEX_CONTAINS_STATUS_BOUNDARY
} from './gpu-spherical-convex-contains';
export {
  GPU_SPHERICAL_CONVEX_INTERSECTION_STATUS_INVALID,
  GPU_SPHERICAL_CONVEX_INTERSECTION_STATUS_CAPACITY,
  GPU_SPHERICAL_CONVEX_INTERSECTION_STATUS_AMBIGUOUS
} from './gpu-spherical-convex-intersection';
export {
  GPU_SPHERICAL_POLYGON_CONTAINS_STATUS_VALID,
  GPU_SPHERICAL_POLYGON_CONTAINS_STATUS_INVALID,
  GPU_SPHERICAL_POLYGON_CONTAINS_STATUS_BOUNDARY
} from './gpu-spherical-polygon-contains';
export {
  GPU_SPHERICAL_POLYGON_OVERLAY_COMPLEMENT_NONE,
  GPU_SPHERICAL_POLYGON_OVERLAY_COMPLEMENT_SPHERE
} from './gpu-spherical-polygon-overlay';
export {
  GPU_SPHERICAL_ARC_OVERLAY_STATUS_INVALID,
  GPU_SPHERICAL_ARC_OVERLAY_STATUS_CAPACITY,
  GPU_SPHERICAL_ARC_OVERLAY_STATUS_UNRESOLVED
} from './gpu-spherical-arc-overlay';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {
  computeSphericalArcOverlayOnCPU,
  CPUSphericalArcOverlayOperation
} from './cpu-spherical-arc-overlay-operation';
export type {
  CPUSphericalArcOverlayInput,
  CPUSphericalArcOverlayOptions,
  CPUSphericalArcOverlayResult
} from './cpu-spherical-arc-overlay-operation';
export {GPUSphericalArcOverlayOperation} from './gpu-spherical-arc-overlay-operation';
export type {SphericalArcOverlayOperationProps} from './gpu-spherical-arc-overlay-operation';
