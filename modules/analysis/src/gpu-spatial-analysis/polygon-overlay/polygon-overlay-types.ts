// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import type {
  GPUBoundedResultStatusPort,
  GPUPackedGeometryPort,
  GPUPolygonGeometryPort,
  GPURingCountStatusPort
} from '../../analysis-core/contracts/index';
import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import type {GPUPolygonOverlayFeatureSources} from './overlay-feature-sources';

/** Boolean set operation applied by {@link GPUPolygonOverlay}. */
export type GPUPolygonOverlayOperation =
  | 'intersection'
  | 'union'
  | 'difference'
  | 'symmetric-difference'
  | 'dissolve';

/**
 * Rule that decides whether a point is inside one polygon of an operand, from the winding number
 * `w` of all rings of that polygon about the point (counter-clockwise rings count +1, clockwise
 * rings -1). A point is inside the operand when it is inside any of its polygons.
 *
 * - `'even-odd'` (default): `w` is odd. For valid polygons this is the OGC shell-and-holes reading;
 *   it differs only for holes outside their shell or overlapping holes.
 * - `'positive'`, `'negative'` and `'nonzero'`: `w > 0`, `w < 0` or `w != 0`. Use them for raw,
 *   self-overlapping boundaries such as offset curves, where regions covered twice must stay inside
 *   (`'even-odd'` would carve them out as holes).
 */
export type GPUPolygonOverlayFillRule = 'even-odd' | 'positive' | 'negative' | 'nonzero';

/** Explicit capacities for the bounded arrangement built by {@link GPUPolygonOverlay}. */
export type GPUPolygonOverlayCapacity = {
  /**
   * Crossings retained while noding both operands (each becomes a hot pixel). Both operands
   * together have at most {@link getGPUSegmentIntersectionMaximumVertexCount} vertex rows plus
   * one per ring, and those rows plus this capacity at most a quarter of that count (four diagonal
   * vertices per hot pixel share one segment table), 1048576 on a default device.
   */
  intersections: number;
  /** Atomic segments retained after noding. */
  nodedSegments: number;
};

/** Boundary provenance emitted before the selected edges are assembled into rings. */
export type GPUPolygonOverlayBoundaryPort = {
  endpoints: GraphDataView<'float32x4'>;
  /** Zero for the left operand and one for the right operand. */
  operandIds: GraphDataView<'uint32'>;
  /** Feature that owns the source edge within its operand. */
  sourceFeatureIds: GraphDataView<'uint32'>;
  status: Pick<GPUBoundedResultStatusPort, 'count' | 'overflow'> & {
    requiredCount: GraphDataView<'uint32'>;
    candidateOverflow: GraphDataView<'uint32'>;
  };
};

/**
 * Canonical, capacity-bounded polygon result of {@link GPUPolygonOverlay}.
 *
 * Features are polygons (one shell and its holes each); `status.count` counts them. The result is a
 * Boolean combination of operand unions, so one output polygon can lie over several input features
 * of each operand: it has no single source row, and `geometry` has no `sourceIds` (contributor
 * contract rule 8). Provenance is per boundary edge (`boundary.operandIds`,
 * `boundary.sourceFeatureIds`). A label point joined to each operand identifies only features
 * containing that point; it cannot enumerate an output polygon's sources (a dissolved shell can
 * cover adjacent input parcels). For source membership, join output polygons to each operand
 * with an explicitly chosen polygon relation, such as positive-area interior intersection,
 * keeping boundary-only contacts separate. Attribute-preserving membership needs a relation
 * carrying output feature row, operand and source row; `featureSources` is only a representative
 * retained boundary owner, not that relation.
 *
 * **Breaking (2026-11-10):** `geometry.sourceIds` (it held the output row index) is removed, and
 * `status.requiredCount`, which held the unclamped ring count, is removed: the unclamped feature
 * count is not known when rings were dropped, so there is no feature requirement distinct from
 * `count`. Size the ring capacity from `status.requiredRingCount`.
 */
export type GPUPolygonOverlayOutput = {
  /** Optional per-output-feature representative boundary source row. Mixed provenance is explicit;
   * this does not enumerate interior features or every owner of coincident boundary edges.
   * Valid only when both output.status.overflow and output.status.candidateOverflow are zero,
   * like the generated polygon itself. */
  featureSources?: GPUPolygonOverlayFeatureSources;
  /** Geometry is complete only when both status.overflow and status.candidateOverflow are zero. */
  geometry: Omit<GPUPackedGeometryPort<GPUPolygonGeometryPort>, 'sourceIds'>;
  status: Pick<GPUBoundedResultStatusPort, 'count' | 'overflow'> &
    GPURingCountStatusPort & {
      /**
       * Optional one-row unclamped number of rings (shells plus holes) the assembly needed, which
       * sizes every view of `geometry` (they share one ring capacity). `ringCount` (optional) is the
       * number of rings written, at most the ring capacity.
       */
      requiredRingCount?: GraphDataView<'uint32'>;
      /** Also set when assembly drops open segments or encounters continuation conflicts. */
      candidateOverflow: GraphDataView<'uint32'>;
      /**
       * Optional one-row count of segment pairs the intersection predicates could not certify.
       * They set `candidateOverflow` whether or not this view is given; `overflow` alone does
       * not indicate noding uncertainty. Reject geometry and provenance while either flag is set.
       */
      uncertainCount?: GraphDataView<'uint32'>;
      /** Optional unclamped noded-piece requirement; exact when candidateOverflow is zero. */
      nodedSegmentRequiredCount?: GraphDataView<'uint32'>;
    };
  boundary: GPUPolygonOverlayBoundaryPort;
};

export type PackedPolygons = GPUPackedGeometryPort<GPUPolygonGeometryPort> & {
  /** Optional source feature ID per feature. */
  sourceIds?: GraphDataView<'uint32'>;
};

/** Properties for {@link GPUPolygonOverlay}. */
export type GPUPolygonOverlayProps = {
  id?: string;
  /**
   * Planar only (default: native planar coordinates). A longitude-latitude context throws at
   * construction: project positions to a planar frame first, for example with
   * `createGPULocalFrame` (`contracts/local-frame`) and `GPULocalFrameProjection`
   * (`geometry-measures`), and pass the frame's context.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Context of `right` when declared separately from `spatialContext`. Must be compatible in
   * datum, coordinate space, metric, units and planar frame.
   * Omitted means `spatialContext` describes both inputs.
   */
  rightSpatialContext?: GPUSpatialContext;
  left: PackedPolygons;
  /** Required except for `dissolve`, which unions every polygon in `left`. */
  right?: PackedPolygons;
  operation: GPUPolygonOverlayOperation;
  /**
   * Inside rule applied to both operands. Default `'even-odd'`. Compile-time. See
   * {@link GPUPolygonOverlayFillRule}.
   */
  fillRule?: GPUPolygonOverlayFillRule;
  capacity: GPUPolygonOverlayCapacity;
  /**
   * Snap grid of the overlay. Every vertex and every crossing is rounded to a lattice of step
   * `cell`, the smallest power of two that is at least `vertexTolerance`, at least 2^-22 of the
   * input extent (the larger of the x and y extents of the finite vertices) and at least 2^-24 of
   * the largest coordinate magnitude (one `float32` step there), anchored at a multiple of `cell`
   * (a power-of-two `vertexTolerance` therefore reproduces shapely `grid_size` exactly whenever
   * `float32` can store that grid at the coordinates of the data). The result lies on that
   * lattice: its vertices are lattice points, every lattice point is a `float32` value, and edges
   * closer than one cell collapse or merge. For data farther from zero than about four times its
   * extent the last bound decides: at a projected northing of 4,000,000 the step is 0.25 to 0.5
   * whatever the tolerance, so use coordinates relative to a local origin for a finer result.
   * Finite and positive.
   */
  vertexTolerance: number;
  output: GPUPolygonOverlayOutput;
  spatialSort?: boolean;
  leafCapacity?: number;
};
