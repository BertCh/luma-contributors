// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import type {
  GPUCardinalityPort,
  GPUOverflowStatusPort,
  GPUPackedGeometryPort,
  GPUPolygonGeometryPort
} from '../contracts/index';
import type {GPUSpatialContext} from '../contracts/spatial-context';
import type {GPUSegmentRingInputStatus} from './ring-input-completeness';

/** Which side of a directed segment the filled region lies on. */
export type GPUSegmentRingAssemblyInteriorSide = 'left' | 'right';

/**
 * Polygon layout of {@link GPUSegmentRingAssemblyOutput}: rings regrouped so every polygon feature
 * is its shell followed by its holes, in GeoArrow layout with implicitly closed rings (the repeated
 * closing vertex is dropped). Offsets are GPU-written and padded flat past the data, so views of
 * fixed length stay valid for compile-time topologies: unused polygons and rings are empty. Holes
 * with no enclosing shell are left out. The ordering is deterministic: polygons by shell ring
 * index, holes by ring index.
 */
export type GPUSegmentRingPolygonOutput = GPUPackedGeometryPort<GPUPolygonGeometryPort> & {
  /** Ring vertices in polygon order, without closing duplicates. Length at most `positions.length`. */
  positions: GraphDataView<'float32x2'>;
  /** Ring-to-vertex offsets, length `ringCapacity + 1`. */
  ringOffsets: GraphDataView<'uint32'>;
  /** Polygon-to-ring offsets, length `ringCapacity + 1` (the first ring of a polygon is its shell). */
  polygonOffsets: GraphDataView<'uint32'>;
  /** Identity feature-to-polygon offsets, padded with `polygonCount`; one feature per polygon. */
  featureOffsets: GraphDataView<'uint32'>;
  /**
   * Optional stable source ID of each polygon feature (its shell's group). Unused rows hold
   * `0xffffffff` ({@link GPU_SEGMENT_RING_ASSEMBLY_NONE}). Needs the `groups` input.
   */
  sourceIds?: GraphDataView<'uint32'>;
};

/**
 * Caller-owned, capacity-bounded outputs of {@link GPUSegmentRingAssembly}.
 *
 * Ring capacity is `ringOffsets.length - 1`; vertex capacity is `positions.length`. Rings are
 * written whole or not at all: the written rings are a prefix of the full ring list, `status.count` is
 * the number written and `status.overflow` is 1 when more rings existed.
 */
export type GPUSegmentRingAssemblyOutput = {
  /**
   * Ring `r` owns `positions[ringOffsets[r] .. ringOffsets[r + 1])`, closed (the last vertex
   * repeats the first, GeoJSON style). Entries past `status.count` repeat the final offset. Length
   * `ringCapacity + 1`.
   */
  ringOffsets: GraphDataView<'uint32'>;
  /** Ring vertices in ring order. */
  positions: GraphDataView<'float32x2'>;
  /**
   * Optional signed shoelace measure per ring in input units squared; counter-clockwise positive.
   * For longitude/latitude input this is an orientation-only measure in degrees², not an area.
   * Use it for winding/sign only and measure areas with `GPUGeometryMeasures`. Zero means an
   * empty walk (out and back along one path), and such rings are left out of `polygons`. A ring
   * whose float32 sum is exactly 0 although it encloses area (a face thinner than one float32
   * step; it has an exact non-zero turn at a lowest vertex it passes once) reports the smallest
   * normal float, `1.17549435e-38`, with the sign of that turn, so it is never taken for empty.
   */
  ringAreas?: GraphDataView<'float32'>;
  /**
   * Optional 1 for rings that are holes (interior on the outside of the ring), else 0. The ring's
   * orientation is the exact turn at its lowest vertex (smallest y, then x), so it is correct for
   * thin rings whose f32 `ringAreas` sign is not; a zero turn falls back to the area sign.
   */
  ringIsHole?: GraphDataView<'uint32'>;
  /**
   * Optional owning shell per ring: a shell's own ring index; a hole's innermost enclosing shell
   * (same group when `groups` is set), or {@link GPU_SEGMENT_RING_ASSEMBLY_NONE} when no shell
   * contains it. Containment is the exact ray test of a point beside the hole's first edge. Of two
   * shells that contain it, the inner one is the one whose bounding box lies inside the other's
   * (exact for a planar boundary, whose rings do not cross); only shells whose boxes are equal or
   * do not nest are ordered by float32 area (smaller first, lowest index on ties). Float32 areas
   * are not otherwise used: a thin shell and its hole can have areas that round to the same value
   * or the wrong order. A full-coordinate Morton-sorted BVH supplies shells whose boxes contain
   * the hole box; only those candidates receive the exact vertex walk.
   */
  ringShells?: GraphDataView<'uint32'>;
  /** Optional group label per ring (needs the `groups` input). */
  ringGroups?: GraphDataView<'uint32'>;
  /** Optional ring index per input segment, or NONE when the segment is in no written ring. */
  segmentRings?: GraphDataView<'uint32'>;
  /** Optional index into `positions` of each segment's start vertex, or NONE. */
  segmentVertices?: GraphDataView<'uint32'>;
  /** Optional `GPU_SEGMENT_RING_ASSEMBLY_FLAG_*` bits per input segment. */
  segmentFlags?: GraphDataView<'uint32'>;
  /**
   * Optional polygon layout of the written rings for consumers that need GeoArrow polygon
   * topology, such as `GPUPointInPolygonJoin`.
   */
  polygons?: GPUSegmentRingPolygonOutput;
  /**
   * One-row status scalars. `count` receives the number of rings written. `overflow` receives 1
   * when ring or vertex capacity dropped rings, or when a valid segment has a non-finite
   * coordinate (such vertices cannot be matched). The optional `requiredCount` receives the
   * unclamped ring count.
   */
  status: GPUCardinalityPort &
    GPUOverflowStatusPort & {
      /** Optional inherited outline loss, open-edge or continuation-conflict flag. */
      candidateOverflow?: GraphDataView<'uint32'>;
      /** Live polygon/feature count of the padded polygons layout; requires output.polygons. */
      polygonCount?: GraphDataView<'uint32'>;
      /** 1 when upstream input, assembly and final capacity are all complete. */
      complete?: GraphDataView<'uint32'>;
      /** Optional one-row number of valid segments on no closed ring. */
      openSegmentCount?: GraphDataView<'uint32'>;
      /** Optional one-row scalar receiving the number of segments whose end vertex is shared. */
      touchingSegmentCount?: GraphDataView<'uint32'>;
    };
};

/** Properties for {@link GPUSegmentRingAssembly}. */
export type GPUSegmentRingAssemblyProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'segment-ring-assembly'`. */
  id?: string;
  /** Directed segments `(x0, y0, x1, y1)`, for example `GPUCellSetOutline` `output.endpoints`. */
  endpoints: GraphDataView<'float32x4'>;
  /** Optional one-row count of valid segment rows. Defaults to every row. */
  count?: GraphDataView<'uint32'>;
  /** Counted outline provenance. If count is also supplied it must be the same view. */
  segmentStatus?: GPUSegmentRingInputStatus;
  /** Optional group label per segment; a ring never mixes groups. */
  groups?: GraphDataView<'uint32'>;
  /**
   * Treat consecutive even/odd segment rows as opposite directed half-edges, collapse duplicate
   * directed geometry within a group for ring traversal, and never immediately return along any
   * geometrically equivalent reverse edge. Duplicate input rows remain present and receive their
   * owner's ring, vertex and flag outputs. Directed duplicates require exact stored-coordinate
   * equality; vertex matching for ring connectivity still uses `vertexTolerance`. Used by general
   * polygonization. Defaults to false.
   */
  pairedOpposites?: boolean;
  /**
   * Compile-time vertex matching distance (Chebyshev, input units). Two vertices match when
   * both coordinates differ by at most this value. Choose at least four times the f32 spacing of
   * the coordinates (7.6e-6 degrees near longitude 100) and well below the shortest segment.
   * Defaults to `extent * 2^-21`, where the GPU measures the largest axis difference from the
   * first segment's start. With an explicit tolerance, distant groups leave local cells
   * unchanged. Where float32 spacing reaches the cell width, a cell can hold two consecutive
   * representable coordinates. Candidates are compared against the matching distance.
   */
  vertexTolerance?: number;
  /**
   * Which side of each segment is filled. Decides which ring is a shell and the rule at shared
   * vertices. `GPUCellSetOutline` of H3 is `'left'` (counter-clockwise shells), of Quadbin
   * `'right'`. Defaults to `'left'`.
   */
  interiorSide?: GPUSegmentRingAssemblyInteriorSide;
  /**
   * Reverse rings of `interiorSide: 'right'` input so every shell is counter-clockwise and every
   * hole clockwise (RFC 7946). Defaults to false (rings follow the segment direction).
   */
  normalizeWinding?: boolean;
  /**
   * Coordinate context of the segment endpoints, default native planar. A `longitude-latitude`
   * context (metric `'none'`) scales x differences by `cos(y)` when comparing turn angles at
   * shared vertices (degrees), so a turn is judged on the ground and not in stretched degrees;
   * the factor is clamped at 0.01 near the poles. The order of the turns at a vertex does not
   * depend on that scale (it is positive), so the continuation is chosen by the same exact
   * orientation tests as for planar input and the scale only enters the f32 pseudo-angle that
   * orders non-finite coordinates. This replaces the former `geographic` flag,
   * which defaulted to true: omitting the context is now planar. The output context preserves the
   * input `edgeModel` for longitude/latitude; an omitted model stays omitted. Prepare global or
   * antimeridian-crossing rings with `GPULonLatPolygonPreparation` before polygon consumers.
   * Only turn comparison is affected; `vertexTolerance` stays in position units.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Remove pairs of exactly opposite segments (`a -> b` and `b -> a` in one group, within the
   * vertex tolerance) before chaining. Such pairs are zero-width slits or ridges where a region is
   * cut by a line of zero thickness, for example band boundaries of `GPUIsobandRings` on samples
   * that equal a break; left in, they pinch rings into spikes and strand chains. Pairs match one
   * to one by lowest segment index; both segments get
   * {@link GPU_SEGMENT_RING_ASSEMBLY_FLAG_CANCELLED}. Defaults to false.
   */
  cancelOpposingSegments?: boolean;
  /**
   * Split rings that touch themselves at a vertex into separate rings (two holes meeting at a
   * corner, a hole pinched to its shell) at the cost of a second tracing pass. Regions touching at
   * a vertex are always separate rings. Without it the rule keeps interior wedges tight, so
   * touching holes join into one self-touching ring. Defaults to true.
   */
  splitTouchingRings?: boolean;
  /** Caller-owned output. */
  output: GPUSegmentRingAssemblyOutput;
};
