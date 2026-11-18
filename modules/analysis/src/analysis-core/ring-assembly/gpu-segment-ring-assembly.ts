// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  GPUSort,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../kernels/wgsl-kernel-nodes';
import {
  validateGPUStorageBindingByteLength,
  validateCapacity,
  validateFiniteNumber,
  validateGraphOutputsDisjointFromInputs,
  validateSinglePackedView
} from '../kernels/gpu-contributor-utils';
import type {
  GPUCardinalityPort,
  GPUOverflowStatusPort,
  GPUPackedGeometryPort,
  GPUPolygonGeometryPort
} from '../contracts/index';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext,
  type GPUSpatialContextSupport
} from '../contracts/spatial-context';
import {
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges
} from '../kernels/graph-chunk-dispatch';
import {IS_FINITE_VALUE_WGSL} from '../kernels/wgsl-common';
import {getRingQuantizerWGSL} from './ring-vertex-quantizer';
import {getRingShellIndexNodes, RING_SHELL_RECORD_WORDS} from './ring-shell-index';
import {EXACT_ORIENTATION_WGSL} from '../kernels/exact-orientation-wgsl';
import {EXACT_GEOMETRY_WGSL, getExactRingWindingWGSL} from '../kernels/exact-geometry-wgsl';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../kernels/construction-snapshot';

/** A positive float32 tolerance must survive rounding before division in the vertex quantizer. */
const GPU_SEGMENT_RING_ASSEMBLY_MINIMUM_VERTEX_TOLERANCE = 2 ** -149;

/** Planar coordinates, or longitude-latitude positions that only need a cos(latitude) turn scale. */
const GPU_RING_ASSEMBLY_SPATIAL_CONTEXT_SUPPORT: GPUSpatialContextSupport = {
  coordinateSpaces: ['planar', 'longitude-latitude'],
  metrics: ['none', 'native']
};

const OPERATION = 'GPUSegmentRingAssembly';
/** Lanes per workgroup; a workgroup owns this many rings and sums its larger rings cooperatively. */
const RING_LANES = 64;

/**
 * Exact turn classification for the planar successor choice (`match` kernel).
 *
 * `ringExactTurnClass(origin, vertex, end, side)` places the turn from `vertex - origin` to
 * `end - vertex` in the order of its angle toward the interior side, in (-pi, pi]: 0 for a turn
 * away from the interior side, 1 for straight on (also a zero-length incoming or outgoing
 * segment), 2 for a turn toward the interior side or a u-turn. The caller supplies `side` through
 * its single `exactOrientSigns` call chain and disables exact choice when it cannot be certified.
 * It agrees with the sign classes of `ringTurnKey`, so the exact order
 * (class, then `orientSign` between two candidates of one open half-plane) is the f32 key order
 * without its rounding. The collinear case compares coordinates only.
 *
 * The two interior sides are mirror images: the orientation signs are multiplied by `SIDE_ORDER`
 * (1 for `'left'`, -1 for `'right'`) before they are classed or compared, so the kernel for
 * `'right'` decides as the kernel for `'left'` does on the mirror image of the input. In
 * particular an exact u-turn is the last class for both: the face rule takes the first edge met
 * when turning from the reversed incoming direction toward the interior side, and an edge that
 * exactly reverses the incoming one is met first whichever way that turn goes. (Until the
 * re-review of 2026-11-11 the turn was classed left or right and the order reversed for
 * `'right'`, which put the u-turn first for `'left'` and last for `'right'`.)
 *
 * `ringSegmentTurnClass(origin, vertex, start, end)` gives the same classes for the turn from the
 * segment `origin -> vertex` to the segment `start -> end`, which need not start at `vertex`. It
 * is used when a candidate starts within the tolerance of the end vertex but not on it, so that
 * there is no common vertex to measure the turn at. The order is that of the directions
 * `vertex - origin` and `end - start` as real vectors, not of their f32 differences:
 * `ringCrossSign(a, b, c, d)` is the exact sign of `cross(b - a, d - c)` for four f32 points
 * (`-1`, `0`, `1`, or `2` when it cannot be certified). The eight products of raw coordinates of
 *
 *     bx dy - bx cy - ax dy + ax cy - by dx + by cx + ay dx - ay cx
 *
 * are summed exactly with the integer machinery of `orientSign` (`multiply24`,
 * `accumulateShifted`: 48-bit products aligned to the smallest exponent in two 256-bit
 * accumulators), behind a float filter on the rounded differences. With `c == b` it is
 * `orientSign(a, b, d)`. The float filter and the exact sum hold no table: the terms are
 * recomputed instead of stored in a local array (contributor contract, section 11).
 *
 * None of these signs changes when one axis is scaled by a positive factor, so the same order
 * holds for longitude/latitude input, whose turns are measured with longitude differences scaled
 * by the cosine of the latitude.
 */
const RING_EXACT_TURN_WGSL = /* wgsl */ `${EXACT_ORIENTATION_WGSL}
${EXACT_GEOMETRY_WGSL}
fn ringExactTurnClass(origin: vec2f, vertex: vec2f, end: vec2f, side: i32) -> i32 {
  // Positive toward the interior side.
  let toward = side * SIDE_ORDER;
  if (toward < 0) { return 0; }
  if (toward > 0) { return 2; }
  // Collinear: behind the vertex (dot product negative) is the u-turn.
  var behind = false;
  if (vertex.x != origin.x) {
    behind = (vertex.x > origin.x && end.x < vertex.x) || (vertex.x < origin.x && end.x > vertex.x);
  } else if (vertex.y != origin.y) {
    behind = (vertex.y > origin.y && end.y < vertex.y) || (vertex.y < origin.y && end.y > vertex.y);
  }
  return select(1, 2, behind);
}
// Bit patterns of the two factors of term 'term' of
// cross(b - a, d - c) = bx dy - bx cy - ax dy + ax cy - by dx + by cx + ay dx - ay cx.
// The sign of the term is folded into the first factor.
fn ringCrossFactors(term: u32, a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> vec2u {
  let negate = 0x80000000u;
  switch (term) {
    case 0u: { return vec2u(bitcast<u32>(b.x), bitcast<u32>(d.y)); }
    case 1u: { return vec2u(bitcast<u32>(b.x) ^ negate, bitcast<u32>(c.y)); }
    case 2u: { return vec2u(bitcast<u32>(a.x) ^ negate, bitcast<u32>(d.y)); }
    case 3u: { return vec2u(bitcast<u32>(a.x), bitcast<u32>(c.y)); }
    case 4u: { return vec2u(bitcast<u32>(b.y) ^ negate, bitcast<u32>(d.x)); }
    case 5u: { return vec2u(bitcast<u32>(b.y), bitcast<u32>(c.x)); }
    case 6u: { return vec2u(bitcast<u32>(a.y), bitcast<u32>(d.x)); }
    default: { return vec2u(bitcast<u32>(a.y) ^ negate, bitcast<u32>(c.x)); }
  }
}

// One term as an exact 48-bit product times 2^exponent; 'present' is false for a zero term.
struct RingCrossTerm {
  low: u32,
  high: u32,
  exponent: i32,
  negative: bool,
  present: bool,
}

fn ringCrossTerm(term: u32, a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> RingCrossTerm {
  let factors = ringCrossFactors(term, a, b, c, d);
  var result = RingCrossTerm(0u, 0u, 0, false, false);
  let firstMagnitude = factors.x & 0x7fffffffu;
  let secondMagnitude = factors.y & 0x7fffffffu;
  if (firstMagnitude == 0u || secondMagnitude == 0u) { return result; }
  let firstExponentBits = firstMagnitude >> 23u;
  let secondExponentBits = secondMagnitude >> 23u;
  let firstFraction = firstMagnitude & 0x7fffffu;
  let secondFraction = secondMagnitude & 0x7fffffu;
  let firstSignificand = select(0x800000u | firstFraction, firstFraction, firstExponentBits == 0u);
  let secondSignificand = select(0x800000u | secondFraction, secondFraction, secondExponentBits == 0u);
  let firstExponent = select(i32(firstExponentBits) - 150, -149, firstExponentBits == 0u);
  let secondExponent = select(i32(secondExponentBits) - 150, -149, secondExponentBits == 0u);
  let product = multiply24(firstSignificand, secondSignificand);
  result.low = product.x;
  result.high = product.y;
  result.exponent = firstExponent + secondExponent;
  result.negative = ((factors.x ^ factors.y) >> 31u) == 1u;
  result.present = true;
  return result;
}

fn ringCrossExact(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> i32 {
  var minimumExponent = 100000;
  var maximumExponent = -100000;
  for (var term = 0u; term < 8u; term++) {
    let value = ringCrossTerm(term, a, b, c, d);
    if (!value.present) { continue; }
    minimumExponent = min(minimumExponent, value.exponent);
    maximumExponent = max(maximumExponent, value.exponent);
  }
  if (minimumExponent > maximumExponent) { return 0; }
  if (maximumExponent - minimumExponent > ORIENT_MAXIMUM_SPAN) { return 2; }
  var positive: array<u32, 8>;
  var negative: array<u32, 8>;
  for (var term = 0u; term < 8u; term++) {
    let value = ringCrossTerm(term, a, b, c, d);
    if (!value.present) { continue; }
    let shift = u32(value.exponent - minimumExponent);
    if (value.negative) {
      accumulateShifted(&negative, value.low, value.high, shift);
    } else {
      accumulateShifted(&positive, value.low, value.high, shift);
    }
  }
  for (var slot = 8u; slot > 0u; slot--) {
    let plus = positive[slot - 1u];
    let minus = negative[slot - 1u];
    if (plus != minus) { return select(-1, 1, plus > minus); }
  }
  return 0;
}

// Exact sign of cross(b - a, d - c) for four f32 points: -1, 0, 1, or 2 when it cannot be
// certified (non-finite coordinates, or terms that span more than 200 binary orders).
fn ringCrossSign(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> i32 {
  if (!(orientIsFinite(a.x) && orientIsFinite(a.y) && orientIsFinite(b.x) && orientIsFinite(b.y) &&
        orientIsFinite(c.x) && orientIsFinite(c.y) && orientIsFinite(d.x) && orientIsFinite(d.y))) {
    return 2;
  }
  // Each difference is rounded once and each product once, so the float determinant is within
  // 2.5e-7 of the magnitude sum of the exact one; the bound keeps a factor of two. Products near
  // the smallest f32 values lose more than that and go to the exact sum.
  let first = b - a;
  let second = d - c;
  let detLeft = first.x * second.y;
  let detRight = first.y * second.x;
  let det = detLeft - detRight;
  let magnitude = abs(detLeft) + abs(detRight);
  let bound = 5.0e-7 * magnitude;
  if (magnitude > 1.0e-30) {
    if (det > bound) { return 1; }
    if (-det > bound) { return -1; }
  }
  return ringCrossExact(a, b, c, d);
}

// The same classes for the turn from segment origin -> vertex to segment start -> end (neither
// of zero length), by the exact sign of the cross product of their directions.
fn ringSegmentTurnClass(origin: vec2f, vertex: vec2f, start: vec2f, end: vec2f) -> i32 {
  let side = ringCrossSign(origin, vertex, start, end);
  if (side == 2) { return -1; }
  // Positive toward the interior side.
  let toward = side * SIDE_ORDER;
  if (toward < 0) { return 0; }
  if (toward > 0) { return 2; }
  // Parallel: opposite signs of a non-zero component are the u-turn. A component of one
  // direction is zero exactly when it is zero in the other.
  var behind = false;
  if (vertex.x != origin.x) {
    behind = (vertex.x > origin.x) != (end.x > start.x);
  } else {
    behind = (vertex.y > origin.y) != (end.y > start.y);
  }
  return select(1, 2, behind);
}
`;

/** Sentinel for "no ring", "no segment" and "no shell" in uint32 outputs (`0xffffffff`). */
export const GPU_SEGMENT_RING_ASSEMBLY_NONE = 0xffffffff;
/** `segmentFlags` bit: more than one segment starts at this segment's end vertex. */
export const GPU_SEGMENT_RING_ASSEMBLY_FLAG_TOUCHING = 1;
/**
 * `segmentFlags` bit: no segment starts at this segment's end vertex (open chain), or the segment
 * has zero length (identical start and end), which is on no ring.
 */
export const GPU_SEGMENT_RING_ASSEMBLY_FLAG_DANGLING = 2;
/** `segmentFlags` bit: the chosen continuation was already taken by a lower-index segment. */
export const GPU_SEGMENT_RING_ASSEMBLY_FLAG_CONFLICT = 4;

/**
 * `segmentFlags` bit: the segment and an exactly opposite segment (same group, end and start
 * vertices matching within the tolerance) were cancelled by `cancelOpposingSegments`; neither is on
 * a ring and neither counts as open.
 */
export const GPU_SEGMENT_RING_ASSEMBLY_FLAG_CANCELLED = 8;

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
      /** Optional one-row scalar receiving the number of valid segments that lie on no closed ring. */
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
   * Defaults to four f32 ulps of the frame extent (`extent * 2^-21`, measured on the GPU), so only
   * separately rounded copies of one vertex match (**behavior change:** the default was an
   * absolute 5e-5). Vertex cells are never smaller than this, and grow with the frame extent so
   * that projected coordinates cannot collide (see {@link GPUSegmentRingAssembly}).
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

/**
 * Chains directed boundary segments into closed rings on the GPU.
 *
 * Every segment's end vertex is matched to the segment starting there (coordinates within
 * `vertexTolerance`, found through a sort of quantized vertex hashes and a 3 x 3 neighborhood
 * lookup). Vertices are quantized relative to the start vertex of segment row 0 into unsigned cell
 * indices whose size is `max(vertexTolerance, extent * 2^-22)` (extent: the largest coordinate
 * offset from that origin, measured on the GPU), so large projected coordinates (UTM, Web
 * Mercator) hash to distinct cells at any tolerance; candidates are still compared by their true
 * distance against `vertexTolerance`. Only non-finite coordinates set `output.status.overflow`. At a vertex shared by several rings (cells touching at a corner) the continuation is
 * the outgoing segment that turns tightest toward the interior side (the face-traversal rule, a
 * bijection on planar boundaries), which keeps regions that touch only at a point as separate
 * rings. A candidate that exactly reverses the incoming segment is the tightest turn for either
 * interior side, so a scene and its mirror image with the interior on the other side give the
 * same rings (until 2026-11-11 such a candidate was the first choice for `'left'` and the last
 * for `'right'`). Turns are ordered exactly. Where the candidates all start exactly at the segment's end
 * vertex (lattice input such as `GPUPolygonOverlay`'s), the order is by half-plane against the
 * incoming direction, then by the exact orientation of the two candidates about that vertex, so
 * edges whose directions differ by less than f32 resolution are still told apart. When a
 * candidate starts within the tolerance of the end vertex but not on it (a tolerance match),
 * there is no common vertex, and the same order is taken from the directions of the segments
 * themselves: the exact sign of the cross product of `end - start` of one segment with
 * `end - start` of the other, computed from the four endpoints and not from the f32 differences,
 * which round when the two ends of a segment differ in magnitude. Candidates whose directions
 * are exactly parallel go to the lowest index. The same order holds for longitude/latitude
 * input: its turns are measured with longitude differences scaled by the cosine of the latitude,
 * a positive scale of one axis that changes no orientation sign, so the exact tests run on the
 * degrees as they are. Only non-finite coordinates are ordered by an f32 pseudo-angle (of the
 * scaled differences for longitude/latitude). Where a choice is still taken twice the lowest segment
 * index wins, the loser is marked
 * {@link GPU_SEGMENT_RING_ASSEMBLY_FLAG_CONFLICT} and left open. That rule makes holes that touch at a
 * corner (or a hole pinched to its shell) one self-touching ring, so a second tracing pass
 * (`splitTouchingRings`, default) swaps the continuations of two segments that end at the same
 * vertex whenever both lie on one ring, which splits it into simple rings.
 *
 * The successor graph is a set of disjoint paths and cycles. Pointer jumping
 * (`ceil(log2(rows))` rounds per phase) finds each cycle's lowest segment index, which becomes its
 * leader and gives the ring ID (rank among leaders, so rings are ordered by lowest segment index),
 * then ranks every segment from the leader. Scans of ring and vertex counts place each ring; a
 * ring starts at its leader segment and follows the segment direction (or is reversed by
 * `normalizeWinding`). Chains that never close (dangling ends, truncated input) are reported
 * through `openSegmentCount` and `segmentFlags` and emit no ring. A zero-length segment (identical
 * start and end) has no direction: it is never a continuation, takes none, and is reported the same
 * way, so it cannot break the ring through its vertex.
 *
 * Rings are written as GeoArrow-style `ringOffsets` plus `positions`. The exact orientation at each
 * ring's lowest vertex identifies shells and holes (library rule 1); `ringShells` assigns each hole to its innermost enclosing shell by
 * ray casting a point of the hole's first edge against shells found through a full-coordinate Morton-sorted BVH
 * (only nodes whose boxes contain the hole box are visited; ring statistics give each
 * lane its own ring and sum rings above 64 vertices cooperatively, so a single huge ring does not serialize them). `polygons` regroups the rings into polygon topology for
 * `GPUPointInPolygonJoin`. Output order and results are deterministic (integer atomics only).
 *
 * Limits: matching is geometric, so shared vertices must agree within the tolerance (exact for
 * Quadbin, within f32 round-off for H3); rings that span the antimeridian keep the longitudes of
 * the input. Prepare global or antimeridian-crossing rings with `GPULonLatPolygonPreparation`
 * before polygon consumers. Area uses f32 arithmetic relative to each ring's first vertex; it is
 * reported, and decides nothing except the order of two shells with equal bounding boxes.
 */
export class GPUSegmentRingAssembly implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSegmentRingAssemblyProps;
  /**
   * Resolved input and output context: planar, or longitude/latitude for cos(latitude) turn
   * comparison. Preserves the input longitude/latitude `edgeModel`, including its omission.
   */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUSegmentRingAssemblyProps) {
    props = Object.freeze({
      ...snapshotConstructionFields(props, [
        'id',
        'endpoints',
        'count',
        'groups',
        'pairedOpposites',
        'vertexTolerance',
        'interiorSide',
        'normalizeWinding',
        'spatialContext',
        'cancelOpposingSegments',
        'splitTouchingRings',
        'output'
      ]),
      spatialContext:
        props.spatialContext && snapshotConstructionSpatialContext(props.spatialContext),
      output: Object.freeze({
        ...snapshotConstructionFields(props.output, [
          'ringOffsets',
          'positions',
          'ringAreas',
          'ringIsHole',
          'ringShells',
          'ringGroups',
          'segmentRings',
          'segmentVertices',
          'segmentFlags',
          'polygons',
          'status'
        ]),
        status: snapshotConstructionFields(props.output.status, [
          'count',
          'overflow',
          'requiredCount',
          'openSegmentCount',
          'touchingSegmentCount'
        ]),
        polygons:
          props.output.polygons &&
          snapshotConstructionFields(props.output.polygons, [
            'kind',
            'positions',
            'ringOffsets',
            'polygonOffsets',
            'featureOffsets',
            'sourceIds'
          ])
      })
    });
    this.id = props.id ?? 'segment-ring-assembly';
    this.spatialContext = snapshotConstructionSpatialContext(
      resolveGPUSpatialContext(
        this.id,
        props.spatialContext,
        GPU_RING_ASSEMBLY_SPATIAL_CONTEXT_SUPPORT
      )
    );
    this.props = props;
    const id = this.id;
    const {output} = props;
    validateSinglePackedView(props.endpoints, ['float32x4'], `${id} endpoints`);
    const segments = props.endpoints.length;
    if (segments < 1) {
      throw new Error(`${id} needs at least one segment row`);
    }
    if (props.vertexTolerance !== undefined) {
      validateFiniteNumber(
        id,
        'vertexTolerance',
        props.vertexTolerance,
        GPU_SEGMENT_RING_ASSEMBLY_MINIMUM_VERTEX_TOLERANCE
      );
    }
    if (props.interiorSide !== undefined && !['left', 'right'].includes(props.interiorSide)) {
      throw new Error(`${id} interiorSide must be 'left' or 'right'`);
    }
    for (const [name, view] of [
      ['count', props.count],
      ['output.status.count', output.status.count],
      ['output.status.overflow', output.status.overflow],
      ['output.status.requiredCount', output.status.requiredCount],
      ['output.status.openSegmentCount', output.status.openSegmentCount],
      ['output.status.touchingSegmentCount', output.status.touchingSegmentCount]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length < 1) {
          throw new Error(`${id} ${name} must contain one uint32 row`);
        }
      }
    }
    if (props.groups) {
      validatePackedUint32View(props.groups, `${id} groups`);
      if (props.groups.length !== segments) {
        throw new Error(`${id} groups length must equal the segment row count`);
      }
    }
    if (output.ringGroups && !props.groups) {
      throw new Error(`${id} output.ringGroups needs the groups input`);
    }
    validatePackedUint32View(output.ringOffsets, `${id} output.ringOffsets`);
    if (output.ringOffsets.length < 2) {
      throw new Error(`${id} output.ringOffsets needs at least two rows (one ring)`);
    }
    validateSinglePackedView(output.positions, ['float32x2'], `${id} output.positions`);
    if (output.positions.length < 1) {
      throw new Error(`${id} output.positions must not be empty`);
    }
    const ringCapacity = output.ringOffsets.length - 1;
    for (const [name, view] of [
      ['ringIsHole', output.ringIsHole],
      ['ringShells', output.ringShells],
      ['ringGroups', output.ringGroups]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} output.${name}`);
        if (view.length !== ringCapacity) {
          throw new Error(`${id} output.${name} length must equal the ring capacity`);
        }
      }
    }
    if (output.ringAreas) {
      validateSinglePackedView(output.ringAreas, ['float32'], `${id} output.ringAreas`);
      if (output.ringAreas.length !== ringCapacity) {
        throw new Error(`${id} output.ringAreas length must equal the ring capacity`);
      }
    }
    if (output.polygons) {
      const {polygons} = output;
      validateSinglePackedView(
        polygons.positions,
        ['float32x2'],
        `${id} output.polygons.positions`
      );
      if (polygons.positions.length !== output.positions.length) {
        throw new Error(`${id} output.polygons.positions must match output.positions length`);
      }
      if (polygons.sourceIds) {
        if (!props.groups) {
          throw new Error(`${id} output.polygons.sourceIds needs the groups input`);
        }
        validatePackedUint32View(polygons.sourceIds, `${id} output.polygons.sourceIds`);
        if (polygons.sourceIds.length !== ringCapacity) {
          throw new Error(
            `${id} output.polygons.sourceIds must have ${ringCapacity} rows (the ring capacity)`
          );
        }
      }
      for (const [name, view, length] of [
        ['ringOffsets', polygons.ringOffsets, ringCapacity + 1],
        ['polygonOffsets', polygons.polygonOffsets, ringCapacity + 1],
        ['featureOffsets', polygons.featureOffsets, ringCapacity + 1]
      ] as const) {
        validatePackedUint32View(view, `${id} output.polygons.${name}`);
        if (view.length !== length) {
          throw new Error(`${id} output.polygons.${name} must have ${length} rows`);
        }
      }
    }
    for (const [name, view] of [
      ['segmentRings', output.segmentRings],
      ['segmentVertices', output.segmentVertices],
      ['segmentFlags', output.segmentFlags]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} output.${name}`);
        if (view.length !== segments) {
          throw new Error(`${id} output.${name} length must equal the segment row count`);
        }
      }
    }
    validateRingOutputDispatches(id, output, Boolean(props.pairedOpposites));
    validateGraphOutputsDisjointFromInputs(id, getOutputViews(output), [
      props.endpoints,
      props.count,
      props.groups
    ]);
  }

  /** Returns the hash, sort, match, pointer jumping, ranking, placement and ring-stat nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    validateGraphChunkBindingRanges(graph, id, {
      endpoints: props.endpoints,
      count: props.count,
      groups: props.groups,
      ...Object.fromEntries(getOutputViews(output).map((view, index) => [`output ${index}`, view]))
    });
    const segments = props.endpoints.length;
    const ringCapacity = output.ringOffsets.length - 1;
    const vertexCapacity = output.positions.length;
    const validateScratch = (name: string, byteLength: number) => {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        byteLength,
        'reduce the segment or ring capacity'
      );
      validateCapacity(id, `${name} byteLength`, byteLength, 0, graph.device.limits.maxBufferSize);
    };
    // Endpoint storage bounds per-segment sort/scan words. Ring words can be wider when callers
    // reserve more output rings than input segments; include every fixed/derived allocation.
    validateScratch('ring counters scratch', 6 * Uint32Array.BYTES_PER_ELEMENT);
    validateScratch('segment word scratch', segments * Uint32Array.BYTES_PER_ELEMENT);
    validateScratch('ring word scratch', ringCapacity * Uint32Array.BYTES_PER_ELEMENT);
    if (output.ringShells || output.polygons) {
      const leafCapacity = 2 ** Math.ceil(Math.log2(ringCapacity));
      validateScratch(
        'shell records',
        ringCapacity * RING_SHELL_RECORD_WORDS * Uint32Array.BYTES_PER_ELEMENT
      );
      validateScratch('shell bounds', ringCapacity * 4 * Float32Array.BYTES_PER_ELEMENT);
      validateScratch(
        'shell BVH nodes',
        (2 * leafCapacity - 1) * 2 * Float32Array.BYTES_PER_ELEMENT
      );
    }
    // Without an explicit tolerance, vertices match within 4 ulps of the frame extent (derived on
    // the GPU from the measured extent, rule 4), so separately rounded copies of one vertex meet.
    const tolerance = props.vertexTolerance ?? 0;
    const interiorSide = props.interiorSide ?? 'left';
    const flip = Boolean(props.normalizeWinding) && interiorSide === 'right';
    // Orientation sign of a shell in the written rings: +1 counter-clockwise.
    const shellSign = props.normalizeWinding || interiorSide === 'left' ? 1 : -1;
    const geographic = this.spatialContext.coordinateSpace === 'longitude-latitude';
    const hasGroups = Boolean(props.groups);
    const none = `${GPU_SEGMENT_RING_ASSEMBLY_NONE}u`;
    const transient = <Format extends 'uint32' | 'float32' | 'float32x2'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, length);
    const u32 = (name: string, length: number = segments) => transient(name, 'uint32', length);

    const keys = u32('keys');
    const ids = u32('ids');
    const sortedKeys = u32('sorted-keys');
    const sortedIds = u32('sorted-ids');
    const duplicateOwner = props.pairedOpposites ? u32('duplicate-owner') : undefined;
    const claim = u32('claim');
    const cancelOpposing = Boolean(props.cancelOpposingSegments);
    const counters = u32('counters', 6);
    const next = u32('next');
    const alternatives = u32('alternatives');
    const swapped = u32('swapped');
    const segmentFlags = output.segmentFlags ?? u32('flags');
    const leader = u32('leader');
    const leaderFlags = u32('leader-flags');
    const leaderVertices = u32('leader-vertices');
    const ringScan = u32('ring-scan');
    const vertexScan = u32('vertex-scan');
    const ringStarts = u32('ring-starts');
    const ringLeaders = u32('ring-leaders');
    const ringCount = u32('ring-count', 1);
    const segmentRings = output.segmentRings ?? u32('segment-rings');
    const segmentVertices = output.segmentVertices ?? u32('segment-vertices');
    const closeVertices = u32('close-vertices');
    const pointerSets = [u32('pointer-a'), u32('pointer-b')];
    const valueSets = [u32('value-a'), u32('value-b')];

    const toleranceLiteral = getWGSLFloatLiteral(tolerance);
    const toleranceWGSL =
      props.vertexTolerance === undefined
        ? `fn ringTolerance() -> f32 { return ringFrameExtent() * ${getWGSLFloatLiteral(2 ** -21)}; }`
        : `fn ringTolerance() -> f32 { return ${toleranceLiteral}; }`;
    // counters[4] holds the frame extent (f32 bits); counters[5] holds the valid row count so that
    // the match kernel can read both from one binding and stay within 8 storage buffers.
    const hashFunctions = getRingQuantizerWGSL(toleranceLiteral, false, 4);
    const atomicHashFunctions = getRingQuantizerWGSL(toleranceLiteral, true, 4);
    const validRows = (name: string) => `min(${name}[${name}Offset], ${segments}u)`;
    const countBinding: WGSLKernelBinding[] = props.count
      ? [{name: 'rowCount', view: props.count, type: 'u32', access: 'read'}]
      : [];
    const groupBinding: WGSLKernelBinding[] = props.groups
      ? [{name: 'groups', view: props.groups, type: 'u32', access: 'read'}]
      : [];
    const validRowsExpression = props.count ? validRows('rowCount') : `${segments}u`;
    const groupOf = (row: string) => (hasGroups ? `groups[groupsOffset + ${row}]` : '0u');
    const kernel = (
      name: string,
      bindings: WGSLKernelBinding[],
      invocationCount: number,
      body: string,
      declarations: string = ''
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${name}`,
        operation: OPERATION,
        variant: name,
        bindings,
        invocationCount,
        declarations,
        body
      });
    const read = (name: string, view: GraphDataView, type: 'u32' | 'f32' = 'u32') =>
      ({name, view, type, access: 'read'}) as WGSLKernelBinding;
    const write = (name: string, view: GraphDataView, type: 'u32' | 'f32' = 'u32') =>
      ({name, view, type, access: 'read_write'}) as WGSLKernelBinding;

    const nodes: GPUCommandNode<Parameters>[] = [];

    // 0. Zero the counters, then measure the frame extent: the largest |x - ox| or |y - oy| of any
    // endpoint relative to the origin (non-negative f32 bits order like u32, so atomicMax works).
    // Non-finite vertices are counted into counters[3] and reported as overflow.
    nodes.push(
      kernel(
        'quantize-init',
        [...countBinding, write('counters', counters)],
        1,
        `counters[countersOffset] = 0u;
  counters[countersOffset + 1u] = 0u;
  counters[countersOffset + 2u] = 0u;
  counters[countersOffset + 3u] = 0u;
  counters[countersOffset + 4u] = 0u;
  counters[countersOffset + 5u] = ${validRowsExpression};`
      ),
      kernel(
        'quantize-extent',
        [
          read('endpoints', props.endpoints, 'f32'),
          ...countBinding,
          {name: 'counters', view: counters, type: 'atomic<u32>', access: 'read_write'}
        ],
        segments,
        `if (index >= ${validRowsExpression}) { return; }
  let origin = vec2f(endpoints[endpointsOffset], endpoints[endpointsOffset + 1u]);
  let start = vec2f(endpoints[endpointsOffset + 4u * index], endpoints[endpointsOffset + 4u * index + 1u]) - origin;
  let end = vec2f(endpoints[endpointsOffset + 4u * index + 2u], endpoints[endpointsOffset + 4u * index + 3u]) - origin;
  if (!isFiniteValue(start.x) || !isFiniteValue(start.y) || !isFiniteValue(end.x) || !isFiniteValue(end.y)) {
    atomicAdd(&counters[countersOffset + 3u], 1u);
    return;
  }
  let extent = max(max(abs(start.x), abs(start.y)), max(abs(end.x), abs(end.y)));
  atomicMax(&counters[countersOffset + 4u], bitcast<u32>(extent));`,
        IS_FINITE_VALUE_WGSL
      )
    );

    // 1. Hash start vertices (invalid rows get the maximum key so they sort last).
    nodes.push(
      kernel(
        'hash',
        [
          read('endpoints', props.endpoints, 'f32'),
          ...countBinding,
          ...groupBinding,
          write('keys', keys),
          write('ids', ids),
          write('claim', claim),
          // Read only: the quantizer reads the frame extent in counters[4].
          read('counters', counters)
        ],
        segments,
        `ids[idsOffset + index] = index;
  claim[claimOffset + index] = 0xffffffffu;
  var key = 0xffffffffu;
  if (index < ${validRowsExpression}) {
    let start = vec2f(endpoints[endpointsOffset + 4u * index], endpoints[endpointsOffset + 4u * index + 1u]);
    let origin = ringOrigin();
    key = ringHash(ringQuantize(start.x, origin.x), ringQuantize(start.y, origin.y), ${groupOf('index')});
  }
  keys[keysOffset + index] = key;`,
        hashFunctions
      )
    );
    nodes.push(
      ...new GPUSort({
        id: `${id}-sort`,
        keys,
        values: ids,
        outputKeys: sortedKeys,
        outputValues: sortedIds
      }).getCommandNodes(graph)
    );

    // In polygonization, overlapping input boundaries can produce multiple rows for the same
    // directed atomic edge. Give each such row one deterministic traversal owner. The sorted start
    // vertex index bounds the search to the 3 x 3 neighboring hash cells already used by matching;
    // original rows and their source provenance are never removed or reordered.
    if (duplicateOwner) {
      // The high flag bit is an internal marker and is replaced by the owner's public flags later.
      nodes.push(
        kernel(
          'duplicate-owners',
          [
            read('endpoints', props.endpoints, 'f32'),
            read('counters', counters),
            ...groupBinding,
            read('sortedKeys', sortedKeys),
            read('sortedIds', sortedIds),
            write('duplicateOwner', duplicateOwner)
          ],
          segments,
          `duplicateOwner[duplicateOwnerOffset + index] = index;
  let valid = min(counters[countersOffset + 5u], ${segments}u);
  if (index >= valid) { return; }
  let start = ringStart(index);
  let end = ringEnd(index);
  let group = ${groupOf('index')};
  let cellOrigin = ringOrigin();
  let cellX = ringQuantize(start.x, cellOrigin.x);
  let cellY = ringQuantize(start.y, cellOrigin.y);
  var seen = array<u32, 9>();
  var seenCount = 0u;
  var owner = index;
  for (var deltaY = -1; deltaY <= 1; deltaY++) {
    for (var deltaX = -1; deltaX <= 1; deltaX++) {
      let hash = ringHash(cellX + bitcast<u32>(deltaX), cellY + bitcast<u32>(deltaY), group);
      var duplicateHash = false;
      for (var seenIndex = 0u; seenIndex < seenCount; seenIndex++) {
        duplicateHash = duplicateHash || seen[seenIndex] == hash;
      }
      if (duplicateHash) { continue; }
      seen[seenCount] = hash;
      seenCount++;
      var low = 0u;
      var high = ${segments}u;
      while (low < high) {
        let middle = low + (high - low) / 2u;
        if (sortedKeys[sortedKeysOffset + middle] < hash) { low = middle + 1u; } else { high = middle; }
      }
      var position = low;
      while (position < ${segments}u && sortedKeys[sortedKeysOffset + position] == hash) {
        let candidate = sortedIds[sortedIdsOffset + position];
        position++;
        if (candidate >= valid || candidate >= owner) { continue; }
        if (${hasGroups ? `groups[groupsOffset + candidate] != group` : 'false'}) { continue; }
        let candidateStart = ringStart(candidate);
        let candidateEnd = ringEnd(candidate);
        // Exact stored-coordinate equality makes owner identity transitive, so propagation never
        // depends on another duplicate being written in the same dispatch.
        if (any(candidateStart != start) || any(candidateEnd != end)) { continue; }
        owner = candidate;
      }
    }
  }
  duplicateOwner[duplicateOwnerOffset + index] = owner;`,
          `${hashFunctions}
${toleranceWGSL}
fn ringStart(row: u32) -> vec2f {
  return vec2f(endpoints[endpointsOffset + 4u * row], endpoints[endpointsOffset + 4u * row + 1u]);
}
fn ringEnd(row: u32) -> vec2f {
  return vec2f(endpoints[endpointsOffset + 4u * row + 2u], endpoints[endpointsOffset + 4u * row + 3u]);
}`
        )
      );
    }

    // 1b. Cancel exactly opposite segment pairs.
    if (cancelOpposing) {
      nodes.push(
        kernel(
          'cancel',
          [
            read('endpoints', props.endpoints, 'f32'),
            ...countBinding,
            ...groupBinding,
            read('sortedKeys', sortedKeys),
            read('sortedIds', sortedIds),
            write('flags', segmentFlags),
            {name: 'counters', view: counters, type: 'atomic<u32>', access: 'read_write'}
          ],
          segments,
          `flags[flagsOffset + index] = 0u;
  if (index >= ${validRowsExpression}) { return; }
  let partner = findReverse(index);
  if (partner != ${none} && findReverse(partner) == index) {
    flags[flagsOffset + index] = ${GPU_SEGMENT_RING_ASSEMBLY_FLAG_CANCELLED}u;
    atomicAdd(&counters[countersOffset + 2u], 1u);
  }`,
          `${atomicHashFunctions}
${toleranceWGSL}
fn ringStart(row: u32) -> vec2f {
  return vec2f(endpoints[endpointsOffset + 4u * row], endpoints[endpointsOffset + 4u * row + 1u]);
}
fn ringEnd(row: u32) -> vec2f {
  return vec2f(endpoints[endpointsOffset + 4u * row + 2u], endpoints[endpointsOffset + 4u * row + 3u]);
}
// Lowest segment that starts where 'index' ends and ends where it starts.
fn findReverse(index: u32) -> u32 {
  let valid = ${validRowsExpression};
  let origin = ringStart(index);
  let vertex = ringEnd(index);
  let group = ${groupOf('index')};
  let cellOrigin = ringOrigin();
  let qx = ringQuantize(vertex.x, cellOrigin.x);
  let qy = ringQuantize(vertex.y, cellOrigin.y);
  var seen = array<u32, 9>();
  var seenCount = 0u;
  var best = ${none};
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let hash = ringHash(qx + bitcast<u32>(dx), qy + bitcast<u32>(dy), group);
      var duplicate = false;
      for (var j = 0u; j < seenCount; j++) { duplicate = duplicate || seen[j] == hash; }
      if (duplicate) { continue; }
      seen[seenCount] = hash;
      seenCount++;
      var low = 0u;
      var high = ${segments}u;
      while (low < high) {
        let middle = low + (high - low) / 2u;
        if (sortedKeys[sortedKeysOffset + middle] < hash) { low = middle + 1u; } else { high = middle; }
      }
      var position = low;
      while (position < ${segments}u && sortedKeys[sortedKeysOffset + position] == hash) {
        let candidate = sortedIds[sortedIdsOffset + position];
        position++;
        if (candidate >= valid || candidate == index) { continue; }
        let candidateStart = ringStart(candidate);
        let candidateEnd = ringEnd(candidate);
        if (abs(candidateStart.x - vertex.x) > ringTolerance() || abs(candidateStart.y - vertex.y) > ringTolerance()) { continue; }
        if (abs(candidateEnd.x - origin.x) > ringTolerance() || abs(candidateEnd.y - origin.y) > ringTolerance()) { continue; }
        if (${hasGroups ? `groups[groupsOffset + candidate] != group` : 'false'}) { continue; }
        best = min(best, candidate);
      }
    }
  }
  return best;
}`
        )
      );
    }

    if (duplicateOwner) {
      nodes.push(
        kernel(
          'mark-duplicate-owners',
          [read('duplicateOwner', duplicateOwner), write('flags', segmentFlags)],
          segments,
          `if (duplicateOwner[duplicateOwnerOffset + index] != index ${cancelOpposing ? `&& (flags[flagsOffset + index] & ${GPU_SEGMENT_RING_ASSEMBLY_FLAG_CANCELLED}u) == 0u` : ''}) {
    flags[flagsOffset + index] = flags[flagsOffset + index] | 0x80000000u;
  } ${cancelOpposing ? '' : `else { flags[flagsOffset + index] = 0u; }`}`
        )
      );
    }

    // 2. Choose each segment's continuation.
    const sideSign = interiorSide === 'left' ? '1.0' : '-1.0';
    nodes.push(
      kernel(
        'match',
        [
          read('endpoints', props.endpoints, 'f32'),
          read('counters', counters),
          ...groupBinding,
          read('sortedKeys', sortedKeys),
          read('sortedIds', sortedIds),
          write('next', next),
          {name: 'flags', view: segmentFlags, type: 'atomic<u32>', access: 'read_write'},
          write('alt', alternatives)
        ],
        segments,
        `next[nextOffset + index] = ${none};
  alt[altOffset + index] = ${none};
  ${
    cancelOpposing
      ? `let carried = atomicLoad(&flags[flagsOffset + index]) & ${GPU_SEGMENT_RING_ASSEMBLY_FLAG_CANCELLED}u;
  let duplicateMarker = atomicLoad(&flags[flagsOffset + index]) & ${duplicateOwner ? '0x80000000u' : '0u'};
  atomicStore(&flags[flagsOffset + index], carried | duplicateMarker);
  if (carried != 0u || duplicateMarker != 0u) { return; }`
      : `if ((atomicLoad(&flags[flagsOffset + index]) & ${duplicateOwner ? '0x80000000u' : '0u'}) != 0u) { return; }
  atomicStore(&flags[flagsOffset + index], 0u);`
  }
  let valid = min(counters[countersOffset + 5u], ${segments}u);
  if (index >= valid) { return; }
  let origin = ringStart(index);
  let vertex = ringEnd(index);
  // A zero-length segment has no direction, so no turn can be ordered against it: it takes no
  // continuation here and is no candidate below. Left in, it reads as straight on, and with a
  // lower index than the real segment into its vertex it takes that segment's continuation.
  if (all(origin == vertex)) {
    atomicStore(&flags[flagsOffset + index], ${GPU_SEGMENT_RING_ASSEMBLY_FLAG_DANGLING}u);
    return;
  }
  let group = ${groupOf('index')};
  let incoming = ringScaled(vertex - origin, vertex.y);
  let cellOrigin = ringOrigin();
  let qx = ringQuantize(vertex.x, cellOrigin.x);
  let qy = ringQuantize(vertex.y, cellOrigin.y);
  var seen = array<u32, 9>();
  var seenCount = 0u;
  var best = ${none};
  var bestTurn = 0.0;
  var matches = 0u;
  var first = ${none};
  var second = ${none};
  // Exact choice, used when every candidate starts exactly at this segment's end vertex.
  var exactUsable = true;
  var exactBest = ${none};
  var exactBestClass = 0;
  var exactBestEnd = vec2f(0.0);
  // The same choice on the exact directions of the segments, used when a candidate starts
  // within the tolerance of this segment's end vertex but not on it.
  var directionUsable = true;
  var directionBest = ${none};
  var directionBestClass = 0;
  var directionBestStart = vec2f(0.0);
  var directionBestEnd = vec2f(0.0);
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let hash = ringHash(qx + bitcast<u32>(dx), qy + bitcast<u32>(dy), group);
      var duplicate = false;
      for (var j = 0u; j < seenCount; j++) { duplicate = duplicate || seen[j] == hash; }
      if (duplicate) { continue; }
      seen[seenCount] = hash;
      seenCount++;
      var low = 0u;
      var high = ${segments}u;
      while (low < high) {
        let middle = low + (high - low) / 2u;
        if (sortedKeys[sortedKeysOffset + middle] < hash) { low = middle + 1u; } else { high = middle; }
      }
      var position = low;
      while (position < ${segments}u && sortedKeys[sortedKeysOffset + position] == hash) {
        let candidate = sortedIds[sortedIdsOffset + position];
        position++;
        if (candidate >= valid || candidate == index) { continue; }
        ${duplicateOwner ? 'if ((atomicLoad(&flags[flagsOffset + candidate]) & 0x80000000u) != 0u) { continue; }' : ''}
        ${cancelOpposing ? `if ((atomicLoad(&flags[flagsOffset + candidate]) & ${GPU_SEGMENT_RING_ASSEMBLY_FLAG_CANCELLED}u) != 0u) { continue; }` : ''}
        let candidateStart = ringStart(candidate);
        if (abs(candidateStart.x - vertex.x) > ringTolerance() || abs(candidateStart.y - vertex.y) > ringTolerance()) { continue; }
        if (all(ringEnd(candidate) == candidateStart)) { continue; }
        ${
          props.pairedOpposites
            ? `let candidateEnd = ringEnd(candidate);
        if (abs(candidateEnd.x - origin.x) <= ringTolerance() && abs(candidateEnd.y - origin.y) <= ringTolerance()) { continue; }`
            : ''
        }
        if (${hasGroups ? `groups[groupsOffset + candidate] != group` : 'false'}) { continue; }
        matches++;
        if (first == ${none}) { first = candidate; } else if (second == ${none}) { second = candidate; }
        let outgoing = ringScaled(ringEnd(candidate) - candidateStart, vertex.y);
        let turn = ringTurnKey(SIDE * (incoming.x * outgoing.y - incoming.y * outgoing.x), dot(incoming, outgoing));
        if (best == ${none} || turn > bestTurn || (turn == bestTurn && candidate < best)) {
          best = candidate;
          bestTurn = turn;
        }
        let turnEnd = ringEnd(candidate);
        if (exactUsable) {
          // Both exact decisions share one call chain. The second sign is evaluated only when
          // the classes tie, preserving the original uncertainty and fallback choices.
          var turnClass = -1;
          var candidateSide = 0;
          var orientationStep = 0u;
          loop {
            var signStart = origin;
            var signEnd = vertex;
            var signPoint = turnEnd;
            if (orientationStep == 1u) {
              signStart = vertex;
              signEnd = turnEnd;
              signPoint = exactBestEnd;
            }
            let exact = exactOrientSigns(
              1u,
              array<vec2f, 4>(signStart, vec2f(0.0), vec2f(0.0), vec2f(0.0)),
              array<vec2f, 4>(signEnd, vec2f(0.0), vec2f(0.0), vec2f(0.0)),
              array<vec2f, 4>(signPoint, vec2f(0.0), vec2f(0.0), vec2f(0.0))
            );
            if (exact.uncertain) { exactUsable = false; break; }
            if (orientationStep == 1u) { candidateSide = exact.signs.x; break; }
            turnClass = ringExactTurnClass(origin, vertex, turnEnd, exact.signs.x);
            if (any(candidateStart != vertex) || exactBest == ${none} || turnClass != exactBestClass || turnClass == 1) { break; }
            orientationStep = 1u;
          }
          if (any(candidateStart != vertex) || !exactUsable) {
            exactUsable = false;
          } else if (exactBest == ${none}) {
            exactBest = candidate;
            exactBestClass = turnClass;
            exactBestEnd = turnEnd;
          } else {
            var order = turnClass - exactBestClass;
            if (order == 0 && turnClass != 1) {
              // Same half-plane: the direction further round toward the interior side turns
              // further (counter-clockwise for the interior on the left).
              order = -candidateSide * SIDE_ORDER;
            }
            if (order > 0 || (order == 0 && candidate < exactBest)) {
              exactBest = candidate;
              exactBestClass = turnClass;
              exactBestEnd = turnEnd;
            }
          }
        }
        if (directionUsable) {
          let turnClass = ringSegmentTurnClass(origin, vertex, candidateStart, turnEnd);
          if (turnClass < 0) {
            directionUsable = false;
          } else if (directionBest == ${none}) {
            directionBest = candidate;
            directionBestClass = turnClass;
            directionBestStart = candidateStart;
            directionBestEnd = turnEnd;
          } else {
            var order = turnClass - directionBestClass;
            if (order == 0 && turnClass != 1) {
              // Same half-plane: the direction further round toward the interior side turns
              // further (counter-clockwise for the interior on the left).
              let side = ringCrossSign(candidateStart, turnEnd, directionBestStart, directionBestEnd);
              if (side == 2) { directionUsable = false; }
              order = -side * SIDE_ORDER;
            }
            if (order > 0 || (order == 0 && candidate < directionBest)) {
              directionBest = candidate;
              directionBestClass = turnClass;
              directionBestStart = candidateStart;
              directionBestEnd = turnEnd;
            }
          }
        }
      }
    }
  }
  // The f32 pseudo-angle decides only where an exact sign cannot be certified (non-finite
  // coordinates). The exact orders hold for longitude/latitude input too: the cos(latitude)
  // scale of the longitude differences is positive and changes no orientation sign.
  if (exactUsable && exactBest != ${none}) {
    best = exactBest;
  } else if (directionUsable && directionBest != ${none}) {
    best = directionBest;
  }
  next[nextOffset + index] = best;
  if (matches == 2u) { alt[altOffset + index] = select(first, second, best == first); }
  var flagBits = 0u;
  if (matches > 1u) { flagBits = ${GPU_SEGMENT_RING_ASSEMBLY_FLAG_TOUCHING}u; }
  if (matches == 0u) { flagBits = ${GPU_SEGMENT_RING_ASSEMBLY_FLAG_DANGLING}u; }
  atomicStore(&flags[flagsOffset + index], flagBits);`,
        `${hashFunctions}
${toleranceWGSL}
const SIDE: f32 = ${sideSign};
const SIDE_ORDER: i32 = ${interiorSide === 'left' ? 1 : -1};
// Monotone in the turn angle from the incoming to the outgoing direction, positive toward the
// interior side: 'cross' is the cross product times SIDE, so the key of a scene with the interior
// on the right is the key of its mirror image with the interior on the left, and the u-turn is
// at +2 for both. atan2 is avoided: it misbehaves on signed zeros under fast math.
fn ringTurnKey(cross: f32, dot: f32) -> f32 {
  let denominator = abs(cross) + abs(dot);
  if (denominator == 0.0) { return 0.0; }
  let ratio = cross / denominator;
  if (dot >= 0.0) { return ratio; }
  return select(-2.0, 2.0, cross >= 0.0) - ratio;
}
fn ringStart(row: u32) -> vec2f {
  return vec2f(endpoints[endpointsOffset + 4u * row], endpoints[endpointsOffset + 4u * row + 1u]);
}
fn ringEnd(row: u32) -> vec2f {
  return vec2f(endpoints[endpointsOffset + 4u * row + 2u], endpoints[endpointsOffset + 4u * row + 3u]);
}
fn ringScaled(delta: vec2f, latitude: f32) -> vec2f {
  ${geographic ? 'return vec2f(delta.x * max(cos(radians(latitude)), 0.01), delta.y);' : 'return delta;'}
}
${RING_EXACT_TURN_WGSL}`
      )
    );

    // 3. Make the successor map injective: the lowest-index claimant keeps a continuation.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-claim`,
        operation: OPERATION,
        variant: 'claim',
        bindings: [
          read('next', next),
          {name: 'claim', view: claim, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: segments,
        body: `let claimed = next[nextOffset + index];
  if (claimed != ${none}) { atomicMin(&claim[claimOffset + claimed], index); }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-resolve`,
        operation: OPERATION,
        variant: 'resolve',
        bindings: [
          write('next', next),
          read('claim', claim),
          write('flags', segmentFlags),
          {name: 'counters', view: counters, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: segments,
        body: `let claimed = next[nextOffset + index];
  if (claimed != ${none} && claim[claimOffset + claimed] != index) {
    next[nextOffset + index] = ${none};
    flags[flagsOffset + index] = flags[flagsOffset + index] | ${GPU_SEGMENT_RING_ASSEMBLY_FLAG_CONFLICT}u;
  }
  if ((flags[flagsOffset + index] & ${GPU_SEGMENT_RING_ASSEMBLY_FLAG_TOUCHING}u) != 0u) {
    atomicAdd(&counters[countersOffset + 1u], 1u);
  }`
      })
    );

    // 4. Trace: pointer jumping finds each cycle's lowest segment index (leader) and the hops of
    // every segment to the segment before its leader. Closed segments sit on cycles.
    const rounds = Math.max(1, Math.ceil(Math.log2(segments)));
    const trace = (pass: number, successor: GraphDataView<'uint32'>, countOpen: boolean) => {
      nodes.push(
        kernel(
          `jump-init-${pass}`,
          [read('next', successor), write('pointer', pointerSets[0]), write('value', valueSets[0])],
          segments,
          `pointer[pointerOffset + index] = next[nextOffset + index];
  value[valueOffset + index] = index;`
        )
      );
      let current = 0;
      for (let round = 0; round < rounds; round++) {
        nodes.push(createJumpNode('min', `${pass}-${round}`, current));
        current = 1 - current;
      }
      const cutTarget = 1 - current;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-cut-${pass}`,
          operation: OPERATION,
          variant: 'cut',
          bindings: [
            read('next', successor),
            read('pointerIn', pointerSets[current]),
            read('valueIn', valueSets[current]),
            ...countBinding,
            write('leader', leader),
            write('pointerOut', pointerSets[cutTarget]),
            write('valueOut', valueSets[cutTarget]),
            {name: 'counters', view: counters, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: segments,
          body: `let valid = ${validRowsExpression};
  var leaderIndex = ${none};
  var tail = ${none};
  var hopCount = 0u;
  if (index < valid) {
    if (pointerIn[pointerInOffset + index] != ${none}) {
      leaderIndex = valueIn[valueInOffset + index];
      tail = next[nextOffset + index];
      if (tail == leaderIndex) { tail = ${none}; }
      if (tail != ${none}) { hopCount = 1u; }
    } else {
      ${countOpen ? 'atomicAdd(&counters[countersOffset], 1u);' : ''}
    }
  }
  leader[leaderOffset + index] = leaderIndex;
  pointerOut[pointerOutOffset + index] = tail;
  valueOut[valueOutOffset + index] = hopCount;`
        })
      );
      current = cutTarget;
      for (let round = 0; round < rounds; round++) {
        nodes.push(createJumpNode('sum', `${pass}-${round}`, current));
        current = 1 - current;
      }
      return valueSets[current];
    };
    let hops = trace(1, next, props.splitTouchingRings === false);
    if (props.splitTouchingRings !== false) {
      // Two segments ending at one vertex continue into the same two outgoing segments. When both
      // lie on one ring, swapping their continuations splits the ring in two (a figure eight of
      // holes, or a shell pinched to its hole); when they lie on different rings the swap would
      // merge them, so those stay. Touch points of planar boundaries do not interleave, so all
      // swaps can be applied at once.
      nodes.push(
        kernel(
          'split-touching',
          [
            read('next', next),
            read('alt', alternatives),
            read('claim', claim),
            read('leader', leader),
            write('swapped', swapped)
          ],
          segments,
          `let chosen = next[nextOffset + index];
  var result = chosen;
  let other = alt[altOffset + index];
  if (chosen != ${none} && other != ${none}) {
    let partner = claim[claimOffset + other];
    if (partner != ${none} && partner != index &&
        alt[altOffset + partner] == chosen && next[nextOffset + partner] == other &&
        leader[leaderOffset + index] != ${none} &&
        leader[leaderOffset + index] == leader[leaderOffset + partner]) {
      result = other;
    }
  }
  swapped[swappedOffset + index] = result;`
        )
      );
      hops = trace(2, swapped, true);
    }

    // 6. Ring and vertex counts per leader, scanned.
    nodes.push(
      kernel(
        'leaders',
        [
          read('leader', leader),
          read('hops', hops),
          write('leaderFlags', leaderFlags),
          write('leaderVertices', leaderVertices)
        ],
        segments,
        `let isLeader = leader[leaderOffset + index] == index;
  leaderFlags[leaderFlagsOffset + index] = select(0u, 1u, isLeader);
  leaderVertices[leaderVerticesOffset + index] = select(0u, hops[hopsOffset + index] + 2u, isLeader);`
      ),
      ...new GPUScan({
        id: `${id}-ring-scan`,
        input: leaderFlags,
        output: ringScan,
        mode: 'inclusive'
      }).getCommandNodes(graph),
      ...new GPUScan({
        id: `${id}-vertex-scan`,
        input: leaderVertices,
        output: vertexScan,
        mode: 'inclusive'
      }).getCommandNodes(graph),
      kernel(
        'ring-starts',
        [
          read('leader', leader),
          read('hops', hops),
          read('ringScan', ringScan),
          read('vertexScan', vertexScan),
          write('ringStarts', ringStarts),
          write('ringLeaders', ringLeaders)
        ],
        segments,
        `if (leader[leaderOffset + index] != index) { return; }
  let ring = ringScan[ringScanOffset + index] - 1u;
  ringStarts[ringStartsOffset + ring] = vertexScan[vertexScanOffset + index] - (hops[hopsOffset + index] + 2u);
  ringLeaders[ringLeadersOffset + ring] = index;`
      ),
      // contract-allow: single-invocation — capacity binary search, at most 32 uint32 bisections.
      kernel(
        'ring-count',
        [
          read('ringScan', ringScan),
          read('vertexScan', vertexScan),
          read('ringStarts', ringStarts),
          write('ringCount', ringCount)
        ],
        1,
        `let total = ringScan[ringScanOffset + ${segments - 1}u];
  let vertices = vertexScan[vertexScanOffset + ${segments - 1}u];
  var low = 0u;
  var high = min(total, ${ringCapacity}u);
  while (low < high) {
    let middle = low + (high - low + 1u) / 2u;
    var end = vertices;
    if (middle < total) { end = ringStarts[ringStartsOffset + middle]; }
    if (end <= ${vertexCapacity}u) { low = middle; } else { high = middle - 1u; }
  }
  ringCount[ringCountOffset] = low;`
      )
    );

    // 7. Place and write vertices.
    nodes.push(
      kernel(
        'place',
        [
          read('leader', leader),
          read('hops', hops),
          read('ringScan', ringScan),
          read('vertexScan', vertexScan),
          read('ringCount', ringCount),
          write('segmentRings', segmentRings),
          write('segmentVertices', segmentVertices),
          write('closeVertices', closeVertices)
        ],
        segments,
        `segmentRings[segmentRingsOffset + index] = ${none};
  segmentVertices[segmentVerticesOffset + index] = ${none};
  closeVertices[closeVerticesOffset + index] = ${none};
  let head = leader[leaderOffset + index];
  if (head == ${none}) { return; }
  let ring = ringScan[ringScanOffset + head] - 1u;
  if (ring >= ringCount[ringCountOffset]) { return; }
  let ringLength = hops[hopsOffset + head] + 1u;
  let start = vertexScan[vertexScanOffset + head] - (ringLength + 1u);
  let position = hops[hopsOffset + head] - hops[hopsOffset + index];
  ${flip ? 'let slot = (ringLength - position) % ringLength;' : 'let slot = position;'}
  segmentRings[segmentRingsOffset + index] = ring;
  segmentVertices[segmentVerticesOffset + index] = start + slot;
  if (position == 0u) { closeVertices[closeVerticesOffset + index] = start + ringLength; }`
      ),
      ...(duplicateOwner
        ? [
            kernel(
              'propagate-duplicate-owners',
              [
                read('duplicateOwner', duplicateOwner),
                read('leader', leader),
                write('segmentRings', segmentRings),
                write('segmentVertices', segmentVertices),
                write('flags', segmentFlags),
                {name: 'counters', view: counters, type: 'atomic<u32>', access: 'read_write'}
              ],
              segments,
              `let owner = duplicateOwner[duplicateOwnerOffset + index];
  let originalFlags = flags[flagsOffset + index];
  // Cancelled rows already contribute to the cancellation counter; leave their flags and open
  // summary untouched so the final counter subtraction accounts for them exactly once.
  if ((originalFlags & ${GPU_SEGMENT_RING_ASSEMBLY_FLAG_CANCELLED}u) != 0u) { return; }
  if (owner != index) {
    segmentRings[segmentRingsOffset + index] = segmentRings[segmentRingsOffset + owner];
    segmentVertices[segmentVerticesOffset + index] = segmentVertices[segmentVerticesOffset + owner];
    flags[flagsOffset + index] = flags[flagsOffset + owner];
    let ownerFlags = flags[flagsOffset + owner];
    // Trace counted each duplicate with no successor as open; the owner leader still identifies a
    // closed cycle even when ring or vertex capacity prevented writing its output slots.
    if (leader[leaderOffset + owner] != ${none} ||
        (ownerFlags & ${GPU_SEGMENT_RING_ASSEMBLY_FLAG_CANCELLED}u) != 0u) {
      atomicSub(&counters[countersOffset], 1u);
    }
    if ((flags[flagsOffset + owner] & ${GPU_SEGMENT_RING_ASSEMBLY_FLAG_TOUCHING}u) != 0u) {
      atomicAdd(&counters[countersOffset + 1u], 1u);
    }
  }`
            )
          ]
        : []),
      kernel(
        'write',
        [
          read('endpoints', props.endpoints, 'f32'),
          ...(duplicateOwner ? [read('duplicateOwner', duplicateOwner)] : []),
          read('segmentVertices', segmentVertices),
          read('closeVertices', closeVertices),
          write('positions', output.positions, 'f32')
        ],
        segments,
        `let slot = segmentVertices[segmentVerticesOffset + index];
  ${duplicateOwner ? `if (duplicateOwner[duplicateOwnerOffset + index] != index) { return; }` : ''}
  if (slot == ${none}) { return; }
  let x = endpoints[endpointsOffset + 4u * index];
  let y = endpoints[endpointsOffset + 4u * index + 1u];
  positions[positionsOffset + 2u * slot] = x;
  positions[positionsOffset + 2u * slot + 1u] = y;
  let close = closeVertices[closeVerticesOffset + index];
  if (close != ${none}) {
    positions[positionsOffset + 2u * close] = x;
    positions[positionsOffset + 2u * close + 1u] = y;
  }`
      ),
      kernel(
        'offsets',
        [
          read('ringStarts', ringStarts),
          read('ringScan', ringScan),
          read('vertexScan', vertexScan),
          read('ringCount', ringCount),
          write('ringOffsets', output.ringOffsets)
        ],
        ringCapacity + 1,
        `let total = ringScan[ringScanOffset + ${segments - 1}u];
  let written = min(index, ringCount[ringCountOffset]);
  var offset = vertexScan[vertexScanOffset + ${segments - 1}u];
  if (written < total) { offset = ringStarts[ringStartsOffset + written]; }
  ringOffsets[ringOffsetsOffset + index] = offset;`
      )
    );

    // 8. Ring statistics: signed shoelace measure (orientation only for longitude/latitude),
    // hole flag, bounds, group, shell assignment.
    const ringAreas = output.ringAreas ?? transient('ring-areas', 'float32', ringCapacity);
    const ringIsHole = output.ringIsHole ?? u32('ring-is-hole', ringCapacity);
    const needsShells = Boolean(output.ringShells ?? output.polygons);
    // Lowest vertex row of every ring (smallest y, then x, then row), for the exact orientation.
    const ringLowestVertices = u32('ring-lowest-vertices', ringCapacity);
    // Ring bounds let the shell search reject a shell with four comparisons instead of walking it.
    const ringBounds = needsShells
      ? createTransientView(graph, `${id}-ring-bounds`, 'float32', ringCapacity * 4)
      : undefined;
    nodes.push(
      // Block hybrid: workgroup w owns rings [RING_LANES * w, RING_LANES * w + RING_LANES). Phase 1:
      // every lane sums its own ring serially when it has at most RING_LANES vertices (bit-identical
      // to the former one-thread-per-ring kernel). Phase 2: the workgroup visits the larger rings of
      // its block in ascending order and splits each across the lanes with a fixed-order tree. Thread
      // count stays the ring count rounded up to RING_LANES, and one huge ring no longer serializes
      // the stage.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-ring-stats`,
        operation: OPERATION,
        variant: 'ring-stats',
        bindings: [
          read('ringOffsets', output.ringOffsets),
          read('positions', output.positions, 'f32'),
          read('ringCount', ringCount),
          write('areas', ringAreas, 'f32'),
          write('lowestVertices', ringLowestVertices),
          ...(ringBounds ? [write('bounds', ringBounds, 'f32')] : [])
        ],
        invocationCount: Math.ceil(ringCapacity / RING_LANES) * RING_LANES,
        workgroupSize: RING_LANES,
        guardIndex: false,
        declarations: `const RING_CAPACITY: u32 = ${ringCapacity}u;
const RING_LANES: u32 = ${RING_LANES}u;
const BOUNDS_LARGE: f32 = 3.0e38;
var<workgroup> largeBits: array<atomic<u32>, 2>;
var<workgroup> largeBitsCopy: array<u32, 2>;
var<workgroup> largeFirst: array<u32, ${RING_LANES}>;
var<workgroup> largeLast: array<u32, ${RING_LANES}>;
var<workgroup> partialSum: array<f32, ${RING_LANES}>;
var<workgroup> partialMinimum: array<vec2f, ${RING_LANES}>;
var<workgroup> partialMaximum: array<vec2f, ${RING_LANES}>;
var<workgroup> partialLowest: array<u32, ${RING_LANES}>;
fn getVertex(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * vertex], positions[positionsOffset + 2u * vertex + 1u]);
}
// Lexicographic (y, x, row) order of vertex rows: the lowest vertex is an extreme point of the ring.
fn isLowerVertex(vertex: u32, other: u32) -> bool {
  let p = getVertex(vertex);
  let q = getVertex(other);
  return p.y < q.y || (p.y == q.y && (p.x < q.x || (p.x == q.x && vertex < other)));
}
fn publishRing(ring: u32, sum: f32, minimum: vec2f, maximum: vec2f, lowest: u32) {
  let area = 0.5 * sum;
  areas[areasOffset + ring] = area;
  lowestVertices[lowestVerticesOffset + ring] = lowest;
  ${ringBounds ? 'bounds[boundsOffset + 4u * ring] = minimum.x; bounds[boundsOffset + 4u * ring + 1u] = minimum.y; bounds[boundsOffset + 4u * ring + 2u] = maximum.x; bounds[boundsOffset + 4u * ring + 3u] = maximum.y;' : ''}
}`,
        body: `let lane = localInvocationIndex;
  let ringBase = workgroupIndex * RING_LANES;
  let ownRing = ringBase + lane;
  // Phase 1: this lane's own ring, serially when small; larger rings are queued for phase 2.
  if (ownRing < RING_CAPACITY) {
    if (ownRing >= ringCount[ringCountOffset]) {
      publishRing(ownRing, 0.0, vec2f(0.0), vec2f(0.0), 0u);
    } else {
      let first = ringOffsets[ringOffsetsOffset + ownRing];
      let last = ringOffsets[ringOffsetsOffset + ownRing + 1u];
      if (last - first <= RING_LANES) {
        let origin = getVertex(first);
        var sum = 0.0;
        var minimum = origin;
        var maximum = origin;
        var lowest = first;
        for (var vertex = first; vertex + 1u < last; vertex++) {
          let p = getVertex(vertex);
          let q = getVertex(vertex + 1u);
          minimum = min(minimum, q);
          maximum = max(maximum, q);
          if (isLowerVertex(vertex + 1u, lowest)) { lowest = vertex + 1u; }
          let a = p - origin;
          let b = q - origin;
          sum += a.x * b.y - b.x * a.y;
        }
        publishRing(ownRing, sum, minimum, maximum, lowest);
      } else {
        largeFirst[lane] = first;
        largeLast[lane] = last;
        atomicOr(&largeBits[lane / 32u], 1u << (lane % 32u));
      }
    }
  }
  workgroupBarrier();
  if (lane == 0u) {
    largeBitsCopy[0] = atomicLoad(&largeBits[0]);
    largeBitsCopy[1] = atomicLoad(&largeBits[1]);
  }
  // Phase 2: every barrier below is in workgroup-uniform control flow (the queue is read through
  // workgroupUniformLoad), and the queued rings are visited in ascending order.
  for (var word = 0u; word < 2u; word++) {
    var pending = workgroupUniformLoad(&largeBitsCopy[word]);
    while (pending != 0u) {
      let slot = word * 32u + firstTrailingBit(pending);
      pending = pending & (pending - 1u);
      let first = workgroupUniformLoad(&largeFirst[slot]);
      let last = workgroupUniformLoad(&largeLast[slot]);
      let origin = getVertex(first);
      var sum = 0.0;
      var minimum = vec2f(BOUNDS_LARGE);
      var maximum = vec2f(-BOUNDS_LARGE);
      var lowest = first;
      for (var vertex = first + lane; vertex + 1u < last; vertex += RING_LANES) {
        let p = getVertex(vertex);
        let q = getVertex(vertex + 1u);
        minimum = min(minimum, q);
        maximum = max(maximum, q);
        if (isLowerVertex(vertex + 1u, lowest)) { lowest = vertex + 1u; }
        let a = p - origin;
        let b = q - origin;
        sum += a.x * b.y - b.x * a.y;
      }
      partialSum[lane] = sum;
      partialMinimum[lane] = minimum;
      partialMaximum[lane] = maximum;
      partialLowest[lane] = lowest;
      workgroupBarrier();
      for (var stride = RING_LANES / 2u; stride > 0u; stride = stride / 2u) {
        if (lane < stride) {
          partialSum[lane] += partialSum[lane + stride];
          partialMinimum[lane] = min(partialMinimum[lane], partialMinimum[lane + stride]);
          partialMaximum[lane] = max(partialMaximum[lane], partialMaximum[lane + stride]);
          if (isLowerVertex(partialLowest[lane + stride], partialLowest[lane])) {
            partialLowest[lane] = partialLowest[lane + stride];
          }
        }
        workgroupBarrier();
      }
      if (lane == 0u) {
        publishRing(ringBase + slot, partialSum[0], min(partialMinimum[0], origin), max(partialMaximum[0], origin), partialLowest[0]);
      }
    }
  }`
      })
    );
    // Rule 1: a ring's orientation is the exact turn at its lowest vertex (an extreme point, so the
    // turn there is the ring's winding for any ring that does not pass it twice), not the sign of
    // the f32 shoelace sum, which a thin or near-degenerate ring can get wrong. Neighbors equal to
    // the lowest vertex are skipped; a zero turn (a spike, or a ring with fewer than three distinct
    // vertices) falls back to the area sign.
    nodes.push(
      kernel(
        'ring-orientation',
        [
          read('ringOffsets', output.ringOffsets),
          read('positions', output.positions, 'f32'),
          read('ringCount', ringCount),
          write('areas', ringAreas, 'f32'),
          read('lowestVertices', ringLowestVertices),
          write('isHole', ringIsHole)
        ],
        ringCapacity,
        `isHole[isHoleOffset + index] = 0u;
  if (index >= ringCount[ringCountOffset]) { return; }
  let first = ringOffsets[ringOffsetsOffset + index];
  let last = ringOffsets[ringOffsetsOffset + index + 1u];
  let area = areas[areasOffset + index];
  var orientation = i32(sign(area));
  // Closed ring: rows [first, last - 1) are its distinct vertex slots, row last - 1 repeats first.
  if (last >= first + 4u) {
    let slots = last - first - 1u;
    var lowest = lowestVertices[lowestVerticesOffset + index];
    if (lowest == last - 1u) { lowest = first; }
    let vertex = getRingVertex(lowest);
    var previous = lowest;
    var next = lowest;
    for (var step = 1u; step < slots; step++) {
      previous = select(previous - 1u, last - 2u, previous == first);
      if (any(getRingVertex(previous) != vertex)) { break; }
    }
    for (var step = 1u; step < slots; step++) {
      next = select(next + 1u, first, next == last - 2u);
      if (any(getRingVertex(next) != vertex)) { break; }
    }
    let turn = exactOrientSigns(
      1u,
      array<vec2f, 4>(getRingVertex(previous), vec2f(0.0), vec2f(0.0), vec2f(0.0)),
      array<vec2f, 4>(vertex, vec2f(0.0), vec2f(0.0), vec2f(0.0)),
      array<vec2f, 4>(getRingVertex(next), vec2f(0.0), vec2f(0.0), vec2f(0.0))
    ).signs.x;
    if (turn != 0) { orientation = turn; }
    // A ring that turns at a lowest vertex it passes once encloses area, whatever the f32 sum
    // says: a face thinner than one f32 step can sum to exactly 0, and a ring of area 0 is
    // read downstream as an empty walk and left out of the polygons. Such a ring gets the smallest
    // normal float with the sign of its exact turn. (A walk out and back along one path has no
    // turn at a tip, and passes any other lowest vertex twice: it stays 0.)
    if (area == 0.0 && turn != 0) {
      var passes = 0u;
      var before = getRingVertex(last - 2u);
      for (var row = first; row + 1u < last; row++) {
        let current = getRingVertex(row);
        if (all(current == vertex) && any(before != vertex)) { passes++; }
        before = current;
      }
      if (passes == 1u) { areas[areasOffset + index] = f32(turn) * 1.17549435e-38; }
    }
  }
  isHole[isHoleOffset + index] = select(0u, 1u, orientation * ${shellSign} < 0);`,
        `${EXACT_ORIENTATION_WGSL}
${EXACT_GEOMETRY_WGSL}
fn getRingVertex(row: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}`
      )
    );
    const ringGroups =
      output.ringGroups ?? (hasGroups ? u32('ring-groups', ringCapacity) : undefined);
    if (ringGroups && props.groups) {
      nodes.push(
        kernel(
          'ring-groups',
          [
            read('ringLeaders', ringLeaders),
            read('ringCount', ringCount),
            ...groupBinding,
            write('ringGroups', ringGroups)
          ],
          ringCapacity,
          `ringGroups[ringGroupsOffset + index] = 0u;
  if (index < ringCount[ringCountOffset]) {
    ringGroups[ringGroupsOffset + index] = groups[groupsOffset + ringLeaders[ringLeadersOffset + index]];
  }`
        )
      );
    }
    const ringShells =
      output.ringShells ?? (output.polygons ? u32('ring-shells', ringCapacity) : undefined);
    if (ringShells) {
      const shellIndex = getRingShellIndexNodes<Parameters>(graph, {
        id: `${id}-shell-index`,
        operation: OPERATION,
        bounds: ringBounds as GraphDataView<'float32'>,
        areas: ringAreas,
        isHole: ringIsHole,
        count: ringCount,
        groups: ringGroups
      });
      const {bvh} = shellIndex;
      nodes.push(
        ...shellIndex.nodes,
        kernel(
          'ring-shells',
          [
            read('ringOffsets', output.ringOffsets),
            read('positions', output.positions, 'f32'),
            read('ringBounds', ringBounds as GraphDataView<'float32'>, 'f32'),
            read('records', shellIndex.records),
            read('nodeMinima', bvh.nodeMinima, 'f32'),
            read('nodeMaxima', bvh.nodeMaxima, 'f32'),
            read('leafIds', bvh.leafIds),
            write('shells', ringShells)
          ],
          ringCapacity,
          `shells[shellsOffset + index] = ${none};
  // The records pass marks rows beyond the live count, freeing the count binding for bounds.
  let kind = records[recordsOffset + RECORD_WORDS * index + 1u];
  if (kind == 2u) { return; }
  if (kind == 0u) {
    shells[shellsOffset + index] = index;
    return;
  }
  let first = ringOffsets[ringOffsetsOffset + index];
  let a = getRingVertex(first);
  let b = getRingVertex(first + 1u);
  // The probe is the symbolic point infinitesimally beside the midpoint of the hole's first edge,
  // on the side away from the hole's interior (left of a clockwise hole), so it lies in the face
  // the hole bounds even when that edge runs along another ring. Every decision below is exact.
  // Holes wind against the shells (exact orientation), so a hole is clockwise exactly when shells
  // are counter-clockwise.
  let probe = ExactProbe(0.5 * (a + b), a, b, ${shellSign > 0 ? 1 : -1});
  var best = ${none};
  var bestArea = 0.0;
  var bestBounds = vec4f(0.0);
  let holeBounds = getRingBounds(index);
  var node = 0u;
  loop {
    let hit = findShell(node, holeBounds);
    let shell = hit.x;
    if (shell == ${none}) { break; }
    node = hit.y;
    if (shell >= RING_CAPACITY || records[recordsOffset + RECORD_WORDS * shell + 1u] != 0u) { continue; }
    if (records[recordsOffset + RECORD_WORDS * shell + 2u] != records[recordsOffset + RECORD_WORDS * index + 2u]) { continue; }
    // A shell around the hole is around the hole's bounding box. The boxes are exact (minima and
    // maxima of stored vertices), where the f32 areas of a thin shell and its hole can come out
    // equal or in the wrong order. A bidirected boundary produces a reverse walk of the same box
    // for the unbounded side: the probe, which lies outside that walk, keeps it from owning itself.
    let shellBounds = getRingBounds(shell);
    if (!boundsContain(shellBounds, holeBounds)) { continue; }
    let area = abs(bitcast<f32>(records[recordsOffset + RECORD_WORDS * shell]));
    if (best != ${none}) {
      // Two shells around one point of a planar boundary are nested, and the inner one has its box
      // inside the other's. Only shells whose boxes are equal or do not nest are ordered by their
      // f32 areas (smaller first, lowest index on ties).
      let insideBest = boundsContain(bestBounds, shellBounds);
      let aroundBest = boundsContain(shellBounds, bestBounds);
      if (aroundBest && !insideBest) { continue; }
      if (insideBest == aroundBest && (area > bestArea || (area == bestArea && shell > best))) { continue; }
    }
    // A probe outside the shell's bounding box (or left of it, where a closed ring crosses the ray
    // an even number of times) cannot be inside: skip the vertex walk.
    let point = probe.point;
    if (point.y < shellBounds.y || point.y > shellBounds.w || point.x < shellBounds.x || point.x > shellBounds.z) { continue; }
    let shellFirst = ringOffsets[ringOffsetsOffset + shell];
    let shellLast = ringOffsets[ringOffsetsOffset + shell + 1u];
    let inside = (getShellWinding(probe, shellFirst, shellLast).x & 1) != 0;
    if (inside) {
      best = shell;
      bestArea = area;
      bestBounds = shellBounds;
    }
  }
  shells[shellsOffset + index] = best;`,
          `const RECORD_WORDS: u32 = ${RING_SHELL_RECORD_WORDS}u;
const RING_CAPACITY: u32 = ${ringCapacity}u;
const INTERNAL_NODE_COUNT: u32 = ${bvh.internalNodeCount}u;
${EXACT_ORIENTATION_WGSL}
${EXACT_GEOMETRY_WGSL}
fn getRingVertex(row: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}
// (minimum x, minimum y, maximum x, maximum y) of a ring's vertices.
fn getRingBounds(ring: u32) -> vec4f {
  return vec4f(
    ringBounds[ringBoundsOffset + 4u * ring], ringBounds[ringBoundsOffset + 4u * ring + 1u],
    ringBounds[ringBoundsOffset + 4u * ring + 2u], ringBounds[ringBoundsOffset + 4u * ring + 3u]);
}
// Whether box 'outer' contains box 'inner' (edges may coincide).
fn boundsContain(outer: vec4f, inner: vec4f) -> bool {
  return outer.x <= inner.x && outer.y <= inner.y && outer.z >= inner.z && outer.w >= inner.w;
}
// Stackless traversal: return a leaf and the cursor immediately after it. Rejected candidates
// can continue the caller's loop without revisiting a node, including the final leaf.
fn findShell(startNode: u32, holeBounds: vec4f) -> vec2u {
  var node = startNode;
  while (node != ${none}) {
    let minimum = vec2f(nodeMinima[nodeMinimaOffset + 2u * node], nodeMinima[nodeMinimaOffset + 2u * node + 1u]);
    let maximum = vec2f(nodeMaxima[nodeMaximaOffset + 2u * node], nodeMaxima[nodeMaximaOffset + 2u * node + 1u]);
    let contains = boundsContain(vec4f(minimum, maximum), holeBounds);
    if (contains && node < INTERNAL_NODE_COUNT) {
      node = 2u * node + 1u;
      continue;
    }
    var shell = ${none};
    if (contains) { shell = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT]; }
    loop {
      if (node == 0u || (node & 1u) == 1u) { break; }
      node = (node - 1u) / 2u;
    }
    node = select(node + 1u, ${none}, node == 0u);
    if (shell != ${none}) { return vec2u(shell, node); }
  }
  return vec2u(${none});
}
${getExactRingWindingWGSL('getShellWinding', 'getRingVertex')}`
        )
      );
    }

    if (output.polygons && ringShells) {
      const {polygons} = output;
      const polygonKeys = u32('polygon-keys', ringCapacity);
      const polygonIds = u32('polygon-ids', ringCapacity);
      const sortedPolygonKeys = u32('sorted-polygon-keys', ringCapacity);
      const sortedPolygonIds = u32('sorted-polygon-ids', ringCapacity);
      const lengths = u32('polygon-lengths', ringCapacity);
      const shellFlags = u32('polygon-shell-flags', ringCapacity);
      const lengthScan = u32('polygon-length-scan', ringCapacity);
      const shellScan = u32('polygon-shell-scan', ringCapacity);
      const polygonStarts = u32('polygon-starts', ringCapacity);
      const validRings = u32('polygon-valid-rings', 1);
      nodes.push(
        kernel(
          'polygon-keys',
          [
            read('ringCount', ringCount),
            read('areas', ringAreas, 'f32'),
            read('isHole', ringIsHole),
            read('shells', ringShells),
            write('keys', polygonKeys),
            write('ids', polygonIds)
          ],
          ringCapacity,
          `ids[idsOffset + index] = index;
  var key = ${none};
  if (index < ringCount[ringCountOffset] && abs(areas[areasOffset + index]) > 0.0 && shells[shellsOffset + index] != ${none}) {
    key = shells[shellsOffset + index] * 2u + isHole[isHoleOffset + index];
  }
  keys[keysOffset + index] = key;`
        ),
        ...new GPUSort({
          id: `${id}-polygon-sort`,
          keys: polygonKeys,
          values: polygonIds,
          outputKeys: sortedPolygonKeys,
          outputValues: sortedPolygonIds
        }).getCommandNodes(graph),
        kernel(
          'polygon-lengths',
          [
            read('ringOffsets', output.ringOffsets),
            read('sortedKeys', sortedPolygonKeys),
            read('sortedIds', sortedPolygonIds),
            write('lengths', lengths),
            write('shellFlags', shellFlags)
          ],
          ringCapacity,
          `let key = sortedKeys[sortedKeysOffset + index];
  let ring = sortedIds[sortedIdsOffset + index];
  let isValid = key != ${none};
  lengths[lengthsOffset + index] = select(0u, ringOffsets[ringOffsetsOffset + ring + 1u] - ringOffsets[ringOffsetsOffset + ring] - 1u, isValid);
  shellFlags[shellFlagsOffset + index] = select(0u, 1u, isValid && (key & 1u) == 0u);`
        ),
        ...new GPUScan({
          id: `${id}-polygon-length-scan`,
          input: lengths,
          output: lengthScan,
          mode: 'inclusive'
        }).getCommandNodes(graph),
        ...new GPUScan({
          id: `${id}-polygon-shell-scan`,
          input: shellFlags,
          output: shellScan,
          mode: 'inclusive'
        }).getCommandNodes(graph),
        kernel(
          'polygon-starts',
          [
            read('shellFlags', shellFlags),
            read('shellScan', shellScan),
            write('polygonStarts', polygonStarts)
          ],
          ringCapacity,
          `if (shellFlags[shellFlagsOffset + index] != 0u) {
    polygonStarts[polygonStartsOffset + shellScan[shellScanOffset + index] - 1u] = index;
  }`
        ),
        ...(polygons.sourceIds && ringGroups
          ? [
              kernel(
                'polygon-groups',
                [
                  read('shellScan', shellScan),
                  read('polygonStarts', polygonStarts),
                  read('sortedIds', sortedPolygonIds),
                  read('ringGroups', ringGroups),
                  write('sourceIds', polygons.sourceIds)
                ],
                ringCapacity,
                `var group = ${none};
  if (index < shellScan[shellScanOffset + ${ringCapacity - 1}u]) {
    group = ringGroups[ringGroupsOffset + sortedIds[sortedIdsOffset + polygonStarts[polygonStartsOffset + index]]];
  }
  sourceIds[sourceIdsOffset + index] = group;`
              )
            ]
          : []),
        // contract-allow: single-invocation — capacity binary search, at most 32 uint32 bisections.
        kernel(
          'polygon-counts',
          [read('sortedKeys', sortedPolygonKeys), write('validRings', validRings)],
          1,
          `var low = 0u;
  var high = ${ringCapacity}u;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (sortedKeys[sortedKeysOffset + middle] == ${none}) { high = middle; } else { low = middle + 1u; }
  }
  validRings[validRingsOffset] = low;`
        ),
        kernel(
          'polygon-offsets',
          [
            read('lengthScan', lengthScan),
            read('shellScan', shellScan),
            read('polygonStarts', polygonStarts),
            read('validRings', validRings),
            write('ringOffsetsOut', polygons.ringOffsets),
            write('polygonOffsetsOut', polygons.polygonOffsets),
            write('featureOffsets', polygons.featureOffsets)
          ],
          ringCapacity + 1,
          `let polygonCount = shellScan[shellScanOffset + ${ringCapacity - 1}u];
  // Identity feature table: one feature per polygon, padded flat with the polygon count.
  featureOffsets[featureOffsetsOffset + index] = min(index, polygonCount);
  var ringOffset = 0u;
  if (index > 0u) { ringOffset = lengthScan[lengthScanOffset + index - 1u]; }
  ringOffsetsOut[ringOffsetsOutOffset + index] = ringOffset;
  var polygonOffset = validRings[validRingsOffset];
  if (index < polygonCount) { polygonOffset = polygonStarts[polygonStartsOffset + index]; }
  polygonOffsetsOut[polygonOffsetsOutOffset + index] = polygonOffset;`
        ),
        kernel(
          'polygon-gather',
          [
            read('sortedIds', sortedPolygonIds),
            read('ringOffsets', output.ringOffsets),
            read('positions', output.positions, 'f32'),
            read('ringOffsetsOut', polygons.ringOffsets),
            write('positionsOut', polygons.positions, 'f32')
          ],
          vertexCapacity,
          `if (index >= ringOffsetsOut[ringOffsetsOutOffset + ${ringCapacity}u]) { return; }
  var low = 0u;
  var high = ${ringCapacity}u;
  while (low + 1u < high) {
    let middle = low + (high - low) / 2u;
    if (ringOffsetsOut[ringOffsetsOutOffset + middle] <= index) { low = middle; } else { high = middle; }
  }
  let ring = sortedIds[sortedIdsOffset + low];
  let source = ringOffsets[ringOffsetsOffset + ring] + (index - ringOffsetsOut[ringOffsetsOutOffset + low]);
  positionsOut[positionsOutOffset + 2u * index] = positions[positionsOffset + 2u * source];
  positionsOut[positionsOutOffset + 2u * index + 1u] = positions[positionsOffset + 2u * source + 1u];`
        )
      );
    }

    // 9. Publish scalars.
    nodes.push(
      kernel(
        'publish',
        [
          read('ringScan', ringScan),
          read('ringCount', ringCount),
          read('counters', counters),
          write('outCount', output.status.count),
          write('outOverflow', output.status.overflow),
          ...(output.status.requiredCount ? [write('outTotal', output.status.requiredCount)] : [])
        ],
        1,
        `let total = ringScan[ringScanOffset + ${segments - 1}u];
  let written = ringCount[ringCountOffset];
  outCount[outCountOffset] = written;
  outOverflow[outOverflowOffset] = select(0u, 1u, written < total || counters[countersOffset + 3u] != 0u);
  ${output.status.requiredCount ? 'outTotal[outTotalOffset] = total;' : ''}`
      )
    );
    if (output.status.openSegmentCount || output.status.touchingSegmentCount) {
      nodes.push(
        kernel(
          'publish-counters',
          [
            read('counters', counters),
            ...(output.status.openSegmentCount
              ? [write('outOpen', output.status.openSegmentCount)]
              : []),
            ...(output.status.touchingSegmentCount
              ? [write('outTouching', output.status.touchingSegmentCount)]
              : [])
          ],
          1,
          `${output.status.openSegmentCount ? 'outOpen[outOpenOffset] = counters[countersOffset] - counters[countersOffset + 2u];' : ''}
  ${output.status.touchingSegmentCount ? 'outTouching[outTouchingOffset] = counters[countersOffset + 1u];' : ''}`
        )
      );
    }
    return nodes;

    function createJumpNode(mode: 'min' | 'sum', round: string, source: number) {
      const target = 1 - source;
      return createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${mode}-jump-${round}`,
        operation: OPERATION,
        variant: `${mode}-jump`,
        bindings: [
          read('pointerIn', pointerSets[source]),
          read('valueIn', valueSets[source]),
          write('pointerOut', pointerSets[target]),
          write('valueOut', valueSets[target])
        ],
        invocationCount: segments,
        body: `let successor = pointerIn[pointerInOffset + index];
  var value = valueIn[valueInOffset + index];
  var pointer = successor;
  if (successor != ${none}) {
    value = ${
      mode === 'min'
        ? 'min(value, valueIn[valueInOffset + successor])'
        : 'value + valueIn[valueInOffset + successor]'
    };
    pointer = pointerIn[pointerInOffset + successor];
  }
  pointerOut[pointerOutOffset + index] = pointer;
  valueOut[valueOutOffset + index] = value;`
      });
    }
  }
}

/** Construction-time caller-output access plan; private scratch never aliases these views. */
function validateRingOutputDispatches(
  id: string,
  output: GPUSegmentRingAssemblyOutput,
  pairedOpposites: boolean
): void {
  validateDisjointWritableViews(id, Object.fromEntries(getOutputViewEntries(output)), {
    separateDispatches: true
  });
  const validateDispatch = (
    inputs: readonly (GraphDataView | undefined)[],
    writable: Readonly<Record<string, GraphDataView | undefined>>
  ) => {
    validateDisjointWritableViews(id, writable);
    // A disjoint byte range is still a conflicting read-only/write usage of one buffer.
    validateGraphOutputsDisjointFromInputs(id, Object.values(writable), inputs);
  };
  validateDispatch([], {
    segmentRings: output.segmentRings,
    segmentVertices: output.segmentVertices,
    segmentFlags: pairedOpposites ? output.segmentFlags : undefined
  });
  validateDispatch([output.segmentVertices], {positions: output.positions});
  validateDispatch([output.positions, output.ringOffsets], {
    ringAreas: output.ringAreas,
    ringIsHole: output.ringIsHole
  });
  validateDispatch([output.positions, output.ringOffsets], {ringShells: output.ringShells});
  if (output.polygons) {
    validateDispatch([output.ringGroups], {sourceIds: output.polygons.sourceIds});
    validateDispatch([], {
      'polygons.ringOffsets': output.polygons.ringOffsets,
      'polygons.polygonOffsets': output.polygons.polygonOffsets,
      'polygons.featureOffsets': output.polygons.featureOffsets
    });
    validateDispatch([output.positions, output.ringOffsets, output.polygons.ringOffsets], {
      'polygons.positions': output.polygons.positions
    });
  }
  validateDispatch([], {
    'status.count': output.status.count,
    'status.overflow': output.status.overflow,
    'status.requiredCount': output.status.requiredCount
  });
  validateDispatch([], {
    'status.openSegmentCount': output.status.openSegmentCount,
    'status.touchingSegmentCount': output.status.touchingSegmentCount
  });
}

function getOutputViewEntries(output: GPUSegmentRingAssemblyOutput) {
  const {kind: _kind, ...polygonViews} = output.polygons ?? {};
  const {status, polygons: _polygons, ...views} = output;
  return [
    ...Object.entries(views).map(([name, view]) => [`output.${name}`, view] as const),
    ...Object.entries(polygonViews).map(
      ([name, view]) => [`output.polygons.${name}`, view] as const
    ),
    ...Object.entries(status).map(([name, view]) => [`output.status.${name}`, view] as const)
  ];
}

function getOutputViews(output: GPUSegmentRingAssemblyOutput) {
  return getOutputViewEntries(output).map(([, view]) => view);
}
