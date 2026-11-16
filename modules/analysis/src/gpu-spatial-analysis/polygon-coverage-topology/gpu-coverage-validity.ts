// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validateSinglePackedView,
  validatePowerOfTwo
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUPolygonGeometryFields} from '../../analysis-core/contracts/index';
import {getGPUSegmentIntersectionMaximumVertexCount} from '../segment-intersection/index';
import {getSegmentBVHNodes} from '../segment-intersection/segment-bvh';
import {createSegmentTableNode} from '../segment-intersection/segment-table';
import {
  getSegmentTableAccessorsWGSL,
  SEGMENT_PREDICATES_WGSL
} from '../segment-intersection/segment-intersection-wgsl';
import {createRingOrientationNode} from './coverage-topology-kernels';
import {GPU_COVERAGE_VALIDITY_PARAMETER_LENGTH} from './coverage-validity-parameters';
import {NODE_OVERLAPS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {EXACT_GEOMETRY_WGSL} from '../../analysis-core/kernels/exact-geometry-wgsl';
import {getNextPowerOfTwo} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateGraphChunkBindingRanges,
  validateDisjointWritableViews
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  validateGeometryReadWriteViews,
  validateGeometryScratchByteLengths
} from '../geometry-edit/geometry-storage-validation';
import {validateSegmentBVHScratch} from '../segment-intersection/segment-storage-validation';
import {validateGeographicDistributionViews} from '../geographic-distribution/geographic-distribution-validation';

import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';

const OPERATION = 'GPUCoverageValidity';
/** Vertex count from which `spatialSort` defaults to on, as `GPUSegmentIntersection`. */
const SPATIAL_SORT_MINIMUM_SLOTS = 1024;

/**
 * Bits of `output.segmentFlags` of {@link GPUCoverageValidity}. A segment is valid when its flags
 * are `0`.
 */
export const GPU_COVERAGE_VALIDITY_FLAG = {
  /**
   * The segment overlaps a segment of another polygon along a span that is not identical (the two
   * polygons disagree about the vertices of a shared boundary), or an endpoint of one lies in the
   * interior of the other (a T-junction without a matching vertex).
   */
  unmatched: 1,
  /** The segment properly crosses a segment of another polygon. */
  crossing: 2,
  /**
   * The segment runs side by side with a segment of another polygon, within `gapWidth` of it, so
   * the two polygons are separated by a narrow gap. "Side by side" is the GEOS rule (see
   * {@link GPUCoverageValidity}): each segment projects onto the other over more than `gapWidth`,
   * and the ends of the two projected parts are within `gapWidth` of each other. Segments that
   * are staggered, side by side over only part of their length, are flagged.
   */
  gap: 4,
  /**
   * The segment lies inside another polygon without touching its boundary, or equals a segment
   * of another polygon whose interior is on the same side (a duplicated or overlapping polygon).
   */
  overlap: 8,
  /** An exact orientation test could not be certified for a candidate pair (non-finite input). */
  uncertain: 16
} as const;

/** Caller-owned outputs of {@link GPUCoverageValidity}. */
export type GPUCoverageValidityOutput = {
  /**
   * `GPU_COVERAGE_VALIDITY_FLAG` bits for the segment that starts at each input vertex (a ring edge
   * `i -> next(i)`, one row per input vertex, so rows align with `positions`). Rows of rings with
   * fewer than three vertices and of degenerate or non-finite segments are `0`.
   */
  segmentFlags: GraphDataView<'uint32'>;
  /** Optional invalid segment count per feature, `featureOffsets.length - 1` rows. */
  featureInvalidCounts?: GraphDataView<'uint32'>;
  /** Optional one-row flag: 1 when no segment is invalid (Shapely `coverage_is_valid`), else 0. */
  isValid?: GraphDataView<'uint32'>;
  /** Status scalars. */
  status?: {
    /** Optional one-row total of invalid segments. */
    invalidCount?: GraphDataView<'uint32'>;
  };
};

/**
 * Properties for {@link GPUCoverageValidity}.
 *
 * Per-frame (no recompile): the contents of `parameters` (the gap width) and of the input buffers
 * as long as lengths stay the same. Compile-time: view lengths, `spatialSort`, `leafCapacity` and
 * which optional outputs are present.
 */
export type GPUCoverageValidityProps = GPUPolygonGeometryFields & {
  /** Prefix for generated node and transient IDs. Defaults to `'coverage-validity'`. */
  id?: string;
  /**
   * Coordinate space and metric of the vertices. Omitted means native planar coordinates.
   * `longitude-latitude` throws at construction: coverage topology is evaluated in the plane of the
   * coordinates (edges are straight lines, gaps are measured in coordinate units). Project to a
   * planar frame first.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Layout notes. `positions` are flattened ring vertices with finite coordinates. Rows are
   * features; rings close implicitly. The first ring of a polygon is its shell, the others are
   * holes. Winding is free: orientation is read from the exact turn at an extreme vertex. `ringOffsets`,
   * `polygonOffsets` and `featureOffsets` each carry a terminal entry. Segments of one feature are
   * never compared with each other.
   */
  /**
   * Per-frame packed float32 view of at least {@link GPU_COVERAGE_VALIDITY_PARAMETER_LENGTH}
   * elements written with `getGPUCoverageValidityParameterValues`: the gap width.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Compile-time. Reorders segments along a Morton curve before the BVH build; identical results,
   * different cost, as `GPUSegmentIntersection.spatialSort`. Defaults to `true` from 1024 vertices.
   */
  spatialSort?: boolean;
  /** Power-of-two BVH leaf slots. Defaults to the next power of two of the vertex count. */
  leafCapacity?: number;
  /** Caller-owned outputs. */
  output: GPUCoverageValidityOutput;
};

/**
 * Validates a polygon coverage: Shapely `coverage_is_valid` and `coverage_invalid_edges`
 * (GEOS `CoverageValidator`), reported per ring segment.
 *
 * **Definition.** Every polygon is a member of one coverage (the polygon rows of the input). A
 * segment of a polygon is checked against the segments of every *other* polygon whose bounds are
 * within `gapWidth`:
 * - an identical segment (either direction) is a *matched* edge. It is valid unless both polygons
 *   have their interior on the same side of it (duplicate or overlapping polygons): `overlap`. A
 *   matched edge is not tested against any other segment, as in GEOS;
 * - a proper crossing is `crossing`;
 * - a collinear overlap of different segments, or a touch where an endpoint of either segment lies
 *   in the interior of the other (a T-junction without a matching vertex), is `unmatched`;
 * - when `gapWidth > 0`, two segments that neither cross nor touch in the ways above are `gap` when
 *   they run side by side. Each segment is projected onto the other and clipped to it; the pair is
 *   a gap when both clipped projections are longer than `gapWidth` (strictly) and their matching
 *   ends are within `gapWidth` of each other (inclusive). The segments need not cover each other:
 *   `(0, 0)-(10, 0)` beside `(5, 0.0625)-(15, 0.0625)` is a gap at `gapWidth` 0.125, over
 *   `x` in `[5, 10]`. Every other pair, such as a segment that passes another by no more than
 *   `gapWidth`, a vertex pointing at an edge or edges meeting at a vertex, is valid;
 * - an otherwise clean, unmatched segment whose midpoint lies inside another polygon (an island or
 *   spike inside a neighbor) is `overlap`, found with a winding number over the other polygons'
 *   segments.
 *
 * Segments sharing only a vertex are valid. Segments of the same polygon are never compared with
 * each other: self-intersections within one polygon are the business of `GPUGeometryValidity`.
 * Orientation is normalized from the exact turn at each ring's lexicographically smallest vertex,
 * so mixed clockwise and counter-clockwise rings are fine, including slivers whose float32 area is
 * zero.
 *
 * **Parity with GEOS.** The gap rule is the GEOS one (`coverage/InvalidSegmentDetector`,
 * `isNearlyParallel`; JTS has the same rule), evaluated in float32 relative to the start of the
 * tested segment. With Shapely 2.2.0 on GEOS 3.14.1 the invalid segment sets of
 * `shapely.coverage_invalid_edges` and this contributor are equal on the curated scenes of the
 * specs and on every case of `test/oracles/fixtures/coverage-validity-gaps.json`: staggered
 * rectangles, disjoint polygon pairs and jittered triangulations with gaps and overlaps, near the
 * origin and at projected-metre coordinates, at gap widths from 0 to 0.75 (0 to 24 m). A pair
 * whose distance or projected length equals `gapWidth` to within float32 rounding can fall on
 * either side; GEOS decides the same ties in binary64. Until 2026-11-11 the rule was this
 * contributor's own (both ends of the shorter segment within `gapWidth` of the longer) and did not
 * flag staggered segments. A matched edge lying inside a *third* polygon is not flagged (that
 * polygon's own segments are).
 *
 * **Exactness.** Every orientation sign is exact for the f32 vertices: the pair tests use the shared
 * segment predicates and the winding number of the interior probe uses the shared exact side probe
 * (`analysis-core/kernels/exact-geometry-wgsl.ts`): the symbolic point beside the exact midpoint of
 * the segment, classified against each neighbor edge from the exact signs of the segment's two
 * endpoints. No rounded sample point is formed, so a segment of a polygon thinner than one f32 step
 * is not read as lying inside the neighbor across its other edge. Only when the *line* of a
 * neighbor edge passes between the segment's endpoints does the rounded midpoint decide the side.
 * A pair with no decidable orientation (non-finite coordinate) sets the `uncertain` flag; the
 * validity checker never silently decides it.
 *
 * **Cost.** One BVH built over all segments; each segment probes it with its bounds inflated by
 * `gapWidth`. Output is one `uint32` row per vertex: nothing is capacity-bounded, nothing
 * overflows. Counts use integer atomics, so results are deterministic.
 */
export class GPUCoverageValidity implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCoverageValidityProps;
  /** Resolved BVH leaf capacity. */
  readonly leafCapacity: number;
  /** Whether segments are Morton sorted before the BVH build. */
  readonly spatialSort: boolean;
  /** Resolved spatial context: native planar coordinates. */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUCoverageValidityProps) {
    const id = props.id ?? 'coverage-validity';
    this.id = id;
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(id, props.spatialContext);
    validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 1) {
      throw new Error(`${id} positions must hold at least one vertex`);
    }
    validatePackedUint32View(props.ringOffsets, `${id} ringOffsets`);
    validatePackedUint32View(props.polygonOffsets, `${id} polygonOffsets`);
    validatePackedUint32View(props.featureOffsets, `${id} featureOffsets`);
    for (const [name, view] of [
      ['ringOffsets', props.ringOffsets],
      ['polygonOffsets', props.polygonOffsets],
      ['featureOffsets', props.featureOffsets]
    ] as const) {
      if (view.length < 2) {
        throw new Error(`${id} ${name} must hold at least two entries`);
      }
    }
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_COVERAGE_VALIDITY_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_COVERAGE_VALIDITY_PARAMETER_LENGTH} float32 values`
      );
    }
    this.spatialSort = props.spatialSort ?? props.positions.length >= SPATIAL_SORT_MINIMUM_SLOTS;
    this.leafCapacity = props.leafCapacity ?? getNextPowerOfTwo(props.positions.length);
    validatePowerOfTwo(id, 'leafCapacity', this.leafCapacity, props.positions.length);
    const {output} = props;
    validatePackedUint32View(output.segmentFlags, `${id} output.segmentFlags`);
    if (output.segmentFlags.length !== props.positions.length) {
      throw new Error(`${id} output.segmentFlags length must equal positions length`);
    }
    if (output.featureInvalidCounts) {
      validatePackedUint32View(output.featureInvalidCounts, `${id} output.featureInvalidCounts`);
      if (output.featureInvalidCounts.length !== props.featureOffsets.length - 1) {
        throw new Error(`${id} output.featureInvalidCounts length must equal the feature count`);
      }
    }
    for (const [name, view] of [
      ['status.invalidCount', output.status?.invalidCount],
      ['isValid', output.isValid]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} output.${name}`);
        if (view.length < 1) {
          throw new Error(`${id} output.${name} must hold one uint32`);
        }
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.segmentFlags,
        output.featureInvalidCounts,
        output.status?.invalidCount,
        output.isValid
      ],
      [
        props.positions,
        props.ringOffsets,
        props.polygonOffsets,
        props.featureOffsets,
        props.parameters
      ]
    );
    validateGeographicDistributionViews(
      id,
      {},
      {
        segmentFlags: output.segmentFlags,
        featureInvalidCounts: output.featureInvalidCounts,
        invalidCount: output.status?.invalidCount,
        isValid: output.isValid
      },
      []
    );
    validateDisjointWritableViews(id, {
      featureInvalidCounts: output.featureInvalidCounts,
      invalidCount: output.status?.invalidCount
    });
    validateGeometryReadWriteViews(
      id,
      [output.segmentFlags],
      [output.featureInvalidCounts, output.status?.invalidCount]
    );
    validateGeometryReadWriteViews(id, [output.status?.invalidCount], [output.isValid]);
  }

  /** Returns the coverage-validity nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {positions, ringOffsets, polygonOffsets, featureOffsets, parameters, output} = props;
    validateGraphChunkBindingRanges(graph, id, {
      positions,
      ringOffsets,
      polygonOffsets,
      featureOffsets,
      parameters,
      segmentFlags: output.segmentFlags,
      featureInvalidCounts: output.featureInvalidCounts,
      invalidCount: output.status?.invalidCount,
      isValid: output.isValid
    });
    validateGeometryScratchByteLengths(graph, id, {
      segmentTable: positions.length * 32,
      ringOrientation: (ringOffsets.length - 1) * 4,
      featureCounts: (featureOffsets.length - 1) * 4,
      scalarCount: 4
    });
    validateSegmentBVHScratch(graph, id, positions.length, this.leafCapacity, this.spatialSort);
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      ringOffsets,
      polygonOffsets,
      featureOffsets,
      parameters,
      output.segmentFlags,
      output.featureInvalidCounts,
      output.status?.invalidCount,
      output.isValid
    ]);
    // The segment table holds one row per vertex.
    validateMaximumCount(
      id,
      'positions rows',
      props.positions.length,
      getGPUSegmentIntersectionMaximumVertexCount(graph.device)
    );
    const vertexCount = positions.length;
    const ringCount = ringOffsets.length - 1;
    const featureCount = featureOffsets.length - 1;
    const view = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);

    const ringFlip = view('ring-flip', ringCount);
    const table = view('segment-table', vertexCount * 8);
    const minima = createTransientView(graph, `${id}-bounds-minima`, 'float32x2', vertexCount);
    const maxima = createTransientView(graph, `${id}-bounds-maxima`, 'float32x2', vertexCount);
    const nodes: GPUCommandNode<Parameters>[] = [
      createRingOrientationNode<Parameters>(graph, {
        id: `${id}-ring-orientation`,
        operation: OPERATION,
        positions,
        ringOffsets,
        polygonOffsets,
        ringFlip
      })
    ];
    nodes.push(
      createSegmentTableNode<Parameters>(graph, {
        id: `${id}-segment-table`,
        operation: OPERATION,
        geometry: {kind: 'polygons', positions, ringOffsets, polygonOffsets, featureOffsets},
        table,
        minima,
        maxima
      })
    );
    const {bvh, nodes: bvhNodes} = getSegmentBVHNodes(graph, {
      id,
      operation: OPERATION,
      minima,
      maxima,
      leafCapacity: this.leafCapacity,
      spatialSort: this.spatialSort
    });
    nodes.push(...bvhNodes);

    const traverse = (queryMinimum: string, queryMaximum: string, leafBody: string) => `{
    let queryMinimum = ${queryMinimum};
    let queryMaximum = ${queryMaximum};
    var node = 0u;
    loop {
      if (nodeOverlaps(node, queryMinimum, queryMaximum)) {
        if (node < INTERNAL_NODE_COUNT) {
          node = node * 2u + 1u;
          continue;
        }
        let other = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT];
        if (other < SEGMENT_COUNT && segmentValid(other) && segmentFeature(other) != polygon) {
          ${leafBody}
        }
      }
      loop {
        if (node == 0u || (node & 1u) == 1u) { break; }
        node = (node - 1u) / 2u;
      }
      if (node == 0u) { break; }
      node = node + 1u;
    }
  }`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-probe`,
        operation: OPERATION,
        variant: 'probe',
        bindings: [
          {name: 'segmentTable', view: table, type: 'u32', access: 'read'},
          {name: 'nodeMinima', view: bvh.nodeMinima, type: 'f32', access: 'read'},
          {name: 'nodeMaxima', view: bvh.nodeMaxima, type: 'f32', access: 'read'},
          {name: 'leafIds', view: bvh.leafIds, type: 'u32', access: 'read'},
          {name: 'ringFlip', view: ringFlip, type: 'u32', access: 'read'},
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
          {name: 'segmentFlags', view: output.segmentFlags, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        declarations: `${SEGMENT_PREDICATES_WGSL}
${getSegmentTableAccessorsWGSL('segment')}
const SEGMENT_COUNT: u32 = ${vertexCount}u;
const INTERNAL_NODE_COUNT: u32 = ${bvh.internalNodeCount}u;
const FLAG_UNMATCHED: u32 = ${GPU_COVERAGE_VALIDITY_FLAG.unmatched}u;
const FLAG_CROSSING: u32 = ${GPU_COVERAGE_VALIDITY_FLAG.crossing}u;
const FLAG_GAP: u32 = ${GPU_COVERAGE_VALIDITY_FLAG.gap}u;
const FLAG_OVERLAP: u32 = ${GPU_COVERAGE_VALIDITY_FLAG.overlap}u;
const FLAG_UNCERTAIN: u32 = ${GPU_COVERAGE_VALIDITY_FLAG.uncertain}u;
const MATCHED_BIT: u32 = 0x80000000u;
const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;

${NODE_OVERLAPS_WGSL}
${EXACT_GEOMETRY_WGSL}

fn isFlipped(row: u32) -> bool {
  return ringFlip[ringFlipOffset + segmentRing(row)] != 0u;
}

fn samePoint(first: vec2f, second: vec2f) -> bool {
  return first.x == second.x && first.y == second.y;
}

// Length of the part of segment (start, end) that projects onto segment (otherStart, otherEnd), times
// the length of the latter, so that no division is needed and ties with gapWidth stay exact. Zero
// or negative when the one does not project onto the other at all.
fn projectedLengthScaled(start: vec2f, end: vec2f, otherStart: vec2f, otherEnd: vec2f) -> f32 {
  let direction = otherEnd - otherStart;
  let lengthSquared = dot(direction, direction);
  let first = dot(start - otherStart, direction);
  let second = dot(end - otherStart, direction);
  return min(max(first, second), lengthSquared) - max(min(first, second), 0.0);
}

// The point of segment (start, end) nearest to the projection of point on its line: GEOS
// LineSegment::project of one point, clamped to the segment. An end is returned as it is.
fn projectOntoSegment(point: vec2f, start: vec2f, end: vec2f) -> vec2f {
  let direction = end - start;
  let lengthSquared = dot(direction, direction);
  let along = dot(point - start, direction);
  if (along <= 0.0) { return start; }
  if (along >= lengthSquared) { return end; }
  return start + direction * (along / lengthSquared);
}

// The GEOS gap rule (coverage/InvalidSegmentDetector::isNearlyParallel): each segment is projected
// onto the other and clipped to it. The pair is a gap when both clipped projections are longer than
// gapWidth and their matching ends lie within gapWidth of each other. (a, b) is the tested segment
// and (c, d) the neighbor's, both in input order, as GEOS reads them. Everything is formed relative
// to a, so that the f32 step of large projected coordinates does not enter the distances.
fn isNearlyParallel(a: vec2f, b: vec2f, c: vec2f, d: vec2f, gapWidth: f32) -> bool {
  let origin = vec2f(0.0);
  let localB = b - a;
  let localC = c - a;
  let localD = d - a;
  if (projectedLengthScaled(localC, localD, origin, localB) <= gapWidth * length(localB) ||
      projectedLengthScaled(origin, localB, localC, localD) <= gapWidth * distance(localC, localD)) {
    return false;
  }
  // Projection of (c, d) on (a, b), and of (a, b) on (c, d).
  let onRowStart = projectOntoSegment(localC, origin, localB);
  let onRowEnd = projectOntoSegment(localD, origin, localB);
  var onOtherStart = projectOntoSegment(origin, localC, localD);
  var onOtherEnd = projectOntoSegment(localB, localC, localD);
  if (distance(onRowStart, onOtherEnd) < distance(onRowStart, onOtherStart)) {
    let swap = onOtherStart;
    onOtherStart = onOtherEnd;
    onOtherEnd = swap;
  }
  return distance(onRowStart, onOtherStart) <= gapWidth &&
    distance(onRowEnd, onOtherEnd) <= gapWidth;
}

// Flags of segment row against segment other (another polygon). Bit 31 reports a matched edge.
fn comparePair(row: u32, other: u32, gapWidth: f32) -> u32 {
  let a = segmentStart(row);
  let b = segmentEnd(row);
  let c = segmentStart(other);
  let d = segmentEnd(other);
  let sameDirection = samePoint(a, c) && samePoint(b, d);
  if (sameDirection || (samePoint(a, d) && samePoint(b, c))) {
    let effectivelySame = sameDirection != (isFlipped(row) != isFlipped(other));
    return select(MATCHED_BIT, FLAG_OVERLAP, effectivelySame);
  }
  let hit = classifySegments(a, b, c, d, false);
  if (hit.kind == KIND_UNCERTAIN) { return FLAG_UNCERTAIN; }
  if (hit.kind == KIND_PROPER) { return FLAG_CROSSING; }
  if (hit.kind == KIND_OVERLAP) { return FLAG_UNMATCHED; }
  if (hit.kind == KIND_TOUCH) {
    // A shared vertex is fine; an endpoint of either segment inside the other is a T-junction.
    let onSegmentRow = samePoint(hit.point, a) || samePoint(hit.point, b);
    let onSegmentOther = samePoint(hit.point, c) || samePoint(hit.point, d);
    if (!(onSegmentRow && onSegmentOther)) { return FLAG_UNMATCHED; }
  }
  if (gapWidth > 0.0 && isNearlyParallel(a, b, c, d, gapWidth)) { return FLAG_GAP; }
  return 0u;
}

// Contribution of segment other to the winding number of the probe (effective interior on the
// left). The probe is the symbolic point beside the exact midpoint of the tested segment (the
// shared side probe), so a neighbor's edge closer to the segment than one f32 step is still on
// the right side of it. An uncertain sign (non-finite input) counts as no crossing.
fn windingContribution(other: u32, probe: ExactProbe) -> i32 {
  var start = segmentStart(other);
  var end = segmentEnd(other);
  if (isFlipped(other)) {
    let swap = start;
    start = end;
    end = swap;
  }
  return exactProbeCrossing(probe, start, end).x;
}`,
        body: `var flags = 0u;
  if (segmentValid(index)) {
    let gapWidth = max(parameters[parametersOffset], 0.0);
    let polygon = segmentFeature(index);
    let a = segmentStart(index);
    let b = segmentEnd(index);
    var matched = false;
    ${traverse(
      'min(a, b) - vec2f(gapWidth)',
      'max(a, b) + vec2f(gapWidth)',
      `let result = comparePair(index, other, gapWidth);
          matched = matched || (result & MATCHED_BIT) != 0u;
          flags = flags | (result & ~MATCHED_BIT);`
    )}
    // A segment matched by an identical segment of another polygon is not tested against the rest,
    // as in GEOS; only a same-side duplicate (FLAG_OVERLAP) stays flagged.
    if (matched) {
      flags = flags & FLAG_OVERLAP;
    }
    if (flags == 0u && !matched) {
      // The rounded midpoint only selects BVH candidates: every edge the ray from the exact
      // midpoint can cross has that height and a larger x inside its bounds.
      let midpoint = a * 0.5 + b * 0.5;
      let probe = ExactProbe(midpoint, a, b, 1);
      var winding = 0;
      ${traverse(
        'vec2f(midpoint.x, midpoint.y)',
        'vec2f(FLOAT32_MAXIMUM, midpoint.y)',
        'winding = winding + windingContribution(other, probe);'
      )}
      if (winding > 0) { flags = FLAG_OVERLAP; }
    }
  }
  segmentFlags[segmentFlagsOffset + index] = flags;`
      })
    );

    // Counts per polygon and in total, with integer atomics.
    const needsCounts = Boolean(
      output.featureInvalidCounts || output.status?.invalidCount || output.isValid
    );
    if (needsCounts) {
      const polygonCounts =
        output.featureInvalidCounts ?? view('feature-invalid-counts', featureCount);
      const total = output.status?.invalidCount ?? view('invalid-count', 1);
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-clear-polygon-counts`,
          operation: OPERATION,
          view: polygonCounts,
          type: 'u32',
          value: '0u'
        }),
        createFillNode<Parameters>(graph, {
          id: `${id}-clear-total`,
          operation: OPERATION,
          view: total,
          type: 'u32',
          value: '0u'
        })
      );
      const countBindings: WGSLKernelBinding[] = [
        {name: 'segmentTable', view: table, type: 'u32', access: 'read'},
        {name: 'segmentFlags', view: output.segmentFlags, type: 'u32', access: 'read'},
        {name: 'polygonCounts', view: polygonCounts, type: 'atomic<u32>', access: 'read_write'},
        {name: 'total', view: total, type: 'atomic<u32>', access: 'read_write'}
      ];
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-count`,
          operation: OPERATION,
          variant: 'count',
          bindings: countBindings,
          invocationCount: vertexCount,
          declarations: `const SEGMENT_NONE: u32 = 0xffffffffu;
${getSegmentTableAccessorsWGSL('segment')}`,
          body: `if (segmentFlags[segmentFlagsOffset + index] != 0u && segmentValid(index)) {
    atomicAdd(&polygonCounts[polygonCountsOffset + segmentFeature(index)], 1u);
    atomicAdd(&total[totalOffset], 1u);
  }`
        })
      );
      if (output.isValid) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-finalize`,
            operation: OPERATION,
            variant: 'finalize',
            bindings: [
              {name: 'total', view: total, type: 'u32', access: 'read'},
              {name: 'isValid', view: output.isValid, type: 'u32', access: 'read_write'}
            ],
            invocationCount: 1,
            body: 'isValid[isValidOffset] = select(0u, 1u, total[totalOffset] == 0u);'
          })
        );
      }
    }
    return nodes;
  }
}
