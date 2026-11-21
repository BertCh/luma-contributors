// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getEmptyGeometryNodes} from '../group-geometry/empty-geometry-nodes';

import {validateGeometryAuxiliaryOutput} from '../group-geometry/geometry-output-destinations';

import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';

import {
  captureGeometryPartitionProvenance,
  getGeometryPartitionProvenanceNodes
} from '../group-geometry/partition-provenance';

import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateRequirement,
  validateGraphOutputsDisjointFromInputs,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUPolygonGeometryFields} from '../../analysis-core/contracts/index';
import {PHILOX_PURPOSES, PHILOX_WGSL} from '../../analysis-core/kernels/philox-random';
import {EXACT_ORIENTATION_WGSL} from '../../analysis-core/kernels/exact-orientation-wgsl';
import {TRIANGLE_CIRCUMCIRCLE_WGSL} from '../shared/triangle-circumcircle-wgsl';
import {validateGroupConvexHullScratch} from '../group-geometry/gpu-group-convex-hull';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH} from '../geographic-distribution/index';
import {GPUGroupGeometry} from '../group-geometry/gpu-group-geometry';
import {GPUGroupConvexHull} from '../group-geometry/index';

const OPERATION = 'GPUMinimumBounds';
const MAXIMUM_ROWS = 2 ** 24 - 1;
// Detailed coastlines and buffered shapes commonly have hulls of several hundred vertices; the cap
// only bounds the work of one thread per group (the hull storage is sized by the vertex count).
const DEFAULT_MAXIMUM_HULL_VERTICES = 2048;
/** Hulls up to this size scan every pair (exact original behavior); larger ones use calipers. */
const CALIPER_MINIMUM_HULL_VERTICES = 16;

/**
 * Caller-owned outputs of {@link GPUMinimumBounds}. Every output is optional and rewritten on every
 * encoding. Empty groups hold NaN shape rows. A hull exceeding `maximumHullVertices` suppresses
 * hull-dependent shapes; sourceBounds still encloses every finite source position.
 */
export type GPUMinimumBoundsOutput = {
  /** Optional one-word partition completion; geometry remains available for provisional labels. */
  labelsComplete?: GraphDataView<'uint32'>;
  /**
   * Corners of the minimum-area rotated rectangle, 4 rows per group, counter-clockwise. The first
   * two corners lie on the side parallel to the hull edge that defined the rectangle.
   */
  rectangleCorners?: GraphDataView<'float32x2'>;
  /**
   * `[width, height, angle, area]` per group of the rotated rectangle. `width` is the extent along
   * the defining hull edge, `height` the extent across it, `angle` the direction of that edge in
   * radians in `(-pi, pi]`.
   */
  rectangleSizes?: GraphDataView<'float32x4'>;
  /** Minimum bounding circle `[centerX, centerY, radius]` per group. */
  circles?: GraphDataView<'float32x3'>;
  /** Longest line (maximum distance pair) `[x0, y0, x1, y1]` per group. */
  longestLines?: GraphDataView<'float32x4'>;
  /** Length of {@link GPUMinimumBoundsOutput.longestLines} per group (the diameter). */
  diameters?: GraphDataView<'float32'>;
  /** Axis-aligned finite source envelope, independent of hull loss. */
  sourceBounds?: GraphDataView<'float32x4'>;
  /** Per group: the global lattice moved a finite source coordinate; hull shapes may lose detail. */
  precisionLimited?: GraphDataView<'uint32'>;
  /** True convex hull vertex count per group, before the `maximumHullVertices` cap. */
  hullSizes?: GraphDataView<'uint32'>;
  /**
   * Completeness scalars. There is no `count`: every column has one row per group (feature),
   * never a variable number of rows.
   */
  status: {
    /** One row: 1 when some hull exceeded `maximumHullVertices`, otherwise 0. Its shape rows are NaN. */
    overflow: GraphDataView<'uint32'>;
    /** Optional one row identifying the per-group hull limit; the same 0/1 value as `overflow`. */
    groupOverflow?: GraphDataView<'uint32'>;
  };
};

/** Props shared by both inputs of {@link GPUMinimumBounds}. */
type GPUMinimumBoundsSharedProps = {
  /** Optional solver completion word; omission denotes ordinary categorical labels. */
  labelsComplete?: GraphDataView<'uint32'>;
  /** Prefix for generated node and transient IDs. Defaults to `'minimum-bounds'`. */
  id?: string;
  /**
   * Planar only. Spatial context of `positions`; omitted means native planar coordinates.
   * Longitude/latitude (or any other non-planar) contexts throw at construction: project
   * positions to a planar frame first, for example with `createGPULocalFrame` and
   * `GPULocalFrameProjection`, and pass the frame's context so its `frame` identity is kept.
   * Measures both distance and area, so neither local frame is exact: use an
   * azimuthal-equidistant frame for study areas small enough that its area distortion is
   * negligible, or a projected CRS.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Largest convex hull handled per group (at least 3, default 2048). Larger hulls are flagged in
   * `overflow` and produce NaN. Hulls above 16 vertices use rotating calipers for the rectangle and
   * longest line (linear per group); smaller hulls scan all pairs.
   */
  maximumHullVertices?: number;
  /** Caller-owned outputs. */
  output: GPUMinimumBoundsOutput;
};

/**
 * Properties for {@link GPUMinimumBounds}.
 *
 * Give either polygons (the four-level layout of {@link GPUPolygonGeometryFields}, one group per
 * feature; vertices outside every ring are excluded) or `positions` with explicit `labels` and
 * `groupCount` (any geometry: points, lines or vertices). All input contents are per-frame; view
 * lengths, `groupCount`, `maximumHullVertices` and the set of requested outputs need a new graph.
 */
export type GPUMinimumBoundsProps = GPUMinimumBoundsSharedProps &
  (
    | (GPUPolygonGeometryFields & {labels?: undefined; groupCount?: undefined})
    | {
        /** Planar points or vertices (fewer than `2^24` rows). Non-finite rows are excluded. */
        positions: GraphDataView<'float32x2'>;
        /** A group label per position row. Labels `>= groupCount` are excluded. */
        labels: GraphDataView<'uint32'>;
        /** Number of groups. */
        groupCount: number;
      }
  );

/** Narrows polygon inputs even when the caller compiles without strict null checks. */
function isPolygonMinimumBoundsInput(
  props: GPUMinimumBoundsProps
): props is Extract<GPUMinimumBoundsProps, GPUPolygonGeometryFields> {
  return !props.labels;
}

/**
 * Minimum bounding shapes of every feature (or label group): the minimum-area rotated rectangle
 * (`oriented_envelope`, `minimum_rotated_rectangle`), the minimum bounding circle
 * (`minimum_bounding_circle`, `minimum_bounding_radius`), the longest line / diameter (Sedona
 * `ST_LongestLine`, `ST_MaxDistance`) and the axis-aligned envelope.
 *
 * ## Semantics
 *
 * - The lattice convex hull of every group comes from {@link GPUGroupConvexHull} (vertex to
 *   feature labels are derived from the offsets). Rectangle, circle and line use hull vertices
 *   in group-relative coordinates. Source envelopes independently reduce every finite source
 *   position, including positions lost by lattice quantization or a hull cap.
 * - Rectangle: for every hull edge the bounding rectangle aligned with it is measured (rotating
 *   calipers: the three extreme vertices only move forward as the edge turns, so a hull costs O(h)
 *   instead of O(h^2); hulls of at most 16 vertices scan every vertex), and the
 *   smallest area wins. An area within `1e-6` relative of the best so far does not replace it, so
 *   equal areas (squares) resolve to the lowest edge index, counting hull edges counter-clockwise
 *   from the smallest `(x, y)` vertex. Rectangles agree with GEOS in area, not in corner order.
 *   A single point gives four equal corners; collinear points a zero-height rectangle.
 * - Circle: all distance squares use one power-of-two-scaled group frame; center and radius are
 *   restored only after the search, so representable radii survive squared-distance overflow and
 *   underflow. Welzl's algorithm uses the hull vertices in a fixed hash-shuffled order (deterministic,
 *   expected linear), then the radius is recomputed as the largest hull distance so the circle
 *   always covers the hull. Degenerate groups give radius 0.
 * - Longest line: the pair of hull vertices with the largest distance; ties go to the lowest
 *   vertex pair in hull order. Large hulls test only the antipodal candidates found by a rotating
 *   caliper (a farthest pair is always antipodal), O(h) instead of O(h^2).
 * - Envelope: from all finite source positions, independently of the hull lattice and hull limits.
 * Optional precisionLimited marks groups whose source coordinates moved on the global lattice;
 * their hull sizes, diameters, rectangles and circles may lose detail.
 *
 * Output is fixed size per group. Hulls over `maximumHullVertices` are flagged in `overflow`.
 *
 * The contributor never compiles, encodes, submits or reads back.
 *
 * Planar only: `spatialContext` defaults to native planar coordinates and a longitude/latitude
 * context throws at construction. Project first, for example with `createGPULocalFrame` and
 * `GPULocalFrameProjection`, and pass the frame's context so its `frame` identity is kept.
 */
export class GPUMinimumBounds implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Solver provenance independent of hull/medoid limits; absent means categorical labels. */
  readonly partitionProvenance: Readonly<{labelsComplete?: GraphDataView<'uint32'>}>;
  /** Validated properties. */
  readonly props: GPUMinimumBoundsProps;
  /** Resolved spatial context (planar). */
  readonly spatialContext: GPUSpatialContext;
  /** Number of groups (features, or `groupCount`). */
  readonly groupCount: number;
  /** Resolved hull vertex cap. */
  readonly maximumHullVertices: number;

  constructor(props: GPUMinimumBoundsProps) {
    const captured = snapshotConstructionFields(
      props as GPUMinimumBoundsProps & Partial<GPUPolygonGeometryFields>,
      [
        'id',
        'spatialContext',
        'positions',
        'labels',
        'labelsComplete',
        'groupCount',
        'ringOffsets',
        'polygonOffsets',
        'featureOffsets',
        'maximumHullVertices',
        'output'
      ]
    );
    validateRequirement(
      captured.id ?? 'minimum-bounds',
      !('bounds' in captured.output),
      'output.bounds was removed; use output.sourceBounds'
    );
    const capturedOutput = snapshotConstructionFields(captured.output, [
      'rectangleCorners',
      'rectangleSizes',
      'circles',
      'longestLines',
      'diameters',
      'sourceBounds',
      'labelsComplete',
      'precisionLimited',
      'hullSizes',
      'status'
    ]);
    props = Object.freeze({
      ...captured,
      ...(captured.spatialContext
        ? {spatialContext: snapshotConstructionSpatialContext(captured.spatialContext)}
        : {}),
      output: Object.freeze({
        ...capturedOutput,
        ...(capturedOutput.status
          ? {
              status: snapshotConstructionFields(capturedOutput.status, [
                'overflow',
                'groupOverflow'
              ])
            }
          : {})
      })
    });
    this.id = props.id ?? 'minimum-bounds';
    this.props = props;
    this.partitionProvenance = captureGeometryPartitionProvenance(
      this.id,
      props.labelsComplete,
      props.output.labelsComplete
    );
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);
    const {id} = this;
    const {output} = props;
    validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);
    const rows = props.positions.length;

    if (rows > MAXIMUM_ROWS) {
      throw new Error(`${id} supports fewer than 2^24 rows`);
    }
    if (!isPolygonMinimumBoundsInput(props)) {
      validateSinglePackedView(props.labels, ['uint32'], `${id} labels`);
      if (props.labels.length !== rows) {
        throw new Error(`${id} labels length must equal positions length`);
      }
      validateCapacity(id, 'groupCount', props.groupCount ?? 0, 0, MAXIMUM_ROWS);
      this.groupCount = props.groupCount;
    } else {
      for (const name of ['ringOffsets', 'polygonOffsets', 'featureOffsets'] as const) {
        validateSinglePackedView(props[name], ['uint32'], `${id} ${name}`);
        if (props[name].length < 1) {
          throw new Error(`${id} ${name} must contain a terminal row`);
        }
      }
      this.groupCount = props.featureOffsets.length - 1;
    }
    this.maximumHullVertices = props.maximumHullVertices ?? DEFAULT_MAXIMUM_HULL_VERTICES;
    validateCapacity(id, 'maximumHullVertices', this.maximumHullVertices, 3);
    const groups = this.groupCount;
    const checks = [
      ['labelsComplete', output.labelsComplete, ['uint32'], 1],
      ['rectangleCorners', output.rectangleCorners, ['float32x2'], groups * 4],
      ['rectangleSizes', output.rectangleSizes, ['float32x4'], groups],
      ['circles', output.circles, ['float32x3'], groups],
      ['longestLines', output.longestLines, ['float32x4'], groups],
      ['diameters', output.diameters, ['float32'], groups],
      ['precisionLimited', output.precisionLimited, ['uint32'], groups],
      ['sourceBounds', output.sourceBounds, ['float32x4'], groups],
      ['hullSizes', output.hullSizes, ['uint32'], groups],
      ['status.overflow', output.status.overflow, ['uint32'], 1],
      ['status.groupOverflow', output.status.groupOverflow, ['uint32'], 1]
    ] as const;
    for (const [name, view, formats, length] of checks) {
      if (!view) {
        continue;
      }
      validateSinglePackedView(view, formats, `${id} output.${name}`);
      validateRequirement(
        id,
        view.length >= length,
        `output.${name} must hold at least ${length} rows`
      );
    }
    if (
      !output.rectangleCorners &&
      !output.rectangleSizes &&
      !output.circles &&
      !output.longestLines &&
      !output.diameters &&
      !output.sourceBounds
    ) {
      throw new Error(`${id} needs at least one shape output`);
    }
    for (const name of ['labelsComplete', 'precisionLimited'] as const)
      validateGeometryAuxiliaryOutput(
        id,
        name,
        output[name],
        Object.fromEntries(
          checks
            .filter(([otherName]) => otherName !== name)
            .map(([otherName, view]) => [otherName, view])
        )
      );
    for (const [name, scalar] of Object.entries(output.status)) {
      if (!scalar) continue;
      for (const [otherName, other] of checks) {
        if (!other || otherName === `status.${name}`) continue;
        const overlaps =
          scalar.buffer === other.buffer &&
          scalar.byteOffset < other.byteOffset + other.length * other.rowByteLength &&
          other.byteOffset < scalar.byteOffset + scalar.length * scalar.rowByteLength;
        validateRequirement(
          id,
          !overlaps,
          `output.status.${name} must not overlap output.${otherName}`
        );
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      checks.map(([, view]) => view),
      [
        props.positions,
        'ringOffsets' in props ? props.ringOffsets : undefined,
        'polygonOffsets' in props ? props.polygonOffsets : undefined,
        'featureOffsets' in props ? props.featureOffsets : undefined,
        props.labels,
        props.labelsComplete
      ]
    );
  }

  /** Returns label, hull and one node per requested shape in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, groupCount} = this;
    const {output, positions} = props;
    const {status, ...columns} = output;
    const polygons = 'ringOffsets' in props ? props : undefined;
    validateGraphChunkBindingRanges(graph, id, {
      positions,
      labels: props.labels,
      ringOffsets: props.labels ? undefined : polygons?.ringOffsets,
      polygonOffsets: props.labels ? undefined : polygons?.polygonOffsets,
      featureOffsets: props.labels ? undefined : polygons?.featureOffsets,
      ...columns,
      ...Object.fromEntries(Object.entries(status).map(([name, view]) => [`status.${name}`, view]))
    });
    if (positions.length === 0)
      return [
        ...getGeometryPartitionProvenanceNodes(
          graph,
          id,
          props.labelsComplete,
          output.labelsComplete
        ),
        ...getEmptyGeometryNodes(
          graph,
          id,
          OPERATION,
          Object.fromEntries([
            ...Object.entries(columns)
              .filter(([name]) => name !== 'labelsComplete')
              .map(([name, view]) => [
                name,
                {view, value: ['hullSizes', 'precisionLimited'].includes(name) ? 0 : NaN}
              ]),
            ...Object.entries(status).map(([name, view]) => [name, {view, value: 0}])
          ])
        )
      ];
    validateGroupConvexHullScratch(graph.device, id, positions.length, groupCount);
    const rows = positions.length;
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const read = (name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding => ({
      name,
      view,
      type,
      access: 'read'
    });
    const write = (name: string, view: GraphDataView, type: 'u32' | 'f32' = 'f32') =>
      ({name, view, type, access: 'read_write'}) as WGSLKernelBinding;
    const kernel = (
      step: string,
      invocationCount: number,
      bindings: WGSLKernelBinding[],
      body: string,
      declarations = ''
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${step}`,
        operation: OPERATION,
        variant: step,
        bindings,
        invocationCount,
        declarations,
        body
      });
    const nodes: GPUCommandNode<Parameters>[] = getGeometryPartitionProvenanceNodes(
      graph,
      id,
      props.labelsComplete,
      output.labelsComplete
    );

    // 1. Vertex to group labels.
    let labels = props.labels;
    if (!labels) {
      // Without labels the props are the polygon layout.
      const {ringOffsets, polygonOffsets, featureOffsets} = props as GPUPolygonGeometryFields;
      labels = createTransientView(graph, `${id}-labels`, 'uint32', rows);
      // The last entry that starts at or before `value`, within the covered range of `offsets`.
      const ownerSearch = (offsets: string, count: number, value: string, owner: string) =>
        `if (${value} >= ${offsets}[${offsets}Offset] && ${value} < ${offsets}[${offsets}Offset + ${count}u]) {
    var ${owner}Low = 0u;
    var ${owner}High = ${count - 1}u;
    while (${owner}Low < ${owner}High) {
      let middle = (${owner}Low + ${owner}High + 1u) >> 1u;
      if (${offsets}[${offsets}Offset + middle] <= ${value}) {
        ${owner}Low = middle;
      } else {
        ${owner}High = middle - 1u;
      }
    }
    ${owner} = ${owner}Low;`;
      nodes.push(
        kernel(
          'labels',
          rows,
          [
            read('ringOffsets', ringOffsets, 'u32'),
            read('polygonOffsets', polygonOffsets, 'u32'),
            read('featureOffsets', featureOffsets, 'u32'),
            write('labels', labels, 'u32')
          ],
          // Vertex row -> ring -> polygon -> feature; any miss leaves the row unlabeled.
          `var ring = 0xffffffffu;
  var polygon = 0xffffffffu;
  var feature = 0xffffffffu;
  ${ownerSearch('ringOffsets', ringOffsets.length - 1, 'index', 'ring')}
    ${ownerSearch('polygonOffsets', polygonOffsets.length - 1, 'ring', 'polygon')}
      ${ownerSearch('featureOffsets', featureOffsets.length - 1, 'polygon', 'feature')}
      }
    }
  }
  labels[labelsOffset + index] = feature;`
        )
      );
    }

    // 2. Per-group convex hulls.
    const hullIndices = u32('hull-indices', rows);
    const hullPositions = createTransientView(graph, `${id}-hull-positions`, 'float32x2', rows);
    const hullOffsets = u32('hull-offsets', groupCount + 1);
    const hullCounts = u32('hull-counts', groupCount);
    nodes.push(
      ...new GPUGroupConvexHull({
        id: `${id}-hull`,
        positions,
        labels,
        groupCount,
        maximumVerticesPerGroup: this.maximumHullVertices,
        vertexCapacity: rows,
        labelsComplete: props.labelsComplete,
        output: {
          vertexIndices: hullIndices,
          vertexPositions: hullPositions,
          offsets: hullOffsets,
          counts: hullCounts,
          sizes: output.hullSizes,
          precisionLimited: output.precisionLimited,
          status: {overflow: output.status.overflow, groupOverflow: output.status.groupOverflow}
        }
      }).getCommandNodes(graph)
    );

    // 3. Shapes, one thread per group over its hull in anchor-relative coordinates.
    const hullBindings = () => [
      read('offsets', hullOffsets, 'u32'),
      read('counts', hullCounts, 'u32'),
      read('hull', hullPositions, 'f32')
    ];
    const prelude = `let count = counts[countsOffset + index];
  let begin = offsets[offsetsOffset + index];
  let nan = bitcast<f32>(0x7fc00000u | (index & 0u));
  var anchor = vec2<f32>(0.0, 0.0);
  if (count > 0u) {
    anchor = vec2<f32>(hull[hullOffset + begin * 2u], hull[hullOffset + begin * 2u + 1u]);
  }`;
    const declarations = `const CALIPER_MINIMUM: u32 = ${CALIPER_MINIMUM_HULL_VERTICES + 1}u;
fn loadPoint(slot: u32, anchor: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(hull[hullOffset + slot * 2u], hull[hullOffset + slot * 2u + 1u]) - anchor;
}
// Rotating-caliper step: the hull is strictly convex, so a linear function along it rises to one
// maximum and falls again. Starting on the rising side, walk forward while one of the next two
// vertices is higher (two, to ride over float noise at the plateau of a perpendicular edge).
fn advanceMaximum(start: u32, begin: u32, count: u32, anchor: vec2<f32>, axis: vec2<f32>) -> u32 {
  var slot = start;
  var value = dot(loadPoint(begin + slot, anchor), axis);
  for (var step = 0u; step < count; step++) {
    var moved = false;
    for (var lookahead = 1u; lookahead <= 2u; lookahead++) {
      let candidate = (slot + lookahead) % count;
      let candidateValue = dot(loadPoint(begin + candidate, anchor), axis);
      if (candidateValue > value) {
        slot = candidate;
        value = candidateValue;
        moved = true;
        break;
      }
    }
    if (!moved) { break; }
  }
  return slot;
}
fn findMaximum(begin: u32, count: u32, anchor: vec2<f32>, axis: vec2<f32>) -> u32 {
  var best = 0u;
  var bestValue = dot(loadPoint(begin, anchor), axis);
  for (var slot = 1u; slot < count; slot++) {
    let value = dot(loadPoint(begin + slot, anchor), axis);
    if (value > bestValue) {
      best = slot;
      bestValue = value;
    }
  }
  return best;
}`;
    if (output.rectangleCorners || output.rectangleSizes) {
      const bindings = hullBindings();
      if (output.rectangleCorners) {
        bindings.push(write('corners', output.rectangleCorners));
      }
      if (output.rectangleSizes) {
        bindings.push(write('sizes', output.rectangleSizes));
      }
      nodes.push(
        kernel(
          'rectangle',
          groupCount,
          bindings,
          `${prelude}
  var bestArea = 0.0;
  var bestEdge = 0xffffffffu;
  var bestDirection = vec2<f32>(1.0, 0.0);
  var bestRange = vec4<f32>(0.0);
  var rightSlot = 0u;
  var topSlot = 0u;
  var leftSlot = 0u;
  for (var edge = 0u; edge < count; edge++) {
    let from0 = loadPoint(begin + edge, anchor);
    let to0 = loadPoint(begin + (edge + 1u) % count, anchor);
    let delta = to0 - from0;
    let edgeLength = sqrt(dot(delta, delta));
    var direction = vec2<f32>(1.0, 0.0);
    if (edgeLength > 0.0) {
      direction = delta / edgeLength;
    }
    let normal = vec2<f32>(-direction.y, direction.x);
    var range = vec4<f32>(3.0e38, 3.0e38, -3.0e38, -3.0e38);
    if (count < CALIPER_MINIMUM) {
      for (var vertex = 0u; vertex < count; vertex++) {
        let point = loadPoint(begin + vertex, anchor);
        let along = dot(point, direction);
        let across = dot(point, normal);
        range = vec4<f32>(min(range.x, along), min(range.y, across), max(range.z, along), max(range.w, across));
      }
    } else {
      // Calipers: the edge's own side is the minimum across; the other three extremes turn
      // counter-clockwise with the edge, so each pointer only moves forward.
      if (edge == 0u) {
        rightSlot = findMaximum(begin, count, anchor, direction);
        topSlot = findMaximum(begin, count, anchor, normal);
        leftSlot = findMaximum(begin, count, anchor, -direction);
      } else {
        rightSlot = advanceMaximum(rightSlot, begin, count, anchor, direction);
        topSlot = advanceMaximum(topSlot, begin, count, anchor, normal);
        leftSlot = advanceMaximum(leftSlot, begin, count, anchor, -direction);
      }
      range = vec4<f32>(
        dot(loadPoint(begin + leftSlot, anchor), direction),
        min(dot(from0, normal), dot(to0, normal)),
        dot(loadPoint(begin + rightSlot, anchor), direction),
        dot(loadPoint(begin + topSlot, anchor), normal)
      );
    }
    let area = (range.z - range.x) * (range.w - range.y);
    if (bestEdge == 0xffffffffu || area < bestArea * (1.0 - 1.0e-6)) {
      bestArea = area;
      bestEdge = edge;
      bestDirection = direction;
      bestRange = range;
    }
  }
  let normal = vec2<f32>(-bestDirection.y, bestDirection.x);
  let width = bestRange.z - bestRange.x;
  let height = bestRange.w - bestRange.y;
  var angle = 0.0;
  if (bestDirection.x == 0.0) {
    angle = select(-1.5707964, 1.5707964, bestDirection.y > 0.0);
  } else {
    angle = atan2(bestDirection.y, bestDirection.x);
  }
  ${
    output.rectangleCorners
      ? `for (var corner = 0u; corner < 4u; corner++) {
    let alongValue = select(bestRange.x, bestRange.z, corner == 1u || corner == 2u);
    let acrossValue = select(bestRange.y, bestRange.w, corner >= 2u);
    let point = anchor + bestDirection * alongValue + normal * acrossValue;
    corners[cornersOffset + (index * 4u + corner) * 2u] = select(nan, point.x, count > 0u);
    corners[cornersOffset + (index * 4u + corner) * 2u + 1u] = select(nan, point.y, count > 0u);
  }`
      : ''
  }
  ${
    output.rectangleSizes
      ? `let isValid = count > 0u;
  sizes[sizesOffset + index * 4u] = select(nan, width, isValid);
  sizes[sizesOffset + index * 4u + 1u] = select(nan, height, isValid);
  sizes[sizesOffset + index * 4u + 2u] = select(nan, angle, isValid);
  sizes[sizesOffset + index * 4u + 3u] = select(nan, width * height, isValid);`
      : ''
  }`,
          declarations
        )
      );
    }

    if (output.circles) {
      const order = u32('circle-order', rows);
      nodes.push(
        kernel(
          'circle',
          groupCount,
          [...hullBindings(), write('order', order, 'u32'), write('circles', output.circles)],
          `${prelude}
  if (count == 0u) {
    circles[circlesOffset + index * 3u] = nan;
    circles[circlesOffset + index * 3u + 1u] = nan;
    circles[circlesOffset + index * 3u + 2u] = nan;
    return;
  }
  // All containment, diameter and covering-radius squares use this shared group frame.
  // A representable source radius can have a square below f32's minimum or above its maximum.
  // Scaled edge descriptors also avoid overflowing source-minus-anchor for opposite extrema.
  var circleExponent = -1000i;
  for (var slot = 0u; slot < count; slot++) {
    let point = vec2f(hull[hullOffset + (begin + slot) * 2u], hull[hullOffset + (begin + slot) * 2u + 1u]);
    let edge = getScaledTriangleEdge(anchor, point);
    if (any(edge.direction != vec2f(0.0))) {
      circleExponent = max(circleExponent, edge.exponent);
    }
    order[orderOffset + begin + slot] = slot;
  }
  if (circleExponent == -1000i) { circleExponent = 0i; }
  // Deterministic Fisher-Yates shuffle: one Philox block per (group, slot). The circle does not
  // depend on the order in exact arithmetic; the shuffle only bounds the expected running time.
  for (var slot = count - 1u; slot > 0u; slot--) {
    let pick = philox4x32(vec4<u32>(index, slot, 0u, 0u), vec2<u32>(0u, CIRCLE_SHUFFLE_PURPOSE)).x % (slot + 1u);
    let held = order[orderOffset + begin + slot];
    order[orderOffset + begin + slot] = order[orderOffset + begin + pick];
    order[orderOffset + begin + pick] = held;
  }
  var center = loadCirclePoint(begin + order[orderOffset + begin], anchor, circleExponent);
  var radiusSquared = 0.0;
  for (var i = 1u; i < count; i++) {
    let pointI = loadCirclePoint(begin + order[orderOffset + begin + i], anchor, circleExponent);
    if (isOutside(pointI, center, radiusSquared)) {
      center = pointI;
      radiusSquared = 0.0;
      for (var j = 0u; j < i; j++) {
        let pointJ = loadCirclePoint(begin + order[orderOffset + begin + j], anchor, circleExponent);
        if (isOutside(pointJ, center, radiusSquared)) {
          center = 0.5 * (pointI + pointJ);
          radiusSquared = dot(pointI - center, pointI - center);
          for (var k = 0u; k < j; k++) {
            let pointK = loadCirclePoint(begin + order[orderOffset + begin + k], anchor, circleExponent);
            if (isOutside(pointK, center, radiusSquared)) {
              center = getCircumcenter(pointI, pointJ, pointK);
              radiusSquared = max(
                dot(pointI - center, pointI - center),
                max(dot(pointJ - center, pointJ - center), dot(pointK - center, pointK - center))
              );
            }
          }
        }
      }
    }
  }
  // Recompute the radius from the hull so the circle covers every vertex.
  var coverSquared = 0.0;
  for (var slot = 0u; slot < count; slot++) {
    let delta = loadCirclePoint(begin + slot, anchor, circleExponent) - center;
    coverSquared = max(coverSquared, dot(delta, delta));
  }
  circles[circlesOffset + index * 3u] = sumTriangleCoordinate(anchor.x, center.x, circleExponent, 0.0, 0i);
  circles[circlesOffset + index * 3u + 1u] = sumTriangleCoordinate(anchor.y, center.y, circleExponent, 0.0, 0i);
  circles[circlesOffset + index * 3u + 2u] = scaleTriangleComponent(sqrt(coverSquared), circleExponent);`,
          `${declarations}
${PHILOX_WGSL}
${EXACT_ORIENTATION_WGSL}
${TRIANGLE_CIRCUMCIRCLE_WGSL}
const CIRCLE_SHUFFLE_PURPOSE: u32 = ${PHILOX_PURPOSES.minimumBoundsCircleShuffle.id}u;
fn loadCirclePoint(slot: u32, anchor: vec2f, exponent: i32) -> vec2f {
  let point = vec2f(hull[hullOffset + slot * 2u], hull[hullOffset + slot * 2u + 1u]);
  let edge = getScaledTriangleEdge(anchor, point);
  let alignedExponent = edge.exponent - exponent;
  // Tiny aligned terms may round to zero in the shared frame; do not pass an invalid ldexp power.
  return vec2f(getTriangleAlignedTerm(edge.direction.x, alignedExponent), getTriangleAlignedTerm(edge.direction.y, alignedExponent));
}
fn isOutside(point: vec2<f32>, center: vec2<f32>, radiusSquared: f32) -> bool {
  let delta = point - center;
  return dot(delta, delta) > radiusSquared * (1.0 + 4.0e-6);
}
fn getDiametralCenter(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> vec2<f32> {
  let ab = dot(a - b, a - b);
  let ac = dot(a - c, a - c);
  let bc = dot(b - c, b - c);
  if (ab >= ac && ab >= bc) {
    return 0.5 * (a + b);
  }
  if (ac >= bc) {
    return 0.5 * (a + c);
  }
  return 0.5 * (b + c);
}
// The near-collinear test and the circumcentre use power-of-two scaled edges (the shared
// circumcircle of the hull and Voronoi kernels), so neither overflows above extent 2^32 nor
// underflows below 2^-42.
fn getCircumcenter(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> vec2<f32> {
  let ab = getScaledTriangleEdge(a, b).direction;
  let ac = getScaledTriangleEdge(a, c).direction;
  let determinant = 2.0 * (ab.x * ac.y - ab.y * ac.x);
  let lengths = sqrt(dot(ab, ab) * dot(ac, ac));
  if (abs(determinant) <= 2.0e-7 * lengths) {
    return getDiametralCenter(a, b, c);
  }
  let circle = getTriangleCircumcircle(a, b, c);
  if (circle.orientation == 0 || circle.orientation == 2) {
    return getDiametralCenter(a, b, c);
  }
  return circle.center;
}`
        )
      );
    }

    if (output.longestLines || output.diameters) {
      const bindings = hullBindings();
      if (output.longestLines) {
        bindings.push(write('lines', output.longestLines));
      }
      if (output.diameters) {
        bindings.push(write('diameters', output.diameters));
      }
      nodes.push(
        kernel(
          'longest-line',
          groupCount,
          bindings,
          `${prelude}
  bestSquared = -1.0;
  bestFrom = 0u;
  bestTo = 0u;
  if (count < CALIPER_MINIMUM) {
    for (var i = 0u; i < count; i++) {
      for (var j = i; j < count; j++) {
        considerPair(begin, anchor, i, j);
      }
    }
  } else {
    // Rotating calipers: a farthest pair has parallel supporting lines, so it is found among the
    // vertices opposite each edge. Pairs are compared exactly as the full scan does.
    var opposite = 0u;
    for (var edge = 0u; edge < count; edge++) {
      let next = (edge + 1u) % count;
      let edgeStart = loadPoint(begin + edge, anchor);
      let edgeVector = loadPoint(begin + next, anchor) - edgeStart;
      let normal = vec2<f32>(-edgeVector.y, edgeVector.x);
      if (edge == 0u) {
        opposite = findMaximum(begin, count, anchor, normal);
      } else {
        opposite = advanceMaximum(opposite, begin, count, anchor, normal);
      }
      for (var offset = count - 1u; offset <= count + 1u; offset++) {
        let candidate = (opposite + offset) % count;
        considerPair(begin, anchor, edge, candidate);
        considerPair(begin, anchor, next, candidate);
      }
    }
  }
  let isValid = count > 0u;
  ${
    output.longestLines
      ? `let startPoint = loadPoint(begin + bestFrom, anchor) + anchor;
  let endPoint = loadPoint(begin + bestTo, anchor) + anchor;
  lines[linesOffset + index * 4u] = select(nan, startPoint.x, isValid);
  lines[linesOffset + index * 4u + 1u] = select(nan, startPoint.y, isValid);
  lines[linesOffset + index * 4u + 2u] = select(nan, endPoint.x, isValid);
  lines[linesOffset + index * 4u + 3u] = select(nan, endPoint.y, isValid);`
      : ''
  }
  ${
    output.diameters
      ? 'diameters[diametersOffset + index] = select(nan, sqrt(max(bestSquared, 0.0)), isValid);'
      : ''
  }`,
          `${declarations}
var<private> bestSquared: f32;
var<private> bestFrom: u32;
var<private> bestTo: u32;
// Keeps the farthest pair; equal distances go to the lowest (first, second) pair in hull order.
fn considerPair(begin: u32, anchor: vec2<f32>, one: u32, other: u32) {
  let first = min(one, other);
  let second = max(one, other);
  let delta = loadPoint(begin + second, anchor) - loadPoint(begin + first, anchor);
  let distanceSquared = dot(delta, delta);
  if (distanceSquared > bestSquared ||
      (distanceSquared == bestSquared && (first < bestFrom || (first == bestFrom && second < bestTo)))) {
    bestSquared = distanceSquared;
    bestFrom = first;
    bestTo = second;
  }
}`
        )
      );
    }

    if (output.sourceBounds) {
      nodes.push(
        ...new GPUGroupGeometry({
          id: `${id}-source-bounds`,
          positions,
          labels,
          groupCount,
          parameters: createTransientView(
            graph,
            `${id}-unused-distribution-parameters`,
            'float32',
            GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH
          ),
          output: {bounds: output.sourceBounds}
        }).getCommandNodes(graph)
      );
    }
    return nodes;
  }
}
