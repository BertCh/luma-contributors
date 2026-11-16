// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import type {
  GPULineGeometryFields,
  GPUPolygonGeometryFields
} from '../../analysis-core/contracts/index';
import {
  validateCapacity,
  validateFiniteNumber,
  validateCommandNodeBudget,
  validateGraphOutputsDisjointFromInputs,
  validateGPUStorageBindingByteLength,
  validateMaximumCount,
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {EXACT_ORIENTATION_WGSL} from '../../analysis-core/kernels/exact-orientation-wgsl';
import {EXACT_GEOMETRY_WGSL} from '../../analysis-core/kernels/exact-geometry-wgsl';
import {validateOutlineOutputPasses} from './validate-outline-output-passes';
import {FIND_LINE_WGSL} from './gpu-outline-geometry';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import type {GPULineGeometryOutput} from '../line-segmentize/line-segmentize-types';
import {
  createLineGeometryCountNode,
  getLineGeometryOutputViews,
  validateLineGeometryOutput
} from '../line-segmentize/line-geometry-output';

/** Spatial contexts supported by this distance-measuring operation. */
const SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance'
} as const);

const OPERATION = 'GPUOffsetCurve';

/** Two scalar f32 coordinates per output row must be addressable by a uint32 index. */
const MAXIMUM_OUTPUT_VERTEX_COUNT = 0x7fffffff;
/** The exclusive count scan and requiredVertexCount use uint32, without wrapping. */
const MAXIMUM_REQUIRED_VERTEX_COUNT = 0xffffffff;
/** Parameter values are encoded as f32 and must remain finite in the GPU buffer. */
const MAXIMUM_PARAMETER_MAGNITUDE = 3.4028234663852886e38;
/** A uint32-sized scan needs at most four 256-row hierarchy levels and three offset passes. */
const COMPACT_SCAN_MAXIMUM_COMMAND_NODE_COUNT = 7;

/** Number of float32 elements in a {@link GPUOffsetCurve} parameter buffer. */
export const GPU_OFFSET_CURVE_PARAMETER_LENGTH = 4;

/** Default mitre limit, matching `shapely.offset_curve`. */
export const GPU_OFFSET_CURVE_DEFAULT_MITRE_LIMIT = 5;

/** Default number of segments per quarter circle of round joins, matching `shapely`. */
export const GPU_OFFSET_CURVE_DEFAULT_QUAD_SEGMENTS = 8;

/**
 * Largest `quadSegments`. Each vertex keeps its `2 * quadSegments + 1` join points in a private
 * array, which must stay within the 8,192 B per-function limit of core WebGPU (513 x 8 B here).
 */
export const GPU_OFFSET_CURVE_MAXIMUM_QUAD_SEGMENTS = 256;

/** Corner treatment of {@link GPUOffsetCurve}, matching shapely `join_style`. */
export type GPUOffsetCurveJoinStyle = 'round' | 'mitre' | 'bevel';

/** Input geometry kind of {@link GPUOffsetCurve}. */
export type GPUOffsetCurveGeometryType =
  /** Open lines: the first and last vertex offset along their only segment. */
  | 'lines'
  /**
   * Polygon rings, each a closed ring with an implicit closing edge (do not repeat the first
   * vertex). Takes the four-level polygon layout; the ring count is preserved.
   */
  | 'polygons';

/** CPU description of the per-frame parameters of {@link GPUOffsetCurve}. */
export type GPUOffsetCurveSettings = {
  /** Signed offset in position units: positive is the left of the line direction, negative the right. */
  distance: number;
  /** Mitre length over `abs(distance)` beyond which a mitre corner is clipped. Default 5, at least 1. */
  mitreLimit?: number;
};

/**
 * Packs {@link GPUOffsetCurveSettings} into the 4-element float32 layout
 * `[distance, mitreLimit, 0, 0]`.
 *
 * @param settings Settings to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If a value is not finite, `mitreLimit` is below 1, or `target` is too short.
 */
export function getGPUOffsetCurveParameterValues(
  settings: GPUOffsetCurveSettings,
  target: Float32Array = new Float32Array(GPU_OFFSET_CURVE_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_OFFSET_CURVE_PARAMETER_LENGTH) {
    throw new Error(`Offset curve target must hold ${GPU_OFFSET_CURVE_PARAMETER_LENGTH} elements`);
  }
  const mitreLimit = settings.mitreLimit ?? GPU_OFFSET_CURVE_DEFAULT_MITRE_LIMIT;
  if (!Number.isFinite(settings.distance)) {
    throw new Error('Offset curve distance must be a finite number');
  }
  if (!Number.isFinite(mitreLimit) || mitreLimit < 1) {
    throw new Error('Offset curve mitreLimit must be a finite number of at least 1');
  }
  validateFiniteNumber(
    OPERATION,
    'distance',
    Math.fround(settings.distance),
    -MAXIMUM_PARAMETER_MAGNITUDE,
    MAXIMUM_PARAMETER_MAGNITUDE
  );
  validateFiniteNumber(
    OPERATION,
    'mitreLimit',
    Math.fround(mitreLimit),
    1,
    MAXIMUM_PARAMETER_MAGNITUDE
  );
  target.set([settings.distance, mitreLimit, 0, 0]);
  return target;
}

/**
 * Number of output rows each input vertex owns: `2 * quadSegments + 1` for round joins (a corner
 * turns by at most half a circle), 2 for mitre and bevel joins.
 */
export function getGPUOffsetCurveRowsPerVertex(
  joinStyle: GPUOffsetCurveJoinStyle,
  quadSegments: number,
  insideTurns: GPUOffsetCurveInsideTurns = 'intersection'
): number {
  // A buffer-style inside turn emits three points (offset end, vertex, offset start).
  const minimum = insideTurns === 'buffer' ? 3 : 2;
  return Math.max(joinStyle === 'round' ? 2 * quadSegments + 1 : 2, minimum);
}

/**
 * Inside-of-turn rule of {@link GPUOffsetCurve}.
 *
 * - `'intersection'` (default): the intersection of the two offset lines (`shapely.offset_curve`).
 * - `'buffer'`: the intersection only while it lies on both offset segments; otherwise the corner
 *   is routed through the input vertex (`a + dn1, a, a + dn2`), as the GEOS buffer builder does, so
 *   a raw buffer boundary keeps a winding of at least one everywhere inside the buffer instead of
 *   the inverted far-away corner a tight switchback produces.
 */
export type GPUOffsetCurveInsideTurns = 'intersection' | 'buffer';

/** Caller-owned outputs of {@link GPUOffsetCurve}. */
export type GPUOffsetCurveOutput = {
  /** Existing padded output layout (default). */
  layout?: 'fixed-stride';
  /**
   * Offset points, exactly `positions.length * getGPUOffsetCurveRowsPerVertex(...)` rows. Input
   * vertex `i` owns rows `[i * k, (i + 1) * k)`; its first `counts[i]` rows are the offset points in
   * line order and any remaining rows repeat the last of them (zero-length, safe to draw).
   */
  positions: GraphDataView<'float32x2'>;
  /** Optional per-input-vertex number of distinct offset points, `positions.length` rows. */
  counts?: GraphDataView<'uint32'>;
};

/**
 * Compact offset points, with one output line per input line or ring and no stride padding.
 * Ring closure remains implicit. `sourceRows` identifies the input corner that emitted each
 * point, and `sourceLineIndices` identifies the input line or ring. Measures are unsupported:
 * an offset corner need not lie on the source line. Capacity truncates a suffix, with clamped
 * line offsets and exact `status.requiredVertexCount`. This avoids a worst-case stride-sized
 * output allocation; scratch uses two uint32 words per input vertex, plus scan summaries.
 */
export type GPUOffsetCurveCompactOutput = Omit<GPULineGeometryOutput, 'measures'> & {
  layout: 'compact';
};

/**
 * Properties for {@link GPUOffsetCurve}.
 *
 * Per-frame (no recompile): the contents of `parameters` (signed distance, mitre limit) and every
 * input buffer. Compile-time: view lengths, `geometryType`, `joinStyle`, `quadSegments`.
 */
export type GPUOffsetCurveProps = (
  | ({
      /** Open lines (default). Takes `positions` and `lineOffsets`. */
      geometryType?: 'lines';
    } & GPULineGeometryFields)
  | ({
      /**
       * Closed rings with an implicit closing edge. Takes the four-level polygon layout
       * (`positions`, `featureOffsets`, `polygonOffsets`, `ringOffsets`); every ring is offset on
       * its own, so the ring count is preserved: output ring `r` owns the rows
       * `[ringOffsets[r] * k, ringOffsets[r + 1] * k)` and `featureOffsets` and `polygonOffsets`
       * apply to it unchanged.
       */
      geometryType: 'polygons';
    } & GPUPolygonGeometryFields)
) & {
  /** Prefix for generated node IDs. Defaults to `'offset-curve'`. */
  id?: string;
  /**
   * Planar only. Spatial context of `positions`; omitted means native planar coordinates.
   * Longitude/latitude (or any other non-planar) contexts throw at construction: project
   * positions to a planar frame first, for example with `createGPULocalFrame` and
   * `GPULocalFrameProjection`, and pass the frame's context so its `frame` identity is kept.
   */
  spatialContext?: GPUSpatialContext;
  /** Corner treatment on the outside of a turn. Default `'round'`. */
  joinStyle?: GPUOffsetCurveJoinStyle;
  /**
   * Segments per quarter circle of round joins, 1 to {@link GPU_OFFSET_CURVE_MAXIMUM_QUAD_SEGMENTS}.
   * Default 8, like shapely.
   */
  quadSegments?: number;
  /** Inside-of-turn rule, see {@link GPUOffsetCurveInsideTurns}. Default `'intersection'`. */
  insideTurns?: GPUOffsetCurveInsideTurns;
  /** Per-frame packed float32 view written with {@link getGPUOffsetCurveParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Output points. */
  output: GPUOffsetCurveOutput | GPUOffsetCurveCompactOutput;
};

/**
 * One-sided offset of lines and rings (`shapely.offset_curve`, GeoPandas `offset_curve`, geo
 * `OffsetCurve`-style raw offset).
 *
 * For every input vertex the kernel emits the offset corner: on the outside of a turn a round arc
 * (`quadSegments` per quarter circle, GEOS rounding rule), a mitre point (clipped like GEOS when
 * the mitre exceeds `mitreLimit`) or a bevel pair; on the inside of a turn the single intersection
 * of the two offset segments; at line ends the plain normal offset. Positive distance offsets to the
 * left of the line direction, negative to the right. Planar coordinates only. Which side of a
 * vertex is the inside of its turn is the exact orientation sign of the three vertices
 * (`orientSign`), so it is right where the two edges nearly reverse and a float32 cross product
 * of their directions has no sign left; the corner points themselves are float32.
 *
 * **Render-grade, not GEOS-clean.** GEOS removes the loops that form when an offset segment is
 * shorter than the inset at an inside corner, and drops pieces closer than `distance` to the input.
 * This contributor does not: tight inside corners, distances larger than the local segment length
 * and concave regions can leave self-crossing loops. Away from loops the points equal
 * `shapely.offset_curve` (checked in the spec against GEOS, including mitre clipping and ring
 * closure). GEOS clamps `quad_segs` to at least 8 for offset curves, so compare round joins at
 * `quadSegments >= 8`; lower values here give coarser, cheaper arcs. On the outside of a ring GEOS
 * leaves the corner at the first vertex open (a quirk); this contributor joins it. Repeated vertices are treated as line ends on that side, so remove duplicates first,
 * and rings must not repeat the first vertex.
 *
 * The default output reserves {@link getGPUOffsetCurveRowsPerVertex} padded rows per input vertex.
 * With `output.layout: 'compact'`, count, exclusive scan and emit passes write only real corner
 * points into caller capacity. Both layouts use one invocation per input vertex, deterministically.
 * The distance and mitre limit are read per frame. Precision: f32 from the f32 positions.
 *
 * Planar only: `spatialContext` defaults to native planar coordinates and a longitude/latitude
 * context throws at construction. Project first, for example with `createGPULocalFrame` and
 * `GPULocalFrameProjection`, and pass the frame's context so its `frame` identity is kept.
 */
export class GPUOffsetCurve implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUOffsetCurveProps;
  /** Resolved spatial context (planar). */
  readonly spatialContext: GPUSpatialContext;
  /** Resolved geometry kind. */
  readonly geometryType: GPUOffsetCurveGeometryType;
  /** Line offsets, or the polygon ring offsets, as read by the kernel. */
  readonly boundaryOffsets: GraphDataView<'uint32'>;
  /** Resolved join style. */
  readonly joinStyle: GPUOffsetCurveJoinStyle;
  /** Resolved quarter-circle segment count. */
  readonly quadSegments: number;
  /** Output rows per input vertex. */
  readonly rowsPerVertex: number;
  /** Resolved inside-of-turn rule. */
  readonly insideTurns: GPUOffsetCurveInsideTurns;
  /** Upper bound including the published scan's hierarchy nodes. */
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPUOffsetCurveProps) {
    this.id = props.id ?? 'offset-curve';
    this.props = props;
    this.estimatedCommandNodeCount =
      props.output.layout === 'compact' ? 4 + COMPACT_SCAN_MAXIMUM_COMMAND_NODE_COUNT : 1;
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext,
      SPATIAL_CONTEXT_SUPPORT
    );
    this.geometryType = props.geometryType ?? 'lines';
    // Rings of a polygon are offset like closed lines, so the kernel reads them as line offsets.
    this.boundaryOffsets =
      props.geometryType === 'polygons' ? props.ringOffsets : props.lineOffsets;
    this.joinStyle = props.joinStyle ?? 'round';
    this.quadSegments = props.quadSegments ?? GPU_OFFSET_CURVE_DEFAULT_QUAD_SEGMENTS;
    this.insideTurns = props.insideTurns ?? 'intersection';
    this.rowsPerVertex = getGPUOffsetCurveRowsPerVertex(
      this.joinStyle,
      this.quadSegments,
      this.insideTurns
    );
    const {id} = this;
    if (this.insideTurns !== 'intersection' && this.insideTurns !== 'buffer') {
      throw new Error(`${id} insideTurns must be 'intersection' or 'buffer'`);
    }
    if (this.geometryType !== 'lines' && this.geometryType !== 'polygons') {
      throw new Error(`${id} geometryType must be 'lines' or 'polygons'`);
    }
    if (!['round', 'mitre', 'bevel'].includes(this.joinStyle)) {
      throw new Error(`${id} joinStyle must be 'round', 'mitre' or 'bevel'`);
    }
    validateCapacity(
      id,
      'quadSegments',
      this.quadSegments,
      1,
      GPU_OFFSET_CURVE_MAXIMUM_QUAD_SEGMENTS
    );
    validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 1) {
      throw new Error(`${id} needs at least one position`);
    }
    const offsetsName = props.geometryType === 'polygons' ? 'ringOffsets' : 'lineOffsets';
    validateSinglePackedView(this.boundaryOffsets, ['uint32'], `${id} ${offsetsName}`);
    if (this.boundaryOffsets.length < 2) {
      throw new Error(`${id} ${offsetsName} needs two rows`);
    }
    if (props.geometryType === 'polygons') {
      validateSinglePackedView(props.featureOffsets, ['uint32'], `${id} featureOffsets`);
      validateSinglePackedView(props.polygonOffsets, ['uint32'], `${id} polygonOffsets`);
    }
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_OFFSET_CURVE_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_OFFSET_CURVE_PARAMETER_LENGTH} float32 values`
      );
    }
    const maximumRequiredCount = props.positions.length * this.rowsPerVertex;
    validateMaximumCount(
      id,
      'positions.length * rowsPerVertex',
      maximumRequiredCount,
      MAXIMUM_REQUIRED_VERTEX_COUNT
    );
    validateCapacity(id, `${offsetsName}.length`, this.boundaryOffsets.length, 2);
    if (props.output.layout === 'compact') {
      validateLineGeometryOutput(id, props.output, this.boundaryOffsets.length - 1);
      validateRequirement(
        id,
        !('measures' in props.output && props.output.measures),
        'output.measures is unsupported for offset corners'
      );
    } else {
      validateRequirement(
        id,
        props.output.layout === undefined || props.output.layout === 'fixed-stride',
        "output.layout must be 'fixed-stride' or 'compact'"
      );
      validateSinglePackedView(props.output.positions, ['float32x2'], `${id} output.positions`);
      validateRequirement(
        id,
        props.output.positions.length === maximumRequiredCount,
        `output.positions must hold ${maximumRequiredCount} rows`
      );
      if (props.output.counts) {
        validateSinglePackedView(props.output.counts, ['uint32'], `${id} output.counts`);
        validateRequirement(
          id,
          props.output.counts.length === props.positions.length,
          `output.counts must hold ${props.positions.length} rows`
        );
      }
    }
    validateMaximumCount(
      id,
      'output.positions.length',
      props.output.positions.length,
      MAXIMUM_OUTPUT_VERTEX_COUNT
    );
    const outputs = this.getOutputViews();
    const output = props.output;
    if (output.layout === 'compact') {
      validateOutlineOutputPasses(
        id,
        {
          positions: output.positions,
          sourceRows: output.sourceRows,
          lineOffsets: output.lineOffsets,
          sourceLineIndices: output.sourceLineIndices,
          ...output.status
        },
        [
          ['positions', 'sourceRows'],
          ['lineOffsets', 'overflow', 'vertexCount', 'requiredVertexCount', 'sourceLineIndices'],
          ['count', 'requiredCount']
        ]
      );
    } else {
      validateOutlineOutputPasses(id, {positions: output.positions, counts: output.counts}, [
        ['positions', 'counts']
      ]);
    }
    validateGraphOutputsDisjointFromInputs(id, outputs, [
      props.positions,
      this.boundaryOffsets,
      props.parameters
    ]);
  }

  private getOutputViews(): (GraphDataView | undefined)[] {
    const {output} = this.props;
    return output.layout === 'compact'
      ? getLineGeometryOutputViews(output)
      : [output.positions, output.counts];
  }

  /** Returns padded generation, or compact count, scan, generation and publication. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const output = props.output;
    const compactOutput = output.layout === 'compact' ? output : undefined;
    const outputCounts = output.layout === 'compact' ? undefined : output.counts;
    validateGraphChunkBindingRanges(graph, id, {
      positions: props.positions,
      boundaryOffsets: this.boundaryOffsets,
      parameters: props.parameters,
      ...Object.fromEntries(this.getOutputViews().map((view, index) => [`output.${index}`, view]))
    });
    if (compactOutput) {
      // Both count columns include a terminal zero; all upstream scan summaries are smaller.
      const byteLength = (props.positions.length + 1) * Uint32Array.BYTES_PER_ELEMENT;
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'compact count scratch',
        byteLength,
        'reduce the input vertex count'
      );
      validateCapacity(
        id,
        'compact count scratch byteLength',
        byteLength,
        0,
        graph.device.limits.maxBufferSize
      );
    }
    const counts = compactOutput
      ? createTransientView(graph, `${id}-counts`, 'uint32', props.positions.length + 1)
      : undefined;
    const starts = compactOutput
      ? createTransientView(graph, `${id}-starts`, 'uint32', props.positions.length + 1)
      : undefined;
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (counts && starts) {
      nodes.push(this.createGenerationNode(graph, {counts, countOnly: true}));
      nodes.push(
        ...new GPUScan({id: `${id}-scan`, input: counts, output: starts}).getCommandNodes(graph)
      );
    }
    nodes.push(this.createGenerationNode(graph, {counts: outputCounts, starts, compactOutput}));
    if (compactOutput && starts) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-publish`,
          operation: OPERATION,
          variant: 'compact-publish',
          invocationCount: this.boundaryOffsets.length,
          bindings: [
            {name: 'starts', view: starts, type: 'u32', access: 'read'},
            {name: 'boundaries', view: this.boundaryOffsets, type: 'u32', access: 'read'},
            {name: 'offsets', view: compactOutput.lineOffsets, type: 'u32', access: 'read_write'},
            {
              name: 'overflow',
              view: compactOutput.status.overflow,
              type: 'u32',
              access: 'read_write'
            },
            ...(
              [
                ['vertexCount', compactOutput.status.vertexCount],
                ['requiredVertexCount', compactOutput.status.requiredVertexCount],
                ['sourceLineIndices', compactOutput.sourceLineIndices]
              ] as const
            ).flatMap(([name, view]) =>
              view ? [{name, view, type: 'u32' as const, access: 'read_write' as const}] : []
            )
          ],
          declarations: `const ROW_COUNT: u32 = ${props.positions.length}u;
const LINE_COUNT: u32 = ${this.boundaryOffsets.length - 1}u;
const CAPACITY: u32 = ${compactOutput.positions.length}u;`,
          body: `let row = min(boundaries[boundariesOffset + index], ROW_COUNT);
offsets[offsetsOffset + index] = min(starts[startsOffset + row], CAPACITY);
${compactOutput.sourceLineIndices ? 'if (index < LINE_COUNT) { sourceLineIndices[sourceLineIndicesOffset + index] = index; }' : ''}
if (index == 0u) {
  let required = starts[startsOffset + ROW_COUNT];
  overflow[overflowOffset] = select(0u, 1u, required > CAPACITY);
  ${compactOutput.status.vertexCount ? 'vertexCount[vertexCountOffset] = min(required, CAPACITY);' : ''}
  ${compactOutput.status.requiredVertexCount ? 'requiredVertexCount[requiredVertexCountOffset] = required;' : ''}
}`
        })
      );
      nodes.push(
        createLineGeometryCountNode(graph, {
          id,
          operation: OPERATION,
          output: compactOutput,
          lineCapacity: this.boundaryOffsets.length - 1
        })
      );
    }
    return nodes;
  }

  /** Same corner computation for padded generation, compact counting and compact emission. */
  private createGenerationNode<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    options: {
      counts?: GraphDataView<'uint32'>;
      countOnly?: boolean;
      starts?: GraphDataView<'uint32'>;
      compactOutput?: GPUOffsetCurveCompactOutput;
    }
  ): GPUCommandNode<Parameters> {
    const {props, id} = this;
    const {counts, countOnly, starts, compactOutput} = options;
    const bindings: WGSLKernelBinding[] = [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'lineOffsets', view: this.boundaryOffsets, type: 'u32', access: 'read'},
      {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'}
    ];
    if (!countOnly) {
      bindings.push({
        name: 'outputPositions',
        view: props.output.positions,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (counts) {
      bindings.push({
        name: 'outputCounts',
        view: counts,
        type: 'u32',
        access: 'read_write'
      });
    }
    if (starts) bindings.push({name: 'starts', view: starts, type: 'u32', access: 'read'});
    if (compactOutput?.sourceRows)
      bindings.push({
        name: 'sourceRows',
        view: compactOutput.sourceRows,
        type: 'u32',
        access: 'read_write'
      });
    return createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-${countOnly ? 'count' : 'generate'}`,
      operation: OPERATION,
      variant: `${this.geometryType}-${this.joinStyle}-${this.insideTurns}`,
      bindings,
      invocationCount: props.positions.length + (countOnly ? 1 : 0),
      declarations: /* wgsl */ `
const LINE_COUNT: u32 = ${this.boundaryOffsets.length - 1}u;
const ROW_COUNT: u32 = ${props.positions.length}u;
const ROWS_PER_VERTEX: u32 = ${this.rowsPerVertex}u;
const QUAD_SEGMENTS: f32 = ${this.quadSegments}.0;
const IS_RING: bool = ${this.geometryType === 'polygons'};
const HALF_PI: f32 = 1.5707963267948966;

fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

fn getLeftNormal(direction: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(-direction.y, direction.x);
}

fn rotateVector(v: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec2<f32>(c * v.x - s * v.y, s * v.x + c * v.y);
}
${FIND_LINE_WGSL}
${EXACT_ORIENTATION_WGSL}
${EXACT_GEOMETRY_WGSL}`,
      body: /* wgsl */ `${countOnly ? 'if (index == ROW_COUNT) { outputCounts[outputCountsOffset + index] = 0u; return; }' : ''}
  let offsetDistance = parameters[parametersOffset];
  let mitreLimit = max(parameters[parametersOffset + 1u], 1.0);
  let magnitude = abs(offsetDistance);
  let center = getPosition(index);
  var points = array<vec2<f32>, ${this.rowsPerVertex}>();
  var count = 1u;
  points[0] = center;

  var hasIncoming = false;
  var hasOutgoing = false;
  var incoming = vec2<f32>(0.0);
  var outgoing = vec2<f32>(0.0);
  var incomingLength = 0.0;
  var outgoingLength = 0.0;
  var previousPoint = center;
  var nextPoint = center;
  let line = findLine(index);
  if (line != NO_LINE) {
    let lineStart = lineOffsets[lineOffsetsOffset + line];
    let lineEnd = min(lineOffsets[lineOffsetsOffset + line + 1u], ROW_COUNT);
    let lineLength = lineEnd - lineStart;
    var previousRow = ROW_COUNT;
    var nextRow = ROW_COUNT;
    if (index > lineStart) {
      previousRow = index - 1u;
    } else if (IS_RING && lineLength >= 3u) {
      previousRow = lineEnd - 1u;
    }
    if (index + 1u < lineEnd) {
      nextRow = index + 1u;
    } else if (IS_RING && lineLength >= 3u) {
      nextRow = lineStart;
    }
    if (previousRow < ROW_COUNT) {
      previousPoint = getPosition(previousRow);
      let delta = center - previousPoint;
      let deltaLength = length(delta);
      if (deltaLength > 0.0) {
        incoming = delta / deltaLength;
        incomingLength = deltaLength;
        hasIncoming = true;
      }
    }
    if (nextRow < ROW_COUNT) {
      nextPoint = getPosition(nextRow);
      let delta = nextPoint - center;
      let deltaLength = length(delta);
      if (deltaLength > 0.0) {
        outgoing = delta / deltaLength;
        outgoingLength = deltaLength;
        hasOutgoing = true;
      }
    }
  }

  if (hasIncoming && !hasOutgoing) {
    points[0] = center + getLeftNormal(incoming) * offsetDistance;
  } else if (!hasIncoming && hasOutgoing) {
    points[0] = center + getLeftNormal(outgoing) * offsetDistance;
  } else if (hasIncoming && hasOutgoing) {
    let normalIn = getLeftNormal(incoming);
    let normalOut = getLeftNormal(outgoing);
    let a = normalIn * offsetDistance;
    let b = normalOut * offsetDistance;
    // Which side is the inside of the turn is an exact orientation sign of the three vertices.
    // The f32 cross product of the two unit directions loses that sign where the edges nearly
    // reverse (the tip of a sliver thinner than one f32 step): the join then went to the wrong
    // side and a buffer lost its end cap there. The f32 sign is kept only for input the exact
    // predicate cannot certify (code 2: non-finite coordinates).
    let exactTurn = exactOrientSigns(
      1u,
      array<vec2f, 4>(previousPoint, vec2f(0.0), vec2f(0.0), vec2f(0.0)),
      array<vec2f, 4>(center, vec2f(0.0), vec2f(0.0), vec2f(0.0)),
      array<vec2f, 4>(nextPoint, vec2f(0.0), vec2f(0.0), vec2f(0.0))
    );
    let turn = select(f32(exactTurn.signs.x), sign(incoming.x * outgoing.y - incoming.y * outgoing.x), exactTurn.uncertain);
    let cosTurn = clamp(dot(incoming, outgoing), -1.0, 1.0);
    if (turn * offsetDistance > 0.0 && 1.0 + cosTurn > 1e-6) {
      // Inside of the turn: intersection of the two offset segments.
      ${getInsideTurnSource(this.insideTurns)}${
        this.insideTurns === 'buffer'
          ? `
    } else if (turn * offsetDistance > 0.0) {
      // A full reversal on the inside: route through the vertex.
      points[0] = center + a;
      points[1] = center;
      points[2] = center + b;
      count = 3u;`
          : ''
      }
    } else if (length(a - b) < magnitude * 1e-3) {
      points[0] = center + a;
    } else {
      ${getJoinSource(this.joinStyle)}
    }
  }

  ${
    countOnly
      ? ''
      : `let base = ${starts ? 'starts[startsOffset + index]' : 'index * ROWS_PER_VERTEX'};
  for (var slot = 0u; slot < ${starts ? 'count' : 'ROWS_PER_VERTEX'}; slot++) {
    ${starts ? `if (base + slot >= ${props.output.positions.length}u) { break; }` : ''}
    let point = points[min(slot, count - 1u)];
    let destination = base + slot;
    let coordinateOffset = outputPositionsOffset + 2u * destination;
    outputPositions[coordinateOffset] = point.x;
    outputPositions[coordinateOffset + 1u] = point.y;
    ${compactOutput?.sourceRows ? 'sourceRows[sourceRowsOffset + destination] = index;' : ''}
  }`
  }
  ${counts ? 'outputCounts[outputCountsOffset + index] = count;' : ''}`
    });
  }
}

/** WGSL statements for the inside-of-turn corner; set `points` and `count`. @internal */
function getInsideTurnSource(insideTurns: GPUOffsetCurveInsideTurns): string {
  if (insideTurns === 'intersection') {
    return 'points[0] = center + (a + b) / (1.0 + cosTurn);';
  }
  // The intersection lies |d| tan(theta / 2) back along each segment (theta the turn angle); where
  // that overshoots either segment the corner goes through the vertex (GEOS buffer rule).
  return `let inset = magnitude * sqrt(max(1.0 - cosTurn, 0.0) / (1.0 + cosTurn));
      if (inset > incomingLength || inset > outgoingLength) {
        points[0] = center + a;
        points[1] = center;
        points[2] = center + b;
        count = 3u;
      } else {
        points[0] = center + (a + b) / (1.0 + cosTurn);
      }`;
}

/** WGSL statements for the outside-of-turn corner; set `points` and `count`. @internal */
function getJoinSource(joinStyle: GPUOffsetCurveJoinStyle): string {
  if (joinStyle === 'bevel') {
    return `points[0] = center + a;
      points[1] = center + b;
      count = 2u;`;
  }
  if (joinStyle === 'round') {
    // GEOS: round(turnAngle / (pi / 2 / quadSegments)) uniform steps, both ends included.
    return `let turnAngle = acos(cosTurn);
      let steps = u32(floor(turnAngle * QUAD_SEGMENTS / HALF_PI + 0.5));
      let direction = select(1.0, -1.0, offsetDistance > 0.0);
      points[0] = center + a;
      if (steps >= 1u) {
        for (var arcStep = 1u; arcStep < steps; arcStep++) {
          points[arcStep] = center + rotateVector(a, direction * turnAngle * f32(arcStep) / f32(steps));
        }
        points[steps] = center + b;
        count = steps + 1u;
      } else {
        points[1] = center + b;
        count = 2u;
      }`;
  }
  return `let cosHalf = sqrt(max(0.5 * (1.0 + cosTurn), 0.0));
      if (cosHalf > 0.0 && 1.0 / cosHalf <= mitreLimit) {
        points[0] = center + (a + b) / (1.0 + cosTurn);
      } else {
        // Clipped mitre: the offset lines cut by the line perpendicular to the bisector at mitreLimit * distance.
        var bisector = a + b;
        if (length(bisector) < 1e-6 * magnitude) {
          bisector = incoming * magnitude;
        }
        let tipDirection = normalize(bisector);
        let clipMiddle = center + tipDirection * (mitreLimit * magnitude);
        let halfLength = magnitude * (1.0 - mitreLimit * cosHalf) / sqrt(max(1.0 - cosHalf * cosHalf, 1e-12));
        let perpendicular = getLeftNormal(tipDirection) * select(-1.0, 1.0, dot(a, getLeftNormal(tipDirection)) >= 0.0);
        points[0] = clipMiddle + perpendicular * halfLength;
        points[1] = clipMiddle - perpendicular * halfLength;
        count = 2u;
      }`;
}
