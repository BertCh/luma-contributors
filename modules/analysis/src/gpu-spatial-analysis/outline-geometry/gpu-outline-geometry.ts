// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  getGPUSphereRadius,
  resolveGPUSpatialContext,
  warnGPUShortLongitudeLatitudeDistance,
  type GPUSpatialContext,
  type GPUSpatialContextSupport
} from '../../analysis-core/contracts/spatial-context';
import type {
  GPULineGeometryFields,
  GPUPolygonGeometryFields
} from '../../analysis-core/contracts/index';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength,
  validateRequirement,
  validateGraphOutputsDisjointFromInputs,
  validateFiniteNumber,
  validateMaximumCount,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';

import type {GPUCardinalityPort, GPUOverflowStatusPort} from '../../analysis-core/contracts/status';
import {validateOutlineOutputPasses} from './validate-outline-output-passes';

const OPERATION = 'GPUOutlineGeometry';
/** Counts are scanned as uint32 and must not wrap even when vertex output capacity is small. */
const MAXIMUM_REQUIRED_VERTEX_COUNT = 0xffffffff;
/** Four hierarchy scans and three offset passes cover every uint32-sized count column. */
const MAXIMUM_SCAN_COMMAND_NODE_COUNT = 7;

/** Two scalar f32 coordinates per output vertex must be addressable with uint32 indices. */
const MAXIMUM_OUTPUT_VERTEX_COUNT = 0x7fffffff;
/** Smallest normal f32: smaller radii can flush to zero before the shader divides by them. */
const MINIMUM_SHADER_RADIUS = 2 ** -126;
/** The sphere radius is emitted as an f32 shader literal and must not round to infinity. */
const MAXIMUM_SHADER_RADIUS = 3.4028234663852886e38;

/** Planar coordinates, or longitude/latitude with a sphere (great-circle) metric. */
const SPATIAL_CONTEXT_SUPPORT: GPUSpatialContextSupport = Object.freeze({
  coordinateSpaces: Object.freeze(['planar', 'longitude-latitude'] as const),
  metricsByCoordinateSpace: Object.freeze({
    planar: Object.freeze(['native'] as const),
    'longitude-latitude': Object.freeze(['great-circle'] as const)
  }),
  measure: 'distance'
});

/** Number of float32 elements in a {@link GPUOutlineGeometry} parameter buffer. */
export const GPU_OUTLINE_GEOMETRY_PARAMETER_LENGTH = 4;

/** Default number of triangles in the round join (and cap) disc of one vertex. */
export const GPU_OUTLINE_GEOMETRY_DEFAULT_JOIN_SEGMENTS = 16;

/** Input geometry kind of {@link GPUOutlineGeometry}. */
export type GPUOutlineGeometryType =
  /** Every position is a point; the outline is a disc. */
  | 'points'
  /** Open lines; the outline is a round-capped stroke. */
  | 'lines'
  /** Polygon rings (the closing edge is implicit); the outline is a round-joined stroke. */
  | 'polygons';

/** CPU description of the per-frame parameters of {@link GPUOutlineGeometry}. */
export type GPUOutlineGeometrySettings = {
  /**
   * Spatial context of the positions; when it is longitude/latitude, a distance under 10 float32
   * quanta (about 17 m) logs a warning (see Precision of longitude/latitude positions). Does not
   * change the packed values.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Offset distance, in position units (planar context) or in the context's `units` on its sphere
   * (longitude/latitude context, meters by default). Must be non-negative; `0` draws nothing visible.
   */
  distance: number;
};

/**
 * Packs {@link GPUOutlineGeometrySettings} into the 4-element float32 layout
 * `[distance, 0, 0, 0]`. Write the result into a `GPUParameterBuffer` between encodings to change
 * the distance without recompiling.
 *
 * @param settings Settings to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If the distance is negative or not finite, or `target` is too short.
 */
export function getGPUOutlineGeometryParameterValues(
  settings: GPUOutlineGeometrySettings,
  target: Float32Array = new Float32Array(GPU_OUTLINE_GEOMETRY_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_OUTLINE_GEOMETRY_PARAMETER_LENGTH) {
    throw new Error(
      `Outline geometry target must hold ${GPU_OUTLINE_GEOMETRY_PARAMETER_LENGTH} elements`
    );
  }
  if (!Number.isFinite(settings.distance) || settings.distance < 0) {
    throw new Error('Outline geometry distance must be a non-negative finite number');
  }
  warnGPUShortLongitudeLatitudeDistance(
    'GPUOutlineGeometry',
    settings.spatialContext,
    settings.distance
  );
  validateFiniteNumber(
    OPERATION,
    'distance',
    Math.fround(settings.distance),
    0,
    MAXIMUM_SHADER_RADIUS
  );
  target.set([settings.distance, 0, 0, 0]);
  return target;
}

/**
 * Number of output vertices each input vertex owns: `joinSegments` disc triangles plus two
 * triangles of the quad toward the next vertex, three vertices each.
 */
export function getGPUOutlineGeometryVerticesPerInput(joinSegments: number): number {
  return 3 * (joinSegments + 2);
}

/** Caller-owned outputs of {@link GPUOutlineGeometry}. */
export type GPUOutlineGeometryOutput = {
  layout?: 'fixed-stride';
  /**
   * Triangle-list vertices, exactly `positions.length * getGPUOutlineGeometryVerticesPerInput(joinSegments)`
   * rows. Input vertex `i` owns rows `[i * k, (i + 1) * k)` with `k` the per-input vertex count.
   * Unused triangles are degenerate (every vertex at the input vertex), so draw the whole buffer.
   */
  positions: GraphDataView<'float32x2'>;
};

/**
 * Triangle-list output with no degenerate end quads, and no triangles at distance zero.
 * Capacity must be divisible by three, so truncation never publishes a partial triangle.
 * `status.count` is the clamped vertex count for drawing; `requiredCount` is the exact vertex
 * requirement. `vertexOffsets` has one start per input vertex and a terminal total, clamped to
 * capacity. `sourceRows` identifies the input vertex for every emitted triangle vertex.
 */
export type GPUOutlineGeometryCompactOutput = {
  layout: 'compact';
  positions: GraphDataView<'float32x2'>;
  vertexOffsets: GraphDataView<'uint32'>;
  sourceRows?: GraphDataView<'uint32'>;
  status: GPUCardinalityPort & GPUOverflowStatusPort;
};

/**
 * Properties for {@link GPUOutlineGeometry}.
 *
 * Per-frame (no recompile): the contents of `parameters` (distance) and every input buffer.
 * Compile-time: view lengths, `geometryType`, `spatialContext`, `joinSegments`, output layout and optional columns.
 */
export type GPUOutlineGeometryProps = (
  | {
      /** Every position is a point. Takes only `positions`. */
      geometryType: 'points';
      /** Packed positions: planar coordinates, or longitude/latitude degrees for a sphere context. */
      positions: GraphDataView<'float32x2'>;
    }
  | ({
      /** Open lines. Takes `positions` and `lineOffsets`. */
      geometryType: 'lines';
    } & GPULineGeometryFields)
  | ({
      /**
       * Polygon rings, each stroked as a closed ring. Takes the four-level polygon layout
       * (`positions`, `featureOffsets`, `polygonOffsets`, `ringOffsets`). The output has one
       * block of vertices per input vertex, so the ring structure is that of the input.
       */
      geometryType: 'polygons';
    } & GPUPolygonGeometryFields)
) & {
  /** Prefix for generated node IDs. Defaults to `'outline-geometry'`. */
  id?: string;
  /**
   * Spatial context of `positions` and of the offset `distance`. Omitted means native planar
   * coordinates with distances in position units. `{coordinateSpace: 'longitude-latitude', metric:
   * 'great-circle'}` makes positions longitude/latitude degrees and distances meters (or the
   * context's `units`) on a sphere of `sphereRadius` (default the mean Earth radius), applied with
   * the local east/north scale of each vertex: an equirectangular approximation, accurate for
   * distances small against the radius and wrong near the poles and across the antimeridian.
   * This replaces the removed `coordinateSystem` and `radius` props. Other contexts throw.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Triangles in each round join and cap disc, at least 3. Default
   * {@link GPU_OUTLINE_GEOMETRY_DEFAULT_JOIN_SEGMENTS}. The disc is inscribed, so the picture is
   * up to `distance * (1 - cos(pi / joinSegments))` narrower than the true offset (1.9% at 16).
   */
  joinSegments?: number;
  /** Per-frame packed float32 view written with {@link getGPUOutlineGeometryParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Output triangles. */
  output: GPUOutlineGeometryOutput | GPUOutlineGeometryCompactOutput;
};

/**
 * Builds render-only buffer geometry for points, lines and rings: for each input vertex a round
 * join disc and a rectangle toward the next vertex, as non-indexed triangles ready to draw
 * (turf `buffer` and `lineOffset` for the picture only).
 *
 * **Picture only.** Triangles of neighboring vertices overlap, and nothing is unioned or
 * dissolved, so the geometry is correct for drawing opaque or stencil-combined fills but wrong for
 * area, overlay, selection or point-in-polygon queries on the buffered shape. Use
 * `GPUBufferSelection` and `dwithin` predicates for queries, and `GPUDistanceField` for a unioned
 * look. Translucent fills show overlaps as darker bands; draw into a stencil or a mask first.
 * Self-intersections and holes are not interpreted. No negative buffers and no one-sided
 * (`lineOffset`) strokes; both are open.
 *
 * The default fixed-stride output reserves {@link getGPUOutlineGeometryVerticesPerInput} vertices
 * per input vertex, including degenerate end quads. With `output.layout: 'compact'`, count, scan
 * and emit passes omit those quads and omit every triangle at zero distance, publishing clamped
 * offsets and exact vertex requirements. Truncation keeps complete triangles. Both routes run
 * one invocation per input vertex without atomics; distance is read per frame. Triangles are
 * counter-clockwise for a positive distance.
 *
 * Precision: f32 from the f32 positions.
 */
export class GPUOutlineGeometry implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUOutlineGeometryProps;
  /** Resolved spatial context (planar, or longitude/latitude with a great-circle metric). */
  readonly spatialContext: GPUSpatialContext;
  /** Sphere radius in the context's length unit; only used by longitude/latitude contexts. */
  readonly radius: number;
  /** Resolved join disc triangle count. */
  readonly joinSegments: number;
  /** Output vertices per input vertex. */
  readonly verticesPerInput: number;
  /** Line offsets or polygon ring offsets as read by the kernel; absent for points. */
  readonly boundaryOffsets?: GraphDataView<'uint32'>;
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPUOutlineGeometryProps) {
    this.id = props.id ?? 'outline-geometry';
    this.props = props;
    this.estimatedCommandNodeCount =
      props.output.layout === 'compact' ? 4 + MAXIMUM_SCAN_COMMAND_NODE_COUNT : 1;
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext,
      SPATIAL_CONTEXT_SUPPORT
    );
    this.radius = getGPUSphereRadius(this.spatialContext);
    validateFiniteNumber(
      this.id,
      'spatialContext sphere radius',
      Math.fround(this.radius),
      MINIMUM_SHADER_RADIUS,
      MAXIMUM_SHADER_RADIUS
    );
    this.joinSegments = props.joinSegments ?? GPU_OUTLINE_GEOMETRY_DEFAULT_JOIN_SEGMENTS;
    this.verticesPerInput = getGPUOutlineGeometryVerticesPerInput(this.joinSegments);
    const {id} = this;
    this.boundaryOffsets =
      props.geometryType === 'lines'
        ? props.lineOffsets
        : props.geometryType === 'polygons'
          ? props.ringOffsets
          : undefined;
    if (!['points', 'lines', 'polygons'].includes(props.geometryType)) {
      throw new Error(`${id} geometryType must be 'points', 'lines' or 'polygons'`);
    }
    validateCapacity(id, 'joinSegments', this.joinSegments, 3);
    validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 1) {
      throw new Error(`${id} needs at least one position`);
    }
    if (props.geometryType !== 'points') {
      const offsetsName = props.geometryType === 'lines' ? 'lineOffsets' : 'ringOffsets';
      if (!this.boundaryOffsets) {
        throw new Error(`${id} ${offsetsName} is required`);
      }
      validateSinglePackedView(this.boundaryOffsets, ['uint32'], `${id} ${offsetsName}`);
      if (this.boundaryOffsets.length < 2) {
        throw new Error(`${id} ${offsetsName} needs two rows`);
      }
      if (props.geometryType === 'polygons') {
        validateSinglePackedView(props.featureOffsets, ['uint32'], `${id} featureOffsets`);
        validateSinglePackedView(props.polygonOffsets, ['uint32'], `${id} polygonOffsets`);
      }
    }
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_OUTLINE_GEOMETRY_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_OUTLINE_GEOMETRY_PARAMETER_LENGTH} float32 values`
      );
    }
    validateSinglePackedView(props.output.positions, ['float32x2'], `${id} output.positions`);
    const expectedLength = props.positions.length * this.verticesPerInput;
    validateMaximumCount(
      id,
      'positions.length * verticesPerInput',
      expectedLength,
      MAXIMUM_REQUIRED_VERTEX_COUNT
    );
    const output = props.output;
    if (output.layout === 'compact') {
      validateRequirement(
        id,
        output.positions.length > 0 && output.positions.length % 3 === 0,
        'output.positions.length must be positive and divisible by three'
      );
      validateSinglePackedView(output.vertexOffsets, ['uint32'], `${id} output.vertexOffsets`);
      validateRequirement(
        id,
        output.vertexOffsets.length === props.positions.length + 1,
        'output.vertexOffsets.length must equal positions.length + 1'
      );
      if (output.sourceRows) {
        validateSinglePackedView(output.sourceRows, ['uint32'], `${id} output.sourceRows`);
        validateRequirement(
          id,
          output.sourceRows.length === output.positions.length,
          'output.sourceRows.length must equal output.positions.length'
        );
      }
      for (const [name, view] of Object.entries(output.status)) {
        if (!view) continue;
        validateSinglePackedView(view, ['uint32'], `${id} output.status.${name}`);
        validateCapacity(id, `output.status.${name}.length`, view.length, 1);
      }
      validateOutlineOutputPasses(
        id,
        {
          positions: output.positions,
          sourceRows: output.sourceRows,
          vertexOffsets: output.vertexOffsets,
          ...output.status
        },
        [
          ['positions', 'sourceRows'],
          ['vertexOffsets', 'count', 'requiredCount', 'overflow']
        ]
      );
    } else {
      validateRequirement(
        id,
        output.layout === undefined || output.layout === 'fixed-stride',
        "output.layout must be 'compact' or 'fixed-stride'"
      );
      validateRequirement(
        id,
        output.positions.length === expectedLength,
        `output.positions must hold ${expectedLength} rows`
      );
    }
    validateMaximumCount(
      id,
      'output.positions.length',
      output.positions.length,
      MAXIMUM_OUTPUT_VERTEX_COUNT
    );
    validateGraphOutputsDisjointFromInputs(id, this.getOutputViews(), [
      props.positions,
      this.boundaryOffsets,
      props.parameters
    ]);
  }

  /** Returns the single generation node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const output = props.output;
    validateGraphChunkBindingRanges(graph, id, {
      positions: props.positions,
      boundaryOffsets: this.boundaryOffsets,
      parameters: props.parameters,
      ...Object.fromEntries(this.getOutputViews().map((view, index) => [`output.${index}`, view]))
    });
    if (output.layout !== 'compact') return [this.createGenerationNode(graph)];
    const byteLength = (props.positions.length + 1) * 4;
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'compact counts and scan scratch',
      byteLength,
      'reduce input vertex count'
    );
    validateCapacity(
      id,
      'compact scratch byteLength',
      byteLength,
      1,
      graph.device.limits.maxBufferSize
    );
    const counts = createTransientView(graph, `${id}-counts`, 'uint32', props.positions.length + 1);
    const starts = createTransientView(graph, `${id}-starts`, 'uint32', props.positions.length + 1);
    const nodes: GPUCommandNode<Parameters>[] = [
      this.createGenerationNode(graph, {counts}),
      ...new GPUScan({id: `${id}-scan`, input: counts, output: starts}).getCommandNodes(graph),
      this.createGenerationNode(graph, {starts})
    ];
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        invocationCount: output.vertexOffsets.length,
        bindings: [
          {name: 'starts', view: starts, type: 'u32', access: 'read'},
          {name: 'offsets', view: output.vertexOffsets, type: 'u32', access: 'read_write'},
          {name: 'count', view: output.status.count, type: 'u32', access: 'read_write'},
          {name: 'overflow', view: output.status.overflow, type: 'u32', access: 'read_write'},
          ...(output.status.requiredCount
            ? [
                {
                  name: 'required',
                  view: output.status.requiredCount,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        body: `let requiredVertices = starts[startsOffset + ${props.positions.length}u];
offsets[offsetsOffset + index] = min(starts[startsOffset + index], ${output.positions.length}u);
if (index == 0u) {
 count[countOffset] = min(requiredVertices, ${output.positions.length}u);
 overflow[overflowOffset] = select(0u, 1u, requiredVertices > ${output.positions.length}u);
 ${output.status.requiredCount ? 'required[requiredOffset] = requiredVertices;' : ''}
}`
      })
    );
    return nodes;
  }

  private getOutputViews(): (GraphDataView | undefined)[] {
    const {output} = this.props;
    return output.layout === 'compact'
      ? [
          output.positions,
          output.vertexOffsets,
          output.sourceRows,
          output.status.count,
          output.status.requiredCount,
          output.status.overflow
        ]
      : [output.positions];
  }

  private createGenerationNode<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    options: {
      counts?: GraphDataView<'uint32'>;
      starts?: GraphDataView<'uint32'>;
    } = {}
  ): GPUCommandNode<Parameters> {
    const {props, id, joinSegments} = this;
    const compact = props.output.layout === 'compact' ? props.output : undefined;
    const {counts, starts} = options;
    const hasLines = props.geometryType !== 'points';
    const bindings: WGSLKernelBinding[] = [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'}
    ];
    if (hasLines && this.boundaryOffsets) {
      bindings.push({name: 'lineOffsets', view: this.boundaryOffsets, type: 'u32', access: 'read'});
    }
    bindings.push({name: 'parameters', view: props.parameters, type: 'f32', access: 'read'});
    if (counts) bindings.push({name: 'counts', view: counts, type: 'u32', access: 'read_write'});
    else
      bindings.push({
        name: 'outputPositions',
        view: props.output.positions,
        type: 'f32',
        access: 'read_write'
      });
    if (starts) bindings.push({name: 'starts', view: starts, type: 'u32', access: 'read'});
    if (compact?.sourceRows && !counts)
      bindings.push({
        name: 'sourceRows',
        view: compact.sourceRows,
        type: 'u32',
        access: 'read_write'
      });
    const spherical = this.spatialContext.coordinateSpace === 'longitude-latitude';
    return createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-${counts ? 'count' : 'generate'}`,
      operation: OPERATION,
      variant: props.geometryType,
      bindings,
      invocationCount: props.positions.length + (counts ? 1 : 0),
      declarations: /* wgsl */ `
const LINE_COUNT: u32 = ${this.boundaryOffsets ? this.boundaryOffsets.length - 1 : 0}u;
const ROW_COUNT: u32 = ${props.positions.length}u;
const JOIN_SEGMENTS: u32 = ${joinSegments}u;
const VERTICES_PER_INPUT: u32 = ${this.verticesPerInput}u;
const RADIUS: f32 = ${getWGSLFloatLiteral(this.radius)};
const DEGREES_PER_RADIAN: f32 = 57.29577951308232;
const TAU: f32 = 6.283185307179586;

fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

${
  counts
    ? ''
    : `fn writeVertex(slot: u32, base: u32, point: vec2<f32>, owner: u32) {
  let destination = base + slot;
  if (destination >= ${props.output.positions.length}u) { return; }
  let coordinateOffset = outputPositionsOffset + 2u * destination;
  outputPositions[coordinateOffset] = point.x;
  outputPositions[coordinateOffset + 1u] = point.y;
  ${compact?.sourceRows ? 'sourceRows[sourceRowsOffset + destination] = owner;' : ''}
}`
}

// Moves p by distance d along the metric direction (east, north).
fn offsetPoint(p: vec2<f32>, direction: vec2<f32>, d: f32) -> vec2<f32> {
  ${
    spherical
      ? `let latitudeScale = d / RADIUS * DEGREES_PER_RADIAN;
  let longitudeScale = latitudeScale / max(cos(radians(p.y)), 1e-6);
  return p + vec2<f32>(direction.x * longitudeScale, direction.y * latitudeScale);`
      : 'return p + direction * d;'
  }
}

// Unit metric direction of the segment a to b, or zero for a zero-length segment.
fn getDirection(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  ${
    spherical
      ? `let middleLatitude = radians(0.5 * (a.y + b.y));
  let metric = vec2<f32>((b.x - a.x) * cos(middleLatitude), b.y - a.y);`
      : 'let metric = b - a;'
  }
  let metricLength = length(metric);
  if (metricLength <= 0.0) {
    return vec2<f32>(0.0);
  }
  return metric / metricLength;
}
${hasLines ? FIND_LINE_WGSL : ''}`,
      body: /* wgsl */ `${counts ? 'if (index == ROW_COUNT) { counts[countsOffset + index] = 0u; return; }' : ''}
  let distance = max(parameters[parametersOffset], 0.0);
  let center = getPosition(index);
  var nextRow = ROW_COUNT;
  ${hasLines ? getNextRowSource(props.geometryType) : ''}
  var direction = vec2<f32>(0.0);
  if (nextRow < ROW_COUNT) { direction = getDirection(center, getPosition(nextRow)); }
  let hasQuad = dot(direction, direction) > 0.0;
  ${
    counts
      ? `counts[countsOffset + index] = select(0u, 3u * JOIN_SEGMENTS + select(0u, 6u, hasQuad), distance > 0.0);`
      : `
  ${compact ? 'if (!(distance > 0.0)) { return; }' : ''}
  let base = ${starts ? 'starts[startsOffset + index]' : 'index * VERTICES_PER_INPUT'};
  var rimA = offsetPoint(center, vec2<f32>(1.0, 0.0), distance);
  for (var corner = 0u; corner < JOIN_SEGMENTS; corner++) {
    let angle = TAU * f32(corner + 1u) / f32(JOIN_SEGMENTS);
    let rimB = offsetPoint(center, vec2<f32>(cos(angle), sin(angle)), distance);
    writeVertex(3u * corner, base, center, index);
    writeVertex(3u * corner + 1u, base, rimA, index);
    writeVertex(3u * corner + 2u, base, rimB, index);
    rimA = rimB;
  }
  var quad = array<vec2<f32>, 6>(center, center, center, center, center, center);
  if (hasQuad) {
    let next = getPosition(nextRow);
    let normal = vec2<f32>(-direction.y, direction.x);
    let aLeft = offsetPoint(center, normal, distance);
    let aRight = offsetPoint(center, -normal, distance);
    let bRight = offsetPoint(next, -normal, distance);
    let bLeft = offsetPoint(next, normal, distance);
    quad = array<vec2<f32>, 6>(aLeft, aRight, bRight, aLeft, bRight, bLeft);
  }
  ${compact ? 'if (!hasQuad) { return; }' : ''}
  for (var vertex = 0u; vertex < 6u; vertex++) {
    writeVertex(3u * JOIN_SEGMENTS + vertex, base, quad[vertex], index);
  }`
  }`
    });
  }
}

/**
 * WGSL `findLine(row) -> u32`: upper-bound binary search over the bound `lineOffsets`.
 *
 * @internal
 */
export const FIND_LINE_WGSL = /* wgsl */ `
const NO_LINE: u32 = 0xffffffffu;

fn findLine(row: u32) -> u32 {
  var low = 0u;
  var high = LINE_COUNT + 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (lineOffsets[lineOffsetsOffset + middle] <= row) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low == 0u || low > LINE_COUNT) {
    return NO_LINE;
  }
  return low - 1u;
}
`;

/** WGSL statements that set `nextRow` for lines or rings. @internal */
function getNextRowSource(geometryType: GPUOutlineGeometryType): string {
  return `let line = findLine(index);
  if (line != NO_LINE) {
    let lineStart = lineOffsets[lineOffsetsOffset + line];
    let lineEnd = min(lineOffsets[lineOffsetsOffset + line + 1u], ROW_COUNT);
    if (index + 1u < lineEnd) {
      nextRow = index + 1u;
    }${
      geometryType === 'polygons'
        ? ` else if (lineEnd - lineStart >= 2u) {
      nextRow = lineStart;
    }`
        : ''
    }
  }`;
}
