// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  resolveGeometryUint32StatusView,
  validateGeometryRetainedViews
} from '../geometry-edit/geometry-storage-validation';
import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateCapacity,
  validateMaximumCount,
  validateRequirement,
  validateGPUStorageBindingByteLength,
  validateCommandNodeBudget,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {
  GPUGeneratedGeometryPort,
  GPUPackedGeometryPort,
  GPUPolygonGeometryPort,
  GPULineGeometryFields,
  GPUPolygonGeometryFields
} from '../../analysis-core/contracts/index';
import {
  createLineGeometryCountNode,
  getLineGeometryOutputViews,
  validateLineGeometryOutput
} from '../line-segmentize/line-geometry-output';
import type {GPULineGeometryOutput} from '../line-segmentize/line-segmentize-types';
import {
  createLineClipCountNode,
  createLineClipEmitNode,
  createLineClipPublishNode,
  createLineTailNode,
  createPolygonPublishNode,
  createPolygonStageCountNode,
  createPolygonStageEmitNode,
  createPolygonStageOffsetsNode,
  GPU_RECTANGLE_CLIP_PARAMETER_LENGTH
} from './rectangle-clip-kernels';
import {
  GPU_LINEAR_EDGE_MODELS,
  resolveGPUSpatialContext,
  type GPUSpatialContext,
  type GPUSpatialContextSupport
} from '../../analysis-core/contracts/spatial-context';

import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';
import {
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges
} from '../../analysis-core/kernels/graph-chunk-dispatch';

import {validateGeometryFloat32Setting} from '../geometry-edit/float32-setting-validation';

const OPERATION = 'GPURectangleClip';
/** Each clip input vertex emits at most two vertices in a uint32 count scan. */
const MAXIMUM_VERTEX_COUNT = 0x7fffffff;
/** Four stages with at most seven upstream scan nodes each, plus publication and feature IDs. */
const MAXIMUM_COMMAND_NODE_COUNT = 43;

// The clip is the usual straight-edge clip in the plane of the coordinates, which for degrees
// means edges are straight in longitude and latitude (the linear edge model).
const SPATIAL_CONTEXT_SUPPORT: GPUSpatialContextSupport = Object.freeze({
  coordinateSpaces: Object.freeze(['planar', 'longitude-latitude'] as const),
  edgeModels: GPU_LINEAR_EDGE_MODELS
});

export {GPU_RECTANGLE_CLIP_PARAMETER_LENGTH};

/** CPU description of the per-frame clip rectangle of {@link GPURectangleClip}. */
export type GPURectangleClipSettings = {
  /** Left edge. */
  minX: number;
  /** Bottom edge. */
  minY: number;
  /** Right edge, at least `minX`. */
  maxX: number;
  /** Top edge, at least `minY`. */
  maxY: number;
};

/**
 * Packs {@link GPURectangleClipSettings} into the 4-element float32 layout
 * `[minX, minY, maxX, maxY]`. Write the result into a `GPUParameterBuffer` between encodings to
 * move the clip rectangle (for example the viewport) without recompiling.
 *
 * @param settings Settings to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If a value is not finite, the rectangle is inverted, or `target` is too short.
 */
export function getGPURectangleClipParameterValues(
  settings: GPURectangleClipSettings,
  target: Float32Array = new Float32Array(GPU_RECTANGLE_CLIP_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_RECTANGLE_CLIP_PARAMETER_LENGTH) {
    throw new Error(
      `Rectangle clip target must hold ${GPU_RECTANGLE_CLIP_PARAMETER_LENGTH} elements`
    );
  }
  const {minX, minY, maxX, maxY} = settings;
  for (const [name, value] of Object.entries({minX, minY, maxX, maxY})) {
    validateGeometryFloat32Setting(OPERATION, name, value);
  }
  if (maxX < minX || maxY < minY) {
    throw new Error('Rectangle clip requires maxX >= minX and maxY >= minY');
  }
  target.set([minX, minY, maxX, maxY]);
  return target;
}

/** Feature cardinality of polygon clipping, separate from the existing ring status. */
export type GPURectangleClipPolygonFeatureStatus = {
  count: GraphDataView<'uint32'>;
  requiredCount?: GraphDataView<'uint32'>;
};

/** Canonical feature status, with the original fields retained as deprecated aliases. */
export type GPURectangleClipPolygonFeatures = {
  sourceRows: GraphDataView<'uint32'>;
} & (
  | {
      status: GPURectangleClipPolygonFeatureStatus;
      /** @deprecated Use status.count. If both are provided they must identify the same view. */
      count?: GPURectangleClipPolygonFeatureStatus['count'];
      /** @deprecated Use status.requiredCount. */
      requiredCount?: GPURectangleClipPolygonFeatureStatus['requiredCount'];
    }
  | {
      status?: undefined;
      /** @deprecated Use status.count. */
      count: GPURectangleClipPolygonFeatureStatus['count'];
      /** @deprecated Use status.requiredCount. */
      requiredCount?: GPURectangleClipPolygonFeatureStatus['requiredCount'];
    }
);

/** Existing ring/line layout, with optional feature-level polygon publication. */
export type GPURectangleClipOutput = Omit<
  GPULineGeometryOutput,
  'sourceRows' | 'measures' | 'status'
> & {
  sourceRows?: never;
  measures?: never;
  status: Omit<GPULineGeometryOutput['status'], 'approximation'> & {approximation?: never};
  /**
   * Polygons only. One source row per input feature, including empty clipped features. Counts
   * here count features; the existing top-level status continues to count rings. The hierarchy
   * is unchanged, so getPolygonOutput() reuses input featureOffsets and polygonOffsets and the
   * clipped positions/ringOffsets without copies or readback.
   */
  polygonFeatures?: GPURectangleClipPolygonFeatures;
};

/**
 * Properties for {@link GPURectangleClip}.
 *
 * Per-frame (no recompile): the rectangle in `parameters` and every input buffer. Compile-time:
 * view lengths, `geometryType`, and the output capacities.
 */
export type GPURectangleClipProps = (
  | ({
      /** Open lines clipped with Liang-Barsky. Takes `positions` and `lineOffsets`. */
      geometryType: 'lines';
    } & GPULineGeometryFields)
  | ({
      /**
       * Polygon rings clipped with Sutherland-Hodgman. Takes the four-level polygon layout
       * (`positions`, `featureOffsets`, `polygonOffsets`, `ringOffsets`). Rings close implicitly.
       */
      geometryType: 'polygons';
    } & GPUPolygonGeometryFields)
) & {
  /** Prefix for generated node and transient IDs. Defaults to `'rectangle-clip'`. */
  id?: string;
  /**
   * Default: native planar coordinates. A longitude-latitude context is accepted: the rectangle
   * is read in the same coordinates as the input (degrees) and edges are straight in longitude
   * and latitude, as in turf `bboxClip`. There is no antimeridian handling: a rectangle crossing
   * longitude 180 is not supported. A context declaring great-circle edges throws.
   */
  spatialContext?: GPUSpatialContext;
  /** Per-frame packed float32 view written with {@link getGPURectangleClipParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /**
   * Bounded output. `positions.length` is the vertex capacity. `output.status.count` and
   * `requiredCount` count output lines, or for `'polygons'` output rings (the input ring count,
   * since every ring is kept, possibly empty); `vertexCount` and `requiredVertexCount` count
   * vertices. For polygons the vertex requirement is the last stage's total, a lower bound when an
   * earlier stage overflowed. For `'polygons'`,
   * `output.lineOffsets` holds the output ring offsets and must have the length of the input
   * `ringOffsets` (one output ring per input ring, possibly empty), so the input `featureOffsets`
   * and `polygonOffsets` describe the output rings unchanged; `sourceLineIndices`, `sourceRows` and
   * `measures` are not supported. For
   * `'lines'`, `lineOffsets.length - 1` is the output line capacity; `sourceLineIndices` gives each
   * output line's input line.
   */
  output: GPURectangleClipOutput;
};

/**
 * Clips lines and polygon rings to an axis-aligned rectangle (turf `bboxClip`, Shapely
 * `clip_by_rect`, PostGIS `ST_ClipByBox2D`), with the rectangle read per frame, so a viewport clip
 * runs without recompiling.
 *
 * **Lines** use Liang-Barsky per segment. Surviving segments are joined back into lines: a
 * line piece continues across a vertex when both neighboring segments survive unclipped there,
 * and a line that leaves and re-enters the rectangle becomes several output lines. Output
 * vertices and lines come from scans, so order follows the input; single-vertex lines and
 * segments that only touch the rectangle at a point are dropped.
 *
 * **Polygons** use Sutherland-Hodgman against the four edges in turn (left, right, bottom, top),
 * each stage a count, scan and emit over the vertices of the previous stage, so a ring keeps its
 * position in `ringOffsets` (possibly with zero vertices). Concave polygons that leave and
 * re-enter the rectangle keep zero-width bridges along the clip edges, as always with
 * Sutherland-Hodgman: the filled area is exact, the outline may trace along the border. Rings are
 * clipped independently, so a polygon with holes keeps working under any hole rule.
 *
 * Output is bounded by `output.positions.length` (all four stages share that capacity); when it
 * overflows, `overflow` is 1 and later vertices and lines are truncated while offsets stay
 * valid. Deterministic, no atomics. Precision: f32. Polygon intersections lie on the rectangle
 * edges exactly in the clipped coordinate (the bound itself is written) and within an f32 rounding
 * of the true crossing in the other; the crossing is interpolated from the lexicographically
 * smaller endpoint of the edge, so two polygons that share an edge get bit-identical clip points
 * and a clipped coverage still matches vertex for vertex. Line clip points are `a + (b - a) * t`
 * from the segment's own start: both coordinates are within an f32 rounding of the true crossing,
 * so the clipped coordinate can be one f32 step off the rectangle edge (about 1 in 100 random
 * segments), and a segment clipped from its two directions can differ by one step.
 */
export class GPURectangleClip implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPURectangleClipProps;
  /** Resolved spatial context (planar unless given). */
  readonly spatialContext: GPUSpatialContext;
  /** Line offsets, or the polygon ring offsets, as read by the kernels. */
  readonly boundaryOffsets: GraphDataView<'uint32'>;
  readonly estimatedCommandNodeCount = MAXIMUM_COMMAND_NODE_COUNT;
  /** Flat kernel bindings resolved once from canonical status and compatibility aliases. */
  private readonly polygonFeatures?: {
    sourceRows: GraphDataView<'uint32'>;
    count: GraphDataView<'uint32'>;
    requiredCount?: GraphDataView<'uint32'>;
  };

  constructor(props: GPURectangleClipProps) {
    this.id = props.id ?? 'rectangle-clip';
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext,
      SPATIAL_CONTEXT_SUPPORT
    );
    this.props = props;
    const {id} = this;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    this.boundaryOffsets =
      props.geometryType === 'polygons' ? props.ringOffsets : props.lineOffsets;
    if (props.geometryType !== 'lines' && props.geometryType !== 'polygons') {
      throw new Error(`${id} geometryType must be 'lines' or 'polygons'`);
    }
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
    if (props.parameters.length < GPU_RECTANGLE_CLIP_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_RECTANGLE_CLIP_PARAMETER_LENGTH} float32 values`
      );
    }
    validateLineGeometryOutput(id, props.output, props.output.lineOffsets.length - 1);
    validateMaximumCount(id, 'positions.length', props.positions.length, MAXIMUM_VERTEX_COUNT);
    validateMaximumCount(
      id,
      'output.positions.length',
      props.output.positions.length,
      MAXIMUM_VERTEX_COUNT
    );
    const featureOutput = props.output.polygonFeatures;
    validateRequirement(
      id,
      !featureOutput || props.geometryType === 'polygons',
      'output.polygonFeatures requires geometryType polygons'
    );
    if (featureOutput && props.geometryType === 'polygons') {
      const count = resolveGeometryUint32StatusView(
        id,
        'output.polygonFeatures.status.count',
        featureOutput.status?.count,
        featureOutput.count
      );
      const requiredCount = resolveGeometryUint32StatusView(
        id,
        'output.polygonFeatures.status.requiredCount',
        featureOutput.status?.requiredCount,
        featureOutput.requiredCount
      );
      validateRequirement(
        id,
        count !== undefined,
        'output.polygonFeatures.status.count is required'
      );
      this.polygonFeatures = {sourceRows: featureOutput.sourceRows, count, requiredCount};
      const features = this.polygonFeatures;
      validateCapacity(id, 'featureOffsets.length', props.featureOffsets.length, 1);
      validateCapacity(id, 'polygonOffsets.length', props.polygonOffsets.length, 1);
      for (const [name, view] of Object.entries(features)) {
        if (!view) continue;
        const field = name === 'sourceRows' ? name : `status.${name}`;
        validateSinglePackedView(view, ['uint32'], `${id} output.polygonFeatures.${field}`);
        validateRequirement(
          id,
          view.length === (name === 'sourceRows' ? props.featureOffsets.length - 1 : 1),
          `output.polygonFeatures.${field} length must match ${name === 'sourceRows' ? 'feature count' : 'one scalar'}`
        );
      }
    }
    this.validateWritableOutputs();
    if (props.geometryType === 'polygons') {
      if (props.output.lineOffsets.length !== this.boundaryOffsets.length) {
        throw new Error(`${id} output.lineOffsets must hold ${this.boundaryOffsets.length} rows`);
      }
      for (const name of ['sourceLineIndices', 'sourceRows', 'measures'] as const) {
        if ((props.output as GPULineGeometryOutput)[name]) {
          throw new Error(`${id} output.${name} is not supported for polygons`);
        }
      }
    } else {
      for (const name of ['sourceRows', 'measures'] as const) {
        if ((props.output as GPULineGeometryOutput)[name]) {
          throw new Error(`${id} output.${name} is not supported`);
        }
      }
    }
    const inputs = [props.positions, this.boundaryOffsets, props.parameters];
    // Line emission reads the original geometry. Polygon emission's last stage reads private
    // intermediate geometry and caller parameters; status publication reads only private scratch.
    validateGraphOutputsDisjointFromInputs(
      id,
      props.geometryType === 'lines'
        ? [props.output.positions, props.output.lineOffsets, props.output.sourceLineIndices]
        : [props.output.positions],
      props.geometryType === 'lines' ? inputs : [props.parameters]
    );
    validateGeometryRetainedViews(
      id,
      [
        ...inputs,
        ...(props.geometryType === 'polygons' ? [props.featureOffsets, props.polygonOffsets] : [])
      ],
      this.getOutputViews()
    );
  }

  /** Standard generated-polygon port, preserving empty features and the input hierarchy. */
  getPolygonOutput(): GPUGeneratedGeometryPort<GPUPackedGeometryPort<GPUPolygonGeometryPort>> {
    const {props, id} = this;
    const features = this.polygonFeatures;
    if (props.geometryType !== 'polygons' || !features)
      throw new Error(`${id} getPolygonOutput requires output.polygonFeatures`);
    return {
      geometry: {
        kind: 'polygons',
        positions: props.output.positions,
        ringOffsets: props.output.lineOffsets,
        featureOffsets: props.featureOffsets,
        polygonOffsets: props.polygonOffsets
      },
      sourceIds: features.sourceRows,
      status: {
        ...props.output.status,
        count: features.count,
        requiredCount: features.requiredCount,
        ringCount: props.output.status.count,
        requiredRingCount: props.output.status.requiredCount
      }
    };
  }

  private getOutputViews(): (GraphDataView | undefined)[] {
    return [
      ...getLineGeometryOutputViews(this.props.output),
      ...Object.values(this.polygonFeatures ?? {}).filter(view => view && view.length > 0)
    ];
  }

  private validateWritableOutputs(): void {
    const {props, id} = this;
    const fields = this.getOutputViews().filter((view): view is GraphDataView =>
      Boolean(view?.length)
    );
    // A bounded number of fields: compare logical ranges across passes, aligned ranges within each pass.
    for (let first = 0; first < fields.length; first++) {
      const firstView = fields[first];
      const firstBuffer = firstView.buffer.defaultBuffer
        ? getCoreDefaultBuffer(firstView.buffer.defaultBuffer)
        : firstView.buffer;
      for (let second = first + 1; second < fields.length; second++) {
        const secondView = fields[second];
        const secondBuffer = secondView.buffer.defaultBuffer
          ? getCoreDefaultBuffer(secondView.buffer.defaultBuffer)
          : secondView.buffer;
        validateRequirement(
          id,
          firstBuffer !== secondBuffer ||
            firstView.byteOffset + firstView.length * firstView.rowByteLength <=
              secondView.byteOffset ||
            secondView.byteOffset + secondView.length * secondView.rowByteLength <=
              firstView.byteOffset,
          'output logical byte ranges must not overlap'
        );
      }
    }
    const {output} = props;
    if (props.geometryType === 'polygons') {
      validateDisjointWritableViews(id, {
        overflow: output.status.overflow,
        vertexCount: output.status.vertexCount,
        requiredVertexCount: output.status.requiredVertexCount
      });
      validateDisjointWritableViews(id, {
        count: output.status.count,
        requiredCount: output.status.requiredCount
      });
    } else {
      validateDisjointWritableViews(id, {
        positions: output.positions,
        lineOffsets: output.lineOffsets,
        sourceLineIndices: output.sourceLineIndices
      });
      validateDisjointWritableViews(id, {
        lineOffsets: output.lineOffsets,
        sourceLineIndices: output.sourceLineIndices,
        count: output.status.count,
        requiredCount: output.status.requiredCount,
        vertexCount: output.status.vertexCount,
        requiredVertexCount: output.status.requiredVertexCount
      });
    }
    if (this.polygonFeatures)
      validateDisjointWritableViews(id, {
        count: this.polygonFeatures.count,
        requiredCount: this.polygonFeatures.requiredCount,
        sourceRows: this.polygonFeatures.sourceRows.length
          ? this.polygonFeatures.sourceRows
          : undefined
      });
  }

  /** Returns the line nodes or the four polygon stages followed by the publish node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphChunkBindingRanges(graph, id, {
      positions: props.positions,
      offsets: this.boundaryOffsets,
      parameters: props.parameters,
      ...Object.fromEntries(this.getOutputViews().map((view, index) => [`output.${index}`, view]))
    });
    const capacity = props.output.positions.length;
    const scratch =
      props.geometryType === 'polygons'
        ? {
            stageFlags: 16,
            stagePositions: capacity * 8,
            stageOffsets: this.boundaryOffsets.length * 4,
            countsAndScan: Math.max(capacity, props.positions.length) * 4
          }
        : {totals: 16, countsAndScans: props.positions.length * 4};
    for (const [name, byteLength] of Object.entries(scratch)) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        byteLength,
        'reduce geometry or output capacity'
      );
      validateCapacity(id, `${name} byteLength`, byteLength, 0, graph.device.limits.maxBufferSize);
    }
    const nodes =
      props.geometryType === 'lines' ? this._getLineNodes(graph) : this._getPolygonNodes(graph);
    const features = this.polygonFeatures;
    if (features) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-polygon-features`,
          operation: OPERATION,
          invocationCount: Math.max(features.sourceRows.length, 1),
          bindings: [
            {name: 'count', view: features.count, type: 'u32', access: 'read_write'},
            ...(features.sourceRows.length
              ? [
                  {
                    name: 'sourceRows',
                    view: features.sourceRows,
                    type: 'u32' as const,
                    access: 'read_write' as const
                  }
                ]
              : []),
            ...(features.requiredCount
              ? [
                  {
                    name: 'requiredCount',
                    view: features.requiredCount,
                    type: 'u32' as const,
                    access: 'read_write' as const
                  }
                ]
              : [])
          ],
          body: `${features.sourceRows.length ? 'sourceRows[sourceRowsOffset + index] = index;' : ''}
if (index == 0u) { count[countOffset] = ${features.sourceRows.length}u;
${features.requiredCount ? `requiredCount[requiredCountOffset] = ${features.sourceRows.length}u;` : ''}
}`
        })
      );
    }
    return nodes;
  }

  private _getLineNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {output} = props;
    const rowCount = props.positions.length;
    const capacity = output.positions.length;
    const lineCapacity = output.lineOffsets.length - 1;
    const inputs = {
      operation: OPERATION,
      positions: props.positions,
      lineOffsets: this.boundaryOffsets,
      parameters: props.parameters
    };
    const counts = createTransientView(graph, `${id}-counts`, 'uint32', rowCount);
    const startFlags = createTransientView(graph, `${id}-start-flags`, 'uint32', rowCount);
    const starts = createTransientView(graph, `${id}-starts`, 'uint32', rowCount);
    const lineIndices = createTransientView(graph, `${id}-line-indices`, 'uint32', rowCount);
    const totals = createTransientView(graph, `${id}-totals`, 'uint32', 4);
    return [
      createLineClipCountNode<Parameters>(graph, {
        ...inputs,
        id: `${id}-count`,
        counts,
        startFlags
      }),
      ...new GPUScan({
        id: `${id}-vertex-scan`,
        input: counts,
        output: starts,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      ...new GPUScan({
        id: `${id}-line-scan`,
        input: startFlags,
        output: lineIndices,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createLineClipEmitNode<Parameters>(graph, {
        ...inputs,
        id: `${id}-emit`,
        starts,
        lineIndices,
        outputPositions: output.positions,
        outputLineOffsets: output.lineOffsets,
        outputSourceLineIndices: output.sourceLineIndices
      }),
      createLineClipPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        capacity,
        lineCapacity,
        starts,
        counts,
        lineIndices,
        startFlags,
        totals,
        overflow: output.status.overflow
      }),
      createLineTailNode<Parameters>(graph, {
        id: `${id}-line-tail`,
        operation: OPERATION,
        totals,
        lineOffsets: output.lineOffsets,
        sourceLineIndices: output.sourceLineIndices,
        count: output.status.count,
        requiredCount: output.status.requiredCount,
        vertexCount: output.status.vertexCount,
        requiredVertexCount: output.status.requiredVertexCount
      })
    ];
  }

  private _getPolygonNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {output} = props;
    const capacity = output.positions.length;
    const ringCount = this.boundaryOffsets.length - 1;
    const stageFlags = createTransientView(graph, `${id}-stage-totals`, 'uint32', 4);
    const positionBuffers = [
      createTransientView(graph, `${id}-positions-a`, 'float32x2', capacity),
      createTransientView(graph, `${id}-positions-b`, 'float32x2', capacity)
    ];
    const offsetBuffers = [
      createTransientView(graph, `${id}-offsets-a`, 'uint32', ringCount + 1),
      createTransientView(graph, `${id}-offsets-b`, 'uint32', ringCount + 1)
    ];
    const nodes: GPUCommandNode<Parameters>[] = [];
    let inputPositions = props.positions;
    let inputOffsets = this.boundaryOffsets;
    for (let stage = 0; stage < 4; stage++) {
      const isLast = stage === 3;
      const outputPositions = isLast ? output.positions : positionBuffers[stage % 2];
      const outputOffsets = isLast ? output.lineOffsets : offsetBuffers[stage % 2];
      const rows = inputPositions.length;
      const counts = createTransientView(graph, `${id}-counts-${stage}`, 'uint32', rows);
      const starts = createTransientView(graph, `${id}-starts-${stage}`, 'uint32', rows);
      const stageInputs = {
        operation: OPERATION,
        boundary: stage,
        positions: inputPositions,
        ringOffsets: inputOffsets,
        parameters: props.parameters
      };
      nodes.push(
        createPolygonStageCountNode<Parameters>(graph, {
          ...stageInputs,
          id: `${id}-count-${stage}`,
          counts
        }),
        ...new GPUScan({
          id: `${id}-scan-${stage}`,
          input: counts,
          output: starts,
          mode: 'exclusive'
        }).getCommandNodes(graph),
        createPolygonStageEmitNode<Parameters>(graph, {
          ...stageInputs,
          id: `${id}-emit-${stage}`,
          starts,
          outputPositions
        }),
        createPolygonStageOffsetsNode<Parameters>(graph, {
          id: `${id}-offsets-${stage}`,
          operation: OPERATION,
          capacity,
          inputOffsets,
          starts,
          counts,
          outputOffsets,
          stageFlags,
          stage
        })
      );
      inputPositions = outputPositions;
      inputOffsets = outputOffsets;
    }
    nodes.push(
      createPolygonPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        capacity,
        stageFlags,
        overflow: output.status.overflow,
        vertexCount: output.status.vertexCount,
        requiredVertexCount: output.status.requiredVertexCount
      }),
      createLineGeometryCountNode<Parameters>(graph, {
        id,
        operation: OPERATION,
        output,
        lineCapacity: ringCount
      })
    );
    return nodes;
  }
}
