// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphVectorView,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  getStorageBindingElementLimit,
  getGraphViewChunks,
  validateGPUStorageBindingByteLength,
  validateCommandNodeBudget,
  validateMaximumCount,
  validateNonZeroFiniteNumber,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  planGraphChunkDispatch,
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges,
  type GraphChunkDispatchPlan
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateGraphChunkGroupKernelSupport} from '../../analysis-core/kernels/graph-chunk-groups';
import {
  getGPUSphereRadius,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import type {GPULineGeometryFields} from '../../analysis-core/contracts/index';
import {createLinePrefixNode} from '../line-segmentize/line-segmentize-kernels';
import {
  createLineLocateNode,
  createPackedScatterNodes,
  LINE_LOCATE_RESULT_STRIDE
} from './linear-referencing-kernels';

const OPERATION = 'GPULineLocate';

// Event placement uses cumulative line lengths and lateral distances.
const SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance'
});

/**
 * Largest per-chunk `eventLineIndices` row count of {@link GPULineLocate} on `device`: the packed result
 * table holds 6 `uint32` words (24 bytes) per event in one storage binding, so 5592405 per chunk on a default
 * device (128 MiB bindings).
 */
export function getGPULineLocateMaximumEventCount(device: Device): number {
  return getStorageBindingElementLimit(device, LINE_LOCATE_RESULT_STRIDE * 4);
}

/** Number of float32 elements in a `GPULineLocate` parameter buffer. */
export const GPU_LINE_LOCATE_PARAMETER_LENGTH = 4;

/** CPU description of the per-frame parameters of `GPULineLocate`. */
export type GPULineLocateSettings = {
  /** Multiplies every event measure before locating. Default 1. */
  measureScale?: number;
  /** Added to every scaled event measure, for example elapsed distance in an animation. Default 0. */
  measureOffset?: number;
};

/**
 * Packs `GPULineLocate` parameters into the 4-element float32 layout
 * `[measureScale, measureOffset, 0, 0]`.
 *
 * @param settings Settings to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If a value is not finite or `target` is too short.
 */
export function getGPULineLocateParameterValues(
  settings: GPULineLocateSettings,
  target: Float32Array = new Float32Array(GPU_LINE_LOCATE_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_LINE_LOCATE_PARAMETER_LENGTH) {
    throw new Error(`Line locate target must hold ${GPU_LINE_LOCATE_PARAMETER_LENGTH} elements`);
  }
  const measureScale = settings.measureScale ?? 1;
  const measureOffset = settings.measureOffset ?? 0;
  if (!Number.isFinite(measureScale) || !Number.isFinite(measureOffset)) {
    throw new Error('Line locate measureScale and measureOffset must be finite');
  }
  target.set([measureScale, measureOffset, 0, 0]);
  return target;
}

/** Kernel model of {@link GPULineLocate}, derived from its `spatialContext` (not a prop). */
export type GPULineLocateCoordinateSystem = 'planar' | 'spherical';

/**
 * Per-event outputs of {@link GPULineLocate}; `eventLineIndices.length` rows each. There is no
 * `status.count`: every row is one event, and `statuses` reports each event individually.
 */
export type GPULineLocateOutput<
  PositionRows extends
    | GraphDataView<'float32x2'>
    | GraphVectorView<'float32x2'> = GraphDataView<'float32x2'>,
  SegmentRows extends GraphDataView<'uint32'> | GraphVectorView<'uint32'> = GraphDataView<'uint32'>,
  TangentRows extends
    | GraphDataView<'float32x2'>
    | GraphVectorView<'float32x2'> = GraphDataView<'float32x2'>,
  AngleRows extends
    | GraphDataView<'float32'>
    | GraphVectorView<'float32'> = GraphDataView<'float32'>,
  StatusRows extends GraphDataView<'uint32'> | GraphVectorView<'uint32'> = GraphDataView<'uint32'>
> = {
  /** Located (and laterally offset) positions; NaN for invalid events. */
  positions: PositionRows;
  /** Segment index within the line, or `0xffffffff` for invalid events. */
  segmentIndices?: SegmentRows;
  /**
   * Unit direction of the segment (`(east, north)` compass components at the located point for
   * `'spherical'`), `(0, 0)` on single-vertex lines and zero-length segments. */
  tangents?: TangentRows;
  /**
   * Segment direction in degrees counter-clockwise from the +x axis (deck.gl `getAngle`
   * convention for planar or Web Mercator-like coordinates); `0` where the tangent is `(0, 0)`.
   */
  angles?: AngleRows;
  /** `GPU_LINE_LOCATE_STATUS` code per event. */
  statuses?: StatusRows;
  /** Optional per-vertex cumulative line measure, `positions.length` rows. */
  vertexMeasures?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPULineLocate}.
 *
 * Per-frame (no recompile): the contents of `parameters` and of every input buffer. Compile-time:
 * view lengths, the line and event counts, `measureMode`, and which optional views are present.
 */
export type GPULineLocateProps<
  PositionRows extends
    | GraphDataView<'float32x2'>
    | GraphVectorView<'float32x2'> = GraphDataView<'float32x2'>,
  SegmentRows extends GraphDataView<'uint32'> | GraphVectorView<'uint32'> = GraphDataView<'uint32'>,
  TangentRows extends
    | GraphDataView<'float32x2'>
    | GraphVectorView<'float32x2'> = GraphDataView<'float32x2'>,
  AngleRows extends
    | GraphDataView<'float32'>
    | GraphVectorView<'float32'> = GraphDataView<'float32'>,
  StatusRows extends GraphDataView<'uint32'> | GraphVectorView<'uint32'> = GraphDataView<'uint32'>,
  EventRows extends GraphDataView<'uint32'> | GraphVectorView<'uint32'> = GraphDataView<'uint32'>,
  MeasureRows extends
    | GraphDataView<'float32'>
    | GraphVectorView<'float32'> = GraphDataView<'float32'>,
  OffsetRows extends
    | GraphDataView<'float32'>
    | GraphVectorView<'float32'> = GraphDataView<'float32'>
> = GPULineGeometryFields & {
  /** Prefix for generated node and transient IDs. Defaults to `'line-locate'`. */
  id?: string;
  /**
   * Planar (default): Euclidean segments in position units. `{coordinateSpace: 'longitude-latitude',
   * metric: 'great-circle'}`: positions are longitude/latitude degrees, segments are great-circle
   * arcs, and measures, distances and offsets are in the context unit on its `sphereRadius`
   * (meters on the mean Earth radius by default). Replaces the former `coordinateSystem` and
   * `sphereRadius` props.
   */
  spatialContext?: GPUSpatialContext;
  /** Packed or chunked line indices; each chunk has at most {@link getGPULineLocateMaximumEventCount} rows (5592405 on a default device). */
  eventLineIndices: EventRows;
  /** Measure of each event: a distance from the line start, or a fraction of the line length. */
  eventMeasures: MeasureRows;
  /** Optional lateral offset of each event, positive to the left of the line direction. */
  eventOffsets?: OffsetRows;
  /** `'distance'` (default) or `'fraction'` of each line's length (turf `along` vs. `ST_LineInterpolatePoint`). */
  measureMode?: 'distance' | 'fraction';
  /**
   * Optional per-frame float32 view of {@link GPU_LINE_LOCATE_PARAMETER_LENGTH} elements written
   * with `getGPULineLocateParameterValues`; transforms every measure as `measure * scale + offset`.
   */
  parameters?: GraphDataView<'float32'>;
  /** Per-event outputs. */
  output: GPULineLocateOutput<PositionRows, SegmentRows, TangentRows, AngleRows, StatusRows>;
};

/**
 * Places events along lines by measure (turf `along`, PostGIS `ST_LineInterpolatePoint` and
 * `ST_LocateAlong` with offset): route events, mile markers, vehicles moving along routes.
 *
 * A per-line sequential prefix computes cumulative vertex measures (Neumaier-compensated,
 * deterministic). Each event clamps its measure into `[0, lineLength]` (status `clamped`), finds
 * the segment with an upper-bound binary search over the measures (so a measure that lands on a
 * vertex continues on the following segment, and zero-length segments are skipped), and
 * interpolates by measure. Lateral offsets move the point along the segment's left normal.
 * Events on empty or out-of-range lines, and events whose measure is NaN, are `invalid` with NaN
 * positions; an infinite measure clamps to a line end.
 *
 * A great-circle `spatialContext` measures in sphere-radius units (meters by default) and places
 * events on great-circle arcs (Geod `npts`, `ST_LineInterpolatePoint` on a sphere). Tangents are
 * the compass direction of the arc at the located point as `(east, north)`, offsets move along the
 * left compass bearing, and `angles` follow the same CCW-from-east convention. `measureMode:
 * 'fraction'` is shapely `line_interpolate_point(normalized=True)`. Precision is f32 (about 1 m).
 *
 * Planar by default: measures are Euclidean in position units. Event inputs and row outputs
 * may be chunked with matching ordered lengths. Geometry and vertex measures stay packed; the
 * shared measure prefix runs once. Each chunk dispatches its own workgroup count with runtime
 * lengths and binding offsets. Scratch is bounded by the largest chunk, not the total events.
 */
export class GPULineLocate<
  PositionRows extends
    | GraphDataView<'float32x2'>
    | GraphVectorView<'float32x2'> = GraphDataView<'float32x2'>,
  SegmentRows extends GraphDataView<'uint32'> | GraphVectorView<'uint32'> = GraphDataView<'uint32'>,
  TangentRows extends
    | GraphDataView<'float32x2'>
    | GraphVectorView<'float32x2'> = GraphDataView<'float32x2'>,
  AngleRows extends
    | GraphDataView<'float32'>
    | GraphVectorView<'float32'> = GraphDataView<'float32'>,
  StatusRows extends GraphDataView<'uint32'> | GraphVectorView<'uint32'> = GraphDataView<'uint32'>,
  EventRows extends GraphDataView<'uint32'> | GraphVectorView<'uint32'> = GraphDataView<'uint32'>,
  MeasureRows extends
    | GraphDataView<'float32'>
    | GraphVectorView<'float32'> = GraphDataView<'float32'>,
  OffsetRows extends
    | GraphDataView<'float32'>
    | GraphVectorView<'float32'> = GraphDataView<'float32'>
> implements GPUCommandNodeProducer
{
  /** Prefix for every node and transient ID. */
  readonly id: string;
  readonly estimatedCommandNodeCount: number;
  private readonly chunkPlan: GraphChunkDispatchPlan;
  /** Validated properties. */
  readonly props: GPULineLocateProps<
    PositionRows,
    SegmentRows,
    TangentRows,
    AngleRows,
    StatusRows,
    EventRows,
    MeasureRows,
    OffsetRows
  >;
  /** Resolved measure mode. */
  readonly measureMode: 'distance' | 'fraction';
  /** Resolved spatial context (planar unless given). */
  readonly spatialContext: GPUSpatialContext;
  /** Kernel model derived from `spatialContext`; not a prop. */
  private readonly coordinateSystem: GPULineLocateCoordinateSystem;
  /** Resolved sphere radius in the context's length unit (1 for planar). */
  private readonly sphereRadius: number;

  constructor(
    props: GPULineLocateProps<
      PositionRows,
      SegmentRows,
      TangentRows,
      AngleRows,
      StatusRows,
      EventRows,
      MeasureRows,
      OffsetRows
    >
  ) {
    this.id = props.id ?? 'line-locate';
    this.props = props;
    this.measureMode = props.measureMode ?? 'distance';
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext,
      SPATIAL_CONTEXT_SUPPORT
    );
    this.coordinateSystem =
      this.spatialContext.coordinateSpace === 'longitude-latitude' ? 'spherical' : 'planar';
    this.sphereRadius =
      this.coordinateSystem === 'spherical' ? getGPUSphereRadius(this.spatialContext) : 1;
    validateNonZeroFiniteNumber(this.id, 'spatialContext.sphereRadius', this.sphereRadius);
    const {id} = this;
    const {output} = props;
    if (this.measureMode !== 'distance' && this.measureMode !== 'fraction') {
      throw new Error(`${id} measureMode must be 'distance' or 'fraction'`);
    }
    validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 1) {
      throw new Error(`${id} needs at least one position`);
    }
    validateSinglePackedView(props.lineOffsets, ['uint32'], `${id} lineOffsets`);
    if (props.lineOffsets.length < 2) {
      throw new Error(`${id} lineOffsets needs two rows`);
    }
    for (const chunk of getGraphViewChunks(props.eventLineIndices))
      validateSinglePackedView(chunk, ['uint32'], `${id} eventLineIndices`);
    const eventCount = props.eventLineIndices.length;
    if (eventCount < 1) {
      throw new Error(`${id} needs at least one event`);
    }
    for (const [name, view] of [
      ['eventMeasures', props.eventMeasures],
      ['eventOffsets', props.eventOffsets]
    ] as const) {
      if (view) {
        for (const chunk of getGraphViewChunks(view))
          validateSinglePackedView(chunk, ['float32'], `${id} ${name}`);
        if (view.length !== eventCount) {
          throw new Error(`${id} ${name} length must equal eventLineIndices length`);
        }
      }
    }
    if (props.parameters) {
      validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
      if (props.parameters.length < GPU_LINE_LOCATE_PARAMETER_LENGTH) {
        throw new Error(
          `${id} parameters must hold ${GPU_LINE_LOCATE_PARAMETER_LENGTH} float32 values`
        );
      }
    }
    const formats = {
      positions: 'float32x2',
      segmentIndices: 'uint32',
      tangents: 'float32x2',
      angles: 'float32',
      statuses: 'uint32',
      vertexMeasures: 'float32'
    } as const;
    for (const [name, view] of Object.entries(output) as [
      keyof typeof formats,
      GraphDataView | GraphVectorView | undefined
    ][]) {
      if (!view) {
        continue;
      }
      if (!(name in formats)) {
        throw new Error(`${id} output.${name} is not a line-locate column`);
      }
      if (name === 'vertexMeasures')
        validateSinglePackedView(view, ['float32'], `${id} output.vertexMeasures`);
      else
        for (const chunk of getGraphViewChunks(view))
          validateSinglePackedView(chunk, [formats[name]], `${id} output.${name}`);
      const length = name === 'vertexMeasures' ? props.positions.length : eventCount;
      if (view.length !== length) {
        throw new Error(`${id} output.${name} must hold ${length} rows`);
      }
    }
    const {vertexMeasures: _sharedVertexMeasures, ...rowOutputs} = output;
    this.chunkPlan = planGraphChunkDispatch(id, {
      eventLineIndices: props.eventLineIndices,
      eventMeasures: props.eventMeasures,
      eventOffsets: props.eventOffsets,
      outputPositions: rowOutputs.positions,
      segmentIndices: rowOutputs.segmentIndices,
      tangents: rowOutputs.tangents,
      angles: rowOutputs.angles,
      statuses: rowOutputs.statuses
    });
    validateDisjointWritableViews(id, output);
    this.estimatedCommandNodeCount = 1 + 2 * this.chunkPlan.nonemptyChunkCount;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateGraphOutputsDisjointFromInputs(id, Object.values(output), [
      props.positions,
      props.lineOffsets,
      props.eventLineIndices,
      props.eventMeasures,
      props.eventOffsets,
      props.parameters
    ]);
  }

  /** Returns the measure prefix node, the locate node and the scatter node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {output} = props;
    validateGraphChunkBindingRanges(graph, id, {
      positions: props.positions,
      lineOffsets: props.lineOffsets,
      eventLineIndices: props.eventLineIndices,
      eventMeasures: props.eventMeasures,
      eventOffsets: props.eventOffsets,
      parameters: props.parameters,
      outputPositions: output.positions,
      segmentIndices: output.segmentIndices,
      tangents: output.tangents,
      angles: output.angles,
      statuses: output.statuses,
      vertexMeasures: output.vertexMeasures
    });
    validateGraphChunkGroupKernelSupport(
      graph,
      id,
      6 + Number(Boolean(props.eventOffsets)) + Number(Boolean(props.parameters))
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'vertexMeasures scratch',
      props.positions.length * 4,
      'reduce the packed geometry row count'
    );
    const maximumChunkLength = this.chunkPlan.chunks.reduce(
      (maximum, chunk) => Math.max(maximum, chunk.length),
      0
    );
    validateMaximumCount(
      id,
      'eventLineIndices chunk rows',
      maximumChunkLength,
      getGPULineLocateMaximumEventCount(graph.device)
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'results scratch',
      maximumChunkLength * LINE_LOCATE_RESULT_STRIDE * 4,
      'reduce the event chunk row count'
    );
    const vertexMeasures =
      output.vertexMeasures ??
      createTransientView(graph, `${id}-vertex-measures`, 'float32', props.positions.length);
    const results = createTransientView(
      graph,
      `${id}-results`,
      'uint32',
      maximumChunkLength * LINE_LOCATE_RESULT_STRIDE
    );
    const nodes: GPUCommandNode<Parameters>[] = [
      createLinePrefixNode<Parameters>(graph, {
        id: `${id}-vertex-measures`,
        operation: OPERATION,
        positions: props.positions,
        lineOffsets: props.lineOffsets,
        coordinateSystem: this.coordinateSystem,
        radius: this.coordinateSystem === 'spherical' ? this.sphereRadius : 1,
        rowMeasures: vertexMeasures
      })
    ];
    for (const chunk of this.chunkPlan.chunks) {
      if (chunk.length === 0) continue;
      const suffix = this.chunkPlan.chunks.length === 1 ? '' : `-chunk-${chunk.chunkIndex}`;
      nodes.push(
        createLineLocateNode<Parameters>(graph, {
          id: `${id}${suffix}-locate`,
          operation: OPERATION,
          positions: props.positions,
          lineOffsets: props.lineOffsets,
          vertexMeasures,
          eventLineIndices: chunk.views['eventLineIndices'] as GraphDataView<'uint32'>,
          eventMeasures: chunk.views['eventMeasures'] as GraphDataView<'float32'>,
          eventOffsets: chunk.views['eventOffsets'] as GraphDataView<'float32'> | undefined,
          parameters: props.parameters,
          measureMode: this.measureMode,
          coordinateSystem: this.coordinateSystem,
          sphereRadius: this.sphereRadius,
          results
        }),
        ...createPackedScatterNodes<Parameters>(graph, {
          id: `${id}${suffix}-output`,
          operation: OPERATION,
          rowCount: chunk.length,
          stride: LINE_LOCATE_RESULT_STRIDE,
          results,
          columns: [
            {
              name: 'positions',
              view: chunk.views['outputPositions'] as GraphDataView<'float32x2'> | undefined,
              type: 'f32',
              statement: `positions[positionsOffset + 2u * index] = bitcast<f32>(results[base]);
  positions[positionsOffset + 2u * index + 1u] = bitcast<f32>(results[base + 1u]);`
            },
            {
              name: 'segmentIndices',
              view: chunk.views['segmentIndices'] as GraphDataView<'uint32'> | undefined,
              type: 'u32',
              statement: 'segmentIndices[segmentIndicesOffset + index] = results[base + 2u];'
            },
            {
              name: 'tangents',
              view: chunk.views['tangents'] as GraphDataView<'float32x2'> | undefined,
              type: 'f32',
              statement: `tangents[tangentsOffset + 2u * index] = bitcast<f32>(results[base + 3u]);
  tangents[tangentsOffset + 2u * index + 1u] = bitcast<f32>(results[base + 4u]);`
            },
            {
              name: 'angles',
              view: chunk.views['angles'] as GraphDataView<'float32'> | undefined,
              type: 'f32',
              statement: `{
    let tangent = vec2<f32>(bitcast<f32>(results[base + 3u]), bitcast<f32>(results[base + 4u]));
    // atan2(0, 0) is undefined in WGSL: degenerate tangents report angle 0.
    angles[anglesOffset + index] = select(
      guardedAtan2(tangent.y, tangent.x) * 57.2957795130823,
      0.0,
      tangent.x == 0.0 && tangent.y == 0.0
    );
  }`
            },
            {
              name: 'statuses',
              view: chunk.views['statuses'] as GraphDataView<'uint32'> | undefined,
              type: 'u32',
              statement: 'statuses[statusesOffset + index] = results[base + 5u];'
            }
          ]
        })
      );
    }
    return nodes;
  }
}
