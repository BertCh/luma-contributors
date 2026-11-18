// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {toRadians} from '@math.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphVectorView,
  validatePackedView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  getGraphViewChunks,
  validateCapacity,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GRAPH_CHUNK_MAXIMUM_ROW_COUNT,
  planGraphChunkDispatch,
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges,
  validateGraphChunkDispatchBindings,
  type GraphChunkDispatchPlan
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {createAffineChunkNodes} from './gpu-affine-transform-chunks';
import {GPUGeometryMeasures} from '../geometry-measures/gpu-geometry-measures';
import {GEOMETRY_STATS_STRIDE} from '../geometry-measures/geometry-measures-kernels';
import type {GPUPolygonGeometryFields} from '../../analysis-core/contracts/index';
import {validateGPUMultipartGeometryPort} from '../../analysis-core/contracts/index';
import {createAffineTransformNode} from './geometry-edit-kernels';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';

const OPERATION = 'GPUAffineTransform';

/** Number of float32 elements in a `GPUAffineTransform` parameter buffer. */
export const GPU_AFFINE_TRANSFORM_PARAMETER_LENGTH = 12;

/** Per-feature origin kinds that {@link GPUAffineTransform} can compute on the GPU. */
export type GPUAffineTransformOrigin = 'center' | 'centroid';

/** CPU description of the per-frame settings of `GPUAffineTransform`. */
export type GPUAffineTransformSettings = {
  /** Rotation angle, counter-clockwise in a y-up system. Degrees unless `useRadians`. Default 0. */
  rotate?: number;
  /** Interpret `rotate`, `skew` as radians. Default false (degrees, as shapely). */
  useRadians?: boolean;
  /** Scale factors `[x, y]`, or one number for both. Default 1. */
  scale?: number | readonly [number, number];
  /** Shear angles `[x, y]` (`shapely.affinity.skew`: `xs`, `ys`). Default 0. */
  skew?: readonly [number, number];
  /** Translation `[x, y]` applied last. Default `[0, 0]`. */
  translate?: readonly [number, number];
  /**
   * Fixed point of the scale, skew and rotation: a coordinate, the per-feature bounding box
   * `'center'`, or the per-feature `'centroid'` of the input geometry. Default `[0, 0]` (shapely
   * defaults to `'center'`; ask for it explicitly). `'center'` and `'centroid'` only work when the
   * contributor was created with the matching entry in `origins`; otherwise they act as `[0, 0]`.
   */
  origin?: 'center' | 'centroid' | readonly [number, number];
};

/**
 * Packs `GPUAffineTransform` settings into the 12-element float32 layout
 * `[a, b, d, e, translateX, translateY, originX, originY, originMode, 0, 0, 0]` where
 * `originMode` is 0 (point), 1 (`'center'`) or 2 (`'centroid'`), and every vertex maps to
 * `[a b; d e] * (p - origin) + origin + translate`. The matrix is the composition shapely applies
 * when calling `scale`, then `skew`, then `rotate`, then `translate` about the same origin:
 * `R * K * S` with `S = diag(scale)`, `K = [1 tan(xs); tan(ys) 1]`, `R = [cos -sin; sin cos]`,
 * evaluated in f64 before narrowing to f32.
 *
 * @param settings Settings to encode.
 * @param target Optional destination of at least 12 elements.
 * @throws If a number is not finite, or `target` is too short.
 */
export function getGPUAffineTransformParameterValues(
  settings: GPUAffineTransformSettings = {},
  target: Float32Array = new Float32Array(GPU_AFFINE_TRANSFORM_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_AFFINE_TRANSFORM_PARAMETER_LENGTH) {
    throw new Error(
      `Affine transform target must hold ${GPU_AFFINE_TRANSFORM_PARAMETER_LENGTH} elements`
    );
  }
  const angleScale = settings.useRadians ? 1 : toRadians(1);
  const rotate = (settings.rotate ?? 0) * angleScale;
  const scale = settings.scale ?? 1;
  const [scaleX, scaleY] = typeof scale === 'number' ? [scale, scale] : scale;
  const [skewX, skewY] = settings.skew ?? [0, 0];
  const [translateX, translateY] = settings.translate ?? [0, 0];
  const origin = settings.origin ?? [0, 0];
  const originPoint = typeof origin === 'string' ? [0, 0] : origin;
  const originMode = origin === 'center' ? 1 : origin === 'centroid' ? 2 : 0;
  if (
    ![rotate, scaleX, scaleY, skewX, skewY, translateX, translateY, ...originPoint].every(
      Number.isFinite
    )
  ) {
    throw new Error('Affine transform parameters must be finite numbers');
  }
  const tanX = Math.tan(skewX * angleScale);
  const tanY = Math.tan(skewY * angleScale);
  const cosine = Math.cos(rotate);
  const sine = Math.sin(rotate);
  // M = R * K * S, with K * S = [[sx, tanX * sy], [tanY * sx, sy]].
  const ksa = scaleX;
  const ksb = tanX * scaleY;
  const ksd = tanY * scaleX;
  const kse = scaleY;
  target.set([
    cosine * ksa - sine * ksd,
    cosine * ksb - sine * kse,
    sine * ksa + cosine * ksd,
    sine * ksb + cosine * kse,
    translateX,
    translateY,
    originPoint[0],
    originPoint[1],
    originMode,
    0,
    0,
    0
  ]);
  return target;
}

/** Caller-owned output of {@link GPUAffineTransform}. */
export type GPUAffineTransformOutput<
  Rows extends
    | GraphDataView<'float32x2'>
    | GraphVectorView<'float32x2'> = GraphDataView<'float32x2'>
> = {
  /** Transformed vertices, same length and layout as the input `positions`. */
  positions: Rows;
};

/** Props shared by every geometry type of {@link GPUAffineTransform}. */
type GPUAffineTransformSharedProps<
  Rows extends
    | GraphDataView<'float32x2'>
    | GraphVectorView<'float32x2'> = GraphDataView<'float32x2'>
> = {
  /** Prefix for generated node and transient IDs. Defaults to `'affine-transform'`. */
  id?: string;
  /**
   * Planar only (default: native planar coordinates). A longitude-latitude context throws at
   * construction: project positions to a planar frame first, for example with
   * `createGPULocalFrame` (`contracts/local-frame`) and `GPULocalFrameProjection`
   * (`geometry-measures`), and pass the frame's context.
   */
  spatialContext?: GPUSpatialContext;
  /** Packed or chunked planar vertices. Chunked input requires no computed origins. */
  positions: Rows;
  /** Per-feature origins that `parameters` may select. Default none. */
  origins?: readonly GPUAffineTransformOrigin[];
  /**
   * Per-frame packed float32 view of at least {@link GPU_AFFINE_TRANSFORM_PARAMETER_LENGTH}
   * elements written with {@link getGPUAffineTransformParameterValues}: one matrix for every
   * feature. Exactly one of `parameters` and `featureTransforms` is required.
   */
  parameters?: GraphDataView<'float32'>;
  /**
   * `6 * featureCount` float32 rows of `[a, b, d, e, xoff, yoff]` per feature, with
   * `x' = a x + b y + xoff` and `y' = d x + e y + yoff` (`shapely.affinity.affine_transform`
   * order), applied to absolute coordinates.
   */
  featureTransforms?: GraphDataView<'float32'>;
  /** Output view. */
  output: GPUAffineTransformOutput<Rows>;
};

/**
 * Properties for {@link GPUAffineTransform}, discriminated by `geometryType`.
 *
 * - `'polygons'`: the four-level layout of {@link GPUPolygonGeometryFields}, one row per feature.
 * - `'lines'` (default): `lineOffsets`, `lineCount + 1` monotonic vertex offsets, optional for a
 *   global transform with no per-feature origin; and optional `featureOffsets`, `featureCount + 1`
 *   monotonic line offsets so a feature owns several lines. Required with `featureTransforms` and
 *   with per-feature `origins`. Without `featureOffsets` every line is its own feature.
 * - `'points'`: optional `featureOffsets`, `featureCount + 1` monotonic point offsets (multi-points);
 *   without it every point is its own feature. Supports only the `'center'` origin.
 *
 * Per-frame (no recompile): the contents of `parameters` or `featureTransforms` and of every
 * input buffer. Compile-time: view lengths, `geometryType`, `origins`, and which of `parameters`
 * or `featureTransforms` is used.
 */
export type GPUAffineTransformProps<
  Rows extends
    | GraphDataView<'float32x2'>
    | GraphVectorView<'float32x2'> = GraphDataView<'float32x2'>
> = GPUAffineTransformSharedProps<Rows> &
  (
    | ({geometryType: 'polygons'} & Omit<GPUPolygonGeometryFields, 'positions'>)
    | {
        geometryType?: 'lines';
        lineOffsets?: GraphDataView<'uint32'>;
        featureOffsets?: GraphDataView<'uint32'>;
      }
    | {geometryType: 'points'; featureOffsets?: GraphDataView<'uint32'>}
  );

/**
 * Applies 2D affine transforms to many features at once on the GPU (`shapely.affinity.affine_transform`,
 * `translate`, `scale`, `rotate`, `skew`; PostGIS `ST_Affine`, `ST_Translate`, `ST_Scale`,
 * `ST_Rotate`).
 *
 * Two forms. **Global** (`parameters`): one per-frame matrix with translation and an origin,
 * packed by {@link getGPUAffineTransformParameters}; sliders and animation rewrite the buffer
 * without recompiling. The origin may be a point, or computed per feature from the input geometry
 * (`'center'` bounding-box center, `'centroid'`) by composing `GPUGeometryMeasures`. Vertices are
 * transformed relative to the origin, so rotating large projected coordinates about a feature's
 * own center keeps f32 precision. **Per feature** (`featureTransforms`): one `[a b d e xoff yoff]`
 * row set per feature, applied to absolute coordinates.
 *
 * Unlike shapely, which recomputes `'center'` after every step, the origin is computed once from
 * the input geometry; the two agree for scale, rotate and `'centroid'`, and differ for skew about
 * `'center'` (shapely re-centers on the sheared bounding box). Pin shapely to a fixed origin to
 * compare. The vertex count and
 * ring layout never change, so the input offsets stay valid for the output. Vertices outside every
 * ring use the nearest ring's feature.
 */
export class GPUAffineTransform<
  Rows extends
    | GraphDataView<'float32x2'>
    | GraphVectorView<'float32x2'> = GraphDataView<'float32x2'>
> implements GPUCommandNodeProducer
{
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUAffineTransformProps<Rows>;
  /** Resolved spatial context (planar only). */
  readonly spatialContext: GPUSpatialContext;
  /** Number of features, or 0 when no ring offsets are given. */
  readonly featureCount: number;
  /** Resolved geometry type. */
  readonly geometryType: 'points' | 'lines' | 'polygons';
  /** Resolved per-feature origins. */
  readonly origins: readonly GPUAffineTransformOrigin[];
  /** Chunk-route bound including the parameter upload; undefined for the existing measurement route. */
  readonly estimatedCommandNodeCount: number | undefined;
  private readonly chunkPlan: GraphChunkDispatchPlan;

  constructor(props: GPUAffineTransformProps<Rows>) {
    this.id = props.id ?? 'affine-transform';
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);
    this.props = props;
    this.geometryType = props.geometryType ?? 'lines';
    this.origins = props.origins ?? [];
    const {id} = this;
    if (!['points', 'lines', 'polygons'].includes(this.geometryType)) {
      throw new Error(`${id} geometryType must be 'points', 'lines' or 'polygons'`);
    }
    for (const origin of this.origins) {
      if (origin !== 'center' && origin !== 'centroid') {
        throw new Error(`${id} origins must contain only 'center' or 'centroid'`);
      }
    }
    if (this.origins.includes('centroid') && this.geometryType === 'points') {
      throw new Error(`${id} the 'centroid' origin needs geometryType 'lines' or 'polygons'`);
    }
    if (Boolean(props.parameters) === Boolean(props.featureTransforms)) {
      throw new Error(`${id} needs exactly one of parameters and featureTransforms`);
    }
    if (props.featureTransforms && this.origins.length > 0) {
      throw new Error(`${id} origins apply only to parameters, not featureTransforms`);
    }
    this.chunkPlan = planGraphChunkDispatch(id, {
      positions: props.positions,
      'output.positions': props.output.positions
    });
    if (this.origins.length > 0) {
      validateSinglePackedView(
        props.positions,
        ['float32x2'],
        `${id} positions with computed origins`
      );
      validateSinglePackedView(
        props.output.positions,
        ['float32x2'],
        `${id} output.positions with computed origins`
      );
    }
    for (const view of [props.positions, props.output.positions]) {
      for (const chunk of getGraphViewChunks(view))
        validatePackedView(chunk, ['float32x2'], `${id} positions`);
    }
    this.estimatedCommandNodeCount =
      this.origins.length > 0 ? undefined : this.chunkPlan.nonemptyChunkCount + 1;
    if (this.estimatedCommandNodeCount !== undefined)
      validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateCapacity(
      id,
      'positions.length',
      props.positions.length,
      1,
      GRAPH_CHUNK_MAXIMUM_ROW_COUNT
    );
    const layout = getAffineLayout(props);
    if (props.geometryType === 'points' && props.featureOffsets) {
      validateGPUMultipartGeometryPort(id, {
        kind: 'points',
        positions: props.positions,
        featureOffsets: props.featureOffsets
      });
    } else if (props.geometryType !== 'polygons' && layout.featureOffsets) {
      if (!layout.ringOffsets) {
        throw new Error(`${id} featureOffsets requires lineOffsets`);
      }
      validateGPUMultipartGeometryPort(id, {
        kind: 'lines',
        positions: props.positions,
        lineOffsets: layout.ringOffsets,
        featureOffsets: layout.featureOffsets
      });
    }
    for (const name of ['ringOffsets', 'featureOffsets', 'polygonOffsets'] as const) {
      const view = layout[name];
      if (view) {
        validateSinglePackedView(view, ['uint32'], `${id} ${name}`);
        // Binary-search upper bounds are emitted as u32 literals in both transform routes.
        validateCapacity(id, `${name}.length`, view.length, 2, GRAPH_CHUNK_MAXIMUM_ROW_COUNT);
      }
    }
    this.featureCount = layout.featureOffsets
      ? layout.featureOffsets.length - 1
      : layout.ringOffsets
        ? layout.ringOffsets.length - 1
        : 0;
    if ((props.featureTransforms || this.origins.length > 0) && !layout.ringOffsets) {
      throw new Error(`${id} per-feature transforms and origins require offsets`);
    }
    if (props.parameters) {
      validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
      if (props.parameters.length < GPU_AFFINE_TRANSFORM_PARAMETER_LENGTH) {
        throw new Error(
          `${id} parameters must hold ${GPU_AFFINE_TRANSFORM_PARAMETER_LENGTH} float32 values`
        );
      }
    }
    if (props.featureTransforms) {
      validateSinglePackedView(props.featureTransforms, ['float32'], `${id} featureTransforms`);
      if (props.featureTransforms.length !== 6 * this.featureCount) {
        throw new Error(`${id} featureTransforms must hold 6 rows per feature`);
      }
    }
    validateDisjointWritableViews(id, {'output.positions': props.output.positions});
    if (props.output.positions.length !== props.positions.length) {
      throw new Error(`${id} output.positions length must equal positions length`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.positions],
      [
        props.positions,
        layout.ringOffsets,
        layout.featureOffsets,
        layout.polygonOffsets,
        props.parameters,
        props.featureTransforms
      ]
    );
  }

  /** Returns the optional measurement nodes and the per-vertex transform node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, featureCount, origins} = this;
    const layout = getAffineLayout(props);
    validateGraphChunkDispatchBindings(graph, id, this.chunkPlan);
    validateGraphChunkBindingRanges(graph, id, {
      ringOffsets: layout.ringOffsets,
      featureOffsets: layout.featureOffsets,
      polygonOffsets: layout.polygonOffsets,
      parameters: props.parameters,
      featureTransforms: props.featureTransforms
    });
    if (
      origins.length === 0 &&
      (!('buffer' in props.positions) || !('buffer' in props.output.positions))
    ) {
      return createAffineChunkNodes(graph, {
        id,
        plan: this.chunkPlan,
        ...layout,
        parameters: props.parameters,
        featureTransforms: props.featureTransforms,
        featureCount
      });
    }
    for (const [name, byteLength] of [
      ['bounds', origins.includes('center') ? featureCount * 16 : 0],
      ['centroids', origins.includes('centroid') ? featureCount * 8 : 0],
      // GPUGeometryMeasures binds one complete shared-layout statistics record per feature.
      [
        'origin feature stats',
        origins.length > 0
          ? featureCount * GEOMETRY_STATS_STRIDE * Uint32Array.BYTES_PER_ELEMENT
          : 0
      ]
    ] as const) {
      validateCapacity(id, `${name} byteLength`, byteLength, 0, graph.device.limits.maxBufferSize);
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        byteLength,
        'reduce the feature count'
      );
    }
    const nodes: GPUCommandNode<Parameters>[] = [];
    let bounds: GraphDataView<'float32x4'> | undefined;
    let centroids: GraphDataView<'float32x2'> | undefined;
    if (origins.length > 0) {
      const wantsCenter = origins.includes('center');
      const wantsCentroid = origins.includes('centroid');
      // Centroids are only valid for lines and polygons; bounds work for any geometry.
      bounds = wantsCenter
        ? createTransientView(graph, `${id}-bounds`, 'float32x4', featureCount)
        : undefined;
      centroids = wantsCentroid
        ? createTransientView(graph, `${id}-centroids`, 'float32x2', featureCount)
        : undefined;
      nodes.push(
        ...new GPUGeometryMeasures({
          spatialContext: {...this.spatialContext, metric: 'native', units: 'native'},
          id: `${id}-measures`,
          positions: props.positions as GraphDataView<'float32x2'>,
          ...getAffineMeasuresLayout(props),
          output: {bounds, centroids}
        }).getCommandNodes(graph)
      );
    }
    nodes.push(
      createAffineTransformNode<Parameters>(graph, {
        id: `${id}-transform`,
        operation: OPERATION,
        positions: props.positions as GraphDataView<'float32x2'>,
        ringOffsets: layout.ringOffsets,
        featureOffsets: layout.featureOffsets,
        polygonOffsets: layout.polygonOffsets,
        parameters: props.parameters,
        featureTransforms: props.featureTransforms,
        bounds,
        centroids,
        featureCount,
        outputPositions: props.output.positions as GraphDataView<'float32x2'>
      })
    );
    return nodes;
  }
}

/**
 * Maps the public layout to the transform kernel: polygons keep the four levels; lines use
 * `lineOffsets` as ring offsets and `featureOffsets` as feature-to-ring; a point feature is one
 * ring spanning its points, so `featureOffsets` doubles as the ring offsets.
 */
function getAffineLayout(
  props: GPUAffineTransformProps<GraphDataView<'float32x2'> | GraphVectorView<'float32x2'>>
): {
  ringOffsets?: GraphDataView<'uint32'>;
  featureOffsets?: GraphDataView<'uint32'>;
  polygonOffsets?: GraphDataView<'uint32'>;
} {
  switch (props.geometryType) {
    case 'polygons':
      return {
        ringOffsets: props.ringOffsets,
        featureOffsets: props.featureOffsets,
        polygonOffsets: props.polygonOffsets
      };
    case 'points':
      return {ringOffsets: props.featureOffsets};
    default:
      return {ringOffsets: props.lineOffsets, featureOffsets: props.featureOffsets};
  }
}

/** The geometry fields `GPUGeometryMeasures` needs to measure the transformed geometry. */
function getAffineMeasuresLayout(
  props: GPUAffineTransformProps<GraphDataView<'float32x2'> | GraphVectorView<'float32x2'>>
) {
  switch (props.geometryType) {
    case 'polygons':
      return {
        geometryType: 'polygons' as const,
        ringOffsets: props.ringOffsets,
        featureOffsets: props.featureOffsets,
        polygonOffsets: props.polygonOffsets
      };
    case 'points':
      return {geometryType: 'points' as const, featureOffsets: props.featureOffsets};
    default:
      // Origins need the offsets (validated by the constructor).
      return {
        geometryType: 'lines' as const,
        lineOffsets: props.lineOffsets as GraphDataView<'uint32'>,
        featureOffsets: props.featureOffsets
      };
  }
}
