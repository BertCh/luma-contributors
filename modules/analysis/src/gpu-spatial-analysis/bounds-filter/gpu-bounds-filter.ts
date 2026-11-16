// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  resolveGPUSpatialContext,
  type GPUSpatialContext,
  type GPUSpatialContextSupport
} from '../../analysis-core/contracts/spatial-context';
import {
  createTransientView,
  type GraphVectorView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {
  GPUCompactOutput,
  GPUUint32Rows
} from '../../analysis-core/kernels/gpu-contributor-types';
import {
  getGraphViewChunks,
  validateCapacity,
  validateCommandNodeBudget,
  validateGraphViewTopology,
  validateCompactOutput,
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createPublishNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  planGraphChunkDispatch,
  validateGraphChunkBindingRanges
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateSpatialJoinOutputGroups} from '../spatial-join/spatial-join-output-validation';
import {createBoundsFilterMaskNode} from './bounds-filter-mask-node';
import {
  getBoundsFilterCompactionNodes,
  planBoundsFilterCompaction
} from './bounds-filter-compaction';
import {validateGraphChunkGroupKernelSupport} from '../../analysis-core/kernels/graph-chunk-groups';

/** Planar coordinates, or longitude/latitude used as plain coordinates (no metric). */
const SPATIAL_CONTEXT_SUPPORT: GPUSpatialContextSupport = Object.freeze({
  coordinateSpaces: Object.freeze(['planar', 'longitude-latitude'] as const),
  metrics: Object.freeze(['none', 'native'] as const)
});

const OPERATION = 'GPUBoundsFilter';

/** Identity IDs and the required count are uint32, including carries between chunks. */
const GPU_BOUNDS_FILTER_MAXIMUM_ROWS = 0xffffffff;
/**
 * Upper bound per chunk (plus one fixed allowance) for classify, bit-count and scatter,
 * plus the global hierarchical scan nodes: a uint32 row count needs at most four 256-row scan levels.
 */
const GPU_BOUNDS_FILTER_COMMAND_NODE_ALLOWANCE = 32;

/** Number of float32 elements of the per-frame box: `[minX, minY, maxX, maxY]`. */
export const GPU_BOUNDS_FILTER_PARAMETER_LENGTH = 4;

/**
 * Relation between each feature's bounding box `F` and the query box `B`.
 *
 * - `'intersects'`: `F` and `B` overlap or touch (GeoPandas `.cx` on bounds).
 * - `'within'`: `F` lies inside `B`, borders included.
 * - `'contains'`: `F` covers `B`, borders included.
 */
export type GPUBoundsFilterMode = 'intersects' | 'within' | 'contains';

/** CPU settings of the per-frame box of {@link GPUBoundsFilter}. */
export type GPUBoundsFilterSettings = {
  /** Left edge. */
  minX: number;
  /** Bottom edge. */
  minY: number;
  /** Right edge, at least `minX` unless `wrapX` is set. */
  maxX: number;
  /** Top edge, at least `minY`. */
  maxY: number;
  /**
   * Allows `minX > maxX`, meaning the box wraps through the longitude seam: it covers
   * `x >= minX` and `x <= maxX` (for example `minX = 170, maxX = -170`). Intended for
   * longitude/latitude contexts. Feature bounds must not wrap themselves. A wrapped box never
   * satisfies `'contains'`, because a finite feature box cannot cover both sides of the seam.
   */
  wrapX?: boolean;
};

/**
 * Packs a {@link GPUBoundsFilterSettings} into the 4-element float32 layout `[minX, minY, maxX, maxY]`.
 * Write the result into a `GPUParameterBuffer` between encodings to move the box without
 * recompiling.
 *
 * @param settings Box to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If a value is not finite, the box is inverted (`maxX < minX` needs `wrapX`), or
 *   `target` is too short.
 */
export function getGPUBoundsFilterParameterValues(
  settings: GPUBoundsFilterSettings,
  target: Float32Array = new Float32Array(GPU_BOUNDS_FILTER_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_BOUNDS_FILTER_PARAMETER_LENGTH) {
    throw new Error(
      `Bounds filter target must hold ${GPU_BOUNDS_FILTER_PARAMETER_LENGTH} elements`
    );
  }
  const {minX, minY, maxX, maxY, wrapX} = settings;
  if (![minX, minY, maxX, maxY].every(Number.isFinite)) {
    throw new Error('Bounds filter box must be finite');
  }
  if ((maxX < minX && !wrapX) || maxY < minY) {
    throw new Error('Bounds filter requires maxX >= minX (unless wrapX) and maxY >= minY');
  }
  target.set([minX, minY, maxX, maxY]);
  return target;
}

/** Caller-owned outputs of {@link GPUBoundsFilter}. */
export type GPUBoundsFilterOutput<MaskRows extends GPUUint32Rows = GraphDataView<'uint32'>> =
  GPUCompactOutput & {
    /** Optional canonical 0/1 mask of accepted features, with the same chunk topology as `bounds`. */
    outputMask?: MaskRows;
  };

/**
 * Properties for {@link GPUBoundsFilter}.
 *
 * Per-frame (no recompile): the box and every input buffer. Compile-time: view lengths and chunking, `mode`,
 * `output.ids.length` and which optional views are present.
 */
export type GPUBoundsFilterProps<MaskRows extends GPUUint32Rows = GraphDataView<'uint32'>> = {
  /** Prefix for generated node and transient IDs. Defaults to `'bounds-filter'`. */
  id?: string;
  /**
   * Spatial context of the coordinates; omitted means native planar coordinates. Planar and
   * longitude/latitude (metric `'none'`) are accepted. Longitude/latitude is treated as plain
   * coordinates (no metric). A box crossing the antimeridian is expressed with
   * `minX > maxX` (see {@link GPUBoundsFilterSettings.wrapX}); feature bounds must be non-wrapped.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Packed or chunked per-feature bounds `[minX, minY, maxX, maxY]`, for example `GPUGeometryMeasures`
   * bounds. Rows with a non-finite component (NaN or infinite, the empty-geometry marker) are
   * rejected in every mode.
   */
  bounds: GraphDataView<'float32x4'> | GraphVectorView<'float32x4'>;
  /** Per-frame box: a packed float32 view written with {@link getGPUBoundsFilterParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Relation tested between each feature's bounds and the box. Defaults to `'intersects'`. */
  mode?: GPUBoundsFilterMode;
  /** Caller-owned bounded result of accepted feature rows, ascending. `ids.length` is the capacity. */
  output: GPUBoundsFilterOutput<MaskRows>;
};

/**
 * Selects features whose bounding boxes relate to a per-frame box: a GPU `GeoDataFrame.cx[...]`
 * (bounds level) and `gdf.bounds` comparison.
 *
 * Note this is a bounds test, not an exact geometry test: `'intersects'` accepts a feature whose
 * bounding box overlaps the box even when the geometry misses it. Refine the survivors with
 * `GPUSpatialPredicateJoin` against a one-row box polygon for exact `cx` semantics.
 *
 * One mask kernel and a stable compaction publish the accepted row IDs, a clamped count and an
 * overflow flag. The box is read per frame, so a viewport or slider moves without recompiling.
 * Comparisons are closed (borders count) and exact in f32 (no arithmetic).
 *
 * Bounds may be a `GraphVectorView` of packed chunks; an optional `outputMask` must have the
 * same topology. One classify kernel runs per nonempty chunk. Compact IDs remain packed and
 * ascending in global source-row order across chunks, including the retained prefix on overflow.
 * Each chunk stores one bit per bound. A global scan of one count per 256 bounds provides
 * stable compact positions; no scratch allocation grows as one uint32 per source row. Each
 * scratch chunk fits the device binding and buffer limits. The packed output must fit one binding.
 * Runtime chunk settings reuse classify, count and scatter shaders and pipelines.
 *
 * Coordinates only, no metric: `spatialContext` accepts planar coordinates or longitude/latitude with
 * metric `'none'`; great-circle and other metric contexts throw. Longitude/latitude degrees are used
 * as plain coordinates. Feature bounds never wrap, but a query box with `minX > maxX` is read as
 * wrapping through the seam (a viewport crossing the antimeridian).
 */
export class GPUBoundsFilter<MaskRows extends GPUUint32Rows = GraphDataView<'uint32'>>
  implements GPUCommandNodeProducer
{
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUBoundsFilterProps<MaskRows>;
  /** Resolved spatial context (planar or longitude/latitude without metric). */
  readonly spatialContext: GPUSpatialContext;
  /** Resolved relation mode. */
  readonly mode: GPUBoundsFilterMode;

  /** Conservative upper bound including the published visibility workflow. */
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPUBoundsFilterProps<MaskRows>) {
    this.id = props.id ?? 'bounds-filter';
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext,
      SPATIAL_CONTEXT_SUPPORT
    );
    this.mode = props.mode ?? 'intersects';
    const {id} = this;
    if (!['intersects', 'within', 'contains'].includes(this.mode)) {
      throw new Error(`${id} mode must be 'intersects', 'within' or 'contains'`);
    }
    planGraphChunkDispatch(id, {bounds: props.bounds});
    const chunks = getGraphViewChunks(props.bounds);
    for (const chunk of chunks) {
      validateSinglePackedView(chunk, ['float32x4'], `${id} bounds`);
    }
    validateCapacity(id, 'bounds.length', props.bounds.length, 1, GPU_BOUNDS_FILTER_MAXIMUM_ROWS);
    this.estimatedCommandNodeCount =
      GPU_BOUNDS_FILTER_COMMAND_NODE_ALLOWANCE *
      (chunks.filter(chunk => chunk.length > 0).length + 1);
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_BOUNDS_FILTER_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_BOUNDS_FILTER_PARAMETER_LENGTH} float32 values`
      );
    }
    validateCompactOutput(id, props.output);
    if (props.output.ids.length < 1) {
      throw new Error(`${id} output.ids must contain at least one row`);
    }
    if (props.output.outputMask) {
      for (const chunk of getGraphViewChunks(props.output.outputMask)) {
        validateSinglePackedView(chunk, ['uint32'], `${id} output.outputMask`);
      }
      validateGraphViewTopology(id, 'output.outputMask', props.bounds, props.output.outputMask);
    }
    const outputs = {
      'output.ids': props.output.ids,
      'output.status.count': props.output.status.count,
      'output.status.overflow': props.output.status.overflow,
      'output.status.requiredCount': props.output.status.requiredCount,
      'output.outputMask': props.output.outputMask
    };
    for (const [name, view] of Object.entries({...outputs, parameters: props.parameters})) {
      if (view)
        validateCapacity(id, `${name}.length`, view.length, 0, GPU_BOUNDS_FILTER_MAXIMUM_ROWS);
    }
    validateSpatialJoinOutputGroups(id, outputs, [{...props.output.status}]);
    validateGraphOutputsDisjointFromInputs(id, Object.values(outputs), [
      props.bounds,
      props.parameters
    ]);
  }

  /** Returns the mask kernel, the stable compaction nodes and the publish node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {output} = props;
    const chunks = getGraphViewChunks(props.bounds);
    const views = {
      bounds: props.bounds,
      parameters: props.parameters,
      'output.ids': output.ids,
      'output.outputMask': output.outputMask,
      'output.status.count': output.status.count,
      'output.status.overflow': output.status.overflow,
      'output.status.requiredCount': output.status.requiredCount
    };
    validateGraphChunkBindingRanges(graph, id, views);
    for (const [name, view] of Object.entries(views)) {
      for (const chunk of view ? getGraphViewChunks(view) : []) {
        validateCapacity(
          id,
          `${name}.buffer.byteLength (device.limits.maxBufferSize)`,
          chunk.buffer.byteLength,
          0,
          graph.device.limits.maxBufferSize
        );
      }
    }
    validateGraphChunkGroupKernelSupport(graph, id, 4);
    const plan = planBoundsFilterCompaction(
      graph.device,
      id,
      chunks.map(chunk => chunk.length)
    );
    const masks = plan.chunks.map((chunk, index) =>
      createTransientView(graph, `${id}-bits-${index}`, 'uint32', Math.max(chunk.wordCount, 1))
    );
    const maskChunks = output.outputMask ? getGraphViewChunks(output.outputMask) : [];
    // A box with minX > maxX wraps through the seam: its x extent is the union of two rays.
    const comparison = {
      intersects: `select(fMax.x >= boxMin.x && fMin.x <= boxMax.x, fMax.x >= boxMin.x || fMin.x <= boxMax.x, wrapped) && fMin.y <= boxMax.y && fMax.y >= boxMin.y`,
      within: `select(fMin.x >= boxMin.x && fMax.x <= boxMax.x, fMin.x >= boxMin.x || fMax.x <= boxMax.x, wrapped) && fMin.y >= boxMin.y && fMax.y <= boxMax.y`,
      contains: `!wrapped && fMin.x <= boxMin.x && fMax.x >= boxMax.x && fMin.y <= boxMin.y && fMax.y >= boxMax.y`
    }[this.mode];
    const nodes: GPUCommandNode<Parameters>[] = [];
    for (const [chunkIndex, chunk] of chunks.entries()) {
      if (chunk.length > 0) {
        nodes.push(
          createBoundsFilterMaskNode(graph, {
            id: `${id}-mask-${chunkIndex}`,
            bounds: chunk,
            box: props.parameters,
            mask: masks[chunkIndex],
            callerMask: maskChunks[chunkIndex],
            comparison
          })
        );
      }
    }
    const total = createTransientView(graph, `${id}-total`, 'uint32', 1);
    nodes.push(
      ...getBoundsFilterCompactionNodes(graph, {
        id: `${id}-compact`,
        plan,
        masks,
        output: output.ids,
        total
      }),
      createPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        requiredCount: total,
        output
      })
    );
    return nodes;
  }
}
