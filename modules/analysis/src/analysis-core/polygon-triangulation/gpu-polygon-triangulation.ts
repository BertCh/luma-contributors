// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {clamp} from '@math.gl/core';
import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../kernels/wgsl-kernel-nodes';
import {
  GPU_POLYGON_TRIANGULATION_DEFAULT_MONOTONE_MINIMUM_VERTICES,
  getMonotoneTriangulationNodes,
  getGPUPolygonTriangulationMonotoneMaximumVertexCount,
  validatePolygonTriangulationMonotoneCapacity
} from './polygon-triangulation-monotone';
import {getTriangulationWGSL} from './polygon-triangulation-earcut-wgsl';
import {
  validateCapacity,
  validateOption,
  getStorageBindingElementLimit,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateMaximumCount,
  validateSinglePackedView
} from '../kernels/gpu-contributor-utils';
import {EXACT_GEOMETRY_WGSL} from '../kernels/exact-geometry-wgsl';
import {EXACT_ORIENTATION_WGSL} from '../kernels/exact-orientation-wgsl';
import {resolveGPUSpatialContext, type GPUSpatialContext} from '../contracts/spatial-context';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../kernels/construction-snapshot';
import type {GPUPolygonGeometryFields} from '../contracts/index';

import {
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges
} from '../kernels/graph-chunk-dispatch';
import {getPolygonTriangulationLayoutNodes} from './polygon-triangulation-layout';

const OPERATION = 'GPUPolygonTriangulation';

/** Default `maximumWork` of {@link GPUPolygonTriangulation}: about 4 million elementary steps. */
export const GPU_POLYGON_TRIANGULATION_DEFAULT_MAXIMUM_WORK = 4_000_000;

/**
 * Safe index capacity for a layout that may contain empty polygons or degenerate rings.
 * With at least one polygon, reserve `3 * max(0, vertexCount + 2 * ringCount - 4)` rows:
 * only one non-empty polygon can be assumed from host lengths. GPU-computed triangle offsets
 * give the used slices; non-degenerate polygons retain `vertices + 2 * holes - 2` triangles.
 */
export function getPolygonTriangulationIndexCount(
  vertexCount: number,
  ringCount: number,
  polygonCount: number
): number {
  return polygonCount > 0 ? Math.max(0, 3 * (vertexCount + 2 * ringCount - 4)) : 0;
}

/**
 * Largest scratch node count of {@link GPUPolygonTriangulation} on `device`: each node takes 7
 * `uint32` words (28 bytes) with the z-order hash (the default) or 4 words (16 bytes) without it,
 * and the scratch must fit one storage binding, so 4793490 (8388608 without the hash) on a default
 * device (128 MiB bindings). The node count of a layout is
 * `2 * vertices + 5 * (rings - polygons) + 8 * polygons`, see
 * {@link getPolygonTriangulationNodeCount}, about `2` to `3` per input vertex.
 *
 * @param device Device whose `maxStorageBufferBindingSize` bounds the scratch.
 * @param useZOrderHash Whether the z-order hash is used (the default).
 */
export function getGPUPolygonTriangulationMaximumNodeCount(
  device: Device,
  useZOrderHash: boolean = true
): number {
  return getStorageBindingElementLimit(device, useZOrderHash ? 28 : 16);
}

/**
 * Number of scratch nodes {@link GPUPolygonTriangulation} allocates for a layout:
 * `2 * vertices + 5 * (rings - polygons) + 8 * polygons`. That is twice the minimum node count
 * for diagonal splits (`vertices + 2 * holes`), a margin of 8 per polygon, and one node per hole
 * for the vertex that is inserted into an edge of another ring when the hole's leftmost vertex
 * lies inside that edge.
 */
export function getPolygonTriangulationNodeCount(
  vertexCount: number,
  ringCount: number,
  polygonCount: number
): number {
  const holeCount = ringCount - polygonCount;
  return 2 * (vertexCount + 2 * holeCount) + 8 * polygonCount + holeCount;
}

/**
 * Throws unless the scratch of a polygon layout fits one storage binding of `device`: its
 * {@link getPolygonTriangulationNodeCount} is at most
 * {@link getGPUPolygonTriangulationMaximumNodeCount}. Contributors that triangulate internally call
 * it under their own ID, at declaration.
 *
 * @param device Device the graph runs on.
 * @param id Contributor ID, for the message.
 * @param vertexCount Vertex rows of the layout.
 * @param ringCount Rings of the layout.
 * @param polygonCount Polygons of the layout.
 * @param useZOrderHash Whether the z-order hash is used (the default).
 */
export function validatePolygonTriangulationScratch(
  device: Device,
  id: string,
  vertexCount: number,
  ringCount: number,
  polygonCount: number,
  useZOrderHash: boolean = true
): void {
  validateMaximumCount(
    id,
    'scratch nodes',
    getPolygonTriangulationNodeCount(vertexCount, ringCount, polygonCount),
    getGPUPolygonTriangulationMaximumNodeCount(device, useZOrderHash)
  );
  validateCapacity(
    id,
    'scratch byteLength',
    Math.max(
      (useZOrderHash ? 28 : 16) *
        getPolygonTriangulationNodeCount(vertexCount, ringCount, polygonCount),
      16
    ),
    0,
    device.limits.maxBufferSize
  );
}

/** Caller-owned outputs of {@link GPUPolygonTriangulation}. */
export type GPUPolygonTriangulationOutput = {
  /**
   * Triangle indices; {@link getPolygonTriangulationIndexCount} gives a safe capacity.
   * Polygon `p` starts at `3 * triangleOffsets[p]`. Degenerate rings are excluded before
   * computing slices; non-degenerate layouts retain their original slice offsets. Indices refer
   * to rows of `positions`. Unused polygon slots hold `(first, first, first)`; spare capacity is
   * zeroed. Smaller legacy buffers are accepted, with insufficient slices flagged invalid.
   * When there are no positions, the used triangle count is zero and nothing should be drawn.
   */
  indices: GraphDataView<'uint32'>;
  /**
   * Flag per polygon, `polygonCount` rows: 1 when the triangulation completed, 0 when
   * it failed (self-intersecting or otherwise untriangulable rings, a hole that cannot be
   * joined to the shell, non-finite coordinates, scratch overflow, or `maximumWork` exceeded).
   * For a valid polygon (simple rings, holes inside the shell and outside each other; rings may
   * touch at single points) a result flagged 1 is a tiling: the triangles cover the polygon once
   * and nothing else, see {@link GPUPolygonTriangulation}. Empty polygons and degenerate shells
   * complete with zero triangles. A failed polygon's slice
   * is entirely degenerate. The polygons of feature `f` are
   * `[featureOffsets[f], featureOffsets[f + 1])`.
   */
  valid: GraphDataView<'uint32'>;
  /**
   * Optional count of emitted triangle rows per polygon, including zero-area rows and excluding
   * unused padding; `polygonCount` rows.
   */
  triangleCount?: GraphDataView<'uint32'>;
  /** Optional per-polygon triangle slice offsets, `polygonCount + 1` rows including the terminal. */
  triangleOffsets?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUPolygonTriangulation}: the four-level layout of
 * {@link GPUPolygonGeometryFields} (`positions`, `featureOffsets`, `polygonOffsets`,
 * `ringOffsets`). The first ring is the shell and later rings are holes. Rings with fewer than
 * three vertices or only exact-zero first-vertex fan orientations are skipped. An uncertified
 * orientation is retained. A degenerate shell makes the entire polygon empty. Empty polygon and
 * ring slots are accepted directly from clipping. Rings close implicitly, a repeated first vertex
 * is tolerated. Winding of either kind is accepted. Triangulation runs per polygon, so a
 * multipolygon feature contributes one slice per polygon and `output.valid` and
 * `output.triangleCount` have one row per polygon.
 *
 * Per-frame: the contents of every input buffer. Topology: view lengths and `maximumWork`.
 */
export type GPUPolygonTriangulationProps = GPUPolygonGeometryFields & {
  /** Prefix for generated node and transient IDs. Defaults to `'polygon-triangulation'`. */
  id?: string;
  /**
   * Planar only (default: native planar coordinates). A longitude-latitude context throws at
   * construction: project positions to a planar frame first, for example with
   * `createGPULocalFrame` (`contracts/local-frame`) and `GPULocalFrameProjection`
   * (`geometry-measures`), and pass the frame's context.
   */
  spatialContext?: GPUSpatialContext;
  /** Caller-owned outputs. */
  output: GPUPolygonTriangulationOutput;
  /**
   * Work budget per polygon in elementary steps (ring walks and predicate scans). A polygon that
   * exceeds it is flagged invalid instead of hanging the device, which bounds the O(n^2) worst
   * case of ear clipping. Defaults to {@link GPU_POLYGON_TRIANGULATION_DEFAULT_MAXIMUM_WORK},
   * which covers rings of a few thousand vertices. Baked into the shader.
   */
  maximumWork?: number;
  /**
   * Index ear tests with Earcut's z-order curve hash for polygons with more than 80 vertices
   * (default `true`). Ring nodes are kept in a second list sorted by Morton code, so an ear test
   * only visits the vertices whose code lies between the codes of the ear's bounding-box corners
   * instead of the whole ring. This turns the O(n^2) ear clipping of one large ring into about
   * O(n log n) for typical outlines, with identical output. Costs 12 extra scratch bytes per
   * node (28 bytes of scratch per node instead of 16). Pass `false` to keep the footprint small
   * when every polygon is tiny. Baked into the shader.
   */
  useZOrderHash?: boolean;
  /**
   * What happens to polygons that per-invocation Earcut cannot finish (default `'monotone'`).
   *
   * - `'monotone'`: polygons with at least {@link GPUPolygonTriangulationProps.monotoneMinimumVertices}
   *   vertices skip Earcut, and every polygon for which Earcut fails (`maximumWork` exceeded, or
   *   rings it cannot split) is retried, by a parallel y-monotone decomposition followed by a
   *   linear stack triangulation of each monotone piece. A polygon that skipped Earcut and that
   *   the monotone path flags invalid (most polygons whose rings touch, input beyond
   *   {@link GPUPolygonTriangulationProps.monotoneMaximumVertices}) is then given to Earcut within
   *   `maximumWork`. The monotone path flags a polygon whenever one of its triangles would come
   *   out clockwise, which is what a wrong decomposition of touching rings produces, so a polygon
   *   it returns as valid is a tiling and the method never does worse than `'none'`: it is valid
   *   wherever `'none'` is, and never valid with a wrong triangulation. Valid monotone output can
   *   include zero-area triangles where holes touch; these rows count toward `triangleCount`
   *   and retain their input vertex indices. The positive-area triangles tile the polygon outside
   *   its holes. The output contract is unchanged: the same slice, `valid` and `triangleCount`.
   *   Polygons that Earcut triangulates are
   *   bit-identical to `'none'`. The path costs fixed O(vertices) scans and sorts (sized by
   *   {@link GPUPolygonTriangulationProps.monotoneMaximumVertices}) and a few hundred bytes of
   *   scratch per vertex, even when no polygon is large.
   * - `'none'`: today's behavior, a polygon over `maximumWork` is flagged `valid = 0` with a
   *   degenerate slice. Use it when all polygons are known to be small.
   *
   * Baked into the graph.
   */
  largePolygonMethod?: 'monotone' | 'none';
  /**
   * Polygons with at least this many vertices (all rings) are routed straight to the monotone path
   * and skip Earcut (default 4096). Earcut with its z-order hash is near-linear for typical
   * outlines up to a few thousand vertices; its worst case, bounded only by `maximumWork`, is
   * quadratic, so a lower value trades Earcut's output for guaranteed linear-time-per-piece work.
   * Used with `largePolygonMethod: 'monotone'`. Baked into the shader.
   */
  monotoneMinimumVertices?: number;
  /**
   * Capacity of the monotone path in cleaned vertices (default: all `positions` rows). Scratch,
   * scans and sorts are sized by it, so a smaller value cuts the fixed cost for inputs whose large
   * polygons are known to total fewer vertices. Large polygons (in row order) whose vertices do not
   * fit skip the monotone path and are given to Earcut within `maximumWork` (`valid = 0` with a
   * degenerate slice if that fails too). At most
   * {@link getGPUPolygonTriangulationMonotoneMaximumVertexCount}; only an explicit larger value
   * throws. Inputs with more rows than that limit skip the monotone path entirely and behave like
   * `'none'`. Baked into the graph.
   */
  monotoneMaximumVertices?: number;
};

/** Device-independent triangulation work limits, shared by composite contributors. @internal */
export function validatePolygonTriangulationWorkLimits(
  id: string,
  props: Pick<
    GPUPolygonTriangulationProps,
    'maximumWork' | 'largePolygonMethod' | 'monotoneMinimumVertices' | 'monotoneMaximumVertices'
  >
): void {
  if (props.maximumWork !== undefined) validateCapacity(id, 'maximumWork', props.maximumWork);
  validateOption(
    id,
    'largePolygonMethod',
    props.largePolygonMethod === undefined ? 'monotone' : props.largePolygonMethod,
    ['monotone', 'none']
  );
  if (props.monotoneMinimumVertices !== undefined)
    validateCapacity(id, 'monotoneMinimumVertices', props.monotoneMinimumVertices);
  if (props.monotoneMaximumVertices !== undefined)
    validateCapacity(id, 'monotoneMaximumVertices', props.monotoneMaximumVertices);
}

/**
 * Triangulates polygons with holes on the GPU: the Earcut algorithm behind deck.gl's polygon
 * tessellation, comparable to `geo` `TriangulateEarcut` and Sedona `ST_TriangulatePolygon`.
 *
 * Each polygon runs on one invocation (the ear-clipping list is sequential), so throughput comes
 * from many polygons, not from one huge ring. Polygons that are too large for that, or on which
 * Earcut fails within `maximumWork`, take the monotone path
 * ({@link GPUPolygonTriangulationProps.largePolygonMethod}): a parallel y-monotone decomposition
 * and a linear stack triangulation of each monotone piece, with the same output contract. The
 * Earcut port follows Earcut: holes are bridged into the
 * shell by Eberly's rightmost-vertex rule (holes processed left to right), ears are clipped from a
 * circular list, and stalls fall back to filtering collinear points, curing local
 * self-intersections and splitting along a valid diagonal. Orientation tests use the exact
 * predicate of `GPUSegmentIntersection`, so collinear and near-degenerate input does not depend
 * on f32 rounding. The hole bridge is chosen exactly too: which shell edge the ray from the
 * hole's leftmost vertex meets first, which vertices lie in the bridge triangle, which candidate
 * makes the smallest angle with the ray (collinear candidates keep the nearer one), and the order
 * of holes that share their leftmost vertex are orientation tests, not f32 quotients. Polygons
 * above 80 vertices use Earcut's z-order hash
 * ({@link GPUPolygonTriangulationProps.useZOrderHash}), so ear tests visit only nearby vertices.
 *
 * **A valid result is a tiling.** Rings may touch at single points: a hole vertex on a vertex of
 * another ring, or inside an edge of another ring. A hole whose leftmost vertex lies inside an
 * edge is joined there: the vertex is inserted into the edge and the bridge has zero length.
 * The port departs from Earcut in three places so that `valid = 1` can be relied on. Earcut cures
 * a local self-intersection of the working ring by writing a triangle across it; that is the
 * only step that does not keep the boundary of the remaining ring plus the triangles written
 * equal to the boundary of the polygon, so the port flags the polygon instead. A hole that
 * cannot be joined to the shell flags the polygon instead of being dropped. A predicate that
 * cannot be certified (non-finite coordinates) flags it too. Every triangle written is then a
 * counter-clockwise ear of exact positive area and the triangle boundaries sum to the polygon
 * boundary, so the number of triangles over any point is the polygon's winding number there: 1
 * inside, 0 outside. This holds for every valid polygon; an invalid one (rings that cross, a hole
 * outside its shell) is flagged or covered according to its winding number. The monotone path
 * has the same property: its triangles sum to the polygon boundary by construction, and it flags
 * a polygon when a triangle would be clockwise.
 *
 * GPU layout preparation counts non-degenerate rings and scans polygon slices: exactly
 * `3 * (vertices + 2 * holes - 2)` indices per non-empty polygon, see {@link GPUPolygonTriangulationOutput.indices}. Like Earcut the result is not a
 * Delaunay triangulation and may contain slivers.
 *
 * **Limits.** Earcut's work is budgeted by `maximumWork`; a larger polygon is retried by the
 * monotone path, or with `largePolygonMethod: 'none'` flagged in `valid` rather than hung. Scratch memory is 56 bytes per vertex of input (twice the minimum node count for diagonal splits;
 * 32 bytes without the z-order hash) and must fit one storage binding: at most
 * {@link getGPUPolygonTriangulationMaximumNodeCount} scratch nodes (4793490 on a default device;
 * 8388608 with `useZOrderHash: false`), see {@link getPolygonTriangulationNodeCount}.
 * Empty polygons, degenerate shells and degenerate holes contribute no triangle slots.
 */
export class GPUPolygonTriangulation implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUPolygonTriangulationProps;
  /** Resolved spatial context (planar only). */
  readonly spatialContext: GPUSpatialContext;
  /** Number of polygons, `polygonOffsets.length - 1`. */
  readonly polygonCount: number;
  /** Number of rings, `ringOffsets.length - 1`. */
  readonly ringCount: number;
  /** Number of index rows the layout requires. */
  readonly indexCount: number;

  constructor(props: GPUPolygonTriangulationProps) {
    const contextProps = snapshotConstructionFields(props, ['id', 'spatialContext']);
    this.id = contextProps.id ?? 'polygon-triangulation';
    const spatialContext =
      contextProps.spatialContext &&
      snapshotConstructionSpatialContext(contextProps.spatialContext);
    // Reject unsupported coordinates before evaluating geometry or output getters.
    this.spatialContext = snapshotConstructionSpatialContext(
      resolveGPUSpatialContext(this.id, spatialContext)
    );
    const constructionProps = snapshotConstructionFields(props, [
      'positions',
      'featureOffsets',
      'polygonOffsets',
      'ringOffsets',
      'output',
      'maximumWork',
      'useZOrderHash',
      'largePolygonMethod',
      'monotoneMinimumVertices',
      'monotoneMaximumVertices'
    ]);
    props = Object.freeze({
      ...contextProps,
      ...constructionProps,
      spatialContext,
      output: snapshotConstructionFields(constructionProps.output, [
        'indices',
        'valid',
        'triangleCount',
        'triangleOffsets'
      ])
    });
    this.props = props;
    const {id} = this;
    const {output} = props;
    validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);
    for (const name of ['featureOffsets', 'polygonOffsets', 'ringOffsets'] as const) {
      validateSinglePackedView(props[name], ['uint32'], `${id} ${name}`);
      if (props[name].length < 1) {
        throw new Error(`${id} ${name} requires a terminal entry`);
      }
    }
    this.polygonCount = props.polygonOffsets.length - 1;
    this.ringCount = props.ringOffsets.length - 1;
    this.indexCount = getPolygonTriangulationIndexCount(
      props.positions.length,
      this.ringCount,
      this.polygonCount
    );
    validateSinglePackedView(output.indices, ['uint32'], `${id} output.indices`);
    // Preserve acceptance of legacy buffers for non-degenerate layouts. The GPU checks each
    // actual slice; callers handling empty polygons should use the safe capacity helper.
    const minimumIndexCount =
      this.polygonCount === 0
        ? 0
        : Math.max(0, 3 * (props.positions.length + 2 * this.ringCount - 4 * this.polygonCount));
    if (output.indices.length < minimumIndexCount) {
      throw new Error(`${id} output.indices needs at least ${minimumIndexCount} rows`);
    }
    if (output.triangleOffsets) {
      validateSinglePackedView(output.triangleOffsets, ['uint32'], `${id} output.triangleOffsets`);
      if (output.triangleOffsets.length !== this.polygonCount + 1) {
        throw new Error(`${id} triangleOffsets length mismatch`);
      }
    }
    validateSinglePackedView(output.valid, ['uint32'], `${id} output.valid`);
    if (output.valid.length !== this.polygonCount) {
      throw new Error(`${id} output.valid length must equal the polygon count`);
    }
    if (output.triangleCount) {
      validateSinglePackedView(output.triangleCount, ['uint32'], `${id} output.triangleCount`);
      if (output.triangleCount.length !== this.polygonCount) {
        throw new Error(`${id} output.triangleCount length must equal the polygon count`);
      }
    }
    validatePolygonTriangulationWorkLimits(id, props);
    const writtenOutputs = {
      indices: output.indices,
      triangleOffsets: output.triangleOffsets,
      valid: this.polygonCount > 0 ? output.valid : undefined,
      triangleCount: this.polygonCount > 0 ? output.triangleCount : undefined
    };
    validateDisjointWritableViews(id, writtenOutputs, {separateDispatches: true});
    // The scan publishes triangleOffsets in an earlier dispatch; Earcut/monotone bind only
    // the writable result columns together and read triangle prefixes from private layout.
    validateDisjointWritableViews(id, {
      indices: output.indices,
      valid: writtenOutputs.valid,
      triangleCount: writtenOutputs.triangleCount
    });
    validateGraphOutputsDisjointFromInputs(id, Object.values(writtenOutputs), [
      props.positions,
      props.featureOffsets,
      props.polygonOffsets,
      props.ringOffsets
    ]);
  }

  private getViews(): (GraphDataView | undefined)[] {
    const {positions, featureOffsets, polygonOffsets, ringOffsets, output} = this.props;
    const {indices, valid, triangleCount, triangleOffsets} = output;
    return [
      positions,
      featureOffsets,
      polygonOffsets,
      ringOffsets,
      indices,
      this.polygonCount > 0 ? valid : undefined,
      this.polygonCount > 0 ? triangleCount : undefined,
      triangleOffsets
    ];
  }

  /**
   * Returns the Earcut node and, with `largePolygonMethod: 'monotone'`, the routing node before it
   * and the monotone-path nodes after it.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, polygonCount, ringCount} = this;
    const {output} = props;
    validateGraphChunkBindingRanges(
      graph,
      id,
      Object.fromEntries(this.getViews().map((view, index) => [`view ${index}`, view]))
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'triangle offsets scratch',
      (polygonCount + 1) * 4,
      'reduce the polygon count'
    );
    validatePolygonTriangulationScratch(
      graph.device,
      id,
      props.positions.length,
      ringCount,
      polygonCount,
      props.useZOrderHash ?? true
    );
    const vertexCount = props.positions.length;
    // The monotone path binds up to 40 bytes per input row; larger inputs keep today's behavior
    // (Earcut within `maximumWork`), so the default never makes an accepted input throw.
    const useMonotone =
      (props.largePolygonMethod ?? 'monotone') === 'monotone' &&
      vertexCount <= getGPUPolygonTriangulationMonotoneMaximumVertexCount(graph.device);
    if (useMonotone && props.monotoneMaximumVertices !== undefined) {
      validatePolygonTriangulationMonotoneCapacity(graph.device, id, props.monotoneMaximumVertices);
    }
    // Scratch nodes: 4 words each, with a capacity of 2 * vertices + 5 * holes + 8 per polygon.
    const nodeCount = getPolygonTriangulationNodeCount(vertexCount, ringCount, polygonCount);
    const useZOrderHash = props.useZOrderHash ?? true;
    // Words 0..4n-1 hold the ring nodes; the z-order list (z, previous, next) of node n follows at
    // 4 * nodeCount + 3 * n.
    const scratch = createTransientView(
      graph,
      `${id}-scratch`,
      'uint32',
      Math.max((useZOrderHash ? 7 : 4) * nodeCount, 4)
    );
    const triangleOffsets =
      output.triangleOffsets ??
      createTransientView(graph, `${id}-polygon-triangle-offsets`, 'uint32', polygonCount + 1);
    const {nodes, layout} = getPolygonTriangulationLayoutNodes(graph, {
      id,
      positions: props.positions,
      polygonOffsets: props.polygonOffsets,
      ringOffsets: props.ringOffsets,
      triangleOffsets
    });
    nodes.push(
      createFillNode(graph, {
        id: `${id}-clear-indices`,
        operation: OPERATION,
        view: output.indices,
        type: 'u32',
        value: '0u'
      })
    );
    if (polygonCount === 0) {
      return nodes;
    }
    const bindings: WGSLKernelBinding[] = [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'ringOffsets', view: layout.ringOffsets, type: 'u32', access: 'read'},
      {name: 'polygonOffsets', view: layout.polygonOffsets, type: 'u32', access: 'read'},
      {name: 'indices', view: output.indices, type: 'u32', access: 'read_write'},
      {name: 'valid', view: output.valid, type: 'u32', access: 'read_write'},
      {name: 'scratch', view: scratch, type: 'u32', access: 'read_write'}
    ];
    if (output.triangleCount) {
      bindings.push({
        name: 'triangleCount',
        view: output.triangleCount,
        type: 'u32',
        access: 'read_write'
      });
    }
    // Monotone method: the Earcut kernel also reads and writes the routing word of each polygon.
    const routing = useMonotone
      ? createTransientView(graph, `${id}-routing`, 'uint32', Math.max(polygonCount, 1))
      : undefined;
    if (routing) {
      bindings.push({name: 'routing', view: routing, type: 'u32', access: 'read_write'});
    }
    let monotoneNodes: GPUCommandNode<Parameters>[] = [];
    if (routing) {
      const monotoneProps = {
        id: `${id}-monotone`,
        operation: OPERATION,
        positions: props.positions,
        polygonOffsets: props.polygonOffsets,
        ringOffsets: props.ringOffsets,
        routing,
        layout,
        output,
        capacity: clamp(vertexCount, 1, props.monotoneMaximumVertices ?? vertexCount),
        minimumVertices:
          props.monotoneMinimumVertices ??
          GPU_POLYGON_TRIANGULATION_DEFAULT_MONOTONE_MINIMUM_VERTICES
      };
      monotoneNodes = getMonotoneTriangulationNodes(graph, monotoneProps);
    }
    const monotoneMinimumVertices =
      props.monotoneMinimumVertices ?? GPU_POLYGON_TRIANGULATION_DEFAULT_MONOTONE_MINIMUM_VERTICES;
    const createEarcutNode = (nodeId: string, routingPass?: 'first' | 'fallback') =>
      createWGSLKernelNode<Parameters>(graph, {
        id: nodeId,
        operation: OPERATION,
        variant: 'earcut',
        bindings,
        invocationCount: polygonCount,
        workgroupSize: 64,
        declarations: `${EXACT_ORIENTATION_WGSL}
// Only sosPointSide and exactCompareSign are called; exactOrientSigns (the helper's orientSign
// call site) is never called here and is dropped as dead code.
${EXACT_GEOMETRY_WGSL}
${getTriangulationWGSL({
  vertexCount,
  polygonCount,
  ringCount,
  indexLength: output.indices.length,
  nodeCount,
  useZOrderHash,
  maximumWork: props.maximumWork ?? GPU_POLYGON_TRIANGULATION_DEFAULT_MAXIMUM_WORK,
  writeTriangleCount: Boolean(output.triangleCount),
  routingPass,
  minimumVertices: monotoneMinimumVertices
})}`,
        body: 'triangulatePolygon(index);'
      });
    nodes.push(createEarcutNode(`${id}-earcut`, routing ? 'first' : undefined));
    // Polygons that the monotone path flagged invalid get a second chance with Earcut (route 4),
    // so the method never does worse than `'none'`: a valid polygon with rings that touch, or one
    // beyond the monotone capacity, is still triangulated within `maximumWork`.
    if (routing) {
      monotoneNodes.push(createEarcutNode(`${id}-earcut-fallback`, 'fallback'));
    }
    return [...nodes, ...monotoneNodes];
  }
}
