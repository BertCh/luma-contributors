// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createStorageTopKNode} from '../../analysis-core/kernels/storage-backed-top-k';

import {
  createTransientView,
  GPUSort,
  type GPUBVH,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateSpatialJoinReadWriteBindings} from './spatial-join-binding-preflight';
import type {GPUFloat32Positions} from '@luma.gl/graph/geospatial';
import {
  getGraphViewChunks,
  validateGPUStorageBindingByteLength,
  validateCommandNodeBudget
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  NEAREST_COMMON_WGSL,
  NEAREST_DISTANCE_RANK_WGSL,
  NEAREST_GEODESIC_WGSL,
  NEAREST_HEADER_WORDS,
  NEAREST_NODE_WORDS,
  getNearestPairWGSL,
  getNearestSideWGSL,
  getNearestTraversalWGSL
} from './nearest-wgsl';
import type {GPUNearestFeatureGeometry, NearestSide} from './nearest-types';
import {
  createBoundsNode as createSpatialJoinGeometryBoundsNode,
  isMultipartJoinGeometry
} from './spatial-join-geometry';
import type {GPUSpatialJoinPrepared} from './gpu-spatial-join-prepared';
import type {GPUSpatialJoinOnAttribute} from './spatial-join-types';
import {
  createSpatialJoinBoundsNode,
  QUANTIZE_AXIS_WGSL,
  HILBERT_INDEX_WGSL,
  SPATIAL_JOIN_WGSL_HELPERS,
  getChunkNodeId,
  getHilbertQueryOrderNodes,
  getSortedFeatureBVHNodes,
  type SpatialSortCurve
} from './spatial-join-passes';
import type {
  GPULineGeometryPort,
  GPUMultipartGeometryPort,
  GPUPackedGeometryPort,
  GPUPolygonGeometryPort
} from '../../analysis-core/contracts/index';

/** u32 words per `(query, slot)` of the foot-point scratch: foot x/y, segment, query-point x/y. */
const EXTRA_WORDS = 5;

/** Scratch batches reuse row zero, so even one query is a valid aligned storage binding. */
const MINIMUM_SCRATCH_QUERY_COUNT = 1;

/** Covers published BVH construction, radix sorting, rings, bounds and fixed status passes. */
const NEAREST_BUILD_COMMAND_NODE_ALLOWANCE = 4096;
/** One scratch chunk needs at most clear, traversal, feet, two column passes and cursor nodes. */
const NEAREST_SCRATCH_COMMAND_NODE_ALLOWANCE = 8;
/** Covers two shared radix sorts, their scans and optional geometry-center passes. */
const NEAREST_QUERY_SORT_COMMAND_NODE_ALLOWANCE = 1024;

/** Conservative command count including device-sized scratch chunks. @internal */
export function getNearestEstimatedCommandNodeCount(
  queryCount: number,
  capacity: number,
  needsFootPass: boolean,
  queryChunkCount: number,
  bindingLimitBytes: number,
  forcedScratchRows?: number,
  querySort = false
): number {
  const wordsPerQuery = Math.max(
    NEAREST_HEADER_WORDS + capacity * 2,
    needsFootPass ? capacity * EXTRA_WORDS : 0
  );
  const scratchRows =
    forcedScratchRows ?? Math.max(1, Math.floor((bindingLimitBytes / 4 - 3) / wordsPerQuery));
  return (
    NEAREST_BUILD_COMMAND_NODE_ALLOWANCE +
    (querySort ? NEAREST_QUERY_SORT_COMMAND_NODE_ALLOWANCE : 0) +
    NEAREST_SCRATCH_COMMAND_NODE_ALLOWANCE * (Math.ceil(queryCount / scratchRows) + queryChunkCount)
  );
}

/** Query side of the k-nearest mode: chunked points or one geometry set. @internal */
export type NearestQuerySource =
  | {kind: 'points'; points: GPUFloat32Positions}
  | {
      kind: 'geometry';
      geometry: GPUPackedGeometryPort<
        GPULineGeometryPort | GPUPolygonGeometryPort | GPUMultipartGeometryPort
      >;
    };

/** The BVH facts the nearest kernels read; satisfied by `GPUBVH` and by prepared storage. @internal */
export type NearestBVH = Pick<
  GPUBVH,
  | 'nodeMinima'
  | 'nodeMaxima'
  | 'leafIds'
  | 'overflow'
  | 'internalNodeCount'
  | 'nodeCount'
  | 'levelCount'
>;

/** Inputs of {@link getNearestNeighborNodes}. @internal */
export type NearestNeighborNodeProps = {
  /** Prepared right-hand side whose BVH and ring ranges replace the per-encoding build. */
  prepared?: GPUSpatialJoinPrepared<never>;
  id: string;
  operation: string;
  queries: NearestQuerySource;
  queryCount: number;
  features: GPUNearestFeatureGeometry;
  featureCount: number;
  neighborCount: number;
  capacity: number;
  ties?: 'all' | 'lowest-row';
  leafCapacity: number;
  spatialSort: boolean;
  spatialSortCurve?: SpatialSortCurve;
  maxDistance?: GraphDataView<'float32'>;
  featureIds?: GraphDataView<'uint32'>;
  neighborIds: GraphDataView<'uint32'>;
  /** Original feature rows before optional application-ID remapping. */
  neighborRows?: GraphDataView<'uint32'>;
  neighborCounts: GraphDataView<'uint32'>;
  neighborDistances?: GraphDataView<'float32'>;
  neighborFootPoints?: GraphDataView<'float32x2'>;
  neighborQueryPoints?: GraphDataView<'float32x2'>;
  exclusive?: boolean;
  queryIds?: GraphDataView<'uint32'>;
  onAttribute?: GPUSpatialJoinOnAttribute;
  neighborSegmentIndices?: GraphDataView<'uint32'>;
  overflow: GraphDataView<'uint32'>;
  /**
   * When set, positions are longitude/latitude degrees and distances are great-circle distances in
   * the units of this sphere radius (meters for the mean Earth radius), including `maxDistance`
   * and `neighborDistances`. Omitted means planar distances in coordinate units.
   */
  sphereRadius?: number;
  /**
   * Write feature rows (not `featureIds` values) to `neighborIds`; `featureIds` then only feeds
   * the `exclusive` filter.
   */
  writeFeatureRows?: boolean;
  /**
   * Queries per scratch batch, a positive integer. By default the per-query scratch
   * (`topk` and, with foot-point outputs, `extras`) is one binding for every query while that
   * fits `maxStorageBufferBindingSize`, and otherwise the largest batch that fits.
   * Setting it forces chunking once `queryCount` exceeds it, which lets specs exercise the
   * chunked path with few queries.
   */
  scratchChunkQueryCount?: number;
  /**
   * Process queries in Hilbert order of points or geometry bounds centers over the BVH root bounds
   * (see `getHilbertQueryOrderNodes`) instead of input order. Outputs are identical; only
   * traversal coherence changes. Ignored for chunks of fewer than two queries. Default false here; the joins
   * resolve their own default.
   */
  querySort?: boolean;
};

/** Returns the number of feature rows of a nearest geometry. @internal */
export function getNearestGeometryCount(geometry: GPUNearestFeatureGeometry): number {
  if (isMultipartJoinGeometry(geometry)) return geometry.featureOffsets.length - 1;
  switch (geometry.kind) {
    case 'points':
      return geometry.positions.length;
    case 'segments':
      return geometry.starts.length;
    case 'lines':
      return geometry.lineOffsets.length - 1;
    case 'polygons':
      return geometry.featureOffsets.length - 1;
  }
}

/** Returns every input view of a nearest geometry. @internal */
export function getNearestGeometryViews(geometry: GPUNearestFeatureGeometry): GraphDataView[] {
  switch (geometry.kind) {
    case 'points':
      return [
        geometry.positions,
        ...(isMultipartJoinGeometry(geometry) ? [geometry.featureOffsets] : [])
      ];
    case 'segments':
      return [geometry.starts, geometry.ends];
    case 'lines':
      return [
        geometry.positions,
        geometry.lineOffsets,
        ...(isMultipartJoinGeometry(geometry) ? [geometry.featureOffsets] : [])
      ];
    case 'polygons':
      return [
        geometry.positions,
        geometry.featureOffsets,
        geometry.polygonOffsets,
        geometry.ringOffsets
      ];
  }
}

function getSideVertexCount(geometry: GPUNearestFeatureGeometry): number {
  return geometry.kind === 'segments' ? geometry.starts.length * 2 : geometry.positions.length;
}

/** Storage bindings read by the accessors of one geometry side. */
function getSideBindings(
  prefix: 'q' | 'r',
  geometry: GPUNearestFeatureGeometry,
  featureRings: GraphDataView<'uint32x2'> | undefined
): WGSLKernelBinding[] {
  const read = (name: string, view: GraphDataView, type: 'f32' | 'u32'): WGSLKernelBinding => ({
    name: `${prefix}${name}`,
    view,
    type,
    access: 'read'
  });
  switch (geometry.kind) {
    case 'points':
      return [
        read('Positions', geometry.positions, 'f32'),
        ...(isMultipartJoinGeometry(geometry)
          ? [read('FeatureOffsets', geometry.featureOffsets, 'u32')]
          : [])
      ];
    case 'segments':
      return [read('Starts', geometry.starts, 'f32'), read('Ends', geometry.ends, 'f32')];
    case 'lines':
      return [
        read('Positions', geometry.positions, 'f32'),
        read('LineOffsets', geometry.lineOffsets, 'u32'),
        ...(isMultipartJoinGeometry(geometry)
          ? [read('FeatureOffsets', geometry.featureOffsets, 'u32')]
          : [])
      ];
    case 'polygons':
      return [
        read('Positions', geometry.positions, 'f32'),
        read('FeatureRings', featureRings!, 'u32'),
        read('RingOffsets', geometry.ringOffsets, 'u32')
      ];
  }
}

/** Writes `[ringStart, ringEnd)` of every polygon feature, clamped against malformed offsets. */
function createPolygonRingsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  operation: string,
  geometry: Extract<GPUNearestFeatureGeometry, {kind: 'polygons'}>,
  featureRings: GraphDataView<'uint32x2'>
): GPUCommandNode<Parameters> {
  return createStorageTopKNode<Parameters>(graph, {
    id,
    operation,
    variant: 'polygon-rings',
    bindings: [
      {name: 'featureOffsets', view: geometry.featureOffsets, type: 'u32', access: 'read'},
      {name: 'polygonOffsets', view: geometry.polygonOffsets, type: 'u32', access: 'read'},
      {name: 'featureRings', view: featureRings, type: 'u32', access: 'read_write'}
    ],
    invocationCount: featureRings.length,
    declarations: `const POLYGON_COUNT: u32 = ${geometry.polygonOffsets.length - 1}u;
const RING_COUNT: u32 = ${geometry.ringOffsets.length - 1}u;`,
    body: `let polygonStart = featureOffsets[featureOffsetsOffset + index];
  let polygonEnd = featureOffsets[featureOffsetsOffset + index + 1u];
  var ringStart = 0u;
  var ringEnd = 0u;
  if (polygonStart <= polygonEnd && polygonEnd <= POLYGON_COUNT) {
    let first = polygonOffsets[polygonOffsetsOffset + polygonStart];
    let last = polygonOffsets[polygonOffsetsOffset + polygonEnd];
    if (first <= last && last <= RING_COUNT) { ringStart = first; ringEnd = last; }
  }
  featureRings[featureRingsOffset + index * 2u] = ringStart;
  featureRings[featureRingsOffset + index * 2u + 1u] = ringEnd;`
  });
}

/** Writes per-feature bounds for every feature kind, including linestrings. */
function createBoundsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  operation: string,
  features: GPUNearestFeatureGeometry,
  featureCount: number,
  minima: GraphDataView<'float32x2'>,
  maxima: GraphDataView<'float32x2'>
): GPUCommandNode<Parameters> {
  if (features.kind === 'lines' || isMultipartJoinGeometry(features)) {
    return createSpatialJoinGeometryBoundsNode<Parameters>(graph, id, {
      operation,
      featureCount,
      geometry: features,
      minima,
      maxima
    });
  }
  return createSpatialJoinBoundsNode<Parameters>(graph, {
    id,
    operation,
    featureCount,
    source: features,
    minima,
    maxima
  });
}

/**
 * Returns the feature BVH and the nodes that build it. With a prepared handle there are no
 * nodes: the handle's own (conditional) build nodes fill its persistent storage.
 *
 * @internal
 */
export function getNearestBVHNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    features: GPUNearestFeatureGeometry;
    featureCount: number;
    leafCapacity: number;
    spatialSort: boolean;
    spatialSortCurve?: SpatialSortCurve;
    prepared?: GPUSpatialJoinPrepared<never>;
  }
): {bvh: NearestBVH; nodes: readonly GPUCommandNode<Parameters>[]} {
  const {id, operation, features, featureCount, prepared} = props;
  if (prepared) {
    if (!prepared.isDeclaredIn(graph)) {
      throw new Error(`${id} prepared handle must be added to the graph before the join`);
    }
    const {storage} = prepared;
    return {
      bvh: {
        nodeMinima: storage.nodeMinima,
        nodeMaxima: storage.nodeMaxima,
        leafIds: storage.leafIds,
        overflow: storage.overflow,
        internalNodeCount: prepared.leafCapacity - 1,
        nodeCount: 2 * prepared.leafCapacity - 1,
        levelCount: Math.log2(prepared.leafCapacity) + 1
      },
      nodes: []
    };
  }
  const minima = createTransientView(graph, `${id}-feature-minima`, 'float32x2', featureCount);
  const maxima = createTransientView(graph, `${id}-feature-maxima`, 'float32x2', featureCount);
  const nodes: GPUCommandNode<Parameters>[] = [];
  if (featureCount > 0) {
    nodes.push(
      createBoundsNode(graph, `${id}-bounds`, operation, features, featureCount, minima, maxima)
    );
  }
  const built = getSortedFeatureBVHNodes(
    graph,
    id,
    operation,
    minima,
    maxima,
    props.leafCapacity,
    props.spatialSort,
    undefined,
    props.spatialSortCurve
  );
  nodes.push(...built.nodes);
  return {bvh: built.bvh, nodes};
}

/** Validates all storage ranges before nearest nodes or resources are declared. @internal */
export function validateNearestNeighborBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: NearestNeighborNodeProps
) {
  const {id, capacity, queryCount, featureCount} = props;
  validateSpatialJoinReadWriteBindings(
    id,
    [props.writeFeatureRows ? undefined : props.featureIds],
    [
      props.neighborIds,
      props.neighborRows,
      props.neighborCounts,
      props.neighborDistances,
      props.overflow
    ]
  );
  const stride = NEAREST_HEADER_WORDS + 2 * capacity;
  const needsFootPass = Boolean(
    props.neighborFootPoints || props.neighborQueryPoints || props.neighborSegmentIndices
  );

  // The per-query scratch is `topk` (stride words) and, with foot-point outputs, `extras`
  // (capacity x EXTRA_WORDS words). When one binding cannot hold every query, the scratch holds a
  // fixed chunk of queries and each chunk of the query side reuses it in turn.
  const bindingLimitWords = Math.floor(graph.device.limits.maxStorageBufferBindingSize / 4);
  const scratchWordsPerQuery = Math.max(stride, needsFootPass ? capacity * EXTRA_WORDS : 0);
  const forcedChunkQueryCount = props.scratchChunkQueryCount;
  if (
    forcedChunkQueryCount !== undefined &&
    (!Number.isSafeInteger(forcedChunkQueryCount) ||
      forcedChunkQueryCount < MINIMUM_SCRATCH_QUERY_COUNT)
  ) {
    throw new Error(
      `${id} scratchChunkQueryCount must be an integer at least MINIMUM_SCRATCH_QUERY_COUNT (1)`
    );
  }
  // Three tail words hold the cursor, dispatch first row and query count. Each batch starts
  // at scratch row zero; its row count need not be a multiple of a workgroup or binding alignment.
  const maximumScratchQueryCount = Math.floor((bindingLimitWords - 3) / scratchWordsPerQuery);
  const scratchQueryCount = Math.min(
    Math.max(queryCount, 1),
    forcedChunkQueryCount ?? maximumScratchQueryCount,
    maximumScratchQueryCount
  );
  const isChunked = queryCount > scratchQueryCount;
  if (scratchQueryCount < 1) {
    throw new Error(
      `${id} needs more scratch per query than one binding holds; reduce neighborCount`
    );
  }
  // Word of `topk` that holds the first query (relative to the current query-side chunk) of the
  // scratch chunk being processed. A 1-thread node advances it, so the chunk index never reaches
  // the WGSL text or the pipeline constants.
  const cursorWord = scratchQueryCount * stride;
  const dispatchWord = cursorWord + 1;
  validateGPUStorageBindingByteLength(
    graph.device,
    id,
    'topk',
    (scratchQueryCount * stride + 3) * 4,
    'reduce the query count or neighborCount'
  );

  const queryViews =
    props.queries.kind === 'points'
      ? getGraphViewChunks(props.queries.points)
      : getNearestGeometryViews(props.queries.geometry);
  const boundViews = [
    ...queryViews,
    ...getNearestGeometryViews(props.features),
    props.featureIds,
    props.maxDistance,
    props.queryIds,
    props.onAttribute?.left,
    props.onAttribute?.right,
    props.neighborIds,
    props.neighborRows,
    props.neighborCounts,
    props.neighborDistances,
    props.neighborFootPoints,
    props.neighborQueryPoints,
    props.neighborSegmentIndices,
    props.overflow,
    ...(props.prepared ? Object.values(props.prepared.storage) : [])
  ];
  validateGraphChunkBindingRanges(
    graph,
    id,
    Object.fromEntries(
      boundViews
        .filter((view): view is GraphDataView =>
          Boolean(view && typeof view === 'object' && 'rowByteLength' in view)
        )
        .map((view, index) => [`nearest view ${index}`, view])
    )
  );
  const nodeCount = 2 * props.leafCapacity - 1;
  const hasFilter = Boolean(props.exclusive || props.onAttribute);
  for (const [name, words] of [
    [
      'nodeData',
      nodeCount * NEAREST_NODE_WORDS + 2 * queryCount + (hasFilter ? 2 * featureCount : 0)
    ],
    ['BVH node bounds and children', nodeCount * 2],
    ['feature bounds and sorting scratch', featureCount * 2],
    ['query sorting scratch and bounds', props.querySort ? queryCount * 2 : 0],
    ['extras', needsFootPass ? scratchQueryCount * capacity * EXTRA_WORDS : 0],
    ['feature rings', props.features.kind === 'polygons' ? featureCount * 2 : 0],
    [
      'query rings',
      props.queries.kind === 'geometry' && props.queries.geometry.kind === 'polygons'
        ? queryCount * 2
        : 0
    ],
    [
      'query metadata',
      props.querySort &&
      queryCount >= 2 &&
      props.queries.kind === 'geometry' &&
      props.queries.geometry.kind === 'polygons'
        ? queryCount * 2 + props.queries.geometry.ringOffsets.length
        : props.querySort &&
            queryCount >= 2 &&
            props.queries.kind === 'geometry' &&
            props.queries.geometry.kind === 'lines' &&
            isMultipartJoinGeometry(props.queries.geometry)
          ? props.queries.geometry.featureOffsets.length + props.queries.geometry.lineOffsets.length
          : 0
    ]
  ] as const) {
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      name,
      words * 4,
      name === 'query metadata'
        ? 'reduce query topology rows or disable querySort'
        : 'reduce feature count, query count or neighborCapacity'
    );
  }
  validateCommandNodeBudget(
    id,
    getNearestEstimatedCommandNodeCount(
      queryCount,
      capacity,
      needsFootPass,
      props.queries.kind === 'points' ? getGraphViewChunks(props.queries.points).length : 1,
      graph.device.limits.maxStorageBufferBindingSize,
      props.scratchChunkQueryCount,
      props.querySort
    )
  );
  return {stride, needsFootPass, scratchQueryCount, isChunked, cursorWord, dispatchWord};
}

/**
 * Classify each actual point chunk, sort all rows by Hilbert key, then stably group that order by
 * input chunk. Chunk slices of the final permutation contain global rows in Hilbert order without
 * padding short input chunks or compiling a different sort pipeline for every chunk length.
 */
function getNearestPointQueryOrderNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  operation: string,
  chunks: readonly GraphDataView<'float32x2'>[],
  queryCount: number,
  bvh: NearestBVH
): {nodes: GPUCommandNode<Parameters>[]; order: GraphDataView<'uint32'>} {
  const keys = createTransientView(graph, `${id}-query-order-keys`, 'uint32', queryCount);
  const rows = createTransientView(graph, `${id}-query-order-rows`, 'uint32', queryCount);
  const chunkIds = createTransientView(graph, `${id}-query-order-chunks`, 'uint32', queryCount);
  const sortedKeys = createTransientView(
    graph,
    `${id}-query-order-sorted-keys`,
    'uint32',
    queryCount
  );
  const sortedRows = createTransientView(
    graph,
    `${id}-query-order-sorted-rows`,
    'uint32',
    queryCount
  );
  const groupKeys = createTransientView(
    graph,
    `${id}-query-order-group-keys`,
    'uint32',
    queryCount
  );
  const outputKeys = createTransientView(
    graph,
    `${id}-query-order-output-keys`,
    'uint32',
    queryCount
  );
  const order = createTransientView(graph, `${id}-query-order`, 'uint32', queryCount);
  const nodes: GPUCommandNode<Parameters>[] = [];
  let firstRow = 0;
  for (const [chunkIndex, points] of chunks.entries()) {
    if (points.length)
      nodes.push(
        createStorageTopKNode(graph, {
          id: `${id}-query-keys-${chunkIndex}`,
          operation,
          variant: 'query-keys-hilbert',
          firstRow,
          batchIndex: chunkIndex,
          invocationCount: points.length,
          bindings: [
            {name: 'points', view: points, type: 'f32', access: 'read'},
            {name: 'nodeMinima', view: bvh.nodeMinima, type: 'f32', access: 'read'},
            {name: 'nodeMaxima', view: bvh.nodeMaxima, type: 'f32', access: 'read'},
            {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
            {name: 'rows', view: rows, type: 'u32', access: 'read_write'},
            {name: 'chunkIds', view: chunkIds, type: 'u32', access: 'read_write'}
          ],
          declarations: `${SPATIAL_JOIN_WGSL_HELPERS}${QUANTIZE_AXIS_WGSL}${HILBERT_INDEX_WGSL}`,
          body: `let row = batchFirstRow + index;
  rows[rowsOffset + row] = row;
  chunkIds[chunkIdsOffset + row] = batchIndex;
  let point = vec2f(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  var key = 0xffffffffu;
  if (isFiniteValue(point.x) && isFiniteValue(point.y)) {
    let column = quantizeAxis(point.x, nodeMinima[nodeMinimaOffset], nodeMaxima[nodeMaximaOffset]);
    let rasterRow = quantizeAxis(point.y, nodeMinima[nodeMinimaOffset + 1u], nodeMaxima[nodeMaximaOffset + 1u]);
    key = getHilbertIndex(column, rasterRow);
  }
  keys[keysOffset + row] = key;`
        })
      );
    firstRow += points.length;
  }
  nodes.push(
    ...new GPUSort({
      id: `${id}-query-hilbert-sort`,
      keys,
      values: rows,
      outputKeys: sortedKeys,
      outputValues: sortedRows,
      algorithm: 'radix'
    }).getCommandNodes(graph)
  );
  nodes.push(
    createStorageTopKNode(graph, {
      id: `${id}-query-chunk-keys`,
      operation,
      variant: 'query-chunk-keys',
      invocationCount: queryCount,
      bindings: [
        {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'},
        {name: 'chunkIds', view: chunkIds, type: 'u32', access: 'read'},
        {name: 'groupKeys', view: groupKeys, type: 'u32', access: 'read_write'}
      ],
      body: 'groupKeys[groupKeysOffset + index] = chunkIds[chunkIdsOffset + sortedRows[sortedRowsOffset + index]];'
    })
  );
  nodes.push(
    ...new GPUSort({
      id: `${id}-query-chunk-sort`,
      keys: groupKeys,
      values: sortedRows,
      outputKeys,
      outputValues: order,
      algorithm: 'radix'
    }).getCommandNodes(graph)
  );
  return {nodes, order};
}

/**
 * Builds the best-first k-nearest pipeline: bounds, BVH, packed node table, polygon ring ranges,
 * header clear, one traversal per query chunk, optional foot-point pass, and column writes.
 *
 * Expensive foot-point evaluation dispatches once per query and visits only the neighbors actually
 * found. Column passes still initialize every caller-owned output slot.
 *
 * @internal
 */
export function getNearestNeighborNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: NearestNeighborNodeProps
): {nodes: GPUCommandNode<Parameters>[]; bvh: NearestBVH} {
  if (props.prepared) {
    props = {
      ...props,
      features: {
        ...props.prepared.getGeometry(graph),
        sourceIds: props.features.sourceIds
      }
    };
  }
  const {id, operation, features, featureCount, queryCount, neighborCount, capacity, sphereRadius} =
    props;
  const {stride, needsFootPass, scratchQueryCount, isChunked, cursorWord, dispatchWord} =
    validateNearestNeighborBindings(graph, props);
  const greatCircle = sphereRadius !== undefined;
  const sphereRadiusLiteral = greatCircle ? getWGSLFloatLiteral(Math.fround(sphereRadius)) : '';
  const nodes: GPUCommandNode<Parameters>[] = [];

  const {bvh, nodes: bvhNodes} = getNearestBVHNodes(graph, {
    id,
    operation,
    features,
    featureCount,
    leafCapacity: props.leafCapacity,
    spatialSort: props.spatialSort,
    spatialSortCurve: props.spatialSortCurve,
    prepared: props.prepared
  });
  nodes.push(...bvhNodes);

  // One packed node table: min.xy, max.xy as f32 bits, and the feature row of leaves.
  // The tail holds the candidate filter: `[id, key]` per query row, then per feature row. It rides
  // in this buffer because a polygon/polygon traversal already binds eight storage buffers.
  const hasFilter = Boolean(props.exclusive || props.onAttribute);
  const queryFilterBase = bvh.nodeCount * NEAREST_NODE_WORDS;
  const featureFilterBase = queryFilterBase + 2 * queryCount;
  const nodeData = createTransientView(
    graph,
    `${id}-node-data`,
    'uint32',
    featureFilterBase + (hasFilter ? 2 * featureCount : 0)
  );
  nodes.push(
    createStorageTopKNode<Parameters>(graph, {
      id: `${id}-node-data`,
      operation,
      variant: 'node-data',
      bindings: [
        {name: 'nodeMinima', view: bvh.nodeMinima, type: 'f32', access: 'read'},
        {name: 'nodeMaxima', view: bvh.nodeMaxima, type: 'f32', access: 'read'},
        {name: 'leafIds', view: bvh.leafIds, type: 'u32', access: 'read'},
        {name: 'nodeData', view: nodeData, type: 'u32', access: 'read_write'}
      ],
      invocationCount: bvh.nodeCount,
      declarations: `const INTERNAL_NODE_COUNT: u32 = ${bvh.internalNodeCount}u;
const NO_FEATURE: u32 = 0xffffffffu;`,
      body: `let base = nodeDataOffset + index * ${NEAREST_NODE_WORDS}u;
  nodeData[base] = bitcast<u32>(nodeMinima[nodeMinimaOffset + index * 2u]);
  nodeData[base + 1u] = bitcast<u32>(nodeMinima[nodeMinimaOffset + index * 2u + 1u]);
  nodeData[base + 2u] = bitcast<u32>(nodeMaxima[nodeMaximaOffset + index * 2u]);
  nodeData[base + 3u] = bitcast<u32>(nodeMaxima[nodeMaximaOffset + index * 2u + 1u]);
  nodeData[base + 4u] = select(NO_FEATURE, leafIds[leafIdsOffset + index - INTERNAL_NODE_COUNT], index >= INTERNAL_NODE_COUNT);`
    })
  );

  if (hasFilter) {
    const filterBindings: WGSLKernelBinding[] = [
      {name: 'nodeData', view: nodeData, type: 'u32', access: 'read_write'}
    ];
    const optionalInputs: [string, GraphDataView<'uint32'> | undefined][] = [
      ['queryIds', props.exclusive ? props.queryIds : undefined],
      ['featureIds', props.exclusive ? props.featureIds : undefined],
      ['queryKeys', props.onAttribute?.left],
      ['featureKeys', props.onAttribute?.right]
    ];
    for (const [name, view] of optionalInputs) {
      if (view) {
        filterBindings.push({name, view, type: 'u32', access: 'read'});
      }
    }
    const read = (name: string, fallback: string) =>
      optionalInputs.find(([inputName, view]) => inputName === name && view)
        ? `${name}[${name}Offset + index]`
        : fallback;
    nodes.push(
      createStorageTopKNode<Parameters>(graph, {
        id: `${id}-filter`,
        operation,
        variant: 'filter',
        bindings: filterBindings,
        invocationCount: Math.max(queryCount, featureCount, 1),
        declarations: `const QUERY_COUNT: u32 = ${queryCount}u;
const FEATURE_COUNT: u32 = ${featureCount}u;`,
        body: `if (index < QUERY_COUNT) {
    let base = nodeDataOffset + ${queryFilterBase}u + index * 2u;
    nodeData[base] = ${read('queryIds', 'index')};
    nodeData[base + 1u] = ${read('queryKeys', '0u')};
  }
  if (index < FEATURE_COUNT) {
    let base = nodeDataOffset + ${featureFilterBase}u + index * 2u;
    nodeData[base] = ${read('featureIds', 'index')};
    nodeData[base + 1u] = ${read('featureKeys', '0u')};
  }`
      })
    );
  }

  // Polygon ring ranges for the feature side and for a polygon query side.
  let featureRings: GraphDataView<'uint32x2'> | undefined;
  if (features.kind === 'polygons' && props.prepared) {
    featureRings = props.prepared.storage.featureRings;
  } else if (features.kind === 'polygons') {
    featureRings = createTransientView(graph, `${id}-feature-rings`, 'uint32x2', featureCount);
    if (featureCount > 0) {
      nodes.push(
        createPolygonRingsNode(graph, `${id}-feature-rings`, operation, features, featureRings)
      );
    }
  }
  const queryGeometry = props.queries.kind === 'geometry' ? props.queries.geometry : undefined;
  let queryRings: GraphDataView<'uint32x2'> | undefined;
  if (queryGeometry?.kind === 'polygons') {
    queryRings = createTransientView(graph, `${id}-query-rings`, 'uint32x2', queryCount);
    if (queryCount > 0) {
      nodes.push(
        createPolygonRingsNode(graph, `${id}-query-rings`, operation, queryGeometry, queryRings)
      );
    }
  }

  const topk = createTransientView(graph, `${id}-topk`, 'uint32', scratchQueryCount * stride + 3);
  const clearBindings: WGSLKernelBinding[] = [
    {name: 'topk', view: topk, type: 'u32', access: 'read_write'},
    {name: 'overflow', view: props.overflow, type: 'u32', access: 'read_write'},
    {name: 'bvhOverflow', view: bvh.overflow, type: 'u32', access: 'read'}
  ];
  if (props.maxDistance) {
    clearBindings.push({name: 'maxDistance', view: props.maxDistance, type: 'f32', access: 'read'});
  }
  // Keep the original planar radius for exact threshold fallback. Great-circle searches store
  // their central-angle bound; the planar traversal encodes its rank after reading the radius.
  const clearBoundWGSL = `// A NaN, negative or infinite maxDistance sets a reserved finite rank that matches nothing.
    ${
      props.maxDistance
        ? `let limit = maxDistance[maxDistanceOffset];
    var bound = NEAREST_INVALID_DISTANCE;
    if (limit >= 0.0 && limit <= FLOAT32_MAXIMUM) { bound = ${
      greatCircle
        ? '// Canonicalize negative zero before storing the central-angle rank.\n    abs(limit / SPHERE_RADIUS)'
        : 'abs(limit)'
    }; }`
        : 'let bound = NEAREST_NO_DISTANCE;'
    }
    topk[header] = 0u;
    topk[header + 1u] = 0u;
    topk[header + 2u] = bitcast<u32>(bound);`;
  const clearDeclarations = `${NEAREST_DISTANCE_RANK_WGSL}
const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;
const STRIDE: u32 = ${stride}u;${greatCircle ? `\nconst SPHERE_RADIUS: f32 = ${sphereRadiusLiteral};` : ''}`;
  if (isChunked) {
    // The overflow flag is initialized once; the per-chunk clears below only reset the scratch.
    nodes.push(
      createStorageTopKNode<Parameters>(graph, {
        id: `${id}-clear-overflow`,
        operation,
        variant: 'clear-overflow',
        bindings: clearBindings.filter(binding => binding.name.endsWith('verflow')),
        invocationCount: 1,
        body: 'overflow[overflowOffset] = bvhOverflow[bvhOverflowOffset];'
      })
    );
  } else {
    nodes.push(
      createStorageTopKNode<Parameters>(graph, {
        id: `${id}-clear`,
        operation,
        variant: greatCircle ? 'clear-great-circle' : 'clear',
        bindings: clearBindings,
        invocationCount: Math.max(queryCount, 1),
        declarations: `${clearDeclarations}
const QUERY_COUNT: u32 = ${queryCount}u;`,
        body: `if (index == 0u) { overflow[overflowOffset] = bvhOverflow[bvhOverflowOffset]; }
  if (index < QUERY_COUNT) {
    let header = topkOffset + index * STRIDE;
    ${clearBoundWGSL}
  }`
      })
    );
  }

  // Query dispatches: one per nonempty point chunk, or one for the whole geometry set.
  type QueryDispatch = {
    chunkId: string;
    side: NearestSide;
    bindings: WGSLKernelBinding[];
    count: number;
    firstRow: number;
    kind: NearestSide['kind'];
    /** Hilbert processing order of this dispatch (dispatch-local rows), when sorted. */
    order?: GraphDataView<'uint32'>;
  };
  const dispatches: QueryDispatch[] = [];
  if (props.queries.kind === 'points') {
    const queryChunks = getGraphViewChunks(props.queries.points);
    const ordered =
      props.querySort && queryCount > 1
        ? getNearestPointQueryOrderNodes(graph, id, operation, queryChunks, queryCount, bvh)
        : undefined;
    if (ordered) nodes.push(...ordered.nodes);
    let firstRow = 0;
    for (const [chunkIndex, chunk] of queryChunks.entries()) {
      if (chunk.length > 0) {
        const geometry = {kind: 'points', positions: chunk} as const;
        const order = ordered
          ? graph.createDataView(ordered.order.buffer, {
              format: 'uint32',
              byteOffset: firstRow * 4,
              length: chunk.length
            })
          : undefined;
        dispatches.push({
          chunkId: getChunkNodeId(id, props.queries.points, chunkIndex),
          side: {prefix: 'q', kind: 'points', vertexCount: chunk.length},
          bindings: getSideBindings('q', geometry, undefined),
          count: chunk.length,
          firstRow,
          kind: 'points',
          order
        });
      }
      firstRow += chunk.length;
    }
  } else if (queryCount > 0) {
    const geometry = props.queries.geometry;
    const side: NearestSide = {
      prefix: 'q',
      kind: geometry.kind,
      vertexCount: getSideVertexCount(geometry),
      partCount: isMultipartJoinGeometry(geometry)
        ? geometry.kind === 'points'
          ? geometry.positions.length
          : geometry.lineOffsets.length - 1
        : undefined
    };
    let bindings = getSideBindings('q', geometry, queryRings);
    let order: GraphDataView<'uint32'> | undefined;
    if (props.querySort && queryCount >= 2) {
      // Polygon/polygon traversal and foot-point passes already bind eight buffers. Pack the
      // query's small read-only topology arrays together to make room for the permutation.
      if (geometry.kind === 'polygons') {
        const ringOffsetsBase = queryCount * 2;
        validateGPUStorageBindingByteLength(
          graph.device,
          id,
          'query metadata',
          (ringOffsetsBase + geometry.ringOffsets.length) * 4,
          'reduce the polygon query count or disable querySort'
        );
        const metadata = createTransientView(
          graph,
          `${id}-query-metadata`,
          'uint32',
          ringOffsetsBase + geometry.ringOffsets.length
        );
        nodes.push(
          createStorageTopKNode<Parameters>(graph, {
            id: `${id}-query-metadata`,
            operation,
            variant: 'query-metadata',
            bindings: [
              {name: 'featureRings', view: queryRings!, type: 'u32', access: 'read'},
              {name: 'ringOffsets', view: geometry.ringOffsets, type: 'u32', access: 'read'},
              {name: 'metadata', view: metadata, type: 'u32', access: 'read_write'}
            ],
            invocationCount: metadata.length,
            body: `if (index < ${ringOffsetsBase}u) {
    metadata[metadataOffset + index] = featureRings[featureRingsOffset + index];
  } else {
    metadata[metadataOffset + index] = ringOffsets[ringOffsetsOffset + index - ${ringOffsetsBase}u];
  }`
          })
        );
        side.packedRingOffsetsBase = ringOffsetsBase;
        bindings = [
          {name: 'qPositions', view: geometry.positions, type: 'f32', access: 'read'},
          {name: 'qFeatureRings', view: metadata, type: 'u32', access: 'read'}
        ];
      } else if (geometry.kind === 'lines' && isMultipartJoinGeometry(geometry)) {
        const lineOffsetsBase = geometry.featureOffsets.length;
        validateGPUStorageBindingByteLength(
          graph.device,
          id,
          'multipart query metadata',
          (lineOffsetsBase + geometry.lineOffsets.length) * 4,
          'reduce query parts or disable querySort'
        );
        const metadata = createTransientView(
          graph,
          `${id}-query-metadata`,
          'uint32',
          lineOffsetsBase + geometry.lineOffsets.length
        );
        nodes.push(
          createStorageTopKNode(graph, {
            id: `${id}-query-metadata`,
            operation,
            variant: 'multipart-query-metadata',
            bindings: [
              {name: 'featureOffsets', view: geometry.featureOffsets, type: 'u32', access: 'read'},
              {name: 'lineOffsets', view: geometry.lineOffsets, type: 'u32', access: 'read'},
              {name: 'metadata', view: metadata, type: 'u32', access: 'read_write'}
            ],
            invocationCount: metadata.length,
            body: `if (index < ${lineOffsetsBase}u) { metadata[metadataOffset + index] = featureOffsets[featureOffsetsOffset + index]; }
  else { metadata[metadataOffset + index] = lineOffsets[lineOffsetsOffset + index - ${lineOffsetsBase}u]; }`
          })
        );
        side.packedRingOffsetsBase = lineOffsetsBase;
        bindings = [
          {name: 'qPositions', view: geometry.positions, type: 'f32', access: 'read'},
          {name: 'qFeatureOffsets', view: metadata, type: 'u32', access: 'read'}
        ];
      }
      const minima = createTransientView(graph, `${id}-query-minima`, 'float32x2', queryCount);
      const maxima = createTransientView(graph, `${id}-query-maxima`, 'float32x2', queryCount);
      const centers = createTransientView(graph, `${id}-query-centers`, 'float32x2', queryCount);
      nodes.push(
        createBoundsNode(
          graph,
          `${id}-query-bounds`,
          operation,
          geometry,
          queryCount,
          minima,
          maxima
        ),
        createStorageTopKNode<Parameters>(graph, {
          id: `${id}-query-centers`,
          operation,
          variant: 'query-centers',
          bindings: [
            {name: 'minima', view: minima, type: 'f32', access: 'read'},
            {name: 'maxima', view: maxima, type: 'f32', access: 'read'},
            {name: 'centers', view: centers, type: 'u32', access: 'read_write'}
          ],
          invocationCount: queryCount,
          body: `let minimum = vec2f(minima[minimaOffset + index * 2u], minima[minimaOffset + index * 2u + 1u]);
  let maximum = vec2f(maxima[maximaOffset + index * 2u], maxima[maximaOffset + index * 2u + 1u]);
  // Halve before adding to avoid overflow for large finite bounds. Empty bounds sort last.
  let center = minimum * 0.5 + maximum * 0.5;
  let valid = all(minimum <= maximum);
  centers[centersOffset + index * 2u] = select(0x7fc00000u, bitcast<u32>(center.x), valid);
  centers[centersOffset + index * 2u + 1u] = select(0x7fc00000u, bitcast<u32>(center.y), valid);`
        })
      );
      const ordered = getHilbertQueryOrderNodes<Parameters>(graph, {
        id: `${id}-order`,
        operation,
        points: centers,
        bvh
      });
      nodes.push(...ordered.nodes);
      order = ordered.order;
    }
    dispatches.push({
      chunkId: id,
      side,
      bindings,
      count: queryCount,
      firstRow: 0,
      kind: geometry.kind,
      order
    });
  }
  const featureSide: NearestSide = {
    prefix: 'r',
    kind: features.kind,
    vertexCount: getSideVertexCount(features),
    partCount: isMultipartJoinGeometry(features)
      ? features.kind === 'points'
        ? features.positions.length
        : features.lineOffsets.length - 1
      : undefined
  };
  const featureBindings = getSideBindings('r', features, featureRings);
  const nodeDataBinding: WGSLKernelBinding = {
    name: 'nodeData',
    view: nodeData,
    type: 'u32',
    access: 'read'
  };
  const topkBinding = (access: 'read' | 'read_write'): WGSLKernelBinding => ({
    name: 'topk',
    view: topk,
    type: 'u32',
    access
  });
  const stackSize = bvh.levelCount + 2;
  if (needsFootPass) {
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'extras',
      scratchQueryCount * capacity * EXTRA_WORDS * 4,
      'reduce the query count or neighborCount, or omit the neighbor foot-point outputs'
    );
  }
  const extras = needsFootPass
    ? createTransientView(
        graph,
        `${id}-extras`,
        'uint32',
        scratchQueryCount * capacity * EXTRA_WORDS
      )
    : undefined;

  // Column writes: IDs, distances, counts, overflow flag. `prologue` defines `query` (scratch row),
  // `slot`, `row` (global query row) and `outputIndex` (flat output slot); `queryGuard` skips the
  // invocations past the end of a chunk. GPU words carry each dispatch range.
  const addColumnNodes = (
    nodeIdPrefix: string,
    queryCountInChunk: number,
    chunk?: {firstRow: number; queryCount: number},
    order?: GraphDataView<'uint32'>
  ) => {
    // Without a scratch chunk the traversal keeps each query's own `topk` row, so the column
    // passes never see the order; with one, scratch rows are order positions.
    const chunkOrder = chunk ? order : undefined;
    const orderBinding: WGSLKernelBinding[] = chunkOrder
      ? [{name: 'queryOrder', view: chunkOrder, type: 'u32', access: 'read'}]
      : [];
    const columnBindings: WGSLKernelBinding[] = [
      topkBinding('read'),
      {name: 'neighborIds', view: props.neighborIds, type: 'u32', access: 'read_write'},
      {name: 'neighborCounts', view: props.neighborCounts, type: 'u32', access: 'read_write'},
      {name: 'overflow', view: props.overflow, type: 'atomic<u32>', access: 'read_write'}
    ];
    if (props.neighborRows) {
      columnBindings.push({
        name: 'neighborRows',
        view: props.neighborRows,
        type: 'u32',
        access: 'read_write'
      });
    }
    if (props.featureIds && !props.writeFeatureRows) {
      columnBindings.push({
        name: 'featureIds',
        view: props.featureIds,
        type: 'u32',
        access: 'read'
      });
    }
    if (props.neighborDistances) {
      columnBindings.push({
        name: 'neighborDistances',
        view: props.neighborDistances,
        type: 'f32',
        access: 'read_write'
      });
    }
    const prologue = chunk
      ? `let query = index / CAPACITY;
  let slot = index % CAPACITY;
  let position = topk[topkOffset + CURSOR_WORD] + query;
  if (position >= CHUNK_QUERY_COUNT) { return; }
  let localQuery = ${chunkOrder ? 'queryOrder[queryOrderOffset + position] - CHUNK_FIRST_ROW' : 'position'};
  let row = CHUNK_FIRST_ROW + localQuery;
  let outputIndex = row * CAPACITY + slot;`
      : `let query = index / CAPACITY;
  let slot = index % CAPACITY;
  let row = query;
  let outputIndex = index;`;
    nodes.push(
      createStorageTopKNode<Parameters>(graph, {
        id: `${nodeIdPrefix}-columns`,
        operation,
        variant: `${greatCircle ? 'columns-great-circle' : 'columns'}${chunkOrder ? '-ordered' : ''}`,
        bindings: [...columnBindings, ...orderBinding],
        invocationCount: queryCountInChunk * capacity,
        declarations: `${NEAREST_DISTANCE_RANK_WGSL}
const CAPACITY: u32 = ${capacity}u;
const STRIDE: u32 = ${stride}u;
const NO_FEATURE: u32 = 0xffffffffu;${greatCircle ? `\nconst SPHERE_RADIUS: f32 = ${sphereRadiusLiteral};` : ''}${
          chunk
            ? `\nconst CURSOR_WORD: u32 = ${cursorWord}u;
`
            : ''
        }`,

        body: `let CHUNK_FIRST_ROW = topk[topkOffset + ${dispatchWord}u];
  let CHUNK_QUERY_COUNT = topk[topkOffset + ${dispatchWord + 1}u];
  ${prologue}
  let header = topkOffset + query * STRIDE;
  let count = topk[header];
  if (slot == 0u) {
    neighborCounts[neighborCountsOffset + row] = count;
    if (topk[header + 1u] != 0u) { atomicOr(&overflow[overflowOffset], 1u); }
  }
  var featureRow = NO_FEATURE;
  var featureId = NO_FEATURE;
  var distance = -1.0;
  if (slot < count) {
    featureRow = topk[header + 3u + slot * 2u];
    featureId = featureRow;
    ${props.featureIds && !props.writeFeatureRows ? 'featureId = featureIds[featureIdsOffset + featureId];' : ''}
    distance = ${greatCircle ? 'bitcast<f32>(topk[header + 4u + slot * 2u]) * SPHERE_RADIUS' : 'decodeNearestDistance(bitcast<f32>(topk[header + 4u + slot * 2u]))'};
  }
  neighborIds[neighborIdsOffset + outputIndex] = featureId;
  ${props.neighborRows ? 'neighborRows[neighborRowsOffset + outputIndex] = featureRow;' : ''}
  ${props.neighborDistances ? 'neighborDistances[neighborDistancesOffset + outputIndex] = distance;' : ''}`
      })
    );
    if (extras) {
      const bindings: WGSLKernelBinding[] = [
        topkBinding('read'),
        {name: 'extras', view: extras, type: 'u32', access: 'read'}
      ];
      if (props.neighborFootPoints) {
        bindings.push({
          name: 'footPoints',
          view: props.neighborFootPoints,
          type: 'u32',
          access: 'read_write'
        });
      }
      if (props.neighborQueryPoints) {
        bindings.push({
          name: 'queryPoints',
          view: props.neighborQueryPoints,
          type: 'u32',
          access: 'read_write'
        });
      }
      if (props.neighborSegmentIndices) {
        bindings.push({
          name: 'segmentIndices',
          view: props.neighborSegmentIndices,
          type: 'u32',
          access: 'read_write'
        });
      }
      nodes.push(
        createStorageTopKNode<Parameters>(graph, {
          id: `${nodeIdPrefix}-foot-columns`,
          operation,
          variant: chunkOrder ? 'foot-columns-ordered' : 'foot-columns',
          bindings: [...bindings, ...orderBinding],
          invocationCount: queryCountInChunk * capacity,
          declarations: `const CAPACITY: u32 = ${capacity}u;
const STRIDE: u32 = ${stride}u;
const NO_SEGMENT: u32 = 0xffffffffu;${
            chunk
              ? `\nconst CURSOR_WORD: u32 = ${cursorWord}u;
`
              : ''
          }`,

          body: `let CHUNK_FIRST_ROW = topk[topkOffset + ${dispatchWord}u];
  let CHUNK_QUERY_COUNT = topk[topkOffset + ${dispatchWord + 1}u];
  ${prologue}
  let valid = slot < topk[topkOffset + query * STRIDE];
  let source = extrasOffset + index * ${EXTRA_WORDS}u;
  ${
    props.neighborFootPoints
      ? `// Unused slots hold a quiet NaN so a missing neighbor is never mistaken for the origin.
  footPoints[footPointsOffset + outputIndex * 2u] = select(0x7fc00000u, extras[source], valid);
  footPoints[footPointsOffset + outputIndex * 2u + 1u] = select(0x7fc00000u, extras[source + 1u], valid);`
      : ''
  }
  ${
    props.neighborQueryPoints
      ? `queryPoints[queryPointsOffset + outputIndex * 2u] = select(0x7fc00000u, extras[source + 3u], valid);
  queryPoints[queryPointsOffset + outputIndex * 2u + 1u] = select(0x7fc00000u, extras[source + 4u], valid);`
      : ''
  }
  ${props.neighborSegmentIndices ? 'segmentIndices[segmentIndicesOffset + outputIndex] = select(NO_SEGMENT, extras[source + 2u], valid);' : ''}`
        })
      );
    }
  };

  // Traversal, foot-point pass and (when chunked) the column writes of one scratch chunk. With
  // `chunk` unset the whole dispatch is one chunk, addressed by global row.
  const addTraversalNodes = (
    dispatch: QueryDispatch,
    nodeIdPrefix: string,
    queryCountInChunk: number,
    chunk?: {firstRow: number; queryCount: number}
  ) => {
    const orderBinding: WGSLKernelBinding[] = dispatch.order
      ? [{name: 'queryOrder', view: dispatch.order, type: 'u32', access: 'read'}]
      : [];
    const traversal = getNearestTraversalWGSL({
      ties: props.ties,
      neighborCount,
      capacity,
      featureCount,
      internalNodeCount: bvh.internalNodeCount,
      stackSize,
      dispatchWord,
      scratchChunk: chunk ? {cursorWord} : undefined,
      queryOrder: Boolean(dispatch.order),
      queryKind: dispatch.kind,
      greatCircle,
      filter: hasFilter
        ? {
            exclusive: Boolean(props.exclusive),
            onAttribute: Boolean(props.onAttribute),
            queryBase: queryFilterBase,
            featureBase: featureFilterBase
          }
        : undefined
    });
    const sideWGSL = `${greatCircle ? NEAREST_GEODESIC_WGSL : NEAREST_COMMON_WGSL}${getNearestSideWGSL(dispatch.side)}${getNearestSideWGSL(featureSide)}${getNearestPairWGSL(dispatch.side, featureSide, greatCircle)}`;

    nodes.push(
      createStorageTopKNode<Parameters>(graph, {
        id: `${nodeIdPrefix}-traverse`,
        operation,
        variant: `traverse-${dispatch.kind}-${features.kind}${greatCircle ? '-great-circle' : ''}${dispatch.order ? '-ordered' : ''}`,
        bindings: [
          ...dispatch.bindings,
          nodeDataBinding,
          ...featureBindings,
          topkBinding('read_write'),
          ...orderBinding
        ],
        invocationCount: queryCountInChunk,
        declarations: `${sideWGSL}${traversal.declarations}`,

        body: traversal.body
      })
    );
    if (extras) {
      nodes.push(
        createStorageTopKNode<Parameters>(graph, {
          id: `${nodeIdPrefix}-foot-points`,
          operation,
          variant: `foot-${dispatch.kind}-${features.kind}${greatCircle ? '-great-circle' : ''}${chunk && dispatch.order ? '-ordered' : ''}`,
          bindings: [
            ...dispatch.bindings,
            ...featureBindings,
            topkBinding('read'),
            {name: 'extras', view: extras, type: 'u32', access: 'read_write'},
            // Unchunked scratch keeps original rows, so only the chunked pass reads the order.
            ...(chunk ? orderBinding : [])
          ],
          invocationCount: queryCountInChunk,
          declarations: `${sideWGSL}
const CAPACITY: u32 = ${capacity}u;
const STRIDE: u32 = ${stride}u;${
            chunk
              ? `\nconst CURSOR_WORD: u32 = ${cursorWord}u;
`
              : ''
          }`,

          body: `let CHUNK_FIRST_ROW = topk[topkOffset + ${dispatchWord}u];
  let CHUNK_QUERY_COUNT = topk[topkOffset + ${dispatchWord + 1}u];
  ${
    chunk
      ? `let position = topk[topkOffset + CURSOR_WORD] + index;
  if (position >= CHUNK_QUERY_COUNT) { return; }
  let localQuery = ${dispatch.order ? 'queryOrder[queryOrderOffset + position] - CHUNK_FIRST_ROW' : 'position'};
  let row = index;`
      : `if (index >= CHUNK_QUERY_COUNT) { return; }
  let localQuery = index;
  let row = CHUNK_FIRST_ROW + localQuery;`
  }
  let header = topkOffset + row * STRIDE;
  let count = topk[header];
  for (var slot = 0u; slot < count; slot++) {
    let result = pairResult(localQuery, topk[header + 3u + slot * 2u]);
    let output = extrasOffset + (row * CAPACITY + slot) * ${EXTRA_WORDS}u;
    extras[output] = bitcast<u32>(result.foot.x);
    extras[output + 1u] = bitcast<u32>(result.foot.y);
    extras[output + 2u] = result.segment;
    extras[output + 3u] = bitcast<u32>(result.queryFoot.x);
    extras[output + 4u] = bitcast<u32>(result.queryFoot.y);
  }`
        })
      );
    }
  };

  nodes.push(
    createStorageTopKNode(graph, {
      id: `${id}-initialize-dispatch`,
      operation,
      variant: 'initialize-dispatch',
      bindings: [topkBinding('read_write')],
      invocationCount: 1,
      body: `topk[topkOffset + ${dispatchWord}u] = 0u;
  topk[topkOffset + ${dispatchWord + 1}u] = 0u;`
    })
  );
  const addDispatchNode = (dispatch: QueryDispatch) => {
    const countView =
      props.queries.kind === 'points'
        ? dispatch.bindings[0].view
        : props.queries.geometry.kind === 'lines'
          ? props.queries.geometry.lineOffsets
          : props.queries.geometry.featureOffsets;
    const components = props.queries.kind === 'points' ? 2 : 1;
    const sentinel = props.queries.kind === 'points' ? 0 : 1;
    nodes.push(
      createStorageTopKNode(graph, {
        id: `${dispatch.chunkId}-dispatch`,
        operation,
        variant: 'dispatch-parameters',
        bindings: [
          topkBinding('read_write'),
          {name: 'queryShape', view: countView, type: 'u32', access: 'read'}
        ],
        invocationCount: 1,
        body: `topk[topkOffset + ${dispatchWord}u] += topk[topkOffset + ${dispatchWord + 1}u];
  topk[topkOffset + ${dispatchWord + 1}u] = (arrayLength(&queryShape) - queryShapeOffset) / ${components}u - ${sentinel}u;`
      })
    );
  };
  if (!isChunked) {
    for (const dispatch of dispatches) {
      addDispatchNode(dispatch);
      addTraversalNodes(dispatch, dispatch.chunkId, dispatch.count);
    }
    if (queryCount > 0) {
      addColumnNodes(id, queryCount);
    }
    return {nodes, bvh};
  }

  // Chunked scratch: every query-side chunk is processed `scratchQueryCount` queries at a time.
  // All chunks of one dispatch run the same kernels; the cursor word selects the queries.
  for (const dispatch of dispatches) {
    addDispatchNode(dispatch);
    const chunkCount = Math.ceil(dispatch.count / scratchQueryCount);
    const chunk = {firstRow: dispatch.firstRow, queryCount: dispatch.count};
    nodes.push(
      createStorageTopKNode<Parameters>(graph, {
        id: `${dispatch.chunkId}-reset-cursor`,
        operation,
        variant: 'reset-cursor',
        bindings: [topkBinding('read_write')],
        invocationCount: 1,
        declarations: `const CURSOR_WORD: u32 = ${cursorWord}u;`,
        body: 'topk[topkOffset + CURSOR_WORD] = 0u;'
      })
    );
    for (let scratchChunk = 0; scratchChunk < chunkCount; scratchChunk++) {
      const queryCountInChunk = Math.min(
        scratchQueryCount,
        dispatch.count - scratchChunk * scratchQueryCount
      );
      const nodeIdPrefix = `${dispatch.chunkId}-scratch-${scratchChunk}`;
      nodes.push(
        createStorageTopKNode<Parameters>(graph, {
          id: `${nodeIdPrefix}-clear`,
          operation,
          variant: greatCircle ? 'clear-chunk-great-circle' : 'clear-chunk',
          bindings: clearBindings.filter(
            binding => binding.name === 'topk' || binding.name === 'maxDistance'
          ),
          invocationCount: queryCountInChunk,
          declarations: `${clearDeclarations}
const CURSOR_WORD: u32 = ${cursorWord}u;
`,
          body: `let CHUNK_QUERY_COUNT = topk[topkOffset + ${dispatchWord + 1}u];
  if (topk[topkOffset + CURSOR_WORD] + index < CHUNK_QUERY_COUNT) {
    let header = topkOffset + index * STRIDE;
    ${clearBoundWGSL}
  }`
        })
      );
      addTraversalNodes(dispatch, nodeIdPrefix, queryCountInChunk, chunk);
      addColumnNodes(nodeIdPrefix, queryCountInChunk, chunk, dispatch.order);
      if (scratchChunk < chunkCount - 1) {
        nodes.push(
          createStorageTopKNode<Parameters>(graph, {
            id: `${nodeIdPrefix}-advance-cursor`,
            operation,
            variant: 'advance-cursor',
            bindings: [topkBinding('read_write')],
            invocationCount: 1,
            declarations: `const CURSOR_WORD: u32 = ${cursorWord}u;
const SCRATCH_QUERY_COUNT: u32 = ${scratchQueryCount}u;`,
            body: 'topk[topkOffset + CURSOR_WORD] = topk[topkOffset + CURSOR_WORD] + SCRATCH_QUERY_COUNT;'
          })
        );
      }
    }
  }
  return {nodes, bvh};
}
