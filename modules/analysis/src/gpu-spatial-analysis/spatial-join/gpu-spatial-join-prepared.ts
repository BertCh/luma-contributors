// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  GPUBVH,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  importGraphBuffer,
  getStorageBindingElementLimit,
  validateCapacity,
  validateSinglePackedView,
  validatePowerOfTwo,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateGPUPartitionDescriptor,
  type GPUPartitionDescriptor
} from '../../analysis-core/contracts/partitioning';
import {
  createBoundsNode,
  createFeatureRingsNode,
  getSpatialJoinFeatureCount,
  getSpatialJoinGeometryViews,
  validateSpatialJoinGeometry
} from './spatial-join-geometry';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getDefaultSpatialSort,
  getNextPowerOfTwo,
  getPolygonEdgeLayoutLength,
  getPolygonEdgeLayoutNodes,
  getSortedFeatureBVHNodes,
  validateDisjointOutputs,
  type SpatialJoinBVHStorage
} from './spatial-join-passes';
import {
  getPolygonEdgeSlabNodes,
  POLYGON_EDGE_SLAB_PARAMS_STRIDE,
  type PolygonEdgeSlabs
} from './polygon-edge-slabs';
import {
  validateSpatialJoinBindings,
  validateSpatialJoinReadWriteBindings
} from './spatial-join-binding-preflight';
import {validateGraphChunkGroupKernelSupport} from '../../analysis-core/kernels/graph-chunk-groups';
import type {GPUSpatialJoinGeometry} from './spatial-join-types';
import {
  advancePreparedGeometryGeneration,
  createPreparedGeometrySnapshot
} from './prepared-geometry-snapshot';

/** Geometry lengths and every generated row address are uint32. */
const SPATIAL_JOIN_PREPARED_MAXIMUM_ROWS = 0xffffffff;
/** A complete binary tree has 2 * leafCapacity - 1 uint32-addressed nodes. */
const SPATIAL_JOIN_PREPARED_MAXIMUM_LEAF_CAPACITY = 0x80000000;
/** Published radix sort processes 256 keys per workgroup and four key bits per digit. */
const SPATIAL_JOIN_PREPARED_SORT_WORKGROUP_SIZE = 256;
/** Four-bit radix digits have sixteen histogram buckets. */
const SPATIAL_JOIN_PREPARED_RADIX_BUCKET_COUNT = 16;
/** Matches the shared edge-slab builder's default entry budget, including small geometries. */
const SPATIAL_JOIN_PREPARED_SLAB_ENTRIES_PER_VERTEX = 4;
/** The shared slab builder reserves at least this many optional edge entries. */
const SPATIAL_JOIN_PREPARED_MINIMUM_SLAB_ENTRIES = 1024;

/**
 * Persistent output storage of a {@link GPUSpatialJoinPrepared} handle.
 *
 * These views must keep their contents between encodings, so they are imported buffers and never
 * graph transients (the graph may reuse transient memory once a transient's last reader has run).
 */
export type GPUSpatialJoinPreparedStorage = SpatialJoinBVHStorage & {
  /** `(ringStart, ringEnd)` per feature. Present for polygon geometry only. */
  featureRings?: GraphDataView<'uint32x2'>;
};

/** Properties for {@link GPUSpatialJoinPrepared}. */
export type GPUSpatialJoinPreparedProps<Parameters = unknown> = {
  /** Prefix for generated node, buffer and transient IDs. Defaults to `'spatial-join-prepared'`. */
  id?: string;
  /**
   * The right-hand side to index. Joins that use this handle must be given the same geometry
   * views as their `right` (and, for `GPUPointInPolygonJoin`, the same polygon views).
   */
  geometry: GPUSpatialJoinGeometry;
  /** Feature-row partitions retained by the prepared index for tiled and streaming execution. */
  partitioning?: GPUPartitionDescriptor;
  /** Power-of-two BVH leaf slots. Defaults to the next power of two of the feature count. */
  leafCapacity?: number;
  /**
   * Reorder features along a Hilbert curve before the build. Tightens the tree for large,
   * spatially incoherent right-hand sides. Joins that promise sorted output restore candidate
   * source-row order after probing. Defaults on from 256 features.
   */
  spatialSort?: boolean;
  /**
   * Polygon geometry only. Also builds, once per rebuild, the polygon edge-slab index and edge
   * layout that `GPUPointInPolygonJoin` uses for its exact refine (`edgeSlabs`), so repeated joins
   * reuse it like the BVH. A join given this handle defaults its `edgeSlabs` to the handle's
   * value. Costs memory of about `(vertexCount + 4 * vertexCount)` pairs of words in persistent
   * storage. Default false: other join kinds do not read it.
   */
  edgeSlabs?: boolean;
  /**
   * Caller-owned persistent storage. When omitted the handle allocates and imports its own buffers
   * when it first produces nodes, and {@link GPUSpatialJoinPrepared.destroy} frees them.
   */
  storage?: GPUSpatialJoinPreparedStorage;
  /**
   * Optional CPU predicate bypassed by a queued refresh request.
   * The first encoding and first encoding after invalidate() bypass it; other encodings call it
   * once with their parameters. True requests a new geometry generation. Requested generations
   * remain pending on the GPU until their complete preparation executes, including after discard.
   */
  rebuildWhen?: (parameters: Parameters) => boolean;
};

/**
 * A prepared static right-hand side whose geometry snapshot changes only when invalidated.
 * Animated left features query bounds of the retained geometry. Published BVH/sort builders replay
 * against that snapshot on every encoding; geometry refresh alone uses GPU generation gating.
 *
 * It reuses geometry. It never caches results: every join still evaluates every candidate against
 * the current left geometry on every encoding.
 *
 * Usage: add the handle to the graph before the joins that use it, and hand it to each join as
 * `prepared`. The first submitted encoding captures the geometry. Invalidate or rebuildWhen requests
 * a new generation, committed only after the complete build executes; discarded encodings retry
 * automatically. Replaying public builders avoids private dispatch APIs and repeats build work.
 * Resident overhead is one copy of every geometry view plus 24 generation-control bytes.
 * If the right-hand side changes and the index is
 * not invalidated, joins keep querying the stale index (by design: the contract of a static
 * right-hand side). Changing view lengths still requires a new handle and graph.
 *
 * One handle belongs to one graph. Several joins in that graph may share it.
 */
export class GPUSpatialJoinPrepared<Parameters = unknown>
  implements GPUCommandNodeProducer<Parameters>
{
  /** Prefix for every node, buffer and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialJoinPreparedProps<Parameters>;
  /** Indexed geometry. */
  readonly geometry: GPUSpatialJoinGeometry;
  /** Number of indexed features. */
  readonly featureCount: number;
  /** Resolved power-of-two BVH leaf capacity. */
  readonly leafCapacity: number;
  /** Whether features are spatially sorted before the build. */
  readonly spatialSort: boolean;
  /** Whether the handle also builds the polygon edge-slab index (polygon geometry only). */
  readonly edgeSlabs: boolean;
  /** Feature-row partitions associated with this index, when it is built per partition. */
  readonly partitioning?: GPUPartitionDescriptor;
  /** CPU refresh-request encodings, including discarded requests; excludes automatic GPU retries. */
  encodedBuildCount = 0;

  private graph: unknown;
  private preparedStorage: GPUSpatialJoinPreparedStorage | undefined;
  private ownsPreparedStorage = false;
  private ownedBuffers: Buffer[] = [];
  private readonly rebuildGeneration = new Uint32Array([1, 0]);
  private encodingRequestPending = true;
  private internalNodeCount = 0;
  private slabStorage: {slabs: PolygonEdgeSlabs; layout: GraphDataView<'uint32'>} | undefined;

  constructor(props: GPUSpatialJoinPreparedProps<Parameters>) {
    this.id = props.id ?? 'spatial-join-prepared';
    this.props = props;
    this.geometry = props.geometry;
    validateSpatialJoinGeometry(this.id, 'geometry', props.geometry, {allowMultipart: true});
    for (const view of getSpatialJoinGeometryViews(props.geometry))
      validateCapacity(
        this.id,
        'geometry view length',
        view.length,
        0,
        SPATIAL_JOIN_PREPARED_MAXIMUM_ROWS
      );
    this.featureCount = getSpatialJoinFeatureCount(props.geometry);
    this.partitioning = props.partitioning;
    if (this.partitioning) {
      validateGPUPartitionDescriptor(`${this.id} partitioning`, this.partitioning);
      if (this.partitioning.length !== this.featureCount) {
        throw new Error(`${this.id} partitioning needs one row per feature`);
      }
    }
    this.edgeSlabs = props.edgeSlabs ?? false;
    if (this.edgeSlabs && props.geometry.kind !== 'polygons') {
      throw new Error(`${this.id} edgeSlabs needs polygon geometry`);
    }
    this.leafCapacity = props.leafCapacity ?? getNextPowerOfTwo(Math.max(this.featureCount, 1));
    validatePowerOfTwo(
      this.id,
      'leafCapacity',
      this.leafCapacity,
      1,
      SPATIAL_JOIN_PREPARED_MAXIMUM_LEAF_CAPACITY
    );
    this.spatialSort = props.spatialSort ?? getDefaultSpatialSort(this.featureCount);
    this.preparedStorage = props.storage;
    if (props.storage) {
      this.validateStorage(props.storage);
      const {nodeMinima, nodeMaxima, nodeChildren, leafIds, count, overflow} = props.storage;
      const featureRings =
        props.geometry.kind === 'polygons' ? props.storage.featureRings : undefined;
      validateDisjointOutputs(
        this.id,
        getSpatialJoinGeometryViews(props.geometry),
        [nodeMinima, nodeMaxima, nodeChildren, leafIds, count, overflow, featureRings],
        [{nodeMinima, nodeMaxima, nodeChildren, leafIds, count, overflow}]
      );
      // Ask the published builder which strategy this device uses. Fused builds bind every
      // tree field writable; only the level route reads children while updating node bounds.
      const hierarchy = new GPUBVH({
        id: `${this.id}-storage-layout`,
        minima: this.geometry.positions,
        maxima: this.geometry.positions,
        leafCapacity: this.leafCapacity,
        nodeMinima,
        nodeMaxima,
        nodeChildren,
        leafIds,
        count,
        overflow
      });
      if (hierarchy.resolvedStrategy === 'level' && this.leafCapacity > 1)
        validateSpatialJoinReadWriteBindings(this.id, [nodeChildren], [nodeMinima, nodeMaxima]);
      if (props.geometry.kind === 'polygons')
        validateSpatialJoinReadWriteBindings(
          this.id,
          [props.geometry.featureOffsets, props.geometry.polygonOffsets],
          [featureRings]
        );
    }
  }

  /**
   * Persistent BVH storage. Throws until the handle has produced nodes for a graph, because
   * self-allocated storage is imported into that graph.
   */
  get storage(): GPUSpatialJoinPreparedStorage {
    if (!this.preparedStorage) {
      throw new Error(`${this.id} storage needs graph.add first`);
    }
    return this.preparedStorage;
  }

  /**
   * Persistent polygon edge-slab index and edge layout built by this handle (`edgeSlabs`), or
   * undefined. Available after the handle has produced nodes.
   */
  get edgeSlabStorage(): {slabs: PolygonEdgeSlabs; layout: GraphDataView<'uint32'>} | undefined {
    return this.slabStorage;
  }

  /** Number of BVH internal nodes. Available after the handle has produced nodes. */
  get bvhInternalNodeCount(): number {
    return this.internalNodeCount;
  }

  /** Returns whether `getCommandNodes` has declared this handle in `graph`. */
  isDeclaredIn(graph: unknown): boolean {
    return this.graph !== undefined && this.graph === graph;
  }

  /** Forces the next encoding to rebuild the index. Call it after the right-hand side changed. */
  invalidate(): void {
    advancePreparedGeometryGeneration(this.rebuildGeneration);
    this.encodingRequestPending = true;
  }

  /** Destroys buffers the handle allocated itself. Destroy compiled graphs that use it first. */
  destroy(): void {
    for (const buffer of this.ownedBuffers) {
      buffer.destroy();
    }
    if (this.ownsPreparedStorage) {
      // The storage views pointed at the buffers just destroyed.
      this.preparedStorage = undefined;
    }
    this.ownsPreparedStorage = false;
    this.slabStorage = undefined;
    this.ownedBuffers = [];
    this.graph = undefined;
    this.rebuildGeneration.set([1, 0]);
    this.encodingRequestPending = true;
  }

  private validateStorage(storage: GPUSpatialJoinPreparedStorage): void {
    const nodeCount = 2 * this.leafCapacity - 1;
    const expectations: [
      string,
      GraphDataView | undefined,
      number,
      'float32x2' | 'uint32x2' | 'uint32'
    ][] = [
      ['nodeMinima', storage.nodeMinima, nodeCount, 'float32x2'],
      ['nodeMaxima', storage.nodeMaxima, nodeCount, 'float32x2'],
      ['nodeChildren', storage.nodeChildren, nodeCount, 'uint32x2'],
      ['leafIds', storage.leafIds, this.leafCapacity, 'uint32'],
      ['count', storage.count, 1, 'uint32'],
      ['overflow', storage.overflow, 1, 'uint32']
    ];
    if (this.geometry.kind === 'polygons') {
      expectations.push(['featureRings', storage.featureRings, this.featureCount, 'uint32x2']);
    }
    for (const [name, view, length, format] of expectations) {
      if (!view || view.length < length) {
        throw new Error(`${this.id} storage.${name} needs ${length} rows`);
      }
      validateSinglePackedView(view, [format], `${this.id} storage.${name}`);
      validateCapacity(
        this.id,
        `storage.${name}.length`,
        view.length,
        length,
        SPATIAL_JOIN_PREPARED_MAXIMUM_ROWS
      );
      if (name !== 'featureRings' && name !== 'count' && name !== 'overflow')
        validateRequirement(
          this.id,
          view.length === length,
          `storage.${name}.length must equal ${length} rows for the published BVH layout`
        );
    }
  }

  private allocateStorage<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): GPUSpatialJoinPreparedStorage {
    const {device} = graph;
    const nodeCount = 2 * this.leafCapacity - 1;
    const create = (
      name: string,
      format: 'float32x2' | 'uint32x2' | 'uint32',
      length: number
    ): GraphDataView<typeof format> => {
      const rowBytes = format === 'uint32' ? 4 : 8;
      const buffer = device.createBuffer({
        id: `${this.id}-${name}`,
        byteLength: Math.max(length, 1) * rowBytes,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      this.ownedBuffers.push(buffer);
      return importGraphBuffer(graph, `${this.id}-${name}`, buffer, format, length);
    };
    return {
      nodeMinima: create('node-minima', 'float32x2', nodeCount) as GraphDataView<'float32x2'>,
      nodeMaxima: create('node-maxima', 'float32x2', nodeCount) as GraphDataView<'float32x2'>,
      nodeChildren: create('node-children', 'uint32x2', nodeCount) as GraphDataView<'uint32x2'>,
      leafIds: create('leaf-ids', 'uint32', this.leafCapacity) as GraphDataView<'uint32'>,
      count: create('count', 'uint32', 1) as GraphDataView<'uint32'>,
      overflow: create('overflow', 'uint32', 1) as GraphDataView<'uint32'>,
      featureRings:
        this.geometry.kind === 'polygons'
          ? (create('feature-rings', 'uint32x2', this.featureCount) as GraphDataView<'uint32x2'>)
          : undefined
    };
  }

  /**
   * Builds the edge-slab index and layout into transients, then copies the slab views into
   * persistent buffers (graph transients may be aliased after their last use). The layout is
   * written to persistent storage directly.
   */
  private getEdgeSlabNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    geometry: Extract<GPUSpatialJoinGeometry, {kind: 'polygons'}>,
    entryCapacity: number
  ): GPUCommandNode<Parameters>[] {
    const {id} = this;
    const create = <Format extends 'uint32' | 'uint32x2'>(
      name: string,
      format: Format,
      length: number
    ): GraphDataView<Format> => {
      const buffer = graph.device.createBuffer({
        id: `${id}-${name}`,
        byteLength: Math.max(length, 1) * (format === 'uint32' ? 4 : 8),
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      this.ownedBuffers.push(buffer);
      return importGraphBuffer(graph, `${id}-${name}`, buffer, format, length);
    };
    const built = getPolygonEdgeSlabNodes<Parameters>(graph, {
      id: `${id}-edges`,
      operation: 'GPUSpatialJoinPrepared',
      owner: 'polygon',
      entryCapacity: entryCapacity || undefined,
      positions: geometry.positions,
      polygonOffsets: geometry.polygonOffsets,
      ringOffsets: geometry.ringOffsets
    });
    const {slabs} = built;
    const persistent: PolygonEdgeSlabs = {
      ...slabs,
      params: create('edge-slab-params', 'uint32', slabs.params.length),
      slabOffsets: create('edge-slab-offsets', 'uint32', slabs.slabOffsets.length),
      slabEntries: create('edge-slab-entries', 'uint32x2', slabs.slabEntries.length)
    };
    const layoutNodes = getPolygonEdgeLayoutNodes<Parameters>(graph, {
      id: `${id}-edges`,
      operation: 'GPUSpatialJoinPrepared',
      polygons: geometry,
      layout: create(
        'edge-layout',
        'uint32',
        getPolygonEdgeLayoutLength(geometry.positions.length, this.featureCount)
      )
    });
    this.slabStorage = {slabs: persistent, layout: layoutNodes.layout};
    const copy = (
      name: string,
      source: GraphDataView<'uint32' | 'uint32x2'>,
      destination: GraphDataView<'uint32' | 'uint32x2'>,
      wordCount: number
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-edges-copy-${name}`,
        operation: 'GPUSpatialJoinPrepared',
        variant: 'copy-words',
        bindings: [
          {name: 'source', view: source, type: 'u32', access: 'read'},
          {name: 'destination', view: destination, type: 'u32', access: 'read_write'}
        ],
        invocationCount: Math.max(wordCount, 1),
        declarations: `const WORD_COUNT: u32 = ${wordCount}u;`,
        body: `if (index >= WORD_COUNT) { return; }
  destination[destinationOffset + index] = source[sourceOffset + index];`
      });
    return [
      ...built.nodes,
      ...layoutNodes.nodes,
      copy('params', slabs.params, persistent.params, slabs.params.length),
      copy('offsets', slabs.slabOffsets, persistent.slabOffsets, slabs.slabOffsets.length),
      copy('entries', slabs.slabEntries, persistent.slabEntries, 2 * slabs.slabEntries.length)
    ];
  }

  /** Mirrors the shared builder's binding-sized budget, before its transient declarations. */
  private getEdgeSlabEntryCapacity<Parameters>(graph: GPUCommandGraph<Parameters>): number {
    const vertexCount = this.geometry.positions.length;
    return Math.min(
      Math.max(0, getStorageBindingElementLimit(graph.device, 8, vertexCount)),
      Math.max(
        SPATIAL_JOIN_PREPARED_SLAB_ENTRIES_PER_VERTEX * vertexCount,
        SPATIAL_JOIN_PREPARED_MINIMUM_SLAB_ENTRIES
      )
    );
  }

  /**
   * Returns gated geometry refresh, replayed public builds, and a final GPU generation commit.
   */
  getCommandNodes(graph: GPUCommandGraph<Parameters>): readonly GPUCommandNode<Parameters>[] {
    const {id, featureCount} = this;
    let geometry = this.geometry;
    if (this.graph) {
      throw new Error(`${id} is already added to a graph`);
    }
    // All caller ranges/usage and nested allocation envelopes precede persistent allocation,
    // graph declaration and handle state changes, so a failed preflight is retryable.
    const caller = this.preparedStorage;
    const inputViews = Object.fromEntries(
      getSpatialJoinGeometryViews(geometry).map((view, index) => [`geometry[${index}]`, view])
    );
    const treeNodeCount = 2 * this.leafCapacity - 1;
    const sort = this.spatialSort && featureCount > 1;
    const scratchBytes: Record<string, number> = {
      'BVH node bounds/children': caller ? 0 : treeNodeCount * 8,
      'BVH leaf IDs': caller ? 0 : this.leafCapacity * 4,
      'BVH status': 4,
      'feature bounds and sorted bounds': featureCount * 8,
      'feature radix histogram': sort
        ? SPATIAL_JOIN_PREPARED_RADIX_BUCKET_COUNT *
          Math.ceil(featureCount / SPATIAL_JOIN_PREPARED_SORT_WORKGROUP_SIZE) *
          4
        : 0,
      'feature sort tile/scene bounds': sort
        ? Math.max(1, Math.ceil(featureCount / SPATIAL_JOIN_PREPARED_SORT_WORKGROUP_SIZE)) * 16
        : 0,
      'feature ring ranges': geometry.kind === 'polygons' && !caller ? featureCount * 8 : 0
    };
    const slabEntryCapacity = this.edgeSlabs ? this.getEdgeSlabEntryCapacity(graph) : 0;
    if (this.edgeSlabs && geometry.kind === 'polygons') {
      const vertexCount = geometry.positions.length;
      const ownerCount = geometry.polygonOffsets.length - 1;
      Object.assign(scratchBytes, {
        'edge slab parameters': Math.max(ownerCount, 1) * POLYGON_EDGE_SLAB_PARAMS_STRIDE * 4,
        'edge slab owner scans': (ownerCount + 1) * 4,
        'edge slab ring owners': Math.max(geometry.ringOffsets.length - 1, 1) * 4,
        'edge slab vertex owners': vertexCount * 4,
        'edge slab counters/offsets': (Math.floor(vertexCount / 4) + 2) * 4,
        'edge slab entries': (vertexCount + slabEntryCapacity) * 8,
        'edge slab layout': getPolygonEdgeLayoutLength(vertexCount, featureCount) * 4
      });
    }
    validateSpatialJoinBindings(
      graph,
      id,
      {
        ...inputViews,
        ...(caller
          ? {
              'storage.nodeMinima': caller.nodeMinima,
              'storage.nodeMaxima': caller.nodeMaxima,
              'storage.nodeChildren': caller.nodeChildren,
              'storage.leafIds': caller.leafIds,
              'storage.count': caller.count,
              'storage.overflow': caller.overflow,
              'storage.featureRings': geometry.kind === 'polygons' ? caller.featureRings : undefined
            }
          : {})
      },
      scratchBytes
    );
    validateGraphChunkGroupKernelSupport(graph, id, 8);
    const snapshotViews: Record<string, GraphDataView> = {positions: geometry.positions};
    if ('featureOffsets' in geometry && geometry.featureOffsets)
      snapshotViews['featureOffsets'] = geometry.featureOffsets;
    if (geometry.kind === 'lines') snapshotViews['lineOffsets'] = geometry.lineOffsets;
    if (geometry.kind === 'polygons') {
      snapshotViews['featureOffsets'] = geometry.featureOffsets;
      snapshotViews['polygonOffsets'] = geometry.polygonOffsets;
      snapshotViews['ringOffsets'] = geometry.ringOffsets;
    }
    const snapshot = createPreparedGeometrySnapshot(graph, {
      id,
      views: snapshotViews,
      generation: this.rebuildGeneration,
      ownedBuffers: this.ownedBuffers,
      recordEncoding: parameters => {
        if (this.encodingRequestPending) this.encodedBuildCount++;
        else if (this.props.rebuildWhen?.(parameters)) {
          advancePreparedGeometryGeneration(this.rebuildGeneration);
          this.encodedBuildCount++;
        }
        this.encodingRequestPending = false;
      }
    });
    geometry = {...geometry, ...snapshot.views} as GPUSpatialJoinGeometry;
    this.graph = graph;
    if (!this.preparedStorage) {
      this.preparedStorage = this.allocateStorage(graph);
      this.ownsPreparedStorage = true;
    }
    const storage = this.preparedStorage;

    const minima = createTransientView(graph, `${id}-minima`, 'float32x2', featureCount);
    const maxima = createTransientView(graph, `${id}-maxima`, 'float32x2', featureCount);
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (geometry.kind === 'polygons') {
      nodes.push(
        createFeatureRingsNode<Parameters>(graph, `${id}-rings`, {
          featureCount,
          geometry,
          featureRings: storage.featureRings as GraphDataView<'uint32x2'>
        })
      );
    }
    nodes.push(
      createBoundsNode<Parameters>(graph, `${id}-bounds`, {
        featureCount,
        geometry,
        featureRings: storage.featureRings,
        minima,
        maxima
      })
    );
    const {bvh, nodes: bvhNodes} = getSortedFeatureBVHNodes(
      graph,
      id,
      'GPUSpatialJoinPrepared',
      minima,
      maxima,
      this.leafCapacity,
      this.spatialSort,
      storage
    );
    this.internalNodeCount = bvh.internalNodeCount;
    nodes.push(...bvhNodes);

    if (this.edgeSlabs && geometry.kind === 'polygons') {
      nodes.push(...this.getEdgeSlabNodes(graph, geometry, slabEntryCapacity));
    }

    return [...snapshot.nodes, ...nodes, snapshot.commit(nodes)];
  }
}
