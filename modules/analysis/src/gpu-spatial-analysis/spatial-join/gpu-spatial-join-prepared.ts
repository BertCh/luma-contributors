// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {snapshotJoinGeometry} from './spatial-join-construction-snapshot';
import {
  createSpatialPreparationRetryViews,
  validateSpatialPreparationRetryView
} from './spatial-preparation-retry';
import {
  getGPUSpatialJoinPreparedMemoryPlan,
  resolveSpatialJoinPreparedShape
} from './spatial-join-prepared-planning';
export {
  getGPUSpatialJoinPreparedMemoryPlan,
  GPU_SPATIAL_JOIN_PREPARED_PLANNER,
  GPU_SPATIAL_JOIN_PREPARED_ROUTES
} from './spatial-join-prepared-planning';
export type {
  GPUSpatialJoinPreparedShape,
  GPUSpatialJoinPreparedDeviceLimits
} from './spatial-join-prepared-planning';
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
  validateCapacity,
  validateSinglePackedView,
  validateRequirement,
  validateGraphViewsBelongToGraph
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
  isMultipartJoinGeometry,
  validateSpatialJoinGeometry
} from './spatial-join-geometry';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getPolygonEdgeLayoutLength,
  getPolygonEdgeLayoutNodes,
  getSortedFeatureBVHNodes,
  validateDisjointOutputs,
  type SpatialJoinBVHStorage
} from './spatial-join-passes';
import {getPolygonEdgeSlabNodes, type PolygonEdgeSlabs} from './polygon-edge-slabs';
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
  /** Feature-row partition metadata. The BVH remains whole-domain: each bounds/children binding needs 8 * (2 * leafCapacity - 1) bytes. At 128 MiB that binding fits at most 8,388,608 features with default padded leaves, even with partitions. Use getGPUSpatialJoinPreparedMemoryPlan for all binding checks. */
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
   * Polygon geometry only. Builds the polygon edge-slab index and edge layout used by
   * `GPUPointInPolygonJoin` exact refinement (`edgeSlabs`) on every encoding, against the retained
   * geometry snapshot. All consumers after this handle in the same graph encoding share that
   * preparation. A join given this handle defaults its `edgeSlabs` to the handle's value.
   * Retains edge layout and slab storage in addition to the geometry snapshot and BVH storage.
   * Default false: other join kinds do not read it.
   */
  edgeSlabs?: boolean;
  /**
   * Caller-owned persistent storage. The handle captures these bindings in its own descriptor,
   * retaining each view and buffer identity; it never frees caller-owned buffers.
   * When omitted the handle allocates and imports its own buffers
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
 * Animated left features query bounds of the retained geometry. Feature-ring mapping (polygons),
 * bounds, optional Hilbert sorting, BVH construction and optional polygon edge-layout/slab stages
 * replay against that snapshot on every encoding; geometry refresh alone uses GPU generation gating.
 *
 * It reuses geometry. It never caches results: every join still evaluates every candidate against
 * the current left geometry on every encoding.
 *
 * Usage: add the handle to the graph before the joins that use it, and hand it to each join as
 * `prepared`. The first submitted encoding captures the geometry. Invalidate or rebuildWhen requests
 * a new generation, committed only after the complete build executes; discarded encodings retry
 * automatically. Replaying public builders avoids private dispatch APIs and repeats build work.
 * Resident overhead is one copy of every geometry view plus 1072 generation-control bytes
 * (a fixed 1024-byte lookup table, 32-byte request and 16-byte state).
 * Partition metadata does not tile the index. Each whole-domain bounds/children binding uses
 * 8 * (2 * leafCapacity - 1) bytes: a 128 MiB binding admits at most 8,388,608 features with
 * default leaf padding for this check. Other bindings and device limits still apply.
 * A failed node construction releases acquired buffers and does not publish readiness. The failed
 * graph may retain imported IDs: retry on a fresh graph. Imported persistent input identities
 * reuse the fresh graph's canonical imports; transient inputs require recreating their producers
 * and this handle. Node-registration errors after getCommandNodes returns cannot be observed by
 * the producer: call recoverFailedDeclaration, discard the graph, then retry.
 * Prepared traversal and exact refinement use the same retained positions and offsets.
 * Source coordinate or topology edits become visible after invalidate plus a submitted rebuild;
 * source-ID contents remain live for consumer output mapping. Changing view lengths requires
 * a new handle and graph. getGeometry(graph) exposes captured views only while declared there.
 *
 * The snapshot retains positions and the geometry's line/feature/polygon/ring offsets when present.
 * Construction captures their view identities; their buffer contents are copied only when a
 * requested generation executes. Source-ID contents remain live for consumer output mapping.
 * The handle also retains prepared output storage. Sort and
 * other build scratch is repopulated on every encoding.
 * One handle belongs to one graph. Add it once before its consumers: several joins in that graph
 * encoding share the single preparation. Separate encodings each replay all preparation stages.
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
  private capturedGeometry: GPUSpatialJoinGeometry | undefined;
  private preparedStorage: GPUSpatialJoinPreparedStorage | undefined;
  private ownsPreparedStorage = false;
  private ownedBuffers: Buffer[] = [];
  private failedDeclaration = false;
  private retryViews:
    | {
        graph: GPUCommandGraph<Parameters>;
        geometry?: GPUSpatialJoinGeometry;
        storage?: GPUSpatialJoinPreparedStorage;
      }
    | undefined;
  private readonly rebuildGeneration = new Uint32Array([1, 0]);
  private encodingRequestPending = true;
  private internalNodeCount = 0;
  private slabStorage: {slabs: PolygonEdgeSlabs; layout: GraphDataView<'uint32'>} | undefined;

  constructor(props: GPUSpatialJoinPreparedProps<Parameters>) {
    props = snapshotConstructionFields(props, [
      'id',
      'geometry',
      'partitioning',
      'leafCapacity',
      'spatialSort',
      'edgeSlabs',
      'storage',
      'rebuildWhen'
    ]);
    const partitioning =
      props.partitioning &&
      snapshotConstructionFields(props.partitioning, [
        'length',
        'partitions',
        'seamOwnership',
        'haloDistance'
      ]);
    props = Object.freeze({
      ...props,
      geometry: snapshotJoinGeometry(props.geometry),
      ...(props.storage
        ? {
            storage: snapshotConstructionFields(props.storage, [
              'nodeMinima',
              'nodeMaxima',
              'nodeChildren',
              'leafIds',
              'count',
              'overflow',
              'featureRings'
            ])
          }
        : {}),
      ...(partitioning
        ? {
            partitioning: Object.freeze({
              ...partitioning,
              partitions: Object.freeze(
                partitioning.partitions.map(partition =>
                  snapshotConstructionFields(partition, [
                    'id',
                    'coreStart',
                    'coreEnd',
                    'haloStart',
                    'haloEnd'
                  ])
                )
              ),
              seamOwnership: snapshotConstructionFields(partitioning.seamOwnership, [
                'kind',
                'sourceSide'
              ] as readonly (keyof typeof partitioning.seamOwnership)[])
            })
          }
        : {})
    });
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
    const shape = resolveSpatialJoinPreparedShape(
      {
        kind: this.geometry.kind,
        featureCount: this.featureCount,
        vertexCount: this.geometry.positions.length,
        leafCapacity: props.leafCapacity,
        spatialSort: props.spatialSort
      },
      this.id
    );
    this.leafCapacity = shape.leafCapacity;
    this.spatialSort = shape.spatialSort;
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
   * Persistent BVH storage. Captured caller storage remains accessible before declaration and
   * after destroy. Internally allocated storage is exposed only after node construction succeeds.
   */
  get storage(): GPUSpatialJoinPreparedStorage {
    if (!this.graph && this.props.storage) return this.props.storage;
    if (!this.graph || !this.preparedStorage) {
      throw new Error(`${this.id} storage needs graph.add first`);
    }
    return this.preparedStorage;
  }

  /**
   * Persistent polygon edge-slab index and edge layout built by this handle (`edgeSlabs`), or
   * undefined. Available after the handle has produced nodes.
   */
  get edgeSlabStorage(): {slabs: PolygonEdgeSlabs; layout: GraphDataView<'uint32'>} | undefined {
    return this.graph ? this.slabStorage : undefined;
  }

  /** Number of BVH internal nodes. Available after the handle has produced nodes. */
  get bvhInternalNodeCount(): number {
    return this.graph ? this.internalNodeCount : 0;
  }

  /** Returns whether `getCommandNodes` has declared this handle in `graph`. */
  isDeclaredIn(graph: unknown): boolean {
    return this.graph !== undefined && this.graph === graph;
  }

  /**
   * Geometry retained by this graph's requested generation. Coordinates and offsets refresh only
   * after invalidation; sourceIds remain live caller data. Available after successful declaration
   * and only until destroy or recoverFailedDeclaration releases the snapshot resources.
   */
  getGeometry(graph: unknown): GPUSpatialJoinGeometry {
    if (!this.isDeclaredIn(graph) || !this.capturedGeometry)
      throw new Error(`${this.id} geometry snapshot needs graph.add on the target graph first`);
    return this.capturedGeometry;
  }

  /**
   * Releases a declaration after graph.add failed while registering returned nodes. The producer
   * API observes node construction, not registration: discard that graph and call this method
   * before retrying on a fresh graph. Do not call it on a successfully submitted graph.
   */
  recoverFailedDeclaration(): void {
    this.destroy();
    this.failedDeclaration = true;
  }

  /**
   * Select the actual fresh producer views after a failed declaration, before graph.add(this).
   * Their physical ranges must equal the original persistent views. Explicit mappings avoid a
   * second logical import when DynamicBuffer/core Buffer or usage descriptors differ. Supplied
   * maps are captured; omitted fields retain automatic import behavior. Transients cannot move.
   */
  setRetryViews(
    graph: GPUCommandGraph<Parameters>,
    views: {
      geometry?: GPUSpatialJoinGeometry;
      storage?: GPUSpatialJoinPreparedStorage;
    }
  ): void {
    validateRequirement(
      this.id,
      this.failedDeclaration && !this.graph,
      'setRetryViews requires a failed declaration and a fresh graph before graph.add'
    );
    const geometry = views.geometry && snapshotJoinGeometry(views.geometry);
    const storage = views.storage && Object.freeze({...views.storage});
    if (geometry) {
      validateRequirement(
        this.id,
        geometry.kind === this.geometry.kind,
        `retry geometry.kind must equal ${this.geometry.kind}; received ${geometry.kind}`
      );
      for (const [name, source] of Object.entries(this.geometry)) {
        if (source && typeof source === 'object' && 'buffer' in source) {
          const replacement = Object.entries(geometry).find(([key]) => key === name)?.[1];
          validateRequirement(
            this.id,
            Boolean(replacement && typeof replacement === 'object' && 'buffer' in replacement),
            `retry geometry.${name} must supply the original view`
          );
          validateSpatialPreparationRetryView(
            graph,
            this.id,
            `geometry.${name}`,
            source,
            replacement as GraphDataView
          );
        }
      }
    }
    if (storage) {
      validateRequirement(
        this.id,
        Boolean(this.props.storage),
        'retry storage requires original caller-owned storage'
      );
      for (const [name, source] of Object.entries(this.props.storage ?? {})) {
        if (!source) continue;
        const replacement = storage[name as keyof GPUSpatialJoinPreparedStorage];
        validateRequirement(
          this.id,
          Boolean(replacement),
          `retry storage.${name} must supply the original view`
        );
        validateSpatialPreparationRetryView(
          graph,
          this.id,
          `storage.${name}`,
          source,
          replacement!
        );
      }
    }
    this.retryViews = {graph, geometry, storage};
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
    this.retryViews = undefined;
    this.capturedGeometry = undefined;
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

  /**
   * Returns gated geometry refresh, replayed builds and final generation commit. Readiness means
   * node construction completed; if graph.add subsequently fails to register those nodes, call
   * recoverFailedDeclaration and discard the graph before retrying.
   */
  getCommandNodes(graph: GPUCommandGraph<Parameters>): readonly GPUCommandNode<Parameters>[] {
    if (this.graph) throw new Error(`${this.id} is already added to a graph`);
    const previousStorage = this.preparedStorage;
    const previousInternalNodeCount = this.internalNodeCount;
    try {
      validateRequirement(
        this.id,
        !this.retryViews || this.retryViews.graph === graph,
        'retry views must belong to the target graph selected by setRetryViews'
      );
      if (this.failedDeclaration && this.preparedStorage)
        this.preparedStorage = createSpatialPreparationRetryViews(
          graph,
          `${this.id}-storage`,
          this.preparedStorage,
          this.retryViews?.storage
        );
      const nodes = this.createCommandNodes(graph);
      this.graph = graph;
      this.failedDeclaration = false;
      this.retryViews = undefined;
      return nodes;
    } catch (error) {
      // Imported IDs can remain in the failed graph. Rebuild that graph before retrying.
      this.destroy();
      this.preparedStorage = previousStorage;
      this.internalNodeCount = previousInternalNodeCount;
      this.failedDeclaration = true;
      throw error;
    }
  }

  private createCommandNodes(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, featureCount} = this;
    let geometry = this.failedDeclaration
      ? createSpatialPreparationRetryViews(graph, this.id, this.geometry, this.retryViews?.geometry)
      : this.geometry;
    if (this.graph) {
      throw new Error(`${id} is already added to a graph`);
    }
    // All caller ranges/usage and nested allocation envelopes precede persistent allocation,
    // graph declaration and handle state changes, so a failed preflight is retryable.
    validateGraphViewsBelongToGraph(id, graph, [geometry.sourceIds]);
    const caller = this.preparedStorage;
    const inputViews = Object.fromEntries(
      getSpatialJoinGeometryViews(geometry).map((view, index) => [`geometry[${index}]`, view])
    );
    const plan = getGPUSpatialJoinPreparedMemoryPlan(
      {
        kind: geometry.kind,
        featureCount,
        vertexCount: geometry.positions.length,
        multipart: isMultipartJoinGeometry(geometry),
        partCount: geometry.kind === 'lines' ? geometry.lineOffsets.length - 1 : undefined,
        polygonCount: geometry.kind === 'polygons' ? geometry.polygonOffsets.length - 1 : undefined,
        ringCount: geometry.kind === 'polygons' ? geometry.ringOffsets.length - 1 : undefined,
        leafCapacity: this.leafCapacity,
        spatialSort: this.spatialSort,
        edgeSlabs: this.edgeSlabs,
        externalStorage: Boolean(caller)
      },
      graph.device.limits
    );
    const scratchBytes = plan.scratchByteLengths;
    const slabEntryCapacity = plan.slabEntryCapacity;
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

    const result = [...snapshot.nodes, ...nodes, snapshot.commit(nodes)];
    this.capturedGeometry = Object.freeze(geometry);
    return result;
  }
}
