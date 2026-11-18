// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  GPU_STUDY_WINDOW_FLAG_EMPTY,
  GPU_STUDY_WINDOW_FLAG_ORIENTATION_UNCERTAIN,
  GPU_STUDY_WINDOW_SUMMARY_LENGTH,
  resolveGPUSpatialContext,
  type GPUPolygonGeometryFields,
  type GPUPolygonStudyWindow,
  type GPUPolygonStudyWindowViews,
  type GPUSpatialContext
} from '../../analysis-core/contracts/index';
import {
  GPU_POLYGON_RASTERIZATION_NO_ZONE,
  GPUPolygonRasterization
} from '../../raster-analysis/polygon-rasterization/index';
import {getOffsetRangeSearchSource} from '../../raster-analysis/polygon-rasterization/polygon-rasterization-parameters';
import {
  importGraphBuffer,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateCapacity,
  validateGPUStorageBindingByteLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {getPolygonEdgeSlabNodes} from '../../analysis-core/kernels/polygon-edge-slabs';
import {
  getStudyWindowPackedLayout,
  getStudyWindowSampleCellWGSL,
  STUDY_WINDOW_BOUNDARY_CELL_BIT
} from '../../analysis-core/kernels/study-window-wgsl';
import {ADDITION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/addition-rounding-error-wgsl';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {EXACT_GEOMETRY_WGSL} from '../../analysis-core/kernels/exact-geometry-wgsl';
import {EXACT_ORIENTATION_WGSL} from '../../analysis-core/kernels/exact-orientation-wgsl';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  advancePreparedGeometryGeneration,
  createPreparedGeometrySnapshot
} from '../spatial-join/prepared-geometry-snapshot';

const OPERATION = 'GPUStudyWindowPreparation';
// Ring records: role-signed area, perimeter, edge count, flags, bbox, reversal sign, ownership.
const RING_STRIDE = 10;
/** Bounds the resident sampling raster and its compaction planes. */
const STUDY_WINDOW_MAXIMUM_SAMPLING_CELLS = 1048576;
/** Raster crossings and slab capacities are emitted as uint32 shader indices. */
const STUDY_WINDOW_MAXIMUM_CAPACITY = 0xfffffffe;
/** Default and slot sizing of the shared polygon edge-slab layout, mirrored for preflight. */
const STUDY_WINDOW_SLAB_MINIMUM_ENTRIES = 1024;
const STUDY_WINDOW_SLAB_ENTRIES_PER_VERTEX = 4;
const STUDY_WINDOW_SLAB_VERTICES_PER_SLOT = 4;
/** Published radix-sort scratch has sixteen buckets per 256-row workgroup. */
const STUDY_WINDOW_SORT_WORKGROUP_SIZE = 256;
const STUDY_WINDOW_SORT_BUCKET_COUNT = 16;

/** Compile-time geometry layout and sampler topology of {@link GPUStudyWindowPreparation}. */
export type GPUStudyWindowPreparationProps<Parameters = unknown> = GPUPolygonGeometryFields & {
  /** Prefix of generated nodes and views. Defaults to `'study-window'`. */
  id?: string;
  /** Planar frame shared with consumers. Omitted means native planar coordinates. */
  spatialContext?: GPUSpatialContext;
  /** Columns and rows; defaults to `[256, 256]`, with at most 1048576 cells. */
  samplingGrid?: readonly [number, number];
  /**
   * Raster crossing capacity; defaults to `4 * samplingGrid[1] * ringCount`. On overflow,
   * a slower exact GPU center test preserves sampler coverage and boundary flags.
   */
  crossingCapacity?: number;
  /** Optional edge-slab entry capacity; a full index falls back to walking edges. */
  slabEntryCapacity?: number;
  /**
   * Optional CPU predicate bypassed by a queued refresh request.
   * The first encoding and first encoding after invalidate() bypass it; other encodings call it
   * once with their parameters. True requests a new geometry generation. Requested generations
   * remain pending on the GPU until their complete preparation executes, including after discard.
   */
  rebuildWhen?: (parameters: Parameters) => boolean;
  /** Optional caller-owned summary copy for readback, containing eight float32 values. */
  output?: {summary?: GraphDataView<'float32'>};
};

/**
 * Prepares one polygon or multipolygon study window: oriented edges with the interior on the
 * left, area, perimeter, finite-vertex bounds, edge slabs and a compact row-major sampling grid.
 * Consumers accept this handle as `studyWindow`; add it to their graph first.
 *
 * The window must be a VALID polygon (consider `GPUGeometryValidity`): self-intersections and
 * overlapping polygons break the fan identity used by edge corrections. Shell/hole role follows
 * GeoArrow ring position, irrespective of winding. Only planar frames are supported; project
 * longitude/latitude first with `GPULocalFrame.projectArray`.
 *
 * Ring determinants use coordinates relative to the first vertex and a fixed pairwise tree,
 * without float atomics. Flag 1 (`GPU_STUDY_WINDOW_FLAG_ORIENTATION_UNCERTAIN`) marks a ring whose
 * doubled area is within `8 * epsilon_f32 * (1 + ceil(log2(vertexCount))) * sum(abs(products))`
 * of zero, or is non-finite. The conservative bound includes determinant cancellation and tree
 * rounding. Flag 2 (`GPU_STUDY_WINDOW_FLAG_EMPTY`) marks non-finite or non-positive total area.
 *
 * Rings with fewer than three vertices or exactly zero orientation for every first-vertex fan
 * triangle contribute nothing: no area, perimeter, edges, bounds or uncertainty flag. Empty ring
 * and polygon slots from `GPURectangleClip` are accepted directly. An all-degenerate window is
 * legal and sets EMPTY at encode time. The raster sampler uses raw positions: degenerate rings
 * can only add boundary flags (extra exact tests), never change coverage.
 *
 * The first submitted encoding captures a persistent geometry snapshot. Invalidate or rebuildWhen
 * requests a new generation, committed only after the complete preparation executes. Discarded
 * encodings retry automatically. Public preparation stages replay against the retained geometry
 * every encoding; geometry refresh alone is GPU-gated, avoiding private dispatch APIs. This repeats
 * preparation work and adds one copy of each geometry view plus 24 generation-control bytes.
 * It reuses geometry, it never caches results: consumers still evaluate their current inputs on
 * every encoding. Changed geometry needs `invalidate()` or `rebuildWhen`; otherwise consumers
 * deliberately keep using the previous build. Changing view lengths requires a new handle and
 * graph. One handle belongs to one graph, whose consumers may share it.
 */
export class GPUStudyWindowPreparation<Parameters = unknown>
  implements GPUCommandNodeProducer<Parameters>, GPUPolygonStudyWindow
{
  /** Discriminant used by study-window consumers. */
  readonly kind = 'polygon' as const;
  /** Prefix of generated node and view IDs. */
  readonly id: string;
  /** Resolved planar frame. */
  readonly spatialContext: GPUSpatialContext;
  /** Number of source vertex rows, including any rows outside rings. */
  readonly vertexCount: number;
  /** Validated compile-time properties. */
  readonly props: GPUStudyWindowPreparationProps<Parameters>;
  /** Sampler columns and rows. */
  readonly sampleGrid: readonly [number, number];
  /** CPU refresh-request encodings, including discarded requests; excludes automatic GPU retries. */
  encodedBuildCount = 0;

  private graph: unknown;
  private views: GPUPolygonStudyWindowViews | undefined;
  private ownedBuffers: Buffer[] = [];
  private readonly rebuildGeneration = new Uint32Array([1, 0]);
  private encodingRequestPending = true;

  constructor(props: GPUStudyWindowPreparationProps<Parameters>) {
    this.id = props.id ?? 'study-window';
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext,
      GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT
    );
    this.props = props;
    this.vertexCount = props.positions.length;
    this.sampleGrid = props.samplingGrid ?? [256, 256];
    const {id, sampleGrid} = this;
    validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);
    for (const name of ['featureOffsets', 'polygonOffsets', 'ringOffsets'] as const) {
      validateSinglePackedView(props[name], ['uint32'], `${id} ${name}`);
    }
    if (props.featureOffsets.length !== 2) {
      throw new Error(`${id} study window needs exactly one feature (featureOffsets of length 2)`);
    }
    if (this.vertexCount < 3 || props.ringOffsets.length < 2 || props.polygonOffsets.length < 2) {
      throw new Error(`${id} study window needs at least one ring of 3 or more vertices`);
    }
    if (
      sampleGrid.length !== 2 ||
      !sampleGrid.every(value => Number.isInteger(value) && value > 0) ||
      sampleGrid[0] * sampleGrid[1] > STUDY_WINDOW_MAXIMUM_SAMPLING_CELLS
    ) {
      throw new Error(
        `${id} samplingGrid must be two positive integers with a product of at most 1048576`
      );
    }
    validateCapacity(
      id,
      'crossingCapacity',
      props.crossingCapacity ?? 4 * sampleGrid[1] * (props.ringOffsets.length - 1),
      2,
      STUDY_WINDOW_MAXIMUM_CAPACITY
    );
    if (props.slabEntryCapacity !== undefined)
      validateCapacity(
        id,
        'slabEntryCapacity',
        props.slabEntryCapacity,
        1,
        STUDY_WINDOW_MAXIMUM_CAPACITY
      );
    if (props.output?.summary) {
      validateSinglePackedView(props.output.summary, ['float32'], `${id} output.summary`);
      if (props.output.summary.length !== GPU_STUDY_WINDOW_SUMMARY_LENGTH) {
        throw new Error(`${id} output.summary must contain 8 float32 values`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output?.summary],
      [props.positions, props.featureOffsets, props.polygonOffsets, props.ringOffsets]
    );
  }

  /** Returns the views registered by adding this producer to the same graph before its consumers. */
  getViews<Parameters>(graph: GPUCommandGraph<Parameters>): GPUPolygonStudyWindowViews {
    const views = this.views;
    if (!this.isDeclaredIn(graph) || !views) {
      throw new Error(`${this.id} study window needs graph.add before its consumers`);
    }
    return views;
  }

  /** Returns whether `getCommandNodes` has declared this handle in `graph`. */
  isDeclaredIn(graph: unknown): boolean {
    return this.graph !== undefined && this.graph === graph;
  }

  /** Forces the next encoding to rebuild the window. Call it after the geometry changed. */
  invalidate(): void {
    advancePreparedGeometryGeneration(this.rebuildGeneration);
    this.encodingRequestPending = true;
  }

  /** Destroys buffers the handle allocated itself. Destroy compiled graphs that use it first. */
  destroy(): void {
    for (const buffer of this.ownedBuffers) {
      buffer.destroy();
    }
    this.views = undefined;
    this.ownedBuffers = [];
    this.graph = undefined;
    this.rebuildGeneration.set([1, 0]);
    this.encodingRequestPending = true;
  }

  private createPersistentView<Parameters, Format extends 'float32' | 'uint32' | 'uint32x2'>(
    graph: GPUCommandGraph<Parameters>,
    name: string,
    format: Format,
    length: number
  ): GraphDataView<Format> {
    const buffer = graph.device.createBuffer({
      id: `${this.id}-${name}`,
      byteLength: Math.max(length, 1) * (format === 'uint32x2' ? 8 : 4),
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    this.ownedBuffers.push(buffer);
    return importGraphBuffer(graph, `${this.id}-${name}`, buffer, format, length);
  }

  /** Returns gated geometry refresh, replayed preparation stages, and final GPU generation commit. */
  getCommandNodes(graph: GPUCommandGraph<Parameters>): readonly GPUCommandNode<Parameters>[] {
    const {id, vertexCount, sampleGrid} = this;
    let props = this.props;
    if (this.graph) {
      throw new Error(`${id} is already added to a graph`);
    }
    const ringCount = props.ringOffsets.length - 1;
    const cellCount = sampleGrid[0] * sampleGrid[1];
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.featureOffsets,
      props.polygonOffsets,
      props.ringOffsets,
      props.output?.summary
    ]);
    validateGraphChunkBindingRanges(graph, id, {
      positions: props.positions,
      featureOffsets: props.featureOffsets,
      polygonOffsets: props.polygonOffsets,
      ringOffsets: props.ringOffsets,
      summary: props.output?.summary
    });
    // The shared slab builder clamps entries to one uint32x2 binding. Packed study-window
    // consumers additionally bind edges, slabs and sampler cells together, so check that layout too.
    const slabCapacity = Math.floor(vertexCount / STUDY_WINDOW_SLAB_VERTICES_PER_SLOT) + 1;
    const slabEntryCapacity = Math.min(
      Math.max(0, Math.floor(graph.device.limits.maxStorageBufferBindingSize / 8) - vertexCount),
      props.slabEntryCapacity ??
        Math.max(
          STUDY_WINDOW_SLAB_ENTRIES_PER_VERTEX * vertexCount,
          STUDY_WINDOW_SLAB_MINIMUM_ENTRIES
        )
    );
    const preparedLayout = getStudyWindowPackedLayout({
      vertexCount,
      slabOffsetCount: slabCapacity + 1,
      slabEntryCount: vertexCount + slabEntryCapacity,
      sampleCellCapacity: cellCount
    });
    const crossingCapacity = props.crossingCapacity ?? 4 * sampleGrid[1] * ringCount;
    for (const [name, byteLength] of [
      ['ring terms', 9 * vertexCount * 4],
      ['edges', 4 * vertexCount * 4],
      ['rings', RING_STRIDE * ringCount * 4],
      ['sample scan', (cellCount + 1) * 4],
      ['slab entries', (vertexCount + slabEntryCapacity) * 8],
      ['slab offsets', (slabCapacity + 1) * 4],
      ['slab parameters', 24],
      ['packed window', preparedLayout.length * 4],
      ['raster crossings', crossingCapacity * 4],
      [
        'raster sort histogram',
        Math.ceil(crossingCapacity / STUDY_WINDOW_SORT_WORKGROUP_SIZE) *
          STUDY_WINDOW_SORT_BUCKET_COUNT *
          4
      ]
    ] as const)
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        byteLength,
        'reduce vertices, samplingGrid or capacities'
      );
    const snapshot = createPreparedGeometrySnapshot(graph, {
      id,
      views: {
        positions: props.positions,
        featureOffsets: props.featureOffsets,
        polygonOffsets: props.polygonOffsets,
        ringOffsets: props.ringOffsets
      },
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
    props = {
      ...snapshotConstructionFields(props, [
        'id',
        'spatialContext',
        'positions',
        'featureOffsets',
        'polygonOffsets',
        'ringOffsets',
        'samplingGrid',
        'crossingCapacity',
        'slabEntryCapacity',
        'rebuildWhen',
        'output'
      ]),
      ...snapshot.views
    };
    this.graph = graph;
    const edges = this.createPersistentView(graph, 'edges', 'float32', 4 * vertexCount);
    const summary = this.createPersistentView(
      graph,
      'summary',
      'float32',
      GPU_STUDY_WINDOW_SUMMARY_LENGTH
    );
    const terms = createTransientView(graph, `${id}-terms`, 'float32', 9 * vertexCount);
    const rings = createTransientView(graph, `${id}-rings`, 'float32', RING_STRIDE * ringCount);
    const sampleCells = this.createPersistentView(graph, 'sample-cells', 'uint32', cellCount);
    const sampleCellCount = this.createPersistentView(graph, 'sample-cell-count', 'uint32', 1);
    const positionsBinding: WGSLKernelBinding = {
      name: 'positions',
      view: props.positions,
      type: 'f32',
      access: 'read'
    };
    const ringOffsetsBinding: WGSLKernelBinding = {
      name: 'ringOffsets',
      view: props.ringOffsets,
      type: 'u32',
      access: 'read'
    };
    const ringsBinding: WGSLKernelBinding = {
      name: 'rings',
      view: rings,
      type: 'f32',
      access: 'read'
    };
    const summaryBinding: WGSLKernelBinding = {
      name: 'summary',
      view: summary,
      type: 'f32',
      access: 'read'
    };
    const ringSearch = getOffsetRangeSearchSource('findRing', 'ringOffsets', ringCount);
    const vertexReader = `fn readVertex(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * vertex], positions[positionsOffset + 2u * vertex + 1u]);
}`;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode(graph, {
        id: `${id}-ring-terms`,
        operation: OPERATION,
        variant: 'ring-terms',
        bindings: [
          positionsBinding,
          ringOffsetsBinding,
          {name: 'terms', view: terms, type: 'f32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        declarations: `${ringSearch}${vertexReader}${EXACT_ORIENTATION_WGSL}${GET_NAN_WGSL}`,
        body: `let base = termsOffset + 9u * index;
  for (var slot = 0u; slot < 4u; slot++) { terms[base + slot] = 0.0; }
  terms[base + 4u] = 3.402823466e38;
  terms[base + 5u] = 3.402823466e38;
  terms[base + 6u] = -3.402823466e38;
  terms[base + 7u] = -3.402823466e38;
  terms[base + 8u] = 0.0;
  let ring = findRing(index);
  if (ring == 0xffffffffu) { return; }
  let first = ringOffsets[ringOffsetsOffset + ring];
  let end = ringOffsets[ringOffsetsOffset + ring + 1u];
  let next = select(index + 1u, first, index + 1u == end);
  let startPosition = readVertex(index);
  let endPosition = readVertex(next);
  let origin = readVertex(first);
  // One exact predicate call site; uncertified orientations must not discard a ring.
  terms[base + 8u] = select(0.0, 1.0, orientSign(origin, startPosition, endPosition) != 0);
  let relativeStart = startPosition - origin;
  let relativeEnd = endPosition - origin;
  let forward = relativeStart.x * relativeEnd.y;
  let backward = relativeStart.y * relativeEnd.x;
  var determinant = forward - backward;
  if (!isFiniteBits(startPosition.x) || !isFiniteBits(startPosition.y) ||
      !isFiniteBits(endPosition.x) || !isFiniteBits(endPosition.y)) { determinant = getNaN(); }
  terms[base] = determinant;
  terms[base + 1u] = abs(forward) + abs(backward);
  terms[base + 2u] = distance(startPosition, endPosition);
  terms[base + 3u] = 1.0;
  if (isFiniteBits(startPosition.x) && isFiniteBits(startPosition.y)) {
    terms[base + 4u] = startPosition.x;
    terms[base + 5u] = startPosition.y;
    terms[base + 6u] = startPosition.x;
    terms[base + 7u] = startPosition.y;
  }`
      })
    ];
    // At each level only the first row of a pair writes; its partner is read-only in that
    // dispatch. Thus the in-place tree has disjoint readers/writers and no float atomics.
    for (let stride = 1; stride < vertexCount; stride *= 2) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-ring-reduce-${stride}`,
          operation: OPERATION,
          variant: 'ring-reduce',
          bindings: [
            ringOffsetsBinding,
            {name: 'terms', view: terms, type: 'f32', access: 'read_write'}
          ],
          invocationCount: vertexCount,
          declarations: `${ringSearch}\noverride TREE_STRIDE: u32;`,
          constants: {TREE_STRIDE: stride},
          body: `let ring = findRing(index);
  if (ring == 0xffffffffu) { return; }
  let first = ringOffsets[ringOffsetsOffset + ring];
  let end = ringOffsets[ringOffsetsOffset + ring + 1u];
  if ((index - first) % (2u * TREE_STRIDE) != 0u || index + TREE_STRIDE >= end) { return; }
  let base = termsOffset + 9u * index;
  let partner = termsOffset + 9u * (index + TREE_STRIDE);
  for (var slot = 0u; slot < 4u; slot++) { terms[base + slot] += terms[partner + slot]; }
  for (var slot = 4u; slot < 6u; slot++) { terms[base + slot] = min(terms[base + slot], terms[partner + slot]); }
  for (var slot = 6u; slot < 8u; slot++) { terms[base + slot] = max(terms[base + slot], terms[partner + slot]); }
  terms[base + 8u] += terms[partner + 8u];`
        })
      );
    }
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-ring-orientation`,
        operation: OPERATION,
        variant: 'ring-orientation',
        bindings: [
          ringOffsetsBinding,
          {name: 'polygonOffsets', view: props.polygonOffsets, type: 'u32', access: 'read'},
          {name: 'featureOffsets', view: props.featureOffsets, type: 'u32', access: 'read'},
          {name: 'terms', view: terms, type: 'f32', access: 'read'},
          {...ringsBinding, access: 'read_write'}
        ],
        invocationCount: ringCount,
        declarations: `${IS_FINITE_BITS_WGSL}${getOffsetRangeSearchSource('findPolygon', 'polygonOffsets', props.polygonOffsets.length - 1)}`,
        body: `let base = ringsOffset + ${RING_STRIDE}u * index;
  for (var slot = 0u; slot < ${RING_STRIDE}u; slot++) { rings[base + slot] = 0.0; }
  let polygon = findPolygon(index);
  if (polygon == 0xffffffffu || polygon < featureOffsets[featureOffsetsOffset] ||
      polygon >= featureOffsets[featureOffsetsOffset + 1u]) { return; }
  let first = ringOffsets[ringOffsetsOffset + index];
  let end = ringOffsets[ringOffsetsOffset + index + 1u];
  if (end - first < 3u) { return; }
  let source = termsOffset + 9u * first;
  if (terms[source + 8u] == 0.0) { return; }
  rings[base + 9u] = 1.0;
  let doubledArea = terms[source];
  let role = select(-1.0, 1.0, index == polygonOffsets[polygonOffsetsOffset + polygon]);
  rings[base] = select(0.0, abs(doubledArea) * 0.5 * role, end - first >= 3u);
  rings[base + 1u] = terms[source + 2u];
  rings[base + 2u] = terms[source + 3u];
  let errorBound = ${8 * 2 ** -23 * (1 + Math.ceil(Math.log2(vertexCount)))} * terms[source + 1u];
  if (!isFiniteBits(doubledArea) || abs(doubledArea) <= errorBound) {
    rings[base + 3u] = ${GPU_STUDY_WINDOW_FLAG_ORIENTATION_UNCERTAIN}.0;
  }
  for (var slot = 4u; slot < 8u; slot++) { rings[base + slot] = terms[source + slot]; }
  rings[base + 8u] = sign(doubledArea) * role;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-orient-edges`,
        operation: OPERATION,
        variant: 'orient-edges',
        bindings: [
          positionsBinding,
          ringOffsetsBinding,
          ringsBinding,
          {name: 'edges', view: edges, type: 'f32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        declarations: `${ringSearch}${vertexReader}${GET_NAN_WGSL}`,
        body: `let base = edgesOffset + 4u * index;
  for (var slot = 0u; slot < 4u; slot++) { edges[base + slot] = getNaN(); }
  let ring = findRing(index);
  if (ring == 0xffffffffu) { return; }
  let ringBase = ringsOffset + ${RING_STRIDE}u * ring;
  if (rings[ringBase + 9u] == 0.0) { return; }
  let first = ringOffsets[ringOffsetsOffset + ring];
  let end = ringOffsets[ringOffsetsOffset + ring + 1u];
  let next = select(index + 1u, first, index + 1u == end);
  let reverse = rings[ringBase + 8u] < 0.0;
  let startPosition = readVertex(select(index, next, reverse));
  let endPosition = readVertex(select(next, index, reverse));
  edges[base] = startPosition.x;
  edges[base + 1u] = startPosition.y;
  edges[base + 2u] = endPosition.x;
  edges[base + 3u] = endPosition.y;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-summarize`,
        operation: OPERATION,
        variant: 'summarize',
        bindings: [ringsBinding, {...summaryBinding, access: 'read_write'}],
        invocationCount: 1,
        declarations: `${IS_FINITE_BITS_WGSL}${GET_NAN_WGSL}${ADDITION_ROUNDING_ERROR_WGSL}`,
        // The ring totals carry the exact rounding error of every addition, read off the integer
        // significands: a float Kahan term is algebraically zero and a fast-math shader compiler
        // removes it, which lost every small ring that followed a large one.
        body: `var totals = vec3f(0.0);
  var roundingErrors = vec3f(0.0);
  var minimum = vec2f(3.402823466e38);
  var maximum = vec2f(-3.402823466e38);
  var flags = 0u;
  for (var ring = 0u; ring < ${ringCount}u; ring++) {
    let base = ringsOffset + ${RING_STRIDE}u * ring;
    if (rings[base + 9u] == 0.0) { continue; }
    let values = vec3f(rings[base], rings[base + 1u], rings[base + 2u]);
    let nextTotals = totals + values;
    roundingErrors += vec3f(
      getAdditionRoundingError(totals.x, values.x, nextTotals.x),
      getAdditionRoundingError(totals.y, values.y, nextTotals.y),
      getAdditionRoundingError(totals.z, values.z, nextTotals.z)
    );
    totals = nextTotals;
    flags |= u32(rings[base + 3u]);
    if (values.z > 0.0) {
      minimum = min(minimum, vec2f(rings[base + 4u], rings[base + 5u]));
      maximum = max(maximum, vec2f(rings[base + 6u], rings[base + 7u]));
    }
  }
  totals += roundingErrors;
  if (!isFiniteBits(totals.x) || totals.x <= 0.0) { flags |= ${GPU_STUDY_WINDOW_FLAG_EMPTY}u; }
  if (any(minimum > maximum)) { minimum = vec2f(getNaN()); maximum = minimum; }
  summary[summaryOffset] = minimum.x;
  summary[summaryOffset + 1u] = minimum.y;
  summary[summaryOffset + 2u] = maximum.x;
  summary[summaryOffset + 3u] = maximum.y;
  summary[summaryOffset + 4u] = totals.x;
  summary[summaryOffset + 5u] = totals.y;
  summary[summaryOffset + 6u] = totals.z;
  summary[summaryOffset + 7u] = f32(flags);`
      })
    );
    const {nodes: slabNodes, slabs: transientSlabs} = getPolygonEdgeSlabNodes(graph, {
      id,
      operation: OPERATION,
      owner: 'feature',
      positions: props.positions,
      featureOffsets: props.featureOffsets,
      polygonOffsets: props.polygonOffsets,
      ringOffsets: props.ringOffsets,
      entryCapacity: props.slabEntryCapacity
    });
    nodes.push(...slabNodes);
    const slabs = {
      ...transientSlabs,
      params: this.createPersistentView(
        graph,
        'edge-slab-params',
        'uint32',
        transientSlabs.params.length
      ),
      slabOffsets: this.createPersistentView(
        graph,
        'edge-slab-offsets',
        'uint32',
        transientSlabs.slabOffsets.length
      ),
      slabEntries: this.createPersistentView(
        graph,
        'edge-slab-entries',
        'uint32x2',
        transientSlabs.slabEntries.length
      )
    };
    for (const name of ['params', 'slabOffsets', 'slabEntries'] as const) {
      const wordCount = transientSlabs[name].length * (name === 'slabEntries' ? 2 : 1);
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-edges-copy-${name}`,
          operation: OPERATION,
          variant: 'copy-words',
          bindings: [
            {name: 'source', view: transientSlabs[name], type: 'u32', access: 'read'},
            {name: 'destination', view: slabs[name], type: 'u32', access: 'read_write'}
          ],
          invocationCount: Math.max(wordCount, 1),
          declarations: `const WORD_COUNT: u32 = ${wordCount}u;`,
          body: `if (index >= WORD_COUNT) { return; }
  destination[destinationOffset + index] = source[sourceOffset + index];`
        })
      );
    }
    const parameters = createTransientView(graph, `${id}-sample-parameters`, 'float32', 4);
    const zones = createTransientView(graph, `${id}-sample-zones`, 'uint32', cellCount);
    const boundary = createTransientView(graph, `${id}-sample-boundary`, 'uint32', cellCount);
    const overflow = createTransientView(graph, `${id}-sample-overflow`, 'uint32', 1);
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-sample-parameters`,
        operation: OPERATION,
        variant: 'sample-parameters',
        bindings: [
          summaryBinding,
          {name: 'parameters', view: parameters, type: 'f32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `fn readSummary(slot: u32) -> f32 { return summary[summaryOffset + slot]; }
${getStudyWindowSampleCellWGSL(sampleGrid, 'readSummary')}`,
        body: `let cell = getStudyWindowSampleCell(0u);
  for (var slot = 0u; slot < 4u; slot++) { parameters[parametersOffset + slot] = cell[slot]; }`
      }),
      ...new GPUPolygonRasterization({
        id: `${id}-raster`,
        spatialContext: this.spatialContext,
        grid: {
          width: sampleGrid[0],
          height: sampleGrid[1],
          affine: [1, 0, 0, 0, 1, 0],
          pixelInterpretation: 'area'
        },
        parameters,
        positions: props.positions,
        featureOffsets: props.featureOffsets,
        polygonOffsets: props.polygonOffsets,
        ringOffsets: props.ringOffsets,
        crossingCapacity: props.crossingCapacity ?? 4 * sampleGrid[1] * ringCount,
        output: {zones, boundary, status: {overflow}}
      }).getCommandNodes(graph)
    );
    // A raster overflow leaves all zones empty while boundary flags remain complete. Recover
    // centers directly in that exceptional case so an undersized crossing capacity never biases CSR.
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-sample-overflow-recovery`,
        operation: OPERATION,
        variant: 'sample-overflow-recovery',
        bindings: [
          summaryBinding,
          {name: 'edges', view: edges, type: 'f32', access: 'read'},
          {name: 'overflow', view: overflow, type: 'u32', access: 'read'},
          {name: 'zones', view: zones, type: 'u32', access: 'read_write'}
        ],
        invocationCount: cellCount,
        declarations: `${EXACT_ORIENTATION_WGSL}${EXACT_GEOMETRY_WGSL}
fn readSummary(slot: u32) -> f32 { return summary[summaryOffset + slot]; }
${getStudyWindowSampleCellWGSL(sampleGrid, 'readSummary')}`,
        body: `if (overflow[overflowOffset] == 0u) { return; }
  let cell = getStudyWindowSampleCell(index);
  let center = cell.xy + 0.5 * cell.zw;
  let probe = ExactProbe(center, center, center, EXACT_PROBE_POINT);
  var winding = 0;
  for (var vertex = 0u; vertex < ${vertexCount}u; vertex++) {
    let base = edgesOffset + 4u * vertex;
    let startPosition = vec2f(edges[base], edges[base + 1u]);
    let endPosition = vec2f(edges[base + 2u], edges[base + 3u]);
    if (!isFiniteBits(startPosition.x) || !isFiniteBits(startPosition.y) ||
        !isFiniteBits(endPosition.x) || !isFiniteBits(endPosition.y)) { continue; }
    winding += exactProbeCrossing(probe, startPosition, endPosition).x;
  }
  zones[zonesOffset + index] = select(${GPU_POLYGON_RASTERIZATION_NO_ZONE}u, 0u, (winding & 1) != 0);`
      })
    );
    const counts = createTransientView(graph, `${id}-sample-counts`, 'uint32', cellCount + 1);
    const offsets = createTransientView(graph, `${id}-sample-offsets`, 'uint32', cellCount + 1);
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-sample-count`,
        operation: OPERATION,
        variant: 'sample-count',
        bindings: [
          {name: 'zones', view: zones, type: 'u32', access: 'read'},
          {name: 'boundary', view: boundary, type: 'u32', access: 'read'},
          {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
        ],
        invocationCount: cellCount + 1,
        body: `var count = 0u;
  if (index < ${cellCount}u) {
    count = select(0u, 1u, zones[zonesOffset + index] != ${GPU_POLYGON_RASTERIZATION_NO_ZONE}u || boundary[boundaryOffset + index] != 0u);
  }
  counts[countsOffset + index] = count;`
      }),
      ...new GPUScan({
        id: `${id}-sample-scan`,
        input: counts,
        output: offsets,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode(graph, {
        id: `${id}-sample-compact`,
        operation: OPERATION,
        variant: 'sample-compact',
        bindings: [
          {name: 'counts', view: counts, type: 'u32', access: 'read'},
          {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
          {name: 'boundary', view: boundary, type: 'u32', access: 'read'},
          {name: 'sampleCells', view: sampleCells, type: 'u32', access: 'read_write'},
          {name: 'sampleCellCount', view: sampleCellCount, type: 'u32', access: 'read_write'}
        ],
        invocationCount: cellCount + 1,
        body: `let count = offsets[offsetsOffset + ${cellCount}u];
  if (index == ${cellCount}u) { sampleCellCount[sampleCellCountOffset] = count; return; }
  if (index >= count) { sampleCells[sampleCellsOffset + index] = 0u; }
  if (counts[countsOffset + index] != 0u) {
    sampleCells[sampleCellsOffset + offsets[offsetsOffset + index]] = index |
      select(0u, ${STUDY_WINDOW_BOUNDARY_CELL_BIT}u, boundary[boundaryOffset + index] != 0u);
  }`
      })
    );
    const layout = preparedLayout;
    const packed = this.createPersistentView(graph, 'packed', 'uint32', layout.length);
    // Each copy binds only its source and destination. Binding floats as words preserves all bits.
    for (const [name, source, length] of [
      ['summary', summary, summary.length],
      ['edges', edges, edges.length],
      ['slabParams', slabs.params, slabs.params.length],
      ['slabOffsets', slabs.slabOffsets, slabs.slabOffsets.length],
      ['slabEntries', slabs.slabEntries, 2 * slabs.slabEntries.length],
      ['sampleCellCount', sampleCellCount, 1],
      ['sampleCells', sampleCells, cellCount]
    ] as const) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-pack-${name}`,
          operation: OPERATION,
          variant: 'pack',
          bindings: [
            {name: 'source', view: source, type: 'u32', access: 'read'},
            {name: 'packed', view: packed, type: 'u32', access: 'read_write'}
          ],
          invocationCount: length,
          declarations: 'override DESTINATION_OFFSET: u32;',
          constants: {DESTINATION_OFFSET: layout[name]},
          body: 'packed[packedOffset + DESTINATION_OFFSET + index] = source[sourceOffset + index];'
        })
      );
    }
    if (props.output?.summary) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-copy-summary`,
          operation: OPERATION,
          variant: 'copy-summary',
          bindings: [
            summaryBinding,
            {name: 'destination', view: props.output.summary, type: 'f32', access: 'read_write'}
          ],
          invocationCount: GPU_STUDY_WINDOW_SUMMARY_LENGTH,
          body: 'destination[destinationOffset + index] = summary[summaryOffset + index];'
        })
      );
    }
    this.views = {
      edges,
      summary,
      slabs,
      sampleCells,
      sampleCellCount,
      sampleGrid,
      packed
    };
    return [...snapshot.nodes, ...nodes, snapshot.commit(nodes)];
  }
}
