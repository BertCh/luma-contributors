// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type CommandEncoder, type Device} from '@luma.gl/core';
import {Kernel} from '@luma-contributors/graph';
import {
  GPUData,
  GPUVector,
  getGPUVectorFormatInfo,
  type GPUVectorFormat,
  type VertexList
} from '@luma-contributors/graph/gpu-data';
import {
  GPUCommandGraph,
  GraphVectorView,
  GPUGather,
  GPUMask,
  createTransientView,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {
  prepareGeometrySource,
  makeGeometryGPUVectors,
  type PreparedGeometryGPUVectors
} from '@luma.gl/engine/arrow';
import type {KeplerDatasetRegistry, KeplerMemoryReservation} from '@luma-contributors/kepler';
import {KeplerAllocationError} from '@luma-contributors/kepler';
import {getH3Boundary, type PreparedH3Source, type PreparedH3ViewportMesh} from './h3-source';
import {getH3HoveredOutline} from './h3-highlight';
import {GeometryRuntime} from './geometry-runtime';
import {splitFloat64Coordinates} from './h3-fp64-split';
import type {ResidentH3Binding} from './resident-layers';
export type {ResidentH3Binding} from './resident-layers';

export type H3GPUVectors = {
  source: PreparedH3Source;
  centers: GPUVector<'float32x2'>;
  /** Float32 residual of each canonical Float64 center; present only when boundaries collapse in Float32. */
  centersLow?: GPUVector<'float32x2'>;
  /** True when Float32 boundaries collapse: columns use hi + lo centers, polygons hi + lo vertex streams. */
  requiresFp64: boolean;
  validity: GPUVector<'uint32'>;
  resolutions: GPUVector<'uint32'>;
  pentagons: GPUVector<'uint32'>;
  primitiveSourceRows: GPUVector<'uint32'>;
  outlineVertexRows: GPUVector<'uint32'>;
  geometry: PreparedGeometryGPUVectors;
  gpuByteLength: number;
  typedStorageByteLength: number;
  isCurrent(): boolean;
  resolveSourceRow(primitiveIndex: number): number;
  destroy(completion: PromiseLike<unknown>): Promise<void>;
};

/** Source preparation/upload is explicit; interaction never re-decodes cells or re-tessellates them. */
export function makeH3GPUVectors(
  device: Device,
  source: PreparedH3Source,
  options: {
    workingMemoryBudgetBytes: number;
    memoryRegistry?: KeplerDatasetRegistry;
  }
): H3GPUVectors {
  const requiresFp64 = source.precision.collapsedRows.length > 0;
  const buffers: Buffer[] = [];
  const reservations: KeplerMemoryReservation[] = [];
  let geometry: PreparedGeometryGPUVectors | undefined;
  try {
    if (
      !Number.isSafeInteger(options.workingMemoryBudgetBytes) ||
      options.workingMemoryBudgetBytes < 0
    )
      throw new Error('H3 upload requires an explicit memory budget');
    const minimumUploadBytes = source.batches.reduce(
      (sum, batch) =>
        sum +
        Math.max(4, batch.rowCount * 8) +
        3 * Math.max(4, batch.rowCount * 4) +
        Math.max(4, batch.primitiveSourceRows.byteLength),
      0
    );
    if (minimumUploadBytes > options.workingMemoryBudgetBytes)
      throw new KeplerAllocationError('working-memory', {
        message: 'H3 upload exceeds its memory reservation'
      });
    // H3 cells are single rings without holes. Reserve a conservative bound for their source
    // channels, tessellated top vertices/triangles, outline vertices/maps and empty batch buffers
    // before any GPU allocation. Actual retained bytes remain separately measured below.
    const conservativeUploadBytes =
      source.vertexCount * 96 + source.rowCount * 96 + source.batches.length * 256;
    if (conservativeUploadBytes > options.workingMemoryBudgetBytes)
      throw new Error('H3 upload budget does not cover its conservative geometry preflight');
    if (options.memoryRegistry) {
      reservations.push(
        options.memoryRegistry.reserveMemory(
          source.typedStorageByteLength + source.identityByteLength,
          'cpu',
          'source'
        )
      );
      reservations.push(
        options.memoryRegistry.reserveMemory(options.workingMemoryBudgetBytes, 'gpu', 'scratch')
      );
    }
    const upload = <Format extends GPUVectorFormat>(
      name: string,
      format: Format,
      chunks: readonly (Float32Array | Uint32Array)[],
      widths = 1
    ): GPUVector<Format> =>
      new GPUVector({
        type: 'data',
        name,
        format,
        ownsData: false,
        data: chunks.map(data => {
          const buffer = device.createBuffer({
            ...(data.byteLength ? {data} : {byteLength: 4}),
            usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
          });
          buffers.push(buffer);
          return new GPUData({buffer, format, length: data.length / widths, ownsBuffer: false});
        })
      });
    const centers = upload(
      'h3-centers',
      'float32x2',
      source.batches.map(batch => Float32Array.from(batch.centers)),
      2
    );
    const centersLow = requiresFp64
      ? upload(
          'h3-centers-low',
          'float32x2',
          source.batches.map(batch => splitFloat64Coordinates(batch.centers).low),
          2
        )
      : undefined;
    const validity = upload(
      'h3-validity',
      'uint32',
      source.batches.map(batch => batch.validity)
    );
    const resolutions = upload(
      'h3-resolutions',
      'uint32',
      source.batches.map(batch => batch.resolutions)
    );
    const pentagons = upload(
      'h3-pentagons',
      'uint32',
      source.batches.map(batch => batch.pentagons)
    );
    const primitiveSourceRows = upload(
      'h3-primitive-rows',
      'uint32',
      source.batches.map(batch => batch.primitiveSourceRows)
    );
    const prepared = prepareGeometrySource(
      source.batches.map(batch => ({
        id: batch.id,
        rows: Array.from({length: batch.rowCount}, (_, row) => {
          if (!batch.validity[row]) return null;
          const start = batch.boundaryOffsets[row];
          const end = batch.boundaryOffsets[row + 1];
          const firstLongitude = batch.boundaryPositions[start * 2];
          const ring = Array.from({length: end - start}, (_, vertex) => {
            let longitude = batch.boundaryPositions[(start + vertex) * 2];
            const difference = longitude - firstLongitude;
            if (difference > 180) longitude -= 360;
            else if (difference < -180) longitude += 360;
            return [longitude, batch.boundaryPositions[(start + vertex) * 2 + 1]];
          });
          return {type: 'Polygon', coordinates: [ring]};
        })
      })),
      // Cells that collapse in Float32 keep a low vertex stream for every render branch.
      {fp64: requiresFp64}
    );
    geometry = makeGeometryGPUVectors(device, prepared);
    const outlineRows = geometry.polygonOutlines.getPath.data.map((chunk, index) => {
      const rows = new Uint32Array(chunk.valueLength);
      const offsets = chunk.valueOffsets;
      if (!offsets) throw new Error('H3 outlines require retained topology offsets');
      const sourceRows = geometry!.polygonOutlines.sourceRows[index];
      for (let row = 0; row < chunk.length; row++)
        rows.fill(sourceRows[row], offsets[row], offsets[row + 1]);
      return rows;
    });
    const outlineVertexRows = upload('h3-outline-vertex-rows', 'uint32', outlineRows);
    const gpuByteLength =
      geometry.gpuByteLength + buffers.reduce((total, buffer) => total + buffer.byteLength, 0);
    if (gpuByteLength > options.workingMemoryBudgetBytes)
      throw new KeplerAllocationError('working-memory', {
        message: 'H3 upload exceeds its memory reservation'
      });
    if (options.memoryRegistry)
      reservations.push(
        options.memoryRegistry.reserveMemory(geometry.mappingByteLength, 'cpu', 'source')
      );
    const retainedGeometry = geometry;
    const deviceLoss = device.lost;
    let current = true;
    let destruction: Promise<void> | undefined;
    const result: H3GPUVectors = {
      source,
      centers,
      centersLow,
      requiresFp64,
      validity,
      resolutions,
      pentagons,
      primitiveSourceRows,
      outlineVertexRows,
      geometry,
      gpuByteLength,
      typedStorageByteLength:
        source.typedStorageByteLength + source.identityByteLength + geometry.mappingByteLength,
      isCurrent: () => current,
      resolveSourceRow(index) {
        if (!Number.isSafeInteger(index) || index < 0) return -1;
        for (const batch of source.batches) {
          if (index < batch.primitiveSourceRows.length) return batch.primitiveSourceRows[index];
          index -= batch.primitiveSourceRows.length;
        }
        return -1;
      },
      destroy(completion) {
        if (!destruction) {
          current = false;
          destruction = Promise.race([completion, deviceLoss])
            .finally(() => {
              retainedGeometry.destroy();
              for (const buffer of buffers) buffer.destroy();
              for (const reservation of reservations) reservation.release();
            })
            .then(() => undefined);
        }
        return destruction;
      }
    };
    void deviceLoss.then(() => result.destroy(Promise.resolve())).catch(() => {});
    return result;
  } catch (error) {
    geometry?.destroy();
    for (const buffer of buffers) buffer.destroy();
    for (const reservation of reservations) reservation.release();
    throw error;
  }
}

export type H3SelectedMetadata = {
  revision: number;
  count: number;
  minimumResolution: number;
  maximumResolution: number;
  hasPentagon: boolean;
};

/** Native topology is a separate scope from display filtering; one bounded summary chooses the branch. */
export class H3MetadataRuntime {
  private readonly graph!: CompiledGPUCommandGraph;
  private readonly summary!: Buffer;
  private readonly reservation?: KeplerMemoryReservation;
  private revision = 1;
  private encodedRevision = 0;
  private generation = 0;
  private current?: H3SelectedMetadata;
  private pending?: Promise<H3SelectedMetadata | null>;
  private destroyed = false;
  private released = false;
  private destruction?: Promise<void>;
  private transferred = 0;
  constructor(
    private readonly props: {
      device: Device;
      source: H3GPUVectors;
      mask: GPUVector<'uint32'>;
      revision: number;
      workingMemoryBudgetBytes: number;
      memoryRegistry?: KeplerDatasetRegistry;
    }
  ) {
    this.setRevision(props.revision);
    if (props.mask.length !== props.source.source.rowCount || !props.source.isCurrent())
      throw new Error('H3 metadata source does not match its topology mask');
    if (
      !Number.isSafeInteger(props.workingMemoryBudgetBytes) ||
      props.workingMemoryBudgetBytes < 32
    )
      throw new Error('H3 metadata requires an explicit memory budget');
    this.reservation = props.memoryRegistry?.reserveMemory(
      props.workingMemoryBudgetBytes,
      'gpu',
      'scratch'
    );
    try {
      this.summary = props.device.createBuffer({
        data: new Uint32Array([0xffffffff, 0, 0, 0]),
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      const graph = new GPUCommandGraph(props.device, {id: 'h3-selected-metadata'});
      const sourceValidity = graph.importGPUVector('source-validity', props.source.validity);
      const scope = graph.importGPUVector('native-topology', props.mask);
      const masks = sourceValidity.data.map((chunk, index) =>
        createTransientView(graph, `valid-${index}`, 'uint32', chunk.length)
      );
      const validMask = new GraphVectorView({
        id: 'valid-masks',
        name: 'valid-masks',
        format: 'uint32',
        length: props.mask.length,
        valueLength: props.mask.length,
        stride: 1,
        byteStride: 4,
        rowByteLength: 4,
        data: masks
      });
      graph.add(
        new GPUMask({id: 'h3-valid-scope', inputs: [sourceValidity, scope], output: validMask})
      );
      const resolutions = graph.importGPUVector('resolutions', props.source.resolutions);
      const pentagons = graph.importGPUVector('pentagons', props.source.pentagons);
      const summary = importView(graph, 'summary', this.summary, 'uint32', 4);
      for (const [index, mask] of masks.entries()) {
        if (!mask.length) continue;
        const resolution = resolutions.data[index];
        const pentagon = pentagons.data[index];
        const dispatch = getBoundedDispatchLayout(
          'H3Metadata',
          mask.length,
          256,
          props.device.limits.maxComputeWorkgroupsPerDimension
        );
        graph.addComputePass({
          id: `reduce-${index}`,
          resources: [
            {buffer: mask, usage: 'storage-read'},
            {buffer: resolution, usage: 'storage-read'},
            {buffer: pentagon, usage: 'storage-read'},
            {buffer: summary, usage: 'storage-read-write'}
          ],
          compile: ({device}) => {
            const kernel = new Kernel(device, {
              id: 'h3-metadata-reduction',
              source: /* wgsl */ `
@group(0) @binding(0) var<storage, read> mask: array<u32>;
@group(0) @binding(1) var<storage, read> resolutions: array<u32>;
@group(0) @binding(2) var<storage, read> pentagons: array<u32>;
@group(0) @binding(3) var<storage, read_write> summary: array<atomic<u32>>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) localInvocationIndex: u32, @builtin(workgroup_id) workgroupId: vec3<u32>) {
  ${getBoundedInvocationIndexSource(dispatch, 256)}
  if (index >= ${mask.length}u || mask[${mask.byteOffset / 4}u + index] == 0u) { return; }
  let resolution = resolutions[${resolution.byteOffset / 4}u + index];
  atomicMin(&summary[0], resolution); atomicMax(&summary[1], resolution);
  atomicOr(&summary[2], pentagons[${pentagon.byteOffset / 4}u + index]); atomicAdd(&summary[3], 1u);
}`,
              shaderLayout: {
                bindings: ['mask', 'resolutions', 'pentagons', 'summary'].map((name, location) => ({
                  name,
                  location,
                  group: 0,
                  type: 'storage' as const
                }))
              }
            });
            return {
              encode: ({computePass, getBuffer}) =>
                kernel.dispatch(computePass, {
                  bindings: {
                    mask: getBuffer(mask.buffer),
                    resolutions: getBuffer(resolution.buffer),
                    pentagons: getBuffer(pentagon.buffer),
                    summary: getBuffer(summary.buffer)
                  },
                  ...dispatch
                }),
              destroy: () => kernel.destroy()
            };
          }
        });
      }
      this.graph = graph.compile();
      if (this.accountedResidentByteLength + 16 > props.workingMemoryBudgetBytes)
        throw new KeplerAllocationError('working-memory', {
          message: 'H3 metadata exceeds its reserved memory'
        });
      void props.device.lost.then(() => this.destroy(Promise.resolve())).catch(() => {});
    } catch (error) {
      this.graph?.destroy();
      this.summary?.destroy();
      this.reservation?.release();
      throw error;
    }
  }
  /** Bounded control state for host diagnostics; no source reads or GPU transfers. */
  get status(): {
    requestedRevision: number;
    encodedRevision: number;
    pending: boolean;
    snapshot?: H3SelectedMetadata;
  } {
    return {
      requestedRevision: this.revision,
      encodedRevision: this.encodedRevision,
      pending: Boolean(this.pending),
      snapshot: this.snapshot
    };
  }
  get snapshot(): H3SelectedMetadata | undefined {
    return !this.destroyed &&
      this.props.source.isCurrent() &&
      this.current?.revision === this.revision
      ? this.current
      : undefined;
  }
  get readbackByteLength(): number {
    return this.transferred;
  }
  get accountedResidentByteLength(): number {
    return this.released
      ? 0
      : this.summary.byteLength + (this.graph?.stats.physicalTransientResourceBytes ?? 0);
  }
  setRevision(revision: number): void {
    if (!Number.isSafeInteger(revision) || revision < 1)
      throw new Error('H3 topology revision must be positive');
    if (revision !== this.revision) {
      this.current = undefined;
      this.encodedRevision = 0;
      this.generation++;
    }
    this.revision = revision;
  }
  encode(encoder: CommandEncoder): void {
    if (this.destroyed || !this.props.source.isCurrent() || encoder.device !== this.props.device)
      throw new Error('H3 metadata source is closed');
    if (this.encodedRevision === this.revision) return;
    this.props.device.writeBufferViaCommandEncoder(
      encoder,
      this.summary,
      new Uint32Array([0xffffffff, 0, 0, 0])
    );
    this.graph.encode(encoder, {parameters: undefined});
    this.encodedRevision = this.revision;
    this.generation++;
  }
  discardEncoding(): void {
    this.encodedRevision = 0;
    this.current = undefined;
    this.generation++;
  }
  readMetadata(): Promise<H3SelectedMetadata | null> {
    if (this.destroyed || !this.props.source.isCurrent() || this.encodedRevision !== this.revision)
      return Promise.resolve(null);
    if (this.current?.revision === this.revision) return Promise.resolve(this.current);
    if (this.pending) return this.pending;
    const revision = this.revision;
    const generation = this.generation;
    this.transferred += 16;
    const pending = this.summary
      .readAsync(0, 16)
      .then(bytes => {
        if (
          this.destroyed ||
          !this.props.source.isCurrent() ||
          revision !== this.revision ||
          revision !== this.encodedRevision ||
          generation !== this.generation
        )
          return null;
        const values = new Uint32Array(bytes.buffer, bytes.byteOffset, 4);
        this.current = {
          revision,
          count: values[3],
          minimumResolution: values[0],
          maximumResolution: values[1],
          hasPentagon: values[2] !== 0
        };
        return this.current;
      })
      .catch(error => {
        if (this.destroyed || !this.props.source.isCurrent()) return null;
        throw error;
      });
    this.pending = pending;
    void pending
      .finally(() => {
        if (this.pending === pending) this.pending = undefined;
      })
      .catch(() => {});
    return pending;
  }
  destroy(completion: PromiseLike<unknown>): Promise<void> {
    if (!this.destruction) {
      this.destroyed = true;
      this.destruction = Promise.race([completion, this.props.device.lost])
        .finally(async () => {
          await Promise.allSettled(this.pending ? [this.pending] : []);
          this.released = true;
          this.graph.destroy();
          this.summary.destroy();
          this.reservation?.release();
        })
        .then(() => undefined);
    }
    return this.destruction;
  }
}

function importView<Format extends GPUVectorFormat>(
  graph: GPUCommandGraph,
  id: string,
  buffer: Buffer,
  format: Format,
  length: number
): GraphDataView<Format> {
  return graph.createDataView(
    graph.importBuffer({id, byteLength: buffer.byteLength, usage: buffer.usage}, buffer),
    {format, length}
  );
}

export type H3RuntimeProps = {
  device: Device;
  column: string;
  source: H3GPUVectors;
  branch: 'polygons' | 'columns';
  selectionMask: GPUVector<'uint32'>;
  colors?: GPUVector<'unorm8x4'>;
  strokeColors?: GPUVector<'unorm8x4'>;
  coverages?: GPUVector<'float32'>;
  elevations?: GPUVector<'float32'>;
  coverage: number;
  viewportMesh?: PreparedH3ViewportMesh;
  workingMemoryBudgetBytes: number;
  memoryRegistry?: KeplerDatasetRegistry;
};

/** Native branch outputs borrow immutable H3 source buffers and preserve global source picking IDs. */
export class H3Runtime {
  readonly binding: ResidentH3Binding;
  private readonly graph: GPUCommandGraph;
  private compiled?: CompiledGPUCommandGraph;
  private readonly buffers: Buffer[] = [];
  private readonly reservation?: KeplerMemoryReservation;
  private readonly coverageBuffer: Buffer;
  private geometryRuntime?: GeometryRuntime;
  private meshBuffer?: Buffer;
  private meshSnapshot?: Float32Array;
  private viewportMesh?: PreparedH3ViewportMesh;
  private coverage = 1;
  private coverageDirty = true;
  private destroyed = false;
  private released = false;
  private destruction?: Promise<void>;
  constructor(private readonly props: H3RuntimeProps) {
    if (!props.source.isCurrent() || props.selectionMask.length !== props.source.source.rowCount)
      throw new Error('H3 render source and display mask must agree');
    for (const vector of [
      props.selectionMask,
      props.colors,
      props.strokeColors,
      props.coverages,
      props.elevations
    ]) {
      if (vector && vector.length !== props.source.source.rowCount)
        throw new Error('H3 style source row count mismatch');
      for (const chunk of vector?.data ?? [])
        if (chunk.buffer.device !== props.device)
          throw new Error('H3 style source device mismatch');
    }
    this.reservation = props.memoryRegistry?.reserveMemory(
      props.workingMemoryBudgetBytes,
      'gpu',
      'scratch'
    );
    this.graph = new GPUCommandGraph(props.device, {id: `h3-${props.column}-${props.branch}`});
    try {
      this.coverageBuffer = this.allocate('coverage', 4);
      this.setCoverage(props.coverage);
      if (!props.source.source.validRowCount) {
        this.binding = {column: props.column, branch: 'empty'};
      } else if (props.branch === 'columns') {
        if (!props.viewportMesh)
          throw new Error('Native H3 columns require current viewport geometry');
        const mesh = props.viewportMesh;
        // Native H3 fixes diskResolution=6 even when the viewport boundary has extra face vertices.
        const vertexCount = 6;
        if (mesh.positions.length < vertexCount * 2)
          throw new Error('Native H3 footprint requires six supplied vertices');
        this.meshBuffer = this.allocate('viewport-mesh', vertexCount * 8);
        this.binding = {
          column: props.column,
          branch: props.branch,
          columns: {
            positions: this.gather(
              'centers',
              props.source.centers,
              props.source.primitiveSourceRows
            ),
            vertices: new GPUData({
              format: 'float32x2',
              buffer: this.meshBuffer,
              length: vertexCount,
              ownsBuffer: false
            }),
            positionsLow: props.source.centersLow
              ? this.gather(
                  'centers-low',
                  props.source.centersLow,
                  props.source.primitiveSourceRows
                )
              : undefined,
            edgeDistance: 0,
            selectionMask: this.gather(
              'display',
              props.selectionMask,
              props.source.primitiveSourceRows
            ),
            colors: props.colors
              ? this.gather('colors', props.colors, props.source.primitiveSourceRows)
              : undefined,
            strokeColors: props.strokeColors
              ? this.gather('stroke-colors', props.strokeColors, props.source.primitiveSourceRows)
              : undefined,
            coverages: props.coverages
              ? this.gather('coverages', props.coverages, props.source.primitiveSourceRows)
              : undefined,
            elevations: props.elevations
              ? this.gather('elevations', props.elevations, props.source.primitiveSourceRows)
              : undefined,
            resolveSourceRow: props.source.resolveSourceRow
          }
        };
        this.setViewportMesh(mesh);
      } else {
        const sourceGeometry = props.source.geometry;
        const vertexRows = flattenListVector(sourceGeometry.polygons.rowIndices, 'uint32');
        // fp64 sources carry low vertex streams; the coverage kernel transforms both parts together.
        const {output: positions, outputLow: positionsLow} = this.transformGeometry(
          'polygons',
          sourceGeometry.polygons.positions,
          'float32x4',
          vertexRows,
          sourceGeometry.polygons.positionsLow
        );
        const {output: outlines, outputLow: outlinesLow} = this.transformGeometry(
          'outlines',
          sourceGeometry.polygonOutlines.getPath,
          'float32x3',
          props.source.outlineVertexRows,
          sourceGeometry.polygonOutlines.getPathLow
        );
        const geometry = {
          ...sourceGeometry,
          polygons: {...sourceGeometry.polygons, positions, positionsLow},
          polygonOutlines: {
            ...sourceGeometry.polygonOutlines,
            getPath: outlines,
            getPathLow: outlinesLow
          }
        };
        this.geometryRuntime = new GeometryRuntime({
          device: props.device,
          column: props.column,
          geometry,
          selectionMask: props.selectionMask,
          colors: props.colors,
          strokeColors: props.strokeColors,
          stylesResolved: true
        });
        this.geometryRuntime.prepare();
        this.binding = {
          column: props.column,
          branch: props.branch,
          polygons: {
            ...this.geometryRuntime.binding.polygons!,
            getElevation: props.elevations,
            extrusionRings: {
              positions: outlines,
              positionsLow: outlinesLow,
              sourceRowIds: geometry.polygonOutlines.sourceRowIds,
              sideOrientations: geometry.polygonOutlines.sideOrientations,
              windingOrder: 'CCW'
            }
          },
          polygonOutlines:
            this.geometryRuntime.binding.polygonOutlines &&
            'getPath' in this.geometryRuntime.binding.polygonOutlines
              ? this.geometryRuntime.binding.polygonOutlines
              : undefined
        };
      }
      // Hovered-outline footprint of one source row (canonical CPU boundary; no GPU readback).
      this.binding.getCellBoundary = (sourceRow: number) =>
        getH3HoveredOutline(props.source.source, sourceRow, false, (batchIndex, rowIndex) =>
          getH3Boundary(props.source.source, batchIndex, rowIndex)
        );
      this.prepare();
      void props.device.lost.then(() => this.destroy(Promise.resolve())).catch(() => {});
    } catch (error) {
      this.release();
      throw error;
    }
  }
  get accountedResidentByteLength(): number {
    return this.released
      ? 0
      : this.buffers.reduce((sum, buffer) => sum + buffer.byteLength, 0) +
          (this.compiled?.stats.physicalTransientResourceBytes ?? 0) +
          (this.geometryRuntime?.accountedResidentByteLength ?? 0);
  }
  get readbackByteLength(): number {
    return 0;
  }
  setCoverage(value: number): void {
    if (!Number.isFinite(value) || value < 0 || value > 1)
      throw new Error('H3 coverage must be between zero and one');
    if (this.coverage !== Math.fround(value)) {
      this.coverage = Math.fround(value);
      this.coverageDirty = true;
    }
  }
  setViewportMesh(mesh: PreparedH3ViewportMesh): void {
    if (mesh === this.viewportMesh) return;
    const columns = this.binding.columns;
    if (!columns || mesh.positions.length < 12)
      throw new Error('Native H3 viewport footprint requires six supplied vertices');
    const positions = Float64Array.from(mesh.positions);
    if (positions.some(value => !Number.isFinite(Math.fround(value))))
      throw new Error('Native H3 viewport footprint requires finite Float32 coordinates');
    let area = 0;
    for (let index = 0, previous = positions.length - 2; index < positions.length; index += 2) {
      area +=
        (positions[index] - positions[previous]) * (positions[index + 1] + positions[previous + 1]);
      previous = index;
    }
    // Match ColumnGeometry: winding normalization runs on the FULL supplied closed perimeter,
    // then its fixed six vertices are selected. Reversing only an open prefix rotates the mesh.
    if (Math.sign(area) !== -1) {
      for (let index = 0; index < Math.floor(positions.length / 4); index++) {
        const opposite = positions.length / 2 - 1 - index;
        for (let component = 0; component < 2; component++) {
          const value = positions[index * 2 + component];
          positions[index * 2 + component] = positions[opposite * 2 + component];
          positions[opposite * 2 + component] = value;
        }
      }
    }
    const snapshot = Float32Array.from(positions.subarray(0, 12));
    let meanRadius = 0;
    for (let index = 0; index < 12; index += 2)
      meanRadius += Math.sqrt(mesh.positions[index] ** 2 + mesh.positions[index + 1] ** 2) / 6;
    this.viewportMesh = mesh;
    this.meshSnapshot = snapshot;
    columns.edgeDistance = Math.cos(Math.PI / 6) * meanRadius;
  }
  prepare(): void {
    if (this.destroyed) throw new Error('H3 runtime is closed');
    this.compiled ??= this.graph.compile();
    if (this.accountedResidentByteLength > this.props.workingMemoryBudgetBytes)
      throw new KeplerAllocationError('working-memory', {
        message: 'H3 runtime exceeds its memory reservation'
      });
  }
  encode(encoder: CommandEncoder): void {
    if (this.destroyed || !this.props.source.isCurrent() || encoder.device !== this.props.device)
      throw new Error('H3 runtime source is closed or mismatched');
    const transformNeeded = this.coverageDirty;
    if (this.coverageDirty) {
      this.props.device.writeBufferViaCommandEncoder(
        encoder,
        this.coverageBuffer,
        new Float32Array([this.coverage])
      );
      this.coverageDirty = false;
    }
    if (this.meshSnapshot && this.meshBuffer) {
      this.props.device.writeBufferViaCommandEncoder(encoder, this.meshBuffer, this.meshSnapshot);
      this.meshSnapshot = undefined;
    }
    if (this.props.branch === 'columns' || transformNeeded)
      this.compiled!.encode(encoder, {parameters: undefined});
    this.geometryRuntime?.encode(encoder);
  }
  discardEncoding(): void {
    this.coverageDirty = true;
    if (this.viewportMesh) {
      const mesh = this.viewportMesh;
      this.viewportMesh = undefined;
      this.setViewportMesh(mesh);
    }
  }
  destroy(completion: PromiseLike<unknown>): Promise<void> {
    if (!this.destruction) {
      this.destroyed = true;
      this.destruction = Promise.race([completion, this.props.device.lost])
        .finally(() => this.release())
        .then(() => undefined);
    }
    return this.destruction;
  }
  private release(): void {
    if (this.released) return;
    this.released = true;
    this.compiled?.destroy();
    this.geometryRuntime?.destroy();
    for (const buffer of this.buffers) buffer.destroy();
    this.reservation?.release();
  }
  private allocate(id: string, byteLength: number): Buffer {
    const buffer = this.props.device.createBuffer({
      id: `h3-${id}`,
      byteLength: Math.max(4, byteLength),
      usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    this.buffers.push(buffer);
    return buffer;
  }
  private gather<Format extends 'float32x2' | 'float32' | 'uint32' | 'unorm8x4'>(
    name: string,
    source: GPUVector<Format>,
    indices: GPUVector<'uint32'>
  ): GPUVector<Format> {
    const format = source.format;
    if (!format) throw new Error('H3 gathered channels require an explicit format');
    const width = getGPUVectorFormatInfo(format).byteLength;
    const output = new GPUVector({
      type: 'data',
      name,
      format: source.format,
      ownsData: false,
      data: indices.data.map(
        (chunk, index) =>
          new GPUData({
            format: source.format,
            buffer: this.allocate(`${name}-${index}`, chunk.length * width),
            length: chunk.length,
            ownsBuffer: false
          })
      )
    });
    this.graph.add(
      new GPUGather({
        id: name,
        source: this.graph.importGPUVector(`${name}-source`, source),
        indices: this.graph.importGPUVector(`${name}-indices`, indices),
        output: this.graph.importGPUVector(`${name}-output`, output)
      })
    );
    return output;
  }
  private transformGeometry<Format extends 'float32x3' | 'float32x4'>(
    name: string,
    source: GPUVector<VertexList<Format>>,
    format: Format,
    rows: GPUVector<'uint32'>,
    lowSource?: GPUVector<VertexList<'float32x2'>>
  ): {
    output: GPUVector<VertexList<Format>>;
    outputLow?: GPUVector<VertexList<'float32x2'>>;
  } {
    const centersLow =
      lowSource && this.props.source.centersLow
        ? this.gather(`${name}-centers-low`, this.props.source.centersLow, rows)
        : undefined;
    const lowInput = lowSource
      ? this.graph.importGPUVector(
          `${name}-positions-low`,
          flattenListVector(lowSource, 'float32x2')
        )
      : undefined;
    const centerLowViews = centersLow
      ? this.graph.importGPUVector(`${name}-center-low-values`, centersLow)
      : undefined;
    const outputLow = lowSource
      ? new GPUVector({
          type: 'data',
          name: `${name}-low`,
          format: 'vertex-list<float32x2>' as VertexList<'float32x2'>,
          ownsData: false,
          data: lowSource.data.map(
            (chunk, index) =>
              new GPUData({
                format: chunk.format,
                buffer: this.allocate(`${name}-low-${index}`, Math.max(4, chunk.valueLength * 8)),
                length: chunk.length,
                valueLength: chunk.valueLength,
                valueOffsets: chunk.valueOffsets,
                ownsBuffer: false
              })
          )
        })
      : undefined;
    const outputLowViews = outputLow
      ? this.graph.importGPUVector(`${name}-output-low`, flattenListVector(outputLow, 'float32x2'))
      : undefined;
    const centers = this.gather(`${name}-centers`, this.props.source.centers, rows);
    const width = format === 'float32x3' ? 3 : 4;
    const input = this.graph.importGPUVector(
      `${name}-positions`,
      flattenListVector(source, format)
    );
    const centerViews = this.graph.importGPUVector(`${name}-center-values`, centers);
    const coverage = importView(this.graph, `${name}-coverage`, this.coverageBuffer, 'float32', 1);
    const output = new GPUVector({
      type: 'data',
      name,
      format: `vertex-list<${format}>` as VertexList<Format>,
      ownsData: false,
      data: source.data.map(
        (chunk, index) =>
          new GPUData({
            format: chunk.format,
            buffer: this.allocate(`${name}-${index}`, chunk.valueLength * width * 4),
            length: chunk.length,
            valueLength: chunk.valueLength,
            valueOffsets: chunk.valueOffsets,
            ownsBuffer: false
          })
      )
    });
    const outputViews = this.graph.importGPUVector(
      `${name}-output`,
      flattenListVector(output, format)
    );
    for (const [index, positions] of input.data.entries()) {
      if (!positions.length) continue;
      const outputPosition = outputViews.data[index];
      const center = centerViews.data[index];
      const lowPositions = lowInput?.data[index];
      const centerLow = centerLowViews?.data[index];
      const outputLowPosition = outputLowViews?.data[index];
      const useLow = Boolean(lowPositions && centerLow && outputLowPosition);
      const dispatch = getBoundedDispatchLayout(
        'H3Coverage',
        positions.length,
        256,
        this.props.device.limits.maxComputeWorkgroupsPerDimension
      );
      this.graph.addComputePass({
        id: `${name}-coverage-${index}`,
        resources: [
          {buffer: positions, usage: 'storage-read'},
          {buffer: center, usage: 'storage-read'},
          {buffer: coverage, usage: 'storage-read'},
          {buffer: outputPosition, usage: 'storage-write'},
          ...(useLow
            ? [
                {buffer: lowPositions!, usage: 'storage-read' as const},
                {buffer: centerLow!, usage: 'storage-read' as const},
                {buffer: outputLowPosition!, usage: 'storage-write' as const}
              ]
            : [])
        ],
        compile: ({device}) => {
          const kernel = new Kernel(device, {
            id: useLow ? 'h3-native-coverage-fp64' : 'h3-native-coverage',
            source: /* wgsl */ `
@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> centers: array<f32>;
@group(0) @binding(2) var<storage, read> coverage: array<f32>;
@group(0) @binding(3) var<storage, read_write> outputPositions: array<f32>;
${
  useLow
    ? `@group(0) @binding(4) var<storage, read> positionsLow: array<f32>;
@group(0) @binding(5) var<storage, read> centersLow: array<f32>;
@group(0) @binding(6) var<storage, read_write> outputPositionsLow: array<f32>;`
    : ''
}
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) localInvocationIndex: u32, @builtin(workgroup_id) workgroupId: vec3<u32>) {
  ${getBoundedInvocationIndexSource(dispatch, 256)}
  if (index >= ${positions.length}u) { return; }
  let inputOffset = ${positions.byteOffset / 4}u + index * ${width}u;
  let outputOffset = ${outputPosition.byteOffset / 4}u + index * ${width}u;
  let centerOffset = ${center.byteOffset / 4}u + index * 2u;
  var longitude = positions[inputOffset]; var latitude = positions[inputOffset + 1u];
  ${
    useLow
      ? `let lowOffset = ${lowPositions!.byteOffset / 4}u + index * 2u;
  let outputLowOffset = ${outputLowPosition!.byteOffset / 4}u + index * 2u;
  let centerLowOffset = ${centerLow!.byteOffset / 4}u + index * 2u;
  var longitudeLow = positionsLow[lowOffset]; var latitudeLow = positionsLow[lowOffset + 1u];`
      : ''
  }
  if (coverage[0] != 1.0) {
    let difference = longitude - centers[centerOffset];
    if (difference > 180.0) {longitude -= 360.0;} else if (difference < -180.0) {longitude += 360.0;}
    ${
      useLow
        ? `// Scale the high/low difference from the center, then add it back with an error-free two-sum.
    let scaledLongitude = ((longitude - centers[centerOffset]) + (longitudeLow - centersLow[centerLowOffset])) * coverage[0];
    let scaledLatitude = ((latitude - centers[centerOffset + 1u]) + (latitudeLow - centersLow[centerLowOffset + 1u])) * coverage[0];
    let sumLongitude = centers[centerOffset] + scaledLongitude;
    let virtualLongitude = sumLongitude - centers[centerOffset];
    longitudeLow = ((centers[centerOffset] - (sumLongitude - virtualLongitude)) + (scaledLongitude - virtualLongitude)) + centersLow[centerLowOffset];
    longitude = sumLongitude;
    let sumLatitude = centers[centerOffset + 1u] + scaledLatitude;
    let virtualLatitude = sumLatitude - centers[centerOffset + 1u];
    latitudeLow = ((centers[centerOffset + 1u] - (sumLatitude - virtualLatitude)) + (scaledLatitude - virtualLatitude)) + centersLow[centerLowOffset + 1u];
    latitude = sumLatitude;`
        : `longitude = mix(centers[centerOffset], longitude, coverage[0]);
    latitude = mix(centers[centerOffset + 1u], latitude, coverage[0]);`
    }
  }
  outputPositions[outputOffset] = longitude; outputPositions[outputOffset + 1u] = latitude;
  ${
    useLow
      ? 'outputPositionsLow[outputLowOffset] = longitudeLow; outputPositionsLow[outputLowOffset + 1u] = latitudeLow;'
      : ''
  }
  outputPositions[outputOffset + 2u] = positions[inputOffset + 2u];
  ${width === 4 ? 'outputPositions[outputOffset + 3u] = positions[inputOffset + 3u];' : ''}
}`,
            shaderLayout: {
              bindings: [
                'positions',
                'centers',
                'coverage',
                'outputPositions',
                ...(useLow ? ['positionsLow', 'centersLow', 'outputPositionsLow'] : [])
              ].map((name, location) => ({name, location, group: 0, type: 'storage' as const}))
            }
          });
          return {
            encode: ({computePass, getBuffer}) =>
              kernel.dispatch(computePass, {
                bindings: {
                  positions: getBuffer(positions.buffer),
                  centers: getBuffer(center.buffer),
                  coverage: getBuffer(coverage.buffer),
                  outputPositions: getBuffer(outputPosition.buffer),
                  ...(useLow
                    ? {
                        positionsLow: getBuffer(lowPositions!.buffer),
                        centersLow: getBuffer(centerLow!.buffer),
                        outputPositionsLow: getBuffer(outputLowPosition!.buffer)
                      }
                    : {})
                },
                ...dispatch
              }),
            destroy: () => kernel.destroy()
          };
        }
      });
    }
    return {output, outputLow};
  }
}

function flattenListVector<Format extends 'float32x2' | 'float32x3' | 'float32x4' | 'uint32'>(
  source: GPUVector<VertexList<Format>>,
  format: Format
): GPUVector<Format> {
  return new GPUVector({
    type: 'data',
    name: source.name,
    format,
    ownsData: false,
    data: source.data.map(
      chunk =>
        new GPUData({
          format,
          buffer: chunk.buffer,
          byteOffset: chunk.byteOffset,
          length: chunk.valueLength,
          ownsBuffer: false
        })
    )
  });
}
