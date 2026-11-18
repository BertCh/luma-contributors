// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readBufferUint32Scalar} from '../runtime/gpu-readback';
import {Buffer, type CommandEncoder, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {GPUCrossfilterSelection} from '@luma.gl/graph/gpu-crossfilter';
import {
  GPUPointInPolygonJoin,
  type GPUPointInPolygonBoundaryRule
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  prepareKeplerSpatialSelection,
  type KeplerSelectionCoordinateSpace,
  type KeplerSpatialSelection
} from './prepare-spatial-selection';

/** Bytes of the private five-word control state owned by each published `GPUCrossfilterSelection`. */
const SELECTION_CONTROL_BYTE_LENGTH = 20;

export type GPUKeplerSpatialSelectionProps = {
  device: Device;
  id?: string;
  /** Borrowed point positions. Original physical chunks and source row order are preserved. */
  positions: GPUVector<'float32x2'>;
  /** Optional borrowed destination with the same chunk topology; enables stable masks on rebuild. */
  outputMask?: GPUVector<'uint32'>;
  selection: KeplerSpatialSelection;
  /** Linear planar edges, or prepared longitude/latitude edges in degrees. */
  coordinateSpace?: KeplerSelectionCoordinateSpace;
  /** Defaults to closed: both outer and hole boundaries count as selected. */
  boundaryRule?: GPUPointInPolygonBoundaryRule;
  /** Called before owned buffers and planned graph scratch are allocated. Throws to reject growth.
   * Covers total owned/planned bytes for this selection, excluding borrowed inputs and driver state. */
  admitMemory?: (byteLength: number) => void;
};

/** Explicitly read completeness diagnostics, aligned with an encoded geometry revision. */
export type GPUKeplerSpatialSelectionStatus = {
  revision: number;
  stale: boolean;
  complete: boolean;
  overflow: number;
  uncertainCount: number;
  candidateCount: number;
};

/**
 * Drawn geometry -> GPU point selection without reading points or row indices back to the CPU.
 *
 * Reuses GPUPointInPolygonJoin for containment (including holes and multipolygon union), then
 * GPUCrossfilterSelection for source-aligned flags. All drawn parts belong to ONE feature; there
 * is at most one candidate per input point, so the candidate buffer covers the full point set.
 *
 * Encode before an explorer's external mask consumer on the same command encoder. Submission is
 * caller-owned. Unless outputMask is supplied, the mask is owned here and borrowed by consumers;
 * release it only after their GPU work finishes. Empty chunks stay in that public mask but are
 * omitted from internal selection bindings. Each encoding captures its drawn vertices. Changing topology
 * requires a new adapter, which can reuse a caller-owned outputMask after prior work finishes.
 * Call invalidate() after changing source point contents so asynchronous diagnostics remain versioned.
 */
export class GPUKeplerSpatialSelection {
  readonly mask: GPUVector<'uint32'>;
  readonly coordinateSpace: KeplerSelectionCoordinateSpace;
  readonly boundaryRule: GPUPointInPolygonBoundaryRule;
  private readonly device: Device;
  private readonly positions: GPUVector<'float32x2'>;
  private readonly positionChunks: readonly GPUData<'float32x2'>[];
  private readonly maskChunks: readonly GPUData<'uint32'>[];
  private readonly rowCount: number;
  private readonly graph: GPUCommandGraph;
  private readonly ownedBuffers: Buffer[] = [];
  private readonly topologyKey: string;
  private readonly geometryBuffer: Buffer;
  private geometryParameters: Float32Array;
  private readonly statusBuffers: {
    overflow: Buffer;
    uncertainCount: Buffer;
    candidateCount: Buffer;
  };
  private maskSelection?: GPUCrossfilterSelection;
  private compiled?: CompiledGPUCommandGraph;
  private plannedTransientByteLength = 0;
  private revision = 0;
  private encodedRevision = -1;
  private encodingCount = 0;
  private activeReads = 0;
  private destroyed = false;
  private released = false;

  constructor(props: GPUKeplerSpatialSelectionProps) {
    const id = props.id ?? 'kepler-spatial-selection';
    this.device = props.device;
    this.positions = props.positions;
    this.positionChunks = [...props.positions.data];
    this.rowCount = props.positions.length;
    this.coordinateSpace = props.coordinateSpace ?? 'planar';
    this.boundaryRule = props.boundaryRule ?? 'closed';
    if (!['closed', 'open', 'half-open'].includes(this.boundaryRule)) {
      throw new Error('Unsupported selection boundary rule');
    }
    if (props.device.type !== 'webgpu') throw new Error('Spatial selection requires WebGPU');
    if (props.positions.format !== 'float32x2') {
      throw new Error('Spatial selection requires Float32 point positions');
    }
    for (const chunk of props.positions.data) {
      if (chunk.buffer.device !== props.device || chunk.byteStride !== 8) {
        throw new Error('Spatial selection positions must be packed on the same device');
      }
    }
    if (props.outputMask) {
      const mask = props.outputMask;
      if (
        mask.format !== 'uint32' ||
        mask.length !== props.positions.length ||
        mask.data.length !== props.positions.data.length
      ) {
        throw new Error('Spatial selection output mask must match point chunk topology');
      }
      const sourceBuffers = new Set(
        props.positions.data.map(chunk =>
          chunk.buffer instanceof Buffer ? chunk.buffer : chunk.buffer.buffer
        )
      );
      for (const [chunkIndex, chunk] of mask.data.entries()) {
        const buffer = chunk.buffer instanceof Buffer ? chunk.buffer : chunk.buffer.buffer;
        if (
          chunk.length !== props.positions.data[chunkIndex].length ||
          chunk.byteStride !== 4 ||
          buffer.device !== props.device ||
          (buffer.usage & Buffer.STORAGE) === 0 ||
          sourceBuffers.has(buffer)
        ) {
          throw new Error('Spatial selection output mask requires separate matching storage');
        }
        for (const previous of mask.data.slice(0, chunkIndex)) {
          const previousBuffer =
            previous.buffer instanceof Buffer ? previous.buffer : previous.buffer.buffer;
          if (
            buffer === previousBuffer &&
            chunk.length > 0 &&
            previous.length > 0 &&
            chunk.byteOffset < previous.byteOffset + previous.length * 4 &&
            previous.byteOffset < chunk.byteOffset + chunk.length * 4
          ) {
            throw new Error('Spatial selection output mask chunks must not overlap');
          }
        }
      }
    }
    const geometry = prepareKeplerSpatialSelection(props.selection, this.coordinateSpace);
    this.topologyKey = geometry.topologyKey;
    this.geometryParameters = geometry.positions;
    this.graph = new GPUCommandGraph(props.device, {id});

    try {
      // The range selector owns five uint32 control words in addition to the explicit buffers.
      const ownedByteLength =
        geometry.positions.byteLength +
        geometry.featureOffsets.byteLength +
        geometry.polygonOffsets.byteLength +
        geometry.ringOffsets.byteLength +
        props.positions.data.reduce(
          (bytes, chunk) => bytes + Math.max(4, chunk.length * 4) * (props.outputMask ? 1 : 2),
          0
        ) +
        12 +
        20;
      props.admitMemory?.(ownedByteLength);
      this.geometryBuffer = this.createBuffer(`${id}-vertices`, geometry.positions);
      const featureOffsets = this.importUint32(`${id}-features`, geometry.featureOffsets);
      const polygonOffsets = this.importUint32(`${id}-polygons`, geometry.polygonOffsets);
      const ringOffsets = this.importUint32(`${id}-rings`, geometry.ringOffsets);
      const geometryPositions = this.graph.createDataView(
        this.graph.importBuffer(
          {
            id: `${id}-vertices`,
            byteLength: this.geometryBuffer.byteLength,
            usage: this.geometryBuffer.usage
          },
          this.geometryBuffer
        ),
        {format: 'float32x2', length: geometry.positions.length / 2}
      );
      this.mask =
        props.outputMask ??
        new GPUVector({
          type: 'data',
          name: `${id}-mask`,
          format: 'uint32',
          data: props.positions.data.map((chunk, chunkIndex) => {
            const buffer = this.createBuffer(
              `${id}-mask-${chunkIndex}`,
              undefined,
              chunk.length * 4
            );
            return new GPUData({buffer, format: 'uint32', length: chunk.length, ownsBuffer: false});
          }),
          ownsData: false
        });
      this.maskChunks = [...this.mask.data];
      const featureChunks = props.positions.data.map((chunk, chunkIndex) => {
        const buffer = this.createBuffer(
          `${id}-feature-ids-${chunkIndex}`,
          undefined,
          chunk.length * 4
        );
        return new GPUData({buffer, format: 'uint32', length: chunk.length, ownsBuffer: false});
      });
      const featureIds = new GPUVector({
        type: 'data',
        name: `${id}-feature-ids`,
        format: 'uint32',
        data: featureChunks,
        ownsData: false
      });
      this.statusBuffers = {
        overflow: this.createBuffer(`${id}-overflow`, new Uint32Array(1)),
        uncertainCount: this.createBuffer(`${id}-uncertain`, new Uint32Array(1)),
        candidateCount: this.createBuffer(`${id}-candidates`, new Uint32Array(1))
      };
      const pointFeatureIds = this.importSelectionVector(`${id}-feature-ids`, featureIds);
      this.graph.add(
        new GPUPointInPolygonJoin({
          id: `${id}-containment`,
          points: {
            kind: 'points',
            positions: this.importSelectionVector(`${id}-points`, props.positions)
          },
          polygons: {
            kind: 'polygons',
            positions: geometryPositions,
            featureOffsets,
            polygonOffsets,
            ringOffsets
          },
          candidateCapacity: Math.max(1, props.positions.length),
          leafCapacity: 1,
          boundaryRule: this.boundaryRule,
          spatialContext:
            this.coordinateSpace === 'longitude-latitude'
              ? {coordinateSpace: 'longitude-latitude', metric: 'none', edgeModel: 'linear'}
              : {coordinateSpace: 'planar', metric: 'native'},
          spatialSort: false,
          querySort: false,
          output: {
            pointFeatureIds,
            status: {
              overflow: this.importStatus(`${id}-overflow`, this.statusBuffers.overflow),
              uncertainCount: this.importStatus(
                `${id}-uncertain`,
                this.statusBuffers.uncertainCount
              ),
              candidateCount: this.importStatus(
                `${id}-candidates`,
                this.statusBuffers.candidateCount
              )
            }
          }
        })
      );
      this.maskSelection = new GPUCrossfilterSelection(this.graph, {
        id: `${id}-selected`,
        kind: 'range',
        input: pointFeatureIds,
        mask: this.importSelectionVector(`${id}-mask`, this.mask)
      });
      this.maskSelection.setRange([0, 0]);
      this.maskSelection.addToGraph(this.graph);
      // Published graphs report transient bytes only after compilation, so scratch is counted
      // from the first encode and admission here covers owned buffers only.
      props.admitMemory?.(this.workingMemoryByteLength);
    } catch (error) {
      this.maskSelection?.destroy();
      for (const buffer of this.ownedBuffers) buffer.destroy();
      throw error;
    }
  }

  /** Owned buffers and planned/compiled scratch, excluding borrowed points/masks and driver allocations.
   * Per-encode upload staging is owned by the caller's command encoder and is not counted here.
   * Reading the count never compiles or allocates GPU resources.
   */
  get workingMemoryByteLength(): number {
    if (this.released) return 0;
    return (
      [...new Set(this.ownedBuffers)].reduce((total, buffer) => total + buffer.byteLength, 0) +
      (this.maskSelection ? SELECTION_CONTROL_BYTE_LENGTH : 0) +
      this.plannedTransientByteLength
    );
  }

  /** Stores drawn vertices for the next encode; topology changes leave the old selection untouched. */
  updateSelection(selection: KeplerSpatialSelection): number {
    this.assertAlive();
    const geometry = prepareKeplerSpatialSelection(selection, this.coordinateSpace);
    if (geometry.topologyKey !== this.topologyKey) {
      throw new Error('Spatial selection topology changed; recreate the selection adapter');
    }
    this.geometryParameters = geometry.positions;
    return ++this.revision;
  }

  /** Marks caller-written point contents as a new revision without uploading or reading rows. */
  invalidate(): number {
    this.assertAlive();
    return ++this.revision;
  }

  /** Encodes on the caller's device; no queue submission or readback occurs here. */
  encode(commandEncoder: CommandEncoder): number {
    this.assertAlive();
    if (
      this.positions.length !== this.rowCount ||
      this.positions.data.length !== this.positionChunks.length ||
      this.positions.data.some((chunk, index) => chunk !== this.positionChunks[index]) ||
      this.mask.length !== this.rowCount ||
      this.mask.data.length !== this.maskChunks.length ||
      this.mask.data.some((chunk, index) => chunk !== this.maskChunks[index])
    ) {
      throw new Error('Spatial selection source topology changed; recreate the selection adapter');
    }
    if (commandEncoder.device !== this.device) {
      throw new Error('Spatial selection encoder belongs to another device');
    }
    if (!this.compiled) {
      this.compiled = this.graph.compile();
      this.plannedTransientByteLength = this.compiled.stats.physicalTransientResourceBytes;
    }
    // Each recorded pass retains its geometry even if later updates precede queue submission.
    this.device.writeBufferViaCommandEncoder(
      commandEncoder,
      this.geometryBuffer,
      this.geometryParameters
    );
    this.compiled.encode(commandEncoder, {parameters: undefined});
    this.encodedRevision = this.revision;
    this.encodingCount++;
    return this.encodedRevision;
  }

  /**
   * Reads three diagnostic words after caller submission. A stale or incomplete status cannot
   * certify an analysis result; invalid/uncertain pairs are excluded by the underlying join.
   */
  async readStatus(): Promise<GPUKeplerSpatialSelectionStatus> {
    this.assertAlive();
    if (this.encodedRevision < 0) throw new Error('Encode spatial selection before reading status');
    const revision = this.encodedRevision;
    const encodingCount = this.encodingCount;
    this.activeReads++;
    try {
      // Wait for every mapping before retiring resources, even if one diagnostic read rejects.
      const results = await Promise.allSettled(
        [
          this.statusBuffers.overflow,
          this.statusBuffers.uncertainCount,
          this.statusBuffers.candidateCount
        ].map(async buffer => {
          return readBufferUint32Scalar(buffer);
        })
      );
      const [overflow, uncertainCount, candidateCount] = results.map(result => {
        if (result.status === 'rejected') throw result.reason;
        return result.value;
      });
      const stale =
        this.destroyed || revision !== this.revision || encodingCount !== this.encodingCount;
      return {
        revision,
        stale,
        complete: !stale && overflow === 0 && uncertainCount === 0,
        overflow,
        uncertainCount,
        candidateCount
      };
    } finally {
      this.activeReads--;
      this.releaseResources();
    }
  }

  /** Caller must finish dependent GPU work first. Pending diagnostics retain owned resources. */
  destroy(): void {
    this.destroyed = true;
    this.releaseResources();
  }

  private releaseResources(): void {
    if (!this.destroyed || this.activeReads || this.released) return;
    this.released = true;
    this.compiled?.destroy();
    this.maskSelection?.destroy();
    for (const buffer of this.ownedBuffers) buffer.destroy();
  }

  private createBuffer(id: string, data?: Float32Array | Uint32Array, byteLength = 4): Buffer {
    const buffer = this.device.createBuffer({
      id,
      ...(data ? {data} : {byteLength: Math.max(4, byteLength)}),
      usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    this.ownedBuffers.push(buffer);
    return buffer;
  }

  private importUint32(id: string, values: Uint32Array): GraphDataView<'uint32'> {
    const buffer = this.createBuffer(id, values);
    return this.graph.createDataView(
      this.graph.importBuffer({id, byteLength: buffer.byteLength, usage: buffer.usage}, buffer),
      {
        format: 'uint32',
        length: values.length
      }
    );
  }

  private importSelectionVector<Format extends 'float32x2' | 'uint32'>(
    id: string,
    vector: GPUVector<Format>
  ): GraphVectorView<Format> {
    // Empty upload chunks may have only a four-byte placeholder, too short for a bound XY row.
    // They contribute no rows: omit them consistently without copying buffers or changing the
    // source-row offsets. Preserve metadata and malformed nonzero lengths for graph validation.
    return this.graph.importGPUVector(id, {
      ...vector,
      data: vector.data.filter(chunk => chunk.length !== 0)
    });
  }

  private importStatus(id: string, buffer: Buffer): GraphDataView<'uint32'> {
    return this.graph.createDataView(
      this.graph.importBuffer({id, byteLength: buffer.byteLength, usage: buffer.usage}, buffer),
      {format: 'uint32', length: 1}
    );
  }

  private assertAlive(): void {
    if (this.destroyed) throw new Error('Spatial selection has been destroyed');
  }
}
