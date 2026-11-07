// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readBufferUint32Scalar} from '../runtime/gpu-readback';
import {Buffer, type CommandEncoder, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {
  GPUTimeWindowFilter,
  getGPUTimeWindowFilterWordParameterValues
} from '@luma-contributors/graph/gpu-dataframe';

export type GPUKeplerTimeUnit = 'second' | 'millisecond' | 'microsecond' | 'nanosecond';

/** Integer UTC epoch instants. Numbers must be safe integers; bigint preserves every Int64 tick. */
export type GPUKeplerTimeWindow = {
  start: number | bigint;
  end: number | bigint;
  unit: GPUKeplerTimeUnit;
};

export type GPUKeplerTimeSelectionProps = {
  device: Device;
  id?: string;
  /** Borrowed signed Int64 words (low, high), preserving source rows and physical chunks. */
  timestamps: GPUVector<'uint32x2'>;
  /** Optional inclusive interval endpoints, in the same source unit and chunk topology. */
  endTimestamps?: GPUVector<'uint32x2'>;
  /** Zero rejects a null start/instant. Missing validity means every source row is valid. */
  validity?: GPUVector<'uint32'>;
  /** Zero rejects a null interval endpoint; requires endTimestamps. */
  endValidity?: GPUVector<'uint32'>;
  unit: GPUKeplerTimeUnit;
  window: GPUKeplerTimeWindow;
};

export type GPUKeplerTimeSelectionStatus = {
  revision: number;
  stale: boolean;
  selectedCount: number;
  overflow: number;
  complete: boolean;
};

const INT64_MINIMUM = -(1n << 63n);
const INT64_MAXIMUM = (1n << 63n) - 1n;
const NANOSECONDS_PER_UNIT: Record<GPUKeplerTimeUnit, bigint> = {
  second: 1_000_000_000n,
  millisecond: 1_000_000n,
  microsecond: 1_000n,
  nanosecond: 1n
};

/**
 * Converts epoch ticks without rounding. Finer-to-coarser conversions must divide exactly;
 * unsafe/fractional numbers and values outside the target signed Int64 range are rejected.
 */
export function convertKeplerTimeInstant(
  value: number | bigint,
  sourceUnit: GPUKeplerTimeUnit,
  targetUnit: GPUKeplerTimeUnit
): bigint {
  validateTimeUnit(sourceUnit);
  validateTimeUnit(targetUnit);
  if (typeof value !== 'bigint' && !Number.isSafeInteger(value)) {
    throw new Error('Time instants require bigint or a safe integer');
  }
  const nanoseconds = BigInt(value) * NANOSECONDS_PER_UNIT[sourceUnit];
  const divisor = NANOSECONDS_PER_UNIT[targetUnit];
  if (nanoseconds % divisor !== 0n) throw new Error('Time unit conversion would lose precision');
  const converted = nanoseconds / divisor;
  if (converted < INT64_MINIMUM || converted > INT64_MAXIMUM) {
    throw new Error('Time instant exceeds signed Int64 range');
  }
  return converted;
}

/**
 * Exact inclusive time selection for Kepler external crossfilter dimensions.
 *
 * Updates upload only eight parameter words. Encode before the external-mask consumer; submission
 * is caller-owned. Masks retain source identity and are usable by GPU layers without readback.
 * Clearing expands the time window but continues rejecting nulls. To disable the filter including
 * its null policy, deactivate the consuming external dimension instead.
 *
 * Interval membership follows GPUTimeWindowFilter: end >= window.start AND start <= window.end.
 * Source intervals must have ordered endpoints; callers can mark malformed intervals invalid.
 */
export class GPUKeplerTimeSelection {
  readonly mask: GPUVector<'uint32'>;
  readonly unit: GPUKeplerTimeUnit;
  readonly rowCount: number;
  private readonly device: Device;
  private readonly graph: GPUCommandGraph;
  private readonly sourceTopology: {vector: GPUVector; chunks: readonly GPUData[]}[];
  private readonly ownedBuffers: Buffer[] = [];
  private readonly windowBuffer: Buffer;
  private windowParameters: Uint32Array;
  private readonly countBuffer: Buffer;
  private readonly overflowBuffer: Buffer;
  private compiled?: CompiledGPUCommandGraph;
  private revision = 0;
  private encodedRevision = -1;
  private encodingCount = 0;
  private activeReads = 0;
  private readbackBytes = 0;
  private destroyed = false;
  private released = false;

  constructor(props: GPUKeplerTimeSelectionProps) {
    validateTimeSelectionProps(props);
    const parameters = makeWindowParameters(props.window, props.unit);
    const id = props.id ?? 'kepler-time-selection';
    this.device = props.device;
    this.unit = props.unit;
    this.rowCount = props.timestamps.length;
    this.windowParameters = parameters;
    this.sourceTopology = [
      props.timestamps,
      props.endTimestamps,
      props.validity,
      props.endValidity
    ].flatMap(vector => (vector ? [{vector, chunks: [...vector.data]}] : []));
    this.graph = new GPUCommandGraph(this.device, {id});
    try {
      this.windowBuffer = this.createBuffer(`${id}-window`, parameters);
      this.countBuffer = this.createBuffer(`${id}-count`, new Uint32Array(1));
      this.overflowBuffer = this.createBuffer(`${id}-overflow`, new Uint32Array(1));
      const idsBuffer = this.createBuffer(`${id}-ids`, undefined, this.rowCount * 4);
      this.mask = new GPUVector({
        type: 'data',
        name: `${id}-mask`,
        format: 'uint32',
        ownsData: false,
        data: props.timestamps.data.map(
          (chunk, chunkIndex) =>
            new GPUData({
              buffer: this.createBuffer(`${id}-mask-${chunkIndex}`, undefined, chunk.length * 4),
              format: 'uint32',
              length: chunk.length,
              ownsBuffer: false
            })
        )
      });
      const validity = [props.validity, props.endValidity].flatMap((vector, index) =>
        vector
          ? [
              {
                kind: 'selection' as const,
                mask: this.graph.importGPUVector(`${id}-validity-${index}`, vector)
              }
            ]
          : []
      );
      this.graph.add(
        new GPUTimeWindowFilter({
          id,
          timestamps: this.graph.importGPUVector(`${id}-timestamps`, props.timestamps),
          endTimestamps: props.endTimestamps
            ? this.graph.importGPUVector(`${id}-end-timestamps`, props.endTimestamps)
            : undefined,
          window: this.importUint32(`${id}-window`, this.windowBuffer, parameters.length),
          additionalPredicates: validity,
          output: {
            ids: this.importUint32(`${id}-ids`, idsBuffer, this.rowCount),
            outputMask: this.graph.importGPUVector(`${id}-mask`, this.mask),
            status: {
              count: this.importUint32(`${id}-count`, this.countBuffer, 1),
              overflow: this.importUint32(`${id}-overflow`, this.overflowBuffer, 1)
            }
          }
        })
      );
    } catch (error) {
      for (const buffer of this.ownedBuffers) buffer.destroy();
      throw error;
    }
  }

  /** Converts host units losslessly before writing; rejected updates leave the previous window intact. */
  setWindow(window: GPUKeplerTimeWindow): number {
    this.assertAlive();
    const parameters = makeWindowParameters(window, this.unit);
    this.windowParameters = parameters;
    return ++this.revision;
  }

  /** Includes every valid timestamp/interval; deactivating an external dimension also bypasses nulls. */
  clear(): number {
    return this.setWindow({start: INT64_MINIMUM, end: INT64_MAXIMUM, unit: this.unit});
  }

  /** Call after changing borrowed timestamps or validity to invalidate result revision metadata. */
  invalidate(): number {
    this.assertAlive();
    return ++this.revision;
  }

  /** Compile without submitting work, so a containing adapter can check its admission budget. */
  prepare(): void {
    this.assertAlive();
    this.compiled ??= this.graph.compile();
  }

  encode(commandEncoder: CommandEncoder): number {
    this.assertAlive();
    if (commandEncoder.device !== this.device)
      throw new Error('Time selection encoder device mismatch');
    for (const {vector, chunks} of this.sourceTopology) {
      if (
        vector.length !== this.rowCount ||
        vector.data.length !== chunks.length ||
        chunks.some((chunk, index) => chunk !== vector.data[index])
      ) {
        throw new Error('Time selection source topology changed; recreate the adapter');
      }
    }
    this.prepare();
    // Capture this window in the encoder, including when several windows are recorded before submit.
    this.device.writeBufferViaCommandEncoder(
      commandEncoder,
      this.windowBuffer,
      this.windowParameters
    );
    this.compiled!.encode(commandEncoder, {parameters: undefined});
    this.encodedRevision = this.revision;
    this.encodingCount++;
    return this.encodedRevision;
  }

  /** Explicit two-word diagnostic readback after caller submission; never downloads source rows. */
  async readStatus(): Promise<GPUKeplerTimeSelectionStatus> {
    this.assertAlive();
    if (this.encodedRevision < 0) throw new Error('Encode time selection before reading status');
    const revision = this.encodedRevision;
    const encodingCount = this.encodingCount;
    this.activeReads++;
    this.readbackBytes += 8;
    try {
      const [selectedCount, overflow] = await Promise.all(
        [this.countBuffer, this.overflowBuffer].map(async buffer => {
          return readBufferUint32Scalar(buffer);
        })
      );
      const stale =
        this.destroyed || revision !== this.revision || encodingCount !== this.encodingCount;
      return {revision, stale, selectedCount, overflow, complete: !stale && overflow === 0};
    } finally {
      this.activeReads--;
      this.releaseResources();
    }
  }

  /** Owned outputs/parameters and compiled scratch; excludes borrowed columns and driver allocations. */
  get accountedResidentByteLength(): number {
    if (this.released) return 0;
    return (
      this.ownedBuffers.reduce((total, buffer) => total + buffer.byteLength, 0) +
      (this.compiled?.stats.physicalTransientResourceBytes ?? 0)
    );
  }

  /** Explicit diagnostic bytes requested by readStatus, excluding caller mask readbacks. */
  get readbackByteLength(): number {
    return this.readbackBytes;
  }

  /** Caller must finish dependent GPU work first. Pending diagnostic reads retain owned resources. */
  destroy(): void {
    this.destroyed = true;
    this.releaseResources();
  }

  private releaseResources(): void {
    if (!this.destroyed || this.activeReads || this.released) return;
    this.released = true;
    this.compiled?.destroy();
    for (const buffer of this.ownedBuffers) buffer.destroy();
  }

  private createBuffer(id: string, data?: Uint32Array, byteLength = 4): Buffer {
    const buffer = this.device.createBuffer({
      id,
      ...(data ? {data} : {byteLength: Math.max(4, byteLength)}),
      usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_DST | Buffer.COPY_SRC
    });
    this.ownedBuffers.push(buffer);
    return buffer;
  }

  private importUint32(id: string, buffer: Buffer, length: number): GraphDataView<'uint32'> {
    return this.graph.createDataView(
      this.graph.importBuffer({id, byteLength: buffer.byteLength, usage: buffer.usage}, buffer),
      {
        format: 'uint32',
        length
      }
    );
  }

  private assertAlive(): void {
    if (this.destroyed) throw new Error('Time selection has been destroyed');
  }
}

function makeWindowParameters(window: GPUKeplerTimeWindow, unit: GPUKeplerTimeUnit): Uint32Array {
  return getGPUTimeWindowFilterWordParameterValues({
    start: convertKeplerTimeInstant(window.start, window.unit, unit),
    end: convertKeplerTimeInstant(window.end, window.unit, unit)
  });
}

function validateTimeUnit(unit: GPUKeplerTimeUnit): void {
  if (!Object.hasOwn(NANOSECONDS_PER_UNIT, unit)) throw new Error('Unsupported time unit');
}

function validateTimeSelectionProps(props: GPUKeplerTimeSelectionProps): void {
  validateTimeUnit(props.unit);
  if (props.device.type !== 'webgpu') throw new Error('Time selection requires WebGPU');
  if (props.endValidity && !props.endTimestamps)
    throw new Error('End validity requires interval endpoints');
  for (const [vector, format] of [
    [props.timestamps, 'uint32x2'],
    [props.endTimestamps, 'uint32x2'],
    [props.validity, 'uint32'],
    [props.endValidity, 'uint32']
  ] as const) {
    if (!vector) continue;
    if (vector.format !== format) throw new Error('Time selection input format mismatch');
    if (
      vector.length !== props.timestamps.length ||
      vector.data.length !== props.timestamps.data.length
    ) {
      throw new Error('Time selection source chunks must align');
    }
    for (const [chunkIndex, chunk] of vector.data.entries()) {
      if (chunk.length !== props.timestamps.data[chunkIndex].length)
        throw new Error('Time selection source chunks must align');
      if (
        chunk.buffer.device !== props.device ||
        (chunk.buffer.usage & Buffer.STORAGE) === 0 ||
        chunk.byteStride !== (format === 'uint32x2' ? 8 : 4)
      ) {
        throw new Error('Time selection inputs require packed storage on the same device');
      }
    }
  }
}
