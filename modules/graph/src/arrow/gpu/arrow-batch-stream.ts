// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {type Device} from '@luma.gl/core';
import {GPURecordBatch, GPUTable} from '@luma-contributors/graph/gpu-tables';
import {
  Binary,
  Bool,
  Data,
  DataType,
  Date_,
  Dictionary,
  Field,
  FixedSizeBinary,
  FixedSizeList,
  Float16,
  Float32,
  Float64,
  Int16,
  Int32,
  Int64,
  Int8,
  List,
  RecordBatch,
  Schema,
  Struct,
  Time,
  Timestamp,
  Type,
  Uint16,
  Uint32,
  Uint64,
  Uint8,
  Utf8,
  Vector
} from 'apache-arrow';
import {
  convertArrowRecordBatchesToGPURecordBatches,
  type ArrowGPUTableConversionColumn,
  type PreparedArrowGPURecordBatch
} from './arrow-gpu-table-conversion';

/** Structural Arrow type shape shared by every installed apache-arrow version. */
export type StructuralArrowType = {
  readonly typeId: number;
  readonly children?: readonly StructuralArrowField[];
  readonly bitWidth?: number;
  readonly isSigned?: boolean;
  readonly precision?: number;
  readonly byteWidth?: number;
  readonly unit?: number;
  readonly timezone?: string | null;
  readonly listSize?: number;
  readonly dictionary?: StructuralArrowType;
  readonly indices?: StructuralArrowType;
  readonly id?: number;
  readonly isOrdered?: boolean;
};

/** Structural Arrow field shape. */
export type StructuralArrowField = {
  readonly name: string;
  readonly type: StructuralArrowType;
  readonly nullable: boolean;
  readonly metadata?: ReadonlyMap<string, string>;
};

/** Structural Arrow Data chunk shape. */
export type StructuralArrowData = {
  readonly type: StructuralArrowType;
  readonly offset: number;
  readonly length: number;
  readonly nullCount: number;
  readonly buffers: readonly unknown[];
  readonly children: readonly StructuralArrowData[];
  readonly dictionary?: {readonly data: readonly StructuralArrowData[]};
};

/** Structural Arrow record batch: a struct Data chunk plus schema, from any apache-arrow version. */
export type StructuralArrowRecordBatch = {
  readonly numRows: number;
  readonly schema: {
    readonly fields: readonly StructuralArrowField[];
    readonly metadata?: ReadonlyMap<string, string>;
  };
  readonly data: StructuralArrowData;
};

/** Structural Arrow table: ordered record batches. */
export type StructuralArrowTable = {
  readonly batches: readonly StructuralArrowRecordBatch[];
};

/**
 * Loaders.gl-style batch (`shape: 'arrow-table'`) or a raw Arrow table or record batch.
 * No loaders.gl or matching apache-arrow version is required.
 */
export type StructuralArrowTableBatch =
  | StructuralArrowRecordBatch
  | StructuralArrowTable
  | {
      readonly shape?: string;
      readonly batchType?: string;
      readonly length?: number;
      readonly data?: StructuralArrowRecordBatch | StructuralArrowTable | null;
    };

/** Options for converting a stream of loaders-style Arrow batches into GPU record batches. */
export type ArrowBatchStreamOptions = {
  /** Columns to upload. Defaults to every top-level field of the first non-empty batch. */
  columns?: readonly ArrowGPUTableConversionColumn[];
  /** Stops pending reads and further conversion. */
  signal?: AbortSignal;
};

/** Options for appending a stream of loaders-style Arrow batches to a GPU table. */
export type AppendGPUTableFromArrowBatchesOptions = ArrowBatchStreamOptions & {
  /** Existing table that receives the new batches. A new table is created when omitted. */
  table?: GPUTable;
};

/**
 * Returns the non-empty Arrow record batches carried by one loaders-style batch, in source order.
 * Metadata, schema-only and empty batches yield nothing.
 */
export function getRecordBatchesFromArrowTableBatch(
  batch: StructuralArrowTableBatch
): StructuralArrowRecordBatch[] {
  const wrapper = batch as {batchType?: string; data?: unknown; shape?: string};
  if (wrapper.batchType === 'metadata' || wrapper.batchType === 'schema') {
    return [];
  }
  // Loaders.gl wrappers carry `shape` and `data`; raw Arrow tables and record batches do not.
  const source = ('shape' in batch && 'data' in batch ? batch.data : batch) as
    | StructuralArrowRecordBatch
    | StructuralArrowTable
    | null;
  if (!source) {
    return [];
  }
  const recordBatches =
    'batches' in source ? source.batches : 'numRows' in source ? [source] : ([] as never[]);
  return recordBatches.filter(recordBatch => recordBatch.numRows > 0);
}

/**
 * Converts a stream of loaders-style Arrow batches into GPU record batches, one per source Arrow
 * record batch. Batch boundaries are preserved; nothing is combined or repacked. Consumers own
 * each yielded batch and should destroy it, or hand it to a `GPUTable`.
 */
export async function* iterateGPURecordBatchesFromArrowBatches(
  device: Device,
  batches: AsyncIterable<StructuralArrowTableBatch> | Iterable<StructuralArrowTableBatch>,
  options: ArrowBatchStreamOptions = {}
): AsyncGenerator<PreparedArrowGPURecordBatch> {
  async function* iterateLocalRecordBatches(): AsyncGenerator<RecordBatch> {
    for await (const batch of batches) {
      options.signal?.throwIfAborted();
      for (const recordBatch of getRecordBatchesFromArrowTableBatch(batch)) {
        yield makeLocalRecordBatch(recordBatch);
      }
    }
  }

  const localBatches = iterateLocalRecordBatches();
  const first = await localBatches.next();
  if (first.done) {
    return;
  }
  const columns = options.columns ?? first.value.schema.fields.map(field => ({name: field.name}));

  async function* replayLocalRecordBatches(): AsyncGenerator<RecordBatch> {
    yield first.value as RecordBatch;
    yield* localBatches;
  }

  yield* convertArrowRecordBatchesToGPURecordBatches(
    device,
    replayLocalRecordBatches(),
    {columns},
    {signal: options.signal}
  );
}

/**
 * Appends every non-empty Arrow record batch from a loaders-style batch stream to a `GPUTable`.
 * Each source record batch becomes one `GPURecordBatch` with one `GPUData` chunk per column.
 * Returns the table (the supplied one, or a new one; undefined when the stream had no rows).
 * On failure, batches added by this call are removed from a new table by destroying it, and
 * unadded batches are destroyed; a caller-supplied table keeps its previously added batches.
 */
export async function appendGPUTableFromArrowBatches(
  device: Device,
  batches: AsyncIterable<StructuralArrowTableBatch> | Iterable<StructuralArrowTableBatch>,
  options: AppendGPUTableFromArrowBatchesOptions = {}
): Promise<GPUTable | undefined> {
  const {table: existingTable, ...streamOptions} = options;
  let table = existingTable;
  let sourceBatchIndex = existingTable?.batches.length ?? 0;
  let sourceRowIndexOffset = existingTable?.numRows ?? 0;
  const ownedBatches: GPURecordBatch[] = [];
  try {
    for await (const prepared of iterateGPURecordBatchesFromArrowBatches(
      device,
      batches,
      streamOptions
    )) {
      const batch = prepared.batch;
      try {
        Object.assign(batch.sourceInfo!, {
          sourceBatchIndex,
          sourceRowIndexOffset,
          sourceRowCount: batch.numRows
        });
        if (table) {
          table.addBatch(batch);
        } else {
          table = new GPUTable({batches: [batch]});
        }
      } catch (error) {
        batch.destroy();
        throw error;
      }
      ownedBatches.push(batch);
      sourceBatchIndex++;
      sourceRowIndexOffset += batch.numRows;
    }
  } catch (error) {
    if (!existingTable) {
      table?.destroy();
    } else {
      for (const batch of ownedBatches) {
        const index = existingTable.batches.indexOf(batch);
        if (index >= 0) existingTable.batches.splice(index, 1);
        batch.destroy();
      }
      existingTable.refreshFromBatches();
    }
    throw error;
  }
  return table;
}

/** Rebuilds a structural (possibly foreign-version) record batch as a zero-copy local one. */
function makeLocalRecordBatch(recordBatch: StructuralArrowRecordBatch): RecordBatch {
  const fields = recordBatch.schema.fields.map(makeLocalField);
  const schema = new Schema(fields, new Map(recordBatch.schema.metadata ?? []));
  return new RecordBatch(schema, makeLocalData(recordBatch.data) as Data<Struct>);
}

function makeLocalField(field: StructuralArrowField): Field {
  return new Field(field.name, makeLocalType(field.type), field.nullable, new Map(field.metadata));
}

function makeLocalData(data: StructuralArrowData): Data {
  return new Data(
    makeLocalType(data.type),
    data.offset,
    data.length,
    data.nullCount,
    data.buffers as never,
    data.children.map(makeLocalData),
    data.dictionary ? new Vector(data.dictionary.data.map(makeLocalData)) : undefined
  );
}

const INT_TYPES = {
  '8:true': Int8,
  '16:true': Int16,
  '32:true': Int32,
  '64:true': Int64,
  '8:false': Uint8,
  '16:false': Uint16,
  '32:false': Uint32,
  '64:false': Uint64
} as const;

/** Maps an Arrow type to the locally installed apache-arrow classes by `typeId` and properties. */
function makeLocalType(type: StructuralArrowType): DataType {
  const children = type.children ?? [];
  switch (type.typeId) {
    case Type.Int:
      return new INT_TYPES[`${type.bitWidth as 8}:${type.isSigned as true}`]();
    case Type.Float:
      return [new Float16(), new Float32(), new Float64()][type.precision as number];
    case Type.Bool:
      return new Bool();
    case Type.Utf8:
      return new Utf8();
    case Type.Binary:
      return new Binary();
    case Type.FixedSizeBinary:
      return new FixedSizeBinary(type.byteWidth as number);
    case Type.Date:
      return new Date_(type.unit as number);
    case Type.Time:
      return new Time(type.unit as number, type.bitWidth as 32);
    case Type.Timestamp:
      return new Timestamp(type.unit as number, type.timezone as string | null);
    case Type.List:
      return new List(makeLocalField(children[0]));
    case Type.FixedSizeList:
      return new FixedSizeList(type.listSize as number, makeLocalField(children[0]));
    case Type.Struct:
      return new Struct(children.map(makeLocalField));
    case Type.Dictionary:
      return new Dictionary(
        makeLocalType(type.dictionary!),
        makeLocalType(type.indices!) as never,
        type.id as number,
        type.isOrdered as boolean
      );
    default:
      // Unsupported Arrow type for structural loaders.gl batches; convert it to a supported type upstream.
      throw new Error(`Unsupported structural Arrow type id ${type.typeId}`);
  }
}
