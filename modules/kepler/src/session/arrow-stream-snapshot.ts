// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUData, GPUVector, type GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {GPURecordBatch, GPUTable} from '@luma.gl/graph/gpu-tables';
import {makeGPUAnalyticsTableFromArrowTable} from '@luma.gl/graph/arrow';
import {
  makeGPUTimeVectorFromPreparedArrowColumn,
  type PreparedArrowAnalyticsTable
} from '@luma-contributors/analysis/analysis-core';
import {Data, DataType, RecordBatch, Table} from 'apache-arrow';
import type {KeplerArrowDataset} from './compute-session';
import type {GPUResidentScalarFormat} from '../exploration/gpu-resident-explorer';
import type {KeplerDatasetResource} from '../runtime/dataset-registry';

export type ArrowStreamPart = {
  source: Table;
  prepared: PreparedArrowAnalyticsTable;
  analytics: KeplerArrowDataset['analytics'];
  timeColumns: KeplerArrowDataset['timeColumns'];
  retainedArrayBufferBytes: number;
  uploadedBufferBytes: number;
  resources: readonly KeplerDatasetResource[];
};

/** Upload only this admitted batch; earlier stream parts never enter an upload adapter. */
export function uploadArrowStreamPart(
  device: Device,
  source: Table,
  prepared: PreparedArrowAnalyticsTable
) {
  const analytics: KeplerArrowDataset['analytics'] = prepared.table.schema.fields.length
    ? makeGPUAnalyticsTableFromArrowTable(device, prepared.table)
    : makeGPUAnalyticsTableFromArrowTable(device, source, {
        columns: []
      });
  const timeColumns: Record<string, KeplerArrowDataset['timeColumns'][string]> = {};
  try {
    for (const [name, column] of Object.entries(prepared.timeColumns))
      timeColumns[name] = makeGPUTimeVectorFromPreparedArrowColumn(device, column);
    const buffers = new Set(
      [
        ...Object.values(analytics.table.gpuVectors),
        ...Object.values(analytics.validity),
        ...Object.values(timeColumns).flatMap(column => [column.timestamps, column.validity])
      ].flatMap(vector => vector?.data.map(chunk => chunk.buffer) ?? [])
    );
    return {
      analytics,
      timeColumns,
      uploadedBufferBytes: [...buffers].reduce((sum, buffer) => sum + buffer.byteLength, 0)
    };
  } catch (error) {
    destroyArrowStreamUpload({analytics, timeColumns});
    throw error;
  }
}

export function destroyArrowStreamUpload(
  part: Pick<ArrowStreamPart, 'analytics' | 'timeColumns'>
): void {
  part.analytics.table.destroy();
  for (const vector of Object.values(part.analytics.validity)) vector?.destroy();
  for (const column of Object.values(part.timeColumns)) column.destroy();
}

/** Immutable aggregate metadata/views. Every chunk borrows the batch resource's original buffer. */
export function makeArrowStreamSnapshot(parts: readonly ArrowStreamPart[]): KeplerArrowDataset {
  const first = parts[0];
  const last = parts.at(-1)!;
  const source = new Table(
    first.source.schema,
    parts.flatMap(part => part.source.batches)
  );
  const table = new Table(
    first.prepared.table.schema,
    parts.flatMap(part =>
      part.prepared.table.batches.map(batch => {
        // Rebind dictionary metadata only. Every code/null buffer remains the original chunk.
        // All labels retain their codes, and prior revision Data/dictionaries are never mutated.
        const children = batch.data.children.map((data, index) =>
          DataType.isDictionary(data.type)
            ? new Data(
                data.type,
                data.offset,
                data.length,
                data.nullCount,
                data.buffers,
                data.children,
                last.prepared.table.batches[0].data.children[index].dictionary
              )
            : data
        );
        return new RecordBatch(
          batch.schema,
          new Data(
            batch.data.type,
            batch.data.offset,
            batch.data.length,
            batch.data.nullCount,
            batch.data.buffers,
            children
          )
        );
      })
    )
  );
  let sourceRowOffset = 0;
  const sourceBatches = parts.map((part, sourceBatchIndex) => {
    const value = {
      sourceBatchIndex,
      sourceRowIndexOffset: sourceRowOffset,
      sourceRowCount: part.source.numRows
    };
    sourceRowOffset += part.source.numRows;
    return value;
  });
  const timeColumns: Record<string, KeplerArrowDataset['timeColumns'][string]> = {};
  const preparedTimes: Record<string, PreparedArrowAnalyticsTable['timeColumns'][string]> = {};
  for (const [name, lastColumn] of Object.entries(last.prepared.timeColumns)) {
    const metadata = {
      ...lastColumn,
      chunks: parts.flatMap(part => part.prepared.timeColumns[name].chunks),
      validity: parts.flatMap(part => part.prepared.timeColumns[name].validity)
    };
    preparedTimes[name] = metadata;
    const timestamps = borrowVectors(
      name,
      'uint32x2',
      parts.map(part => part.timeColumns[name].timestamps)
    );
    const validity = borrowVectors(
      `${name}-validity`,
      'uint32',
      parts.map(part => part.timeColumns[name].validity)
    );
    timeColumns[name] = {
      timestamps,
      validity,
      metadata,
      destroy() {
        timestamps.destroy();
        validity.destroy();
      }
    };
  }
  const prepared: PreparedArrowAnalyticsTable = {
    table,
    sourceRowCount: sourceRowOffset,
    dictionaries: last.prepared.dictionaries,
    timeColumns: preparedTimes,
    sourceBatches,
    conversions: first.prepared.conversions.map(conversion => {
      let maximumAbsoluteError = 0;
      let maximumRelativeError = 0;
      for (const part of parts) {
        const current = part.prepared.conversions.find(value => value.name === conversion.name)!;
        maximumAbsoluteError = Math.max(maximumAbsoluteError, current.maximumAbsoluteError);
        maximumRelativeError = Math.max(maximumRelativeError, current.maximumRelativeError);
      }
      return {...conversion, maximumAbsoluteError, maximumRelativeError};
    })
  };
  const batches = parts.map((part, index) => {
    const batch = part.analytics.table.batches[0];
    return new GPURecordBatch<Record<string, GPUResidentScalarFormat>>({
      gpuData: Object.fromEntries(
        Object.entries(batch.gpuData).map(([name, data]) => [name, borrowData(data)])
      ),
      fields: batch.schema.fields,
      bufferLayout: batch.bufferLayout,
      metadata: batch.schema.metadata,
      numRows: sourceBatches[index].sourceRowCount,
      nullCount: batch.nullCount,
      sourceInfo: sourceBatches[index]
    });
  });
  const gpuTable = new GPUTable<Record<string, GPUResidentScalarFormat>>({batches});
  const columns: Record<string, GPUVector<GPUResidentScalarFormat>> = {};
  for (const [name, vector] of Object.entries(gpuTable.gpuVectors)) {
    if (vector.format !== 'float32' && vector.format !== 'sint32' && vector.format !== 'uint32')
      throw new Error('Streaming analytics requires portable scalar storage');
    // Adapter validation above establishes this precise scalar union.
    columns[name] = vector as GPUVector<GPUResidentScalarFormat>;
  }
  const validity: KeplerArrowDataset['analytics']['validity'] = {};
  const nullCounts: KeplerArrowDataset['analytics']['nullCounts'] = {};
  for (const name of Object.keys(first.analytics.validity))
    validity[name] = borrowVectors(
      `${name}-validity`,
      'uint32',
      parts.map(part => part.analytics.validity[name]!)
    );
  for (const name of Object.keys(first.analytics.nullCounts))
    nullCounts[name] = parts.flatMap(part => part.analytics.nullCounts[name] ?? []);
  return {
    source,
    prepared,
    columns,
    timeColumns,
    analytics: {
      table: gpuTable,
      validity,
      dictionaries: last.prepared.dictionaries,
      nullCounts
    },
    retainedArrayBufferBytes: parts.reduce((sum, part) => sum + part.retainedArrayBufferBytes, 0),
    uploadedBufferBytes: parts.reduce((sum, part) => sum + part.uploadedBufferBytes, 0)
  };
}

function borrowData<Format extends GPUVectorFormat>(data: GPUData<Format>): GPUData<Format> {
  return new GPUData({
    format: data.format,
    buffer: data.buffer,
    length: data.length,
    byteOffset: data.byteOffset,
    byteStride: data.byteStride,
    ownsBuffer: false
  });
}
function borrowVectors<Format extends GPUVectorFormat>(
  name: string,
  format: Format,
  vectors: readonly GPUVector<Format>[]
): GPUVector<Format> {
  return new GPUVector({
    name,
    type: 'data',
    format,
    data: vectors.flatMap(vector => vector.data),
    ownsData: false
  });
}
