// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {GPUTable} from '@luma.gl/graph/gpu-tables';
import {getGPUPartitionMemoryPlan} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  getGPUPartitionDescriptorFromArrowTable,
  type PreparedArrowAnalyticsTable,
  type PrepareArrowTableForGPUAnalyticsOptions
} from '@luma.gl/graph/arrow';
import {DataType, Precision, type Table} from 'apache-arrow';
import type {GPUResidentColumns} from '../exploration/gpu-resident-explorer';
import type {KeplerDatasetRequest, KeplerMemoryReservation} from '../runtime/dataset-registry';
import type {KeplerArrowDataset} from './compute-session';

/** Caller verifies result values against its bounded GPU readback before adoption. No upload occurs. */
export type KeplerAdoptedArrowDatasetOptions = {
  id: string;
  table: Table;
  prepared: PreparedArrowAnalyticsTable;
  columns: GPUResidentColumns;
  validity: Readonly<Record<string, GPUVector<'uint32'> | undefined>>;
  normalization?: PrepareArrowTableForGPUAnalyticsOptions;
  metadata?: Readonly<Record<string, unknown>>;
  request: KeplerDatasetRequest;
  cpuReservation: KeplerMemoryReservation;
  gpuReservation: KeplerMemoryReservation;
  /** Owns dedicated output buffers, not borrowed source chunks. Called only after successful adoption. */
  destroy(): void;
};

/** Structural validation cannot prove GPU values: the producer must retain matching readback Arrow. */
export function makeAdoptedArrowAnalytics(
  device: Device,
  options: KeplerAdoptedArrowDatasetOptions
): KeplerArrowDataset['analytics'] {
  if (device.isLost) throw new Error('Cannot adopt outputs from a lost device');
  const {table: source, prepared, columns, validity} = options;
  const fields = prepared.table.schema.fields;
  const names = fields.map(field => field.name);
  if (
    !options.id ||
    !names.length ||
    new Set(names).size !== names.length ||
    prepared.sourceRowCount !== source.numRows ||
    prepared.table.numRows !== source.numRows ||
    source.schema.fields.length !== fields.length ||
    Object.keys(columns).length !== names.length ||
    Object.keys(validity).some(name => !names.includes(name)) ||
    Object.keys(prepared.timeColumns).length ||
    prepared.sourceBatches.length !== source.batches.length ||
    prepared.table.batches.length !== source.batches.length
  )
    throw new Error('Adopted Arrow schema and source rows must match');
  let sourceOffset = 0;
  for (const [index, batch] of source.batches.entries()) {
    const identity = prepared.sourceBatches[index];
    if (
      identity.sourceBatchIndex !== index ||
      identity.sourceRowIndexOffset !== sourceOffset ||
      identity.sourceRowCount !== batch.numRows ||
      prepared.table.batches[index].numRows !== batch.numRows
    )
      throw new Error('Adoption must preserve result batch boundaries');
    sourceOffset += batch.numRows;
  }
  const buffers = new Set<Buffer>();
  const validateVector = (vector: GPUVector | undefined, format: string): void => {
    if (
      !vector ||
      vector.format !== format ||
      vector.length !== source.numRows ||
      vector.data.length !== source.batches.length
    )
      throw new Error('Adopted GPU columns must match the retained Arrow format and batches');
    for (const [index, chunk] of vector.data.entries()) {
      if (
        chunk.length !== source.batches[index].numRows ||
        !(chunk.buffer instanceof Buffer) ||
        chunk.buffer.device !== device ||
        chunk.buffer.destroyed ||
        chunk.byteStride !== 4 ||
        !(chunk.buffer.usage & Buffer.STORAGE) ||
        !(chunk.buffer.usage & Buffer.COPY_SRC)
      )
        throw new Error('Adopted GPU chunks require readable scalar storage on this device');
      if (buffers.has(chunk.buffer))
        throw new Error('Adopted output columns require disjoint buffers');
      buffers.add(chunk.buffer);
    }
  };
  const nullCounts: KeplerArrowDataset['analytics']['nullCounts'] = {};
  for (const [index, field] of fields.entries()) {
    if (source.schema.fields[index].name !== field.name)
      throw new Error('Adopted field order must match retained source');
    const type = field.type;
    const dictionary = prepared.dictionaries[field.name];
    const format = DataType.isDictionary(type)
      ? 'uint32'
      : DataType.isFloat(type) && type.precision === Precision.SINGLE
        ? 'float32'
        : DataType.isInt(type) && type.bitWidth === 32
          ? type.isSigned
            ? 'sint32'
            : 'uint32'
          : undefined;
    if (!format) throw new Error('Adoption supports prepared portable scalar fields only');
    validateVector(columns[field.name], format);
    const counts = prepared.table.batches.map(batch => batch.getChildAt(index)!.nullCount);
    if ((field.nullable || counts.some(count => count > 0)) && !validity[field.name])
      throw new Error('Nullable adopted columns require explicit GPU validity');
    if (validity[field.name]) validateVector(validity[field.name], 'uint32');
    nullCounts[field.name] = counts;
    if (DataType.isDictionary(type)) {
      if (
        !dictionary ||
        !DataType.isUtf8(type.dictionary) ||
        !DataType.isInt(type.indices) ||
        type.indices.bitWidth !== 32 ||
        type.indices.isSigned
      )
        throw new Error('Adopted dictionary codes must be Uint32 UTF-8 indices');
      for (const batch of prepared.table.batches) {
        const data = batch.getChildAt(index)!.data[0];
        const labels = data.dictionary;
        if (
          !labels ||
          labels.length !== dictionary.values.length ||
          dictionary.values.some((label, labelIndex) => labels.get(labelIndex) !== label)
        )
          throw new Error('Retained dictionary labels must match adopted GPU code metadata');
        const codes = data.values;
        for (let row = 0; row < batch.numRows; row++) {
          const value = batch.getChildAt(index)!.get(row);
          if (value !== null && value !== dictionary.values[codes[data.offset + row]])
            throw new Error('Retained dictionary codes do not match their labels');
        }
      }
    } else if (dictionary)
      throw new Error('Dictionary metadata requires an Arrow dictionary field');
  }
  const borrow = <Format extends 'float32' | 'sint32' | 'uint32'>(
    name: string,
    vector: GPUVector<Format>
  ): GPUVector<Format> =>
    new GPUVector({
      name,
      type: 'data',
      format: vector.format,
      ownsData: true,
      data: vector.data.map(
        chunk =>
          new GPUData({
            format: vector.format,
            buffer: chunk.buffer,
            length: chunk.length,
            byteOffset: chunk.byteOffset,
            ownsBuffer: false
          })
      )
    });
  const borrowed: GPUResidentColumns = Object.fromEntries(
    Object.entries(columns).map(([name, vector]) => [name, borrow(name, vector)])
  );
  const borrowedValidity = Object.fromEntries(
    Object.entries(validity).map(([name, vector]) => [
      name,
      vector ? borrow(name, vector) : undefined
    ])
  );
  const table = new GPUTable<Record<string, 'float32' | 'sint32' | 'uint32'>>({vectors: borrowed});
  for (const schema of [table.schema, ...table.batches.map(batch => batch.schema)]) {
    for (const field of schema.fields) {
      const arrowField = fields.find(candidate => candidate.name === field.name)!;
      field.nullable = arrowField.nullable;
      field.metadata = new Map(arrowField.metadata);
    }
  }
  const partitioning = getGPUPartitionDescriptorFromArrowTable(prepared.table);
  return {
    table,
    validity: borrowedValidity,
    dictionaries: {...prepared.dictionaries},
    nullCounts,
    partitioning,
    partitionMemory: getGPUPartitionMemoryPlan(
      partitioning,
      fields.reduce((bytes, field) => bytes + 4 + (field.nullable ? 4 : 0), 0)
    )
  };
}
