// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import type {PreparedArrowAnalyticsTimeColumn} from './arrow-analytics-normalization';

export type GPUTimeVectorFromPreparedArrowColumn = {
  timestamps: GPUVector<'uint32x2'>;
  validity: GPUVector<'uint32'>;
  metadata: PreparedArrowAnalyticsTimeColumn;
  destroy(): void;
};

/** Uploads exact time words and validity independently of dataframe scalar arithmetic. */
export function makeGPUTimeVectorFromPreparedArrowColumn(
  device: Device,
  column: PreparedArrowAnalyticsTimeColumn
): GPUTimeVectorFromPreparedArrowColumn {
  if (
    column.chunks.length !== column.validity.length ||
    column.chunks.some((chunk, index) => chunk.length !== column.validity[index].length * 2)
  ) {
    throw new Error('Time words and validity must preserve identical batch topology');
  }
  const timestamps: GPUData<'uint32x2'>[] = [];
  const validity: GPUData<'uint32'>[] = [];
  try {
    for (let batchIndex = 0; batchIndex < column.chunks.length; batchIndex++) {
      timestamps.push(uploadTimeChunk(device, column.chunks[batchIndex], 'uint32x2', column.name));
      validity.push(
        uploadTimeChunk(device, column.validity[batchIndex], 'uint32', `${column.name}-validity`)
      );
    }
    const timestampVector = new GPUVector<'uint32x2'>({
      name: column.name,
      type: 'data',
      format: 'uint32x2',
      data: timestamps,
      ownsData: true
    });
    const validityVector = new GPUVector<'uint32'>({
      name: `${column.name}-validity`,
      type: 'data',
      format: 'uint32',
      data: validity,
      ownsData: true
    });
    return {
      timestamps: timestampVector,
      validity: validityVector,
      metadata: column,
      destroy() {
        timestampVector.destroy();
        validityVector.destroy();
      }
    };
  } catch (error) {
    for (const data of [...timestamps, ...validity]) data.destroy();
    throw error;
  }
}

function uploadTimeChunk<Format extends 'uint32' | 'uint32x2'>(
  device: Device,
  values: Uint32Array,
  format: Format,
  name: string
): GPUData<Format> {
  const buffer = device.createBuffer({
    id: name,
    data: values.length ? values : new Uint32Array(2),
    usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  try {
    return new GPUData<Format>({
      buffer,
      format,
      length: values.length / (format === 'uint32x2' ? 2 : 1),
      ownsBuffer: true
    });
  } catch (error) {
    buffer.destroy();
    throw error;
  }
}
