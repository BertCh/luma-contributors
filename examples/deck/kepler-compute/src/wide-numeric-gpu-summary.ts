// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import * as arrow from 'apache-arrow';
import {describeNumericNarrowingReport} from './analysis2-policy-binding';
import type {WideNumericGpuFormat} from './wide-numeric-gpu-reference';
import {scanWideNumericColumnOnGpu} from './wide-numeric-gpu-narrowing';

function getWideNumericGpuFormat(type: arrow.DataType): WideNumericGpuFormat | undefined {
  if (arrow.DataType.isFloat(type))
    return type.precision === arrow.Precision.DOUBLE ? 'float64' : undefined;
  if (arrow.DataType.isInt(type) && type.bitWidth === 64) return type.isSigned ? 'int64' : 'uint64';
  return undefined;
}

/** Uploads one column's 64-bit words chunk by chunk (no repacking) with an optional validity vector. */
export function makeWideNumericWordVectors(device: Device, column: arrow.Vector) {
  const wordData: GPUData<'uint32x2'>[] = [];
  const validityData: GPUData<'uint32'>[] = [];
  let hasNulls = false;
  for (const chunk of column.data) {
    const values = chunk.values as unknown as ArrayBufferView;
    const words = new Uint32Array(
      values.buffer,
      values.byteOffset + chunk.offset * 8,
      chunk.length * 2
    );
    wordData.push(
      new GPUData({
        format: 'uint32x2',
        length: chunk.length,
        ownsBuffer: true,
        buffer: device.createBuffer({
          data: words.length ? words : new Uint32Array(2),
          usage: Buffer.STORAGE | Buffer.COPY_DST
        })
      })
    );
    const flags = new Uint32Array(Math.max(1, chunk.length)).fill(1);
    if (chunk.nullCount > 0) {
      hasNulls = true;
      const vector = new arrow.Vector([chunk]);
      for (let row = 0; row < chunk.length; row++) flags[row] = vector.get(row) === null ? 0 : 1;
    }
    validityData.push(
      new GPUData({
        format: 'uint32',
        length: chunk.length,
        ownsBuffer: true,
        buffer: device.createBuffer({data: flags, usage: Buffer.STORAGE | Buffer.COPY_DST})
      })
    );
  }
  const make = <Format extends 'uint32x2' | 'uint32'>(format: Format, data: GPUData<Format>[]) =>
    new GPUVector({name: `narrowing-${format}`, type: 'data', format, ownsData: true, data});
  return {
    words: make('uint32x2', wordData),
    validity: hasNulls ? make('uint32', validityData) : undefined,
    unusedValidity: hasNulls ? undefined : make('uint32', validityData)
  };
}

/**
 * One summary line per Float64, Int64 or Uint64 column, scanned over EVERY row on the GPU in a
 * single pass per column (24 bytes read back each). Same wording as the CPU summary with a GPU
 * marker; informational, the preparation policy is unchanged.
 */
export async function describeWideNumericNarrowingOnGpu(
  device: Device,
  table: arrow.Table
): Promise<string[]> {
  const lines: string[] = [];
  for (const field of table.schema.fields) {
    const format = getWideNumericGpuFormat(field.type);
    const column = format ? table.getChild(field.name) : null;
    if (!format || !column || !column.length) continue;
    const vectors = makeWideNumericWordVectors(device, column);
    try {
      const report = await scanWideNumericColumnOnGpu({
        device,
        words: vectors.words,
        format,
        validity: vectors.validity
      });
      lines.push(`Narrowing ${field.name}: ${describeNumericNarrowingReport(report)} (GPU scan)`);
    } finally {
      vectors.words.destroy();
      vectors.validity?.destroy();
      vectors.unusedValidity?.destroy();
    }
  }
  return lines;
}
