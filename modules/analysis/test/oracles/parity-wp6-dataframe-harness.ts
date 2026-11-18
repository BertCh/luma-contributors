// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/// <reference types="vite/client" />

import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {GPURecordBatch, GPUTable} from '@luma.gl/graph/gpu-tables';
import {GPUDataFrame, column, literal, parameter} from '@luma.gl/graph/gpu-dataframe';
import {decodeOracleFixture} from './oracle-fixture';
import {compareParity} from './parity-wp6-compare';
import fixtureJson from './fixtures/parity-wp6-dataframe-queries.json';

type Payload = {dtype: string; length: number; base64: string};
export type ParityDataFrameBatch = {
  length: number;
  sourceOffset: number;
  category: Payload;
  values: Payload;
  categoryValidity: Payload;
  valueValidity: Payload;
  score: Payload;
  frames: {mask: Payload; ids: Payload}[];
};
export type ParityDataFrameScene = {
  name: string;
  groupCount: number;
  edges: number[];
  domain: [number, number];
  binCount: number;
  batchFiles?: string[];
  batches?: ParityDataFrameBatch[];
  frames: {
    bound: number;
    groups: Record<string, number[]>;
    global: Record<string, number[]>;
    histogram: number[];
    equalHistogram: number[];
  }[];
};
export const dataframeFixture = decodeOracleFixture<{
  reference: string;
  cases: ParityDataFrameScene[];
}>(fixtureJson);
export type ParityDataFrameSchema = {category: 'uint32'; value: 'float32'};

type ParityDataFrameBatchFixture = {batch: ParityDataFrameBatch};

// Vite supplies raw text in both Node and browser Vitest projects. TypeScript sees string
// values instead of building types for 11 MB of JSON. Each scene supplies its own file order;
// lexicographic glob ordering would put batch 10 between batches 1 and 2.
const realBatchSources = import.meta.glob<string>(
  './fixtures/parity-wp6-dataframe-osm-batch-*.json',
  {query: '?raw', import: 'default', eager: true}
);
const parsedBatches = new Map<string, ParityDataFrameBatch>();

/** Lossless little-endian payload decoding, including source 8-bit validity codes. */
export function decodeDataFramePayload(payload: Payload): number[] {
  const bytes = Uint8Array.from(atob(payload.base64), character => character.charCodeAt(0));
  const width = payload.dtype === 'u1' ? 1 : 4;
  if (bytes.length !== width * payload.length) throw new Error('Invalid dataframe fixture payload');
  const view = new DataView(bytes.buffer);
  return Array.from({length: payload.length}, (_, index) =>
    width === 1
      ? view.getUint8(index)
      : payload.dtype === '<f4'
        ? view.getFloat32(index * 4, true)
        : view.getUint32(index * 4, true)
  );
}

export function getDataFrameBatches(scene: ParityDataFrameScene): ParityDataFrameBatch[] {
  if (scene.batches) return scene.batches;
  return scene.batchFiles!.map(fileName => {
    let batch = parsedBatches.get(fileName);
    if (!batch) {
      const fixture: ParityDataFrameBatchFixture = JSON.parse(
        realBatchSources[`./fixtures/${fileName}`]
      );
      batch = fixture.batch;
      parsedBatches.set(fileName, batch);
    }
    return batch;
  });
}

export function createParityDataFrame(device: Device, scene: ParityDataFrameScene) {
  const batches = getDataFrameBatches(scene);
  const categoryValidity: GPUData<'uint32'>[] = [];
  const valueValidity: GPUData<'uint32'>[] = [];
  function createData<Format extends 'float32' | 'uint32'>(
    payload: Payload,
    format: Format
  ): GPUData<Format> {
    const values =
      format === 'float32'
        ? Float32Array.from(decodeDataFramePayload(payload))
        : Uint32Array.from(decodeDataFramePayload(payload));
    const buffer = device.createBuffer({
      byteLength: Math.max(4, values.byteLength),
      usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST,
      ...(values.length ? {data: values} : {})
    });
    return new GPUData({buffer, format, length: values.length, ownsBuffer: true});
  }
  const table = new GPUTable<ParityDataFrameSchema>({
    batches: batches.map((batch, batchIndex) => {
      categoryValidity.push(createData(batch.categoryValidity, 'uint32'));
      valueValidity.push(createData(batch.valueValidity, 'uint32'));
      return new GPURecordBatch<ParityDataFrameSchema>({
        gpuData: {
          category: createData(batch.category, 'uint32'),
          value: createData(batch.values, 'float32')
        },
        fields: [
          {name: 'category', format: 'uint32', nullable: true},
          {name: 'value', format: 'float32', nullable: true}
        ],
        sourceInfo: {
          sourceBatchIndex: batchIndex + 4,
          sourceRowIndexOffset: batch.sourceOffset,
          sourceRowCount: batch.length
        }
      });
    })
  });
  return new GPUDataFrame({
    table,
    validity: {
      category: new GPUVector({
        type: 'data',
        name: 'category-validity',
        format: 'uint32',
        data: categoryValidity,
        ownsData: true
      }),
      value: new GPUVector({
        type: 'data',
        name: 'value-validity',
        format: 'uint32',
        data: valueValidity,
        ownsData: true
      })
    },
    dictionaries: {
      category: {
        values: Array.from({length: scene.groupCount}, (_, index) => `category-${index}`),
        ordered: false
      }
    },
    ownership: 'owned'
  });
}

export function createParityDataFrameQuery(frame: GPUDataFrame<ParityDataFrameSchema>) {
  return frame
    .withColumn('score', column('value').multiply(literal(2)).add(literal(0.5)), {
      format: 'float32'
    })
    .filter(column('category').lessThan(parameter('maximumCategory', 8)));
}

export async function readDataFrameVector(
  vector: GPUVector,
  lengths?: number[]
): Promise<number[][]> {
  return Promise.all(
    vector.data.map(async (chunk, index) => {
      const length = lengths?.[index] ?? chunk.length;
      if (!length) return [];
      const buffer = chunk.buffer instanceof Buffer ? chunk.buffer : chunk.buffer.buffer;
      const byteOffset = chunk.byteOffset;
      const bytes = await buffer.readAsync(byteOffset, length * 4);
      return Array.from(
        chunk.format === 'float32'
          ? new Float32Array(bytes.buffer, bytes.byteOffset, length)
          : new Uint32Array(bytes.buffer, bytes.byteOffset, length)
      );
    })
  );
}

export async function compareDataFrameSelection(
  className: string,
  scene: ParityDataFrameScene,
  frameIndex: number,
  encoding: number,
  compiled: {selectionMask: GPUVector; selectedCounts: GPUVector; rowIndices: GPUVector}
): Promise<void> {
  const batches = getDataFrameBatches(scene);
  const counts = (await readDataFrameVector(compiled.selectedCounts)).flat();
  const masks = await readDataFrameVector(compiled.selectionMask);
  const ids = await readDataFrameVector(compiled.rowIndices, counts);
  for (const [batchIndex, batch] of batches.entries()) {
    const expected = batch.frames[frameIndex];
    const label = `${scene.name}/frame=${frameIndex}/encoding=${encoding}/batch=${batchIndex}`;
    compareParity(
      className,
      label,
      dataframeFixture.reference,
      batch.length,
      'selectedCount',
      [counts[batchIndex]],
      [expected.ids.length],
      true
    );
    compareParity(
      className,
      label,
      dataframeFixture.reference,
      batch.length,
      'selectionMask',
      masks[batchIndex],
      decodeDataFramePayload(expected.mask),
      true
    );
    compareParity(
      className,
      label,
      dataframeFixture.reference,
      batch.length,
      'sourceRowIds',
      ids[batchIndex],
      decodeDataFramePayload(expected.ids),
      true
    );
  }
}
