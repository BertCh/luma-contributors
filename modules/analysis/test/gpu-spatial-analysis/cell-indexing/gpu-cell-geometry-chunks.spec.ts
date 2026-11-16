// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getCellChunkProbeRows,
  fillCellChunkOutput,
  expectCellChunkOutputRecords
} from '../cell-topology/gpu-cell-chunks.test-utils';
import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUCellGeometry} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-cell-geometry';
import {
  createInputBuffer,
  createOutputBuffer,
  createVectorView,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

/** Closed-form zoom-one Web Mercator tile corners, followed by the complete zero padding. */
function getQuadkeyBoundaryRecord(key: bigint, stride: number): number[] {
  const digit = Number(key & 3n);
  const northLatitude = (Math.atan(Math.sinh(Math.PI)) * 180) / Math.PI;
  const west = (digit & 1) === 0 ? -180 : 0;
  const north = (digit & 2) === 0 ? northLatitude : 0;
  const south = (digit & 2) === 0 ? 0 : -northLatitude;
  return [
    west,
    north,
    west + 180,
    north,
    west + 180,
    south,
    west,
    south,
    ...new Array((stride - 4) * 2).fill(0)
  ];
}

async function runGeometry(
  device: Device,
  keys: bigint[],
  lengths: number[],
  wordOrder: 'little-endian' | 'high-low'
) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let rowBase = 0;
  const inputs = lengths.map((length, chunkIndex) => {
    const prefix = [1, 65, 3][chunkIndex % 3];
    const values = new Uint32Array(prefix + Math.max(length * 2, 2) + 1).fill(99);
    keys.slice(rowBase, rowBase + length).forEach((key, row) => {
      const words = [Number(key & 0xffffffffn), Number(key >> 32n)];
      values.set(wordOrder === 'high-low' ? words.reverse() : words, prefix + row * 2);
    });
    rowBase += length;
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    const backing = importGraphBuffer(
      graph,
      `input-${chunkIndex}`,
      buffer,
      'uint32',
      values.length
    );
    return graph.createDataView(backing.buffer, {
      format: 'uint32x2',
      length,
      byteOffset: prefix * 4
    });
  });
  const outputs = lengths.map((length, chunkIndex) => {
    const prefix = [3, 1, 65][chunkIndex % 3];
    const createField = <Format extends 'float32x2' | 'uint32'>(
      name: string,
      format: Format,
      entries: number
    ) => {
      const componentCount = format === 'float32x2' ? 2 : 1;
      const buffer = createOutputBuffer(
        device,
        prefix + Math.max(entries * componentCount, componentCount) + 1
      );
      buffer.write(
        format === 'float32x2'
          ? new Float32Array(prefix + Math.max(entries * componentCount, componentCount) + 1).fill(
              -999
            )
          : new Uint32Array(prefix + Math.max(entries, 1) + 1).fill(99)
      );
      buffers.push(buffer);
      const backing = importGraphBuffer(
        graph,
        `${name}-${chunkIndex}`,
        buffer,
        'uint32',
        prefix + Math.max(entries * componentCount, componentCount) + 1
      );
      return {
        buffer,
        prefix,
        entries,
        componentCount,
        format,
        view: graph.createDataView(backing.buffer, {
          format,
          length: entries,
          byteOffset: prefix * 4
        })
      };
    };
    return {
      centers: createField('centers', 'float32x2', length),
      boundaries: createField('boundaries', 'float32x2', length * 6),
      vertexCounts: createField('vertexCounts', 'uint32', length),
      validity: createField('validity', 'uint32', length)
    };
  });
  const wrap = <Format extends 'float32x2' | 'uint32' | 'uint32x2'>(
    name: string,
    chunks: GraphDataView<Format>[]
  ) => (chunks.length === 1 ? chunks[0] : createVectorView(name, chunks[0].format, chunks));
  graph.add(
    new GPUCellGeometry({
      family: 'quadkey',
      maximumVertexCount: 6,
      wordOrder,
      cells: wrap('cells', inputs),
      output: {
        centers: wrap(
          'centers',
          outputs.map(output => output.centers.view)
        ),
        boundaries: wrap(
          'boundaries',
          outputs.map(output => output.boundaries.view)
        ),
        vertexCounts: wrap(
          'vertexCounts',
          outputs.map(output => output.vertexCounts.view)
        ),
        validity: wrap(
          'validity',
          outputs.map(output => output.validity.view)
        )
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const result = {
      centers: [] as number[],
      boundaries: [] as number[],
      vertexCounts: [] as number[],
      validity: [] as number[]
    };
    for (const output of outputs) {
      for (const name of ['centers', 'boundaries', 'vertexCounts', 'validity'] as const) {
        const field = output[name];
        const values =
          field.format === 'float32x2'
            ? await readFloat32(field.buffer)
            : await readUint32(field.buffer);
        const sentinel = field.format === 'float32x2' ? -999 : 99;
        expect(values.slice(0, field.prefix)).toEqual(new Array(field.prefix).fill(sentinel));
        expect(values.at(-1)).toBe(sentinel);
        result[name].push(
          ...values.slice(field.prefix, field.prefix + field.entries * field.componentCount)
        );
      }
    }
    return result;
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

it.each([
  'little-endian',
  'high-low'
] as const)('matches packed geometry exactly for three uneven chunks (%s), with independent tile values', async wordOrder => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const northwest = 1n << 58n;
  const southeast = northwest | 3n;
  const keys = [0n, northwest, southeast, 1n, northwest, 0n, southeast];
  const packed = await runGeometry(device, keys, [7], wordOrder);
  const chunked = await runGeometry(device, keys, [1, 4, 2], wordOrder);
  expect(chunked).toEqual(packed);
  expect(await runGeometry(device, keys, [0, 1, 0, 4, 2], wordOrder)).toEqual(packed);
  expect(chunked.validity).toEqual([0, 1, 1, 0, 1, 0, 1]);
  expect(chunked.vertexCounts).toEqual([0, 4, 4, 0, 4, 0, 4]);
  const centerLatitude = (Math.atan(Math.sinh(Math.PI / 2)) * 180) / Math.PI;
  const northLatitude = (Math.atan(Math.sinh(Math.PI)) * 180) / Math.PI;
  keys.forEach((key, row) => {
    if (key === northwest || key === southeast) {
      const western = key === northwest;
      expect(chunked.centers[row * 2]).toBe(western ? -90 : 90);
      expect(chunked.centers[row * 2 + 1]).toBeCloseTo(
        western ? centerLatitude : -centerLatitude,
        4
      );
      const expected = western
        ? [-180, northLatitude, 0, northLatitude, 0, 0, -180, 0]
        : [0, 0, 180, 0, 180, -northLatitude, 0, -northLatitude];
      expected.forEach((coordinate, index) =>
        expect(chunked.boundaries[row * 12 + index]).toBeCloseTo(coordinate, 4)
      );
      expect(chunked.boundaries.slice(row * 12 + 8, row * 12 + 12)).toEqual([0, 0, 0, 0]);
    } else {
      expect(chunked.centers.slice(row * 2, row * 2 + 2).every(Number.isNaN)).toBe(true);
      expect(chunked.boundaries.slice(row * 12, row * 12 + 12)).toEqual(new Array(12).fill(0));
    }
  });
});

it('writes more boundary records than fit the former single packed 128 MiB binding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const lengths = [1, 524_287, 524_289]; // 1,048,577 * 16 * 8 bytes exceeds 128 MiB.
  const counts: Buffer[] = [];
  const boundaryBuffers: Buffer[] = [];
  const expectedRecords: Map<number, number[]>[] = [];
  const inputs = lengths.map((length, chunkIndex) => {
    const words = new Uint32Array(length * 2);
    const records = new Map<number, number[]>();
    getCellChunkProbeRows(length).forEach((row, probeIndex) => {
      const key = (1n << 58n) | BigInt((chunkIndex + probeIndex) % 4);
      words.set([Number(key & 0xffffffffn), Number(key >> 32n)], row * 2);
      records.set(row, getQuadkeyBoundaryRecord(key, 16));
    });
    expectedRecords.push(records);
    const cells = createInputBuffer(device, words);
    const boundaries = createOutputBuffer(device, length * 16 * 2);
    fillCellChunkOutput(boundaries, 'float32', -999);
    boundaryBuffers.push(boundaries);
    const vertexCounts = createOutputBuffer(device, length);
    vertexCounts.write(new Uint32Array(length).fill(99));
    buffers.push(cells, boundaries, vertexCounts);
    counts.push(vertexCounts);
    return {
      cells: importGraphBuffer(graph, `cells-${chunkIndex}`, cells, 'uint32x2', length),
      boundaries: importGraphBuffer(
        graph,
        `boundaries-${chunkIndex}`,
        boundaries,
        'float32x2',
        length * 16
      ),
      vertexCounts: importGraphBuffer(
        graph,
        `vertex-counts-${chunkIndex}`,
        vertexCounts,
        'uint32',
        length
      )
    };
  });
  graph.add(
    new GPUCellGeometry({
      family: 'quadkey',
      maximumVertexCount: 16,
      cells: createVectorView(
        'cells',
        'uint32x2',
        inputs.map(input => input.cells)
      ),
      output: {
        boundaries: createVectorView(
          'boundaries',
          'float32x2',
          inputs.map(input => input.boundaries)
        ),
        vertexCounts: createVectorView(
          'vertexCounts',
          'uint32',
          inputs.map(input => input.vertexCounts)
        )
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    for (const [chunkIndex, count] of counts.entries()) {
      const records = expectedRecords[chunkIndex];
      await expectCellChunkOutputRecords(
        boundaryBuffers[chunkIndex],
        'float32',
        lengths[chunkIndex],
        32,
        records,
        {absolute: 1e-4}
      );
      await expectCellChunkOutputRecords(
        count,
        'uint32',
        lengths[chunkIndex],
        1,
        new Map([...records.keys()].map(row => [row, [4]]))
      );
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

it.each([
  'little-endian',
  'high-low'
] as const)('geometry checks complete valid boundary records across invocation seams (%s)', async wordOrder => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 771;
  const keys = Array.from({length: rows}, (_, row) =>
    row === 13 ? 0n : (1n << 58n) | BigInt(row % 4)
  );
  const packed = await runGeometry(device, keys, [rows], wordOrder);
  const result = await runGeometry(device, keys, [0, 1, 0, 257, 513, 0], wordOrder);
  expect(result).toEqual(packed);
  const centerLatitude = (Math.atan(Math.sinh(Math.PI / 2)) * 180) / Math.PI;
  keys.forEach((key, row) => {
    expect(result.validity[row]).toBe(key === 0n ? 0 : 1);
    expect(result.vertexCounts[row]).toBe(key === 0n ? 0 : 4);
    if (key === 0n) {
      expect(result.centers.slice(row * 2, row * 2 + 2).every(Number.isNaN)).toBe(true);
      expect(result.boundaries.slice(row * 12, row * 12 + 12)).toEqual(new Array(12).fill(0));
    } else {
      const digit = Number(key & 3n);
      expect(result.centers[row * 2]).toBe((digit & 1) === 0 ? -90 : 90);
      expect(result.centers[row * 2 + 1]).toBeCloseTo(
        (digit & 2) === 0 ? centerLatitude : -centerLatitude,
        4
      );
      getQuadkeyBoundaryRecord(key, 6).forEach((coordinate, component) =>
        expect(result.boundaries[row * 12 + component]).toBeCloseTo(coordinate, 4)
      );
    }
  });
});
