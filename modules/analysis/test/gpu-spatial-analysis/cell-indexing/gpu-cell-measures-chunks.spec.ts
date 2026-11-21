// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getCellChunkProbeRows,
  fillCellChunkOutput,
  expectCellChunkOutputRecords
} from '../cell-topology/gpu-cell-chunks.test-utils';
import {s2CellToBoundary, getAngularDistance} from './cell-indexing-oracle';
import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  cellArea,
  cellToBoundary,
  greatCircleDistance,
  getPentagons,
  isPentagon,
  latLngToCell
} from 'h3-js';
import {H3_MEASURES_FIXTURES} from './h3-measures-fixtures';
import {expect, it} from 'vitest';
import {GPU_EARTH_AUTHALIC_RADIUS_METERS} from '../../../src/analysis-core/contracts/earth';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUCellMeasures} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-cell-measures';
import {
  createInputBuffer,
  createOutputBuffer,
  createVectorView,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

async function runMeasures(
  device: Device,
  keys: bigint[],
  lengths: number[],
  precision: 'float32' | 'double-single',
  wordOrder: 'little-endian' | 'high-low',
  options: {family?: 'h3' | 's2'; maximumEdgeCount?: number} = {}
) {
  const maximumEdgeCount = options.maximumEdgeCount ?? 6;
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
    const createField = <Format extends 'float32' | 'uint32'>(
      name: string,
      format: Format,
      entries: number
    ) => {
      const buffer = createOutputBuffer(device, prefix + Math.max(entries, 1) + 1);
      buffer.write(
        format === 'float32'
          ? new Float32Array(prefix + Math.max(entries, 1) + 1).fill(-999)
          : new Uint32Array(prefix + Math.max(entries, 1) + 1).fill(99)
      );
      buffers.push(buffer);
      const backing = importGraphBuffer(
        graph,
        `${name}-${chunkIndex}`,
        buffer,
        'uint32',
        prefix + Math.max(entries, 1) + 1
      );
      return {
        buffer,
        prefix,
        entries,
        format,
        view: graph.createDataView(backing.buffer, {
          format,
          length: entries,
          byteOffset: prefix * 4
        })
      };
    };
    return {
      areas: createField('areas', 'float32', length),
      perimeters: createField('perimeters', 'float32', length),
      edgeLengths: createField('edgeLengths', 'float32', length * maximumEdgeCount),
      edgeCounts: createField('edgeCounts', 'uint32', length),
      pentagons: createField('pentagons', 'uint32', length)
    };
  });
  const wrap = <Format extends 'float32' | 'uint32' | 'uint32x2'>(
    name: string,
    chunks: GraphDataView<Format>[]
  ) => (chunks.length === 1 ? chunks[0] : createVectorView(name, chunks[0].format, chunks));
  graph.add(
    new GPUCellMeasures({
      family: options.family ?? 's2',
      maximumEdgeCount,
      precision,
      wordOrder,
      units: 'kilometers',
      cells: wrap('cells', inputs),
      output: {
        areas: wrap(
          'areas',
          outputs.map(output => output.areas.view)
        ),
        perimeters: wrap(
          'perimeters',
          outputs.map(output => output.perimeters.view)
        ),
        edgeLengths: wrap(
          'edgeLengths',
          outputs.map(output => output.edgeLengths.view)
        ),
        edgeCounts: wrap(
          'edgeCounts',
          outputs.map(output => output.edgeCounts.view)
        ),
        pentagons: wrap(
          'pentagons',
          outputs.map(output => output.pentagons.view)
        )
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const result = {
      areas: [] as number[],
      perimeters: [] as number[],
      edgeLengths: [] as number[],
      edgeCounts: [] as number[],
      pentagons: [] as number[]
    };
    for (const output of outputs) {
      for (const name of [
        'areas',
        'perimeters',
        'edgeLengths',
        'edgeCounts',
        'pentagons'
      ] as const) {
        const field = output[name];
        const values =
          field.format === 'float32'
            ? await readFloat32(field.buffer)
            : await readUint32(field.buffer);
        const sentinel = field.format === 'float32' ? -999 : 99;
        expect(values.slice(0, field.prefix)).toEqual(new Array(field.prefix).fill(sentinel));
        expect(values.at(-1)).toBe(sentinel);
        result[name].push(...values.slice(field.prefix, field.prefix + field.entries));
      }
    }
    return result;
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

it.each([
  'float32',
  'double-single'
] as const)('matches packed measures for three uneven chunks (%s) and the closed-form S2 face area and edges', async precision => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const keys = [0n, 1n << 60n, 3n << 60n, 0n, 5n << 60n, 0n, 11n << 60n];
  for (const wordOrder of ['little-endian', 'high-low'] as const) {
    const packed = await runMeasures(device, keys, [7], precision, wordOrder);
    const chunked = await runMeasures(device, keys, [1, 4, 2], precision, wordOrder);
    expect(chunked).toEqual(packed);
    expect(await runMeasures(device, keys, [0, 1, 0, 4, 2], precision, wordOrder)).toEqual(packed);
    const radius = GPU_EARTH_AUTHALIC_RADIUS_METERS / 1000;
    // Six congruent spherical cube faces partition the sphere. Adjacent corners have dot 1/3.
    const expectedArea = (4 * Math.PI * radius ** 2) / 6;
    const expectedEdge = Math.acos(1 / 3) * radius;
    keys.forEach((key, row) => {
      expect(chunked.edgeCounts[row]).toBe(key === 0n ? 0 : 4);
      expect(chunked.pentagons[row]).toBe(0);
      if (key === 0n) {
        expect(chunked.areas[row]).toBe(0);
        expect(chunked.perimeters[row]).toBe(0);
        expect(chunked.edgeLengths.slice(row * 6, row * 6 + 6)).toEqual(new Array(6).fill(0));
      } else {
        expect(Math.abs(chunked.areas[row] / expectedArea - 1)).toBeLessThan(3e-6);
        expect(Math.abs(chunked.perimeters[row] / (4 * expectedEdge) - 1)).toBeLessThan(3e-6);
        for (const edge of chunked.edgeLengths.slice(row * 6, row * 6 + 4))
          expect(Math.abs(edge / expectedEdge - 1)).toBeLessThan(3e-6);
        expect(chunked.edgeLengths.slice(row * 6 + 4, row * 6 + 6)).toEqual([0, 0]);
      }
    });
  }
});

it('writes more edge records than fit the former packed 128 MiB binding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const lengths = [1, 1_677_720, 1_677_723]; // 3,355,444 * 10 * 4 bytes exceeds 128 MiB.
  const counts: Buffer[] = [];
  const edgeBuffers: Buffer[] = [];
  const expectedRecords: Map<number, number[]>[] = [];
  const inputs = lengths.map((length, chunkIndex) => {
    const words = new Uint32Array(length * 2);
    const records = new Map<number, number[]>();
    const keys = [1n << 60n, 1n << 58n, 3n << 58n, 1n << 56n, 3n << 56n];
    getCellChunkProbeRows(length).forEach((row, probeIndex) => {
      const key = keys[(chunkIndex * 2 + probeIndex) % keys.length];
      words.set([Number(key & 0xffffffffn), Number(key >> 32n)], row * 2);
      const boundary = s2CellToBoundary(key);
      records.set(row, [
        ...boundary.map(
          (point, vertex) =>
            getAngularDistance(point, boundary[(vertex + 1) % 4]) * GPU_EARTH_AUTHALIC_RADIUS_METERS
        ),
        ...new Array(6).fill(0)
      ]);
    });
    expectedRecords.push(records);
    const cells = createInputBuffer(device, words);
    const edgeLengths = createOutputBuffer(device, length * 10);
    fillCellChunkOutput(edgeLengths, 'float32', -999);
    edgeBuffers.push(edgeLengths);
    const edgeCounts = createOutputBuffer(device, length);
    edgeCounts.write(new Uint32Array(length).fill(99));
    buffers.push(cells, edgeLengths, edgeCounts);
    counts.push(edgeCounts);
    return {
      cells: importGraphBuffer(graph, `cells-${chunkIndex}`, cells, 'uint32x2', length),
      edgeLengths: importGraphBuffer(
        graph,
        `edges-${chunkIndex}`,
        edgeLengths,
        'float32',
        length * 10
      ),
      edgeCounts: importGraphBuffer(graph, `counts-${chunkIndex}`, edgeCounts, 'uint32', length)
    };
  });
  graph.add(
    new GPUCellMeasures({
      family: 's2',
      maximumEdgeCount: 10,
      cells: createVectorView(
        'cells',
        'uint32x2',
        inputs.map(input => input.cells)
      ),
      output: {
        edgeLengths: createVectorView(
          'edges',
          'float32',
          inputs.map(input => input.edgeLengths)
        ),
        edgeCounts: createVectorView(
          'counts',
          'uint32',
          inputs.map(input => input.edgeCounts)
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
        edgeBuffers[chunkIndex],
        'float32',
        lengths[chunkIndex],
        10,
        records,
        {relative: 3e-6}
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
  'float32',
  'double-single'
] as const)('S2 measures check valid invocation seams and empty chunks (%s)', async precision => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 771;
  const faces = [1n, 3n, 5n, 7n, 9n, 11n].map(face => face << 60n);
  const keys = Array.from({length: rows}, (_, row) =>
    row === 13 ? 0n : faces[row % faces.length]
  );
  const packed = await runMeasures(device, keys, [rows], precision, 'little-endian');
  const result = await runMeasures(
    device,
    keys,
    [0, 1, 0, 257, 513, 0],
    precision,
    'little-endian'
  );
  expect(result).toEqual(packed);
  const radius = GPU_EARTH_AUTHALIC_RADIUS_METERS / 1000;
  keys.forEach((key, row) => {
    expect(result.edgeCounts[row]).toBe(key === 0n ? 0 : 4);
    expect(result.pentagons[row]).toBe(0);
    if (key === 0n) {
      expect(result.areas[row]).toBe(0);
      expect(result.perimeters[row]).toBe(0);
      expect(result.edgeLengths.slice(row * 6, row * 6 + 6)).toEqual(new Array(6).fill(0));
    } else {
      expect(Math.abs(result.areas[row] / ((2 * Math.PI * radius ** 2) / 3) - 1)).toBeLessThan(
        3e-6
      );
      expect(Math.abs(result.perimeters[row] / (4 * Math.acos(1 / 3) * radius) - 1)).toBeLessThan(
        3e-6
      );
      for (const edge of result.edgeLengths.slice(row * 6, row * 6 + 4))
        expect(Math.abs(edge / (Math.acos(1 / 3) * radius) - 1)).toBeLessThan(3e-6);
      expect(result.edgeLengths.slice(row * 6 + 4, row * 6 + 6)).toEqual([0, 0]);
    }
  });
});

it.each([
  'float32',
  'double-single'
] as const)('H3 measures compare complete hexagon, pentagon and face-crossing records to h3-js (%s)', async precision => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Committed Python fixtures identify a coarse face-crossing cell independently of this kernel.
  const crossing = H3_MEASURES_FIXTURES.find(
    row => row[4] === 0 && row[5] === 0 && Number((BigInt(`0x${row[0]}`) >> 52n) & 15n) <= 3
  );
  expect(crossing).toBeDefined();
  if (!crossing) throw new Error('H3 measures fixture needs a coarse face-crossing cell');
  const pool = [latLngToCell(37.775, -122.418, 9), getPentagons(2)[0], crossing[0]];
  const rows = 771;
  const keys = Array.from({length: rows}, (_, row) =>
    row === 13 ? 0n : BigInt(`0x${pool[row % pool.length]}`)
  );
  const expected = new Map(
    pool.map(cell => {
      const boundary = cellToBoundary(cell);
      const edges = boundary.map((point, vertex) =>
        greatCircleDistance(point, boundary[(vertex + 1) % boundary.length], 'km')
      );
      return [
        cell,
        {
          area: cellArea(cell, 'km2'),
          perimeter: edges.reduce((total, edge) => total + edge, 0),
          edges,
          pentagon: Number(isPentagon(cell))
        }
      ];
    })
  );
  const options = {family: 'h3' as const, maximumEdgeCount: 10};
  const packed = await runMeasures(device, keys, [rows], precision, 'little-endian', options);
  const result = await runMeasures(
    device,
    keys,
    [0, 1, 0, 257, 513, 0],
    precision,
    'little-endian',
    options
  );
  expect(result).toEqual(packed);
  keys.forEach((key, row) => {
    if (key === 0n) {
      expect(result.areas[row]).toBe(0);
      expect(result.perimeters[row]).toBe(0);
      expect(result.edgeCounts[row]).toBe(0);
      expect(result.pentagons[row]).toBe(0);
      expect(result.edgeLengths.slice(row * 10, row * 10 + 10)).toEqual(new Array(10).fill(0));
    } else {
      const reference = expected.get(key.toString(16))!;
      const tolerance =
        precision === 'double-single' ? 2e-5 : row % pool.length === 0 ? 3e-6 : 1e-3;
      expect(Math.abs(result.areas[row] / reference.area - 1)).toBeLessThan(tolerance);
      expect(Math.abs(result.perimeters[row] / reference.perimeter - 1)).toBeLessThan(tolerance);
      expect(result.edgeCounts[row]).toBe(reference.edges.length);
      expect(result.pentagons[row]).toBe(reference.pentagon);
      for (let edge = 0; edge < 10; edge++) {
        if (edge < reference.edges.length)
          expect(
            Math.abs(result.edgeLengths[row * 10 + edge] / reference.edges[edge] - 1)
          ).toBeLessThan(tolerance);
        else expect(result.edgeLengths[row * 10 + edge]).toBe(0);
      }
    }
  });
});
