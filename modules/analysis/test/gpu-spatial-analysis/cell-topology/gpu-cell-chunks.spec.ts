// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  cellToVertexes,
  latLngToCell,
  gridDisk,
  gridDistance,
  gridPathCells,
  getPentagons
} from 'h3-js';
import {expect, it} from 'vitest';
import {GPUCellTopology} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-cell-topology';
import {GPUS2CellHierarchy} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cell-hierarchy';
import {GPUHealpixPointToCell} from '../../../src/gpu-spatial-analysis/healpix-cells/gpu-healpix-point-to-cell';
import {getCellTopologyRowOnCPU} from './cell-topology-oracle';
import {GPUCellGridPath} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-cell-grid-path';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createVectorView,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  CELL_CHUNK_CASES,
  getCellChunkProbeRows,
  fillCellChunkOutput,
  expectCellChunkOutputRecords,
  createCellChunkFields,
  type CellChunkCase,
  type CellChunkOutputName,
  type CellChunkViewFields,
  type CellChunkViewFactory
} from './gpu-cell-chunks.test-utils';

function splitKey(key: bigint): number[] {
  return [Number(key & 0xffffffffn), Number(key >> 32n)];
}
const H3_KEYS = [
  0n,
  BigInt(`0x${latLngToCell(37.775, -122.418, 9)}`),
  BigInt(`0x${latLngToCell(38, -122, 5)}`)
];
const S2_KEYS = [0n, 1n << 60n, 3n << 60n, 5n << 60n, 0n, 7n << 60n, 11n << 60n];
const MASK = [1, 1, 1, 0, 1, 1, 1];
const LONGITUDES = [NaN, 0, 90, 180, -90, 0, 90];

type CellChunkInputRows = {
  keys?: bigint[];
  destinations?: bigint[];
  mask?: number[];
  longitudes?: number[];
};
async function runCellChunks(
  device: Device,
  testCase: CellChunkCase,
  lengths: number[],
  inputRows: CellChunkInputRows = {}
) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const isH3 = testCase.name.includes('H3') || testCase.name === 'GPUCellGridPath';
  const totalRows = lengths.reduce((total, length) => total + length, 0);
  const defaultKeys = isH3
    ? [H3_KEYS[0], H3_KEYS[1], H3_KEYS[2], H3_KEYS[1], 0n, H3_KEYS[2], H3_KEYS[1]]
    : S2_KEYS;
  const keys =
    inputRows.keys ??
    Array.from({length: totalRows}, (_, row) => defaultKeys[row % defaultKeys.length]);
  const masks =
    inputRows.mask ?? Array.from({length: totalRows}, (_, row) => MASK[row % MASK.length]);
  const longitudes =
    inputRows.longitudes ??
    Array.from({length: totalRows}, (_, row) => LONGITUDES[row % LONGITUDES.length]);
  const destinations = inputRows.destinations ?? keys;
  const outputs: {
    name: string;
    format: string;
    buffer: Buffer;
    prefix: number;
    componentCount: number;
  }[] = [];
  const createChunkView: CellChunkViewFactory = ({name, format, length, chunkIndex, rowBase}) => {
    const components = format === 'uint32' ? 1 : 2;
    const prefix = [1, 65, 3][chunkIndex % 3];
    const size = prefix + Math.max(length * components, components) + 1;
    const values =
      format === 'float32x2' ? new Float32Array(size).fill(-999) : new Uint32Array(size).fill(99);
    if (name === 'cells' || name === 'destinations') {
      for (let row = 0; row < length; row++)
        values.set(
          splitKey((name === 'destinations' ? destinations : keys)[rowBase + row]),
          prefix + row * 2
        );
    } else if (name === 'positions') {
      for (let row = 0; row < length; row++)
        values.set([longitudes[rowBase + row], 0], prefix + row * 2);
    } else if (name === 'mask') {
      for (let row = 0; row < length; row++) values[prefix + row] = masks[rowBase + row];
    }
    const buffer = createOutputBuffer(device, size);
    buffer.write(values);
    buffers.push(buffer);
    const backing = importGraphBuffer(graph, `${name}-${chunkIndex}`, buffer, 'uint32', size);
    if (testCase.outputs.includes(name as CellChunkOutputName))
      outputs.push({name, format, buffer, prefix, componentCount: length * components});
    return graph.createDataView(backing.buffer, {format, length, byteOffset: prefix * 4});
  };
  const fields = createCellChunkFields(
    graph,
    lengths,
    testCase.stride,
    createChunkView,
    testCase.distanceStride
  );
  const contributorFields =
    lengths.length === 1
      ? (Object.fromEntries(
          Object.entries(fields).map(([name, view]) => [name, view.data[0]])
        ) as CellChunkViewFields)
      : fields;
  graph.add(testCase.create(contributorFields));
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const result: Record<string, number[]> = {};
    for (const output of outputs) {
      const values =
        output.format === 'float32x2'
          ? await readFloat32(output.buffer)
          : await readUint32(output.buffer);
      const sentinel = output.format === 'float32x2' ? -999 : 99;
      expect(values.slice(0, output.prefix)).toEqual(new Array(output.prefix).fill(sentinel));
      expect(values.at(-1)).toBe(sentinel);
      result[output.name] ??= [];
      result[output.name].push(
        ...values.slice(output.prefix, output.prefix + output.componentCount)
      );
    }
    return {result, keys};
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

function expectCellChunkRows(
  testCase: CellChunkCase,
  result: Record<string, number[]>,
  keys: bigint[],
  masks: number[],
  longitudes: number[]
): void {
  if (testCase.name === 'GPUHealpixPointToCell') {
    // Order zero's equatorial base-pixel centers at 0, 90, 180, -90 degrees are 4, 5, 6, 7.
    expect(result['outputCells']).toEqual(
      longitudes.flatMap((longitude, row) => [
        Number.isFinite(longitude) && masks[row] !== 0 ? 4 + ((longitude + 360) % 360) / 90 : 0,
        0
      ])
    );
    expect(result['validity']).toEqual(
      longitudes.map((longitude, row) => Number(Number.isFinite(longitude) && masks[row] !== 0))
    );
    return;
  }
  keys.forEach((key, row) => {
    const valid = key !== 0n;
    if (testCase.name === 'GPUCellGridPath') {
      const kept = valid && masks[row] !== 0;
      expect(result['distances'][row]).toBe(kept ? 0 : 0xffffffff);
      expect(result['counts'][row]).toBe(kept ? 1 : 0);
      expect(result['outputCells'].slice(row * 8, row * 8 + 8)).toEqual([
        ...splitKey(kept ? key : 0n),
        ...new Array(6).fill(0)
      ]);
    } else if (testCase.name === 'GPUCellTopology children') {
      const kept = valid && masks[row] !== 0;
      const expected = kept
        ? Array.from(
            {length: 4},
            (_, child) => key - (1n << 60n) + BigInt(2 * child + 1) * (1n << 58n)
          ).flatMap(splitKey)
        : new Array(8).fill(0);
      expect(result['outputCells'].slice(row * 8, row * 8 + 8)).toEqual(expected);
      expect(result['counts'][row]).toBe(kept ? 4 : 0);
    } else if (testCase.name === 'GPUCellTopology') {
      const kept = valid && masks[row] !== 0;
      // These are face cells already: the level-zero parent is the input itself.
      expect(result['outputCells'].slice(row * 2, row * 2 + 2)).toEqual(splitKey(kept ? key : 0n));
      expect(result['counts'][row]).toBe(kept ? 1 : 0);
    } else if (testCase.name.startsWith('GPUH3CellIdentities')) {
      const expected = !valid
        ? new Array(12).fill(0)
        : testCase.name.endsWith('vertices')
          ? cellToVertexes(key.toString(16)).flatMap(vertex => splitKey(BigInt(`0x${vertex}`)))
          : Array.from(
              {length: 6},
              (_, slot) =>
                (key & 0x80ffffffffffffffn) | 0x1000000000000000n | (BigInt(slot + 1) << 56n)
            ).flatMap(splitKey);
      expect(result['outputCells'].slice(row * 12, row * 12 + 12)).toEqual(expected);
    } else if (testCase.name === 'GPUS2CellHierarchy') {
      // A face's inclusive leaf range removes/adds its level-zero marker minus one.
      const delta = (1n << 60n) - 1n;
      expect(result['outputCells'].slice(row * 4, row * 4 + 4)).toEqual(
        valid ? [...splitKey(key - delta), ...splitKey(key + delta)] : [0, 0, 0, 0]
      );
      expect(result['status'][row]).toBe(valid ? 0 : 1);
    } else {
      expect(result['status'][row]).toBe(valid ? 0 : 1);
      if (valid) {
        // Six congruent cube faces partition the unit sphere; adjacent unit corners have dot 1/3.
        expect(result['measures'][row * 2]).toBeCloseTo((2 * Math.PI) / 3, 5);
        expect(result['measures'][row * 2 + 1]).toBeCloseTo(4 * Math.acos(1 / 3), 5);
      } else {
        expect(result['measures'].slice(row * 2, row * 2 + 2).every(Number.isNaN)).toBe(true);
      }
    }
  });
}

it.each(
  CELL_CHUNK_CASES
)('$name matches packed output exactly for three uneven chunks and independent expected values', async testCase => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const packed = await runCellChunks(device, testCase, [7]);
  const chunked = await runCellChunks(device, testCase, [1, 4, 2]);
  expect(chunked.result).toEqual(packed.result);
  expect((await runCellChunks(device, testCase, [0, 1, 0, 4, 2])).result).toEqual(packed.result);
  expectCellChunkRows(testCase, chunked.result, chunked.keys, MASK, LONGITUDES);
});

it('GPUCellGridPath writes more whole path records than fit the former packed 128 MiB binding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const lengths = [1, 32_767, 32_769];
  const countBuffers: Buffer[] = [];
  const pathBuffers: Buffer[] = [];
  const expectedRecords: Map<number, number[]>[] = [];
  const chunks = lengths.map((length, chunkIndex) => {
    const words = new Uint32Array(length * 2);
    const records = new Map<number, number[]>();
    getCellChunkProbeRows(length).forEach((row, probeIndex) => {
      const key = BigInt(
        `0x${latLngToCell(37 + (chunkIndex * 16 + probeIndex) * 0.01, -122.4, 9)}`
      );
      words.set(splitKey(key), row * 2);
      records.set(row, [...splitKey(key), ...new Array(510).fill(0)]);
    });
    expectedRecords.push(records);
    const cells = createInputBuffer(device, words);
    const outputCells = createOutputBuffer(device, length * 256 * 2);
    fillCellChunkOutput(outputCells, 'uint32', 99);
    pathBuffers.push(outputCells);
    const counts = createOutputBuffer(device, length);
    counts.write(new Uint32Array(length).fill(99));
    buffers.push(cells, outputCells, counts);
    countBuffers.push(counts);
    return {
      cells: importGraphBuffer(graph, `cells-${chunkIndex}`, cells, 'uint32x2', length),
      outputCells: importGraphBuffer(
        graph,
        `paths-${chunkIndex}`,
        outputCells,
        'uint32x2',
        length * 256
      ),
      counts: importGraphBuffer(graph, `counts-${chunkIndex}`, counts, 'uint32', length)
    };
  });
  const cells = createVectorView(
    'cells',
    'uint32x2',
    chunks.map(chunk => chunk.cells)
  );
  graph.add(
    new GPUCellGridPath({
      family: 'h3',
      origins: cells,
      destinations: cells,
      maximumPathLength: 256,
      output: {
        cells: createVectorView(
          'paths',
          'uint32x2',
          chunks.map(chunk => chunk.outputCells)
        ),
        counts: createVectorView(
          'counts',
          'uint32',
          chunks.map(chunk => chunk.counts)
        )
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    for (const [chunkIndex, buffer] of countBuffers.entries()) {
      const records = expectedRecords[chunkIndex];
      await expectCellChunkOutputRecords(
        pathBuffers[chunkIndex],
        'uint32',
        lengths[chunkIndex],
        512,
        records
      );
      await expectCellChunkOutputRecords(
        buffer,
        'uint32',
        lengths[chunkIndex],
        1,
        new Map([...records.keys()].map(row => [row, [1]]))
      );
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

it.each(
  CELL_CHUNK_CASES
)('$name preserves valid records across 255/256 and 511/512 with empty, uneven, unaligned chunks', async testCase => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 771;
  const isH3 = testCase.name.includes('H3') || testCase.name === 'GPUCellGridPath';
  const pool = isH3 ? H3_KEYS.slice(1) : [1n << 60n, 3n << 60n, 5n << 60n, 7n << 60n, 11n << 60n];
  const keys = Array.from({length: rows}, (_, row) => (row === 13 ? 0n : pool[row % pool.length]));
  const mask = Array.from({length: rows}, (_, row) => (row === 17 ? 0 : 1));
  const longitudes = Array.from({length: rows}, (_, row) =>
    row === 13 ? NaN : [0, 90, 180, -90][row % 4]
  );
  const inputRows = {keys, mask, longitudes};
  const packed = await runCellChunks(device, testCase, [rows], inputRows);
  const chunked = await runCellChunks(device, testCase, [0, 1, 0, 257, 513, 0], inputRows);
  expect(chunked.result).toEqual(packed.result);
  expectCellChunkRows(testCase, chunked.result, keys, mask, longitudes);
});

it.each([
  false,
  true
])('grid-path distance-only/combined outputs match CPU endpoints, masks and truncated paths (combined=%s)', async combined => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const origin = latLngToCell(37.775, -122.418, 9);
  // An odd distance avoids the exact half-step interpolation ties documented by this API.
  const endpoints = gridDisk(origin, 3)
    .filter(cell => gridDistance(origin, cell) === 3)
    .slice(0, 2);
  expect(endpoints).toHaveLength(2);
  const rows = 771;
  const keys = Array.from({length: rows}, (_, row) => (row === 19 ? 0n : BigInt(`0x${origin}`)));
  const destinations = Array.from({length: rows}, (_, row) =>
    BigInt(`0x${row % 3 === 0 ? origin : endpoints[row % 2]}`)
  );
  const mask = Array.from({length: rows}, (_, row) => (row === 17 ? 0 : 1));
  const testCase: CellChunkCase = {
    name: 'grid-path endpoints',
    input: 'cells',
    stride: 2,
    outputs: combined ? ['outputCells', 'distances', 'counts'] : ['distances'],
    create: fields =>
      new GPUCellGridPath({
        family: 'h3',
        origins: fields.cells,
        destinations: fields.destinations,
        mask: fields.mask,
        maximumPathLength: combined ? 2 : undefined,
        output: {
          distances: fields.distances,
          cells: combined ? fields.outputCells : undefined,
          counts: combined ? fields.counts : undefined
        }
      })
  };
  const inputRows = {keys, destinations, mask};
  const packed = await runCellChunks(device, testCase, [rows], inputRows);
  for (const lengths of [
    [1, 257, 513],
    [0, 1, 0, 257, 513, 0]
  ]) {
    const {result} = await runCellChunks(device, testCase, lengths, inputRows);
    expect(result).toEqual(packed.result);
    keys.forEach((key, row) => {
      const kept = key !== 0n && mask[row] !== 0;
      const path = kept ? gridPathCells(key.toString(16), destinations[row].toString(16)) : [];
      expect(result['distances'][row]).toBe(kept ? path.length - 1 : 0xffffffff);
      if (combined) {
        expect(result['counts'][row]).toBe(path.length);
        const expected = path.slice(0, 2).flatMap(cell => splitKey(BigInt(`0x${cell}`)));
        while (expected.length < 4) expected.push(0);
        expect(result['outputCells'].slice(row * 4, row * 4 + 4)).toEqual(expected);
      }
    });
  }
});

it.each([
  'disk',
  'ring'
] as const)('H3 topology %s includes ordered pentagon neighborhoods across chunk boundaries', async operation => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 771;
  const pool = [BigInt(`0x${getPentagons(2)[0]}`), H3_KEYS[1], H3_KEYS[2]];
  const keys = Array.from({length: rows}, (_, row) => (row === 13 ? 0n : pool[row % pool.length]));
  const mask = Array.from({length: rows}, (_, row) => (row === 17 ? 0 : 1));
  const stride = operation === 'disk' ? 7 : 6;
  const testCase: CellChunkCase = {
    name: 'H3 neighborhoods',
    input: 'cells',
    stride,
    distanceStride: stride,
    outputs:
      operation === 'disk' ? ['outputCells', 'distances', 'counts'] : ['outputCells', 'counts'],
    create: fields =>
      new GPUCellTopology({
        family: 'h3',
        operation: {type: operation, k: 1},
        cells: fields.cells,
        mask: fields.mask,
        output: {
          cells: fields.outputCells,
          counts: fields.counts,
          distances: operation === 'disk' ? fields.distances : undefined
        }
      })
  };
  const inputRows = {keys, mask};
  const packed = await runCellChunks(device, testCase, [rows], inputRows);
  const {result} = await runCellChunks(device, testCase, [0, 1, 0, 257, 513, 0], inputRows);
  expect(result).toEqual(packed.result);
  keys.forEach((key, row) => {
    const expected = getCellTopologyRowOnCPU(
      'h3',
      {type: operation, k: 1},
      mask[row] === 0 ? 0n : key
    );
    expect(result['counts'][row]).toBe(expected.cells.length);
    const cells = expected.cells.flatMap(splitKey);
    while (cells.length < stride * 2) cells.push(0);
    expect(result['outputCells'].slice(row * stride * 2, (row + 1) * stride * 2)).toEqual(cells);
    if (operation === 'disk')
      expect(result['distances'].slice(row * stride, (row + 1) * stride)).toEqual([
        ...expected.distances,
        ...new Array(stride - expected.distances.length).fill(0)
      ]);
  });
});

it.each([
  'parent',
  'children'
] as const)('S2 hierarchy %s reports unavailable levels and invalid cells independently of chunking', async operation => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 771;
  const pool = [0n, 1n, 1n << 60n, 3n << 58n, (2n << 60n) | 1n];
  const keys = Array.from({length: rows}, (_, row) => pool[row % pool.length]);
  const stride = operation === 'parent' ? 1 : 4;
  const testCase: CellChunkCase = {
    name: 'S2 unavailable levels',
    input: 'cells',
    stride,
    outputs: ['outputCells', 'status'],
    create: fields =>
      new GPUS2CellHierarchy({
        cells: fields.cells,
        operation,
        level: 1,
        output: {cells: fields.outputCells, status: fields.status}
      })
  };
  const packed = await runCellChunks(device, testCase, [rows], {keys});
  const {result} = await runCellChunks(device, testCase, [0, 1, 0, 257, 513, 0], {keys});
  expect(result).toEqual(packed.result);
  keys.forEach((key, row) => {
    const marker = key & -key;
    const unavailable = operation === 'children' ? marker === 1n : marker > 1n << 58n;
    const status = key === 0n ? 1 : unavailable ? 2 : 0;
    expect(result['status'][row]).toBe(status);
    const expected =
      status !== 0
        ? new Array(stride * 2).fill(0)
        : operation === 'parent'
          ? splitKey((key & ~((1n << 58n) - 1n)) | (1n << 58n))
          : Array.from(
              {length: 4},
              (_, child) => key - marker + BigInt(child * 2 + 1) * (marker >> 2n)
            ).flatMap(splitKey);
    expect(result['outputCells'].slice(row * stride * 2, (row + 1) * stride * 2)).toEqual(expected);
  });
});

it('HEALPix UNIQ at order 29 retains exact equatorial axis keys across uneven chunks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 771;
  const order = 29;
  const longitudes = Array.from({length: rows}, (_, row) =>
    row === 13 ? NaN : [0, 90, 180, -90][row % 4]
  );
  const mask = Array.from({length: rows}, (_, row) => (row === 17 ? 0 : 1));
  const testCase: CellChunkCase = {
    name: 'HEALPix high UNIQ',
    input: 'positions',
    stride: 1,
    outputs: ['outputCells', 'validity'],
    create: fields =>
      new GPUHealpixPointToCell({
        order,
        scheme: 'uniq',
        positions: fields.positions,
        mask: fields.mask,
        output: {cells: fields.outputCells, validity: fields.validity}
      })
  };
  const inputRows = {longitudes, mask};
  const packed = await runCellChunks(device, testCase, [rows], inputRows);
  const {result} = await runCellChunks(device, testCase, [0, 1, 0, 257, 513, 0], inputRows);
  expect(result).toEqual(packed.result);
  // At each equatorial axis ix = nside/2 and iy = nside/2 - 1. The Morton suffix is
  // one even high bit plus all lower odd bits, a finite geometric series.
  const suffix = (5n * 4n ** BigInt(order - 1) - 2n) / 3n;
  longitudes.forEach((longitude, row) => {
    const kept = Number.isFinite(longitude) && mask[row] !== 0;
    const face = 4 + ((longitude + 360) % 360) / 90;
    const key = kept ? 4n ** BigInt(order + 1) + BigInt(face) * 4n ** BigInt(order) + suffix : 0n;
    expect(result['outputCells'].slice(row * 2, row * 2 + 2)).toEqual(splitKey(key));
    expect(result['validity'][row]).toBe(Number(kept));
  });
});
