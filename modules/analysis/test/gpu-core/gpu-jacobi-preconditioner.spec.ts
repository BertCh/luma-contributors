// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUJacobiPreconditioner} from '@luma.gl/graph/gpu-core';
import {getGraphVectorData} from '@luma.gl/graph/gpu-core';
import {BatchConformanceFixture} from './batch-conformance-utils';
import {requireTestDevice} from '../utils/require-test-device';

type Pattern =
  | 'positive'
  | 'signed'
  | 'zero'
  | 'missing'
  | 'negative'
  | 'late-signed'
  | 'late-zero';
type Partition = 'contiguous' | 'independent' | 'empty';

async function checkDiagonal(
  pattern: Pattern,
  partition: Partition,
  featureLevel: 'core' | 'max' = 'max',
  boundedDispatch = false,
  allLongRows = false
) {
  const device = await getWebGPUTestDevice(featureLevel);
  requireTestDevice(device);
  const descriptor = Object.getOwnPropertyDescriptor(device, 'limits');
  if (boundedDispatch) {
    const limits = device.limits;
    Object.defineProperty(device, 'limits', {
      configurable: true,
      value: new Proxy(limits, {
        get: (target, property) =>
          property === 'maxComputeWorkgroupsPerDimension'
            ? 2
            : Reflect.get(target, property, target)
      })
    });
  }
  const fixture = new BatchConformanceFixture(device);
  try {
    const rowCount = 321;
    const longRows = new Set([0, 31, 32, 63, 64, 127, 128, 255, 320]);
    const offsets = [0];
    const columns: number[] = [];
    const values: number[] = [];
    const expected: number[] = [];
    for (let row = 0; row < rowCount; row++) {
      let diagonal = 0;
      const longRow = allLongRows || longRows.has(row);
      const length = longRow ? 4103 : row % 7 === 0 ? 0 : 3;
      for (let entry = 0; entry < length; entry++) {
        const lateTerms = pattern === 'late-signed' || pattern === 'late-zero';
        const term =
          lateTerms && longRow
            ? entry === 0
              ? 0
              : entry === 256
                ? 1
                : entry === 257
                  ? 2
                  : -1
            : entry;
        const diagonalEntry =
          pattern === 'positive' ? entry < (longRow ? 4096 : 1) : term >= 0 && term < 3;
        const column = pattern !== 'missing' && diagonalEntry ? row : (row + 1) % rowCount;
        const value =
          pattern === 'positive'
            ? 0.125
            : pattern === 'signed' || pattern === 'late-signed'
              ? [1e8, -1e8, 1][Math.max(0, term) % 3]
              : pattern === 'late-zero'
                ? [1e8, 1, -1e8][Math.max(0, term) % 3]
                : pattern === 'zero'
                  ? [1, -1, 0][entry % 3]
                  : pattern === 'negative'
                    ? [1, -3, 1][entry % 3]
                    : 2;
        columns.push(column);
        values.push(value);
        if (column === row) diagonal = Math.fround(diagonal + value);
      }
      offsets.push(values.length);
      expected.push(diagonal === 0 ? 0 : Math.fround(1 / diagonal));
    }
    const rowOffsets = fixture.column(
      'offsets',
      'uint32',
      offsets,
      partition === 'contiguous' ? [rowCount + 1] : [1, 0, 64, rowCount - 64]
    );
    const columnIndices = fixture.column(
      'columns',
      'uint32',
      columns,
      partition === 'contiguous'
        ? [columns.length]
        : partition === 'independent'
          ? [4097, columns.length - 4097]
          : [0, 4097, 0, columns.length - 4097, 0]
    );
    const matrixValues = fixture.column(
      'values',
      'float32',
      values,
      partition === 'contiguous'
        ? [values.length]
        : partition === 'independent'
          ? [1, 4096, values.length - 4097]
          : [0, 1, 0, 4096, 0, values.length - 4097, 0]
    );
    const inverseDiagonal = fixture.column(
      'inverse',
      'float32',
      new Array(rowCount).fill(77),
      partition === 'contiguous' ? [rowCount] : [64, 0, rowCount - 64]
    );
    fixture.graph.add(
      new GPUJacobiPreconditioner({
        rowOffsets,
        columnIndices,
        values: matrixValues,
        inverseDiagonal
      })
    );
    const compiled = fixture.graph.compile();
    try {
      for (const cleared of [false, true, false]) {
        let offset = 0;
        for (const chunk of getGraphVectorData(matrixValues)) {
          if (chunk.length)
            fixture.storage
              .get(chunk.buffer)!
              .write(
                new Float32Array(
                  cleared
                    ? new Array(chunk.length).fill(0)
                    : values.slice(offset, offset + chunk.length)
                ),
                chunk.byteOffset
              );
          offset += chunk.length;
        }
        const encoder = device.createCommandEncoder();
        compiled.encode(encoder, {parameters: undefined});
        device.submit(encoder.finish());
        expect(await fixture.read(inverseDiagonal)).toEqual(
          cleared ? new Array(rowCount).fill(0) : expected
        );
        for (const chunk of getGraphVectorData(inverseDiagonal)) {
          const bytes = await fixture.storage.get(chunk.buffer)!.readAsync(0, 4);
          expect(new Float32Array(bytes.buffer, bytes.byteOffset, 1)[0]).toBe(999);
        }
      }
    } finally {
      compiled.destroy();
      expect(fixture.buffers.every(buffer => !buffer.destroyed)).toBe(true);
    }
  } finally {
    if (boundedDispatch) {
      if (descriptor) Object.defineProperty(device, 'limits', descriptor);
      else Reflect.deleteProperty(device, 'limits');
    }
    fixture.destroy();
  }
}

for (const pattern of [
  'positive',
  'signed',
  'zero',
  'missing',
  'negative',
  'late-signed',
  'late-zero'
] as const) {
  for (const partition of ['contiguous', 'independent', 'empty'] as const) {
    test(`Jacobi diagonal preserves ${pattern} with ${partition} storage and replay`, async () => {
      await checkDiagonal(pattern, partition);
    });
  }
}
test('Jacobi long rows work at core limits', async () => {
  await checkDiagonal('positive', 'independent', 'core');
});
test('Jacobi local long-row queue handles padded 3D dispatch', async () => {
  await checkDiagonal('positive', 'contiguous', 'max', true);
});
for (const pattern of ['positive', 'late-signed'] as const) {
  test(`Jacobi retains row parallelism when every row is long with ${pattern} diagonals`, async () => {
    await checkDiagonal(pattern, 'contiguous', 'max', false, true);
  });
}

for (const rowCount of [0, 1, 321]) {
  test(`Jacobi clears ${rowCount} rows with a completely empty entry domain on replay`, async () => {
    const device = await getWebGPUTestDevice('core');
    requireTestDevice(device);
    const fixture = new BatchConformanceFixture(device);
    try {
      const rowOffsets = fixture.column('offsets', 'uint32', new Array(rowCount + 1).fill(0), [
        1,
        0,
        rowCount
      ]);
      const columnIndices = fixture.column('columns', 'uint32', [], [0, 0]);
      const values = fixture.column('values', 'float32', [], [0]);
      const inverseDiagonal = fixture.column(
        'inverse',
        'float32',
        new Array(rowCount).fill(77),
        rowCount > 1 ? [64, 0, rowCount - 64] : [0, rowCount]
      );
      fixture.graph.add(
        new GPUJacobiPreconditioner({rowOffsets, columnIndices, values, inverseDiagonal})
      );
      const compiled = fixture.graph.compile();
      try {
        for (const sentinel of [77, -23]) {
          for (const chunk of getGraphVectorData(inverseDiagonal)) {
            if (chunk.length) {
              fixture.storage
                .get(chunk.buffer)!
                .write(new Float32Array(chunk.length).fill(sentinel), chunk.byteOffset);
            }
          }
          const encoder = device.createCommandEncoder();
          compiled.encode(encoder, {parameters: undefined});
          device.submit(encoder.finish());
          expect(await fixture.read(inverseDiagonal)).toEqual(new Array(rowCount).fill(0));
          for (const chunk of getGraphVectorData(inverseDiagonal)) {
            const bytes = await fixture.storage.get(chunk.buffer)!.readAsync(0, 4);
            expect(new Float32Array(bytes.buffer, bytes.byteOffset, 1)[0]).toBe(999);
          }
        }
      } finally {
        compiled.destroy();
        expect(fixture.buffers.every(buffer => !buffer.destroyed)).toBe(true);
      }
    } finally {
      fixture.destroy();
    }
  });
}
