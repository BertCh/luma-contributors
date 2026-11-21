// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUCellPeriodKeys} from '../../../src/dataframe-analysis/value-cube/gpu-cell-period-keys';
import {
  createInputBuffer,
  createOutputBuffer,
  createVectorView,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('GPUCellPeriodKeys matches packed and independent keys/counts for three uneven chunks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const fill of ['ones', 'zeros'] as const) {
    const cells =
      fill === 'ones'
        ? [
            0x1234567800000003n,
            0x1234567800000013n,
            0x1234567800000001n,
            0xffffffffffffffffn,
            0x1234567800000003n,
            0x1234567800000003n,
            0x1234567800000003n,
            0x1234567800000003n,
            0x1234567800000023n
          ]
        : [
            0x1234567800000000n,
            0x1234567800000010n,
            0x1234567800000001n,
            0xffffffffffffffffn,
            0x1234567800000000n,
            0x1234567800000000n,
            0x1234567800000000n,
            0x1234567800000000n,
            0x1234567800000020n
          ];
    const periods = [0, 3, 1, 1, 0xffffffff, 4, 2, 0xfffffffe, 2];
    const mask = [1, 1, 1, 1, 1, 1, 0, 1, 1];
    const expectedKeys = [
      0x1234567800000000n,
      0x1234567800000013n,
      0xffffffffffffffffn,
      0xffffffffffffffffn,
      0xffffffffffffffffn,
      0xffffffffffffffffn,
      0xffffffffffffffffn,
      0xffffffffffffffffn,
      0x1234567800000022n
    ];
    const expectedWords = expectedKeys.flatMap(key => [
      Number(key & 0xffffffffn),
      Number(key >> 32n)
    ]);
    const results: number[][] = [];
    for (const lengths of [[9], [1, 3, 5]]) {
      const graph = new GPUCommandGraph(device);
      const words = Uint32Array.from(
        cells.flatMap(cell => [Number(cell & 0xffffffffn), Number(cell >> 32n)])
      );
      const cellsBuffer = createInputBuffer(device, words);
      const periodsBuffer = createInputBuffer(device, Uint32Array.from(periods));
      const maskBuffer = createInputBuffer(device, Uint32Array.from(mask));
      const keysBuffer = createOutputBuffer(device, cells.length * 2);
      const invalidBuffer = createOutputBuffer(device, 1);
      const split = <Format extends 'uint32' | 'uint32x2'>(view: GraphDataView<Format>) => {
        if (lengths.length === 1) return view;
        let rowBase = 0;
        return createVectorView(
          view.buffer.id,
          view.format,
          lengths.map(length => {
            const chunk = graph.createDataView(view.buffer, {
              format: view.format,
              byteOffset: rowBase * view.rowByteLength,
              length
            });
            rowBase += length;
            return chunk;
          })
        );
      };
      graph.add(
        new GPUCellPeriodKeys({
          cells: split(importGraphBuffer(graph, 'cells', cellsBuffer, 'uint32x2', 9)),
          periods: split(importGraphBuffer(graph, 'periods', periodsBuffer, 'uint32', 9)),
          mask: split(importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', 9)),
          periodCount: 4,
          fill,
          layout: 'legacy-one-bit',
          output: {
            keys: split(importGraphBuffer(graph, 'keys', keysBuffer, 'uint32x2', 9)),
            status: {invalidCount: importGraphBuffer(graph, 'invalid', invalidBuffer, 'uint32', 1)}
          }
        })
      );
      const compiled = graph.compile();
      try {
        for (let encoding = 0; encoding < 2; encoding++) {
          submitGraph(device, compiled, undefined);
          const actual = await readUint32(keysBuffer, 18);
          expect(actual).toEqual(expectedWords);
          expect(await readUint32(invalidBuffer, 1)).toEqual([3]);
          results.push(actual);
        }
      } finally {
        compiled.destroy();
        for (const buffer of [cellsBuffer, periodsBuffer, maskBuffer, keysBuffer, invalidBuffer])
          buffer.destroy();
      }
    }
    for (const result of results) expect(result).toEqual(results[0]);
  }
});
