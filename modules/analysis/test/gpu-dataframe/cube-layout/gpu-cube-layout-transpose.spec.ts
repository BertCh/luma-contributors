// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {describe, expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUCubeLayoutTranspose,
  type GPUCubeLayout
} from '../../../src/dataframe-analysis/cube-layout/index';
import {
  createInputBuffer,
  createOutputBuffer,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const SHAPES: [cellCount: number, periodCount: number][] = [
  [7, 5],
  [1, 9],
  [9, 1],
  [1, 1],
  [300, 17],
  [4099, 33]
];

/** Bits of the transposed cube by explicit (cell, period) addressing, independent of the kernel. */
function transposeOnCPU(
  values: Uint32Array,
  from: GPUCubeLayout,
  cellCount: number,
  periodCount: number
): Uint32Array {
  const expected = new Uint32Array(values.length);
  for (let cell = 0; cell < cellCount; cell++) {
    for (let period = 0; period < periodCount; period++) {
      const cellMajor = cell * periodCount + period;
      const periodMajor = period * cellCount + cell;
      if (from === 'cell-major') {
        expected[periodMajor] = values[cellMajor];
      } else {
        expected[cellMajor] = values[periodMajor];
      }
    }
  }
  return expected;
}

async function transpose(
  values: Uint32Array,
  format: 'uint32' | 'float32',
  from: GPUCubeLayout,
  cellCount: number,
  periodCount: number
): Promise<Uint32Array> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: 'cube-layout-transpose'});
  const source =
    format === 'float32'
      ? createInputBuffer(device, new Float32Array(values.buffer, values.byteOffset, values.length))
      : createInputBuffer(device, values);
  const target = createOutputBuffer(device, values.length);
  graph.add(
    new GPUCubeLayoutTranspose({
      values: importGraphBuffer(graph, 'values', source, format, values.length),
      output: importGraphBuffer(graph, 'transposed', target, format, values.length),
      from,
      cellCount,
      periodCount
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const bytes = await target.readAsync();
  const result = new Uint32Array(bytes.buffer, bytes.byteOffset, values.length).slice();
  compiled.destroy();
  source.destroy();
  target.destroy();
  return result;
}

describe('GPUCubeLayoutTranspose', () => {
  for (const format of ['uint32', 'float32'] as const) {
    for (const from of ['cell-major', 'period-major'] as const) {
      it(`moves a ${format} cube from ${from} bit for bit`, async () => {
        for (const [cellCount, periodCount] of SHAPES) {
          const values = new Uint32Array(cellCount * periodCount);
          for (let index = 0; index < values.length; index++) {
            // Distinct bit patterns, including NaN payloads and signed zero for floats.
            values[index] = Math.imul(index + 1, 2654435761) >>> 0;
          }
          if (format === 'float32' && values.length > 4) {
            values[0] = 0x7fc00123;
            values[1] = 0x80000000;
            values[2] = 0x7f800000;
            // Signalling NaN payloads: only a word copy keeps them.
            values[3] = 0x7f800001;
            values[4] = 0xffa00005;
          }
          const result = await transpose(values, format, from, cellCount, periodCount);
          expect(Array.from(result), `${format} ${from} ${cellCount}x${periodCount}`).toEqual(
            Array.from(transposeOnCPU(values, from, cellCount, periodCount))
          );
        }
      });
    }
  }

  it('returns the original cube after transposing there and back', async () => {
    const [cellCount, periodCount] = [37, 11];
    const values = Uint32Array.from({length: cellCount * periodCount}, (_, index) => index * 3 + 1);
    const forward = await transpose(values, 'uint32', 'cell-major', cellCount, periodCount);
    // Element (cell 5, period 2) moved from 5 * 11 + 2 to 2 * 37 + 5.
    expect(forward[2 * cellCount + 5]).toBe(values[5 * periodCount + 2]);
    const back = await transpose(forward, 'uint32', 'period-major', cellCount, periodCount);
    expect(Array.from(back)).toEqual(Array.from(values));
  });
});
