// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {getInt64TimeWords} from '../../../src/dataframe-analysis/time-window-filter/time-words';
import {addTrajectorySortCheck} from '../../../src/gpu-spatial-analysis/trajectory-analysis/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

async function countOutOfOrder(options: {
  rows: number;
  offsets: number[];
  float32?: number[];
  words?: bigint[];
  low?: number[];
}): Promise<number> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: 'sort-check-test'});
  const output = createOutputBuffer(device, 1);
  // A stale value must be overwritten, including by a zero count.
  output.write(new Uint32Array([77]));
  const buffers = [output];
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const timestamps = options.words
    ? importGraphBuffer(
        graph,
        'times',
        input(getInt64TimeWords(BigInt64Array.from(options.words))),
        'uint32x2',
        options.rows
      )
    : importGraphBuffer(
        graph,
        'times',
        input(Float32Array.from(options.float32 ?? [])),
        'float32',
        options.rows
      );
  addTrajectorySortCheck(graph, {
    positions: importGraphBuffer(
      graph,
      'positions',
      input(new Float32Array(options.rows * 2)),
      'float32x2',
      options.rows
    ),
    timestamps,
    timestampsLow: options.low
      ? importGraphBuffer(
          graph,
          'low',
          input(Float32Array.from(options.low)),
          'float32',
          options.rows
        )
      : undefined,
    trackOffsets: importGraphBuffer(
      graph,
      'offsets',
      input(Uint32Array.from(options.offsets)),
      'uint32',
      options.offsets.length
    ),
    output: {status: {invalidCount: importGraphBuffer(graph, 'out', output, 'uint32', 1)}}
  });
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return (await readUint32(output, 1))[0];
  } finally {
    compiled.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

it('addTrajectorySortCheck counts rows earlier than their predecessor within a track', async () => {
  // Sorted, equal times included, and a track boundary may step back in time.
  expect(await countOutOfOrder({rows: 5, offsets: [0, 3, 5], float32: [1, 2, 2, 0, 1]})).toBe(0);
  expect(await countOutOfOrder({rows: 5, offsets: [0, 3, 5], float32: [1, 3, 2, 5, 4]})).toBe(2);
  // Reversed track offsets disorder the rows they cover.
  expect(await countOutOfOrder({rows: 4, offsets: [3, 1, 4], float32: [0, 1, 2, 3]})).toBe(2);
});

it('addTrajectorySortCheck compares Int64 words and double-single times exactly', async () => {
  const base = 1_700_000_000_000n;
  expect(
    await countOutOfOrder({rows: 3, offsets: [0, 3], words: [base, base + 1n, base + 2n]})
  ).toBe(0);
  // One tick apart at a magnitude no f32 can resolve.
  expect(await countOutOfOrder({rows: 2, offsets: [0, 2], words: [base + 1n, base]})).toBe(1);
  expect(
    await countOutOfOrder({rows: 2, offsets: [0, 2], float32: [1024, 1024], low: [0.5, 0.25]})
  ).toBe(1);
});
