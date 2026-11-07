// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {KEPLER_RATE_CAUSES, addKeplerRateCauseCounts} from '../src/analysis/rate-cause-counts';
import {importGraphBuffer} from '../../analysis/src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32
} from '../../analysis/test/utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';

test('GPU rate cause counts only count selected rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rowCount = 5000;
  const events = new Float32Array(rowCount);
  const populations = new Float32Array(rowCount);
  const selection = new Uint32Array(rowCount);
  const mask = new Uint32Array(rowCount).fill(1);
  const expected = new Array(KEPLER_RATE_CAUSES.length).fill(0);
  for (let row = 0; row < rowCount; row++) {
    events[row] = (row % 5) + 1;
    populations[row] = 10 + (row % 3);
    if (row % 13 === 0) mask[row] = 0;
    if (row % 17 === 1) events[row] = -1;
    if (row % 19 === 2) populations[row] = 0;
    if (row % 23 === 3) events[row] = Infinity;
    selection[row] = row % 3 === 0 || row % 7 === 0 ? 1 : 0;
    if (!selection[row]) continue;
    let cause = 0;
    if (!mask[row]) cause = 1;
    else if (!Number.isFinite(events[row])) cause = 4;
    else if (events[row] < 0) cause = 6;
    else if (populations[row] === 0) cause = 8;
    expected[cause]++;
  }
  const selectedRows = selection.reduce((sum, value) => sum + value, 0);
  const graph = new GPUCommandGraph(device);
  const countsBuffer = createOutputBuffer(device, KEPLER_RATE_CAUSES.length);
  const input = (name: string, data: Float32Array | Uint32Array) =>
    importGraphBuffer(
      graph,
      name,
      createInputBuffer(device, data),
      data instanceof Float32Array ? 'float32' : 'uint32',
      rowCount
    );
  addKeplerRateCauseCounts(graph, {
    events: input('events', events) as never,
    populations: input('populations', populations) as never,
    mask: input('mask', mask) as never,
    selection: input('selection', selection) as never,
    counts: importGraphBuffer(graph, 'counts', countsBuffer, 'uint32', KEPLER_RATE_CAUSES.length)
  });
  const compiled = graph.compile();
  const encoder = device.createCommandEncoder();
  compiled.encode(encoder, {parameters: undefined} as never);
  device.submit(encoder.finish());
  const counts = Array.from(await readUint32(countsBuffer, KEPLER_RATE_CAUSES.length));
  compiled.destroy();
  expect(counts).toEqual(expected);
  expect(counts.reduce((sum, count) => sum + count, 0)).toBe(selectedRows);
});
