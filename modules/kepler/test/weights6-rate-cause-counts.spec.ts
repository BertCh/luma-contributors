// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {KEPLER_RATE_CAUSES, addKeplerRateCauseCounts} from '../src/analysis/rate-cause-counts';
import {classifyRateInvalidCauses} from '../../../examples/deck/kepler-compute/src/analysis-numeric-diagnostics';
import {importGraphBuffer} from '../../graph/src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32
} from '../../graph/test/utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';

/** Every cause, over more rows than one 64-lane workgroup and one 65536-row block. */
test('GPU rate cause counts equal the CPU classifier over every row', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rowCount = 70000 + 13;
  const events = new Float32Array(rowCount);
  const populations = new Float32Array(rowCount);
  const eventValidity = new Uint32Array(rowCount).fill(1);
  const populationValidity = new Uint32Array(rowCount).fill(1);
  const mask = new Uint32Array(rowCount).fill(1);
  for (let row = 0; row < rowCount; row++) {
    events[row] = (row % 7) + 1;
    populations[row] = 100 + (row % 11);
    if (row % 997 === 0) mask[row] = 0;
    if (row % 1009 === 1) eventValidity[row] = 0;
    if (row % 1013 === 2) populationValidity[row] = 0;
    if (row % 1019 === 3) events[row] = Infinity;
    if (row % 1021 === 4) populations[row] = NaN;
    if (row % 1031 === 5) events[row] = -2;
    if (row % 1033 === 6) populations[row] = -5;
    if (row % 1039 === 7) populations[row] = 0;
  }
  const graph = new GPUCommandGraph(device);
  const countsBuffer = createOutputBuffer(device, KEPLER_RATE_CAUSES.length);
  addKeplerRateCauseCounts(graph, {
    events: importGraphBuffer(
      graph,
      'events',
      createInputBuffer(device, events),
      'float32',
      rowCount
    ),
    populations: importGraphBuffer(
      graph,
      'populations',
      createInputBuffer(device, populations),
      'float32',
      rowCount
    ),
    eventValidity: importGraphBuffer(
      graph,
      'event-validity',
      createInputBuffer(device, eventValidity),
      'uint32',
      rowCount
    ),
    populationValidity: importGraphBuffer(
      graph,
      'population-validity',
      createInputBuffer(device, populationValidity),
      'uint32',
      rowCount
    ),
    mask: importGraphBuffer(graph, 'mask', createInputBuffer(device, mask), 'uint32', rowCount),
    counts: importGraphBuffer(graph, 'counts', countsBuffer, 'uint32', KEPLER_RATE_CAUSES.length)
  });
  const compiled = graph.compile();
  const encoder = device.createCommandEncoder();
  compiled.encode(encoder, {parameters: undefined} as never);
  device.submit(encoder.finish());
  const gpuCounts = Array.from(await readUint32(countsBuffer, KEPLER_RATE_CAUSES.length));
  compiled.destroy();

  const cpuCounts = classifyRateInvalidCauses(
    Array.from(events, (value, row) => (eventValidity[row] ? value : null)),
    Array.from(populations, (value, row) => (populationValidity[row] ? value : null)),
    mask
  ).counts;
  expect(gpuCounts).toEqual(KEPLER_RATE_CAUSES.map(cause => cpuCounts[cause]));
  expect(gpuCounts.reduce((sum, count) => sum + count, 0)).toBe(rowCount);
  for (const count of gpuCounts) expect(count).toBeGreaterThan(0);
});
