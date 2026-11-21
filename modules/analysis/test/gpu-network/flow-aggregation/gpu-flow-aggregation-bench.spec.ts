// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// TOOL-120: the 1M-row atomic versus sorted timing moved out of the correctness spec; the `-bench`
// suffix keeps it out of `yarn verify` and runs it under `yarn test-benchmarks`. The ordinary spec
// keeps a 65,536-row count check of both sum orders.

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createInputBuffer} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createHarness,
  createOrderSensitiveWeights,
  createZoneIds
} from './flow-aggregation-harness';

it('GPUFlowAggregation measures atomic versus sorted sum cost on 1M rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rowCount = 1 << 20;
  const zoneCount = 64;
  const {origins, destinations} = createZoneIds(8, rowCount, zoneCount);
  const weightValues = createOrderSensitiveWeights(6, rowCount);
  const buffers = [
    createInputBuffer(device, Uint32Array.from(origins)),
    createInputBuffer(device, Uint32Array.from(destinations)),
    createInputBuffer(device, weightValues)
  ];
  for (const sumOrder of ['atomic', 'sorted'] as const) {
    const harness = createHarness(device, {
      id: `flow-cost-${sumOrder}`,
      zoneCount,
      topCount: 64,
      pairCapacity: 8192,
      maximumProbeCount: 64,
      hasWeights: true,
      createProps: graph => ({
        sumOrder,
        zones: {kind: 'ids', zoneCount},
        originZoneIds: importGraphBuffer(graph, 'origin-ids', buffers[0], 'uint32', rowCount),
        destinationZoneIds: importGraphBuffer(
          graph,
          'destination-ids',
          buffers[1],
          'uint32',
          rowCount
        ),
        weights: importGraphBuffer(graph, 'weights', buffers[2], 'float32', rowCount)
      })
    });
    harness.submit();
    await harness.read();
    const runCount = 3;
    const startTime = performance.now();
    for (let run = 0; run < runCount; run++) {
      harness.submit();
      await harness.read();
    }
    const milliseconds = (performance.now() - startTime) / runCount;
    const result = await harness.read();
    expect(result.zoneOutCounts.reduce((sum, value) => sum + value, 0)).toBe(rowCount);
    console.log(`flow cost 1M rows, ${sumOrder}: ${milliseconds.toFixed(1)} ms per submit+read`);
    harness.destroy();
  }
  for (const buffer of buffers) buffer.destroy();
});
