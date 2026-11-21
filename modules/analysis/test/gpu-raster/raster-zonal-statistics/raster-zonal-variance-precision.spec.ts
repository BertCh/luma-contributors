// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPURasterZonalStatistics} from '../../../src/raster-analysis/raster-zonal-statistics/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// One zone of 100,000 cells with v_i = 1e7 + ((i * 7919) % 5) - 2, exact in float32. Each
// offset -2..2 occurs 20,000 times, so the sample variance is 200000 / 99999 (closed form; NumPy
// float64 var(ddof=1) agrees). A float32 model of the sorted (lanes + tree) and atomic
// (sequential) sums, fixes/fix-raster-embeddings-flow/raster_variance_model.py, gives 123.0 and
// 9.8e7 for the plain two-pass variance around the float32 mean, and 2.00002 after refining the
// mean and using the corrected two-pass sum.
const WIDTH = 1000;
const HEIGHT = 100;
const CELL_COUNT = WIDTH * HEIGHT;
const EXPECTED_VARIANCE = 200000 / 99999;

for (const sumOrder of ['sorted', 'atomic'] as const) {
  it(`GPURasterZonalStatistics variance stays accurate at 1e7 over 100,000 cells (${sumOrder})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const values = Float32Array.from(
      {length: CELL_COUNT},
      (_, index) => 1e7 + ((index * 7919) % 5) - 2
    );
    const graph = new GPUCommandGraph(device);
    const owned = [
      createInputBuffer(device, new Uint32Array(CELL_COUNT)),
      createInputBuffer(device, values),
      createOutputBuffer(device, 1),
      createOutputBuffer(device, 1)
    ];
    graph.add(
      new GPURasterZonalStatistics({
        grid: makeGPURasterMetadata({width: WIDTH, height: HEIGHT}),
        zoneCapacity: 1,
        sumOrder,
        zones: importGraphBuffer(graph, 'zones', owned[0], 'uint32', CELL_COUNT),
        values: {
          id: 'values',
          format: 'float32',
          storage: {
            kind: 'buffer',
            values: importGraphBuffer(graph, 'values', owned[1], 'float32', CELL_COUNT)
          }
        },
        output: {
          variances: importGraphBuffer(graph, 'variances', owned[2], 'float32', 1),
          standardDeviations: importGraphBuffer(graph, 'deviations', owned[3], 'float32', 1)
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const [variance] = await readFloat32(owned[2], 1);
      const [deviation] = await readFloat32(owned[3], 1);
      expect(Number.isFinite(variance)).toBe(true);
      expect(Math.abs(variance - EXPECTED_VARIANCE)).toBeLessThan(1e-3);
      expect(Number.isFinite(deviation)).toBe(true);
      expect(Math.abs(deviation - Math.sqrt(EXPECTED_VARIANCE))).toBeLessThan(1e-3);
    } finally {
      compiled.destroy();
      for (const buffer of owned) buffer.destroy();
    }
  });
}
