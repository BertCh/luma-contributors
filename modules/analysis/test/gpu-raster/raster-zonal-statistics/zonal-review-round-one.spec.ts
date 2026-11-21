// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPURasterZonalStatistics,
  getGPURasterZonalStatisticsParameterValues
} from '../../../src/raster-analysis/raster-zonal-statistics/gpu-raster-zonal-statistics';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('B2 fused-memory boundary preserves counts and extrema at capacities 1364 and 1365', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const capacity of [1364, 1365]) {
    const graph = new GPUCommandGraph(device);
    const owned = [
      createInputBuffer(device, new Uint32Array([0])),
      createInputBuffer(device, new Float32Array([17])),
      createOutputBuffer(device, capacity),
      createOutputBuffer(device, capacity),
      createOutputBuffer(device, capacity)
    ];
    graph.add(
      new GPURasterZonalStatistics({
        grid: makeGPURasterMetadata({width: 1, height: 1}),
        zoneCapacity: capacity,
        zones: importGraphBuffer(graph, 'zones', owned[0], 'uint32', 1),
        values: {
          id: 'values',
          format: 'float32',
          storage: {
            kind: 'buffer',
            values: importGraphBuffer(graph, 'values', owned[1], 'float32', 1)
          }
        },
        output: {
          cellCounts: importGraphBuffer(graph, 'counts', owned[2], 'uint32', capacity),
          minimums: importGraphBuffer(graph, 'minimums', owned[3], 'float32', capacity),
          maximums: importGraphBuffer(graph, 'maximums', owned[4], 'float32', capacity)
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect((await readUint32(owned[2], capacity)).slice(0, 2)).toEqual([1, 0]);
      for (const buffer of [owned[3], owned[4]]) {
        const values = await readFloat32(buffer, capacity);
        expect(Number.isFinite(values[0])).toBe(true);
        expect(values[0]).toBe(17);
        expect(Number.isNaN(values[1])).toBe(true);
      }
    } finally {
      compiled.destroy();
      for (const buffer of owned) buffer.destroy();
    }
  }
});

for (const sumOrder of ['sorted', 'atomic'] as const) {
  it(`B3 opposite-pole last row retains analytic area across a full strip and crop (${sumOrder})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    // Literal analytic area for dLon=1°, dLat=2^-16°, midpoint pole distance 2^-17°:
    // R²*(pi/180)*2*sin((2^-17)*pi/180)^2 = 0.02512224432680754 m².
    // This intentionally exercises a 5,898,240-row public reduction, not just the helper kernel.
    for (const direction of [-1, 1]) {
      for (const height of [5898240, 1]) {
        const rowStep = direction / 65536;
        const grid = {
          ...makeGPURasterMetadata({width: 1, height}),
          affine: [1, 0, 0, 0, rowStep, height === 1 ? direction * (90 - 1 / 65536) : 0] as const
        };
        const zones = new Uint32Array(height).fill(0xffffffff);
        zones[height - 1] = 0;
        const values = new Float32Array(height);
        values[height - 1] = 1;
        const graph = new GPUCommandGraph(device);
        const owned = [
          createInputBuffer(device, zones),
          createInputBuffer(device, values),
          createInputBuffer(device, getGPURasterZonalStatisticsParameterValues(grid)),
          createOutputBuffer(device, 1)
        ];
        graph.add(
          new GPURasterZonalStatistics({
            grid,
            zoneCapacity: 1,
            cellSizeMode: 'geographic',
            sumOrder,
            zones: importGraphBuffer(graph, 'zones', owned[0], 'uint32', height),
            values: {
              id: 'values',
              format: 'float32',
              storage: {
                kind: 'buffer',
                values: importGraphBuffer(graph, 'values', owned[1], 'float32', height)
              }
            },
            parameters: importGraphBuffer(graph, 'parameters', owned[2], 'float32', 7),
            output: {weightSums: importGraphBuffer(graph, 'weights', owned[3], 'float32', 1)}
          })
        );
        const compiled = graph.compile();
        try {
          submitGraph(device, compiled, undefined);
          const [area] = await readFloat32(owned[3], 1);
          expect(Number.isFinite(area)).toBe(true);
          expect(area / 0.02512224432680754).toBeCloseTo(1, 5);
        } finally {
          compiled.destroy();
          for (const buffer of owned) buffer.destroy();
        }
      }
    }
  });
}

for (const sumOrder of ['sorted', 'atomic'] as const) {
  it(`t1 unmasked float32 values retain finite-only statistics (${sumOrder})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const zoneCapacity of [1, 5000]) {
      const graph = new GPUCommandGraph(device);
      const owned = [
        createInputBuffer(device, new Uint32Array([0, 0, 0, 0, 0])),
        createInputBuffer(device, new Float32Array([1, NaN, Infinity, -Infinity, 3])),
        ...Array.from({length: 7}, () => createOutputBuffer(device, zoneCapacity))
      ];
      graph.add(
        new GPURasterZonalStatistics({
          grid: makeGPURasterMetadata({width: 1, height: 5}),
          zoneCapacity,
          sumOrder,
          zones: importGraphBuffer(graph, 'zones', owned[0], 'uint32', 5),
          values: {
            id: 'values',
            format: 'float32',
            storage: {
              kind: 'buffer',
              values: importGraphBuffer(graph, 'values', owned[1], 'float32', 5)
            }
          },
          output: {
            cellCounts: importGraphBuffer(graph, 'cell-counts', owned[2], 'uint32', zoneCapacity),
            valueCounts: importGraphBuffer(graph, 'value-counts', owned[3], 'uint32', zoneCapacity),
            means: importGraphBuffer(graph, 'means', owned[4], 'float32', zoneCapacity),
            minimums: importGraphBuffer(graph, 'minimums', owned[5], 'float32', zoneCapacity),
            maximums: importGraphBuffer(graph, 'maximums', owned[6], 'float32', zoneCapacity),
            variances: importGraphBuffer(graph, 'sample', owned[7], 'float32', zoneCapacity),
            populationVariances: importGraphBuffer(
              graph,
              'population',
              owned[8],
              'float32',
              zoneCapacity
            )
          }
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        // Only [1,3] contribute: mean 2, extrema 1/3, sum squared deviations 2,
        // sample divisor 1 and population divisor 2. All five source cells belong to zone zero.
        expect(await readUint32(owned[2], 1)).toEqual([5]);
        expect(await readUint32(owned[3], 1)).toEqual([2]);
        for (const [index, expected] of [
          [4, 2],
          [5, 1],
          [6, 3],
          [7, 2],
          [8, 1]
        ]) {
          const [value] = await readFloat32(owned[index], 1);
          expect(Number.isFinite(value)).toBe(true);
          expect(value).toBeCloseTo(expected, 6);
        }
      } finally {
        compiled.destroy();
        for (const buffer of owned) buffer.destroy();
      }
    }
  });
}
