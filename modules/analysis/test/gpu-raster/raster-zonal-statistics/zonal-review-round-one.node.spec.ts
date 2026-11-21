// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {getCellWeightWGSL} from '../../../src/raster-analysis/raster-zonal-statistics/raster-zonal-cell-weight';
import {IS_FINITE_VALUE_WGSL} from '../../../src/analysis-core/kernels/wgsl-common';
import {
  getGPURasterZonalStatisticsParameterValues,
  getRasterZonalCellArea
} from '../../../src/raster-analysis/raster-zonal-statistics/gpu-raster-zonal-statistics';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {expect, it} from 'vitest';
import {GPURasterZonalStatistics} from '../../../src/raster-analysis/raster-zonal-statistics/gpu-raster-zonal-statistics';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';

it('B2 rounds each fused workgroup array to 16 bytes at the 1364/1365 boundary', () => {
  for (const capacity of [1364, 1365]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const view = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      length = capacity
    ) => createTransientView(graph, name, format, length);
    const props = {
      grid: makeGPURasterMetadata({width: 1, height: 1}),
      zoneCapacity: capacity,
      zones: view('zones', 'uint32', 1),
      values: {
        id: 'values',
        format: 'float32' as const,
        storage: {kind: 'buffer' as const, values: view('values', 'float32', 1)}
      },
      output: {
        cellCounts: view('counts', 'uint32'),
        minimums: view('minimums', 'float32'),
        maximums: view('maximums', 'float32')
      }
    };
    const nodes = new GPURasterZonalStatistics(props).getCommandNodes(graph);
    const fused = getGPUCommandNodeKernelSources(nodes).find(kernel =>
      kernel.id.endsWith('-zone-reduce')
    );
    if (capacity === 1364) {
      expect(fused).toBeDefined();
      expect(getWorkgroupMemoryBytes(fused!)).toBe(16368);
    } else {
      expect(fused).toBeUndefined();
    }
  }
});

it('B2 construction alias validation selects the same rounded-memory route as dispatch', () => {
  for (const capacity of [1364, 1365]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const pool = createTransientView(graph, 'pool', 'uint32', 3 * capacity);
    const props = {
      grid: makeGPURasterMetadata({width: 1, height: 1}),
      zoneCapacity: capacity,
      zones: createTransientView(graph, 'zones', 'uint32', 1),
      values: {
        id: 'values',
        format: 'float32' as const,
        storage: {
          kind: 'buffer' as const,
          values: createTransientView(graph, 'values', 'float32', 1)
        }
      },
      output: {
        cellCounts: graph.createDataView(pool.buffer, {format: 'uint32', length: capacity}),
        minimums: graph.createDataView(pool.buffer, {
          format: 'float32',
          byteOffset: 4 * capacity,
          length: capacity
        }),
        maximums: graph.createDataView(pool.buffer, {
          format: 'float32',
          byteOffset: 8 * capacity,
          length: capacity
        })
      }
    };
    // Logical ranges are disjoint, but their 256-byte binding prefixes overlap. Fusion binds
    // all three writable columns together; fallback publishes them in separate dispatches.
    if (capacity === 1364) expect(() => new GPURasterZonalStatistics(props)).toThrow(/overlap/);
    else expect(() => new GPURasterZonalStatistics(props).getCommandNodes(graph)).not.toThrow();
  }
});

it('B3 evaluates the last cell with the nearer pole before rounding through 180 degrees', () => {
  // Analytic spherical strip: R²*(pi/180)*2*sin((2^-17)*pi/180)^2,
  // R=6371007.180918475. /usr/bin/python3 gives 0.02512224432680754 m².
  const height = 5898240;
  for (const direction of [-1, 1]) {
    const step = direction / 65536;
    const full: GPURasterMetadata = {
      width: 1,
      height,
      affine: [1, 0, 0, 0, step, 0],
      pixelInterpretation: 'area'
    };
    const crop: GPURasterMetadata = {
      ...full,
      height: 1,
      affine: [1, 0, 0, 0, step, (height - 1) * step]
    };
    for (const [grid, row] of [
      [full, height - 1],
      [crop, 0]
    ] as const) {
      const settings = getGPURasterZonalStatisticsParameterValues(grid);
      const source = `@group(0) @binding(0) var<storage, read> settings: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
const settingsOffset: u32 = 0u;
${IS_FINITE_VALUE_WGSL}
${getCellWeightWGSL(1, grid.height, 2)}
@compute @workgroup_size(1) fn main() { result[0] = getCellWeight(${row}u); }`;
      const result = new Float32Array(1);
      // Execute the actual generated weight WGSL on the host, with float32 storage. This is a
      // regression aid, not device qualification; the public sorted/atomic routes are queued too.
      new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', 1, {
        0: {0: settings, 1: result}
      });
      expect(Number.isFinite(result[0])).toBe(true);
      expect(result[0] / 0.02512224432680754).toBeCloseTo(1, 5);
      const twin = getRasterZonalCellArea(settings, row, grid.height);
      expect(Number.isFinite(twin)).toBe(true);
      expect(twin / 0.02512224432680754).toBeCloseTo(1, 6);
    }
  }
});

it('earlier-fix note: the CPU area twin rejects short explicit layouts', () => {
  expect(() => getRasterZonalCellArea([1, 1, 60], 0, 1, 1)).toThrow(/at least 4/);
  expect(() => getRasterZonalCellArea([1, 1, 60, 59], 0, 1, 2)).toThrow(/at least 7/);
  expect(Number.isFinite(getRasterZonalCellArea([1, 1, 60, 59, 0, 0, 0, 0], 0, 1, 1))).toBe(true);
  expect(Number.isFinite(getRasterZonalCellArea([1, 1, 60, 59, -1, 30, 1, 999], 0, 1, 2))).toBe(
    true
  );
});

it('t1 declares the full opposite-pole strip without a two-dimensional identity copy', () => {
  const height = 5898240;
  for (const sumOrder of ['sorted', 'atomic'] as const) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const grid = makeGPURasterMetadata({
      width: 1,
      height,
      cellSize: [1, 1 / 65536],
      origin: [0, 0]
    });
    const contributor = new GPURasterZonalStatistics({
      grid,
      sumOrder,
      cellSizeMode: 'geographic',
      parameters: createTransientView(graph, 'settings', 'float32', 7),
      zoneCapacity: 1,
      zones: createTransientView(graph, 'zones', 'uint32', height),
      values: {
        id: 'values',
        format: 'float32',
        storage: {kind: 'buffer', values: createTransientView(graph, 'values', 'float32', height)}
      },
      output: {weightSums: createTransientView(graph, 'weights', 'float32', 1)}
    });
    // The independent row-area expectation is pinned above and in the GPU test. This regression
    // must reach those bounded linear reductions without the unrelated 1 x height focal copy.
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.some(node => node.id.includes('values-elevation'))).toBe(false);
    expect(nodes.some(node => node.id.endsWith('-values-validity'))).toBe(true);
  }
});
