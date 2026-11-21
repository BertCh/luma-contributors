// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPUCommandGraph,
  createTransientView,
  GPU_GRID_COORDINATE_WGSL
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUPointDensity} from '../../src/gpu-spatial-analysis/point-density/gpu-point-density';
import {createEncounterScanNode} from '../../src/gpu-spatial-analysis/trajectory-encounters/encounter-kernels';
import {createWGSLKernelNode} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph,
  getGPUCommandNodeKernelSources
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

for (const [centerX, centerY] of [
  [0, 0],
  [500000, 6000000]
]) {
  it.each([
    0, 0.0001
  ])(`R9 GPU range stays bounded for the 2048-square grid at (${centerX},${centerY}), distance=%s`, async distance => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const view = <Format extends 'float32x3' | 'uint32' | 'float32' | 'uint32x2'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, name, format, length);
    // Generate the production scan, then probe its actual range function without allocating
    // a four-million-cell index or executing the scan's candidate loops.
    const scan = createEncounterScanNode(graph, {
      id: 'range-source',
      shape: {
        trackCount: 2,
        bucketCount: 1,
        hitCapacity: 2,
        pairCapacity: 2,
        columns: 41,
        rows: 1,
        bounds: [-20.5, 0, 20.5, 1],
        cellSize: 1
      },
      positions: view('positions', 'float32x3', 2),
      cellOffsets: view('offsets', 'uint32', 42),
      objectIds: view('objects', 'uint32', 2),
      distance: view('distance', 'float32', 1),
      state: view('state', 'uint32', 1),
      hitPairs: view('pairs', 'uint32x2', 2),
      hitBuckets: view('buckets', 'uint32', 2),
      hitDistances: view('distances', 'float32', 2)
    });
    const range = getGPUCommandNodeKernelSources([scan])[0].source.match(
      /fn getEncounterAxisRange\([\s\S]*?\n\}/
    )![0];
    const output = createOutputBuffer(device, 4);
    // An unwritten probe must fail the ordered-range assertions as well as size checks.
    output.write(Uint32Array.of(0xffffffff, 0, 0xffffffff, 0));
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'range-probe',
        operation: 'encounter-range-review',
        invocationCount: 1,
        bindings: [
          {
            name: 'results',
            view: importGraphBuffer(graph, 'results', output, 'uint32', 4),
            type: 'u32',
            access: 'read_write'
          }
        ],
        declarations: `${GPU_GRID_COORDINATE_WGSL}\n${range}`,
        body: `let x = getEncounterAxisRange(${centerX}.0, ${distance}, ${centerX - 1}.0, ${centerX + 1}.0, 2048u);
      let y = getEncounterAxisRange(${centerY}.0, ${distance}, ${centerY - 1}.0, ${centerY + 1}.0, 2048u);
      results[resultsOffset] = x.x; results[resultsOffset + 1u] = x.y;
      results[resultsOffset + 2u] = y.x; results[resultsOffset + 3u] = y.y;`
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const [firstX, lastX, firstY, lastY] = await readUint32(output, 4);
      expect(firstX).toBeLessThanOrEqual(lastX);
      expect(firstY).toBeLessThanOrEqual(lastY);
      expect(lastX).toBeLessThan(2048);
      expect(lastY).toBeLessThan(2048);
      const width = lastX - firstX + 1;
      const height = lastY - firstY + 1;
      // Hand bound: one center cell, one padding cell each side, plus at most one
      // search-distance cell each side for a positive radius below 2/2048.
      expect(width).toBeLessThanOrEqual(distance === 0 ? 3 : 5);
      expect(height).toBeLessThanOrEqual(distance === 0 ? 3 : 5);
      expect(width * height).toBeLessThanOrEqual(distance === 0 ? 9 : 25);
    } finally {
      compiled.destroy();
      output.destroy();
    }
  });
}

it.each([
  'atomic',
  'workgroup'
] as const)('R2 public density bins the inclusive extreme maximum with %s sums', async sumAccumulation => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const positions = createInputBuffer(device, Float32Array.of(3e38, 0.5));
  const weights = createInputBuffer(device, Float32Array.of(7));
  const values = createOutputBuffer(device, 2);
  const counts = createOutputBuffer(device, 2);
  const sums = createOutputBuffer(device, 2);
  const callerBounds: [number, number, number, number] = [-3e38, 0, 3e38, 1];
  for (const buffer of [values, counts, sums]) buffer.write(Uint32Array.of(99, 99));
  const contributor = new GPUPointDensity({
    positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', 1),
    weights: importGraphBuffer(graph, 'weights', weights, 'float32', 1),
    bounds: callerBounds,
    gridSize: [2, 1],
    statistic: 'sum',
    sumAccumulation,
    output: {
      values: importGraphBuffer(graph, 'values', values, 'float32', 2),
      counts: importGraphBuffer(graph, 'counts', counts, 'uint32', 2),
      sums: importGraphBuffer(graph, 'sums', sums, 'float32', 2)
    }
  });
  // R10: editing the caller's tuple after construction must not exclude the original maximum.
  callerBounds[2] = 0;
  graph.add(contributor);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    // Inclusive maximum clamps to column 1. One exact integer needs no sum tolerance.
    expect(Array.from(await readUint32(counts, 2))).toEqual([0, 1]);
    expect(Array.from(await readFloat32(sums, 2))).toEqual([0, 7]);
    expect(Array.from(await readFloat32(values, 2))).toEqual([0, 7]);
  } finally {
    compiled.destroy();
    for (const buffer of [positions, weights, values, counts, sums]) buffer.destroy();
  }
});
