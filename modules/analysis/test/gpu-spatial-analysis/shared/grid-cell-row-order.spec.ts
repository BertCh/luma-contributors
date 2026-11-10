// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView, GPUGridIndex} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {createGridCellRowOrderNodes} from '../../../src/gpu-spatial-analysis/shared/grid-cell-row-order';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const GRID_SIZE: [number, number] = [4, 4];

/** Cell number of a point in the unit square, as the published grid index maps it. */
function getCellNumber(x: number, y: number): number {
  const column = Math.min(GRID_SIZE[0] - 1, Math.floor(x * GRID_SIZE[0]));
  const row = Math.min(GRID_SIZE[1] - 1, Math.floor(y * GRID_SIZE[1]));
  return row * GRID_SIZE[0] + column;
}

it('createGridCellRowOrderNodes sorts a heavily clustered cell in ascending row order', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // 6,000 of 6,500 rows share one cell (the quadratic rank pass compared 3.6e7 pairs there),
  // the rest are spread over the other cells, and a few rows are outside the unit square.
  const rowCount = 6500;
  const positions = new Float32Array(rowCount * 2);
  for (let row = 0; row < rowCount; row++) {
    const isClustered = row % 13 !== 0 || row > 6000;
    const jitter = ((row * 2654435761) % 1000) / 1000;
    positions[2 * row] = isClustered ? 0.3 + 0.05 * jitter : jitter;
    positions[2 * row + 1] = isClustered ? 0.3 + 0.05 * (1 - jitter) : (jitter * 7) % 1;
  }
  positions[2 * 5] = Number.NaN;
  positions[2 * 5 + 1] = Number.NaN;
  positions[2 * 77] = 2;
  positions[2 * 77 + 1] = 2;

  const graph = new GPUCommandGraph(device, {id: 'grid-cell-row-order'});
  const positionsBuffer = createInputBuffer(device, positions);
  const sortedBuffer = createOutputBuffer(device, rowCount);
  const cellCount = GRID_SIZE[0] * GRID_SIZE[1];
  const cellOffsets = createTransientView(graph, 'cell-offsets', 'uint32', cellCount + 1);
  const objectIds = createTransientView(graph, 'object-ids', 'uint32', rowCount);
  const indexCount = createTransientView(graph, 'index-count', 'uint32', 1);
  const indexOverflow = createTransientView(graph, 'index-overflow', 'uint32', 1);
  const order = createGridCellRowOrderNodes(graph, {
    id: 'order',
    operation: 'grid-cell-row-order-test',
    cellOffsets,
    objectIds,
    indexCount,
    rowCount
  });
  for (const node of new GPUGridIndex({
    id: 'index',
    positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rowCount),
    gridSize: GRID_SIZE,
    bounds: [0, 0, 1, 1],
    cellOffsets,
    objectIds,
    count: indexCount,
    overflow: indexOverflow
  }).getCommandNodes(graph)) {
    graph.add(node);
  }
  for (const node of order.nodes) {
    graph.add(node);
  }
  // Copy the transient result into a readable buffer.
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'publish',
      operation: 'grid-cell-row-order-test',
      variant: 'publish',
      bindings: [
        {name: 'sortedRows', view: order.sortedRows, type: 'u32', access: 'read'},
        {
          name: 'published',
          view: importGraphBuffer(graph, 'sorted-out', sortedBuffer, 'uint32', rowCount),
          type: 'u32',
          access: 'read_write'
        }
      ],
      invocationCount: rowCount,
      body: 'published[publishedOffset + index] = sortedRows[sortedRowsOffset + index];'
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const sorted = await readUint32(sortedBuffer, rowCount);

    const expectedByCell: number[][] = Array.from({length: cellCount}, () => []);
    const excluded: number[] = [];
    for (let row = 0; row < rowCount; row++) {
      const x = positions[2 * row];
      const y = positions[2 * row + 1];
      if (x >= 0 && x <= 1 && y >= 0 && y <= 1) {
        expectedByCell[getCellNumber(x, y)].push(row);
      } else {
        excluded.push(row);
      }
    }
    const expected = [...expectedByCell.flat(), ...excluded];
    expect(Array.from(sorted.slice(0, expected.length - excluded.length))).toEqual(
      expected.slice(0, expected.length - excluded.length)
    );
    expect(new Set(sorted).size).toBe(rowCount);
  } finally {
    compiled.destroy();
    positionsBuffer.destroy();
    sortedBuffer.destroy();
  }
});
