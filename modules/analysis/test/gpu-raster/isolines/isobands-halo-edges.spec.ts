// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUIsobandsParameterValues,
  GPUIsobands
} from '../../../src/raster-analysis/isolines/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

async function runZeroIsobandsWindow(
  device: Device,
  bounds: readonly [number, number, number, number]
) {
  const [firstColumn, firstRow, lastColumn, lastRow] = bounds;
  const width = lastColumn - firstColumn;
  const height = lastRow - firstRow;
  const pixelCount = width * height;
  const capacity = 4 * (width - 1) * (height - 1);
  const grid = makeGPURasterMetadata({width, height});
  const graph = new GPUCommandGraph(device, {id: 'isobands-edge-halo'});
  const valuesBuffer = createInputBuffer(device, new Float32Array(pixelCount));
  const breaksBuffer = createInputBuffer(device, new Float32Array([1]));
  const edgesBuffer = createOutputBuffer(device, capacity * 4);
  const bandsBuffer = createOutputBuffer(device, capacity);
  const countBuffer = createOutputBuffer(device, 1);
  const overflowBuffer = createOutputBuffer(device, 1);
  const parameters = new GPUParameterBuffer(device, {
    id: 'isobands-settings',
    format: 'float32',
    length: 12,
    values: getGPUIsobandsParameterValues({breakCount: 1, extent: bounds}, grid)
  });
  const contributor = new GPUIsobands({
    grid,
    values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', pixelCount),
    breaks: importGraphBuffer(graph, 'breaks', breaksBuffer, 'float32', 1),
    parameters: parameters.importToGraph(graph),
    output: {
      edges: importGraphBuffer(graph, 'edges', edgesBuffer, 'float32x4', capacity),
      edgeBands: importGraphBuffer(graph, 'bands', bandsBuffer, 'uint32', capacity),
      status: {
        edgeCount: importGraphBuffer(graph, 'count', countBuffer, 'uint32', 1),
        edgeOverflow: importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1)
      }
    }
  });
  graph.add(contributor);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(overflowBuffer, 1)).toEqual([0]);
    const [count] = await readUint32(countBuffer, 1);
    expect(count).toBe(2 * (width - 1) + 2 * (height - 1));
    const coordinates = await readFloat32(edgesBuffer, count * 4);
    const bands = await readUint32(bandsBuffer, count);
    const coreEdges: number[][] = [];
    for (let index = 0; index < count; index++) {
      const edge = coordinates.slice(4 * index, 4 * index + 4);
      const [startX, startY, endX, endY] = edge;
      // All-zero cells have axis-aligned unit edges. The band is on the left of each
      // directed edge, so stepping left from its midpoint identifies its emitting cell.
      // Vertices are at sample centres (column + 0.5, row + 0.5), in absolute grid coordinates.
      const insideX = (startX + endX) / 2 - Math.sign(endY - startY) * 0.25;
      const insideY = (startY + endY) / 2 + Math.sign(endX - startX) * 0.25;
      const ownerColumn = Math.floor(insideX - 0.5);
      const ownerRow = Math.floor(insideY - 0.5);
      if (ownerColumn === 4 && ownerRow === 4) coreEdges.push([bands[index], ...edge]);
    }
    return {requiredHalo: contributor.requiredHalo, coreEdges};
  } finally {
    compiled.destroy();
    parameters.destroy();
    for (const buffer of [
      valuesBuffer,
      breaksBuffer,
      edgesBuffer,
      bandsBuffer,
      countBuffer,
      overflowBuffer
    ]) {
      buffer.destroy();
    }
  }
}

it('isobands advertised halo cancels shared boundary edges of the owned core cell', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const full = await runZeroIsobandsWindow(device, [0, 0, 9, 9]);
  expect(full.coreEdges).toEqual([]);
  const halo = full.requiredHalo;
  // Core sample ownership is [4, 4, 5, 5]; its marching-squares cell also reads sample 5.
  const expanded = await runZeroIsobandsWindow(device, [4 - halo, 4 - halo, 5 + halo, 5 + halo]);
  expect(expanded.coreEdges).toEqual(full.coreEdges);
  // One pixel cannot inspect the adjacent cells' far samples at column/row 6, so the
  // core cell incorrectly emits its right and upper borders. Pin ownership as well as count.
  const truncated = await runZeroIsobandsWindow(device, [3, 3, 6, 6]);
  expect(truncated.coreEdges).toHaveLength(2);
  expect(truncated.coreEdges).toEqual(
    expect.arrayContaining([
      [0, 5.5, 4.5, 5.5, 5.5],
      [0, 5.5, 5.5, 4.5, 5.5]
    ])
  );
});
