// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GridStatisticsHarness} from './gpu-region-statistics-grid-harness';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {getGridCandidateNodes} from '../../../src/gpu-spatial-analysis/region-statistics/region-statistics-grid';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

async function readCandidates(
  device: Device,
  options: {
    bounds?: number[];
    vertices?: Float32Array;
    activeCount?: number;
    inheritedOverflow?: number;
  }
): Promise<number[]> {
  const graph = new GPUCommandGraph(device);
  const buffers: ReturnType<typeof createInputBuffer>[] = [];
  const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    name: string,
    data: Float32Array | Uint32Array,
    format: Format,
    length = data.length
  ) => {
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format, length);
  };
  // Four occupied rows in a tall one-column grid, with empty runs between them.
  // Row order is [2, 0, 3, 1]; a capacity of two retains [2, 0], while count stays four.
  const occupiedRows = [0, 100, 4096, 8192];
  const offsets = Uint32Array.from(
    {length: 8194},
    (_, row) => occupiedRows.filter(occupied => occupied < row).length
  );
  const positions = input(
    'positions',
    Float32Array.of(0.5, 100.5, 0.5, 8192.5, 0.5, 0.5, 0.5, 4096.5),
    'float32x2',
    4
  );
  const candidates = getGridCandidateNodes(graph, {
    id: 'candidate-fixture',
    rowCount: 4,
    positions,
    region: options.vertices
      ? {
          kind: 'polygon',
          vertices: input('vertices', options.vertices, 'float32x2', options.vertices.length / 2),
          vertexCount: input(
            'active',
            Uint32Array.of(options.activeCount ?? options.vertices.length / 2),
            'uint32'
          )
        }
      : {
          kind: 'rectangle',
          bounds: input('bounds', Float32Array.from(options.bounds ?? [0, 0, 1, 8193]), 'float32')
        },
    regionOverflow: input('region-overflow', Uint32Array.of(0), 'uint32'),
    spatialIndex: {
      kind: 'grid',
      candidateCapacity: 2,
      index: {
        gridSize: [1, 8193],
        bounds: [0, 0, 1, 8193],
        cellOffsets: input('offsets', offsets, 'uint32'),
        objectIds: input('object-ids', Uint32Array.of(2, 0, 3, 1), 'uint32'),
        count: input('index-count', Uint32Array.of(4), 'uint32'),
        overflow: input('index-overflow', Uint32Array.of(options.inheritedOverflow ?? 0), 'uint32')
      }
    }
  });
  const outputBuffer = createOutputBuffer(device, 4);
  buffers.push(outputBuffer);
  const output = importGraphBuffer(graph, 'output', outputBuffer, 'uint32', 4);
  const readbackNode = createWGSLKernelNode(graph, {
    id: 'read-candidates',
    operation: 'region-grid-fixture',
    invocationCount: 2,
    bindings: [
      {name: 'rows', view: candidates.rows, type: 'u32', access: 'read'},
      {name: 'count', view: candidates.count, type: 'u32', access: 'read'},
      // The status pass consumes the candidate overflow after the candidate query nodes.
      {name: 'flag', view: candidates.candidateFlag, type: 'u32', access: 'read'},
      {name: 'output', view: output, type: 'u32', access: 'read_write'}
    ],
    body: `output[outputOffset + index] = rows[rowsOffset + index];
  if (index == 0u) {
    output[outputOffset + 2u] = count[countOffset];
    output[outputOffset + 3u] = flag[flagOffset];
  }`
  });
  // candidateFlag is published by the explicit status node, before the readback node.
  const counts = input('status-counts', Uint32Array.of(0, 0, 0), 'uint32');
  for (const node of candidates.nodes) graph.add(node);
  graph.add(candidates.getStatusNode(counts));
  graph.add(readbackNode);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return await readUint32(outputBuffer, 4);
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

it('region grid preserves tall-grid row order, full count, empty ranges and inherited overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  expect(await readCandidates(device, {})).toEqual([2, 0, 4, 1]);
  // Row 100 is outside [99.1,99.3]; only the one-cell rounding margin includes its object 0.
  expect(await readCandidates(device, {bounds: [0, 99.1, 1, 99.3]})).toEqual([0, 0xffffffff, 1, 0]);
  expect(await readCandidates(device, {bounds: [0, 99.5, 1, 100.5]})).toEqual([
    0, 0xffffffff, 1, 0
  ]);
  expect(await readCandidates(device, {bounds: [2, 0, 3, 8193]})).toEqual([
    0xffffffff, 0xffffffff, 0, 0
  ]);
  expect(await readCandidates(device, {bounds: [2, 0, 3, 8193], inheritedOverflow: 1})).toEqual([
    0xffffffff, 0xffffffff, 0, 1
  ]);
});

it('region grid ignores inactive nonfinite vertices and detects a later active nonfinite tile', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const vertices = new Float32Array(8193 * 2);
  vertices.set([0, 0, 1, 0, 0, 1]);
  vertices[8192 * 2] = Number.NaN;
  expect(await readCandidates(device, {vertices, activeCount: 3})).toEqual([2, 0xffffffff, 1, 0]);
  expect(await readCandidates(device, {vertices, activeCount: 8193})).toEqual([2, 0, 4, 1]);
  expect(await readCandidates(device, {vertices, activeCount: 99999})).toEqual([2, 0, 4, 1]);
  expect(await readCandidates(device, {vertices, activeCount: 2})).toEqual([
    0xffffffff, 0xffffffff, 0, 0
  ]);
});

it('region grid harness preserves exact polygon selection across inactive and later nonfinite vertices', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new GridStatisticsHarness({
    device,
    id: 'parallel-polygon-harness',
    selectionKind: 'polygon',
    positions: Float32Array.of(0.25, 0.25, 0.5, 8192.5),
    values: Float32Array.of(2, 9),
    vertexCapacity: 8193,
    candidateCapacity: 2,
    gridSize: [1, 8193],
    domain: [0, 0, 1, 8193],
    indexMode: 'same-graph',
    withOutput: true
  });
  try {
    const vertices = new Float32Array(8193 * 2);
    vertices.set([0, 0, 1, 0, 0, 1]);
    vertices[8192 * 2] = NaN;
    for (const activeCount of [3, 8193]) {
      harness.setPolygon(vertices, activeCount);
      harness.encodeBoth();
      // The only finite crossing edges enclose the unit right triangle: only (1/4,1/4) is inside.
      for (const route of ['brute', 'grid'] as const) {
        const readback = await harness.read(route);
        expect(readback.ids).toEqual([0]);
        expect(readback.result.selectedCount).toBe(1);
        expect(Number.isFinite(readback.result.sum)).toBe(true);
        expect(readback.result.sum).toBe(2);
        expect(readback.result.candidatesTruncated).toBe(false);
      }
    }
  } finally {
    harness.destroy();
  }
});
