// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {afterAll, beforeEach, afterEach, expect, it, vi} from 'vitest';
import {
  createPointProjectionNode,
  createLineLocateNode
} from '../../../src/gpu-spatial-analysis/linear-referencing/linear-referencing-kernels';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

import * as chunkGroups from '../../../src/analysis-core/kernels/graph-chunk-groups';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

// The interpreter lacks the chunk-group uniform. Keep the real production kernel
// bindings, declarations and body, using constant offsets for this single chunk.
beforeEach(() => {
  vi.spyOn(chunkGroups, 'createGraphChunkGroupKernelNode').mockImplementation((graph, props) =>
    createWGSLKernelNode(graph, props)
  );
});
afterEach(() => vi.restoreAllMocks());
useCPUExecutorTestHooks();
const device = createNullWebGPUDevice();
afterAll(() => device.destroy());
function createLine(smallCount: number) {
  const graph = new GPUCommandGraph(device),
    memory = createCPUGraphMemory();
  const rowCount = 65 + smallCount;
  const coordinates = new Float32Array(rowCount * 2),
    measures = new Float32Array(rowCount);
  for (let row = 0; row < rowCount; row++) {
    coordinates[row * 2] = Math.min(row, 64) * 16;
    coordinates[row * 2 + 1] = Math.max(row - 64, 0) * 2 ** -20;
    // Closed form, rounded once per published row; no implementation prefix is the oracle.
    measures[row] = Math.min(row, 64) * 16 + Math.max(row - 64, 0) * 2 ** -20;
  }
  const positions = createTransientView(graph, 'positions', 'float32x2', rowCount),
    lineOffsets = createTransientView(graph, 'offsets', 'uint32', 2),
    vertexMeasures = createTransientView(graph, 'measures', 'float32', rowCount);
  memory.write(positions, coordinates);
  memory.write(lineOffsets, Uint32Array.of(0, rowCount));
  memory.write(vertexMeasures, measures);
  return {graph, memory, positions, lineOffsets, vertexMeasures, rowCount};
}
it('t1 projection reads the authoritative published endpoint after 65 small increments', () => {
  const {graph, memory, positions, lineOffsets, vertexMeasures, rowCount} = createLine(65);
  const points = createTransientView(graph, 'points', 'float32x2', 1),
    nearestSegments = createTransientView(graph, 'nearest', 'uint32', 1),
    results = createTransientView(graph, 'results', 'uint32', 9);
  memory.write(points, Float32Array.of(1024, 0.00006198883056640625));
  memory.write(nearestSegments, Uint32Array.of(rowCount - 2));
  executeCommandNodesOnCPU(
    [
      createPointProjectionNode(graph, {
        id: 'projection',
        operation: 'GPULinearReferencing',
        points,
        positions,
        lineOffsets,
        nearestSegments,
        vertexMeasures,
        results
      })
    ],
    memory
  );
  const words = memory.readUint32(results),
    values = new Float32Array(words.buffer, words.byteOffset, words.length);
  expect(values).toHaveLength(9);
  expect(values[2]).toBe(1);
  expect(values[6]).toBe(1024.0001220703125);
  expect(values[8]).toBe(1);
});
for (const mode of ['distance', 'fraction'] as const) {
  it(`t1 ${mode} location retains the final coordinate on a rounded measure plateau`, () => {
    const {graph, memory, positions, lineOffsets, vertexMeasures} = createLine(128);
    const eventLineIndices = createTransientView(graph, 'events', 'uint32', 1),
      eventMeasures = createTransientView(graph, 'event-measures', 'float32', 1),
      results = createTransientView(graph, 'located', 'uint32', 6);
    memory.write(eventLineIndices, Uint32Array.of(0));
    memory.write(eventMeasures, Float32Array.of(mode === 'distance' ? 1024.0001220703125 : 1));
    executeCommandNodesOnCPU(
      [
        createLineLocateNode(graph, {
          id: `locate-${mode}`,
          operation: 'GPULineLocate',
          positions,
          lineOffsets,
          vertexMeasures,
          eventLineIndices,
          eventMeasures,
          measureMode: mode,
          results
        })
      ],
      memory
    );
    const words = memory.readUint32(results),
      values = new Float32Array(words.buffer, words.byteOffset, words.length);
    expect(values).toHaveLength(6);
    expect(Array.from(values.slice(0, 2))).toEqual([1024, 0.0001220703125]);
    expect(words[5]).toBe(0);
  });
}

it('B2 location selects the known endpoint without subtract-and-add cancellation', () => {
  const {graph, memory, positions, lineOffsets, vertexMeasures} = createLine(0);
  // One segment goes from (2^25,0) to (1,0). Its rounded length is 2^25,
  // but f32(2^25 + f32(1 - 2^25)) is zero, rather than the known endpoint 1.
  memory.write(positions, Float32Array.of(33554432, 0, 1, 0));
  memory.write(lineOffsets, Uint32Array.of(0, 2));
  memory.write(vertexMeasures, Float32Array.of(0, 33554432));
  const eventLineIndices = createTransientView(graph, 'events', 'uint32', 1);
  const eventMeasures = createTransientView(graph, 'event-measures', 'float32', 1);
  const results = createTransientView(graph, 'located', 'uint32', 6);
  memory.write(eventLineIndices, Uint32Array.of(0));
  memory.write(eventMeasures, Float32Array.of(1));
  executeCommandNodesOnCPU(
    [
      createLineLocateNode(graph, {
        id: 'endpoint-cancellation',
        operation: 'GPULineLocate',
        positions,
        lineOffsets,
        vertexMeasures,
        eventLineIndices,
        eventMeasures,
        measureMode: 'fraction',
        results
      })
    ],
    memory
  );
  const words = memory.readUint32(results);
  const values = new Float32Array(words.buffer, words.byteOffset, words.length);
  expect(Array.from(values.slice(0, 2))).toEqual([1, 0]);
  expect(words[5]).toBe(0);
});
