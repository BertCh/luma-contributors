// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {addTrajectorySortCheck} from '../../../src/gpu-spatial-analysis/trajectory-analysis/trajectory-sort-check';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

function countInvalidTimes(times: number[], low?: number[], offsets = [0, times.length]) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const nodes: GPUCommandNode[] = [];
  graph.add = node => {
    nodes.push(node as GPUCommandNode);
  };
  try {
    const timestamps = createTransientView(graph, 'times', 'float32', times.length);
    const timestampsLow = low
      ? createTransientView(graph, 'low', 'float32', low.length)
      : undefined;
    const trackOffsets = createTransientView(graph, 'offsets', 'uint32', offsets.length);
    const invalidCount = createTransientView(graph, 'invalid', 'uint32', 1);
    addTrajectorySortCheck(graph, {
      positions: createTransientView(graph, 'positions', 'float32x2', times.length),
      timestamps,
      timestampsLow,
      trackOffsets,
      output: {status: {invalidCount}}
    });
    const memory = createCPUGraphMemory();
    memory.write(timestamps, Float32Array.from(times));
    if (timestampsLow) memory.write(timestampsLow, Float32Array.from(low!));
    memory.write(trackOffsets, Uint32Array.from(offsets));
    memory.write(invalidCount, Uint32Array.of(77));
    executeCommandNodesOnCPU(nodes, memory);
    return memory.readUint32(invalidCount)[0];
  } finally {
    device.destroy();
  }
}

it('counts each nonfinite row once, including singleton tracks and either double-single part', () => {
  expect(countInvalidTimes([NaN])).toBe(1);
  expect(countInvalidTimes([0, Infinity])).toBe(1);
  expect(countInvalidTimes([-Infinity, 0])).toBe(1);
  expect(countInvalidTimes([0, NaN, 1, 0])).toBe(2);
  expect(countInvalidTimes([NaN, Infinity, -Infinity, 0])).toBe(3);
  expect(countInvalidTimes([NaN], [0])).toBe(1);
  expect(countInvalidTimes([0], [Infinity])).toBe(1);
  // Different high parts cannot hide an invalid low part.
  expect(countInvalidTimes([0, 1, 2], [0, Infinity, 0])).toBe(1);
  expect(countInvalidTimes([0, NaN, 2], [0, Infinity, 0])).toBe(1);
});

it('keeps track boundaries, ties, reversed offsets and clear output', () => {
  expect(countInvalidTimes([1, 0])).toBe(1);
  expect(countInvalidTimes([1, 2, 2])).toBe(0);
  expect(countInvalidTimes([1, 2, 0, 1], undefined, [0, 2, 4])).toBe(0);
  expect(countInvalidTimes([0, 1, 2, 3], undefined, [3, 1, 4])).toBe(2);
  expect(countInvalidTimes([1024, 1024], [0.5, 0.25])).toBe(1);
});

it('rejects a zero-length status subview before adding either graph node', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const nodes: GPUCommandNode[] = [];
  graph.add = node => {
    nodes.push(node as GPUCommandNode);
  };
  try {
    const storage = createTransientView(graph, 'storage', 'uint32', 3);
    const invalidCount = graph.createDataView(storage.buffer, {
      format: 'uint32',
      byteOffset: 4,
      length: 0
    });
    expect(() =>
      addTrajectorySortCheck(graph, {
        id: 'sort-review',
        positions: createTransientView(graph, 'positions', 'float32x2', 1),
        timestamps: createTransientView(graph, 'timestamps', 'float32', 1),
        trackOffsets: createTransientView(graph, 'offsets', 'uint32', 2),
        output: {status: {invalidCount}}
      })
    ).toThrow(/^sort-review status.invalidCount/);
    expect(nodes).toHaveLength(0);
  } finally {
    device.destroy();
  }
});

it('accepts padded status storage and leaves its tail untouched', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const nodes: GPUCommandNode[] = [];
  graph.add = node => {
    nodes.push(node as GPUCommandNode);
  };
  try {
    const timestamps = createTransientView(graph, 'times', 'float32', 1);
    const trackOffsets = createTransientView(graph, 'offsets', 'uint32', 2);
    const invalidCount = createTransientView(graph, 'status', 'uint32', 2);
    addTrajectorySortCheck(graph, {
      positions: createTransientView(graph, 'positions', 'float32x2', 1),
      timestamps,
      trackOffsets,
      output: {status: {invalidCount}}
    });
    const memory = createCPUGraphMemory();
    memory.write(timestamps, Float32Array.of(NaN));
    memory.write(trackOffsets, Uint32Array.of(0, 1));
    memory.write(invalidCount, Uint32Array.of(77, 99));
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(invalidCount))).toEqual([1, 99]);
  } finally {
    device.destroy();
  }
});
