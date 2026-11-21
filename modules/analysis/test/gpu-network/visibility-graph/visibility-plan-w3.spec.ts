// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {addVisibilityGraphRecipe} from '../../../src/gpu-network/visibility-graph/visibility-graph';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('SCL-04 automatic feasible sampling uses the planned pivots on a five-node path', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const positions = createInputBuffer(device, Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0, 4, 0));
  const segments = createInputBuffer(device, Float32Array.of(100, 100, 101, 100));
  const closeness = createOutputBuffer(device, 3);
  const approximation = createOutputBuffer(device, 1);
  const converged = createOutputBuffer(device, 1);
  const props = {
    mode: 'csr' as const,
    positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', 5),
    segments: importGraphBuffer(graph, 'segments', segments, 'float32x4', 1),
    maximumDistance: 1.1,
    edgeCapacity: 8,
    laneCount: 3,
    maximumIterations: 5,
    output: {
      closeness: importGraphBuffer(graph, 'closeness', closeness, 'float32', 3),
      status: {
        approximation: importGraphBuffer(graph, 'approximation', approximation, 'uint32', 1),
        converged: importGraphBuffer(graph, 'converged', converged, 'uint32', 1)
      }
    }
  };
  const plan = addVisibilityGraphRecipe.getExecutionPlan(graph, props);
  const result = addVisibilityGraphRecipe(graph, props);
  expect(result.provenance.executionPlan).toEqual(plan);
  expect(plan.pivotCount).toBe(3);
  const compiled = graph.compile();
  try {
    await submitGraph(device, compiled, undefined);
    expect(await readUint32(approximation, 1)).toEqual([1]);
    expect(await readUint32(converged, 1)).toEqual([1]);
    // Systematic pivots floor(i*5/3) = [0,1,3]. Four reachable other nodes have
    // total hop depths [10,7,7], hence closeness [4/10,4/7,4/7].
    const values = await readFloat32(closeness, 3);
    expect(values.every(Number.isFinite)).toBe(true);
    for (const [index, expected] of [0.4, 4 / 7, 4 / 7].entries())
      expect(values[index]).toBeCloseTo(expected);
  } finally {
    compiled.destroy();
    for (const buffer of [positions, segments, closeness, approximation, converged])
      buffer.destroy();
  }
});

it('W5-01 writes caller-owned degree and metrics destinations on a three-node path', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const positions = createInputBuffer(device, Float32Array.of(0, 0, 1, 0, 2, 0));
  const segments = createInputBuffer(device, Float32Array.of(100, 100, 101, 100));
  const degree = createOutputBuffer(device, 3);
  const metrics = createOutputBuffer(device, 12);
  const degreeView = importGraphBuffer(graph, 'degree', degree, 'uint32', 3);
  const metricsView = importGraphBuffer(graph, 'metrics', metrics, 'float32x4', 3);
  const result = addVisibilityGraphRecipe(graph, {
    mode: 'csr',
    positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', 3),
    segments: importGraphBuffer(graph, 'segments', segments, 'float32x4', 1),
    maximumDistance: 1.1,
    edgeCapacity: 4,
    output: {degree: degreeView, metrics: metricsView}
  });
  expect(result.outputs.degree).toBe(degreeView);
  expect(result.outputs.metrics).toBe(metricsView);
  const compiled = graph.compile();
  try {
    await submitGraph(device, compiled, undefined);
    expect(await readUint32(degree, 3)).toEqual([1, 2, 1]);
    // Path endpoints have hop depths 1+2=3; middle has 1+1=2, each reaches two others.
    const expected = [1, 2, 3, 2 / 3, 2, 2, 2, 1, 1, 2, 3, 2 / 3];
    const values = await readFloat32(metrics, 12);
    expect(values).toHaveLength(expected.length);
    expect(values.every(Number.isFinite)).toBe(true);
    values.forEach((value, index) => expect(value).toBeCloseTo(expected[index]));
  } finally {
    compiled.destroy();
    for (const buffer of [positions, segments, degree, metrics]) buffer.destroy();
  }
});
