// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {addVisibilityGraphRecipe} from '../../../src/gpu-network/visibility-graph/visibility-graph';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('W5-01 preserves degree and metrics destinations supplied through output', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const degree = createTransientView(graph, 'degree', 'uint32', 3);
    const metrics = createTransientView(graph, 'metrics', 'float32x4', 3);
    const zeroWeightCycle = createTransientView(graph, 'cycle', 'uint32', 1);
    const destinations = {degree, metrics, status: {zeroWeightCycle}};
    const result = addVisibilityGraphRecipe(graph, {
      mode: 'csr',
      positions: createTransientView(graph, 'positions', 'float32x2', 3),
      segments: createTransientView(graph, 'segments', 'float32x4', 1),
      maximumDistance: 1.1,
      edgeCapacity: 4,
      output: destinations
    });
    expect(result.outputs.degree).toBe(degree);
    expect(result.outputs.metrics).toBe(metrics);
    expect(result.status.zeroWeightCycle).toBe(zeroWeightCycle);
  } finally {
    device.destroy();
  }
});

it('W5-01 rejects the retired outputs prop before preflight or construction mutation', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = {
      mode: 'csr' as const,
      positions: createTransientView(graph, 'positions', 'float32x2', 3),
      segments: createTransientView(graph, 'segments', 'float32x4', 1),
      maximumDistance: 1.1,
      edgeCapacity: 4,
      output: {metrics: createTransientView(graph, 'first', 'float32x4', 3)},
      outputs: {metrics: createTransientView(graph, 'second', 'float32x4', 3)}
    };
    const state = graph as unknown as {nodes: unknown[]; buffers: Map<string, unknown>};
    const bufferCount = state.buffers.size;
    for (const invoke of [
      () => addVisibilityGraphRecipe.getExecutionPlan(graph, props),
      () => addVisibilityGraphRecipe(graph, props)
    ]) {
      expect(invoke).toThrow(/outputs was removed; use output/);
      expect(state.nodes).toHaveLength(0);
      expect(state.buffers.size).toBe(bufferCount);
    }
  } finally {
    device.destroy();
  }
});

it('W5-01 captures getter-backed destination maps once and retains their views', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let canonicalReads = 0;
  try {
    const degree = createTransientView(graph, 'degree', 'uint32', 3);
    const metrics = createTransientView(graph, 'metrics', 'float32x4', 3);
    const canonical = {degree, metrics};
    const result = addVisibilityGraphRecipe(graph, {
      mode: 'csr',
      positions: createTransientView(graph, 'positions', 'float32x2', 3),
      segments: createTransientView(graph, 'segments', 'float32x4', 1),
      maximumDistance: 1.1,
      edgeCapacity: 4,
      get output() {
        canonicalReads++;
        return canonical;
      }
    });
    expect(canonicalReads).toBe(1);
    canonical.degree = createTransientView(graph, 'replacement-degree', 'uint32', 3);
    canonical.metrics = createTransientView(graph, 'replacement-metrics', 'float32x4', 3);
    expect(result.outputs.degree).toBe(degree);
    expect(result.outputs.metrics).toBe(metrics);
  } finally {
    device.destroy();
  }
});

it('B2 reads nested destination leaves once', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const metrics = createTransientView(graph, 'metrics', 'float32x4', 3);
    const complete = createTransientView(graph, 'converged', 'uint32', 1);
    let metricReads = 0;
    let statusReads = 0;
    let firstMetrics: GraphDataView<'float32x4'> | undefined;
    let firstStatus: GraphDataView<'uint32'> | undefined;
    const status = {
      get converged() {
        statusReads++;
        const view = new GraphDataView(complete.buffer, {
          format: 'uint32',
          length: 1,
          byteOffset: 0,
          byteStride: 4,
          rowByteLength: 4
        });
        firstStatus ??= view;
        return view;
      }
    };
    const shared = {
      get metrics() {
        metricReads++;
        const view = new GraphDataView(metrics.buffer, {
          format: 'float32x4',
          length: 3,
          byteOffset: 0,
          byteStride: 16,
          rowByteLength: 16
        });
        firstMetrics ??= view;
        return view;
      },
      status
    };
    const result = addVisibilityGraphRecipe(graph, {
      mode: 'csr',
      positions: createTransientView(graph, 'positions', 'float32x2', 3),
      segments: createTransientView(graph, 'segments', 'float32x4', 1),
      maximumDistance: 1.1,
      edgeCapacity: 4,
      output: shared
    });
    expect(metricReads).toBe(1);
    expect(statusReads).toBe(1);
    expect(result.outputs.metrics).toBe(firstMetrics);
    expect(result.status.converged).toBe(firstStatus);
  } finally {
    device.destroy();
  }
});
