// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUTrajectoryHampel} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-hampel';
import {GPUUtilizationVolumeIsopleths} from '../../../src/gpu-spatial-analysis/trajectory-home-range/gpu-utilization-volume-isopleths';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

it('rejects aligned overlap of co-bound Hampel fields even when logical writes are disjoint', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const packed = createTransientView(graph, 'packed', 'uint32', 4);
    const filtered = graph.createDataView(packed.buffer, {format: 'float32', length: 2});
    const flags = graph.createDataView(packed.buffer, {format: 'uint32', length: 2, byteOffset: 8});
    expect(
      () =>
        new GPUTrajectoryHampel({
          id: 'hampel',
          values: createTransientView(graph, 'values', 'float32', 2),
          trackOffsets: createTransientView(graph, 'tracks', 'uint32', 2),
          parameters: createTransientView(graph, 'parameters', 'float32', 4),
          windowSize: 1,
          output: {filtered, flags}
        })
    ).toThrow(/hampel .*aligned binding.*overlap/);
  } finally {
    device.destroy();
  }
});

it('permits adjacent isopleth total mass and volume fields written by separate passes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const packed = createTransientView(graph, 'packed', 'float32', 3);
    const volume = graph.createDataView(packed.buffer, {format: 'float32', length: 2});
    const totalMass = graph.createDataView(packed.buffer, {
      format: 'float32',
      length: 1,
      byteOffset: 8
    });
    recordWGSLKernelNodes(graph);
    const contributor = new GPUUtilizationVolumeIsopleths({
      density: createTransientView(graph, 'density', 'float32', 2),
      parameters: createTransientView(graph, 'parameters', 'float32', 4),
      output: {volume, totalMass}
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    for (const node of nodes) {
      const record = getWGSLKernelNodeRecord(graph, node.id);
      if (!record) continue;
      for (const writable of record.bindings.filter(binding => binding.access === 'read_write')) {
        for (const readable of record.bindings.filter(binding => binding.access === 'read')) {
          expect(writable.view.buffer, `${node.id}: ${writable.name}/${readable.name}`).not.toBe(
            readable.view.buffer
          );
        }
      }
    }
  } finally {
    device.destroy();
  }
});

it('permits aligned disjoint writable Hampel fields in one graph buffer', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const packed = createTransientView(graph, 'packed', 'uint32', 66);
    const filtered = graph.createDataView(packed.buffer, {format: 'float32', length: 2});
    const flags = graph.createDataView(packed.buffer, {
      format: 'uint32',
      length: 2,
      byteOffset: 256
    });
    const contributor = new GPUTrajectoryHampel({
      values: createTransientView(graph, 'values', 'float32', 2),
      trackOffsets: createTransientView(graph, 'tracks', 'uint32', 2),
      parameters: createTransientView(graph, 'parameters', 'float32', 4),
      windowSize: 1,
      output: {filtered, flags}
    });
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});
