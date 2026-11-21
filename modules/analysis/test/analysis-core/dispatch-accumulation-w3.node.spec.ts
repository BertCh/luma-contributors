// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it, vi} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../src/testing/gpu-contributor-test-utils';
import * as kernels from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createGraphChunkGroupKernelNode} from '../../src/analysis-core/kernels/graph-chunk-groups';

it('WGS-11 uses overflow-safe unsigned ceiling division', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    kernels.recordWGSLKernelNodes(graph);
    kernels.createWGSLActiveCountDispatch(graph, {
      id: 'maximum',
      operation: 'Maximum',
      count: createTransientView(graph, 'count', 'uint32', 1),
      maximumItemCount: 0xffffffff,
      workgroupSize: 256
    });
    const record = kernels.getWGSLKernelNodeRecord(graph, 'maximum')!;
    // ceil((2^32 - 1)/256) = 16777216; ceil(16777216/65535) = 257.
    expect(record.constants['MAXIMUM_X']).toBe(65535);
    expect(record.constants['MAXIMUM_Y']).toBe(257);
    expect(record.body).toContain('invocationCount % TARGET_WORKGROUP_SIZE');
    expect(record.body).not.toContain('invocationCount + TARGET_WORKGROUP_SIZE');
  } finally {
    device.destroy();
  }
});

it('WGS-14 rejects generated namespace collisions in both builders', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const view = createTransientView(graph, 'values', 'uint32', 1);
    for (const builder of [kernels.createWGSLKernelNode, createGraphChunkGroupKernelNode]) {
      expect(() =>
        builder(graph, {
          id: 'collision',
          operation: 'Collision',
          invocationCount: 1,
          bindings: ['values', 'valuesOffset'].map(name => ({
            name,
            view,
            type: 'u32',
            access: 'read'
          })),
          body: ''
        })
      ).toThrow(/collision.*values.*valuesOffset/);
    }
  } finally {
    device.destroy();
  }
});

it('WGS-15 statically references every chunk storage binding', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const view = createTransientView(graph, 'values', 'uint32', 1);
    for (const builder of [kernels.createWGSLKernelNode, createGraphChunkGroupKernelNode]) {
      const node = builder(graph, {
        id: 'unused',
        operation: 'Unused',
        invocationCount: 1,
        bindings: ['values', 'weights'].map(name => ({
          name,
          view,
          type: 'u32',
          access: 'read_write'
        })),
        body: 'values[valuesOffset + index] = 1u;'
      });
      const source = getGPUCommandNodeKernelSources([node])[0]!.source;
      expect(source).toContain('_ = &weights;');
      const names = new WgslReflect(source).entry.compute[0]!.resources.map(
        resource => resource.name
      );
      expect(names).toContain('values');
      expect(names).toContain('weights');
    }
  } finally {
    device.destroy();
  }
});

it('ENT-03 shares opt-in records between independently loaded modules in both directions', async () => {
  vi.resetModules();
  const other = await import('../../src/analysis-core/kernels/wgsl-kernel-nodes');
  const device = createNullWebGPUDevice();
  try {
    for (const [reader, writer] of [
      [kernels, other],
      [other, kernels]
    ]) {
      const graph = new GPUCommandGraph(device);
      const second = new GPUCommandGraph(device);
      reader.recordWGSLKernelNodes(graph);
      const props = {
        id: 'recorded',
        operation: 'Recorded',
        invocationCount: 1,
        bindings: [],
        body: ''
      };
      const node = writer.createWGSLKernelNode(graph, props);
      writer.createWGSLKernelNode(second, props);
      expect(writer.isRecordingWGSLKernelNodes(graph)).toBe(true);
      expect(reader.getWGSLKernelNodeRecord(graph, props.id)!.source).toBe(
        getGPUCommandNodeKernelSources([node])[0]!.source
      );
      expect(reader.getWGSLKernelNodeRecord(second, props.id)).toBeUndefined();
      expect(writer.isRecordingWGSLKernelNodes(second)).toBe(false);
    }
  } finally {
    device.destroy();
  }
});

it('COMMON-W3 preserves foreign single-chunk and empty scratch vector identities', async () => {
  const {GraphVectorView} = await import('@luma.gl/graph/gpu-core');
  const device = createNullWebGPUDevice();
  try {
    for (const lengths of [[], [0], [1], [0, 1, 0]]) {
      const graph = new GPUCommandGraph(device);
      const length = lengths.reduce((total, value) => total + value, 0);
      const template = new GraphVectorView({
        id: 'foreign',
        name: 'foreign',
        format: 'float32',
        length,
        valueLength: length,
        stride: 1,
        byteStride: 4,
        rowByteLength: 4,
        data: lengths.map((count, index) =>
          createTransientView(graph, `chunk-${index}`, 'float32', count)
        )
      });
      Object.setPrototypeOf(template, Object.prototype);
      expect(template instanceof GraphVectorView).toBe(false);
      const result = kernels.createTransientUint32Rows(graph, 'scratch', template);
      expect(result).toBeInstanceOf(GraphVectorView);
      expect(
        (result as InstanceType<typeof GraphVectorView>).data.map(chunk => chunk.length)
      ).toEqual(lengths);
    }
  } finally {
    device.destroy();
  }
});

it('UDF-06 grouped scales validate row topology before declaring nodes', async () => {
  const {createFixedPointGroupedScaleNodes} = await import(
    '../../src/analysis-core/kernels/fixed-point-accumulator'
  );
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const values = createTransientView(graph, 'values', 'float32', 2);
    const cellRows = createTransientView(graph, 'cells', 'uint32', 1);
    const scale = createTransientView(graph, 'scale', 'uint32', 2);
    expect(() =>
      createFixedPointGroupedScaleNodes(graph, {
        id: 'groups',
        operation: 'Groups',
        values,
        cellRows,
        scale
      })
    ).toThrow(/groups.*cellRows/);
  } finally {
    device.destroy();
  }
});
