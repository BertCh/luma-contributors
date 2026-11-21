// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {DynamicBuffer} from '@luma.gl/core/dynamic';
import {GPUData} from '@luma.gl/graph/gpu-data';
import {
  createGPUCopyCommandNode,
  GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUStudyWindowPreparation} from '../../../src/gpu-spatial-analysis/study-window/gpu-study-window-preparation';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

for (const dynamic of [false, true])
  it(`review B2 study-window retry uses actual fresh producer view dynamic=${dynamic}`, () => {
    const device = createNullWebGPUDevice();
    const original = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const wrappers: DynamicBuffer[] = [];
    const input = <Format extends 'float32x2' | 'uint32'>(
      name: string,
      format: Format,
      length: number
    ): GraphDataView<Format> => {
      const buffer = device.createBuffer({
        byteLength: 512,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      buffers.push(buffer);
      const wrapper = new DynamicBuffer(device, {buffer, ownsBuffer: false});
      wrappers.push(wrapper);
      return original.importGPUData(
        name,
        new GPUData({
          buffer: dynamic ? wrapper : buffer,
          format,
          length,
          byteOffset: 256,
          ownsBuffer: false
        })
      );
    };
    const inputs = {
      positions: input('positions', 'float32x2', 3),
      featureOffsets: input('features', 'uint32', 2),
      polygonOffsets: input('polygons', 'uint32', 2),
      ringOffsets: input('rings', 'uint32', 2)
    };
    const prepared = new GPUStudyWindowPreparation({
      id: 'retry-window',
      ...inputs,
      samplingGrid: [1, 1]
    });
    const created: Buffer[] = [];
    const allocate = device.createBuffer.bind(device);
    const spy = vi.spyOn(device, 'createBuffer').mockImplementation(settings => {
      if (
        typeof settings === 'object' &&
        settings !== null &&
        'id' in settings &&
        settings.id === 'retry-window-edges'
      )
        throw new Error('injected edges');
      const buffer = allocate(settings);
      created.push(buffer);
      return buffer;
    });
    try {
      try {
        expect(() => original.add(prepared)).toThrow('injected edges');
      } finally {
        spy.mockRestore();
      }
      expect(prepared.isDeclaredIn(original)).toBe(false);
      expect(created.every(buffer => buffer.destroyed)).toBe(true);
      const fresh = new GPUCommandGraph(device);
      const rebind = <Format extends 'float32x2' | 'uint32'>(
        name: string,
        source: GraphDataView<Format>,
        index: number
      ) => {
        const handle = fresh.importBuffer(
          {id: `canonical-${name}`, byteLength: 512, usage: Buffer.STORAGE},
          buffers[index]
        );
        return fresh.createDataView(handle, source);
      };
      const mapped = {
        positions: rebind('positions', inputs.positions, 0),
        featureOffsets: rebind('features', inputs.featureOffsets, 1),
        polygonOffsets: rebind('polygons', inputs.polygonOffsets, 2),
        ringOffsets: rebind('rings', inputs.ringOffsets, 3)
      };
      expect(() =>
        prepared.setRetryViews(fresh, {
          ...mapped,
          positions: fresh.createDataView(mapped.positions.buffer, {
            ...mapped.positions,
            byteOffset: 264
          })
        })
      ).toThrow(/positions.*same physical range/);
      prepared.setRetryViews(fresh, mapped);
      mapped.positions = fresh.createDataView(mapped.positions.buffer, {
        ...mapped.positions,
        byteOffset: 264
      });
      const nodes = prepared.getCommandNodes(fresh);
      const copy = nodes.find(node => node.id === 'retry-window-snapshot-positions')!;
      const resource = copy.resources![0];
      expect('buffer' in resource && 'buffer' in resource.buffer).toBe(true);
      if (!('buffer' in resource) || !('buffer' in resource.buffer))
        throw new Error('snapshot must read a data view');
      const positions = resource.buffer;
      expect(positions.buffer).toBe(mapped.positions.buffer);
      expect(positions.byteOffset).toBe(256); // Captured mapping survives caller descriptor mutation.
      const writer = createWGSLKernelNode(fresh, {
        id: 'window-producer',
        operation: 'StudyWindowRetryReview',
        invocationCount: 1,
        bindings: [{name: 'positions', view: positions, type: 'f32', access: 'read_write'}],
        body: 'positions[positionsOffset] = 0.0; positions[positionsOffset + 1u] = 0.0; positions[positionsOffset + 2u] = 2.0; positions[positionsOffset + 3u] = 0.0; positions[positionsOffset + 4u] = 0.0; positions[positionsOffset + 5u] = 2.0;'
      });
      // Preserve actual graph resources/hazards in host encoding shells: NullDevice has no compute
      // pipelines. The real emitted producer and snapshot copy are executed below in the interpreter.
      for (const node of [writer, copy])
        fresh.add(
          createGPUCopyCommandNode({
            id: node.id,
            resources: node.resources,
            compile: () => ({encode: () => {}})
          })
        );
      const compiled = fresh.compile();
      const encoder = device.createCommandEncoder();
      try {
        expect(() => compiled.encode(encoder, {parameters: undefined})).not.toThrow();
      } finally {
        encoder.destroy();
        compiled.destroy();
      }
      const memory = createCPUGraphMemory();
      executeCommandNodesOnCPU([writer, {...copy, condition: undefined}], memory);
      const destination = copy.resources![1];
      if (!('buffer' in destination) || !('buffer' in destination.buffer))
        throw new Error('snapshot must write a data view');
      expect(Array.from(memory.readFloat32(destination.buffer))).toEqual([0, 0, 2, 0, 0, 2]);
    } finally {
      spy.mockRestore();
      prepared.destroy();
      for (const wrapper of wrappers) wrapper.destroy();
      for (const buffer of buffers) {
        expect(buffer.destroyed).toBe(false);
        buffer.destroy();
      }
      device.destroy();
    }
  });
