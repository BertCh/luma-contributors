// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {Buffer, Device} from '@luma.gl/core';
import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createNullWebGPUDevice,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
export type WaveFormat = 'uint32' | 'float32' | 'float32x2' | 'float32x4' | 'uint32x4' | 'uint32x2';
export function createWaveHarness(gpuDevice?: Device) {
  const device = gpuDevice ?? createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const buffers = new Map<GraphDataView, Buffer>();
  let serial = 0;
  let compiled: ReturnType<typeof graph.compile> | undefined;
  const allocate = <Format extends WaveFormat>(
    name: string,
    format: Format,
    length: number,
    values?: readonly number[]
  ) => {
    const width = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
    const data = format.startsWith('uint')
      ? Uint32Array.from(values ?? new Array(length * width).fill(99))
      : Float32Array.from(values ?? new Array(length * width).fill(99));
    const buffer = gpuDevice ? createOutputBuffer(device, length * width) : undefined;
    const view = buffer
      ? importGraphBuffer(graph, `${name}-${serial++}`, buffer, format, length)
      : createTransientView(graph, `${name}-${serial++}`, format, length);
    if (buffer) {
      buffer.write(data);
      buffers.set(view, buffer);
    } else memory.write(view, data);
    return view;
  };
  return {
    device,
    graph,
    memory,
    allocate,
    write(view: GraphDataView, values: Uint32Array | Float32Array) {
      const buffer = buffers.get(view);
      if (buffer) buffer.write(values);
      else memory.write(view, values);
    },
    async read(view: GraphDataView) {
      const words = (view.length * view.rowByteLength) / 4;
      const buffer = buffers.get(view);
      return buffer
        ? view.format.startsWith('uint')
          ? readUint32(buffer, words)
          : readFloat32(buffer, words)
        : Array.from(
            view.format.startsWith('uint')
              ? memory.readUint32(view, words)
              : memory.readFloat32(view, words)
          );
    },
    async run(nodes: readonly GPUCommandNode[]) {
      if (!gpuDevice) {
        executeCommandNodesOnCPU(nodes, memory);
        return () => {};
      }
      if (!compiled) {
        for (const node of nodes) graph.add(node);
        compiled = graph.compile();
      }
      submitGraph(device, compiled, undefined);
      return () => {
        compiled?.destroy();
        compiled = undefined;
      };
    },
    destroy() {
      compiled?.destroy();
      buffers.forEach(buffer => buffer.destroy());
      if (!gpuDevice) device.destroy();
    }
  };
}
