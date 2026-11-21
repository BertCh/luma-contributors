// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// biome-ignore-all format: Preserve the literal published Markdown fence.

// BEGIN LITERAL FENCES
import {Buffer, luma} from '@luma.gl/core';
import {webgpuAdapter} from '@luma.gl/core/webgpu';
import {GPUCommandGraph} from '@luma.gl/graph';
import {GPUPointDensity} from '@luma-contributors/analysis/gpu-spatial-analysis';

export async function runPointDensityExample() {
  const device = await luma.createDevice({
    type: 'webgpu',
    adapters: [webgpuAdapter]
  });
  const buffers: Buffer[] = [];
  let compiled: Awaited<ReturnType<GPUCommandGraph['compileAsync']>> | undefined;
  try {
    const graph = new GPUCommandGraph(device, {id: 'first-density'});
    const createBuffer = (id: string, data: Float32Array | Uint32Array) => {
      const buffer = device.createBuffer({
        id,
        data,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      buffers.push(buffer);
      const handle = graph.importBuffer(
        {id, byteLength: buffer.byteLength, usage: buffer.usage},
        buffer
      );
      return {buffer, handle};
    };
    const positions = graph.createDataView(
      createBuffer(
        'positions',
        Float32Array.of(0.25, 0.25, 0.75, 0.75, 0.25, 1.25, 1.25, 1.25, 3, 3)
      ).handle,
      {format: 'float32x2', length: 5}
    );
    const bounds = graph.createDataView(createBuffer('bounds', Float32Array.of(0, 0, 2, 2)).handle, {
      format: 'float32',
      length: 4
    });
    const valuesAllocation = createBuffer('values', new Float32Array(4));
    const overflowAllocation = createBuffer('overflow', new Uint32Array(1));
    const values = graph.createDataView(valuesAllocation.handle, {format: 'float32', length: 4});
    const overflow = graph.createDataView(overflowAllocation.handle, {format: 'uint32', length: 1});
    graph.add(
      new GPUPointDensity({
        positions,
        bounds,
        gridSize: [2, 2],
        statistic: 'count',
        output: {values, status: {overflow}}
      })
    );
    compiled = await graph.compileAsync();
    const commandEncoder = device.createCommandEncoder();
    compiled.encode(commandEncoder, {parameters: undefined});
    device.submit(commandEncoder.finish());
    const valueBytes = await valuesAllocation.buffer.readAsync();
    const overflowBytes = await overflowAllocation.buffer.readAsync();
    const result = {
      values: new Float32Array(valueBytes.buffer, valueBytes.byteOffset, 4).slice(),
      overflow: new Uint32Array(overflowBytes.buffer, overflowBytes.byteOffset, 1)[0]
    };
    if (result.overflow !== 0) throw new Error('Point density output is incomplete');
    return result;
  } finally {
    compiled?.destroy();
    for (const buffer of buffers) buffer.destroy();
    device.destroy();
  }
}

const density = await runPointDensityExample();
console.log(Array.from(density.values), density.overflow);