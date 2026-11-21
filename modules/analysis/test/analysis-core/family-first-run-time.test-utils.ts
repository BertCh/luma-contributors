// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// biome-ignore-all format: Preserve the literal published Markdown fences.

// BEGIN LITERAL FENCES
import {Buffer, type Device, luma} from '@luma.gl/core';
import {webgpuAdapter} from '@luma.gl/core/webgpu';
import {GPUCommandGraph} from '@luma.gl/graph';
import {readPackedBuffer} from '@luma-contributors/analysis/analysis-core';

function createFirstRunContext(device: Device, graph: GPUCommandGraph, buffers: Buffer[]) {
  const allocate = (id: string, data: Float32Array | Uint32Array) => {
    const buffer = device.createBuffer({
      id, data, usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    const handle = graph.importBuffer(
      {id, byteLength: buffer.byteLength, usage: buffer.usage}, buffer
    );
    return {buffer, handle};
  };
  const float32 = (id: string, data: Float32Array) => {
    const {buffer, handle} = allocate(id, data);
    return {
      handle,
      view: graph.createDataView(handle, {format: 'float32', length: data.length}),
      read: (elementCount = data.length) => readPackedBuffer(buffer, 0, 'float32', elementCount)
    };
  };
  const uint32 = (id: string, data: Uint32Array) => {
    const {buffer, handle} = allocate(id, data);
    return {
      view: graph.createDataView(handle, {format: 'uint32', length: data.length}),
      read: (elementCount = data.length) => readPackedBuffer(buffer, 0, 'uint32', elementCount)
    };
  };
  return {graph, float32, uint32};
}

export async function runFirstComputationOnDevice<Result>(
  device: Device,
  id: string,
  construct: (context: ReturnType<typeof createFirstRunContext>) => () => Promise<Result>
) {
  const buffers: Buffer[] = [];
  let compiled: Awaited<ReturnType<GPUCommandGraph['compileAsync']>> | undefined;
  let encoder: ReturnType<Device['createCommandEncoder']> | undefined;
  try {
    const graph = new GPUCommandGraph(device, {id});
    const readResult = construct(createFirstRunContext(device, graph, buffers));
    compiled = await graph.compileAsync();
    encoder = device.createCommandEncoder();
    compiled.encode(encoder, {parameters: undefined});
    const commandBuffer = encoder.finish();
    encoder = undefined; // Successful finish releases the encoder and transfers its resources.
    device.submit(commandBuffer);
    return await readResult();
  } finally {
    encoder?.destroy(); // Encoding or finishing may throw before releasing transient resources.
    compiled?.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

export async function runFirstComputation<Result>(
  id: string,
  construct: (context: ReturnType<typeof createFirstRunContext>) => () => Promise<Result>
) {
  const device = await luma.createDevice({type: 'webgpu', adapters: [webgpuAdapter]});
  try {
    return await runFirstComputationOnDevice(device, id, construct);
  } finally {
    device.destroy();
  }
}

import {
  GPUTimeWindowFilter, getGPUTimeWindowFilterParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';

export async function runTimeWindowFilterExample() {
  return runFirstComputation('first-time-window', ({graph, float32, uint32}) => {
    const t = Date.UTC(2026, 10, 19);
    const events = [t, t + 1000, t + 2000];
    const timestamps = float32('timestamps', Float32Array.from(events, time => time - t));
    const window = float32('window', getGPUTimeWindowFilterParameterValues({
      start: (t + 1000) - t, end: (t + 1000) - t
    }));
    const rows = uint32('rows', new Uint32Array(3));
    const count = uint32('count', new Uint32Array(1));
    const requiredCount = uint32('required-count', new Uint32Array(1));
    const overflow = uint32('overflow', new Uint32Array(1).fill(0xffffffff));
    graph.add(new GPUTimeWindowFilter({
      timestamps: timestamps.view, window: window.view,
      output: {rows: rows.view, status: {
        count: count.view, requiredCount: requiredCount.view, overflow: overflow.view
      }}
    }));
    return async () => {
      const acceptedCount = (await count.read())[0];
      const required = (await requiredCount.read())[0];
      const lost = (await overflow.read())[0];
      if (lost !== 0) throw new Error('Time-window output is incomplete; resize and rerun');
      return {rows: await rows.read(acceptedCount), count: acceptedCount, requiredCount: required, overflow: lost};
    };
  });
}

const timeWindow = await runTimeWindowFilterExample();
console.log(Array.from(timeWindow.rows), timeWindow.count, timeWindow.requiredCount, timeWindow.overflow);
// [1], 1, 1, 0
// END LITERAL FENCES

export {timeWindow};