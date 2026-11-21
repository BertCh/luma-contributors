// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// biome-ignore-all format: Preserve the literal published Markdown fences.

export const displayedPointCloudResults: [number, number[], number[]][] = [];
const console = {
  log(multiplier: number, means: number[], validity: number[]) {
    displayedPointCloudResults.push([multiplier, means.slice(), validity.slice()]);
  }
};

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

import {GPUPointCloudStatisticalFilter} from '@luma-contributors/analysis/pointcloud-analysis';

export async function runPointCloudStatisticalFilterExample(standardDeviations: number) {
  return runFirstComputation('first-point-cloud-filter', ({graph, float32, uint32}) => {
    const squaredDistances = float32('squared-distances', Float32Array.of(1, 1, 100, 0));
    const counts = uint32('counts', Uint32Array.of(1, 1, 1, 0));
    const meanDistances = float32('mean-distances', new Float32Array(4));
    const validity = uint32('validity', new Uint32Array(4));
    graph.add(new GPUPointCloudStatisticalFilter({
      squaredDistances: squaredDistances.view, counts: counts.view,
      capacity: 1, standardDeviations,
      output: {meanDistances: meanDistances.view, validity: validity.view}
    }));
    return async () => ({meanDistances: await meanDistances.read(), validity: await validity.read()});
  });
}

for (const multiplier of [2, 0, 1.5]) {
  const filtered = await runPointCloudStatisticalFilterExample(multiplier);
  console.log(multiplier, Array.from(filtered.meanDistances), Array.from(filtered.validity));
}
// 2:   [1, 1, 10, -1], [1, 1, 1, 0]
// 0:   [1, 1, 10, -1], [1, 1, 0, 0]
// 1.5: [1, 1, 10, -1], [1, 1, 1, 0]
// END LITERAL FENCES

export {};