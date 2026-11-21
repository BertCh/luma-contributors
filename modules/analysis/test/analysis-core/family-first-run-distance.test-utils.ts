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
  GPUDistanceField, makeGPURasterMetadata, getGPUDistanceFieldParameterValues
} from '@luma-contributors/analysis/raster-analysis';

export async function runExactDistanceFieldExample() {
  return runFirstComputation('first-distance', ({graph, float32, uint32}) => {
    const grid = makeGPURasterMetadata({width: 3, height: 3, cellSize: 1, origin: [0, 3]});
    const seeds = float32('seeds', Float32Array.of(1.5, 1.5));
    const seedPositions = graph.createDataView(seeds.handle, {format: 'float32x2', length: 1});
    const parameters = float32('parameters', getGPUDistanceFieldParameterValues({
      maximumDistance: 5
    }, grid));
    const distances = float32('distances', new Float32Array(9));
    const allocation = uint32('allocation', new Uint32Array(9).fill(0xffffffff));
    const withinDistance = uint32('within-distance', new Uint32Array(9));
    graph.add(new GPUDistanceField({
      grid, seedPositions, parameters: parameters.view, mode: 'exact',
      output: {distances: distances.view, allocation: allocation.view, withinDistance: withinDistance.view}
    }));
    return async () => ({
      distances: await distances.read(), allocation: await allocation.read(),
      withinDistance: await withinDistance.read()
    });
  });
}

const distanceField = await runExactDistanceFieldExample();
console.log(Array.from(distanceField.distances));
// [sqrt(2), 1, sqrt(2), 1, 0, 1, sqrt(2), 1, sqrt(2)] within float32 tolerance
console.log(Array.from(distanceField.allocation), Array.from(distanceField.withinDistance));
// [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1]
// END LITERAL FENCES

export {distanceField};