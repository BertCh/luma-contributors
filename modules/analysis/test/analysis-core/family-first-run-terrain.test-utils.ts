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
  GPUTerrainDerivatives, getGPUTerrainDerivativesParameterValues
} from '@luma-contributors/analysis/gpu-terrain';
import {makeGPURasterBand, makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';

export async function runTerrainDerivativesExample() {
  return runFirstComputation('first-terrain', ({graph, float32}) => {
    const grid = makeGPURasterMetadata({width: 3, height: 3, cellSize: 30, origin: [0, 90]});
    const elevations = float32('elevations', Float32Array.of(0, 30, 60, 0, 30, 60, 0, 30, 60));
    const elevation = makeGPURasterBand(elevations.view, grid, {id: 'elevation'});
    const parameters = float32('parameters', getGPUTerrainDerivativesParameterValues({
      azimuthDegrees: 270, altitudeDegrees: 45
    }, grid));
    const slope = float32('slope', new Float32Array(9).fill(-1));
    const hillshade = float32('hillshade', new Float32Array(9));
    graph.add(new GPUTerrainDerivatives({
      grid, elevation, parameters: parameters.view,
      output: {slope: slope.view, hillshade: hillshade.view}
    }));
    return async () => ({slope: await slope.read(), hillshade: await hillshade.read()});
  });
}

const terrain = await runTerrainDerivativesExample();
console.log(terrain.slope[4], terrain.hillshade[4]);
// 45, approximately 1 (finite float32 results)
// END LITERAL FENCES

export {terrain};

export async function runTerrainLightingControlExample(azimuthDegrees: number, flat = false) {
  return runFirstComputation('terrain-lighting-control', ({graph, float32}) => {
    const grid = makeGPURasterMetadata({width: 3, height: 3, cellSize: 30, origin: [0, 90]});
    const elevations = float32('elevations', flat
      ? new Float32Array(9)
      : Float32Array.of(0, 30, 60, 0, 30, 60, 0, 30, 60));
    const elevation = makeGPURasterBand(elevations.view, grid, {id: 'elevation'});
    const parameters = float32('parameters', getGPUTerrainDerivativesParameterValues({
      azimuthDegrees, altitudeDegrees: 45
    }, grid));
    const slope = float32('slope', new Float32Array(9).fill(-1));
    const hillshade = float32('hillshade', new Float32Array(9).fill(-1));
    graph.add(new GPUTerrainDerivatives({
      grid, elevation, parameters: parameters.view,
      output: {slope: slope.view, hillshade: hillshade.view}
    }));
    return async () => ({slope: await slope.read(), hillshade: await hillshade.read()});
  });
}