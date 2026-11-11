// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  getGPUKernelDensityParameterValues,
  GPUKernelDensity
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';

// example:hide-start
declare const device: Device;
/** Observations as x, y pairs in metres of a local frame. */
declare const observations: Float32Array;
/** Study extent in the same metres, a whole number of cells. */
declare const extent: readonly [number, number, number, number];
// example:hide-end
// 1. Buffers: your data and your output, in GPU memory you own.
const [minX, minY, maxX, maxY] = extent;
const cellSize = 125; // metres
const [width, height] = [(maxX - minX) / cellSize, (maxY - minY) / cellSize].map(Math.round);
const pointBuffer = device.createBuffer({usage: Buffer.STORAGE, data: observations});
const densityBuffer = device.createBuffer({
  usage: Buffer.STORAGE | Buffer.COPY_SRC, // COPY_SRC: read the result back
  byteLength: width * height * 4
});

// 2. Views: tell the graph what the bytes are.
const graph = new GPUCommandGraph<void>(device, {id: 'kernel-density'});
const positions = graph.createDataView(
  graph.importBuffer(
    {id: 'observations', byteLength: pointBuffer.byteLength, usage: pointBuffer.usage},
    pointBuffer
  ),
  {format: 'float32x2', length: observations.length / 2}
);
const density = graph.createDataView(
  graph.importBuffer(
    {id: 'density', byteLength: densityBuffer.byteLength, usage: densityBuffer.usage},
    densityBuffer
  ),
  {format: 'float32', length: width * height}
);

// 3. Build: one operation adds all of its kernels.
const grid = makeGPURasterMetadata({width, height, cellSize, origin: [minX, maxY]});
const values = getGPUKernelDensityParameterValues({bandwidth: 400}, grid);
const parameters = new GPUParameterBuffer(device, {
  id: 'kde-parameters',
  format: 'float32',
  length: values.length,
  values
});
graph.add(
  new GPUKernelDensity({
    positions,
    parameters: parameters.importToGraph(graph),
    grid,
    indexGridSize: [32, 32],
    indexBounds: extent,
    output: {values: density}
  })
);

// 4. Compile once: order the kernels, create the pipelines.
const compiled = await graph.compileAsync();

// 5. Encode and submit: this is all a frame costs.
const commandEncoder = device.createCommandEncoder();
compiled.encode(commandEncoder, {parameters: undefined});
device.submit(commandEncoder.finish());

// 6. Read back, only if the CPU needs the numbers. A layer can draw densityBuffer directly.
const bytes = await densityBuffer.readAsync();
const observationsPerSquareMetre = new Float32Array(bytes.buffer, bytes.byteOffset, width * height);

// A new bandwidth is a 36-byte write. Nothing recompiles.
parameters.write(getGPUKernelDensityParameterValues({bandwidth: 250}, grid));
// example:hide-start
export {observationsPerSquareMetre};
// example:hide-end
