// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  getGPUKernelDensityParameterValues,
  GPU_KERNEL_DENSITY_PARAMETER_LENGTH,
  GPUKernelDensity,
  GPUParameterBuffer
} from '@luma-contributors/analysis/gpu-spatial-analysis';

// example:hide-start
import type {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';

declare const device: Device;
declare const graph: GPUCommandGraph<void>;
declare const positions: GraphDataView<'float32x2'>;
declare const density: GraphDataView<'float32'>;
declare const grid: ReturnType<typeof makeGPURasterMetadata>;
declare const extent: readonly [number, number, number, number];
// example:hide-end
// Construct: props are validated here, once.
const parameters = new GPUParameterBuffer(device, {
  id: 'kde-parameters',
  format: 'float32',
  length: GPU_KERNEL_DENSITY_PARAMETER_LENGTH, // 9 float32 values
  values: getGPUKernelDensityParameterValues({bandwidth: 400}, grid)
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

// Compile, once: order the kernels, create the pipelines. The graph freezes.
const compiled = await graph.compileAsync();

// Every frame: record the fixed order and submit it. Nothing is re-planned.
function encodeFrame(): void {
  const commandEncoder = device.createCommandEncoder();
  compiled.encode(commandEncoder, {parameters: undefined});
  device.submit(commandEncoder.finish());
}

// Between frames: a new bandwidth is a buffer write, not a recompile.
function setBandwidth(bandwidth: number): void {
  parameters.write(getGPUKernelDensityParameterValues({bandwidth}, grid));
}
// example:hide-start
export {encodeFrame, setBandwidth};
// example:hide-end
