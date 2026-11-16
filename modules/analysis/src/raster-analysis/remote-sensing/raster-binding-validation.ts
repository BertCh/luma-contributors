// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getGPUFFT2DSupport,
  makeGPUFFT2DStats,
  type GraphDataView,
  type GPUCommandGraph
} from '@luma.gl/graph/gpu-core';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Preflight caller bindings and every derived allocation before declaring resources or nodes. */
export function validateRasterStorageBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  views: readonly GraphDataView[],
  scratch: Readonly<Record<string, number>> = {}
): void {
  validateGraphChunkBindingRanges(
    graph,
    id,
    Object.fromEntries(views.map((view, index) => [`view${index}`, view]))
  );
  for (const [name, byteLength] of Object.entries(scratch)) {
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      name,
      byteLength,
      'reduce the raster dimensions or refinement window'
    );
    validateCapacity(
      id,
      `${name} byteLength (maxBufferSize)`,
      byteLength,
      0,
      graph.device.limits.maxBufferSize
    );
  }
}

/** Includes the published FFT's nested complex scratch and immutable uniform allocations. */
export function validateRasterRegistrationFFT<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  width: number,
  height: number,
  batchCount = 1
): void {
  const support = getGPUFFT2DSupport(graph.device, {width, height, batchCount});
  validateRequirement(
    id,
    support.supported,
    `grid dimensions exceed GPUFFT2D device limits: ${support.reason}`
  );
  const statistics = makeGPUFFT2DStats(width, height, batchCount);
  validateRasterStorageBindings(graph, id, [], {
    'GPUFFT2D scratch': statistics.scratchBufferByteLength
  });
  validateCapacity(
    id,
    'GPUFFT2D parameter allocation byteLength (maxBufferSize)',
    statistics.parameterBufferByteLength / statistics.parameterBufferCount,
    1,
    graph.device.limits.maxBufferSize
  );
  validateCapacity(
    id,
    'GPUFFT2D parameter binding byteLength (maxUniformBufferBindingSize)',
    statistics.parameterBufferByteLength / statistics.parameterBufferCount,
    1,
    graph.device.limits.maxUniformBufferBindingSize
  );
}
