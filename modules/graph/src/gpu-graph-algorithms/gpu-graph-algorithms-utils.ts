// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuGraph.

import type {Buffer} from '@luma.gl/core';
import {DynamicBuffer} from '@luma.gl/core/dynamic';
import type {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import type {GPUGraphAlgorithmsAdjacency} from './gpu-graph-topology';

/** Resolves a stable dynamic wrapper to the current concrete physical GPU allocation. @internal */
export function getPhysicalBuffer(
  chunk: GPUData<'uint32'> | GPUData<'float32'> | GPUData<'float32x2'>
): Buffer {
  return chunk.buffer instanceof DynamicBuffer ? chunk.buffer.buffer : chunk.buffer;
}

/** Lists every GPU column owned by one CSR adjacency for ownership and overlap checks. @internal */
export function getAdjacencyVectors(
  adjacency: GPUGraphAlgorithmsAdjacency
): (GPUVector<'uint32'> | GPUVector<'float32'>)[] {
  return [
    adjacency.offsets,
    adjacency.neighbors,
    adjacency.edgeIds,
    ...(adjacency.edgeWeights ? [adjacency.edgeWeights] : []),
    adjacency.count,
    adjacency.overflow
  ];
}
