// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {GPUCommandGraph, GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUSpatialWeightsOutput} from './spatial-weights';

/** @internal Publish the retained prefix after all CSR writers, independently for each chunk. */
export function createWeightsCountNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  output: GPUSpatialWeightsOutput
): GPUCommandNode<Parameters>[] {
  if (!output.status.count) return [];
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-count`,
      operation: 'SpatialWeightsCount',
      invocationCount: 1,
      bindings: [
        {name: 'offsets', view: output.offsets, type: 'u32', access: 'read'},
        {name: 'count', view: output.status.count, type: 'u32', access: 'read_write'}
      ],
      body: `count[countOffset] = min(offsets[offsetsOffset + ${output.offsets.length - 1}u], ${output.neighbors.length}u);`
    })
  ];
}
