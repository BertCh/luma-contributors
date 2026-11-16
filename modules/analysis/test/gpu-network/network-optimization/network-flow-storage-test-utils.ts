// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLKernelBinding
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

/** Test-owned compute readback: the solver's caller-owned source needs STORAGE only.
 * COPY_SRC belongs exclusively to the separate test readback allocation. */
export function createNetworkFlowStorageReadbackNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  source: GraphDataView<'uint32' | 'uint32x2'>,
  destination: GraphDataView<'uint32'>
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode(graph, {
    id,
    operation: 'GPUNetworkFlowTestReadback',
    invocationCount: source.length * (source.format === 'uint32x2' ? 2 : 1),
    bindings: [
      getWGSLKernelBinding('flowStorageSource', source),
      getWGSLKernelBinding('flowStorageReadback', destination, 'read_write')
    ],
    body: `flowStorageReadback[flowStorageReadbackOffset + index] = flowStorageSource[flowStorageSourceOffset + index];`
  });
}
