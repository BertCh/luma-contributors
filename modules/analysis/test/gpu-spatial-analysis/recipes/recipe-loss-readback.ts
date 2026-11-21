// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

/** Shared by the device loss fixtures and their construction-only Node check. */
export function createRecipeLossReadback(
  graph: GPUCommandGraph,
  id: string,
  source: GraphDataView,
  destination: GraphDataView,
  type: 'u32' | 'f32'
) {
  return createWGSLKernelNode(graph, {
    id,
    operation: 'RecipeLossReadback',
    invocationCount: destination.length,
    bindings: [
      {name: 'source', view: source, type, access: 'read'},
      {name: 'destination', view: destination, type, access: 'read_write'}
    ],
    body: 'destination[destinationOffset + index] = source[sourceOffset + index];'
  });
}
