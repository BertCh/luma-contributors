// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {TRANSITION_INTEGER_PRODUCTS_WGSL} from '../../../src/gpu-spatial-analysis/distribution-dynamics/transition-integer-products-wgsl';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
it('subtracts exact 128-bit count products before conversion', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const output = createTransientView(graph, 'output', 'float32', 6);
    const nodes = [
      createWGSLKernelNode(graph, {
        id: 'products',
        operation: 'test',
        variant: 'test',
        invocationCount: 1,
        bindings: [{name: 'output', view: output, type: 'f32', access: 'read_write'}],
        declarations: TRANSITION_INTEGER_PRODUCTS_WGSL,
        body: `
output[outputOffset] = getTransitionProductDifference(134217728u, TransitionCount(0u,536870912u), TransitionCount(0u,268435457u), TransitionCount(0u,268435455u));
output[outputOffset+1u] = getTransitionProductDifference(134217729u, TransitionCount(0u,536870912u), TransitionCount(0u,268435457u), TransitionCount(0u,268435457u));
let product = multiplyTransitionCounts(TransitionCount(0u,268435457u),TransitionCount(0u,268435455u));
output[outputOffset+2u] = f32(product.first);
output[outputOffset+3u] = f32(product.second);
output[outputOffset+4u] = getTransitionProductDifference(134217727u, TransitionCount(0u,536870912u), TransitionCount(0u,268435455u), TransitionCount(0u,268435455u));
output[outputOffset+5u] = getTransitionProductDifference(134217728u, TransitionCount(0u,536870912u), TransitionCount(0u,268435455u), TransitionCount(0u,268435457u));
`
      })
    ];
    const memory = createCPUGraphMemory();
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readFloat32(output))).toEqual([1, -1, 4294967296, 16777215, -1, 1]);
  } finally {
    device.destroy();
  }
});
