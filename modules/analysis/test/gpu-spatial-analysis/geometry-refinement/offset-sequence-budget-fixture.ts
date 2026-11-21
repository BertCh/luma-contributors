// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createBoundedOffsetAssemblySequences} from '../../../src/gpu-spatial-analysis/geometry-refinement/offset-assembly-sequence';

export function createOffsetSequenceBudgetFixture(
  graph: GPUCommandGraph,
  result: GraphDataView<'uint32'>
) {
  const command = (digit: number) => {
    const node = createWGSLKernelNode(graph, {
      id: `budget-digit-${digit}`,
      operation: 'OffsetSequenceBudgetTest',
      invocationCount: 1,
      bindings: [{name: 'result', view: result, type: 'u32', access: 'read_write'}],
      body: `result[resultOffset] = result[resultOffset] * 10u + ${digit}u;`
    });
    // Stand-in budget forces a split after two complete rounds while the real dispatch
    // stays one lane, so device coverage needs only six tiny commands.
    return {...node, workload: {...node.workload!, maximumInvocationCount: 1000000000}};
  };
  const batches = createBoundedOffsetAssemblySequences('budget-sequence', [
    {nodes: [command(1), command(2)], repetitions: 3}
  ]);
  const initialize = createWGSLKernelNode(graph, {
    id: 'budget-initialize',
    operation: 'OffsetSequenceBudgetTest',
    invocationCount: 1,
    bindings: [{name: 'result', view: result, type: 'u32', access: 'read_write'}],
    body: 'result[resultOffset] = 0u;'
  });
  return {batches, nodes: [initialize, ...batches]};
}
