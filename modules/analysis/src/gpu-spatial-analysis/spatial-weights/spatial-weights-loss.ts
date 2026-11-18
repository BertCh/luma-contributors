// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  getViewBindingRange,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateSinglePackedView,
  validateGPUStorageBindingByteLength,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** @internal Loss metadata can be carried at matrix level and by individual CSR chunks. */
type WeightsLossInput = {
  status?: {overflow?: GraphDataView<'uint32'>; candidateOverflow?: GraphDataView<'uint32'>};
  chunks?: readonly {weights: WeightsLossInput}[];
};

/** @internal Collect metadata only, never input rows, preserving the two distinct causes. */
export function getWeightsLossSources(inputs: readonly WeightsLossInput[]) {
  const ports = inputs.flatMap(input => [
    input.status,
    ...(input.chunks?.map(chunk => chunk.weights.status) ?? [])
  ]);
  const collect = (name: 'overflow' | 'candidateOverflow') => [
    ...new Set(ports.flatMap(port => (port?.[name] ? [port[name]!] : [])))
  ];
  return {overflow: collect('overflow'), candidateOverflow: collect('candidateOverflow')};
}

/** @internal Construction-time validation of every supplied scalar loss view. */
export function validateWeightsLossSources(id: string, inputs: readonly WeightsLossInput[]) {
  const sources = getWeightsLossSources(inputs);
  for (const [name, views] of Object.entries(sources))
    for (const view of views) {
      validateSinglePackedView(view, ['uint32'], `${id} weights.status.${name}`);
      validateCapacity(id, `weights.status.${name} length`, view.length, 1);
    }
  return [...sources.overflow, ...sources.candidateOverflow];
}

/** @internal GPU OR in bounded batches, optionally retaining an operation's own loss word. */
export function createWeightsLossNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  inputs: readonly GraphDataView<'uint32'>[],
  output: GraphDataView<'uint32'>,
  merge = false
): GPUCommandNode<Parameters>[] {
  if (!inputs.length && merge) return [];
  validateGraphViewsBelongToGraph(id, graph, [...inputs, output]);
  for (const view of [...inputs, output])
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'weights loss binding',
      getViewBindingRange(view).size,
      'use a smaller packed status view'
    );
  const nodes: GPUCommandNode<Parameters>[] = [];
  // Seven read-only scalar inputs plus one read/write result fit baseline WebGPU.
  for (let first = 0; first === 0 || first < inputs.length; first += 7) {
    const batch = inputs.slice(first, first + 7);
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-${first / 7}`,
        operation: 'SpatialWeightsLoss',
        invocationCount: 1,
        bindings: [
          ...batch.map((view, index) => ({
            name: `lossInput${index}`,
            view,
            type: 'u32' as const,
            access: 'read' as const
          })),
          {name: 'loss', view: output, type: 'u32', access: 'read_write'}
        ],
        body: `var flag = ${merge || first > 0 ? 'u32(loss[lossOffset] != 0u)' : '0u'};
${batch.map((_, index) => `flag |= u32(lossInput${index}[lossInput${index}Offset] != 0u);`).join('\n')}
loss[lossOffset] = flag;`
      })
    );
  }
  return nodes;
}

/** @internal One aggregate input gate for inference kernels already at eight bindings. */
export function createWeightsIncompleteNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  inputs: readonly WeightsLossInput[]
) {
  const sources = getWeightsLossSources(inputs);
  const views = [...sources.overflow, ...sources.candidateOverflow];
  validateGraphViewsBelongToGraph(id, graph, views);
  for (const input of views)
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'weights loss binding',
      getViewBindingRange(input).size,
      'use a smaller packed status view'
    );
  const view = views.length
    ? createTransientView(graph, `${id}-weights-loss`, 'uint32', 1)
    : undefined;
  return {
    view,
    nodes: view ? createWeightsLossNodes(graph, `${id}-weights-loss`, views, view) : []
  };
}
