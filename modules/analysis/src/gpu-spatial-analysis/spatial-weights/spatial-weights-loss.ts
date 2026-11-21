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
  validateGraphViewsBelongToGraph,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';
import type {GPUChunkedSpatialWeights} from './gpu-chunked-spatial-weights';
import type {GPUSpatialWeights} from './spatial-weights';

/** @internal Loss metadata can be carried at matrix level and by individual CSR chunks. */
type WeightsLossInput = {
  status?: {
    overflow?: GraphDataView<'uint32'>;
    candidateOverflow?: GraphDataView<'uint32'>;
    complete?: GraphDataView<'uint32'>;
  };
  chunks?: readonly {weights: WeightsLossInput}[];
};

/** @internal For consumers that read final overflow but cannot compare counts or qualify stopping evidence. */
export function validateWeightsOverflowEvidence(
  id: string,
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights
): void {
  const validate = (
    name: string,
    status: GPUSpatialWeights['status'],
    hasAggregateOverflow = false
  ) => {
    validateRequirement(
      id,
      status?.requiredCount === undefined || Boolean(status.overflow) || hasAggregateOverflow,
      `${name}.requiredCount needs the producer's final overflow flag covering these slots; supply that flag or verify the CSR is complete and omit requiredCount`
    );
    for (const field of ['converged', 'iterationLimitReached', 'unresolvedCount'] as const) {
      validateRequirement(
        id,
        status?.[field] === undefined,
        `${name}.${field} cannot be qualified; verify the CSR is complete, then omit this view and retain overflow and candidateOverflow`
      );
    }
  };
  validate('weights.status', weights.status);
  if ('chunks' in weights) {
    for (const [index, chunk] of weights.chunks.entries()) {
      // A chunked producer can publish the OR of every chunk's final overflow at the root.
      validate(
        `weights.chunks[${index}].weights.status`,
        chunk.weights.status,
        Boolean(weights.status?.overflow)
      );
    }
  }
}

/** @internal Collect metadata only, never input rows, preserving the two distinct causes. */
export function getWeightsLossSources(
  inputs: readonly WeightsLossInput[],
  id = 'spatial-weights',
  readCompleteness = false
) {
  const ports = inputs.flatMap(input => [
    input.status,
    ...(input.chunks?.map(chunk => chunk.weights.status) ?? [])
  ]);
  for (const port of ports)
    validateRequirement(
      id,
      !port?.complete || readCompleteness,
      'weights.status.complete requires a completeness-aware consumer; a candidateOverflow view alone does not certify equivalence'
    );
  const collect = (name: 'overflow' | 'candidateOverflow' | 'complete') => [
    ...new Set(ports.flatMap(port => (port?.[name] ? [port[name]!] : [])))
  ];
  return {
    overflow: collect('overflow'),
    candidateOverflow: collect('candidateOverflow'),
    complete: collect('complete')
  };
}

/** @internal Construction-time validation of every supplied scalar loss view. */
export function validateWeightsLossSources(
  id: string,
  inputs: readonly WeightsLossInput[],
  readCompleteness = false
) {
  const sources = getWeightsLossSources(inputs, id, readCompleteness);
  for (const [name, views] of Object.entries(sources))
    for (const view of views) {
      validateSinglePackedView(view, ['uint32'], `${id} weights.status.${name}`);
      validateCapacity(id, `weights.status.${name} length`, view.length, 1);
    }
  return [...sources.overflow, ...sources.candidateOverflow, ...sources.complete];
}

/** @internal GPU OR in bounded batches, optionally retaining an operation's own loss word. */
export function createWeightsLossNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  inputs: readonly GraphDataView<'uint32'>[],
  output: GraphDataView<'uint32'>,
  merge = false,
  completeInputs: readonly GraphDataView<'uint32'>[] = []
): GPUCommandNode<Parameters>[] {
  const checks = [
    ...inputs.map(view => ({view, invert: false})),
    ...completeInputs.map(view => ({view, invert: true}))
  ];
  const views = checks.map(check => check.view);
  if (!views.length && merge) return [];
  validateGraphViewsBelongToGraph(id, graph, [...views, output]);
  const outputBuffer = output.buffer.defaultBuffer
    ? getCoreDefaultBuffer(output.buffer.defaultBuffer)
    : output.buffer;
  for (const input of views) {
    const inputBuffer = input.buffer.defaultBuffer
      ? getCoreDefaultBuffer(input.buffer.defaultBuffer)
      : input.buffer;
    validateRequirement(
      id,
      inputBuffer !== outputBuffer,
      'inputs and output must use separate buffers because WebGPU cannot bind one buffer read-only and read-write; use merge: true to retain the output flag'
    );
  }
  for (const view of [...views, output])
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'weights loss binding',
      getViewBindingRange(view).size,
      'use a smaller packed status view'
    );
  const nodes: GPUCommandNode<Parameters>[] = [];
  // Seven read-only scalar inputs plus one read/write result fit baseline WebGPU.
  for (let first = 0; first === 0 || first < checks.length; first += 7) {
    const batch = checks.slice(first, first + 7);
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-${first / 7}`,
        operation: 'SpatialWeightsLoss',
        invocationCount: 1,
        bindings: [
          ...batch.map(({view}, index) => ({
            name: `lossInput${index}`,
            view,
            type: 'u32' as const,
            access: 'read' as const
          })),
          {name: 'loss', view: output, type: 'u32', access: 'read_write'}
        ],
        body: `var flag = ${merge || first > 0 ? 'u32(loss[lossOffset] != 0u)' : '0u'};
${batch.map(({invert}, index) => `flag |= u32(lossInput${index}[lossInput${index}Offset] ${invert ? '==' : '!='} 0u);`).join('\n')}
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
  const sources = getWeightsLossSources(inputs, id, true);
  const views = [...sources.overflow, ...sources.candidateOverflow, ...sources.complete];
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
    nodes: view
      ? createWeightsLossNodes(
          graph,
          `${id}-weights-loss`,
          [...sources.overflow, ...sources.candidateOverflow],
          view,
          false,
          sources.complete
        )
      : []
  };
}
