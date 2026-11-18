// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUGlobalSpatialStatisticsProps} from './gpu-global-spatial-statistics';

/** Folds weights loss for inference while forwarding both producer flags independently. @internal */
export function getGlobalSpatialStatisticsStatusNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUGlobalSpatialStatisticsProps
): {nodes: GPUCommandNode<Parameters>[]; incomplete?: GraphDataView<'uint32'>} {
  const id = props.id ?? 'global-spatial-statistics';
  const input = props.weights.status;
  const output = props.output.status;
  const incomplete =
    input?.overflow || input?.candidateOverflow
      ? createTransientView(graph, `${id}-weights-incomplete`, 'uint32', 1)
      : undefined;
  if (!incomplete && !output?.overflow && !output?.candidateOverflow) return {nodes: []};
  const bindings: WGSLKernelBinding[] = [];
  for (const [name, view, access] of [
    ['inputOverflow', input?.overflow, 'read'],
    ['inputCandidateOverflow', input?.candidateOverflow, 'read'],
    ['incomplete', incomplete, 'read_write'],
    ['overflow', output?.overflow, 'read_write'],
    ['candidateOverflow', output?.candidateOverflow, 'read_write']
  ] as const) {
    if (view) bindings.push({name, view, type: 'u32', access});
  }
  return {
    incomplete,
    nodes: [
      createWGSLKernelNode(graph, {
        id: `${id}-weights-status`,
        operation: 'GPUGlobalSpatialStatistics',
        invocationCount: 1,
        bindings,
        body: `let finalLoss = ${input?.overflow ? 'u32(inputOverflow[inputOverflowOffset] != 0u)' : '0u'};
  let candidateLoss = ${input?.candidateOverflow ? 'u32(inputCandidateOverflow[inputCandidateOverflowOffset] != 0u)' : '0u'};
  ${incomplete ? 'incomplete[incompleteOffset] = finalLoss | candidateLoss;' : ''}
  ${output?.overflow ? 'overflow[overflowOffset] = finalLoss;' : ''}
  ${output?.candidateOverflow ? 'candidateOverflow[candidateOverflowOffset] = candidateLoss;' : ''}`
      })
    ]
  };
}
