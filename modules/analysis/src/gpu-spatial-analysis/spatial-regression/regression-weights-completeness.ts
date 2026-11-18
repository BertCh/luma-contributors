// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  getGPUSpatialWeightsChunks,
  type GPUSpatialWeights,
  type GPUChunkedSpatialWeights
} from '../spatial-weights/index';

/** Whole/chunk-root and per-chunk loss flags; a shared scalar is read only once. */
export function getRegressionLossViews(
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights
): GraphDataView<'uint32'>[] {
  return [
    ...new Set(
      [weights.status, ...getGPUSpatialWeightsChunks(weights).map(chunk => chunk.weights.status)]
        .flatMap(status => [status?.overflow, status?.candidateOverflow])
        .filter((view): view is GraphDataView<'uint32'> => view !== undefined)
    )
  ];
}

/** Suppress every dependent result after its normal writer; reset the loss fold each submission. */
export function getRegressionWeightsSuppressionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights,
  output: object,
  robustOutput?: object
): GPUCommandNode<Parameters>[] {
  const flags = getRegressionLossViews(weights);
  if (!flags.length) return [];
  const loss = createTransientView(graph, `${id}-weights-loss`, 'uint32', 1);
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-weights-loss-reset`,
      operation: 'RegressionWeightsCompleteness',
      invocationCount: 1,
      bindings: [{name: 'loss', view: loss, type: 'u32', access: 'read_write'}],
      body: 'loss[lossOffset] = 0u;'
    })
  ];
  flags.forEach((flag, index) =>
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-weights-loss-${index}`,
        operation: 'RegressionWeightsCompleteness',
        invocationCount: 1,
        bindings: [
          {name: 'flag', view: flag, type: 'u32', access: 'read'},
          {name: 'loss', view: loss, type: 'u32', access: 'read_write'}
        ],
        body: 'loss[lossOffset] = loss[lossOffset] | flag[flagOffset];'
      })
    )
  );
  for (const [group, values] of [output, robustOutput].entries()) {
    if (!values) continue;
    for (const [name, value] of Object.entries(values)) {
      if (value && typeof value === 'object' && 'buffer' in value) {
        const view = value as GraphDataView;
        if (view.format !== 'float32') continue;
        nodes.push(
          createWGSLKernelNode(graph, {
            id: `${id}-suppress-incomplete-${group}-${name}`,
            operation: 'RegressionWeightsCompleteness',
            invocationCount: view.length,
            bindings: [
              {name: 'loss', view: loss, type: 'u32', access: 'read'},
              {name: 'output', view, type: 'f32', access: 'read_write'}
            ],
            declarations: GET_NAN_WGSL,
            body: 'if (loss[lossOffset] != 0u) { output[outputOffset + index] = getNaN(); }'
          })
        );
      }
    }
    const status = (
      values as {
        status?:
          | {code?: GraphDataView<'uint32'>; converged?: GraphDataView<'uint32'>}
          | GraphDataView<'uint32'>;
      }
    ).status;
    if (!status) continue;
    const entries = 'buffer' in status ? [['code', status] as const] : Object.entries(status);
    for (const [name, view] of entries) {
      if (!view || (name !== 'code' && name !== 'converged')) continue;
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-suppress-incomplete-${group}-${name}`,
          operation: 'RegressionWeightsCompleteness',
          invocationCount: 1,
          bindings: [
            {name: 'loss', view: loss, type: 'u32', access: 'read'},
            {name: 'output', view, type: 'u32', access: 'read_write'}
          ],
          body: `if (loss[lossOffset] != 0u) { output[outputOffset] = ${name === 'code' ? 3 : 0}u; }`
        })
      );
    }
  }
  return nodes;
}
