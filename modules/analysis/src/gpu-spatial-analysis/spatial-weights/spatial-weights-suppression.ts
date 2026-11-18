// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_QUIET_NAN_SEEDED_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  getGPUSpatialWeightsLossViews,
  type GPUChunkedSpatialWeights
} from './gpu-chunked-spatial-weights';
import type {GPUSpatialWeights} from './spatial-weights';

/** Suppresses public numbers on incomplete neighborhoods; normal producer writes restore them next frame. @internal */
export function createSpatialWeightsSuppressionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  operation: string,
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights,
  outputs: Readonly<Record<string, GraphDataView | undefined>>,
  logicalLengths: Readonly<Record<string, number>> = {}
): GPUCommandNode<Parameters>[] {
  const losses = getGPUSpatialWeightsLossViews(weights);
  const fields = Object.entries({
    output: outputs['output'],
    statistics: outputs['statistics'],
    summary: outputs['summary'],
    permutationCount: outputs['permutationCount']
  })
    .filter((field): field is [string, GraphDataView] => Boolean(field[1]))
    .map(
      ([name, view]) =>
        [name, view, Math.min(view.length, logicalLengths[name] ?? view.length)] as const
    );
  validateGraphChunkBindingRanges(
    graph,
    id,
    Object.fromEntries(losses.map((view, index) => [`weights loss ${index}`, view]))
  );
  return losses.map((loss, lossIndex) =>
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-incomplete-${lossIndex}`,
      operation,
      variant: 'incomplete-input',
      bindings: [
        {name: 'loss', view: loss, type: 'u32', access: 'read'},
        ...fields.map(([name, view]) => ({
          name,
          view,
          type: view.format === 'float32' ? ('f32' as const) : ('u32' as const),
          access: 'read_write' as const
        }))
      ],
      invocationCount: Math.max(1, ...fields.map(([, , length]) => length)),
      declarations: GET_QUIET_NAN_SEEDED_WGSL,
      body: `if (loss[lossOffset] != 0u) {
${fields.map(([name, view, length]) => `  if (index < ${length}u) { ${name}[${name}Offset + index] = ${view.format === 'float32' ? 'getQuietNaN(index)' : '0u'}; }`).join('\n')}
}`
    })
  );
}
