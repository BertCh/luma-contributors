// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';

/** Validated solver provenance, separate from geometric completeness. @internal */
export function captureGeometryPartitionProvenance(
  id: string,
  input?: GraphDataView<'uint32'>,
  output?: GraphDataView<'uint32'>
): Readonly<{labelsComplete?: GraphDataView<'uint32'>}> {
  for (const [name, view] of [
    ['labelsComplete', input],
    ['output.labelsComplete', output]
  ] as const) {
    if (!view) continue;
    validateSinglePackedView(view, ['uint32'], `${id} ${name}`);
    validateRequirement(
      id,
      view.length === 1,
      `${name} must contain one completion word; received ${view.length}`
    );
  }
  return Object.freeze({labelsComplete: output ?? input});
}

/** Copies current completion without suppressing the incumbent geometry. @internal */
export function getGeometryPartitionProvenanceNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  input?: GraphDataView<'uint32'>,
  output?: GraphDataView<'uint32'>
): GPUCommandNode<Parameters>[] {
  if (!output) return [];
  validateGraphChunkBindingRanges(graph, id, {
    labelsComplete: input,
    'output.labelsComplete': output
  });
  return [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-partition-provenance`,
      operation: 'GeometryPartitionProvenance',
      variant: 'publish',
      invocationCount: 1,
      bindings: [
        ...(input
          ? [{name: 'inputComplete', view: input, type: 'u32' as const, access: 'read' as const}]
          : []),
        {name: 'complete', view: output, type: 'u32', access: 'read_write'}
      ],
      body: `complete[completeOffset + index] = ${input ? 'select(0u, 1u, inputComplete[inputCompleteOffset] != 0u)' : '1u'};`
    })
  ];
}
