// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_QUIET_NAN_SEEDED_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUSpatialWeightsLossViews,
  isGPUChunkedSpatialWeights,
  type GPUChunkedSpatialWeights
} from '../spatial-weights/gpu-chunked-spatial-weights';
import type {GPUSpatialWeights} from '../spatial-weights/spatial-weights';

/** Optional CSR loss evidence. A required count is accepted alongside the final overflow flag. */
export type RegionalizationWeightsStatus<Word> = {
  candidateOverflow?: Word;
} & ({overflow?: Word; requiredCount?: undefined} | {overflow: Word; requiredCount?: Word});

/** Check completeness fields on views, CPU arrays or named graph values. @internal */
export function validateRegionalizationWeightsStatus(
  id: string,
  status?: {
    overflow?: unknown;
    requiredCount?: unknown;
    complete?: unknown;
    converged?: unknown;
    iterationLimitReached?: unknown;
    unresolvedCount?: unknown;
  },
  name = 'weights.status'
): void {
  validateRequirement(
    id,
    status?.complete === undefined,
    `${name}.complete requires a completeness-aware consumer; verify completeness before supplying the CSR and omit this port`
  );
  validateRequirement(
    id,
    status?.requiredCount === undefined || Boolean(status.overflow),
    `${name}.requiredCount requires ${name}.overflow to certify final capacity; supply the producer overflow flag or provide a verified complete CSR without requiredCount`
  );
  for (const field of ['converged', 'iterationLimitReached', 'unresolvedCount'] as const) {
    validateRequirement(
      id,
      status?.[field] === undefined,
      `${name}.${field} cannot be checked by this contributor; verify the CSR is complete and omit this port`
    );
  }
}

/** Reject completeness evidence the overflow-only regionalization routes cannot use. @internal */
export function validateRegionalizationWeightsCompleteness(
  id: string,
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights
): void {
  validateRegionalizationWeightsStatus(id, weights.status);
  if (isGPUChunkedSpatialWeights(weights)) {
    for (const [index, chunk] of weights.chunks.entries()) {
      validateRegionalizationWeightsStatus(
        id,
        chunk.weights.status,
        `weights.chunks[${index}].weights.status`
      );
    }
  }
}

/** Suppress only results depending on incomplete adjacency; separate outputs permit buffer slices. @internal */
export function createRegionalizationLossSuppressionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  operation: string,
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights,
  outputs: readonly {
    view: GraphDataView<'uint32'> | GraphDataView<'float32'>;
    length: number;
    wordOffset?: number;
    value: 'missing' | 'nan' | 'zero';
  }[]
): GPUCommandNode<Parameters>[] {
  const losses = getGPUSpatialWeightsLossViews(weights);
  validateGraphChunkBindingRanges(
    graph,
    id,
    Object.fromEntries(losses.map((view, index) => [`loss${index}`, view]))
  );
  return losses.flatMap((loss, lossIndex) =>
    outputs.map((output, outputIndex) =>
      createWGSLKernelNode(graph, {
        id: `${id}-incomplete-${lossIndex}-${outputIndex}`,
        operation,
        variant: 'incomplete-adjacency',
        invocationCount: output.length,
        bindings: [
          {name: 'loss', view: loss, type: 'u32', access: 'read'},
          {
            name: 'result',
            view: output.view,
            type: output.view.format === 'float32' ? 'f32' : 'u32',
            access: 'read_write'
          }
        ],
        declarations: output.value === 'nan' ? GET_QUIET_NAN_SEEDED_WGSL : '',
        body: `if (loss[lossOffset] != 0u) {
  result[resultOffset + ${output.wordOffset ?? 0}u + index] = ${output.value === 'nan' ? 'getQuietNaN(index)' : output.value === 'missing' ? '0xffffffffu' : '0u'};
}`
      })
    )
  );
}
