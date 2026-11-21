// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_QUIET_NAN_SEEDED_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {snapshotWeightsDescription} from '../spatial-weights/weights-construction-snapshot';
import {getGPUSpatialWeightsLossAndCompletenessViews} from '../spatial-weights/gpu-chunked-spatial-weights';
import type {GPUSpatialWeights} from '../spatial-weights/spatial-weights';

/** Capture extended producer status (including complete) as well as declared categorical fields. @internal */
export function snapshotCategoricalAssociationProps<
  Props extends {weights: GPUSpatialWeights; output: object}
>(input: Props, outputFields: readonly string[]): Props {
  const props = snapshotConstructionFields(input, [
    'id',
    'weights',
    'categories',
    'categoryCount',
    'output'
  ] as readonly (keyof Props)[]);
  return Object.freeze({
    ...props,
    weights: snapshotWeightsDescription(props.weights),
    output: snapshotConstructionFields(props.output, outputFields as never[])
  });
}

/** Suppress on either capacity loss or a supplied zero completeness certificate. @internal */
export function createCategoricalSuppressionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  operation: string,
  weights: GPUSpatialWeights,
  output: GraphDataView<'float32'>,
  length: number
) {
  const sources = getGPUSpatialWeightsLossAndCompletenessViews(weights, id);
  const checks = [
    ...sources.loss.map(view => ({view, incompleteWhenZero: false})),
    ...sources.complete.map(view => ({view, incompleteWhenZero: true}))
  ];
  validateGraphChunkBindingRanges(
    graph,
    id,
    Object.fromEntries(checks.map(({view}, index) => [`weights loss ${index}`, view]))
  );
  return checks.map(({view, incompleteWhenZero}, index) =>
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-incomplete-${index}`,
      operation,
      variant: incompleteWhenZero ? 'incomplete-certificate' : 'incomplete-input',
      bindings: [
        {name: 'loss', view, type: 'u32', access: 'read'},
        {name: 'output', view: output, type: 'f32', access: 'read_write'}
      ],
      invocationCount: length,
      declarations: GET_QUIET_NAN_SEEDED_WGSL,
      body: `if (loss[lossOffset] ${incompleteWhenZero ? '==' : '!='} 0u) { output[outputOffset + index] = getQuietNaN(index); }`
    })
  );
}

/** Validate which upstream completeness evidence the outputs can honor. @internal */
export function validateCategoricalCompleteness(id: string, weights: GPUSpatialWeights): void {
  for (const field of ['converged', 'iterationLimitReached', 'unresolvedCount'] as const) {
    validateRequirement(
      id,
      weights.status?.[field] === undefined,
      `weights.status.${field} cannot be checked by this contributor; verify the CSR is complete and omit this port`
    );
  }
  validateRequirement(
    id,
    !weights.status?.requiredCount || Boolean(weights.status.overflow),
    'weights.status.requiredCount requires weights.status.overflow to certify final capacity; provide the producer overflow flag, or a verified complete CSR without requiredCount'
  );
  if (weights.status?.complete) {
    validateSinglePackedView(weights.status.complete, ['uint32'], `${id} weights.status.complete`);
  }
}
