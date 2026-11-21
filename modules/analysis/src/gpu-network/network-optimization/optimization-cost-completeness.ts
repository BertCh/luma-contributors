// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** @internal Preserve the existing optimizer diagnostic publisher's vocabulary. */
export function withoutComplete<Status extends {complete?: unknown}>(
  status?: Status
): Omit<Status, 'complete'> | undefined {
  if (!status) return undefined;
  const {complete, ...diagnostics} = status;
  return diagnostics;
}
/** @internal Input loss must have a readable decision certificate. */
export function validateOptimizationCosts(
  id: string,
  props: {
    costsComplete?: GraphDataView<'uint32'>;
    costLimit?: GraphDataView<'float32'>;
    output: {status?: {complete?: GraphDataView<'uint32'>}};
  }
) {
  validateRequirement(
    id,
    !(props.costsComplete || props.costLimit) || Boolean(props.output.status?.complete),
    'costsComplete or costLimit requires output.status.complete; inspect it before using the decision'
  );
  for (const [name, view, format] of [
    ['costsComplete', props.costsComplete, 'uint32'],
    ['costLimit', props.costLimit, 'float32'],
    ['output.status.complete', props.output.status?.complete, 'uint32']
  ] as const)
    if (view) {
      validateSinglePackedView(view, [format], `${id} ${name}`);
      validateRequirement(
        id,
        view.length === 1,
        `${name} must contain one row, received ${view.length}`
      );
    }
}
/** @internal A separate pass preserves bounded solver kernel fingerprints and diagnostics. */
export function createOptimizationCostCompletionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: {
    costsComplete?: GraphDataView<'uint32'>;
    costLimit?: GraphDataView<'float32'>;
    threshold?: GraphDataView<'float32'>;
    summary: GraphDataView<'uint32'>;
    complete?: GraphDataView<'uint32'>;
    facility?: boolean;
  }
) {
  if (!props.complete) return [];
  const bindings: WGSLKernelBinding[] = [
    {name: 'summary', view: props.summary, type: 'u32', access: 'read'},
    {name: 'complete', view: props.complete, type: 'u32', access: 'read_write'}
  ];
  let condition = `summary[summaryOffset] != 0u && summary[summaryOffset + ${props.facility ? 2 : 1}u] != 0u`;
  if (props.costsComplete) {
    bindings.push({name: 'costsComplete', view: props.costsComplete, type: 'u32', access: 'read'});
    condition += ' && costsComplete[costsCompleteOffset] != 0u';
  }
  if (props.costLimit) {
    bindings.push({name: 'costLimit', view: props.costLimit, type: 'f32', access: 'read'});
    if (props.threshold) {
      bindings.push({name: 'threshold', view: props.threshold, type: 'f32', access: 'read'});
      condition += ' && threshold[thresholdOffset] <= costLimit[costLimitOffset]';
    } else condition += ' && bitcast<u32>(costLimit[costLimitOffset]) == 0x7f800000u';
  }
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-cost-completion`,
      operation: 'NetworkOptimizationCostCompletion',
      variant: 'complete',
      invocationCount: 1,
      workgroupSize: 2,
      bindings,
      body: `complete[completeOffset] = u32(${condition});`
    })
  ];
}

/** @internal Shared runtime discriminants for array, named and direct forms. */
export function validateRationalAccessOptions(
  id: string,
  options: {demandMode?: string; stepMode?: string}
) {
  for (const [name, values] of [
    ['demandMode', ['integral', 'fractional']],
    ['stepMode', ['fraction', 'absolute']]
  ] as const) {
    validateRequirement(
      id,
      options[name] === undefined || values.includes(options[name] as never),
      `${name} must be ${values.join(' or ')}; received ${options[name]}`
    );
  }
}
