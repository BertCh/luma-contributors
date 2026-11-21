// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';

/** Same-submission status guard: maps never consume finite residuals left by a failed fit. */
export function createRegressionResidualDisplayNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    residuals: GraphDataView<'float32'>;
    code: GraphDataView<'uint32'>;
    output: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode(graph, {
    id: props.id,
    operation: 'RegressionResidualDisplay',
    variant: 'fit-status',
    invocationCount: props.residuals.length,
    bindings: [
      {name: 'residuals', view: props.residuals, type: 'f32', access: 'read'},
      {name: 'code', view: props.code, type: 'u32', access: 'read'},
      {name: 'display', view: props.output, type: 'f32', access: 'read_write'}
    ],
    body: `let fitCode = code[codeOffset];
  var value = bitcast<f32>(0x7fc00000u | (fitCode & 1u));
  if (fitCode == 0u) { value = residuals[residualsOffset + index]; }
  display[displayOffset + index] = value;`
  });
}

/** CPU tooltips use the fit code read in the same ticket as the residual value. */
export function getRegressionResidualForDisplay(
  fitCode: number | undefined,
  residual: number
): number | undefined {
  return fitCode === 0 && Number.isFinite(residual) ? residual : undefined;
}
