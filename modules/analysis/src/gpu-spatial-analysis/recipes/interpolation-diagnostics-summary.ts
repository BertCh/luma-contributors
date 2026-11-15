// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUReduction, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';
import {
  addRecipeCountNodes,
  createRecipeKernelNode,
  RecipeBuilder,
  validateRecipeBoundary
} from './recipe-utils';
import type {GPUInterpolationDiagnosticsOutput} from './interpolation-diagnostics-recipe';

/** Adds parallel finite counts and scaled error metrics, without reading results back. @internal */
export function addInterpolationDiagnosticsSummaryNodes<Parameters>(
  builder: RecipeBuilder<Parameters>,
  id: string,
  outputs: Pick<
    GPUInterpolationDiagnosticsOutput,
    | 'residuals'
    | 'standardizedResiduals'
    | 'validCount'
    | 'standardizedCount'
    | 'bias'
    | 'meanAbsoluteError'
    | 'rootMeanSquaredError'
  >,
  invalidCount: GraphDataView<'uint32'>
): void {
  const {graph} = builder;
  const rowCount = outputs.residuals.length;
  validateRecipeBoundary(
    graph,
    id,
    [outputs.residuals, outputs.standardizedResiduals],
    [
      outputs.validCount,
      outputs.standardizedCount,
      outputs.bias,
      outputs.meanAbsoluteError,
      outputs.rootMeanSquaredError,
      invalidCount
    ]
  );
  const magnitudes = createTransientView(graph, `${id}-magnitudes`, 'float32', rowCount);
  const scale = createTransientView(graph, `${id}-scale`, 'float32', 1);
  addRecipeCountNodes(builder, {
    id: `${id}-counts`,
    operation: 'GPUInterpolationDiagnostics',
    rowCount,
    declarations: IS_FINITE_VALUE_WGSL,
    bindings: [
      {name: 'residuals', view: outputs.residuals, type: 'f32', access: 'read'},
      {name: 'standardized', view: outputs.standardizedResiduals, type: 'f32', access: 'read'},
      {name: 'magnitudes', view: magnitudes, type: 'f32', access: 'read_write'}
    ],
    prelude: `let value = residuals[residualsOffset + index];
let valid = isFiniteValue(value);
magnitudes[magnitudesOffset + index] = select(0.0, abs(value), valid);`,
    counts: [
      {name: 'valid', output: outputs.validCount, value: 'u32(valid)'},
      {
        name: 'standardized',
        output: outputs.standardizedCount,
        value: 'u32(isFiniteValue(standardized[standardizedOffset + index]))'
      },
      {name: 'invalid', output: invalidCount, value: 'u32(!valid)'}
    ]
  });
  builder.add(
    new GPUReduction({id: `${id}-scale`, input: magnitudes, output: scale, operation: 'max'})
  );
  const signedValues = createTransientView(graph, `${id}-signed-values`, 'float32', rowCount);
  const absolute = createTransientView(graph, `${id}-absolute`, 'float32', rowCount);
  const squared = createTransientView(graph, `${id}-squared`, 'float32', rowCount);
  builder.addGlue(
    'normalized-errors',
    createRecipeKernelNode(graph, {
      id: `${id}-normalize`,
      operation: 'GPUInterpolationDiagnostics',
      invocationCount: rowCount,
      declarations: `${IS_FINITE_VALUE_WGSL}\n${SCALED_QUOTIENT_WGSL}`,
      bindings: [
        {name: 'residuals', view: outputs.residuals, type: 'f32', access: 'read'},
        {name: 'scale', view: scale, type: 'f32', access: 'read'},
        {name: 'count', view: outputs.validCount, type: 'u32', access: 'read'},
        {name: 'signedValues', view: signedValues, type: 'f32', access: 'read_write'},
        {name: 'absolute', view: absolute, type: 'f32', access: 'read_write'},
        {name: 'squared', view: squared, type: 'f32', access: 'read_write'}
      ],
      // Scale before squaring and divide before summing: even maximum finite residuals have
      // bounded contributions and finite means/RMSE. All-invalid and all-zero inputs write zero.
      // Split the normalization quotient into mantissas and integer exponents: a raw division
      // can multiply by an underflowing reciprocal and turn near-maximum residuals into zero.
      body: `let value = residuals[residualsOffset + index];
var normalized = 0.0;
if (isFiniteValue(value) && scale[scaleOffset] > 0.0) {
  normalized = getScaledQuotient(value, scale[scaleOffset], 1.0, 0);
}
let denominator = max(1.0, f32(count[countOffset]));
signedValues[signedValuesOffset + index] = normalized / denominator;
absolute[absoluteOffset + index] = abs(normalized) / denominator;
squared[squaredOffset + index] = normalized * normalized / denominator;`
    })
  );
  const means = [signedValues, absolute, squared].map((input, index) => {
    const output = createTransientView(graph, `${id}-mean-${index}`, 'float32', 1);
    builder.add(new GPUReduction({id: `${id}-mean-${index}`, input, output, operation: 'sum'}));
    return output;
  });
  builder.addGlue(
    'summary',
    createRecipeKernelNode(graph, {
      id: `${id}-publish`,
      operation: 'GPUInterpolationDiagnostics',
      invocationCount: 1,
      bindings: [
        {name: 'scale', view: scale, type: 'f32', access: 'read'},
        {name: 'count', view: outputs.validCount, type: 'u32', access: 'read'},
        {name: 'signedValues', view: means[0], type: 'f32', access: 'read'},
        {name: 'absolute', view: means[1], type: 'f32', access: 'read'},
        {name: 'squared', view: means[2], type: 'f32', access: 'read'},
        {name: 'bias', view: outputs.bias, type: 'f32', access: 'read_write'},
        {name: 'absoluteError', view: outputs.meanAbsoluteError, type: 'f32', access: 'read_write'},
        {name: 'rootError', view: outputs.rootMeanSquaredError, type: 'f32', access: 'read_write'}
      ],
      // Clamp normalized means to their mathematical bounds against float32 reduction rounding.
      body: `let missing = bitcast<f32>(0x7fc00000u | (index & 0u));
let valid = count[countOffset] > 0u;
let magnitude = scale[scaleOffset];
bias[biasOffset] = select(missing, magnitude * clamp(signedValues[signedValuesOffset], -1.0, 1.0), valid);
absoluteError[absoluteErrorOffset] = select(missing, magnitude * clamp(absolute[absoluteOffset], 0.0, 1.0), valid);
rootError[rootErrorOffset] = select(missing, magnitude * sqrt(clamp(squared[squaredOffset], 0.0, 1.0)), valid);`
    })
  );
}
