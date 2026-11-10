// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRecipeBoundary, getRecipeView as getOrCreateView} from './expansion-recipe-utils';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUKriging, type GPUKrigingProps} from '../spatial-interpolation/gpu-kriging';
import {createRecipeKernelNode as createWGSLKernelNode} from './expansion-recipe-utils';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {RecipeBuilder, type GPURecipeResult} from './recipe-utils';

export type GPUInterpolationDiagnosticsOutput = {
  values: GraphDataView<'float32'>;
  variance: GraphDataView<'float32'>;
  predictions: GraphDataView<'float32'>;
  predictionVariance: GraphDataView<'float32'>;
  residuals: GraphDataView<'float32'>;
  standardizedResiduals: GraphDataView<'float32'>;
  /** One-row metrics, calculated over finite validation residuals. */
  validCount: GraphDataView<'uint32'>;
  standardizedCount: GraphDataView<'uint32'>;
  bias: GraphDataView<'float32'>;
  meanAbsoluteError: GraphDataView<'float32'>;
  rootMeanSquaredError: GraphDataView<'float32'>;
};

export type GPUInterpolationDiagnosticsRecipeProps = {
  id?: string;
  kriging: Omit<GPUKrigingProps, 'id' | 'output'>;
  outputs?: Partial<GPUInterpolationDiagnosticsOutput>;
};

/** Ordinary kriging and leave-one-out diagnostics, with no hidden fitting or readback. */
export function addInterpolationDiagnosticsRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUInterpolationDiagnosticsRecipeProps
): GPURecipeResult<GPUInterpolationDiagnosticsOutput> {
  const id = props.id ?? 'interpolation-diagnostics-recipe';
  const builder = new RecipeBuilder(graph);
  const sampleCount = props.kriging.values.length;
  const cellCount = props.kriging.grid.width * props.kriging.grid.height;
  const provided = props.outputs ?? {};
  const outputs: GPUInterpolationDiagnosticsOutput = {
    values: getOrCreateView(graph, `${id}-values`, 'float32', cellCount, provided.values),
    variance: getOrCreateView(graph, `${id}-variance`, 'float32', cellCount, provided.variance),
    predictions: getOrCreateView(
      graph,
      `${id}-predictions`,
      'float32',
      sampleCount,
      provided.predictions
    ),
    predictionVariance: getOrCreateView(
      graph,
      `${id}-prediction-variance`,
      'float32',
      sampleCount,
      provided.predictionVariance
    ),
    residuals: getOrCreateView(
      graph,
      `${id}-residuals`,
      'float32',
      sampleCount,
      provided.residuals
    ),
    standardizedResiduals: getOrCreateView(
      graph,
      `${id}-standardized`,
      'float32',
      sampleCount,
      provided.standardizedResiduals
    ),
    validCount: getOrCreateView(graph, `${id}-valid-count`, 'uint32', 1, provided.validCount),
    standardizedCount: getOrCreateView(
      graph,
      `${id}-standardized-count`,
      'uint32',
      1,
      provided.standardizedCount
    ),
    bias: getOrCreateView(graph, `${id}-bias`, 'float32', 1, provided.bias),
    meanAbsoluteError: getOrCreateView(
      graph,
      `${id}-absolute-error`,
      'float32',
      1,
      provided.meanAbsoluteError
    ),
    rootMeanSquaredError: getOrCreateView(
      graph,
      `${id}-root-error`,
      'float32',
      1,
      provided.rootMeanSquaredError
    )
  };
  validateRecipeBoundary(
    graph,
    id,
    [props.kriging.positions, props.kriging.values, props.kriging.mask, props.kriging.parameters],
    Object.values(outputs)
  );
  const krigingStatus = {
    invalidCount: getOrCreateView(graph, `${id}-invalid`, 'uint32', 1),
    uncertainCount: getOrCreateView(graph, `${id}-uncertain`, 'uint32', 1)
  };
  builder.add(
    new GPUKriging({
      ...props.kriging,
      id: `${id}-kriging`,
      output: {
        values: outputs.values,
        variance: outputs.variance,
        crossValidationPrediction: outputs.predictions,
        crossValidationVariance: outputs.predictionVariance,
        status: krigingStatus
      }
    })
  );
  builder.addGlue(
    'residuals',
    createWGSLKernelNode(graph, {
      id: `${id}-residuals`,
      operation: 'GPUInterpolationDiagnostics',
      bindings: [
        {name: 'observed', view: props.kriging.values, type: 'f32', access: 'read'},
        {name: 'predicted', view: outputs.predictions, type: 'f32', access: 'read'},
        {name: 'variance', view: outputs.predictionVariance, type: 'f32', access: 'read'},
        {name: 'residual', view: outputs.residuals, type: 'f32', access: 'read_write'},
        {
          name: 'standardized',
          view: outputs.standardizedResiduals,
          type: 'f32',
          access: 'read_write'
        }
      ],
      invocationCount: sampleCount,
      declarations: IS_FINITE_VALUE_WGSL,
      body: `let observedValue = observed[observedOffset + index];
let predictedValue = predicted[predictedOffset + index];
let varianceValue = variance[varianceOffset + index];
let difference = observedValue - predictedValue;
let valid = isFiniteValue(observedValue) && isFiniteValue(predictedValue) && isFiniteValue(difference);
let missing = bitcast<f32>(0x7fc00000u | (index & 0u));
residual[residualOffset + index] = select(missing, difference, valid);
var score = missing;
if (valid && isFiniteValue(varianceValue) && varianceValue > 0.0) {
  let candidate = difference / sqrt(varianceValue);
  if (isFiniteValue(candidate)) { score = candidate; }
}
standardized[standardizedOffset + index] = score;`
    })
  );
  const invalidCount = getOrCreateView(graph, `${id}-validation-invalid`, 'uint32', 1);
  // Streaming means and a scaled sum of squares avoid fixed-point and squared-error overflow.
  // This diagnostic pass is O(sampleCount), independent of output-grid size. It stays one
  // invocation on purpose: the streaming mean and scaled sum of squares are order-dependent float
  // recurrences, so a parallel reduction would change the results (not bit-identical). One thread
  // reading about 10^6 samples is the size at which it starts to dominate; validation sets are
  // normally thousands of samples. The integer counts alone could move to GPUReduction.
  builder.addGlue(
    'summary',
    createWGSLKernelNode(graph, {
      id: `${id}-summary`,
      operation: 'GPUInterpolationDiagnostics',
      invocationCount: 1,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'residuals', view: outputs.residuals, type: 'f32', access: 'read'},
        {name: 'standardized', view: outputs.standardizedResiduals, type: 'f32', access: 'read'},
        {name: 'validCount', view: outputs.validCount, type: 'u32', access: 'read_write'},
        {
          name: 'standardizedCount',
          view: outputs.standardizedCount,
          type: 'u32',
          access: 'read_write'
        },
        {name: 'bias', view: outputs.bias, type: 'f32', access: 'read_write'},
        {name: 'absoluteError', view: outputs.meanAbsoluteError, type: 'f32', access: 'read_write'},
        {name: 'rootError', view: outputs.rootMeanSquaredError, type: 'f32', access: 'read_write'},
        {name: 'invalidCount', view: invalidCount, type: 'u32', access: 'read_write'}
      ],
      body: `var count = 0u; var scoreCount = 0u; var mean = 0.0; var absoluteMean = 0.0;
var scale = 0.0; var squares = 0.0;
for (var row = 0u; row < ${sampleCount}u; row++) {
  let value = residuals[residualsOffset + row];
  scoreCount += select(0u, 1u, isFiniteValue(standardized[standardizedOffset + row]));
  if (!isFiniteValue(value)) { continue; }
  count++; let fraction = 1.0 / f32(count); let magnitude = abs(value);
  mean = mean * (1.0 - fraction) + value * fraction;
  absoluteMean = absoluteMean * (1.0 - fraction) + magnitude * fraction;
  if (magnitude > scale) { let ratio = scale / magnitude; squares = 1.0 + squares * ratio * ratio; scale = magnitude; }
  else if (scale > 0.0) { let ratio = magnitude / scale; squares += ratio * ratio; }
}
let missing = bitcast<f32>(0x7fc00000u | (index & 0u));
validCount[validCountOffset] = count; standardizedCount[standardizedCountOffset] = scoreCount;
invalidCount[invalidCountOffset] = ${sampleCount}u - count;
bias[biasOffset] = select(missing, mean, count > 0u);
absoluteError[absoluteErrorOffset] = select(missing, absoluteMean, count > 0u);
rootError[rootErrorOffset] = select(missing, scale * sqrt(squares / max(1.0, f32(count))), count > 0u);`
    })
  );
  return {
    contributors: builder.contributors,
    outputs,
    intermediates: {grid: props.kriging.grid, spatialContext: props.kriging.spatialContext},
    status: {
      stages: [
        {stage: 'kriging', status: krigingStatus},
        {stage: 'validation-summary', status: {invalidCount}}
      ]
    }
  };
}
