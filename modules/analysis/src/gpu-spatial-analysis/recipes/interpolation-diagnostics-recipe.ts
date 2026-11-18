// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {addPreparedRecipe} from './prepared-recipe';

import {validateRecipeBoundary, getRecipeView as getOrCreateView} from './recipe-utils';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUKriging, type GPUKrigingProps} from '../spatial-interpolation/gpu-kriging';
import {createRecipeKernelNode as createWGSLKernelNode} from './recipe-utils';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {RecipeBuilder, type GPURecipeResult} from './recipe-utils';
import {addInterpolationDiagnosticsSummaryNodes} from './interpolation-diagnostics-summary';

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
  return addPreparedRecipe(graph, props, declareInterpolationDiagnosticsRecipe);
}

function declareInterpolationDiagnosticsRecipe<Parameters>(
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
  addInterpolationDiagnosticsSummaryNodes(builder, `${id}-summary`, outputs, invalidCount);
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
