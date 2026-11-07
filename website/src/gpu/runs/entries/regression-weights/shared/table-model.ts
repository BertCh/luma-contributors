// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The spatial models that write a coefficient table (estimate, standard error, z, p per row) and
// optional residuals share one run: data generated with a spatial lag or error process, the
// operation added by the caller, a residual map and the coefficient bars with the true values.

import type {GPUSpatialWeights} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  defineOperationRun,
  type ParameterValues,
  type RunBuffer,
  type RunBuilder
} from '../../../define-run';
import {
  createCoefficientPanel,
  createRegressionInputs,
  createResponse,
  drawCoefficientPanel,
  drawTractMap,
  TRUE_COEFFICIENTS,
  uploadRegressionMap,
  uploadStandardizedWeights
} from './regression';
import type {GpuSceneParameter} from '../../../../types';

/** Everything the caller needs to add its operation. */
export type TableModelContext = {
  builder: RunBuilder;
  values: ParameterValues;
  weights: GPUSpatialWeights;
  predictors: RunBuffer<'float32'>;
  response: RunBuffer<'float32'>;
  rowCount: number;
  /** Rows of the coefficient table: intercept, two predictors and the spatial parameter. */
  table: RunBuffer<'float32'>;
  summary: RunBuffer<'float32'>;
  status: RunBuffer<'uint32'>;
  residuals: RunBuffer<'float32'>;
};

export type TableModelSpec = {
  parameters: readonly GpuSceneParameter[];
  caption: string;
  /** The data generating process; the `strength` parameter id sets rho or lambda. */
  process: 'lag' | 'error';
  rebuildOn?: readonly string[];
  /** Adds the real operation, writing into the context's table, summary, status and residuals. */
  addOperation(context: TableModelContext): void;
};

/** Defines a run for a model with a four-column coefficient table. */
export function defineTableModelRun(spec: TableModelSpec) {
  return defineOperationRun({
    parameters: spec.parameters,
    caption: spec.caption,
    rebuildOn: spec.rebuildOn,
    createInputs: ({random, world}) => createRegressionInputs(random, world),
    build(builder, inputs, values) {
      const rowCount = inputs.tracts.count;
      const weights = uploadStandardizedWeights(builder, inputs.weights);
      const predictors = builder.upload('predictors', 'float32', inputs.predictors);
      const response = builder.allocate('response', 'float32', rowCount);
      const table = builder.allocate('table', 'float32', 16);
      const summary = builder.allocate('summary', 'float32', 32);
      const status = builder.allocate('status', 'uint32', 1);
      const residuals = builder.allocate('residuals', 'float32', rowCount);
      spec.addOperation({
        builder,
        values,
        weights: weights.input,
        predictors,
        response,
        rowCount,
        table,
        summary,
        status,
        residuals
      });
      const truth = [...TRUE_COEFFICIENTS, 0];
      const panel = createCoefficientPanel(builder, table, 4, truth);
      return {
        outputs: {mesh: uploadRegressionMap(builder, inputs), residuals, panel},
        update: current => {
          const strength = current.strength;
          response.write(
            createResponse(inputs, {
              noise: current.noise ?? 0.4,
              ...(spec.process === 'lag' ? {lagRho: strength} : {errorLambda: strength})
            })
          );
          panel.truth.write(Float32Array.of(...TRUE_COEFFICIENTS, strength));
        }
      };
    },
    draw: ({inputs, outputs}) => [
      ...drawTractMap(inputs, outputs.mesh, outputs.residuals),
      ...drawCoefficientPanel(outputs.panel)
    ]
  });
}
