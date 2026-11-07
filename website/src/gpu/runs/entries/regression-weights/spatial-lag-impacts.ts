// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialLagImpacts turns a spatial-lag fit and its power traces into direct, indirect and total
// impacts per predictor. The fit and traces come from GPUSpatialLagMaximumLikelihood in the same
// graph. Right: three bar charts (direct, indirect, total) with one bar per predictor; the indirect
// impact (spillover) grows with rho.

import {
  GPUSpatialLagImpacts,
  GPUSpatialLagMaximumLikelihood
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './spatial-lag-impacts.parameters';
import {
  createRegressionInputs,
  createResponse,
  drawTractMap,
  MAP_SHARE,
  uploadRegressionMap,
  uploadStandardizedWeights
} from './shared/regression';
import {addTableColumn} from './shared/weights';

const IMPACT_COLORS = [
  [0.4, 0.65, 1, 0.95],
  [1, 0.6, 0.25, 0.95],
  [0.55, 0.9, 0.5, 0.95]
] as const;

export default defineOperationRun({
  parameters,
  caption: 'Direct, indirect and total impacts of each predictor (right), residuals (left).',
  createInputs: ({random, world}) => createRegressionInputs(random, world),
  build(builder, inputs, values) {
    const rowCount = inputs.tracts.count;
    const maximumPower = values.maximumPower;
    const weights = uploadStandardizedWeights(builder, inputs.weights);
    const predictors = builder.upload('predictors', 'float32', inputs.predictors);
    const response = builder.allocate('response', 'float32', rowCount);
    const table = builder.allocate('table', 'float32', 16);
    const summary = builder.allocate('summary', 'float32', 32);
    const status = builder.allocate('status', 'uint32', 1);
    const residuals = builder.allocate('residuals', 'float32', rowCount);
    const powerTraces = builder.allocate('power-traces', 'float32', maximumPower);
    const powerSums = builder.allocate('power-sums', 'float32', maximumPower);
    const infinityNorm = builder.allocate('infinity-norm', 'float32', 1);
    builder.graph.add(
      new GPUSpatialLagMaximumLikelihood({
        weights: weights.input,
        predictors: predictors.view,
        response: response.view,
        predictorCount: 2,
        maximumPower,
        output: {
          status: {code: status.view},
          table: table.view,
          summary: summary.view,
          residuals: residuals.view,
          powerTraces: powerTraces.view,
          powerSums: powerSums.view,
          infinityNorm: infinityNorm.view
        }
      })
    );
    const impacts = builder.allocate('impacts', 'float32', 6);
    builder.graph.add(
      new GPUSpatialLagImpacts({
        coefficients: table.view,
        coefficientStride: 4,
        rhoIndex: 3,
        predictorCount: 2,
        rowCount,
        maximumPower,
        traces: {
          powerTraces: powerTraces.view,
          powerSums: powerSums.view,
          infinityNorm: infinityNorm.view
        },
        status: status.view,
        output: {impacts: impacts.view}
      })
    );
    return {
      outputs: {
        mesh: uploadRegressionMap(builder, inputs),
        residuals,
        components: [0, 1, 2].map(column =>
          addTableColumn(builder, 'impacts', impacts, 3, column, 2)
        )
      },
      update: current =>
        response.write(createResponse(inputs, {noise: 0.4, lagRho: current.strength}))
    };
  },
  rebuildOn: ['maximumPower'],
  draw: ({inputs, outputs}) => [
    ...drawTractMap(inputs, outputs.mesh, outputs.residuals),
    ...outputs.components.map(
      (values, index) =>
        ({
          layer: 'chart',
          kind: 'bars',
          values,
          range: [-4, 6],
          color: IMPACT_COLORS[index],
          rectangle: [MAP_SHARE + 0.08, 0.7 - index * 0.31, 0.97, 0.94 - index * 0.31]
        }) as const
    )
  ]
});
