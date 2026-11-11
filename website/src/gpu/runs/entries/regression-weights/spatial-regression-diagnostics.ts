// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialRegressionDiagnostics tests the residuals of an OLS fit for spatial dependence. The OLS
// fit runs in the same graph. Left: the residual map. Right: the six test statistics the operation
// wrote (LM lag, LM error, robust lag, robust error, LM SARMA and Moran's I z); the lag and error
// tests rise with the matching spatial process in the data.

import {
  GPUOrdinaryLeastSquares,
  GPUSpatialRegressionDiagnostics,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createChartAxes, getChartRectangle, mergeText} from '../../text';
import {createBarNames} from './shared/labels';
import {parameters} from './spatial-regression-diagnostics.parameters';
import {
  createRegressionInputs,
  createResponse,
  drawResidualMap,
  MAP_SHARE,
  uploadRegressionMap,
  uploadStandardizedWeights
} from './shared/regression';
import {addKernelPass} from './shared/kernel';
import {addTableColumn} from './shared/weights';

const TEST_NAMES = ['LM lag', 'LM error', 'robust lag', 'robust error', 'SARMA', 'Moran z'];
const TEST_RECTANGLE = [MAP_SHARE + 0.08, 0.14, 0.97, 0.86] as const;
const TEST_LIMIT = 12;
/** The 5% critical value of a chi-square with one degree of freedom. */
const CRITICAL_VALUE = 3.84;

export default defineOperationRun({
  parameters,
  caption: 'Residual map (left) and the spatial dependence tests on it (right).',
  createInputs: ({random, world}) => createRegressionInputs(random, world),
  build(builder, inputs, values) {
    const rowCount = inputs.tracts.count;
    const weights = uploadStandardizedWeights(builder, inputs.weights);
    const predictors = builder.upload('predictors', 'float32', inputs.predictors);
    const response = builder.allocate('response', 'float32', rowCount);
    const residuals = builder.allocate('residuals', 'float32', rowCount);
    // A third of the tracts is left out of the fit; their links are dropped, which leaves rows of
    // the row-standardized weights short of 1 unless `restandardizeRows` rescales them.
    const mask = builder.upload(
      'mask',
      'uint32',
      Uint32Array.from({length: rowCount}, (_, row) =>
        ((row * 2654435761) >>> 0) % 100 < 35 ? 0 : 1
      )
    );
    builder.graph.add(
      new GPUOrdinaryLeastSquares({
        predictors: predictors.view,
        response: response.view,
        predictorCount: 2,
        mask: mask.view,
        output: {
          status: {code: builder.allocate('ols-status', 'uint32', 1).view},
          coefficients: builder.allocate('ols-coefficients', 'float32', 3).view,
          standardErrors: builder.allocate('ols-standard-errors', 'float32', 3).view,
          tStatistics: builder.allocate('ols-t-statistics', 'float32', 3).view,
          summary: builder.allocate(
            'ols-summary',
            'float32',
            GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH
          ).view,
          residuals: residuals.view
        }
      })
    );
    const tests = builder.allocate('tests', 'float32', 18);
    builder.graph.add(
      new GPUSpatialRegressionDiagnostics({
        weights: weights.input,
        predictors: predictors.view,
        response: response.view,
        residuals: residuals.view,
        mask: mask.view,
        predictorCount: 2,
        restandardizeRows: values.restandardizeRows === 1,
        output: {
          tests: tests.view,
          summary: builder.allocate('summary', 'float32', 16).view,
          status: {code: builder.allocate('status', 'uint32', 1).view}
        }
      })
    );
    // Tracts left out of the fit draw transparent instead of as zero residuals.
    const shown = builder.allocate('shown-residuals', 'float32', rowCount);
    addKernelPass(builder, {
      id: 'masked-residuals',
      invocationCount: rowCount,
      bindings: [
        {name: 'residuals', buffer: residuals, access: 'read'},
        {name: 'mask', buffer: mask, access: 'read'},
        {name: 'shown', buffer: shown, access: 'read_write'}
      ],
      body: `var nanBits = 0x7fc00000u;
  shown[shownOffset + index] = select(bitcast<f32>(nanBits), residuals[residualsOffset + index], mask[maskOffset + index] != 0u);`
    });
    return {
      outputs: {
        mesh: uploadRegressionMap(builder, inputs),
        residuals: shown,
        critical: builder.upload('critical', 'float32', new Float32Array(64).fill(CRITICAL_VALUE)),
        statistics: addTableColumn(builder, 'tests', tests, 3, 0, 6)
      },
      update: current =>
        response.write(
          createResponse(inputs, {
            noise: 0.4,
            lagRho: current.lagRho,
            errorLambda: current.errorLambda > 0 ? current.errorLambda : undefined
          })
        )
    };
  },
  rebuildOn: ['restandardizeRows'],
  // The pointer along x picks one of the six tests: its bar is boxed and read against the 5% critical value.
  interaction: {
    kind: 'probe',
    hint: 'hover a test',
    rest: [0.3, 0.5],
    resolve(pointer, {outputs, world, peek}) {
      const statistics = peek(outputs.statistics);
      if (!statistics) return {};
      const test = Math.min(
        TEST_NAMES.length - 1,
        Math.max(0, Math.floor(((pointer.unitX - 0.05) / 0.9) * TEST_NAMES.length))
      );
      const [left, bottom, right, top] = TEST_RECTANGLE;
      const slot = ((right - left) * world.width) / TEST_NAMES.length;
      const x = left * world.width + test * slot;
      return {
        focus: test,
        glyphs: [
          {
            shape: 'box',
            minimum: [x, bottom * world.height],
            maximum: [x + slot, top * world.height],
            dashed: true
          }
        ],
        readout: `${TEST_NAMES[test]} ${statistics[test].toFixed(1)} · critical ${CRITICAL_VALUE} (${statistics[test] > CRITICAL_VALUE ? 'reject' : 'keep'})`
      };
    }
  },
  draw: ({inputs, outputs, world}) => {
    // The six tests by name under the bars, the statistic on y, the 5% critical value on the rule.
    const rectangle = getChartRectangle(world, TEST_RECTANGLE);
    const axes = createChartAxes({
      rectangle,
      y: {range: [0, TEST_LIMIT], ticks: [0, 4, 8, 12]},
      references: [{value: CRITICAL_VALUE, label: '5% critical'}]
    });
    const text = mergeText(axes, createBarNames(rectangle, TEST_NAMES, 'top'));
    return [
      ...drawResidualMap(inputs, outputs.mesh, outputs.residuals),
      {layer: 'text', ...axes.grid},
      {
        layer: 'chart',
        kind: 'bars',
        values: outputs.statistics,
        color: [0.55, 0.65, 0.95, 0.9],
        range: [0, TEST_LIMIT],
        rectangle: TEST_RECTANGLE
      },
      {
        layer: 'chart',
        kind: 'line',
        values: outputs.critical,
        range: [0, TEST_LIMIT],
        color: [1, 0.55, 0.3, 0.9],
        radius: 1,
        rectangle: TEST_RECTANGLE
      },
      {layer: 'text', ...text}
    ];
  }
});
