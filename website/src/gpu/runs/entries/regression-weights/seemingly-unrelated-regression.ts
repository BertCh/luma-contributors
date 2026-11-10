// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSeeminglyUnrelatedRegression fits two equations on the same tracts whose errors are
// correlated. Left: residuals of the first equation. Right: the six coefficients (bars) with the
// true values (dots); below, every tract as a point of its two residuals, which tilt together as
// the correlation grows.

import {GPUSeeminglyUnrelatedRegression} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './seemingly-unrelated-regression.parameters';
import {
  createCoefficientPanel,
  createRegressionInputs,
  drawCoefficientPanel,
  drawResidualMap,
  MAP_SHARE,
  RESIDUAL_RANGE,
  uploadRegressionMap
} from './shared/regression';
import {findTractUnder, getTractOutline} from './shared/probes';
import {addKernelPass} from './shared/kernel';

const SCATTER_RECTANGLE = [MAP_SHARE + 0.08, 0.06, 0.97, 0.36] as const;

const COEFFICIENTS = [1, 2, -1.5, 0.5, -1, 2] as const;

export default defineOperationRun({
  parameters,
  caption:
    'Two-equation SUR: coefficients (bars) against the truth (dots), residual covariance below.',
  createInputs: ({random, world}) => createRegressionInputs(random, world),
  build(builder, inputs) {
    const rowCount = inputs.tracts.count;
    const predictors = builder.upload(
      'predictors',
      'float32',
      Float32Array.from([...inputs.predictors, ...inputs.predictors])
    );
    const responses = builder.allocate('responses', 'float32', rowCount * 2);
    const coefficients = builder.allocate('coefficients', 'float32', 6);
    const residuals = builder.allocate('residuals', 'float32', rowCount * 2);
    const residualCovariance = builder.allocate('residual-covariance', 'float32', 4);
    builder.graph.add(
      new GPUSeeminglyUnrelatedRegression({
        equationCount: 2,
        predictorCount: 2,
        rowCount,
        predictors: predictors.view,
        responses: responses.view,
        output: {
          coefficients: coefficients.view,
          covariance: builder.allocate('covariance', 'float32', 36).view,
          residualCovariance: residualCovariance.view,
          residuals: residuals.view,
          status: builder.allocate('status', 'uint32', 1).view
        }
      })
    );
    const panel = createCoefficientPanel(builder, coefficients, 1, COEFFICIENTS);
    // The two equations' residuals as separate columns, for the scatter.
    const firstResiduals = builder.allocate('first-residuals', 'float32', rowCount);
    const secondResiduals = builder.allocate('second-residuals', 'float32', rowCount);
    addKernelPass(builder, {
      id: 'split-residuals',
      invocationCount: rowCount,
      bindings: [
        {name: 'residuals', buffer: residuals, access: 'read'},
        {name: 'first', buffer: firstResiduals, access: 'read_write'},
        {name: 'second', buffer: secondResiduals, access: 'read_write'}
      ],
      declarations: `const ROW_COUNT: u32 = ${rowCount}u;`,
      body: `first[firstOffset + index] = residuals[residualsOffset + index];
  second[secondOffset + index] = residuals[residualsOffset + ROW_COUNT + index];`
    });
    return {
      outputs: {
        mesh: uploadRegressionMap(builder, inputs),
        panel,
        residualCovariance,
        residualBuffer: residuals,
        firstResiduals,
        secondResiduals
      },
      update: current => {
        const first = new Float32Array(rowCount);
        const second = new Float32Array(rowCount);
        const mix = Math.sqrt(1 - current.correlation ** 2);
        for (let tract = 0; tract < rowCount; tract++) {
          const own = inputs.normals[(tract * 7 + 3) % rowCount];
          const error = inputs.normals[tract];
          const signalFirst =
            COEFFICIENTS[0] +
            COEFFICIENTS[1] * inputs.predictors[tract * 2] +
            COEFFICIENTS[2] * inputs.predictors[tract * 2 + 1];
          const signalSecond =
            COEFFICIENTS[3] +
            COEFFICIENTS[4] * inputs.predictors[tract * 2] +
            COEFFICIENTS[5] * inputs.predictors[tract * 2 + 1];
          first[tract] = signalFirst + error * current.noise;
          second[tract] = signalSecond + (current.correlation * error + mix * own) * current.noise;
        }
        responses.write(Float32Array.from([...first, ...second]));
      }
    };
  },
  // The pointer picks a tract: its outline on the map and its point in the residual scatter, linked.
  interaction: {
    kind: 'probe',
    hint: 'hover a tract',
    rest: [0.3, 0.5],
    resolve(pointer, {inputs, outputs, world, peek}) {
      const tract = findTractUnder(inputs.tracts, pointer.x, pointer.y);
      const first = peek(outputs.firstResiduals);
      const second = peek(outputs.secondResiduals);
      const covariance = peek(outputs.residualCovariance);
      if (tract < 0 || !first || !second) return {};
      const [low, high] = RESIDUAL_RANGE;
      const along = (value: number) => Math.min(1, Math.max(0, (value - low) / (high - low)));
      const [left, bottom, right, top] = SCATTER_RECTANGLE;
      return {
        focus: tract,
        glyphs: [
          getTractOutline(inputs.tracts, tract),
          {
            shape: 'ring',
            at: [
              (left + along(first[tract]) * (right - left)) * world.width,
              (bottom + along(second[tract]) * (top - bottom)) * world.height
            ],
            radius: 0.012
          }
        ],
        readout: `tract ${tract} · residuals ${first[tract].toFixed(2)}, ${second[tract].toFixed(2)} · sigma12 ${covariance ? covariance[1].toFixed(2) : '-'}`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    ...drawResidualMap(inputs, outputs.mesh, outputs.firstResiduals),
    ...drawCoefficientPanel(outputs.panel, {rectangle: [MAP_SHARE + 0.08, 0.42, 0.97, 0.9]}),
    // Each tract as a point of its two equations' residuals; the cloud tilts with the correlation.
    {
      layer: 'chart',
      kind: 'scatter',
      values: outputs.secondResiduals,
      xValues: outputs.firstResiduals,
      xRange: RESIDUAL_RANGE,
      range: RESIDUAL_RANGE,
      radius: 2.6,
      color: [0.55, 0.65, 0.95, 0.9],
      rectangle: SCATTER_RECTANGLE
    }
  ]
});
