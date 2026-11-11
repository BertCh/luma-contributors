// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUInstrumentalVariablesCovariance of an OLS fit used as the structural model (the OLS runs in
// the same graph). Left: residuals per tract. Right: the standard errors of the three coefficients
// (intercept first) from the covariance the operation wrote (bars), against the non-robust OLS
// standard errors (dots). Errors that grow with x1 widen the slope's error; `HAC` adds the
// covariance of neighboring residuals through the kernel.

import {
  GPUInstrumentalVariablesCovariance,
  GPUOrdinaryLeastSquares,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {createKnnWeights} from '../../inputs';
import {createChartAxes, getChartRectangle} from '../../text';
import {parameters} from './instrumental-variables-covariance.parameters';
import {
  createRegressionInputs,
  createResponse,
  drawResidualMap,
  MAP_SHARE,
  uploadRegressionMap
} from './shared/regression';
import {addKernelPass} from './shared/kernel';
import {findTractUnder, getTractNeighbors, getTractOutline} from './shared/probes';
import {uploadWeightsBuffers} from './shared/weights';

const CHART_RECTANGLE = [MAP_SHARE + 0.08, 0.14, 0.97, 0.86] as const;

const STANDARD_ERROR_LIMIT = 0.8;
const METHODS = ['HC0', 'HC1', 'HAC'] as const;

export default defineOperationRun({
  parameters,
  caption:
    'Residuals (left); standard errors from the IV covariance (bars) against plain OLS standard errors (dots).',
  createInputs: ({random, world}) => createRegressionInputs(random, world),
  build(builder, inputs, values) {
    const rowCount = inputs.tracts.count;
    const design = builder.upload('design', 'float32', inputs.predictors);
    const instruments = builder.upload(
      'instruments',
      'float32',
      Float32Array.from({length: rowCount * 3}, (_, slot) => {
        const tract = Math.floor(slot / 3);
        const column = slot % 3;
        return column < 2
          ? inputs.predictors[tract * 2 + column]
          : inputs.predictors[tract * 2] * inputs.predictors[tract * 2 + 1];
      })
    );
    const response = builder.allocate('response', 'float32', rowCount);
    const residuals = builder.allocate('residuals', 'float32', rowCount);
    const olsStandardErrors = builder.allocate('ols-standard-errors', 'float32', 3);
    builder.graph.add(
      new GPUOrdinaryLeastSquares({
        predictors: design.view,
        response: response.view,
        predictorCount: 2,
        output: {
          status: {code: builder.allocate('ols-status', 'uint32', 1).view},
          coefficients: builder.allocate('ols-coefficients', 'float32', 3).view,
          standardErrors: olsStandardErrors.view,
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
    // HAC kernel: symmetric, unit diagonal, small weight on the neighbors (diagonally dominant).
    const neighbors = inputs.weights;
    const kernel = uploadWeightsBuffers(builder, 'kernel', createKernelCsr(neighbors, 1));
    const covariance = builder.allocate('covariance', 'float32', 9);
    const method = METHODS[values.method];
    builder.graph.add(
      new GPUInstrumentalVariablesCovariance({
        design: design.view,
        coefficientCount: 2,
        instruments: instruments.view,
        instrumentCount: 3,
        residuals: residuals.view,
        method,
        kernel: method === 'HAC' ? {...kernel.input, distances: undefined} : undefined,
        output: {covariance: covariance.view, status: builder.allocate('status', 'uint32', 1).view}
      })
    );
    // The robust standard errors are the square roots of the covariance diagonal.
    const robustStandardErrors = builder.allocate('robust-standard-errors', 'float32', 3);
    addKernelPass(builder, {
      id: 'robust-standard-errors',
      invocationCount: 3,
      bindings: [
        {name: 'covariance', buffer: covariance, access: 'read'},
        {name: 'errors', buffer: robustStandardErrors, access: 'read_write'}
      ],
      body: 'errors[errorsOffset + index] = sqrt(abs(covariance[covarianceOffset + index * 4u]));'
    });
    return {
      outputs: {
        mesh: uploadRegressionMap(builder, inputs),
        residuals,
        covariance,
        robustStandardErrors,
        olsStandardErrors,
        slots: builder.upload('slots', 'float32', Float32Array.of(0.5 / 3, 1.5 / 3, 2.5 / 3))
      },
      update: current => {
        // Errors are shared with the neighbors, so only `HAC` accounts for their covariance.
        const errors = Float32Array.from(inputs.normals, (value, tract) => {
          let shared = value;
          const first = neighbors.offsets[tract];
          const last = neighbors.offsets[tract + 1];
          for (let slot = first; slot < last; slot++)
            shared += inputs.normals[neighbors.neighbors[slot]];
          return (
            (shared / Math.sqrt(1 + last - first)) *
            0.4 *
            (1 + current.heteroskedasticity * inputs.predictors[tract * 2])
          );
        });
        const signal = createResponse(inputs, {noise: 0});
        response.write(signal.map((value, tract) => value + errors[tract]));
        kernel.weights.write(createKernelCsr(neighbors, current.kernelWeight).weights);
      }
    };
  },
  rebuildOn: ['method'],
  // The pointer picks a tract: the neighbors its HAC kernel row couples it to, and the slope's robust standard error.
  interaction: {
    kind: 'probe',
    hint: 'hover a tract',
    rest: [0.3, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const tract = findTractUnder(inputs.tracts, pointer.x, pointer.y);
      const residuals = peek(outputs.residuals);
      const robust = peek(outputs.robustStandardErrors);
      const plain = peek(outputs.olsStandardErrors);
      if (tract < 0 || !residuals || !robust || !plain) return {};
      const neighbors = getTractNeighbors(inputs.tracts, tract);
      const glyphs: RunGlyph[] = [
        getTractOutline(inputs.tracts, tract),
        ...(values.method === 2
          ? Array.from(neighbors, neighbor => getTractOutline(inputs.tracts, neighbor, 0, true))
          : [])
      ];
      return {
        focus: tract,
        glyphs,
        readout: `residual ${residuals[tract].toFixed(2)} · ${values.method === 2 ? neighbors.length : 0} HAC neighbors · se(x1) ${robust[1].toFixed(2)} vs OLS ${plain[1].toFixed(2)}`
      };
    }
  },
  draw: ({inputs, outputs, world}) => {
    const axes = createChartAxes({
      rectangle: getChartRectangle(world, CHART_RECTANGLE),
      x: {categories: ['int', 'x1', 'x2']},
      y: {
        range: [0, STANDARD_ERROR_LIMIT],
        ticks: [0.4, 0.8],
        title: 'standard error (dot: OLS)'
      }
    });
    return [
      {layer: 'text', ...axes.grid},
      ...drawResidualMap(inputs, outputs.mesh, outputs.residuals),
      {
        layer: 'chart',
        kind: 'bars',
        values: outputs.robustStandardErrors,
        color: [0.55, 0.65, 0.95, 0.9],
        // Fixed range: the methods differ in scale, which an automatic range would hide.
        range: [0, STANDARD_ERROR_LIMIT],
        rectangle: CHART_RECTANGLE
      },
      {
        layer: 'chart',
        kind: 'scatter',
        values: outputs.olsStandardErrors,
        xValues: outputs.slots,
        xRange: [0, 1],
        range: [0, STANDARD_ERROR_LIMIT],
        color: [1, 1, 1, 0.95],
        radius: 5,
        rectangle: CHART_RECTANGLE
      },
      {layer: 'text', ...axes}
    ];
  }
});

/** The symmetric neighbors CSR (the kernel must be) with a unit self slot and `weight` on links. */
function createKernelCsr(neighbors: ReturnType<typeof createKnnWeights>, weight: number) {
  const linked = Array.from({length: neighbors.rowCount}, () => new Set<number>());
  for (let row = 0; row < neighbors.rowCount; row++) {
    for (let slot = neighbors.offsets[row]; slot < neighbors.offsets[row + 1]; slot++) {
      const column = neighbors.neighbors[slot];
      if (column === row) continue;
      linked[row].add(column);
      linked[column].add(row);
    }
  }
  const offsets = new Uint32Array(neighbors.rowCount + 1);
  const columns: number[] = [];
  const weights: number[] = [];
  for (let row = 0; row < neighbors.rowCount; row++) {
    for (const column of [...linked[row], row].sort((first, second) => first - second)) {
      columns.push(column);
      weights.push(column === row ? 1 : weight);
    }
    offsets[row + 1] = columns.length;
  }
  return {
    rowCount: neighbors.rowCount,
    offsets,
    neighbors: Uint32Array.from(columns),
    weights: Float32Array.from(weights),
    distances: new Float32Array(columns.length)
  };
}
