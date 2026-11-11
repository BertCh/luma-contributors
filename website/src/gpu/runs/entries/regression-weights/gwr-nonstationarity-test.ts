// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGeographicallyWeightedRegressionNonstationarityTest asks whether each coefficient surface varies
// more than chance: it compares the spatial standard deviation of the fitted surface (bars) with
// the same statistic after shuffling locations (dots, the simulated mean). The first slope drifts
// in the data, so its bar rises above the dots when `drift` is large; the pseudo p-value of every
// coefficient is drawn below. The regression runs in the same graph.

import {
  GPUGeographicallyWeightedRegression,
  GPUGeographicallyWeightedRegressionNonstationarityTest,
  GPU_GWR_NONSTATIONARITY_SUMMARY,
  GPU_GWR_NONSTATIONARITY_TABLE_STRIDE,
  getGPUGeographicallyWeightedRegressionParameterLength,
  getGPUGeographicallyWeightedRegressionParameterValues,
  getGPUPermutationParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {bindAxes} from '../../interaction';
import {createChartAxes, getChartRectangle} from '../../text';
import {parameters} from './gwr-nonstationarity-test.parameters';
import {
  createRegressionInputs,
  createResponse,
  drawTractMap,
  getDriftingCoefficients,
  MAP_SHARE,
  uploadRegressionMap
} from './shared/regression';
import {addTableColumn} from './shared/weights';

const LADDER_CAPACITY = 4;
const MAXIMUM_PERMUTATIONS = 32;
const COEFFICIENT_NAMES = ['int', 'x1', 'x2'];
const OBSERVED_RECTANGLE = [MAP_SHARE + 0.08, 0.42, 0.97, 0.9] as const;
const PSEUDO_P_RECTANGLE = [MAP_SHARE + 0.08, 0.08, 0.97, 0.34] as const;

export default defineOperationRun({
  parameters,
  caption:
    'Local slope (left); observed variation (bars), shuffled variation (dots), p-values (below).',
  createInputs: ({random, world}) => createRegressionInputs(random, world, 8, MAP_SHARE),
  build(builder, inputs) {
    const rowCount = inputs.tracts.count;
    const positions = builder.upload('positions', 'float32x2', inputs.tracts.centers);
    const predictors = builder.upload('predictors', 'float32', inputs.predictors);
    const response = builder.allocate('response', 'float32', rowCount);
    const bandwidthParameters = builder.allocate(
      'bandwidth-parameters',
      'float32',
      getGPUGeographicallyWeightedRegressionParameterLength(LADDER_CAPACITY)
    );
    const permutationParameters = builder.allocate('permutation-parameters', 'uint32', 4);
    const coefficients = builder.allocate('coefficients', 'float32', rowCount * 3);
    const selectedBandwidth = builder.allocate('selected-bandwidth', 'float32', 2);
    builder.graph.add(
      new GPUGeographicallyWeightedRegression({
        positions: positions.view,
        predictors: predictors.view,
        predictorCount: 2,
        response: response.view,
        parameters: bandwidthParameters.view,
        maximumBandwidthCount: LADDER_CAPACITY,
        maximumNeighborCount: 32,
        refinement: false,
        output: {coefficients: coefficients.view, selectedBandwidth: selectedBandwidth.view}
      })
    );
    const table = builder.allocate('table', 'float32', 3 * GPU_GWR_NONSTATIONARITY_TABLE_STRIDE);
    builder.graph.add(
      new GPUGeographicallyWeightedRegressionNonstationarityTest({
        positions: positions.view,
        predictors: predictors.view,
        predictorCount: 2,
        response: response.view,
        bandwidthParameters: bandwidthParameters.view,
        selectedBandwidth: selectedBandwidth.view,
        coefficients: coefficients.view,
        parameters: permutationParameters.view,
        maximumPermutations: MAXIMUM_PERMUTATIONS,
        maximumBandwidthCount: LADDER_CAPACITY,
        maximumNeighborCount: 32,
        output: {
          table: table.view,
          summary: builder.allocate('summary', 'float32', GPU_GWR_NONSTATIONARITY_SUMMARY.length)
            .view,
          standardDeviations: builder.allocate(
            'standard-deviations',
            'float32',
            (MAXIMUM_PERMUTATIONS + 1) * 3
          ).view
        }
      })
    );
    const markerPositions = builder.upload(
      'marker-positions',
      'float32',
      Float32Array.from([0.5 / 3, 1.5 / 3, 2.5 / 3])
    );
    return {
      outputs: {
        mesh: uploadRegressionMap(builder, inputs),
        slope: addTableColumn(builder, 'local-slope', coefficients, 3, 1, rowCount),
        observed: addTableColumn(
          builder,
          'observed',
          table,
          GPU_GWR_NONSTATIONARITY_TABLE_STRIDE,
          0,
          3
        ),
        simulated: addTableColumn(
          builder,
          'simulated',
          table,
          GPU_GWR_NONSTATIONARITY_TABLE_STRIDE,
          3,
          3
        ),
        pseudoP: addTableColumn(
          builder,
          'pseudo-p',
          table,
          GPU_GWR_NONSTATIONARITY_TABLE_STRIDE,
          1,
          3
        ),
        markerPositions,
        // The conventional 5% level, drawn as a line across the p-value chart.
        significance: builder.upload('significance', 'float32', new Float32Array(64).fill(0.05))
      },
      update: current => {
        response.write(
          createResponse(inputs, {
            noise: 0.3,
            coefficients: getDriftingCoefficients(inputs, current.drift)
          })
        );
        bandwidthParameters.write(
          getGPUGeographicallyWeightedRegressionParameterValues(
            {bandwidths: [current.bandwidth], kernel: 'bisquare'},
            LADDER_CAPACITY
          )
        );
        permutationParameters.write(
          getGPUPermutationParameterValues({seed: 7, permutationCount: current.permutationCount})
        );
      }
    };
  },
  // x sets how many shuffles the test runs, y the bandwidth of the fits it shuffles.
  interaction: bindAxes<unknown, {pseudoP: RunBuffer<'float32'>}>(parameters, {
    kind: 'pad',
    hint: 'permutations and bandwidth',
    x: 'permutationCount',
    y: 'bandwidth',
    decorate: (values, _pointer, {outputs, peek}) => {
      const pseudoP = peek(outputs.pseudoP);
      return {
        readout: `permutationCount ${values.permutationCount} · bandwidth ${values.bandwidth} · p(x1) ${pseudoP ? pseudoP[1].toFixed(2) : '-'}`
      };
    }
  }),
  draw: ({inputs, outputs, world, theme}) => {
    const observedAxes = createChartAxes({
      rectangle: getChartRectangle(world, OBSERVED_RECTANGLE),
      x: {categories: COEFFICIENT_NAMES},
      y: {range: [0, 1], ticks: [0.5, 1], title: 'spatial SD (dot: shuffled)'}
    });
    const pseudoPAxes = createChartAxes({
      rectangle: getChartRectangle(world, PSEUDO_P_RECTANGLE),
      x: {categories: COEFFICIENT_NAMES},
      y: {range: [0, 1], ticks: [1], title: 'pseudo p'},
      references: [{value: 0.05, label: '0.05'}]
    });
    return [
      {layer: 'text', ...observedAxes.grid},
      {layer: 'text', ...pseudoPAxes.grid},
      ...drawTractMap(inputs, outputs.mesh, outputs.slope, {ramp: 'sequential', range: [0, 3.2]}),
      {
        layer: 'chart',
        kind: 'bars',
        values: outputs.observed,
        color: [0.55, 0.65, 0.95, 0.9],
        range: [0, 1],
        rectangle: OBSERVED_RECTANGLE
      },
      {
        layer: 'chart',
        kind: 'scatter',
        values: outputs.simulated,
        xValues: outputs.markerPositions,
        xRange: [0, 1],
        range: [0, 1],
        color: [...theme.ink, 0.95],
        radius: 5,
        rectangle: OBSERVED_RECTANGLE
      },
      {
        layer: 'chart',
        kind: 'bars',
        values: outputs.pseudoP,
        range: [0, 1],
        color: [0.55, 0.65, 0.95, 0.9],
        rectangle: PSEUDO_P_RECTANGLE
      },
      {
        layer: 'chart',
        kind: 'line',
        values: outputs.significance,
        range: [0, 1],
        color: [1, 0.55, 0.3, 0.9],
        radius: 1,
        rectangle: PSEUDO_P_RECTANGLE
      },
      {layer: 'text', ...observedAxes},
      {layer: 'text', ...pseudoPAxes}
    ];
  }
});
