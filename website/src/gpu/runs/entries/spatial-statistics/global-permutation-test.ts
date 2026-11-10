// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGlobalPermutationTest: the statistic of the observed values against its distribution when the
// values are shuffled over the points. The bars are the GPU-written histogram of the shuffled
// statistics; the small chart is the summary (observed value, pseudo p-value, exceedances, ...).

import {
  getGPUPermutationParameterValues,
  GPUGlobalPermutationTest,
  GPU_GLOBAL_PERMUTATION_SUMMARY,
  GPU_PERMUTATION_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {findParameter, getFractionOf, getParameterAt} from '../../interaction';
import {uploadSpatialWeights} from '../../inputs';
import {drawReferenceHistogram} from './shared/permutation-chart';
import {createNeighborGraph, createPointField, mixNoise} from './shared/point-field';
import {parameters} from './global-permutation-test.parameters';

const STATISTICS = ['moran', 'geary', 'getisOrdG'] as const;
const STATISTIC_LABELS = ['Moran I', 'Geary C', 'Getis-Ord G'] as const;
const MAXIMUM_PERMUTATIONS = 999;
const HISTOGRAM_BINS = 28;
const COUNT_PARAMETER = findParameter(parameters, 'permutationCount');
const CHART = [0.1, 0.04, 0.9, 0.22] as const;

export default defineOperationRun({
  parameters,
  caption: 'Observed statistic against shuffles of the values: more shuffles settle the p-value.',
  createInputs: ({random, world}) => createPointField(random, world, 500),
  build(builder, points, values) {
    const graph = createNeighborGraph(points.positions, 6);
    const valueBuffer = builder.allocate('values', 'float32', points.count);
    const parameterBuffer = builder.allocate(
      'permutation',
      'uint32',
      GPU_PERMUTATION_PARAMETER_LENGTH
    );
    const summary = builder.allocate('summary', 'float32', GPU_GLOBAL_PERMUTATION_SUMMARY.length);
    const histogram = builder.allocate('histogram', 'uint32', HISTOGRAM_BINS);
    const reference = builder.allocate('reference', 'float32', MAXIMUM_PERMUTATIONS);
    const bins = builder.allocate('bins', 'float32', HISTOGRAM_BINS);
    builder.graph.add(
      new GPUGlobalPermutationTest({
        weights: uploadSpatialWeights(builder, graph.weights),
        values: valueBuffer.view,
        statistic: STATISTICS[values.statistic],
        parameters: parameterBuffer.view,
        maximumPermutations: MAXIMUM_PERMUTATIONS,
        output: {
          summary: summary.view,
          histogram: histogram.view,
          referenceDistribution: reference.view
        }
      })
    );
    return {
      outputs: {
        valueBuffer,
        summary,
        reference,
        bins,
        positions: builder.upload('positions', 'float32x2', points.positions)
      },
      update: current => {
        valueBuffer.write(mixNoise(points.field, points.noise, current.noise));
        parameterBuffer.write(
          getGPUPermutationParameterValues(
            {seed: current.seed, permutationCount: current.permutationCount},
            {maximumPermutations: MAXIMUM_PERMUTATIONS}
          )
        );
      }
    };
  },
  rebuildOn: ['statistic'],
  // The pointer's x is the permutation count: the histogram fills and the p-value settles.
  interaction: {
    kind: 'time',
    hint: 'sweep the shuffles',
    rest: [getFractionOf(COUNT_PARAMETER, COUNT_PARAMETER.default), 0.5],
    resolve(pointer, {outputs, values, world, peek}) {
      const permutationCount = getParameterAt(COUNT_PARAMETER, pointer.unitX);
      const summary = peek(outputs.summary);
      const observed = summary?.[GPU_GLOBAL_PERMUTATION_SUMMARY.observed] ?? NaN;
      const {glyphs} = drawReferenceHistogram({
        histogram: outputs.bins,
        samples: peek(outputs.reference),
        sampleCount: permutationCount,
        observed,
        rectangle: CHART,
        world
      });
      const pseudoP = summary?.[GPU_GLOBAL_PERMUTATION_SUMMARY.pseudoPValue];
      return {
        values: {permutationCount},
        glyphs,
        readout: `${permutationCount} shuffles · ${STATISTIC_LABELS[values.statistic]} ${
          Number.isFinite(observed) ? observed.toFixed(3) : '…'
        } · p ${pseudoP === undefined ? '…' : pseudoP.toFixed(3)}`
      };
    }
  },
  draw: ({outputs, theme}) => [
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.valueBuffer,
      radius: 4,
      range: [0, 1]
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.bins,
      range: [0, 1],
      rectangle: CHART,
      color: [...theme.categories[0], 0.9]
    }
  ]
});
