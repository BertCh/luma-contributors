// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUVariogram of observations of a spatial field (left panel, colored by value). Each bar is the
// semivariance of the point pairs in one distance bin: it rises with distance until the sill, and
// its intercept at zero distance is the nugget (noise).

import {
  GPUVariogram,
  GPU_VARIOGRAM_PARAMETER_LENGTH,
  getGPUVariogramParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {
  DISTANCE_PLOT,
  createPairProbe,
  createRuleSegments,
  getAxisFraction,
  getAxisPointer,
  getBinBox,
  getBinIndex
} from './_lib/distance-probe';
import {createFieldSample, getFieldValues, getVariance} from './_lib/helpers';
import {parameters} from './variogram.parameters';

const POINT_COUNT = 700;
const PLOT = DISTANCE_PLOT;
const PAIR_LIMIT = 40;
const RANGE = [0, 2.2] as const;
const REST_FRACTION = 0.3;

export default defineOperationRun({
  parameters,
  caption:
    'Semivariance of log flash energy by distance bin; the white rule is the sill (field variance).',
  createInputs: ({random, world}) => {
    const sample = createFieldSample(random, world.width * 0.5, POINT_COUNT);
    return {...sample, world, pairs: createPairProbe(sample.positions, PAIR_LIMIT)};
  },
  build(builder, sample, values) {
    const positions = builder.upload('positions', 'float32x2', sample.positions);
    const observed = builder.allocate('values', 'float32', POINT_COUNT);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      GPU_VARIOGRAM_PARAMETER_LENGTH
    );
    const semivariances = builder.allocate('semivariances', 'float32', values.lagCount);
    builder.graph.add(
      new GPUVariogram({
        positions: positions.view,
        values: observed.view,
        parameters: parameterBuffer.view,
        gridSize: [32, 32],
        lagCount: values.lagCount,
        output: {semivariances: semivariances.view}
      })
    );
    const pairs = builder.allocate('pairs', 'float32x4', PAIR_LIMIT);
    const sill = builder.allocate('sill', 'float32x4', 1);
    return {
      outputs: {positions, observed, semivariances, pairs, sill},
      update: current => {
        const noisy = getFieldValues(sample, current.noise);
        observed.write(noisy);
        sill.write(createRuleSegments(PLOT, sample.world, RANGE, [getVariance(noisy)]));
        parameterBuffer.write(
          getGPUVariogramParameterValues({
            bounds: [0, 0, sample.windowWidth, 1],
            maximumDistance: current.maximumDistance
          })
        );
      }
    };
  },
  rebuildOn: ['lagCount'],
  // The pointer is a lag on the chart's distance axis; the pairs of points that fall in that bin
  // are drawn on the map.
  interaction: {
    kind: 'probe',
    hint: 'hover a lag',
    rest: [getAxisPointer(REST_FRACTION), 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const lagCount = values.lagCount;
      const bin = getBinIndex(getAxisFraction(pointer.unitX), lagCount);
      const binWidth = values.maximumDistance / lagCount;
      const found = inputs.pairs.find(bin * binWidth, (bin + 1) * binWidth);
      outputs.pairs.write(found.segments);
      const gamma = peek(outputs.semivariances)?.[bin];
      return {
        spotlight: null,
        glyphs: [getBinBox(PLOT, inputs.world, bin, lagCount)],
        readout: `lag ${(bin * binWidth).toFixed(2)}-${((bin + 1) * binWidth).toFixed(2)} · γ ${
          gamma === undefined ? '…' : gamma.toFixed(2)
        } · ${found.total} pairs`
      };
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'dots',
      context: true,
      positions: outputs.positions,
      values: outputs.observed,
      range: [1, 7],
      radius: 2.8
    },
    {layer: 'lines', segments: outputs.pairs, count: PAIR_LIMIT, color: [1, 0.6, 0.2, 0.55]},
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.semivariances,
      range: RANGE,
      rectangle: PLOT,
      color: [0.3, 0.75, 1, 0.9]
    },
    {layer: 'lines', context: true, segments: outputs.sill, count: 1, color: [1, 1, 1, 0.7]}
  ]
});
