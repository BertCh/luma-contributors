// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCrossVariogram of two variables observed at the same points (dots: first variable). Each
// bar is half the mean product of the two variables' differences over the pairs in a distance
// bin: positive when they vary together, negative when they oppose.

import {GPUCrossVariogram} from '@luma-contributors/analysis/dataframe-analysis';
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
import {createFieldSample, getFieldValues} from './_lib/helpers';
import {parameters} from './cross-variogram.parameters';

const POINT_COUNT = 700;
const PLOT = DISTANCE_PLOT;
const PAIR_LIMIT = 40;
const RANGE = [-0.5, 2.5] as const;
const REST_FRACTION = 0.3;

export default defineOperationRun({
  parameters,
  caption:
    'Cross semivariance of log flash energy and log footprint area by distance bin, against the zero line.',
  createInputs: ({random, world}) => {
    const sample = createFieldSample(random, world.width * 0.5, POINT_COUNT);
    return {...sample, world, pairs: createPairProbe(sample.positions, PAIR_LIMIT)};
  },
  build(builder, sample, values) {
    const positions = builder.upload('positions', 'float32x2', sample.positions);
    const first = builder.upload('first', 'float32', getFieldValues(sample, 0));
    const second = builder.allocate('second', 'float32', POINT_COUNT);
    const parameterBuffer = builder.allocate('parameters', 'float32', 2);
    const semivariances = builder.allocate('semivariances', 'float32', values.lagCount);
    builder.graph.add(
      new GPUCrossVariogram({
        positions: positions.view,
        values: first.view,
        secondValues: second.view,
        parameters: parameterBuffer.view,
        lagCount: values.lagCount,
        output: {
          semivariances: semivariances.view,
          pairCounts: builder.allocate('pair-counts', 'uint32', values.lagCount).view,
          status: builder.allocate('status', 'uint32', values.lagCount).view
        }
      })
    );
    const pairs = builder.allocate('pairs', 'float32x4', PAIR_LIMIT);
    const zero = builder.upload(
      'zero',
      'float32x4',
      createRuleSegments(PLOT, sample.world, RANGE, [0])
    );
    const centers = builder.allocate('centers', 'float32', values.lagCount);
    return {
      outputs: {positions, first, semivariances, pairs, zero, centers},
      update: current => {
        centers.write(
          Float32Array.from(
            {length: values.lagCount},
            (_, bin) => ((bin + 0.5) / values.lagCount) * current.maximumDistance
          )
        );
        const mix = Math.sqrt(1 - current.mix ** 2);
        second.write(
          Float32Array.from(
            sample.field,
            (value, index) => 5 + current.mix * (value - 5) + mix * (sample.secondField[index] - 5)
          )
        );
        parameterBuffer.write(Float32Array.of(current.maximumDistance, 1));
      }
    };
  },
  rebuildOn: ['lagCount'],
  // The pointer is a lag on the chart's distance axis; the pairs in that bin are drawn on the map.
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
        readout: `lag ${(bin * binWidth).toFixed(2)}-${((bin + 1) * binWidth).toFixed(2)} · cross γ ${
          gamma === undefined ? '…' : gamma.toFixed(2)
        } · ${found.total} pairs`
      };
    }
  },
  draw: ({outputs, values}) => [
    {
      layer: 'dots',
      context: true,
      positions: outputs.positions,
      values: outputs.first,
      range: [1, 7],
      radius: 2.8
    },
    {layer: 'lines', segments: outputs.pairs, count: PAIR_LIMIT, color: [1, 0.6, 0.2, 0.55]},
    {layer: 'lines', context: true, segments: outputs.zero, count: 1, color: [1, 1, 1, 0.7]},
    {
      layer: 'chart',
      kind: 'scatter',
      values: outputs.semivariances,
      xValues: outputs.centers,
      xRange: [0, values.maximumDistance],
      range: RANGE,
      rectangle: PLOT,
      radius: 3.4,
      color: [0.3, 0.75, 1, 1]
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.semivariances,
      range: RANGE,
      rectangle: PLOT,
      radius: 2.6,
      color: [0.3, 0.75, 1, 1]
    }
  ]
});
