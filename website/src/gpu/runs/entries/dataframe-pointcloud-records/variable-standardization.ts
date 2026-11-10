// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUVariableStandardization of a skewed record column. Dots on the map take the transformed
// value (diverging about zero).

import {GPUVariableStandardization} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {sampleFlashes} from '../../../data/lightning';
import {createHistogram, getEnergyValues} from './_lib/helpers';
import {parameters} from './variable-standardization.parameters';

const ROW_COUNT = 1500;
const BIN_COUNT = 24;
const MAP_BOTTOM = 0.3;
const STRIP = [0.07, 0.04, 0.95, 0.26] as const;
const METHODS = [
  'deviation-from-mean',
  'z-score',
  'mad',
  'range-adjust',
  'range-standardize'
] as const;

type Inputs = ReturnType<typeof createInputs>;
type Outputs = {
  column: RunBuffer;
  positions: RunBuffer;
  standardized: RunBuffer;
  rawHistogram: RunBuffer;
  standardHistogram: RunBuffer;
};

function createInputs({world}: {world: {width: number; height: number}}) {
  const points = sampleFlashes({width: world.width, height: 1 - MAP_BOTTOM}, ROW_COUNT);
  for (let row = 0; row < ROW_COUNT; row++) points.positions[row * 2 + 1] += MAP_BOTTOM;
  return {...points, world, raw: {mean: 0, deviation: 1, tallest: 1, standardTallest: 1}};
}

function getMoments(values: ArrayLike<number>) {
  let sum = 0;
  let sumOfSquares = 0;
  let count = 0;
  for (let index = 0; index < values.length; index++) {
    if (!Number.isFinite(values[index])) continue;
    sum += values[index];
    sumOfSquares += values[index] ** 2;
    count++;
  }
  const mean = sum / count;
  return {mean, deviation: Math.sqrt(Math.max(sumOfSquares / count - mean * mean, 0))};
}

function getHistogramOf(values: ArrayLike<number>): Float32Array {
  let minimum = Infinity;
  let maximum = -Infinity;
  for (let index = 0; index < values.length; index++) {
    if (!Number.isFinite(values[index])) continue;
    minimum = Math.min(minimum, values[index]);
    maximum = Math.max(maximum, values[index]);
  }
  return createHistogram(values, BIN_COUNT, minimum, maximum + (maximum - minimum) * 1e-6 + 1e-9);
}

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'Raw flash energy (left of the curtain) and its standardized value (right, diverging about zero), with the histogram of each below.',
  createInputs,
  build(builder, points, values) {
    const column = builder.allocate('values', 'float32', ROW_COUNT);
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const standardized = builder.allocate('standardized', 'float32', ROW_COUNT);
    builder.graph.add(
      new GPUVariableStandardization({
        values: column.view,
        method: METHODS[values.method],
        variance: 'sample',
        output: standardized.view
      })
    );
    const rawHistogram = builder.allocate('raw-histogram', 'float32', BIN_COUNT);
    const standardHistogram = builder.allocate('standard-histogram', 'float32', BIN_COUNT);
    return {
      outputs: {column, positions, standardized, rawHistogram, standardHistogram},
      update: current => {
        const data = getEnergyValues(points.energy, current.spread);
        column.write(data);
        const histogram = getHistogramOf(data);
        rawHistogram.write(histogram);
        points.raw = {
          ...getMoments(data),
          tallest: Math.max(...histogram),
          standardTallest: points.raw.standardTallest
        };
      }
    };
  },
  rebuildOn: ['method'],
  // The pointer is a curtain: the raw column on its left, the standardized column on its right,
  // and the histogram strip below changes from one to the other at the same line.
  interaction: {
    kind: 'swipe',
    hint: 'swipe raw and standardized',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const standardized = peek(outputs.standardized);
      if (standardized) {
        const histogram = getHistogramOf(standardized);
        outputs.standardHistogram.write(histogram);
        inputs.raw.standardTallest = Math.max(...histogram);
      }
      const moments = standardized ? getMoments(standardized) : null;
      return {
        curtain: pointer.x,
        glyphs: [{shape: 'curtain', x: pointer.x}],
        readout: moments
          ? `${METHODS[values.method]} · mean ${Math.round(inputs.raw.mean)} -> ${moments.mean.toFixed(1)} · sd ${Math.round(inputs.raw.deviation)} -> ${moments.deviation.toFixed(1)}`
          : METHODS[values.method]
      };
    }
  },
  draw: ({inputs, outputs, interaction, world}) => {
    const curtain = interaction.curtain ?? world.width / 2;
    const tallest = Math.max(inputs.raw.tallest, inputs.raw.standardTallest);
    return [
      {
        layer: 'dots',
        positions: outputs.positions,
        values: outputs.column,
        range: [0, 500],
        radius: 4.4,
        clip: {side: 'left', x: curtain}
      },
      {
        layer: 'dots',
        positions: outputs.positions,
        values: outputs.standardized,
        ramp: 'diverging',
        range: [-4, 4],
        radius: 4.4,
        clip: {side: 'right', x: curtain}
      },
      {
        layer: 'chart',
        kind: 'bars',
        values: outputs.rawHistogram,
        range: [0, tallest],
        rectangle: STRIP,
        color: [0.9, 0.6, 0.3, 1],
        clip: {side: 'left', x: curtain}
      },
      {
        layer: 'chart',
        kind: 'bars',
        values: outputs.standardHistogram,
        range: [0, tallest],
        rectangle: STRIP,
        color: [0.3, 0.7, 1, 1],
        clip: {side: 'right', x: curtain}
      }
    ];
  }
});
