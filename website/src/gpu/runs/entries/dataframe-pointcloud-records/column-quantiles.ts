// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUColumnQuantiles of a record column. The bars are the quantile values the GPU selected; the
// map shows which rows the percentile filter (outputMask) keeps.

import {
  GPUColumnQuantiles,
  getGPUColumnQuantilesParameterLength,
  getGPUColumnQuantilesParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {clamp, snapToParameter, findParameter} from '../../interaction';
import {sampleFlashes} from '../../../data/lightning';
import {parameters} from './column-quantiles.parameters';

const ROW_COUNT = 2000;
const STRIP = [0.07, 0.06, 0.95, 0.36] as const;
const KEPT_COLOR = [1, 0.8, 0.3] as const;
const DROPPED_COLOR = [0.35, 0.4, 0.5] as const;

type Inputs = {
  world: {width: number; height: number};
  points: ReturnType<typeof sampleFlashes>;
  column: Float32Array;
  /** Sorted values: the strip draws them against their rank, on a log scale. */
  sorted: Float32Array;
  logMaximum: number;
};
type Outputs = {
  column: RunBuffer;
  positions: RunBuffer;
  quantiles: RunBuffer;
  outputMask: RunBuffer;
  curve: RunBuffer;
  curveKept: RunBuffer;
  marks: RunBuffer;
};

function getStripPoint(inputs: Inputs, fraction: number, value: number): [number, number] {
  const height = Math.log1p(value) / inputs.logMaximum;
  return [
    (STRIP[0] + fraction * (STRIP[2] - STRIP[0])) * inputs.world.width,
    (STRIP[1] + height * (STRIP[3] - STRIP[1])) * inputs.world.height
  ];
}

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'Flash energy quantiles in units of 10 fJ; the strip is the sorted column on a log scale and the map shows the rows the percentile filter keeps.',
  createInputs: ({world}) => {
    // The map sits above the strip.
    const points = sampleFlashes({width: world.width, height: 1 - STRIP[3] - 0.04}, ROW_COUNT);
    for (let row = 0; row < ROW_COUNT; row++) points.positions[row * 2 + 1] += STRIP[3] + 0.04;
    const column = Float32Array.from(points.energy, value => value / 10);
    const sorted = Float32Array.from(column).sort();
    return {world, points, column, sorted, logMaximum: Math.log1p(sorted[ROW_COUNT - 1])};
  },
  build(builder, inputs, values) {
    const quantileCount = values.quantileCount;
    const column = builder.upload('flash-energy', 'float32', inputs.column);
    const positions = builder.upload('positions', 'float32x2', inputs.points.positions);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      getGPUColumnQuantilesParameterLength(quantileCount)
    );
    const quantiles = builder.allocate('quantiles', 'float32', quantileCount);
    const outputMask = builder.allocate('output-mask', 'uint32', ROW_COUNT);
    builder.graph.add(
      new GPUColumnQuantiles({
        values: column.view,
        quantileCount,
        parameters: parameterBuffer.view,
        output: {
          quantiles: quantiles.view,
          validCount: builder.allocate('valid-count', 'uint32', 1).view,
          outputMask: outputMask.view
        }
      })
    );
    const curve = builder.upload(
      'curve',
      'float32x2',
      Float32Array.from({length: ROW_COUNT * 2}, (_, index) => {
        const row = index >> 1;
        return getStripPoint(inputs, row / (ROW_COUNT - 1), inputs.sorted[row])[index & 1];
      })
    );
    const curveKept = builder.allocate('curve-kept', 'uint32', ROW_COUNT);
    const marks = builder.allocate('marks', 'float32x2', quantileCount);
    return {
      outputs: {column, positions, quantiles, outputMask, curve, curveKept, marks},
      update: current => {
        parameterBuffer.write(
          getGPUColumnQuantilesParameterValues({
            quantiles: Array.from(
              {length: quantileCount},
              (_, index) => (index + 1) / (quantileCount + 1)
            ),
            filterRange: [current.filterLower / 100, current.filterUpper / 100]
          })
        );
        curveKept.write(
          Uint32Array.from({length: ROW_COUNT}, (_, row) => {
            const rank = (row / (ROW_COUNT - 1)) * 100;
            return rank >= current.filterLower && rank <= current.filterUpper ? 1 : 0;
          })
        );
      }
    };
  },
  rebuildOn: ['quantileCount'],
  // The pointer is a window on the sorted column: x is the center of the percentile range, y its
  // width. The GPU's quantiles are marked on the curve.
  interaction: {
    kind: 'window',
    hint: 'center and width of the range',
    rest: [0.5, 0.25],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const lowerParameter = findParameter(parameters, 'filterLower');
      const upperParameter = findParameter(parameters, 'filterUpper');
      const center = clamp((pointer.unitX - 0.05) / 0.9, 0, 1) * 100;
      const halfWidth = 5 + 45 * clamp((0.95 - pointer.unitY) / 0.9, 0, 1);
      const lower = snapToParameter(lowerParameter, center - halfWidth);
      const upper = snapToParameter(upperParameter, center + halfWidth);
      const marks = peek(outputs.quantiles);
      if (marks) {
        outputs.marks.write(
          Float32Array.from({length: marks.length * 2}, (_, index) => {
            const row = index >> 1;
            return getStripPoint(inputs, (row + 1) / (marks.length + 1), marks[row])[index & 1];
          })
        );
      }
      const mask = peek(outputs.outputMask);
      let kept = 0;
      for (let row = 0; mask && row < mask.length; row++) kept += mask[row];
      const lowValue = inputs.sorted[Math.floor((lower / 100) * (ROW_COUNT - 1))] * 10;
      const highValue = inputs.sorted[Math.floor((upper / 100) * (ROW_COUNT - 1))] * 10;
      const [left, bottom] = getStripPoint(inputs, lower / 100, 0);
      const [right] = getStripPoint(inputs, upper / 100, 0);
      return {
        values: {filterLower: lower, filterUpper: upper},
        glyphs: [
          {
            shape: 'box',
            minimum: [left, bottom],
            maximum: [right, STRIP[3] * inputs.world.height]
          }
        ],
        readout: `filterRange ${lower}-${upper} %${mask ? ` · ${kept} kept` : ''} · ${Math.round(lowValue)}-${Math.round(highValue)} fJ`
      };
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'dots',
      context: true,
      positions: outputs.positions,
      values: outputs.column,
      range: [0, 50],
      radius: 3
    },
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.outputMask,
      ramp: 'categorical',
      categories: [DROPPED_COLOR, KEPT_COLOR],
      radius: 2.8
    },
    {
      layer: 'dots',
      positions: outputs.curve,
      values: outputs.curveKept,
      ramp: 'categorical',
      categories: [DROPPED_COLOR, KEPT_COLOR],
      radius: 1.6
    },
    {layer: 'dots', positions: outputs.marks, color: [1, 1, 1, 1], radius: 5.5}
  ]
});
