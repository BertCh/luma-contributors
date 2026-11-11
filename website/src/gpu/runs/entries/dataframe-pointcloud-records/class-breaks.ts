// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUClassBreaks on a heavy-tailed column. The bars are the row count of each class the chosen
// method produced (GPU-written classCounts); the sorted column above them is the input.

import {
  GPUClassBreaks,
  getGPUClassBreaksParameterLength,
  getGPUClassBreaksParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {bindAxes} from '../../interaction';
import {
  createBarValueLabels,
  createChartAxes,
  getChartRectangle,
  mergeText,
  type RunLabel
} from '../../text';
import {sampleFlashes} from '../../../data/lightning';
import {addIntegerToFloat} from './_lib/convert';
import {createRuleSegments} from './_lib/distance-probe';
import {getEnergyValues} from './_lib/helpers';
import {parameters} from './class-breaks.parameters';

const MAXIMUM_CLASS_COUNT = 9;
const ROW_COUNT = 1500;
const METHODS = [
  'equal-interval',
  'quantile',
  'standard-deviation',
  'natural-breaks',
  'head-tail'
] as const;
const CURVE_RECTANGLE = [0.07, 0.56, 0.95, 0.92] as const;
const COUNT_RECTANGLE = [0.07, 0.1, 0.95, 0.5] as const;

type Inputs = {
  world: {width: number; height: number};
  source: Float32Array;
  top: number;
  sorted: Float32Array;
  /** The interior class breaks the GPU found (log1p of the energy), once read back. */
  breaks: number[];
};
type Outputs = {
  column: RunBuffer;
  curve: RunBuffer;
  curveClasses: RunBuffer;
  rules: RunBuffer;
  breaks: RunBuffer;
  classSizes: RunBuffer;
};

/** The column is log1p of the (spread-adjusted) energy, as the variogram runs use. */
/** An energy for a tick or break label: whole femtojoules, thousands as `k`. */
function formatEnergy(energy: number): string {
  const rounded = Math.round(energy);
  return rounded >= 1000 ? `${Number((rounded / 1000).toFixed(1))}k` : String(rounded);
}

function getColumnValues(energy: Float32Array, spread: number): Float32Array {
  return getEnergyValues(energy, spread).map(Math.log1p);
}

export default defineOperationRun<Inputs, Outputs, Float32Array>({
  parameters,
  caption:
    'Class counts below, and the sorted log flash energies above with the GPU breaks drawn as rules.',
  createInputs: ({world}) => {
    const energy = sampleFlashes(world, ROW_COUNT).energy;
    const largest = energy.reduce((maximum, value) => Math.max(maximum, value), 0);
    return {
      world,
      source: energy,
      // The curve's y axis stays fixed while the spread knob reshapes the column.
      top: Math.log1p(largest),
      sorted: new Float32Array(ROW_COUNT),
      breaks: []
    };
  },
  build(builder, inputs, values) {
    const column = builder.allocate('values', 'float32', ROW_COUNT);
    const curve = builder.allocate('curve', 'float32x2', ROW_COUNT);
    const curveClasses = builder.allocate('curve-classes', 'float32', ROW_COUNT);
    const rules = builder.allocate('rules', 'float32x4', MAXIMUM_CLASS_COUNT);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      getGPUClassBreaksParameterLength(MAXIMUM_CLASS_COUNT)
    );
    const breaks = builder.allocate('breaks', 'float32', MAXIMUM_CLASS_COUNT + 1);
    const classCount = builder.allocate('class-count', 'uint32', 1);
    const classCounts = builder.allocate('class-counts', 'uint32', MAXIMUM_CLASS_COUNT);
    builder.graph.add(
      new GPUClassBreaks({
        values: column.view,
        parameters: parameterBuffer.view,
        maximumClassCount: MAXIMUM_CLASS_COUNT,
        methods: METHODS,
        output: {breaks: breaks.view, classCount: classCount.view, classCounts: classCounts.view}
      })
    );
    const classSizes = addIntegerToFloat(builder, 'class-sizes', classCounts);
    return {
      outputs: {column, curve, curveClasses, rules, breaks, classSizes},
      update: current => {
        const data = getColumnValues(inputs.source, current.spread);
        column.write(data);
        inputs.sorted = Float32Array.from(data).sort();
        curve.write(
          Float32Array.from({length: ROW_COUNT * 2}, (_, index) => {
            const row = index >> 1;
            const rectangle = CURVE_RECTANGLE;
            if (index % 2 === 0) {
              return (
                (rectangle[0] + (row / (ROW_COUNT - 1)) * (rectangle[2] - rectangle[0])) *
                inputs.world.width
              );
            }
            const fraction = Math.min(1, inputs.sorted[row] / inputs.top);
            return (rectangle[1] + fraction * (rectangle[3] - rectangle[1])) * inputs.world.height;
          })
        );
        parameterBuffer.write(
          getGPUClassBreaksParameterValues(
            {method: METHODS[current.method], classCount: current.classCount},
            MAXIMUM_CLASS_COUNT
          )
        );
      }
    };
  },
  rebuildOn: [],
  // The row count of every class, read back so each bar can carry its number.
  async afterRun({outputs}) {
    const bytes = await outputs.classSizes.buffer.readAsync();
    return new Float32Array(bytes.slice().buffer);
  },
  // The pointer is a pad: x is the number of classes, y the method (five bands). The breaks the
  // GPU found are read back and drawn as rules across the sorted curve, and the curve takes the
  // class colors.
  interaction: bindAxes<Inputs, Outputs>(parameters, {
    kind: 'pad',
    hint: 'classes across, method up',
    x: 'classCount',
    y: 'method',
    decorate: (values, _pointer, {inputs, outputs, peek}) => {
      const breaks = peek(outputs.breaks);
      const classCount = values.classCount;
      if (!breaks) return {readout: `${METHODS[values.method]} · ${classCount} classes`};
      const interior = Array.from(breaks.slice(1, classCount));
      inputs.breaks = interior;
      const rules = new Float32Array(MAXIMUM_CLASS_COUNT * 4);
      rules.set(createRuleSegments(CURVE_RECTANGLE, inputs.world, [0, inputs.top], interior));
      outputs.rules.write(rules);
      outputs.curveClasses.write(
        Float32Array.from(inputs.sorted, value => interior.filter(edge => value >= edge).length)
      );
      const energies = interior.slice(0, 4).map(edge => Math.round(Math.expm1(edge)));
      return {
        readout: `${METHODS[values.method]} · ${classCount} classes · ${energies.join(', ')}${
          interior.length > 4 ? '…' : ''
        } fJ`
      };
    }
  }),
  draw: ({inputs, outputs, values, extra, world}) => {
    // Top: the sorted column on a log1p axis labelled in energy, with each GPU break named on its rule.
    const curveRectangle = getChartRectangle(world, CURVE_RECTANGLE);
    const decades = [1, 10, 100, 1000, 10000, 100000].filter(
      energy => Math.log1p(energy) < inputs.top
    );
    const curveAxes = createChartAxes({
      rectangle: curveRectangle,
      y: {
        range: [0, inputs.top],
        ticks: decades.map(Math.log1p),
        format: value => formatEnergy(Math.expm1(value)),
        title: 'energy (fJ)',
        grid: false
      }
    });
    const breakLabels: RunLabel[] = inputs.breaks.map(edge => ({
      text: formatEnergy(Math.expm1(edge)),
      at: [
        curveRectangle[2],
        curveRectangle[1] + (edge / inputs.top) * (curveRectangle[3] - curveRectangle[1])
      ],
      anchor: 'bottom-right',
      offset: [0, 1],
      role: 'value',
      group: 'class-breaks'
    }));
    // Bottom: the rows in each class, numbered, with their count above the bar.
    const countRectangle = getChartRectangle(world, COUNT_RECTANGLE);
    const classNames = Array.from({length: MAXIMUM_CLASS_COUNT}, (_, index) =>
      index < values.classCount ? String(index + 1) : ''
    );
    const countAxes = createChartAxes({
      rectangle: countRectangle,
      x: {categories: classNames, title: 'class'},
      y: {range: [0, ROW_COUNT / 2], ticks: 2, title: 'rows'}
    });
    const countText = mergeText(
      countAxes,
      createBarValueLabels({
        rectangle: countRectangle,
        values: extra ?? [],
        range: [0, ROW_COUNT / 2],
        indices: Array.from({length: values.classCount}, (_, index) => index)
      })
    );
    return [
      {layer: 'text', ...countAxes.grid},
      {
        layer: 'dots',
        positions: outputs.curve,
        values: outputs.curveClasses,
        ramp: 'categorical',
        radius: 2.6
      },
      {layer: 'lines', segments: outputs.rules, count: MAXIMUM_CLASS_COUNT, color: [1, 1, 1, 0.8]},
      {
        layer: 'chart',
        kind: 'bars',
        values: outputs.classSizes,
        color: [0.2, 0.55, 0.9, 1],
        range: [0, ROW_COUNT / 2],
        rectangle: COUNT_RECTANGLE
      },
      {layer: 'text', ...curveAxes},
      {layer: 'text', labels: breakLabels},
      {layer: 'text', ...countText}
    ];
  }
});
