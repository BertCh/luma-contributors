// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUScalarScale mapping a record column to line widths. Each dot is one record: x is the input
// value, y the width the GPU scale wrote. The faint diagonal is the unscaled input, clipped at the top of the output axis.

import {
  GPUScalarScale,
  getGPUScalarScaleParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {bindAxes} from '../../interaction';
import {sampleFlashes} from '../../../data/lightning';
import {parameters} from './scalar-scale.parameters';

const ROW_COUNT = 1200;
const SCALES = ['linear', 'sqrt', 'log'] as const;
const PLOT = [0.08, 0.1, 0.94, 0.9] as const;
const INPUT_RANGE = 80;
const OUTPUT_RANGE = 30;

export default defineOperationRun({
  parameters,
  caption: 'Measured flash energy in units of 10 fJ (x) and the scaled width (y).',
  createInputs: ({world}) =>
    Float32Array.from(sampleFlashes(world, ROW_COUNT).energy, value => value / 10),
  build(builder, column) {
    const input = builder.upload('values', 'float32', column);
    const domain = builder.allocate('domain', 'float32', 2);
    const range = builder.allocate('range', 'float32', 2);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const widths = builder.allocate('widths', 'float32', ROW_COUNT);
    builder.graph.add(
      new GPUScalarScale({
        values: input.view,
        domain: domain.view,
        range: range.view,
        parameters: parameterBuffer.view,
        maximumDomainCount: 2,
        output: {values: widths.view}
      })
    );
    return {
      outputs: {input, widths},
      update: current => {
        domain.write(Float32Array.of(1, current.domainMaximum));
        range.write(Float32Array.of(1, current.rangeMaximum));
        parameterBuffer.write(
          getGPUScalarScaleParameterValues({scale: SCALES[current.scale], domainCount: 2})
        );
      }
    };
  },
  rebuildOn: [],
  // The pointer is the clamp knee of the scale: x is the domain maximum, y the range maximum.
  interaction: bindAxes(parameters, {
    kind: 'pad',
    hint: 'move the clamp knee',
    x: 'domainMaximum',
    y: 'rangeMaximum',
    decorate: (values, _pointer, {world}) => {
      const x =
        (PLOT[0] + (values.domainMaximum / INPUT_RANGE) * (PLOT[2] - PLOT[0])) * world.width;
      const y =
        (PLOT[1] + (values.rangeMaximum / OUTPUT_RANGE) * (PLOT[3] - PLOT[1])) * world.height;
      return {
        glyphs: [{shape: 'crosshair', at: [x, y], size: 10}],
        readout: `${SCALES[values.scale]} · domain 1-${values.domainMaximum} · range 1-${values.rangeMaximum}`
      };
    }
  }),
  draw: ({outputs}) => [
    {
      layer: 'chart',
      kind: 'scatter',
      context: true,
      values: outputs.input,
      xValues: outputs.input,
      xRange: [0, INPUT_RANGE],
      radius: 2,
      range: [0, OUTPUT_RANGE],
      rectangle: PLOT
    },
    {
      layer: 'chart',
      kind: 'scatter',
      values: outputs.widths,
      xValues: outputs.input,
      xRange: [0, INPUT_RANGE],
      range: [0, OUTPUT_RANGE],
      rectangle: PLOT,
      radius: 7,
      color: [0.2, 0.7, 1, 1]
    }
  ]
});
