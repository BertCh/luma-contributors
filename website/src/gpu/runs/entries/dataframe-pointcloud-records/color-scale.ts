// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUColorScale fed by the breaks of a GPUClassBreaks on the same column, with no readback between
// them. Each record takes the palette class the scale gave it; the bars count rows per class.

import {
  GPUClassBreaks,
  GPUColorScale,
  getGPUClassBreaksParameterLength,
  getGPUClassBreaksParameterValues,
  getGPUColorScaleParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {createUniformPoints} from '../../inputs';
import {addIntegerToFloat} from './_lib/convert';
import {createRecordColumn} from './_lib/helpers';
import {parameters} from './color-scale.parameters';

const ROW_COUNT = 2000;
const MAXIMUM_CLASS_COUNT = 8;
const SCALES = ['quantize', 'threshold', 'quantile'] as const;

export default defineOperationRun({
  parameters,
  caption: 'Palette class per record from the scale; bars count rows per class.',
  createInputs: ({random, world}) => {
    const points = createUniformPoints(random, world, ROW_COUNT, 0.15);
    return {points, column: createRecordColumn(points.values)};
  },
  build(builder, inputs) {
    const column = builder.upload('fare', 'float32', inputs.column);
    const positions = builder.upload('positions', 'float32x2', inputs.points.positions);
    const breakParameters = builder.allocate(
      'break-parameters',
      'float32',
      getGPUClassBreaksParameterLength(MAXIMUM_CLASS_COUNT)
    );
    const breaks = builder.allocate('breaks', 'float32', MAXIMUM_CLASS_COUNT + 1);
    const classCount = builder.allocate('class-count', 'uint32', 1);
    builder.graph.add(
      new GPUClassBreaks({
        values: column.view,
        parameters: breakParameters.view,
        maximumClassCount: MAXIMUM_CLASS_COUNT,
        methods: ['natural-breaks'],
        output: {breaks: breaks.view, classCount: classCount.view}
      })
    );
    const palette = builder.allocate('palette', 'uint32', MAXIMUM_CLASS_COUNT);
    const scaleParameters = builder.allocate('scale-parameters', 'float32', 9);
    const classIndices = builder.allocate('class-indices', 'uint32', ROW_COUNT);
    const classCounts = builder.allocate('class-counts', 'uint32', MAXIMUM_CLASS_COUNT);
    builder.graph.add(
      new GPUColorScale({
        values: column.view,
        domain: breaks.view,
        domainCount: classCount.view,
        palette: palette.view,
        parameters: scaleParameters.view,
        maximumDomainCount: MAXIMUM_CLASS_COUNT + 1,
        maximumPaletteCount: MAXIMUM_CLASS_COUNT,
        output: {classIndices: classIndices.view, classCounts: classCounts.view}
      })
    );
    const classSizes = addIntegerToFloat(builder, 'class-sizes', classCounts);
    return {
      outputs: {column, positions, classIndices, classSizes},
      update: current => {
        breakParameters.write(
          getGPUClassBreaksParameterValues(
            {method: 'natural-breaks', classCount: current.classCount},
            MAXIMUM_CLASS_COUNT
          )
        );
        palette.write(
          Uint32Array.from({length: MAXIMUM_CLASS_COUNT}, (_, index) => 0xff000000 | (index * 30))
        );
        scaleParameters.write(
          getGPUColorScaleParameterValues({
            scale: SCALES[current.scale],
            domainCount: current.classCount + 1,
            paletteCount: current.paletteCount,
            clamp: true
          })
        );
      }
    };
  },
  rebuildOn: [],
  draw: ({outputs, values}) => [
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
      values: outputs.classIndices,
      ramp: 'sequential',
      range: [0, Math.max(values.paletteCount - 1, 1)],
      radius: 3
    },
    {
      layer: 'chart',
      kind: 'bars',
      color: [0.2, 0.55, 0.9, 1],
      values: outputs.classSizes,
      range: [0, ROW_COUNT / 2],
      rectangle: [0.62, 0.06, 0.97, 0.4]
    }
  ]
});
