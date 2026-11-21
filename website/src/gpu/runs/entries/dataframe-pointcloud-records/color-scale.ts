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
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {bindAxes} from '../../interaction';
import {createChartAxes, getChartRectangle, type RunLabel} from '../../text';
import {sampleFlashes} from '../../../data/lightning';
import {addIntegerToFloat} from './_lib/convert';
import {parameters} from './color-scale.parameters';

const ROW_COUNT = 2000;
const MAXIMUM_CLASS_COUNT = 8;
const SCALES = ['quantize', 'threshold', 'quantile'] as const;
const MAP_BOTTOM = 0.3;
const SWATCH_RECTANGLE = [0.07, 0.08, 0.55, 0.2] as const;
const COUNT_RECTANGLE = [0.62, 0.05, 0.97, 0.24] as const;
// A categorical palette of eight distinguishable colors, the scale's output range.
const PALETTE: readonly (readonly [number, number, number])[] = [
  [0.27, 0.55, 0.85],
  [0.95, 0.6, 0.2],
  [0.35, 0.72, 0.4],
  [0.85, 0.35, 0.55],
  [0.6, 0.45, 0.85],
  [0.9, 0.8, 0.3],
  [0.25, 0.75, 0.75],
  [0.8, 0.3, 0.25]
];

function packColor(color: readonly [number, number, number]): number {
  const [red, green, blue] = color.map(channel => Math.round(channel * 255));
  return (0xff000000 | (blue << 16) | (green << 8) | red) >>> 0;
}

/** One square swatch per palette color along the legend strip; unused slots collapse. */
function getSwatchPositions(world: {width: number; height: number}, paletteCount: number) {
  const positions = new Float32Array(MAXIMUM_CLASS_COUNT * 8);
  const left = SWATCH_RECTANGLE[0] * world.width;
  const width = ((SWATCH_RECTANGLE[2] - SWATCH_RECTANGLE[0]) * world.width) / MAXIMUM_CLASS_COUNT;
  const bottom = SWATCH_RECTANGLE[1] * world.height;
  const top = SWATCH_RECTANGLE[3] * world.height;
  for (let swatch = 0; swatch < MAXIMUM_CLASS_COUNT; swatch++) {
    const size = swatch < paletteCount ? 0.9 : 0;
    const x = left + swatch * width;
    positions.set([x, bottom, x + width * size, bottom, x + width * size, top, x, top], swatch * 8);
  }
  return positions;
}

type Inputs = {
  world: {width: number; height: number};
  points: ReturnType<typeof sampleFlashes>;
  column: Float32Array;
};
type Outputs = {
  column: RunBuffer;
  positions: RunBuffer;
  classIndices: RunBuffer;
  classSizes: RunBuffer;
  swatches: RunBuffer;
  swatchIndices: RunBuffer;
  swatchOwners: RunBuffer;
  swatchClasses: RunBuffer;
};

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'Flash energies mapped to palette classes: the legend strip is the scale’s palette, the bars count rows per class.',
  createInputs: ({world}) => {
    const points = sampleFlashes({width: world.width, height: 1 - MAP_BOTTOM}, ROW_COUNT);
    for (let row = 0; row < ROW_COUNT; row++) points.positions[row * 2 + 1] += MAP_BOTTOM;
    // Log energy keeps the classes balanced: raw energies put nearly every flash in class 0.
    return {world, points, column: Float32Array.from(points.energy, Math.log1p)};
  },
  build(builder, inputs) {
    const column = builder.upload('flash-energy', 'float32', inputs.column);
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
        classCount: classCount.view,
        palette: palette.view,
        parameters: scaleParameters.view,
        maximumDomainCount: MAXIMUM_CLASS_COUNT + 1,
        maximumPaletteCount: MAXIMUM_CLASS_COUNT,
        output: {classIndices: classIndices.view, classCounts: classCounts.view}
      })
    );
    const classSizes = addIntegerToFloat(builder, 'class-sizes', classCounts);
    const swatches = builder.allocate('swatches', 'float32x2', MAXIMUM_CLASS_COUNT * 4);
    const swatchIndices = builder.upload(
      'swatch-indices',
      'uint32',
      Uint32Array.from({length: MAXIMUM_CLASS_COUNT * 6}, (_, index) => {
        const corner = [0, 1, 2, 0, 2, 3][index % 6];
        return Math.floor(index / 6) * 4 + corner;
      })
    );
    const swatchOwners = builder.upload(
      'swatch-owners',
      'uint32',
      Uint32Array.from({length: MAXIMUM_CLASS_COUNT * 2}, (_, index) => index >> 1)
    );
    const swatchClasses = builder.upload(
      'swatch-classes',
      'uint32',
      Uint32Array.from({length: MAXIMUM_CLASS_COUNT}, (_, index) => index)
    );
    return {
      outputs: {
        column,
        positions,
        classIndices,
        classSizes,
        swatches,
        swatchIndices,
        swatchOwners,
        swatchClasses
      },
      update: current => {
        breakParameters.write(
          getGPUClassBreaksParameterValues(
            {method: 'natural-breaks', classCount: current.classCount},
            MAXIMUM_CLASS_COUNT
          )
        );
        palette.write(Uint32Array.from(PALETTE, packColor));
        swatches.write(getSwatchPositions(inputs.world, current.paletteCount));
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
  // The pointer is a pad on two real props of the scale: x is paletteCount, y is the scale type.
  interaction: bindAxes<Inputs, Outputs>(parameters, {
    kind: 'pad',
    hint: 'colors across, scale up',
    x: 'paletteCount',
    y: 'scale',
    decorate: (values, _pointer, {outputs, peek}) => {
      const sizes = peek(outputs.classSizes);
      let largest = 0;
      let total = 0;
      for (let index = 0; sizes && index < values.paletteCount; index++) {
        largest = Math.max(largest, sizes[index]);
        total += sizes[index];
      }
      return {
        readout: `${SCALES[values.scale]} · ${values.paletteCount} colors${
          total > 0 ? ` · biggest class ${Math.round((100 * largest) / total)}%` : ''
        }`
      };
    }
  }),
  draw: ({outputs, values, world}) => {
    // The palette strip is the legend: each swatch is numbered, and the bars below use the same
    // numbers, so a bar reads as the rows the scale gave that color.
    const [left, bottom, right, top] = SWATCH_RECTANGLE;
    const swatchWidth = ((right - left) * world.width) / MAXIMUM_CLASS_COUNT;
    const swatchLabels: RunLabel[] = Array.from({length: values.paletteCount}, (_, swatch) => ({
      text: String(swatch + 1),
      at: [left * world.width + (swatch + 0.45) * swatchWidth, bottom * world.height],
      anchor: 'top',
      offset: [0, -2],
      role: 'tick',
      group: 'swatches'
    }));
    swatchLabels.push({
      text: 'palette class',
      at: [left * world.width, top * world.height],
      anchor: 'bottom-left',
      offset: [0, 3],
      role: 'title'
    });
    const countAxes = createChartAxes({
      rectangle: getChartRectangle(world, COUNT_RECTANGLE),
      x: {
        categories: Array.from({length: MAXIMUM_CLASS_COUNT}, (_, index) =>
          index < values.paletteCount ? String(index + 1) : ''
        )
      },
      y: {range: [0, ROW_COUNT / 2], ticks: 2, title: 'rows'}
    });
    return [
      {layer: 'text', ...countAxes.grid},
      {
        layer: 'dots',
        context: true,
        positions: outputs.positions,
        values: outputs.column,
        range: [0, 6],
        radius: 3
      },
      {
        layer: 'dots',
        positions: outputs.positions,
        values: outputs.classIndices,
        ramp: 'categorical',
        categories: PALETTE,
        radius: 3
      },
      {
        layer: 'polygons',
        positions: outputs.swatches,
        indices: outputs.swatchIndices,
        owners: outputs.swatchOwners,
        values: outputs.swatchClasses,
        triangleCount: MAXIMUM_CLASS_COUNT * 2,
        ramp: 'categorical',
        categories: PALETTE
      },
      {
        layer: 'chart',
        kind: 'bars',
        color: [0.2, 0.55, 0.9, 1],
        values: outputs.classSizes,
        range: [0, ROW_COUNT / 2],
        rectangle: COUNT_RECTANGLE
      },
      {layer: 'text', ...countAxes},
      {layer: 'text', labels: swatchLabels}
    ];
  }
});
