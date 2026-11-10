// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addPeriodComparisonRecipe: two periods of points aggregated into Quadbin cells and compared.
// Red cells gained events from the first period to the second, blue cells lost them.

import {
  getGPUClassBreaksParameterValues,
  getGPUColorScaleParameterValues,
  packGPUColor
} from '@luma-contributors/analysis/dataframe-analysis';
import {addPeriodComparisonRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {earthquakeEvents, getEventDots, getEventPosition} from './lib/earthquakes';
import {defineOperationRun} from '../../define-run';
import {clamp} from '../../interaction';
import {addCellFill} from './lib/cells';
import {parameters} from './period-comparison-recipe.parameters';

const TABLE_CAPACITY = 1024;
const UNION_CAPACITY = 2048;
const PALETTE = [
  packGPUColor(33, 102, 172),
  packGPUColor(146, 197, 222),
  packGPUColor(247, 247, 247),
  packGPUColor(244, 165, 130),
  packGPUColor(178, 24, 43)
];
const QUARTERS = ['Jan-Mar', 'Apr-Jun', 'Jul-Sep', 'Oct-Dec'];
const EDGES = [-1e6, -8, -2, 2, 8, 1e6];

export default defineOperationRun({
  parameters,
  buildCache: 3,
  caption: 'Earthquake counts in one quarter of 2023 compared with the next.',
  createInputs: ({world}) => ({
    dots: getEventDots(world.width, world.height),
    positions: earthquakeEvents.map(event => getEventPosition(event, world.width, world.height)),
    months: earthquakeEvents.map(event => new Date(event[2] * 1000).getUTCMonth())
  }),
  build(builder, {positions, months, dots}, values) {
    const select = (start: number) =>
      Float32Array.from(
        positions.filter((_, index) => months[index] >= start && months[index] < start + 3).flat()
      );
    const before = builder.upload('before-positions', 'float32x2', select(values.quarter * 3));
    const after = builder.upload('after-positions', 'float32x2', select(values.quarter * 3 + 3));
    const cells = builder.allocate('union-cells', 'uint32x2', UNION_CAPACITY);
    const delta = builder.allocate('delta', 'float32', UNION_CAPACITY);
    const zScore = builder.allocate('z-score', 'float32', UNION_CAPACITY);
    const percentChange = builder.allocate('percent-change', 'float32', UNION_CAPACITY);
    const count = builder.allocate('union-count', 'uint32', 1);
    const overflow = builder.allocate('union-overflow', 'uint32', 1);
    const breaks = builder.allocate('breaks', 'float32', 6);
    const classCount = builder.allocate('class-count', 'uint32', 1);
    const colors = builder.allocate('colors', 'uint32', UNION_CAPACITY);
    const classIndices = builder.allocate('class-indices', 'uint32', UNION_CAPACITY);
    const variable = (['delta', 'zScore', 'percentChange'] as const)[values.classify];
    addPeriodComparisonRecipe(builder.graph, {
      family: 'quadbin',
      resolution: values.resolution,
      tableCapacity: TABLE_CAPACITY,
      unionCapacity: UNION_CAPACITY,
      before: {positions: before.view},
      after: {positions: after.view},
      zScore: 'poisson',
      outputs: {
        comparison: {
          cells: cells.view,
          delta: delta.view,
          zScore: zScore.view,
          percentChange: percentChange.view,
          status: {count: count.view, overflow: overflow.view}
        },
        breaks: breaks.view,
        classCount: classCount.view,
        colors: colors.view,
        classIndices: classIndices.view
      },
      classify: variable,
      classBreaksParameters: builder.upload(
        'class-breaks-parameters',
        'float32',
        getGPUClassBreaksParameterValues({method: 'custom', customEdges: EDGES}, 5)
      ).view,
      maximumClassCount: 5,
      methods: ['custom'],
      palette: builder.upload('palette', 'uint32', Uint32Array.from(PALETTE)).view,
      colorScaleParameters: builder.upload(
        'color-scale-parameters',
        'float32',
        getGPUColorScaleParameterValues({
          scale: 'threshold',
          domainCount: 6,
          paletteCount: 5,
          noDataColor: packGPUColor(0, 0, 0, 0)
        })
      ).view,
      maximumPaletteCount: 5
    });
    return {
      outputs: {
        count,
        delta,
        events: builder.upload('events', 'float32x2', dots),
        shown: variable === 'delta' ? delta : variable === 'zScore' ? zScore : percentChange,
        fill: addCellFill(builder, 'quadbin', cells)
      }
    };
  },
  rebuildOn: ['resolution', 'classify', 'quarter'],
  // x chooses the pair of consecutive quarters; the compile-time quarter settles when the pointer rests.
  interaction: {
    kind: 'time',
    hint: 'choose the quarters',
    rest: [1 / 6, 0.5],
    resolve(pointer, {outputs, peek, world}) {
      const quarter = clamp(Math.floor(pointer.unitX * 3), 0, 2);
      const count = peek(outputs.count);
      const delta = peek(outputs.delta);
      let lost = 0;
      let gained = 0;
      if (count && delta) {
        for (let cell = 0; cell < Math.min(count[0], delta.length); cell++) {
          if (delta[cell] < 0) lost++;
          if (delta[cell] > 0) gained++;
        }
      }
      const stripX = (tick: number) => world.width * (0.1 + 0.8 * (tick / 4));
      return {
        settle: {quarter},
        glyphs: [
          {shape: 'line', from: [stripX(0), 0.03], to: [stripX(4), 0.03]},
          {shape: 'line', from: [stripX(quarter), 0.03], to: [stripX(quarter + 2), 0.03]},
          {shape: 'crosshair', at: [stripX(quarter + 1), 0.03], size: 5}
        ],
        readout: `${QUARTERS[quarter]} -> ${QUARTERS[quarter + 1]} · ${lost} lost, ${gained} gained`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.events, radius: 2.2},
    {
      layer: 'polygons',
      positions: outputs.fill.boundaries,
      indices: outputs.fill.indices,
      owners: outputs.fill.owners,
      values: outputs.shown,
      triangleCount: outputs.fill.triangleCount,
      ramp: 'diverging'
    }
  ]
});
