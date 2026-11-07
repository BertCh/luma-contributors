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
import {defineOperationRun} from '../../define-run';
import {addCellFill} from './lib/cells';
import {parameters} from './period-comparison-recipe.parameters';

const TABLE_CAPACITY = 1024;
const UNION_CAPACITY = 2048;
const POINTS_PER_PERIOD = 700;
const PALETTE = [
  packGPUColor(33, 102, 172),
  packGPUColor(146, 197, 222),
  packGPUColor(247, 247, 247),
  packGPUColor(244, 165, 130),
  packGPUColor(178, 24, 43)
];
const EDGES = [-1e6, -8, -2, 2, 8, 1e6];

export default defineOperationRun({
  parameters,
  caption: 'Change per cell between two periods: gains red, losses blue.',
  createInputs: ({random, world}) => ({
    world,
    background: Float32Array.from(
      {length: POINTS_PER_PERIOD * 2},
      (_, index) => random() * (index % 2 === 0 ? world.width : world.height)
    ),
    jitterX: Float32Array.from({length: POINTS_PER_PERIOD}, () => random() + random() - 1),
    jitterY: Float32Array.from({length: POINTS_PER_PERIOD}, () => random() + random() - 1)
  }),
  build(builder, {world, background, jitterX, jitterY}, values) {
    const before = builder.allocate('before-positions', 'float32x2', POINTS_PER_PERIOD * 2);
    const after = builder.allocate('after-positions', 'float32x2', POINTS_PER_PERIOD * 2);
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
        shown: variable === 'delta' ? delta : variable === 'zScore' ? zScore : percentChange,
        fill: addCellFill(builder, 'quadbin', cells)
      },
      update: current => {
        // Half the points are background in both periods; the rest form a cluster that moves.
        const first = new Float32Array(POINTS_PER_PERIOD * 4);
        const second = new Float32Array(POINTS_PER_PERIOD * 4);
        for (let index = 0; index < POINTS_PER_PERIOD; index++) {
          first.set([background[index * 2], background[index * 2 + 1]], index * 2);
          second.set([background[index * 2], background[index * 2 + 1]], index * 2);
          const wobbleX = jitterX[index] * 0.22;
          const wobbleY = jitterY[index] * 0.22;
          first.set([0.25 * world.width + wobbleX, 0.5 + wobbleY], (POINTS_PER_PERIOD + index) * 2);
          second.set(
            [(0.25 + 0.5 * current.shift) * world.width + wobbleX, 0.5 + wobbleY],
            (POINTS_PER_PERIOD + index) * 2
          );
        }
        before.write(first);
        after.write(second);
      }
    };
  },
  rebuildOn: ['resolution', 'classify'],
  draw: ({outputs}) => [
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
