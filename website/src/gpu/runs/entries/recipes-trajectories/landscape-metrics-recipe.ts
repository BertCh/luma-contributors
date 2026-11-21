// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addLandscapeMetricsRecipe on a categorical landscape (muted map). The left bars are the class
// adjacency counts (how often each pair of classes touches), the right bars the aggregation
// index of each class.

import {addLandscapeMetricsRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {getRasterCell} from '../../pick';
import {createLandscape} from './lib/landscape';
import {addReadout} from './lib/readout';
import {parameters} from './landscape-metrics-recipe.parameters';

const COLUMNS = 64;
const ROWS = 36;

export default defineOperationRun({
  parameters,
  caption: 'Class adjacency and aggregation of the Hyde Park WorldCover landscape.',
  createInputs: () => ({focus: {buffer: null as unknown, className: -1}}),
  build(builder, _inputs, values) {
    const landscape = createLandscape(COLUMNS, ROWS, values.grain, values.classCount);
    const recipe = addLandscapeMetricsRecipe(builder.graph, {
      patches: builder.upload('patches', 'uint32', landscape.patches).view,
      patchClasses: builder.upload('patch-classes', 'uint32', landscape.patchClasses).view,
      width: COLUMNS,
      height: ROWS,
      classCount: values.classCount,
      cellSize: [landscape.cellSizeMeters, landscape.cellSizeMeters],
      connectivity: 4
    });
    return {
      outputs: {
        classValues: landscape.classes,
        highlight: builder.allocate('highlight', 'uint32', landscape.classes.length),
        classes: builder.upload('classes', 'uint32', landscape.classes),
        adjacency: addReadout(builder, 'adjacency', recipe.outputs.classAdjacency),
        aggregation: addReadout(builder, 'aggregation', recipe.outputs.aggregation)
      },
      update: () => undefined
    };
  },
  rebuildOn: ['classCount', 'grain'],
  // The pointer picks a cell: every patch of its class lights up, with the class's aggregation.
  interaction: {
    kind: 'probe',
    hint: 'hover a class',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, world, peek}) {
      const cell = getRasterCell(
        COLUMNS,
        ROWS,
        [0, 0, world.width, world.height],
        pointer.x,
        pointer.y
      );
      const className = outputs.classValues[cell.index];
      if (inputs.focus.buffer !== outputs.highlight || inputs.focus.className !== className) {
        inputs.focus = {buffer: outputs.highlight, className};
        outputs.highlight.write(
          Uint32Array.from(outputs.classValues, value => (value === className ? value : 0xffffffff))
        );
      }
      const aggregation = peek(outputs.aggregation);
      return {
        glyphs: [{shape: 'crosshair', at: [pointer.x, pointer.y]}],
        readout: `class ${className} · aggregation ${aggregation ? aggregation[className].toFixed(2) : '...'}`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'raster', values: outputs.classes, columns: COLUMNS, rows: ROWS, alpha: 0.4},
    {layer: 'raster', values: outputs.highlight, columns: COLUMNS, rows: ROWS},
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.adjacency,
      ramp: 'sequential',
      // Class adjacency counts reach the hundreds on this 64 x 36 grid; the range follows the data
      // (a fixed [0, 60] clipped most bars).
      rectangle: [0.04, 0.05, 0.48, 0.34]
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.aggregation,
      ramp: 'sequential',
      rectangle: [0.56, 0.05, 0.96, 0.34]
    }
  ]
});
