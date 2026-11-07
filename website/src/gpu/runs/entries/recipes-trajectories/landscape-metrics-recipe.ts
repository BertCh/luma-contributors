// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addLandscapeMetricsRecipe on a categorical landscape (muted map). The left bars are the class
// adjacency counts (how often each pair of classes touches), the right bars the aggregation
// index of each class.

import {addLandscapeMetricsRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createLandscape, createSeededRandom} from './lib/landscape';
import {addReadout} from './lib/readout';
import {parameters} from './landscape-metrics-recipe.parameters';

const COLUMNS = 64;
const ROWS = 36;

export default defineOperationRun({
  parameters,
  caption: 'Class adjacency and aggregation of a landscape.',
  createInputs: ({random}) => ({seed: Math.floor(random() * 1e9)}),
  build(builder, {seed}, values) {
    const landscape = createLandscape(
      createSeededRandom(seed),
      COLUMNS,
      ROWS,
      values.fragmentation,
      values.classCount
    );
    const recipe = addLandscapeMetricsRecipe(builder.graph, {
      patches: builder.upload('patches', 'uint32', landscape.patches).view,
      patchClasses: builder.upload('patch-classes', 'uint32', landscape.patchClasses).view,
      width: COLUMNS,
      height: ROWS,
      classCount: values.classCount,
      cellSize: [30, 30],
      connectivity: 4
    });
    return {
      outputs: {
        classes: builder.upload('classes', 'uint32', landscape.classes),
        adjacency: addReadout(builder, 'adjacency', recipe.outputs.classAdjacency),
        aggregation: addReadout(builder, 'aggregation', recipe.outputs.aggregation)
      },
      update: () => undefined
    };
  },
  rebuildOn: ['classCount', 'fragmentation'],
  draw: ({outputs}) => [
    {layer: 'raster', values: outputs.classes, columns: COLUMNS, rows: ROWS, alpha: 0.85},
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.adjacency,
      ramp: 'sequential',
      range: [0, 60],
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
