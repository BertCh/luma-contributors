// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCellPyramid aggregates points once at the finest resolution and rolls the result up through
// every coarser level. All four level tables exist after one run; the slider picks the one drawn.

import {GPUCellPyramid} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints} from '../../inputs';
import {
  addCellFill,
  addPointKeys,
  allocateCellTable,
  getCellFamily,
  getCellTableViews
} from './support/cells';
import {parameters} from './cell-pyramid.parameters';

const LEVEL_COUNT = 4;
const LEVEL_CAPACITY = 2048;

export default defineOperationRun({
  parameters,
  caption: 'One aggregation pass, four zoom levels: finest first.',
  createInputs: ({random, world}) => createClusteredPoints(random, world, 900, 4, 0.12),
  build(builder, points, values) {
    const family = getCellFamily(values.family);
    const resolutions = family === 'h3' ? [6, 5, 4, 3] : [11, 10, 9, 8];
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const pointValues = builder.upload('values', 'float32', points.values);
    const keys = addPointKeys(builder, family, resolutions[0], positions);
    const levels = resolutions.map((_, level) =>
      allocateCellTable(builder, `level-${level}`, LEVEL_CAPACITY)
    );
    const levelCounts = builder.allocate('level-counts', 'uint32', LEVEL_COUNT);
    builder.graph.add(
      new GPUCellPyramid({
        family,
        cells: keys.view,
        values: pointValues.view,
        sumScale: 65536,
        resolutions,
        output: {levels: levels.map(getCellTableViews), levelCounts: levelCounts.view}
      })
    );
    return {
      outputs: {
        levels,
        levelFills: levels.map((level, index) =>
          addCellFill(builder, family, level.cells, `fill-${index}`)
        )
      },
      update: () => undefined
    };
  },
  rebuildOn: ['family'],
  draw: ({outputs, values}) => {
    const fill = outputs.levelFills[values.level];
    return [
      {
        layer: 'polygons',
        positions: fill.boundaries,
        indices: fill.indices,
        owners: fill.owners,
        values: outputs.levels[values.level].sumValues,
        triangleCount: fill.triangleCount,
        range: [0, 8 * Math.pow(3, values.level)]
      }
    ];
  }
});
