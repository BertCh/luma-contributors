// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUS2CellMeasures writes the spherical area (steradians) and perimeter (radians) of S2 cells.
// A grid of points is keyed at one S2 level; the cells are filled by the measure the operation
// wrote (the colors stretch over the small variation across the window).

import {GPUS2CellMeasures} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addCellFill, addPointKeys} from './support/cells';
import {parameters} from './s2-cell-measures.parameters';

const GRID_COLUMNS = 44;
const GRID_ROWS = 24;

export default defineOperationRun({
  parameters,
  caption: 'Spherical area or perimeter of each S2 cell, written on the GPU.',
  createInputs: ({world}) => {
    const positions = new Float32Array(GRID_COLUMNS * GRID_ROWS * 2);
    for (let row = 0; row < GRID_ROWS; row++) {
      for (let column = 0; column < GRID_COLUMNS; column++) {
        const index = (row * GRID_COLUMNS + column) * 2;
        positions[index] = ((column + 0.5) / GRID_COLUMNS) * world.width;
        positions[index + 1] = (row + 0.5) / GRID_ROWS;
      }
    }
    return {count: GRID_COLUMNS * GRID_ROWS, positions};
  },
  build(builder, grid, values) {
    const positions = builder.upload('positions', 'float32x2', grid.positions);
    const cells = addPointKeys(builder, 's2', values.level, positions);
    const measures = builder.allocate('measures', 'float32x2', grid.count);
    const status = builder.allocate('measure-status', 'uint32', grid.count);
    builder.graph.add(
      new GPUS2CellMeasures({
        cells: cells.view,
        output: {measures: measures.view, status: status.view}
      })
    );
    // `measures` rows are (area, perimeter): polygon `2 * row + measure` reads that word.
    return {
      outputs: {measures, fill: addCellFill(builder, 's2', cells, 'fill', 2, values.measure)}
    };
  },
  draw: ({outputs, values}) => {
    // The buffer interleaves both columns, so the range is set from the S2 average cell area.
    const averageArea = (4 * Math.PI) / 6 / 4 ** values.level;
    const range: [number, number] =
      values.measure === 0
        ? [0.84 * averageArea, 0.92 * averageArea]
        : [0.92 * 4 * Math.sqrt(averageArea), 0.96 * 4 * Math.sqrt(averageArea)];
    return [
      {
        layer: 'polygons',
        positions: outputs.fill.boundaries,
        indices: outputs.fill.indices,
        owners: outputs.fill.owners,
        values: outputs.measures,
        triangleCount: outputs.fill.triangleCount,
        range
      }
    ];
  }
});
