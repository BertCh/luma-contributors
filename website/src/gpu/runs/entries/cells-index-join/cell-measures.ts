// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCellMeasures computes the geodesic area and perimeter of H3 cells. A regular grid of points
// is keyed with GPUPointToCell; the cells are filled by the measure the operation wrote.

import {GPUCellMeasures, GPUPointToCell} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addCellFill} from './support/cells';
import {parameters} from './cell-measures.parameters';

const GRID_COLUMNS = 48;
const GRID_ROWS = 27;

export default defineOperationRun({
  parameters,
  caption: 'Geodesic cell area or perimeter, written per H3 cell on the GPU.',
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
  build(builder, points, values) {
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const cells = builder.allocate('cells', 'uint32x2', points.count);
    builder.graph.add(
      new GPUPointToCell({
        family: 'h3',
        resolution: values.resolution,
        positions: positions.view,
        output: {cells: cells.view}
      })
    );
    const areas = builder.allocate('areas', 'float32', points.count);
    const perimeters = builder.allocate('perimeters', 'float32', points.count);
    const edgeCounts = builder.allocate('edge-counts', 'uint32', points.count);
    builder.graph.add(
      new GPUCellMeasures({
        family: 'h3',
        cells: cells.view,
        units: 'kilometers',
        output: {areas: areas.view, perimeters: perimeters.view, edgeCounts: edgeCounts.view}
      })
    );
    return {
      outputs: {areas, perimeters, edgeCounts, fill: addCellFill(builder, 'h3', cells)},
      update: () => undefined
    };
  },
  rebuildOn: ['resolution'],
  draw: ({outputs, values}) => [
    {
      layer: 'polygons',
      positions: outputs.fill.boundaries,
      indices: outputs.fill.indices,
      owners: outputs.fill.owners,
      values: [outputs.areas, outputs.perimeters, outputs.edgeCounts][values.measure],
      triangleCount: outputs.fill.triangleCount,
      ramp: 'sequential'
    }
  ]
});
