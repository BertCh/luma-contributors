// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUH3CellIdentities lists the six directed-edge or vertex identifiers of every H3 cell. Each
// edge starts at a cell corner and is colored by the direction (or vertex) number that the
// operation encoded in the identifier it wrote for that slot.

import {GPUH3CellIdentities} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addCellGeometry, addPointKeys} from './support/cells';
import {addHighWordField} from './support/kernels';
import {parameters} from './h3-cell-identities.parameters';

const GRID_COLUMNS = 40;
const GRID_ROWS = 22;

export default defineOperationRun({
  parameters,
  caption: 'Six identifiers per cell: each corner shows the number encoded in its identifier.',
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
    const cells = addPointKeys(builder, 'h3', values.resolution, positions);
    const identifiers = builder.allocate('identifiers', 'uint32x2', grid.count * 6);
    builder.graph.add(
      new GPUH3CellIdentities({
        cells: cells.view,
        operation: values.operation === 0 ? 'directed-edges' : 'vertices',
        output: identifiers.view
      })
    );
    // Boundary slots 0 to 5 of cell `c` line up with identifier rows `6 * c + slot`.
    const geometry = addCellGeometry(builder, 'h3', cells, 'corners', 6);
    const edges = new Uint32Array(grid.count * 6 * 2);
    for (let slot = 0; slot < grid.count * 6; slot++) {
      const cell = Math.floor(slot / 6);
      edges[slot * 2] = slot;
      edges[slot * 2 + 1] = cell * 6 + ((slot + 1) % 6);
    }
    return {
      outputs: {
        corners: geometry.boundaries,
        numbers: addHighWordField(builder, 'identifiers', identifiers, 24),
        edges: builder.upload('edges', 'uint32x2', edges)
      }
    };
  },
  rebuildOn: ['operation', 'resolution'],
  draw: ({outputs}) => [
    {
      layer: 'segments',
      nodes: outputs.corners,
      edges: outputs.edges,
      values: outputs.numbers,
      ramp: 'sequential',
      range: [0, 6],
      radius: 3.5
    }
  ]
});
