// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULatticeWeights on a small grid. Cells are colored by how many neighbors the operation wrote
// for them (fewer at the border and next to masked holes), and every link is drawn.

import {GPULatticeWeights} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './lattice-weights.parameters';
import {addCardinality, addLinkSegments, allocateWeights, drawLinks} from './shared/weights';

const ROWS = 9;

export default defineOperationRun({
  parameters,
  caption: 'Neighbors of every grid cell, written by the lattice operation.',
  createInputs: ({random, world}) => {
    const columns = Math.round(ROWS * world.width);
    const centers = new Float32Array(columns * ROWS * 2);
    for (let row = 0; row < ROWS; row++) {
      for (let column = 0; column < columns; column++) {
        const cell = row * columns + column;
        // Row 0 is the top row, like the raster that draws the cells.
        centers[cell * 2] = ((column + 0.5) * world.width) / columns;
        centers[cell * 2 + 1] = 1 - (row + 0.5) / ROWS;
      }
    }
    const chance = Float32Array.from({length: columns * ROWS}, () => random());
    return {columns, centers, chance};
  },
  build(builder, {columns, centers, chance}, values) {
    const cellCount = columns * ROWS;
    const mask = builder.allocate('mask', 'uint32', cellCount);
    const weights = allocateWeights(builder, 'lattice', cellCount, cellCount * 24);
    builder.graph.add(
      new GPULatticeWeights({
        width: columns,
        height: ROWS,
        criterion: values.criterion === 0 ? 'rook' : 'queen',
        radius: values.radius,
        mask: mask.view,
        output: weights.port
      })
    );
    const cellCenters = builder.upload('centers', 'float32x2', centers);
    const cardinality = addCardinality(builder, 'lattice', weights);
    const links = addLinkSegments(builder, 'lattice', weights, cellCenters);
    return {
      outputs: {cardinality, links, cellCenters},
      update: current =>
        mask.write(Uint32Array.from(chance, value => (value < current.holes ? 0 : 1)))
    };
  },
  rebuildOn: ['criterion', 'radius'],
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      values: outputs.cardinality,
      columns: inputs.columns,
      rows: ROWS,
      alpha: 0.9
    },
    drawLinks(outputs.links, [0.95, 0.55, 0.2, 0.5]),
    {layer: 'dots', context: true, positions: outputs.cellCenters, radius: 2}
  ]
});
