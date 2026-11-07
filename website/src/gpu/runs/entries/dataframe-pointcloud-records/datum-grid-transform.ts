// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUDatumGridTransform applying a bilinear shift grid to a lattice of points. Gray dots are the
// input lattice (as longitude and latitude); orange dots are where the GPU moved them. The inverse
// direction undoes the shift by iterating, so `maximumIterations` matters where the shift varies fast.

import {GPUDatumGridTransform} from '@luma-contributors/analysis/geospatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addFloatCopy} from './_lib/convert';
import {parameters} from './datum-grid-transform.parameters';

const COLUMNS = 40;
const ROWS = 22;
const GRID_COLUMNS = 14;
const GRID_ROWS = 9;
const GRID_SPACING = 0.16;

export default defineOperationRun({
  parameters,
  caption: 'Lattice before (gray) and after (orange) the datum shift grid.',
  createInputs: ({world}) => {
    const lattice = new Float32Array(COLUMNS * ROWS * 3);
    for (let row = 0; row < ROWS; row++) {
      for (let column = 0; column < COLUMNS; column++) {
        lattice.set(
          [((column + 0.5) / COLUMNS) * world.width, (row + 0.5) / ROWS, 0],
          (row * COLUMNS + column) * 3
        );
      }
    }
    return {lattice};
  },
  build(builder, inputs, values) {
    const pointCount = COLUMNS * ROWS;
    const positions = builder.upload('positions', 'float32x3', inputs.lattice);
    const samples = builder.allocate('samples', 'float32x3', GRID_COLUMNS * GRID_ROWS);
    const shifted = builder.allocate('shifted', 'float32x3', pointCount);
    builder.graph.add(
      new GPUDatumGridTransform({
        positions: positions.view,
        samples: samples.view,
        validity: builder.upload(
          'validity',
          'uint32',
          new Uint32Array(GRID_COLUMNS * GRID_ROWS).fill(1)
        ).view,
        grids: [
          {
            width: GRID_COLUMNS,
            height: GRID_ROWS,
            sampleOffset: 0,
            origin: [-0.1, -0.1],
            spacing: [GRID_SPACING, GRID_SPACING]
          }
        ],
        operation: 'horizontal',
        direction: values.direction === 0 ? 'forward' : 'inverse',
        maximumIterations: values.maximumIterations,
        output: {
          positions: shifted.view,
          status: builder.allocate('status', 'uint32', pointCount).view
        }
      })
    );
    return {
      outputs: {
        before: addFloatCopy(builder, 'before', positions, pointCount, {outputComponents: 2}),
        after: addFloatCopy(builder, 'after', shifted, pointCount, {outputComponents: 2})
      },
      update: current =>
        samples.write(
          Float32Array.from({length: GRID_COLUMNS * GRID_ROWS * 3}, (_, index) => {
            const node = Math.floor(index / 3);
            const column = node % GRID_COLUMNS;
            const row = Math.floor(node / GRID_COLUMNS);
            if (index % 3 === 0) return current.shift * Math.sin(row * 0.35 + column * 0.22);
            if (index % 3 === 1) return current.shift * Math.cos(column * 0.3 - row * 0.25);
            return 0;
          })
        )
    };
  },
  rebuildOn: ['direction', 'maximumIterations'],
  draw: ({outputs}) => [
    {
      layer: 'dots',
      context: true,
      positions: outputs.before,
      color: [0.6, 0.65, 0.75, 0.8],
      radius: 3
    },
    {layer: 'dots', positions: outputs.after, color: [1, 0.6, 0.2, 0.95], radius: 4.6}
  ]
});
