// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointDensity bins clustered points into square cells (count, sum or mean of the point
// weights); the dots are the input events, the raster is the operation's `values` output.

import {
  createGPUPointDensityGaussianKernel,
  GPUPointDensity
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints} from '../../inputs';
import {createPointIndex, getRasterCell} from '../../pick';
import {parameters} from './point-density.parameters';

const STATISTICS = ['count', 'sum', 'mean'] as const;

export default defineOperationRun({
  parameters,
  caption: 'Events binned into cells; the dots are the input points.',
  createInputs: ({random, world}) => ({
    ...(points => ({...points, index: createPointIndex(points.positions)}))(
      createClusteredPoints(random, world, 900, 4, 0.12)
    ),
    width: world.width
  }),
  build(builder, points, values) {
    const rows = values.gridSize;
    const columns = Math.round(rows * points.width);
    // Row 0 of the operation's grid is the lowest y; the raster layer draws row 0 at the top.
    const flipped = Float32Array.from(points.positions, (value, index) =>
      index % 2 === 1 ? 1 - value : value
    );
    const positions = builder.upload('positions', 'float32x2', flipped);
    const weights = builder.upload('weights', 'float32', points.values);
    const cells = builder.allocate('values', 'float32', columns * rows);
    const radius = values.smoothing;
    builder.graph.add(
      new GPUPointDensity({
        positions: positions.view,
        weights: weights.view,
        bounds: [0, 0, points.width, 1],
        gridSize: [columns, rows],
        statistic: STATISTICS[values.statistic],
        smoothing:
          radius > 0
            ? {
                kernel: builder.upload(
                  'kernel',
                  'float32',
                  createGPUPointDensityGaussianKernel(radius)
                ).view,
                kernelWidth: 2 * radius + 1,
                kernelHeight: 2 * radius + 1
              }
            : undefined,
        output: {values: cells.view}
      })
    );
    return {
      outputs: {
        cells,
        columns,
        rows,
        dots: builder.upload('dots', 'float32x2', points.positions),
        // The events of the hovered cell; unused rows stay NaN and are not drawn.
        binned: builder.upload('binned', 'float32x2', new Float32Array(points.count * 2).fill(NaN))
      }
    };
  },
  // The hovered cell outlines itself and lights the events that were binned into it.
  interaction: {
    kind: 'probe',
    hint: 'hover a cell',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const bounds = [0, 0, inputs.width, 1] as const;
      // Snap to the cell of the nearest event, so the probe always lands on something binned.
      const nearest = Math.max(0, inputs.index.nearest(pointer.x, pointer.y));
      const cell = getRasterCell(
        outputs.columns,
        outputs.rows,
        bounds,
        inputs.positions[nearest * 2],
        inputs.positions[nearest * 2 + 1]
      );
      const cellWidth = inputs.width / outputs.columns;
      const cellHeight = 1 / outputs.rows;
      const minimum: [number, number] = [cell.column * cellWidth, 1 - (cell.row + 1) * cellHeight];
      const lit = new Float32Array(inputs.count * 2).fill(NaN);
      let count = 0;
      for (let row = 0; row < inputs.count; row++) {
        const x = inputs.positions[row * 2];
        const y = inputs.positions[row * 2 + 1];
        if (
          x < minimum[0] ||
          x >= minimum[0] + cellWidth ||
          y < minimum[1] ||
          y >= minimum[1] + cellHeight
        )
          continue;
        lit.set([x, y], count * 2);
        count++;
      }
      outputs.binned.write(lit);
      const value = peek(outputs.cells)?.[cell.index];
      return {
        glyphs: [
          {shape: 'box', minimum, maximum: [minimum[0] + cellWidth, minimum[1] + cellHeight]}
        ],
        readout: `cell ${cell.column}, ${cell.row} · ${count} events · ${STATISTICS[values.statistic]} ${value === undefined ? '...' : value.toFixed(2)}`
      };
    }
  },
  draw: ({outputs, values}) => [
    {
      layer: 'raster',
      values: outputs.cells,
      columns: outputs.columns,
      rows: outputs.rows,
      range: [
        [0, 10],
        [0, 5],
        [0, 1]
      ][values.statistic] as [number, number]
    },
    {layer: 'dots', context: true, positions: outputs.dots, radius: 2.4},
    {layer: 'dots', positions: outputs.binned, radius: 4, color: [1, 0.9, 0.35, 1]}
  ]
});
