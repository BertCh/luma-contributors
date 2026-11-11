// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterExtremaPyramid: a min/max pyramid of a raster whose first level has blocks of
// `firstBlockSize` cells; every further level doubles the block. Level 0 is drawn (the pyramid
// stores levels back to back, so level 0 is the first rows of each output): coarse blocks hold the
// extremes of the cells they cover, so peaks survive downsampling. `bilinear` footprints widen
// each block so the extremes also cover bilinear resampling.

import {
  getGPURasterExtremaPyramidLayout,
  GPURasterExtremaPyramid,
  makeGPURasterBand
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createElevationRaster} from '../../inputs';
import type {RunLabel} from '../../text';
import {parameters} from './extrema-pyramid.parameters';
import {formatNumber} from '../../interaction';
import {makeGridOfRaster} from './support/bands';
import {getSwipeResult, getCellAt, getCurtainX} from './support/gestures';

export default defineOperationRun({
  parameters,
  rebuildOn: ['firstBlockSize', 'footprint'],
  caption: 'A raster in; the maximum and minimum of every block out (level 0).',
  createInputs: ({world}) => createElevationRaster(world, 64),
  build(builder, inputs, values) {
    const grid = makeGridOfRaster(inputs);
    const firstBlockSize = 2 ** Math.round(values.firstBlockSize);
    const options = {
      firstBlockSize,
      footprint: values.footprint === 1 ? ('bilinear' as const) : ('cell' as const)
    };
    const layout = getGPURasterExtremaPyramidLayout(grid, options);
    const source = builder.upload('source', 'float32', inputs.values);
    const maximum = builder.allocate('maximum', 'float32', layout.length);
    const minimum = builder.allocate('minimum', 'float32', layout.length);
    builder.graph.add(
      new GPURasterExtremaPyramid({
        grid,
        input: makeGPURasterBand(source.view, grid, {id: 'source'}),
        ...options,
        output: {maximum: maximum.view, minimum: minimum.view}
      })
    );
    return {
      outputs: {
        source,
        maximum,
        minimum,
        columns: layout.levels[0].width,
        rows: layout.levels[0].height
      }
    };
  },
  // The pointer is a curtain between the block minima (left) and maxima (right), with the block.
  interaction: {
    kind: 'swipe',
    hint: 'swipe minimum and maximum',
    rest: [0.5, 0.5],
    resolve: (pointer, {inputs, values, world}) => {
      const size = 2 ** Math.round(values.firstBlockSize);
      const cell = getCellAt(inputs, world, pointer);
      const column = Math.floor(cell.column / size) * size;
      const row = Math.floor(cell.row / size) * size;
      const blockWidth = Math.min(size, inputs.columns - column);
      const blockHeight = Math.min(size, inputs.rows - row);
      const cellWidth = world.width / inputs.columns;
      const cellHeight = world.height / inputs.rows;
      let minimum = Infinity;
      let maximum = -Infinity;
      for (let rowIndex = row; rowIndex < row + blockHeight; rowIndex++) {
        for (let columnIndex = column; columnIndex < column + blockWidth; columnIndex++) {
          const value = inputs.values[rowIndex * inputs.columns + columnIndex];
          minimum = Math.min(minimum, value);
          maximum = Math.max(maximum, value);
        }
      }
      return getSwipeResult(
        pointer,
        `block ${size}x${size} | min ${formatNumber(minimum)} | max ${formatNumber(maximum)}`,
        [
          {
            shape: 'box',
            minimum: [column * cellWidth, world.height - (row + blockHeight) * cellHeight],
            maximum: [(column + blockWidth) * cellWidth, world.height - row * cellHeight]
          }
        ]
      );
    }
  },
  draw: ({outputs, interaction, values, world}) => {
    const curtain = getCurtainX(interaction, world);
    const look = {
      layer: 'raster',
      columns: outputs.columns,
      rows: outputs.rows,
      range: [0, 1]
    } as const;
    const size = 2 ** Math.round(values.firstBlockSize);
    const labels: RunLabel[] = [
      {
        text: `minimum of each ${size} x ${size} block`,
        at: [curtain - 0.01, 0.97],
        anchor: 'top-right',
        role: 'title'
      },
      {
        text: `maximum of each ${size} x ${size} block`,
        at: [curtain + 0.01, 0.97],
        anchor: 'top-left',
        role: 'title'
      }
    ];
    return [
      {...look, values: outputs.minimum, clip: {side: 'left', x: curtain}},
      {...look, values: outputs.maximum, clip: {side: 'right', x: curtain}},
      {layer: 'text', labels}
    ];
  }
});
