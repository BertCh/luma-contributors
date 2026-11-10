// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUColumnProfile of one numeric column. The bright bars are the GPU histogram of the masked rows
// over the per-frame domain; the faint bars are the same bins over all rows.

import {
  GPUColumnProfile,
  getGPUColumnProfileParameterValues,
  GPU_COLUMN_PROFILE_STATISTIC_COUNT
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {sampleFlashes} from '../../../data/lightning';
import {addIntegerToFloat} from './_lib/convert';
import {createBoxMask, createHistogram, getBoxMaskBounds, type BoxBounds} from './_lib/helpers';
import {parameters} from './column-profile.parameters';

const ROW_COUNT = 2000;
const MAP_BOTTOM = 0.45;
const HISTOGRAM_RECTANGLE = [0.07, 0.06, 0.95, 0.38] as const;

type Inputs = {
  world: {width: number; height: number};
  mapHeight: number;
  points: ReturnType<typeof sampleFlashes>;
  column: Float32Array;
  /** The window's center in the drawing world: the pointer, when there is one. */
  center: [number, number];
  /** Rows inside the window, for the bright dots. */
  kept: {count: number; sum: number; bounds: BoxBounds};
};
type Outputs = {
  column: RunBuffer;
  positions: RunBuffer;
  keptPositions: RunBuffer;
  keptValues: RunBuffer;
  bars: RunBuffer;
  allRows: RunBuffer;
};

function getWindowBounds(inputs: Inputs, share: number): BoxBounds {
  const local = getBoxMaskBounds(inputs.world.width, inputs.mapHeight, share, [
    inputs.center[0],
    inputs.center[1] - MAP_BOTTOM
  ]);
  return {
    minimumX: local.minimumX,
    maximumX: local.maximumX,
    minimumY: local.minimumY + MAP_BOTTOM,
    maximumY: local.maximumY + MAP_BOTTOM
  };
}

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'Flash energy histogram in units of 10 fJ for the flashes inside the box (bright bars), over all flashes (faint bars).',
  createInputs: ({world}) => {
    // The map fills the top of the frame; the histogram strip sits below it.
    const mapHeight = 1 - MAP_BOTTOM;
    const points = sampleFlashes({width: world.width, height: mapHeight}, ROW_COUNT);
    for (let row = 0; row < ROW_COUNT; row++) points.positions[row * 2 + 1] += MAP_BOTTOM;
    return {
      world,
      mapHeight,
      points,
      column: Float32Array.from(points.energy, value => value / 10),
      center: [world.width / 2, MAP_BOTTOM + mapHeight / 2],
      kept: {count: 0, sum: 0, bounds: {minimumX: 0, minimumY: 0, maximumX: 0, maximumY: 0}}
    };
  },
  build(builder, inputs, values) {
    const binCount = values.histogramBinCount;
    const column = builder.upload('flash-energy', 'float32', inputs.column);
    const positions = builder.upload('positions', 'float32x2', inputs.points.positions);
    const keptPositions = builder.allocate('kept-positions', 'float32x2', ROW_COUNT);
    const keptValues = builder.allocate('kept-values', 'float32', ROW_COUNT);
    const mask = builder.allocate('mask', 'uint32', ROW_COUNT);
    const domain = builder.allocate('domain', 'float32', 2);
    const histograms = builder.allocate('histograms', 'uint32', binCount);
    const allRows = builder.allocate('all-rows', 'float32', binCount);
    builder.graph.add(
      new GPUColumnProfile({
        columns: [{values: column.view}],
        mask: mask.view,
        parameters: domain.view,
        histogramBinCount: binCount,
        output: {
          statistics: builder.allocate('statistics', 'float32', GPU_COLUMN_PROFILE_STATISTIC_COUNT)
            .view,
          counts: builder.allocate('counts', 'uint32', 2).view,
          histograms: histograms.view
        }
      })
    );
    const bars = addIntegerToFloat(builder, 'bars', histograms);
    return {
      outputs: {column, positions, keptPositions, keptValues, bars, allRows},
      update: current => {
        const bounds = getWindowBounds(inputs, current.mask);
        const maskRows = createBoxMask(inputs.points.positions, bounds);
        mask.write(maskRows);
        let count = 0;
        let sum = 0;
        const rows = new Float32Array(ROW_COUNT * 2);
        const rowValues = new Float32Array(ROW_COUNT);
        for (let row = 0; row < ROW_COUNT; row++) {
          if (!maskRows[row]) continue;
          rows.set(
            [inputs.points.positions[row * 2], inputs.points.positions[row * 2 + 1]],
            count * 2
          );
          rowValues[count] = inputs.column[row];
          sum += inputs.column[row];
          count++;
        }
        keptPositions.write(rows);
        keptValues.write(rowValues);
        inputs.kept = {count, sum, bounds};
        domain.write(getGPUColumnProfileParameterValues([[0, current.domainMaximum]]));
        allRows.write(createHistogram(inputs.column, binCount, 0, current.domainMaximum));
      }
    };
  },
  rebuildOn: ['histogramBinCount'],
  // The pointer is the center of the box on the map: the rows inside it are the mask.
  interaction: {
    kind: 'window',
    hint: 'move the box',
    rest: ({inputs}) => [0.5, 1 - (MAP_BOTTOM + inputs.mapHeight / 2)],
    resolve(pointer, {inputs, values}) {
      inputs.center = [pointer.x, Math.max(pointer.y, MAP_BOTTOM)];
      const bounds = getWindowBounds(inputs, values.mask);
      const {count, sum} = inputs.kept;
      return {
        rerun: true,
        glyphs: [
          {
            shape: 'box',
            minimum: [bounds.minimumX, bounds.minimumY],
            maximum: [bounds.maximumX, bounds.maximumY]
          }
        ],
        readout: `mask ${count} of ${ROW_COUNT} rows · mean ${count ? Math.round((sum / count) * 10) : 0} fJ`
      };
    }
  },
  draw: ({inputs, outputs, values}) => {
    const tallest = Math.max(
      ...createHistogram(inputs.column, values.histogramBinCount, 0, values.domainMaximum)
    );
    const range = [0, tallest * 1.05] as const;
    return [
      {
        layer: 'dots',
        context: true,
        positions: outputs.positions,
        values: outputs.column,
        range: [0, 50],
        radius: 3
      },
      {
        layer: 'dots',
        positions: outputs.keptPositions,
        values: outputs.keptValues,
        count: Math.max(1, inputs.kept.count),
        range: [0, 50],
        radius: 3.6
      },
      {
        layer: 'chart',
        kind: 'bars',
        context: true,
        values: outputs.allRows,
        range,
        rectangle: HISTOGRAM_RECTANGLE
      },
      {
        layer: 'chart',
        kind: 'bars',
        color: [0.2, 0.55, 0.9, 1],
        values: outputs.bars,
        range,
        rectangle: HISTOGRAM_RECTANGLE
      }
    ];
  }
});
