// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUEmergingHotSpots: Gi* hot spot z-scores in every time slice of a space-time cube, then a
// Mann-Kendall trend per cell. The picture is the trend z-score per cell: red where the hot spot is
// intensifying (the blob that grows over the 12 slices), blue where it fades. The muted layer is
// the last slice of the input cube. Hovering a cell boxes it and its Gi* neighborhood, charts its
// twelve Gi* z-scores (the line through the bars is z = 0) and names its category.

import {
  getGPUEmergingHotSpotsParameterValues,
  GPUEmergingHotSpots,
  GPU_EMERGING_HOT_SPOTS_CATEGORIES,
  GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {sampleFlashes} from '../../../data/lightning';
import {getRasterCell, getRasterCellCenter} from '../../pick';
import {parameters} from './emerging-hot-spots.parameters';

const WIDTH = 48;
const HEIGHT = 27;
const SLICES = 12;
const SERIES = [0.03, 0.04, 0.3, 0.24] as const;
const CATEGORY_NAMES = new Map<number, string>(
  Object.entries(GPU_EMERGING_HOT_SPOTS_CATEGORIES).map(([name, code]) => [
    code,
    name.toLowerCase().replaceAll('_', ' ')
  ])
);

export default defineOperationRun({
  parameters,
  caption: 'Evolving concentrations of Midwest lightning flashes in twelve 90-minute windows.',
  createInputs: () => {
    const flashes = sampleFlashes({width: 1, height: 1}, 4096);
    const counts = new Float32Array(WIDTH * HEIGHT * SLICES);
    for (let row = 0; row < flashes.count; row++) {
      const column = Math.min(WIDTH - 1, Math.floor(flashes.positions[row * 2] * WIDTH));
      const gridRow = Math.min(
        HEIGHT - 1,
        Math.floor((1 - flashes.positions[row * 2 + 1]) * HEIGHT)
      );
      const period = Math.min(SLICES - 1, Math.floor((flashes.hours[row] / 18) * SLICES));
      counts[(gridRow * WIDTH + column) * SLICES + period]++;
    }
    return counts;
  },
  build(builder, inputs) {
    const cellCount = WIDTH * HEIGHT;
    const cube = builder.allocate('cube', 'float32', cellCount * SLICES);
    const lastSlice = builder.allocate('last-slice', 'float32', cellCount);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH
    );
    const giZScores = builder.allocate('gi-z-scores', 'float32', cellCount * SLICES);
    const trendZ = builder.allocate('trend-z', 'float32', cellCount);
    const trendP = builder.allocate('trend-p', 'float32', cellCount);
    const trendS = builder.allocate('trend-s', 'sint32', cellCount);
    const category = builder.allocate('category', 'uint32', cellCount);
    const hotSliceCount = builder.allocate('hot-slice-count', 'uint32', cellCount);
    const coldSliceCount = builder.allocate('cold-slice-count', 'uint32', cellCount);
    builder.graph.add(
      new GPUEmergingHotSpots({
        values: cube.view,
        gridWidth: WIDTH,
        gridHeight: HEIGHT,
        sliceCount: SLICES,
        maximumRadius: 3,
        parameters: parameterBuffer.view,
        output: {
          giZScores: giZScores.view,
          trendZ: trendZ.view,
          trendP: trendP.view,
          trendS: trendS.view,
          category: category.view,
          hotSliceCount: hotSliceCount.view,
          coldSliceCount: coldSliceCount.view
        }
      })
    );
    return {
      outputs: {
        trendZ,
        lastSlice,
        giZScores,
        category,
        // Render-only: the hovered cell's Gi* z-score in each time slice.
        series: builder.allocate('series', 'float32', SLICES)
      },
      update: current => {
        const data = new Float32Array(inputs.length);
        const last = new Float32Array(cellCount);
        for (let cell = 0; cell < cellCount; cell++) {
          const offset = cell * SLICES;
          const slices = inputs.subarray(offset, offset + SLICES);
          const average = slices.reduce((sum, count) => sum + count, 0) / SLICES;
          for (let slice = 0; slice < SLICES; slice++) {
            data[offset + slice] = average + (slices[slice] - average) * current.trend;
          }
          last[cell] = data[offset + SLICES - 1];
        }
        cube.write(data);
        lastSlice.write(last);
        parameterBuffer.write(
          getGPUEmergingHotSpotsParameterValues({
            radius: current.radius,
            temporalWindow: current.temporalWindow
          })
        );
      }
    };
  },
  interaction: {
    kind: 'probe',
    hint: 'hover a cell',
    rest: [0.5, 0.5],
    resolve(pointer, {outputs, values, world, peek}) {
      const bounds = [0, 0, world.width, world.height] as const;
      const cell = getRasterCell(WIDTH, HEIGHT, bounds, pointer.x, pointer.y);
      const giZScores = peek(outputs.giZScores);
      const [centerX, centerY] = getRasterCellCenter(WIDTH, HEIGHT, bounds, cell.column, cell.row);
      const cellWidth = world.width / WIDTH;
      const cellHeight = world.height / HEIGHT;
      const reach = values.radius + 0.5;
      const zero = (SERIES[1] + (SERIES[3] - SERIES[1]) / 2) * world.height;
      const glyphs: RunGlyph[] = [
        {
          shape: 'line',
          from: [SERIES[0] * world.width, zero],
          to: [SERIES[2] * world.width, zero]
        },
        {
          shape: 'box',
          minimum: [centerX - reach * cellWidth, centerY - reach * cellHeight],
          maximum: [centerX + reach * cellWidth, centerY + reach * cellHeight],
          dashed: true
        },
        {
          shape: 'box',
          minimum: [centerX - cellWidth / 2, centerY - cellHeight / 2],
          maximum: [centerX + cellWidth / 2, centerY + cellHeight / 2]
        }
      ];
      if (!giZScores) return {glyphs};
      outputs.series.write(giZScores.slice(cell.index * SLICES, (cell.index + 1) * SLICES));
      const trend = peek(outputs.trendZ)?.[cell.index];
      const category = peek(outputs.category)?.[cell.index];
      return {
        glyphs,
        readout: `cell ${cell.column},${cell.row} · trend z ${
          trend === undefined || !Number.isFinite(trend)
            ? '–'
            : (trend >= 0 ? '+' : '') + trend.toFixed(1)
        } · ${category === undefined ? '…' : (CATEGORY_NAMES.get(category) ?? 'no pattern')}`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'raster', context: true, values: outputs.lastSlice, columns: WIDTH, rows: HEIGHT},
    {
      layer: 'raster',
      values: outputs.trendZ,
      columns: WIDTH,
      rows: HEIGHT,
      ramp: 'diverging',
      range: [-4, 4],
      alpha: 0.9
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.series,
      ramp: 'diverging',
      range: [-4, 4],
      rectangle: SERIES
    }
  ]
});
