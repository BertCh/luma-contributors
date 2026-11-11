// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUValueCube: flash energy binned into one dense (cell, period) cube. Periods come from
// GPUPeriodIndex (one hour each), cells are a lattice over the Midwest. The heatmap is the cube as
// stored, one row per cell and one column per hour; the dots are one hour of it on the map.

import {defineOperationRun, type RunBuffer} from '../../define-run';
import {
  addCubeHeatmap,
  addCubeSelection,
  addPeriodSlice,
  CUBE_STATISTICS,
  createMapCenters,
  getBinAt,
  getBinBox,
  getCubeFrame,
  getHeatmapBounds
} from './_lib/cube-display';
import {addEventCube, writeEnergyMask} from './_lib/event-cube';
import {
  createFlashEvents,
  createLatticeCells,
  FLASH_HOURS,
  summarizeCube
} from './_lib/flash-events';
import {parameters} from './value-cube.parameters';

const GRIDS = [
  [4, 3],
  [6, 4],
  [8, 5]
] as const;
const DOT_RADII = [26, 20, 15];

/** Hour of day (UTC) of a period: the flashes start at 12:00. */
const formatPeriod = (period: number) =>
  `${String((12 + period) % 24).padStart(2, '0')}:00-${String((13 + period) % 24).padStart(2, '0')}:00`;

type Inputs = ReturnType<typeof createInputs>;
type Outputs = {
  heatmap: RunBuffer;
  slice: RunBuffer;
  centers: RunBuffer;
  counts: RunBuffer;
  sums: RunBuffer;
  means: RunBuffer;
  invalidCount: RunBuffer;
};

function createInputs({world}: {world: {width: number; height: number}}) {
  const events = createFlashEvents();
  const defaultGrid = GRIDS[parameters[1].default];
  const summaries = GRIDS.map(([columns, rows]) => {
    const cells = createLatticeCells(events.positions, columns, rows);
    return {cells, ...summarizeCube(events, cells, columns * rows, FLASH_HOURS)};
  });
  const busiest = summaries[parameters[1].default];
  return {
    world,
    events,
    summaries,
    defaultGrid,
    mask: new Uint32Array(events.count),
    selected: {cell: busiest.busiestCell, period: busiest.busiestPeriod}
  };
}

const getGrid = (values: Readonly<Record<string, number>>) => GRIDS[values.gridSize];

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'Midwest GLM flashes binned by lattice cell and hour since 12:00 UTC, May 21, 2024; the value is flash energy in fJ.',
  createInputs,
  build(builder, inputs, values) {
    const [columns, rows] = getGrid(values);
    const cellCount = columns * rows;
    const frame = getCubeFrame(cellCount, FLASH_HOURS);
    const {events} = inputs;
    const cube = addEventCube(builder, events, inputs.summaries[values.gridSize].cells, cellCount);
    const {counts, sums, means, invalidCount} = cube;
    const selection = addCubeSelection(builder);
    const cubeViews = {frame, counts, sums, means, selection};
    return {
      outputs: {
        heatmap: addCubeHeatmap(builder, cubeViews),
        slice: addPeriodSlice(builder, cubeViews),
        centers: builder.upload(
          'centers',
          'float32x2',
          createMapCenters(frame, inputs.world, columns, rows)
        ),
        counts,
        sums,
        means,
        invalidCount
      },
      update: current => {
        writeEnergyMask(cube, inputs.mask, events.energy, current.mask);
        selection.write(
          Uint32Array.of(
            current.statistic,
            inputs.selected.period,
            Math.min(inputs.selected.cell, cellCount - 1),
            0
          )
        );
      }
    };
  },
  rebuildOn: ['gridSize'],
  // Probe: a bin of the heatmap picks a cell and a period; over the map the nearest cell is picked.
  interaction: {
    kind: 'probe',
    hint: 'hover a cell and hour',
    rest: ({inputs, toUnit}) => {
      const frame = getCubeFrame(inputs.defaultGrid[0] * inputs.defaultGrid[1], FLASH_HOURS);
      const box = getBinBox(frame, inputs.world, inputs.selected.cell, inputs.selected.period);
      return toUnit([(box.minimum[0] + box.maximum[0]) / 2, (box.minimum[1] + box.maximum[1]) / 2]);
    },
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const [columns, rows] = getGrid(values);
      const cellCount = columns * rows;
      const frame = getCubeFrame(cellCount, FLASH_HOURS);
      const bin = getBinAt(frame, world, pointer.x, pointer.y);
      let cell = Math.min(inputs.selected.cell, cellCount - 1);
      let period = inputs.selected.period;
      if (bin.isInside) {
        cell = bin.cell;
        period = bin.period;
      } else {
        // Over the map: the cell under the pointer, in the period already chosen.
        const centers = createMapCenters(frame, world, columns, rows);
        let nearest = Infinity;
        for (let candidate = 0; candidate < cellCount; candidate++) {
          const distance = Math.hypot(
            centers[candidate * 2] - pointer.x,
            centers[candidate * 2 + 1] - pointer.y
          );
          if (distance < nearest) {
            nearest = distance;
            cell = candidate;
          }
        }
      }
      const changed = cell !== inputs.selected.cell || period !== inputs.selected.period;
      inputs.selected = {cell, period};
      const centers = createMapCenters(frame, world, columns, rows);
      const bounds = getHeatmapBounds(frame, world);
      const box = getBinBox(frame, world, cell, period);
      const index = cell * FLASH_HOURS + period;
      const count = peek(outputs.counts)?.[index];
      const sum = peek(outputs.sums)?.[index];
      const mean = peek(outputs.means)?.[index];
      const invalid = peek(outputs.invalidCount)?.[0];
      const statistic = CUBE_STATISTICS[values.statistic];
      return {
        rerun: changed,
        spotlight: null,
        glyphs: [
          {shape: 'box', minimum: box.minimum, maximum: box.maximum},
          {
            shape: 'box',
            minimum: [bounds[0], box.minimum[1]],
            maximum: [bounds[2], box.maximum[1]],
            dashed: true
          },
          {
            shape: 'box',
            minimum: [box.minimum[0], 0],
            maximum: [box.maximum[0], 1],
            dashed: true
          },
          {
            shape: 'ring',
            at: [centers[cell * 2], centers[cell * 2 + 1]],
            radius: ((world.width * 0.3) / columns) * 0.5
          }
        ],
        readout:
          count === undefined
            ? `cell ${cell} · ${formatPeriod(period)}`
            : `cell ${cell} · ${formatPeriod(period)} · ${
                count === 0
                  ? 'empty'
                  : `n ${count}, sum ${Math.round(sum ?? 0).toLocaleString('en-US')} fJ, mean ${(mean ?? 0).toFixed(1)} fJ (${statistic} drawn)`
              }${invalid ? ` · invalidCount ${invalid}` : ''}`
      };
    }
  },
  draw: ({inputs, outputs, values}) => {
    const [columns, rows] = getGrid(values);
    const frame = getCubeFrame(columns * rows, FLASH_HOURS);
    const summary = inputs.summaries[values.gridSize];
    const maximum = [summary.maximumCount, summary.maximumSum, summary.maximumMean][
      values.statistic
    ];
    const range = [0, Math.log2(Math.max(2, maximum))] as const;
    return [
      {
        layer: 'dots',
        context: true,
        positions: outputs.centers,
        radius: DOT_RADII[values.gridSize] + 2,
        color: [0.4, 0.45, 0.55, 0.35]
      },
      {
        layer: 'raster',
        values: outputs.heatmap,
        columns: frame.columns,
        rows: frame.rows,
        ramp: 'sequential',
        logarithmic: true,
        range
      },
      {
        layer: 'dots',
        positions: outputs.centers,
        values: outputs.slice,
        radius: DOT_RADII[values.gridSize],
        ramp: 'sequential',
        logarithmic: true,
        range
      }
    ];
  }
});
