// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCubeLayoutTranspose: one cell-by-hour flash cube in both layouts. The left block is the cube as
// the value cube wrote it (cell-major, one row per cell); the right block is the transposed copy
// (period-major, one row per period). A second transpose back gives the original bit for bit.

import {GPUCubeLayoutTranspose} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {
  addLayoutComposite,
  getLayoutBinAt,
  getLayoutBinBox,
  getLayoutFrame
} from './_lib/cube-display';
import {addEventCube, writeEnergyMask} from './_lib/event-cube';
import {
  createFlashEvents,
  createLatticeCells,
  FLASH_HOURS,
  summarizeCube
} from './_lib/flash-events';
import {parameters} from './cube-layout-transpose.parameters';

const GRIDS = [
  [4, 3],
  [6, 4],
  [8, 5]
] as const;

type Inputs = ReturnType<typeof createInputs>;
type Outputs = {
  composite: RunBuffer;
  cellMajor: RunBuffer;
  periodMajor: RunBuffer;
  roundTrip: RunBuffer;
};

function createInputs({world}: {world: {width: number; height: number}}) {
  const events = createFlashEvents();
  const summaries = GRIDS.map(([columns, rows]) => {
    const cells = createLatticeCells(events.positions, columns, rows);
    return {cells, ...summarizeCube(events, cells, columns * rows, FLASH_HOURS)};
  });
  return {world, events, summaries, mask: new Uint32Array(events.count)};
}

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'Flash counts and mean energy per cell and hour: the same cube stored cell-major (left, row per cell) and period-major (right, row per period). The lines trace the stored row of the hovered bin.',
  createInputs,
  build(builder, inputs, values) {
    const [columns, rows] = GRIDS[values.cellCount];
    const cellCount = columns * rows;
    const cube = addEventCube(
      builder,
      inputs.events,
      inputs.summaries[values.cellCount].cells,
      cellCount
    );
    const isInteger = values.values === 0;
    const format = isInteger ? 'uint32' : 'float32';
    const cellMajor = (isInteger ? cube.counts : cube.means) as RunBuffer;
    const periodMajor = builder.allocate('period-major', format, cellCount * FLASH_HOURS);
    const roundTrip = builder.allocate('round-trip', format, cellCount * FLASH_HOURS);
    builder.graph.add(
      new GPUCubeLayoutTranspose({
        id: 'to-period-major',
        values: cellMajor.view as never,
        from: 'cell-major',
        cellCount,
        periodCount: FLASH_HOURS,
        output: periodMajor.view as never
      })
    );
    builder.graph.add(
      new GPUCubeLayoutTranspose({
        id: 'to-cell-major',
        values: periodMajor.view as never,
        from: 'period-major',
        cellCount,
        periodCount: FLASH_HOURS,
        output: roundTrip.view as never
      })
    );
    const frame = getLayoutFrame(cellCount, FLASH_HOURS);
    return {
      outputs: {
        composite: addLayoutComposite(builder, {
          frame,
          cellMajor: cellMajor as never,
          periodMajor: periodMajor as never
        }),
        cellMajor,
        periodMajor,
        roundTrip
      },
      update: current => writeEnergyMask(cube, inputs.mask, inputs.events.energy, current.mask)
    };
  },
  rebuildOn: ['values', 'cellCount'],
  // Probe: a bin in either block; the other block shows where the same value was moved to.
  interaction: {
    kind: 'probe',
    hint: 'hover a bin',
    rest: ({inputs, toUnit}) => {
      const summary = inputs.summaries[parameters[1].default];
      const frame = getLayoutFrame(
        GRIDS[parameters[1].default][0] * GRIDS[parameters[1].default][1],
        FLASH_HOURS
      );
      const box = getLayoutBinBox(
        frame,
        inputs.world,
        summary.busiestCell,
        summary.busiestPeriod,
        'cell-major'
      );
      return toUnit([(box.minimum[0] + box.maximum[0]) / 2, (box.minimum[1] + box.maximum[1]) / 2]);
    },
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const [columns, rows] = GRIDS[values.cellCount];
      const cellCount = columns * rows;
      const frame = getLayoutFrame(cellCount, FLASH_HOURS);
      const summary = inputs.summaries[values.cellCount];
      const bin = getLayoutBinAt(frame, world, pointer.x, pointer.y) ?? {
        cell: summary.busiestCell,
        period: summary.busiestPeriod
      };
      const first = getLayoutBinBox(frame, world, bin.cell, bin.period, 'cell-major');
      const second = getLayoutBinBox(frame, world, bin.cell, bin.period, 'period-major');
      const from = bin.cell * FLASH_HOURS + bin.period;
      const to = bin.period * cellCount + bin.cell;
      const original = peek(outputs.cellMajor);
      const moved = peek(outputs.periodMajor);
      const returned = peek(outputs.roundTrip);
      let differing = 0;
      if (original && returned) {
        for (let index = 0; index < original.length; index++) {
          if (!Object.is(original[index], returned[index])) differing++;
        }
      }
      const value = moved?.[to];
      const center = (box: typeof first): [number, number] => [
        (box.minimum[0] + box.maximum[0]) / 2,
        (box.minimum[1] + box.maximum[1]) / 2
      ];
      // Storage order: the row holding the bin is contiguous in memory, left to right in both blocks.
      const traceRow = (block: 'cell-major' | 'period-major') => {
        const first = block === 'cell-major' ? [bin.cell, 0] : [0, bin.period];
        const last =
          block === 'cell-major' ? [bin.cell, FLASH_HOURS - 1] : [cellCount - 1, bin.period];
        const [x0, y0] = center(getLayoutBinBox(frame, world, first[0], first[1], block));
        const [x1, y1] = center(getLayoutBinBox(frame, world, last[0], last[1], block));
        return [x0, y0, x1, y1];
      };
      return {
        spotlight: null,
        glyphs: [
          {shape: 'polyline', points: traceRow('cell-major'), dashed: true},
          {shape: 'polyline', points: traceRow('period-major'), dashed: true},
          {shape: 'box', minimum: first.minimum, maximum: first.maximum},
          {shape: 'box', minimum: second.minimum, maximum: second.maximum},
          {shape: 'arrow', from: center(first), to: center(second)}
        ],
        readout: `cell ${bin.cell}, period ${bin.period} → cell-major ${from} · period-major ${to}${
          value === undefined ? '' : ` · ${Number.isNaN(value) ? 'empty' : Math.round(value)}`
        }${original ? (differing === 0 ? ' · round trip exact' : ` · ${differing} differ`) : ''}`
      };
    }
  },
  draw: ({inputs, outputs, values}) => {
    const [columns, rows] = GRIDS[values.cellCount];
    const frame = getLayoutFrame(columns * rows, FLASH_HOURS);
    const summary = inputs.summaries[values.cellCount];
    const maximum = values.values === 0 ? summary.maximumCount : summary.maximumMean;
    return [
      {
        layer: 'raster',
        values: outputs.composite,
        columns: frame.columns,
        rows: frame.rows,
        ramp: 'sequential',
        logarithmic: true,
        range: [0, Math.log2(Math.max(2, maximum))]
      }
    ];
  }
});
