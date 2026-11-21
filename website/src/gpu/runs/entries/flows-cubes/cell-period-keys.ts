// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCellPeriodKeys: each flash's 64-bit H3 cell key and its period are packed into one group key, and
// GPUGroupStatistics counts the flashes per key, a sparse cell-by-period cube. The table is the cube:
// one row per occupied cell in key order, one column per period. On the map, muted flashes were not
// packed (masked out, or a cell finer than the period bits allow).

import {
  GPUCellPeriodKeys,
  GPUGroupStatistics,
  GPUPeriodIndex,
  getGPUCellKeyFreeBits,
  getGPUCellPeriodKeyPeriodBits,
  unpackGPUCellPeriodKeyChecked
} from '@luma-contributors/analysis/dataframe-analysis';
import {createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';
import {GPUPointToCell} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {writeEnergyMask} from './_lib/event-cube';
import {
  createFlashEvents,
  FLASH_HOURS,
  FLASH_ORIGIN_MILLISECONDS,
  FLASH_TEMPORAL_CONTEXT,
  MILLISECONDS_PER_HOUR
} from './_lib/flash-events';
import {parameters} from './cell-period-keys.parameters';

const PERIOD_COUNTS = [6, 18, 72];
const KEY_LAYOUT = 'legacy-one-bit';
/** Occupied cells shown as table rows (the first in key order). */
const ROW_COUNT = 32;
/** Enough for one group per flash: the extract has 4,096 flashes. */
const GROUP_CAPACITY = 4096;
const MAP_WIDTH_SHARE = 0.4;

type Inputs = ReturnType<typeof createInputs>;
type Outputs = {
  mapPositions: RunBuffer;
  colors: RunBuffer;
  highlight: RunBuffer;
  keys: RunBuffer;
  invalidCount: RunBuffer;
  groupKeys: RunBuffer;
  groupCounts: RunBuffer;
  groupCount: RunBuffer;
  groupOverflow: RunBuffer;
};

function createInputs({world}: {world: {width: number; height: number}}) {
  const events = createFlashEvents();
  // Longitude and latitude of the extract: 99 to 83 degrees west, 36 to 46 degrees north.
  const degrees = Float32Array.from(events.positions, (value, index) =>
    index % 2 === 0 ? -99 + value * 16 : 36 + value * 10
  );
  const mapPositions = Float32Array.from(events.positions, (value, index) =>
    index % 2 === 0
      ? 0.02 * world.width + value * (MAP_WIDTH_SHARE * 0.94 * world.width)
      : 0.08 + value * 0.84
  );
  return {
    world,
    events,
    degrees,
    mapPositions,
    mask: new Uint32Array(events.count),
    focus: {row: -1, period: -1, key: ''}
  };
}

const getFrame = (periodCount: number) => {
  const left = Math.round(periodCount * (MAP_WIDTH_SHARE / (1 - MAP_WIDTH_SHARE)));
  return {left, columns: left + periodCount};
};

/** Rows of the sparse table: occupied cells in key order, each with its (period, count, key) groups. */
function readGroups(
  keys: ArrayLike<number>,
  counts: ArrayLike<number>,
  groupCount: number,
  periodCount: number
) {
  const cells: bigint[] = [];
  const bins = new Map<number, {key: bigint; count: number}>();
  for (let group = 0; group < groupCount; group++) {
    const key = (BigInt(keys[group * 2 + 1]) << 32n) | BigInt(keys[group * 2]);
    const unpacked = unpackGPUCellPeriodKeyChecked(key, periodCount, 'ones', KEY_LAYOUT);
    if (!unpacked) continue;
    const {cellKey, period} = unpacked;
    if (cells.at(-1) !== cellKey) cells.push(cellKey);
    bins.set((cells.length - 1) * periodCount + period, {key, count: counts[group]});
  }
  return {cells, bins};
}

export default defineOperationRun<Inputs, Outputs, {table: RunBuffer}>({
  parameters,
  caption:
    'Flash counts per H3 cell and period from packed cell-period keys; rows are occupied cells in key order, columns are periods over 18 hours.',
  createInputs,
  build(builder, inputs, values) {
    const periodCount = PERIOD_COUNTS[values.periodCount];
    const {events} = inputs;
    const cells = builder.allocate('cells', 'uint32x2', events.count);
    builder.graph.add(
      new GPUPointToCell({
        family: 'h3',
        resolution: values.resolution,
        positions: builder.upload('degrees', 'float32x2', inputs.degrees).view,
        output: {cells: cells.view}
      })
    );
    const periods = builder.allocate('periods', 'uint32', events.count);
    builder.graph.add(
      new GPUPeriodIndex({
        temporalContext: FLASH_TEMPORAL_CONTEXT,
        periods: {
          kind: 'fixed',
          width: BigInt((FLASH_HOURS * MILLISECONDS_PER_HOUR) / periodCount),
          origin: FLASH_ORIGIN_MILLISECONDS,
          count: periodCount
        },
        timestamps: builder.upload('timestamps', 'uint32x2', events.timestampWords).view,
        output: {periods: periods.view}
      })
    );
    const mask = builder.allocate('mask', 'uint32', events.count);
    const keys = builder.allocate('keys', 'uint32x2', events.count);
    const invalidCount = builder.allocate('invalid-count', 'uint32', 1);
    builder.graph.add(
      new GPUCellPeriodKeys({
        cells: cells.view,
        periods: periods.view,
        periodCount,
        fill: 'ones',
        layout: KEY_LAYOUT,
        mask: mask.view,
        output: {keys: keys.view, status: {invalidCount: invalidCount.view}}
      })
    );
    const groupKeys = builder.allocate('group-keys', 'uint32x2', GROUP_CAPACITY);
    const groupCounts = builder.allocate('group-counts', 'uint32', GROUP_CAPACITY);
    const groupCount = builder.allocate('group-count', 'uint32', 1);
    const groupOverflow = builder.allocate('group-overflow', 'uint32', 1);
    builder.graph.add(
      new GPUGroupStatistics({
        keys: keys.view,
        columns: [],
        output: {
          keys: groupKeys.view,
          counts: groupCounts.view,
          status: {
            count: groupCount.view,
            overflow: groupOverflow.view
          }
        }
      })
    );
    // Dot colors: the period of a packed flash; a flash that was not packed (all-ones key) is muted.
    const colors = builder.allocate('colors', 'uint32', events.count);
    builder.graph.add(
      createWGSLKernelNode(builder.graph, {
        id: 'packed-colors',
        operation: 'display-packed-colors',
        invocationCount: events.count,
        bindings: [
          {name: 'keys', view: keys.view, type: 'u32', access: 'read'},
          {name: 'colors', view: colors.view, type: 'u32', access: 'read_write'}
        ],
        body: `
  let low = keys[keysOffset + index * 2u];
  let high = keys[keysOffset + index * 2u + 1u];
  let isPacked = !(low == 0xffffffffu && high == 0xffffffffu);
  colors[colorsOffset + index] = select(0xffffffffu, low & ${(1 << getGPUCellPeriodKeyPeriodBits(periodCount, KEY_LAYOUT)) - 1}u, isPacked);`
      })
    );
    return {
      outputs: {
        mapPositions: builder.upload('map-positions', 'float32x2', inputs.mapPositions),
        colors,
        highlight: builder.allocate('highlight', 'float32', events.count),
        keys,
        invalidCount,
        groupKeys,
        groupCounts,
        groupCount,
        groupOverflow
      },
      update: current => writeEnergyMask({mask}, inputs.mask, events.energy, current.mask)
    };
  },
  rebuildOn: ['resolution', 'periodCount'],
  async afterRun({outputs, values, upload}) {
    const periodCount = PERIOD_COUNTS[values.periodCount];
    const [keyBytes, countBytes] = await Promise.all([
      outputs.groupKeys.buffer.readAsync(),
      outputs.groupCounts.buffer.readAsync()
    ]);
    const keys = new Uint32Array(
      keyBytes.buffer.slice(keyBytes.byteOffset, keyBytes.byteOffset + keyBytes.byteLength)
    );
    const counts = new Uint32Array(
      countBytes.buffer.slice(countBytes.byteOffset, countBytes.byteOffset + countBytes.byteLength)
    );
    let groupCount = 0;
    while (
      groupCount < GROUP_CAPACITY &&
      !(keys[groupCount * 2] === 0xffffffff && keys[groupCount * 2 + 1] === 0xffffffff) &&
      counts[groupCount] > 0
    )
      groupCount++;
    const {cells, bins} = readGroups(keys, counts, groupCount, periodCount);
    const frame = getFrame(periodCount);
    const table = new Float32Array(frame.columns * ROW_COUNT).fill(Number.NaN);
    for (const [bin, {count}] of bins) {
      const row = Math.floor(bin / periodCount);
      if (row < ROW_COUNT) table[row * frame.columns + frame.left + (bin % periodCount)] = count;
    }
    void cells;
    return {table: upload('table', 'float32', table)};
  },
  // Probe: a bin of the table is one (cell, period) group; its flashes light up on the map.
  interaction: {
    kind: 'probe',
    hint: 'hover a cell and period',
    rest: [0.9, 0.97],
    resolve(pointer, {inputs, outputs, values, world, build, peek}) {
      const periodCount = PERIOD_COUNTS[values.periodCount];
      const frame = getFrame(periodCount);
      const left = (world.width * frame.left) / frame.columns;
      const groupKeys = peek(outputs.groupKeys);
      const groupCounts = peek(outputs.groupCounts);
      const flashKeys = peek(outputs.keys);
      const invalid = peek(outputs.invalidCount)?.[0];
      const overflow = peek(outputs.groupOverflow)?.[0];
      const freeBits = getGPUCellKeyFreeBits('h3', values.resolution);
      const bits = getGPUCellPeriodKeyPeriodBits(periodCount, KEY_LAYOUT);
      let summary = `res ${values.resolution} · ${bits} of ${freeBits} free bits${
        invalid === undefined ? '' : ` · invalidCount ${invalid}`
      }`;
      if (!groupKeys || !groupCounts || !flashKeys) return {spotlight: null, readout: summary};
      let groupCount = 0;
      while (groupCount < GROUP_CAPACITY && groupCounts[groupCount] > 0) groupCount++;
      const {cells, bins} = readGroups(groupKeys, groupCounts, groupCount, periodCount);
      // The reason to use keys: only the occupied bins of the dense cells-by-periods cube are stored.
      if (overflow) summary = `group capacity exceeded, bins undercounted · ${summary}`;
      summary = `${groupCount.toLocaleString('en-US')} of ${(cells.length * periodCount).toLocaleString('en-US')} bins occupied · ${summary}`;
      const pointerPeriod = ((pointer.x - left) / (world.width - left)) * periodCount;
      const pointerRow = (1 - pointer.y) * ROW_COUNT;
      // Snap to the nearest occupied bin, measured in bins of the picture.
      let row = Math.min(ROW_COUNT - 1, Math.floor(pointerRow));
      let period = Math.min(periodCount - 1, Math.floor(pointerPeriod));
      let nearest = Infinity;
      for (const bin of bins.keys()) {
        const binRow = Math.floor(bin / periodCount);
        if (binRow >= ROW_COUNT) continue;
        const binPeriod = bin % periodCount;
        const distance = Math.hypot(
          ((binPeriod + 0.5 - pointerPeriod) * (world.width - left)) / periodCount,
          (binRow + 0.5 - pointerRow) / ROW_COUNT
        );
        if (distance < nearest) {
          nearest = distance;
          row = binRow;
          period = binPeriod;
        }
      }
      if (pointer.x < left) {
        // Over the map: the group of the flash nearest the pointer.
        let nearestFlash = -1;
        let nearestDistance = Infinity;
        for (let flash = 0; flash < inputs.events.count; flash++) {
          const distance = Math.hypot(
            inputs.mapPositions[flash * 2] - pointer.x,
            inputs.mapPositions[flash * 2 + 1] - pointer.y
          );
          if (distance < nearestDistance) {
            nearestDistance = distance;
            nearestFlash = flash;
          }
        }
        const flashKey =
          (BigInt(flashKeys[nearestFlash * 2 + 1]) << 32n) | BigInt(flashKeys[nearestFlash * 2]);
        for (const [bin, candidate] of bins) {
          if (candidate.key === flashKey) {
            row = Math.floor(bin / periodCount);
            period = bin % periodCount;
          }
        }
      }
      const group = bins.get(row * periodCount + period);
      const key = `${build}:${group?.key ?? 'none'}`;
      if (key !== inputs.focus.key) {
        const highlight = new Float32Array(inputs.events.count);
        if (group) {
          for (let flash = 0; flash < inputs.events.count; flash++) {
            const value = (BigInt(flashKeys[flash * 2 + 1]) << 32n) | BigInt(flashKeys[flash * 2]);
            if (value === group.key) highlight[flash] = 1;
          }
        }
        outputs.highlight.write(highlight);
        inputs.focus = {row, period, key};
      }
      const cell = cells[row];
      return {
        spotlight: null,
        focus: group ? row * periodCount + period : -1,
        glyphs: [
          {
            shape: 'box',
            minimum: [left, 1 - (row + 1) / ROW_COUNT],
            maximum: [world.width, 1 - row / ROW_COUNT],
            dashed: true
          },
          {
            shape: 'box',
            minimum: [
              left + (period / periodCount) * (world.width - left),
              1 - (row + 1) / ROW_COUNT
            ],
            maximum: [
              left + ((period + 1) / periodCount) * (world.width - left),
              1 - row / ROW_COUNT
            ]
          }
        ],
        readout: group
          ? `cell ${cell?.toString(16).slice(0, 9)} · period ${period} · ${group.count} flashes · ${summary}`
          : `no flashes here · ${summary}`
      };
    }
  },
  draw: ({inputs, outputs, values, extra}) => {
    const periodCount = PERIOD_COUNTS[values.periodCount];
    const frame = getFrame(periodCount);
    return [
      {
        layer: 'dots',
        context: true,
        positions: outputs.mapPositions,
        color: [0.4, 0.45, 0.55, 0.4],
        radius: 2.2
      },
      {
        layer: 'raster',
        values: extra.table,
        columns: frame.columns,
        rows: ROW_COUNT,
        ramp: 'sequential',
        logarithmic: true,
        range: [0, 6]
      },
      {layer: 'dots', positions: outputs.mapPositions, values: outputs.colors, radius: 3},
      {
        layer: 'dots',
        positions: outputs.mapPositions,
        values: outputs.highlight,
        range: [0, 1],
        fade: true,
        ramp: 'sequential',
        radius: 9
      }
    ];
  }
});
