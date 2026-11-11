// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The real operations that turn timestamped flashes into a dense cube: GPUPeriodIndex gives each
// flash its hour, GPUValueCube bins flash energy by lattice cell and hour. Runs of the cube
// operations share it, so each shows its own step of the same pipeline.

import {GPUPeriodIndex, GPUValueCube} from '@luma-contributors/analysis/dataframe-analysis';
import type {RunBuffer, RunBuilder} from '../../../define-run';
import {
  FLASH_HOURS,
  FLASH_ORIGIN_MILLISECONDS,
  FLASH_TEMPORAL_CONTEXT,
  MILLISECONDS_PER_HOUR,
  type FlashEvents
} from './flash-events';

export type EventCube = {
  mask: RunBuffer<'uint32'>;
  periods: RunBuffer<'uint32'>;
  counts: RunBuffer<'uint32'>;
  sums: RunBuffer<'float32'>;
  means: RunBuffer<'float32'>;
  invalidCount: RunBuffer<'uint32'>;
};

/** Adds GPUPeriodIndex (one-hour periods) and GPUValueCube (`cellCount` by 18 hours) to the graph. */
export function addEventCube(
  builder: RunBuilder,
  events: FlashEvents,
  cells: Uint32Array,
  cellCount: number
): EventCube {
  const timestamps = builder.upload('timestamps', 'uint32x2', events.timestampWords);
  const periods = builder.allocate('periods', 'uint32', events.count);
  builder.graph.add(
    new GPUPeriodIndex({
      temporalContext: FLASH_TEMPORAL_CONTEXT,
      periods: {
        kind: 'fixed',
        width: BigInt(MILLISECONDS_PER_HOUR),
        origin: FLASH_ORIGIN_MILLISECONDS,
        count: FLASH_HOURS
      },
      timestamps: timestamps.view,
      output: {periods: periods.view}
    })
  );
  const mask = builder.allocate('mask', 'uint32', events.count);
  const counts = builder.allocate('counts', 'uint32', cellCount * FLASH_HOURS);
  const sums = builder.allocate('sums', 'float32', cellCount * FLASH_HOURS);
  const means = builder.allocate('means', 'float32', cellCount * FLASH_HOURS);
  const invalidCount = builder.allocate('invalid-count', 'uint32', 1);
  builder.graph.add(
    new GPUValueCube({
      cellCount,
      periodCount: FLASH_HOURS,
      cells: builder.upload('cells', 'uint32', cells).view,
      periods: periods.view,
      values: builder.upload('energy', 'float32', events.energy).view,
      mask: mask.view,
      output: {
        counts: counts.view,
        sums: sums.view,
        means: means.view,
        status: {invalidCount: invalidCount.view}
      }
    })
  );
  return {mask, periods, counts, sums, means, invalidCount};
}

/** Writes the row mask: 1 for flashes whose energy is at least `minimumEnergy`. */
export function writeEnergyMask(
  cube: Pick<EventCube, 'mask'>,
  scratch: Uint32Array,
  energy: Float32Array,
  minimumEnergy: number
): void {
  for (let row = 0; row < energy.length; row++) scratch[row] = energy[row] >= minimumEnergy ? 1 : 0;
  cube.mask.write(scratch);
}
