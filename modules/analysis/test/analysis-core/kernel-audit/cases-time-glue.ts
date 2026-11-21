// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUCubeLayoutTranspose,
  GPUPeriodIndex,
  GPUTimeWordsRelative
} from '@luma-contributors/analysis/dataframe-analysis';
import type {GPUTemporalContext} from '../../../src/analysis-core/contracts/index';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';

const TEMPORAL_CONTEXT: GPUTemporalContext = {
  unit: 'millisecond',
  epoch: 'unix',
  encoding: 'int64-words',
  axis: {kind: 'irregular'}
};

function addCalendarPeriodIndex({graph, view, construct}: KernelAuditContext, rows: number): void {
  graph.add(
    construct(GPUPeriodIndex, {
      temporalContext: TEMPORAL_CONTEXT,
      periods: {
        kind: 'calendar',
        step: {calendarUnit: 'hour', interval: 6},
        origin: {year: 1969, month: 12, day: 30, hour: 22},
        count: 100_000
      },
      columns: {
        year: view('sint32', rows),
        month: view('uint32', rows),
        dayOfMonth: view('uint32', rows),
        hour: view('uint32', rows)
      },
      mask: view('uint32', rows),
      output: {periods: view('uint32', rows), status: {invalidCount: view('uint32', 1)}}
    })
  );
}

function addFixedPeriodIndex({graph, view, construct}: KernelAuditContext, rows: number): void {
  graph.add(
    construct(GPUPeriodIndex, {
      temporalContext: TEMPORAL_CONTEXT,
      periods: {kind: 'fixed', width: 2n ** 40n + 1n, origin: -(2n ** 50n), count: 0xfffffffd},
      timestamps: view('uint32x2', rows),
      mask: view('uint32', rows),
      output: {periods: view('uint32', rows), status: {invalidCount: view('uint32', 1)}}
    })
  );
}

function addTranspose(
  {graph, view, construct}: KernelAuditContext,
  format: 'uint32' | 'float32',
  from: 'cell-major' | 'period-major',
  periodCount: number
): void {
  const cellCount = 4096;
  const length = cellCount * periodCount;
  graph.add(
    construct(GPUCubeLayoutTranspose, {
      values: view(format, length) as never,
      output: view(format, length) as never,
      from,
      cellCount,
      periodCount
    })
  );
}

function addTimeWordsRelative({graph, view, construct}: KernelAuditContext, rows: number): void {
  graph.add(
    construct(GPUTimeWordsRelative, {
      timestamps: view('uint32x2', rows),
      temporalContext: TEMPORAL_CONTEXT,
      origin: 1_700_000_000_000n,
      output: view('float32', rows)
    })
  );
}

/** Cases of the glue contributors between time columns, period indices and cube layouts. */
export const TIME_GLUE_AUDIT_CASES: readonly KernelAuditCase[] = [
  {
    exportName: 'GPUPeriodIndex',
    name: 'calendar hour step, mask and status, binding-limited rows',
    build: context => addCalendarPeriodIndex(context, context.maximumRows(4))
  },
  {
    exportName: 'GPUPeriodIndex',
    name: 'calendar hour step, one row above the binding limit',
    build: context => addCalendarPeriodIndex(context, context.maximumRows(4) + 1)
  },
  {
    exportName: 'GPUPeriodIndex',
    name: 'fixed width, mask and status, binding-limited rows',
    build: context => addFixedPeriodIndex(context, context.maximumRows(8))
  },
  {
    exportName: 'GPUPeriodIndex',
    name: 'fixed width, one row above the binding limit',
    build: context => addFixedPeriodIndex(context, context.maximumRows(8) + 1)
  },
  {
    exportName: 'GPUCubeLayoutTranspose',
    name: 'float32 cell-major, binding-limited cube',
    build: context => addTranspose(context, 'float32', 'cell-major', context.maximumRows(4) / 4096)
  },
  {
    exportName: 'GPUCubeLayoutTranspose',
    name: 'uint32 period-major, binding-limited cube',
    build: context => addTranspose(context, 'uint32', 'period-major', context.maximumRows(4) / 4096)
  },
  {
    exportName: 'GPUCubeLayoutTranspose',
    name: 'cube one period above the binding limit',
    build: context =>
      addTranspose(context, 'uint32', 'cell-major', context.maximumRows(4) / 4096 + 1)
  },
  {
    exportName: 'GPUTimeWordsRelative',
    name: 'binding-limited rows',
    build: context => addTimeWordsRelative(context, context.maximumRows(8))
  },
  {
    exportName: 'GPUTimeWordsRelative',
    name: 'one row above the binding limit',
    build: context => addTimeWordsRelative(context, context.maximumRows(8) + 1)
  }
];
