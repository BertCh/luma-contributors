// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCellPeriodKeys, GPUValueCube} from '@luma-contributors/analysis/dataframe-analysis';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';

/** Bytes per bin of the fixed-point words of `GPUValueCube`: three uint32 words. */
const BYTES_PER_BIN = 12;

function addValueCube(
  {graph, view}: KernelAuditContext,
  cellCount: number,
  periodCount: number,
  rows: number,
  withStatus: boolean
): void {
  const binCount = cellCount * periodCount;
  graph.add(
    new GPUValueCube({
      cellCount,
      periodCount,
      cells: view('uint32', rows),
      periods: view('uint32', rows),
      values: view('float32', rows),
      mask: view('uint32', rows),
      output: {
        counts: view('uint32', binCount),
        sums: view('float32', binCount),
        means: view('float32', binCount),
        status: withStatus ? {invalidCount: view('uint32', 1)} : undefined
      }
    })
  );
}

function addCellPeriodKeys({graph, view}: KernelAuditContext, rows: number): void {
  graph.add(
    new GPUCellPeriodKeys({
      cells: view('uint32x2', rows),
      periods: view('uint32', rows),
      periodCount: 0xfffffffd,
      fill: 'zeros',
      mask: view('uint32', rows),
      output: {keys: view('uint32x2', rows), status: {invalidCount: view('uint32', 1)}}
    })
  );
}

/** Cases of the value cube and the cell-period key adapter. */
export const VALUE_CUBE_AUDIT_CASES: readonly KernelAuditCase[] = [
  {
    exportName: 'GPUValueCube',
    name: 'all outputs, mask and status, binding-limited cube',
    build: context => addValueCube(context, context.maximumRows(BYTES_PER_BIN), 1, 100_000, true)
  },
  {
    exportName: 'GPUValueCube',
    name: 'one bin above the binding limit',
    build: context =>
      addValueCube(context, context.maximumRows(BYTES_PER_BIN) + 1, 1, 100_000, false)
  },
  {
    exportName: 'GPUCellPeriodKeys',
    name: 'binding-limited rows, widest period axis',
    build: context => addCellPeriodKeys(context, context.maximumRows(8))
  },
  {
    exportName: 'GPUCellPeriodKeys',
    name: 'one row above the binding limit',
    build: context => addCellPeriodKeys(context, context.maximumRows(8) + 1)
  }
];
