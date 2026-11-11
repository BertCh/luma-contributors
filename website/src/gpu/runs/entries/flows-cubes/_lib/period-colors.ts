// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Display adapter: a period column as dot colors. GPUPeriodIndex writes 0xfffffffe for a valid time
// outside the periods and 0xffffffff for a masked row; both draw muted here (the unreached value).

import {createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';
import type {RunBuffer, RunBuilder} from '../../../define-run';

/** Copies `periods` and maps every value at or above `periodCount` to the muted 0xffffffff. */
export function addPeriodColors(
  builder: RunBuilder,
  options: {id: string; periods: RunBuffer<'uint32'>; periodCount: number}
): RunBuffer<'uint32'> {
  const colors = builder.allocate(`${options.id}-colors`, 'uint32', options.periods.length);
  builder.graph.add(
    createWGSLKernelNode(builder.graph, {
      id: `${options.id}-colors`,
      operation: 'display-period-colors',
      invocationCount: options.periods.length,
      bindings: [
        {name: 'periods', view: options.periods.view, type: 'u32', access: 'read'},
        {name: 'colors', view: colors.view, type: 'u32', access: 'read_write'}
      ],
      body: `
  let period = periods[periodsOffset + index];
  colors[colorsOffset + index] = select(0xffffffffu, period, period < ${options.periodCount}u);`
    })
  );
  return colors;
}
