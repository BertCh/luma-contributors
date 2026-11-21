// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUTimeWindowFilter,
  getGPUTimeWindowFilterParameterValues
} from '../../../src/dataframe-analysis/time-window-filter/index';
import {RecipeTestFixture} from '../../gpu-spatial-analysis/recipes/recipe-harness';
import {requireTestDevice} from '../../utils/require-test-device';

it('sorted time windows write literal draw first instances on repeated submissions without rowRange', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const firstPresent of [true, false]) {
    const fixture = new RecipeTestFixture(device, 'time-first-oracle');
    const rows = fixture.output('rows', 'uint32', 4);
    const count = fixture.output('count', 'uint32', 1);
    const overflow = fixture.output('overflow', 'uint32', 1);
    const first = fixture.output('first', 'uint32', 1);
    const window = fixture.output('window', 'float32', 8);
    fixture.graph.add(
      new GPUTimeWindowFilter({
        sorted: true,
        timestamps: fixture.input('times', new Float32Array([10, 20, 30, 40]), 'float32', 4),
        window: window.view,
        output: {
          rows: rows.view,
          status: {count: count.view, overflow: overflow.view},
          drawFirstInstance: firstPresent ? first.view : undefined
        }
      })
    );
    try {
      // First selected row differs from both the selected count and the timestamp window start.
      for (const [start, end, expectedRows] of [
        [29, 31, [2]],
        [9, 21, [0, 1]]
      ] as const) {
        window.buffer.write(getGPUTimeWindowFilterParameterValues({start, end}));
        first.buffer.write(new Uint32Array([123]));
        rows.buffer.write(new Uint32Array(4).fill(123));
        count.buffer.write(new Uint32Array([123]));
        overflow.buffer.write(new Uint32Array([123]));
        fixture.run();
        if (firstPresent) expect(await fixture.readUint32(first, 1)).toEqual([expectedRows[0]]);
        expect(await fixture.readUint32(count, 1)).toEqual([expectedRows.length]);
        expect(await fixture.readUint32(overflow, 1)).toEqual([0]);
        expect(await fixture.readUint32(rows, expectedRows.length)).toEqual([...expectedRows]);
      }
    } finally {
      fixture.destroy();
    }
  }
});
