// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {makeSpatialFixture, readIntegers} from '../spatial-test-utils';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {GPURipleyEnvelope} from '../../src/dataframe-analysis/pair-statistics/gpu-ripley-envelope';
import {getGPURipleyParameterValues} from '../../src/dataframe-analysis/pair-statistics/ripley-parameters';
import {GPUPeriodIndex} from '../../src/dataframe-analysis/period-index/gpu-period-index';
import {GPUCellPeriodKeys} from '../../src/dataframe-analysis/value-cube/gpu-cell-period-keys';
import {GPUValueCube} from '../../src/dataframe-analysis/value-cube/gpu-value-cube';
import {
  GPUColumnProfile,
  GPU_COLUMN_PROFILE_STATISTIC_COUNT
} from '../../src/dataframe-analysis/column-profile/index';
import {GPUGroupStatistics} from '../../src/dataframe-analysis/group-statistics/index';

for (const subject of ['envelope', 'periods', 'keys', 'cube'] as const) {
  it(`R1 ${subject} preserves trailing scalar status capacity and clears rejection on replay`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = makeSpatialFixture(device);
    try {
      const status = fixture.view('uint32', new Uint32Array([99, 41, 42]));
      // Standard status ports expose one row; the backing allocation still has
      // two sentinel words so readback detects writes past the scalar view.
      const invalidCount = fixture.graph.createDataView(status.view.buffer, {
        format: 'uint32',
        length: 1
      });
      const input = fixture.view(
        'uint32',
        new Uint32Array([subject === 'periods' || subject === 'keys' ? 2 : 1, 0])
      );
      if (subject === 'envelope') {
        const coordinates = fixture.view('float32', new Float32Array([0, 0, 1, 1]));
        const positions = fixture.graph.createDataView(coordinates.view.buffer, {
          format: 'float32x2',
          length: 2
        });
        const parameters = fixture.view(
          'float32',
          getGPURipleyParameterValues({
            bounds: [0, 0, 1, 1],
            maximumDistance: 2,
            edgeCorrection: 'none'
          })
        );
        fixture.graph.add(
          new GPURipleyEnvelope({
            positions,
            parameters: parameters.view,
            gridSize: [2, 2],
            radiusCount: 1,
            simulationCount: 2,
            output: {
              lower: fixture.view('float32', new Float32Array(1)).view,
              unplacedCount: status.view
            }
          })
        );
      } else if (subject === 'periods') {
        fixture.graph.add(
          new GPUPeriodIndex({
            temporalContext: {
              encoding: 'int64-words',
              epoch: 'unix',
              unit: 'millisecond',
              axis: {kind: 'irregular'}
            },
            timestamps: fixture.graph.createDataView(input.view.buffer, {
              format: 'uint32x2',
              length: 1
            }),
            periods: {kind: 'fixed', origin: 0n, width: 1n, count: 1},
            output: {
              periods: fixture.view('uint32', new Uint32Array(1)).view,
              status: {invalidCount}
            }
          })
        );
      } else if (subject === 'keys') {
        const cells = fixture.view('uint32', new Uint32Array([0, 0]));
        const keys = fixture.view('uint32', new Uint32Array(2));
        fixture.graph.add(
          new GPUCellPeriodKeys({
            cells: fixture.graph.createDataView(cells.view.buffer, {format: 'uint32x2', length: 1}),
            periods: fixture.graph.createDataView(input.view.buffer, {format: 'uint32', length: 1}),
            periodCount: 2,
            fill: 'zeros',
            layout: 'legacy-one-bit',
            output: {
              keys: fixture.graph.createDataView(keys.view.buffer, {format: 'uint32x2', length: 1}),
              status: {invalidCount}
            }
          })
        );
      } else {
        fixture.graph.add(
          new GPUValueCube({
            cells: fixture.view('uint32', new Uint32Array([0])).view,
            periods: fixture.graph.createDataView(input.view.buffer, {format: 'uint32', length: 1}),
            values: fixture.view('float32', new Float32Array([1])).view,
            cellCount: 1,
            periodCount: 1,
            output: {
              counts: fixture.view('uint32', new Uint32Array(1)).view,
              status: {invalidCount}
            }
          })
        );
      }
      // Rectangle mapping rejects zero draws. Otherwise exactly one row
      // exceeds the declared period range, so the invalid count is one.
      // The status contract owns only word 0; 41 and 42 must survive.
      await fixture.run();
      expect(await readIntegers(status.buffer)).toEqual([subject === 'envelope' ? 0 : 1, 41, 42]);
      input.buffer.write(new Uint32Array([0, 0]));
      await fixture.run();
      expect(await readIntegers(status.buffer)).toEqual([0, 41, 42]);
    } finally {
      fixture.destroy();
    }
  });
}

it('R3 profile overwrites numeric-column and unused top-category count slots', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const topCategoryCounts = fixture.view('uint32', new Uint32Array(6).fill(99));
    fixture.graph.add(
      new GPUColumnProfile({
        columns: [
          {values: fixture.view('float32', new Float32Array([10, 20, 30])).view},
          {
            kind: 'category',
            categoryCount: 2,
            values: fixture.view('uint32', new Uint32Array([0, 1, 1])).view
          }
        ],
        histogramBinCount: 2,
        hyperLogLogPrecision: 4,
        topCategoryCount: 3,
        output: {
          statistics: fixture.view(
            'float32',
            new Float32Array(2 * GPU_COLUMN_PROFILE_STATISTIC_COUNT)
          ).view,
          topCategories: fixture.view('uint32', new Uint32Array(6)).view,
          topCategoryCounts: topCategoryCounts.view
        }
      })
    );
    // Numeric columns have no category ranks: all three zeros must be
    // written. Codes [0,1,1] give rank supports [2,1,0], including padding.
    await fixture.run();
    expect(await readIntegers(topCategoryCounts.buffer)).toEqual([0, 0, 0, 2, 1, 0]);
  } finally {
    fixture.destroy();
  }
});

it('R3 compact group count, column count and unique count tails clear on replay', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const keys = fixture.view('uint32', new Uint32Array([10, 20, 30]));
    const values = fixture.view('float32', new Float32Array([2, 4, 8]));
    const mask = fixture.view('uint32', new Uint32Array([1, 1, 1]));
    const counts = fixture.view('uint32', new Uint32Array(3).fill(99));
    const columnCounts = fixture.view('uint32', new Uint32Array(3).fill(99));
    const uniqueCounts = fixture.view('uint32', new Uint32Array(3).fill(99));
    fixture.graph.add(
      new GPUGroupStatistics({
        keys: keys.view,
        mask: mask.view,
        columns: [
          {
            values: values.view,
            statistics: ['count', 'uniqueCount'],
            output: {counts: columnCounts.view, uniqueCounts: uniqueCounts.view}
          }
        ],
        output: {
          keys: fixture.view('uint32', new Uint32Array(3)).view,
          counts: counts.view,
          status: {
            count: fixture.view('uint32', new Uint32Array(1)).view,
            overflow: fixture.view('uint32', new Uint32Array(1)).view
          }
        }
      })
    );
    // Each distinct key initially has exactly one finite, unique value.
    await fixture.run();
    for (const output of [counts, columnCounts, uniqueCounts])
      expect(await readIntegers(output.buffer)).toEqual([1, 1, 1]);
    // Retain only key 10 on the same graph. Compact slots 1/2 are now tail
    // rows, and every one of their three count products must be overwritten.
    mask.buffer.write(new Uint32Array([1, 0, 0]));
    await fixture.run();
    for (const output of [counts, columnCounts, uniqueCounts])
      expect(await readIntegers(output.buffer)).toEqual([1, 0, 0]);
  } finally {
    fixture.destroy();
  }
});
