// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {describe, expect, it} from 'vitest';
import {
  getGPUTemporalContextFromPreparedArrowColumn,
  makeGPUTimeVectorFromPreparedArrowColumn,
  prepareArrowTableForGPUAnalytics,
  type ArrowAnalyticsTimeUnit,
  type PreparedArrowAnalyticsTimeColumn
} from '../../src/analysis-core/graph-support/index';
import {
  getGPUTemporalContextCompatibility,
  resolveGPUTemporalContext
} from '../../src/analysis-core/contracts/index';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

const ARROW_TYPES: readonly [ArrowAnalyticsTimeUnit, arrow.TimeUnit][] = [
  ['second', arrow.TimeUnit.SECOND],
  ['millisecond', arrow.TimeUnit.MILLISECOND],
  ['microsecond', arrow.TimeUnit.MICROSECOND],
  ['nanosecond', arrow.TimeUnit.NANOSECOND]
];

function prepareColumn(type: arrow.DataType): PreparedArrowAnalyticsTimeColumn {
  const column = arrow.makeVector({type, data: new BigInt64Array([1n, 2n]), length: 2} as never);
  const table = new arrow.Table({time: column});
  return prepareArrowTableForGPUAnalytics(table).timeColumns['time'];
}

describe('temporal context of prepared Arrow time columns', () => {
  for (const [unit, arrowUnit] of ARROW_TYPES) {
    it(`carries the ${unit} unit through the upload`, () => {
      const column = prepareColumn(new arrow.Timestamp(arrowUnit, null));
      expect(getGPUTemporalContextFromPreparedArrowColumn(column)).toEqual({
        unit,
        epoch: 'unix',
        encoding: 'int64-words',
        axis: {kind: 'irregular'}
      });
      const device = createNullWebGPUDevice();
      const upload = makeGPUTimeVectorFromPreparedArrowColumn(device, column);
      expect(upload.temporalContext).toEqual(getGPUTemporalContextFromPreparedArrowColumn(column));
      expect(upload.timestamps.format).toBe('uint32x2');
      expect(upload.metadata).toBe(column);
      expect(() => resolveGPUTemporalContext('test', upload.temporalContext)).not.toThrow();
      upload.destroy();
      device.destroy();
    });
  }

  it('keeps a declared Arrow time zone and leaves an absent one undeclared', () => {
    const zoned = prepareColumn(new arrow.Timestamp(arrow.TimeUnit.MILLISECOND, 'Asia/Kolkata'));
    expect(getGPUTemporalContextFromPreparedArrowColumn(zoned).timeZone).toBe('Asia/Kolkata');
    const naive = prepareColumn(new arrow.Timestamp(arrow.TimeUnit.MILLISECOND, null));
    expect(getGPUTemporalContextFromPreparedArrowColumn(naive)).not.toHaveProperty('timeZone');
  });

  it('declares a signed Int64 column in its declared unit', () => {
    const column = arrow.makeVector({
      type: new arrow.Int64(),
      data: new BigInt64Array([1n]),
      length: 1
    } as never);
    const prepared = prepareArrowTableForGPUAnalytics(new arrow.Table({time: column}), {
      timeColumns: {time: 'microsecond'}
    }).timeColumns['time'];
    expect(getGPUTemporalContextFromPreparedArrowColumn(prepared).unit).toBe('microsecond');
  });

  it('lets two uploads of different units be told apart', () => {
    const seconds = prepareColumn(new arrow.Timestamp(arrow.TimeUnit.SECOND, null));
    const nanoseconds = prepareColumn(new arrow.Timestamp(arrow.TimeUnit.NANOSECOND, null));
    expect(
      getGPUTemporalContextCompatibility(
        getGPUTemporalContextFromPreparedArrowColumn(seconds),
        getGPUTemporalContextFromPreparedArrowColumn(nanoseconds)
      )
    ).toEqual({compatible: false, reason: 'units differ (second and nanosecond)'});
  });
});
