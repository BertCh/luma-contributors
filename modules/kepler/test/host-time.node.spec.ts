// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {prepareArrowTableForGPUAnalytics} from '@luma.gl/graph/arrow';
import {KeplerComputeSession} from '../src/session/compute-session';
import {createNullWebGPUDevice} from '../../analysis/test/utils/gpu-contributor-test-utils';
import * as arrow from 'apache-arrow';
import {expect, test, vi} from 'vitest';
import {
  getPreparedTimeDomainMilliseconds,
  makeSampleTimestampColumn,
  prepareKeplerTimeRangeMilliseconds,
  SAMPLE_TIMESTAMP_EPOCH_MILLISECONDS,
  TimeRuntime
} from '../../../examples/deck/kepler-compute/src/time-runtime';

test('native time endpoints convert exact epoch milliseconds to source units and reject unsupported precision', () => {
  expect(prepareKeplerTimeRangeMilliseconds(-1, 1, 'microsecond')).toEqual({
    start: -1000n,
    end: 1000n,
    unit: 'microsecond'
  });
  expect(
    prepareKeplerTimeRangeMilliseconds(1_700_000_000_000, 1_700_000_000_001, 'nanosecond')
  ).toEqual({
    start: 1_700_000_000_000_000_000n,
    end: 1_700_000_000_001_000_000n,
    unit: 'nanosecond'
  });
  for (const endpoint of [0.5, NaN, Infinity, Number.MAX_SAFE_INTEGER, 8_640_000_000_000_001]) {
    expect(() => prepareKeplerTimeRangeMilliseconds(0, endpoint, 'millisecond')).toThrow(
      /integral milliseconds/
    );
  }
  expect(() => prepareKeplerTimeRangeMilliseconds(1, 2, 'second')).toThrow(/precision/);
  expect(() => prepareKeplerTimeRangeMilliseconds(2, 1, 'millisecond')).toThrow(/ordered/);
  expect(() => prepareKeplerTimeRangeMilliseconds(0, 9_223_372_036_855, 'nanosecond')).toThrow(
    /Int64/
  );
});

test('sample timestamps are a real Arrow Timestamp with exact native values and preserved timezone', () => {
  const sample = makeSampleTimestampColumn(new Uint32Array([1439, 0, 1]), {
    timeZone: 'America/New_York'
  });
  expect(sample.column.type).toBeInstanceOf(arrow.TimestampMillisecond);
  expect(sample.column.type.timezone).toBe('America/New_York');
  expect(Array.from(sample.values)).toEqual([
    SAMPLE_TIMESTAMP_EPOCH_MILLISECONDS + 1439 * 60_000,
    SAMPLE_TIMESTAMP_EPOCH_MILLISECONDS,
    SAMPLE_TIMESTAMP_EPOCH_MILLISECONDS + 60_000
  ]);
  const prepared = prepareArrowTableForGPUAnalytics(new arrow.Table({timestamp: sample.column}));
  const column = prepared.timeColumns.timestamp;
  expect(column.unit).toBe('millisecond');
  expect(column.timeZone).toBe('America/New_York');
  expect(getPreparedTimeDomainMilliseconds(column)).toEqual(sample.domainMilliseconds);
  expect(() =>
    makeSampleTimestampColumn(new Uint32Array([1]), {epochMilliseconds: 8_640_000_000_000_000})
  ).toThrow(/Date range/);
});

test('domain preparation preserves signed64 words, nullable slices and original batch boundaries', () => {
  const first = makeTimestampTable([-9999n, -1000n, null], 'America/New_York').slice(1);
  const second = makeTimestampTable([1000n, 2000n], 'America/New_York');
  const prepared = prepareArrowTableForGPUAnalytics(first.concat(second));
  const column = prepared.timeColumns.timestamp;
  expect(column.chunks.map(words => words.length)).toEqual([4, 4]);
  expect(column.validity.map(mask => Array.from(mask))).toEqual([
    [1, 0],
    [1, 1]
  ]);
  expect(getPreparedTimeDomainMilliseconds(column)).toEqual([-1, 2]);
  expect(column.timeZone).toBe('America/New_York');
  expect(
    getPreparedTimeDomainMilliseconds(
      prepareArrowTableForGPUAnalytics(makeTimestampTable([null, null])).timeColumns.timestamp
    )
  ).toBeNull();
  expect(
    getPreparedTimeDomainMilliseconds(
      prepareArrowTableForGPUAnalytics(makeTimestampTable([1001n])).timeColumns.timestamp
    )
  ).toEqual([1, 2]);
  expect(
    getPreparedTimeDomainMilliseconds(
      prepareArrowTableForGPUAnalytics(makeTimestampTable([-1001n])).timeColumns.timestamp
    )
  ).toEqual([-2, -1]);
  const invalid = {...column, validity: [new Uint32Array(1), column.validity[1]]};
  expect(() => getPreparedTimeDomainMilliseconds(invalid)).toThrow(/source batches/);
});

test('runtime borrows exact time columns, separates display/domain masks and rejects stale revisions', async () => {
  const device = createNullWebGPUDevice();
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 16 * 1024 * 1024});
  const source = makeTimestampTable([-1000n, 0n, 1000n]);
  const dataset = session.importArrowDataset({id: 'observations', table: source});
  const runtime = new TimeRuntime({
    session,
    datasetId: dataset.id,
    column: 'timestamp',
    dimension: 'absolute-time'
  });
  try {
    expect(runtime.domainMilliseconds).toEqual([-1, 1]);
    expect(runtime.externalRange.displayDimension.id).toBe('absolute-time');
    expect(runtime.externalRange.domainDimension.id).toBe('absolute-time');
    expect(runtime.externalRange.displayDimension.mask).not.toBe(
      runtime.externalRange.domainDimension.mask
    );
    expect(runtime.externalRange.displayDimension.mask.data.map(chunk => chunk.length)).toEqual([
      3
    ]);
    const allocation = vi.spyOn(device, 'createBuffer');
    runtime.externalRange.setDisplayRange(0, 1);
    runtime.externalRange.setDomainRange(-1, 1);
    runtime.externalRange.setDisplayEmpty();
    expect(() => runtime.externalRange.setDisplayRange(0.5, 1)).toThrow(/integral milliseconds/);
    expect(allocation).not.toHaveBeenCalled();
    allocation.mockRestore();
    session.importArrowDataset({id: 'observations', table: source});
    expect(() => runtime.externalRange.setDisplayRange(0, 1)).toThrow(/no longer active/);
    expect(
      dataset.value.timeColumns.timestamp.timestamps.data.every(chunk => !chunk.buffer.destroyed)
    ).toBe(true);
  } finally {
    await runtime.destroy(Promise.resolve());
    expect(
      dataset.value.timeColumns.timestamp.timestamps.data.every(chunk => chunk.buffer.destroyed)
    ).toBe(true);
    await session.destroy(Promise.resolve());
    device.destroy();
  }
});

test('runtime constructor failure releases leases and reservation without adopting source ownership', async () => {
  const device = createNullWebGPUDevice();
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 16 * 1024 * 1024});
  const dataset = session.importArrowDataset({
    id: 'observations',
    table: makeTimestampTable([0n, 1000n])
  });
  const before = session.registry.memoryUsage.totalBytes;
  expect(
    () =>
      new TimeRuntime({
        session,
        datasetId: dataset.id,
        column: 'timestamp',
        workingMemoryBudgetBytes: 0
      })
  ).toThrow(/reservation/);
  expect(session.registry.memoryUsage.totalBytes).toBe(before);
  expect(() => new TimeRuntime({session, datasetId: dataset.id, column: 'missing'})).toThrow(
    /prepared exact/
  );
  expect(session.registry.memoryUsage.totalBytes).toBe(before);
  expect(
    dataset.value.timeColumns.timestamp.timestamps.data.every(chunk => !chunk.buffer.destroyed)
  ).toBe(true);
  await session.destroy(Promise.resolve());
  expect(
    dataset.value.timeColumns.timestamp.timestamps.data.every(chunk => chunk.buffer.destroyed)
  ).toBe(true);
  device.destroy();
});

function makeTimestampTable(values: readonly (bigint | null)[], timeZone = 'UTC'): arrow.Table {
  const bitmap = new Uint8Array(Math.ceil(values.length / 8));
  const timestamps = new BigInt64Array(values.length);
  let nullCount = 0;
  for (const [index, value] of values.entries()) {
    if (value === null) nullCount++;
    else {
      timestamps[index] = value;
      bitmap[index >>> 3] |= 1 << (index % 8);
    }
  }
  return new arrow.Table({
    timestamp: arrow.makeVector(
      arrow.makeData({
        type: new arrow.TimestampMicrosecond(timeZone),
        data: timestamps,
        length: values.length,
        nullCount,
        nullBitmap: bitmap
      })
    )
  });
}
