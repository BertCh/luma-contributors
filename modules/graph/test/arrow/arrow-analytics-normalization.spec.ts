// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  prepareArrowTableForGPUAnalytics,
  makeGPUAnalyticsTableFromArrowTable,
  makeArrowTableFromGPUAnalyticsTable,
  makeGPUTimeVectorFromPreparedArrowColumn
} from '@luma-contributors/graph/arrow';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import * as arrow from 'apache-arrow';
import {expect, test} from 'vitest';

test('normalized Arrow categories, nullable integers and exact time words survive WebGPU upload', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('WebGPU is required for the Arrow normalization integration test');
  const type = new arrow.TimestampMicrosecond('UTC');
  const time = new arrow.Vector([
    arrow.makeData({
      type,
      data: new BigInt64Array([-1n, 9007199254740993n, 9223372036854775807n]),
      length: 3
    })
  ]);
  const source = new arrow.Table({
    count: arrow.vectorFromArray([255, null, 1], new arrow.Uint8()),
    category: arrow.vectorFromArray(['first', null, 'second'], new arrow.Utf8()),
    time
  });
  const prepared = prepareArrowTableForGPUAnalytics(source);
  const uploaded = makeGPUAnalyticsTableFromArrowTable(device, prepared.table);
  const uploadedTime = makeGPUTimeVectorFromPreparedArrowColumn(device, prepared.timeColumns.time);
  try {
    const readback = await makeArrowTableFromGPUAnalyticsTable(uploaded);
    expect(Array.from(readback.getChild('count')!)).toEqual([255, null, 1]);
    expect(Array.from(readback.getChild('category')!)).toEqual(['first', null, 'second']);
    const bytes = await uploadedTime.timestamps.data[0].buffer.readAsync();
    expect(new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)).toEqual(
      new Uint32Array([0xffffffff, 0xffffffff, 1, 2097152, 0xffffffff, 0x7fffffff])
    );
  } finally {
    uploadedTime.destroy();
    uploaded.table.destroy();
    Object.values(uploaded.validity).forEach(vector => vector?.destroy());
  }
});
