// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import * as arrow from 'apache-arrow';
import {expect, test, vi} from 'vitest';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../analysis/test/utils/gpu-contributor-test-utils';
import {makeResidentCalendarTimeView} from '../../../examples/deck/kepler-compute/src/resident-calendar-time';

function makeFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'calendar-time-node'});
  const source = (id: string, length: number, byteOffset = 0) =>
    graph.createDataView(
      graph.createTransientBuffer({
        id,
        byteLength: Math.max(1, length) * 8 + byteOffset,
        usage: Buffer.STORAGE
      }),
      {format: 'uint32x2', length, byteOffset}
    );
  return {device, graph, source};
}

test('calendar normalization borrows millisecond/empty chunks and never admits declared Int64 dates', () => {
  const {device, graph, source} = makeFixture();
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const addPass = vi.spyOn(graph, 'addComputePass');
  try {
    const milliseconds = source('milliseconds', 3, 4);
    const empty = source('empty', 0);
    allocate.mockClear();
    expect(
      makeResidentCalendarTimeView(graph, {
        id: 'milliseconds',
        timestamps: milliseconds,
        unit: 'millisecond',
        sourceType: 'Timestamp<MILLISECOND>'
      })
    ).toBe(milliseconds);
    expect(
      makeResidentCalendarTimeView(graph, {
        id: 'empty',
        timestamps: empty,
        unit: 'nanosecond',
        sourceType: 'Timestamp<NANOSECOND>'
      })
    ).toBe(empty);
    expect(allocate).not.toHaveBeenCalled();
    expect(addPass).not.toHaveBeenCalled();
    expect(() =>
      makeResidentCalendarTimeView(graph, {
        id: 'declared-int64',
        timestamps: milliseconds,
        unit: 'millisecond',
        sourceType: 'Int64'
      })
    ).toThrow();
  } finally {
    device.destroy();
  }
});

test('calendar unit graph planning preserves each source chunk and produces bounded integer-only WGSL', () => {
  const {device, graph, source} = makeFixture();
  const addPass = vi.spyOn(graph, 'addComputePass');
  const allocate = vi.spyOn(device, 'createBuffer');
  try {
    const chunks = [
      source('seconds', 3, 4),
      source('microseconds', 1, 16),
      source('nanoseconds', 9, 0)
    ];
    const outputs = chunks.map((timestamps, index) =>
      makeResidentCalendarTimeView(graph, {
        id: `unit-${index}`,
        timestamps,
        unit: (['second', 'microsecond', 'nanosecond'] as const)[index],
        sourceType: 'Timestamp'
      })
    );
    expect(outputs.map(view => view.length)).toEqual([3, 1, 9]);
    expect(outputs.map(view => view.byteOffset)).toEqual([0, 0, 0]);
    expect(outputs.every((view, index) => view.buffer !== chunks[index].buffer)).toBe(true);
    expect(allocate).not.toHaveBeenCalled();
    expect(
      addPass.mock.calls.map(([pass]) => pass.resources?.map(resource => resource.usage))
    ).toEqual(Array.from({length: 3}, () => ['storage-read', 'storage-write']));
    const sources = [...getParsedKernelSources(addPass.mock.calls.map(([pass]) => pass)).values()];
    expect(sources).toHaveLength(3);
    for (const shader of sources) {
      expect(getStorageBindingCount(shader)).toBe(2);
      expect(shader).not.toMatch(/\bf32\b/);
      expect(shader).toContain('0x7fffffffu');
    }
    expect(sources[0]).toContain('let sourceIndex = 1u + index * 2u');
    expect(sources[1]).toContain('let sourceIndex = 4u + index * 2u');
  } finally {
    device.destroy();
  }
});

test('native Arrow fractional Number rounding differs from exact time at contemporary month and epoch boundaries', () => {
  const boundary = BigInt(Date.parse('2024-03-01T00:00:00Z'));
  const nanoseconds = boundary * 1000000n - 1n;
  const vector = arrow.makeVector(
    arrow.makeData({
      type: new arrow.TimestampNanosecond(),
      data: new BigInt64Array([nanoseconds, -1n])
    })
  );
  const native = new Date(vector.get(0)!).getTime();
  const exact = Number(nanoseconds / 1000000n);
  expect(new Date(native).toISOString()).toBe('2024-03-01T00:00:00.000Z');
  expect(new Date(exact).toISOString()).toBe('2024-02-29T23:59:59.999Z');
  // Date's truncation maps a negative fraction to epoch; an exact native-unit [0, end] filter does not.
  expect(new Date(vector.get(1)!).getTime()).toBe(0);
  expect(-1n / 1000000n).toBe(0n);
  expect(-1n >= 0n).toBe(false);
  const outsideDate = 8640000000000000001n;
  const microseconds = arrow.makeVector(
    arrow.makeData({type: new arrow.TimestampMicrosecond(), data: new BigInt64Array([outsideDate])})
  );
  expect(new Date(microseconds.get(0)!).getTime()).toBe(8640000000000000);
  expect(outsideDate > 8640000000000000000n).toBe(true);
});
