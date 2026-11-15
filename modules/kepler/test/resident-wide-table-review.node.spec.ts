// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {expect, test, vi} from 'vitest';
import {createRequire} from 'node:module';
import {dirname} from 'node:path';
import {ComputePipeline, type ComputePipelineProps, type Device} from '@luma.gl/core';
import {NullDevice} from '@luma.gl/core/test-utils';
import {
  KeplerComputeSession,
  type KeplerSessionExplorer,
  type KeplerMemoryReservation
} from '../src';
import {
  readWideWords,
  ResidentTableGPUSorter,
  type ResidentTableGPUSortRequest
} from '../../../examples/deck/kepler-compute/src/resident-table-gpu-sorter';
import {
  encodeWideSortWord,
  getPublishedTableSortMemoryBudgetBytes
} from '../../../examples/deck/kepler-compute/src/resident-table-sort-keys';
import {
  makeRelativeFloat32Column,
  ResidentTableWideStatistics
} from '../../../examples/deck/kepler-compute/src/resident-table-wide-statistics';
import {previewArrowSource} from '../../../examples/deck/kepler-compute/src/source-preview';
import {
  createColumnStatisticsReader,
  setColumnStatsComputer
} from '../../../examples/deck/kepler-compute/src/resident-column-statistics';
import type {ResidentTableProvider} from '../../../examples/deck/kepler-compute/src/resident-data-table';

test.each([
  'wide-int64',
  'float64'
] as const)('sliced %s keys address the sliced values from zero', kind => {
  const vector: arrow.Vector =
    kind === 'wide-int64'
      ? arrow.vectorFromArray([10n, 20n, 30n, 40n], new arrow.Int64()).slice(1, 3)
      : arrow.vectorFromArray([10, 20, 30, 40], new arrow.Float64()).slice(1, 3);
  expect(vector.data[0].offset).toBe(1);
  const keys = readWideWords(vector, kind, vector.length);
  for (let row = 0; row < vector.length; row++) {
    const value = vector.get(row);
    const words = new Uint32Array(
      kind === 'wide-int64'
        ? new BigInt64Array([value as bigint]).buffer
        : new Float64Array([value as number]).buffer
    );
    const expected = encodeWideSortWord(kind, words[0], words[1]);
    expect({lo: keys.lo[row], hi: keys.hi[row], valid: keys.valid[row]}).toEqual({
      lo: expected.lo,
      hi: expected.hi,
      valid: 1
    });
  }
});

test.each([
  'wide-int64',
  'float64'
] as const)('sliced nullable/chunked %s keys retain bitmap offsets', kind => {
  const vector: arrow.Vector =
    kind === 'wide-int64'
      ? arrow.vectorFromArray(
          Array.from({length: 20}, (_, row) => (row % 3 ? BigInt(row - 10) : null)),
          new arrow.Int64()
        )
      : arrow.vectorFromArray(
          Array.from({length: 20}, (_, row) => (row % 3 ? row - 10.5 : null)),
          new arrow.Float64()
        );
  const sliced = vector.slice(7, 12).concat(vector.slice(13, 19));
  const keys = readWideWords(sliced, kind, sliced.length);
  for (let row = 0; row < sliced.length; row++) {
    expect(keys.valid[row]).toBe(sliced.isValid(row) ? 1 : 0);
    if (!sliced.isValid(row)) continue;
    const value = sliced.get(row);
    const words = new Uint32Array(
      kind === 'wide-int64'
        ? new BigInt64Array([value as bigint]).buffer
        : new Float64Array([value as number]).buffer
    );
    const expected = encodeWideSortWord(kind, words[0], words[1]);
    expect([keys.lo[row], keys.hi[row]]).toEqual([expected.lo, expected.hi]);
  }
});

// Compile real sort graphs and allocate CPU-backed buffers, but never execute a GPU kernel.
class PlanningPipeline extends ComputePipeline {
  readonly device: Device;
  readonly handle = null;
  constructor(device: Device, props: ComputePipelineProps) {
    super(device, props);
    this.device = device;
  }
  setBindings(): void {}
  protected override destroyHandle(): void {}
}

test.each([
  'production',
  'former-too-small',
  'registry-limited'
] as const)('300k exact-sort compilation enforces the %s memory ceiling', async ceiling => {
  const rowCount = 300_000;
  const oldBudget = 8 * 1024 * 1024 + rowCount * 32;
  const budget =
    ceiling === 'former-too-small'
      ? oldBudget
      : ceiling === 'registry-limited'
        ? 512 * 1024 * 1024
        : getPublishedTableSortMemoryBudgetBytes(rowCount);
  const device = new NullDevice({_cachePipelines: false, _cacheShaders: false});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.defineProperty(device.limits, 'maxComputeWorkgroupsPerDimension', {value: 65535});
  Object.defineProperty(device.limits, 'maxBufferSize', {value: 2 ** 30});
  Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 2 ** 28});
  Object.defineProperty(device.limits, 'maxStorageBuffersPerShaderStage', {value: 16});
  vi.spyOn(device, 'createComputePipeline').mockImplementation(props => {
    expect(session.registry.memoryUsage.reservedBytes.gpu).toBe(admissionBytes);
    return new PlanningPipeline(device, props!);
  });
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 256 * 1024 * 1024});
  const source = new arrow.Table({
    i64: arrow.makeVector(
      BigInt64Array.from(
        {length: rowCount + 1},
        (_, row) => 10_000_000_000n + BigInt((row * 7919) % rowCount)
      )
    )
  }).slice(1);
  const dataset = session.importArrowDataset({
    id: 'sliced',
    table: source,
    normalization: {float64: {mode: 'round', maximumAbsoluteError: 0, maximumRelativeError: 1e-6}}
  });
  const admissionBytes = Math.min(budget, session.registry.memoryUsage.availableBytes);
  const explorer = {
    dataset,
    isCurrent: true,
    allowIncompletePopulation: false
  } as unknown as KeplerSessionExplorer;
  const sorter = new ResidentTableGPUSorter({session, explorer, memoryBudgetBytes: budget});
  try {
    expect(dataset.value.source.getChild('i64')!.data[0].offset).toBe(1);
    expect(sorter.planSort('i64', rowCount).route).toBe('gpu');
    const keys = readWideWords(dataset.value.source.getChild('i64')!, 'wide-int64', rowCount);
    const order = Uint32Array.from({length: rowCount}, (_, row) => row).sort(
      (first, second) => keys.hi[first] - keys.hi[second] || keys.lo[first] - keys.lo[second]
    );
    expect([...order.subarray(0, 3)]).toEqual([299999, 217678, 135357]);
    // Preparing the private program avoids encoding/reading nonexistent GPU results in this probe.
    const preparing = (
      sorter as unknown as {
        getProgram(
          request: ResidentTableGPUSortRequest
        ): Promise<{reservation: KeplerMemoryReservation}>;
      }
    ).getProgram({column: 'i64', direction: 'ascending', population: 'all', offset: 0, limit: 3});
    if (ceiling === 'former-too-small') {
      await expect(preparing).rejects.toThrow(/including compiled scratch.*working memory budget/);
      expect(session.registry.memoryUsage.reservedBytes.gpu).toBe(0);
    } else {
      const program = await preparing;
      expect(program.reservation.byteLength).toBeGreaterThan(oldBudget);
      expect(program.reservation.byteLength).toBeLessThanOrEqual(budget);
      expect(session.registry.memoryUsage.reservedBytes.gpu).toBe(program.reservation.byteLength);
    }
  } finally {
    await sorter.destroy();
    await session.destroy(Promise.resolve());
    device.destroy();
  }
});

test('finite accepted Float64 values refuse an overflowing relative Float32 range', () => {
  const source = new arrow.Table({value: arrow.makeVector(new Float64Array([-3e38, 3e38]))});
  expect(
    previewArrowSource(source, {
      float64: {mode: 'round', maximumAbsoluteError: 0, maximumRelativeError: 1e-6}
    }).canUpload
  ).toBe(true);
  expect(() => makeRelativeFloat32Column(source.getChild('value')!, source.numRows)).toThrow(
    /range.*Float32/
  );
});

test.each([
  262_144, 262_145
])('finite wide ranges at %i rows use bounded CPU statistics or explicitly refuse GPU overflow', async rowCount => {
  // Install the same pinned native computer as the browser, without importing the deck-loading
  // utils barrel. This validates the CPU threshold using real native statistics.
  const nativeRequire = createRequire(import.meta.url);
  const nativeStatistics = nativeRequire(
    `${dirname(nativeRequire.resolve('@kepler.gl/utils'))}/column-statistics.js`
  );
  setColumnStatsComputer(nativeStatistics.computeColumnStats);
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 128 * 1024 * 1024});
  const source = new arrow.Table({
    value: arrow.makeVector(
      Float64Array.from({length: rowCount}, (_, row) => (row % 2 ? 3e38 : -3e38))
    )
  });
  const dataset = session.importArrowDataset({
    id: 'extreme',
    table: source,
    normalization: {float64: {mode: 'round', maximumAbsoluteError: 0, maximumRelativeError: 1e-6}}
  });
  const explorer = {
    dataset,
    isCurrent: true,
    allowIncompletePopulation: false,
    readSelectedCount: async () => ({
      revision: 1,
      datasetRevision: dataset.revision,
      selectedCount: rowCount
    })
  } as unknown as KeplerSessionExplorer;
  const wide = new ResidentTableWideStatistics({
    session,
    explorer,
    memoryBudgetBytes: 32 * 1024 * 1024
  });
  const readPopulationRows = vi.fn(async () =>
    Uint32Array.from({length: rowCount}, (_, row) => row)
  );
  const provider: ResidentTableProvider = {
    datasetId: dataset.id,
    sourceRevision: dataset.revision,
    maximumPageSize: 50,
    fields: [{name: 'value', sortable: true}],
    getRevision: () => '1',
    subscribe: () => () => {},
    readPage: vi.fn()
  };
  const reader = createColumnStatisticsReader({
    source,
    provider,
    wide,
    readPopulationRows,
    isCurrent: () => true,
    getPopulationRows: () => rowCount,
    gpu: {plan: () => ({supported: true}), read: vi.fn()}
  });
  const allocateBuffer = vi.spyOn(device, 'createBuffer');
  try {
    if (rowCount === 262_144) {
      const result = await reader.readColumnStatistics({column: 'value', population: 'all'});
      expect(result).toMatchObject({source: 'cpu', rowCount, nullCount: 0});
      expect(Number.isFinite((result!.stats as {mean: number}).mean)).toBe(true);
      expect(readPopulationRows).toHaveBeenCalledOnce();
    } else {
      await expect(
        reader.readColumnStatistics({column: 'value', population: 'all'})
      ).rejects.toThrow(/range.*Float32.*262,144.*CPU/);
      expect(readPopulationRows).not.toHaveBeenCalled();
    }
    expect(allocateBuffer).not.toHaveBeenCalled();
    expect(session.registry.memoryUsage.reservedBytes.gpu).toBe(0);
  } finally {
    allocateBuffer.mockRestore();
    await wide.destroy();
    await session.destroy(Promise.resolve());
    device.destroy();
  }
});
