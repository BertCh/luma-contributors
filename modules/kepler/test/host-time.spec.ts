// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession} from '../src/session/compute-session';
import * as arrow from 'apache-arrow';
import {expect, test, vi} from 'vitest';
import {TimeRuntime} from '../../../examples/deck/kepler-compute/src/time-runtime';
import {ResidentTableBinding} from '../../../examples/deck/kepler-compute/src/resident-table-binding';
import {
  installKeplerComputeTableBridge,
  type KeplerComputeFilter
} from '../../../examples/deck/kepler-compute/src/kepler-table-bridge';

test('native absolute time filters preserve exact64 instants and independent display/domain populations', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('WebGPU is required for native timestamp integration');
  const epochMilliseconds = Date.UTC(2026, 0, 1);
  const epochMicroseconds = BigInt(epochMilliseconds) * 1000n;
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 40 * 1024 * 1024});
  const source = makeBatch([epochMicroseconds, epochMicroseconds + 1n], [10, 20]).concat(
    makeBatch(
      [epochMicroseconds + 1000n, epochMicroseconds + 2000n, epochMicroseconds + 3000n, null],
      [30, 40, 50, 60]
    )
  );
  const dataset = session.importArrowDataset({id: 'observations', table: source});
  const sourceBuffers = dataset.value.timeColumns.timestamp.timestamps.data.map(
    chunk => chunk.buffer
  );
  const sourceReads = sourceBuffers.map(buffer => vi.spyOn(buffer, 'readAsync'));
  const positions = new GPUVector({
    type: 'data',
    name: 'timestamp-points',
    format: 'float32x2',
    ownsData: true,
    data: [2, 4].map(
      length =>
        new GPUData({
          format: 'float32x2',
          length,
          ownsBuffer: true,
          buffer: device.createBuffer({
            data: new Float32Array(length * 2),
            usage: Buffer.STORAGE | Buffer.COPY_DST
          })
        })
    )
  });
  const time = new TimeRuntime({session, datasetId: dataset.id, column: 'timestamp'});
  const errors: string[] = [];
  const provider = new ResidentTableBinding({
    device,
    session,
    datasetId: dataset.id,
    positions,
    layerId: 'points',
    fields: {
      value: {dimension: 'value', kind: 'range', domain: [10, 60]},
      timestamp: {
        dimension: 'timestamp',
        kind: 'time',
        domain: [epochMilliseconds, epochMilliseconds + 3],
        validateRange: time.validateRange
      }
    },
    externalRanges: {timestamp: time.externalRange},
    onUpdate() {},
    onUnsupported: reason => errors.push(reason)
  });
  const explorer = session.openExplorer(dataset.id, {
    workingMemoryBudgetBytes: 8 * 1024 * 1024,
    dimensions: [
      {id: 'value', kind: 'range', column: 'value'},
      time.externalRange.displayDimension,
      provider.externalDimension
    ]
  });
  const table = installKeplerComputeTableBridge(
    {
      id: dataset.id,
      fields: [
        {name: 'value', type: 'real'},
        {name: 'timestamp', type: 'timestamp'}
      ],
      allIndexes: [0, 1, 2, 3, 4, 5],
      filteredIndex: [],
      filteredIndexForDomain: [],
      filteredIndexByLayer: {},
      filterTable(_filters: readonly KeplerComputeFilter[], _layers: readonly unknown[]) {
        return this;
      },
      getColumnLayerDomain(_field: {name: string}, _scale: string): readonly unknown[] | null {
        return null;
      }
    },
    provider.bind(explorer)
  );
  const dynamic: KeplerComputeFilter = {
    id: 'dynamic-time',
    dataId: [dataset.id],
    name: ['timestamp'],
    type: 'timeRange',
    fixedDomain: false,
    value: [epochMilliseconds, epochMilliseconds + 3]
  };
  const fixed: KeplerComputeFilter = {
    ...dynamic,
    id: 'fixed-time',
    fixedDomain: true,
    value: [epochMilliseconds + 1, epochMilliseconds + 2]
  };
  async function execute(filters: readonly KeplerComputeFilter[]): Promise<number[]> {
    table.filterTable(filters, [{id: 'points', type: 'point', config: {dataId: dataset.id}}]);
    expect(table.lumaCompute.status).not.toBe('unsupported');
    const encoder = device.createCommandEncoder();
    explorer.encode(encoder);
    device.submit(encoder.finish());
    expect(await provider.readSummary()).toBe(true);
    const page = await explorer.readPage(0, 6, ['value']);
    return Array.from(page!.sourceRowIndices);
  }
  try {
    expect(time.metadata.unit).toBe('microsecond');
    expect(time.metadata.timeZone).toBe('America/New_York');
    expect(time.domainMilliseconds).toEqual([epochMilliseconds, epochMilliseconds + 3]);
    expect(time.externalRange.displayDimension.mask.data.map(chunk => chunk.length)).toEqual([
      2, 4
    ]);
    expect(await execute([dynamic, fixed])).toEqual([2, 3]);
    expect(table.getColumnLayerDomain({name: 'value'}, 'linear')).toEqual([10, 50]);
    const status = await time.readStatus();
    expect(status.display).toMatchObject({complete: true, selectedCount: 2});
    expect(status.domain).toMatchObject({complete: true, selectedCount: 5});
    expect(time.readbackByteLength).toBe(16);

    // Adjacent source microseconds cannot collapse onto the same absolute Float32 epoch.
    expect(await execute([{...fixed, value: [epochMilliseconds, epochMilliseconds]}])).toEqual([0]);
    expect(
      await execute([
        dynamic,
        fixed,
        {...fixed, id: 'disjoint', value: [epochMilliseconds + 3, epochMilliseconds + 3]}
      ])
    ).toEqual([]);
    expect(table.getColumnLayerDomain({name: 'value'}, 'linear')).toEqual([10, 50]);

    // Removing all time filters bypasses both masks, including their null rejection policy.
    expect(await execute([])).toEqual([0, 1, 2, 3, 4, 5]);
    const revision = table.lumaCompute.revision;
    table.filterTable([{...fixed, value: [epochMilliseconds + 0.5, epochMilliseconds + 1]}], []);
    expect(table.lumaCompute.status).toBe('unsupported');
    expect(table.lumaCompute.revision).toBe(revision);
    expect(errors).toHaveLength(1);
    expect(errors[0]).toMatch(/integral milliseconds/);
    for (const read of sourceReads) expect(read).not.toHaveBeenCalled();
    expect(dataset.value.timeColumns.timestamp.timestamps.data.map(chunk => chunk.buffer)).toEqual(
      sourceBuffers
    );
  } finally {
    sourceReads.forEach(read => read.mockRestore());
    const fence = device.createFence();
    try {
      await Promise.all([
        explorer.close(fence.signaled),
        provider.destroy(fence.signaled),
        time.destroy(fence.signaled)
      ]);
      await session.destroy(fence.signaled);
      positions.destroy();
    } finally {
      fence.destroy();
    }
  }
});

function makeBatch(timestamps: readonly (bigint | null)[], values: number[]): arrow.Table {
  const bitmap = new Uint8Array(Math.ceil(timestamps.length / 8));
  const words = new BigInt64Array(timestamps.length);
  let nullCount = 0;
  for (const [index, timestamp] of timestamps.entries()) {
    if (timestamp === null) nullCount++;
    else {
      words[index] = timestamp;
      bitmap[index >>> 3] |= 1 << (index % 8);
    }
  }
  return new arrow.Table({
    value: arrow.makeVector(new Float32Array(values)),
    timestamp: arrow.makeVector(
      arrow.makeData({
        type: new arrow.TimestampMicrosecond('America/New_York'),
        data: words,
        length: words.length,
        nullCount,
        nullBitmap: bitmap
      })
    )
  });
}
