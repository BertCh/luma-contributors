// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import * as arrow from 'apache-arrow';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession} from '../src/session/compute-session';
import {
  DuckDBRuntime,
  DEFAULT_DUCKDB_SQL
} from '../../../examples/deck/kepler-compute/src/duckdb-runtime';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';

it('real worker SQL returns Arrow before an explicit resident GPU import', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  session.importArrowDataset({
    id: 'observations',
    table: arrow.tableFromArrays({
      category: new Uint32Array([0, 1, 0, 1]),
      value: new Float32Array([1, 10, 3, 20])
    })
  });
  const runtime = new DuckDBRuntime({session, datasetId: 'observations'});
  try {
    expect(runtime.status).toBe('disabled');
    await runtime.initialize();
    await runtime.registerSource();
    await runtime.runSql(DEFAULT_DUCKDB_SQL);
    expect(runtime.previewRows()).toEqual([
      {category: 0, value: 2, count: 2},
      {category: 1, value: 15, count: 2}
    ]);
    expect(runtime.statistics.uploads).toBe(0);
    expect(runtime.statistics.retainedResultBytes).toBeGreaterThan(0);
    const result = runtime.importResult();
    expect(result.rowCount).toBe(2);
    expect(runtime.statistics.uploadedBytes).toBeGreaterThan(0);
    expect(runtime.statistics.sourceArrowBytes).toBeGreaterThan(0);
    expect(runtime.statistics.resultArrowBytes).toBeGreaterThan(0);
    expect(runtime.statistics.retainedResultBytes).toBe(0);
    const values = result.value.columns['value'];
    expect(values.format).toBe('float32');
    const chunk = values.data[0];
    const bytes = await chunk.buffer.readAsync(chunk.byteOffset, 8);
    expect(Array.from(new Float32Array(bytes.buffer, bytes.byteOffset, 2))).toEqual([2, 15]);
    session.importArrowDataset({
      id: 'observations',
      table: arrow.tableFromArrays({value: new Float32Array([8])})
    });
    await expect(runtime.runSql(DEFAULT_DUCKDB_SQL)).rejects.toThrow('stale');
    expect(() => runtime.importResult()).toThrow('stale');
  } finally {
    await runtime.destroy();
    const fence = device.createFence();
    try {
      await session.destroy(fence.signaled);
    } finally {
      fence.destroy();
    }
  }
}, 60000);
