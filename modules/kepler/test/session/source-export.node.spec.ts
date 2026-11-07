// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {expect, test, vi} from 'vitest';
import {NullDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession, type KeplerSessionExplorer} from '../../src/session/compute-session';
import {exportSelectedSource} from '../../../../examples/deck/kepler-compute/src/source-export';

function fixture() {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 1024 * 1024});
  const source = new arrow.Table({
    amount: arrow.vectorFromArray([0.1, 0.123456789012345, 0.3, 0.4], new arrow.Float64()),
    geometry: arrow.vectorFromArray([{x: 0}, {x: 1}, {x: 2}, {x: 3}])
  });
  const dataset = session.importArrowDataset({
    id: 'source',
    table: source,
    normalization: {
      columns: ['amount'],
      float64: {mode: 'round', maximumAbsoluteError: 0.000001, maximumRelativeError: 0}
    }
  });
  const inspector = {
    dataset,
    currentRevision: 4,
    isCurrent: true,
    maximumInspectionPageSize: 2,
    maximumInspectionReadbackBytes: 12,
    readSelectedCount: vi.fn(async () => ({
      revision: 4,
      datasetRevision: dataset.revision,
      selectedCount: 2
    })),
    readPage: vi.fn(async (offset: number, limit: number, columns: readonly string[]) => {
      expect(columns).toEqual([]);
      return {
        revision: 4,
        datasetRevision: dataset.revision,
        sourceRowIndices: new Uint32Array([1, 3].slice(offset, offset + limit)),
        columns: {},
        validity: {}
      };
    })
  };
  return {
    device,
    session,
    source,
    inspector,
    explorer: inspector as unknown as KeplerSessionExplorer,
    cleanup: async () => {
      await session.destroy(Promise.resolve());
      device.destroy();
    }
  };
}

const options = {
  mode: 'all-selected' as const,
  maximumRows: 4,
  maximumReadbackBytes: 16,
  maximumCpuBytes: 64,
  maximumOutputBytes: 65536,
  format: 'arrow' as const
};

test('source export reads only GPU identities and owns a bounded serialization reservation', async () => {
  const context = fixture();
  try {
    const baseline = context.session.registry.memoryUsage.totalBytes;
    const result = await exportSelectedSource(context.session, context.explorer, options);
    const table = arrow.tableFromIPC(result.bytes);
    expect([...table.getChild('amount')!]).toEqual([0.123456789012345, 0.4]);
    expect([...table.getChild(result.sourceRowColumn)!]).toEqual([1, 3]);
    expect(table.getChild('geometry')!.get(0).x).toBe(1);
    expect(result.readbackBytes).toBe(16);
    expect(result.rowCount).toBe(2);
    expect(context.inspector.readPage).toHaveBeenCalledWith(0, 2, []);
    expect(context.session.registry.memoryUsage.totalBytes).toBe(
      baseline + options.maximumOutputBytes * 2
    );
    result.destroy();
    result.destroy();
    expect(result.bytes.byteLength).toBe(0);
    expect(context.session.registry.memoryUsage.totalBytes).toBe(baseline);
  } finally {
    await context.cleanup();
  }
});

test('failed or canceled source exports release reservations and do not publish partial output', async () => {
  const context = fixture();
  try {
    const baseline = context.session.registry.memoryUsage.totalBytes;
    await expect(
      exportSelectedSource(context.session, context.explorer, {...options, maximumOutputBytes: 64})
    ).rejects.toThrow(/byte budget/);
    expect(context.session.registry.memoryUsage.totalBytes).toBe(baseline);
    const controller = new AbortController();
    controller.abort();
    await expect(
      exportSelectedSource(context.session, context.explorer, {
        ...options,
        signal: controller.signal
      })
    ).rejects.toThrow();
    expect(context.session.registry.memoryUsage.totalBytes).toBe(baseline);
    context.inspector.readSelectedCount.mockImplementation(async () => {
      context.inspector.currentRevision++;
      return {revision: 4, datasetRevision: context.inspector.dataset.revision, selectedCount: 2};
    });
    await expect(exportSelectedSource(context.session, context.explorer, options)).rejects.toThrow(
      /stale/
    );
    expect(context.session.registry.memoryUsage.totalBytes).toBe(baseline);
  } finally {
    await context.cleanup();
  }
});
