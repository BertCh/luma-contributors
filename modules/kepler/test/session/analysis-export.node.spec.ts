// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {expect, test, vi} from 'vitest';
import {NullDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession, type KeplerSessionExplorer} from '../../src/session/compute-session';
import type {AnalysisRuntime} from '../../../../examples/deck/kepler-compute/src/analysis-runtime';
import {exportSelectedAnalysis} from '../../../../examples/deck/kepler-compute/src/analysis-export';

const inspection = vi.hoisted(() => ({readPage: vi.fn(), destroy: vi.fn()}));
vi.mock('@luma-contributors/kepler', async importOriginal => ({
  ...(await importOriginal<object>()),
  GPUKeplerTableInspector: class {
    readPage = inspection.readPage;
    destroy = inspection.destroy;
  }
}));

function fixture() {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 2 * 1024 * 1024});
  const table = new arrow.Table({value: arrow.makeVector(new Float32Array([1, 2, 3, 4]))});
  const dataset = session.importArrowDataset({id: 'analysis-source', table});
  const state = {current: true};
  const inspector = {
    dataset,
    currentRevision: 4,
    isCurrent: true,
    maximumInspectionPageSize: 1,
    maximumInspectionReadbackBytes: 24,
    readSelectedCount: vi.fn(async () => ({
      revision: 4,
      datasetRevision: dataset.revision,
      selectedCount: 2
    }))
  };
  const snapshot = {
    revision: '2:4',
    columns: {score: dataset.value.columns.value},
    validity: {score: dataset.value.columns.value},
    isCurrent: () => state.current
  };
  const analysis = {
    withResultSnapshot: vi.fn(async read => read(snapshot)),
    serializeRecipe: () =>
      JSON.stringify({
        population: 'all',
        operation: {kind: 'standardize', method: 'z-score', column: 'value'}
      })
  };
  inspection.destroy.mockReset().mockResolvedValue(undefined);
  inspection.readPage.mockReset().mockImplementation(async ({offset, columns}) => {
    expect(columns).toEqual(['score']);
    return {
      revision: 4,
      datasetRevision: dataset.revision,
      totalCount: 2,
      sourceRowIndices: new Uint32Array([[1], [3]][offset]),
      columns: {score: new Float32Array(offset ? [99] : [-1])},
      validity: {score: new Uint32Array(offset ? [0] : [1])}
    };
  });
  return {
    device,
    session,
    state,
    inspector,
    explorer: inspector as unknown as KeplerSessionExplorer,
    analysis: analysis as unknown as AnalysisRuntime,
    cleanup: async () => {
      await session.destroy(Promise.resolve());
      device.destroy();
    }
  };
}
const options = {
  mode: 'all-selected' as const,
  maximumRows: 4,
  maximumReadbackBytes: 44,
  maximumCpuBytes: 1024,
  maximumOutputBytes: 65536,
  format: 'arrow' as const
};

test('analysis export pages computed values with original identities and nullable typed metadata', async () => {
  const context = fixture();
  try {
    const baseline = context.session.registry.memoryUsage.totalBytes;
    const result = await exportSelectedAnalysis(
      context.session,
      context.explorer,
      context.analysis,
      options
    );
    const table = arrow.tableFromIPC(result.bytes);
    expect([...table.getChild('score')!]).toEqual([-1, null]);
    expect([...table.getChild(result.sourceRowColumn)!]).toEqual([1, 3]);
    expect(JSON.parse(table.schema.metadata.get('luma:analysis-export')!).recipe.population).toBe(
      'all'
    );
    expect(result.readbackBytes).toBe(44);
    const joined = await exportSelectedAnalysis(
      context.session,
      context.explorer,
      context.analysis,
      {...options, includeOriginalSource: true}
    );
    const joinedTable = arrow.tableFromIPC(joined.bytes);
    expect([...joinedTable.getChild('value')!]).toEqual([2, 4]);
    expect([...joinedTable.getChild('score')!]).toEqual([-1, null]);
    expect(joined.readbackBytes).toBe(result.readbackBytes);
    joined.destroy();
    expect(inspection.readPage).toHaveBeenCalledTimes(4);
    expect(inspection.destroy).toHaveBeenCalledTimes(2);
    expect(context.session.registry.memoryUsage.totalBytes).toBeGreaterThan(baseline);
    context.state.current = false;
    expect(result.assertCurrent).toThrow(/stale/);
    result.destroy();
    result.destroy();
    expect(result.bytes.byteLength).toBe(0);
    expect(context.session.registry.memoryUsage.totalBytes).toBe(baseline);
  } finally {
    await context.cleanup();
  }
});

test('analysis export rejects budgets before pages and cleans up canceled or stale reads', async () => {
  const context = fixture();
  try {
    const baseline = context.session.registry.memoryUsage.totalBytes;
    for (const reduced of [{maximumRows: 1}, {maximumReadbackBytes: 43}, {maximumCpuBytes: 1}])
      await expect(
        exportSelectedAnalysis(context.session, context.explorer, context.analysis, {
          ...options,
          ...reduced
        })
      ).rejects.toThrow(/budget/);
    expect(inspection.readPage).not.toHaveBeenCalled();
    await expect(
      exportSelectedAnalysis(context.session, context.explorer, context.analysis, {
        ...options,
        maximumOutputBytes: 16
      })
    ).rejects.toThrow(/byte budget/);
    expect(context.session.registry.memoryUsage.totalBytes).toBe(baseline);
    const controller = new AbortController();
    inspection.readPage.mockImplementationOnce(async () => {
      controller.abort();
      return null;
    });
    await expect(
      exportSelectedAnalysis(context.session, context.explorer, context.analysis, {
        ...options,
        signal: controller.signal
      })
    ).rejects.toThrow();
    inspection.readPage.mockImplementationOnce(async () => {
      context.state.current = false;
      return null;
    });
    await expect(
      exportSelectedAnalysis(context.session, context.explorer, context.analysis, options)
    ).rejects.toThrow(/stale/);
    expect(context.session.registry.memoryUsage.totalBytes).toBe(baseline);
    expect(inspection.destroy).toHaveBeenCalledTimes(3);
  } finally {
    await context.cleanup();
  }
});
