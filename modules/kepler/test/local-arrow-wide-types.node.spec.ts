// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {expect, test, vi} from 'vitest';
import {createRequire} from 'node:module';
import {NullDevice} from '@luma.gl/core/test-utils';
import {GPUKeplerTableInspector, KeplerComputeSession, type KeplerSessionExplorer} from '../src';
import {
  loadLocalSource,
  makeSourceCandidateFromTable
} from '../../../examples/deck/kepler-compute/src/remote-source-loader';
import {previewArrowSource} from '../../../examples/deck/kepler-compute/src/source-preview';
import {SourcePreviewPanel} from '../../../examples/deck/kepler-compute/src/source-preview-panel';
import {PublishedDatasetRuntime} from '../../../examples/deck/kepler-compute/src/published-dataset-runtime';
import {getOriginalFieldType} from '../../../examples/deck/kepler-compute/src/original-column-values';
import {createColumnStatisticsReader} from '../../../examples/deck/kepler-compute/src/resident-column-statistics';
import type {ResidentTableProvider} from '../../../examples/deck/kepler-compute/src/resident-data-table';
import {
  makeRelativeFloat32Column,
  makeWideColumnStats,
  ResidentTableWideStatistics
} from '../../../examples/deck/kepler-compute/src/resident-table-wide-statistics';
import {
  readWideWords,
  ResidentTableGPUSorter
} from '../../../examples/deck/kepler-compute/src/resident-table-gpu-sorter';
import {
  classifyTableSortType,
  getPublishedTableSortMemoryBudgetBytes
} from '../../../examples/deck/kepler-compute/src/resident-table-sort-keys';
import {
  chooseColumnStatisticsRoute,
  chooseTableSortRoute
} from '../../../examples/deck/kepler-compute/src/resident-table-population-routes';

test.each([
  100_000, 300_000
])('all six wide-statistics fields at %i rows are selected by default and retained through IPC publication', async rowCount => {
  const names = ['longitude', 'latitude', 'i64', 'stamp', 'f64', 'kind'];
  const source = new arrow.Table({
    longitude: arrow.makeVector(new Float32Array(rowCount).fill(-74)),
    latitude: arrow.makeVector(new Float32Array(rowCount).fill(40)),
    i64: arrow.makeVector(
      BigInt64Array.from(
        {length: rowCount},
        (_, row) => 10_000_000_000n + BigInt((row * 7919) % rowCount)
      )
    ),
    stamp: arrow.vectorFromArray(
      Array.from(
        {length: rowCount},
        (_, row) => 1_600_000_000_000 + ((row * 104729) % rowCount) * 1000
      ),
      new arrow.TimestampMillisecond()
    ),
    f64: arrow.vectorFromArray(
      Array.from({length: rowCount}, (_, row) =>
        row % 1000 === 0 ? null : Math.sin(row * 0.001) * 500 + row * 1e-4
      ),
      new arrow.Float64()
    ),
    kind: arrow.vectorFromArray(Array.from({length: rowCount}, (_, row) => `label-${row % 1000}`))
  });
  const file = new File([new Uint8Array(arrow.tableToIPC(source, 'stream')).buffer], 'wide.arrow');
  const candidate = await loadLocalSource(file, {preserveArrowTypes: true});
  expect(candidate.table.schema.fields.map(field => field.name)).toEqual(names);
  const normalization = {
    float64: {mode: 'round' as const, maximumAbsoluteError: 0.001, maximumRelativeError: 0.000001}
  };
  const preview = previewArrowSource(candidate.table, normalization);
  expect(preview.canUpload).toBe(true);
  expect(preview.fields.filter(field => field.selected).map(field => field.name)).toEqual(names);
  const hostRequire = createRequire(
    new URL('../../../examples/deck/kepler-compute/package.json', import.meta.url)
  );
  const {createElement} = hostRequire('react') as typeof import('react');
  const {renderToStaticMarkup} = hostRequire(
    'react-dom/server'
  ) as typeof import('react-dom/server');
  const markup = renderToStaticMarkup(
    createElement(SourcePreviewPanel, {
      source: candidate.table,
      label: 'wide',
      onPublish: async () => {}
    })
  );
  expect(markup.match(/type="checkbox" checked=""/g)).toHaveLength(6);
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 128 * 1024 * 1024});
  try {
    const dataset = session.importArrowDataset({id: 'wide', table: candidate.table, normalization});
    expect(dataset.value.source.schema.fields.map(field => field.name)).toEqual(names);
    expect(dataset.value.prepared.table.schema.fields.map(field => field.name)).toEqual(
      names.filter(name => name !== 'stamp')
    );
    expect(Object.keys(dataset.value.columns)).toEqual(names.filter(name => name !== 'stamp'));
    expect(Object.keys(dataset.value.timeColumns)).toEqual(['stamp']);
    expect(dataset.value.source.getChild('f64')!.nullCount).toBe(rowCount / 1000);
    expect(dataset.value.prepared.table.getChild('f64')!.nullCount).toBe(rowCount / 1000);
    expect(dataset.value.prepared.dictionaries['kind'].values).toHaveLength(1000);
    // The published table's default prepared view omits time words, not nullable Float64 fields.
    const explorer = {
      dataset,
      currentRevision: 1,
      isCurrent: true,
      maximumInspectionPageSize: 50,
      allowIncompletePopulation: false,
      close: vi.fn(async () => {})
    } as unknown as KeplerSessionExplorer;
    const openExplorer = vi.spyOn(session, 'openExplorer').mockReturnValue(explorer);
    const consumer = new PublishedDatasetRuntime({
      session,
      datasetId: dataset.id,
      explorer: {dimensions: [], workingMemoryBudgetBytes: 16 * 1024 * 1024},
      sortMemoryBudgetBytes: getPublishedTableSortMemoryBudgetBytes(rowCount),
      requestUpdate() {}
    });
    try {
      expect(consumer.table.fields.map(field => field.name)).toEqual(
        names.filter(name => name !== 'stamp')
      );
      expect(consumer.table.originalFields!.map(field => field.name)).toEqual(names);
      if (rowCount === 300_000) {
        const profile = {
          populationRows: rowCount,
          count: rowCount - rowCount / 1000,
          nullCount: rowCount / 1000,
          minimum: -500,
          maximum: 530,
          sum: 0,
          mean: 16.7,
          standardDeviation: 354,
          sampleVariance: 354 ** 2,
          distinctEstimate: rowCount,
          histogram: new Uint32Array([rowCount])
        };
        const readStatistics = vi
          .spyOn(ResidentTableWideStatistics.prototype, 'read')
          .mockResolvedValue({
            fieldType: 'real',
            rowCount,
            nullCount: rowCount / 1000,
            stats: makeWideColumnStats('real', profile),
            routeReason: 'Original-source wide profile'
          });
        const sortRows = vi
          .spyOn(ResidentTableGPUSorter.prototype, 'readSortedRows')
          .mockResolvedValue({
            rows: new Uint32Array([1, 3]),
            totalCount: rowCount
          });
        const readRows = vi.spyOn(GPUKeplerTableInspector.prototype, 'readRows').mockResolvedValue({
          revision: 1,
          datasetRevision: dataset.revision,
          totalCount: rowCount,
          sourceRowIndices: new Uint32Array([1, 3]),
          columns: {},
          validity: {}
        });
        try {
          expect(
            await consumer.table.readColumnStatistics!({column: 'f64', population: 'all'})
          ).toMatchObject({source: 'gpu', rowCount, nullCount: rowCount / 1000});
          expect(readStatistics).toHaveBeenCalledWith({column: 'f64', population: 'all'});
          const page = await consumer.table.readPage({
            view: 'original',
            population: 'all',
            columns: ['i64'],
            limit: 2,
            offset: 0,
            sort: {column: 'i64', direction: 'ascending'}
          });
          expect(page!.columns['i64']).toEqual([
            String(source.getChild('i64')!.get(1)),
            String(source.getChild('i64')!.get(3))
          ]);
          expect(sortRows).toHaveBeenCalledWith({
            column: 'i64',
            direction: 'ascending',
            population: 'all',
            offset: 0,
            limit: 2
          });
        } finally {
          readStatistics.mockRestore();
          sortRows.mockRestore();
          readRows.mockRestore();
        }
      }
    } finally {
      await consumer.destroy();
      expect(explorer.close).toHaveBeenCalledOnce();
      openExplorer.mockRestore();
    }
  } finally {
    await session.destroy(Promise.resolve());
    device.destroy();
  }
});

test('local IPC import keeps decimal labels by default and preserves numeric words only by choice', async () => {
  const source = new arrow.Table({
    i64: arrow.vectorFromArray([10_000_000_001n, null, 10_000_000_000n], new arrow.Int64()),
    longitude: arrow.makeVector(new Float32Array([-74, -73, -72])),
    latitude: arrow.makeVector(new Float32Array([40, 41, 42]))
  });
  const batches = source.slice(0, 1).concat(source.slice(1));
  const file = new File([new Uint8Array(arrow.tableToIPC(batches, 'stream')).buffer], 'wide.arrow');
  const defaultCandidate = await loadLocalSource(file);
  expect(String(defaultCandidate.table.getChild('i64')!.type)).toBe('Utf8');
  expect(defaultCandidate.table.getChild('i64')!.get(0)).toBe('10000000001');
  const candidate = await loadLocalSource(file, {preserveArrowTypes: true});
  expect(String(candidate.table.getChild('i64')!.type)).toBe('Int64');
  expect([...candidate.table.getChild('i64')!]).toEqual([10_000_000_001n, null, 10_000_000_000n]);
  expect(candidate.table.batches.map(batch => batch.numRows)).toEqual([1, 2]);
  expect(candidate.suggestedCoordinates).toEqual({longitude: 'longitude', latitude: 'latitude'});
  expect(previewArrowSource(candidate.table).canUpload).toBe(false);
  expect(previewArrowSource(candidate.table, {float64: {mode: 'exact'}}).canUpload).toBe(false);
  const preview = previewArrowSource(candidate.table, {
    float64: {mode: 'round', maximumAbsoluteError: 0.001, maximumRelativeError: 0.000001}
  });
  expect(preview.canUpload).toBe(true);
  expect(preview.fields.find(field => field.name === 'i64')).toMatchObject({
    sourceType: 'Int64',
    kind: 'scalar',
    targetType: 'Float32'
  });
});

test('preserved IPC enforces original-value precision budgets and leaves other format routing unchanged', () => {
  const source = new arrow.Table({
    i64: arrow.vectorFromArray([9_007_199_254_740_993n], new arrow.Int64())
  });
  const candidate = makeSourceCandidateFromTable(source, 'unsafe.arrow', 'arrow', {
    preserveArrowTypes: true
  });
  expect(previewArrowSource(candidate.table).canUpload).toBe(false);
  expect(previewArrowSource(candidate.table, {float64: {mode: 'exact'}}).canUpload).toBe(false);
  expect(
    previewArrowSource(candidate.table, {
      float64: {mode: 'round', maximumAbsoluteError: 0.5, maximumRelativeError: 0}
    }).canUpload
  ).toBe(false);
  const preview = previewArrowSource(candidate.table, {
    float64: {mode: 'round', maximumAbsoluteError: 1, maximumRelativeError: 0}
  });
  expect(preview.canUpload).toBe(true);
  expect(preview.fields).toMatchObject([
    {
      name: 'i64',
      sourceType: 'Int64',
      targetType: 'Float32',
      maximumAbsoluteError: 1,
      maximumRelativeError: 1.1102230246251564e-16
    }
  ]);
  expect(
    String(
      makeSourceCandidateFromTable(source, 'default.parquet', 'parquet', {
        preserveArrowTypes: true
      }).table.getChild('i64')!.type
    )
  ).toBe('Utf8');
  expect(readWideWords(candidate.table.getChild('i64')!, 'wide-int64', 1)).toMatchObject({
    lo: new Uint32Array([1]),
    hi: new Uint32Array([0x80200000]),
    valid: new Uint32Array([1])
  });
});

test.each([
  100_000, 300_000
])('preserved %i-row Int64 uses original numeric statistics and exact sort routing', rowCount => {
  const values = BigInt64Array.from(
    {length: rowCount},
    (_, row) => 10_000_000_000n + BigInt((row * 7919) % rowCount)
  );
  const candidate = makeSourceCandidateFromTable(
    new arrow.Table({i64: arrow.makeVector(values)}),
    'wide.arrow',
    'arrow',
    {preserveArrowTypes: true}
  );
  const vector = candidate.table.getChild('i64')!;
  expect(getOriginalFieldType(vector.type)).toBe('integer');
  const relative = makeRelativeFloat32Column(vector, rowCount);
  expect(relative.base).toBe(10_000_000_000);
  expect(relative.validCount).toBe(rowCount);
  expect(relative.values.reduce((sum, value) => sum + value, 0) / rowCount + relative.base).toBe(
    10_000_000_000 + (rowCount - 1) / 2
  );
  expect(
    chooseColumnStatisticsRoute({
      gpuSupported: false,
      wideSupported: true,
      populationRows: rowCount
    }).route
  ).toBe(rowCount === 100_000 ? 'cpu' : 'gpu');
  const sorting = chooseTableSortRoute({
    orderKind: classifyTableSortType(vector.type),
    hasPreparedGPUColumn: true,
    hasRetainedWideData: true,
    populationRows: rowCount
  });
  expect(sorting.route).toBe(rowCount === 100_000 ? 'cpu' : 'gpu');
  if (rowCount === 300_000) expect(sorting.reason).toContain('hi/lo 32-bit words');
});

test('large original numeric columns use the wide profile even when their prepared copy supports GPU statistics', async () => {
  for (const type of [new arrow.Int64(), new arrow.Float64()]) {
    const source = new arrow.Table({
      amount: arrow.vectorFromArray(
        type instanceof arrow.Int64 ? [10_000_000_000n] : [10_000_000_000],
        type
      )
    });
    const preparedRead = vi.fn();
    const readPopulationRows = vi.fn();
    const stats = makeWideColumnStats('integer', {
      populationRows: 300_000,
      count: 300_000,
      nullCount: 0,
      minimum: 10_000_000_000,
      maximum: 10_000_299_999,
      sum: 0,
      mean: 10_000_149_999.5,
      standardDeviation: 1,
      sampleVariance: 1,
      distinctEstimate: 300_000,
      histogram: new Uint32Array([300_000])
    });
    const wideRead = vi.fn(async () => ({
      fieldType: getOriginalFieldType(type),
      rowCount: 300_000,
      nullCount: 0,
      stats,
      routeReason: 'Relative to the original minimum'
    }));
    const provider: ResidentTableProvider = {
      datasetId: 'wide',
      sourceRevision: 1,
      maximumPageSize: 50,
      fields: [{name: 'amount', label: 'amount', sortable: true}],
      getRevision: () => '1',
      subscribe: () => () => {},
      readPage: vi.fn()
    };
    const reader = createColumnStatisticsReader({
      source,
      provider,
      isCurrent: () => true,
      readPopulationRows,
      getPopulationRows: () => 300_000,
      gpu: {plan: () => ({supported: true}), read: preparedRead},
      wide: {
        plan: () => ({supported: true, kind: 'wide-numeric', reason: 'Wide source'}),
        read: wideRead
      }
    });
    expect(await reader.readColumnStatistics({column: 'amount', population: 'all'})).toMatchObject({
      source: 'gpu',
      stats,
      routeReason: 'Relative to the original minimum'
    });
    expect(wideRead).toHaveBeenCalledOnce();
    expect(preparedRead).not.toHaveBeenCalled();
    expect(readPopulationRows).not.toHaveBeenCalled();
  }
});
