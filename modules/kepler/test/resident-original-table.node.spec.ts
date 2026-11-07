// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {expect, test, vi} from 'vitest';
import {
  formatOriginalTableValue,
  withOriginalSourceTable
} from '../../../examples/deck/kepler-compute/src/resident-table-provider';
import type {
  ResidentTablePage,
  ResidentTablePageRequest,
  ResidentTableProvider
} from '../../../examples/deck/kepler-compute/src/resident-data-table';

function fixture() {
  const timestamp = arrow.makeVector(
    arrow.makeData({
      type: new arrow.TimestampNanosecond('UTC'),
      length: 4,
      data: new BigInt64Array([
        -1n,
        1767225600000000001n,
        1767225600000000002n,
        1767225600000000003n
      ])
    })
  );
  const original = new arrow.Table({
    time: timestamp,
    amount: arrow.vectorFromArray(
      [0.1, 0.123456789012345, null, 0.3333333333333333],
      new arrow.Float64()
    ),
    value: arrow.makeVector(new Float32Array([1, 2, 3, 4])),
    geometry: arrow.vectorFromArray(
      [
        '{"type":"Point","coordinates":[1,2]}',
        null,
        '{"type":"Point","coordinates":[3,4]}',
        '{"type":"Point","coordinates":[5,6]}'
      ],
      new arrow.Utf8()
    )
  });
  const source = original.slice(0, 2).concat(original.slice(2));
  let revision = 'selection-3';
  let current = true;
  let derived = false;
  const readPage = vi.fn(
    async (request: ResidentTablePageRequest): Promise<ResidentTablePage> => ({
      revision: 3,
      datasetRevision: 2,
      totalCount: 2,
      sourceRowIndices: new Uint32Array(
        [3, 1].slice(request.offset, request.offset + request.limit)
      ),
      columns: Object.fromEntries(
        request.columns.map(name => [
          name,
          new Float32Array(name === 'analysis' ? [30, 10] : [0.33, 0.12])
        ])
      ),
      validity: {}
    })
  );
  const base: ResidentTableProvider = {
    datasetId: 'source',
    sourceRevision: 2,
    maximumPageSize: 2,
    get fields() {
      return [
        {name: 'amount', sortable: true},
        {name: 'value', sortable: true},
        ...(derived ? [{name: 'analysis', sortable: true}] : [])
      ];
    },
    getRevision: () => revision,
    subscribe: () => () => {},
    readPage
  };
  const provider = withOriginalSourceTable(base, source, () => current);
  return {
    provider,
    source,
    readPage,
    invalidate: () => {
      revision = 'selection-4';
    },
    stale: () => {
      current = false;
    },
    derive: () => {
      derived = true;
    }
  };
}

test('original source-only table reads bounded identities without a synthetic prepared column', async () => {
  const source = new arrow.Table({
    geometry: arrow.vectorFromArray(['first', null, 'third'], new arrow.Utf8())
  });
  const readPage = vi.fn(
    async (): Promise<ResidentTablePage> => ({
      revision: 1,
      datasetRevision: 2,
      totalCount: 2,
      sourceRowIndices: new Uint32Array([0, 2]),
      columns: {},
      validity: {}
    })
  );
  const provider = withOriginalSourceTable(
    {
      datasetId: 'geometry',
      sourceRevision: 2,
      maximumPageSize: 2,
      fields: [],
      getRevision: () => '2:1',
      subscribe: () => () => {},
      readPage
    },
    source,
    () => true
  );
  const page = await provider.readPage({
    view: 'original',
    population: 'selected',
    offset: 0,
    limit: 2,
    columns: ['geometry']
  });
  expect(page?.columns.geometry).toEqual(['first', 'third']);
  expect(readPage).toHaveBeenCalledWith(expect.objectContaining({columns: [], view: 'prepared'}));
  expect(provider.fields).toEqual([]);
});

test('original source pages retain exact Float64, timestamp words, geometry and source IDs without scanning rows', async () => {
  const context = fixture();
  vi.spyOn(context.source, 'toArray').mockImplementation(() => {
    throw new Error('bulk source scan');
  });
  vi.spyOn(context.source, 'get').mockImplementation(() => {
    throw new Error('whole row scan');
  });
  const getChild = context.source.getChild.bind(context.source);
  const vectors = new Map(
    context.source.schema.fields.map(field => [field.name, getChild(field.name)!])
  );
  const valueReads = new Map([...vectors].map(([name, vector]) => [name, vi.spyOn(vector, 'get')]));
  for (const vector of vectors.values())
    vi.spyOn(vector, 'toArray').mockImplementation(() => {
      throw new Error('bulk vector scan');
    });
  vi.spyOn(context.source, 'getChild').mockImplementation(name => vectors.get(name) ?? null);
  for (let iteration = 0; iteration < 3; iteration++) {
    const page = await context.provider.readPage({
      view: 'original',
      population: 'selected',
      offset: 0,
      limit: 2,
      columns: ['time', 'amount', 'geometry']
    });
    expect(Array.from(page!.sourceRowIndices)).toEqual([3, 1]);
    expect(page!.columns['amount']).toEqual(['0.3333333333333333', '0.123456789012345']);
    expect(page!.columns['time']).toEqual([
      '1767225600000000003 ns since Unix epoch (UTC)',
      '1767225600000000001 ns since Unix epoch (UTC)'
    ]);
    expect(page!.columns['geometry']).toEqual(['{"type":"Point","coordinates":[5,6]}', '']);
    expect(Array.from(page!.validity['geometry'])).toEqual([1, 0]);
    expect(context.readPage.mock.calls[iteration][0].columns).toEqual([]);
  }
  expect(valueReads.get('amount')!.mock.calls.map(([row]) => row)).toEqual([3, 1, 3, 1, 3, 1]);
  expect(valueReads.get('geometry')).not.toHaveBeenCalled();
  expect(valueReads.get('time')).not.toHaveBeenCalled();
  expect(context.source.batches.map(batch => batch.numRows)).toEqual([2, 2]);
});

test('original rich columns reject normalized sorting, while exact Float32 and derived keys use bounded GPU pages', async () => {
  const context = fixture();
  for (const column of ['time', 'amount', 'geometry']) {
    expect(context.provider.originalFields!.find(field => field.name === column)?.sortable).toBe(
      false
    );
    await expect(
      context.provider.readPage({
        view: 'original',
        population: 'selected',
        offset: 0,
        limit: 2,
        columns: [column],
        sort: {column, direction: 'ascending'}
      })
    ).rejects.toThrow(/original-value sorting/);
  }
  expect(context.readPage).not.toHaveBeenCalled();
  context.derive();
  expect(context.provider.originalFields!.at(-1)?.name).toBe('analysis');
  const page = await context.provider.readPage({
    view: 'original',
    population: 'selected',
    offset: 0,
    limit: 2,
    columns: ['amount', 'analysis'],
    sort: {column: 'value', direction: 'descending'}
  });
  expect(Array.from(page!.columns['analysis'])).toEqual([30, 10]);
  expect(context.readPage.mock.calls[0][0].columns).toEqual(['analysis']);
  expect(context.readPage.mock.calls[0][0].sort).toEqual({
    column: 'value',
    direction: 'descending'
  });
});

test('original page publication rejects stale source or selection and enforces bounds before any read', async () => {
  const context = fixture();
  const request: ResidentTablePageRequest = {
    view: 'original',
    population: 'selected',
    offset: 0,
    limit: 2,
    columns: ['amount']
  };
  await expect(context.provider.readPage({...request, limit: 3})).rejects.toThrow(/bounded/);
  expect(context.readPage).not.toHaveBeenCalled();
  context.readPage.mockImplementationOnce(async () => {
    context.invalidate();
    return {
      revision: 3,
      datasetRevision: 2,
      totalCount: 1,
      sourceRowIndices: new Uint32Array([1]),
      columns: {},
      validity: {}
    };
  });
  expect(await context.provider.readPage(request)).toBeNull();
  context.stale();
  expect(await context.provider.readPage(request)).toBeNull();
  expect(context.readPage).toHaveBeenCalledOnce();
});

test('exact timestamp formatting respects sliced Arrow offsets, negative ticks and timezone metadata', () => {
  const source = fixture().source.getChild('time')!;
  expect(formatOriginalTableValue(source, 0)).toBe('-1 ns since Unix epoch (UTC)');
  expect(formatOriginalTableValue(source.slice(1, 4), 0)).toBe(
    '1767225600000000001 ns since Unix epoch (UTC)'
  );
  const timestamp = arrow.makeVector(
    arrow.makeData({
      type: new arrow.TimestampMicrosecond('America/New_York'),
      length: 1,
      data: new BigInt64Array([9007199254740993n])
    })
  );
  expect(formatOriginalTableValue(timestamp, 0)).toBe(
    '9007199254740993 us since Unix epoch (America/New_York)'
  );
});

test('original rich-cell formatting retains nested geometry and exact Int64 values', () => {
  const geometry = arrow.vectorFromArray([
    {
      type: 'LineString',
      coordinates: [
        [-73, 40],
        [-74, 41]
      ]
    }
  ]);
  expect(JSON.parse(formatOriginalTableValue(geometry, 0))).toEqual({
    type: 'LineString',
    coordinates: [
      [-73, 40],
      [-74, 41]
    ]
  });
  const integers = arrow.makeVector(new BigInt64Array([9007199254740993n]));
  expect(formatOriginalTableValue(integers, 0)).toBe('9007199254740993');
});

test('original preview never fully decodes a huge UTF8 cell or materializes nested vectors', () => {
  const text = arrow.vectorFromArray(['x'.repeat(100_000)], new arrow.Utf8());
  vi.spyOn(text, 'get').mockImplementation(() => {
    throw new Error('unbounded string decode');
  });
  const formatted = formatOriginalTableValue(text, 0);
  expect(formatted.length).toBeLessThanOrEqual(4096);
  expect(formatted).toMatch(/… \[preview\]$/);
  const nested = arrow.vectorFromArray([
    {coordinates: Array.from({length: 10000}, (_, index) => [index, index + 1])}
  ]);
  const toArray = vi.spyOn(arrow.Vector.prototype, 'toArray').mockImplementation(() => {
    throw new Error('bulk nested copy');
  });
  const toJSON = vi.spyOn(arrow.StructRow.prototype, 'toJSON').mockImplementation(() => {
    throw new Error('bulk object copy');
  });
  try {
    const rich = formatOriginalTableValue(nested, 0);
    expect(rich).toMatch(/… \[preview\]$/);
    expect(rich.length).toBeLessThanOrEqual(4096);
  } finally {
    toArray.mockRestore();
    toJSON.mockRestore();
  }
});

test('original and computed fields with the same name retain independent display and sorting', async () => {
  const source = new arrow.Table({
    analysis: arrow.makeVector(new Float32Array([111, 222])),
    analysis_analysis: arrow.makeVector(new Float32Array([333, 444]))
  });
  const readPage = vi.fn(async (_request: ResidentTablePageRequest) => ({
    revision: 1,
    datasetRevision: 1,
    totalCount: 2,
    sourceRowIndices: new Uint32Array([1, 0]),
    columns: {analysis: new Float32Array([2, 1])},
    validity: {analysis: new Uint32Array([1, 0])}
  }));
  const base: ResidentTableProvider = {
    datasetId: 'source',
    sourceRevision: 1,
    maximumPageSize: 2,
    fields: [{name: 'analysis', label: 'Analysis result', sortable: true, derived: true}],
    getRevision: () => '1',
    subscribe: () => () => {},
    readPage
  };
  const provider = withOriginalSourceTable(base, source, () => true);
  expect(provider.originalFields!.map(field => field.name)).toEqual([
    'analysis',
    'analysis_analysis',
    'analysis_analysis_analysis'
  ]);
  expect(provider.originalFields![0].sortable).toBe(false);
  const page = await provider.readPage({
    view: 'original',
    population: 'selected',
    offset: 0,
    limit: 2,
    columns: ['analysis', 'analysis_analysis_analysis'],
    sort: {column: 'analysis_analysis_analysis', direction: 'descending'}
  });
  expect(page!.columns.analysis).toEqual(['222', '111']);
  expect(Array.from(page!.columns.analysis_analysis_analysis)).toEqual([2, 1]);
  expect(Array.from(page!.validity.analysis_analysis_analysis)).toEqual([1, 0]);
  expect(readPage.mock.calls[0][0].columns).toEqual(['analysis']);
  expect(readPage.mock.calls[0][0].sort.column).toBe('analysis');
});

test('wide original tables reject excessive cell counts before any GPU read', async () => {
  const source = new arrow.Table(
    Object.fromEntries(
      Array.from({length: 200}, (_, index) => [
        `field_${index}`,
        arrow.makeVector(new Float32Array([1]))
      ])
    )
  );
  const readPage = vi.fn();
  const base: ResidentTableProvider = {
    datasetId: 'source',
    sourceRevision: 1,
    maximumPageSize: 100,
    fields: [{name: 'field_0', sortable: true}],
    getRevision: () => '1',
    subscribe: () => () => {},
    readPage
  };
  const provider = withOriginalSourceTable(base, source, () => true);
  await expect(
    provider.readPage({
      view: 'original',
      population: 'selected',
      offset: 0,
      limit: 100,
      columns: source.schema.fields.map(field => field.name)
    })
  ).rejects.toThrow(/too many cells/);
  expect(readPage).not.toHaveBeenCalled();
});

test('bounded string formatting preserves sliced Unicode, dictionary values and nulls', () => {
  const strings = arrow.vectorFromArray([null, '北😀', 'suffix'], new arrow.Utf8());
  expect(formatOriginalTableValue(strings.slice(1), 0)).toBe('北😀');
  expect(formatOriginalTableValue(strings, 0)).toBe('null');
  const dictionary = arrow.vectorFromArray(['alpha', '😀'.repeat(10000)]);
  vi.spyOn(dictionary, 'get').mockImplementation(() => {
    throw new Error('unbounded dictionary decode');
  });
  expect(formatOriginalTableValue(dictionary, 0)).toBe('alpha');
  const preview = formatOriginalTableValue(dictionary, 1);
  expect(preview.length).toBeLessThanOrEqual(4096);
  expect(preview).toMatch(/… \[preview\]$/);
});

test('original pages of a resident source page the selected rows in before reading cells', async () => {
  const source = new arrow.Table({
    value: arrow.vectorFromArray([null, null, null], new arrow.Float32())
  });
  const loaded = new Set<number>();
  const ensureRows = vi.fn(async (rows: readonly number[]) => {
    for (const row of rows) loaded.add(row);
  });
  const provider = withOriginalSourceTable(
    {
      datasetId: 'resident',
      sourceRevision: 2,
      maximumPageSize: 2,
      fields: [],
      getRevision: () => '2:1',
      subscribe: () => () => {},
      readPage: async () => ({
        revision: 1,
        datasetRevision: 2,
        totalCount: 2,
        sourceRowIndices: new Uint32Array([2, 0]),
        columns: {},
        validity: {}
      })
    },
    source,
    () => true,
    {ensureRows}
  );
  const page = await provider.readPage({
    view: 'original',
    population: 'selected',
    offset: 0,
    limit: 2,
    columns: ['value']
  });
  expect(ensureRows).toHaveBeenCalledWith([2, 0]);
  expect([...loaded]).toEqual([2, 0]);
  expect(page?.sourceRowIndices).toEqual(new Uint32Array([2, 0]));
});
