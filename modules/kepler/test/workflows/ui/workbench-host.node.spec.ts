// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterAll, beforeEach, describe, expect, test, vi} from 'vitest';
import {createRequire} from 'node:module';
import * as arrow from 'apache-arrow';
import {fitBounds, wrapTo} from '@kepler.gl/actions';
import {createConnection} from '../../../../../examples/deck/kepler-compute/src/duckdb-browser';
import type {DatasetOperationQueue} from '../../../../../examples/deck/kepler-compute/src/binding2-dataset-operations';
import {
  createWorkbenchHost,
  runWorkflowSqlOnConnection
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-host';
import {
  prepareArrowTableForGPUAnalytics,
  type PrepareArrowTableForGPUAnalyticsOptions
} from '@luma-contributors/analysis/analysis-core';
import {publishSource} from '../../../../../examples/deck/kepler-compute/src/publication';
import type {WorkflowDatasetRequest} from '../../../src/workflows/workflow-types';
import {makeFixturePoints} from '../workflow-fixtures';
import {
  readNativeDatasetData,
  type NativeWorkflowDataset,
  type WorkflowDrawingFeature
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-host-data';

const hostRequire = createRequire(
  new URL('../../../../../examples/deck/kepler-compute/package.json', import.meta.url)
);

type ProbeDataset = NativeWorkflowDataset & {
  dataRevision?: number;
  filteredIdxCPU?: number[];
  filterTableCPU(filters: unknown[], layers: unknown[]): unknown;
};
type ProbePublication = {dataset: {id: string; revision: number; deviceEpoch: number}};
type ProbeRuntime = {
  publications: Map<string, ProbePublication>;
  session: {deviceLost: boolean};
  removePublication: ReturnType<typeof vi.fn<(datasetId: string) => Promise<void>>>;
};
// Host modules must bind to this suite's doubles even when a neighbour imported them first.
vi.hoisted(() => vi.resetModules());
afterAll(() => {
  vi.doUnmock('../../../../../examples/deck/kepler-compute/src/app-state');
  vi.doUnmock('../../../../../examples/deck/kepler-compute/src/app-store');
  vi.doUnmock('../../../../../examples/deck/kepler-compute/src/app-services');
  vi.doUnmock('../../../../../examples/deck/kepler-compute/src/resident-drawing-actions');
  vi.doUnmock('../../../../../examples/deck/kepler-compute/src/publication');
  vi.doUnmock('../../../../../examples/deck/kepler-compute/src/duckdb-browser');
  vi.doUnmock('@kepler.gl/actions');
  vi.doUnmock('@kepler.gl/constants');
  vi.doUnmock('../../../../../examples/deck/kepler-compute/src/binding3-field-preparation');
  vi.doUnmock('../../../../../examples/deck/kepler-compute/src/sample-data');
  vi.resetModules();
  const version = (globalThis as typeof globalThis & {deck?: {VERSION?: unknown}}).deck?.VERSION;
  expect(
    String(version ?? ''),
    'Workbench specs must not load deck.gl 9 into the shared worker; mock Kepler and deck runtimes.'
  ).not.toMatch(/^9\./);
});
const hostState = vi.hoisted(() => ({
  runtime: null as ProbeRuntime | null,
  datasets: {} as Record<string, ProbeDataset>,
  filters: [] as unknown[],
  layers: [] as unknown[],
  requestedPublicationId: undefined as string | undefined,
  queue: null as DatasetOperationQueue | null,
  publish: null as
    | null
    | ((
        table: arrow.Table,
        normalization: PrepareArrowTableForGPUAnalyticsOptions,
        isCurrent: () => boolean
      ) => Promise<void>),
  removals: [] as string[],
  dispatchedActions: [] as unknown[],
  onRemove: null as null | (() => void),
  onDispatch: null as null | ((action: unknown) => void),
  features: [] as WorkflowDrawingFeature[],
  residentIds: new Set<string>()
}));
vi.mock('../../../../../examples/deck/kepler-compute/src/duckdb-browser', () => ({
  createConnection: vi.fn()
}));
vi.mock('../../../../../examples/deck/kepler-compute/src/app-state', () => ({appState: hostState}));
vi.mock('../../../../../examples/deck/kepler-compute/src/app-store', () => ({
  store: {
    getState: () => ({
      keplerGl: {
        compute: {
          visState: {
            datasets: hostState.datasets,
            filters: hostState.filters,
            layers: hostState.layers,
            editor: {features: hostState.features}
          }
        }
      }
    }),
    dispatch: (action: {dataId?: string}) => {
      hostState.dispatchedActions.push(action);
      hostState.onDispatch?.(action);
      if (action.dataId) {
        hostState.removals.push(action.dataId);
        delete hostState.datasets[action.dataId];
        hostState.onRemove?.();
      }
    }
  }
}));
vi.mock('../../../../../examples/deck/kepler-compute/src/app-services', () => ({
  residentDatasetIds: hostState.residentIds,
  residentEviction: {forget: vi.fn()},
  notify: vi.fn()
}));
vi.mock('../../../../../examples/deck/kepler-compute/src/resident-drawing-actions', () => ({
  extractSelectedRows: vi.fn()
}));
vi.mock('../../../../../examples/deck/kepler-compute/src/publication', async () => {
  const {DatasetOperationQueue} = await import(
    '../../../../../examples/deck/kepler-compute/src/binding2-dataset-operations'
  );
  const datasetOperations = new DatasetOperationQueue();
  hostState.queue = datasetOperations;
  return {
    datasetOperations,
    isResidentDataset: (datasetId: string) => hostState.residentIds.has(datasetId),
    publishSource: vi.fn(
      async (
        candidate: {table: arrow.Table; isCurrent(): boolean},
        normalization: PrepareArrowTableForGPUAnalyticsOptions
      ) => hostState.publish?.(candidate.table, normalization, candidate.isCurrent)
    ),
    // Reproduce the production helper's late lookup inside the real operation queue.
    removePublishedDataset: async (datasetId: string) =>
      datasetOperations.run(datasetId, 'remove', async () => {
        hostState.removals.push(datasetId);
        delete hostState.datasets[datasetId];
        await hostState.runtime?.removePublication(datasetId);
        hostState.residentIds.delete(datasetId);
      })
  };
});

// Action shapes are explicit test doubles: importing Kepler registers its incompatible deck runtime.
vi.mock('@kepler.gl/actions', () => ({
  fitBounds: (bounds: readonly number[]) => ({type: 'workbench-test/FIT_BOUNDS', bounds}),
  wrapTo: (instance: string, action: object) => ({
    ...action,
    meta: {_addr_: `@@KG_${instance.toUpperCase()}`}
  }),
  removeDataset: (dataId: string) => ({type: 'workbench-test/REMOVE_DATASET', dataId}),
  layerVisualChannelConfigChange: (
    oldLayer: unknown,
    newConfig: unknown,
    channel: string,
    newVisConfig: unknown
  ) => ({
    type: '@@kepler.gl/LAYER_VISUAL_CHANNEL_CHANGE',
    payload: {oldLayer, newConfig, channel, newVisConfig}
  })
}));
vi.mock('@kepler.gl/constants', () => ({
  KEPLER_COLOR_PALETTES: [
    {name: 'Blues', type: 'sequential'},
    {name: 'RdBu', type: 'diverging'},
    {name: 'Set2', type: 'qualitative'}
  ],
  colorPaletteToColorRange: (palette: {name: string; type: string}, options: object) => ({
    ...palette,
    ...options
  })
}));
vi.mock('../../../../../examples/deck/kepler-compute/src/binding3-field-preparation', () => ({
  fieldPreparationCoordinator: {forget: vi.fn()}
}));
vi.mock('../../../../../examples/deck/kepler-compute/src/sample-data', () => ({
  DATASET_ID: 'observations'
}));

function makeDataset(
  fields: readonly {name: string; type: string}[],
  rows: readonly (readonly unknown[])[]
): NativeWorkflowDataset {
  return {
    id: 'native',
    label: 'Taxi trips',
    fields,
    dataContainer: {numRows: () => rows.length, valueAt: (row, column) => rows[row][column]}
  };
}

function makeRequest(
  geometry: WorkflowDatasetRequest['geometry'] = {mode: 'none'}
): WorkflowDatasetRequest {
  return {datasetId: 'native', geometry, population: 'all', maximumRows: 100};
}

function makeBoundsData(geometries: readonly unknown[]) {
  const data = makeFixturePoints(geometries.length);
  data.table.columns['geometry'] = geometries.map(geometry =>
    geometry === null ? null : (JSON.stringify(geometry) ?? null)
  );
  return data;
}

const DRAWING_FEATURES: WorkflowDrawingFeature[] = [
  {
    id: 'polygon',
    geometry: {
      type: 'Polygon',
      coordinates: [
        [
          [0, 0],
          [2, 0],
          [2, 2],
          [0, 0]
        ]
      ]
    }
  },
  {
    id: 'rectangle',
    properties: {shape: 'Rectangle'},
    geometry: {
      type: 'Polygon',
      coordinates: [
        [
          [3, 3],
          [4, 3],
          [4, 4],
          [3, 3]
        ]
      ]
    }
  },
  {
    id: 7,
    geometry: {
      type: 'LineString',
      coordinates: [
        [0, 0],
        [1, 1]
      ]
    }
  },
  {
    id: 'named',
    properties: {name: 'Study area'},
    geometry: {
      type: 'Polygon',
      coordinates: [
        [
          [0, 0],
          [1, 0],
          [1, 1],
          [0, 0]
        ]
      ]
    }
  },
  {geometry: {type: 'Point', coordinates: [1, 2]}}
];

describe('workflow SQL connection exchange', () => {
  test('registers named Arrow inputs, bounds SQL, and drops request tables deterministically', async () => {
    const input = makeFixturePoints(2);
    const query = "SELECT geometry, value FROM input1 WHERE 'a;b' = 'a;b'; -- tail comment";
    const table = new arrow.Table({
      geometry: arrow.vectorFromArray(
        input.table.columns['geometry'] as (string | null)[],
        new arrow.Utf8()
      ),
      value: arrow.vectorFromArray([2, 3], new arrow.Int32())
    });
    const connection = {
      insertArrowTable: vi.fn(async (_table: arrow.Table, _options: {name: string}) => undefined),
      query: vi.fn(async (sql: string) => (sql.startsWith('DROP ') ? new arrow.Table() : table))
    };
    const data = await runWorkflowSqlOnConnection(
      connection,
      {query, tables: {input1: input}, maximumRows: 2},
      new AbortController().signal
    );
    expect(data.kind).toBe('point');
    expect(data.table.columns['value']).toEqual([2, 3]);
    expect(connection.insertArrowTable).toHaveBeenCalledExactlyOnceWith(expect.any(arrow.Table), {
      name: 'input1'
    });
    expect(connection.insertArrowTable.mock.calls[0][0].getChild('geometry')!.get(0)).toBe(
      input.table.columns['geometry'][0]
    );
    expect(connection.query.mock.calls.map(([sql]) => sql)).toEqual([
      `SELECT * FROM (\nSELECT geometry, value FROM input1 WHERE 'a;b' = 'a;b'  -- tail comment\n) AS workflow_query LIMIT 3`,
      'DROP TABLE IF EXISTS "input1"'
    ]);
  });

  test('cleans all temporary inputs after query errors, row-cap refusal and cancellation', async () => {
    for (const outcome of ['query-error', 'capacity', 'cancelled'] as const) {
      const controller = new AbortController();
      const connection = {
        insertArrowTable: vi.fn(async (_table: arrow.Table, _options: {name: string}) => undefined),
        query: vi.fn(async (sql: string) => {
          if (sql.startsWith('DROP ')) return new arrow.Table();
          if (outcome === 'query-error') throw new Error('Syntax error\nLINE 1: invalid query');
          if (outcome === 'cancelled') controller.abort();
          return new arrow.Table({count: arrow.vectorFromArray([1, 2], new arrow.Int32())});
        })
      };
      await expect(
        runWorkflowSqlOnConnection(
          connection,
          {
            query: 'SELECT * FROM input1',
            tables: {input1: makeFixturePoints(1), input2: makeFixturePoints(1)},
            maximumRows: 1
          },
          controller.signal
        )
      ).rejects.toMatchObject({
        code:
          outcome === 'query-error'
            ? 'operation-failed'
            : outcome === 'capacity'
              ? 'capacity-exceeded'
              : 'cancelled'
      });
      expect(connection.query.mock.calls.slice(-2)).toEqual([
        ['DROP TABLE IF EXISTS "input2"'],
        ['DROP TABLE IF EXISTS "input1"']
      ]);
    }
  });

  test('refuses a mutating statement before accessing the connection', async () => {
    const connection = {
      query: vi.fn(async () => new arrow.Table()),
      insertArrowTable: vi.fn(async () => undefined)
    };
    await expect(
      runWorkflowSqlOnConnection(
        connection,
        {query: 'DELETE FROM input1', tables: {}, maximumRows: 10},
        new AbortController().signal
      )
    ).rejects.toMatchObject({
      code: 'parameter-out-of-range',
      message: 'SQL: Query must be a single SELECT statement.'
    });
    expect(connection.query).not.toHaveBeenCalled();
    expect(connection.insertArrowTable).not.toHaveBeenCalled();
  });
});

function makeProbeRuntime(): ProbeRuntime {
  const publications = new Map<string, ProbePublication>();
  return {
    publications,
    session: {deviceLost: false},
    removePublication: vi.fn(async datasetId => {
      publications.delete(datasetId);
    })
  };
}
function makeProbeDataset(id: string, label = 'Trips'): ProbeDataset {
  return {
    ...makeDataset([{name: 'value', type: 'integer'}], [[1]]),
    id,
    label,
    filterTableCPU: () => undefined
  };
}
const PUBLICATION_ID = 'workflow-review-points-data';
function makePublishRequest() {
  return {
    workflowId: 'review',
    workflowTitle: 'Review',
    stepId: 'points',
    output: 'data',
    label: 'Review points',
    data: makeFixturePoints(1)
  };
}
function makeGate() {
  let resolve!: () => void;
  const promise = new Promise<void>(complete => {
    resolve = complete;
  });
  return {promise, resolve};
}
function attachProbePublication(
  runtime: ProbeRuntime,
  revision = 1,
  datasetId = PUBLICATION_ID
): ProbePublication {
  const publication = {dataset: {id: datasetId, revision, deviceEpoch: 1}};
  runtime.publications.set(datasetId, publication);
  if (hostState.runtime === runtime) hostState.datasets[datasetId] = makeProbeDataset(datasetId);
  return publication;
}

describe('workbench host ownership and replay regressions', () => {
  beforeEach(async () => {
    await hostState.queue?.whenIdle();
    hostState.runtime = makeProbeRuntime();
    hostState.datasets = {};
    hostState.filters = [];
    hostState.layers = [];
    hostState.removals = [];
    hostState.dispatchedActions = [];
    hostState.onRemove = null;
    hostState.onDispatch = null;
    hostState.features = [];
    hostState.residentIds.clear();
    hostState.publish = null;
    vi.mocked(publishSource).mockClear();
    vi.mocked(createConnection).mockReset();
    hostState.requestedPublicationId = undefined;
  });

  test('wave 2 F1/F2 runs real DuckDB WASM without an app runtime and converts decimal aggregates', async () => {
    const {createDuckDB, NODE_RUNTIME, VoidLogger} = await import('@duckdb/duckdb-wasm/blocking');
    const database = await createDuckDB(
      {
        mvp: {
          mainModule: hostRequire.resolve('@duckdb/duckdb-wasm/dist/duckdb-mvp.wasm'),
          mainWorker: hostRequire.resolve('@duckdb/duckdb-wasm/dist/duckdb-node-mvp.worker.cjs')
        },
        eh: {
          mainModule: hostRequire.resolve('@duckdb/duckdb-wasm/dist/duckdb-eh.wasm'),
          mainWorker: hostRequire.resolve('@duckdb/duckdb-wasm/dist/duckdb-node-eh.worker.cjs')
        }
      },
      new VoidLogger(),
      NODE_RUNTIME
    );
    await database.instantiate();
    database.open({});
    const connection = database.connect();
    vi.mocked(createConnection).mockResolvedValue({
      query: async query => connection.query(query),
      insertArrowTable: async (table, options) => {
        connection.insertArrowFromIPCStream(arrow.tableToIPC(table, 'stream'), {
          ...options,
          create: true
        });
      },
      destroy: async () => connection.close()
    });
    try {
      hostState.runtime = null;
      const host = createWorkbenchHost();
      const result = await host.runSql!(
        {
          query: 'SELECT 1.25 AS amount, SUM(value) AS total FROM (VALUES (1), (2)) t(value)',
          tables: {},
          maximumRows: 1
        },
        new AbortController().signal
      );
      expect(result.table.columns).toEqual({amount: [1.25], total: [3], row_index: [0]});
      expect(result.table.columnTypes).toEqual({
        amount: 'float64',
        total: 'float64',
        row_index: 'int32'
      });
      const points = await host.runSql!(
        {
          query: 'SELECT * FROM input1 ORDER BY point_id LIMIT 2',
          tables: {input1: makeFixturePoints(3)},
          maximumRows: 2
        },
        new AbortController().signal
      );
      expect(points.table.rowCount).toBe(2);
      expect(points.kind).toBe('point');
      const count = await host.runSql!(
        {
          query: 'SELECT COUNT(*) AS n FROM input1',
          tables: {input1: makeFixturePoints(3)},
          maximumRows: 1
        },
        new AbortController().signal
      );
      expect(count.table.columns['n']).toEqual([3]);
      expect(count.table.columnTypes['n']).toBe('float64');
      expect(count.kind).toBe('table');
      expect(
        connection.query(
          "SELECT table_name FROM information_schema.tables WHERE table_name = 'input1'"
        ).numRows
      ).toBe(0);
    } finally {
      connection.close();
      database.reset();
    }
  }, 20000);

  test('wave 2 F3 separates ambiguous component tuples and retracts only the requested output', async () => {
    const owner = hostState.runtime!;
    hostState.publish = async () => {
      attachProbePublication(owner, 1, hostState.requestedPublicationId!);
    };
    const host = createWorkbenchHost();
    const first = await host.publishOutput!(
      {...makePublishRequest(), workflowId: 'a-b', stepId: 'c', output: 'result'},
      new AbortController().signal
    );
    const second = await host.publishOutput!(
      {...makePublishRequest(), workflowId: 'a', stepId: 'b-c', output: 'result'},
      new AbortController().signal
    );
    expect(first.datasetId).toBe('workflow-a%2Db-c-result');
    expect(second.datasetId).toBe('workflow-a-b%2Dc-result');
    expect(owner.publications.size).toBe(2);
    await host.retractOutput!(first.datasetId);
    expect(owner.publications.has(first.datasetId)).toBe(false);
    expect(owner.publications.has(second.datasetId)).toBe(true);
  });

  test('wave 2 F3 refuses occupied first-publication ids and another output replacement id', async () => {
    const owner = hostState.runtime!;
    hostState.publish = async () => {
      attachProbePublication(owner);
    };
    const host = createWorkbenchHost();
    await host.publishOutput!(makePublishRequest(), new AbortController().signal);
    const original = owner.publications.get(PUBLICATION_ID);
    for (const request of [
      makePublishRequest(),
      {...makePublishRequest(), stepId: 'another', previousDatasetId: PUBLICATION_ID}
    ]) {
      await expect(
        host.publishOutput!(request, new AbortController().signal)
      ).rejects.toMatchObject({code: 'publish-failed'});
      expect(owner.publications.get(PUBLICATION_ID)).toBe(original);
    }
    expect(publishSource).toHaveBeenCalledTimes(1);
    expect(owner.removePublication).not.toHaveBeenCalled();
  });

  test.each([
    'same runtime',
    'replacement runtime'
  ])('wave 2 F6 preserves recovered output exclusion after obsolete retraction: %s', async restoration => {
    const owner = hostState.runtime!;
    hostState.publish = async () => {
      attachProbePublication(owner);
    };
    const host = createWorkbenchHost();
    await host.publishOutput!(makePublishRequest(), new AbortController().signal);
    const recovered = restoration === 'same runtime' ? owner : makeProbeRuntime();
    hostState.runtime = recovered;
    const publication = attachProbePublication(recovered, 2);
    hostState.residentIds.clear();
    expect(host.listDatasets()).toEqual([]);
    await host.retractOutput!(PUBLICATION_ID);
    expect(recovered.publications.get(PUBLICATION_ID)).toBe(publication);
    expect(host.listDatasets()).toEqual([]);
    expect(host.getDatasetRevisionKey({...makeRequest(), datasetId: PUBLICATION_ID})).toBe(
      'missing'
    );
  });

  test('wave 2 F7 drawing callers cannot mutate cached data or profiles', async () => {
    hostState.features = DRAWING_FEATURES;
    const host = createWorkbenchHost();
    const revision = host.getDrawingRevisionKey!();
    const first = await host.readDrawings!('all', new AbortController().signal);
    const expected = structuredClone(first);
    first.table.columns['geometry'] = [JSON.stringify({type: 'Point', coordinates: [99, 99]})];
    first.table.columnTypes['label'] = 'float64';
    first.table.label = 'Changed';
    host.listDrawings!()[0].label = 'Changed';
    const subset = await host.readDrawings!(['rectangle'], new AbortController().signal);
    subset.table.columnTypes['geometry'] = 'boolean';
    expect(await host.readDrawings!('all', new AbortController().signal)).toEqual(expected);
    expect(host.listDrawings!()[0].label).toBe('Polygon 1');
    expect(host.getDrawingRevisionKey!()).toBe(revision);
  });

  test('wave 2 F5 cancellation waits for an uninterruptible query, cleans up, then releases SQL queue', async () => {
    const gate = makeGate();
    let queries = 0;
    const connection = {
      destroy: vi.fn(async () => undefined),
      insertArrowTable: vi.fn(async () => undefined),
      query: vi.fn(async (query: string) => {
        if (query.startsWith('DROP ')) return new arrow.Table();
        queries++;
        if (queries === 1) await gate.promise;
        return new arrow.Table({n: arrow.vectorFromArray([queries], new arrow.Int32())});
      })
    };
    vi.mocked(createConnection).mockResolvedValue(connection);
    const host = createWorkbenchHost();
    const request = {
      query: 'SELECT * FROM input1',
      tables: {input1: makeFixturePoints(1)},
      maximumRows: 1
    };
    const controller = new AbortController();
    let settled = false;
    const first = host.runSql!(request, controller.signal).then(
      () => {
        settled = true;
        return 'unexpected success';
      },
      (failure: unknown) => {
        settled = true;
        return failure;
      }
    );
    const second = host.runSql!(request, new AbortController().signal);
    await vi.waitFor(() => expect(queries).toBe(1));
    controller.abort();
    await new Promise<void>(resolve => setTimeout(resolve, 0));
    expect(settled).toBe(false);
    expect(connection.query).toHaveBeenCalledTimes(1);
    expect(connection.insertArrowTable).toHaveBeenCalledTimes(1);
    gate.resolve();
    expect(await first).toMatchObject({code: 'cancelled'});
    expect((await second).table.columns['n']).toEqual([2]);
    expect(connection.query.mock.calls.map(([query]) => query.startsWith('DROP '))).toEqual([
      false,
      true,
      false,
      true
    ]);
    expect(connection.destroy).not.toHaveBeenCalled();
  });

  test('SQL owns one lazy connection and serializes registration, queries and cleanup', async () => {
    const gate = makeGate();
    let queries = 0;
    const connection = {
      insertArrowTable: vi.fn(async (_table: arrow.Table, _options: {name: string}) => undefined),
      destroy: vi.fn(async () => undefined),
      query: vi.fn(async (sql: string) => {
        if (sql.startsWith('DROP ')) return new arrow.Table();
        queries++;
        if (queries === 1) await gate.promise;
        return new arrow.Table({
          total: arrow.vectorFromArray([BigInt(queries)], new arrow.Int64())
        });
      })
    };
    vi.mocked(createConnection).mockResolvedValue(connection);
    hostState.runtime = null;
    const host = createWorkbenchHost();
    expect(createConnection).not.toHaveBeenCalled();
    const request = {
      query: 'SELECT COUNT(*) AS total FROM input1',
      tables: {input1: makeFixturePoints(1)},
      maximumRows: 2
    };
    const first = host.runSql!(request, new AbortController().signal);
    const second = host.runSql!(request, new AbortController().signal);
    await vi.waitFor(() => expect(queries).toBe(1));
    expect(connection.insertArrowTable).toHaveBeenCalledTimes(1);
    gate.resolve();
    const results = await Promise.all([first, second]);
    expect(results.map(result => result.table.columns['total'])).toEqual([[1], [2]]);
    expect(results.map(result => result.table.columnTypes['total'])).toEqual([
      'float64',
      'float64'
    ]);
    expect(createConnection).toHaveBeenCalledTimes(1);
    expect(connection.query.mock.calls.map(([query]) => query)).toEqual([
      'SELECT * FROM (\nSELECT COUNT(*) AS total FROM input1\n) AS workflow_query LIMIT 3',
      'DROP TABLE IF EXISTS "input1"',
      'SELECT * FROM (\nSELECT COUNT(*) AS total FROM input1\n) AS workflow_query LIMIT 3',
      'DROP TABLE IF EXISTS "input1"'
    ]);
    expect(connection.destroy).not.toHaveBeenCalled();
  });

  test('SQL keeps a healthy connection after a query error and reports only its first line', async () => {
    const connection = {
      insertArrowTable: vi.fn(async () => undefined),
      destroy: vi.fn(async () => undefined),
      query: vi.fn(async () => new arrow.Table({n: arrow.vectorFromArray([1], new arrow.Int32())}))
    };
    connection.query.mockRejectedValueOnce(
      new Error('Binder Error: column missing\nLINE 1: missing')
    );
    vi.mocked(createConnection).mockResolvedValue(connection);
    const host = createWorkbenchHost();
    const request = {
      query: 'SELECT missing FROM input1',
      tables: {input1: makeFixturePoints(1)},
      maximumRows: 2
    };
    await expect(host.runSql!(request, new AbortController().signal)).rejects.toMatchObject({
      code: 'operation-failed',
      message: 'SQL: Binder Error: column missing'
    });
    await host.runSql!({...request, query: 'SELECT 1 AS n'}, new AbortController().signal);
    expect(createConnection).toHaveBeenCalledTimes(1);
    expect(connection.destroy).not.toHaveBeenCalled();
  });

  test('SQL recreates a broken connection and retries lazy initialization after factory errors', async () => {
    const broken = {
      insertArrowTable: vi.fn(async () => undefined),
      destroy: vi.fn(async () => undefined),
      query: vi.fn(async () => {
        throw new Error('Connection is closed\nWorker terminated');
      })
    };
    const healthy = {
      insertArrowTable: vi.fn(async () => undefined),
      destroy: vi.fn(async () => undefined),
      query: vi.fn(async () => new arrow.Table({n: arrow.vectorFromArray([1n], new arrow.Int64())}))
    };
    vi.mocked(createConnection)
      .mockRejectedValueOnce(new Error('WASM load failed\nnetwork'))
      .mockResolvedValueOnce(broken)
      .mockResolvedValueOnce(healthy);
    const host = createWorkbenchHost();
    const request = {query: 'SELECT 1 AS n', tables: {}, maximumRows: 2};
    await expect(host.runSql!(request, new AbortController().signal)).rejects.toMatchObject({
      code: 'operation-failed',
      message: 'SQL: WASM load failed'
    });
    await expect(host.runSql!(request, new AbortController().signal)).rejects.toMatchObject({
      code: 'operation-failed',
      message: 'SQL: Connection is closed'
    });
    expect(broken.destroy).toHaveBeenCalledTimes(1);
    expect((await host.runSql!(request, new AbortController().signal)).table.columns['n']).toEqual([
      1
    ]);
    expect(createConnection).toHaveBeenCalledTimes(3);
  });

  test('SQL drops inputs after cancellation and reuses the healthy connection', async () => {
    const controller = new AbortController();
    const connection = {
      insertArrowTable: vi.fn(async () => undefined),
      destroy: vi.fn(async () => undefined),
      query: vi.fn(async (query: string) => {
        if (!query.startsWith('DROP ')) controller.abort();
        return new arrow.Table({n: arrow.vectorFromArray([1], new arrow.Int32())});
      })
    };
    vi.mocked(createConnection).mockResolvedValue(connection);
    const host = createWorkbenchHost();
    const request = {
      query: 'SELECT * FROM input1',
      tables: {input1: makeFixturePoints(1)},
      maximumRows: 2
    };
    await expect(host.runSql!(request, controller.signal)).rejects.toMatchObject({
      code: 'cancelled',
      message: 'SQL: The workflow was cancelled.'
    });
    expect(connection.query.mock.calls.at(-1)).toEqual(['DROP TABLE IF EXISTS "input1"']);
    await host.runSql!(request, new AbortController().signal);
    expect(createConnection).toHaveBeenCalledTimes(1);
    expect(connection.destroy).not.toHaveBeenCalled();
  });

  test('SQL discards a connection when input cleanup fails so later requests cannot see old inputs', async () => {
    const connection = {
      insertArrowTable: vi.fn(async () => undefined),
      destroy: vi.fn(async () => undefined),
      query: vi.fn(async (query: string) => {
        if (query.startsWith('DROP ')) throw new Error('Cleanup failed\nDetails');
        return new arrow.Table({n: arrow.vectorFromArray([1], new arrow.Int32())});
      })
    };
    const nextConnection = {
      ...connection,
      query: vi.fn(async () => new arrow.Table({n: arrow.vectorFromArray([2], new arrow.Int32())}))
    };
    vi.mocked(createConnection)
      .mockResolvedValueOnce(connection)
      .mockResolvedValueOnce(nextConnection);
    const host = createWorkbenchHost();
    const request = {
      query: 'SELECT 1 AS n',
      tables: {input1: makeFixturePoints(1)},
      maximumRows: 2
    };
    await expect(host.runSql!(request, new AbortController().signal)).rejects.toMatchObject({
      code: 'operation-failed',
      message: 'SQL: Cleanup failed'
    });
    expect(connection.destroy).toHaveBeenCalledTimes(1);
    expect((await host.runSql!(request, new AbortController().signal)).table.columns['n']).toEqual([
      2
    ]);
    expect(createConnection).toHaveBeenCalledTimes(2);
  });

  test('styles the published layer and restyles its replacement using the requested field and palette', async () => {
    const owner = hostState.runtime!;
    hostState.publish = async () => {
      attachProbePublication(owner);
      hostState.datasets[PUBLICATION_ID].fields = [{name: 'value', type: 'real'}];
      hostState.layers = [
        {id: 'output-layer', config: {dataId: PUBLICATION_ID}},
        {id: 'other-layer', config: {dataId: 'other'}}
      ];
    };
    const host = createWorkbenchHost();
    const request = {
      ...makePublishRequest(),
      style: {colorColumn: 'value', colorScale: 'quantize' as const, palette: 'diverging' as const}
    };
    await host.publishOutput!(request, new AbortController().signal);
    await host.publishOutput!(
      {
        ...request,
        previousDatasetId: PUBLICATION_ID,
        style: {colorColumn: 'value', colorScale: 'ordinal'}
      },
      new AbortController().signal
    );
    const actions = hostState.dispatchedActions.filter(action =>
      Boolean(
        action &&
          typeof action === 'object' &&
          'type' in action &&
          action.type === '@@kepler.gl/LAYER_VISUAL_CHANNEL_CHANGE'
      )
    );
    expect(actions).toHaveLength(2);
    expect(actions[0]).toMatchObject({
      meta: {_addr_: '@@KG_COMPUTE'},
      payload: {
        oldLayer: {id: 'output-layer'},
        channel: 'color',
        newConfig: {colorField: {name: 'value', type: 'real'}, colorScale: 'quantize'},
        newVisConfig: {colorRange: {name: 'RdBu', type: 'diverging'}}
      }
    });
    expect(actions[1]).toMatchObject({
      payload: {
        oldLayer: {id: 'output-layer'},
        newConfig: {colorScale: 'ordinal'},
        newVisConfig: {colorRange: {name: 'Set2', type: 'qualitative'}}
      }
    });
  });

  test('unsupported or renderer-rejected colour styles do not fail publication', async () => {
    const owner = hostState.runtime!;
    hostState.publish = async () => {
      attachProbePublication(owner);
      hostState.layers = [{id: 'output-layer', config: {dataId: PUBLICATION_ID}}];
    };
    const host = createWorkbenchHost();
    await expect(
      host.publishOutput!(
        {...makePublishRequest(), style: {colorColumn: 'missing'}},
        new AbortController().signal
      )
    ).resolves.toMatchObject({datasetId: PUBLICATION_ID});
    hostState.onDispatch = action => {
      if (
        action &&
        typeof action === 'object' &&
        'type' in action &&
        action.type === '@@kepler.gl/LAYER_VISUAL_CHANNEL_CHANGE'
      )
        throw new Error('Unsupported field');
    };
    await expect(
      host.publishOutput!(
        {...makePublishRequest(), previousDatasetId: PUBLICATION_ID, style: {colorColumn: 'value'}},
        new AbortController().signal
      )
    ).resolves.toMatchObject({datasetId: PUBLICATION_ID});
    expect(owner.publications.has(PUBLICATION_ID)).toBe(true);
  });

  test('drawing host reads subsets without relabelling and keys geometry and name edits', async () => {
    hostState.features = DRAWING_FEATURES;
    const host = createWorkbenchHost();
    expect(host.listDrawings!().map(drawing => drawing.label)).toEqual([
      'Polygon 1',
      'Rectangle 1',
      'Line 1',
      'Study area',
      'Point 1'
    ]);
    const profiles = host.listDrawings!();
    expect(host.listDrawings!()).toEqual(profiles);
    expect(host.listDrawings!()).not.toBe(profiles);
    const key = host.getDrawingRevisionKey!();
    expect(host.getDrawingRevisionKey!()).toBe(key);
    const subset = await host.readDrawings!(
      ['drawing-5', 'rectangle'],
      new AbortController().signal
    );
    expect(subset.table.columns['drawing_id']).toEqual(['rectangle', 'drawing-5']);
    expect(subset.table.columns['label']).toEqual(['Rectangle 1', 'Point 1']);
    expect((await host.readDrawings!(['7'], new AbortController().signal)).kind).toBe('line');
    hostState.features = DRAWING_FEATURES.map((feature, index) =>
      index === 4 ? {...feature, geometry: {type: 'Point', coordinates: [10, 20]}} : feature
    );
    expect(host.getDrawingRevisionKey!()).not.toBe(key);
    const movedKey = host.getDrawingRevisionKey!();
    hostState.features = hostState.features.map((feature, index) =>
      index === 0 ? {...feature, properties: {name: 'Renamed'}} : feature
    );
    expect(host.getDrawingRevisionKey!()).not.toBe(movedKey);
    await expect(
      host.readDrawings!(['missing'], new AbortController().signal)
    ).rejects.toMatchObject({code: 'dataset-not-found'});
    const controller = new AbortController();
    controller.abort();
    await expect(host.readDrawings!('all', controller.signal)).rejects.toMatchObject({
      code: 'cancelled'
    });
  });

  test('H1 fits the compute map once on first display and preserves the view on replacement', async () => {
    const owner = hostState.runtime!;
    hostState.publish = async () => {
      attachProbePublication(owner);
    };
    const host = createWorkbenchHost();
    const request = {
      ...makePublishRequest(),
      data: makeBoundsData([
        {type: 'Point', coordinates: [-2, 3]},
        {type: 'Point', coordinates: [5, 7]}
      ])
    };
    await host.publishOutput!(request, new AbortController().signal);
    const fitted = wrapTo('compute', fitBounds([-2, 3, 5, 7]));
    expect(hostState.dispatchedActions).toEqual([fitted]);
    await host.publishOutput!(
      {...request, previousDatasetId: PUBLICATION_ID},
      new AbortController().signal
    );
    expect(
      hostState.dispatchedActions.filter(action =>
        Boolean(
          action && typeof action === 'object' && 'type' in action && action.type === fitted.type
        )
      )
    ).toEqual([fitted]);
  });

  test('focusOutput uses the latest successful bounds and forgets retracted outputs', async () => {
    const owner = hostState.runtime!;
    hostState.publish = async () => {
      attachProbePublication(owner);
    };
    const host = createWorkbenchHost();
    await host.publishOutput!(
      {...makePublishRequest(), data: makeBoundsData([{type: 'Point', coordinates: [1, 2]}])},
      new AbortController().signal
    );
    hostState.dispatchedActions = [];
    host.focusOutput!(PUBLICATION_ID);
    host.focusOutput!('unknown');
    expect(hostState.dispatchedActions).toEqual([wrapTo('compute', fitBounds([1, 2, 1, 2]))]);

    await host.publishOutput!(
      {
        ...makePublishRequest(),
        previousDatasetId: PUBLICATION_ID,
        data: makeBoundsData([
          {type: 'Point', coordinates: [-4, -5]},
          {type: 'Point', coordinates: [6, 7]}
        ])
      },
      new AbortController().signal
    );
    hostState.dispatchedActions = [];
    host.focusOutput!(PUBLICATION_ID);
    expect(hostState.dispatchedActions).toEqual([wrapTo('compute', fitBounds([-4, -5, 6, 7]))]);

    await host.retractOutput!(PUBLICATION_ID);
    hostState.dispatchedActions = [];
    host.focusOutput!(PUBLICATION_ID);
    expect(hostState.dispatchedActions).toEqual([]);
  });

  test('focusOutput clears geometry bounds when an output becomes a table', async () => {
    const owner = hostState.runtime!;
    hostState.publish = async () => {
      attachProbePublication(owner);
    };
    const host = createWorkbenchHost();
    await host.publishOutput!(makePublishRequest(), new AbortController().signal);
    await host.publishOutput!(
      {
        ...makePublishRequest(),
        previousDatasetId: PUBLICATION_ID,
        data: readNativeDatasetData(
          makeDataset([{name: 'value', type: 'real'}], [[1]]),
          makeRequest()
        )
      },
      new AbortController().signal
    );
    hostState.dispatchedActions = [];
    host.focusOutput!(PUBLICATION_ID);
    expect(hostState.dispatchedActions).toEqual([]);
  });

  test('H1 does not fit a table or a failed display', async () => {
    const owner = hostState.runtime!;
    hostState.publish = async () => {
      attachProbePublication(owner);
    };
    const host = createWorkbenchHost();
    const request = {
      ...makePublishRequest(),
      data: readNativeDatasetData(
        makeDataset([{name: 'value', type: 'real'}], [[1]]),
        makeRequest()
      )
    };
    await host.publishOutput!(request, new AbortController().signal);
    expect(hostState.dispatchedActions).toEqual([]);
    hostState.publish = async () => {
      throw new Error('GPU import failed');
    };
    await expect(
      host.publishOutput!(makePublishRequest(), new AbortController().signal)
    ).rejects.toMatchObject({code: 'publish-failed'});
    expect(hostState.dispatchedActions).toEqual([]);
  });

  test.each([
    'device recovery',
    'workspace replay'
  ])('H2 excludes recreated workflow output ids after %s, including a fresh host', async restoration => {
    const owner = hostState.runtime!;
    hostState.publish = async () => {
      attachProbePublication(owner);
    };
    const host = createWorkbenchHost();
    await host.publishOutput!(makePublishRequest(), new AbortController().signal);
    const restoredRuntime = restoration === 'device recovery' ? owner : makeProbeRuntime();
    hostState.runtime = restoredRuntime;
    attachProbePublication(restoredRuntime, 2);
    // Recovery temporarily clears resident registration, but the host still knows this id.
    expect(host.listDatasets()).toEqual([]);
    hostState.residentIds.add(PUBLICATION_ID);
    hostState.datasets['source'] = makeProbeDataset('source', 'Source');
    const readers = [host, createWorkbenchHost()];
    for (const reader of readers) {
      expect(reader.listDatasets().map(dataset => dataset.id)).toEqual(['source']);
      expect(reader.getDatasetRevisionKey({...makeRequest(), datasetId: PUBLICATION_ID})).toBe(
        'missing'
      );
      await expect(
        reader.readDataset(
          {...makeRequest(), datasetId: PUBLICATION_ID},
          new AbortController().signal
        )
      ).rejects.toMatchObject({code: 'dataset-not-found'});
      await expect(
        reader.readDataset(
          {...makeRequest(), datasetId: 'old-id', datasetLabel: 'Trips'},
          new AbortController().signal
        )
      ).rejects.toMatchObject({code: 'dataset-not-found'});
    }
    // A fresh host must remember discovered output ids through a later recovery, too.
    hostState.residentIds.delete(PUBLICATION_ID);
    for (const reader of readers)
      expect(reader.listDatasets().map(dataset => dataset.id)).toEqual(['source']);
  });

  test('display publication rounds only the map copy using automatic preparation budgets', async () => {
    const owner = hostState.runtime!;
    const request = makePublishRequest();
    request.data = makeFixturePoints(2);
    request.data.table.columns['value'] = [0.1, 1234567.891];
    const original = structuredClone(request.data);
    let mapValues: unknown[] | undefined;
    hostState.publish = async (table, normalization) => {
      const prepared = prepareArrowTableForGPUAnalytics(table, normalization);
      mapValues = Array.from(prepared.table.getChild('value')!);
      attachProbePublication(owner);
    };
    const result = await createWorkbenchHost().publishOutput!(
      request,
      new AbortController().signal
    );
    expect(vi.mocked(publishSource)).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({table: expect.any(arrow.Table), label: 'Review points'}),
      {
        columns: ['point_id', 'value', 'longitude', 'latitude'],
        float64: {
          mode: 'round',
          maximumAbsoluteError: 1e-37,
          maximumRelativeError: 2 ** -23
        }
      },
      {longitude: 'longitude', latitude: 'latitude'},
      undefined
    );
    expect(mapValues).toEqual([Math.fround(0.1), Math.fround(1234567.891)]);
    expect(request.data).toEqual(original);
    expect(result).toEqual({datasetId: PUBLICATION_ID, label: 'Review points'});
  });

  test('display publication refuses Float32 overflow with an actionable column message and no pending id', async () => {
    const owner = hostState.runtime!;
    const request = makePublishRequest();
    request.data.table.columns['area'] = [Number.MAX_VALUE];
    request.data.table.columnTypes['area'] = 'float64';
    hostState.publish = async (table, normalization) => {
      prepareArrowTableForGPUAnalytics(table, normalization);
      attachProbePublication(owner);
    };
    const host = createWorkbenchHost();
    await expect(host.publishOutput!(request, new AbortController().signal)).rejects.toMatchObject({
      code: 'publish-failed',
      message:
        'Review points: Column "area" cannot be shown at 32-bit precision; remove it with Select columns before displaying.'
    });
    expect(owner.publications.size).toBe(0);
    expect(Object.keys(hostState.datasets)).toEqual([]);
    expect(hostState.requestedPublicationId).toBeUndefined();
    hostState.datasets[PUBLICATION_ID] = makeProbeDataset(PUBLICATION_ID);
    expect(host.listDatasets().map(dataset => dataset.id)).toEqual([PUBLICATION_ID]);
  });

  test.each([
    false,
    true
  ])('display publication failure cleans partial native data when service already cleaned GPU: %s', async serviceCleanedGpu => {
    const owner = hostState.runtime!;
    hostState.publish = async (_table, _normalization, isCurrent) => {
      attachProbePublication(owner);
      expect(isCurrent()).toBe(true);
      if (serviceCleanedGpu) await owner.removePublication(PUBLICATION_ID);
      throw new Error('Native registration failed');
    };
    const host = createWorkbenchHost();
    await expect(
      host.publishOutput!(makePublishRequest(), new AbortController().signal)
    ).rejects.toMatchObject({
      code: 'publish-failed',
      message: 'Review points: Publication failed: Native registration failed'
    });
    expect(owner.publications.size).toBe(0);
    expect(Object.keys(hostState.datasets)).toEqual([]);
    expect(hostState.residentIds.has(PUBLICATION_ID)).toBe(false);
    hostState.datasets[PUBLICATION_ID] = makeProbeDataset(PUBLICATION_ID);
    expect(host.listDatasets().map(dataset => dataset.id)).toEqual([PUBLICATION_ID]);
    delete hostState.datasets[PUBLICATION_ID];
    hostState.publish = async () => {
      attachProbePublication(owner);
    };
    await expect(
      host.publishOutput!(makePublishRequest(), new AbortController().signal)
    ).resolves.toEqual({datasetId: PUBLICATION_ID, label: 'Review points'});
  });

  test.each([
    false,
    true
  ])('orphan cleanup preserves a replacement native table across queued cleanup; new workspace: %s', async replaceWorkspace => {
    const owner = hostState.runtime!;
    const gate = makeGate();
    let blocker: Promise<unknown> | undefined;
    hostState.publish = async (_table, _normalization, isCurrent) => {
      attachProbePublication(owner);
      expect(isCurrent()).toBe(true);
      await owner.removePublication(PUBLICATION_ID);
      blocker = hostState.queue!.run(PUBLICATION_ID, 'remove', () => gate.promise);
      throw new Error('Native registration failed');
    };
    const host = createWorkbenchHost();
    const failed = expect(
      host.publishOutput!(makePublishRequest(), new AbortController().signal)
    ).rejects.toMatchObject({code: 'publish-failed'});
    await vi.waitFor(() => expect(hostState.queue!.busyDatasetIds).toContain(PUBLICATION_ID));
    await new Promise<void>(resolve => setTimeout(resolve, 0));
    if (replaceWorkspace) hostState.runtime = makeProbeRuntime();
    const replacement = makeProbeDataset(PUBLICATION_ID, 'Replacement');
    hostState.datasets[PUBLICATION_ID] = replacement;
    gate.resolve();
    await blocker;
    await failed;
    expect(hostState.datasets[PUBLICATION_ID]).toBe(replacement);
    expect(host.listDatasets().map(dataset => dataset.label)).toEqual(['Replacement']);
    expect(hostState.removals).toEqual([]);
  });

  test('F5 refuses ambiguous replay labels but permits unique labels and exact ids', async () => {
    hostState.datasets = {a: makeProbeDataset('a'), b: makeProbeDataset('b')};
    const host = createWorkbenchHost();
    const request = {...makeRequest(), datasetId: 'old-id', datasetLabel: 'Trips'};
    await expect(host.readDataset(request, new AbortController().signal)).rejects.toMatchObject({
      code: 'dataset-not-found',
      message: 'Trips: Dataset label matches multiple datasets; select the intended dataset.'
    });
    expect(host.getDatasetRevisionKey(request)).toBe('missing');
    expect(
      (await host.readDataset({...request, datasetId: 'b'}, new AbortController().signal)).table.id
    ).toBe('b');
    delete hostState.datasets['a'];
    expect((await host.readDataset(request, new AbortController().signal)).table.id).toBe('b');
  });

  test('F8 restores native accessibility after a failed publication', async () => {
    hostState.datasets[PUBLICATION_ID] = makeProbeDataset(PUBLICATION_ID);
    hostState.publish = async () => {
      throw new Error('GPU import failed');
    };
    const host = createWorkbenchHost();
    await expect(
      host.publishOutput!(makePublishRequest(), new AbortController().signal)
    ).rejects.toMatchObject({code: 'publish-failed'});
    expect(host.listDatasets().map(dataset => dataset.id)).toEqual([PUBLICATION_ID]);
    expect(host.getDatasetRevisionKey({...makeRequest(), datasetId: PUBLICATION_ID})).not.toBe(
      'missing'
    );
    expect(hostState.requestedPublicationId).toBeUndefined();
  });

  test('F1 cancellation cleanup stays bound to its original workspace across a queued removal', async () => {
    const owner = hostState.runtime!;
    const controller = new AbortController();
    const gate = makeGate();
    let blocker: Promise<unknown> | undefined;
    hostState.publish = async () => {
      attachProbePublication(owner);
      blocker = hostState.queue!.run(PUBLICATION_ID, 'remove', () => gate.promise);
      controller.abort();
    };
    const host = createWorkbenchHost();
    const failed = expect(
      host.publishOutput!(makePublishRequest(), controller.signal)
    ).rejects.toMatchObject({code: 'cancelled'});
    await vi.waitFor(() => expect(hostState.queue!.busyDatasetIds).toContain(PUBLICATION_ID));
    // Let the host queue its cleanup before changing workspaces.
    await new Promise<void>(resolve => setTimeout(resolve, 0));
    const replacement = makeProbeRuntime();
    hostState.runtime = replacement;
    const replacementPublication = attachProbePublication(replacement);
    const replacementNative = hostState.datasets[PUBLICATION_ID];
    expect(host.listDatasets().map(dataset => dataset.id)).toEqual([PUBLICATION_ID]);
    gate.resolve();
    await blocker;
    await failed;
    expect(owner.publications.has(PUBLICATION_ID)).toBe(false);
    expect(replacement.publications.get(PUBLICATION_ID)).toBe(replacementPublication);
    expect(replacement.removePublication).not.toHaveBeenCalled();
    expect(hostState.datasets[PUBLICATION_ID]).toBe(replacementNative);
    expect(hostState.removals).toEqual([]);
  });

  test('F1 retraction does not remove a newer generation in the same workspace', async () => {
    const owner = hostState.runtime!;
    hostState.publish = async () => {
      attachProbePublication(owner);
    };
    const host = createWorkbenchHost();
    await host.publishOutput!(makePublishRequest(), new AbortController().signal);
    const gate = makeGate();
    const blocker = hostState.queue!.run(PUBLICATION_ID, 'remove', () => gate.promise);
    const retracted = host.retractOutput!(PUBLICATION_ID);
    await new Promise<void>(resolve => setTimeout(resolve, 0));
    const newer = attachProbePublication(owner, 2);
    const native = hostState.datasets[PUBLICATION_ID];
    gate.resolve();
    await blocker;
    await retracted;
    expect(owner.publications.get(PUBLICATION_ID)).toBe(newer);
    expect(owner.removePublication).not.toHaveBeenCalled();
    expect(hostState.datasets[PUBLICATION_ID]).toBe(native);
  });

  test('F1 native removal notifications cannot retire a synchronous replacement generation', async () => {
    const owner = hostState.runtime!;
    hostState.publish = async () => {
      attachProbePublication(owner);
    };
    const host = createWorkbenchHost();
    await host.publishOutput!(makePublishRequest(), new AbortController().signal);
    let newer: ProbePublication | undefined;
    hostState.onRemove = () => {
      newer = attachProbePublication(owner, 2);
    };
    await host.retractOutput!(PUBLICATION_ID);
    expect(newer).toBeDefined();
    expect(owner.publications.get(PUBLICATION_ID)).toBe(newer);
    expect(owner.removePublication).not.toHaveBeenCalled();
    expect(hostState.datasets[PUBLICATION_ID]).toBeDefined();
  });

  test('F8 a failed attempt preserves genuinely pre-existing publication ownership', async () => {
    const owner = hostState.runtime!;
    hostState.publish = async () => {
      attachProbePublication(owner);
    };
    const host = createWorkbenchHost();
    await host.publishOutput!(makePublishRequest(), new AbortController().signal);
    const original = owner.publications.get(PUBLICATION_ID);
    hostState.datasets['source'] = makeProbeDataset('source', 'Source');
    hostState.publish = async () => {
      throw new Error('GPU import failed');
    };
    await expect(
      host.publishOutput!(makePublishRequest(), new AbortController().signal)
    ).rejects.toMatchObject({code: 'publish-failed'});
    expect(owner.publications.get(PUBLICATION_ID)).toBe(original);
    expect(host.listDatasets().map(dataset => dataset.id)).toEqual(['source']);
    await host.retractOutput!(PUBLICATION_ID);
    expect(owner.publications.has(PUBLICATION_ID)).toBe(false);
    expect(owner.removePublication).toHaveBeenCalledExactlyOnceWith(PUBLICATION_ID);
  });

  test('F6 refuses success after publication switches workspaces and cleans only the old owner', async () => {
    const owner = hostState.runtime!;
    const replacement = makeProbeRuntime();
    hostState.publish = async () => {
      attachProbePublication(owner);
      hostState.runtime = replacement;
      attachProbePublication(replacement);
    };
    const host = createWorkbenchHost();
    await expect(
      host.publishOutput!(makePublishRequest(), new AbortController().signal)
    ).rejects.toMatchObject({code: 'publish-failed'});
    expect(owner.publications.has(PUBLICATION_ID)).toBe(false);
    expect(replacement.publications.has(PUBLICATION_ID)).toBe(true);
    expect(replacement.removePublication).not.toHaveBeenCalled();
    expect(hostState.datasets[PUBLICATION_ID]).toBeDefined();
  });

  test('F2 different polygon-selected rows require different revision keys when coordinate roles change', async () => {
    const table: ProbeDataset = {
      ...makeDataset(
        ['lon', 'lat', 'other_lon', 'other_lat'].map(name => ({name, type: 'real'})),
        [
          [0, 0, 10, 10],
          [10, 10, 0, 0]
        ]
      ),
      label: 'Trips',
      // Polygon CPU selection changes while the native table's coarse index stays unchanged.
      filteredIndex: [0, 1],
      filteredIdxCPU: [],
      filterTableCPU: vi.fn((_filters: unknown[], layers: unknown[]) => {
        const layer = layers[0] as ReturnType<typeof makeLayer>;
        table.filteredIdxCPU = layer.config.columns.lng.fieldIdx === 0 ? [0] : [1];
      })
    };
    const polygon = {
      id: 'polygon',
      dataId: ['native'],
      layerId: ['points'],
      name: [],
      type: 'polygon',
      enabled: true,
      value: {
        id: 'drawn-polygon',
        type: 'Feature',
        properties: {},
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [-1, -1],
              [1, -1],
              [1, 1],
              [-1, 1],
              [-1, -1]
            ]
          ]
        }
      }
    };
    function makeLayer(longitude: number, latitude: number) {
      return {
        id: 'points',
        type: 'point',
        config: {
          dataId: 'native',
          columns: {lng: {fieldIdx: longitude}, lat: {fieldIdx: latitude}}
        }
      };
    }
    hostState.datasets['native'] = table;
    hostState.filters = [polygon];
    hostState.layers = [makeLayer(0, 1)];
    const host = createWorkbenchHost();
    const request = {...makeRequest(), population: 'selected' as const};
    const firstKey = host.getDatasetRevisionKey(request);
    const allKey = host.getDatasetRevisionKey({...request, population: 'all'});
    expect(
      (await host.readDataset(request, new AbortController().signal)).table.columns['row_index']
    ).toEqual([0]);
    hostState.layers = [makeLayer(2, 3)];
    expect(
      (await host.readDataset(request, new AbortController().signal)).table.columns['row_index']
    ).toEqual([1]);
    expect(host.getDatasetRevisionKey(request)).not.toBe(firstKey);
    expect(host.getDatasetRevisionKey({...request, population: 'all'})).toBe(allKey);
    expect(table.filterTableCPU).toHaveBeenCalledTimes(2);
    expect(table.filterTableCPU).toHaveBeenLastCalledWith(hostState.filters, hostState.layers);
  });
});
