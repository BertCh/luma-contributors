// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {beforeEach, afterEach, expect, test, vi} from 'vitest';
import type {UnknownAction} from 'redux';

type Presentation = ReturnType<typeof makePresentation>;
type Table = ReturnType<typeof makeTable>;
type Dataset = {
  info: {id: string; label: string};
  data: {fields: {name: string}[]; rows: number[][]};
  metadata?: Record<string, unknown>;
};
type Payload = {
  datasets: Dataset | Dataset[];
  config: Presentation;
  options: object;
  limitations: never[];
};
const context = vi.hoisted(() => ({
  presentation: undefined as unknown as Presentation,
  tables: {} as Record<string, Table>,
  resident: new Set<string>(),
  runtime: {session: {deviceLost: false}, device: {isLost: false}, publications: new Map()},
  phase: 'ready',
  requests: [] as {payload: Payload; onFinish: unknown}[],
  actions: [] as UnknownAction[],
  dispatch: undefined as unknown as (action: UnknownAction) => unknown,
  load: vi.fn(),
  catalog: vi.fn(),
  prepare: vi.fn(),
  checkpoint: vi.fn()
}));
vi.mock('../../../examples/deck/kepler-compute/src/app-store', () => ({
  store: {
    dispatch: (action: UnknownAction) => context.dispatch(action),
    getState: () => ({
      keplerGl: {
        compute: {
          visState: {
            datasets: context.tables,
            layers: context.presentation.config.visState.layers
          }
        }
      }
    })
  }
}));
vi.mock('../../../examples/deck/kepler-compute/src/app-state', () => ({
  appState: {
    get runtime() {
      return context.runtime;
    },
    deviceRecovery: {
      get status() {
        return {phase: context.phase};
      },
      subscribe: () => () => {},
      checkpoint: context.checkpoint
    }
  }
}));
vi.mock('../../../examples/deck/kepler-compute/src/app-services', () => ({notify: vi.fn()}));
vi.mock('../../../examples/deck/kepler-compute/src/sample-data', () => ({DATASET_ID: 'synthetic'}));
vi.mock('../../../examples/deck/kepler-compute/src/sample-map-loader', () => ({
  fetchSampleMap: (...parameters: unknown[]) => context.load(...parameters),
  fetchSampleMaps: (...parameters: unknown[]) => context.catalog(...parameters)
}));
vi.mock('../../../examples/deck/kepler-compute/src/native-hosts', () => ({
  nativeAutoPrepare: {
    republish: (...parameters: unknown[]) => context.prepare(...parameters)
  }
}));
vi.mock('../../../examples/deck/kepler-compute/src/publication', () => ({
  isResidentDataset: (id: string) => context.resident.has(id),
  removePublishedDataset: async (id: string) => {
    delete context.tables[id];
  }
}));
vi.mock('@kepler.gl/actions', () => ({
  ActionTypes: {LOAD_FILES: 'LOAD_FILES', RESET_MAP_CONFIG: 'RESET_MAP_CONFIG'},
  addDataToMap: (payload: unknown) => ({type: 'ADD', payload}),
  updateDatasetProps: (id: string, props: unknown) => ({type: 'PROPS', id, props}),
  removeDataset: (id: string) => ({type: 'REMOVE', id}),
  toggleModal: (payload: unknown) => ({type: 'MODAL', payload}),
  wrapTo: (_id: string, action: unknown) => action
}));
vi.mock('@kepler.gl/schemas', () => ({
  default: {
    getConfigToSave: () => structuredClone(context.presentation),
    parseSavedConfig: (value: unknown) => value
  }
}));

function makePresentation(id = 'synthetic') {
  return {
    version: 'v1',
    config: {
      visState: {
        layers: [{id: `${id}-layer`, type: 'point', config: {dataId: id, isVisible: true}}]
      },
      mapState: {latitude: id === 'synthetic' ? 12 : 42, longitude: 9, zoom: 5}
    }
  };
}
function makeTable(id: string, value = 1) {
  return {
    id,
    label: id,
    color: [1, 2, 3],
    type: undefined,
    metadata: {} as Record<string, unknown>,
    dataRevision: 1,
    fields: [{name: 'value'}],
    dataContainer: {numRows: () => 1, valueAt: () => value}
  };
}
function sample(id = 'earthquakes') {
  return {
    id,
    label: id,
    description: '',
    detail: '',
    visible: true,
    dataUrl: `https://example.com/${id}.csv`,
    configUrl: 'https://example.com/config.json'
  };
}
function payload(id: string, value = 2): Payload {
  return {
    datasets: {info: {id, label: id}, data: {fields: [{name: 'value'}], rows: [[value]]}},
    config: makePresentation(id),
    options: {keepExistingConfig: false},
    limitations: []
  };
}
function deferred<Value>() {
  let resolve!: (value: Value) => void;
  const promise = new Promise<Value>(complete => {
    resolve = complete;
  });
  return {promise, resolve};
}
function complete(index: number, error?: Error) {
  const request = context.requests[index]!;
  const datasets = Array.isArray(request.payload.datasets)
    ? request.payload.datasets
    : [request.payload.datasets];
  context.dispatch({
    type: 'CREATE_SUCCESS',
    payload: {
      onFinish: request.onFinish,
      results: datasets.map((dataset, datasetIndex) =>
        error && datasetIndex === 0
          ? {status: 'rejected', reason: error}
          : {status: 'fulfilled', value: dataset}
      )
    }
  });
}

beforeEach(async () => {
  vi.resetModules();
  context.presentation = makePresentation();
  context.tables = {synthetic: makeTable('synthetic')};
  context.resident = new Set();
  context.runtime = {
    session: {deviceLost: false},
    device: {isLost: false},
    publications: new Map()
  };
  context.phase = 'ready';
  context.requests = [];
  context.actions = [];
  context.load.mockReset().mockImplementation(async (entry: {id: string}) => payload(entry.id));
  context.catalog.mockReset().mockResolvedValue([sample()]);
  context.prepare.mockReset().mockResolvedValue(undefined);
  context.checkpoint.mockReset();
  vi.stubGlobal('location', {search: ''});
  const {sampleMapNativeMiddleware} = await import(
    '../../../examples/deck/kepler-compute/src/sample-map-native-transaction'
  );
  const next = (value: unknown) => {
    const action = value as UnknownAction;
    context.actions.push(action);
    if (action.type === 'ADD') {
      const incoming = action['payload'] as Payload;
      if (incoming.config) {
        context.presentation = incoming.config;
        context.tables = {};
      }
      context.requests.push({payload: incoming, onFinish: action['onFinish']});
    } else if (action.type === 'PROPS') {
      Object.assign(context.tables[String(action['id'])]!, action['props']);
    } else if (action.type === 'REMOVE') delete context.tables[String(action['id'])];
    else if (action.type === 'CREATE_SUCCESS') {
      for (const result of (action['payload'] as {results: {status: string; value?: Dataset}[]})
        .results) {
        if (result.status === 'fulfilled') {
          const dataset = result.value!;
          context.tables[dataset.info.id] = makeTable(dataset.info.id, dataset.data.rows[0]![0]);
          context.tables[dataset.info.id]!.metadata = dataset.metadata ?? {};
        }
      }
    }
    return action;
  };
  context.dispatch = sampleMapNativeMiddleware({} as never)(next) as typeof context.dispatch;
});
afterEach(() => {
  vi.unstubAllGlobals();
});

test('resident reload waits for its native update and its explicit GPU republish', async () => {
  const host = await import('../../../examples/deck/kepler-compute/src/sample-map-host');
  context.tables['earthquakes'] = makeTable('earthquakes');
  context.resident.add('earthquakes');
  const gpu = deferred<void>();
  context.prepare.mockReturnValue(gpu.promise);
  const loading = host.loadSampleMap(sample());
  await vi.waitFor(() => expect(context.requests).toHaveLength(1));
  expect(context.prepare).not.toHaveBeenCalled();
  expect(
    context.actions.some(action => action.type === 'REMOVE' && action['id'] === 'earthquakes')
  ).toBe(true);
  expect(host.sampleMapStatus.loaded).toBeUndefined();
  complete(0);
  await vi.waitFor(() =>
    expect(context.prepare).toHaveBeenCalledWith('earthquakes', expect.any(AbortSignal))
  );
  expect(host.sampleMapStatus.loaded).toBeUndefined();
  gpu.resolve();
  await loading;
  expect(context.tables['earthquakes']!.dataContainer.valueAt()).toBe(2);
  expect(host.sampleMapStatus.loaded?.id).toBe('earthquakes');
});

test('retained native datasets and attribution survive Kepler presentation reset', async () => {
  const host = await import('../../../examples/deck/kepler-compute/src/sample-map-host');
  const retained = makeTable('movement');
  retained.metadata = {lumaSampleMap: {sample: {id: 'movement'}, licenseNotice: 'Noncommercial'}};
  context.tables['movement'] = retained;
  const loading = host.loadSampleMap(sample());
  await vi.waitFor(() => expect(context.requests).toHaveLength(1));
  expect(context.tables).toEqual({}); // The real Kepler reset clears native datasets before tasks finish.
  complete(0);
  await loading;
  expect(context.tables['movement']!.metadata).toEqual(retained.metadata);
  expect(context.tables['synthetic']).toBeDefined();
});

test('ownership is revalidated after GPU preparation before completion mutations', async () => {
  const host = await import('../../../examples/deck/kepler-compute/src/sample-map-host');
  const preparation = deferred<void>();
  context.prepare.mockReturnValue(preparation.promise);
  const loading = host.loadSampleMap(sample());
  const rejected = expect(loading).rejects.toThrow('workspace changed');
  await vi.waitFor(() => expect(context.requests).toHaveLength(1));
  complete(0);
  await vi.waitFor(() => expect(context.prepare).toHaveBeenCalledOnce());
  context.runtime = {...context.runtime};
  const tables = context.tables;
  preparation.resolve();
  await rejected;
  expect(context.tables).toBe(tables);
  expect(context.tables['synthetic']).toBeDefined();
  expect(host.sampleMapStatus.loaded).toBeUndefined();
  expect(context.actions.filter(action => action.type === 'MODAL')).toEqual([]);
});

test('failed startup registration removes the partial dataset and restores synthetic layers and camera', async () => {
  const host = await import('../../../examples/deck/kepler-compute/src/sample-map-host');
  const original = structuredClone(context.presentation);
  const loading = host.initializeStartupSample();
  await vi.waitFor(() => expect(context.requests).toHaveLength(1));
  expect(context.presentation).not.toEqual(original);
  context.tables['earthquakes'] = makeTable('earthquakes', 999);
  complete(0, new Error('native import rejected'));
  await vi.waitFor(() => expect(context.requests).toHaveLength(2));
  complete(1);
  await loading;
  expect(context.presentation).toEqual(original);
  expect(Object.keys(context.tables)).toEqual(['synthetic']);
  expect(host.sampleMapStatus.notice).toContain('Showing synthetic squares');
  expect(host.sampleMapStatus.loaded).toBeUndefined();
});

test('failed rollback reports its failure and never claims a synthetic fallback', async () => {
  const host = await import('../../../examples/deck/kepler-compute/src/sample-map-host');
  const loading = host.initializeStartupSample();
  await vi.waitFor(() => expect(context.requests).toHaveLength(1));
  complete(0, new Error('initial failure'));
  await vi.waitFor(() => expect(context.requests).toHaveLength(2));
  complete(1, new Error('rollback failure'));
  await loading;
  expect(host.sampleMapStatus.error).toContain('restoring the previous presentation failed');
  expect(host.sampleMapStatus.error).toContain('initial failure');
  expect(host.sampleMapStatus.error).toContain('rollback failure');
  expect(host.sampleMapStatus.loadFailure?.message).toContain('initial failure');
  expect(host.sampleMapStatus.loadFailure?.stack).toContain('initial failure');
  expect(host.sampleMapStatus.rollbackFailure?.message).toContain('rollback failure');
  expect(host.sampleMapStatus.rollbackFailure?.stack).toContain('rollback failure');
  expect(host.sampleMapStatus.notice).not.toContain('Showing synthetic squares');
  expect(host.sampleMapStatus.loaded).toBeUndefined();
});

test('synchronous load and rollback exceptions retain the first thrower and both stacks', async () => {
  const host = await import('../../../examples/deck/kepler-compute/src/sample-map-host');
  const dispatch = context.dispatch;
  const firstFailure = new TypeError('load view is undefined');
  const rollbackFailure = new TypeError('rollback view is undefined');
  firstFailure.stack = 'TypeError: load view is undefined\n    at firstThrower (split-view.ts:44)';
  rollbackFailure.stack = 'TypeError: rollback view is undefined\n    at rollbackThrower';
  let attempts = 0;
  context.dispatch = action => {
    if (action.type === 'ADD') throw attempts++ === 0 ? firstFailure : rollbackFailure;
    return dispatch(action);
  };
  const failure = await host.loadSampleMap(sample()).catch(error => error);
  expect(failure).toBeInstanceOf(AggregateError);
  expect(failure.errors).toEqual([firstFailure, rollbackFailure]);
  expect(failure.cause).toBe(firstFailure);
  expect(host.sampleMapStatus.error).toBe(
    'Sample load failed: TypeError: load view is undefined; restoring the previous presentation failed: TypeError: rollback view is undefined'
  );
  expect(host.sampleMapStatus.loadFailure?.stack).toBe(firstFailure.stack);
  expect(host.sampleMapStatus.rollbackFailure?.stack).toBe(rollbackFailure.stack);
  expect(host.sampleMapStatus.notice).not.toContain('Showing synthetic squares');
});

test('failed reload restores the previous native data and presentation', async () => {
  const host = await import('../../../examples/deck/kepler-compute/src/sample-map-host');
  context.tables['earthquakes'] = makeTable('earthquakes', 17);
  const original = structuredClone(context.presentation);
  const loading = host.loadSampleMap(sample());
  const rejected = expect(loading).rejects.toThrow('native import rejected');
  await vi.waitFor(() => expect(context.requests).toHaveLength(1));
  context.tables['earthquakes'] = makeTable('earthquakes', 999);
  complete(0, new Error('native import rejected'));
  await vi.waitFor(() => expect(context.requests).toHaveLength(2));
  complete(1);
  await rejected;
  expect(context.tables['earthquakes']!.dataContainer.valueAt()).toBe(17);
  expect(context.presentation).toEqual(original);
});

test('a delayed startup catalog cannot overwrite an explicit user load', async () => {
  const host = await import('../../../examples/deck/kepler-compute/src/sample-map-host');
  const catalog = deferred<ReturnType<typeof sample>[]>();
  context.catalog.mockReturnValue(catalog.promise);
  const startup = host.initializeStartupSample();
  const explicit = host.loadSampleMap(sample('nyctrips'));
  await vi.waitFor(() => expect(context.requests).toHaveLength(1));
  complete(0);
  await explicit;
  catalog.resolve([sample()]);
  await startup;
  expect(
    context.requests.map(
      request =>
        (Array.isArray(request.payload.datasets)
          ? request.payload.datasets[0]!
          : request.payload.datasets
        ).info.id
    )
  ).toEqual(['nyctrips']);
  expect(host.sampleMapStatus.loaded?.id).toBe('nyctrips');
  expect(context.tables['synthetic']).toBeDefined();
});

test('native file loading invalidates a pending startup catalog generation', async () => {
  const host = await import('../../../examples/deck/kepler-compute/src/sample-map-host');
  const catalog = deferred<ReturnType<typeof sample>[]>();
  context.catalog.mockReturnValue(catalog.promise);
  const startup = host.initializeStartupSample();
  context.dispatch({type: 'LOAD_FILES'});
  catalog.resolve([sample()]);
  await startup;
  expect(context.requests).toEqual([]);
  expect(host.sampleMapStatus.loaded).toBeUndefined();
});

test('an explicit native file load waits for an in-progress sample rollback', async () => {
  const host = await import('../../../examples/deck/kepler-compute/src/sample-map-host');
  const loading = host.loadSampleMap(sample());
  const rejected = expect(loading).rejects.toThrow('superseded');
  await vi.waitFor(() => expect(context.requests).toHaveLength(1));
  const fileLoad = context.dispatch({type: 'LOAD_FILES'});
  await vi.waitFor(() => expect(context.requests).toHaveLength(2));
  expect(context.actions.some(action => action.type === 'LOAD_FILES')).toBe(false);
  complete(1);
  await rejected;
  await fileLoad;
  expect(context.presentation).toEqual(makePresentation());
  expect(context.actions.some(action => action.type === 'LOAD_FILES')).toBe(true);
  complete(0);
  expect(context.tables['earthquakes']).toBeUndefined();
});

test('superseding a native task rolls back before the new load and discards late task results', async () => {
  const host = await import('../../../examples/deck/kepler-compute/src/sample-map-host');
  const previous = host.loadSampleMap(sample());
  const rejected = expect(previous).rejects.toThrow('superseded');
  await vi.waitFor(() => expect(context.requests).toHaveLength(1));
  const next = host.loadSampleMap(sample('nyctrips'));
  await vi.waitFor(() => expect(context.requests).toHaveLength(2));
  complete(1); // The old presentation is restored before the new load may mutate it.
  await rejected;
  await vi.waitFor(() => expect(context.requests).toHaveLength(3));
  complete(2);
  await next;
  complete(0);
  expect(context.tables['earthquakes']).toBeUndefined();
  expect(context.presentation.config.visState.layers[0]!.config.dataId).toBe('nyctrips');
  expect(host.sampleMapStatus.loaded?.id).toBe('nyctrips');
});

test.each([
  'loss',
  'replacement',
  'recovery'
] as const)('completion after %s cannot mutate status or remove bootstrap', async condition => {
  const host = await import('../../../examples/deck/kepler-compute/src/sample-map-host');
  const loading = host.loadSampleMap(sample());
  const rejected = expect(loading).rejects.toThrow('workspace changed');
  await vi.waitFor(() => expect(context.requests).toHaveLength(1));
  if (condition === 'loss') context.runtime.session.deviceLost = true;
  if (condition === 'replacement') context.runtime = {...context.runtime};
  if (condition === 'recovery') context.phase = 'rebuilding';
  const tablesBeforeCompletion = context.tables;
  const presentationBeforeCompletion = context.presentation;
  complete(0);
  await rejected;
  expect(context.tables['earthquakes']).toBeUndefined();
  expect(context.tables).toBe(tablesBeforeCompletion);
  expect(context.presentation).toBe(presentationBeforeCompletion);
  expect(
    context.actions.some(action => action.type === 'REMOVE' && action['id'] === 'synthetic')
  ).toBe(false);
  expect(host.sampleMapStatus.loaded).toBeUndefined();
  expect(context.actions.filter(action => action.type === 'MODAL')).toEqual([]);
});
