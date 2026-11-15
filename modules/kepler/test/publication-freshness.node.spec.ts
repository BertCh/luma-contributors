// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {afterAll, beforeEach, expect, test, vi} from 'vitest';
import type {ComputeRuntime} from '../../../examples/deck/kepler-compute/src/compute-runtime';

vi.hoisted(() => vi.resetModules());
const context = vi.hoisted(() => ({
  runtime: undefined as unknown as ComputeRuntime,
  resident: new Set<string>(),
  datasets: {} as Record<string, unknown>,
  register: () => {},
  notify: vi.fn(),
  touch: vi.fn(),
  forget: vi.fn()
}));
vi.mock('../../../examples/deck/kepler-compute/src/app-state', () => ({
  appState: {
    get runtime() {
      return context.runtime;
    },
    publicationSequence: 0,
    requestedPublicationId: undefined as string | undefined,
    workflowNotice: ''
  }
}));
vi.mock('../../../examples/deck/kepler-compute/src/native-config', () => ({
  getActiveRuntime: () => context.runtime
}));
vi.mock('../../../examples/deck/kepler-compute/src/app-services', () => ({
  residentDatasetIds: context.resident,
  plotDiagnostics: new Map(),
  notify: context.notify,
  residentEviction: {touch: context.touch, forget: context.forget}
}));
vi.mock('../../../examples/deck/kepler-compute/src/app-store', () => ({
  store: {
    dispatch: (action: {type: string; id?: string; payload?: {datasets: {info: {id: string}}}}) => {
      if (action.type === 'REMOVE') delete context.datasets[action.id!];
      if (action.type === 'ADD') {
        context.datasets[action.payload!.datasets.info.id] = action.payload!.datasets;
        context.register();
      }
    },
    getState: () => ({keplerGl: {compute: {visState: {datasets: context.datasets}}}}),
    subscribe: () => () => {}
  }
}));
vi.mock('@kepler.gl/actions', () => ({
  removeDataset: (id: string) => ({type: 'REMOVE', id}),
  addDataToMap: (payload: unknown) => ({type: 'ADD', payload})
}));
vi.mock('../../../examples/deck/kepler-compute/src/binding3-field-preparation', () => ({
  fieldPreparationCoordinator: {markSettled: vi.fn(), forget: vi.fn()}
}));
vi.mock('../../../examples/deck/kepler-compute/src/acceptance-instrumentation', () => ({
  recordUploadMeasure: vi.fn()
}));
vi.mock('../../../examples/deck/kepler-compute/src/prepared-layer-classes', () => ({
  PREPARE_DATA_MESSAGE: 'Prepare data'
}));
vi.mock('../../../examples/deck/kepler-compute/src/recovery', () => ({
  scheduleRecoveryCheckpoint: vi.fn()
}));

import {
  publishSource,
  publicationsInProgress
} from '../../../examples/deck/kepler-compute/src/publication';
import {appState} from '../../../examples/deck/kepler-compute/src/app-state';

beforeEach(() => {
  context.resident.clear();
  context.datasets = {};
  context.register = () => {};
});

test.each([
  'attachment',
  'registration',
  'current'
])('source guard survives asynchronous %s and cleans up stale publication', async phase => {
  const table = arrow.tableFromArrays({value: [1, 2]});
  const dataset = {id: 'guarded-publication', value: {source: table}};
  const publications = new Map();
  let imported = true;
  let current = true;
  const removePublication = vi.fn(async (id: string) => {
    publications.delete(id);
  });
  const removeSource = vi.fn(async () => {
    imported = false;
  });
  const attachDataset = vi.fn(async () => {
    await Promise.resolve();
    const published = {consumer: {explorer: {isCurrent: true}}, nativeData: {fields: []}};
    publications.set(dataset.id, published);
    if (phase === 'attachment') current = false;
    return published;
  });
  context.runtime = {
    publications,
    attachDataset,
    removePublication,
    requestUpdate: vi.fn(),
    session: {
      importArrowDataset: vi.fn(() => dataset),
      registry: {getDataset: () => (imported ? dataset : undefined)},
      removeDataset: removeSource
    },
    device: {isLost: false}
  } as unknown as ComputeRuntime;
  context.register = () => {
    if (phase === 'registration') current = false;
  };
  appState.requestedPublicationId = dataset.id;
  const result = publishSource({table, label: 'Guarded', isCurrent: () => current}, {});
  if (phase === 'current') {
    await result;
    expect(publications.has(dataset.id)).toBe(true);
    expect(context.datasets[dataset.id]).toBeDefined();
    expect(context.resident.has(dataset.id)).toBe(true);
    expect(removePublication).not.toHaveBeenCalled();
  } else {
    await expect(result).rejects.toThrow(/changed/);
    expect(publications.size).toBe(0);
    expect(context.datasets[dataset.id]).toBeUndefined();
    expect(context.resident.has(dataset.id)).toBe(false);
    expect(removePublication).toHaveBeenCalledWith(dataset.id);
    expect(removeSource).toHaveBeenCalled();
  }
  expect(appState.requestedPublicationId).toBeUndefined();
  expect(publicationsInProgress.size).toBe(0);
});

afterAll(() => {
  for (const name of [
    'app-state',
    'native-config',
    'app-services',
    'app-store',
    'binding3-field-preparation',
    'acceptance-instrumentation',
    'prepared-layer-classes',
    'recovery'
  ])
    vi.doUnmock(`../../../examples/deck/kepler-compute/src/${name}`);
  vi.doUnmock('@kepler.gl/actions');
  vi.resetModules();
});
