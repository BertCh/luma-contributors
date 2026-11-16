// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import type {CatalogOutputTable} from '../src/analysis/catalog-geojson';
import {
  getAnalysisResultReportView,
  dismissAnalysisResultReport
} from '../../../examples/deck/kepler-compute/src/analysis-result-publication';
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

test.each([
  'planar',
  'EPSG:3857'
] as const)('upload boundary rechecks %s on a legacy map candidate', async sourceCrs => {
  const geometry = JSON.stringify({
    type: 'Point',
    coordinates: [1113194.9079327357, 2273030.926987689]
  });
  const table: CatalogOutputTable = {
    id: 'crs-result',
    label: 'CRS result',
    rowCount: 1,
    coordinateSystem: 'planar',
    geometryColumn: 'geometry',
    columns: {geometry: [geometry], value: [9]},
    columnTypes: {geometry: 'string', value: 'float64'},
    ...(sourceCrs === 'EPSG:3857'
      ? {
          raster: {
            type: 'raster' as const,
            id: 'mercator',
            crs: sourceCrs,
            cellSizeMode: 'web-mercator' as const,
            grid: makeGPURasterMetadata({width: 1, height: 1}),
            band: {
              index: 1,
              values: new Float32Array([1]),
              validity: new Uint8Array([1]),
              noDataValue: null
            }
          }
        }
      : {})
  };
  const source = arrow.tableFromArrays({geometry: [geometry], value: [9]});
  const dataset = {id: 'crs-result', value: {source}};
  const publications = new Map();
  const importArrowDataset = vi.fn(
    ({
      table: displayTable,
      normalization
    }: {
      table: arrow.Table;
      normalization: {columns: string[]};
    }) => {
      expect(normalization.columns).toEqual(['value']);
      const point = JSON.parse(String(displayTable.getChild('geometry')!.get(0)));
      expect(point.coordinates[0]).toBeCloseTo(10, 10);
      expect(point.coordinates[1]).toBeCloseTo(20, 10);
      return dataset;
    }
  );
  context.runtime = {
    publications,
    device: {isLost: false},
    requestUpdate: vi.fn(),
    removePublication: vi.fn(),
    attachDataset: vi.fn(async () => {
      const published = {consumer: {explorer: {isCurrent: true}}, nativeData: {fields: []}};
      publications.set(dataset.id, published);
      return published;
    }),
    session: {importArrowDataset, registry: {getDataset: () => dataset}, removeDataset: vi.fn()}
  } as unknown as ComputeRuntime;
  appState.requestedPublicationId = dataset.id;
  // A saved/staged result can still say "map"; it must pass the current shared CRS guard.
  await publishSource(
    {
      table: source,
      label: table.label,
      isCurrent: () => true,
      analysisResult: {table, destination: 'map'}
    },
    {columns: ['value', 'longitude', 'latitude']},
    undefined,
    'geometry'
  );
  if (sourceCrs === 'planar') {
    expect(importArrowDataset).not.toHaveBeenCalled();
    expect(publications.size).toBe(0);
    const report = getAnalysisResultReportView()!;
    expect(report.analysisResult.destination).toBe('report');
    expect(report.analysisResult.mapMessage).toMatch(/planar.*no declared projection/);
    expect(report.analysisResult.table).toBe(table);
    dismissAnalysisResultReport(report);
    appState.requestedPublicationId = undefined;
  } else expect(importArrowDataset).toHaveBeenCalledOnce();
  expect(table.columns['geometry']).toEqual([geometry]);
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
