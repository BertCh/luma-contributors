// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import type {ReactElement, ReactNode} from 'react';
import {afterEach, describe, expect, test, vi} from 'vitest';
import {
  createKeplerCuratedAnalysisMapping,
  getKeplerCuratedAnalysis,
  getWorkflowFeatures,
  type KeplerDatasetProfile
} from '../../../src/index';
import {analysisDatasetContext} from '../../../../../examples/deck/kepler-compute/src/analysis-dataset-context';
import {
  attachWorkbenchAnalysisProfile,
  getWorkbenchAnalysisProfile,
  makeWorkbenchNativeAnalysisDatasets
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-analysis-profile';
import {
  readNativeDatasetData,
  type NativeWorkflowDataset
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-host-data';
import {getWorkbenchAnalysisProfiles} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-analysis-launch';
import {WorkbenchAnalysisMapping} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-analysis-mapping';
import {WorkbenchParameterValue} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-parameter-form';
import type {WorkbenchStore} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-store';

const hostRequire = createRequire(
  new URL('../../../../../examples/deck/kepler-compute/package.json', import.meta.url)
);
const react = hostRequire('react') as typeof import('react') & {
  __CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE: {H: unknown};
};
type Element = ReactElement<Record<string, unknown>>;
function elements(node: ReactNode): Element[] {
  if (Array.isArray(node)) return node.flatMap(elements);
  return react.isValidElement<Record<string, unknown>>(node)
    ? [node, ...elements(node.props['children'] as ReactNode)]
    : [];
}

// Drive the production handlers with deterministic hook state; this does not claim browser timing.
function createProbe<Properties>(component: (properties: Properties) => React.JSX.Element) {
  const states: unknown[] = [];
  const references: {current: unknown}[] = [];
  const dependencies = new Map<number, readonly unknown[]>();
  const cleanups = new Map<number, () => void>();
  return {
    render(properties: Properties) {
      let stateIndex = 0;
      let referenceIndex = 0;
      let effectIndex = 0;
      const effects: (() => void)[] = [];
      const internals = react.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;
      const previous = internals.H;
      internals.H = {
        useState(initial: unknown) {
          const index = stateIndex++;
          if (!(index in states))
            states[index] = typeof initial === 'function' ? initial() : initial;
          return [
            states[index],
            (next: unknown) => {
              states[index] = typeof next === 'function' ? next(states[index]) : next;
            }
          ];
        },
        useRef(initial: unknown) {
          const index = referenceIndex++;
          references[index] ??= {current: initial};
          return references[index];
        },
        useId: () => 'review-probe',
        useEffect(effect: () => void | (() => void), next: readonly unknown[]) {
          const index = effectIndex++;
          const prior = dependencies.get(index);
          if (
            prior &&
            prior.length === next.length &&
            next.every((value, position) => Object.is(value, prior[position]))
          )
            return;
          dependencies.set(index, next);
          effects.push(() => {
            cleanups.get(index)?.();
            const cleanup = effect();
            if (cleanup) cleanups.set(index, cleanup);
          });
        }
      };
      try {
        const tree = component(properties);
        for (const effect of effects) effect();
        return elements(tree);
      } finally {
        internals.H = previous;
      }
    },
    unmount() {
      for (const cleanup of cleanups.values()) cleanup();
    }
  };
}

function makeDataset(): NativeWorkflowDataset {
  const rows = [
    [10, 20, 100, 4],
    [11, 21, 200, 5],
    [12, 22, 300, 6]
  ];
  return {
    id: 'review-native',
    label: 'Unprepared native observations',
    dataRevision: 1,
    fields: [
      {name: 'longitude', type: 'real'},
      {name: 'latitude', type: 'real'},
      {name: 'row_index', type: 'integer'},
      {name: 'row_index_source', type: 'integer'}
    ],
    dataContainer: {numRows: () => rows.length, valueAt: (row, column) => rows[row][column]}
  };
}
afterEach(() => {
  analysisDatasetContext.setDeclarations('review-native', undefined);
  analysisDatasetContext.synchronize([]);
});

describe('review-fix-host native profile handoffs', () => {
  test('registers native coordinates without reading values and refines through the shared service', async () => {
    let current = makeDataset();
    const reads = vi.spyOn(current.dataContainer, 'valueAt');
    const register = () =>
      analysisDatasetContext.synchronize(
        makeWorkbenchNativeAnalysisDatasets([current], dataset => dataset === current)
      );
    register();
    expect(reads).not.toHaveBeenCalled();
    const store = {
      host: {
        listDatasets: () => [{id: current.id}],
        getAnalysisProfile: getWorkbenchAnalysisProfile
      }
    } as unknown as WorkbenchStore;
    expect(getWorkbenchAnalysisProfiles(store)).toMatchObject([
      {
        datasetId: current.id,
        kind: 'points',
        pointColumns: {longitude: 'longitude', latitude: 'latitude'}
      }
    ]);
    await analysisDatasetContext.readStatistics(current.id);
    expect(reads).toHaveBeenCalledTimes(12);
    expect(
      getWorkbenchAnalysisProfiles(store)[0].columns.find(column => column.name === 'row_index')
        ?.maximum
    ).toBe(300);
    current.dataRevision = 2;
    expect(getWorkbenchAnalysisProfiles(store)).toEqual([]);
    register();
    expect(analysisDatasetContext.get(current.id)?.stage).toBe('metadata');
    const previous = analysisDatasetContext.get(current.id);
    current = {...makeDataset(), dataRevision: 2};
    register();
    expect(analysisDatasetContext.get(current.id)).not.toBe(previous);
    analysisDatasetContext.synchronize([]);
    expect(getWorkbenchAnalysisProfiles(store)).toEqual([]);
  });

  test('refuses oversized native refinement before reading a source cell', async () => {
    const dataset = {...makeDataset(), dataContainer: {numRows: () => 100001, valueAt: vi.fn()}};
    analysisDatasetContext.synchronize(makeWorkbenchNativeAnalysisDatasets([dataset], () => true));
    await expect(analysisDatasetContext.readStatistics(dataset.id)).rejects.toThrow(/capacity/);
    expect(dataset.dataContainer.valueAt).not.toHaveBeenCalled();
  });

  test.each([
    'all',
    'selected'
  ] as const)('reprofiles aliased columns and translates declarations for %s rows', async population => {
    const dataset = makeDataset();
    analysisDatasetContext.synchronize(makeWorkbenchNativeAnalysisDatasets([dataset], () => true));
    await analysisDatasetContext.readStatistics(dataset.id);
    analysisDatasetContext.setDeclarations(dataset.id, {
      measurement: {row_index: 'amount'},
      rowSemantics: 'objects',
      coordinateModel: 'geographic'
    });
    const geometry = {mode: 'point-columns' as const, longitude: 'longitude', latitude: 'latitude'};
    const data = attachWorkbenchAnalysisProfile(
      readNativeDatasetData(
        dataset,
        {
          datasetId: dataset.id,
          geometry,
          population,
          maximumRows: 100
        },
        population === 'selected' ? [2] : undefined
      ),
      population,
      geometry
    );
    const column = data.analysisProfile!.columns.find(
      column => column.name === 'row_index_source_2'
    );
    expect(column?.maximum).toBe(300);
    expect(column?.minimum).toBe(population === 'all' ? 100 : 300);
    expect(column?.measurement).toBe('amount');
    expect(column?.measurementSource).toBe('declared');
    expect(data.analysisProfile!.rowSemantics).toMatchObject({
      value: 'objects',
      source: 'declared'
    });
    expect(data.analysisProfile!.geometryColumns).toEqual(['geometry']);
    expect(data.analysisProfile!.columns.find(column => column.name === 'row_index')?.maximum).toBe(
      2
    );
    expect(getWorkflowFeatures(data).features[0].properties?.['row_index_source_2']).toBe(
      population === 'all' ? 100 : 300
    );
    for (const column of data.analysisProfile!.columns)
      expect(Object.hasOwn(data.table.columns, column.name)).toBe(true);
  });
});

describe('review-fix-host mapping validity', () => {
  test('invalid cell size blocks the rendered button and its production handler until corrected', () => {
    const analysis = getKeplerCuratedAnalysis('density-surface')!;
    const profile: KeplerDatasetProfile = {
      datasetId: 'points',
      label: 'Points',
      kind: 'points',
      catalogGeometry: 'point',
      rowCount: 3,
      geometryColumns: [],
      columns: [],
      pointColumns: {longitude: 'longitude', latitude: 'latitude'},
      coordinates: {model: 'planar', source: 'declared'},
      extent: [0, 0, 10, 10]
    };
    const mapping = createKeplerCuratedAnalysisMapping(analysis, [profile]);
    expect(mapping.ready).toBe(true);
    const onRun = vi.fn();
    const onParametersChange = vi.fn();
    const properties = {
      analysis,
      mapping,
      profiles: [profile],
      onRun,
      onParametersChange,
      onChange: () => {},
      onCancel: () => {}
    };
    const parent = createProbe(WorkbenchAnalysisMapping);
    let nodes = parent.render(properties);
    const parameter = nodes.find(
      node =>
        node.type === WorkbenchParameterValue &&
        node.props['testId'] === 'ax-mapping-parameter-cellSize'
    )!;
    const control = createProbe(WorkbenchParameterValue);
    const controlProperties = parameter.props as unknown as Parameters<
      typeof WorkbenchParameterValue
    >[0];
    let controls = control.render(controlProperties);
    const change = (text: string) => {
      const input = controls.find(
        node => node.props['data-testid'] === 'ax-mapping-parameter-cellSize'
      )!;
      (input.props['onChange'] as (event: {target: {value: string}}) => void)({
        target: {value: text}
      });
      controls = control.render(controlProperties);
    };
    const oldRun = nodes.find(node => node.props['data-testid'] === 'ax-mapping-run')!.props[
      'onClick'
    ] as () => void;
    change('-1');
    oldRun(); // Even a handler retained before the parent rerender must see current validity.
    nodes = parent.render(properties);
    expect(
      nodes.find(node => node.props['data-testid'] === 'ax-mapping-run')!.props['disabled']
    ).toBe(true);
    expect(onParametersChange).not.toHaveBeenCalled();
    expect(onRun).not.toHaveBeenCalled();
    change('1');
    nodes = parent.render(properties);
    expect(
      nodes.find(node => node.props['data-testid'] === 'ax-mapping-run')!.props['disabled']
    ).toBe(false);
    (
      nodes.find(node => node.props['data-testid'] === 'ax-mapping-run')!.props[
        'onClick'
      ] as () => void
    )();
    expect(onRun).toHaveBeenCalledOnce();
    control.unmount();
    parent.unmount();
  });
});
