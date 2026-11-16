// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {afterAll, describe, expect, test, vi} from 'vitest';
import * as ReactHooks from 'react';
import type {ReactElement} from 'react';
import {
  WorkbenchSidePanel,
  formatSidePanelDuration,
  getSidePanelDatasetSource,
  getSidePanelRecommendations,
  getSidePanelDisplayLabel,
  getSidePanelInputProblem,
  getSidePanelRunLabel,
  getSidePanelRunSummary,
  getSidePanelSteps,
  parseSidePanelInputText
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-side-panel';
import {WORKBENCH_TEST_IDS} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-contract';
import {
  createWorkbenchStore,
  type WorkbenchState,
  type WorkbenchStore
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-store';
import {updateWorkflowInput} from '../../../src/workflows/workflow-edit';
import {getWorkflowSourceOperations} from '../../../src/workflows/workflow-source-operations';
import {
  WorkflowError,
  WORKFLOW_LIMITS,
  type WorkflowDatasetProfile,
  type WorkflowInput,
  type WorkflowOperation,
  type WorkflowRunRecord,
  type WorkflowRunnerState,
  type WorkflowStepResult,
  type WorkflowStepStatus
} from '../../../src/workflows/workflow-types';
import {
  getCatalogBinding,
  getWorkflowOperationRegistry,
  profileKeplerFeatureCollection,
  suggestKeplerAnalyses
} from '@luma-contributors/kepler';
import {listWorkflowTemplates} from '../../../src/workflows/workflow-templates';
import {
  makeFixtureDocument,
  makeFixtureHost,
  makeFixturePoints,
  makeFixtureRegistry
} from '../workflow-fixtures';

// Shared workers may have cached useWorkbenchState with unmocked hooks in another suite.
vi.hoisted(() => vi.resetModules());
vi.mock('react', async importOriginal => {
  const original = await importOriginal<typeof import('react')>();
  return {
    ...original,
    useSyncExternalStore: vi.fn(original.useSyncExternalStore),
    useMemo: vi.fn(original.useMemo),
    useId: vi.fn(original.useId),
    useState: vi.fn(original.useState)
  };
});
afterAll(() => {
  vi.doUnmock('react');
  vi.resetModules();
});

const hostRequire = createRequire(
  new URL('../../../../../examples/deck/kepler-compute/package.json', import.meta.url)
);
const react = hostRequire('react') as typeof import('react');
const {createElement} = react;
const {renderToStaticMarkup} = hostRequire('react-dom/server') as typeof import('react-dom/server');

function makeRunner(
  statuses: WorkflowStepStatus[] = ['ready', 'ready', 'ready']
): WorkflowRunnerState {
  return {
    revision: 1,
    running: false,
    order: ['source', 'scale', 'count'],
    issues: [],
    steps: Object.fromEntries(
      ['source', 'scale', 'count'].map((stepId, index) => [
        stepId,
        {status: statuses[index], key: stepId, issues: []}
      ])
    )
  };
}

function makeRecord(patch: Partial<WorkflowRunRecord> = {}): WorkflowRunRecord {
  return {
    schemaVersion: 1,
    runId: 'fixture-run',
    workflowId: 'fixture-workflow',
    startedAt: 0,
    durationMilliseconds: 1400,
    status: 'done',
    steps: [],
    ...patch
  };
}

function makeResult(): WorkflowStepResult {
  const points = makeFixturePoints(1);
  return {
    key: 'scale',
    outputs: {scaled: {...points, table: {...points.table, rowCount: 1204}}},
    diagnostics: [],
    summary: {},
    engine: 'cpu-reference',
    dispatchedContributors: [],
    parameters: {},
    durationMilliseconds: 42,
    cacheHit: false,
    completedAt: 0
  };
}

function makeStore(patch: Partial<WorkbenchState> = {}) {
  const document = makeFixtureDocument();
  const state: WorkbenchState = {
    revision: 1,
    workflows: [document],
    activeWorkflowId: document.id,
    document,
    runner: makeRunner(),
    selectedStepIds: [],
    canUndo: false,
    canRedo: false,
    dock: {open: false, height: 380, inspectorTab: 'result', logOpen: false},
    palette: {open: false},
    dialog: null,
    draft: null,
    notice: null,
    runLog: [],
    templates: [],
    clipboardStepCount: 0,
    ...patch
  };
  const calls: unknown[][] = [];
  const recordCall = (...arguments_: unknown[]) => {
    calls.push(arguments_);
  };
  const store: WorkbenchStore = {
    registry: makeFixtureRegistry(),
    host: makeFixtureHost(),
    getState: () => state,
    subscribe: () => () => {},
    createWorkflow: options => {
      recordCall('createWorkflow', options);
      return '';
    },
    addWorkflow: added => {
      recordCall('addWorkflow', added);
      return '';
    },
    duplicateWorkflow: workflowId => {
      recordCall('duplicateWorkflow', workflowId);
      return '';
    },
    deleteWorkflow: workflowId => recordCall('deleteWorkflow', workflowId),
    setActiveWorkflow: workflowId => recordCall('setActiveWorkflow', workflowId),
    edit: (tag, change) => {
      recordCall('edit', tag, change);
      return true;
    },
    undo: () => recordCall('undo'),
    redo: () => recordCall('redo'),
    select: stepIds => recordCall('select', stepIds),
    run: async options => {
      recordCall('run', options);
      return null;
    },
    cancel: () => recordCall('cancel'),
    notifyExternalChange: () => recordCall('notifyExternalChange'),
    getRunner: () => null,
    setDock: dock => recordCall('setDock', dock),
    openPalette: options => recordCall('openPalette', options),
    closePalette: () => recordCall('closePalette'),
    setDialog: dialog => recordCall('setDialog', dialog),
    setNotice: notice => recordCall('setNotice', notice),
    proposeDraft: draft => recordCall('proposeDraft', draft),
    acceptDraft: () => {
      recordCall('acceptDraft');
      return null;
    },
    rejectDraft: () => recordCall('rejectDraft'),
    getReviewedAssistantIssues: () => [],
    recordAssistantReview: () => {},
    saveTemplate: () => '',
    deleteTemplate: () => {},
    useTemplate: templateId => {
      recordCall('useTemplate', templateId);
      return 'used-template';
    },
    insertWorkflow: () => false,
    copySelection: () => false,
    cutSelection: () => false,
    paste: () => false,
    serialize: () => undefined,
    restore: collection => recordCall('restore', collection),
    clear: async () => recordCall('clear')
  };
  return {store, state, calls};
}

function render(patch: Partial<WorkbenchState> = {}) {
  const fixture = makeStore(patch);
  return {
    ...fixture,
    markup: renderToStaticMarkup(createElement(WorkbenchSidePanel, {store: fixture.store}))
  };
}

const distanceInput: WorkflowInput = {
  id: 'distance',
  label: 'Distance',
  kind: 'number',
  value: 50,
  unit: 'meters',
  minimum: 0,
  maximum: 1000
};

type PanelElement = ReactElement<Record<string, unknown>>;

function findElement(tree: unknown, matches: (element: PanelElement) => boolean): PanelElement {
  const found: PanelElement[] = [];
  function visit(child: unknown): void {
    if (Array.isArray(child)) child.forEach(visit);
    else if (react.isValidElement<Record<string, unknown>>(child)) {
      if (matches(child)) found.push(child);
      visit(child.props['children']);
    }
  }
  visit(tree);
  expect(found).toHaveLength(1);
  return found[0];
}

function renderPanelTree(store: WorkbenchStore): PanelElement {
  const mocks = [
    vi
      .mocked(ReactHooks.useSyncExternalStore)
      .mockImplementation((_subscribe, getSnapshot) => getSnapshot()),
    vi.mocked(ReactHooks.useMemo).mockImplementation(factory => factory()),
    vi.mocked(ReactHooks.useId).mockReturnValue('side-panel-handler-test')
  ];
  try {
    return WorkbenchSidePanel({store}) as PanelElement;
  } finally {
    for (const mock of mocks) mock.mockReset();
  }
}

function clickPanelControl(store: WorkbenchStore, testId: string): void {
  const control = findElement(
    renderPanelTree(store),
    element => element.props['data-testid'] === testId
  );
  const onClick = control.props['onClick'] as () => void;
  onClick();
}

function makeQuickStartStore(
  dataset: WorkflowDatasetProfile,
  additionalOperations: readonly WorkflowOperation[] = []
) {
  const operations = [
    ...makeFixtureRegistry().list(),
    ...getWorkflowSourceOperations(),
    ...additionalOperations
  ];
  return createWorkbenchStore({
    host: makeFixtureHost([dataset]),
    registry: {
      list: () => operations,
      get: operationId => operations.find(operation => operation.id === operationId)
    }
  });
}

const quickStartDataset: WorkflowDatasetProfile = {
  id: 'observations',
  label: 'Observations',
  rowCount: 1204,
  resident: true,
  fields: [{name: 'geometry', type: 'geojson'}],
  geometryFields: ['geometry', 'other-geometry'],
  pointColumns: {longitude: 'longitude', latitude: 'latitude'}
};

function makeAnalysisRecommendationStore(
  rowSemantics: 'events' | 'surface-samples' = 'events',
  label = 'Observations'
) {
  const dataset = {...quickStartDataset, label, rowCount: 12};
  const profile = profileKeplerFeatureCollection(
    {
      type: 'FeatureCollection',
      features: Array.from({length: 12}, (_, index) => ({
        type: 'Feature',
        geometry: {type: 'Point', coordinates: [index % 4, Math.floor(index / 4)]},
        properties: {rainfall: index + 1}
      }))
    },
    {datasetId: dataset.id, label, coordinateSystem: 'planar', declarations: {rowSemantics}}
  );
  const host = makeFixtureHost([dataset]);
  host.getAnalysisProfile = () => profile;
  host.getDevice = () => ({}) as NonNullable<ReturnType<typeof host.getDevice>>;
  const store = createWorkbenchStore({host, registry: getWorkflowOperationRegistry()});
  return {store, dataset, profile};
}

function makeRecommendationHarness(store: WorkbenchStore, dataset: WorkflowDatasetProfile) {
  const element = findElement(renderPanelTree(store), child => child.props['dataset'] === dataset);
  const component = element.type as (props: {
    store: WorkbenchStore;
    dataset: WorkflowDatasetProfile;
  }) => PanelElement;
  let open = false;
  let tree: PanelElement;
  const memoEntries: Array<{dependencies: readonly unknown[]; value: unknown}> = [];
  function renderRecommendations(profile = dataset): void {
    let memoIndex = 0;
    const mocks = [
      vi.mocked(ReactHooks.useState).mockImplementation((() => [
        open,
        (value: boolean) => {
          open = value;
        }
      ]) as typeof react.useState),
      vi.mocked(ReactHooks.useMemo).mockImplementation(((
        factory: () => unknown,
        dependencies: readonly unknown[]
      ) => {
        const index = memoIndex++;
        const previous = memoEntries[index];
        if (
          !previous ||
          previous.dependencies.length !== dependencies.length ||
          dependencies.some((value, offset) => !Object.is(value, previous.dependencies[offset]))
        )
          memoEntries[index] = {dependencies, value: factory()};
        return memoEntries[index].value;
      }) as typeof react.useMemo)
    ];
    try {
      tree = component({store, dataset: profile});
    } finally {
      for (const mock of mocks) mock.mockReset();
    }
  }
  renderRecommendations();
  return {
    render: renderRecommendations,
    markup: () => renderToStaticMarkup(tree),
    toggle(nextOpen: boolean) {
      const onToggle = tree.props['onToggle'] as (event: {currentTarget: {open: boolean}}) => void;
      onToggle({currentTarget: {open: nextOpen}});
      renderRecommendations();
    },
    click(recommendationId: string) {
      const button = findElement(
        tree,
        child => child.props['data-testid'] === WORKBENCH_TEST_IDS.recommendation(recommendationId)
      );
      const onClick = button.props['onClick'] as () => void;
      onClick();
    }
  };
}

describe('suggested analyses in the side panel', () => {
  test('uses shared AX suggestions and shows readiness/remedies for declared rain gauges', async () => {
    const {store, dataset, profile} = makeAnalysisRecommendationStore('surface-samples');
    try {
      const suggestions = makeRecommendationHarness(store, dataset);
      expect(suggestions.markup()).not.toContain('data-status=');
      suggestions.toggle(true);
      const bindings = store.registry.list().flatMap(operation => {
        const binding = getCatalogBinding(operation.id);
        return binding ? [binding] : [];
      });
      const expected = suggestKeplerAnalyses(bindings, [profile], {
        requiredDatasetId: profile.datasetId,
        maximumCount: 5,
        executionEngine: 'gpu'
      });
      expect(getSidePanelRecommendations(store, dataset.id)).toEqual([
        ...expected.suggestions,
        ...expected.blocked
      ]);
      const markup = suggestions.markup();
      for (const recommendation of [...expected.suggestions, ...expected.blocked]) {
        expect(markup).toContain(
          WORKBENCH_TEST_IDS.recommendation(`${dataset.id}-${recommendation.id}`)
        );
        expect(markup).toContain(`data-status="${recommendation.status}"`);
      }
      expect(
        expected.suggestions.some(suggestion => suggestion.bindingId === 'kernel-density-grid')
      ).toBe(false);
      expect(
        expected.suggestions.some(suggestion => suggestion.bindingId === 'recipe-hot-spot-lattice')
      ).toBe(false);
      expect(markup).toMatch(/Ready|Review mapping|Requirements unmet/);
      for (const recommendation of [...expected.suggestions, ...expected.blocked]) {
        for (const missing of recommendation.applicability.missing) {
          expect(markup).toContain(
            renderToStaticMarkup(createElement(react.Fragment, null, missing.what))
          );
          if (missing.remedy)
            expect(markup).toContain(
              renderToStaticMarkup(createElement(react.Fragment, null, missing.remedy))
            );
        }
      }
      // A changed declaration must replace the recommendations even when the dataset card is unchanged.
      store.host.getAnalysisProfile = () => ({
        ...profile,
        rowSemantics: {value: 'events', source: 'declared'}
      });
      suggestions.render();
      expect(getSidePanelRecommendations(store, dataset.id)).not.toEqual([
        ...expected.suggestions,
        ...expected.blocked
      ]);
    } finally {
      await store.clear();
    }
  });

  test.each([
    false,
    true
  ])('adds shared AX assignments and fields as one undoable edit (active: %s)', async active => {
    const {store, dataset} = makeAnalysisRecommendationStore();
    const edit = vi.spyOn(store, 'edit');
    try {
      if (active) store.addWorkflow(makeFixtureDocument());
      const originalSteps = store.getState().document?.steps ?? [];
      const recommendation = getSidePanelRecommendations(store, dataset.id).find(
        candidate => candidate.status !== 'blocked'
      )!;
      expect(recommendation).toBeDefined();
      const suggestions = makeRecommendationHarness(store, dataset);
      suggestions.toggle(true);
      suggestions.click(`${dataset.id}-${recommendation.id}`);
      const state = store.getState();
      expect(state.document?.steps).toHaveLength(originalSteps.length + 2);
      const added = state.document!.steps.slice(originalSteps.length);
      expect(added[0]).toMatchObject({
        operationId: 'source-dataset',
        parameters: {dataset: {from: 'static', value: dataset.id}}
      });
      expect(added[1].operationId).toBe(recommendation.bindingId);
      expect(Object.values(added[1].inputs)).toEqual([{stepId: added[0].id, output: 'data'}]);
      for (const [name, columns] of Object.entries(recommendation.applicability.fieldBindings))
        expect(added[1].parameters[name]).toEqual({
          from: 'static',
          value: typeof columns === 'string' ? columns : columns.join(',')
        });
      expect(edit.mock.calls.map(([tag]) => tag)).toEqual(['quick-start:source-dataset']);
      expect(state.selectedStepIds).toEqual([added[1].id]);
      expect(state.dock).toMatchObject({open: true, inspectorTab: 'parameters'});
      expect(state.runLog).toEqual([]);
      store.undo();
      expect(store.getState().document?.steps).toEqual(originalSteps);
    } finally {
      edit.mockRestore();
      await store.clear();
    }
  });

  test('connects both datasets of a shared AX comparison recommendation in one edit', async () => {
    const {store, dataset, profile} = makeAnalysisRecommendationStore('events', 'Before');
    const partner = {...dataset, id: 'after', label: 'After'};
    const partnerProfile = {...profile, datasetId: partner.id, label: partner.label};
    store.host.listDatasets = () => [dataset, partner];
    store.host.getAnalysisProfile = datasetId =>
      datasetId === dataset.id ? profile : partnerProfile;
    try {
      const recommendation = getSidePanelRecommendations(store, dataset.id).find(
        candidate => candidate.status !== 'blocked' && new Set(candidate.datasetIds).size === 2
      );
      expect(recommendation).toBeDefined();
      const suggestions = makeRecommendationHarness(store, dataset);
      suggestions.toggle(true);
      suggestions.click(`${dataset.id}-${recommendation!.id}`);
      const document = store.getState().document!;
      expect(document.steps).toHaveLength(3);
      const analysis = document.steps.find(step => step.operationId === recommendation!.bindingId)!;
      for (const [input, datasetId] of Object.entries(
        recommendation!.applicability.inputAssignments
      )) {
        const source = document.steps.find(
          step =>
            step.parameters['dataset']?.from === 'static' &&
            step.parameters['dataset'].value === datasetId
        )!;
        expect(analysis.inputs[input]).toEqual({stepId: source.id, output: 'data'});
      }
      expect(store.getState().runLog).toEqual([]);
      store.undo();
      expect(store.getState().document?.steps).toEqual([]);
    } finally {
      await store.clear();
    }
  });

  test('refuses the whole edit when the analysis would exceed the step limit', async () => {
    const {store, dataset} = makeAnalysisRecommendationStore();
    try {
      const document = makeFixtureDocument();
      document.steps = Array.from({length: WORKFLOW_LIMITS.maximumSteps - 1}, (_, index) => ({
        ...document.steps[0],
        id: `full-${index}`
      }));
      store.addWorkflow(document);
      store.select(['full-0']);
      store.setDock({open: false});
      const before = store.getState().document;
      const suggestions = makeRecommendationHarness(store, dataset);
      suggestions.toggle(true);
      suggestions.click(
        `${dataset.id}-${getSidePanelRecommendations(store, dataset.id).find(candidate => candidate.status !== 'blocked')!.id}`
      );
      expect(store.getState().document).toBe(before);
      expect(store.getState().document?.steps).toHaveLength(63);
      expect(store.getState().notice).toEqual({
        kind: 'error',
        message: 'Fixture workflow: Steps must not exceed maximumSteps (64).'
      });
      expect(store.getState().canUndo).toBe(false);
      expect(store.getState().selectedStepIds).toEqual(['full-0']);
      expect(store.getState().dock.open).toBe(false);
      expect(store.getState().palette.open).toBe(false);
    } finally {
      await store.clear();
    }
  });
});

describe('side panel quick starts and saved templates', () => {
  test('F2 refuses an oversized replay label before creating a workflow or opening the palette', async () => {
    const dataset = {...quickStartDataset, label: 'x'.repeat(121)};
    const store = makeQuickStartStore(dataset);
    const operation = store.registry.get('source-dataset')!;
    const normalize = vi.spyOn(operation, 'normalizeParameters');
    try {
      expect(getSidePanelDatasetSource(dataset).parameters?.['datasetLabel']).toBe(dataset.label);
      clickPanelControl(store, WORKBENCH_TEST_IDS.datasetQuickStart(dataset.id));
      const state = store.getState();
      expect(normalize.mock.results.find(result => result.type === 'throw')?.value).toMatchObject({
        code: 'parameter-out-of-range',
        details: {parameter: 'datasetLabel'},
        message: 'Dataset: Dataset label must be at most 120 characters; got 121.'
      });
      expect
        .soft({
          workflows: state.workflows.length,
          sourceStatus: state.runner?.steps['source-dataset']?.status ?? null,
          paletteOpen: state.palette.open,
          dockOpen: state.dock.open
        })
        .toEqual({workflows: 0, sourceStatus: null, paletteOpen: false, dockOpen: false});
      expect.soft(state.notice).toEqual({
        kind: 'error',
        message:
          'Dataset: Dataset label must be at most 120 characters; got 121. Rename the dataset to use a shorter label, then try again.'
      });
      expect(renderToStaticMarkup(createElement(WorkbenchSidePanel, {store}))).toContain(
        state.notice!.message
      );
    } finally {
      normalize.mockRestore();
      await store.clear();
    }
  });

  test('F2 refuses the same hidden-label problem for recommendations without editing an active workflow', async () => {
    const {store, dataset} = makeAnalysisRecommendationStore('events', 'x'.repeat(121));
    try {
      store.addWorkflow(makeFixtureDocument());
      store.select(['source']);
      store.setDock({open: false});
      const before = store.getState().document;
      const suggestions = makeRecommendationHarness(store, dataset);
      suggestions.toggle(true);
      suggestions.click(
        `${dataset.id}-${getSidePanelRecommendations(store, dataset.id).find(candidate => candidate.status !== 'blocked')!.id}`
      );
      expect(store.getState().document).toBe(before);
      expect(store.getState().selectedStepIds).toEqual(['source']);
      expect(store.getState().canUndo).toBe(false);
      expect(store.getState().dock.open).toBe(false);
      expect(store.getState().palette.open).toBe(false);
      expect(store.getState().notice).toEqual({
        kind: 'error',
        message:
          'Dataset: Dataset label must be at most 120 characters; got 121. Rename the dataset to use a shorter label, then try again.'
      });
    } finally {
      await store.clear();
    }
  });

  test('preserves a 120-character replay label exactly and resolves it after the host dataset id changes', async () => {
    const dataset = {...quickStartDataset, label: 'x'.repeat(120)};
    const store = makeQuickStartStore(dataset);
    const readDataset = vi.fn(async () => makeFixturePoints(4));
    store.host.readDataset = readDataset;
    try {
      clickPanelControl(store, WORKBENCH_TEST_IDS.datasetQuickStart(dataset.id));
      expect(store.getState().runner?.steps['source-dataset']?.status).toBe('ready');
      expect(store.getState().document?.steps[0].parameters['datasetLabel']).toEqual({
        from: 'static',
        value: dataset.label
      });
      expect(store.serialize()).toMatchObject({
        workflows: [{steps: [{parameters: {datasetLabel: {from: 'static', value: dataset.label}}}]}]
      });
      store.host.listDatasets = () => [{...dataset, id: 'replayed-observations'}];
      await store.run();
      expect(readDataset).toHaveBeenCalledExactlyOnceWith(
        {
          datasetId: 'replayed-observations',
          datasetLabel: dataset.label,
          geometry: {mode: 'geojson-field', field: 'geometry'},
          population: 'all',
          maximumRows: 20000
        },
        expect.any(AbortSignal)
      );
      expect(store.getState().runner?.steps['source-dataset']?.status).toBe('done');
    } finally {
      await store.clear();
    }
  });

  test('chooses the first geometry field, then point columns, then no geometry', () => {
    expect(getSidePanelDatasetSource(quickStartDataset)).toEqual({
      operationId: 'source-dataset',
      label: 'Observations',
      parameters: {
        dataset: 'observations',
        datasetLabel: 'Observations',
        geometryMode: 'geojson-field',
        geometryField: 'geometry'
      }
    });
    expect(
      getSidePanelDatasetSource({...quickStartDataset, geometryFields: []}).parameters
    ).toEqual({
      dataset: 'observations',
      datasetLabel: 'Observations',
      geometryMode: 'point-columns',
      longitudeField: 'longitude',
      latitudeField: 'latitude'
    });
    expect(
      getSidePanelDatasetSource({...quickStartDataset, geometryFields: [], pointColumns: undefined})
        .parameters
    ).toEqual({
      dataset: 'observations',
      datasetLabel: 'Observations',
      geometryMode: 'none'
    });
  });

  test('lists datasets and drawing guidance and collapses only once the workflow has steps', async () => {
    const store = makeQuickStartStore(quickStartDataset);
    try {
      let markup = renderToStaticMarkup(createElement(WorkbenchSidePanel, {store}));
      expect(markup).toContain('Start from data');
      expect(markup).toMatch(/<details open=""/);
      expect(markup).toContain('Observations');
      expect(markup).toContain('1,204 rows · resident');
      expect(markup).toContain(
        `data-testid="${WORKBENCH_TEST_IDS.datasetQuickStart('observations')}"`
      );
      expect(markup).toContain('aria-label="Use Observations in a workflow"');
      expect(markup).toContain('Draw a polygon on the map to use it as an input.');
      expect(markup).not.toContain(WORKBENCH_TEST_IDS.drawingQuickStart);
      store.addWorkflow(makeFixtureDocument());
      markup = renderToStaticMarkup(createElement(WorkbenchSidePanel, {store}));
      expect(markup).toMatch(/<details(?![^>]*\bopen\b)[^>]*>/);
      expect(markup.indexOf('Start from data')).toBeLessThan(
        markup.indexOf('aria-label="Workflow steps"')
      );
      const document = makeFixtureDocument();
      document.steps = [];
      store.addWorkflow(document);
      expect(renderToStaticMarkup(createElement(WorkbenchSidePanel, {store}))).toMatch(
        /<details open=""/
      );
      store.host.listDatasets = () => [{...quickStartDataset, resident: false}];
      store.host.listDrawings = () => [];
      markup = renderToStaticMarkup(createElement(WorkbenchSidePanel, {store}));
      expect(markup).toContain('1,204 rows · table');
      expect(markup).toContain('Draw a polygon on the map to use it as an input.');
    } finally {
      await store.clear();
    }
  });

  test('starts a dataset workflow with one undoable edit and offers analysis from its data output', async () => {
    const store = makeQuickStartStore(quickStartDataset);
    const edit = vi.spyOn(store, 'edit');
    try {
      clickPanelControl(store, WORKBENCH_TEST_IDS.datasetQuickStart('observations'));
      const state = store.getState();
      expect(state.document?.title).toBe('Observations');
      expect(state.document?.steps).toHaveLength(1);
      expect(state.document?.steps[0]).toMatchObject({
        id: 'source-dataset',
        operationId: 'source-dataset',
        label: 'Observations',
        parameters: {
          dataset: {from: 'static', value: 'observations'},
          datasetLabel: {from: 'static', value: 'Observations'},
          geometryMode: {from: 'static', value: 'geojson-field'},
          geometryField: {from: 'static', value: 'geometry'}
        }
      });
      expect(edit.mock.calls.map(([tag]) => tag)).toEqual(['quick-start:source-dataset']);
      expect(state.selectedStepIds).toEqual(['source-dataset']);
      expect(state.dock).toMatchObject({open: true, inspectorTab: 'parameters'});
      expect(state.palette).toEqual({
        open: true,
        connectFrom: {stepId: 'source-dataset', output: 'data'}
      });
      expect(state.runLog).toEqual([]);
      store.undo();
      expect(store.getState().document?.steps).toEqual([]);
      expect(store.getState().canUndo).toBe(false);
    } finally {
      edit.mockRestore();
      await store.clear();
    }
  });

  test('appends to the active workflow and leaves the interface alone when the step limit refuses it', async () => {
    const store = makeQuickStartStore(quickStartDataset);
    const create = vi.spyOn(store, 'createWorkflow');
    try {
      store.addWorkflow(makeFixtureDocument());
      const workflowId = store.getState().activeWorkflowId;
      clickPanelControl(store, WORKBENCH_TEST_IDS.datasetQuickStart('observations'));
      expect(create).not.toHaveBeenCalled();
      expect(store.getState().activeWorkflowId).toBe(workflowId);
      expect(store.getState().document?.title).toBe('Fixture workflow');
      expect(store.getState().document?.steps).toHaveLength(4);
      const document = makeFixtureDocument();
      document.steps = Array.from({length: WORKFLOW_LIMITS.maximumSteps}, (_, index) => ({
        ...document.steps[0],
        id: `full-${index}`
      }));
      store.addWorkflow(document);
      store.setDock({open: false});
      store.closePalette();
      store.select(['full-0']);
      const before = store.getState().document;
      clickPanelControl(store, WORKBENCH_TEST_IDS.datasetQuickStart('observations'));
      expect(store.getState().document).toBe(before);
      expect(store.getState().notice?.kind).toBe('error');
      expect(store.getState().notice?.message).toContain('maximumSteps (64)');
      expect(store.getState().selectedStepIds).toEqual(['full-0']);
      expect(store.getState().dock.open).toBe(false);
      expect(store.getState().palette.open).toBe(false);
    } finally {
      create.mockRestore();
      await store.clear();
    }
  });

  test('offers drawn shapes and creates a drawing source connected to the analysis palette', async () => {
    const store = makeQuickStartStore(quickStartDataset);
    store.host.listDrawings = () => [
      {id: 'shape-1', label: 'Area', kind: 'polygon'},
      {id: 'shape-2', label: 'Path', kind: 'line'}
    ];
    try {
      const markup = renderToStaticMarkup(createElement(WorkbenchSidePanel, {store}));
      expect(markup).toContain('2 drawn shapes');
      expect(markup).toContain('aria-label="Use drawn shapes in a workflow"');
      expect(markup).not.toContain('Draw a polygon on the map to use it as an input.');
      clickPanelControl(store, WORKBENCH_TEST_IDS.drawingQuickStart);
      const state = store.getState();
      expect(state.document?.title).toBe('Drawn shapes');
      expect(state.document?.steps).toHaveLength(1);
      expect(state.document?.steps[0]).toMatchObject({
        id: 'source-drawing',
        operationId: 'source-drawing',
        parameters: {shapes: {from: 'static', value: 'all'}}
      });
      expect(state.selectedStepIds).toEqual(['source-drawing']);
      expect(state.dock.open).toBe(true);
      expect(state.palette).toEqual({
        open: true,
        connectFrom: {stepId: 'source-drawing', output: 'data'}
      });
      store.undo();
      expect(store.getState().document?.steps).toEqual([]);
    } finally {
      await store.clear();
    }
  });

  test('lists saved templates below the built-ins and delegates their buttons to useTemplate', () => {
    const template = {...makeFixtureDocument(), id: 'daily-analysis', title: 'Daily analysis'};
    const {store, calls} = makeStore({
      workflows: [],
      document: null,
      activeWorkflowId: null,
      runner: null,
      templates: [template]
    });
    const markup = renderToStaticMarkup(createElement(WorkbenchSidePanel, {store}));
    expect(markup).toContain('Your templates');
    expect(markup).toContain(`data-testid="${WORKBENCH_TEST_IDS.userTemplate('daily-analysis')}"`);
    expect(markup.indexOf(WORKBENCH_TEST_IDS.template(listWorkflowTemplates()[0].id))).toBeLessThan(
      markup.indexOf(WORKBENCH_TEST_IDS.userTemplate('daily-analysis'))
    );
    clickPanelControl(store, WORKBENCH_TEST_IDS.userTemplate('daily-analysis'));
    expect(calls).toEqual([['useTemplate', 'daily-analysis']]);
  });
});

/** Preserves hook state and invokes the real field handlers, without simulating browser layout. */
function makeInputHarness(store: WorkbenchStore, inputId = 'distance') {
  const states: unknown[] = [];
  let stateIndex = 0;
  let rendering = false;
  let renderAgain = false;
  let tree: PanelElement;
  function renderField(): void {
    const mocks = [
      vi
        .mocked(ReactHooks.useSyncExternalStore)
        .mockImplementation((_subscribe, getSnapshot) => getSnapshot()),
      vi.mocked(ReactHooks.useMemo).mockImplementation(factory => factory()),
      vi.mocked(ReactHooks.useId).mockReturnValue('side-panel-field-test'),
      vi.mocked(ReactHooks.useState).mockImplementation(((initial: unknown) => {
        const index = stateIndex++;
        if (!(index in states)) states[index] = typeof initial === 'function' ? initial() : initial;
        return [
          states[index],
          (value: unknown) => {
            const next = typeof value === 'function' ? value(states[index]) : value;
            if (!Object.is(next, states[index])) {
              states[index] = next;
              if (rendering) renderAgain = true;
            }
          }
        ];
      }) as typeof react.useState)
    ];
    try {
      const panel = WorkbenchSidePanel({store});
      const field = findElement(panel, element => {
        const input = element.props['input'] as WorkflowInput | undefined;
        return input?.id === inputId;
      });
      const component = field.type as (props: {
        input: WorkflowInput;
        store: WorkbenchStore;
      }) => PanelElement;
      let renderCount = 0;
      do {
        renderAgain = false;
        stateIndex = 0;
        rendering = true;
        tree = component(field.props as {input: WorkflowInput; store: WorkbenchStore});
        rendering = false;
        expect(++renderCount).toBeLessThan(10);
      } while (renderAgain);
    } finally {
      rendering = false;
      for (const mock of mocks) mock.mockReset();
    }
  }
  const control = () =>
    findElement(
      tree,
      element => element.props['data-testid'] === WORKBENCH_TEST_IDS.workflowInput(inputId)
    );
  renderField();
  return {
    render: renderField,
    control,
    markup: () => renderToStaticMarkup(tree),
    change(value: string | boolean) {
      const handler = control().props['onChange'] as (event: {
        currentTarget: {value: string; checked: boolean};
      }) => void;
      handler({currentTarget: {value: String(value), checked: value === true}});
      renderField();
    },
    blur() {
      const handler = control().props['onBlur'] as () => void;
      handler();
      renderField();
    }
  };
}

function makeBoundInputStore() {
  const document = makeFixtureDocument();
  document.inputs = [{...distanceInput, maximum: 100}];
  document.steps[1].parameters['factor'] = {from: 'input', inputId: 'distance'};
  const store = createWorkbenchStore({host: makeFixtureHost(), registry: makeFixtureRegistry()});
  store.addWorkflow(document);
  return store;
}

describe('side panel review regressions', () => {
  test('F1 distinguishes a display request from confirmed publication', async () => {
    const store = createWorkbenchStore({host: makeFixtureHost(), registry: makeFixtureRegistry()});
    try {
      store.addWorkflow(makeFixtureDocument());
      let markup = renderToStaticMarkup(createElement(WorkbenchSidePanel, {store}));
      expect(markup).not.toContain('aria-label="Displayed on map"');
      expect(markup).toContain('Display requested');
      await store.run();
      expect(store.getState().runner?.steps['scale'].published?.['scaled']).toEqual({
        datasetId: 'published-1',
        label: 'Fixture scale · Scaled',
        key: store.getState().runner?.steps['scale'].key
      });
      markup = renderToStaticMarkup(createElement(WorkbenchSidePanel, {store}));
      expect(markup).toContain('aria-label="Displayed on map"');
      expect(markup).not.toContain('Display requested');
    } finally {
      await store.clear();
    }
  });

  test('F1 exposes a rejected publication without claiming the output is displayed', async () => {
    const host = makeFixtureHost();
    host.publishOutput = async () => {
      throw new Error('Map refused output');
    };
    const store = createWorkbenchStore({host, registry: makeFixtureRegistry()});
    try {
      store.addWorkflow(makeFixtureDocument());
      await store.run();
      const state = store.getState().runner?.steps['scale'];
      const message = 'Fixture scale: could not display Scaled: Map refused output';
      expect(state?.status).toBe('done');
      expect(state?.published).toBeUndefined();
      expect(state?.issues).toContainEqual({
        severity: 'warning',
        code: 'publish-failed',
        message,
        stepId: 'scale',
        port: 'scaled'
      });
      const markup = renderToStaticMarkup(createElement(WorkbenchSidePanel, {store}));
      expect(markup).not.toContain('aria-label="Displayed on map"');
      expect(markup).toContain(message);
      expect(markup).toContain('Display requested');
      expect(markup).toMatch(/aria-describedby="[^"]*-scale-publication-0"/);
      expect(markup).toMatch(/id="[^"]*-scale-publication-0" aria-live="polite"/);
    } finally {
      await store.clear();
    }
  });

  test('F2 reconciles a typed value with an external edit and the value actually executed', async () => {
    const store = makeBoundInputStore();
    try {
      const field = makeInputHarness(store);
      field.change('60');
      expect(field.control().props['value']).toBe('60');
      expect(store.getState().document?.inputs[0].value).toBe(60);
      expect(
        store.edit('other-view', document =>
          updateWorkflowInput(document, 'distance', {value: 100})
        )
      ).toBe(true);
      field.render();
      await store.run();
      expect(store.getState().runner?.steps['scale'].result?.parameters['factor']).toBe(100);
      expect(field.control().props['value']).toBe('100');
    } finally {
      await store.clear();
    }
  });

  test('F2 coalesces real input handler edits and synchronizes undo and redo before blur', async () => {
    const clock = vi.spyOn(Date, 'now').mockReturnValue(100);
    const store = makeBoundInputStore();
    const edit = vi.spyOn(store, 'edit');
    try {
      const field = makeInputHarness(store);
      field.change('6');
      field.change('060');
      expect(field.control().props['value']).toBe('060');
      expect(edit.mock.calls.map(([tag]) => tag)).toEqual(['input:distance', 'input:distance']);
      expect(store.getState().document?.inputs[0].value).toBe(60);
      store.undo();
      field.render();
      expect(store.getState().document?.inputs[0].value).toBe(50);
      expect(store.getState().canUndo).toBe(false);
      expect(field.control().props['value']).toBe('50');
      store.redo();
      field.render();
      expect(field.control().props['value']).toBe('60');
      store.undo();
      field.render();
      expect(field.control().props['value']).toBe('50');
    } finally {
      edit.mockRestore();
      clock.mockRestore();
      await store.clear();
    }
  });

  test('real field handlers retain incomplete text and show store refusals without committing it', async () => {
    const store = makeBoundInputStore();
    const errors: unknown[] = [];
    const editDocument = store.edit;
    const edit = vi.spyOn(store, 'edit').mockImplementation((tag, change) =>
      editDocument(tag, document => {
        try {
          return change(document);
        } catch (error) {
          errors.push(error);
          throw error;
        }
      })
    );
    try {
      const field = makeInputHarness(store);
      const original = store.getState().document;
      field.change('');
      expect(field.control().props['value']).toBe('');
      expect(field.control().props['aria-invalid']).toBe(true);
      expect(field.markup()).toContain('Distance: Enter a finite number.');
      expect(store.getState().document).toBe(original);
      expect(store.getState().notice).toEqual({
        kind: 'error',
        message: 'Distance: Value must be a finite number.'
      });
      field.change('101');
      expect(field.control().props['value']).toBe('101');
      expect(field.markup()).toContain(
        'Distance: Value must be between 0 and 100 meters; got 101.'
      );
      expect(store.getState().document).toBe(original);
      expect(store.getState().notice).toEqual({
        kind: 'error',
        message: 'Distance: Value must be at most 100; got 101.'
      });
      expect(errors).toHaveLength(2);
      for (const error of errors) {
        expect(error).toBeInstanceOf(WorkflowError);
        expect((error as WorkflowError).code).toBe('parameter-out-of-range');
      }
      expect(store.getState().canUndo).toBe(false);
      store.setDock({open: false});
      field.render();
      expect(field.control().props['value']).toBe('101');
      field.blur();
      expect(field.control().props['value']).toBe('50');
      field.change('75');
      expect(store.getState().document?.inputs[0].value).toBe(75);
      expect(store.getState().notice).toBeNull();
      expect(field.control().props['aria-invalid']).toBe(false);
    } finally {
      edit.mockRestore();
      await store.clear();
    }
  });

  test('commits boolean, enum, integer, string and list values through their actual handlers', async () => {
    const cases: Array<{
      input: WorkflowInput;
      typed: string | boolean;
      expected: WorkflowInput['value'];
    }> = [
      {
        input: {id: 'enabled', label: 'Enabled', kind: 'boolean', value: false},
        typed: true,
        expected: true
      },
      {
        input: {id: 'mode', label: 'Mode', kind: 'enum', value: 'all', options: ['all', 'nearest']},
        typed: 'nearest',
        expected: 'nearest'
      },
      {
        input: {id: 'count', label: 'Count', kind: 'integer', value: 2, minimum: 1, maximum: 10},
        typed: '3',
        expected: 3
      },
      {
        input: {id: 'name', label: 'Name', kind: 'string', value: 'Initial'},
        typed: 'Changed',
        expected: 'Changed'
      },
      {
        input: {id: 'weights', label: 'Weights', kind: 'number-list', value: [1, 2]},
        typed: '3, 4, 5',
        expected: [3, 4, 5]
      }
    ];
    const document = makeFixtureDocument();
    document.inputs = cases.map(entry => entry.input);
    const store = createWorkbenchStore({host: makeFixtureHost(), registry: makeFixtureRegistry()});
    try {
      store.addWorkflow(document);
      for (const entry of cases) {
        const field = makeInputHarness(store, entry.input.id);
        field.change(entry.typed);
        expect(
          store.getState().document?.inputs.find(input => input.id === entry.input.id)?.value
        ).toEqual(entry.expected);
        expect(field.control().props[entry.input.kind === 'boolean' ? 'checked' : 'value']).toBe(
          entry.typed
        );
        expect(store.getState().notice).toBeNull();
      }
    } finally {
      await store.clear();
    }
  });

  test('preserves partial list text across unrelated document edits and replaces it on an external value edit', async () => {
    const document = makeFixtureDocument();
    document.inputs = [{id: 'weights', label: 'Weights', kind: 'number-list', value: [1, 2]}];
    const store = createWorkbenchStore({host: makeFixtureHost(), registry: makeFixtureRegistry()});
    try {
      store.addWorkflow(document);
      const field = makeInputHarness(store, 'weights');
      field.change('3,');
      expect(field.control().props['value']).toBe('3,');
      expect(store.getState().document?.inputs[0].value).toEqual([1, 2]);
      store.edit('other-view', current =>
        updateWorkflowInput(current, 'weights', {label: 'New weights'})
      );
      field.render();
      expect(field.control().props['value']).toBe('3,');
      store.edit('other-view', current => updateWorkflowInput(current, 'weights', {value: [8, 9]}));
      field.render();
      expect(field.control().props['value']).toBe('8, 9');
      expect(field.control().props['aria-invalid']).toBe(false);
    } finally {
      await store.clear();
    }
  });

  test('shows a publication warning that belongs to the whole workflow', () => {
    const runner = makeRunner();
    const message =
      'Fixture workflow: could not retract displayed dataset old-output: Map refused removal';
    runner.issues = [{severity: 'warning', code: 'publish-failed', message}];
    expect(render({runner}).markup).toContain(message);
  });
});

describe('workbench side panel', () => {
  test('lists template titles and summaries with no workflows, without changing the store', () => {
    const {markup, calls} = render({
      workflows: [],
      activeWorkflowId: null,
      document: null,
      runner: null
    });
    expect(markup).toContain(`data-testid="${WORKBENCH_TEST_IDS.sidePanel}"`);
    expect(markup).toContain('Workflows');
    expect(markup).toContain(
      'Chain analysis operations and re-run them when data or settings change.'
    );
    expect(markup).toContain('New workflow');
    expect(markup).toContain('More templates');
    for (const template of listWorkflowTemplates().slice(0, 3)) {
      expect(markup).toContain(`data-testid="${WORKBENCH_TEST_IDS.template(template.id)}"`);
      expect(markup).toContain(template.title);
      expect(markup).toContain(template.summary);
    }
    expect(markup).not.toContain(`data-testid="${WORKBENCH_TEST_IDS.runAll}"`);
    expect(calls).toEqual([]);
  });

  test('renders the chooser, runner order, readable statuses and pending display request', () => {
    const runner = makeRunner();
    runner.order = ['count', 'source', 'scale'];
    const {markup} = render({runner});
    expect(markup).toContain('aria-label="Workflow"');
    expect(markup).toContain('Run 3 steps');
    expect(markup).toContain('role="list" aria-label="Workflow steps"');
    expect(markup.indexOf(WORKBENCH_TEST_IDS.stepRow('count'))).toBeLessThan(
      markup.indexOf(WORKBENCH_TEST_IDS.stepRow('source'))
    );
    expect(markup.indexOf(WORKBENCH_TEST_IDS.stepRow('source'))).toBeLessThan(
      markup.indexOf(WORKBENCH_TEST_IDS.stepRow('scale'))
    );
    expect(markup).toContain('data-status="ready"');
    expect(markup).toContain('Ready');
    expect(markup).toContain('aria-label="Select Fixture scale"');
    expect(markup).toContain('Display requested');
    expect(markup).not.toContain('aria-label="Displayed on map"');
    expect(markup).toContain('aria-label="Run to here: Fixture scale"');
    expect(markup).toContain('Open workbench');
  });

  test('marks only the selected row and reveals the operation title of a renamed step', () => {
    const document = makeFixtureDocument();
    document.steps[1].label = 'Scale values';
    const {markup} = render({document, workflows: [document], selectedStepIds: ['scale']});
    expect(markup).toMatch(/data-testid="workflow-step-row-scale"[^>]*aria-current="true"/);
    expect(markup.match(/aria-current="true"/g)).toHaveLength(1);
    expect(markup).toContain('Scale values');
    expect(markup).toContain('Fixture scale');
  });

  test('announces progress and presents Cancel during a run', () => {
    const runner = {
      ...makeRunner(['done', 'running', 'queued']),
      running: true,
      runningStepId: 'scale'
    };
    const {markup} = render({
      runner,
      dock: {open: true, height: 380, inspectorTab: 'parameters', logOpen: false}
    });
    expect(markup).toContain(`data-testid="${WORKBENCH_TEST_IDS.cancel}"`);
    expect(markup).toContain('Running Fixture scale');
    expect(markup).toContain('data-status="running"');
    expect(markup).toContain('Queued');
    expect(markup).toContain('aria-live="polite" role="status"');
    expect(markup).toMatch(/data-testid="workflow-run-all"[^>]*disabled=""/);
    expect(markup).toMatch(/data-testid="workflow-run-to-here"[^>]*disabled=""/);
    expect(markup).toContain('Hide workbench');
  });

  test('shows a failed step error below the row and the first last-run error', () => {
    const runner = makeRunner(['done', 'error', 'blocked']);
    const error = {
      severity: 'error' as const,
      code: 'operation-failed' as const,
      message: 'Fixture scale: Factor must be at most 100; got 101.'
    };
    runner.steps['scale'].issues = [
      {severity: 'warning', code: 'gpu-fallback', message: 'Fallback warning'},
      error
    ];
    runner.steps['scale'].error = error;
    runner.lastRun = makeRecord({
      status: 'error',
      steps: [
        {
          stepId: 'scale',
          operationId: 'fixture-scale',
          key: 'scale',
          status: 'error',
          cacheHit: false,
          durationMilliseconds: 42,
          inputs: [],
          error
        }
      ]
    });
    const {markup} = render({runner});
    expect(markup).toContain('data-status="error"');
    expect(markup).toContain('Error');
    expect(markup).toContain(`Failed: ${error.message}`);
    const rowMarkup = markup.slice(markup.indexOf(WORKBENCH_TEST_IDS.stepRow('scale')));
    expect(rowMarkup).toMatch(
      new RegExp(`</button><div[^>]*aria-live="polite"[^>]*>${error.message}`)
    );
    expect(rowMarkup).not.toContain('Fallback warning');
  });

  test('shows done row counts but never presents stale counts as current', () => {
    const runner = makeRunner(['done', 'done', 'done']);
    runner.steps['scale'].result = makeResult();
    runner.lastRun = makeRecord();
    let markup = render({runner}).markup;
    expect(markup).toContain('1,204 rows');
    expect(markup).toContain('Done in 1.4 s');
    expect(markup).toContain('Up to date');
    expect(markup).toMatch(/data-testid="workflow-run-all"[^>]*disabled=""/);
    runner.steps['scale'].status = 'stale';
    runner.steps['scale'].key = 'new-scale';
    markup = render({runner}).markup;
    expect(markup).toContain('data-status="stale"');
    expect(markup).toContain('Stale');
    expect(markup).toContain('Run 1 step');
    expect(markup).not.toContain('1,204 rows');
  });

  test('renders every input kind with labels, units and numeric limits', () => {
    const document = makeFixtureDocument();
    document.inputs = [
      distanceInput,
      {id: 'count', label: 'Count', kind: 'integer', value: 4, minimum: 1, maximum: 50},
      {id: 'mode', label: 'Mode', kind: 'enum', value: 'nearest', options: ['nearest', 'all']},
      {id: 'enabled', label: 'Enabled', kind: 'boolean', value: true},
      {id: 'name', label: 'Name', kind: 'string', value: 'Observations'},
      {id: 'weights', label: 'Weights', kind: 'number-list', value: [1, 2, 3]}
    ];
    const {markup} = render({document, workflows: [document]});
    for (const input of document.inputs) {
      expect(markup).toContain(`data-testid="${WORKBENCH_TEST_IDS.workflowInput(input.id)}"`);
      expect(markup).toContain(input.label);
    }
    expect(markup).toContain('meters');
    expect(markup).toContain('type="number"');
    expect(markup).toContain('min="0" max="1000" step="any"');
    expect(markup).toContain('step="1"');
    expect(markup).toContain('<option value="nearest" selected="">nearest</option>');
    expect(markup).toContain('type="checkbox" checked=""');
    expect(markup).toContain('value="1, 2, 3"');
  });

  test('associates restored input range problems with the affected field', () => {
    const document = makeFixtureDocument();
    document.inputs = [{...distanceInput, value: -1}];
    const {markup} = render({document});
    expect(markup).toContain('aria-invalid="true"');
    expect(markup).toMatch(/aria-describedby="[^"]+-problem"/);
    expect(markup).toContain('Distance: Value must be between 0 and 1,000 meters; got -1.');
  });

  test('handles an empty workflow, a missing runner and an unknown operation', () => {
    const document = makeFixtureDocument();
    document.steps = [];
    expect(render({document, runner: {...makeRunner(), order: [], steps: {}}}).markup).toContain(
      'No steps yet. Open the workbench to add a step.'
    );
    document.steps = [{...makeFixtureDocument().steps[0], operationId: 'removed-operation'}];
    const {markup} = render({document, runner: null});
    expect(markup).toContain('removed-operation');
    expect(markup).toContain('Unknown operation');
    expect(markup).toContain('data-status="not-configured"');
    expect(markup).toContain('Not configured');
    expect(markup).toMatch(/data-testid="workflow-run-to-here"[^>]*disabled=""/);
    expect(markup).toMatch(/data-testid="workflow-run-all"[^>]*disabled=""/);
  });
});

describe('side panel model helpers', () => {
  test('distinguishes requested, current, partial and previous map publication', () => {
    const step = makeFixtureDocument().steps[1];
    const state = makeRunner(['done', 'done', 'done']).steps['scale'];
    expect(getSidePanelDisplayLabel(step, undefined)).toBe('Display requested');
    expect(getSidePanelDisplayLabel({...step, display: []}, state)).toBeNull();
    state.published = {scaled: {datasetId: 'map-output', label: 'Scaled', key: 'scale'}};
    expect(getSidePanelDisplayLabel(step, state)).toBe('Displayed on map');
    expect(getSidePanelDisplayLabel({...step, display: ['scaled', 'other']}, state)).toBe(
      'Some outputs displayed on map'
    );
    expect(getSidePanelDisplayLabel({...step, display: ['other']}, state)).toBe(
      'Display requested'
    );
    state.key = 'new-scale';
    state.status = 'stale';
    expect(getSidePanelDisplayLabel(step, state)).toBe('Previous result displayed on map');
    const markup = render({runner: {...makeRunner(), steps: {scale: state}}}).markup;
    expect(markup).toContain('aria-label="Previous result displayed on map"');
    expect(markup).not.toContain('aria-label="Displayed on map"');
  });

  test('counts runnable work only and uses singular, busy and absent-runner labels', () => {
    expect(getSidePanelRunLabel(null)).toBe('Run');
    expect(getSidePanelRunLabel(makeRunner(['ready', 'stale', 'error']))).toBe('Run 3 steps');
    expect(getSidePanelRunLabel(makeRunner(['done', 'ready', 'disabled']))).toBe('Run 1 step');
    expect(getSidePanelRunLabel(makeRunner(['not-configured', 'blocked', 'done']))).toBe(
      'Up to date'
    );
    expect(getSidePanelRunLabel({...makeRunner(), running: true})).toBe('Running');
    expect(getSidePanelRunLabel({...makeRunner(), order: []})).toBe('Up to date');
  });

  test('orders steps without dropping unplanned placeholders or duplicating runner ids', () => {
    const document = makeFixtureDocument();
    const runner = {...makeRunner(), order: ['count', 'missing', 'source', 'source']};
    expect(getSidePanelSteps(document, runner).map(step => step.id)).toEqual([
      'count',
      'source',
      'scale'
    ]);
    expect(getSidePanelSteps(document, null).map(step => step.id)).toEqual([
      'source',
      'scale',
      'count'
    ]);
    expect(getSidePanelSteps(null, runner)).toEqual([]);
    expect(document.steps.map(step => step.id)).toEqual(['source', 'scale', 'count']);
  });

  test('formats durations and last-run outcomes exactly', () => {
    expect([0, 42, 999, 1000, 1750].map(formatSidePanelDuration)).toEqual([
      '0 ms',
      '42 ms',
      '999 ms',
      '1.0 s',
      '1.8 s'
    ]);
    expect(getSidePanelRunSummary(undefined)).toBe('');
    expect(getSidePanelRunSummary(makeRecord())).toBe('Done in 1.4 s');
    expect(
      getSidePanelRunSummary(makeRecord({status: 'cancelled', durationMilliseconds: 42}))
    ).toBe('Cancelled in 42 ms');
    expect(getSidePanelRunSummary(makeRecord({status: 'error'}))).toBe(
      'Failed: Workflow could not run.'
    );
  });

  test('preserves empty and malformed numeric tokens as invalid values', () => {
    expect(parseSidePanelInputText(distanceInput, '12.5')).toBe(12.5);
    expect(parseSidePanelInputText(distanceInput, '')).toBeNaN();
    expect(parseSidePanelInputText(distanceInput, '-')).toBeNaN();
    expect(parseSidePanelInputText({...distanceInput, kind: 'number-list'}, '1, 2, 3')).toEqual([
      1, 2, 3
    ]);
    expect(parseSidePanelInputText({...distanceInput, kind: 'number-list'}, '1,,3')).toEqual([
      1,
      Number.NaN,
      3
    ]);
    expect(parseSidePanelInputText({...distanceInput, kind: 'string'}, 'name')).toBe('name');
  });

  test('checks inclusive ranges, finite numbers, integer values and every list member', () => {
    expect(getSidePanelInputProblem(distanceInput, 0)).toBeNull();
    expect(getSidePanelInputProblem(distanceInput, 1000)).toBeNull();
    expect(getSidePanelInputProblem(distanceInput, -1)).toBe(
      'Distance: Value must be between 0 and 1,000 meters; got -1.'
    );
    expect(getSidePanelInputProblem(distanceInput, 1001)).toBe(
      'Distance: Value must be between 0 and 1,000 meters; got 1,001.'
    );
    expect(getSidePanelInputProblem(distanceInput, Number.POSITIVE_INFINITY)).toBe(
      'Distance: Enter a finite number.'
    );
    expect(getSidePanelInputProblem({...distanceInput, kind: 'integer'}, 1.5)).toBe(
      'Distance: Enter a whole number; got 1.5.'
    );
    expect(getSidePanelInputProblem({...distanceInput, kind: 'number-list'}, [])).toBe(
      'Distance: Enter a non-empty list of finite numbers.'
    );
    expect(getSidePanelInputProblem({...distanceInput, kind: 'number-list'}, [1, -1])).toBe(
      'Distance: Value must be between 0 and 1,000 meters; got -1.'
    );
    expect(getSidePanelInputProblem({...distanceInput, kind: 'number-list'}, [1, 1000])).toBeNull();
    expect(getSidePanelInputProblem({...distanceInput, maximum: undefined}, -1)).toBe(
      'Distance: Value must be at least 0 meters; got -1.'
    );
    expect(getSidePanelInputProblem({...distanceInput, minimum: undefined}, 1001)).toBe(
      'Distance: Value must be at most 1,000 meters; got 1,001.'
    );
  });

  test('reports boolean, enum and string type problems', () => {
    expect(getSidePanelInputProblem({...distanceInput, kind: 'boolean'}, false)).toBeNull();
    expect(getSidePanelInputProblem({...distanceInput, kind: 'boolean'}, 'false')).toBe(
      'Distance: Choose on or off.'
    );
    expect(
      getSidePanelInputProblem(
        {...distanceInput, kind: 'enum', options: ['nearest', 'all']},
        'other'
      )
    ).toBe('Distance: Choose one of nearest, all.');
    expect(
      getSidePanelInputProblem({...distanceInput, kind: 'enum', options: ['nearest', 'all']}, 'all')
    ).toBeNull();
    expect(getSidePanelInputProblem({...distanceInput, kind: 'enum', options: ['1']}, 1)).toBe(
      'Distance: Choose one of 1.'
    );
    expect(getSidePanelInputProblem({...distanceInput, kind: 'string'}, 1)).toBe(
      'Distance: Enter text.'
    );
  });
});
