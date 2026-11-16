// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {afterAll, describe, expect, test, vi} from 'vitest';
import * as ReactHooks from 'react';
import type {DependencyList, EffectCallback, KeyboardEvent, ReactElement} from 'react';
import {
  addPaletteOperation,
  getPaletteActiveDatasetId,
  openPaletteAnalysis,
  subscribePaletteShortcut,
  WorkbenchAnalysisPaletteDialog,
  getPaletteCategoryCounts,
  getPaletteConnection,
  getPalettePortLine,
  groupPaletteOperations,
  movePaletteHighlight,
  updatePaletteRecentOperations,
  WorkbenchPalette,
  WorkbenchPaletteDialog
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-palette';
import {
  WORKBENCH_PORT_COLORS,
  WORKBENCH_TEST_IDS
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-contract';
import type {
  WorkbenchState,
  WorkbenchStore
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-store';
import {searchWorkflowOperations} from '../../../src/workflows/workflow-operations';
import {createWorkbenchAnalysisDocument} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-analysis-launch';
import * as WorkbenchAnalysisLaunch from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-analysis-launch';
import {
  getWorkflowOperationRegistry,
  getKeplerCuratedAnalysis,
  profileKeplerFeatureCollection,
  searchWorkflowAnalyses,
  type KeplerDatasetProfile,
  type WorkflowRunRecord
} from '@luma-contributors/kepler';
import {
  WorkflowError,
  WORKFLOW_LIMITS,
  type WorkflowOperationRegistry,
  type WorkflowPortKind,
  type WorkflowRunnerState
} from '../../../src/workflows/workflow-types';
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
    useState: vi.fn(original.useState),
    useRef: vi.fn(original.useRef),
    useEffect: vi.fn(original.useEffect)
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
const {ServerStyleSheet} = hostRequire('styled-components') as typeof import('styled-components');

function makeStore(
  patch: Partial<WorkbenchState> = {},
  registry: WorkflowOperationRegistry = makeFixtureRegistry()
) {
  const document = makeFixtureDocument();
  let state: WorkbenchState = {
    revision: 1,
    workflows: [document],
    activeWorkflowId: document.id,
    document,
    runner: null,
    selectedStepIds: [],
    canUndo: false,
    canRedo: false,
    dock: {open: true, height: 380, inspectorTab: 'result', logOpen: false},
    palette: {open: true},
    dialog: null,
    draft: null,
    notice: null,
    runLog: [],
    templates: [],
    clipboardStepCount: 0,
    ...patch
  };
  const calls: string[] = [];
  const errors: unknown[] = [];
  const store: WorkbenchStore = {
    registry,
    host: makeFixtureHost(),
    getState: () => state,
    subscribe: () => () => {},
    createWorkflow: () => '',
    addWorkflow: () => '',
    duplicateWorkflow: () => '',
    deleteWorkflow: () => {},
    setActiveWorkflow: () => {},
    edit(tag, change) {
      calls.push(`edit:${tag}`);
      try {
        if (!state.document) throw new Error('Workflows: select a workflow before editing.');
        state = {...state, document: change(state.document)};
        return true;
      } catch (error) {
        errors.push(error);
        return false;
      }
    },
    undo: () => {},
    redo: () => {},
    select: stepIds => calls.push(`select:${stepIds.join(',')}`),
    run: async () => null,
    cancel: () => {},
    notifyExternalChange: () => {},
    getRunner: () => null,
    setDock: patch => calls.push(`dock:${patch.inspectorTab}`),
    openPalette: () => {},
    closePalette: () => calls.push('close'),
    setDialog: () => {},
    setNotice: () => {},
    proposeDraft: () => {},
    acceptDraft: () => null,
    rejectDraft: () => {},
    saveTemplate: () => '',
    deleteTemplate: () => {},
    useTemplate: () => '',
    insertWorkflow: () => false,
    copySelection: () => false,
    cutSelection: () => false,
    paste: () => false,
    serialize: () => undefined,
    restore: () => {},
    getReviewedAssistantIssues: () => [],
    recordAssistantReview: () => {},
    clear: async () => {}
  };
  return {store, calls, errors};
}

function render(store: WorkbenchStore, query?: string): string {
  return renderToStaticMarkup(
    query === undefined
      ? createElement(WorkbenchPalette, {store})
      : createElement(WorkbenchPaletteDialog, {store, initialQuery: query})
  );
}

function renderWithStyles(store: WorkbenchStore): {markup: string; styles: string} {
  const stylesheet = new ServerStyleSheet();
  try {
    const markup = renderToStaticMarkup(
      stylesheet.collectStyles(createElement(WorkbenchPalette, {store}))
    );
    return {markup, styles: stylesheet.getStyleTags()};
  } finally {
    stylesheet.seal();
  }
}

function readElementStyles(tag: string, styles: string): string {
  const classes = tag.match(/class="([^"]+)"/)?.[1].split(' ') ?? [];
  expect(classes.length).toBeGreaterThan(0);
  return classes
    .flatMap(className => [...styles.matchAll(new RegExp(`\\.${className}\\{([^{}]*)\\}`, 'g'))])
    .map(match => match[1])
    .join('');
}

function makeRunner(
  kind: 'point' | 'polygon',
  status: 'done' | 'stale' | 'running'
): WorkflowRunnerState {
  const result = makeFixturePoints(2);
  return {
    revision: 1,
    running: status === 'running',
    runningStepId: status === 'running' ? 'source' : undefined,
    order: ['source'],
    issues: [],
    steps: {
      source: {
        key: 'current',
        status,
        issues: [],
        result: {
          key: status === 'stale' ? 'older' : 'current',
          outputs: {data: {...result, kind}},
          diagnostics: [],
          summary: {},
          engine: 'host',
          dispatchedContributors: [],
          parameters: {},
          durationMilliseconds: 0,
          cacheHit: false,
          completedAt: 0
        }
      }
    }
  };
}

type PaletteElement = ReactElement<Record<string, unknown>>;

function findPaletteElement(tree: unknown, property: string, value: unknown): PaletteElement {
  const matches: PaletteElement[] = [];
  function visit(child: unknown): void {
    if (Array.isArray(child)) {
      child.forEach(visit);
    } else if (react.isValidElement<Record<string, unknown>>(child)) {
      if (child.props[property] === value) matches.push(child);
      visit(child.props['children']);
    }
  }
  visit(tree);
  expect(matches).toHaveLength(1);
  return matches[0];
}

/** Runs the component's handlers and effects without claiming to simulate a browser renderer. */
function makePaletteHarness(
  store: WorkbenchStore,
  component: typeof WorkbenchPaletteDialog = WorkbenchPaletteDialog,
  initialQuery = 'fixture'
) {
  const states: unknown[] = [];
  const references: Array<{current: unknown}> = [];
  const effects: Array<{
    callback: EffectCallback;
    dependencies?: DependencyList;
    cleanup?: ReturnType<EffectCallback>;
    pending: boolean;
  }> = [];
  let tree: PaletteElement;
  function renderComponent(): void {
    let stateIndex = 0;
    let referenceIndex = 0;
    let effectIndex = 0;
    const spies = [
      vi
        .mocked(ReactHooks.useSyncExternalStore)
        .mockImplementation((_subscribe, getSnapshot) => getSnapshot()),
      vi.mocked(ReactHooks.useMemo).mockImplementation(factory => factory()),
      vi.mocked(ReactHooks.useId).mockReturnValue('palette-handler-test'),
      vi.mocked(ReactHooks.useState).mockImplementation(((initialValue: unknown) => {
        const index = stateIndex++;
        if (!(index in states))
          states[index] = typeof initialValue === 'function' ? initialValue() : initialValue;
        return [
          states[index],
          (value: unknown) => {
            states[index] = typeof value === 'function' ? value(states[index]) : value;
          }
        ];
      }) as typeof react.useState),
      vi.mocked(ReactHooks.useRef).mockImplementation(((initialValue: unknown) => {
        const index = referenceIndex++;
        references[index] ??= {current: initialValue};
        return references[index];
      }) as typeof react.useRef),
      vi.mocked(ReactHooks.useEffect).mockImplementation((callback, dependencies) => {
        const index = effectIndex++;
        const previous = effects[index];
        if (
          !previous ||
          !dependencies ||
          !previous.dependencies ||
          dependencies.some((value, offset) => !Object.is(value, previous.dependencies?.[offset]))
        ) {
          effects[index] = {callback, dependencies, cleanup: previous?.cleanup, pending: true};
        }
      })
    ];
    try {
      tree = component({store, initialQuery}) as PaletteElement;
    } finally {
      for (const spy of spies) spy.mockReset();
    }
  }
  renderComponent();
  function getSearch() {
    return findPaletteElement(tree, 'data-testid', WORKBENCH_TEST_IDS.paletteSearch);
  }
  return {
    render: renderComponent,
    find: (property: string, value: unknown) => findPaletteElement(tree, property, value),
    getSearch,
    getDialog: () => findPaletteElement(tree, 'role', 'dialog'),
    getHighlight: () => getSearch().props['aria-activedescendant'],
    pressKey(key: string, isComposing = false) {
      const event = {
        key,
        nativeEvent: {isComposing},
        preventDefault: vi.fn(),
        stopPropagation: vi.fn()
      };
      const searchHandler = getSearch().props['onKeyDown'] as (
        event: KeyboardEvent<HTMLInputElement>
      ) => void;
      const dialogHandler = findPaletteElement(tree, 'role', 'dialog').props['onKeyDown'] as (
        event: KeyboardEvent<HTMLDivElement>
      ) => void;
      searchHandler(event as unknown as KeyboardEvent<HTMLInputElement>);
      dialogHandler(event as unknown as KeyboardEvent<HTMLDivElement>);
      renderComponent();
      return event;
    },
    flushEffects() {
      for (const effect of effects) {
        if (!effect.pending) continue;
        effect.cleanup?.();
        effect.cleanup = effect.callback();
        effect.pending = false;
      }
    },
    unmount() {
      for (const effect of effects) effect.cleanup?.();
    }
  };
}

function makePaletteDocument() {
  class ElementDouble {
    ownerDocument: unknown;
    isConnected = true;
    focus = vi.fn();
    querySelector = vi.fn((_selector: string): ElementDouble | null => null);
    contains = vi.fn((target: unknown) => target === this);
  }
  const opener = new ElementDouble();
  const canvas = new ElementDouble();
  const search = new ElementDouble();
  const palette = new ElementDouble();
  const scrollIntoView = vi.fn();
  const listeners = new Map<string, (event: {target: unknown}) => void>();
  const ownerDocument = {
    activeElement: opener,
    body: new ElementDouble(),
    documentElement: new ElementDouble(),
    querySelector: vi.fn((_selector: string): ElementDouble | null => canvas),
    getElementById: vi.fn((_id: string) => ({scrollIntoView})),
    addEventListener: vi.fn((name: string, listener: (event: {target: unknown}) => void) => {
      listeners.set(name, listener);
    }),
    removeEventListener: vi.fn((name: string, listener: (event: {target: unknown}) => void) => {
      expect(listeners.get(name)).toBe(listener);
      listeners.delete(name);
    })
  };
  palette.ownerDocument = ownerDocument;
  palette.contains.mockImplementation(target => target === palette || target === search);
  return {ElementDouble, opener, canvas, search, palette, scrollIntoView, listeners, ownerDocument};
}

function attachPaletteReferences(
  harness: ReturnType<typeof makePaletteHarness>,
  documentDouble: ReturnType<typeof makePaletteDocument>
): void {
  const paletteReference = harness.getDialog().props['ref'] as {current: unknown};
  const searchReference = harness.getSearch().props['ref'] as {current: unknown};
  paletteReference.current = documentDouble.palette;
  searchReference.current = documentDouble.search;
}

describe('workbench palette component handlers', () => {
  test.each([
    'ArrowDown',
    'ArrowUp',
    'Home',
    'End',
    'Enter',
    'Escape'
  ])('leaves composing %s events to the input method', key => {
    const {store, calls} = makeStore();
    const harness = makePaletteHarness(store);
    const before = harness.getHighlight();
    const original = store.getState().document;
    const event = harness.pressKey(key, true);
    expect({
      prevented: event.preventDefault.mock.calls.length,
      highlight: harness.getHighlight(),
      calls,
      documentUnchanged: store.getState().document === original
    }).toEqual({prevented: 0, highlight: before, calls: [], documentUnchanged: true});
  });

  test('navigates, wraps and requests scrolling through the actual search handler', () => {
    const {store, calls} = makeStore();
    const harness = makePaletteHarness(store);
    const documentDouble = makePaletteDocument();
    attachPaletteReferences(harness, documentDouble);
    harness.flushEffects();
    const navigation = [
      ['ArrowDown', 'fixture-scale'],
      ['End', 'fixture-source'],
      ['ArrowDown', 'fixture-count'],
      ['ArrowUp', 'fixture-source'],
      ['Home', 'fixture-count']
    ];
    for (const [key, operationId] of navigation) {
      const event = harness.pressKey(key);
      harness.flushEffects();
      expect(event.preventDefault).toHaveBeenCalledTimes(1);
      expect(harness.getHighlight()).toBe(`palette-handler-test-${operationId}`);
      expect(documentDouble.ownerDocument.getElementById).toHaveBeenLastCalledWith(
        `palette-handler-test-${operationId}`
      );
      expect(documentDouble.scrollIntoView).toHaveBeenLastCalledWith({block: 'nearest'});
    }
    expect(documentDouble.scrollIntoView).toHaveBeenCalledTimes(6);
    expect(documentDouble.search.focus).toHaveBeenCalledTimes(1);
    expect(documentDouble.ownerDocument.addEventListener).toHaveBeenCalledTimes(1);
    expect(calls).toEqual([]);
    harness.unmount();
  });

  test('preserves native Tab and text-key defaults and closes on ordinary Escape', () => {
    const {store, calls} = makeStore();
    const harness = makePaletteHarness(store);
    const before = harness.getHighlight();
    for (const key of ['Tab', 'a']) {
      const event = harness.pressKey(key);
      expect(event.preventDefault).not.toHaveBeenCalled();
      expect(event.stopPropagation).toHaveBeenCalledTimes(1);
      expect(harness.getHighlight()).toBe(before);
      expect(calls).toEqual([]);
    }
    const event = harness.pressKey('Escape');
    expect(event.preventDefault).toHaveBeenCalledTimes(1);
    expect(calls).toEqual(['close']);
  });

  test('adds the highlighted operation only for non-composing Enter', () => {
    const {store, calls} = makeStore();
    const harness = makePaletteHarness(store);
    harness.pressKey('ArrowDown');
    expect(harness.getHighlight()).toBe('palette-handler-test-fixture-scale');
    harness.pressKey('Enter', true);
    expect(calls).toEqual([]);
    const event = harness.pressKey('Enter');
    expect(event.preventDefault).toHaveBeenCalledTimes(1);
    expect(calls).toEqual(['edit:add-step', 'select:fixture-scale', 'close', 'dock:parameters']);
    expect(store.getState().document?.steps[3]?.operationId).toBe('fixture-scale');
  });

  test('focuses search, ignores inside pointers, closes outside and restores canvas focus', () => {
    const {store, calls} = makeStore();
    const harness = makePaletteHarness(store);
    const documentDouble = makePaletteDocument();
    attachPaletteReferences(harness, documentDouble);
    vi.stubGlobal('Node', documentDouble.ElementDouble);
    try {
      harness.flushEffects();
      expect(documentDouble.search.focus).toHaveBeenCalledTimes(1);
      expect(documentDouble.ownerDocument.addEventListener).toHaveBeenCalledWith(
        'pointerdown',
        expect.any(Function)
      );
      const listener = documentDouble.listeners.get('pointerdown')!;
      listener({target: documentDouble.search});
      listener({target: documentDouble.palette});
      listener({target: null});
      expect(calls).toEqual([]);
      listener({target: documentDouble.canvas});
      expect(calls).toEqual(['close']);
      harness.unmount();
      expect(documentDouble.ownerDocument.removeEventListener).toHaveBeenCalledWith(
        'pointerdown',
        listener
      );
      expect(documentDouble.listeners.size).toBe(0);
      expect(documentDouble.ownerDocument.querySelector).toHaveBeenCalledWith(
        `[data-testid="${WORKBENCH_TEST_IDS.canvas}"]`
      );
      expect(documentDouble.canvas.focus).toHaveBeenCalledTimes(1);
      expect(documentDouble.opener.focus).not.toHaveBeenCalled();
    } finally {
      vi.unstubAllGlobals();
    }
  });

  test.each([
    true,
    false
  ])('restores a connected opener only when the canvas is absent (connected: %s)', isConnected => {
    const harness = makePaletteHarness(makeStore().store);
    const documentDouble = makePaletteDocument();
    attachPaletteReferences(harness, documentDouble);
    documentDouble.ownerDocument.querySelector.mockReturnValue(null);
    documentDouble.opener.isConnected = isConnected;
    vi.stubGlobal('HTMLElement', documentDouble.ElementDouble);
    try {
      harness.flushEffects();
      harness.unmount();
      expect(documentDouble.opener.focus).toHaveBeenCalledTimes(isConnected ? 1 : 0);
      expect(documentDouble.canvas.focus).not.toHaveBeenCalled();
      expect(documentDouble.listeners.size).toBe(0);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

describe('workbench palette models', () => {
  test('counts categories in shared vocabulary order and hides zero-count categories', () => {
    const registry = makeFixtureRegistry();
    expect(getPaletteCategoryCounts(registry, searchWorkflowOperations(registry, ''))).toEqual([
      {category: 'all', label: 'All', count: 3},
      {category: 'source', label: 'Sources', count: 1},
      {category: 'table', label: 'Tables', count: 1},
      {category: 'geometry', label: 'Geometry', count: 1}
    ]);
    expect(getPaletteCategoryCounts(registry, searchWorkflowOperations(registry, 'scale'))).toEqual(
      [
        {category: 'all', label: 'All', count: 1},
        {category: 'geometry', label: 'Geometry', count: 1}
      ]
    );
    expect(getPaletteCategoryCounts({list: () => [], get: () => undefined}, [])).toEqual([
      {category: 'all', label: 'All', count: 0}
    ]);
    expect(
      getPaletteCategoryCounts(registry, searchWorkflowOperations(registry, 'missing'))
    ).toEqual([{category: 'all', label: 'All', count: 0}]);
    expect(
      getPaletteCategoryCounts(
        registry,
        searchWorkflowOperations(registry, '', {compatibleWith: 'polygon'})
      )
    ).toEqual([
      {category: 'all', label: 'All', count: 1},
      {category: 'table', label: 'Tables', count: 1}
    ]);
  });

  test('models unique input and output kinds in declaration order', () => {
    const registry = makeFixtureRegistry();
    expect(getPalettePortLine(registry.get('fixture-source')!)).toEqual({
      inputs: [],
      outputs: ['features']
    });
    expect(getPalettePortLine(registry.get('fixture-count')!)).toEqual({
      inputs: ['point', 'line', 'polygon', 'table'],
      outputs: ['table']
    });
    const operation = registry.get('fixture-scale')!;
    expect(
      getPalettePortLine({
        ...operation,
        inputs: [...operation.inputs, ...operation.inputs],
        outputs: [...operation.outputs, ...operation.outputs]
      })
    ).toEqual({inputs: ['point'], outputs: ['point']});
  });

  test.each<{
    accepts: WorkflowPortKind[];
    expected: string[];
  }>([
    {accepts: ['point', 'line', 'polygon', 'features', 'table'], expected: ['any']},
    {accepts: ['table', 'features', 'polygon', 'line', 'point', 'point'], expected: ['any']},
    {accepts: ['point', 'line', 'polygon'], expected: ['any-geometry']},
    {accepts: ['features', 'polygon', 'point', 'line'], expected: ['any-geometry']},
    {accepts: ['point', 'polygon'], expected: ['point', 'polygon']},
    {accepts: ['features'], expected: ['features']},
    {accepts: ['table'], expected: ['table']},
    {
      accepts: ['point', 'line', 'polygon', 'table'],
      expected: ['point', 'line', 'polygon', 'table']
    },
    {accepts: [], expected: []}
  ])('summarises accepted input kinds $accepts as $expected', ({accepts, expected}) => {
    const operation = makeFixtureRegistry().get('fixture-scale')!;
    const outputs = (['point', 'line', 'polygon', 'features', 'table'] as const).map(kind => ({
      ...operation.outputs[0],
      name: kind,
      kind
    }));
    expect(
      getPalettePortLine({...operation, inputs: [{...operation.inputs[0], accepts}], outputs})
    ).toEqual({inputs: expected, outputs: ['point', 'line', 'polygon', 'features', 'table']});
  });

  test('summarises each input independently and deduplicates entries in port order', () => {
    const operation = makeFixtureRegistry().get('fixture-scale')!;
    const port = operation.inputs[0];
    expect(
      getPalettePortLine({
        ...operation,
        inputs: [
          {...port, name: 'points', accepts: ['point']},
          {...port, name: 'lines', accepts: ['line']},
          {...port, name: 'polygons', accepts: ['polygon']}
        ]
      })
    ).toEqual({inputs: ['point', 'line', 'polygon'], outputs: ['point']});
    expect(
      getPalettePortLine({
        ...operation,
        inputs: [
          {...port, name: 'geometry', accepts: ['point', 'line', 'polygon']},
          {...port, name: 'geometry-again', accepts: ['features', 'polygon', 'point', 'line']},
          {...port, name: 'rows', accepts: ['point', 'line', 'polygon', 'features', 'table']},
          {...port, name: 'points', accepts: ['point']}
        ]
      })
    ).toEqual({inputs: ['any-geometry', 'any', 'point'], outputs: ['point']});
  });

  test('keeps six unique recent ids, most recent first, without changing its input', () => {
    const previous = ['a', 'b', 'c', 'd', 'e', 'f'];
    expect(updatePaletteRecentOperations(previous, 'g')).toEqual(['g', 'a', 'b', 'c', 'd', 'e']);
    expect(updatePaletteRecentOperations(previous, 'c')).toEqual(['c', 'a', 'b', 'd', 'e', 'f']);
    expect(updatePaletteRecentOperations(['a', 'a', 'b'], 'b')).toEqual(['b', 'a']);
    expect(previous).toEqual(['a', 'b', 'c', 'd', 'e', 'f']);
  });

  test('groups matching recents once, omits missing ids, and preserves search ranking', () => {
    const registry = makeFixtureRegistry();
    const matches = searchWorkflowOperations(registry, '');
    const recentIds = ['missing', 'fixture-count', 'fixture-count', 'fixture-source'];
    const groups = groupPaletteOperations(matches, recentIds, '  ');
    expect(groups.recent.map(match => match.operation.id)).toEqual([
      'fixture-count',
      'fixture-source'
    ]);
    expect(groups.operations.map(match => match.operation.id)).toEqual(['fixture-scale']);
    expect(groupPaletteOperations(matches, recentIds, 'fixture')).toEqual({
      recent: [],
      operations: matches
    });
    expect(groupPaletteOperations(matches, recentIds, '', 'geometry')).toEqual({
      recent: [],
      operations: matches
    });
  });

  test('maps navigation with wrapping and leaves unrelated keys alone', () => {
    expect(movePaletteHighlight('ArrowDown', 2, 3)).toBe(0);
    expect(movePaletteHighlight('ArrowUp', 0, 3)).toBe(2);
    expect(movePaletteHighlight('ArrowDown', 0, 3)).toBe(1);
    expect(movePaletteHighlight('ArrowUp', 2, 3)).toBe(1);
    expect(movePaletteHighlight('Home', 2, 3)).toBe(0);
    expect(movePaletteHighlight('End', 0, 3)).toBe(2);
    for (const key of ['ArrowDown', 'ArrowUp', 'Home', 'End'])
      expect(movePaletteHighlight(key, 0, 0)).toBe(-1);
    for (const key of ['Tab', 'Enter', 'Escape', 'a'])
      expect(movePaletteHighlight(key, 0, 3)).toBeUndefined();
  });

  test('reads an observed output kind even for stale or running results, with declared fallback', () => {
    for (const status of ['done', 'stale', 'running'] as const) {
      const {store} = makeStore({
        palette: {open: true, connectFrom: {stepId: 'source', output: 'data'}},
        runner: makeRunner('polygon', status)
      });
      expect(getPaletteConnection(store.getState(), store.registry)).toEqual({
        producerLabel: 'Fixture source',
        outputLabel: 'Data',
        kind: 'polygon'
      });
    }
    const {store} = makeStore({
      palette: {open: true, connectFrom: {stepId: 'source', output: 'data'}}
    });
    expect(getPaletteConnection(store.getState(), store.registry)).toEqual({
      producerLabel: 'Fixture source',
      outputLabel: 'Data',
      kind: 'features'
    });
    expect(
      getPaletteConnection({...store.getState(), palette: {open: true}}, store.registry)
    ).toBeNull();
    expect(
      getPaletteConnection(
        {
          ...store.getState(),
          palette: {open: true, connectFrom: {stepId: 'missing', output: 'missing-output'}}
        },
        store.registry
      )
    ).toEqual({producerLabel: 'missing', outputLabel: 'missing-output', kind: undefined});
  });
});

describe('workbench palette markup', () => {
  test.each<{
    accepts: WorkflowPortKind[];
    label: string;
    swatch: WorkflowPortKind;
  }>([
    {accepts: ['point', 'line', 'polygon', 'features', 'table'], label: 'Any', swatch: 'table'},
    {accepts: ['point', 'line', 'polygon', 'features'], label: 'Any geometry', swatch: 'features'}
  ])('renders the compact input label $label and preserves the output label', ({
    accepts,
    label,
    swatch
  }) => {
    const operation = makeFixtureRegistry().get('fixture-scale')!;
    const adapted = {
      ...operation,
      inputs: [{...operation.inputs[0], accepts}],
      outputs: [{...operation.outputs[0], kind: 'features' as const}]
    };
    const registry = {list: () => [adapted], get: () => adapted};
    const {markup, styles} = renderWithStyles(makeStore({}, registry).store);
    const row = markup.match(/<button[^>]*role="option"[^>]*>[\s\S]*?<\/button>/)![0];
    expect(row).toContain(`title="${label} → Features"`);
    expect(row).toContain(`>${label}</span>`);
    expect(row).toContain('>Features</span>');
    for (const kind of ['Points', 'Lines', 'Polygons', 'Table'])
      expect(row).not.toContain(`>${kind}</span>`);
    const swatchTag = row.match(/<span aria-hidden="true"[^>]*>/)![0];
    expect(readElementStyles(swatchTag, styles)).toContain(
      `background:${WORKBENCH_PORT_COLORS[swatch]};`
    );
  });

  test('uses viewport bounds independent of dock height and separate scroll areas', () => {
    const rootStyles: string[] = [];
    for (const height of [220, 380, 900]) {
      const {store} = makeStore({
        dock: {open: true, height, inspectorTab: 'result', logOpen: false}
      });
      const {markup, styles} = renderWithStyles(store);
      const rootTag = markup.match(/^<div[^>]*>/)![0];
      const root = readElementStyles(rootTag, styles);
      rootStyles.push(root);
      expect(root).toContain('position:fixed;');
      expect(root).toContain('top:12vh;');
      expect(root).toContain(
        'left:calc(var(--workbench-dock-left, 0px) + (100vw - var(--workbench-dock-left, 0px)) / 2);'
      );
      expect(root).toContain('transform:translateX(-50%);');
      expect(root).toContain('z-index:60;');
      expect(root).toContain(
        'width:min(680px, calc(100vw - var(--workbench-dock-left, 0px) - 48px));'
      );
      expect(root).toContain('max-height:72vh;');
      expect(root).toContain('overflow:hidden;');
      expect(root).toContain('box-shadow:');
      expect(root).toContain('container-type:inline-size;');
      expect(styles).toContain('grid-template-columns:minmax(0, 3fr) minmax(0, 2fr);');
      expect(styles).toContain('grid-template-rows:minmax(0, 1fr);');
      expect(styles).toContain('@container (max-width: 580px)');
      const listTag = markup.match(/<div[^>]*role="listbox"[^>]*>/)![0];
      const detailTag = markup.match(/<section[^>]*details"[^>]*>/)![0];
      for (const tag of [listTag, detailTag]) {
        const rules = readElementStyles(tag, styles);
        expect(rules).toContain('min-height:0;');
        expect(rules).toContain('overflow-y:auto;');
        expect(rules).toContain('overflow-x:hidden;');
      }
    }
    expect(rootStyles[0]).toBe(rootStyles[1]);
    expect(rootStyles[1]).toBe(rootStyles[2]);
  });

  test('keeps titles, engine tags and port lines on one line with complete tooltips', () => {
    const title = 'A very long operation title that must leave enough room for its engine tag';
    const operation = {...makeFixtureRegistry().get('fixture-scale')!, title};
    const registry = {list: () => [operation], get: () => operation};
    const {markup, styles} = renderWithStyles(makeStore({}, registry).store);
    const titleTag = markup.match(/<span title="A very long operation title[^>]*>/)![0];
    expect(titleTag).toContain(`title="${title}"`);
    const titleStyles = readElementStyles(titleTag, styles);
    expect(titleStyles).toContain('min-width:0;');
    expect(titleStyles).toContain('white-space:nowrap;');
    expect(titleStyles).toContain('overflow:hidden;');
    expect(titleStyles).toContain('text-overflow:ellipsis;');
    const engineTag = markup.match(/<span class="[^"]+">GPU or CPU<\/span>/)![0];
    const engineStyles = readElementStyles(engineTag, styles);
    expect(engineStyles).toContain('flex:none;');
    expect(engineStyles).toContain('white-space:nowrap;');
    const portTag = markup.match(/<span title="Points → Points"[^>]*>/)![0];
    const portStyles = readElementStyles(portTag, styles);
    expect(portStyles).toContain('display:block;');
    expect(portStyles).toContain('white-space:nowrap;');
    expect(portStyles).toContain('overflow:hidden;');
    expect(portStyles).toContain('text-overflow:ellipsis;');
    expect(portStyles).not.toContain('flex-wrap:wrap;');
    const rowTag = markup.match(/<button[^>]*role="option"[^>]*>/)![0];
    const rowStyles = readElementStyles(rowTag, styles);
    expect(rowStyles).toContain('padding:5px 8px;');
    expect(rowStyles).toContain('line-height:1.2;');
  });

  test('places a one-line keyboard hint below search and associates it with the combobox', () => {
    const {markup, styles} = renderWithStyles(makeStore().store);
    const hint = '↑ ↓ to move · Enter to add · Esc to close';
    const hintId = markup.match(/aria-describedby="([^"]+)"/)?.[1];
    expect(hintId).toBeTruthy();
    const hintTag = markup.match(/<span id="[^"]+-hint"[^>]*>/)![0];
    expect(hintTag).toContain(`id="${hintId}"`);
    expect(markup).toContain(`${hintTag}${hint}</span>`);
    expect(readElementStyles(hintTag, styles)).toContain('white-space:nowrap;');
    expect(markup.indexOf(hintTag)).toBeGreaterThan(
      markup.indexOf(`data-testid="${WORKBENCH_TEST_IDS.paletteSearch}"`)
    );
    expect(markup.indexOf(hintTag)).toBeLessThan(
      markup.indexOf(`data-testid="${WORKBENCH_TEST_IDS.paletteCategory('all')}"`)
    );
  });

  test('renders the dialog, labelled combobox, counted chips, options, engines and detail', () => {
    const {store} = makeStore();
    const markup = render(store);
    expect(markup).toContain(`data-testid="${WORKBENCH_TEST_IDS.palette}"`);
    expect(markup).toContain('role="dialog" aria-label="Add a step"');
    expect(markup).toContain(`data-testid="${WORKBENCH_TEST_IDS.paletteSearch}"`);
    expect(markup).toContain('role="combobox"');
    expect(markup).toContain('aria-autocomplete="list"');
    expect(markup).toContain('Search operations');
    const controlledList = markup.match(/aria-controls="([^"]+)"/)?.[1];
    expect(controlledList).toBeTruthy();
    expect(markup).toContain(`id="${controlledList}" role="listbox" aria-label="Operations"`);
    expect(markup).toContain('aria-activedescendant=');
    expect(markup).toContain(
      `data-testid="${WORKBENCH_TEST_IDS.paletteCategory('all')}" aria-pressed="true"`
    );
    for (const label of ['All (3)', 'Sources (1)', 'Tables (1)', 'Geometry (1)'])
      expect(markup).toContain(label);
    for (const operation of store.registry.list()) {
      expect(markup).toContain(`data-testid="${WORKBENCH_TEST_IDS.paletteItem(operation.id)}"`);
      expect(markup).toContain(operation.title);
    }
    expect(markup.match(/role="option"/g)).toHaveLength(3);
    for (const label of [
      'Host',
      'CPU',
      'GPU or CPU',
      'Features',
      'Points',
      'Lines',
      'Polygons',
      'Table'
    ])
      expect(markup).toContain(label);
    const sourceDetails = render(store, 'source');
    expect(sourceDetails).toContain('aria-label="Fixture source details"');
    expect(sourceDetails).toContain('Points from the fixture host.');
    expect(sourceDetails).toContain('Data: features, up to 50 rows');
    expect(sourceDetails).toContain('Fixture operation.');
    expect(sourceDetails).toContain('Fixture only.');
  });

  test('renders the no-match query and query-specific counts without an active descendant', () => {
    const markup = render(makeStore().store, 'unfindable');
    expect(markup).toContain('No operation matches &quot;unfindable&quot;.');
    expect(markup).toContain('All (0)');
    for (const category of ['source', 'table', 'geometry'])
      expect(markup).not.toContain(`data-testid="${WORKBENCH_TEST_IDS.paletteCategory(category)}"`);
    expect(markup).not.toContain('role="option"');
    expect(markup).not.toContain('aria-activedescendant=');
    expect(markup).not.toContain('details"');
  });

  test('filters by a connection and explains uncertain compatibility', () => {
    const {store} = makeStore({
      palette: {open: true, connectFrom: {stepId: 'source', output: 'data'}}
    });
    const markup = render(store);
    expect(markup).toContain('Reading Fixture source · Data (Features)');
    expect(markup).toContain('Compatible only');
    expect(markup).toContain('type="checkbox" checked=""');
    expect(markup).toContain('checked when run');
    expect(markup).toContain('All (2)');
    expect(markup).not.toContain(`data-testid="${WORKBENCH_TEST_IDS.paletteCategory('source')}"`);
    expect(markup).not.toContain(
      `data-testid="${WORKBENCH_TEST_IDS.paletteItem('fixture-source')}"`
    );
    expect(markup.match(/role="option"/g)).toHaveLength(2);
  });

  test('uses the observed kind, custom producer label and only compatible rows', () => {
    const document = makeFixtureDocument();
    document.steps[0].label = 'My source';
    const {store} = makeStore({
      document,
      runner: makeRunner('polygon', 'stale'),
      palette: {open: true, connectFrom: {stepId: 'source', output: 'data'}}
    });
    const markup = render(store);
    expect(markup).toContain('Reading My source · Data (Polygons)');
    expect(markup).toContain('All (1)');
    expect(markup).not.toContain(`data-testid="${WORKBENCH_TEST_IDS.paletteCategory('geometry')}"`);
    expect(markup).toContain(`data-testid="${WORKBENCH_TEST_IDS.paletteItem('fixture-count')}"`);
    expect(markup).not.toContain(
      `data-testid="${WORKBENCH_TEST_IDS.paletteItem('fixture-scale')}"`
    );
    expect(markup).not.toContain('checked when run');
  });

  test('limits details to two behaviours and shows input capacities and a GPU tag', () => {
    const fixtureRegistry = makeFixtureRegistry();
    const operation = {
      ...fixtureRegistry.get('fixture-scale')!,
      engine: 'gpu' as const,
      inputs: [
        {
          name: 'points',
          label: 'Points',
          accepts: ['point' as const],
          maximumRows: 20000,
          maximumVertices: 40000
        }
      ],
      declaredBehavior: ['First behaviour.', 'Second behaviour.', 'Hidden third behaviour.']
    };
    const registry = {list: () => [operation], get: () => operation};
    const markup = render(makeStore({}, registry).store);
    expect(markup).toContain('Points: points, up to 20,000 rows, 40,000 vertices');
    expect(markup).toContain('First behaviour.');
    expect(markup).toContain('Second behaviour.');
    expect(markup).not.toContain('Hidden third behaviour.');
    expect(markup).toMatch(/>GPU<\/span>/);
  });

  test('renders at most sixty rows and the exact remaining count', () => {
    const base = makeFixtureRegistry().get('fixture-scale')!;
    const operations = Array.from({length: 65}, (_, index) => ({
      ...base,
      id: `operation-${index}`,
      title: `Operation ${index}`
    }));
    const registry = {
      list: () => operations,
      get: (operationId: string) => operations.find(operation => operation.id === operationId)
    };
    const markup = render(makeStore({}, registry).store);
    expect(markup.match(/role="option"/g)).toHaveLength(60);
    expect(markup).toContain('5 more: refine the search');
    expect(markup).toContain('All (65)');
    expect(markup).not.toContain(`data-testid="${WORKBENCH_TEST_IDS.paletteItem('operation-60')}"`);
  });

  test('renders closed, absent, empty, unknown and error documents without a runner', () => {
    expect(render(makeStore({palette: {open: false}}).store)).toBe('');
    const document = makeFixtureDocument();
    document.steps[0].operationId = 'missing-operation';
    for (const patch of [
      {document: null, workflows: [], activeWorkflowId: null},
      {document: {...document, steps: []}},
      {document, palette: {open: true, connectFrom: {stepId: 'source', output: 'data'}}},
      {
        document,
        runner: {
          ...makeRunner('point', 'running'),
          steps: {
            source: {
              key: null,
              status: 'error' as const,
              issues: [
                {
                  code: 'operation-failed' as const,
                  severity: 'error' as const,
                  message: 'My source: failed.'
                }
              ]
            }
          }
        }
      }
    ])
      expect(render(makeStore(patch).store)).toContain('aria-label="Add a step"');
    expect(
      render(
        makeStore({
          document,
          palette: {open: true, connectFrom: {stepId: 'source', output: 'data'}}
        }).store
      )
    ).toContain('Reading missing-operation · data (Unknown kind)');
  });
});

describe('adding a palette operation', () => {
  test('adds once at the requested position, connects, selects, closes and opens parameters', () => {
    const {store, calls, errors} = makeStore({
      palette: {
        open: true,
        position: {x: 123, y: 456},
        connectFrom: {stepId: 'source', output: 'data'}
      }
    });
    expect(addPaletteOperation(store, 'fixture-scale')).toBe(true);
    expect(calls).toEqual(['edit:add-step', 'select:fixture-scale', 'close', 'dock:parameters']);
    expect(errors).toEqual([]);
    expect(store.getState().document?.steps).toHaveLength(4);
    expect(store.getState().document?.steps[3]).toEqual({
      id: 'fixture-scale',
      operationId: 'fixture-scale',
      inputs: {points: {stepId: 'source', output: 'data'}},
      parameters: {},
      position: {x: 123, y: 456}
    });
    const next = makeStore({document: store.getState().document});
    expect(addPaletteOperation(next.store, 'fixture-scale')).toBe(true);
    expect(next.calls).toEqual([
      'edit:add-step',
      'select:fixture-scale-2',
      'close',
      'dock:parameters'
    ]);
    const markup = render(store);
    expect(markup).toContain('Recent');
    expect(
      markup.indexOf(`data-testid="${WORKBENCH_TEST_IDS.paletteItem('fixture-scale')}"`)
    ).toBeLessThan(
      markup.indexOf(`data-testid="${WORKBENCH_TEST_IDS.paletteItem('fixture-count')}"`)
    );
    expect(markup.match(/role="option"/g)).toHaveLength(2);
  });

  test('preserves refusals and does not select or close on unknown operations or capacity', () => {
    const unknown = makeStore();
    const original = unknown.store.getState().document;
    expect(addPaletteOperation(unknown.store, 'does-not-exist')).toBe(false);
    expect(unknown.calls).toEqual(['edit:add-step']);
    expect(unknown.errors[0]).toBeInstanceOf(WorkflowError);
    expect((unknown.errors[0] as WorkflowError).code).toBe('unknown-operation');
    expect(unknown.store.getState().document).toBe(original);
    const document = makeFixtureDocument();
    document.steps = Array.from({length: WORKFLOW_LIMITS.maximumSteps}, (_, index) => ({
      ...document.steps[0],
      id: `step-${index}`
    }));
    const full = makeStore({document});
    expect(addPaletteOperation(full.store, 'fixture-source')).toBe(false);
    expect(full.calls).toEqual(['edit:add-step']);
    expect((full.errors[0] as WorkflowError).code).toBe('limit-exceeded');
    expect((full.errors[0] as Error).message).toContain('maximumSteps');
    expect(full.store.getState().document).toBe(document);
  });
});

function makeAnalysisProfile(
  datasetId = 'uploaded-events',
  geometry: 'Point' | 'Polygon' = 'Point'
): KeplerDatasetProfile {
  return {
    ...profileKeplerFeatureCollection(
      {
        type: 'FeatureCollection',
        features: Array.from({length: 12}, (_, index) => ({
          type: 'Feature',
          geometry:
            geometry === 'Point'
              ? {
                  type: 'Point',
                  coordinates: [100 + (index % 4) * 2, 200 + Math.floor(index / 4) * 3]
                }
              : {
                  type: 'Polygon',
                  coordinates: [
                    [
                      [index, 0],
                      [index + 1, 0],
                      [index + 1, 1],
                      [index, 1],
                      [index, 0]
                    ]
                  ]
                },
          properties: {renamed_value: index + 1}
        }))
      },
      {
        datasetId,
        label: datasetId,
        coordinateSystem: 'planar',
        declarations: {rowSemantics: 'events'}
      }
    ),
    geometryColumns: ['geometry']
  };
}

function makeAnalysisStore(profiles = [makeAnalysisProfile()]) {
  const fixture = makeStore({}, getWorkflowOperationRegistry());
  fixture.store.host.getAnalysisProfile = datasetId =>
    profiles.find(profile => profile.datasetId === datasetId);
  fixture.store.host.listDatasets = () =>
    profiles.map(profile => ({
      id: profile.datasetId,
      label: profile.label,
      rowCount: profile.rowCount,
      resident: false,
      fields: [{name: 'geometry', type: 'string'}],
      geometryFields: ['geometry']
    }));
  return fixture;
}

describe('AX command palette', () => {
  test('renders the AX hook, question matches, dataset scope and mapped template first', () => {
    const {store} = makeAnalysisStore();
    const markup = renderToStaticMarkup(
      createElement(WorkbenchAnalysisPaletteDialog, {store, initialQuery: 'hot'})
    );
    expect(markup).toContain('data-testid="ax-command-palette"');
    // A wrapper with only fixed-position children has a zero-size box and is hidden to a
    // browser locator. The AX hook must identify the actual positioned dialog.
    expect(markup).toMatch(/<div[^>]*role="dialog"[^>]*data-testid="ax-command-palette"/);
    expect(markup).toContain('Search questions, templates and operations');
    expect(markup).toContain('data-testid="ax-command-dataset"');
    expect(markup).toContain('value="uploaded-events"');
    expect(markup.indexOf('data-testid="ax-command-template:hot-spot-map"')).toBeLessThan(
      markup.indexOf('data-testid="workflow-palette-item-recipe-hot-spot-lattice"')
    );
    expect(markup).toContain('Where do values cluster or stand out?');
    expect(render(store)).toContain('data-testid="ax-command-palette"');
  });

  test('a stationary pointer cannot replace the keyboard highlight when search rows move beneath it', () => {
    const {store} = makeAnalysisStore();
    const harness = makePaletteHarness(store, WorkbenchAnalysisPaletteDialog);
    const searchChange = harness.getSearch().props['onChange'] as (event: {
      target: {value: string};
    }) => void;
    searchChange({target: {value: 'hot'}});
    harness.render();
    const operationId = WORKBENCH_TEST_IDS.paletteItem('curated-emerging-hot-spots');
    const movedRow = harness.find('data-testid', operationId);
    const layoutEnter = movedRow.props['onMouseEnter'] as (() => void) | undefined;
    layoutEnter?.();
    harness.render();
    expect(harness.getHighlight()).toContain('template:hot-spot-map');
    const pointerMove = movedRow.props['onMouseMove'] as (event: {
      movementX: number;
      movementY: number;
    }) => void;
    pointerMove({movementX: 0, movementY: 0});
    harness.render();
    expect(harness.getHighlight()).toContain('template:hot-spot-map');
    pointerMove({movementX: 0, movementY: 1});
    harness.render();
    expect(harness.getHighlight()).toContain('operation:curated-emerging-hot-spots');
    searchChange({target: {value: 'hot'}});
    harness.render();
    harness.pressKey('Enter');
    expect(
      harness.find('analysis', getKeplerCuratedAnalysis('hot-spot-map')).props['mapping']
    ).toMatchObject({
      progress: {mapped: 1, required: 1},
      slots: [{slot: 'points', datasetId: 'uploaded-events'}]
    });
  });

  test('Enter opens the existing mapping component with filled slots and never runs', () => {
    const {store, calls} = makeAnalysisStore();
    const run = vi.spyOn(store, 'run');
    const harness = makePaletteHarness(store, WorkbenchAnalysisPaletteDialog, 'hot');
    expect(harness.getHighlight()).toContain('template:hot-spot-map');
    harness.pressKey('Enter', true);
    expect(harness.getHighlight()).toContain('template:hot-spot-map');
    harness.pressKey('Enter');
    const mappingComponent = harness.find('analysis', getKeplerCuratedAnalysis('hot-spot-map'));
    expect(mappingComponent.props['mapping']).toMatchObject({
      analysisId: 'hot-spot-map',
      progress: {mapped: 1, required: 1},
      slots: [{slot: 'points', datasetId: 'uploaded-events'}]
    });
    expect(run).not.toHaveBeenCalled();
    expect(calls).toEqual([]);
    (mappingComponent.props['onCancel'] as () => void)();
    harness.render();
    expect(harness.getSearch()).toBeDefined();
  });

  test('focuses search, transfers focus into mapping and restores the opener on close', () => {
    const {store} = makeAnalysisStore();
    const harness = makePaletteHarness(store, WorkbenchAnalysisPaletteDialog, 'hot');
    const documentDouble = makePaletteDocument();
    attachPaletteReferences(harness, documentDouble);
    const mappingControl = new documentDouble.ElementDouble();
    documentDouble.palette.querySelector.mockReturnValue(mappingControl);
    vi.stubGlobal('Node', documentDouble.ElementDouble);
    vi.stubGlobal('HTMLElement', documentDouble.ElementDouble);
    try {
      harness.flushEffects();
      expect(documentDouble.search.focus).toHaveBeenCalled();
      harness.pressKey('Enter');
      harness.flushEffects();
      expect(mappingControl.focus).toHaveBeenCalledTimes(1);
      harness.unmount();
      expect(documentDouble.opener.focus).toHaveBeenCalledTimes(1);
      expect(documentDouble.listeners.size).toBe(0);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  test.each([
    'body',
    'documentElement',
    'removed',
    'palette-control'
  ])('returns AX palette focus to the canvas when the captured opener is %s', openerKind => {
    const {store} = makeAnalysisStore();
    const harness = makePaletteHarness(store, WorkbenchAnalysisPaletteDialog, 'hot');
    const documentDouble = makePaletteDocument();
    attachPaletteReferences(harness, documentDouble);
    documentDouble.opener.isConnected = openerKind !== 'removed';
    documentDouble.ownerDocument.activeElement =
      openerKind === 'body'
        ? documentDouble.ownerDocument.body
        : openerKind === 'documentElement'
          ? documentDouble.ownerDocument.documentElement
          : openerKind === 'palette-control'
            ? documentDouble.search
            : documentDouble.opener;
    vi.stubGlobal('Node', documentDouble.ElementDouble);
    vi.stubGlobal('HTMLElement', documentDouble.ElementDouble);
    try {
      harness.flushEffects();
      const previousFocusCalls = documentDouble.ownerDocument.activeElement.focus.mock.calls.length;
      harness.pressKey('Escape');
      harness.unmount();
      expect(documentDouble.canvas.focus).toHaveBeenCalledOnce();
      expect(documentDouble.ownerDocument.activeElement.focus).toHaveBeenCalledTimes(
        previousFocusCalls
      );
    } finally {
      vi.unstubAllGlobals();
    }
  });

  test('returns AX palette focus to the dock when no canvas or connected opener remains', () => {
    const {store} = makeAnalysisStore();
    const harness = makePaletteHarness(store, WorkbenchAnalysisPaletteDialog, 'hot');
    const documentDouble = makePaletteDocument();
    attachPaletteReferences(harness, documentDouble);
    const dockElement = new documentDouble.ElementDouble();
    documentDouble.opener.isConnected = false;
    documentDouble.ownerDocument.querySelector.mockImplementation(selector =>
      selector === `[data-testid="${WORKBENCH_TEST_IDS.dock}"]` ? dockElement : null
    );
    vi.stubGlobal('Node', documentDouble.ElementDouble);
    vi.stubGlobal('HTMLElement', documentDouble.ElementDouble);
    try {
      harness.flushEffects();
      harness.unmount();
      expect(dockElement.focus).toHaveBeenCalledOnce();
    } finally {
      vi.unstubAllGlobals();
    }
  });

  test('changing the dataset scopes the next opened template', () => {
    const profiles = [makeAnalysisProfile('first'), makeAnalysisProfile('second')];
    const {store} = makeAnalysisStore(profiles);
    const harness = makePaletteHarness(store, WorkbenchAnalysisPaletteDialog, 'hot');
    const selector = harness.find('data-testid', 'ax-command-dataset');
    (selector.props['onChange'] as (event: unknown) => void)({target: {value: 'second'}});
    harness.render();
    harness.pressKey('Enter');
    expect(
      harness.find('analysis', getKeplerCuratedAnalysis('hot-spot-map')).props['mapping']
    ).toMatchObject({slots: [{datasetId: 'second'}]});
  });

  test('the opened mapping produces the existing launch document on the user dataset', () => {
    const profile = makeAnalysisProfile('renamed-upload');
    const {store} = makeAnalysisStore([profile]);
    const match = searchWorkflowAnalyses(store.registry, 'hot', {profiles: [profile]})[0];
    const selection = openPaletteAnalysis(store, match, profile.datasetId)!;
    const document = createWorkbenchAnalysisDocument(selection.template, store, selection.mapping);
    expect(document.steps.find(step => step.id === 'source-points')?.parameters['dataset']).toEqual(
      {from: 'static', value: profile.datasetId}
    );
    expect(document.steps.find(step => step.id === 'analysis')).toMatchObject({
      operationId: 'recipe-hot-spot-lattice',
      inputs: {points: {stepId: 'source-points', output: 'data'}}
    });
    expect(document.inputs).toEqual([]);
  });

  test('refuses an opening whose active dataset profile disappeared since the search', () => {
    const profile = makeAnalysisProfile();
    const {store} = makeAnalysisStore([profile]);
    const match = searchWorkflowAnalyses(store.registry, 'hot', {profiles: [profile]})[0];
    store.host.getAnalysisProfile = () => undefined;
    const selection = openPaletteAnalysis(store, match, profile.datasetId);
    expect(selection).toBeUndefined();
  });

  test('finds the selected source through an analysis input, with cycle protection', () => {
    const profiles = [makeAnalysisProfile('first'), makeAnalysisProfile('second')];
    const {store} = makeAnalysisStore(profiles);
    const state = store.getState();
    state.document!.steps[0].parameters['dataset'] = {from: 'static', value: 'second'};
    state.selectedStepIds = ['scaled'];
    state.document!.steps.push({
      id: 'scaled',
      operationId: 'fixture-scale',
      parameters: {},
      position: {x: 0, y: 0},
      inputs: {points: {stepId: 'source', output: 'data'}}
    });
    expect(getPaletteActiveDatasetId(state, profiles)).toBe('second');
    state.document!.steps[0].parameters = {};
    state.document!.steps[0].inputs = {points: {stepId: 'scaled', output: 'data'}};
    expect(getPaletteActiveDatasetId(state, profiles)).toBe('first');
  });

  test('resolves a source dataset from a workflow input before opening its template', () => {
    const profiles = [makeAnalysisProfile('first'), makeAnalysisProfile('second')];
    const {store} = makeAnalysisStore(profiles);
    const state = store.getState();
    state.document!.inputs = [{id: 'source-id', label: 'Source', kind: 'string', value: 'second'}];
    state.document!.steps[0].operationId = 'source-dataset';
    state.document!.steps[0].parameters['dataset'] = {from: 'input', inputId: 'source-id'};
    state.selectedStepIds = ['source'];
    expect(getPaletteActiveDatasetId(state, profiles, store.registry)).toBe('second');
    const harness = makePaletteHarness(store, WorkbenchAnalysisPaletteDialog, 'hot');
    expect(harness.find('data-testid', 'ax-command-dataset').props['value']).toBe('second');
    harness.pressKey('Enter');
    expect(
      harness.find('analysis', getKeplerCuratedAnalysis('hot-spot-map')).props['mapping']
    ).toMatchObject({slots: [{datasetId: 'second'}]});
  });

  test('requires an explicit scope switch instead of substituting another compatible dataset', () => {
    const areas = makeAnalysisProfile('areas', 'Polygon');
    const events = makeAnalysisProfile('events');
    const profiles = [areas, events];
    const {store} = makeAnalysisStore(profiles);
    const notice = vi.spyOn(store, 'setNotice');
    const match = searchWorkflowAnalyses(store.registry, 'hot', {
      profiles,
      activeDatasetId: areas.datasetId
    }).find(match => match.id === 'template:hot-spot-map')!;
    expect(match).toMatchObject({
      status: 'blocked',
      reason: expect.stringContaining('active dataset')
    });
    expect(openPaletteAnalysis(store, match, areas.datasetId)).toBeUndefined();
    expect(notice).toHaveBeenCalledWith({
      kind: 'error',
      message: expect.stringContaining('Choose a compatible Active dataset')
    });
    const harness = makePaletteHarness(store, WorkbenchAnalysisPaletteDialog, 'hot');
    (
      harness.find('data-testid', 'ax-command-template:hot-spot-map').props['onClick'] as () => void
    )();
    harness.render();
    expect(harness.getSearch()).toBeDefined();
    expect(harness.find('role', 'alert').props['children']).toContain(
      'Choose a compatible Active dataset'
    );
    (
      harness.find('data-testid', 'ax-command-dataset').props['onChange'] as (
        event: unknown
      ) => void
    )({target: {value: 'events'}});
    harness.render();
    harness.pressKey('Enter');
    expect(
      harness.find('analysis', getKeplerCuratedAnalysis('hot-spot-map')).props['mapping']
    ).toMatchObject({ready: true, slots: [{datasetId: 'events'}]});
  });

  test.each([
    false,
    true
  ])('a launch closes only its still-mounted dialog (dismissed: %s)', async dismissed => {
    const {store} = makeAnalysisStore();
    const close = vi.spyOn(store, 'closePalette');
    let finishLaunch!: (record: WorkflowRunRecord) => void;
    const completion = new Promise<WorkflowRunRecord>(resolve => {
      finishLaunch = resolve;
    });
    const launch = vi
      .spyOn(WorkbenchAnalysisLaunch, 'launchWorkbenchAnalysis')
      .mockReturnValue(completion);
    const harness = makePaletteHarness(store, WorkbenchAnalysisPaletteDialog, 'hot');
    harness.flushEffects();
    harness.pressKey('Enter');
    const mappingComponent = harness.find('analysis', getKeplerCuratedAnalysis('hot-spot-map'));
    (mappingComponent.props['onRun'] as () => void)();
    expect(launch).toHaveBeenCalledTimes(1);
    let replacement: ReturnType<typeof makePaletteHarness> | undefined;
    if (dismissed) {
      (harness.getDialog().props['onKeyDown'] as (event: unknown) => void)({
        key: 'Escape',
        nativeEvent: {isComposing: false},
        preventDefault: vi.fn(),
        stopPropagation: vi.fn()
      });
      harness.unmount();
      store.openPalette();
      replacement = makePaletteHarness(store, WorkbenchAnalysisPaletteDialog, 'hot');
      replacement.flushEffects();
      close.mockClear();
    }
    try {
      finishLaunch({
        schemaVersion: 1,
        runId: 'palette-launch',
        workflowId: 'palette-workflow',
        startedAt: 0,
        durationMilliseconds: 0,
        status: 'done',
        steps: []
      });
      await completion;
      await Promise.resolve();
      expect(close).toHaveBeenCalledTimes(dismissed ? 0 : 1);
      if (replacement) expect(replacement.getSearch().props['value']).toBe('hot');
    } finally {
      if (!dismissed) harness.unmount();
      replacement?.unmount();
      launch.mockRestore();
    }
  });

  test('Cmd/Ctrl-K closes an open palette and leaves other workbench dialogs alone', () => {
    const {store, calls} = makeStore();
    let listener: (event: unknown) => void = () => {};
    const ownerDocument = {
      addEventListener: vi.fn((_name, callback) => {
        listener = callback;
      }),
      removeEventListener: vi.fn()
    };
    const dispose = subscribePaletteShortcut(store, ownerDocument as unknown as Document);
    const event = {key: 'k', ctrlKey: true, preventDefault: vi.fn()};
    listener(event);
    expect(calls).toEqual(['close']);
    store.getState().dialog = 'shortcuts';
    listener(event);
    expect(calls).toEqual(['close']);
    expect(event.preventDefault).toHaveBeenCalledTimes(1);
    dispose();
  });

  test.each([
    'ctrlKey',
    'metaKey'
  ])('Cmd/Ctrl-K opens from a collapsed workbench or text input (%s)', modifier => {
    const {store} = makeStore({
      palette: {open: false},
      dock: {open: false, height: 380, inspectorTab: 'result', logOpen: false}
    });
    const open = vi.spyOn(store, 'openPalette');
    let listener: (event: unknown) => void = () => {};
    const ownerDocument = {
      addEventListener: vi.fn((_name, callback) => {
        listener = callback;
      }),
      removeEventListener: vi.fn()
    };
    const dispose = subscribePaletteShortcut(store, ownerDocument as unknown as Document);
    const event = {key: 'k', [modifier]: true, target: {tagName: 'INPUT'}, preventDefault: vi.fn()};
    listener({...event, isComposing: true});
    listener({...event, altKey: true});
    listener({...event, defaultPrevented: true});
    listener({...event, shiftKey: true});
    expect(open).not.toHaveBeenCalled();
    listener(event);
    expect(open).toHaveBeenCalledTimes(1);
    expect(event.preventDefault).toHaveBeenCalledTimes(1);
    dispose();
    expect(ownerDocument.removeEventListener).toHaveBeenCalledWith('keydown', listener, true);
  });
});
