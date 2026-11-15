// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {readFileSync} from 'node:fs';
import {afterAll, describe, expect, test, vi} from 'vitest';
import type {Context, ReactElement, ReactNode, KeyboardEvent} from 'react';
import * as flowLibrary from '@xyflow/react';
import {ReactFlow, ReactFlowProvider} from '@xyflow/react';
import type {
  WorkflowRunnerState,
  WorkflowStepResult,
  WorkflowStepState
} from '../../../src/workflows/workflow-types';
import {
  makeFixtureDocument,
  makeFixtureHost,
  makeFixturePoints,
  makeFixtureRegistry
} from '../workflow-fixtures';
import {
  makeWorkbenchHandleId,
  parseWorkbenchHandleId,
  makeWorkbenchNodes,
  makeWorkbenchEdges,
  getWorkbenchStepSummary,
  getWorkbenchStepStatus,
  reconcileWorkbenchNodes,
  applyWorkbenchSelectionChanges,
  shouldShowWorkbenchMinimap,
  getWorkbenchFitViewOptions,
  getWorkbenchRevealViewport,
  getWorkbenchGroupRevealViewport,
  getWorkbenchNodeRectangle,
  getWorkbenchNodeBounds,
  WORKBENCH_CANVAS_LAYOUT
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-canvas-model';
import {
  WorkbenchCanvas,
  WorkbenchCanvasContent,
  type WorkbenchCanvasApi
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-canvas';
import {
  WorkbenchNode,
  WorkbenchConnectionContext,
  type WorkbenchKeyboardConnection
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-node';
import {
  moveWorkflowSteps,
  disconnectWorkflowInput,
  setWorkflowDisplayStyle
} from '../../../src/workflows/workflow-edit';
import {getWorkflowOperationRegistry} from '../../../src/workflows/workflow-operations';
import {WorkbenchStoreContext} from '../../../../../examples/deck/kepler-compute/src/workflow/use-workbench';
import {
  WORKBENCH_COLORS,
  WORKBENCH_DOCK,
  WORKBENCH_STATUS,
  WORKBENCH_TEST_IDS
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-contract';
import type {
  WorkbenchState,
  WorkbenchStore
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-store';
import {createWorkbenchStore} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-store';

// The dock can cache the canvas before this suite installs its React Flow probe.
vi.hoisted(() => vi.resetModules());
afterAll(() => {
  vi.doUnmock('@xyflow/react');
  vi.resetModules();
});
const flowProbe = vi.hoisted(() => ({current: null as unknown}));
vi.mock('@xyflow/react', async importOriginal => {
  const actual = await importOriginal<typeof import('@xyflow/react')>();
  return {
    ...actual,
    useReactFlow: () => flowProbe.current ?? actual.useReactFlow(),
    useNodesInitialized: () =>
      flowProbe.current
        ? ((flowProbe.current as {nodesInitialized?: boolean}).nodesInitialized ?? true)
        : actual.useNodesInitialized()
  };
});

const hostRequire = createRequire(
  new URL('../../../../../examples/deck/kepler-compute/package.json', import.meta.url)
);
const {createElement} = hostRequire('react') as typeof import('react');
const {renderToStaticMarkup} = hostRequire('react-dom/server') as typeof import('react-dom/server');
const {ServerStyleSheet} = hostRequire('styled-components') as typeof import('styled-components');

function makeResult(patch: Partial<WorkflowStepResult> = {}): WorkflowStepResult {
  return {
    key: 'current',
    outputs: {data: makeFixturePoints(4)},
    diagnostics: [],
    summary: {},
    engine: 'host',
    dispatchedContributors: [],
    parameters: {},
    durationMilliseconds: 42,
    cacheHit: false,
    completedAt: 0,
    ...patch
  };
}
function makeRunner(steps: Record<string, WorkflowStepState> = {}): WorkflowRunnerState {
  return {revision: 1, running: false, order: ['source', 'scale', 'count'], issues: [], steps};
}
function makeStore(patch: Partial<WorkbenchState> = {}): WorkbenchStore {
  const document = makeFixtureDocument();
  let state: WorkbenchState = {
    revision: 0,
    workflows: [document],
    activeWorkflowId: document.id,
    document,
    runner: null,
    selectedStepIds: [],
    canUndo: false,
    canRedo: false,
    dock: {open: true, height: 380, inspectorTab: 'parameters', logOpen: false},
    palette: {open: false},
    dialog: null,
    draft: null,
    notice: null,
    runLog: [],
    templates: [],
    clipboardStepCount: 0,
    ...patch
  };
  const calls: unknown[] = [];
  return {
    registry: makeFixtureRegistry(),
    host: makeFixtureHost(),
    getState: () => state,
    subscribe: () => () => {},
    createWorkflow: options => {
      calls.push(['create', options]);
      return 'created';
    },
    addWorkflow: next => {
      calls.push(['add', next]);
      return next.id;
    },
    duplicateWorkflow: id => {
      calls.push(['duplicate', id]);
      return 'copy';
    },
    deleteWorkflow: id => {
      calls.push(['delete', id]);
    },
    setActiveWorkflow: id => {
      calls.push(['active', id]);
    },
    edit: (tag, change) => {
      calls.push(['edit', tag, change]);
      if (!state.document) return false;
      try {
        const document = change(state.document);
        if (document !== state.document)
          state = {...state, document, workflows: [document], revision: state.revision + 1};
        return true;
      } catch (error) {
        state = {
          ...state,
          notice: {kind: 'error', message: error instanceof Error ? error.message : String(error)}
        };
        return false;
      }
    },
    undo: () => {
      calls.push('undo');
    },
    redo: () => {
      calls.push('redo');
    },
    select: ids => {
      calls.push(['select', ids]);
      state = {...state, selectedStepIds: ids};
    },
    run: async options => {
      calls.push(['run', options]);
      return null;
    },
    cancel: () => {
      calls.push('cancel');
    },
    notifyExternalChange: () => {
      calls.push('external');
    },
    getRunner: () => null,
    setDock: next => {
      calls.push(['dock', next]);
    },
    openPalette: options => {
      calls.push(['palette', options]);
      state = {...state, palette: {open: true, ...options}};
    },
    closePalette: () => {
      calls.push('close');
    },
    setDialog: dialog => {
      calls.push(['dialog', dialog]);
    },
    setNotice: notice => {
      calls.push(['notice', notice]);
      state = {...state, notice};
    },
    proposeDraft: draft => {
      calls.push(['draft', draft]);
    },
    acceptDraft: () => null,
    rejectDraft: () => {
      calls.push('reject');
    },
    saveTemplate: () => '',
    deleteTemplate: () => {},
    useTemplate: () => '',
    insertWorkflow: () => false,
    copySelection: () => false,
    cutSelection: () => false,
    paste: () => false,
    serialize: () => undefined,
    restore: collection => {
      calls.push(['restore', collection]);
    },
    clear: async () => {
      calls.push('clear');
    }
  };
}

function renderNode(
  store: WorkbenchStore,
  stepId = 'scale',
  selected = true,
  stylesheet?: InstanceType<typeof ServerStyleSheet>
): string {
  const node = makeWorkbenchNodes(store.getState().document!, selected ? [stepId] : []).find(
    candidate => candidate.id === stepId
  )!;
  const tree = createElement(
    WorkbenchStoreContext.Provider,
    {value: store},
    createElement(ReactFlowProvider, {
      initialNodes: [{...node, width: 220, height: 160, measured: {width: 220, height: 160}}],
      children: createElement(ReactFlow, {
        width: 800,
        height: 400,
        nodeTypes: {step: WorkbenchNode}
      })
    })
  );
  return renderToStaticMarkup(stylesheet ? stylesheet.collectStyles(tree) : tree);
}

describe('workbench canvas model', () => {
  test('status falls back safely without a runner', () => {
    const step = makeFixtureDocument().steps[1];
    const operation = makeFixtureRegistry().get(step.operationId)!;
    expect(getWorkbenchStepStatus(step, operation, undefined)).toBe('ready');
    expect(getWorkbenchStepStatus({...step, inputs: {}}, operation, undefined)).toBe(
      'not-configured'
    );
    expect(getWorkbenchStepStatus(step, undefined, undefined)).toBe('not-configured');
    expect(getWorkbenchStepStatus(step, operation, {status: 'queued', key: null, issues: []})).toBe(
      'queued'
    );
    expect(
      getWorkbenchStepStatus({...step, disabled: true}, operation, {
        status: 'done',
        key: 'current',
        issues: []
      })
    ).toBe('disabled');
  });
  test('handle ids round trip without interpreting the port name', () => {
    for (const port of ['points', 'a:b-c::', '', 'in:out: - / 日本語']) {
      for (const direction of ['in', 'out'] as const) {
        const id = makeWorkbenchHandleId(direction, port);
        expect(id).toBe(`${direction}:${port}`);
        expect(parseWorkbenchHandleId(id)).toEqual({direction, port});
      }
    }
    for (const id of ['points', 'in', 'IN:points', 'other:points', ':'])
      expect(parseWorkbenchHandleId(id)).toBeNull();
  });
  test('node identities follow immutable step identity and selection', () => {
    const document = makeFixtureDocument();
    const first = makeWorkbenchNodes(document, ['scale']);
    expect(first[1]).toEqual({
      id: 'scale',
      type: 'step',
      position: {x: 280, y: 0},
      data: {stepId: 'scale'},
      selected: true
    });
    const second = makeWorkbenchNodes({...document, title: 'Renamed'}, ['scale']);
    first.forEach((node, index) => expect(second[index]).toBe(node));
    const deselected = makeWorkbenchNodes(document, []);
    expect(deselected[0]).toBe(first[0]);
    expect(deselected[1]).not.toBe(first[1]);
    expect(makeWorkbenchNodes(document, ['scale'])[1]).toBe(first[1]);
    const moved = {
      ...document,
      steps: document.steps.map(step =>
        step.id === 'scale' ? {...step, position: {x: 99, y: 2}} : step
      )
    };
    expect(makeWorkbenchNodes(moved, ['scale'])[1]).not.toBe(first[1]);
    expect(makeWorkbenchNodes({...document, steps: []}, [])).toEqual([]);
  });
  test('edges use observed kinds and exact handles, and derive running, stale, invalid and ready', () => {
    const document = makeFixtureDocument();
    const registry = makeFixtureRegistry();
    const runner = makeRunner({
      source: {status: 'done', key: 'current', issues: [], result: makeResult()},
      scale: {status: 'running', key: 'scale', issues: []}
    });
    expect(makeWorkbenchEdges(document, registry, runner)[0]).toEqual({
      id: 'scale:points',
      source: 'source',
      sourceHandle: 'out:data',
      target: 'scale',
      targetHandle: 'in:points',
      data: {kind: 'point', state: 'running'}
    });
    runner.steps['scale'].status = 'ready';
    expect(makeWorkbenchEdges(document, registry, runner)[0].data).toEqual({
      kind: 'point',
      state: 'ready'
    });
    runner.steps['source'].status = 'stale';
    expect(makeWorkbenchEdges(document, registry, runner)[0].data?.state).toBe('stale');
    runner.steps['scale'].issues = [
      {severity: 'error', code: 'geometry-kind-not-accepted', message: 'Wrong kind', port: 'points'}
    ];
    expect(makeWorkbenchEdges(document, registry, runner)[0].data?.state).toBe('invalid');
    runner.steps['scale'].issues[0].severity = 'warning';
    expect(makeWorkbenchEdges(document, registry, runner)[0].data?.state).toBe('stale');
    expect(makeWorkbenchEdges(document, registry, null)[0].data).toEqual({
      kind: 'features',
      state: 'stale'
    });
  });
  test('unknown ports stay invalid, missing nodes are skipped', () => {
    const document = makeFixtureDocument();
    document.steps[1].inputs = {'odd:input': {stepId: 'source', output: 'odd:output'}};
    expect(makeWorkbenchEdges(document, makeFixtureRegistry(), null)[0]).toEqual({
      id: 'scale:odd:input',
      source: 'source',
      sourceHandle: 'out:odd:output',
      target: 'scale',
      targetHandle: 'in:odd:input',
      data: {kind: 'table', state: 'invalid'}
    });
    document.steps[1].inputs = {points: {stepId: 'missing', output: 'data'}};
    expect(makeWorkbenchEdges(document, makeFixtureRegistry(), null).map(edge => edge.id)).toEqual([
      'count:rows'
    ]);
    document.steps[0].operationId = 'unknown';
    document.steps[1].inputs = {points: {stepId: 'source', output: 'data'}};
    expect(makeWorkbenchEdges(document, makeFixtureRegistry(), null)[0].data?.state).toBe(
      'invalid'
    );
  });
  test('summaries prioritise errors, then results, then input hints', () => {
    const document = makeFixtureDocument();
    const operation = makeFixtureRegistry().get('fixture-source')!;
    const step = document.steps[0];
    const state: WorkflowStepState = {
      status: 'done',
      key: 'current',
      issues: [],
      result: makeResult()
    };
    expect(getWorkbenchStepSummary(step, operation, state)).toBe('4 rows · Host · 42 ms');
    state.result = makeResult({engine: 'gpu', durationMilliseconds: 1800});
    expect(getWorkbenchStepSummary(step, operation, state)).toBe('4 rows · GPU · 1.8 s');
    state.result = makeResult({engine: 'cpu-reference', cacheHit: true});
    expect(getWorkbenchStepSummary(step, operation, state)).toBe('4 rows · CPU · cached');
    state.issues = [
      {severity: 'warning', code: 'gpu-fallback', message: 'Warning'},
      {severity: 'error', code: 'operation-failed', message: 'Fixture source: failed.'}
    ];
    expect(getWorkbenchStepSummary(step, operation, state)).toBe('Fixture source: failed.');
    expect(getWorkbenchStepSummary(step, operation, undefined)).toBe('Ready to run');
    expect(
      getWorkbenchStepSummary(
        {...document.steps[1], inputs: {}},
        makeFixtureRegistry().get('fixture-scale'),
        undefined
      )
    ).toBe('Needs an input for "Points"');
    expect(getWorkbenchStepSummary(step, undefined, undefined)).toBe('Unknown operation');
  });
  test('summaries name the first declared output, format row counts and include runtime errors', () => {
    const step = makeFixtureDocument().steps[1];
    const operation = makeFixtureRegistry().get('fixture-scale')!;
    const points = makeFixturePoints(1);
    const state: WorkflowStepState = {
      status: 'stale',
      key: 'new',
      issues: [],
      result: makeResult({
        outputs: {unrelated: points, scaled: {...points, table: {...points.table, rowCount: 1204}}},
        durationMilliseconds: 1000
      })
    };
    expect(getWorkbenchStepSummary(step, operation, state)).toBe('1,204 rows · Host · 1 s');
    state.error = {
      severity: 'error',
      code: 'operation-failed',
      message: 'Fixture scale: could not run.'
    };
    expect(getWorkbenchStepSummary(step, operation, state)).toBe('Fixture scale: could not run.');
    expect(
      getWorkbenchStepSummary(
        {...step, inputs: {}},
        {...operation, inputs: operation.inputs.map(port => ({...port, optional: true}))},
        undefined
      )
    ).toBe('Ready to run');
  });
});

describe('workbench canvas markup', () => {
  test('no workflow and empty workflow show named actions', () => {
    const absent = renderToStaticMarkup(
      createElement(WorkbenchCanvas, {
        store: makeStore({document: null, workflows: [], activeWorkflowId: null})
      })
    );
    expect(absent).toContain(`data-testid="${WORKBENCH_TEST_IDS.canvasEmpty}"`);
    expect(absent).toContain('No workflow yet');
    expect(absent).toContain('New workflow');
    expect(absent).toContain('Start from a template');
    expect(absent).toContain('aria-label="Workflow graph"');
    expect(absent).toContain('tabindex="0"');
    const document = {...makeFixtureDocument(), steps: []};
    const empty = renderToStaticMarkup(
      createElement(WorkbenchCanvas, {store: makeStore({document})})
    );
    expect(empty).toContain('Add a step to begin');
    expect(empty).toContain('Add step');
    expect(empty).toContain('Press N, or click here and press Tab');
  });
  test('custom node renders typed ports, status and named toggle buttons through React Flow', () => {
    const markup = renderNode(makeStore());
    expect(markup).toContain(`data-testid="${WORKBENCH_TEST_IDS.node('scale')}"`);
    expect(markup).toContain('Fixture scale');
    expect(markup).toContain('data-status="ready"');
    expect(markup).toContain('aria-label="Show Fixture scale on the map"');
    expect(getMarkupTag(markup, WORKBENCH_TEST_IDS.nodeDisplay('scale'))).toContain(
      'aria-pressed="true"'
    );
    expect(getMarkupTag(markup, WORKBENCH_TEST_IDS.nodeDisable('scale'))).toContain(
      'aria-pressed="false"'
    );
    expect(markup).toContain('aria-label="Disable Fixture scale"');
    expect(markup).toContain(`data-testid="${WORKBENCH_TEST_IDS.inputPort('scale', 'points')}"`);
    expect(markup).toContain(`data-testid="${WORKBENCH_TEST_IDS.outputPort('scale', 'scaled')}"`);
    expect(markup).toContain('Points');
    for (const testId of [
      WORKBENCH_TEST_IDS.inputPort('scale', 'points'),
      WORKBENCH_TEST_IDS.outputPort('scale', 'scaled')
    ]) {
      const handle = getMarkupTag(markup, testId);
      expect(handle).toContain('tabindex="0"');
      expect(handle).toContain('role="button"');
    }
  });
  test('unknown, missing input, disabled, error, running and stale nodes render', () => {
    const unknown = makeFixtureDocument();
    unknown.steps[1].operationId = 'removed-operation';
    const placeholder = renderNode(makeStore({document: unknown}));
    expect(placeholder).toContain('removed-operation');
    expect(placeholder).toContain('Unknown operation');
    expect(placeholder).toContain('data-status="not-configured"');
    const document = makeFixtureDocument();
    document.steps[1].inputs = {};
    expect(renderNode(makeStore({document}))).toContain('required');
    document.steps[1].disabled = true;
    expect(renderNode(makeStore({document}))).toContain('data-status="disabled"');
    for (const status of ['error', 'running', 'stale'] as const) {
      const state: WorkflowStepState = {
        status,
        key: 'new',
        issues:
          status === 'error'
            ? [{severity: 'error', code: 'operation-failed', message: 'Fixture scale: failed.'}]
            : [],
        ...(status === 'stale'
          ? {result: makeResult({outputs: {scaled: makeFixturePoints(4)}})}
          : {})
      };
      const markup = renderNode(makeStore({runner: makeRunner({scale: state})}));
      expect(markup).toContain(`data-status="${status}"`);
      if (status === 'error') expect(markup).toContain('Fixture scale: failed.');
      if (status === 'stale') expect(markup).toContain('4 rows · Host · 42 ms');
    }
  });
  test('saved connections retain handles when their ports disappear', () => {
    const document = makeFixtureDocument();
    document.steps[1].inputs = {'old:input': {stepId: 'source', output: 'old:output'}};
    const store = makeStore({document});
    const target = renderNode(store);
    expect(target).toContain(`data-testid="${WORKBENCH_TEST_IDS.inputPort('scale', 'old:input')}"`);
    expect(target).toContain('old:input (unknown input)');
    const source = renderNode(store, 'source');
    expect(source).toContain(
      `data-testid="${WORKBENCH_TEST_IDS.outputPort('source', 'old:output')}"`
    );
    expect(source).toContain('old:output (unknown output)');
  });
});

// Drive the actual component callbacks without a DOM or a second React renderer dependency.
// React owns static rendering above; this dispatcher is installed only during a probe render.
function makeHookProbe<Result>(
  render: () => Result,
  store: WorkbenchStore,
  keyboardConnection?: () => WorkbenchKeyboardConnection | null,
  beforeEffects?: (result: Result) => void
) {
  const react = hostRequire('react') as {
    __CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE: {H: unknown};
  };
  const internals = react.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;
  const slots = new Map<number, unknown>();
  const dependencies = new Map<number, readonly unknown[] | undefined>();
  const cleanups = new Map<number, (() => void) | undefined>();
  let cursor = 0;
  let dirty = false;
  let effects: Array<() => void> = [];
  function changed(position: number, next: readonly unknown[] | undefined) {
    const previous = dependencies.get(position);
    return (
      !dependencies.has(position) ||
      !next ||
      !previous ||
      next.length !== previous.length ||
      next.some((value, index) => !Object.is(value, previous[index]))
    );
  }
  const dispatcher = {
    useState(initial: unknown) {
      const position = cursor++;
      if (!slots.has(position))
        slots.set(position, typeof initial === 'function' ? initial() : initial);
      return [
        slots.get(position),
        (update: unknown) => {
          const previous = slots.get(position);
          const next = typeof update === 'function' ? update(previous) : update;
          if (!Object.is(previous, next)) {
            slots.set(position, next);
            dirty = true;
          }
        }
      ];
    },
    useRef(initial: unknown) {
      const position = cursor++;
      if (!slots.has(position)) slots.set(position, {current: initial});
      return slots.get(position);
    },
    useMemo(calculate: () => unknown, next: readonly unknown[] | undefined) {
      const position = cursor++;
      if (changed(position, next)) {
        slots.set(position, calculate());
        dependencies.set(position, next);
      }
      return slots.get(position);
    },
    useCallback(callback: unknown, next: readonly unknown[] | undefined) {
      return dispatcher.useMemo(() => callback, next);
    },
    useEffect(effect: () => (() => void) | void, next: readonly unknown[] | undefined) {
      const position = cursor++;
      if (changed(position, next)) {
        dependencies.set(position, next);
        effects.push(() => {
          cleanups.get(position)?.();
          cleanups.set(position, effect() || undefined);
        });
      }
    },
    useSyncExternalStore(_subscribe: unknown, getSnapshot: () => unknown) {
      cursor++;
      return getSnapshot();
    },
    useContext(context: Context<unknown>) {
      cursor++;
      return context === WorkbenchStoreContext
        ? store
        : context === WorkbenchConnectionContext && keyboardConnection
          ? keyboardConnection()
          : (context as unknown as {_currentValue: unknown})._currentValue;
    }
  };
  return {
    read(): Result {
      let result: Result;
      for (let iteration = 0; iteration < 20; iteration++) {
        dirty = false;
        cursor = 0;
        const previous = internals.H;
        internals.H = dispatcher;
        try {
          result = render();
        } finally {
          internals.H = previous;
        }
        beforeEffects?.(result);
        const pending = effects;
        effects = [];
        pending.forEach(effect => effect());
        if (!dirty) return result;
      }
      throw new Error('Component probe did not settle within 20 renders.');
    },
    close() {
      cleanups.forEach(cleanup => cleanup?.());
    }
  };
}

type ProbeElement = ReactElement<Record<string, unknown>>;
function findProbeElement(
  tree: ReactNode,
  predicate: (element: ProbeElement) => boolean
): ProbeElement {
  const react = hostRequire('react') as typeof import('react');
  function visit(child: ReactNode): ProbeElement | undefined {
    if (Array.isArray(child)) {
      for (const entry of child) {
        const found = visit(entry);
        if (found) return found;
      }
    } else if (react.isValidElement<Record<string, unknown>>(child)) {
      if (predicate(child)) return child;
      return visit(child.props['children'] as ReactNode);
    }
    return undefined;
  }
  const found = visit(tree);
  if (!found) throw new Error('Requested component element is missing.');
  return found;
}
function makeKeyboardEvent(key: string) {
  return {
    key,
    target: {closest: () => null},
    defaultPrevented: false,
    preventDefault: vi.fn(),
    stopPropagation: vi.fn(),
    ctrlKey: false,
    metaKey: false,
    altKey: false,
    shiftKey: false
  } as unknown as KeyboardEvent<HTMLDivElement>;
}
function makeLiveStore() {
  const store = createWorkbenchStore({host: makeFixtureHost(), registry: makeFixtureRegistry()});
  store.addWorkflow(makeFixtureDocument());
  return store;
}
function withCanvasProbe(
  run: (probe: {read(): ProbeElement; close(): void}, store: WorkbenchStore) => void
) {
  const store = makeLiveStore();
  const flow = {
    fitView: vi.fn(async () => true),
    getViewport: () => ({x: 0, y: 0, zoom: 1}),
    screenToFlowPosition: (point: {x: number; y: number}) => point
  };
  flowProbe.current = flow;
  vi.stubGlobal('window', {
    matchMedia: () => ({matches: false, addEventListener: () => {}, removeEventListener: () => {}})
  });
  const probe = makeHookProbe(() => WorkbenchCanvasContent({store}), store);
  try {
    run(probe, store);
  } finally {
    probe.close();
    flowProbe.current = null;
    vi.unstubAllGlobals();
  }
}
function getFlowProps(tree: ReactNode) {
  return findProbeElement(tree, element => element.type === ReactFlow)
    .props as unknown as flowLibrary.ReactFlowProps;
}

describe('empty canvas pointer focus', () => {
  test.each([false, true])('overlay passes through clicks with an active workflow=%s', active => {
    const store = makeStore({
      document: active ? {...makeFixtureDocument(), steps: []} : null
    });
    const stylesheet = new ServerStyleSheet();
    try {
      const markup = renderToStaticMarkup(
        stylesheet.collectStyles(createElement(WorkbenchCanvas, {store}))
      );
      const styles = stylesheet.getStyleTags();
      const overlay = getMarkupTag(markup, WORKBENCH_TEST_IDS.canvasEmpty);
      const classes = new Set(overlay.match(/class="([^"]*)"/)![1].split(' '));
      const declarations = [...styles.matchAll(/\.([\w-]+)\{([^}]*)\}/g)]
        .filter(match => classes.has(match[1]))
        .map(match => match[2])
        .join('');
      expect(declarations).toContain('pointer-events:none;');
      expect(
        [...classes].some(className =>
          styles.includes(`.${className} button{pointer-events:auto;}`)
        )
      ).toBe(true);
      expect(markup).toContain(active ? '>Add step</button>' : '>New workflow</button>');
      if (active) expect(markup).toContain('Press N, or click here and press Tab');
    } finally {
      stylesheet.seal();
    }
  });

  test('pane pointerdown focuses the empty canvas before React Flow handles bubbling', () => {
    withCanvasProbe((probe, store) => {
      store.edit('empty', document => ({...document, steps: []}));
      const tree = probe.read();
      const focus = vi.fn();
      (tree.props['ref'] as {current: unknown}).current = {focus};
      class PointerTarget {
        constructor(private readonly interactive = false) {}
        closest() {
          return this.interactive ? this : null;
        }
      }
      vi.stubGlobal('Element', PointerTarget);
      const capture = tree.props['onPointerDownCapture'] as (event: {target: object}) => void;
      expect(typeof capture).toBe('function');
      capture({target: new PointerTarget()});
      expect(focus).toHaveBeenCalledExactlyOnceWith({preventScroll: true});
      capture({target: new PointerTarget(true)});
      expect(focus).toHaveBeenCalledTimes(1);
    });
  });
});

function getNodeContent(
  store: WorkbenchStore,
  stepId = 'scale',
  keyboardConnection?: () => WorkbenchKeyboardConnection | null
) {
  const node = makeWorkbenchNodes(store.getState().document!, [])[
    store.getState().document!.steps.findIndex(step => step.id === stepId)
  ];
  const component = (
    WorkbenchNode as unknown as {type: (props: flowLibrary.NodeProps) => ReactElement}
  ).type;
  return makeHookProbe(
    () => component({...node, data: node.data} as unknown as flowLibrary.NodeProps),
    store,
    keyboardConnection
  );
}

function getMarkupTag(markup: string, testId: string): string {
  const position = markup.indexOf(`data-testid="${testId}"`);
  expect(position).toBeGreaterThanOrEqual(0);
  return markup.slice(markup.lastIndexOf('<', position), markup.indexOf('>', position) + 1);
}
function pressPort(tree: ReactNode, testId: string, key: string) {
  const event = makeKeyboardEvent(key);
  const handle = findProbeElement(tree, element => element.props['data-testid'] === testId);
  (handle.props['onKeyDown'] as (event: KeyboardEvent<HTMLDivElement>) => void)(event);
  return event;
}

describe('review regressions', () => {
  test('F1 completed normalized no-op and refused moves reconcile visible positions', () => {
    withCanvasProbe((probe, store) => {
      for (const [saved, requested] of [
        [280, 280.4],
        [100000, 100001],
        [280, Number.NaN]
      ]) {
        store.edit('setup', document => ({
          ...document,
          steps: document.steps.map(step =>
            step.id === 'scale' ? {...step, position: {x: saved, y: 0}} : step
          )
        }));
        const document = store.getState().document;
        getFlowProps(probe.read()).onNodesChange!([
          {type: 'position', id: 'scale', position: {x: requested, y: 0}, dragging: false}
        ]);
        const visible = getFlowProps(probe.read()).nodes!.find(node => node.id === 'scale')!;
        expect(store.getState().document).toBe(document);
        expect(visible.position).toEqual({x: saved, y: 0});
      }
    });
  });
  test('F2 selected connections stay selected and Delete disconnects them', () => {
    withCanvasProbe((probe, store) => {
      getFlowProps(probe.read()).onEdgesChange!([
        {type: 'select', id: 'scale:points', selected: true}
      ]);
      const tree = probe.read();
      const properties = getFlowProps(tree);
      expect(properties.edges!.find(edge => edge.id === 'scale:points')!.selected).toBe(true);
      expect(properties.onSelectionChange).toBeUndefined();
      expect(properties.deleteKeyCode).toBeNull();
      const event = makeKeyboardEvent('Delete');
      (
        tree.props as unknown as {onKeyDown: (event: KeyboardEvent<HTMLDivElement>) => void}
      ).onKeyDown(event);
      expect(store.getState().document!.steps.find(step => step.id === 'scale')!.inputs).toEqual(
        {}
      );
      expect(event.stopPropagation).toHaveBeenCalledOnce();
    });
  });
  test('F3 input and output handles expose keyboard activation', () => {
    const store = makeLiveStore();
    const probe = getNodeContent(store);
    try {
      const tree = probe.read();
      for (const testId of [
        WORKBENCH_TEST_IDS.inputPort('scale', 'points'),
        WORKBENCH_TEST_IDS.outputPort('scale', 'scaled')
      ]) {
        const handle = findProbeElement(tree, element => element.props['data-testid'] === testId);
        expect(handle.props['tabIndex']).toBe(0);
        expect(handle.props['role']).toBe('button');
        expect(typeof handle.props['onKeyDown']).toBe('function');
      }
    } finally {
      probe.close();
    }
  });
  test('F4 the first declared output may have an empty name', () => {
    const step = makeFixtureDocument().steps[0];
    const operation = makeFixtureRegistry().get(step.operationId)!;
    const first = {...operation.outputs[0], name: ''};
    const state: WorkflowStepState = {
      status: 'done',
      key: 'current',
      issues: [],
      result: makeResult({outputs: {other: makeFixturePoints(99), '': makeFixturePoints(3)}})
    };
    expect(
      getWorkbenchStepSummary(
        step,
        {...operation, outputs: [first, {...first, name: 'other'}]},
        state
      )
    ).toBe('3 rows · Host · 42 ms');
  });
  test('F5 focused node wrappers have an outline independent of selection', () => {
    const stylesheet = new ServerStyleSheet();
    try {
      const markup = renderNode(makeStore(), 'scale', false);
      const wrapper = markup.match(/<div[^>]*class="[^"]*react-flow__node-step[^"]*"[^>]*>/)?.[0];
      expect(wrapper).toContain('tabindex="0"');
      expect(wrapper).not.toContain('selected');
      renderToStaticMarkup(
        stylesheet.collectStyles(createElement(WorkbenchCanvas, {store: makeStore()}))
      );
      const styles = stylesheet.getStyleTags();
      expect(styles).toMatch(/\.react-flow__node:focus-visible\s*\{[^}]*outline:\s*2px solid/);
      const libraryStyles = readFileSync(
        hostRequire.resolve('@xyflow/react/dist/style.css'),
        'utf8'
      );
      expect(libraryStyles).toMatch(
        /\.react-flow__node\.selectable:focus-visible\s*\{\s*outline: none/
      );
    } finally {
      stylesheet.seal();
    }
  });
});

describe('component callback coverage', () => {
  test('the fake store executes edits, preserves no-ops and reports refusals', () => {
    const store = makeStore();
    const original = store.getState().document;
    expect(
      store.edit('move', document => moveWorkflowSteps(document, {scale: {x: 280.4, y: 0}}))
    ).toBe(true);
    expect(store.getState().document).toBe(original);
    expect(
      store.edit('move', document => moveWorkflowSteps(document, {scale: {x: 301.8, y: 9.2}}))
    ).toBe(true);
    expect(store.getState().document!.steps[1].position).toEqual({x: 302, y: 9});
    const moved = store.getState().document;
    expect(
      store.edit('move', document => moveWorkflowSteps(document, {scale: {x: Number.NaN, y: 0}}))
    ).toBe(false);
    expect(store.getState().document).toBe(moved);
    expect(store.getState().notice).toEqual({
      kind: 'error',
      message: 'fixture-scale: Canvas coordinates must be finite numbers.'
    });
  });
  test('dragging is transient and drag stop reconciles the real edit result', () => {
    withCanvasProbe((probe, store) => {
      const original = store.getState().document;
      getFlowProps(probe.read()).onNodesChange!([
        {type: 'position', id: 'scale', position: {x: 301.8, y: 9.2}, dragging: true}
      ]);
      const properties = getFlowProps(probe.read());
      const moved = properties.nodes!.find(node => node.id === 'scale')!;
      expect(moved.position).toEqual({x: 301.8, y: 9.2});
      expect(store.getState().document).toBe(original);
      properties.onNodeDragStop!({} as MouseEvent, moved, [moved]);
      expect(store.getState().document!.steps[1].position).toEqual({x: 302, y: 9});
      expect(getFlowProps(probe.read()).nodes!.find(node => node.id === 'scale')!.position).toEqual(
        {x: 302, y: 9}
      );
    });
  });
  test('deletion consumes mixed edge and node selection in one undoable edit', () => {
    withCanvasProbe((probe, store) => {
      const original = store.getState().document;
      const properties = getFlowProps(probe.read());
      properties.onEdgesChange!([{type: 'select', id: 'scale:points', selected: true}]);
      properties.onNodesChange!([{type: 'select', id: 'count', selected: true}]);
      const tree = probe.read();
      const event = makeKeyboardEvent('Backspace');
      (tree.props['onKeyDown'] as (event: KeyboardEvent<HTMLDivElement>) => void)(event);
      expect(store.getState().document!.steps.map(step => step.id)).toEqual(['source', 'scale']);
      expect(store.getState().document!.steps[1].inputs).toEqual({});
      expect(event.preventDefault).toHaveBeenCalledOnce();
      expect(event.stopPropagation).toHaveBeenCalledOnce();
      expect(getFlowProps(probe.read()).edges).toEqual([]);
      store.undo();
      expect(store.getState().document).toEqual(original);
    });
  });
  test('edge deselection and text-field deletion leave connections intact', () => {
    withCanvasProbe((probe, store) => {
      const original = store.getState().document;
      getFlowProps(probe.read()).onEdgesChange!([
        {type: 'select', id: 'scale:points', selected: true}
      ]);
      const tree = probe.read();
      const event = makeKeyboardEvent('Delete');
      event.target = {closest: () => ({})} as unknown as EventTarget & HTMLDivElement;
      (tree.props['onKeyDown'] as (event: KeyboardEvent<HTMLDivElement>) => void)(event);
      expect(event.preventDefault).not.toHaveBeenCalled();
      expect(store.getState().document).toBe(original);
      getFlowProps(tree).onEdgesChange!([{type: 'select', id: 'scale:points', selected: false}]);
      expect(
        getFlowProps(probe.read()).edges!.find(edge => edge.id === 'scale:points')!.selected
      ).toBe(false);
    });
  });
  test('each toggle updates the real document and its own pressed state', () => {
    const store = makeLiveStore();
    const probe = getNodeContent(store);
    try {
      for (const [testId, initial] of [
        [WORKBENCH_TEST_IDS.nodeDisplay('scale'), true],
        [WORKBENCH_TEST_IDS.nodeDisable('scale'), false]
      ] as const) {
        const button = findProbeElement(
          probe.read(),
          element => element.props['data-testid'] === testId
        );
        expect(button.props['aria-pressed']).toBe(initial);
        (button.props['onClick'] as () => void)();
        const step = store.getState().document!.steps[1];
        expect(
          testId === WORKBENCH_TEST_IDS.nodeDisplay('scale')
            ? Boolean(step.display?.includes('scaled'))
            : Boolean(step.disabled)
        ).toBe(!initial);
        const updated = findProbeElement(
          probe.read(),
          element => element.props['data-testid'] === testId
        );
        expect(updated.props['aria-pressed']).toBe(!initial);
        (updated.props['onClick'] as () => void)();
        expect(
          findProbeElement(probe.read(), element => element.props['data-testid'] === testId).props[
            'aria-pressed'
          ]
        ).toBe(initial);
      }
    } finally {
      probe.close();
    }
  });
  test.each([
    'out',
    'in'
  ] as const)('keyboard connection completes starting from an %s port', direction => {
    withCanvasProbe((probe, store) => {
      store.edit('disconnect', document => disconnectWorkflowInput(document, 'scale', 'points'));
      let connection = findProbeElement(
        probe.read(),
        element => element.type === WorkbenchConnectionContext.Provider
      ).props['value'] as WorkbenchKeyboardConnection;
      const source = getNodeContent(store, 'source', () => connection);
      const target = getNodeContent(store, 'scale', () => connection);
      try {
        const first = direction === 'out' ? source : target;
        const firstId =
          direction === 'out'
            ? WORKBENCH_TEST_IDS.outputPort('source', 'data')
            : WORKBENCH_TEST_IDS.inputPort('scale', 'points');
        const second = direction === 'out' ? target : source;
        const secondId =
          direction === 'out'
            ? WORKBENCH_TEST_IDS.inputPort('scale', 'points')
            : WORKBENCH_TEST_IDS.outputPort('source', 'data');
        const untouched = pressPort(first.read(), firstId, 'Tab');
        expect(untouched.preventDefault).not.toHaveBeenCalled();
        const begin = pressPort(first.read(), firstId, 'Enter');
        expect(begin.stopPropagation).toHaveBeenCalledOnce();
        connection = findProbeElement(
          probe.read(),
          element => element.type === WorkbenchConnectionContext.Provider
        ).props['value'] as WorkbenchKeyboardConnection;
        expect(
          findProbeElement(first.read(), element => element.props['data-testid'] === firstId).props[
            'aria-pressed'
          ]
        ).toBe(true);
        const end = pressPort(second.read(), secondId, ' ');
        expect(end.preventDefault).toHaveBeenCalledOnce();
        expect(store.getState().document!.steps[1].inputs).toEqual({
          points: {stepId: 'source', output: 'data'}
        });
        connection = findProbeElement(
          probe.read(),
          element => element.type === WorkbenchConnectionContext.Provider
        ).props['value'] as WorkbenchKeyboardConnection;
        expect(connection.pending).toBeNull();
        expect(store.getState().palette.open).toBe(false);
      } finally {
        source.close();
        target.close();
      }
    });
  });
  test('keyboard connection reports a refused cycle and Escape cancels the gesture', () => {
    withCanvasProbe((probe, store) => {
      let tree = probe.read();
      let connection = findProbeElement(
        tree,
        element => element.type === WorkbenchConnectionContext.Provider
      ).props['value'] as WorkbenchKeyboardConnection;
      const node = getNodeContent(store, 'scale', () => connection);
      const original = store.getState().document;
      try {
        pressPort(node.read(), WORKBENCH_TEST_IDS.outputPort('scale', 'scaled'), 'Enter');
        connection = findProbeElement(
          probe.read(),
          element => element.type === WorkbenchConnectionContext.Provider
        ).props['value'] as WorkbenchKeyboardConnection;
        pressPort(node.read(), WORKBENCH_TEST_IDS.inputPort('scale', 'points'), 'Enter');
        expect(store.getState().document).toBe(original);
        expect(store.getState().notice).toEqual({
          kind: 'error',
          message: 'Fixture scale: Input Points cannot create a cycle.'
        });
        tree = probe.read();
        const cancel = makeKeyboardEvent('Escape');
        (tree.props['onKeyDown'] as (event: KeyboardEvent<HTMLDivElement>) => void)(cancel);
        expect(cancel.stopPropagation).toHaveBeenCalledOnce();
        connection = findProbeElement(
          probe.read(),
          element => element.type === WorkbenchConnectionContext.Provider
        ).props['value'] as WorkbenchKeyboardConnection;
        expect(connection.pending).toBeNull();
        expect(store.getState().document).toBe(original);
      } finally {
        node.close();
      }
    });
  });
});

describe('headless review regressions', () => {
  test('B1 external selection preserves measurements without another selection owner', () => {
    withCanvasProbe((probe, store) => {
      getFlowProps(probe.read()).onNodesChange!([
        {
          type: 'dimensions',
          id: 'scale',
          dimensions: {width: 220, height: 164},
          setAttributes: true
        }
      ]);
      const measured = getFlowProps(probe.read()).nodes!.find(node => node.id === 'scale')!;
      expect(measured.measured).toEqual({width: 220, height: 164});
      store.select(['scale']);
      const properties = getFlowProps(probe.read());
      expect(properties.nodes!.find(node => node.id === 'scale')!.measured).toBe(measured.measured);
      expect(properties.onSelectionChange).toBeUndefined();
      expect(store.getState().selectedStepIds).toEqual(['scale']);
      const previousNodes = properties.nodes;
      store.select(['scale']);
      expect(getFlowProps(probe.read()).nodes).toBe(previousNodes);
    });
  });
});

describe('canvas reconciliation and layout', () => {
  test('reconciliation is idempotent for repeated store selection and equal model values', () => {
    const document = makeFixtureDocument();
    const initial = makeWorkbenchNodes(document, []);
    expect(reconcileWorkbenchNodes(initial, makeWorkbenchNodes(document, []))).toBe(initial);
    const selected = reconcileWorkbenchNodes(initial, makeWorkbenchNodes(document, ['scale']));
    expect(selected).not.toBe(initial);
    expect(selected[0]).toBe(initial[0]);
    expect(selected[1].selected).toBe(true);
    expect(reconcileWorkbenchNodes(selected, makeWorkbenchNodes(document, ['scale']))).toBe(
      selected
    );
    const copied = structuredClone(document);
    expect(reconcileWorkbenchNodes(selected, makeWorkbenchNodes(copied, ['scale']))).toBe(selected);
    const reordered = reconcileWorkbenchNodes(
      selected,
      [...makeWorkbenchNodes(document, ['scale'])].reverse()
    );
    expect(reordered.map(node => node.id)).toEqual(['count', 'scale', 'source']);
    expect(reordered[0]).toBe(selected[2]);
  });
  test('measurements, dimensions and dragging survive selection and another node moving', () => {
    const document = makeFixtureDocument();
    const initial = makeWorkbenchNodes(document, []).map(node => ({
      ...node,
      measured: {width: 220, height: 164},
      width: 220,
      height: 164,
      dragging: node.id === 'scale'
    }));
    const selected = reconcileWorkbenchNodes(initial, makeWorkbenchNodes(document, ['scale']));
    expect(selected[1]).toMatchObject({selected: true, width: 220, height: 164, dragging: true});
    expect(selected[1].measured).toBe(initial[1].measured);
    const movedDocument = moveWorkflowSteps(document, {count: {x: 600, y: 50}});
    const moved = reconcileWorkbenchNodes(selected, makeWorkbenchNodes(movedDocument, ['scale']));
    expect(moved[1]).toBe(selected[1]);
    expect(moved[2].position).toEqual({x: 600, y: 50});
    expect(moved[2].measured).toBe(initial[2].measured);
    expect(moved[2].width).toBe(220);
    expect(moved[2].height).toBe(164);
  });
  test('reconciliation adds and removes nodes and updates model data without keeping removed nodes', () => {
    const initial = makeWorkbenchNodes(makeFixtureDocument(), []);
    const added = {...initial[0], id: 'extra', data: {stepId: 'extra'}, position: {x: 44, y: 55}};
    const desired = [{...initial[1], data: {stepId: 'replacement'}}, added];
    const reconciled = reconcileWorkbenchNodes(initial, desired);
    expect(reconciled.map(node => node.id)).toEqual(['scale', 'extra']);
    expect(reconciled[0].data).toEqual({stepId: 'replacement'});
    expect(reconciled[1]).toBe(added);
    expect(reconcileWorkbenchNodes(reconciled, [])).toEqual([]);
    const empty: typeof initial = [];
    expect(reconcileWorkbenchNodes(empty, [])).toBe(empty);
  });
  test('selection changes preserve identity for no-ops and apply deselect and multi-select once', () => {
    const selected = ['scale'];
    expect(applyWorkbenchSelectionChanges(selected, [])).toBe(selected);
    expect(
      applyWorkbenchSelectionChanges(selected, [{type: 'select', id: 'scale', selected: true}])
    ).toBe(selected);
    expect(
      applyWorkbenchSelectionChanges(selected, [{type: 'select', id: 'count', selected: false}])
    ).toBe(selected);
    expect(
      applyWorkbenchSelectionChanges(selected, [{type: 'select', id: 'scale', selected: false}])
    ).toEqual([]);
    expect(
      applyWorkbenchSelectionChanges(selected, [
        {type: 'select', id: 'count', selected: true},
        {type: 'select', id: 'source', selected: true}
      ])
    ).toEqual(['scale', 'count', 'source']);
    expect(
      applyWorkbenchSelectionChanges(selected, [
        {type: 'select', id: 'scale', selected: false},
        {type: 'select', id: 'scale', selected: true}
      ])
    ).toBe(selected);
  });
  test('user selection changes update the store once and repeated reports leave nodes untouched', () => {
    withCanvasProbe((probe, store) => {
      const select = vi.spyOn(store, 'select');
      const change: flowLibrary.NodeSelectionChange[] = [
        {type: 'select', id: 'scale', selected: true},
        {type: 'select', id: 'count', selected: true}
      ];
      getFlowProps(probe.read()).onNodesChange!(change);
      expect(select).toHaveBeenCalledTimes(1);
      expect(store.getState().selectedStepIds).toEqual(['scale', 'count']);
      const previous = getFlowProps(probe.read()).nodes;
      getFlowProps(probe.read()).onNodesChange!(change);
      expect(select).toHaveBeenCalledTimes(1);
      expect(getFlowProps(probe.read()).nodes).toBe(previous);
      getFlowProps(probe.read()).onNodesChange!([{type: 'select', id: 'scale', selected: false}]);
      expect(store.getState().selectedStepIds).toEqual(['count']);
      expect(select).toHaveBeenCalledTimes(2);
    });
  });
  test('edge selection no-ops preserve the controlled edge array', () => {
    withCanvasProbe(probe => {
      getFlowProps(probe.read()).onEdgesChange!([
        {type: 'select', id: 'scale:points', selected: true}
      ]);
      const selected = getFlowProps(probe.read()).edges;
      getFlowProps(probe.read()).onEdgesChange!([
        {type: 'select', id: 'scale:points', selected: true}
      ]);
      expect(getFlowProps(probe.read()).edges).toBe(selected);
      getFlowProps(probe.read()).onEdgesChange!([{type: 'select', id: 'absent', selected: false}]);
      expect(getFlowProps(probe.read()).edges).toBe(selected);
    });
  });
  test('minimap thresholds use canvas height and graph size and frame padding reserves the overlays', () => {
    expect(shouldShowWorkbenchMinimap(319, 6)).toBe(false);
    expect(shouldShowWorkbenchMinimap(320, 5)).toBe(false);
    expect(shouldShowWorkbenchMinimap(320, 6)).toBe(true);
    expect(shouldShowWorkbenchMinimap(800, 64)).toBe(true);
    expect(WORKBENCH_CANVAS_LAYOUT).toEqual({
      minimapWidth: 120,
      minimapHeight: 80,
      minimapMargin: 8,
      minimumMinimapCanvasHeight: 320,
      minimumMinimapSteps: 6
    });
    expect(getWorkbenchFitViewOptions(false)).toEqual({
      padding: 0.12,
      minZoom: 0.2,
      maxZoom: 1
    });
    expect(getWorkbenchFitViewOptions(true)).toEqual({
      padding: 0.12,
      minZoom: 0.2,
      maxZoom: 1
    });
    withCanvasProbe(probe => {
      const properties = getFlowProps(probe.read());
      expect(properties.fitViewOptions).toEqual(getWorkbenchFitViewOptions(false));
      const controls = findProbeElement(
        probe.read(),
        element => element.type === flowLibrary.Controls
      );
      expect(controls.props['fitViewOptions']).toBe(properties.fitViewOptions);
      expect(() =>
        findProbeElement(probe.read(), element => element.type === flowLibrary.MiniMap)
      ).toThrow('Requested component element is missing.');
    });
  });
  test('40-character step labels, 30-character port labels and long footers keep full titles and clipping styles', () => {
    const stepLabel = 'A'.repeat(40);
    const portLabel = 'P'.repeat(30);
    const document = makeFixtureDocument();
    document.steps[1].label = stepLabel;
    const registry = makeFixtureRegistry();
    const operation = registry.get('fixture-scale')!;
    const longOperation = {
      ...operation,
      inputs: operation.inputs.map(port => ({...port, label: portLabel})),
      outputs: operation.outputs.map(port => ({...port, label: portLabel}))
    };
    const store = makeStore({
      document,
      runner: makeRunner({
        scale: {
          status: 'error',
          key: null,
          issues: [
            {
              severity: 'error',
              code: 'operation-failed',
              message: `${stepLabel}: ${'A long explanation '.repeat(12)}`
            }
          ]
        }
      })
    });
    const styledStore: WorkbenchStore = {
      ...store,
      registry: {...registry, get: id => (id === operation.id ? longOperation : registry.get(id))}
    };
    const stylesheet = new ServerStyleSheet();
    try {
      // Include the real node subtree so styled-components emits its port and footer rules.
      const node = makeWorkbenchNodes(document, [])[1];
      const markup = renderToStaticMarkup(
        stylesheet.collectStyles(
          createElement(
            WorkbenchStoreContext.Provider,
            {value: styledStore},
            createElement(ReactFlowProvider, {
              initialNodes: [{...node, width: 220, height: 164}],
              children: createElement(ReactFlow, {
                width: 800,
                height: 400,
                nodeTypes: {step: WorkbenchNode}
              })
            })
          )
        )
      );
      expect(markup).toContain(`<strong title="${stepLabel}">${stepLabel}</strong>`);
      expect(markup.match(new RegExp(`title="${portLabel}"`, 'g'))).toHaveLength(2);
      expect(markup).toContain(`title="${stepLabel}: A long explanation`);
      const styles = stylesheet.getStyleTags();
      expect(styles).toContain('grid-template-columns:minmax(0, 1fr) minmax(0, 1fr)');
      expect(styles).toMatch(
        /display:block;min-width:0;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap/
      );
      expect(styles).toMatch(
        /min-width:0;max-width:100%;padding:5px 8px;[^}]*white-space:nowrap;overflow:hidden;text-overflow:ellipsis/
      );
    } finally {
      stylesheet.seal();
    }
  });
});

test('L1 measured canvas height controls the compact minimap and all framing paths reserve its space', () => {
  const store = makeLiveStore();
  store.edit('six-steps', document => ({
    ...document,
    steps: [
      ...document.steps,
      ...Array.from({length: 3}, (_, index) => ({
        ...document.steps[0],
        id: `extra-${index}`,
        position: {x: 840 + index * 280, y: 0}
      }))
    ]
  }));
  let height = 319;
  let notifyResize = () => {};
  const element = {getBoundingClientRect: () => ({width: 1000, height})};
  const flow = {fitView: vi.fn(async () => true), getViewport: () => ({x: 0, y: 0, zoom: 1})};
  flowProbe.current = flow;
  vi.stubGlobal('window', {
    matchMedia: () => ({matches: false, addEventListener: () => {}, removeEventListener: () => {}})
  });
  vi.stubGlobal(
    'ResizeObserver',
    class {
      constructor(callback: () => void) {
        notifyResize = callback;
      }
      observe() {}
      disconnect() {}
    }
  );
  const apiReference: {current: WorkbenchCanvasApi | null} = {current: null};
  let initialFitViewOptions: flowLibrary.FitViewOptions | undefined;
  const onApi = (api: WorkbenchCanvasApi | null) => {
    apiReference.current = api;
  };
  const probe = makeHookProbe(
    () => WorkbenchCanvasContent({store, onApi}),
    store,
    undefined,
    tree => {
      initialFitViewOptions ??= getFlowProps(tree).fitViewOptions;
      (tree.props['ref'] as {current: unknown}).current = element;
      (
        findProbeElement(tree, entry => entry.type === ReactFlow).props['ref'] as {current: unknown}
      ).current = element;
    }
  );
  try {
    expect(store.getState().dock.height).toBe(380);
    expect(() =>
      findProbeElement(probe.read(), entry => entry.type === flowLibrary.MiniMap)
    ).toThrow('Requested component element is missing.');
    expect(initialFitViewOptions).toEqual(getWorkbenchFitViewOptions(true));
    height = 320;
    notifyResize();
    const tree = probe.read();
    const minimap = findProbeElement(tree, entry => entry.type === flowLibrary.MiniMap);
    expect(minimap.props['position']).toBe('bottom-right');
    expect(minimap.props['style']).toEqual({width: 120, height: 80, margin: 8, opacity: 0.75});
    expect(getFlowProps(tree).fitViewOptions).toEqual(getWorkbenchFitViewOptions(true));
    expect(
      findProbeElement(tree, entry => entry.type === flowLibrary.Controls).props['fitViewOptions']
    ).toEqual(getWorkbenchFitViewOptions(true));
    apiReference.current!.frame(['scale']);
    expect(flow.fitView).toHaveBeenLastCalledWith({
      ...getWorkbenchFitViewOptions(true),
      nodes: [{id: 'scale'}],
      duration: 200
    });
    store.edit('five-steps', document => ({...document, steps: document.steps.slice(0, 5)}));
    expect(() =>
      findProbeElement(probe.read(), entry => entry.type === flowLibrary.MiniMap)
    ).toThrow('Requested component element is missing.');
  } finally {
    probe.close();
    expect(apiReference.current).toBeNull();
    flowProbe.current = null;
    vi.unstubAllGlobals();
  }
});

describe('pan to reveal an added step', () => {
  test('a rectangle already inside the 24px margin needs no pan, including the boundary', () => {
    const viewport = {x: 0, y: 0, zoom: 1};
    const canvas = {width: 740, height: 300};
    expect(
      getWorkbenchRevealViewport({x: 24, y: 24, width: 220, height: 120}, viewport, canvas)
    ).toBeNull();
    expect(
      getWorkbenchRevealViewport({x: 496, y: 156, width: 220, height: 120}, viewport, canvas)
    ).toBeNull();
    expect(viewport).toEqual({x: 0, y: 0, zoom: 1});
  });
  test('pan is the minimum needed for each edge and preserves zoom', () => {
    const canvas = {width: 740, height: 300};
    expect(
      getWorkbenchRevealViewport(
        {x: 500, y: 250, width: 220, height: 120},
        {x: 0, y: 0, zoom: 1},
        canvas
      )
    ).toEqual({x: -4, y: -94, zoom: 1});
    expect(
      getWorkbenchRevealViewport(
        {x: -40, y: -20, width: 220, height: 120},
        {x: 0, y: 0, zoom: 1},
        canvas
      )
    ).toEqual({x: 64, y: 44, zoom: 1});
    const rectangle = {x: 1200, y: 500, width: 300, height: 180};
    const next = getWorkbenchRevealViewport(rectangle, {x: 100, y: -10, zoom: 0.5}, canvas)!;
    expect(next).toEqual({x: -34, y: -64, zoom: 0.5});
    expect(getWorkbenchRevealViewport(rectangle, next, canvas)).toBeNull();
  });
  test('oversized nodes are centred without zooming, and uninitialised viewports do nothing', () => {
    const rectangle = {x: 0, y: 0, width: 1000, height: 500};
    const canvas = {width: 740, height: 300};
    expect(getWorkbenchRevealViewport(rectangle, {x: 0, y: 0, zoom: 1}, canvas)).toEqual({
      x: -130,
      y: -100,
      zoom: 1
    });
    expect(getWorkbenchRevealViewport(rectangle, {x: -130, y: -100, zoom: 1}, canvas)).toBeNull();
    expect(
      getWorkbenchRevealViewport(rectangle, {x: 0, y: 0, zoom: 1}, {width: 0, height: 300})
    ).toBeNull();
    expect(getWorkbenchRevealViewport(rectangle, {x: 0, y: 0, zoom: 0}, canvas)).toBeNull();
  });
  test('node rectangles use measured dimensions or the specified 220 by 120 fallback', () => {
    const nodes = makeWorkbenchNodes(makeFixtureDocument(), []);
    expect(getWorkbenchNodeRectangle(nodes[1])).toEqual({
      x: 280,
      y: 0,
      width: WORKBENCH_DOCK.nodeWidth,
      height: 120
    });
    expect(getWorkbenchNodeRectangle({...nodes[1], width: 230, height: 130})).toEqual({
      x: 280,
      y: 0,
      width: WORKBENCH_DOCK.nodeWidth,
      height: 120
    });
    expect(
      getWorkbenchNodeRectangle({
        ...nodes[1],
        width: 230,
        height: 130,
        measured: {width: 300, height: 180}
      })
    ).toEqual({x: 280, y: 0, width: 300, height: 180});
    expect(getWorkbenchNodeBounds(nodes)).toEqual({x: 0, y: 0, width: 780, height: 120});
    expect(getWorkbenchNodeBounds([])).toBeUndefined();
  });
});

function withRevealProbe(
  run: (
    probe: {read(): ProbeElement; close(): void},
    store: WorkbenchStore,
    flow: {
      setViewport: ReturnType<typeof vi.fn>;
      fitView: ReturnType<typeof vi.fn>;
      nodesInitialized: boolean;
      getViewport(): {x: number; y: number; zoom: number};
    }
  ) => void,
  reducedMotion = false,
  initialViewport = {x: 0, y: 0, zoom: 1},
  canvasSize = {width: 740, height: 300}
) {
  const store = makeLiveStore();
  let viewport = initialViewport;
  const flow = {
    viewportInitialized: true,
    nodesInitialized: true,
    getViewport: () => viewport,
    setViewport: vi.fn(async (next: typeof viewport) => {
      viewport = next;
      return true;
    }),
    fitView: vi.fn(async () => true)
  };
  flowProbe.current = flow;
  let frameId = 0;
  const frames = new Map<number, FrameRequestCallback>();
  vi.stubGlobal('window', {
    requestAnimationFrame: (callback: FrameRequestCallback) => {
      frames.set(++frameId, callback);
      return frameId;
    },
    cancelAnimationFrame: (identifier: number) => {
      frames.delete(identifier);
    },
    matchMedia: () => ({
      matches: reducedMotion,
      addEventListener: () => {},
      removeEventListener: () => {}
    })
  });
  vi.stubGlobal(
    'ResizeObserver',
    class {
      observe() {}
      disconnect() {}
    }
  );
  const probe = makeHookProbe(
    () => WorkbenchCanvasContent({store}),
    store,
    undefined,
    tree => {
      (tree.props['ref'] as {current: unknown}).current = {
        getBoundingClientRect: () => ({width: 1100, height: 1000})
      };
      (
        findProbeElement(tree, entry => entry.type === ReactFlow).props['ref'] as {current: unknown}
      ).current = {getBoundingClientRect: () => canvasSize};
    }
  );
  try {
    const read = () => {
      const tree = probe.read();
      const scheduled = [...frames.values()];
      frames.clear();
      scheduled.forEach(callback => callback(0));
      return tree;
    };
    read();
    run({read, close: probe.close}, store, flow);
  } finally {
    probe.close();
    flowProbe.current = null;
    vi.unstubAllGlobals();
  }
}
function appendRevealStep(store: WorkbenchStore, position = {x: 500, y: 250}) {
  store.edit('append', document => ({
    ...document,
    steps: [...document.steps, {...document.steps[0], id: 'added', position}]
  }));
}

test('Z2 an addition waits for measurements and reveals once without a later refit', () => {
  withRevealProbe((probe, store, flow) => {
    appendRevealStep(store);
    probe.read();
    expect(flow.setViewport).not.toHaveBeenCalled();
    store.select(['added']);
    probe.read();
    expect(flow.setViewport).not.toHaveBeenCalled();
    getFlowProps(probe.read()).onNodesChange!([
      {type: 'dimensions', id: 'added', dimensions: {width: 220, height: 180}}
    ]);
    probe.read();
    expect(flow.setViewport).toHaveBeenLastCalledWith({x: -4, y: -154, zoom: 1}, {duration: 200});
    expect(flow.setViewport).toHaveBeenCalledTimes(1);
    store.select(['added']);
    store.edit('move', document => moveWorkflowSteps(document, {added: {x: 900, y: 600}}));
    probe.read();
    expect(flow.setViewport).toHaveBeenCalledTimes(1);
    expect(flow.fitView).not.toHaveBeenCalled();
  });
});

test('C1 visible additions, existing selections and multi-step additions do not pan', () => {
  withRevealProbe((probe, store, flow) => {
    store.select(['count']);
    probe.read();
    expect(flow.setViewport).not.toHaveBeenCalled();
    appendRevealStep(store, {x: 30, y: 30});
    store.select(['added']);
    probe.read();
    getFlowProps(probe.read()).onNodesChange!([
      {type: 'dimensions', id: 'added', dimensions: {width: 220, height: 120}}
    ]);
    probe.read();
    expect(flow.setViewport).not.toHaveBeenCalled();
    store.edit('two', document => ({
      ...document,
      steps: [
        ...document.steps,
        ...['second', 'third'].map(id => ({...document.steps[0], id, position: {x: 1200, y: 1000}}))
      ]
    }));
    store.select(['third']);
    probe.read();
    expect(flow.setViewport).not.toHaveBeenCalled();
  });
});

test('C1 waits for single selection and preserves zoom with reduced motion', () => {
  withRevealProbe(
    (probe, store, flow) => {
      appendRevealStep(store, {x: 1200, y: 500});
      store.select(['source', 'added']);
      probe.read();
      expect(flow.setViewport).not.toHaveBeenCalled();
      store.select(['added']);
      probe.read();
      expect(flow.setViewport).not.toHaveBeenCalled();
      getFlowProps(probe.read()).onNodesChange!([
        {type: 'dimensions', id: 'added', dimensions: {width: 220, height: 120}}
      ]);
      probe.read();
      expect(flow.setViewport).toHaveBeenCalledExactlyOnceWith(
        {x: 6, y: -34, zoom: 0.5},
        {duration: 0}
      );
    },
    true,
    {x: 100, y: -10, zoom: 0.5}
  );
});

function getButtonMarkup(markup: string, testId: string): string {
  const position = markup.indexOf(`data-testid="${testId}"`);
  expect(position).toBeGreaterThanOrEqual(0);
  return markup.slice(
    markup.lastIndexOf('<button', position),
    markup.indexOf('</button>', position) + '</button>'.length
  );
}

test('C2 toggles use SVG icons, shortcut titles and distinct pressed colours', () => {
  const document = makeFixtureDocument();
  document.steps[1].disabled = true;
  const stylesheet = new ServerStyleSheet();
  const store = makeStore({document});
  try {
    const node = makeWorkbenchNodes(document, [])[1];
    const markup = renderToStaticMarkup(
      stylesheet.collectStyles(
        createElement(
          WorkbenchStoreContext.Provider,
          {value: store},
          createElement(ReactFlowProvider, {
            initialNodes: [{...node, width: 220, height: 164}],
            children: createElement(ReactFlow, {
              width: 800,
              height: 400,
              nodeTypes: {step: WorkbenchNode}
            })
          })
        )
      )
    );
    const display = getButtonMarkup(markup, WORKBENCH_TEST_IDS.nodeDisplay('scale'));
    const disable = getButtonMarkup(markup, WORKBENCH_TEST_IDS.nodeDisable('scale'));
    expect(display).toContain('title="Show on the map (D)"');
    expect(disable).toContain('title="Disable step (B)"');
    for (const button of [display, disable]) {
      expect(button).toContain('<svg');
      expect(button).toContain('aria-hidden="true"');
      expect(button).toContain('aria-pressed="true"');
      expect(button).not.toMatch(/>\s*[DB]\s*</);
    }
    expect(display).toContain('fill="currentColor"');
    const styles = stylesheet.getStyleTags();
    expect(styles).toContain(`color:${WORKBENCH_COLORS.textMuted}`);
    expect(styles).toMatch(new RegExp(`aria-pressed='true'[^}]*color:${WORKBENCH_COLORS.accent}`));
    expect(styles).toMatch(new RegExp(`aria-pressed='true'[^}]*color:${WORKBENCH_COLORS.warning}`));
  } finally {
    stylesheet.seal();
  }
  document.steps[1].display = [];
  expect(
    getButtonMarkup(renderNode(makeStore({document})), WORKBENCH_TEST_IDS.nodeDisplay('scale'))
  ).toContain('aria-pressed="false"');
});

test('C2 steps without declared outputs do not show an eye button', () => {
  const store = makeStore();
  const registry = store.registry;
  const operation = registry.get('fixture-scale')!;
  const emptyOutputs: WorkbenchStore = {
    ...store,
    registry: {
      ...registry,
      get: id => (id === operation.id ? {...operation, outputs: []} : registry.get(id))
    }
  };
  const markup = renderNode(emptyOutputs);
  expect(markup).not.toContain(`data-testid="${WORKBENCH_TEST_IDS.nodeDisplay('scale')}"`);
  expect(markup).toContain(`data-testid="${WORKBENCH_TEST_IDS.nodeDisable('scale')}"`);
});

test('C3 framing caps padding and preserves 70% scale when the whole graph fits', () => {
  const canvas = {width: 740, height: 300};
  for (const bounds of [
    {x: 0, y: 0, width: 780, height: 300},
    {x: 0, y: 0, width: 1000, height: 400},
    {x: 0, y: 0, width: 800, height: 425}
  ]) {
    const options = getWorkbenchFitViewOptions(false, canvas, bounds);
    expect(options.padding).toBeLessThanOrEqual(0.12);
    const viewport = flowLibrary.getViewportForBounds(
      bounds,
      canvas.width,
      canvas.height,
      options.minZoom!,
      options.maxZoom!,
      options.padding!
    );
    expect(viewport.zoom).toBeGreaterThanOrEqual(0.7 - 1e-12);
    expect(viewport.x).toBeGreaterThanOrEqual(0);
    expect(viewport.y).toBeGreaterThanOrEqual(0);
    expect(viewport.x + bounds.width * viewport.zoom).toBeLessThanOrEqual(canvas.width);
    expect(viewport.y + bounds.height * viewport.zoom).toBeLessThanOrEqual(canvas.height);
  }
});

test('C3 wider graphs still frame every column instead of enforcing a cropping zoom floor', () => {
  const canvas = {width: 740, height: 300};
  for (const width of [1620, 5000, 200220]) {
    const bounds = {x: -100000, y: 0, width, height: 180};
    const options = getWorkbenchFitViewOptions(true, canvas, bounds);
    expect(options.padding).toBe(0.12);
    const viewport = flowLibrary.getViewportForBounds(
      bounds,
      canvas.width,
      canvas.height,
      options.minZoom!,
      options.maxZoom!,
      options.padding!
    );
    const left = viewport.x + bounds.x * viewport.zoom;
    expect(left).toBeGreaterThanOrEqual(-1e-9);
    expect(left + width * viewport.zoom).toBeLessThanOrEqual(canvas.width + 1e-9);
    if (width >= 5000) expect(options.minZoom).toBeLessThan(0.2);
  }
});

describe('Z1 framing zoom', () => {
  test('one- and two-step graphs never fit above 100%, while manual zoom remains available', () => {
    for (const width of [220, 500]) {
      const bounds = {x: 0, y: 0, width, height: 120};
      const options = getWorkbenchFitViewOptions(false, {width: 740, height: 290}, bounds);
      expect(options.maxZoom).toBe(1);
      expect(
        flowLibrary.getViewportForBounds(
          bounds,
          740,
          290,
          options.minZoom!,
          options.maxZoom!,
          options.padding!
        ).zoom
      ).toBe(1);
    }
    withCanvasProbe((probe, store) => {
      store.edit('empty', document => ({...document, steps: []}));
      const empty = getFlowProps(probe.read());
      expect(empty.defaultViewport).toEqual({x: 0, y: 0, zoom: 1});
      expect(empty.maxZoom).toBe(2);
      expect(empty.fitViewOptions!.maxZoom).toBe(1);
      expect(() =>
        findProbeElement(probe.read(), entry => entry.type === flowLibrary.Controls)
      ).toThrow('Requested component element is missing.');
      store.edit('one-step', document => ({...document, steps: [makeFixtureDocument().steps[0]]}));
      const populated = getFlowProps(probe.read());
      expect(populated.nodes).toHaveLength(1);
      expect(populated.defaultViewport).toBe(empty.defaultViewport);
      expect(populated.fitView).toBe(true);
      expect(populated.fitViewOptions!.maxZoom).toBe(1);
      expect(
        findProbeElement(probe.read(), entry => entry.type === flowLibrary.Controls).props[
          'fitViewOptions'
        ]
      ).toBe(populated.fitViewOptions);
    });
  });
});

describe('Z2 reveal geometry and ordering', () => {
  test.each([
    {
      zoom: 1,
      rectangle: {x: 500, y: 24, width: 220, height: 120},
      expected: {x: -4, y: 0, zoom: 1}
    },
    {
      zoom: 1,
      rectangle: {x: 24, y: 200, width: 220, height: 120},
      expected: {x: 0, y: -54, zoom: 1}
    },
    {
      zoom: 1.7,
      rectangle: {x: 500, y: 24, width: 220, height: 120},
      expected: {x: -508, y: 0, zoom: 1.7}
    },
    {
      zoom: 1.7,
      rectangle: {x: 24, y: 200, width: 220, height: 120},
      expected: {x: 0, y: -278, zoom: 1.7}
    },
    {
      zoom: 1.7,
      rectangle: {x: 500, y: 200, width: 220, height: 120},
      expected: {x: -508, y: -278, zoom: 1.7}
    }
  ])('reveals off-screen rectangles at zoom $zoom: $rectangle', ({zoom, rectangle, expected}) => {
    const result = getWorkbenchRevealViewport(
      rectangle,
      {x: 0, y: 0, zoom},
      {width: 740, height: 290}
    );
    expect(result).toEqual(expected);
    expect(getWorkbenchRevealViewport(rectangle, result!, {width: 740, height: 290})).toBeNull();
  });
  test('waits for nodesInitialized and uses the current zoom at the measured reveal', () => {
    withRevealProbe(
      (probe, store, flow) => {
        flow.nodesInitialized = false;
        appendRevealStep(store, {x: 500, y: 200});
        store.select(['added']);
        probe.read();
        expect(flow.setViewport).not.toHaveBeenCalled();
        flow.getViewport().zoom = 1.7;
        getFlowProps(probe.read()).onNodesChange!([
          {type: 'dimensions', id: 'added', dimensions: {width: 220, height: 120}}
        ]);
        probe.read();
        expect(flow.setViewport).not.toHaveBeenCalled();
        flow.nodesInitialized = true;
        probe.read();
        expect(flow.setViewport).toHaveBeenCalledExactlyOnceWith(
          {x: -508, y: -278, zoom: 1.7},
          {duration: 200}
        );
        const properties = getFlowProps(probe.read());
        const defaultViewport = properties.defaultViewport;
        store.edit('move-existing', document =>
          moveWorkflowSteps(document, {added: {x: 900, y: 600}})
        );
        getFlowProps(probe.read()).onNodesChange!([
          {type: 'dimensions', id: 'added', dimensions: {width: 230, height: 140}}
        ]);
        const after = getFlowProps(probe.read());
        expect(after.fitView).toBe(properties.fitView);
        expect(after.defaultViewport).toBe(defaultViewport);
        expect(flow.setViewport).toHaveBeenCalledTimes(1);
        expect(flow.fitView).not.toHaveBeenCalled();
      },
      false,
      {x: 0, y: 0, zoom: 1},
      {width: 740, height: 290}
    );
  });
  test('reads the React Flow element again after layout, independently of the wider dock', () => {
    const canvasSize = {width: 740, height: 290};
    withRevealProbe(
      (probe, store, flow) => {
        appendRevealStep(store);
        store.select(['added']);
        probe.read();
        expect(flow.setViewport).not.toHaveBeenCalled();
        canvasSize.width = 500;
        getFlowProps(probe.read()).onNodesChange!([
          {type: 'dimensions', id: 'added', dimensions: {width: 220, height: 120}}
        ]);
        probe.read();
        expect(flow.setViewport).toHaveBeenCalledExactlyOnceWith(
          {x: -244, y: -104, zoom: 1},
          {duration: 200}
        );
      },
      false,
      {x: 0, y: 0, zoom: 1},
      canvasSize
    );
  });
  test('the unmeasured fallback is delayed and is used only as a last resort', () => {
    vi.useFakeTimers();
    try {
      withRevealProbe((probe, store, flow) => {
        appendRevealStep(store);
        store.select(['added']);
        probe.read();
        vi.advanceTimersByTime(499);
        probe.read();
        expect(flow.setViewport).not.toHaveBeenCalled();
        vi.advanceTimersByTime(1);
        probe.read();
        expect(flow.setViewport).toHaveBeenCalledExactlyOnceWith(
          {x: -4, y: -94, zoom: 1},
          {duration: 200}
        );
        getFlowProps(probe.read()).onNodesChange!([
          {type: 'dimensions', id: 'added', dimensions: {width: 220, height: 180}}
        ]);
        probe.read();
        expect(flow.setViewport).toHaveBeenLastCalledWith(
          {x: -4, y: -154, zoom: 1},
          {duration: 200}
        );
      });
    } finally {
      vi.useRealTimers();
    }
  });
});

describe('composition on the canvas', () => {
  test('F6 every displayed coloured output has its own palette indicator', () => {
    const operation = getWorkflowOperationRegistry().get('k-means')!;
    expect(operation.outputs.map(port => port.name)).toEqual([
      'kmeans-assignments',
      'kmeans-centroids'
    ]);
    const document = makeFixtureDocument();
    document.steps[1] = {
      ...document.steps[1],
      operationId: operation.id,
      displayStyle: {
        'kmeans-assignments': {colorColumn: 'point_index'},
        'kmeans-centroids': {colorColumn: 'cluster'}
      }
    };
    function readIndicators(display: string[]): string[] {
      document.steps[1] = {...document.steps[1], display};
      const store = makeStore({document});
      const registry = store.registry;
      const button = getButtonMarkup(
        renderNode({
          ...store,
          registry: {
            ...registry,
            get: id => (id === operation.id ? operation : registry.get(id))
          }
        }),
        WORKBENCH_TEST_IDS.nodeDisplay('scale')
      );
      return [...button.matchAll(/title="Coloured by ([^"]*)"/g)].map(match => match[1]);
    }
    expect(readIndicators(['kmeans-centroids'])).toEqual(['cluster']);
    expect(readIndicators(['kmeans-centroids', 'kmeans-assignments'])).toEqual([
      'point_index',
      'cluster'
    ]);
    document.steps[1] = {
      ...document.steps[1],
      displayStyle: {'kmeans-centroids': {colorColumn: 'cluster'}}
    };
    expect(readIndicators(['kmeans-assignments'])).toEqual([]);
    expect(readIndicators([])).toEqual([]);
  });

  test('a displayed colour column adds a labelled dot beside the eye', () => {
    const document = setWorkflowDisplayStyle(makeFixtureDocument(), 'scale', 'scaled', {
      colorColumn: 'Population'
    });
    const button = getButtonMarkup(
      renderNode(makeStore({document})),
      WORKBENCH_TEST_IDS.nodeDisplay('scale')
    );
    expect(button).toContain('aria-pressed="true"');
    expect(button).toContain('title="Coloured by Population"');
    expect(button).toContain('role="img" aria-label="Coloured by Population"');
    expect(button.indexOf('Coloured by Population')).toBeGreaterThan(button.indexOf('</svg>'));
    document.steps[1] = {...document.steps[1], display: []};
    expect(renderNode(makeStore({document}))).not.toContain('Coloured by');
    expect(renderNode(makeStore())).not.toContain('Coloured by');
  });

  test('host capability failures render the error message in the footer', () => {
    const message = 'Fixture source: this host cannot read drawings. Choose another source.';
    const error = {
      severity: 'error' as const,
      code: 'host-capability-missing' as const,
      stepId: 'source',
      message
    };
    const stepState: WorkflowStepState = {status: 'error', key: null, issues: [error], error};
    const store = makeStore({runner: makeRunner({source: stepState})});
    const step = store.getState().document!.steps[0];
    const operation = store.registry.get(step.operationId)!;
    expect(operation.engine).toBe('host');
    expect(getWorkbenchStepSummary(step, operation, stepState)).toBe(message);
    const markup = renderNode(store, 'source');
    expect(getMarkupTag(markup, WORKBENCH_TEST_IDS.node('source'))).toContain(
      'data-status="error"'
    );
    expect(markup).toContain(`title="${message}" aria-live="polite"`);
    expect(markup).toContain(`>${message}</div>`);
  });

  test('group framing fits both axes, keeps a margin and never enlarges a block', () => {
    const rectangle = {x: 1000, y: 500, width: 1000, height: 400};
    const canvasSize = {width: 548, height: 248};
    const fitted = getWorkbenchGroupRevealViewport(rectangle, {x: 0, y: 0, zoom: 1.7}, canvasSize);
    expect(fitted).toEqual({x: -476, y: -226, zoom: 0.5});
    expect(getWorkbenchGroupRevealViewport(rectangle, fitted!, canvasSize)).toBeNull();
    expect(
      getWorkbenchGroupRevealViewport(
        {x: 30, y: 30, width: 500, height: 200},
        {x: 9, y: 9, zoom: 0.5},
        canvasSize
      )
    ).toBeNull();
    expect(
      getWorkbenchGroupRevealViewport(
        {x: 30, y: 30, width: 300, height: 100},
        {x: 0, y: 0, zoom: 1.7},
        canvasSize
      )
    ).toEqual({x: 0, y: 0, zoom: 1});
  });

  test.each([
    'paste',
    'insert'
  ] as const)('%s reveals the selected block after every added node has measurements', action => {
    withRevealProbe(
      (probe, store, flow) => {
        store.edit('setup', document => moveWorkflowSteps(document, {scale: {x: 780, y: 0}}));
        probe.read();
        if (action === 'paste') {
          store.select(['source', 'scale']);
          expect(store.copySelection()).toBe(true);
          expect(store.paste({x: 1000, y: 500})).toBe(true);
        } else {
          const source = makeFixtureDocument();
          source.steps = source.steps.slice(0, 2);
          source.steps[1] = {...source.steps[1], position: {x: 780, y: 0}};
          expect(store.insertWorkflow(source, {x: 1000, y: 500})).toBe(true);
        }
        const selected = store.getState().selectedStepIds;
        expect(selected).toHaveLength(2);
        probe.read();
        expect(flow.setViewport).not.toHaveBeenCalled();
        const measure = (id: string) => {
          getFlowProps(probe.read()).onNodesChange!([
            {type: 'dimensions', id, dimensions: {width: 220, height: 160}}
          ]);
          probe.read();
        };
        measure(selected[0]);
        expect(flow.setViewport).not.toHaveBeenCalled();
        measure(selected[1]);
        expect(flow.setViewport).toHaveBeenCalledExactlyOnceWith(
          {x: -476, y: -106, zoom: 0.5},
          {duration: 0}
        );
        store.select([...selected].reverse());
        probe.read();
        expect(flow.setViewport).toHaveBeenCalledTimes(1);
        expect(flow.fitView).not.toHaveBeenCalled();
      },
      true,
      {x: 0, y: 0, zoom: 1},
      {width: 548, height: 248}
    );
  });

  test('selecting steps added in different changes does not reveal a block', () => {
    withRevealProbe((probe, store, flow) => {
      appendRevealStep(store);
      probe.read();
      store.edit('append-again', document => ({
        ...document,
        steps: [...document.steps, {...document.steps[0], id: 'later', position: {x: 1000, y: 500}}]
      }));
      store.select(['added', 'later']);
      probe.read();
      getFlowProps(probe.read()).onNodesChange!(
        ['added', 'later'].map(id => ({
          type: 'dimensions',
          id,
          dimensions: {width: 220, height: 160}
        }))
      );
      probe.read();
      expect(flow.setViewport).not.toHaveBeenCalled();
    });
  });
});

describe('node text contrast', () => {
  function getContrastRatio(foreground: string, background: string): number {
    function getLuminance(color: string): number {
      const channels = [1, 3, 5].map(offset => {
        const channel = Number.parseInt(color.slice(offset, offset + 2), 16) / 255;
        return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
      });
      return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
    }
    const foregroundLuminance = getLuminance(foreground);
    const backgroundLuminance = getLuminance(background);
    return (
      (Math.max(foregroundLuminance, backgroundLuminance) + 0.05) /
      (Math.min(foregroundLuminance, backgroundLuminance) + 0.05)
    );
  }

  function readDeclarations(markupTag: string, styles: string): string {
    const classes = new Set(markupTag.match(/class="([^"]*)"/)?.[1].split(' '));
    return [...styles.matchAll(/\.([\w-]+)\{([^}]*)\}/g)]
      .filter(match => classes.has(match[1]))
      .map(match => match[2])
      .join('');
  }

  test('every normal and disabled text colour meets WCAG AA on the node surface', () => {
    expect(getContrastRatio('#ffffff', '#000000')).toBe(21);
    expect(getContrastRatio(WORKBENCH_COLORS.surface, WORKBENCH_COLORS.surface)).toBe(1);
    const textColors = {
      text: WORKBENCH_COLORS.text,
      textMuted: WORKBENCH_COLORS.textMuted,
      ...Object.fromEntries(
        Object.entries(WORKBENCH_STATUS).map(([status, value]) => [status, value.color])
      )
    };
    for (const [name, color] of Object.entries(textColors)) {
      expect(getContrastRatio(color, WORKBENCH_COLORS.surface), name).toBeGreaterThanOrEqual(4.5);
    }
  });

  test.each([
    false,
    true
  ])('disabled=%s uses full-contrast text and its border treatment', disabled => {
    const document = makeFixtureDocument();
    document.steps[1] = {...document.steps[1], disabled};
    const stylesheet = new ServerStyleSheet();
    try {
      const markup = renderNode(makeStore({document}), 'scale', false, stylesheet);
      const styles = stylesheet.getStyleTags();
      const root = readDeclarations(getMarkupTag(markup, WORKBENCH_TEST_IDS.node('scale')), styles);
      expect(root).not.toMatch(/(?:^|;)opacity:/);
      expect(root).toContain(
        `border:1px ${disabled ? 'dashed' : 'solid'} ${disabled ? WORKBENCH_COLORS.border : WORKBENCH_COLORS.borderStrong};`
      );
      expect(root).toContain(`color:${WORKBENCH_COLORS.text};`);
      expect(styles).toMatch(
        new RegExp(
          ` strong\\{[^}]*color:${disabled ? WORKBENCH_COLORS.textMuted : WORKBENCH_COLORS.text};`
        )
      );
      const badge = readDeclarations(
        getMarkupTag(markup, WORKBENCH_TEST_IDS.nodeStatus('scale')),
        styles
      );
      expect(badge).toContain(`color:${WORKBENCH_STATUS[disabled ? 'disabled' : 'ready'].color};`);
      expect(badge).not.toMatch(/(?:^|;)opacity:/);
      expect(getMarkupTag(markup, WORKBENCH_TEST_IDS.nodeDisable('scale'))).toContain(
        `aria-pressed="${disabled}"`
      );
      const footer = markup.match(/<div[^>]*aria-live="polite"[^>]*>/)![0];
      expect(readDeclarations(footer, styles)).toContain(`color:${WORKBENCH_COLORS.textMuted};`);
      const textColors = [...styles.matchAll(/[;{]color:(#[0-9a-f]{6})[;}]/gi)].map(
        match => match[1]
      );
      expect(textColors).toContain(WORKBENCH_COLORS.text);
      expect(textColors).toContain(WORKBENCH_COLORS.textMuted);
      for (const color of textColors) {
        expect(getContrastRatio(color, WORKBENCH_COLORS.surface), color).toBeGreaterThanOrEqual(
          4.5
        );
      }
    } finally {
      stylesheet.seal();
    }
  });
});
