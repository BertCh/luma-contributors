// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {afterEach, describe, expect, test, vi} from 'vitest';
import type {ReactElement, ReactNode} from 'react';
import {
  describeWorkflowOperationForAssistant,
  getWorkflowAssistantTools,
  listWorkflowOperationsForAssistant,
  listWorkflowsForAssistant,
  describeWorkflowForAssistant,
  validateWorkflowForAssistant
} from '../../../src/workflows/workflow-assistant';
import {
  exportWorkflowJSON,
  exportWorkflowTypeScript,
  importWorkflowJSON,
  WORKFLOW_JSON_LIMITS,
  describeWorkflow
} from '../../../src/workflows/workflow-export';
import {
  getCatalogBinding,
  serializeCatalogInvocation
} from '../../../src/analysis/catalog-bindings';
import {getWorkflowOperationRegistry} from '../../../src/workflows/workflow-operations';
import {WorkflowRunner} from '../../../src/workflows/workflow-runner';
import {makeWorkflowRunReport, makeWorkflowDot} from '../../../src/workflows/workflow-report';
import {planWorkflow} from '../../../src/workflows/workflow-plan';
import {getCatalogSampleInputs} from '../../../../../examples/deck/kepler-compute/src/analysis-catalog-input';
import {listWorkflowTemplates} from '../../../src/workflows/workflow-templates';
import {
  WorkflowError,
  type WorkflowIssue,
  type WorkflowDocument,
  type WorkflowRunRecord,
  type WorkflowRunnerState
} from '../../../src/workflows/workflow-types';
import {makeFixtureDocument, makeFixtureHost, makeFixtureRegistry} from '../workflow-fixtures';
import {
  getWorkbenchAssistantTools,
  runWorkbenchAssistantTool
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-assistant';
import {WorkbenchRunLog} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-run-log';
import {
  formatWorkbenchShortcut,
  sortWorkbenchDraftIssues,
  WorkbenchDialogs,
  WorkbenchTransfer
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-dialogs';
import {
  WORKBENCH_PORT_LABELS,
  WORKBENCH_SHORTCUTS,
  WORKBENCH_STATUS,
  WORKBENCH_TEST_IDS
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-contract';
import {createWorkbenchStore} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-store';
import type {
  WorkbenchDraft,
  WorkbenchState,
  WorkbenchStore
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-store';
import {
  clearCatalogInvocations,
  restoreCatalogInvocations
} from '../../../../../examples/deck/kepler-compute/src/catalog-invocation-store';

const hostRequire = createRequire(
  new URL('../../../../../examples/deck/kepler-compute/package.json', import.meta.url)
);
const {createElement, isValidElement} = hostRequire('react') as typeof import('react');
const {renderToStaticMarkup} = hostRequire('react-dom/server') as typeof import('react-dom/server');

function makeStore(patch: Partial<WorkbenchState> = {}) {
  const document = makeFixtureDocument();
  const drafts: WorkbenchDraft[] = [];
  const addedDocuments: WorkflowDocument[] = [];
  const calls: string[] = [];
  const state: WorkbenchState = {
    revision: 0,
    workflows: [document],
    activeWorkflowId: document.id,
    document,
    runner: null,
    selectedStepIds: [],
    canUndo: false,
    canRedo: false,
    dock: {open: false, height: 380, inspectorTab: 'parameters', logOpen: false},
    palette: {open: false},
    dialog: null,
    draft: null,
    notice: null,
    runLog: [],
    templates: [],
    clipboardStepCount: 0,
    ...patch
  };
  const store: WorkbenchStore = {
    registry: makeFixtureRegistry(),
    host: makeFixtureHost(),
    getState: () => state,
    subscribe: () => () => {},
    createWorkflow: () => {
      calls.push('create');
      return 'created';
    },
    addWorkflow: document => {
      addedDocuments.push(document);
      calls.push('add');
      return 'added';
    },
    duplicateWorkflow: () => '',
    deleteWorkflow: () => {},
    setActiveWorkflow: () => {},
    edit: () => {
      calls.push('edit');
      return true;
    },
    undo: () => {},
    redo: () => {},
    select: () => {},
    run: async () => {
      calls.push('run');
      return null;
    },
    cancel: () => {},
    notifyExternalChange: () => {},
    getRunner: () => null,
    setDock: () => {},
    openPalette: () => {},
    closePalette: () => {},
    setDialog: () => {
      calls.push('dialog');
    },
    setNotice: () => {},
    proposeDraft: draft => {
      calls.push('propose');
      drafts.push(draft);
    },
    acceptDraft: () => {
      calls.push('accept');
      return 'accepted';
    },
    rejectDraft: () => {
      calls.push('reject');
    },
    getReviewedAssistantIssues: () => [],
    recordAssistantReview: () => {},
    saveTemplate: () => '',
    deleteTemplate: () => {},
    useTemplate: () => '',
    insertWorkflow: () => false,
    copySelection: () => false,
    cutSelection: () => false,
    paste: () => false,
    serialize: () => undefined,
    restore: () => {},
    clear: async () => {}
  };
  return {store, drafts, calls, addedDocuments};
}

function render(store: WorkbenchStore): string {
  return renderToStaticMarkup(createElement(WorkbenchDialogs, {store}));
}

function escapeMarkup(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;');
}

const WARNING: WorkflowIssue = {
  severity: 'warning',
  code: 'unverified-port',
  message: 'Scale: Points will be checked when run.'
};
const ERROR: WorkflowIssue = {
  severity: 'error',
  code: 'input-required',
  message: 'Scale needs an input for "Points".'
};
const INFORMATION: WorkflowIssue = {
  severity: 'info',
  code: 'capacity-may-exceed',
  message: 'Count: Rows may exceed 10.'
};

describe('workbench assistant routing', () => {
  test('advertises the library tool definitions', () => {
    expect(getWorkbenchAssistantTools()).toEqual(getWorkflowAssistantTools());
    expect(getWorkbenchAssistantTools().map(tool => tool.name)).toEqual([
      'listAnalysisOperations',
      'describeAnalysisOperation',
      'defineWorkflow',
      'listWorkflows',
      'describeWorkflow',
      'setWorkflowInputs',
      'runWorkflow',
      'listAnalysisTemplates',
      'describeAnalysisTemplate'
    ]);
  });

  test('returns operation lists and descriptions without changing the store', () => {
    const {store, calls} = makeStore();
    for (const [name, argumentsValue, expected] of [
      [
        'listAnalysisOperations',
        {query: 'scale', limit: 1},
        listWorkflowOperationsForAssistant(store.registry, {query: 'scale', limit: 1})
      ],
      [
        'describeAnalysisOperation',
        {operationId: 'fixture-scale'},
        describeWorkflowOperationForAssistant(store.registry, {operationId: 'fixture-scale'})
      ]
    ] as const) {
      expect(runWorkbenchAssistantTool(store, {name, arguments: argumentsValue})).toEqual(expected);
    }
    expect(calls).toEqual([]);
  });

  test('proposes one valid draft, avoids existing ids and never adds or runs it', () => {
    const {store, drafts, calls} = makeStore();
    const result = runWorkbenchAssistantTool(store, {
      name: 'defineWorkflow',
      arguments: {
        title: 'Fixture workflow',
        steps: [{id: 'source', operationId: 'fixture-source'}]
      }
    });
    expect(result).toEqual({
      status: 'awaiting-approval',
      workflowTitle: 'Fixture workflow',
      stepCount: 1,
      issues: [],
      note: 'The workflow is shown to the user as a draft. It has not been added or run.'
    });
    expect(calls).toEqual(['propose']);
    expect(drafts).toHaveLength(1);
    expect(drafts[0].source).toBe('assistant');
    expect(drafts[0].document.id).toBe('fixture-workflow-2');
    expect(drafts[0].document.steps[0].operationId).toBe('fixture-source');
    expect(drafts[0].outline).toContain('Fixture source');
    expect(store.getState().workflows).toHaveLength(1);
  });

  test('rejects configuration errors before offering a draft for approval', () => {
    const {store, drafts, calls} = makeStore();
    const result = runWorkbenchAssistantTool(store, {
      name: 'defineWorkflow',
      arguments: {title: 'Needs points', steps: [{id: 'scale', operationId: 'fixture-scale'}]}
    });
    expect(result).toEqual({
      status: 'error',
      code: 'malformed',
      message: expect.stringContaining('Points')
    });
    expect(drafts).toEqual([]);
    expect(calls).toEqual([]);
  });

  test.each([
    null,
    {},
    {title: 'No steps', steps: []},
    {title: 'Bad steps', steps: 'source'}
  ])('returns malformed errors for %j without proposing', argumentsValue => {
    const {store, drafts, calls} = makeStore();
    expect(
      runWorkbenchAssistantTool(store, {name: 'defineWorkflow', arguments: argumentsValue})
    ).toEqual({status: 'error', code: 'malformed', message: expect.any(String)});
    expect(drafts).toEqual([]);
    expect(calls).toEqual([]);
  });

  test.each([0, 51])('refuses list limits outside 1 to 50: %s', limit => {
    const {store, calls} = makeStore();
    expect(
      runWorkbenchAssistantTool(store, {name: 'listAnalysisOperations', arguments: {limit}})
    ).toEqual({
      status: 'error',
      code: 'malformed',
      message: 'arguments.limit: must be an integer between 1 and 50.'
    });
    expect(calls).toEqual([]);
  });

  test('refuses more than 64 draft steps before proposing', () => {
    const {store, calls} = makeStore();
    const result = runWorkbenchAssistantTool(store, {
      name: 'defineWorkflow',
      arguments: {
        title: 'Too many steps',
        steps: Array.from({length: 65}, (_, index) => ({
          id: `source-${index}`,
          operationId: 'fixture-source'
        }))
      }
    });
    expect(result).toEqual({
      status: 'error',
      code: 'malformed',
      message: 'arguments.steps: maximumSteps is 64; got 65.'
    });
    expect(calls).toEqual([]);
  });

  test('reports unknown tools and unknown operations with exact codes', () => {
    const {store, calls} = makeStore();
    expect(runWorkbenchAssistantTool(store, {name: 'executeWorkflow', arguments: {}})).toEqual({
      status: 'error',
      code: 'malformed',
      message: 'Assistant tool "executeWorkflow" is not available.'
    });
    expect(
      runWorkbenchAssistantTool(store, {
        name: 'describeAnalysisOperation',
        arguments: {operationId: 'missing'}
      })
    ).toEqual({
      status: 'error',
      code: 'unknown-operation',
      message: 'Operation "missing" is not available.'
    });
    expect(calls).toEqual([]);
  });

  test('returns thrown workflow and unexpected failures to the caller', () => {
    const {store} = makeStore();
    store.proposeDraft = () => {
      throw new WorkflowError('limit-exceeded', 'Workflows: maximumWorkflows is 16.');
    };
    const call = {
      name: 'defineWorkflow',
      arguments: {title: 'A draft', steps: [{id: 'source', operationId: 'fixture-source'}]}
    };
    expect(runWorkbenchAssistantTool(store, call)).toEqual({
      status: 'error',
      code: 'limit-exceeded',
      message: 'Workflows: maximumWorkflows is 16.'
    });
    store.proposeDraft = () => {
      throw new Error('Draft storage failed.');
    };
    expect(runWorkbenchAssistantTool(store, call)).toEqual({
      status: 'error',
      code: 'operation-failed',
      message: 'Draft storage failed.'
    });
    store.proposeDraft = () => {
      throw null;
    };
    expect(runWorkbenchAssistantTool(store, call)).toEqual({
      status: 'error',
      code: 'operation-failed',
      message: 'The assistant tool could not complete.'
    });
  });
});

describe('workbench dialog helpers', () => {
  test('formats platform modifiers and preserves key order', () => {
    expect(formatWorkbenchShortcut('Shift+Mod+Z', 'MacIntel')).toBe('Shift+⌘+Z');
    expect(formatWorkbenchShortcut('Mod+D', 'iPad')).toBe('⌘+D');
    expect(formatWorkbenchShortcut('Mod+D', 'Win32')).toBe('Ctrl+D');
    expect(formatWorkbenchShortcut('Mod+A', 'Linux x86_64')).toBe('Ctrl+A');
    expect(formatWorkbenchShortcut('Shift+Enter', '')).toBe('Shift+Enter');
  });
  test('sorts errors before warnings and information without changing input or tie order', () => {
    const secondError = {...ERROR, message: 'Second error.'};
    const issues = [INFORMATION, WARNING, ERROR, secondError];
    expect(sortWorkbenchDraftIssues(issues)).toEqual([ERROR, secondError, WARNING, INFORMATION]);
    expect(issues).toEqual([INFORMATION, WARNING, ERROR, secondError]);
    expect(sortWorkbenchDraftIssues([])).toEqual([]);
  });
});

describe('workbench dialogs markup', () => {
  test('renders nothing when closed without a draft', () => {
    expect(render(makeStore().store)).toBe('');
  });
  test('renders every real template with a collapsed outline and an accessible modal', () => {
    const {store} = makeStore({dialog: 'templates'});
    const markup = render(store);
    const templates = listWorkflowTemplates();
    expect(markup).toContain('role="dialog" aria-modal="true" aria-labelledby=');
    expect(markup).toContain('Workflow templates');
    expect(markup).toContain('>Close</button>');
    expect(markup.match(/>Use template<\/button>/g)).toHaveLength(templates.length);
    expect(markup.match(/<details>/g)).toHaveLength(templates.length);
    expect(markup).not.toContain('<details open');
    for (const template of templates) {
      expect(markup).toContain(`data-testid="${WORKBENCH_TEST_IDS.template(template.id)}"`);
      expect(markup).toContain(template.title);
      expect(markup).toContain(`${template.document.steps.length} steps`);
    }
  });
  test('exports exactly the canonical JSON in a labelled readonly textarea', () => {
    const {store} = makeStore({dialog: 'import-export'});
    const markup = render(store);
    expect(markup).toContain(`data-testid="${WORKBENCH_TEST_IDS.exportJSON}"`);
    expect(markup).toContain(`data-testid="${WORKBENCH_TEST_IDS.exportText}" readOnly=""`);
    expect(markup).toContain(
      `>${escapeMarkup(exportWorkflowJSON(store.getState().document!))}</textarea>`
    );
    expect(markup.match(/role="tab"/g)).toHaveLength(6);
    expect(markup).toContain('role="tabpanel"');
    expect(markup).toContain('>Copy</button>');
    expect(markup).toContain('>Download</button>');
  });
  test('renders the exact TypeScript export', () => {
    const {store} = makeStore();
    const markup = renderToStaticMarkup(
      createElement(WorkbenchTransfer, {store, initialTab: 'typescript'})
    );
    expect(markup).toContain(
      `>${escapeMarkup(exportWorkflowTypeScript(store.getState().document!, store.registry))}</textarea>`
    );
    expect(markup).toContain('Workflow TypeScript');
  });
  test('explains disabled exports with no active workflow', () => {
    const markup = render(
      makeStore({dialog: 'import-export', document: null, workflows: [], activeWorkflowId: null})
        .store
    );
    expect(markup).toContain('Select or create a workflow to export it.');
    expect(markup).toMatch(/disabled=""[^>]*>Copy<\/button>/);
    expect(markup).toMatch(/disabled=""[^>]*>Download<\/button>/);
  });
  test('renders an import text field, file chooser and initially disabled import action', () => {
    const markup = renderToStaticMarkup(
      createElement(WorkbenchTransfer, {store: makeStore().store, initialTab: 'import'})
    );
    expect(markup).toContain(`data-testid="${WORKBENCH_TEST_IDS.importText}"`);
    expect(markup).toContain('aria-label="Choose workflow JSON file"');
    expect(markup).toContain('>Choose file</button>');
    expect(markup).toMatch(
      new RegExp(`data-testid="${WORKBENCH_TEST_IDS.importApply}"[^>]*disabled=""`)
    );
  });
  test('explains empty history and disables draft creation', () => {
    clearCatalogInvocations();
    const markup = renderToStaticMarkup(
      createElement(WorkbenchTransfer, {store: makeStore().store, initialTab: 'history'})
    );
    expect(markup).toContain(`data-testid="${WORKBENCH_TEST_IDS.promoteHistory}"`);
    expect(markup).toContain('No saved catalog runs.');
    expect(markup).toMatch(/disabled=""[^>]*>Create draft<\/button>/);
  });
  test('lists every saved history record with a checked checkbox', () => {
    restoreCatalogInvocations([
      {
        id: 'fixture-scale#1',
        operationId: 'fixture-scale',
        inputs: {},
        invocation: '{}',
        engine: 'cpu'
      },
      {
        id: 'fixture-count#1',
        operationId: 'fixture-count',
        inputs: {},
        invocation: '{}',
        engine: 'auto'
      }
    ]);
    try {
      const markup = renderToStaticMarkup(
        createElement(WorkbenchTransfer, {store: makeStore().store, initialTab: 'history'})
      );
      expect(markup).toContain('fixture-scale#1 · fixture-scale');
      expect(markup).toContain('fixture-count#1 · fixture-count');
      expect(markup.match(/type="checkbox"[^>]*checked=""/g)).toHaveLength(2);
      expect(markup).not.toContain('No saved catalog runs.');
    } finally {
      clearCatalogInvocations();
    }
  });
  test('renders one shortcut row per contract entry and names every port and status', () => {
    const markup = render(makeStore({dialog: 'shortcuts'}).store);
    expect(markup).toContain('aria-label="Keyboard shortcuts"');
    expect(markup.match(/<tr>/g)).toHaveLength(WORKBENCH_SHORTCUTS.length + 1);
    for (const shortcut of WORKBENCH_SHORTCUTS) expect(markup).toContain(shortcut.label);
    for (const label of Object.values(WORKBENCH_PORT_LABELS)) expect(markup).toContain(label);
    for (const [status, value] of Object.entries(WORKBENCH_STATUS)) {
      expect(markup).toContain(`data-status="${status}"`);
      expect(markup).toContain(value.label);
    }
  });
  test.each([
    'assistant',
    'history',
    'import'
  ] as const)('renders a non-modal %s draft independently of dialog state', source => {
    const {store, calls} = makeStore({
      draft: {
        document: makeFixtureDocument(),
        issues: [],
        outline: '1. Fixture source\n2. Fixture scale',
        source
      }
    });
    const markup = render(store);
    expect(markup).toContain(`data-testid="${WORKBENCH_TEST_IDS.draft}"`);
    expect(markup).toContain('role="region" aria-label="Proposed workflow"');
    expect(markup).toContain(
      {assistant: 'Proposed by the assistant', history: 'From catalog history', import: 'Imported'}[
        source
      ]
    );
    expect(markup).toContain('1. Fixture source\n2. Fixture scale');
    expect(markup).toContain(`data-testid="${WORKBENCH_TEST_IDS.draftAccept}"`);
    expect(markup).toContain('>Add workflow</button>');
    expect(markup).toContain('>Discard</button>');
    expect(markup).not.toContain('role="dialog"');
    expect(markup).not.toContain('This workflow has problems');
    expect(calls).toEqual([]);
  });
  test('renders warning-only drafts without the error warning', () => {
    const markup = render(
      makeStore({
        draft: {
          document: makeFixtureDocument(),
          outline: 'A draft',
          issues: [WARNING],
          source: 'history'
        }
      }).store
    );
    expect(markup).toContain(WARNING.message);
    expect(markup).toContain('Warning:');
    expect(markup).not.toContain('This workflow has problems');
    expect(markup).not.toContain('data-status="error"');
  });

  test('renders error issues first and warns about problems while keeping approval explicit', () => {
    const markup = render(
      makeStore({
        dialog: 'shortcuts',
        draft: {
          document: makeFixtureDocument(),
          outline: 'A draft',
          issues: [WARNING, ERROR, INFORMATION],
          source: 'assistant'
        }
      }).store
    );
    expect(markup).toContain('This workflow has problems and may not run as proposed.');
    expect(markup).toContain('data-status="error"');
    expect(markup.indexOf(escapeMarkup(ERROR.message))).toBeLessThan(
      markup.indexOf(WARNING.message)
    );
    expect(markup.indexOf(WARNING.message)).toBeLessThan(markup.indexOf(INFORMATION.message));
    expect(markup).toContain('role="dialog"');
    expect(markup).toContain('aria-label="Proposed workflow"');
  });
  test('renders empty, unknown, errored, running and stale workflows without requiring a runner', () => {
    const document = makeFixtureDocument();
    document.steps[0].operationId = 'missing-operation';
    for (const status of ['error', 'running', 'stale'] as const) {
      const markup = render(
        makeStore({
          dialog: 'import-export',
          document,
          runner: {
            revision: 1,
            running: status === 'running',
            order: ['source'],
            issues: [],
            steps: {source: {status, key: null, issues: [ERROR]}}
          }
        }).store
      );
      expect(markup).toContain('missing-operation');
    }
    expect(
      render(makeStore({dialog: 'import-export', document: {...document, steps: []}}).store)
    ).toContain('&quot;steps&quot;: []');
  });
});

// The dispatcher executes real component callbacks and effects, with controlled state and
// promises. It does not emulate browser layout, native tab movement or React scheduling.
function makeHookHarness<Properties>(
  component: (properties: Properties) => React.JSX.Element | null
) {
  const internals = (
    hostRequire('react') as {
      __CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE: {H: unknown};
    }
  ).__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;
  const slots: unknown[] = [];
  const cleanups = new Map<number, () => void>();
  let cursor = 0;
  let dirty = false;
  let effects: Array<() => void> = [];
  function useSlot<Value>(create: () => Value): {value: Value; index: number} {
    const index = cursor++;
    if (slots.length <= index) slots.push(create());
    return {value: slots[index] as Value, index};
  }
  function equalDependencies(
    left: readonly unknown[] | undefined,
    right: readonly unknown[] | undefined
  ): boolean {
    return Boolean(
      left &&
        right &&
        left.length === right.length &&
        left.every((value, index) => Object.is(value, right[index]))
    );
  }
  const dispatcher = {
    useState<Value>(initial: Value | (() => Value)) {
      const slot = useSlot(() => ({
        current: typeof initial === 'function' ? (initial as () => Value)() : initial
      })).value;
      return [
        slot.current,
        (next: Value | ((previous: Value) => Value)) => {
          const value =
            typeof next === 'function' ? (next as (previous: Value) => Value)(slot.current) : next;
          if (!Object.is(value, slot.current)) {
            slot.current = value;
            dirty = true;
          }
        }
      ];
    },
    useRef<Value>(initial: Value) {
      return useSlot(() => ({current: initial})).value;
    },
    useId() {
      return useSlot(() => `probe-${cursor}`).value;
    },
    useMemo<Value>(create: () => Value, dependencies: readonly unknown[] | undefined) {
      const slot = useSlot(() => ({
        initialized: false,
        current: undefined as Value | undefined,
        dependencies: undefined as readonly unknown[] | undefined
      })).value;
      if (!slot.initialized || !equalDependencies(slot.dependencies, dependencies)) {
        slot.current = create();
        slot.dependencies = dependencies;
        slot.initialized = true;
      }
      return slot.current;
    },
    useEffect(effect: () => void | (() => void), dependencies: readonly unknown[] | undefined) {
      const {value: slot, index} = useSlot(() => ({
        initialized: false,
        dependencies: undefined as readonly unknown[] | undefined
      }));
      if (!slot.initialized || !equalDependencies(slot.dependencies, dependencies)) {
        slot.initialized = true;
        slot.dependencies = dependencies;
        effects.push(() => {
          cleanups.get(index)?.();
          const cleanup = effect();
          if (cleanup) cleanups.set(index, cleanup);
          else cleanups.delete(index);
        });
      }
    },
    useSyncExternalStore<Value>(_subscribe: unknown, snapshot: () => Value) {
      useSlot(() => null);
      return snapshot();
    }
  };
  return {
    render(properties: Properties): React.JSX.Element | null {
      for (let iteration = 0; iteration < 20; iteration++) {
        cursor = 0;
        dirty = false;
        effects = [];
        const previous = internals.H;
        let tree: React.JSX.Element | null;
        try {
          internals.H = dispatcher;
          tree = component(properties);
        } finally {
          internals.H = previous;
        }
        if (!dirty) return tree;
      }
      throw new Error('Hook probe did not settle within 20 renders.');
    },
    flushEffects() {
      const pending = effects;
      effects = [];
      for (const effect of pending) effect();
    },
    unmount() {
      for (const cleanup of cleanups.values()) cleanup();
    }
  };
}

type ProbeElement = ReactElement<Record<string, unknown>>;

function findElements(node: ReactNode): ProbeElement[] {
  if (Array.isArray(node)) return node.flatMap(findElements);
  if (!isValidElement<Record<string, unknown>>(node)) return [];
  return [node, ...findElements(node.props['children'] as ReactNode)];
}

function findElement(node: ReactNode, property: string, value: unknown): ProbeElement {
  const found = findElements(node).find(element => element.props[property] === value);
  if (!found) throw new Error(`Missing element with ${property} = ${String(value)}`);
  return found;
}

function invokeHandler(element: ProbeElement, name: string, event?: unknown): unknown {
  const handler = element.props[name];
  if (typeof handler !== 'function') throw new Error(`Missing ${name} handler`);
  return handler(event);
}

function makeTransferProbe(store: WorkbenchStore, initialTab: 'json' | 'import' = 'json') {
  const harness = makeHookHarness(WorkbenchTransfer);
  const render = () => {
    const tree = harness.render({store, initialTab});
    harness.flushEffects();
    return tree;
  };
  return {render, unmount: harness.unmount};
}

function makeDeferred<Value>() {
  let resolve!: (value: Value) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<Value>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return {promise, resolve, reject};
}

function makeFileEvent(text: () => Promise<string>, size = 10) {
  return {target: {files: [{size, text}], value: 'selected.json'}};
}

function readImportText(tree: ReactNode): unknown {
  return findElement(tree, 'data-testid', WORKBENCH_TEST_IDS.importText).props['value'];
}

function readConfirmation(tree: ReactNode): unknown {
  return findElement(tree, 'aria-live', 'polite').props['children'];
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('dialog async regressions', () => {
  test('F1: the newest file selection wins when reads complete in reverse order', async () => {
    const probe = makeTransferProbe(makeStore().store, 'import');
    const older = makeDeferred<string>();
    const newer = makeDeferred<string>();
    const file = findElement(probe.render(), 'type', 'file');
    const first = invokeHandler(
      file,
      'onChange',
      makeFileEvent(() => older.promise)
    );
    const second = invokeHandler(
      file,
      'onChange',
      makeFileEvent(() => newer.promise)
    );
    newer.resolve('New selection');
    await second;
    expect(readImportText(probe.render())).toBe('New selection');
    older.resolve('Old selection');
    await first;
    expect(readImportText(probe.render())).toBe('New selection');
    probe.unmount();
  });

  test('F1: manual edits invalidate an outstanding file read', async () => {
    const probe = makeTransferProbe(makeStore().store, 'import');
    const older = makeDeferred<string>();
    const tree = probe.render();
    const pending = invokeHandler(
      findElement(tree, 'type', 'file'),
      'onChange',
      makeFileEvent(() => older.promise)
    );
    invokeHandler(findElement(tree, 'data-testid', WORKBENCH_TEST_IDS.importText), 'onChange', {
      target: {value: 'Manually edited text'}
    });
    older.resolve('Older file contents');
    await pending;
    expect(readImportText(probe.render())).toBe('Manually edited text');
    probe.unmount();
  });

  test('F3: a JSON copy cannot confirm a different visible export', async () => {
    const {store} = makeStore();
    const completion = makeDeferred<void>();
    const writeText = vi.fn(() => completion.promise);
    vi.stubGlobal('navigator', {clipboard: {writeText}});
    const probe = makeTransferProbe(store);
    const tree = probe.render();
    const pending = invokeHandler(findElement(tree, 'children', 'Copy'), 'onClick');
    invokeHandler(findElement(tree, 'children', 'Export TypeScript'), 'onClick');
    const switched = probe.render();
    expect(findElement(switched, 'data-testid', WORKBENCH_TEST_IDS.exportText).props['value']).toBe(
      exportWorkflowTypeScript(store.getState().document!, store.registry)
    );
    expect(writeText).toHaveBeenCalledExactlyOnceWith(
      exportWorkflowJSON(store.getState().document!)
    );
    completion.resolve();
    await pending;
    expect(readConfirmation(probe.render())).toBe('');
    probe.unmount();
  });
});

function makeModalProbe() {
  const {store, calls} = makeStore({dialog: 'import-export', document: null, workflows: []});
  const modal = findElements(renderWorkbenchDialogsTree(store)).find(
    element => typeof element.type === 'function' && element.type.name === 'Modal'
  );
  if (!modal || typeof modal.type !== 'function') throw new Error('Missing modal component');
  const harness = makeHookHarness(
    modal.type as (properties: Record<string, unknown>) => React.JSX.Element
  );
  const tree = harness.render(modal.props);
  const listeners = new Map<string, (event: unknown) => void>();
  const owner = {
    activeElement: null as FocusControl | null,
    addEventListener: (name: string, handler: (event: unknown) => void) =>
      listeners.set(name, handler),
    removeEventListener: (name: string) => listeners.delete(name)
  };
  class FocusControl {
    hidden = false;
    isConnected = true;
    constructor(
      readonly label: string,
      readonly tabIndex: number
    ) {}
    closest() {
      return null;
    }
    focus = vi.fn(() => {
      owner.activeElement = this;
    });
  }
  const opener = new FocusControl('Open dialog', 0);
  const close = new FocusControl('Close', 0);
  const selected = new FocusControl('Export JSON', 0);
  const inactive = ['Export TypeScript', 'Import', 'From catalog history'].map(
    label => new FocusControl(label, -1)
  );
  const panel = {
    ownerDocument: owner,
    querySelectorAll: () => [close, selected, ...inactive],
    contains: (value: unknown) => [close, selected, ...inactive].includes(value as FocusControl),
    focus: vi.fn()
  };
  owner.activeElement = opener;
  const panelElement = findElement(tree, 'role', 'dialog');
  (panelElement.props['ref'] as {current: unknown}).current = panel;
  vi.stubGlobal('HTMLElement', FocusControl);
  vi.stubGlobal('Node', FocusControl);
  harness.flushEffects();
  return {harness, listeners, owner, opener, close, selected, inactive, calls};
}

function renderWorkbenchDialogsTree(store: WorkbenchStore): ReactNode {
  const harness = makeHookHarness(WorkbenchDialogs);
  return harness.render({store});
}

describe('modal focus regressions', () => {
  test('F2: Shift+Tab skips inactive tabs and wraps to the selected export tab', () => {
    const probe = makeModalProbe();
    expect(probe.close.focus).toHaveBeenCalledTimes(1);
    const preventDefault = vi.fn();
    probe.listeners.get('keydown')!({key: 'Tab', shiftKey: true, preventDefault});
    expect(preventDefault).toHaveBeenCalledTimes(1);
    expect(probe.owner.activeElement?.label).toBe('Export JSON');
    expect(probe.selected.focus).toHaveBeenCalledTimes(1);
    for (const inactive of probe.inactive) expect(inactive.focus).not.toHaveBeenCalled();
    probe.harness.unmount();
  });
});

describe('executed import and file handlers', () => {
  test('keeps malformed JSON visible, reports the WorkflowError and allows a corrected import', () => {
    const {store, calls, addedDocuments} = makeStore();
    const probe = makeTransferProbe(store, 'import');
    const invalid = '{';
    invokeHandler(
      findElement(probe.render(), 'data-testid', WORKBENCH_TEST_IDS.importText),
      'onChange',
      {target: {value: invalid}}
    );
    invokeHandler(
      findElement(probe.render(), 'data-testid', WORKBENCH_TEST_IDS.importApply),
      'onClick'
    );
    let expectedMessage = '';
    try {
      importWorkflowJSON(invalid);
    } catch (error) {
      expect(error).toBeInstanceOf(WorkflowError);
      expect((error as WorkflowError).code).toBe('malformed');
      expectedMessage = (error as WorkflowError).message;
    }
    const failed = probe.render();
    expect(findElement(failed, 'role', 'alert').props['children']).toBe(expectedMessage);
    expect(readImportText(failed)).toBe(invalid);
    expect(calls).toEqual([]);
    const document = {...makeFixtureDocument(), title: 'Corrected import'};
    invokeHandler(findElement(failed, 'data-testid', WORKBENCH_TEST_IDS.importText), 'onChange', {
      target: {value: exportWorkflowJSON(document)}
    });
    expect(findElements(probe.render()).some(element => element.props['role'] === 'alert')).toBe(
      false
    );
    invokeHandler(
      findElement(probe.render(), 'data-testid', WORKBENCH_TEST_IDS.importApply),
      'onClick'
    );
    expect(addedDocuments).toEqual([document]);
    expect(calls).toEqual(['add', 'dialog']);
    probe.unmount();
  });

  test('keeps the dialog and text on store refusal', () => {
    const {store, calls} = makeStore({
      notice: {kind: 'error', message: 'Workflows: maximumWorkflows is 16.'}
    });
    store.addWorkflow = () => '';
    const probe = makeTransferProbe(store, 'import');
    const text = exportWorkflowJSON(makeFixtureDocument());
    invokeHandler(
      findElement(probe.render(), 'data-testid', WORKBENCH_TEST_IDS.importText),
      'onChange',
      {target: {value: text}}
    );
    invokeHandler(
      findElement(probe.render(), 'data-testid', WORKBENCH_TEST_IDS.importApply),
      'onClick'
    );
    expect(findElement(probe.render(), 'role', 'alert').props['children']).toBe(
      'Workflows: maximumWorkflows is 16.'
    );
    expect(readImportText(probe.render())).toBe(text);
    expect(calls).toEqual([]);
    probe.unmount();
  });

  test.each([
    -1, 0, 1
  ])('checks the exact file size boundary (%s bytes from the limit)', async offset => {
    const probe = makeTransferProbe(makeStore().store, 'import');
    const maximumBytes = WORKFLOW_JSON_LIMITS.maximumFormattedCharacters * 4;
    const readText = vi.fn(async () => 'File contents');
    const event = makeFileEvent(readText, maximumBytes + offset);
    await invokeHandler(findElement(probe.render(), 'type', 'file'), 'onChange', event);
    expect(event.target.value).toBe('');
    if (offset > 0) {
      expect(readText).not.toHaveBeenCalled();
      expect(readImportText(probe.render())).toBe('');
      expect(findElement(probe.render(), 'role', 'alert').props['children']).toBe(
        `Workflow JSON: File size must be at most ${maximumBytes.toLocaleString()} bytes.`
      );
    } else {
      expect(readText).toHaveBeenCalledTimes(1);
      expect(readImportText(probe.render())).toBe('File contents');
    }
    probe.unmount();
  });

  test('reports read failures and ignores superseded failures', async () => {
    const probe = makeTransferProbe(makeStore().store, 'import');
    const failedRead = makeDeferred<string>();
    const file = findElement(probe.render(), 'type', 'file');
    const pending = invokeHandler(
      file,
      'onChange',
      makeFileEvent(() => failedRead.promise)
    );
    failedRead.reject(new Error('File access failed.'));
    await pending;
    expect(findElement(probe.render(), 'role', 'alert').props['children']).toBe(
      'File access failed.'
    );
    const obsolete = makeDeferred<string>();
    const older = invokeHandler(
      file,
      'onChange',
      makeFileEvent(() => obsolete.promise)
    );
    await invokeHandler(
      file,
      'onChange',
      makeFileEvent(async () => 'New file')
    );
    obsolete.reject(new Error('Obsolete failure.'));
    await older;
    expect(readImportText(probe.render())).toBe('New file');
    expect(findElements(probe.render()).some(element => element.props['role'] === 'alert')).toBe(
      false
    );
    probe.unmount();
  });

  test('a newer oversized selection prevents an older read from replacing the text', async () => {
    const probe = makeTransferProbe(makeStore().store, 'import');
    const older = makeDeferred<string>();
    const file = findElement(probe.render(), 'type', 'file');
    const pending = invokeHandler(
      file,
      'onChange',
      makeFileEvent(() => older.promise)
    );
    await invokeHandler(
      file,
      'onChange',
      makeFileEvent(
        async () => 'Oversized',
        WORKFLOW_JSON_LIMITS.maximumFormattedCharacters * 4 + 1
      )
    );
    older.resolve('Old text');
    await pending;
    expect(readImportText(probe.render())).toBe('');
    expect(findElement(probe.render(), 'role', 'alert').props['children']).toContain(
      'File size must be at most'
    );
    probe.unmount();
  });
});

describe('executed clipboard handlers', () => {
  test('confirms the current export and shows a fallback when the clipboard is unavailable', async () => {
    const probe = makeTransferProbe(makeStore().store);
    const writeText = vi.fn(async () => {});
    vi.stubGlobal('navigator', {clipboard: {writeText}});
    await invokeHandler(findElement(probe.render(), 'children', 'Copy'), 'onClick');
    expect(readConfirmation(probe.render())).toBe('Copied');
    vi.stubGlobal('navigator', {});
    await invokeHandler(findElement(probe.render(), 'children', 'Copy'), 'onClick');
    expect(readConfirmation(probe.render())).toBe(
      'Clipboard unavailable. Select the export text and copy it manually.'
    );
    probe.unmount();
  });

  test('shows current failures but ignores failures after a tab change', async () => {
    const probe = makeTransferProbe(makeStore().store);
    vi.stubGlobal('navigator', {
      clipboard: {
        writeText: async () => {
          throw new Error('Denied');
        }
      }
    });
    await invokeHandler(findElement(probe.render(), 'children', 'Copy'), 'onClick');
    expect(readConfirmation(probe.render())).toBe(
      'Could not copy. Select the export text and copy it manually.'
    );
    const completion = makeDeferred<void>();
    vi.stubGlobal('navigator', {clipboard: {writeText: () => completion.promise}});
    const tree = probe.render();
    const pending = invokeHandler(findElement(tree, 'children', 'Copy'), 'onClick');
    invokeHandler(findElement(tree, 'children', 'Export TypeScript'), 'onClick');
    probe.render();
    completion.reject(new Error('Delayed denial'));
    await pending;
    expect(readConfirmation(probe.render())).toBe('');
    probe.unmount();
  });

  test('ignores pending copy completions when the workflow changes in the same tab', async () => {
    const {store} = makeStore();
    const probe = makeTransferProbe(store);
    const completion = makeDeferred<void>();
    vi.stubGlobal('navigator', {clipboard: {writeText: () => completion.promise}});
    const pending = invokeHandler(findElement(probe.render(), 'children', 'Copy'), 'onClick');
    store.getState().document = {...makeFixtureDocument(), title: 'Updated export'};
    probe.render();
    completion.resolve();
    await pending;
    expect(readConfirmation(probe.render())).toBe('');
    probe.unmount();
  });

  test('does not revive a copy confirmation after switching away and back', async () => {
    const probe = makeTransferProbe(makeStore().store);
    const completion = makeDeferred<void>();
    vi.stubGlobal('navigator', {clipboard: {writeText: () => completion.promise}});
    const tree = probe.render();
    const pending = invokeHandler(findElement(tree, 'children', 'Copy'), 'onClick');
    invokeHandler(findElement(tree, 'children', 'Export TypeScript'), 'onClick');
    invokeHandler(findElement(probe.render(), 'children', 'Export JSON'), 'onClick');
    probe.render();
    completion.resolve();
    await pending;
    expect(readConfirmation(probe.render())).toBe('');
    probe.unmount();
  });
});

describe('executed download handlers', () => {
  test.each([
    'json',
    'typescript'
  ] as const)('downloads the %s export and releases its object URL', async tab => {
    const {store} = makeStore();
    const probe = makeTransferProbe(store);
    if (tab === 'typescript')
      invokeHandler(findElement(probe.render(), 'children', 'Export TypeScript'), 'onClick');
    const anchor = {href: '', download: '', click: vi.fn(), remove: vi.fn()};
    const append = vi.fn();
    vi.stubGlobal('document', {createElement: vi.fn(() => anchor), body: {append}});
    const createObjectURL = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:workflow');
    const revokeObjectURL = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    invokeHandler(findElement(probe.render(), 'children', 'Download'), 'onClick');
    expect(anchor.download).toBe(`fixture-workflow.workflow.${tab === 'json' ? 'json' : 'ts'}`);
    expect(anchor.href).toBe('blob:workflow');
    expect(append).toHaveBeenCalledExactlyOnceWith(anchor);
    expect(anchor.click).toHaveBeenCalledTimes(1);
    expect(anchor.remove).toHaveBeenCalledTimes(1);
    const blob = createObjectURL.mock.calls[0][0] as Blob;
    expect(await blob.text()).toBe(
      tab === 'json'
        ? exportWorkflowJSON(store.getState().document!)
        : exportWorkflowTypeScript(store.getState().document!, store.registry)
    );
    expect(blob.type).toBe(tab === 'json' ? 'application/json' : 'text/typescript');
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(revokeObjectURL).toHaveBeenCalledExactlyOnceWith('blob:workflow');
    probe.unmount();
  });

  test('reports download failures inline and still cleans up', async () => {
    const probe = makeTransferProbe(makeStore().store);
    const anchor = {
      href: '',
      download: '',
      click: () => {
        throw new Error('Download blocked.');
      },
      remove: vi.fn()
    };
    vi.stubGlobal('document', {createElement: () => anchor, body: {append: vi.fn()}});
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:failed');
    const revokeObjectURL = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    invokeHandler(findElement(probe.render(), 'children', 'Download'), 'onClick');
    expect(findElement(probe.render(), 'role', 'alert').props['children']).toBe(
      'Download blocked.'
    );
    expect(anchor.remove).toHaveBeenCalledTimes(1);
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(revokeObjectURL).toHaveBeenCalledExactlyOnceWith('blob:failed');
    probe.unmount();
  });
});

function makeHistoryProbe(store: WorkbenchStore) {
  const transfer = makeTransferProbe(store);
  invokeHandler(findElement(transfer.render(), 'children', 'From catalog history'), 'onClick');
  const history = findElements(transfer.render()).find(
    element => typeof element.type === 'function' && element.type.name === 'HistoryPromotion'
  );
  if (!history || typeof history.type !== 'function') throw new Error('Missing history component');
  const harness = makeHookHarness(
    history.type as (properties: {store: WorkbenchStore}) => React.JSX.Element
  );
  return {
    render: () => harness.render({store}),
    unmount: () => {
      harness.unmount();
      transfer.unmount();
    }
  };
}

describe('executed history handlers', () => {
  test('promotes only selected history to one planned draft without adding or running it', () => {
    const fixture = makeStore();
    const store = {...fixture.store, registry: getWorkflowOperationRegistry()};
    const operationId = 'geometry-point-buffer';
    const binding = getCatalogBinding(operationId)!;
    const sampleText = getCatalogSampleInputs(operationId)['points'];
    const invocation = serializeCatalogInvocation(
      binding,
      {radius: 25},
      {points: JSON.parse(sampleText)}
    );
    restoreCatalogInvocations([
      {id: 'first', operationId, invocation, inputs: {points: {mode: 'sample'}}, engine: 'cpu'},
      {id: 'second', operationId, invocation, inputs: {points: {mode: 'sample'}}, engine: 'auto'}
    ]);
    try {
      const probe = makeHistoryProbe(store);
      const checkboxes = findElements(probe.render()).filter(
        element => element.props['type'] === 'checkbox'
      );
      expect(checkboxes.map(element => element.props['checked'])).toEqual([true, true]);
      invokeHandler(checkboxes[1], 'onChange', {target: {checked: false}});
      expect(
        findElements(probe.render())
          .filter(element => element.props['type'] === 'checkbox')
          .map(element => element.props['checked'])
      ).toEqual([true, false]);
      invokeHandler(findElement(probe.render(), 'children', 'Create draft'), 'onClick');
      expect(fixture.calls).toEqual(['propose', 'dialog']);
      expect(fixture.addedDocuments).toEqual([]);
      expect(fixture.drafts).toHaveLength(1);
      const draft = fixture.drafts[0];
      expect(draft.source).toBe('history');
      expect(draft.document.steps.map(step => step.operationId)).toEqual([
        'source-geojson',
        operationId
      ]);
      expect(draft.document.steps[0].parameters['text']).toEqual({
        from: 'static',
        value: sampleText
      });
      expect(draft.document.steps[1].id).toBe('first');
      expect(draft.document.steps[1].engine).toBe('cpu');
      expect(draft.document.steps[1].parameters['radius']).toEqual({from: 'static', value: 25});
      const plan = planWorkflow(draft.document, {registry: store.registry, host: store.host});
      expect(draft.issues).toEqual([
        ...plan.issues,
        ...plan.order.flatMap(stepId => plan.steps[stepId].issues)
      ]);
      expect(draft.outline).toBe(describeWorkflow(draft.document, store.registry));
      invokeHandler(checkboxes[0], 'onChange', {target: {checked: false}});
      expect(findElement(probe.render(), 'children', 'Create draft').props['disabled']).toBe(true);
      invokeHandler(checkboxes[1], 'onChange', {target: {checked: true}});
      expect(findElement(probe.render(), 'children', 'Create draft').props['disabled']).toBe(false);
      probe.unmount();
    } finally {
      clearCatalogInvocations();
    }
  });

  test('reports unsupported history inline without proposing or closing', () => {
    const {store, calls, drafts} = makeStore();
    restoreCatalogInvocations([
      {id: 'broken', operationId: 'missing', invocation: '{}', inputs: {}, engine: 'cpu'}
    ]);
    try {
      const probe = makeHistoryProbe(store);
      invokeHandler(findElement(probe.render(), 'children', 'Create draft'), 'onClick');
      expect(findElement(probe.render(), 'role', 'alert').props['children']).toBe(
        'Catalog history contains no supported runs to promote.'
      );
      expect(calls).toEqual([]);
      expect(drafts).toEqual([]);
      probe.unmount();
    } finally {
      clearCatalogInvocations();
    }
  });
});

describe('executed modal lifecycle', () => {
  test('wraps forward, closes on Escape, removes listeners and restores the opener', () => {
    const probe = makeModalProbe();
    probe.selected.focus();
    const preventDefault = vi.fn();
    probe.listeners.get('keydown')!({key: 'Tab', shiftKey: false, preventDefault});
    expect(preventDefault).toHaveBeenCalledTimes(1);
    expect(probe.owner.activeElement).toBe(probe.close);
    const stopPropagation = vi.fn();
    probe.listeners.get('keydown')!({key: 'Escape', preventDefault, stopPropagation});
    expect(probe.calls).toEqual(['dialog']);
    expect(stopPropagation).toHaveBeenCalledTimes(1);
    probe.harness.unmount();
    expect(probe.listeners.size).toBe(0);
    expect(probe.opener.focus).toHaveBeenCalledTimes(1);
    expect(probe.owner.activeElement).toBe(probe.opener);
  });
});

describe('saved workflow assistant tools', () => {
  test('lists saved workflows and describes runner state only for the active workflow', () => {
    const document = makeFixtureDocument();
    const other = {...document, id: 'other', title: 'Other workflow'};
    const runner: WorkflowRunnerState = {
      revision: 1,
      running: false,
      order: ['source'],
      issues: [],
      steps: {source: {key: null, status: 'stale', issues: [WARNING]}}
    };
    const {store, calls} = makeStore({workflows: [document, other], runner});
    expect(runWorkbenchAssistantTool(store, {name: 'listWorkflows', arguments: {}})).toEqual(
      listWorkflowsForAssistant([document, other], store.registry)
    );
    for (const selected of [document, other]) {
      expect(
        runWorkbenchAssistantTool(store, {
          name: 'describeWorkflow',
          arguments: {workflowId: selected.id}
        })
      ).toEqual(
        describeWorkflowForAssistant(
          [document, other],
          store.registry,
          {workflowId: selected.id},
          selected.id === document.id ? runner : undefined
        )
      );
    }
    expect(calls).toEqual([]);
  });

  test('previews input changes without effects, then applies them in one approved edit', () => {
    const document: WorkflowDocument = {
      ...makeFixtureDocument(),
      inputs: [{id: 'factor', label: 'Factor', kind: 'number', value: 2, minimum: 0, maximum: 10}]
    };
    const {store, calls} = makeStore({document, workflows: [document]});
    const call = {
      name: 'setWorkflowInputs',
      arguments: {workflowId: document.id, inputs: {Factor: 5}}
    };
    const changes = [{inputId: 'factor', label: 'Factor', previous: 2, next: 5}];
    expect(runWorkbenchAssistantTool(store, call)).toEqual({
      status: 'needs-approval',
      workflowTitle: document.title,
      changes
    });
    expect(document.inputs[0].value).toBe(2);
    expect(calls).toEqual([]);
    store.setActiveWorkflow = id => {
      calls.push(`active:${id}`);
    };
    let edited: WorkflowDocument | undefined;
    store.edit = (tag, change) => {
      calls.push(`edit:${tag}`);
      edited = change(document);
      return true;
    };
    expect(runWorkbenchAssistantTool(store, call, {approved: true})).toEqual({
      status: 'applied',
      changes
    });
    expect(calls).toEqual([`active:${document.id}`, 'edit:assistant-inputs']);
    expect(edited?.inputs[0].value).toBe(5);
    expect(
      runWorkbenchAssistantTool(
        store,
        {...call, arguments: {workflowId: document.id, inputs: {factor: 11}}},
        {approved: true}
      )
    ).toEqual({
      status: 'error',
      code: 'parameter-out-of-range',
      message: expect.stringContaining('Factor')
    });
    expect(calls).toHaveLength(2);
    store.edit = () => false;
    expect(runWorkbenchAssistantTool(store, call, {approved: true})).toEqual({
      status: 'error',
      code: 'operation-failed',
      message: 'Fixture workflow: Workflow inputs could not be applied.'
    });
  });

  test('previews whole and targeted runs, excludes completed steps, and starts only approved runs', async () => {
    const {store, calls} = makeStore();
    const document = store.getState().document!;
    const whole = {name: 'runWorkflow', arguments: {workflowId: document.id}};
    const target = {name: 'runWorkflow', arguments: {workflowId: document.id, target: 'scale'}};
    expect(runWorkbenchAssistantTool(store, whole)).toEqual({
      status: 'needs-approval',
      workflowTitle: document.title,
      steps: 3,
      cachedResultsKnown: true,
      issues: validateWorkflowForAssistant(document, {
        registry: store.registry,
        host: store.host
      }).map(({severity, code, message}) => ({severity, code, message})),
      message: 'Running a workflow needs explicit user approval of this exact call.'
    });
    expect(runWorkbenchAssistantTool(store, target)).toEqual({
      status: 'needs-approval',
      workflowTitle: document.title,
      steps: 2,
      cachedResultsKnown: true,
      issues: validateWorkflowForAssistant(document, {
        registry: store.registry,
        host: store.host,
        target: 'scale'
      }).map(({severity, code, message}) => ({severity, code, message})),
      message: 'Running a workflow needs explicit user approval of this exact call.'
    });
    expect(calls).toEqual([]);
    const runner = new WorkflowRunner({registry: store.registry, host: store.host});
    runner.setDocument(document);
    await runner.run();
    store.getState().runner = runner.getState();
    expect(runWorkbenchAssistantTool(store, whole)).toEqual({
      status: 'needs-approval',
      workflowTitle: document.title,
      steps: 0,
      cachedResultsKnown: true,
      issues: validateWorkflowForAssistant(document, {
        registry: store.registry,
        host: store.host
      }).map(({severity, code, message}) => ({severity, code, message})),
      message: 'Running a workflow needs explicit user approval of this exact call.'
    });
    await runner.destroy();
    const completion = makeDeferred<WorkflowRunRecord | null>();
    store.setActiveWorkflow = id => {
      calls.push(`active:${id}`);
    };
    store.run = options => {
      calls.push(`run:${options?.target}`);
      return completion.promise;
    };
    expect(runWorkbenchAssistantTool(store, target, {approved: true})).toEqual({
      status: 'started',
      workflowTitle: document.title,
      issues: validateWorkflowForAssistant(document, {
        registry: store.registry,
        host: store.host,
        target: 'scale'
      }).map(({severity, code, message}) => ({severity, code, message}))
    });
    expect(calls).toEqual([`active:${document.id}`, 'run:scale']);
    completion.resolve(null);
    expect(
      runWorkbenchAssistantTool(
        store,
        {name: 'runWorkflow', arguments: {workflowId: document.id, target: 'missing'}},
        {approved: true}
      )
    ).toEqual({status: 'error', code: 'unknown-step', message: expect.stringContaining('missing')});
    expect(calls).toHaveLength(2);
  });
});

function makeTemplatesProbe(store: WorkbenchStore) {
  const template = findElements(renderWorkbenchDialogsTree(store)).find(
    element => typeof element.type === 'function' && element.type.name === 'Templates'
  );
  if (!template || typeof template.type !== 'function')
    throw new Error('Missing templates component');
  const harness = makeHookHarness(
    template.type as (properties: {store: WorkbenchStore}) => React.JSX.Element
  );
  return {render: () => harness.render({store}), unmount: harness.unmount};
}

describe('workspace template actions', () => {
  test('renders user templates first, saves and uses them, and confirms deletion inline', () => {
    const document = {...makeFixtureDocument(), id: 'saved-template', title: 'Saved analysis'};
    const {store, calls} = makeStore({dialog: 'templates', templates: [document]});
    const markup = render(store);
    expect(markup.indexOf('Your templates')).toBeLessThan(
      markup.indexOf(listWorkflowTemplates()[0].title)
    );
    expect(markup).toContain('Saved analysis');
    expect(markup).toContain(
      'Adds these steps to the current workflow so you can connect them to your own.'
    );
    expect(markup).toContain('aria-label="Delete template Saved analysis"');
    store.saveTemplate = () => {
      calls.push('save');
      return document.id;
    };
    store.useTemplate = id => {
      calls.push(`use:${id}`);
      return 'new-workflow';
    };
    store.deleteTemplate = id => {
      calls.push(`delete:${id}`);
    };
    const probe = makeTemplatesProbe(store);
    invokeHandler(
      findElement(probe.render(), 'data-testid', WORKBENCH_TEST_IDS.saveTemplate),
      'onClick'
    );
    invokeHandler(
      findElement(probe.render(), 'data-testid', WORKBENCH_TEST_IDS.userTemplate(document.id)),
      'onClick'
    );
    expect(calls).toEqual(['use:saved-template', 'dialog']);
    expect(renderToStaticMarkup(probe.render())).toContain('Save reusable template');
    calls.length = 0;
    invokeHandler(
      findElement(
        probe.render(),
        'data-testid',
        WORKBENCH_TEST_IDS.userTemplateDelete(document.id)
      ),
      'onClick'
    );
    expect(calls).toEqual([]);
    expect(findElement(probe.render(), 'role', 'group').props['aria-label']).toBe(
      'Delete template Saved analysis'
    );
    const confirmationMarkup = renderToStaticMarkup(probe.render());
    expect(confirmationMarkup).toContain('aria-label="Delete template: Saved analysis"');
    expect(confirmationMarkup).toContain('aria-label="Cancel deleting template: Saved analysis"');
    invokeHandler(findElement(probe.render(), 'children', 'Cancel'), 'onClick');
    expect(findElements(probe.render()).some(element => element.props['role'] === 'group')).toBe(
      false
    );
    invokeHandler(
      findElement(
        probe.render(),
        'data-testid',
        WORKBENCH_TEST_IDS.userTemplateDelete(document.id)
      ),
      'onClick'
    );
    invokeHandler(findElement(probe.render(), 'children', 'Delete'), 'onClick');
    expect(calls).toEqual(['delete:saved-template']);
    probe.unmount();
  });

  test('inserts user and instantiated built-in templates and preserves refused insertions', () => {
    const document = {...makeFixtureDocument(), id: 'saved-template'};
    const {store, calls} = makeStore({dialog: 'templates', templates: [document]});
    const inserted: WorkflowDocument[] = [];
    store.insertWorkflow = source => {
      inserted.push(source);
      return true;
    };
    const probe = makeTemplatesProbe(store);
    invokeHandler(
      findElement(probe.render(), 'data-testid', WORKBENCH_TEST_IDS.insertWorkflow(document.id)),
      'onClick'
    );
    expect(inserted).toEqual([document]);
    const builtin = listWorkflowTemplates()[0];
    invokeHandler(
      findElement(probe.render(), 'data-testid', WORKBENCH_TEST_IDS.insertWorkflow(builtin.id)),
      'onClick'
    );
    expect(inserted[1].steps).toEqual(builtin.document.steps);
    expect(inserted[1]).not.toBe(builtin.document);
    expect(calls).toEqual(['dialog', 'dialog']);
    store.insertWorkflow = () => false;
    invokeHandler(
      findElement(probe.render(), 'data-testid', WORKBENCH_TEST_IDS.insertWorkflow(document.id)),
      'onClick'
    );
    expect(findElement(probe.render(), 'role', 'alert').props['children']).toBe(
      'The steps could not be inserted.'
    );
    expect(calls).toHaveLength(2);
    probe.unmount();
  });

  test('disables saving empty workflows and inserting without an active workflow', () => {
    for (const document of [null, {...makeFixtureDocument(), steps: []}]) {
      const {store} = makeStore({dialog: 'templates', document});
      const probe = makeTemplatesProbe(store);
      expect(
        findElement(probe.render(), 'data-testid', WORKBENCH_TEST_IDS.saveTemplate).props[
          'disabled'
        ]
      ).toBe(true);
      expect(
        findElement(
          probe.render(),
          'data-testid',
          WORKBENCH_TEST_IDS.insertWorkflow(listWorkflowTemplates()[0].id)
        ).props['disabled']
      ).toBe(!document);
      probe.unmount();
    }
  });
});

describe('run report and graph exports', () => {
  test('explains an unavailable run report and disables export actions', () => {
    const {store} = makeStore();
    const harness = makeHookHarness(WorkbenchTransfer);
    const tree = harness.render({store, initialTab: 'report'});
    expect(findElement(tree, 'children', 'Run the workflow to get a report.')).toBeDefined();
    expect(findElement(tree, 'children', 'Copy').props['disabled']).toBe(true);
    expect(findElement(tree, 'children', 'Download').props['disabled']).toBe(true);
    expect(findElement(tree, 'data-testid', WORKBENCH_TEST_IDS.runReportText).props['value']).toBe(
      ''
    );
    harness.unmount();
  });

  test.each([
    'report',
    'dot'
  ] as const)('exports the exact %s text through the textarea, clipboard and download', async tab => {
    const runner: WorkflowRunnerState = {
      revision: 1,
      running: false,
      order: [],
      issues: [],
      steps: {}
    };
    const record: WorkflowRunRecord = {
      schemaVersion: 1,
      workflowId: 'fixture-workflow',
      runId: 'fixture-workflow-7',
      startedAt: 0,
      durationMilliseconds: 42,
      status: 'done',
      steps: []
    };
    const {store} = makeStore({runner, runLog: [record]});
    const harness = makeHookHarness(WorkbenchTransfer);
    const tree = harness.render({store, initialTab: tab});
    const expected =
      tab === 'report'
        ? makeWorkflowRunReport(store.getState().document!, store.registry, runner, record)
        : makeWorkflowDot(store.getState().document!, store.registry, runner);
    expect(
      findElement(
        tree,
        'data-testid',
        tab === 'report' ? WORKBENCH_TEST_IDS.runReportText : WORKBENCH_TEST_IDS.exportText
      ).props['value']
    ).toBe(expected);
    expect(
      findElement(
        tree,
        'data-testid',
        tab === 'report' ? WORKBENCH_TEST_IDS.runReport : WORKBENCH_TEST_IDS.graphDot
      ).props['aria-selected']
    ).toBe(true);
    const writeText = vi.fn(async () => {});
    vi.stubGlobal('navigator', {clipboard: {writeText}});
    await invokeHandler(findElement(tree, 'children', 'Copy'), 'onClick');
    expect(writeText).toHaveBeenCalledExactlyOnceWith(expected);
    const anchor = {href: '', download: '', click: vi.fn(), remove: vi.fn()};
    vi.stubGlobal('document', {createElement: () => anchor, body: {append: vi.fn()}});
    const createObjectURL = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:export');
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    invokeHandler(findElement(tree, 'children', 'Download'), 'onClick');
    expect(anchor.download).toBe(
      tab === 'report' ? 'fixture-workflow-run-7.md' : 'fixture-workflow.dot'
    );
    expect(await (createObjectURL.mock.calls[0][0] as Blob).text()).toBe(expected);
    await new Promise(resolve => setTimeout(resolve, 0));
    harness.unmount();
  });
});

describe('latest-run report routing', () => {
  test('F4: the latest-run Report action opens the Run report tab with its Markdown', () => {
    const runner: WorkflowRunnerState = {
      revision: 1,
      running: false,
      order: [],
      issues: [],
      steps: {}
    };
    const record: WorkflowRunRecord = {
      schemaVersion: 1,
      workflowId: 'fixture-workflow',
      runId: 'fixture-workflow-1',
      startedAt: 0,
      durationMilliseconds: 42,
      status: 'done',
      steps: []
    };
    const {store} = makeStore({runner, runLog: [record]});
    store.setDialog = dialog => {
      store.getState().dialog = dialog;
    };
    const runLogHarness = makeHookHarness(WorkbenchRunLog);
    const runLogTree = runLogHarness.render({store});
    invokeHandler(findElement(runLogTree, 'children', 'Report'), 'onClick');
    expect(store.getState().dialog).toBe('import-export');
    const transfer = findElements(renderWorkbenchDialogsTree(store)).find(
      element => element.type === WorkbenchTransfer
    );
    if (!transfer) throw new Error('Missing import and export dialog');
    const transferHarness = makeHookHarness(WorkbenchTransfer);
    const properties = transfer.props as Parameters<typeof WorkbenchTransfer>[0];
    const tree = transferHarness.render(properties);
    expect(
      findElement(tree, 'data-testid', WORKBENCH_TEST_IDS.runReport).props['aria-selected']
    ).toBe(true);
    expect(findElement(tree, 'data-testid', WORKBENCH_TEST_IDS.runReportText).props['value']).toBe(
      makeWorkflowRunReport(store.getState().document!, store.registry, runner, record)
    );
    invokeHandler(findElement(tree, 'children', 'Export JSON'), 'onClick');
    const jsonTree = transferHarness.render(properties);
    expect(
      findElement(jsonTree, 'data-testid', WORKBENCH_TEST_IDS.exportJSON).props['aria-selected']
    ).toBe(true);
    expect(findElement(jsonTree, 'data-testid', WORKBENCH_TEST_IDS.exportText).props['value']).toBe(
      exportWorkflowJSON(store.getState().document!)
    );
    transferHarness.unmount();
    runLogHarness.unmount();
  });
});

describe('assistant approval against reentrant store subscribers', () => {
  test('F1: rejects an approved run if activation is redirected to another workflow', async () => {
    const operationCalls: string[] = [];
    const store = createWorkbenchStore({
      registry: makeFixtureRegistry(operationCalls),
      host: makeFixtureHost()
    });
    const approved = makeFixtureDocument();
    const other: WorkflowDocument = {
      ...approved,
      id: 'other-workflow',
      title: 'Other workflow',
      steps: [{...approved.steps[0], id: 'unapproved-source'}]
    };
    store.addWorkflow(approved);
    store.addWorkflow(other);
    const unsubscribe = store.subscribe(() => {
      if (store.getState().activeWorkflowId === approved.id) store.setActiveWorkflow(other.id);
    });
    const run = vi.spyOn(store, 'run');
    try {
      const result = runWorkbenchAssistantTool(
        store,
        {name: 'runWorkflow', arguments: {workflowId: approved.id}},
        {approved: true}
      );
      await run.mock.results[0]?.value;
      expect(operationCalls).toEqual([]);
      expect(run).not.toHaveBeenCalled();
      expect(result).toEqual({
        status: 'error',
        code: 'operation-failed',
        message:
          'Fixture workflow: Workflow activation changed before the run could start. Try again.'
      });
      expect(store.getState().activeWorkflowId).toBe(other.id);
    } finally {
      unsubscribe();
      await store.clear();
    }
  });

  test('F2: patches current inputs and preserves unrelated activation-time edits', async () => {
    const store = createWorkbenchStore({registry: makeFixtureRegistry(), host: makeFixtureHost()});
    const approved: WorkflowDocument = {
      ...makeFixtureDocument(),
      inputs: [{id: 'factor', label: 'Factor', kind: 'number', value: 2, minimum: 0, maximum: 10}]
    };
    store.addWorkflow(approved);
    store.addWorkflow({...makeFixtureDocument(), id: 'other-workflow', title: 'Other workflow'});
    let changed = false;
    const unsubscribe = store.subscribe(() => {
      if (!changed && store.getState().activeWorkflowId === approved.id) {
        changed = true;
        store.edit('user-title', current => ({
          ...current,
          title: 'User title change',
          description: 'Keep this description.'
        }));
      }
    });
    try {
      const result = runWorkbenchAssistantTool(
        store,
        {name: 'setWorkflowInputs', arguments: {workflowId: approved.id, inputs: {factor: 5}}},
        {approved: true}
      );
      expect(store.getState().document?.title).toBe('User title change');
      expect(store.getState().document?.description).toBe('Keep this description.');
      expect(store.getState().document?.inputs[0].value).toBe(5);
      expect(result).toEqual({
        status: 'applied',
        changes: [{inputId: 'factor', label: 'Factor', previous: 2, next: 5}]
      });
      store.undo();
      expect(store.getState().document?.title).toBe('User title change');
      expect(store.getState().document?.inputs[0].value).toBe(2);
    } finally {
      unsubscribe();
      await store.clear();
    }
  });

  test('marks inactive run counts as uncached estimates and active cached counts as exact', async () => {
    const operationCalls: string[] = [];
    const store = createWorkbenchStore({
      registry: makeFixtureRegistry(operationCalls),
      host: makeFixtureHost()
    });
    const document = makeFixtureDocument();
    store.addWorkflow(document);
    await store.run();
    expect(operationCalls).toEqual([
      'fixture-source:source',
      'fixture-scale:scale',
      'fixture-count:count'
    ]);
    store.addWorkflow({...document, id: 'other-workflow', title: 'Other workflow'});
    operationCalls.length = 0;
    const snapshot = store.getState();
    try {
      expect(
        runWorkbenchAssistantTool(store, {
          name: 'runWorkflow',
          arguments: {workflowId: document.id}
        })
      ).toEqual({
        status: 'needs-approval',
        workflowTitle: document.title,
        steps: 3,
        cachedResultsKnown: false,
        issues: validateWorkflowForAssistant(document, {
          registry: store.registry,
          host: store.host
        }).map(({severity, code, message}) => ({severity, code, message})),
        message: 'Running a workflow needs explicit user approval of this exact call.'
      });
      expect(
        runWorkbenchAssistantTool(store, {
          name: 'runWorkflow',
          arguments: {workflowId: document.id, target: 'scale'}
        })
      ).toEqual({
        status: 'needs-approval',
        workflowTitle: document.title,
        steps: 2,
        cachedResultsKnown: false,
        issues: validateWorkflowForAssistant(document, {
          registry: store.registry,
          host: store.host,
          target: 'scale'
        }).map(({severity, code, message}) => ({severity, code, message})),
        message: 'Running a workflow needs explicit user approval of this exact call.'
      });
      expect(store.getState()).toBe(snapshot);
      store.setActiveWorkflow(document.id);
      expect(
        runWorkbenchAssistantTool(store, {
          name: 'runWorkflow',
          arguments: {workflowId: document.id}
        })
      ).toEqual({
        status: 'needs-approval',
        workflowTitle: document.title,
        steps: 0,
        cachedResultsKnown: true,
        issues: validateWorkflowForAssistant(document, {
          registry: store.registry,
          host: store.host
        }).map(({severity, code, message}) => ({severity, code, message})),
        message: 'Running a workflow needs explicit user approval of this exact call.'
      });
      expect(operationCalls).toEqual([]);
    } finally {
      await store.clear();
    }
  });
});

describe('assistant input patches after activation', () => {
  test('reports current values and preserves changed labels while patching by input id', async () => {
    const store = createWorkbenchStore({registry: makeFixtureRegistry(), host: makeFixtureHost()});
    const document: WorkflowDocument = {
      ...makeFixtureDocument(),
      inputs: [{id: 'factor', label: 'Factor', kind: 'number', value: 2, minimum: 0, maximum: 10}]
    };
    store.addWorkflow(document);
    store.addWorkflow({...makeFixtureDocument(), id: 'other-workflow'});
    let changed = false;
    const unsubscribe = store.subscribe(() => {
      if (!changed && store.getState().activeWorkflowId === document.id) {
        changed = true;
        store.edit('user-input', current => ({
          ...current,
          inputs: current.inputs.map(input => ({...input, label: 'Multiplier', value: 3}))
        }));
      }
    });
    try {
      expect(
        runWorkbenchAssistantTool(
          store,
          {name: 'setWorkflowInputs', arguments: {workflowId: document.id, inputs: {Factor: 5}}},
          {approved: true}
        )
      ).toEqual({
        status: 'applied',
        changes: [{inputId: 'factor', label: 'Multiplier', previous: 3, next: 5}]
      });
      expect(store.getState().document?.inputs[0]).toEqual({
        id: 'factor',
        label: 'Multiplier',
        kind: 'number',
        value: 5,
        minimum: 0,
        maximum: 10
      });
    } finally {
      unsubscribe();
      await store.clear();
    }
  });

  test('revalidates the patch against activation-time input limits', async () => {
    const store = createWorkbenchStore({registry: makeFixtureRegistry(), host: makeFixtureHost()});
    const document: WorkflowDocument = {
      ...makeFixtureDocument(),
      inputs: [{id: 'factor', label: 'Factor', kind: 'number', value: 2, minimum: 0, maximum: 10}]
    };
    store.addWorkflow(document);
    store.addWorkflow({...makeFixtureDocument(), id: 'other-workflow'});
    let changed = false;
    const unsubscribe = store.subscribe(() => {
      if (!changed && store.getState().activeWorkflowId === document.id) {
        changed = true;
        store.edit('user-limit', current => ({
          ...current,
          inputs: current.inputs.map(input => ({...input, maximum: 4}))
        }));
      }
    });
    try {
      expect(
        runWorkbenchAssistantTool(
          store,
          {name: 'setWorkflowInputs', arguments: {workflowId: document.id, inputs: {factor: 5}}},
          {approved: true}
        )
      ).toEqual({
        status: 'error',
        code: 'parameter-out-of-range',
        message: 'Factor: Value must be at most 4; got 5.'
      });
      expect(store.getState().document?.inputs[0].maximum).toBe(4);
      expect(store.getState().document?.inputs[0].value).toBe(2);
    } finally {
      unsubscribe();
      await store.clear();
    }
  });
});

describe('inactive workflow input previews', () => {
  test('returns needs-approval without activating or editing the named workflow', () => {
    const document: WorkflowDocument = {
      ...makeFixtureDocument(),
      inputs: [{id: 'factor', label: 'Factor', kind: 'number', value: 2, minimum: 0, maximum: 10}]
    };
    const other = {...makeFixtureDocument(), id: 'other-workflow'};
    const {store, calls} = makeStore({
      workflows: [document, other],
      document: other,
      activeWorkflowId: other.id
    });
    const snapshot = store.getState();
    expect(
      runWorkbenchAssistantTool(store, {
        name: 'setWorkflowInputs',
        arguments: {workflowId: document.id, inputs: {factor: 5}}
      })
    ).toEqual({
      status: 'needs-approval',
      workflowTitle: document.title,
      changes: [{inputId: 'factor', label: 'Factor', previous: 2, next: 5}]
    });
    expect(store.getState()).toBe(snapshot);
    expect(document.inputs[0].value).toBe(2);
    expect(calls).toEqual([]);
  });
});

function findRenderedControl(markup: string, tag: string, testId: string): string {
  const openingTag = markup
    .match(new RegExp(`<${tag}\\b[^>]*>`, 'g'))
    ?.find(candidate => candidate.includes(`data-testid="${testId}"`));
  if (!openingTag) throw new Error(`Missing rendered ${tag} for ${testId}`);
  return openingTag;
}

describe('dialog control accessible names', () => {
  test.each([
    ['import', WORKBENCH_TEST_IDS.importText, 'Workflow JSON to import'],
    ['json', WORKBENCH_TEST_IDS.exportText, 'Workflow JSON'],
    ['typescript', WORKBENCH_TEST_IDS.exportText, 'Workflow TypeScript'],
    ['report', WORKBENCH_TEST_IDS.runReportText, 'Run report'],
    ['dot', WORKBENCH_TEST_IDS.exportText, 'Graph (DOT)']
  ] as const)('associates the %s textarea with its visible label and explicit name', (tab, testId, label) => {
    const markup = renderToStaticMarkup(
      createElement(WorkbenchTransfer, {store: makeStore().store, initialTab: tab})
    );
    const textarea = findRenderedControl(markup, 'textarea', testId);
    const identifier = textarea.match(/\bid="([^"]+)"/)?.[1];
    expect(identifier).toBeTruthy();
    expect(markup).toContain(`<label for="${identifier}">${label}</label>`);
    expect(textarea).toContain(`aria-label="${label}"`);
    if (tab !== 'import') expect(textarea).toContain('readOnly=""');
  });

  test('names the file chooser and visible file button', () => {
    const markup = renderToStaticMarkup(
      createElement(WorkbenchTransfer, {store: makeStore().store, initialTab: 'import'})
    );
    expect(markup).toMatch(/<input[^>]*type="file"[^>]*aria-label="Choose workflow JSON file"/);
    expect(markup).toMatch(
      /<button[^>]*aria-label="Choose file: Workflow JSON"[^>]*>Choose file<\/button>/
    );
    expect(markup).toContain('>Import</button>');
  });

  test('names every built-in and user template action and outline for its template', () => {
    const userTemplate = {
      ...makeFixtureDocument(),
      id: 'saved-template',
      title: 'Saved "analysis" & comparison'
    };
    const markup = render(makeStore({dialog: 'templates', templates: [userTemplate]}).store);
    const templates = [
      {
        id: userTemplate.id,
        title: userTemplate.title,
        useTestId: WORKBENCH_TEST_IDS.userTemplate(userTemplate.id)
      },
      ...listWorkflowTemplates().map(template => ({
        id: template.id,
        title: template.title,
        useTestId: WORKBENCH_TEST_IDS.template(template.id)
      }))
    ];
    for (const template of templates) {
      const title = escapeMarkup(template.title);
      expect(findRenderedControl(markup, 'button', template.useTestId)).toContain(
        `aria-label="Use template: ${title}"`
      );
      expect(
        findRenderedControl(markup, 'button', WORKBENCH_TEST_IDS.insertWorkflow(template.id))
      ).toContain(`aria-label="Insert into current workflow: ${title}"`);
      expect(markup).toContain(
        `<summary aria-label="View outline: ${title}">View outline</summary>`
      );
    }
    expect(
      findRenderedControl(markup, 'button', WORKBENCH_TEST_IDS.userTemplateDelete(userTemplate.id))
    ).toContain(`aria-label="Delete template ${escapeMarkup(userTemplate.title)}"`);
    expect(markup).toContain('>Save current workflow as a template</button>');
    expect(markup).toContain('>Close</button>');
  });

  test('gives each history checkbox a name and an associated visible record label', () => {
    restoreCatalogInvocations([
      {
        id: 'buffer#1',
        operationId: 'geometry-point-buffer',
        invocation: '{}',
        inputs: {},
        engine: 'cpu'
      },
      {id: 'count#1', operationId: 'fixture-count', invocation: '{}', inputs: {}, engine: 'auto'}
    ]);
    try {
      const markup = renderToStaticMarkup(
        createElement(WorkbenchTransfer, {store: makeStore().store, initialTab: 'history'})
      );
      const checkboxes = markup.match(/<input[^>]*type="checkbox"[^>]*>/g) ?? [];
      expect(checkboxes).toHaveLength(2);
      for (const [index, record] of [
        {id: 'buffer#1', operationId: 'geometry-point-buffer'},
        {id: 'count#1', operationId: 'fixture-count'}
      ].entries()) {
        const checkbox = checkboxes[index];
        const identifier = checkbox.match(/\bid="([^"]+)"/)?.[1];
        expect(identifier).toBeTruthy();
        expect(markup).toContain(
          `<label for="${identifier}">${checkbox}${record.id} · ${record.operationId}</label>`
        );
        expect(checkbox).toContain(
          `aria-label="Include catalog run ${record.id}: ${record.operationId}"`
        );
      }
      expect(markup).toContain('>Create draft</button>');
    } finally {
      clearCatalogInvocations();
    }
  });
});

function expectUniqueDialogIds(markup: string): void {
  const identifiers = [...markup.matchAll(/\sid="([^"]+)"/g)].map(match => match[1]);
  expect(identifiers.length).toBeGreaterThan(0);
  expect(new Set(identifiers).size).toBe(identifiers.length);
  for (const reference of markup.matchAll(/\s(?:aria-controls|aria-labelledby|for)="([^"]+)"/g)) {
    for (const identifier of reference[1].split(/\s+/)) {
      expect(identifiers.filter(candidate => candidate === identifier)).toEqual([identifier]);
    }
  }
}

describe('dialog ids and accessibility references', () => {
  test.each([
    'templates',
    'import-export',
    'shortcuts'
  ] as const)('renders unique ids and resolvable references in %s', dialog => {
    expectUniqueDialogIds(
      render(
        makeStore({dialog, templates: [{...makeFixtureDocument(), id: 'saved-template'}]}).store
      )
    );
  });

  test.each([
    'json',
    'typescript',
    'import',
    'history',
    'report',
    'dot'
  ] as const)('keeps %s tab, panel and field ids distinct in the complete dialog', initialTab => {
    const {store} = makeStore({dialog: 'import-export'});
    restoreCatalogInvocations([
      {
        id: 'fixture-source#1',
        operationId: 'fixture-source',
        invocation: '{}',
        inputs: {},
        engine: 'auto'
      },
      {
        id: 'fixture-count#1',
        operationId: 'fixture-count',
        invocation: '{}',
        inputs: {},
        engine: 'cpu'
      }
    ]);
    try {
      const modal = findElements(renderWorkbenchDialogsTree(store)).find(
        element => typeof element.type === 'function' && element.type.name === 'Modal'
      );
      if (!modal || typeof modal.type !== 'function') throw new Error('Missing dialog shell');
      const markup = renderToStaticMarkup(
        createElement(modal.type as (properties: Record<string, unknown>) => React.JSX.Element, {
          ...modal.props,
          children: createElement(WorkbenchTransfer, {store, initialTab})
        })
      );
      expectUniqueDialogIds(markup);
      const tabs = markup.match(/<button[^>]*role="tab"[^>]*>/g) ?? [];
      const panels = markup.match(/<(?:section|div)[^>]*role="tabpanel"[^>]*>/g) ?? [];
      expect(tabs).toHaveLength(6);
      expect(panels).toHaveLength(6);
      expect(panels.filter(panel => panel.includes(' hidden=""'))).toHaveLength(5);
      for (const tab of tabs) {
        const identifier = tab.match(/\sid="([^"]+)"/)?.[1];
        const controlled = tab.match(/\saria-controls="([^"]+)"/)?.[1];
        expect(identifier).toMatch(/-tab-(json|typescript|import|history|report|dot)$/);
        const panel = panels.find(candidate => candidate.includes(` id="${controlled}"`));
        expect(panel).toContain(`aria-labelledby="${identifier}"`);
      }
    } finally {
      clearCatalogInvocations();
    }
  });
});
