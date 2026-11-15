// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {describe, expect, test} from 'vitest';
import type {FocusEvent, KeyboardEvent, PointerEvent, ReactElement, ReactNode} from 'react';
import {WorkflowRunner} from '../../../src/workflows/workflow-runner';
import type {
  WorkflowRunRecord,
  WorkflowRunnerState,
  WorkflowStepStatus
} from '../../../src/workflows/workflow-types';
import {makeFixtureDocument, makeFixtureHost, makeFixtureRegistry} from '../workflow-fixtures';
import {
  WORKBENCH_SHORTCUTS,
  WORKBENCH_TEST_IDS,
  type WorkbenchShortcutAction
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-contract';
import {
  WorkbenchCanvas,
  type WorkbenchCanvasApi
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-canvas';
import {
  WorkbenchDock,
  applyWorkbenchShortcut,
  clampWorkbenchHeight,
  formatWorkbenchAnnouncement,
  formatWorkbenchSummary,
  shouldHandleTabAsAddStep,
  subscribeWorkbenchPaletteClose,
  truncateWorkbenchTitle
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-dock';
import {
  WorkbenchToolbar,
  WorkbenchDockHeightContext,
  getWorkbenchRunControls,
  getWorkbenchPlatform,
  getWorkbenchHeightToggle,
  isWorkbenchToolbarNarrow
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-toolbar';
import {
  WorkbenchRunLog,
  formatWorkbenchDuration,
  formatWorkbenchRunLine
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-run-log';
import {
  formatWorkbenchShortcut,
  isEditableTarget,
  matchWorkbenchShortcut
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-shortcuts';
import type {
  WorkbenchState,
  WorkbenchStore
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-store';

const hostRequire = createRequire(
  new URL('../../../../../examples/deck/kepler-compute/package.json', import.meta.url)
);
const {Children, cloneElement, createElement, isValidElement} = hostRequire(
  'react'
) as typeof import('react');
const {renderToStaticMarkup} = hostRequire('react-dom/server') as typeof import('react-dom/server');
const {ServerStyleSheet} = hostRequire('styled-components') as typeof import('styled-components');

function makeRunner(
  statuses: WorkflowStepStatus[] = ['ready', 'stale', 'done']
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

function makeStore(patch: Partial<WorkbenchState> = {}) {
  const document = makeFixtureDocument();
  let state: WorkbenchState = {
    revision: 1,
    workflows: [document],
    document,
    activeWorkflowId: document.id,
    runner: makeRunner(),
    selectedStepIds: ['scale'],
    canUndo: true,
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
  const calls: unknown[][] = [];
  const store: WorkbenchStore = {
    registry: makeFixtureRegistry(),
    host: makeFixtureHost(),
    getState: () => state,
    subscribe: () => () => {},
    createWorkflow: options => {
      calls.push(['create', options]);
      return 'new';
    },
    addWorkflow: next => {
      calls.push(['add', next]);
      return next.id;
    },
    duplicateWorkflow: workflowId => {
      calls.push(['duplicate-workflow', workflowId]);
      return 'copy';
    },
    deleteWorkflow: workflowId => {
      calls.push(['delete-workflow', workflowId]);
    },
    setActiveWorkflow: workflowId => {
      calls.push(['active', workflowId]);
    },
    edit: (tag, change) => {
      calls.push(['edit', tag]);
      if (!state.document) return false;
      state = {...state, document: change(state.document)};
      return true;
    },
    undo: () => {
      calls.push(['undo']);
    },
    redo: () => {
      calls.push(['redo']);
    },
    select: stepIds => {
      calls.push(['select', stepIds]);
      state = {...state, selectedStepIds: stepIds};
    },
    run: async options => {
      calls.push(['run', options]);
      return null;
    },
    cancel: () => {
      calls.push(['cancel']);
    },
    notifyExternalChange: () => {},
    getRunner: () => null,
    setDock: next => {
      calls.push(['dock', next]);
    },
    openPalette: options => {
      calls.push(['open-palette', options]);
    },
    closePalette: () => {
      calls.push(['close-palette']);
    },
    setDialog: dialog => {
      calls.push(['dialog', dialog]);
    },
    setNotice: () => {},
    proposeDraft: () => {},
    acceptDraft: () => null,
    rejectDraft: () => {},
    saveTemplate: workflowId => {
      calls.push(['save-template', workflowId]);
      return 'saved-template';
    },
    deleteTemplate: () => {},
    useTemplate: () => '',
    insertWorkflow: () => false,
    copySelection: () => {
      calls.push(['copy']);
      return true;
    },
    cutSelection: () => {
      calls.push(['cut']);
      return true;
    },
    paste: position => {
      calls.push(['paste', position]);
      return true;
    },
    serialize: () => undefined,
    restore: () => {},
    clear: async () => {}
  };
  return {store, calls};
}

function makeRun(): WorkflowRunRecord {
  return {
    schemaVersion: 1,
    runId: 'fixture-workflow-3',
    workflowId: 'fixture-workflow',
    startedAt: 0,
    durationMilliseconds: 1400,
    status: 'error',
    steps: [
      {
        stepId: 'source',
        operationId: 'fixture-source',
        key: 'source',
        status: 'done',
        cacheHit: true,
        engine: 'host',
        durationMilliseconds: 0,
        outputRows: {data: 1204},
        inputs: []
      },
      {
        stepId: 'scale',
        operationId: 'fixture-scale',
        key: 'scale',
        status: 'error',
        cacheHit: false,
        engine: 'cpu-reference',
        durationMilliseconds: 42,
        inputs: [],
        fallbackReason: 'No GPU device supplied',
        error: {
          code: 'operation-failed',
          severity: 'error',
          message: 'Fixture scale: asked to fail'
        }
      }
    ]
  };
}

function render(component: typeof WorkbenchToolbar, patch: Partial<WorkbenchState> = {}) {
  return renderToStaticMarkup(createElement(component, {store: makeStore(patch).store}));
}
function control(markup: string, testId: string): string {
  const opening = markup.match(new RegExp(`<[^>]+data-testid="${testId}"[^>]*>`))?.[0];
  expect(opening, testId).toBeDefined();
  return opening!;
}

const shortcutCases: Array<[string, WorkbenchShortcutAction]> = [
  ['Tab', 'add-step'],
  ['N', 'add-step'],
  ['Enter', 'run-to-here'],
  ['Shift+Enter', 'run-all'],
  ['Escape', 'cancel-or-close'],
  ['B', 'toggle-disabled'],
  ['D', 'toggle-display'],
  ['F', 'frame'],
  ['L', 'layout'],
  ['Delete', 'delete'],
  ['Mod+D', 'duplicate'],
  ['Mod+Z', 'undo'],
  ['Shift+Mod+Z', 'redo'],
  ['Mod+A', 'select-all'],
  ['Mod+C', 'copy'],
  ['Mod+X', 'cut'],
  ['Mod+V', 'paste']
];
function makeKey(keys: string, platform: 'mac' | 'other') {
  const parts = keys.split('+');
  return {
    key: parts.at(-1)!,
    shiftKey: parts.includes('Shift'),
    metaKey: parts.includes('Mod') && platform === 'mac',
    ctrlKey: parts.includes('Mod') && platform === 'other',
    altKey: false
  };
}

describe('workbench shortcut vocabulary', () => {
  test('covers exactly the contract', () => {
    expect(WORKBENCH_SHORTCUTS.map(({keys, action}) => [keys, action])).toEqual(shortcutCases);
  });
  for (const platform of ['mac', 'other'] as const) {
    for (const [keys, action] of shortcutCases) {
      test(`${platform}: ${keys} matches ${action} with exact modifiers`, () => {
        const event = makeKey(keys, platform);
        expect(matchWorkbenchShortcut(event, platform)).toBe(action);
        expect(matchWorkbenchShortcut({...event, key: event.key.toLowerCase()}, platform)).toBe(
          action
        );
        expect(matchWorkbenchShortcut({...event, altKey: true}, platform)).toBeNull();
        expect(
          matchWorkbenchShortcut({...event, metaKey: true, ctrlKey: true}, platform)
        ).toBeNull();
        if (!keys.includes('Mod')) {
          expect(matchWorkbenchShortcut({...event, metaKey: true}, platform)).toBe(
            keys === 'D' && platform === 'mac' ? 'duplicate' : null
          );
          expect(matchWorkbenchShortcut({...event, ctrlKey: true}, platform)).toBe(
            keys === 'D' && platform === 'other' ? 'duplicate' : null
          );
        } else {
          expect(
            matchWorkbenchShortcut(
              {...event, metaKey: !event.metaKey, ctrlKey: !event.ctrlKey},
              platform
            )
          ).toBeNull();
        }
      });
    }
    test(`${platform}: excludes extra Shift and unknown keys; aliases Backspace`, () => {
      expect(matchWorkbenchShortcut(makeKey('Shift+B', platform), platform)).toBeNull();
      expect(matchWorkbenchShortcut(makeKey('Shift+Delete', platform), platform)).toBeNull();
      expect(matchWorkbenchShortcut(makeKey('Shift+Tab', platform), platform)).toBeNull();
      expect(matchWorkbenchShortcut(makeKey('Mod+Enter', platform), platform)).toBeNull();
      expect(matchWorkbenchShortcut(makeKey('Q', platform), platform)).toBeNull();
      expect(matchWorkbenchShortcut(makeKey('Backspace', platform), platform)).toBe('delete');
    });
  }
  test('formats platform modifiers and special keys exactly', () => {
    expect(formatWorkbenchShortcut('Mod+Z', 'mac')).toBe('⌘Z');
    expect(formatWorkbenchShortcut('Mod+Z', 'other')).toBe('Ctrl+Z');
    expect(formatWorkbenchShortcut('Shift+Mod+Z', 'mac')).toBe('⇧⌘Z');
    expect(formatWorkbenchShortcut('Shift+Mod+Z', 'other')).toBe('Shift+Ctrl+Z');
    expect(formatWorkbenchShortcut('Shift+Enter', 'mac')).toBe('⇧↵');
    expect(formatWorkbenchShortcut('Tab', 'mac')).toBe('⇥');
    expect(formatWorkbenchShortcut('Escape', 'mac')).toBe('Esc');
    expect(formatWorkbenchShortcut('Delete', 'mac')).toBe('⌫');
  });
  test('recognizes editable ancestors without a DOM global', () => {
    for (const tagName of ['INPUT', 'textarea', 'SELECT'])
      expect(isEditableTarget({tagName})).toBe(true);
    expect(isEditableTarget({tagName: 'SPAN', isContentEditable: true})).toBe(true);
    expect(isEditableTarget({parentElement: {getAttribute: () => ''}})).toBe(true);
    expect(isEditableTarget({getAttribute: () => 'plaintext-only'})).toBe(true);
    expect(
      isEditableTarget({getAttribute: () => 'false', parentElement: {isContentEditable: true}})
    ).toBe(false);
    for (const target of [null, undefined, 'input', {}, {tagName: 'BUTTON'}])
      expect(isEditableTarget(target)).toBe(false);
  });
});

describe('workbench dock and controls', () => {
  test('renders a collapsed empty-workspace bar with named controls', () => {
    const markup = render(WorkbenchDock, {
      document: null,
      workflows: [],
      activeWorkflowId: null,
      runner: null
    });
    expect(control(markup, WORKBENCH_TEST_IDS.dock)).toContain('aria-label="Analysis workbench"');
    expect(control(markup, WORKBENCH_TEST_IDS.dockToggle)).toContain(
      'aria-label="Expand workbench"'
    );
    expect(control(markup, WORKBENCH_TEST_IDS.runAll)).toContain('disabled');
    expect(control(markup, WORKBENCH_TEST_IDS.cancel)).toContain('disabled');
    expect(control(markup, WORKBENCH_TEST_IDS.workflowNew)).not.toContain('disabled');
    expect(markup).toContain('No workflow');
    expect(markup).not.toContain(`data-testid="${WORKBENCH_TEST_IDS.toolbar}"`);
  });
  test('counts mixed statuses in deterministic words', () => {
    const state = makeStore({runner: makeRunner(['done', 'stale', 'error'])}).store.getState();
    expect(formatWorkbenchSummary(state)).toBe('1 done · 1 stale · 1 error');
    expect(render(WorkbenchDock, {runner: state.runner})).toContain('1 done · 1 stale · 1 error');
    expect(formatWorkbenchSummary({...state, document: null})).toBe('No workflow');
    expect(formatWorkbenchSummary({...state, document: {...state.document!, steps: []}})).toBe(
      'No steps'
    );
    expect(formatWorkbenchSummary({...state, runner: null})).toBe('3 not configured');
  });
  test('renders open shell, named resize separator, notice and run log', () => {
    const markup = render(WorkbenchDock, {
      dock: {open: true, height: 400, inspectorTab: 'parameters', logOpen: false},
      notice: {kind: 'error', message: 'Fixture scale: check Factor.'}
    });
    expect(control(markup, WORKBENCH_TEST_IDS.dockResize)).toContain('role="separator"');
    expect(control(markup, WORKBENCH_TEST_IDS.dockResize)).toContain('aria-valuenow="400"');
    expect(control(markup, WORKBENCH_TEST_IDS.dockResize)).toContain(
      'aria-orientation="horizontal"'
    );
    expect(control(markup, WORKBENCH_TEST_IDS.toolbar)).toContain('aria-label="Workflow"');
    expect(control(markup, WORKBENCH_TEST_IDS.notice)).toContain('role="status"');
    expect(markup).toContain('Fixture scale: check Factor.');
    expect(markup).toContain('aria-label="Dismiss"');
    expect(markup).toContain('aria-live="polite"');
    expect(control(markup, WORKBENCH_TEST_IDS.runLogToggle)).toContain('aria-expanded="false"');
  });
  test('renders dock controls in natural keyboard order and advertises N for Add step', () => {
    const markup = render(WorkbenchDock, {
      dock: {open: true, height: 400, inspectorTab: 'parameters', logOpen: false}
    });
    const positions = [
      WORKBENCH_TEST_IDS.dockResize,
      WORKBENCH_TEST_IDS.toolbar,
      WORKBENCH_TEST_IDS.canvas,
      WORKBENCH_TEST_IDS.inspector,
      WORKBENCH_TEST_IDS.runLogToggle
    ].map(testId => markup.indexOf(control(markup, testId)));
    expect(positions).toEqual([...positions].sort((first, second) => first - second));
    expect(control(markup, WORKBENCH_TEST_IDS.dockResize)).toContain('tabindex="0"');
    expect(control(markup, WORKBENCH_TEST_IDS.canvas)).toContain('tabindex="0"');
    expect(control(markup, WORKBENCH_TEST_IDS.inspectorTab('parameters'))).toContain(
      'tabindex="0"'
    );
    expect(control(markup, WORKBENCH_TEST_IDS.addStep)).toContain('title="Add step (N)"');
  });
  test('renders empty workflow, unknown operation, and step errors without a runner', () => {
    const document = makeFixtureDocument();
    document.steps[1] = {...document.steps[1], operationId: 'removed-operation'};
    const dock = {open: true, height: 380, inspectorTab: 'parameters' as const, logOpen: true};
    expect(render(WorkbenchDock, {document, runner: null, dock})).toContain('Analysis workbench');
    expect(
      render(WorkbenchDock, {document: {...document, steps: []}, runner: null, dock})
    ).toContain('No runs yet');
    const runner = makeRunner(['done', 'error', 'blocked']);
    runner.steps['scale'].issues = [
      {severity: 'error', code: 'input-required', message: 'Fixture scale needs Points.'}
    ];
    expect(render(WorkbenchDock, {document, runner, dock})).toContain('Workflow');
  });
  test('only enables creation, templates and import with no document', () => {
    const markup = render(WorkbenchToolbar, {
      document: null,
      workflows: [],
      activeWorkflowId: null,
      runner: null
    });
    for (const testId of [
      WORKBENCH_TEST_IDS.workflowNew,
      WORKBENCH_TEST_IDS.templates,
      WORKBENCH_TEST_IDS.importExport
    ])
      expect(control(markup, testId)).not.toContain('disabled');
    for (const testId of [
      WORKBENCH_TEST_IDS.workflowSelect,
      WORKBENCH_TEST_IDS.workflowDuplicate,
      WORKBENCH_TEST_IDS.workflowDelete,
      WORKBENCH_TEST_IDS.runAll,
      WORKBENCH_TEST_IDS.runToHere,
      WORKBENCH_TEST_IDS.undo,
      WORKBENCH_TEST_IDS.redo,
      WORKBENCH_TEST_IDS.addStep,
      WORKBENCH_TEST_IDS.layout,
      WORKBENCH_TEST_IDS.engine,
      WORKBENCH_TEST_IDS.autoRun
    ])
      expect(control(markup, testId)).toContain('disabled');
  });
  test('enables runnable selection and undo independently at idle', () => {
    const markup = render(WorkbenchToolbar);
    expect(control(markup, WORKBENCH_TEST_IDS.runAll)).not.toContain('disabled');
    expect(markup).toContain('Run (2)');
    expect(control(markup, WORKBENCH_TEST_IDS.runToHere)).not.toContain('disabled');
    expect(control(markup, WORKBENCH_TEST_IDS.undo)).not.toContain('disabled');
    expect(control(markup, WORKBENCH_TEST_IDS.redo)).toContain('disabled');
    expect(markup).not.toContain(`data-testid="${WORKBENCH_TEST_IDS.cancel}"`);
    expect(markup).toContain('No GPU</span>');
    expect(markup).toMatch(/>GPU<\/option>/);
    expect(markup).toMatch(/>CPU<\/option>/);
    const state = makeStore().store.getState();
    expect(getWorkbenchRunControls(state)).toEqual({count: 2, canRun: true, target: 'scale'});
    for (const selectedStepIds of [[], ['scale', 'source'], ['missing']])
      expect(getWorkbenchRunControls({...state, selectedStepIds}).target).toBeNull();
    expect(
      getWorkbenchRunControls({
        ...state,
        runner: makeRunner(['disabled', 'blocked', 'not-configured'])
      })
    ).toEqual({count: 0, canRun: false, target: null});
  });
  test('disables run controls and enables cancellation during a run', () => {
    const runner = {...makeRunner(['running', 'queued', 'queued']), running: true};
    const markup = render(WorkbenchToolbar, {runner});
    expect(control(markup, WORKBENCH_TEST_IDS.runAll)).toContain('disabled');
    expect(control(markup, WORKBENCH_TEST_IDS.runToHere)).toContain('disabled');
    expect(control(markup, WORKBENCH_TEST_IDS.cancel)).not.toContain('disabled');
    expect(formatWorkbenchAnnouncement(makeStore({runner}).store.getState())).toBe(
      'Run started. Running workflow.'
    );
    expect(
      formatWorkbenchAnnouncement(
        makeStore({runner: {...runner, runningStepId: 'scale'}}).store.getState(),
        makeFixtureRegistry()
      )
    ).toBe('Running Fixture scale.');
  });
  test('excludes completed steps from the run count and refuses an unconfigured target', () => {
    const state = makeStore({runner: makeRunner(['done', 'done', 'done'])}).store.getState();
    expect(getWorkbenchRunControls(state)).toEqual({count: 0, canRun: false, target: 'scale'});
    expect(
      control(render(WorkbenchToolbar, {runner: state.runner}), WORKBENCH_TEST_IDS.runAll)
    ).toContain('disabled');
    const unconfigured = makeStore({runner: makeRunner(['ready', 'not-configured', 'blocked'])});
    applyWorkbenchShortcut('run-to-here', unconfigured.store, null);
    expect(unconfigured.calls).toEqual([]);
  });
  test('clamps height for large and short viewports', () => {
    expect(clampWorkbenchHeight(380, 1000)).toBe(380);
    expect(clampWorkbenchHeight(100, 1000)).toBe(220);
    expect(clampWorkbenchHeight(900, 1000)).toBe(700);
    expect(clampWorkbenchHeight(380, 200)).toBe(140);
  });
});

describe('workbench run log', () => {
  test('formats durations and complete run lines', () => {
    expect(formatWorkbenchDuration(42)).toBe('42 ms');
    expect(formatWorkbenchDuration(999)).toBe('999 ms');
    expect(formatWorkbenchDuration(1800)).toBe('1.8 s');
    expect(formatWorkbenchRunLine(makeRun())).toBe('Run 3 · error · 2 steps · 1.4 s · 1 cached');
  });
  test('renders cached and failed steps, counts, engines and reasons', () => {
    const run = makeRun();
    const markup = render(WorkbenchRunLog, {
      runLog: [run],
      dock: {open: true, height: 380, inspectorTab: 'parameters', logOpen: true}
    });
    expect(control(markup, WORKBENCH_TEST_IDS.runLogToggle)).toContain('aria-expanded="true"');
    expect(control(markup, WORKBENCH_TEST_IDS.runLog)).toContain('aria-label="Run log"');
    for (const text of [
      'Run 3 · error · 2 steps · 1.4 s · 1 cached',
      'Fixture source',
      'Fixture scale',
      'Host',
      'CPU',
      'Cached',
      '1,204',
      'rows',
      '42 ms',
      'No GPU device supplied',
      'Fixture scale: asked to fail',
      'data-status="done"',
      'data-status="error"'
    ])
      expect(markup).toContain(text);
    const state = makeStore({runLog: [run]}).store.getState();
    expect(formatWorkbenchAnnouncement(state)).toBe('Run failed: Fixture scale: asked to fail');
    expect(formatWorkbenchAnnouncement({...state, runLog: [{...run, status: 'done'}]})).toBe(
      'Run finished: 1 steps done in 1.4 s'
    );
    expect(formatWorkbenchAnnouncement({...state, runLog: [{...run, status: 'cancelled'}]})).toBe(
      'Run cancelled.'
    );
  });
});

describe('dock shortcut dispatch', () => {
  test('opens at canvas centre, frames selection and frames the whole graph', () => {
    const {store, calls} = makeStore();
    const framed: unknown[] = [];
    const api = {
      frame: (stepIds?: readonly string[]) => {
        framed.push(stepIds);
      },
      getViewportCenter: () => ({x: 70, y: 80})
    };
    applyWorkbenchShortcut('add-step', store, api);
    expect(calls).toEqual([['open-palette', {position: {x: 70, y: 80}}]]);
    applyWorkbenchShortcut('frame', store, api);
    store.select([]);
    applyWorkbenchShortcut('frame', store, api);
    expect(framed).toEqual([['scale'], undefined]);
  });
  test('runs to one selected runnable step and runs all; refuses while running', () => {
    const {store, calls} = makeStore();
    applyWorkbenchShortcut('run-to-here', store, null);
    applyWorkbenchShortcut('run-all', store, null);
    expect(calls).toEqual([
      ['run', {target: 'scale'}],
      ['run', undefined]
    ]);
    const running = makeStore({runner: {...makeRunner(), running: true}});
    applyWorkbenchShortcut('run-to-here', running.store, null);
    applyWorkbenchShortcut('run-all', running.store, null);
    expect(running.calls).toEqual([]);
  });
  test('closes dialogs, then palette, then cancels runs; idle Escape does nothing', () => {
    for (const [patch, expected] of [
      [{dialog: 'shortcuts', palette: {open: true}}, ['dialog', null]],
      [{palette: {open: true}, runner: {...makeRunner(), running: true}}, ['close-palette']],
      [{runner: {...makeRunner(), running: true}}, ['cancel']]
    ] as Array<[Partial<WorkbenchState>, unknown[]]>) {
      const {store, calls} = makeStore(patch);
      applyWorkbenchShortcut('cancel-or-close', store, null);
      expect(calls).toEqual([expected]);
    }
    const idle = makeStore();
    applyWorkbenchShortcut('cancel-or-close', idle.store, null);
    expect(idle.calls).toEqual([]);
  });
  test('toggles every selected disabled flag and first output without dropping other outputs', () => {
    const {store} = makeStore({selectedStepIds: ['source', 'scale']});
    applyWorkbenchShortcut('toggle-disabled', store, null);
    expect(store.getState().document!.steps.map(step => step.disabled)).toEqual([
      true,
      true,
      undefined
    ]);
    applyWorkbenchShortcut('toggle-disabled', store, null);
    expect(store.getState().document!.steps.map(step => step.disabled)).toEqual([
      false,
      false,
      undefined
    ]);
    applyWorkbenchShortcut('toggle-display', store, null);
    expect(store.getState().document!.steps.map(step => step.display)).toEqual([
      ['data'],
      undefined,
      undefined
    ]);
    applyWorkbenchShortcut('toggle-display', store, null);
    expect(store.getState().document!.steps.map(step => step.display)).toEqual([
      undefined,
      ['scaled'],
      undefined
    ]);
  });
  test('duplicates and selects copies, removes edges on delete, and lays out via edits', () => {
    const {store, calls} = makeStore();
    applyWorkbenchShortcut('duplicate', store, null);
    expect(store.getState().selectedStepIds).toEqual(['scale-2']);
    expect(store.getState().document!.steps.map(step => step.id)).toEqual([
      'source',
      'scale',
      'count',
      'scale-2'
    ]);
    store.select(['scale']);
    applyWorkbenchShortcut('delete', store, null);
    expect(store.getState().document!.steps.find(step => step.id === 'count')!.inputs).toEqual({});
    applyWorkbenchShortcut('layout', store, null);
    expect(calls.at(-1)).toEqual(['edit', 'layout']);
    expect(store.getState().document!.steps.find(step => step.id === 'source')!.position).toEqual({
      x: 0,
      y: 150
    });
  });
  test('respects history availability, selects all, and safely ignores actions without a document', () => {
    const {store, calls} = makeStore();
    applyWorkbenchShortcut('undo', store, null);
    applyWorkbenchShortcut('redo', store, null);
    applyWorkbenchShortcut('select-all', store, null);
    expect(calls).toEqual([['undo'], ['select', ['source', 'scale', 'count']]]);
    const redo = makeStore({canUndo: false, canRedo: true});
    applyWorkbenchShortcut('undo', redo.store, null);
    applyWorkbenchShortcut('redo', redo.store, null);
    expect(redo.calls).toEqual([['redo']]);
    const empty = makeStore({document: null, runner: null});
    for (const [, action] of shortcutCases) applyWorkbenchShortcut(action, empty.store, null);
    expect(empty.calls).toEqual([]);
  });
});

// Render within React's server dispatcher, then invoke the actual listener attached to the dock.
type DockHandlers = {
  onKeyDown: (event: KeyboardEvent<HTMLDivElement>) => void;
  onPointerDownCapture: (event: PointerEvent<HTMLDivElement>) => void;
  onBlurCapture: (event: FocusEvent<HTMLDivElement>) => void;
};

function captureDockHandlers(store: WorkbenchStore) {
  let handlers: DockHandlers | undefined;
  function CaptureDock() {
    const provider = WorkbenchDock({store}) as ReactElement<{
      children: ReactElement<DockHandlers>;
    }>;
    handlers = provider.props.children.props;
    return provider;
  }
  renderToStaticMarkup(createElement(CaptureDock));
  if (!handlers) throw new Error('Dock did not attach its event listeners.');
  return handlers;
}

function captureDockKeyHandler(store: WorkbenchStore) {
  return captureDockHandlers(store).onKeyDown;
}

function makeDockTarget(tagName: string, targetInsideCanvas: boolean) {
  return {
    tagName,
    closest: (selector: string) =>
      selector === `[data-testid="${WORKBENCH_TEST_IDS.canvas}"]`
        ? targetInsideCanvas
          ? {}
          : null
        : ['BUTTON', 'SUMMARY', 'A'].includes(tagName)
          ? {}
          : null,
    // Chromium can keep focus-visible even after a pointer interaction.
    matches: (selector: string) => selector === ':focus-visible'
  };
}

function sendDockKey(
  store: WorkbenchStore,
  keys: string,
  tagName = 'BUTTON',
  defaultPrevented = false,
  focus = {targetInsideCanvas: false, canvasPointerFocused: false}
) {
  const handlers = captureDockHandlers(store);
  if (focus.canvasPointerFocused)
    handlers.onPointerDownCapture({
      target: makeDockTarget('DIV', true)
    } as unknown as PointerEvent<HTMLDivElement>);
  return sendCapturedDockKey(
    handlers,
    keys,
    makeDockTarget(tagName, focus.targetInsideCanvas),
    defaultPrevented
  );
}

function sendCapturedDockKey(
  handlers: DockHandlers,
  keys: string,
  target = makeDockTarget('DIV', true),
  defaultPrevented = false
) {
  let prevented = 0;
  handlers.onKeyDown({
    ...makeKey(keys, getWorkbenchPlatform()),
    target,
    defaultPrevented,
    preventDefault: () => {
      prevented++;
    }
  } as unknown as KeyboardEvent<HTMLDivElement>);
  return prevented;
}

function readClassRules(opening: string, styles: string): string {
  const classNames = opening.match(/class="([^"]+)"/)?.[1].split(' ') ?? [];
  return classNames
    .map(className => styles.match(new RegExp(`\\.${className}\\{([^}]+)\\}`))?.[1] ?? '')
    .join('');
}

describe('independent review regressions', () => {
  test('Tab as Add step requires both a canvas target and explicit pointer focus', () => {
    expect(shouldHandleTabAsAddStep({targetInsideCanvas: true, canvasPointerFocused: true})).toBe(
      true
    );
    expect(shouldHandleTabAsAddStep({targetInsideCanvas: true, canvasPointerFocused: false})).toBe(
      false
    );
    expect(shouldHandleTabAsAddStep({targetInsideCanvas: false, canvasPointerFocused: true})).toBe(
      false
    );
    expect(shouldHandleTabAsAddStep({targetInsideCanvas: false, canvasPointerFocused: false})).toBe(
      false
    );
  });
  test('F1: root listener handles B, Delete and Mod+Z on ordinary controls', () => {
    for (const tagName of ['BUTTON', 'SUMMARY', 'A']) {
      for (const [keys, expected] of [
        ['B', ['edit', 'toggle-disabled:scale']],
        ['Delete', ['edit', 'delete:scale']],
        ['Mod+Z', ['undo']]
      ] as const) {
        const {store, calls} = makeStore();
        expect(sendDockKey(store, keys, tagName)).toBe(1);
        expect(calls).toEqual([expected]);
      }
    }
  });
  test('F1: root listener preserves native activation and traversal, plus editable and dialog exclusions', () => {
    for (const keys of ['Enter', 'Shift+Enter', 'Tab', 'Shift+Tab', ' ']) {
      const {store, calls} = makeStore();
      expect(sendDockKey(store, keys)).toBe(0);
      expect(calls).toEqual([]);
    }
    for (const tagName of ['INPUT', 'TEXTAREA', 'SELECT']) {
      const {store, calls} = makeStore();
      expect(sendDockKey(store, 'B', tagName)).toBe(0);
      expect(calls).toEqual([]);
    }
    const dialog = makeStore({dialog: 'shortcuts'});
    expect(sendDockKey(dialog.store, 'B')).toBe(0);
    expect(dialog.calls).toEqual([]);
    expect(sendDockKey(dialog.store, 'Escape')).toBe(1);
    expect(dialog.calls).toEqual([['dialog', null]]);
    const handled = makeStore();
    expect(sendDockKey(handled.store, 'B', 'BUTTON', true)).toBe(0);
    expect(handled.calls).toEqual([]);
    const canvas = makeStore();
    expect(sendDockKey(canvas.store, 'N', 'DIV')).toBe(1);
    expect(canvas.calls).toEqual([['open-palette', {position: undefined}]]);
  });
  test('canvas Tab opens the palette only with pointer focus and never intercepts Shift+Tab', () => {
    for (const targetInsideCanvas of [false, true]) {
      for (const canvasPointerFocused of [false, true]) {
        const focus = {targetInsideCanvas, canvasPointerFocused};
        const {store, calls} = makeStore();
        const shouldOpen = targetInsideCanvas && canvasPointerFocused;
        expect(sendDockKey(store, 'Tab', 'DIV', false, focus)).toBe(shouldOpen ? 1 : 0);
        expect(calls).toEqual(shouldOpen ? [['open-palette', {position: undefined}]] : []);
        const backwards = makeStore();
        expect(sendDockKey(backwards.store, 'Shift+Tab', 'DIV', false, focus)).toBe(0);
        expect(backwards.calls).toEqual([]);
      }
    }
  });
  test('capture-phase pointer focus survives internal focus movement but clears when focus leaves the canvas', () => {
    const {store, calls} = makeStore();
    const handlers = captureDockHandlers(store);
    const target = makeDockTarget('DIV', true);
    expect(target.matches(':focus-visible')).toBe(true);
    expect(sendCapturedDockKey(handlers, 'Tab')).toBe(0);
    handlers.onPointerDownCapture({target} as unknown as PointerEvent<HTMLDivElement>);
    const inside = {};
    const canvas = {contains: (relatedTarget: unknown) => relatedTarget === inside};
    const blurTarget = {closest: () => canvas};
    handlers.onBlurCapture({
      target: blurTarget,
      relatedTarget: inside
    } as unknown as FocusEvent<HTMLDivElement>);
    expect(sendCapturedDockKey(handlers, 'Tab')).toBe(1);
    expect(calls).toEqual([['open-palette', {position: undefined}]]);
    for (const relatedTarget of [{}, null]) {
      handlers.onPointerDownCapture({target} as unknown as PointerEvent<HTMLDivElement>);
      handlers.onBlurCapture({
        target: blurTarget,
        relatedTarget
      } as unknown as FocusEvent<HTMLDivElement>);
      expect(sendCapturedDockKey(handlers, 'Tab')).toBe(0);
    }
    handlers.onPointerDownCapture({
      target: makeDockTarget('DIV', false)
    } as unknown as PointerEvent<HTMLDivElement>);
    expect(sendCapturedDockKey(handlers, 'Tab')).toBe(0);
    expect(calls).toHaveLength(1);
  });
  test('every unhandled Tab clears pointer focus before keyboard navigation returns to the canvas', () => {
    for (const [keys, target, defaultPrevented] of [
      ['Shift+Tab', makeDockTarget('DIV', true), false],
      ['Mod+Tab', makeDockTarget('DIV', true), false],
      ['Tab', makeDockTarget('DIV', true), true],
      ['Tab', makeDockTarget('DIV', false), false],
      ['Tab', makeDockTarget('BUTTON', true), false],
      ['Tab', makeDockTarget('INPUT', true), false]
    ] as const) {
      const {store, calls} = makeStore();
      const handlers = captureDockHandlers(store);
      handlers.onPointerDownCapture({
        target: makeDockTarget('DIV', true)
      } as unknown as PointerEvent<HTMLDivElement>);
      expect(sendCapturedDockKey(handlers, keys, target, defaultPrevented)).toBe(0);
      expect(sendCapturedDockKey(handlers, 'Tab')).toBe(0);
      expect(calls).toEqual([]);
    }
  });
  test('palette-close subscription resets modality only on an open-to-closed transition and unsubscribes', () => {
    const fixture = makeStore();
    let state = fixture.store.getState();
    const listeners = new Set<() => void>();
    const store: WorkbenchStore = {
      ...fixture.store,
      getState: () => state,
      subscribe: listener => {
        listeners.add(listener);
        return () => listeners.delete(listener);
      }
    };
    let resets = 0;
    const unsubscribe = subscribeWorkbenchPaletteClose(store, () => resets++);
    function setPaletteOpen(open: boolean) {
      state = {...state, palette: {open}};
      for (const listener of listeners) listener();
    }
    setPaletteOpen(false);
    setPaletteOpen(true);
    expect(resets).toBe(0);
    setPaletteOpen(false);
    setPaletteOpen(false);
    expect(resets).toBe(1);
    setPaletteOpen(true);
    setPaletteOpen(false);
    expect(resets).toBe(2);
    unsubscribe();
    expect(listeners.size).toBe(0);
    setPaletteOpen(true);
    setPaletteOpen(false);
    expect(resets).toBe(2);
  });
  test('N opens the palette from ordinary controls but leaves editable fields and dialogs alone', () => {
    for (const tagName of ['BUTTON', 'DIV']) {
      const {store, calls} = makeStore();
      expect(sendDockKey(store, 'n', tagName)).toBe(1);
      expect(calls).toEqual([['open-palette', {position: undefined}]]);
    }
    for (const tagName of ['INPUT', 'TEXTAREA', 'SELECT']) {
      const {store, calls} = makeStore();
      expect(sendDockKey(store, 'N', tagName)).toBe(0);
      expect(calls).toEqual([]);
    }
    const dialog = makeStore({dialog: 'shortcuts'});
    expect(sendDockKey(dialog.store, 'N', 'DIV')).toBe(0);
    expect(dialog.calls).toEqual([]);
  });
  test('F2: targeted and partially completed real runs announce only the executing step', async () => {
    const document = makeFixtureDocument();
    document.steps[0] = {...document.steps[0], label: 'Read points'};
    document.steps[1] = {...document.steps[1], label: 'Scale points'};
    document.steps.unshift({
      ...document.steps[0],
      id: 'unrelated',
      label: 'Unrelated points',
      position: {x: -280, y: 0}
    });
    const registry = makeFixtureRegistry();
    const runner = new WorkflowRunner({registry, host: makeFixtureHost(), now: () => 0});
    const base = makeStore({document}).store.getState();
    const announcements = new Set<string>();
    const unsubscribe = runner.subscribe(() => {
      const state = runner.getState();
      if (state.running && state.runningStepId)
        announcements.add(formatWorkbenchAnnouncement({...base, runner: state}));
    });
    try {
      runner.setDocument(document);
      const first = await runner.run({target: 'source'});
      expect(first.steps.map(step => step.stepId)).toEqual(['source']);
      expect([...announcements]).toEqual(['Running Read points.']);
      announcements.clear();
      const second = await runner.run({target: 'scale'});
      expect(second.steps.map(step => step.stepId)).toEqual(['scale']);
      expect([...announcements]).toEqual(['Running Scale points.']);
    } finally {
      unsubscribe();
      await runner.destroy();
    }
  });
  test('F3: retained runs keep their runner identities after the twentieth run', () => {
    function retained(latest: number) {
      return Array.from({length: 20}, (_, index) => ({
        ...makeRun(),
        runId: `fixture-workflow-${latest - index}`,
        status: 'done' as const,
        steps: []
      }));
    }
    const dock = {open: true, height: 220, inspectorTab: 'parameters' as const, logOpen: true};
    const before = render(WorkbenchRunLog, {dock, runLog: retained(20)});
    const after = render(WorkbenchRunLog, {dock, runLog: retained(21)});
    expect(before).toContain('Run 20 · done · 0 steps · 1.4 s · 0 cached');
    expect(after).toContain('Run 21 · done · 0 steps · 1.4 s · 0 cached');
    expect(after).toContain('Run 20 · done · 0 steps · 1.4 s · 0 cached');
    expect(after).not.toContain('Run 1 ·');
    expect(
      render(WorkbenchRunLog, {dock, runLog: [{...makeRun(), runId: 'opaque-run-id'}]})
    ).toContain('Run opaque-run-id ·');
  });
  test('F3: cumulative labels agree with real runner records after retention', async () => {
    const document = {...makeFixtureDocument(), steps: []};
    const runner = new WorkflowRunner({
      registry: makeFixtureRegistry(),
      host: makeFixtureHost(),
      now: () => 0
    });
    const runLog: WorkflowRunRecord[] = [];
    try {
      runner.setDocument(document);
      for (let number = 1; number <= 21; number++) {
        runLog.unshift(await runner.run());
        if (runLog.length > 20) runLog.pop();
      }
      expect(runLog).toHaveLength(20);
      expect(runLog[0].runId).toBe('fixture-workflow-21');
      const markup = render(WorkbenchRunLog, {
        runLog,
        dock: {open: true, height: 220, inspectorTab: 'parameters', logOpen: true}
      });
      expect(markup).toContain('Run 21 · done · 0 steps · 0 ms · 0 cached');
      expect(markup).toContain('Run 2 · done · 0 steps · 0 ms · 0 cached');
      expect(markup).not.toContain('Run 1 ·');
      for (const runId of [
        'fixture-workflow-0',
        'fixture-workflow-01',
        'fixture-workflow-9007199254740993'
      ]) {
        expect(formatWorkbenchRunLine({...makeRun(), runId})).toMatch(`Run ${runId} ·`);
      }
    } finally {
      await runner.destroy();
    }
  });
  test('F4: populated log has a bounded, shrinkable scroll area at minimum dock height', () => {
    const sheet = new ServerStyleSheet();
    try {
      const store = makeStore({
        runLog: Array.from({length: 20}, (_, index) => ({
          ...makeRun(),
          runId: `fixture-workflow-${index + 1}`
        })),
        dock: {open: true, height: 220, inspectorTab: 'parameters', logOpen: true}
      }).store;
      const markup = renderToStaticMarkup(
        sheet.collectStyles(createElement(WorkbenchRunLog, {store}))
      );
      const styles = sheet.getStyleTags();
      const strip = readClassRules(markup.match(/^<div[^>]*>/)![0], styles);
      const list = readClassRules(control(markup, WORKBENCH_TEST_IDS.runLog), styles);
      expect(strip).toContain('display:flex;');
      expect(strip).toContain('flex:0 1 auto;');
      expect(strip).toContain('max-height:50%;');
      expect(list).toContain('min-height:0;');
      expect(list).toContain('flex:1 1 auto;');
      expect(list).toContain('overflow:auto;');
      expect(list).toContain('max-height:160px;');
      const maximumLogFraction = Number(strip.match(/max-height:([\d.]+)%;/)?.[1]) / 100;
      const dockHeight = clampWorkbenchHeight(220, 1000);
      const remainingCanvasHeight = dockHeight - 1 - 24 - 36 - dockHeight * maximumLogFraction;
      expect(remainingCanvasHeight).toBe(49);
      expect(remainingCanvasHeight).toBeGreaterThanOrEqual(24);
    } finally {
      sheet.seal();
    }
  });
});

function renderWithStyles(component: typeof WorkbenchToolbar, patch: Partial<WorkbenchState> = {}) {
  const sheet = new ServerStyleSheet();
  try {
    const markup = renderToStaticMarkup(
      sheet.collectStyles(createElement(component, {store: makeStore(patch).store}))
    );
    return {markup, styles: sheet.getStyleTags()};
  } finally {
    sheet.seal();
  }
}

function readButtonByLabel(markup: string, label: string): string {
  const button = markup.match(
    new RegExp(`<button[^>]*aria-label="${label}"[^>]*>[\\s\\S]*?</button>`)
  )?.[0];
  expect(button, label).toBeDefined();
  return button!;
}

describe('browser design review regressions', () => {
  test('D1: open dock spans the window below Kepler, which the host shortens by its height', () => {
    const {markup, styles} = renderWithStyles(WorkbenchDock, {
      dock: {open: true, height: 380, inspectorTab: 'parameters', logOpen: false}
    });
    const rules = readClassRules(control(markup, WORKBENCH_TEST_IDS.dock), styles);
    expect(rules).toContain('position:fixed;');
    expect(rules).toContain('left:0;');
    expect(rules).toContain('right:0;');
    expect(rules).toContain('bottom:0;');
    expect(rules).toContain('z-index:50;');
    expect(rules).toContain('border-top:1px solid');
    expect(rules).toContain('box-shadow:0 -6px 20px');
    expect(rules).not.toContain('width:max-content;');
  });
  test('D2: collapsed dock is a pill at the top left of the map with a 28-character title', () => {
    const document = {
      ...makeFixtureDocument(),
      title: 'A workflow title that is much too long to fit'
    };
    const {markup, styles} = renderWithStyles(WorkbenchDock, {document});
    const rules = readClassRules(control(markup, WORKBENCH_TEST_IDS.dock), styles);
    expect(rules).toContain('left:calc(var(--workbench-dock-left, 0px) + 48px);');
    expect(rules).toContain('right:auto;');
    expect(rules).toContain('top:10px;');
    expect(rules).toContain('bottom:auto;');
    expect(rules).toContain('width:max-content;');
    expect(rules).toContain('border-radius:18px;');
    expect(rules).toContain('height:36px;');
    expect(markup).toContain('>A workflow title that is mu…</span>');
    expect(markup).toContain(`title="${document.title}"`);
    expect(control(markup, WORKBENCH_TEST_IDS.dockSummary)).toContain('title=');
    for (const testId of [
      WORKBENCH_TEST_IDS.dockToggle,
      WORKBENCH_TEST_IDS.runAll,
      WORKBENCH_TEST_IDS.cancel
    ])
      expect(control(markup, testId)).toMatch(/^<button/);
    expect(markup).not.toContain(`data-testid="${WORKBENCH_TEST_IDS.toolbar}"`);
    expect(truncateWorkbenchTitle('x'.repeat(28))).toBe('x'.repeat(28));
    expect(truncateWorkbenchTitle('🌍'.repeat(29))).toBe(`${'🌍'.repeat(27)}…`);
  });
  test('D3: icon controls retain their ids, accessible names, SVGs and shortcut hints', () => {
    const {markup, styles} = renderWithStyles(WorkbenchToolbar);
    for (const [testId, label] of [
      [WORKBENCH_TEST_IDS.workflowNew, 'New workflow'],
      [WORKBENCH_TEST_IDS.workflowDuplicate, 'Duplicate workflow'],
      [WORKBENCH_TEST_IDS.workflowDelete, 'Delete workflow'],
      [WORKBENCH_TEST_IDS.undo, 'Undo'],
      [WORKBENCH_TEST_IDS.redo, 'Redo'],
      [WORKBENCH_TEST_IDS.layout, 'Lay out'],
      [WORKBENCH_TEST_IDS.importExport, 'Import and export'],
      [WORKBENCH_TEST_IDS.dockToggle, 'Collapse workbench']
    ]) {
      expect(control(markup, testId)).toContain(`aria-label="${label}"`);
      const button = readButtonByLabel(markup, label);
      expect(button).toContain('title=');
      expect(button).toContain('<svg');
      expect(button).toContain('aria-hidden="true"');
      expect(button.replace(/<[^>]+>/g, '')).toBe('');
      expect(readClassRules(control(markup, testId), styles)).toContain('width:26px;');
    }
    for (const label of ['Shortcuts', 'Maximise workbench', 'Rename workflow']) {
      expect(readButtonByLabel(markup, label)).toContain('title=');
      expect(readButtonByLabel(markup, label)).toContain('<svg');
    }
    expect(control(markup, WORKBENCH_TEST_IDS.undo)).toContain(
      formatWorkbenchShortcut('Mod+Z', getWorkbenchPlatform())
    );
    expect(control(markup, WORKBENCH_TEST_IDS.redo)).toContain(
      formatWorkbenchShortcut('Shift+Mod+Z', getWorkbenchPlatform())
    );
    expect(control(markup, WORKBENCH_TEST_IDS.layout)).toContain('title="Lay out (L)"');
    expect(markup.indexOf('aria-label="Maximise workbench"')).toBeLessThan(
      markup.indexOf('aria-label="Collapse workbench"')
    );
  });
  test('D3: narrow mode is based on dock width and uses a Rename control instead of a second field', () => {
    for (const width of [0, 900, 940, 999.99]) expect(isWorkbenchToolbarNarrow(width)).toBe(true);
    for (const width of [1000, 1100, 1440]) expect(isWorkbenchToolbarNarrow(width)).toBe(false);
    const markup = render(WorkbenchToolbar);
    expect(control(markup, WORKBENCH_TEST_IDS.toolbar)).toContain('data-layout="narrow"');
    expect(control(markup, WORKBENCH_TEST_IDS.workflowSelect)).toContain('aria-label="Workflow"');
    expect(markup).not.toContain(`data-testid="${WORKBENCH_TEST_IDS.workflowTitle}"`);
    expect(readButtonByLabel(markup, 'Rename workflow')).not.toContain('disabled');
    const empty = render(WorkbenchToolbar, {
      document: null,
      runner: null,
      workflows: [],
      activeWorkflowId: null
    });
    expect(readButtonByLabel(empty, 'Rename workflow')).toContain('disabled');
    expect(control(markup, WORKBENCH_TEST_IDS.engine)).toContain('title="Engine"');
    expect(control(markup, WORKBENCH_TEST_IDS.engine)).toContain('aria-label="Engine"');
    expect(markup).toContain('>GPU</option>');
    expect(markup).toContain('>CPU</option>');
    expect(markup).toContain('No GPU</span>');
    expect(markup).toContain('Auto-run</label>');
  });
  test('D3: native selects stay legible within the reviewed 1100px toolbar while running', () => {
    const {markup, styles} = renderWithStyles(WorkbenchToolbar, {
      runner: {...makeRunner(), running: true}
    });
    const toolbarRules = readClassRules(control(markup, WORKBENCH_TEST_IDS.toolbar), styles);
    expect(toolbarRules).toContain('gap:1px;');
    expect(toolbarRules).toContain('padding:4px;');
    expect(toolbarRules.lastIndexOf('overflow:visible;')).toBeGreaterThan(
      toolbarRules.lastIndexOf('overflow-x:auto;')
    );
    const buttonWidths = [...markup.matchAll(/<button[^>]*>/g)].map(([opening]) => {
      const rules = readClassRules(opening, styles);
      return Number([...rules.matchAll(/(?:^|;)width:([\d.]+)px;/g)].at(-1)?.[1]);
    });
    expect(buttonWidths).toHaveLength(19);
    expect(buttonWidths.reduce((sum, width) => sum + width, 0)).toBe(650);
    expect(readClassRules(control(markup, WORKBENCH_TEST_IDS.workflowSelect), styles)).toContain(
      'flex:0 1 180px;'
    );
    const workflowRules = readClassRules(
      control(markup, WORKBENCH_TEST_IDS.workflowSelect),
      styles
    );
    expect(workflowRules).toContain('min-width:120px;');
    expect(workflowRules).toContain('overflow:hidden;');
    expect(workflowRules).toContain('text-overflow:ellipsis;');
    expect(workflowRules).toContain('white-space:nowrap;');
    const engineRules = readClassRules(control(markup, WORKBENCH_TEST_IDS.engine), styles);
    expect(engineRules).toContain('flex:none;');
    expect(engineRules).toContain('width:68px;');
    expect(engineRules).toContain('min-width:68px;');
    expect(engineRules).toContain('max-width:68px;');
    // 27 direct flex items: 19 buttons, selector, engine, auto-run, GPU, spacer, three separators.
    const fixedWidth = 650 + 68 + 64 + 50 + 3 + 26 + 8;
    expect(fixedWidth + 120).toBe(989);
    expect(1100 - fixedWidth).toBeGreaterThanOrEqual(120);
  });
  test('D4: maximise saves the previous height and restore respects a resized window', () => {
    const maximised = getWorkbenchHeightToggle(380, null, 1000);
    expect(maximised).toEqual({height: 700, previousHeight: 380});
    expect(getWorkbenchHeightToggle(maximised.height, maximised.previousHeight, 1000)).toEqual({
      height: 380,
      previousHeight: null
    });
    expect(getWorkbenchHeightToggle(700, 380, 300)).toEqual({height: 210, previousHeight: null});
    expect(getWorkbenchHeightToggle(220, null, 1000)).toEqual({height: 700, previousHeight: 220});
    const markup = renderToStaticMarkup(
      createElement(WorkbenchDockHeightContext.Provider, {
        value: {previousHeight: 380, setPreviousHeight: () => {}},
        children: createElement(WorkbenchToolbar, {store: makeStore().store})
      })
    );
    expect(readButtonByLabel(markup, 'Restore workbench')).toContain('aria-pressed="true"');
    expect(readButtonByLabel(markup, 'Restore workbench')).toContain('title="Restore workbench"');
  });
  test('D5: notice dismissal remains an accessible icon and the live region stays mounted', () => {
    const markup = render(WorkbenchDock, {
      dock: {open: true, height: 380, inspectorTab: 'parameters', logOpen: false},
      notice: {kind: 'error', message: 'Check the workflow.'}
    });
    expect(control(markup, WORKBENCH_TEST_IDS.notice)).toContain('role="status"');
    expect(readButtonByLabel(markup, 'Dismiss')).toContain('title="Dismiss"');
    expect(readButtonByLabel(markup, 'Dismiss')).toContain('<svg');
    expect(markup).toContain('aria-live="polite"');
  });
});

// Reuse React's server dispatcher; do not replace shared React hooks or leak module mocks.
function captureControlClick(
  component: typeof WorkbenchToolbar,
  store: WorkbenchStore,
  testId: string
) {
  let click: (() => unknown) | undefined;
  function CaptureControl() {
    const element = component({store});
    function visit(node: unknown): boolean {
      if (Array.isArray(node)) return node.some(visit);
      if (
        !isValidElement<{'data-testid'?: string; onClick?: () => unknown; children?: unknown}>(node)
      )
        return false;
      if (node.props['data-testid'] === testId) {
        click = node.props.onClick;
        return true;
      }
      return visit(node.props.children);
    }
    visit(element);
    return element;
  }
  renderToStaticMarkup(createElement(CaptureControl));
  if (!click) throw new Error(`Control ${testId} did not attach a click listener.`);
  return click;
}

describe('second-wave dock controls', () => {
  test('dispatches clipboard shortcuts while preserving native editing shortcuts', () => {
    for (const [keys, action] of [
      ['Mod+C', 'copy'],
      ['Mod+X', 'cut'],
      ['Mod+V', 'paste']
    ] as const) {
      const {store, calls} = makeStore({clipboardStepCount: 2});
      expect(sendDockKey(store, keys)).toBe(1);
      expect(calls).toEqual(action === 'paste' ? [['paste', undefined]] : [[action]]);
      for (const tagName of ['INPUT', 'TEXTAREA', 'SELECT']) {
        const editable = makeStore({clipboardStepCount: 2});
        expect(sendDockKey(editable.store, keys, tagName)).toBe(0);
        expect(editable.calls).toEqual([]);
      }
    }
    const editable = makeStore({clipboardStepCount: 2});
    let prevented = 0;
    captureDockKeyHandler(editable.store)({
      ...makeKey('Mod+V', getWorkbenchPlatform()),
      defaultPrevented: false,
      target: {isContentEditable: true},
      preventDefault: () => {
        prevented++;
      }
    } as unknown as KeyboardEvent<HTMLDivElement>);
    expect(prevented).toBe(0);
    expect(editable.calls).toEqual([]);
    const paste = makeStore({clipboardStepCount: 2});
    applyWorkbenchShortcut('paste', paste.store, {
      frame: () => {},
      getViewportCenter: () => ({x: 420, y: 240})
    });
    expect(paste.calls).toEqual([['paste', {x: 420, y: 240}]]);
  });
  test('enables clipboard and template icons from the snapshot and delegates their clicks', () => {
    const disabled = render(WorkbenchToolbar, {selectedStepIds: [], clipboardStepCount: 0});
    expect(control(disabled, WORKBENCH_TEST_IDS.copySteps)).toContain('disabled');
    expect(control(disabled, WORKBENCH_TEST_IDS.pasteSteps)).toContain('disabled');
    const {store, calls} = makeStore({clipboardStepCount: 3});
    const markup = renderToStaticMarkup(createElement(WorkbenchToolbar, {store}));
    for (const [testId, label] of [
      [WORKBENCH_TEST_IDS.copySteps, 'Copy steps'],
      [WORKBENCH_TEST_IDS.pasteSteps, 'Paste steps'],
      [WORKBENCH_TEST_IDS.saveTemplate, 'Save as template']
    ]) {
      expect(control(markup, testId)).not.toContain('disabled');
      expect(control(markup, testId)).toContain(`aria-label="${label}"`);
      expect(readButtonByLabel(markup, label)).toContain('<svg');
      captureControlClick(WorkbenchToolbar, store, testId)();
    }
    expect(control(markup, WORKBENCH_TEST_IDS.copySteps)).toContain(
      formatWorkbenchShortcut('Mod+C', getWorkbenchPlatform())
    );
    expect(control(markup, WORKBENCH_TEST_IDS.pasteSteps)).toContain('title="Paste 3 steps"');
    expect(calls).toEqual([['copy'], ['paste', undefined], ['save-template', 'fixture-workflow']]);
    const empty = render(WorkbenchToolbar, {
      document: {...makeFixtureDocument(), steps: []},
      selectedStepIds: []
    });
    expect(control(empty, WORKBENCH_TEST_IDS.saveTemplate)).toContain('disabled');
    const missing = render(WorkbenchToolbar, {
      document: null,
      runner: null,
      workflows: [],
      activeWorkflowId: null,
      clipboardStepCount: 3
    });
    for (const testId of [
      WORKBENCH_TEST_IDS.copySteps,
      WORKBENCH_TEST_IDS.pasteSteps,
      WORKBENCH_TEST_IDS.saveTemplate
    ])
      expect(control(missing, testId)).toContain('disabled');
  });
  test('offers one latest-run Report button, even with the log closed, and opens the export dialog', () => {
    expect(render(WorkbenchRunLog)).not.toContain(`data-testid="${WORKBENCH_TEST_IDS.runReport}"`);
    const {store, calls} = makeStore({
      runLog: [makeRun(), {...makeRun(), runId: 'fixture-workflow-2'}]
    });
    const markup = renderToStaticMarkup(createElement(WorkbenchRunLog, {store}));
    expect(control(markup, WORKBENCH_TEST_IDS.runLogToggle)).toContain('aria-expanded="false"');
    expect(
      markup.match(new RegExp(`data-testid="${WORKBENCH_TEST_IDS.runReport}"`, 'g'))
    ).toHaveLength(1);
    expect(control(markup, WORKBENCH_TEST_IDS.runReport)).not.toContain('disabled');
    expect(markup).toContain('>Report</button>');
    captureControlClick(WorkbenchRunLog, store, WORKBENCH_TEST_IDS.runReport)();
    expect(calls).toEqual([['dialog', 'import-export']]);
  });
});

// Capture real dock and toolbar handlers under their real providers without replacing React hooks.
function captureDockPasteControls(store: WorkbenchStore) {
  let paste: (() => unknown) | undefined;
  let onApi: ((api: WorkbenchCanvasApi | null) => void) | undefined;
  let onKeyDown: ((event: KeyboardEvent<HTMLDivElement>) => void) | undefined;
  function CaptureToolbar() {
    const element = WorkbenchToolbar({store});
    function visit(node: unknown): void {
      if (Array.isArray(node)) {
        node.forEach(visit);
        return;
      }
      if (
        !isValidElement<{'data-testid'?: string; onClick?: () => unknown; children?: unknown}>(node)
      )
        return;
      if (node.props['data-testid'] === WORKBENCH_TEST_IDS.pasteSteps) paste = node.props.onClick;
      else visit(node.props.children);
    }
    visit(element);
    return element;
  }
  function instrument(children: ReactNode): ReactNode {
    return Children.map(children, child => {
      if (
        !isValidElement<{children?: ReactNode; onApi?: (api: WorkbenchCanvasApi | null) => void}>(
          child
        )
      )
        return child;
      if (child.type === WorkbenchCanvas) {
        onApi = child.props.onApi;
        return child;
      }
      if (child.type === WorkbenchToolbar) return createElement(CaptureToolbar);
      return cloneElement(child, {children: instrument(child.props.children)});
    });
  }
  function CaptureDock() {
    const provider = WorkbenchDock({store}) as ReactElement<{
      children: ReactElement<{onKeyDown: (event: KeyboardEvent<HTMLDivElement>) => void}>;
    }>;
    onKeyDown = provider.props.children.props.onKeyDown;
    return instrument(provider);
  }
  renderToStaticMarkup(createElement(CaptureDock));
  if (!paste || !onApi || !onKeyDown) throw new Error('Dock paste controls were not mounted.');
  return {paste, onApi, onKeyDown};
}

describe('second-wave review regressions', () => {
  test('F5: toolbar and keyboard paste use the live viewport centre and handle canvas unmount', () => {
    const {store, calls} = makeStore({
      clipboardStepCount: 2,
      dock: {open: true, height: 380, inspectorTab: 'parameters', logOpen: false}
    });
    const controls = captureDockPasteControls(store);
    controls.onApi({frame: () => {}, getViewportCenter: () => ({x: 4000, y: 3000})});
    controls.paste();
    let prevented = 0;
    controls.onKeyDown({
      ...makeKey('Mod+V', getWorkbenchPlatform()),
      defaultPrevented: false,
      target: {tagName: 'DIV', closest: () => null},
      preventDefault: () => {
        prevented++;
      }
    } as unknown as KeyboardEvent<HTMLDivElement>);
    expect(prevented).toBe(1);
    expect(calls).toEqual([
      ['paste', {x: 4000, y: 3000}],
      ['paste', {x: 4000, y: 3000}]
    ]);
    controls.onApi({frame: () => {}, getViewportCenter: () => ({x: -200, y: 700})});
    controls.paste();
    expect(calls.at(-1)).toEqual(['paste', {x: -200, y: 700}]);
    controls.onApi(null);
    controls.paste();
    expect(calls.at(-1)).toEqual(['paste', undefined]);
  });
});
