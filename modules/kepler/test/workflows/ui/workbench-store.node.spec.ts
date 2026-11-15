// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test, vi} from 'vitest';
import * as keplerWorkflows from '@luma-contributors/kepler';
import {
  createWorkbenchStore,
  type WorkbenchDraft,
  type WorkbenchStore
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-store';
import {WORKBENCH_DOCK} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-contract';
import {
  WORKFLOW_LIMITS,
  type WorkflowCollection,
  type WorkflowDocument,
  type WorkflowOperationRegistry
} from '../../../src/workflows/workflow-types';
import {makeFixtureDocument, makeFixtureHost, makeFixtureRegistry} from '../workflow-fixtures';

function makeStore() {
  const calls: string[] = [];
  const host = makeFixtureHost();
  const registry = makeFixtureRegistry(calls);
  const store = createWorkbenchStore({host, registry});
  return {store, host, registry, calls};
}

function changeFactor(document: WorkflowDocument, factor: number): WorkflowDocument {
  return {
    ...document,
    steps: document.steps.map(step =>
      step.id === 'scale'
        ? {...step, parameters: {...step.parameters, factor: {from: 'static', value: factor}}}
        : step
    )
  };
}

function makeDraft(source: WorkbenchDraft['source'] = 'assistant'): WorkbenchDraft {
  return {document: makeFixtureDocument(), issues: [], outline: 'Source, scale, count', source};
}

describe('workbench store', () => {
  test('starts empty without needing the default registry, and empty calls are stable', async () => {
    const host = makeFixtureHost();
    const store = createWorkbenchStore({host});
    const initial = store.getState();
    expect(initial).toEqual({
      revision: 0,
      workflows: [],
      activeWorkflowId: null,
      document: null,
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
      clipboardStepCount: 0
    });
    expect(store.host).toBe(host);
    expect(store.getRunner()).toBeNull();
    expect(store.serialize()).toBeUndefined();
    expect(await store.run()).toBeNull();
    store.cancel();
    store.notifyExternalChange();
    store.undo();
    store.redo();
    store.select(['missing']);
    store.deleteWorkflow('missing');
    store.deleteTemplate('missing');
    expect(store.copySelection()).toBe(false);
    expect(store.cutSelection()).toBe(false);
    expect(store.paste()).toBe(false);
    store.setActiveWorkflow(null);
    store.setDock({});
    store.closePalette();
    store.rejectDraft();
    store.setDialog(null);
    store.setNotice(null);
    expect(store.acceptDraft()).toBeNull();
    await store.clear();
    store.restore({schemaVersion: 1, workflows: []});
    expect(store.getState()).toBe(initial);
  });

  test('resolves the default registry lazily once per store and honors a supplied registry', async () => {
    const fixtureRegistry = makeFixtureRegistry();
    const getRegistry = vi
      .spyOn(keplerWorkflows, 'getWorkflowOperationRegistry')
      .mockReturnValue(fixtureRegistry);
    try {
      const store = createWorkbenchStore({host: makeFixtureHost()});
      store.getState();
      store.serialize();
      store.setDock({open: true});
      expect(getRegistry).not.toHaveBeenCalled();
      expect(store.registry).toBe(fixtureRegistry);
      expect(store.registry).toBe(fixtureRegistry);
      store.addWorkflow(makeFixtureDocument());
      store.createWorkflow();
      expect(getRegistry).toHaveBeenCalledTimes(1);
      await store.clear();
      const activeStore = createWorkbenchStore({host: makeFixtureHost()});
      expect(getRegistry).toHaveBeenCalledTimes(1);
      activeStore.addWorkflow(makeFixtureDocument());
      activeStore.addWorkflow(makeFixtureDocument());
      expect(getRegistry).toHaveBeenCalledTimes(2);
      await activeStore.clear();
      const suppliedStore = createWorkbenchStore({
        host: makeFixtureHost(),
        registry: fixtureRegistry
      });
      expect(suppliedStore.registry).toBe(fixtureRegistry);
      suppliedStore.addWorkflow(makeFixtureDocument());
      expect(getRegistry).toHaveBeenCalledTimes(2);
      await suppliedStore.clear();
    } finally {
      getRegistry.mockRestore();
    }
  });

  test('F1 refuses a workflow switch inside an edit without changing either workflow', async () => {
    const {store} = makeStore();
    const firstId = store.addWorkflow(makeFixtureDocument());
    const secondId = store.addWorkflow({...makeFixtureDocument(), id: 'second', title: 'Second'});
    store.setActiveWorkflow(firstId);
    store.select(['scale']);
    const initial = store.getState();
    const runner = store.getRunner();
    expect(
      store.edit('reenter', document => {
        store.setActiveWorkflow(secondId);
        return {...document, title: 'Edited first'};
      })
    ).toBe(false);
    expect(store.getState()).toEqual({
      ...initial,
      revision: initial.revision + 1,
      notice: {
        kind: 'error',
        message:
          'Fixture workflow: edit callbacks cannot change the workbench; return an edited document instead.'
      }
    });
    expect(store.getRunner()).toBe(runner);
    expect(store.serialize()).toMatchObject({activeWorkflowId: firstId});
    store.setActiveWorkflow(secondId);
    expect(store.getState().document?.title).toBe('Second');
    expect(store.getState().canUndo).toBe(false);
    await store.clear();
  });

  test('F1 refuses a nested edit even if the callback catches its failure', async () => {
    const {store} = makeStore();
    store.addWorkflow(makeFixtureDocument());
    const initial = store.getState();
    expect(
      store.edit('outer', document => {
        try {
          store.edit('nested', current => changeFactor(current, 9));
        } catch {
          // A callback cannot turn a refused nested mutation into a partial commit.
        }
        return {...document, title: 'Outer edit'};
      })
    ).toBe(false);
    expect(store.getState().document).toBe(initial.document);
    expect(store.getState().runner).toBe(initial.runner);
    expect(store.getState().canUndo).toBe(false);
    expect(store.getState().canRedo).toBe(false);
    expect(store.getState().notice?.message).toBe(
      'Fixture workflow: edit callbacks cannot change the workbench; return an edited document instead.'
    );
    store.setNotice(null);
    expect(
      store.edit('outer-throws', () => {
        try {
          store.edit('nested', current => changeFactor(current, 9));
        } catch {}
        throw new Error('Outer failed');
      })
    ).toBe(false);
    expect(store.getState().document).toBe(initial.document);
    expect(store.getState().canUndo).toBe(false);
    expect(store.getState().notice).toEqual({kind: 'error', message: 'Outer failed'});
    await store.clear();
  });

  test('F1 blocks every store state mutation during the edit callback', async () => {
    const {store} = makeStore();
    const workflowId = store.addWorkflow(makeFixtureDocument());
    const mutations: Array<(store: WorkbenchStore) => unknown> = [
      current => current.createWorkflow(),
      current => current.addWorkflow(makeFixtureDocument()),
      current => current.duplicateWorkflow(workflowId),
      current => current.deleteWorkflow(workflowId),
      current => current.undo(),
      current => current.redo(),
      current => current.select(['scale']),
      current => current.run(),
      current => current.cancel(),
      current => current.notifyExternalChange(),
      current => current.setDock({open: false}),
      current => current.openPalette(),
      current => current.closePalette(),
      current => current.setDialog('shortcuts'),
      current => current.setNotice({kind: 'info', message: 'Nested notice'}),
      current => current.proposeDraft(makeDraft()),
      current => current.acceptDraft(),
      current => current.rejectDraft(),
      current => current.saveTemplate(),
      current => current.deleteTemplate('missing'),
      current => current.useTemplate('missing'),
      current => current.insertWorkflow(makeFixtureDocument()),
      current => current.copySelection(),
      current => current.cutSelection(),
      current => current.paste(),
      current => current.restore({schemaVersion: 1, workflows: []}),
      current => current.clear()
    ];
    for (const mutation of mutations) {
      store.setNotice(null);
      const initial = store.getState();
      expect(
        store.edit('outer', document => {
          mutation(store);
          return document;
        })
      ).toBe(false);
      expect(store.getState()).toEqual({
        ...initial,
        revision: initial.revision + 1,
        notice: {
          kind: 'error',
          message:
            'Fixture workflow: edit callbacks cannot change the workbench; return an edited document instead.'
        }
      });
    }
    await store.clear();
  });

  test('F2 changes workflow identity atomically and supports undo, redo and deletion', async () => {
    const {store, host} = makeStore();
    const originalId = store.addWorkflow(makeFixtureDocument());
    const runner = store.getRunner();
    await store.run();
    const records = store.getState().runLog;
    store.select(['scale']);
    expect(store.edit('rename-id', document => ({...document, id: 'renamed-workflow'}))).toBe(true);
    expect(store.getState().activeWorkflowId).toBe('renamed-workflow');
    expect(store.getState().document?.id).toBe('renamed-workflow');
    expect(store.getState().workflows.map(document => document.id)).toEqual(['renamed-workflow']);
    expect(store.getRunner()).toBe(runner);
    expect(runner?.document?.id).toBe('renamed-workflow');
    expect(store.getState().runLog).toBe(records);
    expect(store.getState().selectedStepIds).toEqual(['scale']);
    expect(store.serialize()).toMatchObject({activeWorkflowId: 'renamed-workflow'});
    store.undo();
    expect(store.getState().activeWorkflowId).toBe(originalId);
    expect(store.getState().document?.id).toBe(originalId);
    expect(store.getRunner()).toBe(runner);
    expect(store.getState().canRedo).toBe(true);
    expect(store.serialize()).toMatchObject({activeWorkflowId: originalId});
    store.redo();
    expect(store.getState().document?.id).toBe('renamed-workflow');
    expect(store.getState().activeWorkflowId).toBe('renamed-workflow');
    expect(store.getState().canUndo).toBe(true);
    store.setActiveWorkflow(null);
    store.setActiveWorkflow('renamed-workflow');
    expect(store.getRunner()).toBe(runner);
    store.deleteWorkflow('renamed-workflow');
    await store.clear();
    expect(store.getState().workflows).toEqual([]);
    expect(host.retracted).toEqual(['published-1']);
  });

  test('F2 refuses duplicate and invalid ids before changing the document or history', async () => {
    const {store} = makeStore();
    const firstId = store.addWorkflow(makeFixtureDocument());
    const secondId = store.addWorkflow({...makeFixtureDocument(), id: 'second'});
    store.setActiveWorkflow(firstId);
    const initial = store.getState();
    expect(store.edit('duplicate-id', document => ({...document, id: secondId}))).toBe(false);
    expect(store.getState().document).toBe(initial.document);
    expect(store.getState().runner).toBe(initial.runner);
    expect(store.getState().canUndo).toBe(false);
    expect(store.getState().notice).toEqual({
      kind: 'error',
      message: 'Fixture workflow: Id "second" must be unique among workflows.'
    });
    expect(store.edit('invalid-id', document => ({...document, id: 'Not a valid id'}))).toBe(false);
    expect(store.getState().document).toBe(initial.document);
    expect(store.getState().canUndo).toBe(false);
    expect(store.getState().notice).toEqual({
      kind: 'error',
      message:
        'Fixture workflow: Id must use lowercase letters, digits and hyphens and contain at most 48 characters.'
    });
    expect(store.serialize()).toMatchObject({activeWorkflowId: firstId});
    await store.clear();
  });

  test.each([
    'undo',
    'redo'
  ] as const)('F2 refuses an id collision during %s without moving history', async direction => {
    const {store} = makeStore();
    const originalId = store.addWorkflow(makeFixtureDocument());
    store.edit('rename-id', document => ({...document, id: 'renamed-workflow'}));
    if (direction === 'redo') store.undo();
    const activeId = direction === 'undo' ? 'renamed-workflow' : originalId;
    const conflictingId = direction === 'undo' ? originalId : 'renamed-workflow';
    store.addWorkflow({...makeFixtureDocument(), id: conflictingId});
    store.setActiveWorkflow(activeId);
    const initial = store.getState();
    store[direction]();
    expect(store.getState().document).toBe(initial.document);
    expect(store.getState().runner).toBe(initial.runner);
    expect(store.getState().canUndo).toBe(initial.canUndo);
    expect(store.getState().canRedo).toBe(initial.canRedo);
    expect(store.getState().notice).toEqual({
      kind: 'error',
      message: `Fixture workflow: Id "${conflictingId}" must be unique among workflows.`
    });
    expect(store.getState().workflows.map(document => document.id)).toEqual([
      activeId,
      conflictingId
    ]);
    store.deleteWorkflow(conflictingId);
    store[direction]();
    expect(store.getState().activeWorkflowId).toBe(conflictingId);
    expect(store.getState().document?.id).toBe(conflictingId);
    await store.clear();
  });

  test('F3 copies and freezes caller-owned drafts, notices and nested palette options', async () => {
    const {store} = makeStore();
    const notice = {kind: 'info' as const, message: 'Original notice'};
    const position = {x: 10, y: 20};
    const connectFrom = {stepId: 'source', output: 'data'};
    const draft = makeDraft();
    const issue = {
      severity: 'warning' as const,
      code: 'unverified-port' as const,
      message: 'Original issue'
    };
    draft.issues = [issue];
    store.setNotice(notice);
    store.openPalette({position, connectFrom});
    store.proposeDraft(draft);
    const before = store.getState();
    let notifications = 0;
    const unsubscribe = store.subscribe(() => notifications++);
    notice.message = 'Changed outside store';
    position.x = 999;
    connectFrom.output = 'changed';
    draft.document.steps = [];
    draft.outline = 'Changed outline';
    issue.message = 'Changed issue';
    expect(store.getState()).toBe(before);
    expect(notifications).toBe(0);
    expect(before.notice).toEqual({kind: 'info', message: 'Original notice'});
    expect(before.palette).toEqual({
      open: true,
      position: {x: 10, y: 20},
      connectFrom: {stepId: 'source', output: 'data'}
    });
    expect(before.draft?.document.steps).toHaveLength(3);
    expect(before.draft?.outline).toBe('Source, scale, count');
    expect(before.draft?.issues).toEqual([
      {severity: 'warning', code: 'unverified-port', message: 'Original issue'}
    ]);
    expect(Object.isFrozen(before.notice)).toBe(true);
    expect(Object.isFrozen(before.palette.position)).toBe(true);
    expect(Object.isFrozen(before.palette.connectFrom)).toBe(true);
    expect(Object.isFrozen(before.draft)).toBe(true);
    expect(Object.isFrozen(before.draft?.issues[0])).toBe(true);
    expect(Object.isFrozen(before.draft?.document.steps)).toBe(true);
    expect(store.acceptDraft()).toBe('fixture-workflow');
    expect(store.getState().document?.steps).toHaveLength(3);
    expect(notifications).toBe(1);
    unsubscribe();
    await store.clear();
  });

  test('F4 delivers committed changes to every subscriber and reports subscriber failures', () => {
    const {store} = makeStore();
    const failed = vi.fn(() => {
      throw new Error('Subscriber failed');
    });
    const unsubscribe = store.subscribe(failed);
    const received: number[] = [];
    store.subscribe(() => received.push(store.getState().revision));
    expect(() => store.setNotice({kind: 'info', message: 'Committed'})).not.toThrow();
    expect(received).toEqual([1, 2]);
    expect(failed).toHaveBeenCalledTimes(2);
    expect(store.getState().notice).toEqual({
      kind: 'error',
      message: 'Workflows: subscriber failed: Subscriber failed'
    });
    unsubscribe();
    store.setNotice({kind: 'info', message: 'Recovered'});
    expect(received).toEqual([1, 2, 3]);
    expect(store.getState().notice).toEqual({kind: 'info', message: 'Recovered'});
  });

  test('F5 converts unreadable thrown values to a notice without changing the document', async () => {
    const {store} = makeStore();
    store.addWorkflow(makeFixtureDocument());
    const initial = store.getState();
    let accepted: boolean | undefined;
    expect(() => {
      accepted = store.edit('throw-value', () => {
        throw Object.create(null);
      });
    }).not.toThrow();
    expect(accepted).toBe(false);
    expect(store.getState()).toEqual({
      ...initial,
      revision: initial.revision + 1,
      notice: {
        kind: 'error',
        message: 'Workflows: an error occurred, but its message could not be read.'
      }
    });
    await store.clear();
  });

  test('F5 also handles an Error whose message getter throws', async () => {
    const {store} = makeStore();
    store.addWorkflow(makeFixtureDocument());
    const initial = store.getState();
    const failure = new Error('Unreadable');
    Object.defineProperty(failure, 'message', {
      get() {
        throw new Error('Cannot read message');
      }
    });
    expect(
      store.edit('throw-error', () => {
        throw failure;
      })
    ).toBe(false);
    expect(store.getState()).toEqual({
      ...initial,
      revision: initial.revision + 1,
      notice: {
        kind: 'error',
        message: 'Workflows: an error occurred, but its message could not be read.'
      }
    });
    await store.clear();
  });

  test('creates, adds and duplicates with unique ids and titles and opens the dock', async () => {
    const {store, registry} = makeStore();
    expect(store.registry).toBe(registry);
    expect(store.createWorkflow()).toBe('untitled-workflow');
    expect(store.getState().document?.steps).toEqual([]);
    expect(store.getState().document?.title).toBe('Untitled workflow');
    expect(store.createWorkflow()).toBe('untitled-workflow-2');
    expect(store.getState().document?.title).toBe('Untitled workflow 2');
    expect(store.createWorkflow({title: 'Custom title'})).toBe('custom-title');
    const firstId = store.addWorkflow(makeFixtureDocument());
    store.select(['scale']);
    store.setDock({open: false});
    expect(store.duplicateWorkflow(firstId)).toBe('fixture-workflow-2');
    expect(store.getState().document?.title).toBe('Fixture workflow 2');
    expect(store.getState().selectedStepIds).toEqual([]);
    expect(store.getState().dock.open).toBe(true);
    expect(store.addWorkflow(makeFixtureDocument())).toBe('fixture-workflow-3');
    expect(store.getState().document?.title).toBe('Fixture workflow 3');
    expect(store.getState().workflows).toHaveLength(6);
    await store.clear();
  });

  test('keeps generated titles and ids within their limits', async () => {
    const {store} = makeStore();
    const title = 'a'.repeat(WORKFLOW_LIMITS.maximumTitleCharacters);
    const firstId = store.createWorkflow({title});
    const secondId = store.duplicateWorkflow(firstId);
    expect(firstId).toHaveLength(WORKFLOW_LIMITS.maximumIdCharacters);
    expect(secondId).toHaveLength(WORKFLOW_LIMITS.maximumIdCharacters);
    expect(secondId.endsWith('-2')).toBe(true);
    expect(store.getState().document?.title).toBe(`${title.slice(0, -2)} 2`);
    expect(store.duplicateWorkflow(firstId).endsWith('-3')).toBe(true);
    expect(store.getState().document?.title).toBe(`${title.slice(0, -2)} 3`);
    await store.clear();
  });

  test('preserves snapshot identity on no-ops and notifies once for a compound edit', async () => {
    const {store} = makeStore();
    let notifications = 0;
    const unsubscribe = store.subscribe(() => notifications++);
    store.addWorkflow(makeFixtureDocument());
    expect(notifications).toBe(1);
    const initial = store.getState();
    expect(initial.revision).toBe(1);
    store.setActiveWorkflow(initial.activeWorkflowId);
    store.select([]);
    expect(store.edit('identity', document => document)).toBe(true);
    expect(store.getState()).toBe(initial);
    expect(notifications).toBe(1);
    expect(store.edit('factor', document => changeFactor(document, 4))).toBe(true);
    expect(store.getState().revision).toBe(2);
    expect(store.getState().canUndo).toBe(true);
    expect(notifications).toBe(2);
    const edited = store.getState();
    store.undo();
    expect(notifications).toBe(3);
    expect(store.getState().revision).toBe(3);
    expect(edited.document?.steps[1].parameters['factor']).toEqual({from: 'static', value: 4});
    expect(initial.document?.steps[1].parameters['factor']).toEqual({from: 'static', value: 3});
    unsubscribe();
    store.redo();
    expect(notifications).toBe(3);
    await store.clear();
  });

  test('queues listener changes without calling listeners re-entrantly', () => {
    const {store} = makeStore();
    let depth = 0;
    let maximumDepth = 0;
    const revisions: number[] = [];
    store.subscribe(() => {
      depth++;
      maximumDepth = Math.max(maximumDepth, depth);
      revisions.push(store.getState().revision);
      if (store.getState().revision === 1) store.setDialog('shortcuts');
      depth--;
    });
    store.setNotice({kind: 'info', message: 'First change'});
    expect(maximumDepth).toBe(1);
    expect(revisions).toEqual([1, 2]);
    expect(store.getState().dialog).toBe('shortcuts');
  });

  test('refuses edits without a workflow and catches edit errors without changing history', async () => {
    const {store} = makeStore();
    let called = false;
    expect(
      store.edit('empty', document => {
        called = true;
        return document;
      })
    ).toBe(false);
    expect(called).toBe(false);
    expect(store.getState().notice).toEqual({
      kind: 'error',
      message: 'Workflows: select a workflow before editing.'
    });
    store.addWorkflow(makeFixtureDocument());
    store.setNotice(null);
    store.select(['scale']);
    const initial = store.getState();
    let notifications = 0;
    const unsubscribe = store.subscribe(() => notifications++);
    expect(
      store.edit('refuse', document =>
        keplerWorkflows.renameWorkflow(
          document,
          'a'.repeat(WORKFLOW_LIMITS.maximumTitleCharacters + 1)
        )
      )
    ).toBe(false);
    expect(store.getState()).toEqual({
      ...initial,
      revision: initial.revision + 1,
      notice: {
        kind: 'error',
        message: 'Fixture workflow: Title must have 1 to maximumTitleCharacters (80) characters.'
      }
    });
    expect(notifications).toBe(1);
    const refused = store.getState();
    store.undo();
    expect(store.getState()).toBe(refused);
    unsubscribe();
    expect(store.edit('factor', document => changeFactor(document, 4))).toBe(true);
    expect(store.getState().notice).toBeNull();
    store.setNotice({kind: 'info', message: 'Keep this'});
    store.edit('factor-again', document => changeFactor(document, 5));
    expect(store.getState().notice).toEqual({kind: 'info', message: 'Keep this'});
    await store.clear();
  });

  test('a callback cannot mutate the current document and then throw', async () => {
    const {store} = makeStore();
    store.addWorkflow(makeFixtureDocument());
    const document = store.getState().document;
    expect(
      store.edit('mutation', current => {
        current.title = 'Changed';
        throw new Error('Failed');
      })
    ).toBe(false);
    expect(store.getState().document).toBe(document);
    expect(document?.title).toBe('Fixture workflow');
    expect(store.getState().canUndo).toBe(false);
    expect(store.getState().notice?.kind).toBe('error');
    await store.clear();
  });

  test('deduplicates selection, preserves the inspector tab and prunes deleted or undone steps', async () => {
    const {store} = makeStore();
    store.addWorkflow(makeFixtureDocument());
    store.setDock({inspectorTab: 'result'});
    store.select(['count', 'missing', 'scale', 'count']);
    expect(store.getState().selectedStepIds).toEqual(['count', 'scale']);
    store.select(['scale']);
    expect(store.getState().dock.inspectorTab).toBe('result');
    store.edit('delete-count', document => ({
      ...document,
      steps: document.steps.filter(step => step.id !== 'count')
    }));
    store.select(['source', 'scale']);
    store.edit('delete-scale', document => ({
      ...document,
      steps: document.steps.filter(step => step.id !== 'scale')
    }));
    expect(store.getState().selectedStepIds).toEqual(['source']);
    store.undo();
    expect(store.getState().selectedStepIds).toEqual(['source']);
    store.select(['scale']);
    store.redo();
    expect(store.getState().selectedStepIds).toEqual([]);
    store.edit('add-count', document => ({
      ...document,
      steps: [...document.steps, makeFixtureDocument().steps[2]]
    }));
    store.select(['count']);
    store.undo();
    expect(store.getState().selectedStepIds).toEqual([]);
    await store.clear();
  });

  test('undo after running restores cached results without new operation calls', async () => {
    const {store, calls} = makeStore();
    store.addWorkflow(makeFixtureDocument());
    const record = await store.run();
    expect(record?.status).toBe('done');
    expect(calls).toEqual(['fixture-source:source', 'fixture-scale:scale', 'fixture-count:count']);
    store.edit('factor', document => changeFactor(document, 4));
    expect(store.getState().runner?.steps['source'].status).toBe('done');
    expect(store.getState().runner?.steps['scale'].status).toBe('stale');
    await store.run();
    expect(calls).toEqual([
      'fixture-source:source',
      'fixture-scale:scale',
      'fixture-count:count',
      'fixture-scale:scale',
      'fixture-count:count'
    ]);
    const callCount = calls.length;
    store.undo();
    expect(store.getState().canUndo).toBe(false);
    expect(store.getState().canRedo).toBe(true);
    expect(Object.values(store.getState().runner!.steps).map(step => step.status)).toEqual([
      'done',
      'done',
      'done'
    ]);
    expect(calls).toHaveLength(callCount);
    store.redo();
    expect(store.getState().canUndo).toBe(true);
    expect(store.getState().canRedo).toBe(false);
    expect(store.getState().runner?.steps['scale'].status).toBe('done');
    expect(calls).toHaveLength(callCount);
    await store.clear();
  });

  test('forwards every active runner change, but unsubscribes when switching', async () => {
    const {store} = makeStore();
    const firstId = store.addWorkflow(makeFixtureDocument());
    const firstRunner = store.getRunner()!;
    const revisions: number[] = [];
    const unsubscribe = store.subscribe(() => revisions.push(store.getState().revision));
    let runnerChanges = 0;
    const unsubscribeRunner = firstRunner.subscribe(() => runnerChanges++);
    await firstRunner.run();
    expect(runnerChanges).toBeGreaterThan(3);
    expect(revisions).toHaveLength(runnerChanges);
    expect(revisions).toEqual(Array.from({length: runnerChanges}, (_, index) => index + 2));
    expect(store.getState().runner).toBe(firstRunner.getState());
    const secondId = store.addWorkflow(makeFixtureDocument());
    const secondState = store.getState();
    await firstRunner.run({force: true});
    expect(store.getState()).toBe(secondState);
    expect(store.getState().activeWorkflowId).toBe(secondId);
    store.setActiveWorkflow(firstId);
    expect(store.getRunner()).toBe(firstRunner);
    expect(store.getState().runner?.steps['scale'].status).toBe('done');
    unsubscribe();
    unsubscribeRunner();
    await store.clear();
  });

  test('keeps separate histories, cached results and run logs for workflows', async () => {
    const {store, calls} = makeStore();
    const firstId = store.addWorkflow(makeFixtureDocument());
    const firstRunner = store.getRunner();
    const firstRecord = await store.run();
    store.edit('factor', document => changeFactor(document, 4));
    const secondId = store.addWorkflow(makeFixtureDocument());
    const secondRunner = store.getRunner();
    expect(secondRunner).not.toBe(firstRunner);
    expect(store.getState().canUndo).toBe(false);
    expect(store.getState().runLog).toEqual([]);
    const secondRecord = await store.run();
    store.setActiveWorkflow(firstId);
    expect(store.getRunner()).toBe(firstRunner);
    expect(store.getState().canUndo).toBe(true);
    expect(store.getState().runLog).toEqual([firstRecord]);
    store.undo();
    const callCount = calls.length;
    await store.run();
    expect(calls).toHaveLength(callCount);
    store.setActiveWorkflow(secondId);
    expect(store.getRunner()).toBe(secondRunner);
    expect(store.getState().runLog).toEqual([secondRecord]);
    expect(store.getState().runner?.steps['scale'].status).toBe('done');
    store.setActiveWorkflow(null);
    expect(store.getState().document).toBeNull();
    expect(store.getState().runner).toBeNull();
    expect(store.getState().runLog).toEqual([]);
    store.setActiveWorkflow(secondId);
    expect(store.getRunner()).toBe(secondRunner);
    await store.clear();
  });

  test('forwards run target and force and bounds the newest-first run log to 20', async () => {
    const {store, calls} = makeStore();
    store.addWorkflow(makeFixtureDocument());
    const first = await store.run({target: 'scale'});
    expect(first?.target).toBe('scale');
    expect(first?.steps.map(step => step.stepId)).toEqual(['source', 'scale']);
    expect(calls).toEqual(['fixture-source:source', 'fixture-scale:scale']);
    const second = await store.run({target: 'scale', force: true});
    expect(second?.steps.every(step => !step.cacheHit)).toBe(true);
    expect(calls).toEqual([
      'fixture-source:source',
      'fixture-scale:scale',
      'fixture-source:source',
      'fixture-scale:scale'
    ]);
    const records = [first, second];
    for (let index = 0; index < 20; index++) records.push(await store.run());
    expect(store.getState().runLog).toEqual(records.slice(-20).reverse());
    await store.clear();
  });

  test('shows the first failed step message in an error notice', async () => {
    const {store} = makeStore();
    const document = makeFixtureDocument();
    document.steps[1].parameters['fail'] = {from: 'static', value: true};
    store.addWorkflow(document);
    const record = await store.run();
    expect(record?.status).toBe('error');
    const failure = record?.steps.find(step => step.status === 'error')?.error;
    expect(failure?.code).toBe('operation-failed');
    expect(failure?.message).toContain('Fixture scale: asked to fail');
    expect(store.getState().notice).toEqual({kind: 'error', message: failure?.message});
    expect(store.getState().runLog).toEqual([record]);
    await store.clear();
  });

  test('cancels an active run and does not revive logs after clear', async () => {
    const calls: string[] = [];
    const registry = makeFixtureRegistry(calls);
    const source = registry.get('fixture-source')!;
    let markStarted = () => {};
    const started = new Promise<void>(resolve => {
      markStarted = resolve;
    });
    const waitingSource = {
      ...source,
      run: async (...arguments_: Parameters<typeof source.run>) => {
        markStarted();
        await new Promise<void>((_resolve, reject) => {
          arguments_[0].signal.addEventListener('abort', () => reject(new Error('Cancelled')), {
            once: true
          });
        });
        return source.run(...arguments_);
      }
    };
    const waitingRegistry: WorkflowOperationRegistry = {
      list: () => [
        waitingSource,
        ...registry.list().filter(operation => operation.id !== source.id)
      ],
      get: operationId => (operationId === source.id ? waitingSource : registry.get(operationId))
    };
    const store = createWorkbenchStore({host: makeFixtureHost(), registry: waitingRegistry});
    store.addWorkflow(makeFixtureDocument());
    const pending = store.run();
    await started;
    expect(store.getState().runner?.running).toBe(true);
    store.cancel();
    expect((await pending)?.status).toBe('cancelled');
    expect(store.getState().runner?.running).toBe(false);
    expect(store.getState().runLog).toHaveLength(1);
    const next = store.run();
    await store.clear();
    expect((await next)?.status).toBe('cancelled');
    expect(store.getState().runLog).toEqual([]);
    expect(store.getState().workflows).toEqual([]);
  });

  test('refreshes external revisions only on the active runner and on activation', async () => {
    const {store, host} = makeStore();
    const firstId = store.addWorkflow(makeFixtureDocument());
    await store.run();
    const firstRunner = store.getRunner()!;
    const secondId = store.addWorkflow(makeFixtureDocument());
    await store.run();
    const firstState = firstRunner.getState();
    host.revision++;
    store.notifyExternalChange();
    expect(store.getState().runner?.steps['source'].status).toBe('stale');
    expect(firstRunner.getState()).toBe(firstState);
    store.setActiveWorkflow(firstId);
    expect(store.getState().runner?.steps['source'].status).toBe('stale');
    store.setActiveWorkflow(secondId);
    expect(store.getState().runner?.steps['source'].status).toBe('stale');
    await store.clear();
  });

  test('updates dock, palette, dialog and notices and preserves no-op snapshots', () => {
    const {store} = makeStore();
    store.setDock({height: -10, inspectorTab: 'contract', logOpen: true});
    expect(store.getState().dock).toEqual({
      open: false,
      height: WORKBENCH_DOCK.minimumHeight,
      inspectorTab: 'contract',
      logOpen: true
    });
    store.setDock({height: 2000});
    expect(store.getState().dock.height).toBe(900);
    store.setDock({height: Number.NaN});
    expect(store.getState().dock.height).toBe(900);
    store.openPalette({position: {x: 10, y: 20}, connectFrom: {stepId: 'scale', output: 'scaled'}});
    expect(store.getState().dock.open).toBe(true);
    expect(store.getState().palette).toEqual({
      open: true,
      position: {x: 10, y: 20},
      connectFrom: {stepId: 'scale', output: 'scaled'}
    });
    const opened = store.getState();
    store.openPalette({position: {x: 10, y: 20}, connectFrom: {stepId: 'scale', output: 'scaled'}});
    expect(store.getState()).toBe(opened);
    store.closePalette();
    expect(store.getState().palette).toEqual({open: false});
    store.setDialog('templates');
    expect(store.getState().dialog).toBe('templates');
    store.setDialog('import-export');
    expect(store.getState().dialog).toBe('import-export');
    store.setNotice({kind: 'info', message: 'Saved'});
    const noticed = store.getState();
    store.setNotice({kind: 'info', message: 'Saved'});
    expect(store.getState()).toBe(noticed);
    store.setNotice(null);
    expect(store.getState().notice).toBeNull();
  });

  test('replaces, rejects and accepts drafts without running them', async () => {
    const {store, calls} = makeStore();
    store.proposeDraft(makeDraft());
    const imported = makeDraft('import');
    store.proposeDraft(imported);
    expect(store.getState().draft).toEqual(imported);
    expect(store.getState().draft).not.toBe(imported);
    expect(store.getState().workflows).toEqual([]);
    store.rejectDraft();
    expect(store.getState().draft).toBeNull();
    store.proposeDraft(makeDraft('history'));
    let notifications = 0;
    const unsubscribe = store.subscribe(() => notifications++);
    expect(store.acceptDraft()).toBe('fixture-workflow');
    expect(notifications).toBe(1);
    expect(store.getState().draft).toBeNull();
    expect(store.getState().dock.open).toBe(true);
    expect(store.getState().document?.title).toBe('Fixture workflow');
    expect(calls).toEqual([]);
    unsubscribe();
    await store.clear();
  });

  test('serializes and restores documents with fresh ready states and lazy inactive sessions', async () => {
    const {store, host} = makeStore();
    const firstId = store.addWorkflow(makeFixtureDocument());
    await store.run();
    const secondId = store.addWorkflow(makeFixtureDocument());
    await store.run();
    store.setActiveWorkflow(firstId);
    store.select(['scale']);
    store.edit('factor', document => changeFactor(document, 4));
    store.proposeDraft(makeDraft());
    store.setNotice({kind: 'error', message: 'Old error'});
    const saved = store.serialize() as WorkflowCollection;
    let revisionReads = 0;
    const originalRead = host.getDatasetRevisionKey;
    host.getDatasetRevisionKey = request => {
      revisionReads++;
      return originalRead(request);
    };
    store.restore(saved);
    expect(store.serialize()).toEqual(saved);
    expect(store.getState().selectedStepIds).toEqual([]);
    expect(store.getState().draft).toBeNull();
    expect(store.getState().notice).toBeNull();
    expect(store.getState().runLog).toEqual([]);
    expect(store.getState().canUndo).toBe(false);
    expect(store.getState().canRedo).toBe(false);
    expect(Object.values(store.getState().runner!.steps).map(step => step.status)).toEqual([
      'ready',
      'ready',
      'ready'
    ]);
    const activeReads = revisionReads;
    expect(activeReads).toBeGreaterThan(0);
    store.setActiveWorkflow(secondId);
    expect(revisionReads).toBeGreaterThan(activeReads);
    expect(store.getState().runner?.steps['scale'].status).toBe('ready');
    expect(store.getState().runLog).toEqual([]);
    store.restore({schemaVersion: 1, workflows: saved.workflows});
    expect(store.getState().activeWorkflowId).toBe(firstId);
    await store.clear();
    expect([...host.retracted].sort()).toEqual(['published-1', 'published-2']);
  });

  test('deleting workflows and clearing retract all their published datasets', async () => {
    const {store, host} = makeStore();
    const firstId = store.addWorkflow(makeFixtureDocument());
    await store.run();
    const secondId = store.addWorkflow(makeFixtureDocument());
    await store.run();
    expect(host.published.map(output => output.datasetId)).toEqual(['published-1', 'published-2']);
    store.deleteWorkflow(firstId);
    expect(store.getState().activeWorkflowId).toBe(secondId);
    expect(store.getState().workflows.map(document => document.id)).toEqual([secondId]);
    await store.clear();
    expect([...host.retracted].sort()).toEqual(['published-1', 'published-2']);
    expect(store.serialize()).toBeUndefined();
    expect(store.getState().document).toBeNull();
    expect(store.getRunner()).toBeNull();
    expect(store.getState().runLog).toEqual([]);
  });

  test('clear waits for output retractions started by an earlier delete', async () => {
    const {store, host} = makeStore();
    const workflowId = store.addWorkflow(makeFixtureDocument());
    await store.run();
    let finishRetraction = () => {};
    const retraction = new Promise<void>(resolve => {
      finishRetraction = resolve;
    });
    let markStarted = () => {};
    const started = new Promise<void>(resolve => {
      markStarted = resolve;
    });
    host.retractOutput = async datasetId => {
      markStarted();
      await retraction;
      host.retracted.push(datasetId);
    };
    store.deleteWorkflow(workflowId);
    await started;
    let cleared = false;
    const clearing = store.clear().then(() => {
      cleared = true;
    });
    await Promise.resolve();
    expect(cleared).toBe(false);
    expect(host.retracted).toEqual([]);
    finishRetraction();
    await clearing;
    expect(cleared).toBe(true);
    expect(host.retracted).toEqual(['published-1']);
  });

  test('deleting the active workflow activates the first remaining workflow and clears selection', async () => {
    const {store} = makeStore();
    const firstId = store.addWorkflow(makeFixtureDocument());
    const secondId = store.addWorkflow(makeFixtureDocument());
    store.select(['scale']);
    store.deleteWorkflow(secondId);
    expect(store.getState().activeWorkflowId).toBe(firstId);
    expect(store.getState().selectedStepIds).toEqual([]);
    store.deleteWorkflow(firstId);
    expect(store.getState().activeWorkflowId).toBeNull();
    await store.clear();
  });

  test('F6 refuses additions at the limit while clearing accepted drafts and opening the dock', async () => {
    const {store} = makeStore();
    for (let index = 0; index < WORKFLOW_LIMITS.maximumWorkflows; index++) store.createWorkflow();
    const initial = store.getState();
    expect(store.createWorkflow()).toBe('');
    expect(store.getState().workflows).toBe(initial.workflows);
    expect(store.getState().activeWorkflowId).toBe(initial.activeWorkflowId);
    expect(store.getState().notice).toEqual({
      kind: 'error',
      message: 'Workflows: maximumWorkflows is 16.'
    });
    expect(store.addWorkflow(makeFixtureDocument())).toBe('');
    expect(store.duplicateWorkflow(initial.activeWorkflowId!)).toBe('');
    const templateId = store.saveTemplate();
    expect(store.useTemplate(templateId)).toBe('');
    expect(store.getState().activeWorkflowId).toBe(initial.activeWorkflowId);
    const draft = makeDraft();
    store.proposeDraft(draft);
    store.setDock({open: false});
    expect(store.acceptDraft()).toBe('');
    expect(store.getState().draft).toBeNull();
    expect(store.getState().dock.open).toBe(true);
    expect(store.getState().workflows).toHaveLength(WORKFLOW_LIMITS.maximumWorkflows);
    await store.clear();
  });

  test('unknown workflow calls report an error and leave the active document intact', async () => {
    const {store} = makeStore();
    store.addWorkflow(makeFixtureDocument());
    const document = store.getState().document;
    store.setActiveWorkflow('missing');
    expect(store.getState().document).toBe(document);
    expect(store.getState().notice).toEqual({
      kind: 'error',
      message: 'Workflows: workflow "missing" does not exist.'
    });
    expect(store.duplicateWorkflow('missing')).toBe('');
    expect(store.getState().document).toBe(document);
    await store.clear();
  });

  test('saves independent templates with distinct identities and reproduces display settings on use', async () => {
    const {store, calls} = makeStore();
    const original = makeFixtureDocument();
    original.steps[1].displayStyle = {
      scaled: {colorColumn: 'value', colorScale: 'quantile', palette: 'diverging'}
    };
    const workflowId = store.addWorkflow(original);
    store.select(['scale']);
    let notifications = 0;
    const unsubscribe = store.subscribe(() => notifications++);
    const templateId = store.saveTemplate();
    expect(templateId).toBe('fixture-workflow');
    expect(notifications).toBe(1);
    expect(store.getState().notice).toEqual({
      kind: 'info',
      message: 'Saved "Fixture workflow" as a template.'
    });
    expect(store.getState().selectedStepIds).toEqual(['scale']);
    expect(store.getState().canUndo).toBe(false);
    const template = store.getState().templates[0];
    expect(template).toEqual(original);
    expect(template).not.toBe(store.getState().document);
    expect(template.steps[1]).not.toBe(store.getState().document?.steps[1]);
    expect(Object.isFrozen(template.steps[1].displayStyle?.['scaled'])).toBe(true);
    expect(store.saveTemplate()).toBe('fixture-workflow-2');
    expect(store.getState().templates.map(saved => saved.title)).toEqual([
      'Fixture workflow',
      'Fixture workflow 2'
    ]);
    const otherId = store.createWorkflow({title: 'Other'});
    expect(store.saveTemplate(workflowId)).toBe('fixture-workflow-3');
    expect(store.getState().activeWorkflowId).toBe(otherId);
    expect(store.useTemplate(templateId)).toBe('fixture-workflow-2');
    expect(store.getState().dock.open).toBe(true);
    expect(store.getState().document?.steps[1].display).toEqual(['scaled']);
    expect(store.getState().document?.steps[1].displayStyle).toEqual(
      original.steps[1].displayStyle
    );
    store.edit('factor', document => changeFactor(document, 6));
    expect(template.steps[1].parameters['factor']).toEqual({from: 'static', value: 3});
    expect(calls).toEqual([]);
    store.deleteTemplate('fixture-workflow-2');
    expect(store.getState().templates.map(saved => saved.id)).toEqual([
      'fixture-workflow',
      'fixture-workflow-3'
    ]);
    const beforeMissingDelete = store.getState();
    store.deleteTemplate('missing');
    expect(store.getState()).toBe(beforeMissingDelete);
    expect(store.useTemplate('missing')).toBe('');
    expect(store.getState().notice).toEqual({
      kind: 'error',
      message: 'Templates: template "missing" does not exist.'
    });
    unsubscribe();
    await store.clear();
  });

  test('refuses missing template sources and the template limit and bounds generated titles', async () => {
    const {store} = makeStore();
    expect(store.saveTemplate()).toBe('');
    expect(store.getState().notice).toEqual({
      kind: 'error',
      message: 'Templates: select a workflow before saving a template.'
    });
    expect(store.saveTemplate('missing')).toBe('');
    expect(store.getState().notice).toEqual({
      kind: 'error',
      message: 'Templates: workflow "missing" does not exist.'
    });
    const title = 'T'.repeat(WORKFLOW_LIMITS.maximumTitleCharacters);
    store.createWorkflow({title});
    let firstTemplateId = '';
    for (let index = 0; index < WORKFLOW_LIMITS.maximumUserTemplates; index++) {
      const templateId = store.saveTemplate();
      if (index === 0) firstTemplateId = templateId;
    }
    const before = store.getState();
    expect(before.templates).toHaveLength(32);
    expect(before.templates[31].title).toBe(`${title.slice(0, -3)} 32`);
    expect(before.templates[31].id).toHaveLength(WORKFLOW_LIMITS.maximumIdCharacters);
    expect(store.saveTemplate()).toBe('');
    expect(store.getState().templates).toBe(before.templates);
    expect(store.getState().activeWorkflowId).toBe(before.activeWorkflowId);
    expect(store.getState().notice).toEqual({
      kind: 'error',
      message: 'Templates: maximumUserTemplates is 32.'
    });
    store.deleteTemplate(firstTemplateId);
    expect(store.saveTemplate()).toBe(firstTemplateId);
    expect(store.getState().templates).toHaveLength(32);
    await store.clear();
  });

  test('serializes template-only workspaces and replaces templates on restore and clear', async () => {
    const {store} = makeStore();
    const workflowId = store.addWorkflow(makeFixtureDocument());
    store.saveTemplate();
    const templates = [...store.getState().templates];
    store.deleteWorkflow(workflowId);
    const saved = {schemaVersion: 1 as const, workflows: [], templates};
    expect(store.serialize()).toEqual(saved);
    store.restore(saved);
    expect(store.serialize()).toEqual(saved);
    expect(store.getState().templates).toEqual(templates);
    expect(store.getState().templates[0]).not.toBe(templates[0]);
    expect(store.getRunner()).toBeNull();
    store.restore({schemaVersion: 1, workflows: [makeFixtureDocument()]});
    expect(store.getState().templates).toEqual([]);
    store.restore(saved);
    await store.clear();
    expect(store.getState().templates).toEqual([]);
    expect(store.serialize()).toBeUndefined();
  });

  test('inserts a workflow in one edit, remaps connections and inputs, selects the block and undoes it', async () => {
    const {store} = makeStore();
    expect(store.insertWorkflow(makeFixtureDocument())).toBe(false);
    expect(store.getState().notice).toEqual({
      kind: 'error',
      message: 'Workflows: select a workflow before editing.'
    });
    const original = makeFixtureDocument();
    original.inputs = [
      {id: 'factor', label: 'Existing factor', kind: 'number', value: 2, maximum: 10}
    ];
    store.addWorkflow(original);
    const source = makeFixtureDocument();
    source.inputs = [
      {id: 'factor', label: 'Source factor', kind: 'number', value: 7, maximum: 100}
    ];
    source.steps[1].parameters['factor'] = {from: 'input', inputId: 'factor'};
    source.steps[1].displayStyle = {scaled: {colorColumn: 'value'}};
    let notifications = 0;
    const unsubscribe = store.subscribe(() => notifications++);
    expect(store.insertWorkflow(source, {x: 100, y: 200})).toBe(true);
    expect(notifications).toBe(1);
    expect(store.getState().selectedStepIds).toEqual(['source-2', 'scale-2', 'count-2']);
    expect(store.getState().document?.steps.slice(3)).toEqual([
      {...source.steps[0], id: 'source-2', position: {x: 100, y: 200}},
      {
        ...source.steps[1],
        id: 'scale-2',
        inputs: {points: {stepId: 'source-2', output: 'data'}},
        parameters: {factor: {from: 'input', inputId: 'factor-2'}},
        position: {x: 380, y: 200}
      },
      {
        ...source.steps[2],
        id: 'count-2',
        inputs: {rows: {stepId: 'scale-2', output: 'scaled'}},
        position: {x: 660, y: 200}
      }
    ]);
    expect(store.getState().document?.inputs).toEqual([
      original.inputs[0],
      {...source.inputs[0], id: 'factor-2'}
    ]);
    store.undo();
    expect(store.getState().document).toEqual(original);
    expect(store.getState().canUndo).toBe(false);
    expect(store.getState().selectedStepIds).toEqual([]);
    unsubscribe();
    await store.clear();
  });

  test('copies a detached selection across workflows and pastes under free ids with one undo entry', async () => {
    const {store} = makeStore();
    const source = makeFixtureDocument();
    source.inputs = [{id: 'factor', label: 'Factor', kind: 'number', value: 7, maximum: 100}];
    source.steps[1].parameters['factor'] = {from: 'input', inputId: 'factor'};
    source.steps[1].displayStyle = {scaled: {colorColumn: 'value'}};
    store.addWorkflow(source);
    store.select(['scale', 'count']);
    const original = store.getState().document;
    expect(store.copySelection()).toBe(true);
    expect(store.getState().clipboardStepCount).toBe(2);
    expect(store.getState().document).toBe(original);
    expect(store.getState().canUndo).toBe(false);
    store.edit('change-source', document => changeFactor(document, 9));
    const target = {...makeFixtureDocument(), id: 'target', title: 'Target', steps: []};
    target.inputs = [{...source.inputs[0], value: 99}];
    store.addWorkflow(target);
    let notifications = 0;
    const unsubscribe = store.subscribe(() => notifications++);
    expect(store.paste({x: 500, y: 600})).toBe(true);
    expect(notifications).toBe(1);
    expect(store.getState().selectedStepIds).toEqual(['scale', 'count']);
    expect(store.getState().document?.inputs).toEqual(target.inputs);
    const pasted = store.getState().document?.steps;
    expect(pasted).toEqual([
      {...source.steps[1], inputs: {}, display: undefined, position: {x: 500, y: 600}},
      {...source.steps[2], position: {x: 780, y: 600}}
    ]);
    store.undo();
    expect(store.getState().document).toEqual(target);
    expect(store.getState().canUndo).toBe(false);
    expect(store.getState().selectedStepIds).toEqual([]);
    expect(store.getState().clipboardStepCount).toBe(2);
    store.setActiveWorkflow(source.id);
    expect(store.paste()).toBe(true);
    expect(store.getState().selectedStepIds).toEqual(['scale-2', 'count-2']);
    expect(store.getState().document?.steps[3].position).toEqual({x: 320, y: 40});
    unsubscribe();
    await store.clear();
    expect(store.getState().clipboardStepCount).toBe(0);
  });

  test('cuts in one edit, keeps the clipboard through undo and notifies when a same-sized copy changes', async () => {
    const {store} = makeStore();
    store.addWorkflow(makeFixtureDocument());
    store.select(['scale']);
    expect(store.copySelection()).toBe(true);
    store.select(['count']);
    const beforeCopy = store.getState();
    expect(store.copySelection()).toBe(true);
    expect(store.getState().revision).toBe(beforeCopy.revision + 1);
    store.select(['source', 'scale']);
    const original = store.getState().document;
    let notifications = 0;
    const unsubscribe = store.subscribe(() => notifications++);
    expect(store.cutSelection()).toBe(true);
    expect(notifications).toBe(1);
    expect(store.getState().clipboardStepCount).toBe(2);
    expect(store.getState().document?.steps).toEqual([
      {...makeFixtureDocument().steps[2], inputs: {}}
    ]);
    expect(store.getState().selectedStepIds).toEqual([]);
    expect(store.copySelection()).toBe(false);
    expect(store.cutSelection()).toBe(false);
    store.undo();
    expect(store.getState().document).toEqual(original);
    expect(store.getState().canUndo).toBe(false);
    expect(store.getState().clipboardStepCount).toBe(2);
    store.setActiveWorkflow(null);
    expect(store.paste()).toBe(false);
    expect(store.getState().notice).toEqual({
      kind: 'error',
      message: 'Workflows: select a workflow before editing.'
    });
    unsubscribe();
    await store.clear();
  });

  test('composition refusals preserve the document, selection, history and prior clipboard', async () => {
    const {store} = makeStore();
    const malformed = makeFixtureDocument();
    malformed.steps[1].parameters['factor'] = {from: 'input', inputId: 'missing'};
    store.addWorkflow(malformed);
    store.select(['source']);
    expect(store.copySelection()).toBe(true);
    store.select(['scale']);
    const original = store.getState().document;
    expect(store.copySelection()).toBe(false);
    expect(store.cutSelection()).toBe(false);
    expect(store.getState().notice).toEqual({
      kind: 'error',
      message: 'Workflow input missing: The input does not exist.'
    });
    expect(store.getState().document).toBe(original);
    expect(store.getState().selectedStepIds).toEqual(['scale']);
    expect(store.getState().canUndo).toBe(false);
    expect(store.getState().clipboardStepCount).toBe(1);
    const full = {...makeFixtureDocument(), id: 'full', title: 'Full'};
    full.steps = Array.from({length: WORKFLOW_LIMITS.maximumSteps}, (_, index) => ({
      ...makeFixtureDocument().steps[0],
      id: `source-${index}`
    }));
    store.addWorkflow(full);
    store.select(['source-0']);
    const before = store.getState();
    expect(store.paste()).toBe(false);
    expect(store.getState().notice).toEqual({
      kind: 'error',
      message: 'Full: maximumSteps (64) is exceeded by total 65 (64 existing + 1 added).'
    });
    expect(store.insertWorkflow(makeFixtureDocument())).toBe(false);
    expect(store.getState().notice).toEqual({
      kind: 'error',
      message: 'Full: maximumSteps (64) is exceeded by total 67 (64 existing + 3 added).'
    });
    expect(store.getState().document).toBe(before.document);
    expect(store.getState().selectedStepIds).toEqual(['source-0']);
    expect(store.getState().canUndo).toBe(false);
    expect(store.getState().clipboardStepCount).toBe(1);
    const emptyId = store.createWorkflow({title: 'Empty'});
    expect(store.paste()).toBe(true);
    expect(store.getState().document?.id).toBe(emptyId);
    expect(store.getState().document?.steps.map(step => step.operationId)).toEqual([
      'fixture-source'
    ]);
    store.restore({schemaVersion: 1, workflows: []});
    expect(store.getState().clipboardStepCount).toBe(0);
    await store.clear();
  });

  test('F3 refuses a template whose generated names exceed the serialized document limit', async () => {
    const {store} = makeStore();
    const document = makeFixtureDocument();
    document.steps = [
      {
        ...document.steps[0],
        parameters: {
          first: {from: 'static', value: ''},
          second: {from: 'static', value: ''},
          third: {from: 'static', value: ''},
          fourth: {from: 'static', value: ''}
        }
      }
    ];
    const emptySize = JSON.stringify(keplerWorkflows.serializeWorkflowDocument(document)).length;
    let remaining = WORKFLOW_LIMITS.maximumDocumentCharacters - emptySize;
    for (const name of ['first', 'second', 'third', 'fourth']) {
      const length = Math.min(remaining, WORKFLOW_LIMITS.maximumTextCharacters);
      document.steps[0].parameters[name] = {from: 'static', value: 'a'.repeat(length)};
      remaining -= length;
    }
    expect(remaining).toBe(0);
    const parsed = keplerWorkflows.parseWorkflowDocument(document);
    expect(JSON.stringify(parsed)).toHaveLength(WORKFLOW_LIMITS.maximumDocumentCharacters);
    store.addWorkflow(parsed);
    expect(store.saveTemplate()).toBe('fixture-workflow');
    const before = store.getState();
    const saved = store.serialize();
    expect(store.saveTemplate()).toBe('');
    expect(store.getState()).toEqual({
      ...before,
      revision: before.revision + 1,
      notice: {
        kind: 'error',
        message: 'document: maximumDocumentCharacters is 1048576; got 1048580.'
      }
    });
    expect(store.getState().templates).toBe(before.templates);
    expect(store.serialize()).toEqual(saved);
    await store.clear();
  });

  test.each([
    'paste',
    'cut',
    'insert-workflow'
  ] as const)('F4 keeps consecutive %s actions separate in history while ordinary edits still coalesce', async action => {
    const {store} = makeStore();
    const now = vi.spyOn(Date, 'now').mockReturnValue(1000);
    const edit = vi.spyOn(store, 'edit');
    try {
      store.addWorkflow(makeFixtureDocument());
      const original = store.getState().document;
      const source = {...makeFixtureDocument(), steps: [makeFixtureDocument().steps[0]]};
      const applyAction = () => {
        if (action === 'paste') return store.paste();
        if (action === 'insert-workflow') return store.insertWorkflow(source);
        store.select([store.getState().document!.steps[0].id]);
        return store.cutSelection();
      };
      if (action === 'paste') {
        store.select(['source']);
        expect(store.copySelection()).toBe(true);
      }
      expect(applyAction()).toBe(true);
      const first = store.getState().document;
      expect(first?.steps).toHaveLength(action === 'cut' ? 2 : 4);
      expect(applyAction()).toBe(true);
      const second = store.getState().document;
      expect(second?.steps).toHaveLength(action === 'cut' ? 1 : 5);
      expect(edit.mock.calls.map(([tag]) => tag)).toEqual([action, action]);
      store.undo();
      expect(store.getState().document).toEqual(first);
      expect(store.getState().canUndo).toBe(true);
      store.undo();
      expect(store.getState().document).toEqual(original);
      expect(store.getState().canUndo).toBe(false);
      store.redo();
      expect(store.getState().document).toEqual(first);
      store.redo();
      expect(store.getState().document).toEqual(second);
      expect(store.getState().canRedo).toBe(false);
      store.edit('title', document => ({...document, title: 'First title'}));
      store.edit('title', document => ({...document, title: 'Second title'}));
      store.undo();
      expect(store.getState().document).toEqual(second);
    } finally {
      edit.mockRestore();
      now.mockRestore();
      await store.clear();
    }
  });
});
