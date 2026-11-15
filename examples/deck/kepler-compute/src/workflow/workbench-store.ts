// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  KeplerJSONValue,
  WorkflowClipboard,
  WorkflowCollection,
  WorkflowDocument,
  WorkflowHost,
  WorkflowIssue,
  WorkflowOperationRegistry,
  WorkflowRunOptions,
  WorkflowRunRecord,
  WorkflowRunnerState
} from '@luma-contributors/kepler';

import {
  copyWorkflowSteps,
  createWorkflowDocument,
  getWorkflowOperationRegistry,
  insertWorkflowSteps,
  isWorkflowId,
  makeUniqueWorkflowId,
  parseWorkflowDocument,
  pasteWorkflowSteps,
  removeWorkflowSteps,
  serializeWorkflowCollection,
  serializeWorkflowDocument,
  WorkflowError,
  WorkflowHistory,
  WorkflowRunner,
  WORKFLOW_LIMITS
} from '@luma-contributors/kepler';
import {WORKBENCH_DOCK} from './workbench-contract';

export type WorkbenchInspectorTab = 'parameters' | 'result' | 'contract';

export type WorkbenchDialog = 'templates' | 'import-export' | 'shortcuts' | null;

export type WorkbenchPaletteState = {
  open: boolean;
  /** Canvas coordinates where the new step goes; absent: right of the selection or at the centre. */
  position?: {x: number; y: number};
  /** Output the new step should read (dragging a connection into empty canvas). */
  connectFrom?: {stepId: string; output: string};
};

export type WorkbenchDraft = {
  document: WorkflowDocument;
  issues: readonly WorkflowIssue[];
  outline: string;
  source: 'assistant' | 'import' | 'history';
};

export type WorkbenchNotice = {kind: 'info' | 'error'; message: string};

/** Immutable snapshot; a new object whenever `revision` changes. */
export type WorkbenchState = {
  revision: number;
  workflows: readonly WorkflowDocument[];
  activeWorkflowId: string | null;
  /** The active document, or null when there is none. */
  document: WorkflowDocument | null;
  /** Runner state of the active workflow. */
  runner: WorkflowRunnerState | null;
  selectedStepIds: readonly string[];
  canUndo: boolean;
  canRedo: boolean;
  dock: {open: boolean; height: number; inspectorTab: WorkbenchInspectorTab; logOpen: boolean};
  palette: WorkbenchPaletteState;
  dialog: WorkbenchDialog;
  /** A workflow proposed by the assistant, an import or the history; nothing runs until accepted. */
  draft: WorkbenchDraft | null;
  notice: WorkbenchNotice | null;
  /** Run records of the active workflow, newest first, at most 20. */
  runLog: readonly WorkflowRunRecord[];
  /** Workflows the user saved as templates; saved with the workspace. */
  templates: readonly WorkflowDocument[];
  /** Number of steps on the workbench clipboard, 0 when it is empty. */
  clipboardStepCount: number;
};

/**
 * Application state of the workbench: the workflows of the workspace, one runner and one undo
 * history per workflow, selection and dock state. Plain TypeScript, no React, so it is tested in
 * node with a fake host.
 */
export type WorkbenchStore = {
  readonly registry: WorkflowOperationRegistry;
  readonly host: WorkflowHost;
  getState(): WorkbenchState;
  /** Subscriber failures are reported as error notices after the committed change is delivered. */
  subscribe(listener: () => void): () => void;

  // Workflows
  /** Creates an empty workflow, makes it active and returns its id. */
  createWorkflow(options?: {title?: string}): string;
  /** Adds a document (template, import, accepted draft) under a free id, makes it active. */
  addWorkflow(document: WorkflowDocument): string;
  duplicateWorkflow(workflowId: string): string;
  /** Destroys the workflow's runner (retracting its published outputs) and removes it. */
  deleteWorkflow(workflowId: string): void;
  setActiveWorkflow(workflowId: string | null): void;

  // Editing the active workflow
  /**
   * Applies an edit to the active document: pushes it on the undo history under `tag`, hands it
   * to the runner and drops selected ids that no longer exist. An edit that throws leaves the
   * document unchanged and sets an error notice with the message; `edit` returns false.
   */
  edit(tag: string, change: (document: WorkflowDocument) => WorkflowDocument): boolean;
  undo(): void;
  redo(): void;
  select(stepIds: readonly string[]): void;

  // Running
  run(options?: WorkflowRunOptions): Promise<WorkflowRunRecord | null>;
  cancel(): void;
  /** The host calls this when its datasets, filters or selection change. */
  notifyExternalChange(): void;
  /** Runner of the active workflow (for result data that is not part of the snapshot). */
  getRunner(): WorkflowRunner | null;

  // Interface state
  setDock(patch: Partial<WorkbenchState['dock']>): void;
  openPalette(options?: Omit<WorkbenchPaletteState, 'open'>): void;
  closePalette(): void;
  setDialog(dialog: WorkbenchDialog): void;
  setNotice(notice: WorkbenchNotice | null): void;
  proposeDraft(draft: WorkbenchDraft): void;
  /** Adds the draft as a workflow and opens the dock; returns its id. */
  acceptDraft(): string | null;
  rejectDraft(): void;

  // Composition
  /** Saves a copy of a workflow (default: the active one) as a template and returns its id. */
  saveTemplate(workflowId?: string): string;
  deleteTemplate(templateId: string): void;
  /** Adds a workflow from a user template and makes it active; returns its id. */
  useTemplate(templateId: string): string;
  /** Inlines every step of another document into the active workflow and selects them (one undo step). */
  insertWorkflow(source: WorkflowDocument, position?: {x: number; y: number}): boolean;
  /** Copies the selected steps to the workbench clipboard; false when nothing is selected. */
  copySelection(): boolean;
  /** Copies and removes the selected steps (one undo step). */
  cutSelection(): boolean;
  /** Pastes the clipboard into the active workflow and selects the copies (one undo step). */
  paste(position?: {x: number; y: number}): boolean;

  // Persistence
  /** The `settings.extensions.workflows` member; undefined when there are no workflows. */
  serialize(): KeplerJSONValue | undefined;
  /** Replaces every workflow (restore replaces the workspace; it never merges). */
  restore(collection: WorkflowCollection): void;
  /** Removes every workflow and retracts every published output. */
  clear(): Promise<void>;
};

type WorkflowSession = {
  workflowId: string;
  runner: WorkflowRunner;
  history: WorkflowHistory;
  runLog: readonly WorkflowRunRecord[];
};

function readErrorMessage(error: unknown): string {
  try {
    return String(error instanceof Error ? error.message : error);
  } catch {
    return 'Workflows: an error occurred, but its message could not be read.';
  }
}

function copyAndFreeze<Value>(value: Value): Value {
  const copy = structuredClone(value);
  function freezeValue(child: unknown): void {
    if (child && typeof child === 'object' && !Object.isFrozen(child)) {
      Object.freeze(child);
      for (const nested of Object.values(child)) freezeValue(nested);
    }
  }
  freezeValue(copy);
  return copy;
}

export function createWorkbenchStore(options: {
  host: WorkflowHost;
  registry?: WorkflowOperationRegistry;
}): WorkbenchStore {
  let registry = options.registry;
  const sessions = new Map<string, WorkflowSession>();
  const destructions = new Set<Promise<void>>();
  const listeners = new Set<() => void>();
  const notifications: Array<Array<() => void>> = [];
  let notifying = false;
  let mutationDepth = 0;
  let editing = false;
  let editMutationError: WorkflowError | undefined;
  let unsubscribeRunner: (() => void) | undefined;
  let workflows: readonly WorkflowDocument[] = [];
  let templates: readonly WorkflowDocument[] = [];
  let clipboard: WorkflowClipboard | null = null;
  let snapshotClipboard: WorkflowClipboard | null = null;
  let activeWorkflowId: string | null = null;
  let selectedStepIds: readonly string[] = [];
  let dock: WorkbenchState['dock'] = {
    open: false,
    height: WORKBENCH_DOCK.defaultHeight,
    inspectorTab: 'parameters',
    logOpen: false
  };
  let palette: WorkbenchPaletteState = {open: false};
  let dialog: WorkbenchDialog = null;
  let draft: WorkbenchDraft | null = null;
  let notice: WorkbenchNotice | null = null;
  const emptyRunLog: readonly WorkflowRunRecord[] = [];
  let state: WorkbenchState = {
    revision: 0,
    workflows,
    activeWorkflowId,
    document: null,
    runner: null,
    selectedStepIds,
    canUndo: false,
    canRedo: false,
    dock,
    palette,
    dialog,
    draft,
    notice,
    runLog: emptyRunLog,
    templates,
    clipboardStepCount: 0
  };

  function getRegistry(): WorkflowOperationRegistry {
    registry ??= getWorkflowOperationRegistry();
    return registry;
  }

  function getSession(): WorkflowSession | undefined {
    return activeWorkflowId === null ? undefined : sessions.get(activeWorkflowId);
  }

  function getDocument(): WorkflowDocument | null {
    return workflows.find(document => document.id === activeWorkflowId) ?? null;
  }

  function publishState(): void {
    if (mutationDepth) return;
    const session = getSession();
    const next: WorkbenchState = {
      revision: state.revision,
      workflows,
      activeWorkflowId,
      document: getDocument(),
      runner: session?.runner.getState() ?? null,
      selectedStepIds,
      canUndo: session?.history.canUndo ?? false,
      canRedo: session?.history.canRedo ?? false,
      dock,
      palette,
      dialog,
      draft,
      notice,
      runLog: session?.runLog ?? emptyRunLog,
      templates,
      clipboardStepCount: clipboard?.steps.length ?? 0
    };
    if (
      clipboard === snapshotClipboard &&
      (Object.keys(next) as Array<keyof WorkbenchState>).every(key => next[key] === state[key])
    )
      return;
    // A different copy is observable on paste even when it contains the same number of steps.
    snapshotClipboard = clipboard;
    state = {...next, revision: state.revision + 1};
    notifications.push([...listeners]);
    if (notifying) return;
    notifying = true;
    let reportedSubscriberFailure = false;
    try {
      while (notifications.length) {
        let subscriberFailure: string | undefined;
        for (const listener of notifications.shift()!) {
          if (!listeners.has(listener)) continue;
          try {
            listener();
          } catch (error) {
            subscriberFailure ??= readErrorMessage(error);
          }
        }
        // Report once per delivery cycle; the failing subscriber also receives that notice.
        if (subscriberFailure !== undefined && !reportedSubscriberFailure) {
          reportedSubscriberFailure = true;
          setError(`Workflows: subscriber failed: ${subscriberFailure}`);
          publishState();
        }
      }
    } finally {
      notifying = false;
    }
  }

  function requireMutationAllowed(): void {
    if (!editing) return;
    editMutationError ??= new WorkflowError(
      'malformed',
      `${getDocument()?.title ?? 'Workflows'}: edit callbacks cannot change the workbench; return an edited document instead.`
    );
    throw editMutationError;
  }

  // A document edit and its synchronous runner notification form one observable change.
  function mutate<Result>(change: () => Result): Result {
    requireMutationAllowed();
    mutationDepth++;
    try {
      return change();
    } finally {
      mutationDepth--;
      publishState();
    }
  }

  function setError(error: unknown): void {
    const message = readErrorMessage(error);
    if (notice?.kind !== 'error' || notice.message !== message)
      notice = Object.freeze({kind: 'error', message});
  }

  function pruneSelection(): void {
    const existing = new Set(getDocument()?.steps.map(step => step.id));
    const retained = selectedStepIds.filter(stepId => existing.has(stepId));
    if (retained.length !== selectedStepIds.length) selectedStepIds = retained;
  }

  function activate(workflowId: string | null): void {
    unsubscribeRunner?.();
    unsubscribeRunner = undefined;
    activeWorkflowId = workflowId;
    if (selectedStepIds.length) selectedStepIds = [];
    const document = getDocument();
    if (!document) return;
    let session = sessions.get(document.id);
    if (!session) {
      const runner = new WorkflowRunner({host: options.host, registry: getRegistry()});
      session = {
        workflowId: document.id,
        runner,
        history: new WorkflowHistory(document),
        runLog: []
      };
      sessions.set(document.id, session);
      runner.setDocument(document);
    } else {
      session.runner.refreshExternal();
    }
    unsubscribeRunner = session.runner.subscribe(publishState);
  }

  function destroySession(session: WorkflowSession): void {
    const destruction = session.runner.destroy().catch(error => {
      mutate(() => setError(error));
    });
    destructions.add(destruction);
    void destruction.then(() => destructions.delete(destruction));
  }

  function canAddWorkflow(): boolean {
    if (workflows.length < WORKFLOW_LIMITS.maximumWorkflows) return true;
    setError(`Workflows: maximumWorkflows is ${WORKFLOW_LIMITS.maximumWorkflows}.`);
    // UI calls signal a refused addition with an empty id, so callers never select another workflow.
    return false;
  }

  function addDocument(document: WorkflowDocument): string {
    if (!canAddWorkflow()) return '';
    const titles = new Set(workflows.map(workflow => workflow.title));
    const baseTitle = document.title.slice(0, WORKFLOW_LIMITS.maximumTitleCharacters);
    let title = baseTitle;
    for (let number = 2; titles.has(title); number++) {
      const suffix = ` ${number}`;
      title = baseTitle.slice(0, WORKFLOW_LIMITS.maximumTitleCharacters - suffix.length) + suffix;
    }
    const id = makeUniqueWorkflowId(
      document.id,
      workflows.map(workflow => workflow.id)
    );
    const added = copyAndFreeze({...document, id, title});
    workflows = [...workflows, added];
    activate(id);
    if (!dock.open) dock = {...dock, open: true};
    return id;
  }

  function validateDocumentId(document: WorkflowDocument, session: WorkflowSession): void {
    if (!isWorkflowId(document.id))
      throw new WorkflowError(
        'malformed',
        `${document.title}: Id must use lowercase letters, digits and hyphens and contain at most ${WORKFLOW_LIMITS.maximumIdCharacters} characters.`
      );
    if (
      workflows.some(workflow => workflow.id === document.id && workflow.id !== session.workflowId)
    )
      throw new WorkflowError(
        'duplicate-id',
        `${document.title}: Id "${document.id}" must be unique among workflows.`
      );
  }

  function replaceDocument(document: WorkflowDocument, session: WorkflowSession): void {
    const previousId = session.workflowId;
    workflows = workflows.map(workflow => (workflow.id === previousId ? document : workflow));
    if (document.id !== previousId) {
      sessions.delete(previousId);
      sessions.set(document.id, session);
      session.workflowId = document.id;
      if (activeWorkflowId === previousId) activeWorkflowId = document.id;
    }
    session.runner.setDocument(document);
    pruneSelection();
  }

  function moveHistory(direction: 'undo' | 'redo'): void {
    const session = getSession();
    if (!session?.history[direction]()) return;
    try {
      validateDocumentId(session.history.document, session);
    } catch (error) {
      // The old id may have been reused after a rename. Keep both history stacks intact.
      session.history[direction === 'undo' ? 'redo' : 'undo']();
      setError(error);
      return;
    }
    replaceDocument(session.history.document, session);
  }

  function replaceCollection(collection: WorkflowCollection): void {
    unsubscribeRunner?.();
    unsubscribeRunner = undefined;
    for (const session of sessions.values()) destroySession(session);
    sessions.clear();
    if (workflows.length || collection.workflows.length)
      workflows = collection.workflows.map(copyAndFreeze);
    if (templates.length || collection.templates?.length)
      templates = (collection.templates ?? []).map(copyAndFreeze);
    clipboard = null;
    draft = null;
    notice = null;
    const restoredId = workflows.some(document => document.id === collection.activeWorkflowId)
      ? collection.activeWorkflowId!
      : (workflows[0]?.id ?? null);
    activate(restoredId);
  }

  async function runWorkflow(runOptions?: WorkflowRunOptions): Promise<WorkflowRunRecord | null> {
    const session = getSession();
    if (!session || activeWorkflowId === null) return null;
    const record = await session.runner.run(runOptions);
    mutate(() => {
      // A completed run must not repopulate a deleted or restored workspace.
      if (sessions.get(session.workflowId) !== session) return;
      session.runLog = [record, ...session.runLog].slice(0, 20);
      if (record.status === 'error') {
        const failure = record.steps.find(step => step.status === 'error')?.error;
        setError(failure?.message ?? `${session.history.document.title}: workflow run failed.`);
      }
    });
    return record;
  }

  const store: WorkbenchStore = {
    get registry() {
      return getRegistry();
    },
    host: options.host,
    getState: () => state,
    subscribe(listener) {
      requireMutationAllowed();
      listeners.add(listener);
      return () => {
        requireMutationAllowed();
        listeners.delete(listener);
      };
    },
    createWorkflow(workflowOptions) {
      return mutate(() => {
        if (!canAddWorkflow()) return '';
        try {
          return addDocument(createWorkflowDocument(workflowOptions));
        } catch (error) {
          setError(error);
          return '';
        }
      });
    },
    addWorkflow(document) {
      return mutate(() => addDocument(document));
    },
    duplicateWorkflow(workflowId) {
      return mutate(() => {
        const document = workflows.find(workflow => workflow.id === workflowId);
        if (!document) {
          setError(`Workflows: workflow "${workflowId}" does not exist.`);
          return '';
        }
        return addDocument(document);
      });
    },
    deleteWorkflow(workflowId) {
      mutate(() => {
        if (!workflows.some(document => document.id === workflowId)) return;
        const session = sessions.get(workflowId);
        if (activeWorkflowId === workflowId) {
          unsubscribeRunner?.();
          unsubscribeRunner = undefined;
        }
        sessions.delete(workflowId);
        if (session) destroySession(session);
        workflows = workflows.filter(document => document.id !== workflowId);
        if (activeWorkflowId === workflowId) activate(workflows[0]?.id ?? null);
      });
    },
    setActiveWorkflow(workflowId) {
      mutate(() => {
        if (workflowId === activeWorkflowId) return;
        if (workflowId !== null && !workflows.some(document => document.id === workflowId)) {
          setError(`Workflows: workflow "${workflowId}" does not exist.`);
          return;
        }
        activate(workflowId);
      });
    },
    edit(tag, change) {
      return mutate(() => {
        const document = getDocument();
        const session = getSession();
        if (!document || !session) {
          setError('Workflows: select a workflow before editing.');
          return false;
        }
        let next: WorkflowDocument;
        try {
          editing = true;
          editMutationError = undefined;
          try {
            next = change(document);
            if (next !== document) {
              next = copyAndFreeze(next);
              validateDocumentId(next, session);
            }
            if (editMutationError) throw editMutationError;
          } finally {
            editing = false;
          }
          if (next === document) return true;
        } catch (error) {
          setError(error);
          return false;
        }
        // Each composition command needs its own undo entry even when commands share a tag.
        const discrete = tag === 'paste' || tag === 'cut' || tag === 'insert-workflow';
        session.history.apply(next, discrete ? undefined : tag);
        replaceDocument(next, session);
        if (notice?.kind === 'error') notice = null;
        return true;
      });
    },
    undo() {
      mutate(() => moveHistory('undo'));
    },
    redo() {
      mutate(() => moveHistory('redo'));
    },
    select(stepIds) {
      mutate(() => {
        const existing = new Set(getDocument()?.steps.map(step => step.id));
        const selected = [...new Set(stepIds)].filter(stepId => existing.has(stepId));
        if (
          selected.length !== selectedStepIds.length ||
          selected.some((stepId, index) => stepId !== selectedStepIds[index])
        )
          selectedStepIds = selected;
      });
    },
    run(runOptions) {
      requireMutationAllowed();
      return runWorkflow(runOptions);
    },
    cancel() {
      mutate(() => getSession()?.runner.cancel());
    },
    notifyExternalChange() {
      mutate(() => getSession()?.runner.refreshExternal());
    },
    getRunner: () => getSession()?.runner ?? null,
    setDock(patch) {
      mutate(() => {
        const height = patch.height ?? dock.height;
        const next = {
          ...dock,
          ...patch,
          height: Number.isNaN(height)
            ? dock.height
            : Math.max(WORKBENCH_DOCK.minimumHeight, Math.min(900, height))
        };
        if ((Object.keys(next) as Array<keyof typeof dock>).some(key => next[key] !== dock[key]))
          dock = next;
      });
    },
    openPalette(paletteOptions) {
      mutate(() => {
        const next = copyAndFreeze({...paletteOptions, open: true});
        if (
          !palette.open ||
          palette.position?.x !== next.position?.x ||
          palette.position?.y !== next.position?.y ||
          palette.connectFrom?.stepId !== next.connectFrom?.stepId ||
          palette.connectFrom?.output !== next.connectFrom?.output
        )
          palette = next;
        if (!dock.open) dock = {...dock, open: true};
      });
    },
    closePalette() {
      mutate(() => {
        if (palette.open) palette = {open: false};
      });
    },
    setDialog(nextDialog) {
      mutate(() => {
        dialog = nextDialog;
      });
    },
    setNotice(nextNotice) {
      mutate(() => {
        if (notice?.kind !== nextNotice?.kind || notice?.message !== nextNotice?.message)
          notice = nextNotice === null ? null : copyAndFreeze(nextNotice);
      });
    },
    proposeDraft(nextDraft) {
      mutate(() => {
        draft = copyAndFreeze(nextDraft);
      });
    },
    acceptDraft() {
      return mutate(() => {
        if (!draft) return null;
        const id = addDocument(draft.document);
        draft = null;
        if (!dock.open) dock = {...dock, open: true};
        return id;
      });
    },
    rejectDraft() {
      mutate(() => {
        draft = null;
      });
    },
    saveTemplate(workflowId) {
      return mutate(() => {
        const document = workflows.find(
          workflow => workflow.id === (workflowId ?? activeWorkflowId)
        );
        if (!document) {
          setError(
            workflowId === undefined
              ? 'Templates: select a workflow before saving a template.'
              : `Templates: workflow "${workflowId}" does not exist.`
          );
          return '';
        }
        if (templates.length >= WORKFLOW_LIMITS.maximumUserTemplates) {
          setError(`Templates: maximumUserTemplates is ${WORKFLOW_LIMITS.maximumUserTemplates}.`);
          return '';
        }
        try {
          const template = parseWorkflowDocument(serializeWorkflowDocument(document));
          template.id = makeUniqueWorkflowId(
            template.id,
            templates.map(saved => saved.id)
          );
          const titles = new Set(templates.map(saved => saved.title));
          const baseTitle = template.title;
          for (let number = 2; titles.has(template.title); number++) {
            const suffix = ` ${number}`;
            template.title =
              baseTitle.slice(0, WORKFLOW_LIMITS.maximumTitleCharacters - suffix.length) + suffix;
          }
          // Generated names count toward the serialized document size limit.
          serializeWorkflowDocument(template);
          templates = [...templates, copyAndFreeze(template)];
          notice = Object.freeze({
            kind: 'info',
            message: `Saved "${template.title}" as a template.`
          });
          return template.id;
        } catch (error) {
          setError(error);
          return '';
        }
      });
    },
    deleteTemplate(templateId) {
      mutate(() => {
        if (templates.some(template => template.id === templateId))
          templates = templates.filter(template => template.id !== templateId);
      });
    },
    useTemplate(templateId) {
      return mutate(() => {
        const template = templates.find(saved => saved.id === templateId);
        if (!template) {
          setError(`Templates: template "${templateId}" does not exist.`);
          return '';
        }
        return addDocument(template);
      });
    },
    insertWorkflow(source, position) {
      return mutate(() => {
        let stepIds: string[] = [];
        const inserted = store.edit('insert-workflow', document => {
          const result = insertWorkflowSteps(document, getRegistry(), source, position);
          stepIds = result.stepIds;
          return result.document;
        });
        if (inserted) store.select(stepIds);
        return inserted;
      });
    },
    copySelection() {
      return mutate(() => {
        const document = getDocument();
        if (!document || !selectedStepIds.length) return false;
        try {
          clipboard = copyAndFreeze(copyWorkflowSteps(document, selectedStepIds));
          return true;
        } catch (error) {
          setError(error);
          return false;
        }
      });
    },
    cutSelection() {
      return mutate(() => {
        if (!getDocument() || !selectedStepIds.length) return false;
        let copied: WorkflowClipboard | null = null;
        const removed = store.edit('cut', document => {
          copied = copyAndFreeze(copyWorkflowSteps(document, selectedStepIds));
          return removeWorkflowSteps(document, selectedStepIds);
        });
        if (removed) clipboard = copied;
        return removed;
      });
    },
    paste(position) {
      return mutate(() => {
        if (!clipboard) return false;
        const copied = clipboard;
        let stepIds: string[] = [];
        const pasted = store.edit('paste', document => {
          const result = pasteWorkflowSteps(document, getRegistry(), copied, position);
          stepIds = result.stepIds;
          return result.document;
        });
        if (pasted) store.select(stepIds);
        return pasted;
      });
    },
    serialize() {
      return serializeWorkflowCollection({
        schemaVersion: 1,
        workflows: [...workflows],
        templates: [...templates],
        ...(activeWorkflowId === null ? {} : {activeWorkflowId})
      });
    },
    restore(collection) {
      mutate(() => replaceCollection(collection));
    },
    clear() {
      mutate(() => replaceCollection({schemaVersion: 1, workflows: []}));
      return Promise.all([...destructions]).then(() => {});
    }
  };
  return store;
}
