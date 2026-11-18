// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateCatalogRasterInput} from '../analysis/catalog-raster-input';
import {KeplerAllocationError} from '../runtime/dataset-registry';
import {countWorkflowVertices, getWorkflowDataKind} from './workflow-data';
import {getWorkflowPortCompatibility} from './workflow-operations';
import {resolveWorkflowParameterOverrides, planWorkflow} from './workflow-plan';
import {WorkflowError} from './workflow-types';
import type {
  WorkflowData,
  WorkflowDisplayStyle,
  WorkflowDocument,
  WorkflowEngineChoice,
  WorkflowHost,
  WorkflowIssue,
  WorkflowOperation,
  WorkflowOperationRegistry,
  WorkflowPlan,
  WorkflowPortKind,
  WorkflowPublishedOutput,
  WorkflowRunOptions,
  WorkflowRunRecord,
  WorkflowRunRecordStep,
  WorkflowRunnerState,
  WorkflowStep,
  WorkflowStepResult,
  WorkflowStepState
} from './workflow-types';

export type WorkflowRunnerOptions = {
  host: WorkflowHost;
  registry: WorkflowOperationRegistry;
  /** Results kept besides the latest result of each step. Default 32. */
  maximumCachedResults?: number;
  /** Total rows across cached results; least recently used non-latest results go first. Default 2,000,000. */
  maximumCachedRows?: number;
  /** Delay before an automatic run when `settings.autoRun` is on. Default 400. */
  autoRunDelayMilliseconds?: number;
  now?: () => number;
};

type CachedResult = {result: WorkflowStepResult; owners: Set<string>};
type StyledPublishedOutput = WorkflowPublishedOutput & {style?: WorkflowDisplayStyle};
type RunTask = {
  document: WorkflowDocument | null;
  plan: WorkflowPlan | null;
  identityPlan: WorkflowPlan | null;
  options: WorkflowRunOptions;
  controller: AbortController;
  record: WorkflowRunRecord;
  selected: Set<string>;
  completed: Set<string>;
  device: ReturnType<WorkflowHost['getDevice']>;
  runtimeRevisionKey: string | undefined;
};

function freezeValue<Value>(value: Value): Value {
  // Decoded raster views are immutable by contract; JavaScript cannot freeze nonempty views.
  if (ArrayBuffer.isView(value)) return value;
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const child of Object.values(value)) freezeValue(child);
    Object.freeze(value);
  }
  return value;
}

/** Cached computations do not own the search history of their consuming steps. */
function applyStepSensitivity(
  result: WorkflowStepResult,
  step: Pick<WorkflowStep, 'sensitivity'>
): WorkflowStepResult {
  if (JSON.stringify(result.sensitivity) === JSON.stringify(step.sensitivity)) return result;
  const rebound = {...result};
  delete rebound.sensitivity;
  if (step.sensitivity) rebound.sensitivity = structuredClone(step.sensitivity);
  return freezeValue(rebound);
}

function readMessage(error: unknown): string {
  try {
    return String(error instanceof Error ? error.message : error);
  } catch {
    return 'Operation failed with an unreadable error.';
  }
}

function getDisplayStyleKey(style: WorkflowDisplayStyle | undefined): string {
  return JSON.stringify([style?.colorColumn, style?.colorScale, style?.palette]);
}

/**
 * Runs one workflow and owns its results.
 *
 * Step status is derived: the plan gives each step a content key, and a step is `done` when it
 * holds a result for that key, `stale` when it holds one for another key. Editing the document
 * or a change in the host's data therefore invalidates exactly the steps whose key changed and
 * everything downstream. Removal and runtime replacement also invalidate the old lineage,
 * so affected dependents must recompute even if their content keys are unchanged.
 *
 * Steps run one at a time in topological order (the GPU device is shared). A failed step stops
 * its dependents; independent branches continue. Outputs are checked against the consumer's
 * declared row and vertex capacity before they are handed on. Displayed outputs are published
 * through `host.publishOutput` after the step succeeds; a publish failure is a warning on the
 * step, not a step error.
 */
export class WorkflowRunner {
  private readonly host: WorkflowHost;
  private readonly registry: WorkflowOperationRegistry;
  private readonly now: () => number;
  private readonly maximumCachedResults: number;
  private readonly maximumCachedRows: number;
  private readonly autoRunDelayMilliseconds: number;
  private currentDocument: WorkflowDocument | null = null;
  private plan: WorkflowPlan | null = null;
  private state: WorkflowRunnerState = freezeValue({
    revision: 0,
    running: false,
    order: [],
    issues: [],
    steps: {}
  });
  private readonly latest = new Map<string, WorkflowStepResult>();
  private readonly cache = new Map<string, CachedResult>();
  private readonly runtimeIssues = new Map<string, Map<string, WorkflowIssue[]>>();
  private readonly published = new Map<string, Record<string, StyledPublishedOutput>>();
  private readonly documentIssues: WorkflowIssue[] = [];
  private readonly listeners = new Set<() => void>();
  private readonly notifications: Array<Array<() => void>> = [];
  private notifying = false;
  private destroyed = false;
  private sequence = 0;
  private activeTask?: RunTask;
  private readonly tasks = new Set<RunTask>();
  private tail?: Promise<WorkflowRunRecord>;
  private runningStepId?: string;
  private lastRun?: WorkflowRunRecord;
  private autoRunTimer?: ReturnType<typeof setTimeout>;
  // Publishing is serialized independently of execution, including display changes during a run.
  private publishTail: Promise<void> = Promise.resolve();
  private readonly pendingPublications = new Set<string>();
  private readonly publicationChecks = new Map<AbortController, () => boolean>();
  // Removal/runtime replacement break lineage even when content keys are unchanged.
  private readonly invalidatedSteps = new Set<string>();
  private runtimeRevisionKey: string | undefined;
  private publicationGeneration = 0;
  private readonly publicationGenerations = new WeakMap<StyledPublishedOutput, number>();

  constructor(options: WorkflowRunnerOptions) {
    this.host = options.host;
    this.runtimeRevisionKey = this.host.getRuntimeRevisionKey?.();
    this.registry = options.registry;
    this.now = options.now ?? Date.now;
    this.maximumCachedResults = Math.max(0, options.maximumCachedResults ?? 32);
    this.maximumCachedRows = Math.max(0, options.maximumCachedRows ?? 2000000);
    this.autoRunDelayMilliseconds = options.autoRunDelayMilliseconds ?? 400;
  }

  get document(): WorkflowDocument | null {
    return this.currentDocument;
  }

  /** Replaces the document, re-plans and re-derives every status. Keeps results that still match. */
  setDocument(document: WorkflowDocument): void {
    if (this.destroyed) return;
    const previous = this.currentDocument;
    this.currentDocument = freezeValue(structuredClone(document));
    const retained = new Set(document.steps.map(step => step.id));
    const changedWorkflow = previous !== null && previous.id !== document.id;
    if (changedWorkflow) this.invalidatedSteps.clear();
    else if (previous) {
      const removed = new Set(
        previous.steps.filter(step => !retained.has(step.id)).map(step => step.id)
      );
      // Walk the old edges, since an edit may also have removed the dangling connections.
      let expanded: boolean;
      do {
        expanded = false;
        for (const step of previous.steps) {
          if (
            removed.has(step.id) ||
            !Object.values(step.inputs).some(input => removed.has(input.stepId))
          )
            continue;
          removed.add(step.id);
          expanded = true;
        }
      } while (expanded);
      for (const stepId of removed) {
        if (retained.has(stepId)) this.invalidatedSteps.add(stepId);
      }
      for (const [key, entry] of this.cache) {
        for (const stepId of removed) entry.owners.delete(stepId);
        if (!entry.owners.size) this.cache.delete(key);
      }
    }
    for (const stepId of this.latest.keys()) {
      if (changedWorkflow || !retained.has(stepId)) this.dropStepResults(stepId);
    }
    for (const [key, entry] of this.cache) {
      for (const stepId of entry.owners) {
        if (changedWorkflow || !retained.has(stepId)) entry.owners.delete(stepId);
      }
      if (!entry.owners.size) this.cache.delete(key);
    }
    for (const stepId of this.runtimeIssues.keys()) {
      if (changedWorkflow || !retained.has(stepId)) this.runtimeIssues.delete(stepId);
    }
    this.replan();
    this.reconcileDisplay(changedWorkflow);
    this.updateState();
    this.publishDoneSteps();
    this.scheduleAutoRun();
  }

  /** Current state; a new object whenever `revision` changes, so it suits useSyncExternalStore. */
  getState(): WorkflowRunnerState {
    return this.state;
  }

  subscribe(listener: () => void): () => void {
    if (this.destroyed) return () => {};
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /** Re-reads external revision keys (host datasets, filters, selection) and re-derives statuses. */
  refreshExternal(): void {
    if (this.destroyed || !this.currentDocument) return;
    this.replan();
    this.updateState();
    this.publishDoneSteps();
    this.scheduleAutoRun();
  }

  /**
   * Runs every step that is not `done` (or only what `target` needs). Resolves with the run
   * record; never rejects for a step failure. A second call while running cancels the first.
   */
  run(options: WorkflowRunOptions = {}): Promise<WorkflowRunRecord> {
    if (this.destroyed) {
      return Promise.resolve(
        this.state.lastRun ?? {
          schemaVersion: 1,
          runId: `${this.currentDocument?.id ?? 'workflow'}-${this.sequence}`,
          executionId: globalThis.crypto.randomUUID(),
          workflowId: this.currentDocument?.id ?? '',
          startedAt: 0,
          durationMilliseconds: 0,
          status: 'cancelled',
          steps: []
        }
      );
    }
    this.clearAutoRunTimer();
    // Supersede executions, but keep independent display requests for still-current results.
    for (const task of this.tasks) task.controller.abort();
    this.cancelObsoleteWork();
    this.updateState();
    const document = this.currentDocument;
    const plan = document ? this.createPlan(document) : null;
    if (document === this.currentDocument) this.plan = plan;
    const task: RunTask = {
      document,
      plan,
      identityPlan: document
        ? planWorkflow(document, {registry: this.registry, host: this.host})
        : null,
      options: {...options},
      controller: new AbortController(),
      record: {
        schemaVersion: 1,
        runId: `${document?.id ?? 'workflow'}-${++this.sequence}`,
        executionId: globalThis.crypto.randomUUID(),
        workflowId: document?.id ?? '',
        ...(options.target === undefined ? {} : {target: options.target}),
        startedAt: this.now(),
        durationMilliseconds: 0,
        status: 'done',
        steps: []
      },
      selected: new Set(),
      completed: new Set(),
      device: this.host.getDevice(),
      runtimeRevisionKey: this.host.getRuntimeRevisionKey?.()
    };
    this.tasks.add(task);
    const previous = this.tail;
    const promise = (previous ?? Promise.resolve()).then(
      () => this.executeRun(task),
      () => this.executeRun(task)
    );
    this.tail = promise;
    const releaseTail = (): void => {
      if (this.tail === promise) this.tail = undefined;
    };
    void promise.then(releaseTail, releaseTail);
    return promise;
  }

  /** Aborts the run in progress; the running step finishes or is abandoned at its next await. */
  cancel(): void {
    if (this.destroyed) return;
    for (const task of this.tasks) task.controller.abort();
    for (const controller of this.publicationChecks.keys()) controller.abort();
    this.updateState();
  }

  /** Drops cached results of one step, or of all steps. */
  clearResults(stepId?: string): void {
    if (this.destroyed) return;
    if (stepId === undefined) {
      this.latest.clear();
      this.cache.clear();
      this.runtimeIssues.clear();
      this.invalidatedSteps.clear();
    } else {
      this.dropStepResults(stepId);
      this.runtimeIssues.delete(stepId);
    }
    this.replan();
    this.updateState();
  }

  /** Cancels, retracts published outputs and releases results. */
  async destroy(): Promise<void> {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const task of this.tasks) task.controller.abort();
    for (const controller of this.publicationChecks.keys()) controller.abort();
    this.clearAutoRunTimer();
    this.listeners.clear();
    this.notifications.length = 0;
    this.latest.clear();
    this.cache.clear();
    this.runtimeIssues.clear();
    await this.publishTail;
    const outputs = [...this.published.values()].flatMap(Object.values);
    this.published.clear();
    await Promise.all(outputs.map(output => this.retractOutput(output.datasetId)));
  }

  private createPlan(document: WorkflowDocument): WorkflowPlan {
    const observedKinds: Record<string, Record<string, WorkflowPortKind>> = {};
    for (const [stepId, result] of this.latest) {
      observedKinds[stepId] = Object.fromEntries(
        Object.entries(result.outputs).map(([name, data]) => [name, data.kind])
      );
    }
    return planWorkflow(document, {registry: this.registry, host: this.host, observedKinds});
  }

  private replan(): void {
    if (!this.currentDocument) return;
    const runtimeRevisionKey = this.host.getRuntimeRevisionKey?.();
    if (runtimeRevisionKey !== this.runtimeRevisionKey) {
      this.runtimeRevisionKey = runtimeRevisionKey;
      this.publicationGeneration++;
      this.cache.clear();
      for (const stepId of this.latest.keys()) this.invalidatedSteps.add(stepId);
    }
    let restored: boolean;
    do {
      restored = false;
      this.plan = this.createPlan(this.currentDocument);
      for (const stepId of this.plan.order) {
        const planned = this.plan.steps[stepId];
        const key = planned.key;
        const step = this.currentDocument.steps.find(candidate => candidate.id === stepId)!;
        let latest = this.latest.get(stepId);
        if (latest) {
          latest = applyStepSensitivity(latest, step);
          this.latest.set(stepId, latest);
        }
        if (!key || !latest || this.invalidatedSteps.has(stepId)) continue;
        const cached = latest.key === key ? latest : this.readCache(key, stepId);
        if (cached) {
          const issue = this.getFallbackIssue(step, planned.operation!, planned.engine, cached);
          if (issue) this.addRuntimeIssue(stepId, key, issue);
          else
            this.runtimeIssues.get(stepId)?.set(
              key,
              (this.runtimeIssues.get(stepId)?.get(key) ?? []).filter(
                previous => previous.code !== 'gpu-fallback'
              )
            );
          if (issue?.severity === 'error' || latest.key === key) continue;
          this.latest.set(
            stepId,
            applyStepSensitivity(freezeValue({...cached, cacheHit: true}), step)
          );
          restored = true;
        }
      }
      // Restoring a source can change an observed kind and unblock its consumers' cache keys.
    } while (restored);
    this.trimCache();
    this.cancelObsoleteWork();
  }

  private cancelObsoleteWork(): void {
    // Observed output kinds may legitimately change during a run. Execution identity uses
    // document content and external revisions; normal input validation handles observed kinds.
    const identityPlan =
      this.tasks.size && this.currentDocument
        ? planWorkflow(this.currentDocument, {registry: this.registry, host: this.host})
        : null;
    for (const task of this.tasks) {
      if (
        this.currentDocument?.id !== task.document?.id ||
        this.host.getDevice() !== task.device ||
        this.host.getRuntimeRevisionKey?.() !== task.runtimeRevisionKey ||
        task.identityPlan?.order.some(
          stepId =>
            task.identityPlan!.steps[stepId].key !== null &&
            task.identityPlan!.steps[stepId].key !== identityPlan?.steps[stepId]?.key
        )
      )
        task.controller.abort();
    }
    for (const [controller, isCurrent] of this.publicationChecks) {
      if (!isCurrent()) controller.abort();
    }
  }

  private checkTaskCurrent(task: RunTask): boolean {
    if (!this.destroyed) this.replan();
    return !this.destroyed && !task.controller.signal.aborted;
  }

  private updateState(): void {
    if (this.destroyed) return;
    const steps: Record<string, WorkflowStepState> = {};
    for (const stepId of this.plan?.order ?? []) {
      const planned = this.plan!.steps[stepId];
      const result = this.latest.get(stepId);
      const runtimeIssues = planned.key
        ? (this.runtimeIssues.get(stepId)?.get(planned.key) ?? [])
        : [];
      const error = runtimeIssues.find(issue => issue.severity === 'error');
      let status: WorkflowStepState['status'] = planned.status;
      if (status === 'ready') {
        const runningKey = this.activeTask?.plan?.steps[stepId]?.key;
        if (
          this.activeTask?.selected.has(stepId) &&
          !this.activeTask.completed.has(stepId) &&
          runningKey === planned.key &&
          !this.activeTask.controller.signal.aborted
        ) {
          status = this.runningStepId === stepId ? 'running' : 'queued';
        } else if (error) status = 'error';
        else if (result)
          status =
            result.key === planned.key && !this.invalidatedSteps.has(stepId) ? 'done' : 'stale';
      }
      const display = this.currentDocument?.steps.find(step => step.id === stepId)?.display;
      const published = Object.fromEntries(
        Object.entries(this.published.get(stepId) ?? {}).filter(([name]) => display?.includes(name))
      );
      steps[stepId] = {
        status,
        key: planned.key,
        issues: [...planned.issues, ...runtimeIssues],
        ...(result ? {result} : {}),
        ...(error ? {error} : {}),
        ...(Object.keys(published).length ? {published} : {})
      };
    }
    const next = {
      running: Boolean(this.activeTask),
      ...(this.runningStepId ? {runningStepId: this.runningStepId} : {}),
      order: [...(this.plan?.order ?? [])],
      issues: [...(this.plan?.issues ?? []), ...this.documentIssues],
      steps,
      ...(this.lastRun ? {lastRun: this.lastRun} : {})
    };
    const {revision} = this.state;
    const sameSteps = next.order.every(stepId => {
      const previous = this.state.steps[stepId];
      const current = next.steps[stepId];
      return (
        previous &&
        previous.status === current.status &&
        previous.key === current.key &&
        previous.result === current.result &&
        previous.error === current.error &&
        JSON.stringify(previous.issues) === JSON.stringify(current.issues) &&
        JSON.stringify(previous.published) === JSON.stringify(current.published)
      );
    });
    // Tables can contain 100,000 rows. Compare their immutable result references, not their cells.
    if (
      sameSteps &&
      this.state.running === next.running &&
      this.state.runningStepId === next.runningStepId &&
      this.state.lastRun === next.lastRun &&
      JSON.stringify(this.state.order) === JSON.stringify(next.order) &&
      JSON.stringify(this.state.issues) === JSON.stringify(next.issues)
    )
      return;
    this.state = freezeValue({...next, revision: revision + 1});
    this.notifications.push([...this.listeners]);
    if (this.notifying) return;
    this.notifying = true;
    try {
      while (this.notifications.length && !this.destroyed) {
        for (const listener of this.notifications.shift()!) {
          if (this.listeners.has(listener)) listener();
        }
      }
    } finally {
      this.notifying = false;
    }
  }

  private readCache(key: string, stepId: string): WorkflowStepResult | undefined {
    if (this.invalidatedSteps.has(stepId)) return undefined;
    const entry = this.cache.get(key);
    if (!entry) return undefined;
    entry.owners.add(stepId);
    this.cache.delete(key);
    this.cache.set(key, entry);
    return entry.result;
  }

  private keepResult(stepId: string, result: WorkflowStepResult): void {
    const step = this.currentDocument?.steps.find(candidate => candidate.id === stepId);
    if (this.destroyed || !step) return;
    this.latest.set(stepId, applyStepSensitivity(result, step));
    this.invalidatedSteps.delete(stepId);
    const owners = this.cache.get(result.key)?.owners ?? new Set<string>();
    owners.add(stepId);
    this.cache.delete(result.key);
    this.cache.set(result.key, {result: applyStepSensitivity(result, {}), owners});
    this.trimCache();
  }

  private trimCache(): void {
    const protectedKeys = new Set([...this.latest.values()].map(result => result.key));
    const candidates = [...this.cache].filter(([key]) => !protectedKeys.has(key));
    let count = candidates.length;
    let rows = [...this.cache.values()].reduce(
      (total, entry) =>
        total +
        Object.values(entry.result.outputs).reduce((sum, data) => sum + data.table.rowCount, 0),
      0
    );
    for (const [key, entry] of candidates) {
      if (count <= this.maximumCachedResults && rows <= this.maximumCachedRows) break;
      this.cache.delete(key);
      count--;
      rows -= Object.values(entry.result.outputs).reduce(
        (sum, data) => sum + data.table.rowCount,
        0
      );
    }
  }

  private dropStepResults(stepId: string): void {
    this.latest.delete(stepId);
    this.invalidatedSteps.delete(stepId);
    for (const [key, entry] of this.cache) {
      entry.owners.delete(stepId);
      if (!entry.owners.size) this.cache.delete(key);
    }
  }

  private addRuntimeIssue(stepId: string, key: string, issue: WorkflowIssue): void {
    if (this.destroyed || !this.currentDocument?.steps.some(step => step.id === stepId)) return;
    let entries = this.runtimeIssues.get(stepId);
    if (!entries) {
      entries = new Map();
      this.runtimeIssues.set(stepId, entries);
    }
    if (
      (entries.get(key) ?? []).some(previous => JSON.stringify(previous) === JSON.stringify(issue))
    )
      return;
    entries.set(key, [
      ...(entries.get(key) ?? []).filter(
        previous =>
          !(previous.severity === 'error' && issue.severity === 'error') &&
          !(previous.code === issue.code && previous.port === issue.port)
      ),
      issue
    ]);
  }

  private issueFromError(
    error: unknown,
    step: WorkflowStep,
    operation: WorkflowOperation
  ): WorkflowIssue {
    const label = step.label ?? operation.title;
    const message = readMessage(error);
    return {
      severity: 'error',
      code:
        error instanceof WorkflowError
          ? error.code
          : error instanceof KeplerAllocationError && error.reason === 'capacity'
            ? 'capacity-exceeded'
            : 'operation-failed',
      message:
        message.startsWith(label) || message.startsWith(operation.title)
          ? message
          : `${label}: ${message}`,
      stepId: step.id,
      ...(error instanceof WorkflowError
        ? {port: error.details.port, parameter: error.details.parameter}
        : {})
    };
  }

  private getFallbackIssue(
    step: WorkflowStep,
    operation: WorkflowOperation,
    engine: WorkflowEngineChoice,
    result: Pick<WorkflowStepResult, 'fallbackReason'>,
    device?: ReturnType<WorkflowHost['getDevice']>
  ): WorkflowIssue | undefined {
    if (!result.fallbackReason || operation.engine !== 'gpu-or-cpu' || engine !== 'auto')
      return undefined;
    if (!(device === undefined ? this.host.getDevice() : device)) return undefined;
    return {
      severity: step.onFallback === 'stop' ? 'error' : 'warning',
      code: 'gpu-fallback',
      message: `${step.label ?? operation.title}: GPU fallback: ${result.fallbackReason}`,
      stepId: step.id
    };
  }

  private async executeRun(task: RunTask): Promise<WorkflowRunRecord> {
    try {
      await this.executeTask(task);
    } catch (error) {
      task.record.status = task.controller.signal.aborted || this.destroyed ? 'cancelled' : 'error';
      const entry = task.record.steps.at(-1);
      if (entry) {
        const step = task.document?.steps.find(candidate => candidate.id === entry.stepId);
        const operation = task.plan?.steps[entry.stepId].operation;
        const issue: WorkflowIssue = {
          severity: 'error',
          code: 'operation-failed',
          stepId: entry.stepId,
          message: `${step?.label ?? operation?.title ?? entry.stepId}: ${readMessage(error)}`
        };
        entry.status = task.record.status === 'cancelled' ? 'cancelled' : 'error';
        if (entry.status === 'error') {
          entry.error = issue;
          if (this.currentDocument?.id === task.document?.id)
            this.addRuntimeIssue(entry.stepId, entry.key, issue);
        }
      } else if (task.record.status === 'error') {
        this.documentIssues.push({
          severity: 'error',
          code: 'operation-failed',
          message: `${task.document?.title ?? 'Workflow'}: ${readMessage(error)}`
        });
      }
    } finally {
      this.tasks.delete(task);
      if (this.activeTask === task) this.activeTask = undefined;
      this.runningStepId = undefined;
    }
    task.record.durationMilliseconds = this.now() - task.record.startedAt;
    const record = freezeValue(task.record);
    if (this.destroyed) return record;
    this.lastRun = record;
    this.replan();
    this.updateState();
    if (record.status !== 'cancelled') this.publishDoneSteps(true);
    this.scheduleAutoRun();
    return record;
  }

  private async executeTask(task: RunTask): Promise<void> {
    const {document, plan, controller, record, options} = task;
    const signal = controller.signal;
    if (!this.checkTaskCurrent(task)) {
      record.status = 'cancelled';
      return;
    }
    this.activeTask = task;
    const invalidTarget =
      options.target !== undefined && plan?.steps[options.target]?.status !== 'ready';
    if (!document || !plan || invalidTarget) {
      record.status = 'error';
    } else {
      const included = new Set<string>();
      const include = (stepId: string): void => {
        if (included.has(stepId)) return;
        included.add(stepId);
        for (const upstream of plan.steps[stepId].upstream) include(upstream);
      };
      if (options.target) include(options.target);
      else for (const stepId of plan.order) included.add(stepId);
      const results = new Map<string, WorkflowStepResult>();
      for (const stepId of plan.order) {
        const planned = plan.steps[stepId];
        const latest = this.latest.get(stepId);
        if (latest?.key === planned.key && !this.invalidatedSteps.has(stepId))
          results.set(stepId, latest);
        if (
          included.has(stepId) &&
          planned.status === 'ready' &&
          (options.force ||
            latest?.key !== planned.key ||
            this.invalidatedSteps.has(stepId) ||
            this.state.steps[stepId]?.status === 'error')
        )
          task.selected.add(stepId);
      }
      this.updateState();
      const failed = new Set<string>();
      for (const stepId of plan.order) {
        if (!task.selected.has(stepId)) continue;
        const step = document.steps.find(candidate => candidate.id === stepId)!;
        const planned = plan.steps[stepId];
        const operation = planned.operation!;
        const key = planned.key!;
        const entry: WorkflowRunRecordStep = {
          ...(step.sensitivity ? {sensitivity: structuredClone(step.sensitivity)} : {}),
          stepId,
          operationId: step.operationId,
          key,
          status: 'skipped',
          cacheHit: false,
          durationMilliseconds: 0,
          inputs: []
        };
        record.steps.push(entry);
        const inputs: Record<string, WorkflowData> = {};
        let missingInput = false;
        for (const name of Object.keys(step.inputs).sort()) {
          const connection = step.inputs[name];
          const producer = results.get(connection.stepId);
          const data = producer?.outputs[connection.output];
          if (
            failed.has(connection.stepId) ||
            !data ||
            producer?.key !== plan.steps[connection.stepId]?.key
          ) {
            missingInput = true;
            continue;
          }
          inputs[name] = data;
          entry.inputs.push({
            port: name,
            ...connection,
            key: producer.key,
            rowCount: data.table.rowCount
          });
        }
        if (signal.aborted || missingInput) {
          failed.add(stepId);
          task.completed.add(stepId);
          this.updateState();
          continue;
        }
        this.runningStepId = stepId;
        this.updateState();
        const startedAt = this.now();
        try {
          if (!this.checkTaskCurrent(task)) {
            entry.status = 'cancelled';
            failed.add(stepId);
            continue;
          }
          const device = this.host.getDevice();
          let result = options.force ? undefined : this.readCache(key, stepId);
          if (result) {
            result = freezeValue({...result, cacheHit: true});
          } else {
            this.validateInputs(document, step, operation, inputs);
            if (!this.checkTaskCurrent(task)) {
              entry.status = 'cancelled';
              failed.add(stepId);
              continue;
            }
            const output = await operation.run(
              {
                device,
                engine: planned.engine,
                host: this.host,
                signal,
                workflowId: document.id,
                curatedAnalysisId: step.curatedAnalysisId,
                parameterOverrides: resolveWorkflowParameterOverrides(document, step, operation),
                stepId
              },
              inputs,
              planned.parameters!
            );
            if (!this.checkTaskCurrent(task)) {
              entry.status = 'cancelled';
              failed.add(stepId);
              continue;
            }
            for (const port of operation.outputs) {
              const data = output.outputs[port.name];
              if (!data)
                throw new WorkflowError(
                  'operation-failed',
                  `${step.label ?? operation.title}: missing output ${port.label}.`,
                  {port: port.name}
                );
              if (!Number.isSafeInteger(data.table.rowCount) || data.table.rowCount < 0) {
                throw new WorkflowError(
                  'operation-failed',
                  `${step.label ?? operation.title}: ${port.label} row count must be a nonnegative safe integer; got ${data.table.rowCount}.`,
                  {port: port.name}
                );
              }
              if (data.table.rowCount > port.maximumRows) {
                throw new WorkflowError(
                  'capacity-exceeded',
                  `${step.label ?? operation.title}: ${port.label} allows at most ${port.maximumRows} rows; got ${data.table.rowCount}.`,
                  {port: port.name}
                );
              }
              if (data.analysisProfile && operation.category !== 'source') {
                // Table ids name output ports (for example "filtered"), rather than datasets.
                // Independent producers must retain distinct identities behind input aliases.
                output.outputs[port.name] = {
                  ...data,
                  analysisProfile: {
                    ...data.analysisProfile,
                    sourceDatasetId: `workflow-result:${key}:${port.name}`
                  }
                };
              }
            }
            if (this.currentDocument?.id === document.id)
              this.runtimeIssues.get(stepId)?.delete(key);
            result = freezeValue({
              ...structuredClone(output),
              key,
              parameters: structuredClone(output.resolvedParameters ?? planned.parameters!),
              durationMilliseconds: this.now() - startedAt,
              cacheHit: false,
              completedAt: this.now()
            });
          }
          result = applyStepSensitivity(result, step);
          const fallbackIssue = this.getFallbackIssue(
            step,
            operation,
            planned.engine,
            result,
            device
          );
          if (fallbackIssue) {
            if (fallbackIssue.severity === 'error')
              throw new WorkflowError('gpu-fallback', fallbackIssue.message);
            if (this.currentDocument?.id === document.id)
              this.addRuntimeIssue(stepId, key, fallbackIssue);
          }
          if (this.currentDocument?.id === document.id)
            this.runtimeIssues.get(stepId)?.set(
              key,
              (this.runtimeIssues.get(stepId)?.get(key) ?? []).filter(
                issue => issue.severity !== 'error'
              )
            );
          results.set(stepId, result);
          if (this.currentDocument?.id === document.id) this.keepResult(stepId, result);
          entry.status = 'done';
          entry.cacheHit = result.cacheHit;
          entry.engine = result.engine;
          entry.fallbackReason = result.fallbackReason;
          entry.dispatchedContributors = [...result.dispatchedContributors];
          entry.outputRows = Object.fromEntries(
            Object.entries(result.outputs).map(([name, data]) => [name, data.table.rowCount])
          );
          task.completed.add(stepId);
          this.updateState();
          await this.publishStep(document, step, operation, result, signal);
          if (!this.checkTaskCurrent(task)) entry.status = 'cancelled';
        } catch (error) {
          failed.add(stepId);
          if (!this.checkTaskCurrent(task)) entry.status = 'cancelled';
          else {
            entry.status = 'error';
            entry.error = this.issueFromError(error, step, operation);
            if (this.currentDocument?.id === document.id)
              this.addRuntimeIssue(stepId, key, entry.error);
          }
        } finally {
          entry.durationMilliseconds = this.now() - startedAt;
          task.completed.add(stepId);
          this.runningStepId = undefined;
          this.updateState();
        }
      }
      record.status =
        signal.aborted || this.destroyed
          ? 'cancelled'
          : record.steps.some(step => step.status === 'error')
            ? 'error'
            : 'done';
    }
  }

  private validateInputs(
    document: WorkflowDocument,
    step: WorkflowStep,
    operation: WorkflowOperation,
    inputs: Record<string, WorkflowData>
  ): void {
    const label = step.label ?? operation.title;
    for (const port of operation.inputs) {
      if (inputs[port.name]?.kind === 'raster' || inputs[port.name]?.table.rowCount !== 0) continue;
      const producer = document.steps.find(
        candidate => candidate.id === step.inputs[port.name].stepId
      )!;
      const producerLabel =
        producer.label ?? this.registry.get(producer.operationId)?.title ?? producer.id;
      throw new WorkflowError(
        'input-required',
        `${label}: "${port.label}" received no rows from ${producerLabel}. Check that step's result or its parameters.`,
        {port: port.name}
      );
    }
    for (const port of operation.inputs) {
      const data = inputs[port.name];
      if (!data) continue;
      const kind = data.kind === 'raster' ? 'raster' : getWorkflowDataKind(data.table);
      if (kind === 'table' && !port.accepts.includes('table')) {
        throw new WorkflowError(
          'geometry-required',
          `${label}: ${port.label} requires geometry (${port.accepts.join(' or ')}); got table.`,
          {port: port.name}
        );
      }
      if (
        getWorkflowPortCompatibility(kind, port) === 'no' ||
        (kind === 'features' &&
          !port.accepts.includes('features') &&
          !port.accepts.includes('table'))
      ) {
        throw new WorkflowError(
          'geometry-kind-not-accepted',
          `${label}: ${port.label} accepts ${port.accepts.join(' or ')}; got ${kind}.`,
          {port: port.name}
        );
      }
      if (kind === 'raster') {
        if (!data.raster) {
          throw new WorkflowError(
            'input-required',
            `${label}: ${port.label} requires a raster band/grid payload.`,
            {port: port.name}
          );
        }
        try {
          validateCatalogRasterInput(
            `${operation.id}: ${port.name}`,
            data.raster,
            port.maximumCells ?? 0
          );
        } catch (cause) {
          const cellCount = data.raster.grid.width * data.raster.grid.height;
          const exceedsCapacity = port.maximumCells !== undefined && cellCount > port.maximumCells;
          throw new WorkflowError(
            exceedsCapacity ? 'capacity-exceeded' : 'operation-failed',
            `${label}: ${port.label}: ${readMessage(cause)}`,
            {port: port.name, cause}
          );
        }
        // Retained display/provenance rows are independent of the bounded computational grid.
        continue;
      }
      const suggestion = 'Add a filter or sort-with-limit step before it.';
      if (data.table.rowCount > port.maximumRows) {
        throw new WorkflowError(
          'capacity-exceeded',
          `${label}: ${port.label} allows at most ${port.maximumRows} rows; got ${data.table.rowCount}. ${suggestion}`,
          {port: port.name}
        );
      }
      if (port.maximumVertices !== undefined) {
        const count = countWorkflowVertices(data);
        if (count > port.maximumVertices) {
          throw new WorkflowError(
            'capacity-exceeded',
            `${label}: ${port.label} allows at most ${port.maximumVertices} vertices; got ${count}. ${suggestion}`,
            {port: port.name}
          );
        }
      }
    }
  }

  private reconcileDisplay(retractAll: boolean): void {
    const workflowId = this.currentDocument?.id;
    for (const [stepId, outputs] of this.published) {
      const step = this.currentDocument?.steps.find(candidate => candidate.id === stepId);
      for (const [name, published] of Object.entries(outputs)) {
        if (!retractAll && step?.display?.includes(name)) continue;
        this.publishTail = this.publishTail.then(async () => {
          const current = this.published.get(stepId)?.[name];
          // The immutable publication object identifies its generation, even if the host reuses ids.
          if (current !== published) return;
          const retained =
            this.currentDocument?.id === workflowId &&
            this.currentDocument?.steps.some(
              candidate => candidate.id === stepId && candidate.display?.includes(name)
            );
          if (!retractAll && retained) return;
          delete outputs[name];
          if (!Object.keys(outputs).length) this.published.delete(stepId);
          await this.retractOutput(published.datasetId);
          this.updateState();
        });
      }
    }
  }

  private async retractOutput(datasetId: string): Promise<void> {
    try {
      await this.host.retractOutput?.(datasetId);
    } catch (error) {
      if (this.destroyed) return;
      this.documentIssues.push({
        severity: 'warning',
        code: 'publish-failed',
        message: `${this.currentDocument?.title ?? 'Workflow'}: could not retract displayed dataset ${datasetId}: ${readMessage(error)}`
      });
      this.updateState();
    }
  }

  private publishDoneSteps(skipFailed = false): void {
    if (this.destroyed || !this.currentDocument || !this.host.publishOutput) return;
    for (const step of this.currentDocument.steps) {
      const state = this.state.steps[step.id];
      const operation = this.registry.get(step.operationId);
      if (state?.status === 'done' && state.result && operation) {
        void this.publishStep(
          this.currentDocument,
          skipFailed
            ? {
                ...step,
                display: step.display?.filter(
                  name =>
                    !state.issues.some(
                      issue => issue.code === 'publish-failed' && issue.port === name
                    )
                )
              }
            : step,
          operation,
          state.result,
          new AbortController().signal
        );
      }
    }
  }

  private publishStep(
    document: WorkflowDocument,
    step: WorkflowStep,
    operation: WorkflowOperation,
    result: WorkflowStepResult,
    signal: AbortSignal
  ): Promise<void> {
    if (!this.host.publishOutput || this.destroyed) return Promise.resolve();
    for (const name of step.display ?? []) {
      const port = operation.outputs.find(output => output.name === name);
      const data = result.outputs[name];
      const style = step.displayStyle?.[name];
      const styleKey = getDisplayStyleKey(style);
      const runtimeRevisionKey = this.host.getRuntimeRevisionKey?.();
      const publicationGeneration = this.publicationGeneration;
      const token = JSON.stringify([
        document.id,
        step.id,
        name,
        result.key,
        styleKey,
        runtimeRevisionKey,
        publicationGeneration
      ]);
      if (!port || !data || this.pendingPublications.has(token)) continue;
      this.pendingPublications.add(token);
      const controller = new AbortController();
      const device = this.host.getDevice();
      const isCurrent = (): boolean => {
        if (
          this.destroyed ||
          signal.aborted ||
          controller.signal.aborted ||
          this.currentDocument?.id !== document.id ||
          this.host.getDevice() !== device ||
          this.host.getRuntimeRevisionKey?.() !== runtimeRevisionKey ||
          this.publicationGeneration !== publicationGeneration ||
          this.latest.get(step.id)?.key !== result.key ||
          this.invalidatedSteps.has(step.id)
        )
          return false;
        return this.createPlan(this.currentDocument).steps[step.id]?.key === result.key;
      };
      this.publicationChecks.set(controller, isCurrent);
      this.publishTail = this.publishTail.then(async () => {
        try {
          const currentStep = this.currentDocument?.steps.find(
            candidate => candidate.id === step.id
          );
          if (!isCurrent() || !currentStep?.display?.includes(name)) return;
          if (getDisplayStyleKey(currentStep.displayStyle?.[name]) !== styleKey) return;
          const previous = this.published.get(step.id)?.[name];
          if (
            previous?.key === result.key &&
            this.publicationGenerations.get(previous) === publicationGeneration
          ) {
            if (this.plan?.steps[step.id]?.key !== result.key) return;
            if (getDisplayStyleKey(previous.style) === styleKey) {
              this.clearPublishIssue(step.id, result.key, name);
              this.updateState();
              return;
            }
          }
          const label = `${step.label ?? operation.title} · ${port.label}`;
          const output = await this.host.publishOutput!(
            {
              workflowId: document.id,
              workflowTitle: document.title,
              stepId: step.id,
              output: name,
              label,
              data,
              style,
              isCurrent,
              previousDatasetId: previous?.datasetId
            },
            controller.signal
          );
          const retained =
            this.currentDocument?.id === document.id &&
            this.currentDocument.steps.some(
              candidate => candidate.id === step.id && candidate.display?.includes(name)
            );
          if (!isCurrent() || !retained) {
            const outputs = this.published.get(step.id);
            if (outputs?.[name]?.datasetId === output.datasetId) {
              delete outputs[name];
              if (!Object.keys(outputs).length) this.published.delete(step.id);
            }
            await this.retractOutput(output.datasetId);
            return;
          }
          let retirement: Promise<void> | undefined;
          try {
            // Acceptance is synchronous with the currentness check. Once accepted, later
            // cancellation cannot roll back this display while its predecessor is retiring.
            retirement = output.commit?.();
          } catch (error) {
            await this.retractOutput(output.datasetId);
            throw error;
          }
          const outputs = this.published.get(step.id) ?? {};
          outputs[name] = freezeValue({
            datasetId: output.datasetId,
            label: output.label,
            key: result.key,
            ...(style ? {style} : {})
          });
          this.publicationGenerations.set(outputs[name], publicationGeneration);
          this.published.set(step.id, outputs);
          this.clearPublishIssue(step.id, result.key, name);
          this.updateState();
          await retirement;
        } catch (error) {
          if (isCurrent())
            this.addRuntimeIssue(step.id, result.key, {
              severity: 'warning',
              code: 'publish-failed',
              message: `${step.label ?? operation.title}: could not display ${port.label}: ${readMessage(error)}`,
              stepId: step.id,
              port: name
            });
          this.updateState();
        } finally {
          this.pendingPublications.delete(token);
          this.publicationChecks.delete(controller);
        }
      });
    }
    return this.publishTail;
  }

  private clearPublishIssue(stepId: string, key: string, output: string): void {
    this.runtimeIssues.get(stepId)?.set(
      key,
      (this.runtimeIssues.get(stepId)?.get(key) ?? []).filter(
        issue => issue.code !== 'publish-failed' || issue.port !== output
      )
    );
  }

  private clearAutoRunTimer(): void {
    if (this.autoRunTimer !== undefined) clearTimeout(this.autoRunTimer);
    this.autoRunTimer = undefined;
  }

  private scheduleAutoRun(): void {
    this.clearAutoRunTimer();
    if (
      this.destroyed ||
      !this.currentDocument?.settings.autoRun ||
      this.activeTask ||
      this.tasks.size
    )
      return;
    if (
      !Object.entries(this.state.steps).some(
        ([stepId, state]) => this.plan?.steps[stepId].status === 'ready' && state.status !== 'done'
      )
    )
      return;
    this.autoRunTimer = setTimeout(() => {
      this.autoRunTimer = undefined;
      void this.run();
    }, this.autoRunDelayMilliseconds);
  }
}
