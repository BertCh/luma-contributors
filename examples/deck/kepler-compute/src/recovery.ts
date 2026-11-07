// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {appState} from './app-state';
import {DATASET_ID, createSampleData} from './sample-data';
import {
  readPublishedSandboxSource,
  validateSandboxBundle,
  readSandboxFixture
} from './sandbox-state';
import {getStreamImportOptions, getPopulationLabel} from './remote-stream-source';
import {CELL_INDEX_METADATA_KEY} from './cell-layer-types';
import {ComputeRuntime} from './compute-runtime';
import {type ResidentResultRecipe} from './resident-result-recipes';
import {
  type ResidentRecoverySnapshot,
  captureResidentRecoverySnapshot,
  restoreResidentRecoveryState,
  type ResidentRecoveryContext,
  ResidentDeviceRecovery
} from './device-recovery';
import {removeDataset, wrapTo, toggleModal} from '@kepler.gl/actions';
import {type RestoredSandboxBundle} from './sandbox-bundle';
import {
  runRestoreTransaction,
  preflightRestoreReservation,
  describePreviousWorkspaceRestore
} from './restore-transaction';
import {runWithOutOfMemoryScope} from './binding5-device-oom-scope';
import {describeRestoreAllocationFailure} from './binding3-restore-recovery';
import {estimateRestoreBytes} from './retained-bytes-estimate';
import {assertSingleViewConfiguration} from './resident-action-guards';
import KeplerGlSchema from '@kepler.gl/schemas';
import {stageNativeSources, clearStagedNativeSources} from './bundle-native-sources';
import {type AutomaticRecoveryStatus, AutomaticRecoveryPolicy} from './automatic-recovery-policy';
import {type ResidentEvictionStatus} from './resident-eviction-host';
import {
  datasetOperations,
  beginPublication,
  reportPlot,
  discardPublishedResult,
  finishPublication,
  attachPublishedDataset,
  publicationsInProgress
} from './publication';
import {getActiveRuntime, reimportSource, applyNativeConfig} from './native-config';
import {
  residentEviction,
  notify,
  reportError,
  residentDatasetIds,
  memoryManager,
  residentResultRecipes,
  effect,
  destroyClusterRuntimes,
  abandonedDeviceGenerations,
  AUTOMATIC_POLICY
} from './app-services';
import {store} from './app-store';
import {
  type ResidentJoinExecution,
  type ResidentSuitabilityExecution,
  executeSuitabilityRequest,
  executeJoinRequest
} from './resident-operations';
import {SUPPORTED_VIEW_COUNT} from './split-view';

/** A dataset is evictable only while the healthy checkpoint retains its current revision. */
export function getEvictionRebuildSource(datasetId: string) {
  const snapshot = appState.deviceRecovery.checkpointSnapshot;
  const current = appState.runtime?.session.registry.getDataset(datasetId);
  if (!snapshot || !current || datasetId === DATASET_ID) return undefined;
  const descriptor = snapshot.bundle.state.sources.find(
    candidate => candidate.dataset.datasetId === datasetId
  );
  const retained = snapshot.bundle.sources.find(candidate => candidate.datasetId === datasetId);
  const recipe = snapshot.derived?.find(candidate => candidate.resultId === datasetId);
  return descriptor && (retained || recipe) && descriptor.dataset.revision === current.revision
    ? {descriptor, retained, recipe}
    : undefined;
}

/**
 * Re-upload one evicted table from the retained CPU checkpoint. The native table and its layers
 * were never removed, so only the resident publication is recreated and rebound.
 */
export async function rebuildEvictedDataset(datasetId: string): Promise<void> {
  await datasetOperations.run(datasetId, 'replace', () => rebuildEvictedDatasetNow(datasetId));
}

async function rebuildEvictedDatasetNow(datasetId: string): Promise<void> {
  const owner = getActiveRuntime();
  const snapshot = appState.deviceRecovery.checkpointSnapshot;
  const descriptor = snapshot?.bundle.state.sources.find(
    candidate => candidate.dataset.datasetId === datasetId
  );
  const retained = snapshot?.bundle.sources.find(candidate => candidate.datasetId === datasetId);
  const recipe = snapshot?.derived?.find(candidate => candidate.resultId === datasetId);
  if (!descriptor || !(retained || recipe)) throw new Error(`No retained source for ${datasetId}`);
  const roles = readPublishedSandboxSource(descriptor);
  beginPublication(datasetId);
  try {
    if (recipe) {
      // Inputs may themselves be evicted; they rebuild first, then the result is recomputed.
      for (const input of recipe.inputs) await residentEviction.ensureResident(input.datasetId);
      const signal = new AbortController().signal;
      const execution = await executeResidentResultRecipe(
        owner,
        recipe,
        signal,
        () => appState.runtime === owner
      );
      try {
        assertRecomputedResult(execution, recipe, roles);
        await owner.attachDataset(
          execution.result.dataset,
          roles.label,
          undefined,
          reportPlot,
          () => queueMicrotask(notify),
          undefined,
          undefined,
          undefined,
          undefined,
          roles.join,
          roles.suitability
        );
        if (appState.runtime !== owner) throw new Error('Workspace changed during table rebuild');
        execution.result.transfer();
      } catch (error) {
        await discardPublishedResult(execution, 'Resident result').catch(reportError);
        throw error;
      }
      owner.requestUpdate();
      store.dispatch({type: 'LUMA_RENDER_ANALYSIS'});
      return;
    }
    if (!retained) throw new Error(`No retained source for ${datasetId}`);
    const dataset = owner.session.importArrowDataset({
      id: datasetId,
      table: retained.table,
      normalization: descriptor.normalization,
      ...getStreamImportOptions(roles.stream),
      metadata: {
        ...getStreamImportOptions(roles.stream).metadata,
        ...(roles.cellIndex ? {[CELL_INDEX_METADATA_KEY]: roles.cellIndex} : {})
      }
    });
    await owner.attachDataset(
      dataset,
      getPopulationLabel(roles.label, dataset.population),
      roles.coordinates,
      reportPlot,
      () => queueMicrotask(notify),
      roles.geometryColumn,
      roles.tripOptions,
      roles.h3Column,
      roles.groupBy,
      roles.join,
      roles.suitability
    );
    if (appState.runtime !== owner) throw new Error('Workspace changed during table rebuild');
    owner.requestUpdate();
    store.dispatch({type: 'LUMA_RENDER_ANALYSIS'});
  } finally {
    finishPublication(datasetId);
  }
}

type ResidentResultExecution = ResidentJoinExecution | ResidentSuitabilityExecution;

/** Re-run the recorded operation on the current inputs; the result keeps its original dataset ID. */
function executeResidentResultRecipe(
  owner: ComputeRuntime,
  recipe: ResidentResultRecipe,
  signal: AbortSignal,
  isCurrent: () => boolean
): Promise<ResidentResultExecution> {
  const operation = recipe.operation;
  return operation.kind === 'suitability'
    ? executeSuitabilityRequest(owner, operation.request, signal, isCurrent)
    : executeJoinRequest(owner, operation.request, signal, isCurrent);
}

/** A recomputed result must reproduce the saved identity, schema and row count. */
function assertRecomputedResult(
  execution: ResidentResultExecution,
  recipe: ResidentResultRecipe,
  roles: ReturnType<typeof readPublishedSandboxSource>
): void {
  const lineage = roles.join ?? roles.suitability;
  if (
    execution.result.dataset.id !== recipe.resultId ||
    execution.result.rowCount !== recipe.rowCount ||
    !lineage ||
    JSON.stringify(execution.plan.provenance.fields) !== JSON.stringify(lineage.fields)
  )
    throw new Error(`Recomputed result ${recipe.resultId} does not match its saved lineage`);
}

/** Recompute every resident result in dependency order on a recovered runtime. */
async function recomputeResidentResults(
  restored: ComputeRuntime,
  snapshot: ResidentRecoverySnapshot,
  signal: AbortSignal
): Promise<void> {
  for (const recipe of snapshot.derived ?? []) {
    signal.throwIfAborted();
    const descriptor = snapshot.bundle.state.sources.find(
      candidate => candidate.dataset.datasetId === recipe.resultId
    );
    if (!descriptor) throw new Error('Resident result lineage has no saved source');
    const roles = readPublishedSandboxSource(descriptor);
    const isCurrent = () =>
      appState.runtime === restored && !signal.aborted && !restored.session.deviceLost;
    const execution = await executeResidentResultRecipe(restored, recipe, signal, isCurrent);
    try {
      assertRecomputedResult(execution, recipe, roles);
      const current = () => isCurrent() && execution.result.isCurrent();
      // The saved lineage keeps the original source revisions so replayed state matches the manifest.
      await datasetOperations.run(recipe.resultId, 'replace', () =>
        attachPublishedDataset(
          restored,
          execution.result.dataset,
          roles.label,
          undefined,
          undefined,
          undefined,
          undefined,
          undefined,
          current,
          roles.join,
          roles.suitability
        )
      );
      execution.result.transfer();
    } catch (error) {
      await discardPublishedResult(execution, 'Resident result').catch(reportError);
      throw error;
    }
  }
}

/** Removes whatever a failed source creation left behind: native table, publication, dataset lease. */
async function discardPartialRestoreSource(
  restored: ComputeRuntime,
  datasetId: string
): Promise<void> {
  residentEviction.forget(datasetId);
  store.dispatch(removeDataset(datasetId));
  await restored.removePublication(datasetId).catch(reportError);
  if (restored.session.registry.getDataset(datasetId))
    await restored.session.removeDataset(datasetId, Promise.resolve()).catch(reportError);
  residentDatasetIds.delete(datasetId);
  notify();
}

async function populateBundleSources(
  restored: ComputeRuntime,
  bundle: RestoredSandboxBundle,
  signal?: AbortSignal,
  derivedDatasetIds: ReadonlySet<string> = new Set()
): Promise<void> {
  // Transactional: if source k fails, sources created before it are removed in reverse order.
  try {
    await runRestoreTransaction({
      sources: bundle.state.sources.filter(
        descriptor =>
          descriptor.dataset.datasetId !== DATASET_ID &&
          !derivedDatasetIds.has(descriptor.dataset.datasetId)
      ),
      getSourceId: descriptor => descriptor.dataset.datasetId,
      signal,
      async createSource(descriptor, index) {
        // Test-only fault injection used by scripts/restore-rollback.mjs.
        if (index === appState.injectedRestoreFailureIndex) {
          appState.injectedRestoreFailureIndex = -1;
          throw new Error('Injected restore failure');
        }
        const retained = bundle.sources.find(
          candidate => candidate.datasetId === descriptor.dataset.datasetId
        );
        if (!retained) throw new Error('Bundle source is missing');
        const roles = readPublishedSandboxSource(descriptor);
        const dataset = restored.session.importArrowDataset({
          id: descriptor.dataset.datasetId,
          table: retained.table,
          normalization: descriptor.normalization,
          // Restores recorded remote origin, partial population and cell index; no network request.
          ...getStreamImportOptions(roles.stream),
          metadata: {
            ...getStreamImportOptions(roles.stream).metadata,
            ...(roles.cellIndex ? {[CELL_INDEX_METADATA_KEY]: roles.cellIndex} : {})
          }
        });
        try {
          // Device out-of-memory is reported asynchronously; the scope turns it into a thrown
          // allocation error so the rollback below runs.
          await runWithOutOfMemoryScope(restored.session.device, () =>
            datasetOperations.run(descriptor.dataset.datasetId, 'replace', () =>
              attachPublishedDataset(
                restored,
                dataset,
                roles.label,
                roles.coordinates,
                roles.geometryColumn,
                roles.tripOptions,
                roles.h3Column,
                roles.groupBy,
                undefined,
                roles.join,
                roles.suitability
              )
            )
          );
        } catch (error) {
          // A partial upload of the failing source (imported dataset, half-built publication, native
          // table) is not in the transaction's staged list; release it before the rollback runs.
          await discardPartialRestoreSource(restored, descriptor.dataset.datasetId);
          throw error;
        }
        return {
          async undo() {
            residentEviction.forget(descriptor.dataset.datasetId);
            store.dispatch(removeDataset(descriptor.dataset.datasetId));
            await restored.removePublication(descriptor.dataset.datasetId);
            residentDatasetIds.delete(descriptor.dataset.datasetId);
            notify();
          }
        };
      }
    });
  } catch (error) {
    // After rollback, an allocation failure also names every source that cannot be allocated.
    throw describeRestoreAllocationFailure(
      error,
      bundle.sources
        .filter(retained => !derivedDatasetIds.has(retained.datasetId))
        .map(retained => ({
          sourceId: retained.datasetId,
          bytes: estimateRestoreBytes(retained.table).totalBytes
        })),
      restored.session.registry.memoryBudgetBytes
    );
  }
}

export async function restoreBundle(bundle: RestoredSandboxBundle): Promise<ComputeRuntime> {
  assertSingleViewConfiguration(bundle.state.nativeConfig, SUPPORTED_VIEW_COUNT);
  // Queued replacements and removals settle first, so none races the workspace swap.
  await datasetOperations.whenIdle();
  const previous = getActiveRuntime();
  // Complete CPU normalization, schema, role and geometry checks precede source replacement.
  validateSandboxBundle(bundle, previous);
  const fixture = bundle.state.sources.find(
    descriptor => descriptor.dataset.datasetId === DATASET_ID
  );
  if (!fixture) throw new Error('Bundle does not contain the observations fixture');
  // The saved budget applies first so a bundle saved under a larger budget can be admitted again.
  if (bundle.state.settings) memoryManager.settings.apply(bundle.state.settings);
  // Admit the whole bundle against the memory budget before the current workspace is replaced.
  const preflight = preflightRestoreReservation(
    bundle.sources,
    retained => estimateRestoreBytes(retained.table).totalBytes,
    previous.session.registry.memoryBudgetBytes
  );
  if (!preflight.fits)
    throw new Error(
      `Bundle needs about ${preflight.totalBytes.toLocaleString()} bytes of memory but the budget is ${preflight.availableBytes.toLocaleString()}; the current workspace was not changed`
    );
  // Snapshot the previous workspace (retained CPU tables, roles, native config, result lineage) before
  // anything is replaced, so a failed restore can rebuild it instead of leaving a fixture-only workspace.
  let previousSnapshot: ResidentRecoverySnapshot | undefined;
  try {
    previousSnapshot = captureResidentRecoverySnapshot({
      runtime: previous,
      getNativeConfig: () => KeplerGlSchema.getConfigToSave(store.getState().keplerGl['compute']),
      collectResultRecipes: residentResultIds => residentResultRecipes.collect(residentResultIds)
    });
  } catch {
    previousSnapshot = appState.deviceRecovery.checkpointSnapshot;
  }
  stageNativeSources(bundle.nativeOnlySources ?? []);
  try {
    const restored = await reimportSource(fixture);
    await populateBundleSources(restored, bundle);
    return restored;
  } catch (error) {
    clearStagedNativeSources();
    // Created sources were already removed; now bring back what the user had.
    // The workflow dialog unmounts while datasets are replaced, so its own status is lost; the
    // outcome is also published as the persistent workflow notice.
    const publishNotice = (described: Error): Error => {
      appState.workflowNotice = described.message;
      notify();
      return described;
    };
    if (!previousSnapshot)
      throw publishNotice(describePreviousWorkspaceRestore(error, false, 'no snapshot'));
    try {
      await rebuildPreviousWorkspace(previousSnapshot);
    } catch (rebuildFailure) {
      throw publishNotice(describePreviousWorkspaceRestore(error, false, rebuildFailure));
    }
    throw publishNotice(describePreviousWorkspaceRestore(error, true));
  }
}

/** Rebuilds a captured workspace on a fresh fixture runtime: sources, derived results, then native config. */
async function rebuildPreviousWorkspace(snapshot: ResidentRecoverySnapshot): Promise<void> {
  const fixture = snapshot.bundle.state.sources.find(
    descriptor => descriptor.dataset.datasetId === DATASET_ID
  );
  if (!fixture) throw new Error('Snapshot has no observations fixture');
  const restored = await reimportSource(fixture);
  const derivedDatasetIds = new Set(snapshot.derived?.map(recipe => recipe.resultId));
  const signal = new AbortController().signal;
  await populateBundleSources(restored, snapshot.bundle, signal, derivedDatasetIds);
  await recomputeResidentResults(restored, snapshot, signal);
  await restoreResidentRecoveryState(snapshot, {
    runtime: restored,
    applyNativeConfig,
    derivedDatasetIds
  });
  restored.requestUpdate();
  notify();
}

export function scheduleRecoveryCheckpoint(): void {
  if (appState.checkpointScheduled) return;
  appState.checkpointScheduled = true;
  queueMicrotask(() => {
    appState.checkpointScheduled = false;
    if (
      appState.runtime &&
      !appState.runtime.session.deviceLost &&
      !appState.runtime.isLayerRegistrationPending &&
      !publicationsInProgress.size &&
      // A checkpoint taken while a table is evicted would drop the source needed to rebuild it.
      !residentEviction.status.evictedIds.length &&
      appState.deviceRecovery.status.phase === 'ready'
    )
      appState.deviceRecovery.checkpoint(appState.runtime);
  });
}

export function invalidateLostRuntime(previous: ComputeRuntime, reason: string): void {
  if (appState.runtime !== previous) return;
  effect.runtime = null;
  appState.latestSummary = null;

  appState.unsupportedSelection = true;
  previous.deck.setProps({layers: []});
  store.dispatch(wrapTo('compute', toggleModal(null)));
  appState.currentError = reason;
  notify();
}

export async function retireRecoveryRuntime(
  previous: ComputeRuntime,
  completion?: PromiseLike<unknown>
): Promise<void> {
  const datasetIds = [DATASET_ID, ...previous.publications.keys()];
  if (appState.runtime === previous) {
    destroyClusterRuntimes();
    appState.runtime = null;
    effect.runtime = null;
    for (const datasetId of datasetIds) {
      store.dispatch(removeDataset(datasetId));
      if (datasetId !== DATASET_ID) residentDatasetIds.delete(datasetId);
    }
  }
  try {
    await previous.destroy(completion);
  } finally {
    previous.deck.finalize();
    notify();
  }
}

function createReplacementRuntime(context: ResidentRecoveryContext): Promise<ComputeRuntime> {
  context.signal.throwIfAborted();
  const generation = ++appState.deviceGeneration;
  return new Promise((resolve, reject) => {
    let settled = false;
    const cleanup = () => {
      window.clearTimeout(timeout);
      context.signal.removeEventListener('abort', abort);
      if (appState.pendingDeviceInitialization?.generation === generation)
        appState.pendingDeviceInitialization = null;
    };
    const fail = (reason: unknown) => {
      if (settled) return;
      settled = true;
      abandonedDeviceGenerations.add(generation);
      cleanup();
      reject(reason);
    };
    const abort = () => fail(context.signal.reason ?? new Error('Device creation cancelled'));
    const timeout = window.setTimeout(
      () => fail(new Error('A replacement GPU device did not initialize within 30 seconds')),
      30000
    );
    appState.pendingDeviceInitialization = {
      generation,
      signal: context.signal,
      resolve(candidate) {
        if (settled || context.signal.aborted) {
          void retireRecoveryRuntime(candidate).catch(reportError);
          return;
        }
        settled = true;
        cleanup();
        resolve(candidate);
      },
      reject: fail
    };
    context.signal.addEventListener('abort', abort, {once: true});
    notify();
  });
}

export function formatAutomaticRecoveryStatus(status: AutomaticRecoveryStatus): string {
  if (status.state === 'exhausted')
    return `Automatic recovery stopped after ${status.attempts} of ${status.maximumAttempts} attempts; use Recover GPU workspace.`;
  if (status.state === 'waiting' || status.state === 'recovering')
    return `Automatic recovery attempt ${status.attempts + (status.state === 'waiting' ? 1 : 0)} of ${status.maximumAttempts} ${status.state === 'waiting' ? 'scheduled' : 'running'}.`;
  return 'Automatic recovery is enabled.';
}

export function formatEvictionStatus(status: ResidentEvictionStatus): string {
  if (status.failedIds.length)
    return `Rebuilding ${status.failedIds.join(', ')} failed: ${status.lastError ?? 'unknown error'}.`;
  if (status.rebuildingIds.length)
    return `Rebuilding evicted ${status.rebuildingIds.length === 1 ? 'table' : 'tables'} ${status.rebuildingIds.join(', ')} from retained sources.`;
  if (status.evictedIds.length)
    return `${status.evictedIds.length} hidden ${status.evictedIds.length === 1 ? 'table was' : 'tables were'} evicted to free memory and will rebuild when shown.`;
  return 'Automatic table eviction is enabled.';
}

export function createDeviceRecovery(): ResidentDeviceRecovery<ComputeRuntime> {
  const coordinator = new ResidentDeviceRecovery<ComputeRuntime>({
    capture: activeRuntime =>
      captureResidentRecoverySnapshot({
        runtime: activeRuntime,
        getNativeConfig: () => KeplerGlSchema.getConfigToSave(store.getState().keplerGl['compute']),
        collectResultRecipes: residentResultIds => residentResultRecipes.collect(residentResultIds)
      }),
    invalidate: invalidateLostRuntime,
    teardown: (previous, completion) => retireRecoveryRuntime(previous, completion),
    async rebuild(snapshot: ResidentRecoverySnapshot, context) {
      context.signal.throwIfAborted();
      const fixture = snapshot.bundle.state.sources.find(
        descriptor => descriptor.dataset.datasetId === DATASET_ID
      );
      if (!fixture) throw new Error('Recovery checkpoint has no observations fixture');
      const restoredRowCount = readSandboxFixture(fixture).rowCount;
      if (restoredRowCount !== appState.rowCount) {
        appState.rowCount = restoredRowCount;
        appState.source = createSampleData(appState.rowCount);
      }
      return createReplacementRuntime(context);
    },
    async restore(candidate, snapshot, context) {
      context.signal.throwIfAborted();
      const derivedDatasetIds = new Set(snapshot.derived?.map(recipe => recipe.resultId));
      validateSandboxBundle(snapshot.bundle, candidate, derivedDatasetIds);
      await populateBundleSources(candidate, snapshot.bundle, context.signal, derivedDatasetIds);
      // Over-cap results have no CPU source: recompute them on the recovered inputs.
      await recomputeResidentResults(candidate, snapshot, context.signal);
      context.signal.throwIfAborted();
      await restoreResidentRecoveryState(snapshot, {
        runtime: candidate,
        applyNativeConfig,
        derivedDatasetIds
      });
      context.signal.throwIfAborted();
    },
    publish(candidate) {
      appState.runtime = candidate;
      residentEviction.attach(candidate.session.registry);
      memoryManager.bindRegistry();
      effect.runtime = candidate;
      appState.unsupportedSelection = false;
      appState.currentError = '';
      candidate.requestUpdate();
      store.dispatch({type: 'LUMA_RENDER_ANALYSIS'});
      notify();
    },
    discard: candidate => retireRecoveryRuntime(candidate)
  });
  coordinator.subscribe(() => notify());
  appState.automaticRecovery?.dispose();
  appState.automaticRecovery = new AutomaticRecoveryPolicy(coordinator, {
    enabled: AUTOMATIC_POLICY.autoRecovery
  });
  appState.automaticRecovery.subscribe(() => notify());
  return coordinator;
}
