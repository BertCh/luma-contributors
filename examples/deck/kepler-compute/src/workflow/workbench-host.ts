// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Scoped host builds also need the app's declaration for its transitive CSV reader dependency.
/// <reference path="../remote-source-modules.d.ts" />

import {
  WorkflowError,
  hashWorkflowValue,
  makeWorkflowData,
  getWorkflowSqlProblem,
  type KeplerDuckDBConnection,
  type WorkflowSqlRequest,
  type WorkflowHost,
  type WorkflowDatasetRequest,
  type WorkflowDatasetProfile,
  type WorkflowDisplayStyle
} from '@luma-contributors/kepler';
import {store} from '../app-store';
import {appState} from '../app-state';
import {isResidentDataset, publishSource, datasetOperations} from '../publication';
import {fitBounds, layerVisualChannelConfigChange, removeDataset, wrapTo} from '@kepler.gl/actions';
import {KEPLER_COLOR_PALETTES, colorPaletteToColorRange} from '@kepler.gl/constants';
import {residentDatasetIds, residentEviction, notify} from '../app-services';
import {fieldPreparationCoordinator} from '../binding3-field-preparation';
import {extractSelectedRows} from '../resident-drawing-actions';
import {DATASET_ID} from '../sample-data';
import type {OwnedDuckDBConnection} from '../duckdb-runtime';
import {
  chooseWorkflowDisplayChannel,
  getNativeRevisionKey,
  getWorkflowDataBounds,
  makeWorkflowDataFromArrowTable,
  makeWorkflowArrowTable,
  makeWorkflowDataFromSqlTable,
  makeWorkflowDataFromDrawings,
  profileWorkflowDrawings,
  type WorkflowDrawingFeature,
  makeWorkflowPublication,
  profileNativeDataset,
  readNativeDatasetData,
  validateWorkflowRowIdentity
} from './workbench-host-data';

// Same values as AUTO_PREPARE_FLOAT64_POLICY in ../auto-prepare-plan, without loading its
// source-role planner here. The relative budget is twice the Float32 unit roundoff.
const WORKFLOW_DISPLAY_FLOAT64_POLICY = Object.freeze({
  mode: 'round' as const,
  maximumAbsoluteError: 1e-37,
  maximumRelativeError: 2 ** -23
});

/** Remove the optional statement terminator without changing quoted text or SQL comments. */
function stripWorkflowSqlTerminator(query: string): string {
  let result = '';
  let index = 0;
  while (index < query.length) {
    const character = query[index];
    if (character === "'" || character === '"') {
      const start = index++;
      while (index < query.length) {
        if (query[index++] !== character) continue;
        if (query[index] === character) index++;
        else break;
      }
      result += query.slice(start, index);
    } else if (query.startsWith('--', index)) {
      const start = index;
      while (index < query.length && !/[\r\n]/.test(query[index])) index++;
      result += query.slice(start, index);
    } else if (query.startsWith('/*', index)) {
      const start = index;
      index += 2;
      let depth = 1;
      while (index < query.length && depth) {
        if (query.startsWith('/*', index)) {
          depth++;
          index += 2;
        } else if (query.startsWith('*/', index)) {
          depth--;
          index += 2;
        } else index++;
      }
      result += query.slice(start, index);
    } else {
      result += character === ';' ? ' ' : character;
      index++;
    }
  }
  return result;
}

class WorkflowSqlCleanupError extends WorkflowError {}

function getSqlErrorMessage(cause: unknown): string {
  return `SQL: ${(cause instanceof Error ? cause.message : String(cause)).split(/[\r\n]/, 1)[0]}`;
}

/** One bounded SQL exchange on a workbench-owned connection, with request inputs removed afterwards. */
export async function runWorkflowSqlOnConnection(
  connection: KeplerDuckDBConnection,
  request: WorkflowSqlRequest,
  signal: AbortSignal,
  assertCurrent: () => void = () => undefined
) {
  const checkCurrent = () => {
    if (signal.aborted) throw new WorkflowError('cancelled', 'SQL: The workflow was cancelled.');
    assertCurrent();
  };
  checkCurrent();
  const problem = getWorkflowSqlProblem(request.query);
  if (problem)
    throw new WorkflowError('parameter-out-of-range', `SQL: Query ${problem}.`, {
      parameter: 'query'
    });
  if (
    !Number.isSafeInteger(request.maximumRows) ||
    request.maximumRows < 1 ||
    request.maximumRows > 100000
  )
    throw new WorkflowError(
      'parameter-out-of-range',
      `SQL: Maximum rows must be between 1 and 100000; got ${request.maximumRows}.`,
      {parameter: 'maximumRows'}
    );
  const inputs = Object.entries(request.tables).sort(([first], [second]) =>
    first < second ? -1 : first > second ? 1 : 0
  );
  for (const [name] of inputs)
    if (!/^input[123]$/.test(name))
      throw new WorkflowError(
        'parameter-out-of-range',
        `SQL: Input table name must be input1, input2 or input3; got ${name}.`
      );
  const registered: string[] = [];
  let failure: unknown;
  let result: ReturnType<typeof makeWorkflowDataFromSqlTable> | undefined;
  try {
    for (const [name, data] of inputs) {
      registered.push(name);
      await connection.insertArrowTable(makeWorkflowArrowTable(data.table), {name});
      checkCurrent();
    }
    // This separate DuckDB instance contains only this request's inputs, so workflow SQL cannot
    // read tables registered in the app's SQL explorer or left by another workflow request.
    const sql = `SELECT * FROM (\n${stripWorkflowSqlTerminator(request.query)}\n) AS workflow_query LIMIT ${request.maximumRows + 1}`;
    const table = await connection.query(sql);
    checkCurrent();
    result = makeWorkflowDataFromSqlTable(table, request);
  } catch (cause) {
    failure = cause;
  } finally {
    const cleanupErrors: unknown[] = [];
    for (const name of registered.reverse()) {
      try {
        await connection.query(`DROP TABLE IF EXISTS "${name}"`);
      } catch (cause) {
        cleanupErrors.push(cause);
      }
    }
    if (cleanupErrors.length) {
      failure = new WorkflowSqlCleanupError(
        'operation-failed',
        getSqlErrorMessage(failure ?? cleanupErrors[0]),
        {
          cause: new AggregateError([...(failure === undefined ? [] : [failure]), ...cleanupErrors])
        }
      );
    }
  }
  checkCurrent();
  if (failure !== undefined) {
    if (failure instanceof WorkflowError) throw failure;
    throw new WorkflowError('operation-failed', getSqlErrorMessage(failure), {cause: failure});
  }
  if (!result)
    throw new WorkflowError('operation-failed', 'SQL: The SQL engine returned no result.');
  return result;
}

/**
 * The Kepler compute app as a workflow host: native and GPU-resident datasets in (all rows or the
 * current selection), step outputs out as published datasets with a layer.
 */
export function createWorkbenchHost(): WorkflowHost {
  type Runtime = NonNullable<typeof appState.runtime>;
  type Publication = Runtime['publications'] extends Map<string, infer Value> ? Value : never;
  type Ownership = {
    runtime: Runtime;
    publication: Publication;
    nativeDataset?: ReturnType<typeof getDatasets>[string];
  };
  const emptyDrawings: readonly WorkflowDrawingFeature[] = [];
  const drawings = new WeakMap<
    readonly WorkflowDrawingFeature[],
    {
      data: ReturnType<typeof makeWorkflowDataFromDrawings>;
      profiles: ReturnType<typeof profileWorkflowDrawings>;
      revision: string;
    }
  >();
  function getDrawingSnapshot() {
    const features =
      store.getState().keplerGl['compute']?.visState?.editor?.features ?? emptyDrawings;
    const cached = drawings.get(features);
    if (cached) return cached;
    const data = makeWorkflowDataFromDrawings(features);
    const snapshot = {
      data,
      profiles: profileWorkflowDrawings(features),
      // Labels are source data as well: renaming a shape must invalidate its source step.
      revision: hashWorkflowValue([
        data.table.columns['drawing_id'],
        data.table.columns['geometry'],
        data.table.columns['label']
      ])
    };
    drawings.set(features, snapshot);
    return snapshot;
  }
  function applyDisplayStyle(
    datasetId: string,
    schema: Parameters<typeof chooseWorkflowDisplayChannel>[0],
    style: WorkflowDisplayStyle | undefined
  ): void {
    const choice = chooseWorkflowDisplayChannel(schema, style);
    if (!choice) return;
    const palette = KEPLER_COLOR_PALETTES.find(candidate => candidate.name === choice.rangeName);
    if (!palette) return;
    const visState = store.getState().keplerGl['compute']?.visState;
    const field = visState?.datasets[datasetId]?.fields.find(
      candidate => candidate.name === choice.field.name
    );
    if (!field || !['real', 'integer', 'string', 'boolean'].includes(field.type)) return;
    for (const layer of visState?.layers ?? []) {
      if (layer.config.dataId !== datasetId) continue;
      try {
        const colorRange = colorPaletteToColorRange(palette, {steps: 6, reversed: false});
        store.dispatch(
          wrapTo(
            'compute',
            layerVisualChannelConfigChange(
              layer,
              // Kepler's action declaration omits the channel props its own executor accepts.
              {colorField: field, colorScale: choice.scale} as never,
              'color',
              {colorRange}
            )
          )
        );
      } catch {
        // Display style is best effort: a renderer may reject a field without losing its output.
      }
    }
  }
  const publications = new Map<string, Ownership>();
  const publishedBounds = new Map<string, ReturnType<typeof getWorkflowDataBounds>>();
  // Exclusion follows dataset ids across recovery; cleanup follows exact runtime generations.
  const publishedDatasetIds = new Set<string>();
  const pendingIds = new Map<string, Runtime>();
  // Kepler replaces index arrays on selection edits. Hash their contents once per array and
  // small configuration signature, keeping planning independent of the source row count.
  const nativeRevisionKeys = new WeakMap<
    object,
    {signature: string; indices: readonly number[] | undefined; key: string}
  >();
  const profiles = new WeakMap<
    object,
    {resident: boolean; dataRevision: number; profile: WorkflowDatasetProfile}
  >();
  let publicationQueue: Promise<unknown> = Promise.resolve();
  let sqlQueue: Promise<unknown> = Promise.resolve();
  // WorkflowHost has no teardown hook; the healthy connection lives for the page's lifetime.
  let sqlConnection: OwnedDuckDBConnection | undefined;
  const getDatasets = () => store.getState().keplerGl['compute']?.visState?.datasets ?? {};
  const isWorkflowDataset = (datasetId: string) => {
    // A replay can create the host after loading its sources. Remember the reserved resident
    // namespace so those outputs stay excluded while recovery clears resident registration.
    if (datasetId.startsWith('workflow-') && isResidentDataset(datasetId))
      publishedDatasetIds.add(datasetId);
    const owned = publications.get(datasetId);
    return (
      pendingIds.get(datasetId) === appState.runtime ||
      publishedDatasetIds.has(datasetId) ||
      Boolean(
        owned &&
          owned.runtime === appState.runtime &&
          owned.runtime.publications.get(datasetId) === owned.publication
      )
    );
  };
  const findDataset = (request: WorkflowDatasetRequest) => {
    const datasets = getDatasets();
    const direct = datasets[request.datasetId];
    if (direct && !isWorkflowDataset(direct.id)) return direct;
    const matching = request.datasetLabel
      ? Object.values(datasets).filter(
          dataset => dataset.label === request.datasetLabel && !isWorkflowDataset(dataset.id)
        )
      : [];
    if (matching.length > 1)
      throw new WorkflowError(
        'dataset-not-found',
        `${request.datasetLabel}: Dataset label matches multiple datasets; select the intended dataset.`
      );
    return matching[0];
  };
  const getExplorer = (datasetId: string) => {
    const runtime = appState.runtime;
    return datasetId === DATASET_ID
      ? runtime?.explorer
      : runtime?.publications.get(datasetId)?.consumer.explorer;
  };
  function checkAborted(signal: AbortSignal, label: string): void {
    if (signal.aborted)
      throw new WorkflowError('cancelled', `${label}: The workflow was cancelled.`);
  }
  function queuePublication<Result>(action: () => Promise<Result>): Promise<Result> {
    const pending = publicationQueue.then(action, action);
    // Both outcomes release the queue; callers still receive the original rejection.
    publicationQueue = pending.then(
      () => undefined,
      () => undefined
    );
    return pending;
  }
  function captureOwnership(runtime: Runtime, datasetId: string): Ownership | undefined {
    const publication = runtime.publications.get(datasetId);
    return publication
      ? {
          runtime,
          publication,
          nativeDataset: appState.runtime === runtime ? getDatasets()[datasetId] : undefined
        }
      : undefined;
  }
  async function removeOwnedPublication(datasetId: string, owned: Ownership): Promise<void> {
    await datasetOperations.run(datasetId, 'remove', async () => {
      // The queue may wait across a workspace restore or a same-ID replacement. Never resolve
      // ownership from appState after waiting: retire only this exact captured generation.
      if (owned.runtime.publications.get(datasetId) !== owned.publication) return;
      if (appState.runtime === owned.runtime) {
        const nativeDataset = getDatasets()[datasetId];
        if (!nativeDataset || nativeDataset === owned.nativeDataset) {
          residentEviction.forget(datasetId);
          fieldPreparationCoordinator.forget(datasetId);
          if (nativeDataset) store.dispatch(removeDataset(datasetId));
          if (
            appState.runtime === owned.runtime &&
            owned.runtime.publications.get(datasetId) === owned.publication
          )
            residentDatasetIds.delete(datasetId);
        }
      }
      // Redux subscribers can synchronously replace a publication during native removal.
      if (owned.runtime.publications.get(datasetId) !== owned.publication) return;
      await owned.runtime.removePublication(datasetId);
      if (appState.runtime === owned.runtime) notify();
    });
  }
  async function removeOrphanedNativePublication(
    runtime: Runtime,
    datasetId: string,
    nativeDataset: NonNullable<Ownership['nativeDataset']>
  ): Promise<void> {
    await datasetOperations.run(datasetId, 'remove', async () => {
      // publishSource may retire GPU data before returning a registration error. Only remove
      // the exact native table observed during our publication, while it still has no GPU owner.
      if (
        appState.runtime !== runtime ||
        runtime.publications.has(datasetId) ||
        getDatasets()[datasetId] !== nativeDataset
      )
        return;
      residentEviction.forget(datasetId);
      fieldPreparationCoordinator.forget(datasetId);
      store.dispatch(removeDataset(datasetId));
      if (appState.runtime === runtime && !getDatasets()[datasetId])
        residentDatasetIds.delete(datasetId);
      if (appState.runtime === runtime) notify();
    });
  }
  return {
    focusOutput: datasetId => {
      const bounds = publishedBounds.get(datasetId);
      if (bounds) store.dispatch(wrapTo('compute', fitBounds(bounds)));
    },
    runSql: (request, signal) => {
      const pending = sqlQueue.then(async () => {
        checkAborted(signal, 'SQL');
        let queryFailed = false;
        let cleanupFailed = false;
        try {
          if (!sqlConnection) {
            const {createConnection} = await import('../duckdb-browser');
            checkAborted(signal, 'SQL');
            sqlConnection = await createConnection();
          }
          checkAborted(signal, 'SQL');
          const connection = sqlConnection;
          return await runWorkflowSqlOnConnection(
            {
              insertArrowTable: (table, options) => connection.insertArrowTable(table, options),
              query: async query => {
                try {
                  return await connection.query(query);
                } catch (cause) {
                  queryFailed = true;
                  if (query.startsWith('DROP TABLE ')) cleanupFailed = true;
                  throw cause;
                }
              }
            },
            request,
            signal
          );
        } catch (cause) {
          let unusable = cleanupFailed || cause instanceof WorkflowSqlCleanupError;
          if (sqlConnection && queryFailed && !unusable) {
            // A binder/parser error leaves DuckDB usable; a failed health query identifies a
            // lost worker/connection without relying on version-specific error message text.
            try {
              await sqlConnection.query('SELECT 1');
            } catch {
              unusable = true;
            }
          }
          if (sqlConnection && unusable) {
            const connection = sqlConnection;
            sqlConnection = undefined;
            try {
              await connection.destroy();
            } catch (cleanupError) {
              throw new WorkflowError(
                signal.aborted ? 'cancelled' : 'operation-failed',
                signal.aborted ? 'SQL: The workflow was cancelled.' : getSqlErrorMessage(cause),
                {
                  cause: new AggregateError([cause, cleanupError])
                }
              );
            }
          }
          checkAborted(signal, 'SQL');
          if (cause instanceof WorkflowError) throw cause;
          throw new WorkflowError('operation-failed', getSqlErrorMessage(cause), {cause});
        }
      });
      sqlQueue = pending.then(
        () => undefined,
        () => undefined
      );
      return pending;
    },
    getDevice: () => {
      const runtime = appState.runtime;
      return runtime && !runtime.session.deviceLost && !runtime.device.isLost
        ? runtime.session.device
        : null;
    },
    listDrawings: () => getDrawingSnapshot().profiles.map(drawing => ({...drawing})),
    getDrawingRevisionKey: () => getDrawingSnapshot().revision,
    readDrawings: async (drawingIds, signal) => {
      checkAborted(signal, 'Drawn shapes');
      const data = getDrawingSnapshot().data;
      const available = new Set(data.table.columns['drawing_id'].map(String));
      const selected = drawingIds === 'all' ? available : new Set(drawingIds);
      for (const identity of selected)
        if (!available.has(identity))
          throw new WorkflowError(
            'dataset-not-found',
            `Drawn shapes: Shape id ${JSON.stringify(identity)} was not found.`,
            {parameter: 'drawingIds'}
          );
      const rows = data.table.columns['drawing_id'].flatMap((identity, index) =>
        selected.has(String(identity)) ? [index] : []
      );
      return makeWorkflowData(
        {
          ...data.table,
          columnTypes: {...data.table.columnTypes},
          rowCount: rows.length,
          columns: Object.fromEntries(
            Object.entries(data.table.columns).map(([name, values]) => [
              name,
              rows.map(row => values[row])
            ])
          )
        },
        {identityColumn: data.identityColumn}
      );
    },
    listDatasets: () =>
      Object.values(getDatasets())
        .filter(dataset => !isWorkflowDataset(dataset.id))
        .map(dataset => {
          const resident = isResidentDataset(dataset.id);
          const cached = profiles.get(dataset);
          if (
            cached &&
            cached.resident === resident &&
            cached.dataRevision === dataset.dataRevision &&
            cached.profile.rowCount === dataset.dataContainer.numRows()
          )
            return cached.profile;
          const profile = profileNativeDataset(dataset, resident);
          profiles.set(dataset, {resident, dataRevision: dataset.dataRevision, profile});
          return profile;
        }),
    getDatasetRevisionKey: request => {
      try {
        const dataset = findDataset(request);
        if (!dataset) return 'missing';
        if (!isResidentDataset(dataset.id)) {
          const visState = store.getState().keplerGl['compute']?.visState;
          const filters = visState?.filters ?? [];
          const polygonLayerIds = new Set(
            filters
              .filter(
                filter =>
                  filter.type === 'polygon' &&
                  filter.dataId.includes(dataset.id) &&
                  filter.enabled !== false
              )
              .flatMap(filter => filter.layerId ?? [])
          );
          const selectionRevision =
            request.population === 'selected'
              ? hashWorkflowValue(
                  (visState?.layers ?? [])
                    .filter(
                      layer => layer.config.dataId === dataset.id && polygonLayerIds.has(layer.id)
                    )
                    .map(layer => [
                      layer.id,
                      layer.type,
                      layer.config.columnMode,
                      Object.entries(layer.config.columns).map(([name, column]) => [
                        name,
                        column?.fieldIdx,
                        column?.value
                      ])
                    ])
                )
              : undefined;
          const signature = getNativeRevisionKey(
            {...dataset, filteredIndex: undefined, selectionRevision},
            filters,
            request.population
          );
          const indices = request.population === 'selected' ? dataset.filteredIndex : undefined;
          const cached = nativeRevisionKeys.get(dataset);
          if (cached?.signature === signature && cached.indices === indices) return cached.key;
          const key = getNativeRevisionKey(
            {...dataset, selectionRevision},
            filters,
            request.population
          );
          nativeRevisionKeys.set(dataset, {signature, indices, key});
          return key;
        }
        const explorer = getExplorer(dataset.id);
        if (!explorer) return 'missing';
        return JSON.stringify([
          dataset.id,
          explorer.dataset.deviceEpoch,
          explorer.dataset.revision,
          explorer.currentRevision,
          request.population
        ]);
      } catch {
        // A disappearing dataset must synchronously invalidate a plan, never break rendering.
        return 'missing';
      }
    },
    readDataset: async (request, signal) => {
      const label = request.datasetLabel ?? request.datasetId;
      checkAborted(signal, label);
      const dataset = findDataset(request);
      if (!dataset)
        throw new WorkflowError(
          'dataset-not-found',
          `${label}: Dataset is no longer available; choose an existing dataset.`
        );
      try {
        if (!isResidentDataset(dataset.id)) {
          const visState = store.getState().keplerGl['compute']?.visState;
          // filteredIndex omits GPU filters. Kepler's export path materialises every filter,
          // including the per-layer polygon union, into filteredIdxCPU on an explicit read.
          if (request.population === 'selected')
            dataset.filterTableCPU(visState?.filters ?? [], visState?.layers ?? []);
          return readNativeDatasetData(
            dataset,
            request,
            request.population === 'selected'
              ? (dataset.filteredIdxCPU ?? dataset.filteredIndex)
              : undefined
          );
        }
        const runtime = appState.runtime;
        const explorer = getExplorer(dataset.id);
        if (!runtime || !explorer || runtime.session.deviceLost)
          throw new WorkflowError(
            'dataset-read-failed',
            `${dataset.label}: Dataset has no active resident explorer.`
          );
        const revision = explorer.currentRevision;
        const count = await explorer.readSelectedCount();
        checkAborted(signal, dataset.label);
        if (
          appState.runtime !== runtime ||
          explorer.currentRevision !== revision ||
          !explorer.isCurrent
        )
          throw new WorkflowError(
            'dataset-read-failed',
            `${dataset.label}: Dataset or selection changed during extraction; run the step again.`
          );
        if (!count || count.revision !== revision)
          throw new WorkflowError(
            'dataset-read-failed',
            `${dataset.label}: Rows are not submitted yet; try again after the map updates.`
          );
        if (request.population === 'all' && count.selectedCount !== explorer.dataset.rowCount)
          throw new WorkflowError(
            'dataset-read-failed',
            `${dataset.label}: All rows cannot be read while filters exclude rows; choose Rows: selected or clear the filters.`
          );
        if (count.selectedCount > request.maximumRows)
          throw new WorkflowError(
            'capacity-exceeded',
            `${dataset.label}: ${count.selectedCount} rows exceed Maximum rows (${request.maximumRows}); filter the dataset or raise Maximum rows.`
          );
        // The extraction API reports empty selections as a distinct non-ready state.
        if (count.selectedCount === 0)
          return makeWorkflowDataFromArrowTable(
            explorer.dataset.value.source.slice(0, 0),
            request,
            dataset.label
          );
        const extracted = await extractSelectedRows(runtime.session, explorer, {
          maximumRows: request.maximumRows,
          maximumOutputBytes: 256 * 1024 * 1024,
          signal
        });
        checkAborted(signal, dataset.label);
        if (
          appState.runtime !== runtime ||
          explorer.currentRevision !== revision ||
          !explorer.isCurrent
        )
          throw new WorkflowError(
            'dataset-read-failed',
            `${dataset.label}: Dataset or selection changed during extraction; run the step again.`
          );
        if (extracted.status !== 'ready')
          throw new WorkflowError('dataset-read-failed', `${dataset.label}: ${extracted.reason}`);
        const data = makeWorkflowDataFromArrowTable(extracted.value.table, request, dataset.label);
        data.table.columns['row_index'] = Array.from(extracted.value.sourceRowIndices, row => {
          validateWorkflowRowIdentity(row, dataset.label);
          return row;
        });
        return data;
      } catch (cause) {
        checkAborted(signal, dataset.label);
        if (cause instanceof WorkflowError) throw cause;
        throw new WorkflowError(
          'dataset-read-failed',
          `${dataset.label}: Dataset read failed: ${cause instanceof Error ? cause.message : String(cause)}`,
          {cause}
        );
      }
    },
    publishOutput: (request, signal) =>
      queuePublication(async () => {
        checkAborted(signal, request.label);
        const runtime = appState.runtime;
        if (!runtime || runtime.session.deviceLost)
          throw new WorkflowError(
            'publish-failed',
            `${request.label}: Publication requires an active device.`
          );
        // Escape the separator inside each component: a-b/c and a/b-c must own different ids.
        const expectedDatasetId = `workflow-${[request.workflowId, request.stepId, request.output]
          .map(component => encodeURIComponent(component).replace(/-/g, '%2D'))
          .join('-')}`;
        const datasetId = request.previousDatasetId ?? expectedDatasetId;
        const previousPublication = runtime.publications.get(datasetId);
        const previousNativeDataset = getDatasets()[datasetId];
        if (datasetId !== expectedDatasetId)
          throw new WorkflowError(
            'publish-failed',
            `${request.label}: The previous dataset id does not identify this workflow output; hide the output and display it again.`
          );
        if (
          ((previousPublication || previousNativeDataset) && !request.previousDatasetId) ||
          (previousNativeDataset && !previousPublication)
        )
          throw new WorkflowError(
            'publish-failed',
            `${request.label}: Dataset id ${datasetId} is already occupied; hide this output and display it again, or remove the conflicting dataset.`
          );
        let observedNativeDataset: Ownership['nativeDataset'];
        let publicationStarted = false;
        try {
          const publication = makeWorkflowPublication(request.data, request.label);
          if (request.previousDatasetId) {
            const previous = captureOwnership(runtime, datasetId);
            if (previous) await removeOwnedPublication(datasetId, previous);
            if (runtime.publications.has(datasetId) || getDatasets()[datasetId])
              throw new WorkflowError(
                'publish-failed',
                `${request.label}: The dataset changed during replacement; run the step again.`
              );
          }
          checkAborted(signal, request.label);
          if (appState.runtime !== runtime)
            throw new WorkflowError(
              'publish-failed',
              `${request.label}: Workspace changed before publication.`
            );
          // The map copy rounds numeric columns to 32-bit floats using automatic preparation's
          // budget. The workflow keeps its exact table for results, downstream steps and downloads.
          const normalization = {
            columns: publication.table.schema.fields
              .map(field => field.name)
              .filter(name => name !== publication.geometryColumn),
            float64: WORKFLOW_DISPLAY_FLOAT64_POLICY
          };
          pendingIds.set(datasetId, runtime);
          appState.requestedPublicationId = datasetId;
          publicationStarted = true;
          await publishSource(
            {
              table: publication.table,
              label: publication.label,
              isCurrent: () => {
                const current = appState.runtime === runtime && !signal.aborted;
                if (current && runtime.publications.get(datasetId) !== previousPublication) {
                  const nativeDataset = getDatasets()[datasetId];
                  if (nativeDataset !== previousNativeDataset)
                    observedNativeDataset = nativeDataset;
                }
                return current;
              }
            },
            normalization,
            publication.coordinates,
            publication.geometryColumn
          );
          checkAborted(signal, request.label);
          const owned = captureOwnership(runtime, datasetId);
          if (appState.runtime !== runtime || !owned || owned.publication === previousPublication)
            throw new WorkflowError(
              'publish-failed',
              `${request.label}: Workspace or publication changed during publication.`
            );
          applyDisplayStyle(datasetId, publication.table.schema, request.style);
          const bounds = getWorkflowDataBounds(request.data);
          if (!request.previousDatasetId) {
            if (bounds) store.dispatch(wrapTo('compute', fitBounds(bounds)));
          }
          if (
            appState.runtime !== runtime ||
            runtime.publications.get(datasetId) !== owned.publication
          )
            throw new WorkflowError(
              'publish-failed',
              `${request.label}: Workspace or publication changed during publication.`
            );
          publications.set(datasetId, owned);
          publishedBounds.set(datasetId, bounds);
          publishedDatasetIds.add(datasetId);
          return {datasetId, label: publication.label};
        } catch (cause) {
          if (appState.requestedPublicationId === datasetId)
            appState.requestedPublicationId = undefined;
          const partial = captureOwnership(runtime, datasetId);
          try {
            if (publicationStarted && partial && partial.publication !== previousPublication)
              await removeOwnedPublication(datasetId, {
                ...partial,
                nativeDataset: observedNativeDataset ?? partial.nativeDataset
              });
            else if (publicationStarted && !partial && observedNativeDataset)
              await removeOrphanedNativePublication(runtime, datasetId, observedNativeDataset);
          } catch (cleanupError) {
            throw new WorkflowError(
              'publish-failed',
              `${request.label}: Publication failed and its output could not be removed: ${cleanupError instanceof Error ? cleanupError.message : String(cleanupError)}`,
              {cause: new AggregateError([cause, cleanupError])}
            );
          }
          checkAborted(signal, request.label);
          if (cause instanceof WorkflowError) throw cause;
          const message = cause instanceof Error ? cause.message : String(cause);
          const precisionFailure = /^Float64 column "(.+)" exceeds its precision policy$/s.exec(
            message
          );
          throw new WorkflowError(
            'publish-failed',
            precisionFailure
              ? `${request.label}: Column "${precisionFailure[1]}" cannot be shown at 32-bit precision; remove it with Select columns before displaying.`
              : `${request.label}: Publication failed: ${message}`,
            {cause}
          );
        } finally {
          pendingIds.delete(datasetId);
        }
      }),
    retractOutput: datasetId =>
      queuePublication(async () => {
        const owned = publications.get(datasetId);
        if (!owned) return;
        try {
          await removeOwnedPublication(datasetId, owned);
          if (publications.get(datasetId) === owned) {
            publications.delete(datasetId);
            if (!appState.runtime?.publications.has(datasetId)) {
              publishedBounds.delete(datasetId);
              publishedDatasetIds.delete(datasetId);
            }
          }
        } catch (cause) {
          throw new WorkflowError(
            'publish-failed',
            `${datasetId}: Removing the published output failed: ${cause instanceof Error ? cause.message : String(cause)}`,
            {cause}
          );
        }
      })
  };
}
