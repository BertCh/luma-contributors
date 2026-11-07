// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ComputeRuntime} from './compute-runtime';
import {type ResidentGroupByResult, executeResidentGroupBy} from './resident-group-by-runtime';
import {type ResidentGroupByPlan, createResidentGroupByPlan} from './resident-group-by-plan';
import {appState} from './app-state';
import {
  ResidentGroupByRequests,
  RESIDENT_GROUP_BY_STATUS,
  getResidentGroupByRequest,
  getResidentGroupByRequestKey
} from './resident-action-guards';
import {type KeplerArrowDataset, type KeplerDatasetSnapshot} from '@luma-contributors/kepler';
import {
  describeResidentGroupByPolicy,
  describeResidentSpatialJoinPolicy,
  describeResidentJoinPolicy
} from './analysis2-policy-binding';
import {type Middleware} from 'redux';
import {type KeplerGlState} from '@kepler.gl/reducers';
import {runGroupBy, runSpatialJoin, runJoin, runSuitability} from '@kepler.gl/actions';
import {type ResidentJoinResult, executeResidentJoin} from './resident-join-runtime';
import {
  type ResidentSpatialJoinResult,
  executeResidentSpatialJoin
} from './resident-spatial-join-runtime';
import {type ResidentJoinPlan, createResidentJoinPlan} from './resident-join-plan';
import {
  type ResidentSpatialJoinPlan,
  type ResidentSpatialJoinSourceMetadata,
  createResidentSpatialJoinPlan
} from './resident-spatial-join-plan';
import {
  type ResidentJoinRequest,
  type ResidentSpatialJoinRequest,
  RESIDENT_JOIN_STATUS,
  getResidentJoinRequest,
  getResidentSpatialJoinRequest,
  getResidentSpatialJoinRequestKey,
  getResidentJoinRequestKey
} from './resident-join-actions';
import {runTrackedOperation} from './tracked-operations';
import {makeResidentSpatialJoinSourcePositions} from './resident-spatial-join-reproject';
import {packSpatialJoinPolygons} from './resident-spatial-join-polygons';
import {
  type ResidentSuitabilityRequest,
  RESIDENT_SUITABILITY_STATUS,
  getResidentSuitabilityRequest,
  getResidentSuitabilityRequestKey
} from './resident-suitability-actions';
import {createResidentSuitabilityResultPlan} from './resident-suitability-plan';
import {
  executeResidentSuitability,
  type ResidentSuitabilityResult
} from './resident-suitability-runtime';
import {type ResidentResultOperation} from './resident-result-recipes';
import {notify, reportError, residentResultRecipes} from './app-services';
import {store} from './app-store';
import {attachPublishedDataset, discardPublishedResult, isResidentDataset} from './publication';

type ResidentGroupByExecution = {
  owner: ComputeRuntime;
  result: ResidentGroupByResult;
  plan: ResidentGroupByPlan;
};

/** Shows the policy line of a running resident operation in the workflow notice area. */
function reportPolicyNotice(notices: Map<string, string>, operationId: string, policy: string) {
  const current = notices.get(operationId);
  if (current === undefined) return;
  notices.set(operationId, `${current} ${policy}`);
  appState.workflowNotice = [...notices.values()].at(-1) ?? appState.workflowNotice;
  notify();
}

const groupByNotices = new Map<string, string>();

export const groupByRequests = new ResidentGroupByRequests<ResidentGroupByExecution>({
  async execute(request, signal, isCurrent) {
    const owner = appState.runtime;
    const source = owner?.session.registry.getDataset<KeplerArrowDataset>(request.dataId);
    const nativeSource = store.getState().keplerGl['compute']?.visState?.datasets[request.dataId];
    if (!owner || !source || !nativeSource || !isCurrent())
      throw new Error('Group By source is unavailable.');
    const nullCounts: Record<string, readonly number[]> = Object.create(null);
    for (const [name, counts] of Object.entries(source.value.analytics.nullCounts))
      if (counts) nullCounts[name] = counts;
    const columns: Record<string, {format: string}> = Object.create(null);
    for (const [name, column] of Object.entries(source.value.columns))
      if (column.format) columns[name] = {format: column.format};
    const plan = createResidentGroupByPlan(request, {
      datasetId: source.id,
      revision: source.revision,
      rowCount: source.rowCount,
      fields: nativeSource.fields.map(field => ({
        name: field.name,
        type: field.type,
        analyzerType: field.analyzerType
      })),
      columns,
      dictionaries: source.value.prepared.dictionaries,
      nullCounts
    });
    reportPolicyNotice(
      groupByNotices,
      plan.operationId,
      describeResidentGroupByPolicy(plan, source.rowCount)
    );
    const result = await executeResidentGroupBy({
      session: owner.session,
      source: {datasetId: source.id, revision: source.revision},
      plan,
      signal,
      isCurrent
    });
    return {owner, result, plan};
  },
  async attach({owner, result, plan}, isCurrent) {
    const current = () => appState.runtime === owner && isCurrent() && result.isCurrent();
    if (!current()) throw new Error('Group By request changed before publication.');
    await attachPublishedDataset(
      owner,
      result.dataset,
      plan.resultLabel,
      undefined,
      undefined,
      undefined,
      undefined,
      plan.provenance,
      current
    );
  },
  transfer({result}) {
    result.transfer();
  },
  discard: result => discardPublishedResult(result, 'Group By'),
  status(status) {
    store.dispatch({type: RESIDENT_GROUP_BY_STATUS, ...status});
    const previousNotice = appState.groupByNotice;
    if (status.phase === 'running') {
      groupByNotices.set(
        status.id,
        `Running resident Group By (${status.id}) over all source rows…`
      );
    } else groupByNotices.delete(status.id);
    appState.groupByNotice = [...groupByNotices.values()].at(-1) ?? '';
    if (status.phase === 'running' || appState.workflowNotice === previousNotice)
      appState.workflowNotice =
        appState.groupByNotice ||
        (status.phase === 'complete' ? 'Group By result added as a resident dataset.' : '');
    notify();
  },
  cleanupError: reportError
});

export const residentGroupByMiddleware: Middleware<
  {},
  {keplerGl: Record<string, Partial<KeplerGlState>>}
> = () => next => action => {
  if (
    !action ||
    typeof action !== 'object' ||
    !('type' in action) ||
    typeof action.type !== 'string'
  )
    return next(action);
  const request = getResidentGroupByRequest(
    store.getState().keplerGl['compute'],
    {...action, type: action.type},
    isResidentDataset
  );
  if (!request) return next(action);
  const owner = appState.runtime;
  const source = owner?.session.registry.getDataset(request.dataId);
  const key = getResidentGroupByRequestKey(request);
  const current = () => {
    const latest = getResidentGroupByRequest(
      store.getState().keplerGl['compute'],
      runGroupBy(request.id),
      isResidentDataset
    );
    return (
      appState.runtime === owner &&
      !owner?.session.deviceLost &&
      (!source || owner?.session.registry.getDataset(request.dataId) === source) &&
      Boolean(latest && getResidentGroupByRequestKey(latest) === key)
    );
  };
  void groupByRequests.run(request, current);
  return action;
};

export type ResidentJoinExecution = {
  owner: ComputeRuntime;
  result: ResidentJoinResult | ResidentSpatialJoinResult;
  plan: ResidentJoinPlan | ResidentSpatialJoinPlan;
};

function readJoinSourceMetadata(
  owner: ComputeRuntime,
  datasetId: string
): ResidentSpatialJoinSourceMetadata {
  const source = owner.session.registry.getDataset<KeplerArrowDataset>(datasetId);
  const published = owner.publications.get(datasetId);
  const nativeSource = store.getState().keplerGl['compute']?.visState?.datasets[datasetId];
  if (!source || !nativeSource) throw new Error('Join source is unavailable.');
  const nullCounts: Record<string, readonly number[]> = Object.create(null);
  for (const [name, counts] of Object.entries(source.value.analytics.nullCounts))
    if (counts) nullCounts[name] = counts;
  const columns: Record<string, {format: string}> = Object.create(null);
  for (const [name, column] of Object.entries(source.value.columns))
    if (column.format) columns[name] = {format: column.format};
  return {
    datasetId: source.id,
    label: nativeSource.label,
    revision: source.revision,
    rowCount: source.rowCount,
    fields: nativeSource.fields.map(field => ({
      name: field.name,
      type: field.type,
      analyzerType: field.analyzerType
    })),
    columns,
    dictionaries: source.value.prepared.dictionaries,
    nullCounts,
    ...(published?.coordinates
      ? {
          coordinates: {
            longitude: published.coordinates.longitude,
            latitude: published.coordinates.latitude
          }
        }
      : {}),
    ...(published?.geometryColumn ? {geometryColumn: published.geometryColumn} : {})
  };
}

const joinNotices = new Map<string, string>();

export const joinRequests = new ResidentGroupByRequests<
  ResidentJoinExecution,
  ResidentJoinRequest | ResidentSpatialJoinRequest
>({
  async execute(request, signal, isCurrent) {
    const owner = appState.runtime;
    if (!owner || !isCurrent()) throw new Error('Join sources are unavailable.');
    return executeJoinRequest(owner, request, signal, isCurrent);
  },
  async attach({owner, result, plan}, isCurrent) {
    const current = () => appState.runtime === owner && isCurrent() && result.isCurrent();
    if (!current()) throw new Error('Join request changed before publication.');
    await attachPublishedDataset(
      owner,
      result.dataset,
      plan.resultLabel,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      current,
      plan.provenance
    );
  },
  transfer({result}) {
    result.transfer();
  },
  discard: result => discardPublishedResult(result, 'Join'),
  status(status) {
    store.dispatch({type: RESIDENT_JOIN_STATUS, ...status});
    const previousNotice = appState.joinNotice;
    if (status.phase === 'running')
      joinNotices.set(status.id, `Running resident Join (${status.id}) over bounded inputs…`);
    else joinNotices.delete(status.id);
    appState.joinNotice = [...joinNotices.values()].at(-1) ?? '';
    if (status.phase === 'running' || appState.workflowNotice === previousNotice)
      appState.workflowNotice =
        appState.joinNotice ||
        (status.phase === 'complete' ? 'Join result added as a resident dataset.' : '');
    notify();
  },
  cleanupError: reportError
});

/** Joins are tracked operations: the memory manager shows them and can cancel them. */
export function executeJoinRequest(
  owner: ComputeRuntime,
  request: ResidentJoinRequest | ResidentSpatialJoinRequest,
  signal: AbortSignal,
  isCurrent: () => boolean
): Promise<ResidentJoinExecution> {
  return runTrackedOperation(
    `${'kind' in request ? 'Spatial join' : 'Join'} ${request.leftDataId} with ${request.rightDataId}`,
    signal,
    trackedSignal => executeJoinRequestUntracked(owner, request, trackedSignal, isCurrent)
  );
}

async function executeJoinRequestUntracked(
  owner: ComputeRuntime,
  request: ResidentJoinRequest | ResidentSpatialJoinRequest,
  signal: AbortSignal,
  isCurrent: () => boolean
): Promise<ResidentJoinExecution> {
  let execution: ResidentJoinExecution;
  if ('kind' in request) {
    const left = owner.publications.get(request.leftDataId);
    const right = owner.publications.get(request.rightDataId);
    const plan = createResidentSpatialJoinPlan(
      request,
      readJoinSourceMetadata(owner, request.leftDataId),
      readJoinSourceMetadata(owner, request.rightDataId)
    );
    reportPolicyNotice(joinNotices, plan.operationId, describeResidentSpatialJoinPolicy(plan));
    if (!left?.positions || !right?.polygonSource)
      throw new Error(
        'Spatial Join requires resident points and a retained polygon source. Use SQL for other inputs.'
      );
    // A projected left CRS (UTM, 3857) was hidden as NaN at ingestion; upload its raw coordinates.
    const projectedPositions =
      plan.provenance.crs?.points !== 'EPSG:4326' && left.coordinates
        ? makeResidentSpatialJoinSourcePositions(
            owner.session.device,
            left.dataset.value.prepared,
            left.coordinates,
            owner.session.registry
          )
        : undefined;
    let result: ResidentSpatialJoinResult;
    try {
      result = await executeResidentSpatialJoin({
        session: owner.session,
        plan,
        positions: projectedPositions?.positions ?? left.positions,
        polygons: packSpatialJoinPolygons(right.polygonSource.source),
        mergeSource: right.polygonSource.source,
        signal,
        isCurrent
      });
    } finally {
      await projectedPositions?.destroy(Promise.resolve());
    }
    execution = {owner, result, plan};
  } else {
    const plan = createResidentJoinPlan(
      {...request, leftField: request.leftField, rightField: request.rightField},
      readJoinSourceMetadata(owner, request.leftDataId),
      readJoinSourceMetadata(owner, request.rightDataId)
    );
    reportPolicyNotice(joinNotices, plan.operationId, describeResidentJoinPolicy(plan));
    const result = await executeResidentJoin({session: owner.session, plan, signal, isCurrent});
    execution = {owner, result, plan};
  }
  recordResidentResultRecipe(
    execution.result.dataset,
    [execution.plan.left, execution.plan.right],
    'kind' in request ? {kind: 'spatial-join', request} : {kind: 'join', request}
  );
  return execution;
}

export async function executeSuitabilityRequest(
  owner: ComputeRuntime,
  request: ResidentSuitabilityRequest,
  signal: AbortSignal,
  isCurrent: () => boolean
): Promise<ResidentSuitabilityExecution> {
  const metadata = readJoinSourceMetadata(owner, request.dataId);
  const plan = createResidentSuitabilityResultPlan(request, metadata);
  if (plan.scoring.notes.length)
    reportPolicyNotice(suitabilityNotices, plan.operationId, plan.scoring.notes.join(' '));
  const result = await executeResidentSuitability({
    session: owner.session,
    plan,
    signal,
    isCurrent
  });
  recordResidentResultRecipe(result.dataset, [plan.source], {kind: 'suitability', request});
  if (result.partialMissingRowCount > 0)
    reportPolicyNotice(
      suitabilityNotices,
      plan.operationId,
      `${result.partialMissingRowCount} rows were missing in some weighted fields and were ${
        plan.scoring.missingValuePolicy === 'renormalize'
          ? 'scored over their defined fields with renormalized weights'
          : 'scored null'
      }.`
    );
  return {owner, result, plan};
}

/** Over-cap results have no CPU source; their lineage lets recovery and eviction recompute them. */
function recordResidentResultRecipe(
  dataset: KeplerDatasetSnapshot<KeplerArrowDataset>,
  inputs: readonly {datasetId: string; revision: number}[],
  operation: ResidentResultOperation
): void {
  if (!dataset.value.resident) return;
  residentResultRecipes.record({
    resultId: dataset.id,
    rowCount: dataset.rowCount,
    inputs: inputs.map(({datasetId, revision}) => ({datasetId, revision})),
    operation
  });
}

export const residentJoinMiddleware: Middleware<
  {},
  {keplerGl: Record<string, Partial<KeplerGlState>>}
> = () => next => action => {
  if (
    !action ||
    typeof action !== 'object' ||
    !('type' in action) ||
    typeof action.type !== 'string'
  )
    return next(action);
  const request =
    getResidentJoinRequest(
      store.getState().keplerGl['compute'],
      {...action, type: action.type},
      isResidentDataset
    ) ??
    getResidentSpatialJoinRequest(
      store.getState().keplerGl['compute'],
      {...action, type: action.type},
      isResidentDataset
    );
  if (!request) return next(action);
  const owner = appState.runtime;
  const sourceIds = [request.leftDataId, request.rightDataId];
  const sources = sourceIds.map(id => owner?.session.registry.getDataset(id));
  const getKey = (value: ResidentJoinRequest | ResidentSpatialJoinRequest) =>
    'kind' in value ? getResidentSpatialJoinRequestKey(value) : getResidentJoinRequestKey(value);
  const key = getKey(request);
  const current = () => {
    const state = store.getState().keplerGl['compute'];
    const latest =
      'kind' in request
        ? getResidentSpatialJoinRequest(state, runSpatialJoin(request.id), isResidentDataset)
        : getResidentJoinRequest(state, runJoin(request.id), isResidentDataset);
    return (
      appState.runtime === owner &&
      !owner?.session.deviceLost &&
      sources.every(
        (source, index) =>
          !source || owner?.session.registry.getDataset(sourceIds[index]) === source
      ) &&
      Boolean(latest && getKey(latest) === key)
    );
  };
  void joinRequests.run(request, current);
  return action;
};

export type ResidentSuitabilityExecution = {
  owner: ComputeRuntime;
  result: ResidentSuitabilityResult;
  plan: ReturnType<typeof createResidentSuitabilityResultPlan>;
};

const suitabilityNotices = new Map<string, string>();

export const suitabilityRequests = new ResidentGroupByRequests<
  ResidentSuitabilityExecution,
  ResidentSuitabilityRequest
>({
  async execute(request, signal, isCurrent) {
    const owner = appState.runtime;
    if (!owner || !isCurrent()) throw new Error('Suitability source is unavailable.');
    return executeSuitabilityRequest(owner, request, signal, isCurrent);
  },
  async attach({owner, result, plan}, isCurrent) {
    const current = () => appState.runtime === owner && isCurrent() && result.isCurrent();
    if (!current()) throw new Error('Suitability request changed before publication.');
    await attachPublishedDataset(
      owner,
      result.dataset,
      plan.resultLabel,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      current,
      undefined,
      plan.provenance
    );
  },
  transfer({result}) {
    result.transfer();
  },
  discard: result => discardPublishedResult(result, 'Suitability'),
  status(status) {
    store.dispatch({type: RESIDENT_SUITABILITY_STATUS, ...status});
    const previousNotice = appState.suitabilityNotice;
    if (status.phase === 'running')
      suitabilityNotices.set(
        status.id,
        `Running resident Suitability (${status.id}) on GPU kernels (up to 16,777,216 rows)…`
      );
    else suitabilityNotices.delete(status.id);
    appState.suitabilityNotice = [...suitabilityNotices.values()].at(-1) ?? '';
    if (status.phase === 'running' || appState.workflowNotice === previousNotice)
      appState.workflowNotice =
        appState.suitabilityNotice ||
        (status.phase === 'complete' ? 'Suitability result added as a resident dataset.' : '');
    notify();
  },
  cleanupError: reportError
});

export const residentSuitabilityMiddleware: Middleware<
  {},
  {keplerGl: Record<string, Partial<KeplerGlState>>}
> = () => next => action => {
  if (
    !action ||
    typeof action !== 'object' ||
    !('type' in action) ||
    typeof action.type !== 'string'
  )
    return next(action);
  const request = getResidentSuitabilityRequest(
    store.getState().keplerGl['compute'],
    {...action, type: action.type},
    isResidentDataset
  );
  if (!request) return next(action);
  const owner = appState.runtime;
  const source = owner?.session.registry.getDataset(request.dataId);
  const key = getResidentSuitabilityRequestKey(request);
  const current = () => {
    const latest = getResidentSuitabilityRequest(
      store.getState().keplerGl['compute'],
      runSuitability(request.id),
      isResidentDataset
    );
    return (
      appState.runtime === owner &&
      !owner?.session.deviceLost &&
      (!source || owner?.session.registry.getDataset(request.dataId) === source) &&
      Boolean(latest && getResidentSuitabilityRequestKey(latest) === key)
    );
  };
  void suitabilityRequests.run(request, current);
  return action;
};
