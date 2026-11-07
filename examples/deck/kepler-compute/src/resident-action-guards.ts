// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ActionTypes} from '@kepler.gl/actions';
import {LAYER_TYPES} from '@kepler.gl/constants';
import type {UnknownAction} from 'redux';

type DatasetOperation = {
  id: string;
  dataId?: string | null;
  leftDataId?: string | null;
  rightDataId?: string | null;
  fieldName?: string | null;
  aggregations?: Readonly<Record<string, string>>;
  resultId?: string;
  resultLabel?: string;
};
export type ResidentGroupByRequest = {
  id: string;
  dataId: string;
  fieldName: string | null;
  aggregations: Readonly<Record<string, string>>;
  resultId: string;
  resultLabel: string;
};
export type ResidentGroupByStatus = {
  id: string;
  requestId: number;
  phase: 'running' | 'failed' | 'cancelled' | 'complete';
  error?: string;
};
export const RESIDENT_GROUP_BY_STATUS = 'LUMA_GROUP_BY_STATUS';

export function getResidentGroupByStatus(action: UnknownAction): ResidentGroupByStatus | undefined {
  if (action.type !== RESIDENT_GROUP_BY_STATUS) return undefined;
  const {id, requestId, phase, error} = action;
  if (
    typeof id !== 'string' ||
    typeof requestId !== 'number' ||
    (phase !== 'running' && phase !== 'complete' && phase !== 'failed' && phase !== 'cancelled')
  )
    return undefined;
  return {id, requestId, phase, ...(typeof error === 'string' ? {error} : {})};
}

export function updateResidentGroupByStatus<Operation extends {id: string}>(
  operations: readonly Operation[],
  status: ResidentGroupByStatus
): Operation[] {
  return operations.flatMap(operation => {
    if (
      operation.id !== status.id ||
      (status.phase !== 'running' &&
        (!('residentRequestId' in operation) || operation.residentRequestId !== status.requestId))
    )
      return [operation];
    if (status.phase === 'complete') return [];
    return [
      {
        ...operation,
        residentRequestId: status.requestId,
        residentStatus: status.phase,
        error: status.phase === 'failed' ? (status.error ?? 'Resident Group By failed.') : null
      }
    ];
  });
}
export type ResidentOperationState = {
  visState?: {
    groupBys?: readonly DatasetOperation[];
    joins?: readonly DatasetOperation[];
    suitabilities?: readonly DatasetOperation[];
    layers?: readonly {id: string; type?: string | null; config: {dataId?: string | null}}[];
  };
  mapState?: {isSplit?: boolean};
};
export type ResidentActionRejection = {
  reason: string;
  route?: 'sql' | 'analysis';
  operation?: {list: 'groupBys' | 'joins' | 'suitabilities'; id: string};
};
export const SINGLE_VIEW_REASON =
  'This resident workspace owns one Deck device and supports one map view. Split maps and swipe comparison require separate view ownership; keep the current single view.';

/** RUN_GROUP_BY carries an operation ID; snapshot its editable native configuration before work. */
export function getResidentGroupByRequest(
  state: ResidentOperationState | undefined,
  action: UnknownAction,
  isResidentDataset: (datasetId: string) => boolean,
  instanceId = 'compute'
): ResidentGroupByRequest | undefined {
  const nativeAction = unwrapForInstance(action, instanceId);
  if (nativeAction?.type !== ActionTypes.RUN_GROUP_BY) return undefined;
  const operation = state?.visState?.groupBys?.find(
    candidate => candidate.id === nativeAction['id']
  );
  if (!operation?.dataId || !isResidentDataset(operation.dataId)) return undefined;
  return {
    id: operation.id,
    dataId: operation.dataId,
    fieldName: operation.fieldName ?? null,
    aggregations: {...operation.aggregations},
    resultId: operation.resultId ?? '',
    resultLabel: operation.resultLabel ?? ''
  };
}

export function getResidentGroupByRequestKey(request: ResidentGroupByRequest): string {
  // Aggregation insertion order determines native result column order.
  return JSON.stringify([
    request.id,
    request.dataId,
    request.fieldName,
    Object.entries(request.aggregations),
    request.resultId,
    request.resultLabel
  ]);
}

/** Async action ownership stays outside Redux reducers and never invokes a CPU fallback. */
export class ResidentGroupByRequests<
  Result,
  Request extends {id: string} = ResidentGroupByRequest
> {
  private sequence = 0;
  private readonly pending = new Map<
    string,
    {
      requestId: number;
      controller: AbortController;
      isCurrent(): boolean;
    }
  >();

  constructor(
    private readonly callbacks: {
      execute(request: Request, signal: AbortSignal, isCurrent: () => boolean): Promise<Result>;
      attach(result: Result, isCurrent: () => boolean): Promise<void>;
      transfer(result: Result): void;
      discard(result: Result): Promise<void>;
      status(status: ResidentGroupByStatus): void;
      cleanupError(error: unknown): void;
    }
  ) {}

  run(request: Request, ownerIsCurrent: () => boolean): Promise<void> {
    this.pending.get(request.id)?.controller.abort();
    const requestId = ++this.sequence;
    const controller = new AbortController();
    const current = () =>
      this.pending.get(request.id)?.requestId === requestId &&
      !controller.signal.aborted &&
      ownerIsCurrent();
    this.pending.set(request.id, {requestId, controller, isCurrent: current});
    this.callbacks.status({id: request.id, requestId, phase: 'running'});
    return Promise.resolve().then(async () => {
      let result: Result | undefined;
      let transferred = false;
      try {
        if (!current()) return;
        result = await this.callbacks.execute(request, controller.signal, current);
        if (!current()) return;
        await this.callbacks.attach(result, current);
        if (!current()) return;
        this.callbacks.transfer(result);
        transferred = true;
        this.pending.delete(request.id);
        this.callbacks.status({id: request.id, requestId, phase: 'complete'});
      } catch (error) {
        if (current())
          this.callbacks.status({
            id: request.id,
            requestId,
            phase: 'failed',
            error: error instanceof Error ? error.message : String(error)
          });
      } finally {
        if (result !== undefined && !transferred) {
          try {
            await this.callbacks.discard(result);
          } catch (error) {
            this.callbacks.cleanupError(error);
          }
        }
        if (this.pending.get(request.id)?.requestId === requestId) {
          this.pending.delete(request.id);
          if (controller.signal.aborted || !ownerIsCurrent())
            this.callbacks.status({id: request.id, requestId, phase: 'cancelled'});
        }
      }
    });
  }

  reconcile(): void {
    for (const [id, request] of this.pending)
      if (!request.isCurrent()) {
        request.controller.abort();
        this.pending.delete(id);
        this.callbacks.status({id, requestId: request.requestId, phase: 'cancelled'});
      }
  }
}

/** Inspect only configuration metadata, before source teardown or native reducer side effects. */
export function assertSingleViewConfiguration(value: unknown, supportedViewCount = 1): void {
  let configuration = record(value);
  for (let depth = 0; configuration && depth < 4; depth++) {
    const visState = record(configuration['visState']);
    const mapState = record(configuration['mapState']);
    const splitMaps = visState?.['splitMaps'];
    const splitViewports = mapState?.['splitMapViewports'];
    const mode = mapState?.['mapSplitMode'];
    // Two views are admitted only when the host declares shared-device view support.
    const viewCount = Array.isArray(splitMaps) ? splitMaps.length : 0;
    const splitMode = mode !== undefined && mode !== null && mode !== 'SINGLE_MAP';
    if (
      supportedViewCount < 2
        ? viewCount > 0 ||
          mapState?.['isSplit'] === true ||
          (Array.isArray(splitViewports) && splitViewports.length > 1) ||
          splitMode
        : viewCount > supportedViewCount ||
          (Array.isArray(splitViewports) && splitViewports.length > supportedViewCount) ||
          (splitMode && mode !== 'DUAL_MAP' && mode !== 'SWIPE_COMPARE')
    )
      throw new Error(SINGLE_VIEW_REASON);
    configuration = record(configuration['config']);
  }
}

/** Return a rejection before the pinned reducer can synchronously scan retained source rows. */
export function getResidentActionRejection(
  state: ResidentOperationState | undefined,
  action: UnknownAction,
  isResidentDataset: (datasetId: string) => boolean,
  instanceId = 'compute',
  supportedViewCount = 1
): ResidentActionRejection | undefined {
  const nativeAction = unwrapForInstance(action, instanceId);
  if (!nativeAction) return undefined;
  const payload = record(nativeAction['payload']);
  if (nativeAction.type === ActionTypes.EXTRACT_DATA_FROM_FEATURE) {
    const layer = state?.visState?.layers?.find(
      candidate => candidate.id === nativeAction['layerId']
    );
    if (
      layer?.config.dataId &&
      isResidentDataset(layer.config.dataId) &&
      layer.type !== LAYER_TYPES.vectorTile &&
      layer.type !== LAYER_TYPES.rasterTile
    )
      return {
        reason:
          'Native Extract data runs a retained-row CPU filter and is unavailable for resident sources. Use Filter Layers to apply this geometry, then Export Data to explicitly materialize the selected rows.'
      };
  }
  if (
    nativeAction.type === ActionTypes.TOGGLE_SPLIT_MAP &&
    !state?.mapState?.isSplit &&
    supportedViewCount < 2
  )
    return {reason: SINGLE_VIEW_REASON};
  if (
    nativeAction.type === ActionTypes.SET_MAP_SPLIT_MODE &&
    payload?.['mapSplitMode'] !== 'SINGLE_MAP' &&
    (supportedViewCount < 2 ||
      (payload?.['mapSplitMode'] !== 'DUAL_MAP' && payload?.['mapSplitMode'] !== 'SWIPE_COMPARE'))
  )
    return {reason: SINGLE_VIEW_REASON};
  try {
    if (
      nativeAction.type === ActionTypes.RECEIVE_MAP_CONFIG ||
      nativeAction.type === ActionTypes.ADD_DATA_TO_MAP
    )
      assertSingleViewConfiguration(payload?.['config'], supportedViewCount);
    if (nativeAction.type === ActionTypes.UPDATE_MAP)
      assertSingleViewConfiguration({mapState: payload?.['viewport']}, supportedViewCount);
    if (nativeAction.type === ActionTypes.APPLY_MAP_STATE)
      assertSingleViewConfiguration({mapState: nativeAction['payload']}, supportedViewCount);
  } catch (error) {
    return {reason: error instanceof Error ? error.message : SINGLE_VIEW_REASON};
  }
  const operationId = nativeAction['id'];
  if (typeof operationId !== 'string') return undefined;
  let list: 'groupBys' | 'joins' | 'suitabilities';
  let label: string;
  let route: 'sql' | 'analysis';
  switch (nativeAction.type) {
    case ActionTypes.RUN_GROUP_BY:
      list = 'groupBys';
      label = 'Group By';
      route = 'sql';
      break;
    case ActionTypes.RUN_JOIN:
      list = 'joins';
      label = 'Join';
      route = 'sql';
      break;
    case ActionTypes.RUN_SPATIAL_JOIN:
      list = 'joins';
      label = 'Spatial Join';
      route = 'analysis';
      break;
    case ActionTypes.RUN_SUITABILITY:
      list = 'suitabilities';
      label = 'Suitability';
      route = 'analysis';
      break;
    default:
      return undefined;
  }
  const operation = state?.visState?.[list]?.find(candidate => candidate.id === operationId);
  if (
    !operation ||
    ![operation.dataId, operation.leftDataId, operation.rightDataId].some(
      datasetId => typeof datasetId === 'string' && isResidentDataset(datasetId)
    )
  )
    return undefined;
  const guidance =
    nativeAction.type === ActionTypes.RUN_JOIN
      ? 'The bounded resident Join requires a configured simple Join whose two inputs are both resident; use SQL for mixed resident and nonresident inputs.'
      : nativeAction.type === ActionTypes.RUN_SPATIAL_JOIN
        ? 'The bounded resident Spatial Join requires published longitude/latitude points on the left, a published polygon column on the right, the intersects or within predicate, and Count/Sum/Average/Minimum/Maximum aggregations; use SQL otherwise.'
        : nativeAction.type === ActionTypes.RUN_SUITABILITY
          ? 'The bounded resident Suitability scores up to 16 numeric fields of a resident source and is claimed only from the Suitability panel Run action; use SQL to prepare a numeric table otherwise.'
          : route === 'sql'
            ? 'Open Add Data → SQL for supported source queries, then prepare and publish its result. This native operation has no resident executor yet.'
            : 'This native operation needs a resident predicate or weighted-criteria adapter. The Layers sidebar Analysis section lists implemented operations; it does not substitute a different result.';
  return {
    reason: `${label} cannot execute the native retained-row CPU path for a resident source. ${guidance}`,
    route,
    operation: {list, id: operationId}
  };
}

function record(value: unknown): Record<string, unknown> | undefined {
  return isRecord(value) ? value : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export function unwrapForInstance(
  action: UnknownAction,
  instanceId: string
): UnknownAction | undefined {
  let current = action;
  for (let depth = 0; depth < 8; depth++) {
    const meta = record(current['meta']);
    if (meta?.['_forward_'] !== '@redux-forward/FORWARD') return current;
    if (meta['_addr_'] !== `@@KG_${instanceId.toUpperCase()}`) return undefined;
    const payload = record(current['payload']);
    if (!payload || typeof payload['type'] !== 'string') return undefined;
    current = {...payload, type: payload['type']};
  }
  return undefined;
}
