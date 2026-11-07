// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ActionTypes} from '@kepler.gl/actions';
import type {UnknownAction} from 'redux';
import type {ResidentSpatialJoinGeo} from './resident-spatial-join-plan';
import {
  getResidentGroupByStatus,
  RESIDENT_GROUP_BY_STATUS,
  unwrapForInstance,
  type ResidentGroupByStatus
} from './resident-action-guards';

export const RESIDENT_JOIN_STATUS = 'LUMA_JOIN_STATUS';

export type ResidentJoinRequest = {
  id: string;
  leftDataId: string;
  rightDataId: string;
  leftField: string | null;
  rightField: string | null;
  type: string;
  leftColumns?: readonly string[];
  rightColumns?: readonly string[];
  /** Host policy extensions; the native operation has no such settings, so they stay optional. */
  duplicateRightMatches?: 'all' | 'first' | 'last';
  rightColumnSuffix?: string;
  dropRightKey?: boolean;
  resultId: string;
  resultLabel: string;
};
export type ResidentSpatialJoinRequest = {
  kind: 'spatial-join';
  id: string;
  leftDataId: string;
  rightDataId: string;
  leftGeo: ResidentSpatialJoinGeo;
  rightGeo: ResidentSpatialJoinGeo;
  predicate: string;
  /** Host policy extension: declared coordinate systems, which must agree. */
  leftCrs?: string;
  rightCrs?: string;
  aggregations: Readonly<Record<string, string>>;
  leftColumns?: readonly string[];
  resultId: string;
  resultLabel: string;
};
type JoinOperationRecord = {
  id: string;
  implementation?: string;
  leftGeo?: ResidentSpatialJoinGeo;
  rightGeo?: ResidentSpatialJoinGeo;
  predicate?: string;
  leftCrs?: string;
  rightCrs?: string;
  aggregations?: Readonly<Record<string, string>>;
  leftDataId?: string | null;
  rightDataId?: string | null;
  leftField?: string | null;
  rightField?: string | null;
  type?: string;
  leftColumns?: readonly string[];
  rightColumns?: readonly string[];
  duplicateRightMatches?: 'all' | 'first' | 'last';
  rightColumnSuffix?: string;
  dropRightKey?: boolean;
  resultId?: string;
  resultLabel?: string;
};

/** Join status reuses the Group By status shape under its own action type and operation list. */
export function getResidentJoinStatus(action: UnknownAction): ResidentGroupByStatus | undefined {
  if (action.type !== RESIDENT_JOIN_STATUS) return undefined;
  return getResidentGroupByStatus({...action, type: RESIDENT_GROUP_BY_STATUS});
}

/**
 * RUN_JOIN carries an operation ID. Only simple joins whose two inputs are both resident are
 * claimed; anything else keeps the explicit rejection path and never falls back to CPU rows.
 */
export function getResidentJoinRequest(
  state: {visState?: {joins?: readonly JoinOperationRecord[]}} | undefined,
  action: UnknownAction,
  isResidentDataset: (datasetId: string) => boolean,
  instanceId = 'compute'
): ResidentJoinRequest | undefined {
  const nativeAction = unwrapForInstance(action, instanceId);
  if (nativeAction?.type !== ActionTypes.RUN_JOIN) return undefined;
  const operation = state?.visState?.joins?.find(candidate => candidate.id === nativeAction['id']);
  if (
    !operation ||
    operation.implementation === 'spatial' ||
    !operation.leftDataId ||
    !operation.rightDataId ||
    !isResidentDataset(operation.leftDataId) ||
    !isResidentDataset(operation.rightDataId) ||
    !operation.leftField ||
    !operation.rightField
  )
    return undefined;
  return {
    id: operation.id,
    leftDataId: operation.leftDataId,
    rightDataId: operation.rightDataId,
    leftField: operation.leftField,
    rightField: operation.rightField,
    type: operation.type ?? 'LEFT',
    ...(operation.leftColumns ? {leftColumns: [...operation.leftColumns]} : {}),
    ...(operation.rightColumns ? {rightColumns: [...operation.rightColumns]} : {}),
    ...(operation.duplicateRightMatches
      ? {duplicateRightMatches: operation.duplicateRightMatches}
      : {}),
    ...(operation.rightColumnSuffix !== undefined
      ? {rightColumnSuffix: operation.rightColumnSuffix}
      : {}),
    ...(operation.dropRightKey ? {dropRightKey: true} : {}),
    resultId: operation.resultId ?? '',
    resultLabel: operation.resultLabel ?? ''
  };
}

export function getResidentJoinRequestKey(request: ResidentJoinRequest): string {
  return JSON.stringify([
    request.id,
    request.leftDataId,
    request.rightDataId,
    request.leftField,
    request.rightField,
    request.type,
    request.leftColumns ?? null,
    request.rightColumns ?? null,
    request.duplicateRightMatches ?? null,
    request.rightColumnSuffix ?? null,
    request.dropRightKey ?? null,
    request.resultId,
    request.resultLabel
  ]);
}

/** RUN_SPATIAL_JOIN snapshot, claimed only when both inputs are resident. */
export function getResidentSpatialJoinRequest(
  state: {visState?: {joins?: readonly JoinOperationRecord[]}} | undefined,
  action: UnknownAction,
  isResidentDataset: (datasetId: string) => boolean,
  instanceId = 'compute'
): ResidentSpatialJoinRequest | undefined {
  const nativeAction = unwrapForInstance(action, instanceId);
  if (nativeAction?.type !== ActionTypes.RUN_SPATIAL_JOIN) return undefined;
  const operation = state?.visState?.joins?.find(candidate => candidate.id === nativeAction['id']);
  if (
    !operation ||
    operation.implementation !== 'spatial' ||
    !operation.leftDataId ||
    !operation.rightDataId ||
    !isResidentDataset(operation.leftDataId) ||
    !isResidentDataset(operation.rightDataId)
  )
    return undefined;
  return {
    kind: 'spatial-join',
    id: operation.id,
    leftDataId: operation.leftDataId,
    rightDataId: operation.rightDataId,
    leftGeo: operation.leftGeo ? {...operation.leftGeo} : null,
    rightGeo: operation.rightGeo ? {...operation.rightGeo} : null,
    predicate: operation.predicate ?? 'intersects',
    ...(operation.leftCrs ? {leftCrs: operation.leftCrs} : {}),
    ...(operation.rightCrs ? {rightCrs: operation.rightCrs} : {}),
    aggregations: {...operation.aggregations},
    ...(operation.leftColumns ? {leftColumns: [...operation.leftColumns]} : {}),
    resultId: operation.resultId ?? '',
    resultLabel: operation.resultLabel ?? ''
  };
}

export function getResidentSpatialJoinRequestKey(request: ResidentSpatialJoinRequest): string {
  return JSON.stringify([
    request.id,
    request.leftDataId,
    request.rightDataId,
    request.leftGeo,
    request.rightGeo,
    request.predicate,
    request.leftCrs ?? null,
    request.rightCrs ?? null,
    Object.entries(request.aggregations),
    request.leftColumns ?? null,
    request.resultId,
    request.resultLabel
  ]);
}
