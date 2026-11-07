// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ActionTypes} from '@kepler.gl/actions';
import type {UnknownAction} from 'redux';
import {
  getResidentGroupByStatus,
  RESIDENT_GROUP_BY_STATUS,
  unwrapForInstance,
  type ResidentGroupByStatus
} from './resident-action-guards';
import type {ResidentSuitabilityOperation} from './resident-suitability-plan';

export const RESIDENT_SUITABILITY_STATUS = 'LUMA_SUITABILITY_STATUS';

export type ResidentSuitabilityRequest = ResidentSuitabilityOperation;
type SuitabilityOperationRecord = {
  id: string;
  dataId?: string | null;
  weights?: Readonly<Record<string, number>>;
  weightStandardization?: string;
  dataStandardization?: string;
  outputFieldName?: string;
  columns?: readonly string[];
  missingValues?: string;
  categoryScores?: Readonly<Record<string, Readonly<Record<string, number>>>>;
  resultId?: string;
  resultLabel?: string;
};

/** Suitability status reuses the Group By status shape under its own action type and list. */
export function getResidentSuitabilityStatus(
  action: UnknownAction
): ResidentGroupByStatus | undefined {
  if (action.type !== RESIDENT_SUITABILITY_STATUS) return undefined;
  return getResidentGroupByStatus({...action, type: RESIDENT_GROUP_BY_STATUS});
}

/**
 * RUN_SUITABILITY carries an operation ID. Only resident sources are claimed; the editable native
 * configuration is snapshotted before work so later edits cancel or invalidate the request.
 * Unsupported shapes are rejected by the plan with guidance, never run on retained rows.
 */
export function getResidentSuitabilityRequest(
  state: {visState?: {suitabilities?: readonly SuitabilityOperationRecord[]}} | undefined,
  action: UnknownAction,
  isResidentDataset: (datasetId: string) => boolean,
  instanceId = 'compute'
): ResidentSuitabilityRequest | undefined {
  const nativeAction = unwrapForInstance(action, instanceId);
  if (nativeAction?.type !== ActionTypes.RUN_SUITABILITY) return undefined;
  const operation = state?.visState?.suitabilities?.find(
    candidate => candidate.id === nativeAction['id']
  );
  if (!operation?.dataId || !isResidentDataset(operation.dataId)) return undefined;
  return {
    id: operation.id,
    dataId: operation.dataId,
    weights: {...operation.weights},
    // Native falls back to normalize/range only for an absent value; any other string leaves
    // weights or data unstandardized, which is the raw policy.
    weightStandardization:
      !operation.weightStandardization || operation.weightStandardization === 'normalize'
        ? 'normalize'
        : 'raw',
    dataStandardization:
      !operation.dataStandardization || operation.dataStandardization === 'range'
        ? 'range'
        : operation.dataStandardization === 'zScore'
          ? 'zScore'
          : 'raw',
    outputFieldName: operation.outputFieldName ?? '',
    ...(operation.columns ? {columns: [...operation.columns]} : {}),
    ...(operation.missingValues === 'exclude' || operation.missingValues === 'renormalize'
      ? {missingValues: operation.missingValues}
      : {}),
    ...(operation.categoryScores && Object.keys(operation.categoryScores).length
      ? {categoryScores: operation.categoryScores}
      : {}),
    resultId: operation.resultId ?? '',
    resultLabel: operation.resultLabel ?? ''
  };
}

export function getResidentSuitabilityRequestKey(request: ResidentSuitabilityRequest): string {
  return JSON.stringify([
    request.id,
    request.dataId,
    Object.entries(request.weights),
    request.weightStandardization,
    request.dataStandardization,
    request.outputFieldName,
    request.columns ?? null,
    request.missingValues ?? null,
    request.categoryScores ? JSON.stringify(request.categoryScores) : null,
    request.resultId,
    request.resultLabel
  ]);
}
