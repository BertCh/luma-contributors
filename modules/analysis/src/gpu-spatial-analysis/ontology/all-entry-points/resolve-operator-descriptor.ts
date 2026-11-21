// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUAnalysisOperator, GPUSpatialAnalysisOperator} from '../spatial-analysis-ontology';
import {withCatalogStability, type CatalogRecordDefinition} from '../freeze-catalog';
import {OPERATOR_ANNOTATIONS} from './operator-annotations';
import {AUDITED_EMPTY_OPERATOR_STATUSES} from './operator-status-audits';

/** An omitted status has not been audited; an empty list also needs affirmative audit evidence. */
export type OperatorDescriptor = CatalogRecordDefinition<Omit<GPUAnalysisOperator, 'status'>> & {
  status?: GPUAnalysisOperator['status'];
};

/** Historical family evidence must survive resolution without becoming optional. */
export type SpatialOperatorDescriptor = OperatorDescriptor & {
  evidence: GPUSpatialAnalysisOperator['evidence'];
};
export type ResolvedSpatialOperator = GPUAnalysisOperator & {
  evidence: GPUSpatialAnalysisOperator['evidence'];
};

const RESOLVED_DESCRIPTORS = new WeakMap<OperatorDescriptor, GPUAnalysisOperator>();

/** Apply qualified audit evidence and annotation precedence once, before either public projection. */
export function resolveOperatorDescriptor(
  descriptor: SpatialOperatorDescriptor
): ResolvedSpatialOperator;
export function resolveOperatorDescriptor(descriptor: OperatorDescriptor): GPUAnalysisOperator;
export function resolveOperatorDescriptor(descriptor: OperatorDescriptor): GPUAnalysisOperator {
  const resolved = RESOLVED_DESCRIPTORS.get(descriptor);
  if (resolved) return resolved;
  const identity = `${descriptor.package}:${descriptor.exportName}`;
  const annotation = OPERATOR_ANNOTATIONS[identity] ?? OPERATOR_ANNOTATIONS[descriptor.exportName];
  const auditReason = AUDITED_EMPTY_OPERATOR_STATUSES[identity];
  const declaredStatus = descriptor.status;
  const status =
    declaredStatus && (declaredStatus === 'unknown' || declaredStatus.length > 0)
      ? declaredStatus
      : auditReason?.trim()
        ? (descriptor.status ?? [])
        : 'unknown';
  const operator: GPUAnalysisOperator =
    descriptor.stability !== undefined && descriptor.status === status && !annotation
      ? (descriptor as GPUAnalysisOperator)
      : withCatalogStability({...descriptor, status, ...annotation});
  // The final catalog may encounter an already resolved historical descriptor. Keep its identity
  // and avoid applying an annotation again after projection or assembly.
  RESOLVED_DESCRIPTORS.set(descriptor, operator);
  RESOLVED_DESCRIPTORS.set(operator, operator);
  return operator;
}

/** The historical API remains array-valued, including for unaudited descriptors. */
export function projectSpatialOperator(
  operator: ResolvedSpatialOperator
): GPUSpatialAnalysisOperator {
  const evidence = operator.evidence;
  if (
    evidence.representativeSourceRows === undefined ||
    evidence.transientMemory === undefined ||
    evidence.dispatch === undefined ||
    evidence.correctnessOracle === undefined
  ) {
    throw new Error('historical projection requires complete family evidence');
  }
  return {...operator, status: operator.status === 'unknown' ? [] : operator.status};
}
