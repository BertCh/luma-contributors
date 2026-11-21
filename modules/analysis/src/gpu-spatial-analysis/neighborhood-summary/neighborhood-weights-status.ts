// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUBoundedResultStatusPort} from '../../analysis-core/contracts/index';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

const UNSUPPORTED_LOSS_FIELDS = ['unresolvedCount', 'invalidCount', 'approximation'] as const;
export const NEIGHBORHOOD_STOPPING_FIELDS = ['iterationLimitReached', 'converged'] as const;
export const NEIGHBORHOOD_CAPACITY_LOSS_FIELDS = ['overflow', 'candidateOverflow'] as const;

type NeighborhoodWeightsStatus<Value = unknown> = Partial<
  Record<keyof GPUBoundedResultStatusPort, Value>
>;

export type NeighborhoodWeightsChunkMetadata<Value> = {
  chunks?: readonly {weights: {status?: NeighborhoodWeightsStatus<Value>}}[];
};

/** Capture known status ports without dropping inherited getters or cloning their bindings. */
export function snapshotNeighborhoodWeightsStatus<Value>(
  status: NeighborhoodWeightsStatus<Value> | undefined
): NeighborhoodWeightsStatus<Value> | undefined {
  if (!status) return undefined;
  return snapshotConstructionFields(status, [
    ...NEIGHBORHOOD_CAPACITY_LOSS_FIELDS,
    ...NEIGHBORHOOD_STOPPING_FIELDS,
    'complete',
    'requiredCount',
    ...UNSUPPORTED_LOSS_FIELDS,
    'count',
    'iterationCount',
    'uncertainCount'
  ]);
}

export function validateNeighborhoodWeightsLossEvidence(
  id: string,
  path: string,
  status: NeighborhoodWeightsStatus | undefined,
  stoppingEvidence = ''
): void {
  for (const field of UNSUPPORTED_LOSS_FIELDS) {
    validateRequirement(
      id,
      status?.[field] === undefined,
      `${path}.${field} must be omitted; received a status view. Pass the producer's overflow and candidateOverflow for capacity loss${stoppingEvidence}. Verify the CSR is complete before omitting this view`
    );
  }
}

export function validateNeighborhoodChunkLossBindings(
  id: string,
  path: string,
  status: NeighborhoodWeightsStatus | undefined,
  fields: readonly (
    | (typeof NEIGHBORHOOD_STOPPING_FIELDS)[number]
    | (typeof NEIGHBORHOOD_CAPACITY_LOSS_FIELDS)[number]
  )[] = NEIGHBORHOOD_STOPPING_FIELDS
): void {
  for (const field of fields) {
    validateRequirement(
      id,
      status?.[field] === undefined,
      `${path}.${field} must be omitted; received a child status binding. Supply an aggregate weights.status.${field} covering the entire matrix instead`
    );
  }
}

/** Graph forms read loss flags but cannot compare required counts or qualify positive certificates. */
export function validateNeighborhoodWeightsCertificates(
  id: string,
  path: string,
  status: NeighborhoodWeightsStatus | undefined
): void {
  validateNeighborhoodWeightsLossEvidence(
    id,
    path,
    status,
    ', and outer iterationLimitReached and converged for stopping'
  );
  validateRequirement(
    id,
    status?.requiredCount === undefined,
    `${path}.requiredCount must be omitted; received a status view. Supply overflow and candidateOverflow for capacity loss and retain requiredCount for capacity planning`
  );
  validateRequirement(
    id,
    status?.complete === undefined,
    `${path}.complete is unsupported; use computeNeighborhoodSummaryOnCPU with the intact CPU producer result, or verify the CSR is complete before omitting this view and retain any overflow and candidateOverflow views`
  );
}
