// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import type {GPUSpatialWeights} from '../spatial-weights/spatial-weights';

/** @internal Capture declared status properties, including class/proxy getters, exactly once. */
export function snapshotRestrictedRegimeKernelStatus<Status extends object>(
  status: Status
): Status {
  const fields = new Set([
    ...Object.keys(status),
    'count',
    'requiredCount',
    'overflow',
    'candidateOverflow',
    'complete',
    'converged',
    'iterationCount',
    'iterationLimitReached',
    'iterationLimit',
    'invalidInput',
    'invalidCount',
    'uncertainCount',
    'unresolvedCount',
    'approximation'
  ]);
  const captured = snapshotConstructionFields(status, [...fields] as (keyof Status)[]);
  return Object.fromEntries(
    Object.entries(captured).filter(([, value]) => value !== undefined)
  ) as Status;
}

/** @internal Snapshot metadata without spreading a caller's prototype getters away. */
export function snapshotRestrictedRegimeKernelInformation(
  information: GPUSpatialWeights['information']
) {
  if (!information) return undefined;
  const captured = snapshotConstructionFields(information, [
    'rowCount',
    'transform',
    'symmetric',
    'hasSelfWeights',
    'provenance'
  ]);
  return {
    ...captured,
    provenance:
      captured.provenance &&
      snapshotConstructionFields(captured.provenance, [
        'source',
        'criterion',
        'k',
        'radius',
        'order'
      ])
  };
}
