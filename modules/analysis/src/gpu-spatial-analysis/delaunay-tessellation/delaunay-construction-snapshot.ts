// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import {resolveGPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {DelaunayTessellationOperationProps} from './gpu-delaunay-tessellation-operation';

const DELAUNAY_STATUS_FIELDS = [
  'count',
  'requiredCount',
  'overflow',
  'duplicateCount',
  'invalidCount',
  'candidateOverflow',
  'uncertainCount',
  'iterationCount',
  'converged',
  'iterationLimitReached',
  'approximation',
  'complete',
  'unresolvedCount'
] as const;
type DelaunayConstructionStatus = Partial<Record<(typeof DELAUNAY_STATUS_FIELDS)[number], unknown>>;

/** @internal Capture supported diagnostics and reject unwritten destinations before narrowing. */
export function snapshotDelaunayStatus<Status extends DelaunayConstructionStatus>(
  id: string,
  status: Status
): Status {
  const captured = snapshotConstructionFields(status, DELAUNAY_STATUS_FIELDS);
  for (const field of ['complete', 'unresolvedCount'] as const) {
    validateRequirement(
      id,
      captured[field] === undefined,
      `output.status.${field} is unsupported; use output.status.converged and capacity diagnostics`
    );
  }
  return captured;
}

/** @internal Shared CPU/GPU named-operation construction snapshot and status admission. */
export function snapshotDelaunayTessellationOperationProps(
  props: DelaunayTessellationOperationProps
): DelaunayTessellationOperationProps {
  const captured = snapshotConstructionFields(props, [
    'id',
    'spatialContext',
    'positions',
    'capacity',
    'groups',
    'output'
  ]);
  const id = captured.id ?? 'delaunay-tessellation';
  const output = snapshotConstructionFields(captured.output, ['triangles', 'status', 'groups']);
  const status = snapshotDelaunayStatus(id, output.status);
  const outputGroups =
    output.groups &&
    snapshotConstructionFields(output.groups, ['triangleCounts', 'triangleOffsets', 'status']);
  const spatialContext =
    captured.spatialContext && snapshotConstructionSpatialContext(captured.spatialContext);
  resolveGPUSpatialContext(id, spatialContext);
  return Object.freeze({
    ...captured,
    spatialContext,
    groups: captured.groups && snapshotConstructionFields(captured.groups, ['ids', 'count']),
    output: Object.freeze({
      ...output,
      status,
      groups:
        outputGroups &&
        Object.freeze({
          ...outputGroups,
          status: snapshotConstructionFields(outputGroups.status, ['converged'])
        })
    })
  });
}
