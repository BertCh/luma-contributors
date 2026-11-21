// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GraphDataView} from '@luma.gl/graph/gpu-core';

/** Capture every declared destination, including inactive ports, before any side effects. @internal */
export function captureUrbanLandscapeDestinations<Outputs extends object>(
  id: string,
  props: {output?: Outputs},
  fields: readonly (keyof Outputs)[]
): Outputs | undefined {
  validateRequirement(id, !('outputs' in props), 'outputs was removed; use output');
  const output = props.output;
  if (!output) return undefined;
  const statusFields = [
    'count',
    'requiredCount',
    'overflow',
    'candidateOverflow',
    'invalidCount',
    'invalidCellCount',
    'converged',
    'iterationCount',
    'iterationLimitReached',
    'invalidEdgeCount'
  ];
  const captured = {...snapshotConstructionFields(output, fields)};
  const records = captured as Record<string, unknown>;
  const status = records['status'];
  if (status && typeof status === 'object' && !(status instanceof GraphDataView)) {
    records['status'] = snapshotConstructionFields(status as Record<string, unknown>, statusFields);
  }
  // Keep caller-owned overlay records while capturing the array of destinations.
  if (Array.isArray(records['overlays'])) {
    records['overlays'] = Array.from(records['overlays']);
  }
  return captured;
}
