// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUTrajectoryThinningOutput} from './gpu-trajectory-thinning';
import type {TrajectoryThinningOperationProps} from './gpu-trajectory-thinning-operation';

/** @internal Capture declared destinations and reject retired keys on the original map. */
export function snapshotThinningOutput<
  Output extends GPUTrajectoryThinningOutput | TrajectoryThinningOperationProps['output']
>(id: string, input: Output): Output {
  validateRequirement(id, !('rowIds' in input), 'output.rowIds was removed; use output.sourceRows');
  validateRequirement(
    id,
    !('validity' in input),
    'output.validity was removed; use output.sourceSelectionMask'
  );
  const output = snapshotConstructionFields(input, [
    'sourceRows',
    'sourceSelectionMask',
    'trackOffsets',
    'status',
    'positions',
    'timestamps'
  ]);
  validateRequirement(id, output.sourceRows !== undefined, 'output.sourceRows is required');
  const {status} = output;
  return {
    ...output,
    status: snapshotConstructionFields(status ?? ({} as typeof status), [
      'count',
      'overflow',
      'requiredCount',
      'candidateOverflow'
    ])
  };
}
