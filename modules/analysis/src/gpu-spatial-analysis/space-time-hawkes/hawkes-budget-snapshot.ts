// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {snapshotSpaceTimeHawkesProps} from './space-time-hawkes-construction';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import type {GPUSpaceTimeHawkesProps} from './gpu-space-time-hawkes';
import type {SpaceTimeHawkesOperationProps} from './gpu-space-time-hawkes-operation';

/** Capture the additive budget diagnostic along with the established declared-field snapshot. @internal */
export function snapshotHawkesBudgetProps<
  Props extends GPUSpaceTimeHawkesProps | SpaceTimeHawkesOperationProps
>(props: Props): Props {
  // Capture getters on their original receiver once, including inherited/non-enumerable fields.
  const fields = snapshotConstructionFields(props, [
    'id',
    'positions',
    'times',
    'parameters',
    'spatialContext',
    'temporalContext',
    'maximumIterations',
    'maximumParents',
    'output'
  ]);
  const output = snapshotConstructionFields(fields.output, [
    'estimates',
    'logLikelihood',
    'fittedLogLikelihood',
    'backgroundProbability',
    'status'
  ]);
  const status =
    output.status &&
    snapshotConstructionFields(output.status, [
      'converged',
      'iterationLimitReached',
      'iterationCount',
      'candidateOverflow',
      'invalidCount',
      'approximation',
      'maximumIterations'
    ]);
  const captured = snapshotSpaceTimeHawkesProps({
    ...fields,
    ...('rowCount' in props ? {rowCount: props.rowCount} : {}),
    output: {...output, status}
  } as Props);
  return Object.freeze({
    ...captured,
    output: Object.freeze({
      ...captured.output,
      status:
        status &&
        Object.freeze({...captured.output.status, maximumIterations: status.maximumIterations})
    })
  }) as Props;
}
