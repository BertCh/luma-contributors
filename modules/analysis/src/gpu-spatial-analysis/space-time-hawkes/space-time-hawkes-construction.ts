// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import {snapshotSpaceTimeTemporalContext} from '../space-time-tests/space-time-construction';
import type {GPUSpaceTimeHawkesProps} from './gpu-space-time-hawkes';
import type {SpaceTimeHawkesOperationProps} from './gpu-space-time-hawkes-operation';

/** Capture only declared construction options; parameter/data buffers remain live. @internal */
export function snapshotSpaceTimeHawkesProps<
  Props extends GPUSpaceTimeHawkesProps | SpaceTimeHawkesOperationProps
>(props: Props): Props {
  const captured = snapshotConstructionFields(props, [
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
  const output = snapshotConstructionFields(captured.output, [
    'estimates',
    'logLikelihood',
    'fittedLogLikelihood',
    'backgroundProbability',
    'status'
  ]);
  const status = output.status;
  return Object.freeze({
    ...captured,
    ...('rowCount' in props ? {rowCount: props.rowCount} : {}),
    spatialContext:
      captured.spatialContext && snapshotConstructionSpatialContext(captured.spatialContext),
    temporalContext:
      captured.temporalContext && snapshotSpaceTimeTemporalContext(captured.temporalContext),
    output: Object.freeze({
      ...output,
      status:
        status &&
        snapshotConstructionFields(status, [
          'converged',
          'iterationLimitReached',
          'iterationCount',
          'candidateOverflow',
          'invalidCount',
          'approximation'
        ])
    })
  }) as Props;
}
