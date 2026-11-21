// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import type {GPUNestedInterpolationCandidate} from './gpu-nested-interpolation-validation';

/** @internal Capture a nested candidate ladder and port map before validation or lowering. */
export function snapshotNestedValidationProps<
  Props extends {candidates: readonly GPUNestedInterpolationCandidate[]; output?: object}
>(input: Props): Props {
  const props = snapshotConstructionFields(input, [
    'id',
    'mode',
    'rowCount',
    'positions',
    'response',
    'training',
    'testing',
    'innerBlocks',
    'foldCount',
    'innerFoldCount',
    'innerBufferDistance',
    'seed',
    'neighborCount',
    'candidates',
    'output'
  ] as readonly (keyof Props)[]);
  return Object.freeze({
    ...props,
    ...(props.output
      ? {
          output: snapshotConstructionFields(props.output, [
            'selected',
            'scores',
            'predictions',
            'metrics',
            'status'
          ] as never[])
        }
      : {}),
    candidates: Object.freeze(
      props.candidates.map(candidate => {
        const captured = snapshotConstructionFields(candidate, [
          'mode',
          'covariance',
          'kernel',
          'degree',
          'epsilon',
          'smoothing'
        ] as never[]);
        return captured.mode === 'ordinary-kriging'
          ? Object.freeze({
              ...captured,
              covariance: snapshotConstructionFields(captured.covariance, [
                'model',
                'range',
                'sill',
                'nugget',
                'smoothness'
              ])
            })
          : captured;
      })
    )
  });
}
