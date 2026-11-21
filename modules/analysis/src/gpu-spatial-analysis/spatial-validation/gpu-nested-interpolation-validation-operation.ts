// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {snapshotNestedValidationProps} from './nested-validation-construction';
import {
  GPUNestedInterpolationValidation,
  type GPUNestedInterpolationValidationProps
} from './gpu-nested-interpolation-validation';

/** Graph-value names replace views; rowCount makes the inferred response length explicit. */
export type NestedInterpolationValidationOperationProps = Pick<
  GPUNestedInterpolationValidationProps,
  | 'id'
  | 'mode'
  | 'foldCount'
  | 'innerFoldCount'
  | 'innerBufferDistance'
  | 'seed'
  | 'neighborCount'
  | 'candidates'
> & {
  rowCount: number;
  positions: string;
  response: string;
  training: string;
  testing: string;
  innerBlocks: string;
  output: {
    selected: string;
    scores: string;
    predictions: string;
    metrics: string;
    status: string;
  };
};

/** Vector operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUNestedInterpolationValidationOperation extends GPUContributorOperation {
  constructor(props: NestedInterpolationValidationOperationProps) {
    props = snapshotNestedValidationProps(props);
    const {rowCount, foldCount, output} = props;
    super({
      id: props.id ?? 'nested-interpolation-validation',
      inputs: {
        positions: props.positions,
        response: props.response,
        training: props.training,
        testing: props.testing,
        innerBlocks: props.innerBlocks
      },
      outputs: {
        selected: {value: output.selected, format: 'uint32', length: foldCount},
        scores: {
          value: output.scores,
          format: 'float32',
          length: foldCount * props.candidates.length * 3
        },
        predictions: {value: output.predictions, format: 'float32', length: rowCount * foldCount},
        metrics: {value: output.metrics, format: 'float32', length: foldCount * 6},
        status: {value: output.status, format: 'uint32', length: foldCount}
      },
      createContributor: getView =>
        new GPUNestedInterpolationValidation({
          ...props,
          positions: getView<'float32x2'>('positions'),
          response: getView<'float32'>('response'),
          training: getView<'uint32'>('training'),
          testing: getView<'uint32'>('testing'),
          innerBlocks: getView<'uint32'>('innerBlocks'),
          output: {
            selected: getView<'uint32'>('selected'),
            scores: getView<'float32'>('scores'),
            predictions: getView<'float32'>('predictions'),
            metrics: getView<'float32'>('metrics'),
            status: getView<'uint32'>('status')
          }
        })
    });
  }
}
