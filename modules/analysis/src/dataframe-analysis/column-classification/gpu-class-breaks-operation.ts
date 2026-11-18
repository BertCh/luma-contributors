// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUClassBreaks} from './gpu-class-breaks';
import type {GPUClassBreaksMethod} from './class-breaks-parameters';
import {
  validateClassBreaksInput,
  validateClassBreaksInputAssertions,
  validateClassBreaksNaturalOptions
} from './class-breaks-input';
import {snapshotClassBreaksOperationProps} from './class-breaks-operation-snapshot';

/** Shared graph-value props. The packed parameters select a method per frame. */
export type ClassBreaksOperationProps = {
  id?: string;
  values: string;
  /** Optional assertion on the actual values row count; inferred when omitted. */
  rowCount?: number;
  /** Optional assertion on the actual values format; inferred when omitted. */
  valuesFormat?: 'float32' | 'uint32' | 'sint32';
  mask?: string;
  parameters: string;
  maximumClassCount: number;
  methods?: readonly GPUClassBreaksMethod[];
  naturalBreaksBinCount?: number;
  naturalBreaksMode?: 'approximate' | 'exact' | 'monotone';
  naturalBreaksExactStorage?: 'auto' | 'materialized' | 'recompute';
  output: {breaks: string; classCount: string; classCounts?: string};
};

/** Vector operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUClassBreaksOperation extends GPUContributorOperation {
  constructor(inputProps: ClassBreaksOperationProps) {
    const props = snapshotClassBreaksOperationProps(inputProps);
    validateClassBreaksInputAssertions(props.id ?? 'class-breaks', props);
    validateClassBreaksNaturalOptions(props.id ?? 'class-breaks', props);
    super({
      id: props.id ?? 'class-breaks',
      inputs: {
        values: props.values,
        parameters: props.parameters,
        ...(props.mask !== undefined ? {mask: props.mask} : {})
      },
      outputs: {
        breaks: {
          value: props.output.breaks,
          format: 'float32',
          length: props.maximumClassCount + 1
        },
        classCount: {value: props.output.classCount, format: 'uint32', length: 1},
        ...(props.output.classCounts !== undefined
          ? {
              classCounts: {
                value: props.output.classCounts,
                format: 'uint32' as const,
                length: props.maximumClassCount
              }
            }
          : {})
      },
      createContributor: getView => {
        const values = getView<'float32' | 'uint32' | 'sint32'>('values');
        validateClassBreaksInput(props.id ?? 'class-breaks', props, values);
        return new GPUClassBreaks({
          ...props,
          values,
          parameters: getView<'float32'>('parameters'),
          mask: props.mask !== undefined ? getView<'uint32'>('mask') : undefined,
          output: {
            breaks: getView<'float32'>('breaks'),
            classCount: getView<'uint32'>('classCount'),
            classCounts:
              props.output.classCounts !== undefined ? getView<'uint32'>('classCounts') : undefined
          }
        });
      }
    });
  }
}
