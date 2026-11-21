// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPUColumnQuantiles} from './gpu-column-quantiles';

/** Graph-value props shared by the GPU and CPU quantile operations. */
export type ColumnQuantilesOperationProps = {
  id?: string;
  values: string;
  /** Explicit length of values (and mask and outputMask when present). */
  rowCount: number;
  mask?: string;
  /** Packed with getColumnQuantilesParameterValues (either backend) or its GPU-limited spelling. */
  parameters: string;
  rationalProbabilities?: string;
  quantileCount: number;
  output: {
    quantiles: string;
    validCount: string;
    outputMask?: string;
    filterBounds?: string;
  };
};

/** @internal Capture the declared shape and port names for both operation forms. */
export function snapshotColumnQuantilesOperationProps(
  inputProps: ColumnQuantilesOperationProps
): ColumnQuantilesOperationProps {
  const props = snapshotConstructionFields(inputProps, [
    'id',
    'values',
    'rowCount',
    'mask',
    'parameters',
    'rationalProbabilities',
    'quantileCount',
    'output'
  ]);
  return Object.freeze({
    ...props,
    output: snapshotConstructionFields(props.output, [
      'quantiles',
      'validCount',
      'outputMask',
      'filterBounds'
    ])
  });
}

/** Vector operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUColumnQuantilesOperation extends GPUContributorOperation {
  constructor(inputProps: ColumnQuantilesOperationProps) {
    const props = snapshotColumnQuantilesOperationProps(inputProps);
    const id = props.id ?? 'column-quantiles';
    validateCapacity(id, 'rowCount', props.rowCount, 1);
    super({
      id: props.id ?? 'column-quantiles',
      inputs: {
        values: props.values,
        parameters: props.parameters,
        ...(props.mask !== undefined ? {mask: props.mask} : {}),
        ...(props.rationalProbabilities !== undefined
          ? {rationalProbabilities: props.rationalProbabilities}
          : {})
      },
      outputs: {
        quantiles: {value: props.output.quantiles, format: 'float32', length: props.quantileCount},
        validCount: {value: props.output.validCount, format: 'uint32', length: 1},
        ...(props.output.outputMask !== undefined
          ? {
              outputMask: {
                value: props.output.outputMask,
                format: 'uint32' as const,
                length: props.rowCount
              }
            }
          : {}),
        ...(props.output.filterBounds !== undefined
          ? {
              filterBounds: {
                value: props.output.filterBounds,
                format: 'float32' as const,
                length: 2
              }
            }
          : {})
      },
      createContributor: getView => {
        const values = getView<'float32'>('values');
        // The CPU form rejects the same declaration when it runs; without an outputMask nothing
        // else on this form reads rowCount.
        validateRequirement(
          id,
          values.length === props.rowCount,
          `values length must equal rowCount; values length ${values.length} must equal rowCount ${props.rowCount}`
        );
        return new GPUColumnQuantiles({
          id: props.id,
          values,
          parameters: getView<'float32'>('parameters'),
          mask: props.mask !== undefined ? getView<'uint32'>('mask') : undefined,
          rationalProbabilities:
            props.rationalProbabilities !== undefined
              ? getView<'uint32'>('rationalProbabilities')
              : undefined,
          quantileCount: props.quantileCount,
          output: {
            quantiles: getView<'float32'>('quantiles'),
            validCount: getView<'uint32'>('validCount'),
            outputMask:
              props.output.outputMask !== undefined ? getView<'uint32'>('outputMask') : undefined,
            filterBounds:
              props.output.filterBounds !== undefined
                ? getView<'float32'>('filterBounds')
                : undefined
          }
        });
      }
    });
  }
}
