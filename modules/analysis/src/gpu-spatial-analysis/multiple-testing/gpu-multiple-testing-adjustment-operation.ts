// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUMultipleTestingMethod} from '../../analysis-core/contracts/index';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUMultipleTestingAdjustment} from './gpu-multiple-testing-adjustment';

/** Graph value names shared by the GPU adjustment and its CPU twin. */
export type MultipleTestingAdjustmentOperationProps = {
  id?: string;
  pValues: string;
  /** Explicit length of pValues and adjustedPValues. */
  rowCount: number;
  method: GPUMultipleTestingMethod;
  minimumFamilySize?: number;
  output: {adjustedPValues: string; testedCount?: string};
};

/** Operation form; all props are compile-time, input contents are per-frame. */
export class GPUMultipleTestingAdjustmentOperation extends GPUContributorOperation {
  constructor(props: MultipleTestingAdjustmentOperationProps) {
    const id = props.id ?? 'multiple-testing-adjustment';
    super({
      id,
      inputs: {pValues: props.pValues},
      outputs: {
        adjustedPValues: {
          value: props.output.adjustedPValues,
          format: 'float32',
          length: props.rowCount
        },
        ...(props.output.testedCount
          ? {testedCount: {value: props.output.testedCount, format: 'uint32' as const, length: 1}}
          : {})
      },
      createContributor: getView =>
        new GPUMultipleTestingAdjustment({
          id,
          pValues: getView<'float32'>('pValues'),
          method: props.method,
          minimumFamilySize: props.minimumFamilySize,
          output: {
            adjustedPValues: getView<'float32'>('adjustedPValues'),
            testedCount: props.output.testedCount ? getView<'uint32'>('testedCount') : undefined
          }
        })
    });
  }
}
