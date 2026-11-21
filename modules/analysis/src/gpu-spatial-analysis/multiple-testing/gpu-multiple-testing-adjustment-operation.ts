// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUMultipleTestingMethod} from '../../analysis-core/contracts/index';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUMultipleTestingAdjustment,
  GPU_MULTIPLE_TESTING_MAXIMUM_ROWS,
  resolveGPUMultipleTestingMethod
} from './gpu-multiple-testing-adjustment';
import type {GPUMultipleTestingAdjustmentOutput} from './gpu-multiple-testing-adjustment';
import {
  MULTIPLE_TESTING_COUNT_OUTPUT_NAMES,
  validateMultipleTestingCountPorts
} from './multiple-testing-counts';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Graph value names shared by the GPU adjustment and its CPU twin. */
export type MultipleTestingAdjustmentOperationProps = {
  id?: string;
  pValues: string;
  /** Explicit length of pValues and adjustedPValues. */
  rowCount: number;
  method: GPUMultipleTestingMethod;
  minimumFamilySize?: number;
  /** Use distinct destination names for `finiteTestCount` and `effectiveFamilySize`. */
  output: {
    adjustedPValues: string;
    /** Optional one-word uint32 count of finite p-values in [0, 1], before padding. */
    finiteTestCount?: string;
    /** Optional one-word uint32 denominator: max(finiteTestCount, minimumFamilySize). */
    effectiveFamilySize?: string;
  };
};

/** Operation form; all props are compile-time, input contents are per-frame. */
export class GPUMultipleTestingAdjustmentOperation extends GPUContributorOperation {
  constructor(props: MultipleTestingAdjustmentOperationProps) {
    props = snapshotConstructionFields(props, [
      'id',
      'pValues',
      'rowCount',
      'method',
      'minimumFamilySize',
      'output'
    ]);
    const id = props.id ?? 'multiple-testing-adjustment';
    validateMultipleTestingCountPorts(id, props.output);
    props = Object.freeze({
      ...props,
      output: snapshotConstructionFields(props.output, [
        'adjustedPValues',
        ...MULTIPLE_TESTING_COUNT_OUTPUT_NAMES
      ])
    });
    const method = resolveGPUMultipleTestingMethod(id, props.method, true);
    validateCapacity(id, 'rowCount', props.rowCount, 1, GPU_MULTIPLE_TESTING_MAXIMUM_ROWS - 1);
    validateCapacity(
      id,
      'minimumFamilySize',
      props.minimumFamilySize ?? 0,
      0,
      GPU_MULTIPLE_TESTING_MAXIMUM_ROWS - 1
    );
    const countOutputs = MULTIPLE_TESTING_COUNT_OUTPUT_NAMES.filter(
      name => props.output[name] !== undefined
    );
    super({
      id,
      inputs: {pValues: props.pValues},
      outputs: {
        adjustedPValues: {
          value: props.output.adjustedPValues,
          format: 'float32',
          length: props.rowCount
        },
        ...Object.fromEntries(
          countOutputs.map(name => [
            name,
            {value: props.output[name]!, format: 'uint32' as const, length: 1}
          ])
        )
      },
      createContributor: getView =>
        new GPUMultipleTestingAdjustment({
          id,
          pValues: getView<'float32'>('pValues'),
          method,
          minimumFamilySize: props.minimumFamilySize,
          output: {
            adjustedPValues: getView<'float32'>('adjustedPValues'),
            ...Object.fromEntries(countOutputs.map(name => [name, getView<'uint32'>(name)]))
          } satisfies GPUMultipleTestingAdjustmentOutput
        })
    });
  }
}
