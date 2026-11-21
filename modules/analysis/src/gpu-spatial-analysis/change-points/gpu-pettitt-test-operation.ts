// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validatePettittShape, validatePettittOperationPorts} from './change-point-shape';
import {snapshotChangePointProps} from './change-point-construction';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import type {GPUSpaceTimeWeightsLayout} from '../spatial-weights/spatial-weights-space-time-stages';
import {GPUPettittTest} from './gpu-pettitt-test';

/** Graph-value form of `GPUPettittTestProps`; every output has `cellCount` rows. */
export type PettittTestOperationProps = {
  id?: string;
  /** Packed float32 cube in `layout`. */
  values: string;
  layout: GPUSpaceTimeWeightsLayout;
  cellCount: number;
  sliceCount: number;
  output: {statistic: string; changeSlice: string; pValues: string};
};

/** Pettitt contributor operation. Compile with `GPU_CONTRIBUTOR_OPERATION_LOWERING`. */
export class GPUPettittTestOperation extends GPUContributorOperation {
  constructor(props: PettittTestOperationProps) {
    props = snapshotChangePointProps(props);
    const id = props.id ?? 'pettitt-test';
    validatePettittShape(id, props);
    validatePettittOperationPorts(id, props);
    const length = props.cellCount;
    super({
      id,
      inputs: {values: props.values},
      outputs: {
        statistic: {value: props.output.statistic, format: 'uint32', length},
        changeSlice: {value: props.output.changeSlice, format: 'uint32', length},
        pValues: {value: props.output.pValues, format: 'float32', length}
      },
      createContributor: getView =>
        new GPUPettittTest({
          ...props,
          id,
          values: getView<'float32'>('values'),
          output: {
            statistic: getView<'uint32'>('statistic'),
            changeSlice: getView<'uint32'>('changeSlice'),
            pValues: getView<'float32'>('pValues')
          }
        })
    });
  }
}
