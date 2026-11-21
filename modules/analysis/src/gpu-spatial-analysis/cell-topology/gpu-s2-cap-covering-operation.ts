// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateS2CoveringOptions} from './s2-covering-option-validation';
import {snapshotS2CapOperationProps} from './s2-covering-construction-snapshot';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUS2CapCovering, type GPUS2CapCoveringProps} from './gpu-s2-cap-covering';

/** Shared graph-value properties, with explicit cap and output row counts. */
export type S2CapCoveringOperationProps = Omit<GPUS2CapCoveringProps, 'caps' | 'output'> & {
  /** float32x4 unit xyz/radius rows. */
  caps: string;
  capCount: number;
  output: {cells: string; counts: string; status: string};
};

/** Operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUS2CapCoveringOperation extends GPUContributorOperation {
  constructor(props: S2CapCoveringOperationProps) {
    props = snapshotS2CapOperationProps(props);
    validateS2CoveringOptions(props.id ?? 's2-cap-covering', props, [
      'uniform',
      'level-synchronous'
    ]);
    super({
      id: props.id ?? 's2-cap-covering',
      inputs: {caps: props.caps},
      outputs: {
        cells: {
          value: props.output.cells,
          format: 'uint32x2',
          length: props.capCount * props.capacity
        },
        counts: {value: props.output.counts, format: 'uint32x2', length: props.capCount},
        status: {value: props.output.status, format: 'uint32', length: props.capCount}
      },
      createContributor: getView =>
        new GPUS2CapCovering({
          ...props,
          caps: getView<'float32x4'>('caps'),
          output: {
            cells: getView<'uint32x2'>('cells'),
            counts: getView<'uint32x2'>('counts'),
            status: getView<'uint32'>('status')
          }
        })
    });
  }
}
