// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPULocalFrameProjection,
  type GPULocalFrameProjectionProps
} from './gpu-local-frame-projection';

/** Packed single-chunk graph inputs; raw binary64 positions use uint32x4 word rows. */
export type LocalFrameProjectionOperationProps = Pick<
  GPULocalFrameProjectionProps,
  'id' | 'frame' | 'bounds' | 'tolerance'
> & {
  positions: string;
  positionsFormat?: 'float32x2' | 'uint32x4';
  rowCount: number;
  output: string;
  validity?: string;
};

/** Operation form of GPULocalFrameProjection; uses GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPULocalFrameProjectionOperation extends GPUContributorOperation {
  constructor(props: LocalFrameProjectionOperationProps) {
    const id = props.id ?? 'gpu-local-frame-projection';
    super({
      id,
      inputs: {positions: props.positions},
      outputs: {
        output: {value: props.output, format: 'float32x2', length: props.rowCount},
        ...(props.validity === undefined
          ? {}
          : {
              validity: {
                value: props.validity,
                format: 'uint32',
                length: props.rowCount
              }
            })
      },
      createContributor: getView =>
        new GPULocalFrameProjection({
          ...props,
          id,
          positions:
            props.positionsFormat === 'uint32x4'
              ? getView<'uint32x4'>('positions')
              : getView<'float32x2'>('positions'),
          output: getView<'float32x2'>('output'),
          validity: props.validity === undefined ? undefined : getView<'uint32'>('validity')
        })
    });
  }
}
