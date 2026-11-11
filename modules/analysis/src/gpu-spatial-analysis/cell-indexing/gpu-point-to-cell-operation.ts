// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUPointToCell, type GPUPointToCellProps} from './gpu-point-to-cell';

/** Graph-value properties shared by the GPU operation and its CPU twin. */
export type PointToCellOperationProps = Omit<
  GPUPointToCellProps,
  'positions' | 'mask' | 'output'
> & {
  /** Compile-time number of positions and output rows. */
  rowCount: number;
  /** Longitude/latitude degrees, float32x2. */
  positions: string;
  /** Optional uint32 row mask. */
  mask?: string;
  output: {cells: string; validity?: string};
};

/** Vector operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUPointToCellOperation extends GPUContributorOperation {
  constructor(props: PointToCellOperationProps) {
    super({
      id: props.id ?? 'point-to-cell',
      inputs: {positions: props.positions, ...(props.mask === undefined ? {} : {mask: props.mask})},
      outputs: {
        cells: {value: props.output.cells, format: 'uint32x2', length: props.rowCount},
        ...(props.output.validity === undefined
          ? {}
          : {
              validity: {
                value: props.output.validity,
                format: 'uint32' as const,
                length: props.rowCount
              }
            })
      },
      createContributor: getView =>
        new GPUPointToCell({
          ...props,
          positions: getView<'float32x2'>('positions'),
          mask: props.mask === undefined ? undefined : getView<'uint32'>('mask'),
          output: {
            cells: getView<'uint32x2'>('cells'),
            validity:
              props.output.validity === undefined ? undefined : getView<'uint32'>('validity')
          }
        })
    });
  }
}
