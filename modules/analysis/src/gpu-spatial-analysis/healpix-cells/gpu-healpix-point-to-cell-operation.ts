// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUHealpixPointToCell, type GPUHealpixPointToCellProps} from './gpu-healpix-point-to-cell';

/** Same properties as the contributor, with graph names and an explicit position row count. */
export type HealpixPointToCellOperationProps = Pick<
  GPUHealpixPointToCellProps,
  'id' | 'order' | 'scheme' | 'spatialContext'
> & {
  positions: string;
  mask?: string;
  rowCount: number;
  output: {cells: string; validity?: string};
};

/** HEALPix projection as a graph operation; use GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUHealpixPointToCellOperation extends GPUContributorOperation {
  constructor(props: HealpixPointToCellOperationProps) {
    const id = props.id ?? 'healpix-point-to-cell';
    super({
      id,
      inputs: {positions: props.positions, ...(props.mask ? {mask: props.mask} : {})},
      outputs: {
        cells: {value: props.output.cells, format: 'uint32x2', length: props.rowCount},
        ...(props.output.validity
          ? {validity: {value: props.output.validity, format: 'uint32', length: props.rowCount}}
          : {})
      },
      createContributor: getView =>
        new GPUHealpixPointToCell({
          id,
          order: props.order,
          scheme: props.scheme,
          spatialContext: props.spatialContext,
          positions: getView<'float32x2'>('positions'),
          mask: props.mask ? getView<'uint32'>('mask') : undefined,
          output: {
            cells: getView<'uint32x2'>('cells'),
            validity: props.output.validity ? getView<'uint32'>('validity') : undefined
          }
        })
    });
  }
}
