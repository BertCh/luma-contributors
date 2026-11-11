// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPURasterMedoidComposite} from './gpu-raster-medoid-composite';

/** Scalar-vector graph ports and compile-time stack dimensions. */
export type RasterMedoidCompositeOperationProps = {
  id?: string;
  cellCount: number;
  sliceCount: number;
  bandCount: number;
  values: string;
  /** Per-observation uint32 input; nonzero keeps the observation. */
  mask?: string;
  output: {composite: string; sliceIndex: string; status: string};
};

/** Operation form of the medoid contributor; outputs have the exact dense stack shape. */
export class GPURasterMedoidCompositeOperation extends GPUContributorOperation {
  constructor(props: RasterMedoidCompositeOperationProps) {
    const id = props.id ?? 'raster-medoid-composite';
    super({
      id,
      inputs: {
        values: props.values,
        ...(props.mask !== undefined ? {mask: props.mask} : {})
      },
      outputs: {
        composite: {
          value: props.output.composite,
          format: 'float32',
          length: props.cellCount * props.bandCount
        },
        sliceIndex: {value: props.output.sliceIndex, format: 'uint32', length: props.cellCount},
        status: {value: props.output.status, format: 'uint32', length: props.cellCount}
      },
      createContributor: getView =>
        new GPURasterMedoidComposite({
          ...props,
          id,
          values: getView<'float32'>('values'),
          mask: props.mask !== undefined ? getView<'uint32'>('mask') : undefined,
          output: {
            composite: getView<'float32'>('composite'),
            sliceIndex: getView<'uint32'>('sliceIndex'),
            status: getView<'uint32'>('status')
          }
        })
    });
  }
}
