// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {captureRasterMorphologyOperationProps} from './raster-morphology-operation-utils';
import {GPURasterFillHoles, type GPURasterFillHolesProps} from '../gpu-raster-reconstruction';

/** Graph-value properties shared by both hole-fill operations. */
export type RasterFillHolesOperationProps = Pick<
  GPURasterFillHolesProps,
  'id' | 'grid' | 'connectivity' | 'maximumIterations'
> & {
  input: string;
  output: {
    values: string;
    status: {converged: string; iterationLimitReached?: string; iterationCount?: string};
  };
};

/** GPU hole filling as a vector operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPURasterFillHolesOperation extends GPUContributorOperation {
  constructor(props: RasterFillHolesOperationProps) {
    props = captureRasterMorphologyOperationProps(props);
    super({
      id: props.id ?? 'raster-fill-holes',
      inputs: {input: props.input},
      outputs: {
        values: {
          value: props.output.values,
          format: 'uint32',
          length: props.grid.width * props.grid.height
        },
        converged: {value: props.output.status.converged, format: 'uint32', length: 1},
        ...(props.output.status.iterationLimitReached !== undefined
          ? {
              iterationLimitReached: {
                value: props.output.status.iterationLimitReached,
                format: 'uint32',
                length: 1
              }
            }
          : {}),
        ...(props.output.status.iterationCount !== undefined
          ? {
              iterationCount: {
                value: props.output.status.iterationCount,
                format: 'uint32',
                length: 1
              }
            }
          : {})
      },
      createContributor: getView =>
        new GPURasterFillHoles({
          ...props,
          input: getView<'uint32'>('input'),
          output: {
            values: getView<'uint32'>('values'),
            status: {
              converged: getView<'uint32'>('converged'),
              iterationCount:
                props.output.status.iterationCount !== undefined
                  ? getView<'uint32'>('iterationCount')
                  : undefined,
              iterationLimitReached:
                props.output.status.iterationLimitReached !== undefined
                  ? getView<'uint32'>('iterationLimitReached')
                  : undefined
            }
          }
        })
    });
  }
}
