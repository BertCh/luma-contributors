// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPURasterReconstruction,
  type GPURasterReconstructionProps
} from '../gpu-raster-reconstruction';

/** Graph-value properties shared by both reconstruction operations. */
export type RasterReconstructionOperationProps = Pick<
  GPURasterReconstructionProps,
  'id' | 'grid' | 'method' | 'connectivity' | 'maximumIterations'
> & {
  marker: string;
  mask: string;
  output: {
    values: string;
    validity: string;
    status: {converged: string; iterationLimitReached?: string; iterationCount?: string};
  };
};

/** GPU reconstruction as a vector operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPURasterReconstructionOperation extends GPUContributorOperation {
  constructor(props: RasterReconstructionOperationProps) {
    const length = props.grid.width * props.grid.height;
    super({
      id: props.id ?? 'raster-reconstruction',
      inputs: {marker: props.marker, mask: props.mask},
      outputs: {
        values: {value: props.output.values, format: 'float32', length},
        validity: {value: props.output.validity, format: 'uint32', length},
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
        new GPURasterReconstruction({
          ...props,
          marker: getView<'float32'>('marker'),
          mask: getView<'float32'>('mask'),
          output: {
            values: getView<'float32'>('values'),
            validity: getView<'uint32'>('validity'),
            status: {
              converged: getView<'uint32'>('converged'),
              iterationLimitReached:
                props.output.status.iterationLimitReached !== undefined
                  ? getView<'uint32'>('iterationLimitReached')
                  : undefined,
              iterationCount:
                props.output.status.iterationCount !== undefined
                  ? getView<'uint32'>('iterationCount')
                  : undefined
            }
          }
        })
    });
  }
}
