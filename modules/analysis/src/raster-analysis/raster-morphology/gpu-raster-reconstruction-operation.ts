// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {captureRasterMorphologyOperationProps} from './raster-morphology-operation-utils';
import {
  resolveRasterOperationBand,
  type RasterOperationBand
} from '../raster-grid/raster-operation-band';
import {
  GPURasterReconstruction,
  type GPURasterReconstructionProps
} from '../gpu-raster-reconstruction';

/** Graph-value properties shared by both reconstruction operations. */
export type RasterReconstructionOperationProps = Pick<
  GPURasterReconstructionProps,
  'id' | 'grid' | 'method' | 'connectivity' | 'maximumIterations'
> & {
  marker: string | RasterOperationBand;
  mask: string | RasterOperationBand;
  output: {
    values: string;
    validity: string;
    status: {converged: string; iterationLimitReached?: string; iterationCount?: string};
  };
};

/** GPU reconstruction as a vector operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPURasterReconstructionOperation extends GPUContributorOperation {
  constructor(props: RasterReconstructionOperationProps) {
    props = captureRasterMorphologyOperationProps(props);
    const length = props.grid.width * props.grid.height;
    super({
      id: props.id ?? 'raster-reconstruction',
      inputs: {
        marker: typeof props.marker === 'string' ? props.marker : props.marker.storage.values,
        mask: typeof props.mask === 'string' ? props.mask : props.mask.storage.values,
        ...(typeof props.marker !== 'string' && props.marker.validity
          ? {markerValidity: props.marker.validity}
          : {}),
        ...(typeof props.mask !== 'string' && props.mask.validity
          ? {maskValidity: props.mask.validity}
          : {})
      },
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
          marker:
            typeof props.marker === 'string'
              ? getView<'float32'>('marker')
              : resolveRasterOperationBand(props.marker, name =>
                  getView(name === 'values' ? 'marker' : 'markerValidity')
                ),
          mask:
            typeof props.mask === 'string'
              ? getView<'float32'>('mask')
              : resolveRasterOperationBand(props.mask, name =>
                  getView(name === 'values' ? 'mask' : 'maskValidity')
                ),
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
