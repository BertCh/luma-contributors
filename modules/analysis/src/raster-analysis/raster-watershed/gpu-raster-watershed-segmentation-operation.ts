// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {snapshotRasterGrid} from '../raster-grid/raster-construction-snapshot';
import {
  resolveRasterOperationBand,
  snapshotRasterOperationBand,
  validateRasterOperationOutputNames,
  type RasterOperationBand
} from '../raster-grid/raster-operation-band';
import {GPURasterWatershedSegmentation} from '../gpu-raster-watershed-segmentation';

/** Vector-only buffer-band operation; output labels have `grid.width * grid.height` rows. */
export type RasterWatershedSegmentationOperationProps = {
  id?: string;
  grid: GPURasterMetadata;
  input: RasterOperationBand;
  markers: string;
  connectivity?: 4 | 8;
  method?: 'fifo' | 'minimum-spanning-forest';
  maximumSteps?: number;
  maximumIterations?: number;
  output: {labels: string; status: {converged: string; iterationLimitReached?: string}};
};

/** Operation form of the buffer-backed watershed contributor. */
export class GPURasterWatershedSegmentationOperation extends GPUContributorOperation {
  constructor(props: RasterWatershedSegmentationOperationProps) {
    props = snapshotConstructionFields(props, [
      'id',
      'grid',
      'input',
      'markers',
      'connectivity',
      'method',
      'maximumSteps',
      'maximumIterations',
      'output'
    ]);
    const output = snapshotConstructionFields(props.output, ['labels', 'status']);
    props = Object.freeze({
      ...props,
      grid: snapshotRasterGrid(props.grid),
      input: snapshotRasterOperationBand(props.input),
      output: Object.freeze({
        ...output,
        status: snapshotConstructionFields(output.status, ['converged', 'iterationLimitReached'])
      })
    });
    const id = props.id ?? 'raster-watershed-segmentation';
    validateRasterOperationOutputNames(
      id,
      {
        'output.labels': props.output.labels,
        'output.status.converged': props.output.status.converged
      },
      {'output.status.iterationLimitReached': props.output.status.iterationLimitReached},
      {
        'input.storage.values': props.input.storage.values,
        ...(props.input.validity !== undefined ? {'input.validity': props.input.validity} : {}),
        markers: props.markers
      }
    );
    const iterationLimitReached = props.output.status.iterationLimitReached;
    super({
      id,
      inputs: {
        values: props.input.storage.values,
        markers: props.markers,
        ...(props.input.validity !== undefined ? {validity: props.input.validity} : {})
      },
      outputs: {
        labels: {
          value: props.output.labels,
          format: 'uint32',
          length: props.grid.width * props.grid.height
        },
        converged: {value: props.output.status.converged, format: 'uint32', length: 1},
        ...(iterationLimitReached !== undefined
          ? {
              iterationLimitReached: {
                value: iterationLimitReached,
                format: 'uint32' as const,
                length: 1
              }
            }
          : {})
      },
      createContributor: getView =>
        new GPURasterWatershedSegmentation({
          ...props,
          id,
          input: resolveRasterOperationBand(props.input, getView),
          markers: getView<'uint32'>('markers'),
          output: {
            labels: getView<'uint32'>('labels'),
            status: {
              converged: getView<'uint32'>('converged'),
              iterationLimitReached:
                iterationLimitReached !== undefined
                  ? getView<'uint32'>('iterationLimitReached')
                  : undefined
            }
          }
        })
    });
  }
}
