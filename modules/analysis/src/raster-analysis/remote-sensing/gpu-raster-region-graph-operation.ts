// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPURasterRegionGraph} from './gpu-raster-region-graph';

/** Value names replacing views; matrices have exactly `regionCount ** 2` scalar rows. */
export type RasterRegionGraphOperationProps = {
  id?: string;
  grid: GPURasterMetadata;
  labels: string;
  values: string;
  regionCount: number;
  mergeMode?: 'fixed' | 'adaptive';
  mergeThreshold: number;
  maximumIterations: number;
  output: {
    adjacency: string;
    weights: string;
    mapping: string;
    status: {iterationLimitReached: string};
  };
};

/** Operation form of the contributor, with vector-only ports and no additional behaviour. */
export class GPURasterRegionGraphOperation extends GPUContributorOperation {
  constructor(props: RasterRegionGraphOperationProps) {
    const id = props.id ?? 'raster-region-graph';
    super({
      id,
      inputs: {labels: props.labels, values: props.values},
      outputs: {
        adjacency: {
          value: props.output.adjacency,
          format: 'uint32',
          length: props.regionCount ** 2
        },
        weights: {value: props.output.weights, format: 'float32', length: props.regionCount ** 2},
        mapping: {value: props.output.mapping, format: 'uint32', length: props.regionCount},
        iterationLimitReached: {
          value: props.output.status.iterationLimitReached,
          format: 'uint32',
          length: 1
        }
      },
      createContributor: getView =>
        new GPURasterRegionGraph({
          ...props,
          id,
          labels: getView<'uint32'>('labels'),
          values: getView<'float32'>('values'),
          output: {
            adjacency: getView<'uint32'>('adjacency'),
            weights: getView<'float32'>('weights'),
            mapping: getView<'uint32'>('mapping'),
            status: {iterationLimitReached: getView<'uint32'>('iterationLimitReached')}
          }
        })
    });
  }
}
