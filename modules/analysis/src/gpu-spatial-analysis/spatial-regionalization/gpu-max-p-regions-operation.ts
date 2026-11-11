// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import type {SpatialWeightsAlgebraGraphInput} from '../spatial-weights/gpu-spatial-weights-algebra-operation';
import {GPUMaxPRegions, GPU_MAX_P_STATUS} from './gpu-max-p-regions';

/** Graph-value names shared by both twins; graph outputs have their exact minimal lengths. */
export type MaxPRegionsOperationProps = {
  id?: string;
  rowCount: number;
  weights: SpatialWeightsAlgebraGraphInput;
  values: string;
  columnCount?: number;
  standardize?: boolean;
  extensiveValues: string;
  threshold: string;
  startCount?: number;
  maximumGrowthSteps?: number;
  maximumIterations?: number;
  /** uint32 seed layout packed by getGPUMaxPRegionsParameterValues. */
  parameters: string;
  output: {labels: string; objective: string; status: string};
};

/** Vector operation form. All numeric props are compile-time; graph values are per-frame. */
export class GPUMaxPRegionsOperation extends GPUContributorOperation {
  constructor(props: MaxPRegionsOperationProps) {
    super({
      id: props.id ?? 'max-p-regions',
      inputs: {
        offsets: props.weights.offsets,
        neighbors: props.weights.neighbors,
        weights: props.weights.weights,
        ...(props.weights.distances === undefined ? {} : {distances: props.weights.distances}),
        values: props.values,
        extensiveValues: props.extensiveValues,
        threshold: props.threshold,
        parameters: props.parameters
      },
      outputs: {
        labels: {value: props.output.labels, format: 'uint32', length: props.rowCount},
        objective: {value: props.output.objective, format: 'float32', length: 1},
        status: {value: props.output.status, format: 'uint32', length: GPU_MAX_P_STATUS.length}
      },
      createContributor: getView =>
        new GPUMaxPRegions({
          ...props,
          weights: {
            ...props.weights,
            offsets: getView<'uint32'>('offsets'),
            neighbors: getView<'uint32'>('neighbors'),
            weights: getView<'float32'>('weights'),
            distances:
              props.weights.distances === undefined ? undefined : getView<'float32'>('distances')
          },
          values: getView<'float32'>('values'),
          extensiveValues: getView<'float32'>('extensiveValues'),
          threshold: getView<'float32'>('threshold'),
          parameters: getView<'uint32'>('parameters'),
          output: {
            labels: getView<'uint32'>('labels'),
            objective: getView<'float32'>('objective'),
            status: getView<'uint32'>('status')
          }
        })
    });
  }
}
