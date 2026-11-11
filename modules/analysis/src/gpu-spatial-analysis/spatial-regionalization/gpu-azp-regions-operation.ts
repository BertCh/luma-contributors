// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import type {SpatialWeightsAlgebraGraphInput} from '../spatial-weights/gpu-spatial-weights-algebra-operation';
import {GPUAZPRegions, GPU_AZP_STATUS} from './gpu-azp-regions';

/** Graph-value names shared by AZP twins, with view-inferred rowCount made explicit. */
export type AZPRegionsOperationProps = {
  id?: string;
  rowCount: number;
  weights: SpatialWeightsAlgebraGraphInput;
  values: string;
  columnCount?: number;
  standardize?: boolean;
  extensiveValues?: string;
  extensiveThreshold?: string;
  initialLabels: string;
  regionCapacity: number;
  minimumRegionSize?: number;
  maximumIterations: number;
  /** uint32 seed layout packed by getGPUAZPRegionsParameterValues. */
  parameters: string;
  output: {labels: string; objective: string; status: string};
};

/** Vector operation form. Numeric props are compile-time; graph values are per-frame. */
export class GPUAZPRegionsOperation extends GPUContributorOperation {
  constructor(props: AZPRegionsOperationProps) {
    super({
      id: props.id ?? 'azp-regions',
      inputs: {
        offsets: props.weights.offsets,
        neighbors: props.weights.neighbors,
        weights: props.weights.weights,
        ...(props.weights.distances === undefined ? {} : {distances: props.weights.distances}),
        values: props.values,
        initialLabels: props.initialLabels,
        parameters: props.parameters,
        ...(props.extensiveValues === undefined ? {} : {extensiveValues: props.extensiveValues}),
        ...(props.extensiveThreshold === undefined
          ? {}
          : {extensiveThreshold: props.extensiveThreshold})
      },
      outputs: {
        labels: {value: props.output.labels, format: 'uint32', length: props.rowCount},
        objective: {value: props.output.objective, format: 'float32', length: 1},
        status: {value: props.output.status, format: 'uint32', length: GPU_AZP_STATUS.length}
      },
      createContributor: getView =>
        new GPUAZPRegions({
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
          initialLabels: getView<'uint32'>('initialLabels'),
          extensiveValues:
            props.extensiveValues === undefined ? undefined : getView<'float32'>('extensiveValues'),
          extensiveThreshold:
            props.extensiveThreshold === undefined
              ? undefined
              : getView<'float32'>('extensiveThreshold'),
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
