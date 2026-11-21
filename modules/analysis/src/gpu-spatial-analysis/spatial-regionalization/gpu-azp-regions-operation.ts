// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import type {SpatialWeightsAlgebraGraphInput} from '../spatial-weights/gpu-spatial-weights-algebra-operation';
import {GPUAZPRegions, GPU_AZP_STATUS_LAYOUT} from './gpu-azp-regions';
import {snapshotWeightsPort} from '../spatial-weights/weights-construction-snapshot';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  type RegionalizationWeightsStatus,
  validateRegionalizationWeightsStatus
} from './adjacency-loss-suppression';

/** Graph-value names shared by AZP twins, with view-inferred rowCount made explicit. */
export type AZPRegionsOperationProps = {
  id?: string;
  rowCount: number;
  weights: SpatialWeightsAlgebraGraphInput & {
    status?: RegionalizationWeightsStatus<string>;
  };
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

/** @internal Capture declared names/options once, including inherited and non-enumerable getters. */
export function snapshotAZPRegionsOperationProps(
  props: AZPRegionsOperationProps
): AZPRegionsOperationProps {
  const captured = snapshotConstructionFields(props, [
    'id',
    'rowCount',
    'weights',
    'values',
    'columnCount',
    'standardize',
    'extensiveValues',
    'extensiveThreshold',
    'initialLabels',
    'regionCapacity',
    'minimumRegionSize',
    'maximumIterations',
    'parameters',
    'output'
  ]);
  const weights = snapshotWeightsPort(captured.weights);
  validateRegionalizationWeightsStatus(captured.id ?? 'azp-regions', weights.status);
  return Object.freeze({
    ...captured,
    weights,
    output: snapshotConstructionFields(captured.output, ['labels', 'objective', 'status'])
  });
}

/** Vector operation form. Numeric props are compile-time; graph values are per-frame. */
export class GPUAZPRegionsOperation extends GPUContributorOperation {
  constructor(props: AZPRegionsOperationProps) {
    props = snapshotAZPRegionsOperationProps(props);
    super({
      id: props.id ?? 'azp-regions',
      inputs: {
        offsets: props.weights.offsets,
        neighbors: props.weights.neighbors,
        weights: props.weights.weights,
        ...(props.weights.status?.overflow === undefined
          ? {}
          : {weightsOverflow: props.weights.status.overflow}),
        ...(props.weights.status?.candidateOverflow === undefined
          ? {}
          : {weightsCandidateOverflow: props.weights.status.candidateOverflow}),
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
        status: {value: props.output.status, format: 'uint32', length: GPU_AZP_STATUS_LAYOUT.length}
      },
      createContributor: getView =>
        new GPUAZPRegions({
          ...props,
          weights: {
            ...props.weights,
            offsets: getView<'uint32'>('offsets'),
            neighbors: getView<'uint32'>('neighbors'),
            weights: getView<'float32'>('weights'),
            status: {
              overflow:
                props.weights.status?.overflow === undefined
                  ? undefined
                  : getView<'uint32'>('weightsOverflow'),
              candidateOverflow:
                props.weights.status?.candidateOverflow === undefined
                  ? undefined
                  : getView<'uint32'>('weightsCandidateOverflow')
            },
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
