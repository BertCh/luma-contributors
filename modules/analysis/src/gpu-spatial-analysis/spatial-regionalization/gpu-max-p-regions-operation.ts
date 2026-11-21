// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import type {SpatialWeightsAlgebraGraphInput} from '../spatial-weights/gpu-spatial-weights-algebra-operation';
import {GPUMaxPRegions, GPU_MAX_P_STATUS_LAYOUT, GPU_MAX_P_EVIDENCE} from './gpu-max-p-regions';
import {snapshotWeightsPort} from '../spatial-weights/weights-construction-snapshot';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  type RegionalizationWeightsStatus,
  validateRegionalizationWeightsStatus
} from './adjacency-loss-suppression';

/** Graph-value names shared by both twins; graph outputs have their exact minimal lengths. */
export type MaxPRegionsOperationProps = {
  id?: string;
  rowCount: number;
  weights: SpatialWeightsAlgebraGraphInput & {
    status?: RegionalizationWeightsStatus<string>;
  };
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
  output: {labels: string; objective: string; status: string; evidence?: string};
};

/** @internal Capture declared names/options once, including inherited and non-enumerable getters. */
export function snapshotMaxPRegionsOperationProps(
  props: MaxPRegionsOperationProps
): MaxPRegionsOperationProps {
  const captured = snapshotConstructionFields(props, [
    'id',
    'rowCount',
    'weights',
    'values',
    'columnCount',
    'standardize',
    'extensiveValues',
    'threshold',
    'startCount',
    'maximumGrowthSteps',
    'maximumIterations',
    'parameters',
    'output'
  ]);
  const weights = snapshotWeightsPort(captured.weights);
  validateRegionalizationWeightsStatus(captured.id ?? 'max-p-regions', weights.status);
  return Object.freeze({
    ...captured,
    weights,
    output: snapshotConstructionFields(captured.output, [
      'labels',
      'objective',
      'status',
      'evidence'
    ])
  });
}

/** Vector operation form. All numeric props are compile-time; graph values are per-frame. */
export class GPUMaxPRegionsOperation extends GPUContributorOperation {
  constructor(props: MaxPRegionsOperationProps) {
    props = snapshotMaxPRegionsOperationProps(props);
    super({
      id: props.id ?? 'max-p-regions',
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
        extensiveValues: props.extensiveValues,
        threshold: props.threshold,
        parameters: props.parameters
      },
      outputs: {
        labels: {value: props.output.labels, format: 'uint32', length: props.rowCount},
        objective: {value: props.output.objective, format: 'float32', length: 1},
        status: {
          value: props.output.status,
          format: 'uint32',
          length: GPU_MAX_P_STATUS_LAYOUT.length
        },
        ...(props.output.evidence === undefined
          ? {}
          : {
              evidence: {
                value: props.output.evidence,
                format: 'uint32' as const,
                length: GPU_MAX_P_EVIDENCE.length
              }
            })
      },
      createContributor: getView =>
        new GPUMaxPRegions({
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
          extensiveValues: getView<'float32'>('extensiveValues'),
          threshold: getView<'float32'>('threshold'),
          parameters: getView<'uint32'>('parameters'),
          output: {
            labels: getView<'uint32'>('labels'),
            objective: getView<'float32'>('objective'),
            status: getView<'uint32'>('status'),
            evidence:
              props.output.evidence === undefined ? undefined : getView<'uint32'>('evidence')
          }
        })
    });
  }
}
