// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import type {GPUSpatialWeightsInformation} from '../spatial-weights/spatial-weights';
import {
  GPUNeighborhoodSummary,
  type GPUNeighborhoodSummaryStatistic
} from './gpu-neighborhood-summary';
import {
  NEIGHBORHOOD_CAPACITY_LOSS_FIELDS,
  NEIGHBORHOOD_STOPPING_FIELDS,
  snapshotNeighborhoodWeightsStatus,
  validateNeighborhoodChunkLossBindings,
  validateNeighborhoodWeightsCertificates,
  type NeighborhoodWeightsChunkMetadata
} from './neighborhood-weights-status';

const AGGREGATE_LOSS_FIELDS = [
  ...NEIGHBORHOOD_CAPACITY_LOSS_FIELDS,
  ...NEIGHBORHOOD_STOPPING_FIELDS
];

/** Graph-value properties shared by both neighborhood operations. */
export type NeighborhoodSummaryOperationProps = {
  id?: string;
  rowCount: number;
  /** Names of the CSR graph values and the upstream loss views used by the summary. */
  weights: {
    offsets: string;
    neighbors: string;
    weights: string;
    information?: GPUSpatialWeightsInformation;
    /**
     * Matrix-level capacity loss, an exhausted iteration budget or zero convergence suppresses
     * included rows. Aggregate any child loss flags before binding them here.
     * Construction rejects `complete`, `requiredCount`, `unresolvedCount`, `invalidCount` and
     * `approximation`.
     */
    status?: {
      /** Unsupported positive certificate: rejected at construction; use the pure CPU helper. */
      complete?: string;
      overflow?: string;
      candidateOverflow?: string;
      iterationLimitReached?: string;
      converged?: string;
    };
  };
  values?: string;
  valuesFormat?: 'float32' | 'uint32';
  categories?: string;
  mask?: string;
  includeFocal?: boolean;
  focalWeight?: number;
  statistics?: readonly GPUNeighborhoodSummaryStatistic[];
  maximumNeighbors?: number;
  output: {table?: string; modes?: string; entropy?: string; status?: {overflow: string}};
};

/** @internal Capture and validate the binding names shared by both named forms. */
export function snapshotNeighborhoodSummaryOperationProps(
  props: NeighborhoodSummaryOperationProps
): NeighborhoodSummaryOperationProps {
  const id = props.id ?? 'neighborhood-summary';
  const {status, chunks, ...weights} =
    props.weights as NeighborhoodSummaryOperationProps['weights'] &
      NeighborhoodWeightsChunkMetadata<string>;
  const weightsStatus = snapshotNeighborhoodWeightsStatus(status);
  validateNeighborhoodWeightsCertificates(id, 'weights.status', weightsStatus);
  chunks?.forEach((chunk, index) => {
    const chunkStatus = snapshotNeighborhoodWeightsStatus(chunk.weights.status);
    const path = `weights.chunks[${index}].weights.status`;
    validateNeighborhoodWeightsCertificates(id, path, chunkStatus);
    validateNeighborhoodChunkLossBindings(id, path, chunkStatus, AGGREGATE_LOSS_FIELDS);
  });
  return {
    ...props,
    statistics: props.statistics ? [...props.statistics] : undefined,
    weights: {
      ...weights,
      information: props.weights.information ? {...props.weights.information} : undefined,
      status: weightsStatus
    },
    output: {...props.output, status: props.output.status ? {...props.output.status} : undefined}
  };
}

/** Vector operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUNeighborhoodSummaryOperation extends GPUContributorOperation {
  constructor(props: NeighborhoodSummaryOperationProps) {
    props = snapshotNeighborhoodSummaryOperationProps(props);
    const id = props.id ?? 'neighborhood-summary';
    const outputs: Record<string, GPUContributorOperationOutput> = {};
    for (const port of ['table', 'modes', 'entropy'] as const) {
      const value = props.output[port];
      if (value !== undefined)
        outputs[port] = {
          value,
          format: port === 'modes' ? 'uint32' : 'float32',
          length: props.rowCount * (port === 'table' ? (props.statistics ?? []).length : 1)
        };
    }
    if (props.output.status)
      outputs['overflow'] = {value: props.output.status.overflow, format: 'uint32', length: 1};
    super({
      id,
      inputs: {
        offsets: props.weights.offsets,
        neighbors: props.weights.neighbors,
        weights: props.weights.weights,
        ...(props.weights.status?.overflow !== undefined
          ? {weightsOverflow: props.weights.status.overflow}
          : {}),
        ...(props.weights.status?.candidateOverflow !== undefined
          ? {weightsCandidateOverflow: props.weights.status.candidateOverflow}
          : {}),
        ...(props.weights.status?.iterationLimitReached !== undefined
          ? {weightsIterationLimitReached: props.weights.status.iterationLimitReached}
          : {}),
        ...(props.weights.status?.converged !== undefined
          ? {weightsConverged: props.weights.status.converged}
          : {}),
        ...(props.values !== undefined ? {values: props.values} : {}),
        ...(props.categories !== undefined ? {categories: props.categories} : {}),
        ...(props.mask !== undefined ? {mask: props.mask} : {})
      },
      outputs,
      createContributor: getView =>
        new GPUNeighborhoodSummary({
          id: props.id,
          weights: {
            offsets: getView<'uint32'>('offsets'),
            neighbors: getView<'uint32'>('neighbors'),
            weights: getView<'float32'>('weights'),
            information: props.weights.information,
            status:
              props.weights.status !== undefined
                ? {
                    ...(props.weights.status.overflow !== undefined
                      ? {overflow: getView<'uint32'>('weightsOverflow')}
                      : {}),
                    ...(props.weights.status.candidateOverflow !== undefined
                      ? {candidateOverflow: getView<'uint32'>('weightsCandidateOverflow')}
                      : {}),
                    ...(props.weights.status.iterationLimitReached !== undefined
                      ? {iterationLimitReached: getView<'uint32'>('weightsIterationLimitReached')}
                      : {}),
                    ...(props.weights.status.converged !== undefined
                      ? {converged: getView<'uint32'>('weightsConverged')}
                      : {})
                  }
                : undefined
          },
          values: props.values !== undefined ? getView<'float32' | 'uint32'>('values') : undefined,
          categories: props.categories !== undefined ? getView<'uint32'>('categories') : undefined,
          mask: props.mask !== undefined ? getView<'uint32'>('mask') : undefined,
          includeFocal: props.includeFocal,
          focalWeight: props.focalWeight,
          statistics: props.statistics,
          maximumNeighbors: props.maximumNeighbors,
          output: {
            table: props.output.table !== undefined ? getView<'float32'>('table') : undefined,
            modes: props.output.modes !== undefined ? getView<'uint32'>('modes') : undefined,
            entropy: props.output.entropy !== undefined ? getView<'float32'>('entropy') : undefined,
            status:
              props.output.status !== undefined
                ? {overflow: getView<'uint32'>('overflow')}
                : undefined
          }
        })
    });
  }
}
