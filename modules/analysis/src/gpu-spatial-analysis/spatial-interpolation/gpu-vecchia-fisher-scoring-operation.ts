// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotVecchiaLossProps} from './vecchia-neighborhood-status';

import {snapshotVecchiaFisherScoringOperationProps} from './interpolation-operation-snapshots';

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUVecchiaFisherScoring,
  type GPUVecchiaFisherScoringProps,
  GPU_VECCHIA_FISHER_CELL_COUNT
} from './gpu-vecchia-fisher-scoring';

/** Names of graph values in place of packed views. Shared by the GPU operation and CPU twin. */
export type VecchiaFisherScoringOperationProps = Pick<
  GPUVecchiaFisherScoringProps,
  'id' | 'spatialContext' | 'neighborCount'
> & {
  /** Explicit length of positions and all per-site outputs. Compile-time. */
  rowCount: number;
  positions: string;
  observations: string;
  neighborhoodOffsets: string;
  neighborhoods: string;
  neighborhoodStatus?: {overflow?: string; candidateOverflow?: string};
  parameters: string;
  output: {
    siteTerms: string;
    siteStatus: string;
    totals: string;
    status: string;
    neighborhoodStatus?: string;
  };
};

/**
 * Vector operation form of GPUVecchiaFisherScoring; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING.
 * Counts and contexts are compile-time; the packed parameters graph value is per-frame.
 */
export class GPUVecchiaFisherScoringOperation extends GPUContributorOperation {
  constructor(props: VecchiaFisherScoringOperationProps) {
    props = snapshotVecchiaLossProps(props, snapshotVecchiaFisherScoringOperationProps);
    const id = props.id ?? 'vecchia-fisher-scoring';
    super({
      id,
      inputs: {
        positions: props.positions,
        observations: props.observations,
        neighborhoodOffsets: props.neighborhoodOffsets,
        neighborhoods: props.neighborhoods,
        ...(props.neighborhoodStatus?.overflow !== undefined
          ? {neighborhoodOverflow: props.neighborhoodStatus.overflow}
          : {}),
        ...(props.neighborhoodStatus?.candidateOverflow !== undefined
          ? {neighborhoodCandidateOverflow: props.neighborhoodStatus.candidateOverflow}
          : {}),
        parameters: props.parameters
      },
      outputs: {
        siteTerms: {
          value: props.output.siteTerms,
          format: 'float32',
          length: props.rowCount * GPU_VECCHIA_FISHER_CELL_COUNT
        },
        siteStatus: {value: props.output.siteStatus, format: 'uint32', length: props.rowCount},
        totals: {
          value: props.output.totals,
          format: 'float32',
          length: GPU_VECCHIA_FISHER_CELL_COUNT * 2
        },
        ...(props.output.neighborhoodStatus !== undefined
          ? {
              neighborhoodStatus: {
                value: props.output.neighborhoodStatus,
                format: 'uint32' as const,
                length: 3
              }
            }
          : {}),
        status: {value: props.output.status, format: 'uint32', length: 3}
      },
      createContributor: getView =>
        new GPUVecchiaFisherScoring({
          id,
          spatialContext: props.spatialContext,
          neighborCount: props.neighborCount,
          positions: getView<'float32x2'>('positions'),
          observations: getView<'float32x4'>('observations'),
          neighborhoodOffsets: getView<'uint32'>('neighborhoodOffsets'),
          neighborhoods: getView<'uint32'>('neighborhoods'),
          parameters: getView<'float32'>('parameters'),
          neighborhoodStatus: {
            overflow:
              props.neighborhoodStatus?.overflow !== undefined
                ? getView<'uint32'>('neighborhoodOverflow')
                : undefined,
            candidateOverflow:
              props.neighborhoodStatus?.candidateOverflow !== undefined
                ? getView<'uint32'>('neighborhoodCandidateOverflow')
                : undefined
          },
          output: {
            siteTerms: getView<'float32'>('siteTerms'),
            siteStatus: getView<'uint32'>('siteStatus'),
            totals: getView<'float32'>('totals'),
            ...(props.output.neighborhoodStatus !== undefined
              ? {neighborhoodStatus: getView<'uint32'>('neighborhoodStatus')}
              : {}),
            status: getView<'uint32'>('status')
          }
        })
    });
  }
}
