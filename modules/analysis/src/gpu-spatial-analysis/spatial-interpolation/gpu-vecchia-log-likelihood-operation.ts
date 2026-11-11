// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUVecchiaLogLikelihood,
  type GPUVecchiaLogLikelihoodProps
} from './gpu-vecchia-log-likelihood';

/** Names of graph values in place of packed views. Shared by the GPU operation and CPU twin. */
export type VecchiaLogLikelihoodOperationProps = Pick<
  GPUVecchiaLogLikelihoodProps,
  'id' | 'spatialContext' | 'neighborCount'
> & {
  /** Explicit length of positions and all per-site outputs. Compile-time. */
  rowCount: number;
  positions: string;
  observations: string;
  neighborhoodOffsets: string;
  neighborhoods: string;
  parameters: string;
  output: {
    conditionals: string;
    conditionalStatus: string;
    siteTerms: string;
    siteStatus: string;
    logLikelihood: string;
    status: string;
  };
};

/**
 * Vector operation form of GPUVecchiaLogLikelihood; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING.
 * Counts and contexts are compile-time; the packed parameters graph value is per-frame.
 */
export class GPUVecchiaLogLikelihoodOperation extends GPUContributorOperation {
  constructor(props: VecchiaLogLikelihoodOperationProps) {
    const id = props.id ?? 'vecchia-log-likelihood';
    super({
      id,
      inputs: {
        positions: props.positions,
        observations: props.observations,
        neighborhoodOffsets: props.neighborhoodOffsets,
        neighborhoods: props.neighborhoods,
        parameters: props.parameters
      },
      outputs: {
        conditionals: {
          value: props.output.conditionals,
          format: 'float32x4',
          length: props.rowCount
        },
        conditionalStatus: {
          value: props.output.conditionalStatus,
          format: 'uint32',
          length: props.rowCount
        },
        siteTerms: {value: props.output.siteTerms, format: 'float32', length: props.rowCount},
        siteStatus: {value: props.output.siteStatus, format: 'uint32', length: props.rowCount},
        logLikelihood: {value: props.output.logLikelihood, format: 'float32', length: 2},
        status: {value: props.output.status, format: 'uint32', length: 3}
      },
      createContributor: getView =>
        new GPUVecchiaLogLikelihood({
          id,
          spatialContext: props.spatialContext,
          neighborCount: props.neighborCount,
          positions: getView<'float32x2'>('positions'),
          observations: getView<'float32x4'>('observations'),
          neighborhoodOffsets: getView<'uint32'>('neighborhoodOffsets'),
          neighborhoods: getView<'uint32'>('neighborhoods'),
          parameters: getView<'float32'>('parameters'),
          output: {
            conditionals: getView<'float32x4'>('conditionals'),
            conditionalStatus: getView<'uint32'>('conditionalStatus'),
            siteTerms: getView<'float32'>('siteTerms'),
            siteStatus: getView<'uint32'>('siteStatus'),
            logLikelihood: getView<'float32'>('logLikelihood'),
            status: getView<'uint32'>('status')
          }
        })
    });
  }
}
