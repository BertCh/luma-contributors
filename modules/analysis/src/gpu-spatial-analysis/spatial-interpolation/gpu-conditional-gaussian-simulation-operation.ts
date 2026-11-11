// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUConditionalGaussianSimulation,
  type GPUConditionalGaussianSimulationProps
} from './gpu-conditional-gaussian-simulation';

/** Shared graph values and explicit input lengths. Reuse is a GPU preparation optimization only. */
export type ConditionalGaussianSimulationOperationProps = Pick<
  GPUConditionalGaussianSimulationProps,
  | 'id'
  | 'spatialContext'
  | 'covariance'
  | 'mean'
  | 'seed'
  | 'replicateCount'
  | 'replicateOffset'
  | 'reusePreparation'
> & {
  observationCount: number;
  queryCount: number;
  positions: string;
  observations: string;
  queries: string;
  output: {values: string; moments: string; status: string};
};

/** Vector operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUConditionalGaussianSimulationOperation extends GPUContributorOperation {
  constructor(props: ConditionalGaussianSimulationOperationProps) {
    const id = props.id ?? 'conditional-gaussian-simulation';
    super({
      id,
      inputs: {
        positions: props.positions,
        observations: props.observations,
        queries: props.queries
      },
      outputs: {
        values: {
          value: props.output.values,
          format: 'float32',
          length: props.queryCount * props.replicateCount
        },
        moments: {value: props.output.moments, format: 'float32x2', length: props.queryCount},
        status: {value: props.output.status, format: 'uint32', length: props.replicateCount}
      },
      createContributor: getView =>
        new GPUConditionalGaussianSimulation({
          id,
          spatialContext: props.spatialContext,
          covariance: props.covariance,
          mean: props.mean,
          seed: props.seed,
          replicateCount: props.replicateCount,
          replicateOffset: props.replicateOffset,
          reusePreparation: props.reusePreparation,
          positions: getView<'float32x2'>('positions'),
          observations: getView<'float32x4'>('observations'),
          queries: getView<'float32x2'>('queries'),
          output: {
            values: getView<'float32'>('values'),
            moments: getView<'float32x2'>('moments'),
            status: getView<'uint32'>('status')
          }
        })
    });
  }
}
