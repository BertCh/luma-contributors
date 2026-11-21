// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConditionalGaussianSimulationOperationProps} from './interpolation-operation-snapshots';

import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUConditionalGaussianSimulation,
  type GPUConditionalGaussianSimulationProps
} from './gpu-conditional-gaussian-simulation';

/** Shared graph values and explicit input lengths. */
export type ConditionalGaussianSimulationOperationProps = Pick<
  GPUConditionalGaussianSimulationProps,
  'id' | 'spatialContext' | 'covariance' | 'mean' | 'seed' | 'replicateCount' | 'replicateOffset'
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
    props = snapshotConditionalGaussianSimulationOperationProps(props);
    const id = props.id ?? 'conditional-gaussian-simulation';
    validateCapacity(id, 'observationCount', props.observationCount, 0);
    validateCapacity(id, 'queryCount', props.queryCount, 1);
    validateCapacity(id, 'replicateCount', props.replicateCount, 1);
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
      createContributor: getView => {
        const positions = getView<'float32x2'>('positions');
        const observations = getView<'float32x4'>('observations');
        const queries = getView<'float32x2'>('queries');
        validateRequirement(
          id,
          positions.length === props.observationCount,
          'positions.length must equal observationCount'
        );
        validateRequirement(
          id,
          observations.length === props.observationCount,
          'observations.length must equal observationCount'
        );
        validateRequirement(
          id,
          queries.length === props.queryCount,
          'queries.length must equal queryCount'
        );
        return new GPUConditionalGaussianSimulation({
          id,
          spatialContext: props.spatialContext,
          covariance: props.covariance,
          mean: props.mean,
          seed: props.seed,
          replicateCount: props.replicateCount,
          replicateOffset: props.replicateOffset,
          positions,
          observations,
          queries,
          output: {
            values: getView<'float32'>('values'),
            moments: getView<'float32x2'>('moments'),
            status: getView<'uint32'>('status')
          }
        });
      }
    });
  }
}
