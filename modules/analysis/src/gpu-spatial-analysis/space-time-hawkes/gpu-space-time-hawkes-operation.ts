// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUSpaceTimeHawkes, type GPUSpaceTimeHawkesProps} from './gpu-space-time-hawkes';
import {GPU_SPACE_TIME_HAWKES_ESTIMATE_LENGTH} from './space-time-hawkes-parameters';

/**
 * Shared graph-value props; output arrays use the minimum sizes the contributor requires.
 * maximumIterations bounds either operation: default 100 on GPU, 1000 on CPU. CPU has no
 * 1000-iteration unrolling cap. maximumParents caps only the GPU's parent enumeration.
 */
export type SpaceTimeHawkesOperationProps = Pick<
  GPUSpaceTimeHawkesProps,
  'id' | 'spatialContext' | 'temporalContext' | 'maximumIterations' | 'maximumParents'
> & {
  rowCount: number;
  positions: string;
  times: string;
  parameters: string;
  output: {
    estimates: string;
    logLikelihood?: string;
    backgroundProbability?: string;
    status?: {
      converged?: string;
      iterationLimitReached?: string;
      iterationCount?: string;
      candidateOverflow?: string;
      invalidCount?: string;
      approximation?: string;
    };
  };
};

/** Vector operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUSpaceTimeHawkesOperation extends GPUContributorOperation {
  constructor(props: SpaceTimeHawkesOperationProps) {
    const id = props.id ?? 'space-time-hawkes';
    super({
      id,
      inputs: {positions: props.positions, times: props.times, parameters: props.parameters},
      outputs: {
        estimates: {
          value: props.output.estimates,
          format: 'float32',
          length: GPU_SPACE_TIME_HAWKES_ESTIMATE_LENGTH
        },
        ...(props.output.logLikelihood === undefined
          ? {}
          : {
              logLikelihood: {
                value: props.output.logLikelihood,
                format: 'float32' as const,
                length: 1
              }
            }),
        ...(props.output.backgroundProbability === undefined
          ? {}
          : {
              backgroundProbability: {
                value: props.output.backgroundProbability,
                format: 'float32' as const,
                length: props.rowCount
              }
            }),
        ...Object.fromEntries(
          Object.entries(props.output.status ?? {})
            .filter((entry): entry is [string, string] => entry[1] !== undefined)
            .map(([port, value]) => [port, {value, format: 'uint32' as const, length: 1}])
        )
      },
      createContributor: getView =>
        new GPUSpaceTimeHawkes({
          id,
          spatialContext: props.spatialContext,
          temporalContext: props.temporalContext,
          maximumIterations: props.maximumIterations,
          maximumParents: props.maximumParents,
          positions: getView<'float32x2'>('positions'),
          times: getView<'float32'>('times'),
          parameters: getView<'float32'>('parameters'),
          output: {
            estimates: getView<'float32'>('estimates'),
            logLikelihood:
              props.output.logLikelihood === undefined
                ? undefined
                : getView<'float32'>('logLikelihood'),
            backgroundProbability:
              props.output.backgroundProbability === undefined
                ? undefined
                : getView<'float32'>('backgroundProbability'),
            status: Object.fromEntries(
              Object.entries(props.output.status ?? {})
                .filter(([, value]) => value !== undefined)
                .map(([port]) => [port, getView<'uint32'>(port)])
            )
          }
        })
    });
  }
}
