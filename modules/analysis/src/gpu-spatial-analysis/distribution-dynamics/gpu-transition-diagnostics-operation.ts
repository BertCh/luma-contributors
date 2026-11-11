// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUTransitionDiagnostics,
  GPU_TRANSITION_HOMOGENEITY_LENGTH,
  GPU_TRANSITION_STATE_SPARSITY_LENGTH
} from './gpu-transition-diagnostics';

/** Graph names in place of the contributor's views. Output lengths are fixed by class counts. */
export type TransitionDiagnosticsOperationProps = {
  id?: string;
  counts: string;
  classCount: number;
  conditionCount?: number;
  minimumTransitions?: number;
  output: {
    homogeneity?: string;
    steadyStates?: string;
    stateSparsity?: string;
    status?: {converged?: string};
  };
};

/** Transition diagnostics operation; compile with `GPU_CONTRIBUTOR_OPERATION_LOWERING`. */
export class GPUTransitionDiagnosticsOperation extends GPUContributorOperation {
  constructor(props: TransitionDiagnosticsOperationProps) {
    const id = props.id ?? 'transition-diagnostics';
    const {output} = props;
    const conditionCount = props.conditionCount ?? 1;
    const chainCount = conditionCount > 1 ? conditionCount + 1 : 1;
    super({
      id,
      inputs: {counts: props.counts},
      outputs: {
        ...(output.homogeneity
          ? {
              homogeneity: {
                value: output.homogeneity,
                format: 'float32' as const,
                length: GPU_TRANSITION_HOMOGENEITY_LENGTH
              }
            }
          : {}),
        ...(output.steadyStates
          ? {
              steadyStates: {
                value: output.steadyStates,
                format: 'float32' as const,
                length: chainCount * props.classCount
              }
            }
          : {}),
        ...(output.stateSparsity
          ? {
              stateSparsity: {
                value: output.stateSparsity,
                format: 'uint32' as const,
                length: GPU_TRANSITION_STATE_SPARSITY_LENGTH
              }
            }
          : {}),
        ...(output.status?.converged
          ? {converged: {value: output.status.converged, format: 'uint32' as const, length: 1}}
          : {})
      },
      createContributor: getView =>
        new GPUTransitionDiagnostics({
          ...props,
          id,
          counts: getView<'uint32'>('counts'),
          output: {
            homogeneity: output.homogeneity ? getView<'float32'>('homogeneity') : undefined,
            steadyStates: output.steadyStates ? getView<'float32'>('steadyStates') : undefined,
            stateSparsity: output.stateSparsity ? getView<'uint32'>('stateSparsity') : undefined,
            status: {
              converged: output.status?.converged ? getView<'uint32'>('converged') : undefined
            }
          }
        })
    });
  }
}
